import csv
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from urllib.parse import urljoin, quote_plus
import re  # For regular expressions
import time # For rate limiting
from typing import Dict, List, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import logging
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import concurrent.futures
from tqdm import tqdm
import warnings
import urllib3
import os
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='scraper.log'
)

# Suppress warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@dataclass
class CompanyData:
    """Data structure for company information"""
    name: str
    idi: str  # Added IDI field
    address: Optional[str] = None
    phone_numbers: List[str] = None
    email: Optional[str] = None
    website: Optional[str] = None  # Added website field
    detail_url: Optional[str] = None
    search_url: Optional[str] = None

    def __post_init__(self):
        if self.phone_numbers is None:
            self.phone_numbers = []

class LocalChScraper:
    def __init__(self, max_retries: int = 3, timeout: int = 10):
        """Initialize scraper with robust session handling"""
        # Define headers first
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        # Then create session
        self.session = self._create_session(max_retries, timeout)
        self.session.verify = False  # Skip SSL verification for speed
        self.cache = {}  # Simple memory cache

    def _create_session(self, max_retries: int, timeout: int) -> requests.Session:
        """Create a session with retry strategy"""
        session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update(self.headers)
        return session

    def _make_request(self, url: str) -> Optional[BeautifulSoup]:
        """Make a request with error handling and return BeautifulSoup object"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return BeautifulSoup(response.text, 'html.parser')  # Changed to html.parser
        except Exception as e:
            logging.error(f"Error fetching {url}: {str(e)}")
            return None

    def extract_company_details(self, company_detail_url: str) -> Dict:
        """Extract company details from a detail page"""
        soup = self._make_request(company_detail_url)
        if not soup:
            return {}

        details = {
            "address": None,
            "phone_numbers": [],
            "email": None,
            "website": None  # Added website field
        }

        # Extract Address
        address_container = soup.find(class_='DetailMapPreview_addressInfoContainer__qRsKX')
        if address_container:
            details["address"] = address_container.get_text(separator=" ", strip=True)

        # Extract Phone Numbers and Website
        contact_containers = soup.find_all(class_='ContactGroupsAccordion_contactGroupInfo__TWmRu')
        for container in contact_containers:
            text = container.get_text(strip=True)
            if text:
                if self.is_swiss_phone_number(text):
                    digits_only = re.sub(r'\D', '', text)
                    details["phone_numbers"].append(digits_only)
                elif self.is_website(text):
                    details["website"] = self.normalize_website(text)

        # Extract Email
        email_link = soup.find('a', href=lambda href: href and href.startswith('mailto:'))
        if email_link:
            email = email_link['href'].replace('mailto:', '').strip()
            if self.is_valid_email(email):
                details["email"] = email
            else:
                print(f"  - Invalid email format found: '{email}' - Skipping.")
        else:
            # Check if email is listed as a phone number
            for phone_text in list(details["phone_numbers"]):
                if self.is_valid_email(phone_text):
                    details["email"] = phone_text
                    details["phone_numbers"].remove(phone_text)
                    print(f"  - Found email address '{phone_text}' listed as phone number - Correcting.")
                    break

        return details

    def _extract_email(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract and validate email from soup object"""
        email_link = soup.find('a', href=lambda href: href and href.startswith('mailto:'))
        if email_link:
            email = email_link['href'].replace('mailto:', '').strip()
            return email if self.is_valid_email(email) else None
        return None

    @staticmethod
    def is_swiss_phone_number(phone: str) -> bool:
        """Validate Swiss phone numbers"""
        digits = re.sub(r'\D', '', phone)
        return len(digits) == 10

    @staticmethod
    def is_valid_email(email: str) -> bool:
        """Validate email addresses"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, email))

    @staticmethod
    def is_website(text: str) -> bool:
        """Check if text looks like a website"""
        return any(text.lower().startswith(prefix) for prefix in ['www.', 'http://', 'https://'])

    @staticmethod
    def normalize_website(url: str) -> str:
        """Normalize website URL"""
        url = url.lower().strip()
        if url.startswith('www.'):
            url = 'https://' + url
        return url

    def process_company(self, company_name: str, idi: str) -> CompanyData:
        """Process a single company with caching"""
        try:
            # Check cache first
            if company_name in self.cache:
                return self.cache[company_name]

            # Initialize company data
            company_data = CompanyData(name=company_name, idi=idi)
            
            # Get search URL and detail URL
            search_url = self.create_search_url(company_name)
            company_data.search_url = search_url
            
            # Get search results page
            soup = self._make_request(search_url)
            if not soup:
                logging.error(f"Could not fetch search page for {company_name}")
                return company_data
            
            # Extract detail URL
            detail_url = self.extract_detail_url(soup)
            if not detail_url:
                logging.warning(f"No detail URL found for {company_name}")
                return company_data
            
            company_data.detail_url = detail_url
            
            # Get company details
            details = self.extract_company_details(detail_url)
            company_data.address = details.get("address")
            company_data.phone_numbers = details.get("phone_numbers", [])
            company_data.email = details.get("email")
            company_data.website = details.get("website")
            
            # Cache the result
            self.cache[company_name] = company_data
            return company_data
            
        except Exception as e:
            logging.error(f"Error processing company {company_name}: {str(e)}")
            return CompanyData(name=company_name, idi=idi)

    def scrape_companies_from_csv(self, input_file: str, output_file: str, 
                                max_workers: int = 8) -> None:
        """Scrape multiple companies with parallel processing"""
        temp_file = f"temp_{output_file}"
        
        try:
            # Load existing results to avoid re-scraping
            existing_results = self._load_existing_results(output_file)
            temp_results = self._load_existing_results(temp_file)
            
            if temp_results:
                print(f"Found {len(temp_results)} companies in temporary file")
                existing_results.update(temp_results)
            
            # Read companies with their IDIs
            companies = []
            with open(input_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                companies = [(row['Ditta'], row['IDI']) for row in reader 
                            if row['Ditta'] not in existing_results]

            if not companies:
                print("All companies already processed!")
                if temp_results:
                    self._finalize_results(temp_file, output_file)
                return

            chunk_size = 50
            results = list(existing_results.values())
            total = len(companies)
            
            print(f"\nStarting scraping of {total} new companies with {max_workers} workers")
            print(f"Already processed: {len(existing_results)} companies")
            
            for i in range(0, total, chunk_size):
                chunk = companies[i:i + chunk_size]
                with tqdm(total=len(chunk), desc=f"Chunk {i//chunk_size + 1}") as pbar:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(self.process_company, name, idi): name 
                            for name, idi in chunk
                        }
                        
                        for future in concurrent.futures.as_completed(futures):
                            company = futures[future]
                            try:
                                company_data = future.result()
                                results.append(company_data)
                                pbar.update(1)
                                pbar.set_postfix({"Current": company[:20]})
                            except Exception as e:
                                logging.error(f"Error processing {company}: {str(e)}")
                                pbar.update(1)
                    
                    self._save_results(results, temp_file)
                    time.sleep(2)
            
            self._finalize_results(temp_file, output_file)
            
        except Exception as e:
            logging.error(f"Error during scraping: {str(e)}")
            if 'results' in locals():
                self._save_results(results, temp_file)
                print(f"Progress saved to {temp_file}")
            raise

    def _finalize_results(self, temp_file: str, output_file: str) -> None:
        """Move temporary results to final output file and cleanup"""
        try:
            # Copy temp file to output file
            shutil.copy2(temp_file, output_file)
            print(f"\nResults saved to {output_file}")
            
            # Remove temp file
            os.remove(temp_file)
            print("Temporary file cleaned up")
            
        except Exception as e:
            logging.error(f"Error finalizing results: {str(e)}")
            print(f"Warning: Could not cleanup temporary file {temp_file}")

    def _load_existing_results(self, output_file: str) -> Dict[str, CompanyData]:
        """Load existing results to avoid re-scraping"""
        existing_results = {}
        try:
            with open(output_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    company_data = CompanyData(
                        name=row['Company Name'],
                        idi=row['IDI'],
                        address=row['Address'],
                        phone_numbers=row['Phone Numbers'].split(',') if row['Phone Numbers'] else [],
                        email=row['Email'],
                        website=row['Website'],
                        detail_url=row['Detail URL'],
                        search_url=row['Search URL']
                    )
                    existing_results[row['Company Name']] = company_data
        except FileNotFoundError:
            pass
        return existing_results

    def _save_results(self, results: List[CompanyData], output_file: str) -> None:
        """Save results to CSV with backup"""
        backup_file = f"{output_file}.bak"
        
        try:
            with open(backup_file, 'w', encoding='utf-8', newline='') as outfile:
                fieldnames = ["Company Name", "IDI", "Address", "Phone Numbers", 
                             "Email", "Website", "Detail URL", "Search URL"]  # Added Website
                writer = csv.DictWriter(outfile, fieldnames=fieldnames)
                writer.writeheader()
                for data in results:
                    writer.writerow({
                        "Company Name": data.name,
                        "IDI": data.idi,
                        "Address": data.address,
                        "Phone Numbers": ','.join(data.phone_numbers) if data.phone_numbers else '',
                        "Email": data.email,
                        "Website": data.website,  # Added Website
                        "Detail URL": data.detail_url,
                        "Search URL": data.search_url
                    })
            
            shutil.move(backup_file, output_file)
            
        except Exception as e:
            logging.error(f"Error saving results to {output_file}: {str(e)}")
            if os.path.exists(backup_file):
                print(f"Backup file available at: {backup_file}")
            raise

    def create_search_url(self, company_name: str) -> str:
        """Creates the local.ch search URL for a given company name."""
        base_url = "https://www.local.ch/it/s/"
        encoded_company_name = quote_plus(company_name)
        rid_param = "?rid=8e01aa"
        return base_url + encoded_company_name + rid_param

    def extract_detail_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Extracts the detail URL from search results page."""
        element_link = soup.find(class_='ListElement_link__LabW8')

        if element_link:
            href_value = element_link.get('href')
            if href_value:
                base_url = "https://www.local.ch"
                return urljoin(base_url, href_value)
        return None

def main():
    """Main execution function"""
    input_file = 'estratto_results.csv'
    output_file = 'local_ch_company_data.csv'
    
    scraper = LocalChScraper()
    scraper.scrape_companies_from_csv(input_file, output_file)

if __name__ == "__main__":
    main()