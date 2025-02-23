import csv
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor
import threading
import os
from tenacity import retry, stop_after_attempt, wait_exponential

def init_driver():
    """
    Initialize the Firefox WebDriver.
    """
    options = FirefoxOptions()
    options.add_argument("--headless")  # Uncomment to run headless
    driver = webdriver.Firefox(options=options)
    return driver

def read_csv_urls():
    """
    Reads the scraped_query_results.csv file and returns a list of URLs
    from the "Estratto cantonale" column along with their corresponding IDI numbers.
    """
    urls_with_ids = []
    with open('scraped_query_results.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Estratto cantonale'] and row['IDI']:
                urls_with_ids.append({
                    'url': row['Estratto cantonale'],
                    'idi': row['IDI'],
                    'company_name': row['Ditta']
                })
    return urls_with_ids

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def scrape_estratto_page(driver, url):
    """
    Retries up to 3 times with exponential backoff if scraping fails.
    """
    try:
        driver.get(url)
        
        # Use shorter, more specific waits
        wait = WebDriverWait(driver, 5)  # Reduced from 10 seconds
        
        # Wait for specific elements with better conditions
        titel_element = wait.until(
            EC.presence_of_element_located((By.ID, "Titel")),
            message="Timeout waiting for Titel element"
        )
        
        content_panel = wait.until(
            EC.presence_of_element_located((By.ID, "idAuszugForm:auszugContentPanel")),
            message="Timeout waiting for content panel"
        )
        
        # First get the Titel data
        data = {}
        
        if titel_element:
            components = titel_element.text.split('\n')
            if len(components) >= 5:
                data = {
                    'Ditta': components[0].strip(),
                    'Forma giuridica': components[1].strip(),
                    'Sede': components[2].strip(),
                    'IDI': components[3].strip(),
                    'Registration_date': components[4].strip()
                }
        
        # Find tables by their headers
        headers = content_panel.find_elements(By.TAG_NAME, "th")
        for header in headers:
            header_text = header.text.strip()
            
            if "Registered address" in header_text:
                # Get the containing table
                table = header.find_element(By.XPATH, "./ancestor::table")
                # Get all rows
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # Find the first non-struck-through address (current address)
                for row in rows[1:]:  # Skip header row
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == 3:  # Ensure we have all three columns
                        address_cell = cells[2]
                        if 'strike' not in address_cell.get_attribute('class'):
                            # Replace newlines with commas
                            address_text = address_cell.text.strip().replace('\n', ', ')
                            data['Registered_address'] = address_text
                            break
                
            elif "Purpose" in header_text:
                # Get the containing table
                table = header.find_element(By.XPATH, "./ancestor::table")
                # Get rows (excluding header)
                rows = table.find_elements(By.TAG_NAME, "tr")
                
                # Find the first non-struck-through purpose
                for row in rows[1:]:  # Skip header row
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) == 3:  # Ensure we have all three columns
                        purpose_cell = cells[2]
                        if 'strike' not in purpose_cell.get_attribute('class'):
                            # Replace newlines with spaces
                            purpose_text = purpose_cell.text.strip().replace('\n', ' ')
                            data['Purpose'] = purpose_text
                            break
        
        return data
        
    except Exception as e:
        print(f"Error scraping {url}: {str(e)}")
        raise  # This will trigger a retry

def save_to_csv_append(data_list, filename='estratto_results.csv', batch_size=50):
    """
    Saves the scraped data to a CSV file in append mode.
    """
    fieldnames = [
        'idi', 'company_name', 'url', 'Ditta', 'Forma giuridica', 
        'Sede', 'IDI', 'Registration_date', 'Registered_address', 'Purpose'
    ]
    
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()
        
        for item in data_list:
            row = {
                'idi': item['idi'],
                'company_name': item['company_name'],
                'url': item['url']
            }
            if item['data']:
                row.update(item['data'])
            writer.writerow(row)

def scrape_with_thread_pool(urls_with_ids, max_workers=5):
    """
    Scrapes URLs in parallel using a thread pool.
    """
    thread_local = threading.local()
    
    def get_driver():
        if not hasattr(thread_local, "driver"):
            thread_local.driver = init_driver()
        return thread_local.driver
    
    def process_url(item):
        driver = get_driver()
        try:
            print(f"Processing: {item['company_name']}")
            data = scrape_estratto_page(driver, item['url'])
            if data:
                return {
                    'idi': item['idi'],
                    'company_name': item['company_name'],
                    'url': item['url'],
                    'data': data
                }
        except Exception as e:
            print(f"Error processing {item['company_name']}: {str(e)}")
        return None

    all_data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_url, urls_with_ids))
        all_data = [r for r in results if r is not None]
    
    return all_data

def main():
    try:
        # Read URLs from CSV
        urls_with_ids = read_csv_urls()
        print(f"Found {len(urls_with_ids)} URLs to process")
        
        # Process URLs in parallel
        all_data = scrape_with_thread_pool(urls_with_ids)
        
        # Save all data to CSV
        if all_data:
            save_to_csv_append(all_data)
            print(f"\nScraping completed! Processed {len(all_data)} companies")
            print("Results saved to estratto_results.csv")
        else:
            print("\nNo data was scraped successfully")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        print("\nProcess completed.")

if __name__ == "__main__":
    main()
