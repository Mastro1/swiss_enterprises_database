import csv
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import re

BASE_URL = "https://www.zefix.admin.ch"

def init_driver():
    options = FirefoxOptions()
    #options.add_argument("--headless")  # Run Firefox headlessly.
    driver = webdriver.Firefox(options=options)
    return driver

def dismiss_popup(driver):
    """
    Attempts to dismiss the popup by clicking on two buttons as recorded from Selenium IDE.
    It will try twice but will not halt execution if it fails.
    """
    for attempt in range(2):
        try:
            # Click the first button.
            element = driver.find_element(By.CSS_SELECTOR, ".mat-mdc-flat-button:nth-child(1) > .mat-mdc-button-touch-target")
            actions = ActionChains(driver)
            actions.move_to_element(element).click().perform()
            # Move to the body.
            element = driver.find_element(By.CSS_SELECTOR, "body")
            actions.move_to_element(element, 0, 0).perform()
            
            print(f"Popup dismissed successfully (attempt {attempt+1}).")
        except Exception as e:
            print(f"Popup dismissal attempt {attempt+1} failed: {e}")
    time.sleep(1)

def select_all_types(driver):
    """
    Navigates to the Zefix search page with the specified URL, then selects all types by:
    
    1. Opening the URL with searchTypeExact=true.
    2. Clicking the element with ID "mat-select-value-7".
    3. Clicking the element with ID "mat-option-8".
    4. Clicking the overlay backdrop (CSS selector ".cdk-overlay-backdrop").
    5. Moving the mouse to the element with CSS selector ".mat-mdc-stroked-button > .mat-mdc-button-touch-target".
    6. Finally, moving the mouse to the body element at offset (0, 0).
    
    If any of these actions fail, the function will print an error message but continue.
    """
    try:
        # Click on the element with ID "mat-select-value-7".
        driver.find_element(By.ID, "mat-select-value-7").click()

        # Delesect first option
        driver.find_element(By.ID, "mat-option-36").click()

        # Press esc
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)

        # Perform query
        driver.find_element(By.CSS_SELECTOR, "button.ob-button:nth-child(2)").click()
        
        # Wait the results
        time.sleep(10)

        
        print("select_all_types: Successfully performed all actions.")
    except Exception as e:
        print(f"select_all_types: An error occurred: {e}")

def set_items_per_page(driver):
    """
    Sets the number of displayed items per page to 100.
    Ensures proper scrolling before interacting with elements.
    """
    try:
        # First scroll all the way to the bottom
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)
        
        # Find the dropdown element
        dropdown = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "mat-select-2")))
        
        # Click the dropdown
        dropdown.click()
        
        # Select the "100" option
        driver.find_element(By.ID, "mat-option-7").click()
        time.sleep(2)
        print("Set items per page to 100.")
    except Exception as e:
        print(f"Could not set items per page to 100: {e}")

def scrape_query_results_page(driver):
    """
    Scrapes the current query results page and returns results immediately.
    Extracts URLs for "Estratto cantonale" and formats IDI numbers.
    """
    results = []
    soup = BeautifulSoup(driver.page_source, "html.parser")
    table = soup.find("table")
    if table:
        # Extract header cells
        header = []
        thead = table.find("thead")
        if thead:
            header_row = thead.find("tr")
            header = [th.get_text(strip=True) for th in header_row.find_all("th")]
        
        # Extract data rows
        tbody = table.find("tbody")
        if tbody:
            rows = tbody.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    row_data = {}
                    for i, cell in enumerate(cells):
                        if header[i] == "Estratto cantonale":
                            # Extract URL from the anchor tag if it exists
                            link = cell.find('a')
                            value = link.get('href') if link else ''
                        elif header[i] == "IDI":
                            # Extract CHE pattern using regex
                            text = cell.get_text(strip=True)
                            match = re.search(r'CHE-\d{3}\.\d{3}\.\d{3}', text)
                            value = match.group(0) if match else ''
                        else:
                            # For all other columns, get the text
                            value = cell.get_text(strip=True)
                        row_data[header[i]] = value
                    results.append(row_data)
    return results, header

def get_next_page(driver):
    """
    Checks for the presence of a pagination button labeled "Avanti" (Next)
    and returns its URL if available.
    """
    try:
        driver.find_element(By.TAG_NAME, "body").send_keys(Keys.END)

        next_elem = driver.find_element(By.CSS_SELECTOR, "button.mat-mdc-tooltip-trigger:nth-child(4)")
        next_elem.click()
        return True
    except Exception:
        return False

def append_to_csv(results, csv_filename):
    """
    Appends scraped results to a CSV file.
    Writes the header only if the CSV file doesn't already exist.
    """
    if os.path.exists(csv_filename):
        with open(csv_filename, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                existing_header = next(reader)
            except StopIteration:
                existing_header = None
    else:
        existing_header = None

    if not existing_header:
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        fieldnames = list(all_keys)
    else:
        fieldnames = existing_header

    with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not existing_header:
            writer.writeheader()
        for result in results:
            row = {key: result.get(key, "") for key in fieldnames}
            writer.writerow(row)
    print(f"Data saved to {csv_filename}")

def main():
    driver = init_driver()
    # Use the URL with searchTypeExact=true.
    start_url = "https://www.zefix.admin.ch/it/search/entity/list?registryOffice=501&legalForms=1&searchTypeExact=true"
    print("Loading start URL and dismissing popup (if present).")
    driver.get(start_url)

    # Dismiss popup
    dismiss_popup(driver)

    # Select all types
    select_all_types(driver)

    # Immediately set the number of items per page.
    set_items_per_page(driver)
    
    csv_filename = "scraped_query_results_nek.csv"
    header_written = False
    total_results = 0
    
    while True:
        print(f"Scraping query results from: {driver.current_url}")
        results, header = scrape_query_results_page(driver)
        print(f"Found {len(results)} results on this page.")
        
        if results:
            # Write results immediately to CSV
            with open(csv_filename, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=header)
                if not header_written:
                    writer.writeheader()
                    header_written = True
                writer.writerows(results)
            
            total_results += len(results)
            print(f"Total results scraped so far: {total_results}")
        
        if not get_next_page(driver):
            break
        time.sleep(1)

    driver.quit()
    print(f"Scraping completed. Total results: {total_results}")

if __name__ == "__main__":
    main()
