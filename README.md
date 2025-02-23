# Zefix & Local.ch Company Data Scraper

This project retrieves comprehensive Swiss company data by combining web scraping with API-based classification. It leverages Selenium to extract detailed information from the Swiss Central Business Name Index (Zefix) and then enriches this data with contact information obtained from Local.ch. Finally, it uses Google's Gemini API to categorize the companies.

## What it Does

1.  **Zefix Data Extraction:**
    -   Uses Selenium to automate browsing and data extraction from the Zefix website (www.zefix.admin.ch).
    -   Navigates through search results, handling pagination and dynamic page elements.
    -   Extracts key company details, including name, registration number, legal form, registered office, and purpose.

2.  **Local.ch Contact Information Retrieval:**
    -   Takes the company name and address information obtained from Zefix.
    -   Uses this information to query Local.ch (www.local.ch) to find matching company entries.
    -   Extracts available contact information, such as phone numbers and email addresses.

3.  **Company Classification:**
    -   Utilizes Google's Gemini API to classify companies into predefined categories and subcategories.
    -   Provides a structured classification based on the company's name and stated purpose.
    -   Handles multilingual input (German, French, Italian) for classification.

4.  **Data Output:**
    -   Stores the raw scraped data from Zefix in a CSV file (`data/scraped_data.csv`).
    -   Combines the Zefix data, Local.ch contact information, and Gemini API classifications.
    -   Saves the final, enriched data to a CSV file (`data/classified_companies.csv`).
    -   Saves all the files into a SQL database.

## Key Features

-   **Automated Web Scraping:** Employs Selenium for robust and reliable data extraction from dynamic websites.
-   **Data Enrichment:** Combines data from multiple sources (Zefix and Local.ch) to provide a more complete dataset.
-   **AI-Powered Classification:** Leverages Google's Gemini API for accurate and efficient company categorization.
-   **Multilingual Support:** Handles company descriptions in German, French, and Italian.
-   **Resilient Operation:** Includes error handling, rate limiting, and progress saving for robust performance.
-   **Modular Design:** Separates scraping, classification, and main execution logic for better maintainability.


## Notes

-   This project is work in progress. It already scrapes the data from Zefix and Local.ch and saves it to a SQL database but only for the companies in Ticino. The classification is not yet optimized.
-   You need to install firefox and geckodriver to run the selenium scraper.

