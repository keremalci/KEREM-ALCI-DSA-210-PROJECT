"""
Sahibinden.com Scraper for Kurtköy Rentals.

This script safely scrapes rental listings from Sahibinden.com using undetected-chromedriver
and BeautifulSoup. It includes conservative delays and human-like interactions to avoid
detection by anti-bot measures.

Usage:
    Ensure 'sahibinden_urls.txt' exists in the project root with a list of URLs to scrape
    (see find_sahibinden_urls.py to populate it automatically).
    Run directly, from any directory: python scraper_sahibinden.py
"""

import time
import random
import re
from datetime import datetime
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
import undetected_chromedriver as uc
from geopy.distance import geodesic

# ==========================================
# Configuration & Constants
# ==========================================

# Anchored to this script's own location (not the current working directory),
# so it works whether you run it as `python scraper_sahibinden.py` from this
# folder or as `python scrapers/sahibinden/scraper_sahibinden.py` from
# anywhere else.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
URL_FILE = PROJECT_ROOT / "sahibinden_urls.txt"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "sahibinden"

# Reference coordinates for distance calculations
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

# Scraper Safety Settings
MIN_DELAY = 10                # Minimum seconds between requests
MAX_DELAY = 25                # Maximum seconds between requests
PAGE_LOAD_WAIT = 8            # Seconds to wait for page load to ensure content renders
MAX_LISTINGS_PER_SESSION = 15 # Maximum number of listings to scrape in one run

def setup_driver():
    """
    Initialize an undetected Chrome driver instance with custom options
    to mimic a real user browser.
    """
    print("Initializing browser session...")
    
    options = uc.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    # Use a standard user agent
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = uc.Chrome(options=options, version_main=None)
    driver.maximize_window()
    
    return driver

def human_like_scroll(driver):
    """
    Perform random scrolling actions to simulate human behavior.
    """
    scroll_pause = random.uniform(0.5, 1.5)
    scroll_height = random.randint(300, 600)
    
    driver.execute_script(f"window.scrollBy(0, {scroll_height});")
    time.sleep(scroll_pause)
    
    # Occasionally scroll back up to appear more natural
    if random.random() > 0.7:
        driver.execute_script(f"window.scrollBy(0, -{random.randint(50, 150)});")
        time.sleep(random.uniform(0.3, 0.8))

def extract_listing_details(driver, url, index, total):
    """
    Extract data from a single listing page.
    
    Args:
        driver: The Selenium/Undetected-Chromedriver instance.
        url (str): The URL of the listing to scrape.
        index (int): Current listing number.
        total (int): Total listings to scrape.
        
    Returns:
        dict: A dictionary containing extracted property details or None if failed.
    """
    print(f"\n[{index}/{total}] Processing URL: {url[:70]}...")
    
    try:
        # Navigate to listing
        driver.get(url)
        time.sleep(PAGE_LOAD_WAIT)
        
        # Check for manual verification/CAPTCHA
        if "Verify" in driver.title or "challenge" in driver.title.lower():
            print("\n  [!] CAPTCHA detected. Please solve it in the browser window.")
            input("  Press Enter once resolved...")
        
        # Simulate human reading/scrolling
        for _ in range(3):
            human_like_scroll(driver)
        
        # Parse content
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        details = {
            'Listing URL': url,
            'Collection Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # 1. Extract Price
        price_input = soup.find('input', {'id': 'favoriteClassifiedPrice'})
        if price_input and price_input.get('value'):
            details['Price'] = price_input['value'].strip()
        
        # 2. Extract Property Details (Area, Rooms, Age, etc.)
        info_list = soup.find('ul', class_='classifiedInfoList')
        if info_list:
            items = info_list.find_all('li')
            for item in items:
                try:
                    strong = item.find('strong')
                    span = item.find('span')
                    
                    if not strong or not span:
                        continue
                    
                    label = strong.text.strip()
                    value = span.text.strip()
                    
                    if "İlan Tarihi" in label:
                        details["Listing Date"] = value
                    elif "m² (Brüt)" in label or "m² (Net)" in label:
                        # Prioritize the first area value found
                        if "Area(m2)" not in details:
                            details["Area(m2)"] = value.replace('.', '').strip()
                    elif "Oda Sayısı" in label:
                        details["Rooms"] = value
                    elif "Banyo Sayısı" in label:
                        details["Bathrooms"] = value
                    elif "Bina Yaşı" in label:
                        details["Building Age"] = value
                    elif "Eşyalı" in label:
                        details["Furnishment"] = value
                    elif "Kimden" in label:
                        details["Listing Type"] = value
                except Exception:
                    continue
        
        # 3. Extract Location & Calculate Distances
        try:
            scripts = soup.find_all('script', type='text/javascript')
            for script in scripts:
                if script.string and 'mapOptions' in script.string:
                    lat_match = re.search(r'"lat":\s*([0-9.]+)', script.string)
                    lon_match = re.search(r'"lng":\s*([0-9.]+)', script.string)
                    
                    if lat_match and lon_match:
                        lat = float(lat_match.group(1))
                        lon = float(lon_match.group(1))
                        
                        property_coords = (lat, lon)
                        # Calculate distances to key points
                        details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
                        details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
                        details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
                        break
        except Exception as e:
            print(f"    [!] Location extraction failed: {e}")
        
        # Summary log
        price_log = details.get('Price', 'N/A')
        area_log = details.get('Area(m2)', 'N/A')
        print(f"  > Success: Price={price_log}, Area={area_log}m²")
        
        return details
        
    except Exception as e:
        print(f"  [X] Failed to process listing: {e}")
        return None

def main():
    """
    Main execution function.
    Loads URLs, initializes the scraper, and saves results.
    """
    print("-" * 60)
    print("Sahibinden.com Scraper (Standard Mode)")
    print("-" * 60)
    print(f"Configuration: Max {MAX_LISTINGS_PER_SESSION} listings, {MIN_DELAY}-{MAX_DELAY}s delay.")
    
    print(f"\nReading URL list from: {URL_FILE}")

    try:
        with open(URL_FILE, 'r') as f:
            urls = [line.strip() for line in f if 'sahibinden.com/ilan/' in line.strip()]
        print(f"Found {len(urls)} valid URLs.")
    except Exception as e:
        print(f"Error reading URL file: {e}")
        return
    
    if not urls:
        print("No URLs to scrape. Exiting.")
        return
    
    # Enforce safety limits
    if len(urls) > MAX_LISTINGS_PER_SESSION:
        print(f"Limiting scraping to first {MAX_LISTINGS_PER_SESSION} URLs for safety.")
        urls = urls[:MAX_LISTINGS_PER_SESSION]
    
    # Confirmation
    print(f"Ready to scrape {len(urls)} listings.")
    input("Press Enter to start...")
    
    driver = setup_driver()
    
    try:
        all_data = []
        
        for i, url in enumerate(urls, 1):
            result = extract_listing_details(driver, url, i, len(urls))
            if result:
                all_data.append(result)
            
            # Respectful delay between requests
            if i < len(urls):
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                print(f"  Waiting {delay:.1f} seconds...")
                time.sleep(delay)
        
        # Save Results
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            output_path = OUTPUT_DIR / f"sahibinden_safe_scrape_{timestamp}.xlsx"

            df.to_excel(output_path, index=False)
            
            print("\n" + "-" * 60)
            print("Scraping Completed Successfully")
            print("-" * 60)
            print(f"Total Listings: {len(all_data)}")
            print(f"Output File:    {output_path}")
        else:
            print("\nScraping finished but no data was collected.")
            
    finally:
        print("\nShutting down browser...")
        time.sleep(3)
        driver.quit()

if __name__ == "__main__":
    main()
