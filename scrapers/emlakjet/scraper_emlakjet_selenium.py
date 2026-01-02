"""
Emlakjet.com Selenium Scraper for Kurtköy Rental Listings
Uses Selenium to handle dynamic content and avoid blocking
"""

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc
import pandas as pd
import re
import time
import random
from datetime import datetime
from geopy.distance import geodesic

# Constants
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

# Search URL
BASE_URL = "https://www.emlakjet.com"

# Possible search URLs to try
SEARCH_URLS = [
    "https://www.emlakjet.com/satilik-kiralik-konut/istanbul-pendik-kurtkoy/",
    "https://www.emlakjet.com/kiralik-daire/istanbul-pendik-kurtkoy/",
    "https://www.emlakjet.com/ilan/kiralik-konut/istanbul-pendik-kurtkoy/",
]

def setup_driver():
    """
    Set up undetected Chrome driver
    """
    print("Setting up browser...")
    options = uc.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--start-maximized')
    # Uncomment next line to run headless (no browser window)
    # options.add_argument('--headless')
    
    driver = uc.Chrome(options=options)
    return driver

def get_listing_urls_manual(driver):
    """
    Get listing URLs - requires manual navigation or URL list
    Since search pages might be blocked, this function helps guide manual collection
    """
    print("\n" + "="*60)
    print("MANUAL URL COLLECTION MODE")
    print("="*60)
    print("\nSince Emlakjet may block automated searches, here are your options:")
    print("\n1. MANUAL NAVIGATION:")
    print("   - The browser window will open")
    print("   - Navigate to: https://www.emlakjet.com")
    print("   - Search for 'Kurtköy Pendik kiralik'")
    print("   - Scroll through listings")
    print("   - Copy listing URLs manually")
    print("\n2. USE EXISTING URLs:")
    print("   - Provide a text file with one URL per line")
    
    choice = input("\nEnter '1' for manual mode, '2' to load from file: ").strip()
    
    if choice == '2':
        filepath = input("Enter path to file with URLs: ").strip()
        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and 'emlakjet.com' in line]
            print(f"✓ Loaded {len(urls)} URLs from file")
            return urls
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return []
    else:
        # Manual mode
        driver.get("https://www.emlakjet.com")
        print("\n✓ Browser opened. Navigate to Kurtköy rental listings.")
        print("   When done, come back here and press Enter...")
        input()
        
        # Try to find listing links on current page
        try:
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/ilan/']")
            urls = list(set([link.get_attribute('href') for link in links if link.get_attribute('href')]))
            print(f"✓ Found {len(urls)} listing URLs on current page")
            return urls
        except:
            print("⚠️ Could not automatically extract URLs")
            return []

def extract_listing_details(driver, url):
    """
    Extract details from a single listing
    """
    print(f"\n  Scraping: {url[:80]}...")
    
    try:
        driver.get(url)
        time.sleep(random.uniform(3, 5))  # Wait for page load
        
        details = {
            'Listing URL': url,
            'Collection Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Extract price
        try:
            price_selectors = [
                "span[class*='price']",
                "div[class*='price']",
                "span[class*='fiyat']",
                "*[class*='_2TxNQv']"
            ]
            for selector in price_selectors:
                try:
                    price_elem = driver.find_element(By.CSS_SELECTOR, selector)
                    details['Price'] = price_elem.text.strip()
                    break
                except:
                    continue
        except Exception as e:
            print(f"    ⚠️ Could not find price: {e}")
        
        # Extract features list
        try:
            feature_items = driver.find_elements(By.CSS_SELECTOR, "li, div[class*='feature'], div[class*='ozellik']")
            
            for item in feature_items:
                try:
                    text = item.text.strip()
                    
                    # Area
                    if 'm²' in text or 'Net' in text or 'Brüt' in text:
                        match = re.search(r'(\d+)\s*m²', text)
                        if match:
                            details['Area(m2)'] = int(match.group(1))
                    
                    # Rooms
                    if '+' in text and ('Oda' in text or 'oda' in text):
                        match = re.search(r'(\d+\+\d+)', text)
                        if match:
                            details['Rooms'] = match.group(1)
                    
                    # Building Age
                    if 'Bina Yaşı' in text or 'Yapım' in text:
                        details['Building Age'] = text.split(':')[-1].strip()
                    
                    # Floor
                    if 'Bulunduğu Kat' in text or ('Kat' in text and 'Sayısı' not in text):
                        details['Floor'] = text.split(':')[-1].strip()
                    
                    # Bathrooms
                    if 'Banyo' in text:
                        match = re.search(r'(\d+)', text)
                        if match:
                            details['Bathrooms'] = int(match.group(1))
                    
                    # Furnishment
                    if 'Eşya' in text or 'Mobilya' in text:
                        if 'Eşyalı' in text or 'Var' in text:
                            details['Furnishment'] = 'Eşyalı'
                        else:
                            details['Furnishment'] = 'Boş'
                
                except:
                    continue
        
        except Exception as e:
            print(f"    ⚠️ Error extracting features: {e}")
        
        # Try to extract coordinates from page source
        try:
            page_source = driver.page_source
            
            # Pattern 1: coordinate{lat...,lon...}
            match = re.search(r"coordinate\{lat([\d\.]+),lon([\d\.]+)\}", page_source)
            if match:
                lat, lon = float(match.group(1)), float(match.group(2))
            else:
                # Pattern 2: Standard JSON
                match = re.search(r'"lat":\s*([\d\.]+).*?"lon":\s*([\d\.]+)', page_source)
                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                else:
                    lat, lon = None, None
            
            if lat and lon:
                property_coords = (lat, lon)
                details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
                details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
                details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
        
        except Exception as e:
            print(f"    ⚠️ Could not extract coordinates: {e}")
        
        print(f"    ✓ Price: {details.get('Price')}, Area: {details.get('Area(m2)')}, Rooms: {details.get('Rooms')}")
        return details
    
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def scrape_with_url_list():
    """
    Simple mode: provide URLs directly
    """
    print("\n" + "="*60)
    print("EMLAKJET SCRAPER - URL LIST MODE")
    print("="*60)
    print("\nThis mode allows you to provide a list of Emlakjet listing URLs")
    print("to scrape (one per line in a text file)")
    
    filepath = input("\nEnter path to URL list file (or 'skip' to enter URLs manually): ").strip()
    
    urls = []
    if filepath.lower() != 'skip':
        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip() and 'emlakjet.com/ilan/' in line]
            print(f"✓ Loaded {len(urls)} URLs from file")
        except Exception as e:
            print(f"❌ Error: {e}")
    
    if not urls:
        print("\nEnter URLs one per line (empty line to finish):")
        while True:
            url = input().strip()
            if not url:
                break
            if 'emlakjet.com' in url:
                urls.append(url)
    
    if not urls:
        print("❌ No URLs provided")
        return
    
    # Set up driver
    driver = setup_driver()
    
    try:
        all_data = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]")
            details = extract_listing_details(driver, url)
            
            if details:
                all_data.append(details)
            
            # Delay between requests
            if i < len(urls):
                sleep_time = random.uniform(4, 8)
                print(f"    ⏱️ Sleeping {sleep_time:.1f}s...")
                time.sleep(sleep_time)
        
        # Save results
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"../../data/raw/emlakjet/emlakjet_listings_{timestamp}.xlsx"
            df.to_excel(output_file, index=False)
            
            print("\n" + "="*60)
            print("SCRAPING COMPLETE!")
            print("="*60)
            print(f"✓ Scraped {len(all_data)} listings")
            print(f"✓ Saved to: {output_file}")
            print(f"\nPreview:")
            print(df[['Price', 'Area(m2)', 'Rooms', 'Distance to Metro (km)']].head())
        else:
            print("\n❌ No data collected")
    
    finally:
        driver.quit()
        print("\n✓ Browser closed")

def main():
    """
    Main function
    """
    print("="*60)
    print("EMLAKJET SCRAPER - KURTKÖY RENTALS")
    print("="*60)
    print("\nNote: Emlakjet may block automated searches.")
    print("This scraper works best with a list of URLs.")
    
    scrape_with_url_list()

if __name__ == "__main__":
    main()
    