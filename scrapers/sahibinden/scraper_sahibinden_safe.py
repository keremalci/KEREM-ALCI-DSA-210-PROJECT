"""
Sahibinden.com ULTRA-SAFE Scraper for Kurtköy Rentals
EXTREME CAUTION MODE - Very slow, very safe, very human-like
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import time
import random
import os
import re

# Constants
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

# SAFETY SETTINGS - Be VERY conservative
MIN_DELAY = 10  # Minimum seconds between requests
MAX_DELAY = 25  # Maximum seconds between requests
PAGE_LOAD_WAIT = 8  # Seconds to wait for page load
MAX_LISTINGS_PER_SESSION = 15  # Don't scrape too many at once!

def setup_ultra_safe_driver():
    """
    Set up undetected Chrome with maximum stealth
    """
    print("Setting up ultra-safe browser...")
    print("⚠️  SAFETY MODE: This will be VERY SLOW to avoid detection")
    
    options = uc.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--start-maximized')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    # Add user agent
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = uc.Chrome(options=options, version_main=None)
    
    # Give browser time to fully load
    time.sleep(2)
    
    return driver

def human_like_scroll(driver):
    """
    Scroll like a human - slowly and irregularly
    """
    scroll_pause = random.uniform(0.5, 1.5)
    scroll_height = random.randint(300, 600)
    
    driver.execute_script(f"window.scrollBy(0, {scroll_height});")
    time.sleep(scroll_pause)
    
    # Sometimes scroll back up a bit
    if random.random() > 0.7:
        driver.execute_script(f"window.scrollBy(0, -{random.randint(50, 150)});")
        time.sleep(random.uniform(0.3, 0.8))

def collect_listing_urls_manual(driver):
    """
    Manual URL collection - safest method
    Opens search page and lets you manually collect URLs
    """
    print("\n" + "="*60)
    print("MANUAL URL COLLECTION - ULTRA SAFE MODE")
    print("="*60)
    print("\nSahibinden.com has VERY strict bot detection.")
    print("We'll use the safest approach:\n")
    print("1. Browser will open to Sahibinden.com")
    print("2. YOU manually navigate and search for 'Kurtköy kiralık'")
    print("3. Browse listings naturally (use mouse, scroll, etc.)")
    print("4. Copy URLs you want to scrape")
    print("5. Save them to a text file")
    
    driver.get("https://www.sahibinden.com")
    time.sleep(5)
    
    print("\n✓ Browser opened at Sahibinden.com")
    print("\nOptions:")
    print("  A) Manually navigate and collect URLs (RECOMMENDED)")
    print("  B) Provide a file with URLs you collected earlier")
    
    choice = input("\nChoice (A/B): ").strip().upper()
    
    urls = []
    if choice == 'B':
        filepath = input("Enter path to URL file: ").strip()
        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if 'sahibinden.com/ilan/' in line]
            print(f"✓ Loaded {len(urls)} URLs")
        except Exception as e:
            print(f"❌ Error: {e}")
    else:
        print("\n" + "="*60)
        print("When you've collected URLs, save them and press Enter...")
        input()
        
        # Try to extract from current page
        try:
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/ilan/emlak-konut-kiralik']")
            urls = list(set([link.get_attribute('href') for link in links if link.get_attribute('href')]))
            print(f"✓ Found {len(urls)} URLs on current page")
        except:
            print("⚠️ Could not auto-extract. Please provide URLs manually.")
            print("\nEnter URLs (one per line, empty line to finish):")
            while True:
                url = input().strip()
                if not url:
                    break
                if 'sahibinden.com/ilan/' in url:
                    urls.append(url)
    
    return urls

def extract_listing_details(driver, url, index, total):
    """
    Extract details from ONE listing - VERY CAREFULLY
    """
    print(f"\n{'='*60}")
    print(f"[{index}/{total}] SCRAPING LISTING")
    print("="*60)
    print(f"URL: {url[:70]}...")
    
    try:
        # Navigate to listing
        print("  → Loading page...")
        driver.get(url)
        
        # LONG wait for page load
        time.sleep(PAGE_LOAD_WAIT)
        
        # Check for CAPTCHA/challenge
        if "Verify" in driver.title or "challenge" in driver.title.lower():
            print("\n  ⚠️  CAPTCHA DETECTED!")
            print("  Please solve it in the browser window.")
            input("  Press Enter when you've solved it...")
        
        # Scroll like a human
        print("  → Scrolling (human-like)...")
        for _ in range(3):
            human_like_scroll(driver)
        
        details = {
            'Listing URL': url,
            'Collection Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Extract data
        print("  → Extracting data...")
        
        # Get price (try multiple methods)
        try:
            price_elem = driver.find_element(By.CSS_SELECTOR, "div.classifiedInfo h3")
            details["Price"] = price_elem.text.strip()
        except:
            try:
                price_elem = driver.find_element(By.CSS_SELECTOR, "h3")
                if "TL" in price_elem.text:
                    details["Price"] = price_elem.text.strip()
            except:
                pass
        
        # Get all info items
        try:
            info_items = driver.find_elements(By.CSS_SELECTOR, "ul.classifiedInfoList li")
            
            for item in info_items:
                try:
                    label = item.find_element(By.TAG_NAME, "strong").text.strip()
                    value = item.find_element(By.TAG_NAME, "span").text.strip()
                    
                    if "İlan Tarihi" in label:
                        details["Listing Date"] = value
                    elif "m²" in label or "Net" in label:
                        details["Area(m2)"] = value.replace(' m²', '').replace('.', '').strip()
                    elif "Oda Sayısı" in label:
                        details["Rooms"] = value
                    elif "Banyo" in label:
                        details["Bathrooms"] = value
                    elif "Bina Yaşı" in label:
                        details["Building Age"] = value
                    elif "Eşya" in label:
                        details["Furnishment"] = value
                    elif "Kimden" in label:
                        details["Listing Type"] = value
                except:
                    continue
        except Exception as e:
            print(f"    ⚠️ Info extraction error: {e}")
        
        # Get coordinates from map
        try:
            map_elem = driver.find_element(By.ID, "gmap")
            lat = map_elem.get_attribute("data-lat")
            lon = map_elem.get_attribute("data-lon")
            
            if lat and lon:
                property_coords = (float(lat), float(lon))
                details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
                details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
                details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
        except Exception as e:
            print(f"    ⚠️ Location extraction error: {e}")
        
        print(f"  ✓ Extracted - Price: {details.get('Price', 'N/A')}, Area: {details.get('Area(m2)', 'N/A')}m²")
        return details
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """
    Ultra-safe Sahibinden scraper
    """
    print("="*60)
    print("SAHIBINDEN.COM ULTRA-SAFE SCRAPER")
    print("⚠️  EXTREME CAUTION MODE")
    print("="*60)
    print("\nThis scraper is VERY SLOW by design to avoid detection.")
    print(f"Delays: {MIN_DELAY}-{MAX_DELAY} seconds between listings")
    print(f"Max listings per session: {MAX_LISTINGS_PER_SESSION}")
    
    input("\nPress Enter to continue...")
    
    # Setup
    driver = setup_ultra_safe_driver()
    
    try:
        # Collect URLs (manual/safe method)
        urls = collect_listing_urls_manual(driver)
        
        if not urls:
            print("\n❌ No URLs provided")
            return
        
        # Limit to prevent detection
        if len(urls) > MAX_LISTINGS_PER_SESSION:
            print(f"\n⚠️  Warning: {len(urls)} URLs provided")
            print(f"   Limiting to {MAX_LISTINGS_PER_SESSION} for safety")
            urls = urls[:MAX_LISTINGS_PER_SESSION]
        
        print(f"\n✓ Will scrape {len(urls)} listings")
        print(f"⏱️  Estimated time: {len(urls) * 20 / 60:.0f}-{len(urls) * 35 / 60:.0f} minutes")
        
        confirm = input("\nContinue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Cancelled")
            return
        
        # Scrape each listing
        all_data = []
        
        for i, url in enumerate(urls, 1):
            details = extract_listing_details(driver, url, i, len(urls))
            
            if details:
                all_data.append(details)
            
            # LONG delay between requests
            if i < len(urls):
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                print(f"\n  ⏱️  Waiting {delay:.1f} seconds (safety delay)...")
                time.sleep(delay)
        
        # Save results
        if all_data:
            df = pd.DataFrame(all_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"../../data/raw/sahibinden/sahibinden_safe_scrape_{timestamp}.xlsx"
            df.to_excel(output_file, index=False)
            
            print("\n" + "="*60)
            print("SCRAPING COMPLETE!")
            print("="*60)
            print(f"✓ Scraped {len(all_data)} listings")
            print(f"✓ Saved to: {output_file}")
            print(f"\nPreview:")
            print(df[['Price', 'Area(m2)', 'Rooms']].head())
        else:
            print("\n❌ No data collected")
    
    finally:
        print("\nClosing browser in 10 seconds...")
        time.sleep(10)
        driver.quit()

if __name__ == "__main__":
    main()
    