"""
Sahibinden.com ULTRA-SAFE Scraper for Kurtköy Rentals
EXTREME CAUTION MODE - Very slow, very safe, BeautifulSoup parsing
"""

import undetected_chromedriver as uc
from bs4 import BeautifulSoup
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import time
import random
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
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    
    # Add user agent
    options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    driver = uc.Chrome(options=options, version_main=None)
    driver.maximize_window()
    
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

def extract_listing_details(driver, url, index, total):
    """
    Extract details from ONE listing - VERY CAREFULLY using BeautifulSoup
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
        
        # Get page source and parse with BeautifulSoup
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        details = {
            'Listing URL': url,
            'Collection Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Extract data using BeautifulSoup
        print("  → Extracting data...")
        
        # First, try to get price from hidden input (most reliable)
        price_input = soup.find('input', {'id': 'favoriteClassifiedPrice'})
        if price_input and price_input.get('value'):
            details['Price'] = price_input['value'].strip()
        
        # Get all info items from classifiedInfoList
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
                        if "Area(m2)" not in details:  # Use first m² value found
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
                except Exception as e:
                    continue
        
        # Get coordinates from map script
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
                        details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
                        details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
                        details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
                        break
        except Exception as e:
            print(f"    ⚠️ Location extraction error: {e}")
        
        print(f"  ✓ Extracted - Price: {details.get('Price', 'N/A')}, Area: {details.get('Area(m2)', 'N/A')}m², Rooms: {details.get('Rooms', 'N/A')}")
        return details
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

def main():
    """
    Ultra-safe Sahibinden scraper
    """
    print("="*60)
    print("SAHIBINDEN.COM ULTRA-SAFE SCRAPER v2")
    print("⚠️  EXTREME CAUTION MODE - BeautifulSoup Parsing")
    print("="*60)
    print("\nThis scraper is VERY SLOW by design to avoid detection.")
    print(f"Delays: {MIN_DELAY}-{MAX_DELAY} seconds between listings")
    print(f"Max listings per session: {MAX_LISTINGS_PER_SESSION}")
    
    # Load URLs from file
    url_file = "../../sahibinden_urls.txt"
    print(f"\nLoading URLs from: {url_file}")
    
    try:
        with open(url_file, 'r') as f:
            urls = [line.strip() for line in f if 'sahibinden.com/ilan/' in line.strip()]
        print(f"✓ Loaded {len(urls)} URLs")
    except Exception as e:
        print(f"❌ Error loading URLs: {e}")
        return
    
    if not urls:
        print("❌ No URLs found in file")
        return
    
    # Limit to prevent detection
    if len(urls) > MAX_LISTINGS_PER_SESSION:
        print(f"\n⚠️  Warning: {len(urls)} URLs provided")
        print(f"   Limiting to {MAX_LISTINGS_PER_SESSION} for safety")
        urls = urls[:MAX_LISTINGS_PER_SESSION]
    
    print(f"\n✓ Will scrape {len(urls)} listings")
    print(f"⏱️  Estimated time: {len(urls) * 20 / 60:.0f}-{len(urls) * 35 / 60:.0f} minutes")
    
    input("\nPress Enter to continue...")
    
    # Setup
    driver = setup_ultra_safe_driver()
    
    try:
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
    