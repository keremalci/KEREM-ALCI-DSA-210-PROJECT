import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import time
import random
import os

# Constants
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

INPUT_FILE = "sahibinden_local_listings.xlsx"
OUTPUT_FILE = "sahibinden_enriched_listings.xlsx"

def extract_details(driver, url):
    print(f"Scraping: {url}")
    try:
        driver.get(url)
        # Random sleep to mimic human behavior (increased to 15-20s)
        time.sleep(random.uniform(15, 20))

        # Check for Cloudflare title or challenge
        if "Verify you are human" in driver.title:
            print("Hit Cloudflare challenge. Please solve it in the browser window!")
            # Wait until title changes or a specific element appears
            try:
                WebDriverWait(driver, 120).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "classifiedDetailTitle"))
                )
            except:
                print("Timeout waiting for CAPTCHA solution.")
                return None

        details = {}
        details["Listing URL"] = url
        details["Collection Date"] = datetime.now().strftime("%Y-%m-%d")

        # 1. Extract Standard Fields
        try:
            info_items = driver.find_elements(By.CSS_SELECTOR, "ul.classifiedInfoList li")
            for item in info_items:
                try:
                    label = item.find_element(By.TAG_NAME, "strong").text.strip()
                    value = item.find_element(By.TAG_NAME, "span").text.strip()
                    
                    if "İlan Tarihi" in label:
                        details["Listing Date"] = value
                    elif "Fiyat" in label:
                        details["Price"] = value
                    elif "m²" in label and "Brüt" in label:
                        details["Area(m2)"] = value
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
                except:
                    continue
        except:
            pass

        # Price fallback
        if "Price" not in details:
            try:
                price_elem = driver.find_element(By.CSS_SELECTOR, "div.classifiedInfo > h3")
                details["Price"] = price_elem.text.strip()
            except:
                pass

        # 2. Extract Location (Lat/Lon)
        try:
            map_elem = driver.find_element(By.ID, "gmap")
            lat = map_elem.get_attribute("data-lat")
            lon = map_elem.get_attribute("data-lon")
            
            if lat and lon:
                house_coords = (float(lat), float(lon))
                details["Distance to Metro (km)"] = round(geodesic(house_coords, KURTKOY_METRO_COORDS).km, 2)
                details["Distance to University (km)"] = round(geodesic(house_coords, SABANCI_UNIV_COORDS).km, 2)
                details["Distance to Bus Station (km)"] = round(geodesic(house_coords, BUS_STATION_COORDS).km, 2)
            else:
                details["Distance to Metro (km)"] = "N/A"
                details["Distance to University (km)"] = "N/A"
                details["Distance to Bus Station (km)"] = "N/A"
        except:
            details["Distance to Metro (km)"] = "N/A"
            details["Distance to University (km)"] = "N/A"
            details["Distance to Bus Station (km)"] = "N/A"

        return details

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def main():
    # Load URLs
    try:
        df = pd.read_excel(INPUT_FILE)
        urls = df["Listing URL"].dropna().unique().tolist()
        urls = [u for u in urls if u.startswith("http")]
        print(f"Loaded {len(urls)} URLs to scrape.")
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Initialize Undetected Chromedriver
    options = uc.ChromeOptions()
    # options.add_argument('--headless') # Do NOT use headless with UC if you want to bypass CF easily
    
    print("Starting Chrome...")
    driver = uc.Chrome(options=options)

    # Navigate to home page to allow login
    driver.get("https://www.sahibinden.com/login")
    print("Browser opened.")
    print("IMPORTANT: Please log in to your account in the browser window.")
    print("Solve any CAPTCHAs if they appear.")
    input("Press Enter here in the terminal once you are logged in and ready to start scraping...")


    enriched_data = []
    
    try:
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing...")
            data = extract_details(driver, url)
            
            if data:
                enriched_data.append(data)
            
            # Save progress every 10
            if (i + 1) % 10 == 0:
                pd.DataFrame(enriched_data).to_excel(OUTPUT_FILE, index=False)
                print(f"Saved progress to {OUTPUT_FILE}")
                
    except KeyboardInterrupt:
        print("Interrupted! Saving current progress...")
    finally:
        if enriched_data:
            pd.DataFrame(enriched_data).to_excel(OUTPUT_FILE, index=False)
            print(f"Final data saved to {OUTPUT_FILE}")
        
        driver.quit()

if __name__ == "__main__":
    main()
    