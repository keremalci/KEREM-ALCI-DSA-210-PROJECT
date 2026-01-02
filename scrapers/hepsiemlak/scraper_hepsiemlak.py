import asyncio
import json
import random
from datetime import datetime
from playwright.async_api import async_playwright
from geopy.distance import geodesic
import pandas as pd

# Constants
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

BASE_URL = "https://www.hepsiemlak.com/tr/pendik-kurtkoy-kiralik"

async def extract_listing_details(page, url):
    print(f"Scraping: {url}")
    try:
        await page.goto(url, timeout=600000)
        await page.wait_for_load_state("domcontentloaded")
        await asyncio.sleep(random.uniform(1, 3))

        details = {}
        details["Listing URL"] = url
        details["Collection Date"] = datetime.now().strftime("%Y-%m-%d")

        # Extract from __NUXT__
        try:
            nuxt_data = await page.evaluate('() => window.__NUXT__')
            if nuxt_data:
                # The structure might vary, but usually it's in data or state
                # Based on inspection: nuxt.data[0].list[0] or similar
                # We will try to find the listing object recursively or by checking common keys
                
                listing_data = None
                
                # Check if we are on a detail page, the data might be in a different place
                # Often in nuxt.data[0].currentListing or similar
                
                # Let's dump the keys to see where it might be if we fail
                # But for now, let's try to find an object that looks like a listing
                
                # Helper to search for listing data
                def find_listing(data):
                    if isinstance(data, dict):
                        if "price" in data and "mapLocation" in data:
                            return data
                        for k, v in data.items():
                            res = find_listing(v)
                            if res: return res
                    elif isinstance(data, list):
                        for item in data:
                            res = find_listing(item)
                            if res: return res
                    return None

                listing_data = find_listing(nuxt_data)
                
                if listing_data:
                    details["Listing Date"] = listing_data.get('updatedDate', 'N/A') # or createdDate
                    details["Price"] = listing_data.get('price', 'N/A')
                    details["Area(m2)"] = listing_data.get('squareMeter', 'N/A') # might be 'sqm'
                    if details["Area(m2)"] == 'N/A':
                         details["Area(m2)"] = listing_data.get('sqm', 'N/A')

                    details["Rooms"] = listing_data.get('roomCount', 'N/A') # might be 'roomAndLivingRoom'
                    if details["Rooms"] == 'N/A':
                        details["Rooms"] = listing_data.get('roomAndLivingRoom', 'N/A')

                    details["Bathrooms"] = listing_data.get('bathroomCount', 'N/A')
                    details["Building Age"] = listing_data.get('buildingAge', 'N/A')
                    details["Furnishment"] = listing_data.get('furnished', 'N/A') # might be boolean
                    details["Listing Type"] = listing_data.get('ownerType', 'N/A')
                    
                    # Location
                    map_loc = listing_data.get('mapLocation', {})
                    lat = map_loc.get('lat')
                    lon = map_loc.get('lon')
                    
                    if lat and lon:
                        house_coords = (float(lat), float(lon))
                        details["Distance to Metro (km)"] = round(geodesic(house_coords, KURTKOY_METRO_COORDS).km, 2)
                        details["Distance to University (km)"] = round(geodesic(house_coords, SABANCI_UNIV_COORDS).km, 2)
                        details["Distance to Bus Station (km)"] = round(geodesic(house_coords, BUS_STATION_COORDS).km, 2)
                    else:
                        details["Distance to Metro (km)"] = "N/A"
                        details["Distance to University (km)"] = "N/A"
                        details["Distance to Bus Station (km)"] = "N/A"
                        
                    return details
        except Exception as e:
            print(f"NUXT extraction failed: {e}")

        # Fallback to CSS
        print("Falling back to CSS selectors...")
        # ... (Implement CSS fallback if needed, but NUXT is preferred)
        
        return details

    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        print("Navigating to Hepsi Emlak...")
        await page.goto(BASE_URL)
        
        print("Waiting for listings...")
        await page.wait_for_selector("li.listing-item", timeout=30000)

        # Collect URLs
        listings_locator = page.locator("li.listing-item a.img-link")
        count = await listings_locator.count()
        print(f"Found {count} potential listings.")
        
        listing_urls = set()
        for i in range(count):
            href = await listings_locator.nth(i).get_attribute("href")
            if href:
                full_url = f"https://www.hepsiemlak.com{href}"
                listing_urls.add(full_url)
        
        print(f"Collected {len(listing_urls)} unique URLs.")

        all_data = []
        for url in listing_urls:
            data = await extract_listing_details(page, url)
            if data:
                all_data.append(data)
            await asyncio.sleep(random.uniform(2, 5))

        # Export
        if all_data:
            df = pd.DataFrame(all_data)
            output_file = "hepsiemlak_listings.xlsx"
            df.to_excel(output_file, index=False)
            print(f"Data saved to {output_file}")
            print(df.head())
        else:
            print("No data collected.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())    