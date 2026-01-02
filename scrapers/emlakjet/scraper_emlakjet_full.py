"""
Emlakjet.com Full Scraper for Kurtköy Rental Listings
Collects comprehensive rental data from Emlakjet
"""

import requests
from bs4 import BeautifulSoup
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

# Search URL for Kurtköy rentals
BASE_URL = "https://www.emlakjet.com"
SEARCH_URL = f"{BASE_URL}/kiralik-konut/istanbul-pendik-kurtkoy/"

# Headers to mimic real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Referer': 'https://www.google.com/',
    'Upgrade-Insecure-Requests': '1'
}

def get_listing_urls(max_pages=5):
    """
    Scrape listing URLs from search results pages
    """
    print(f"Collecting listing URLs from Emlakjet (max {max_pages} pages)...")
    all_urls = []
    
    for page in range(1, max_pages + 1):
        try:
            url = f"{SEARCH_URL}?page={page}"
            print(f"  Fetching page {page}: {url}")
            
            response = requests.get(url, headers=HEADERS, timeout=15)
            
            if response.status_code != 200:
                print(f"  ⚠️ Failed to fetch page {page} (status {response.status_code})")
                break
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find listing cards - Emlakjet uses different selectors
            listings = soup.find_all('a', {'class': re.compile(r'listing-item|_3qUI9|link')})
            
            if not listings:
                # Try alternative selectors
                listings = soup.find_all('a', href=re.compile(r'/ilan/'))
            
            page_urls = []
            for listing in listings:
                href = listing.get('href', '')
                if href and '/ilan/' in href:
                    full_url = href if href.startswith('http') else BASE_URL + href
                    if full_url not in all_urls:
                        page_urls.append(full_url)
                        all_urls.append(full_url)
            
            print(f"  ✓ Found {len(page_urls)} listings on page {page}")
            
            if len(page_urls) == 0:
                print(f"  ℹ️ No more listings found, stopping at page {page}")
                break
            
            # Respectful delay
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"  ❌ Error on page {page}: {e}")
            break
    
    print(f"\n✓ Collected {len(all_urls)} unique listing URLs")
    return all_urls

def extract_coordinates(soup, html_content):
    """
    Extract latitude and longitude from listing page
    """
    try:
        # Pattern 1: coordinate{lat...,lon...}
        match = re.search(r"coordinate\{lat([\d\.]+),lon([\d\.]+)\}", html_content)
        if match:
            return float(match.group(1)), float(match.group(2))
        
        # Pattern 2: Standard JSON
        match = re.search(r'"lat":\s*([\d\.]+).*?"lon":\s*([\d\.]+)', html_content)
        if match:
            return float(match.group(1)), float(match.group(2))
        
        # Pattern 3: coordinates array
        match = re.search(r'coordinates":\s*\[([\d\.]+),\s*([\d\.]+)\]', html_content)
        if match:
            return float(match.group(2)), float(match.group(1))  # lon, lat reversed
        
    except Exception as e:
        print(f"    Coordinate extraction error: {e}")
    
    return None, None

def extract_listing_details(url):
    """
    Extract detailed information from a single listing page
    """
    print(f"  Scraping: {url[:80]}...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        
        if response.status_code != 200:
            print(f"    ⚠️ Failed (status {response.status_code})")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        html_content = response.text
        
        details = {
            'Listing URL': url,
            'Collection Date': datetime.now().strftime("%Y-%m-%d")
        }
        
        # Extract price
        price_elem = soup.find(['span', 'div'], {'class': re.compile(r'price|fiyat|_2TxNQv', re.I)})
        if not price_elem:
            price_elem = soup.find(text=re.compile(r'[\d\.,]+\s*TL'))
        if price_elem:
            price_text = price_elem.get_text() if hasattr(price_elem, 'get_text') else str(price_elem)
            details['Price'] = price_text.strip()
        
        # Extract features from feature list
        feature_items = soup.find_all(['li', 'div'], {'class': re.compile(r'feature|ozellik|spec|_', re.I)})
        
        for item in feature_items:
            text = item.get_text(strip=True)
            
            # Area (m²)
            if 'm²' in text or 'Net Alan' in text or 'Brüt Alan' in text:
                match = re.search(r'(\d+)\s*m²', text)
                if match:
                    details['Area(m2)'] = int(match.group(1))
            
            # Rooms
            if '+' in text and any(x in text for x in ['Oda', 'oda', 'Salon']):
                match = re.search(r'(\d+\+\d+)', text)
                if match:
                    details['Rooms'] = match.group(1)
            
            # Building Age
            if 'Bina Yaşı' in text or 'Yapım Yılı' in text:
                details['Building Age'] = text.split(':')[-1].strip() if ':' in text else text
            
            # Floor
            if 'Kat' in text and 'Sayısı' not in text:
                details['Floor'] = text.split(':')[-1].strip() if ':' in text else text
            
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
        
        # Listing date
        date_elem = soup.find(text=re.compile(r'İlan Tarihi|Güncellenme'))
        if date_elem:
            date_text = date_elem.find_parent().get_text()
            details['Listing Date'] = date_text.split(':')[-1].strip() if ':' in date_text else date_text
        
        # Listing type (Owner vs Agency)
        agent_elem = soup.find(text=re.compile(r'Sahibinden|Emlakçıdan|Emlak Ofisi'))
        if agent_elem:
            details['Listing Type'] = agent_elem.strip()
        
        # Extract coordinates
        lat, lon = extract_coordinates(soup, html_content)
        
        if lat and lon:
            # Calculate distances
            property_coords = (lat, lon)
            details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
            details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
            details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
        
        print(f"    ✓ Extracted: Price={details.get('Price')}, Area={details.get('Area(m2)')}, Rooms={details.get('Rooms')}")
        return details
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def main():
    """
    Main scraping function
    """
    print("="*60)
    print("EMLAKJET SCRAPER - KURTKÖY RENTALS")
    print("="*60)
    
    # Step 1: Collect listing URLs
    listing_urls = get_listing_urls(max_pages=10)
    
    if not listing_urls:
        print("❌ No listings found. Exiting.")
        return
    
    # Step 2: Scrape each listing
    print(f"\nScraping {len(listing_urls)} listings...")
    all_data = []
    
    for i, url in enumerate(listing_urls, 1):
        print(f"\n[{i}/{len(listing_urls)}]")
        details = extract_listing_details(url)
        
        if details:
            all_data.append(details)
        
        # Respectful delay (longer for detailed pages)
        sleep_time = random.uniform(3, 7)
        print(f"    ⏱️ Sleeping {sleep_time:.1f}s...")
        time.sleep(sleep_time)
    
    # Step 3: Save to Excel
    if all_data:
        df = pd.DataFrame(all_data)
        
        # Save to data/raw/emlakjet folder
        output_file = "../../data/raw/emlakjet/emlakjet_listings_new.xlsx"
        df.to_excel(output_file, index=False)
        
        print("\n" + "="*60)
        print("SCRAPING COMPLETE!")
        print("="*60)
        print(f"✓ Scraped {len(all_data)} listings")
        print(f"✓ Saved to: {output_file}")
        print(f"\nSummary:")
        print(df[['Price', 'Area(m2)', 'Rooms']].describe())
    else:
        print("\n❌ No data collected")

if __name__ == "__main__":
    main()
    