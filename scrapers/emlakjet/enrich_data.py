"""
Enrich Emlakjet data with missing prices and other details
Re-scrapes listings to add missing information
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from geopy.distance import geodesic

# Constants
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
}

def extract_price_and_details(url):
    """Extract price and additional details from listing"""
    print(f"  Fetching: {url[:70]}...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"    ❌ Status {response.status_code}")
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        html = response.text
        details = {}
        
        # Extract price - multiple patterns
        price_elem = soup.find('span', {'class': re.compile(r'priceWrap', re.I)})
        if price_elem:
            details['Price'] = price_elem.get_text(strip=True)
        else:
            # Try finding in page title/meta
            match = re.search(r'(\d+[\.,]\d+)\s*TL', html)
            if match:
                details['Price'] = match.group(1) + ' TL'
        
        # Extract coordinates for distances
        match = re.search(r'"lat":\s*([\d\.]+).*?"lon":\s*([\d\.]+)', html)
        if match:
            lat, lon = float(match.group(1)), float(match.group(2))
            property_coords = (lat, lon)
            details['Distance to Metro (km)'] = round(geodesic(property_coords, KURTKOY_METRO_COORDS).km, 2)
            details['Distance to University (km)'] = round(geodesic(property_coords, SABANCI_UNIV_COORDS).km, 2)
            details['Distance to Bus Station (km)'] = round(geodesic(property_coords, BUS_STATION_COORDS).km, 2)
        
        # Extract bathrooms
        bath_match = re.search(r'Banyo Sayısı</span[^>]*>\s*<span[^>]*>(\d+)', html, re.I)
        if bath_match:
            details['Bathrooms'] = int(bath_match.group(1))
        
        # Extract building age
        age_match = re.search(r'Bina Yaşı</span[^>]*>\s*<span[^>]*>([^<]+)', html, re.I)
        if age_match:
            details['Building Age'] = age_match.group(1).strip()
        
        # Extract listing date
        date_match = re.search(r'İlan Tarihi</span[^>]*>\s*<span[^>]*>([^<]+)', html, re.I)
        if date_match:
            details['Listing Date'] = date_match.group(1).strip()
        
        # Extract listing type
        if 'Emlak Ofisi' in html or 'Emlakçı' in html:
            details['Listing Type'] = 'Emlak Ofisinden'
        elif 'Sahibinden' in html:
            details['Listing Type'] = 'Sahibinden'
        
        print(f"    ✓ Price: {details.get('Price', 'N/A')}")
        return details
    
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return None

def main():
    # Load the scraped data
    input_file = '../../data/raw/emlakjet/emlakjet_listings_20251230_225742.xlsx'
    df = pd.read_excel(input_file)
    
    print("="*60)
    print("EMLAKJET DATA ENRICHMENT")
    print("="*60)
    print(f"\nLoading: {input_file}")
    print(f"Total listings: {len(df)}")
    print(f"Missing prices: {df['Price'].isna().sum()}")
    
    # Enrich each listing
    for idx, row in df.iterrows():
        url = row['Listing URL']
        print(f"\n[{idx+1}/{len(df)}]")
        
        details = extract_price_and_details(url)
        
        if details:
            for key, value in details.items():
                df.at[idx, key] = value
        
        # Sleep to be respectful
        if idx < len(df) - 1:
            sleep_time = random.uniform(2, 4)
            print(f"    ⏱️  Sleeping {sleep_time:.1f}s...")
            time.sleep(sleep_time)
    
    # Save enriched data
    output_file = '../../data/raw/emlakjet/emlakjet_listings_enriched.xlsx'
    df.to_excel(output_file, index=False)
    
    print("\n" + "="*60)
    print("ENRICHMENT COMPLETE!")
    print("="*60)
    print(f"✓ Saved to: {output_file}")
    print(f"\nData summary:")
    print(f"  Total listings: {len(df)}")
    print(f"  With prices: {df['Price'].notna().sum()}")
    print(f"  With distances: {df['Distance to Metro (km)'].notna().sum()}")
    print(f"\nPreview:")
    print(df[['Price', 'Area(m2)', 'Rooms', 'Distance to Metro (km)']].head())

if __name__ == "__main__":
    main()
    