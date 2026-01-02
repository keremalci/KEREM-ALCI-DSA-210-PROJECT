import os
import sys
import pandas as pd
import requests
import re
import time
import random

# 1. LOAD YOUR FILE
# ---------------------------------------------------------
input_file = 'emlakjet_listings.xlsx' # Make sure this matches your file name

# Robust file loading with helpful error messages
if not os.path.exists(input_file):
    csv_file = input_file.replace('.xlsx', '.csv')
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"ERROR: Failed to read fallback CSV '{csv_file}': {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Neither '{input_file}' nor '{csv_file}' were found in {os.getcwd()}")
        print("Please place your Excel/CSV file in the project directory or update 'input_file'.")
        sys.exit(1)
else:
    try:
        # Use openpyxl engine explicitly for .xlsx files
        df = pd.read_excel(input_file, engine='openpyxl')
    except PermissionError:
        print(f"ERROR: Permission denied when trying to open '{input_file}'.")
        print(" - Close the file if it's open in Excel or another program, or run the script with sufficient privileges.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read '{input_file}': {e}")
        # Try CSV fallback if exists
        csv_file = input_file.replace('.xlsx', '.csv')
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e2:
                print(f"ERROR: Failed to read fallback CSV '{csv_file}': {e2}")
                sys.exit(1)
        else:
            sys.exit(1)

print(f"Loaded {len(df)} listings. Starting scrape...")

# 2. SETUP HEADERS (To look like a real browser and avoid blocks)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7',
    'Referer': 'https://www.google.com/'
}

# 3. DEFINE EXTRACTION FUNCTION
# ---------------------------------------------------------
def fetch_coordinates(url):
    if pd.isna(url): return None, None
    
    try:
        # Request the page
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # SEARCH FOR COORDINATES using the pattern we found
            # Pattern: coordinate{lat40.91...,lon29.30...}
            match = re.search(r"coordinate\{lat([\d\.]+),lon([\d\.]+)\}", content)
            
            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))
                return lat, lon
            else:
                # Alternate pattern check (Standard JSON)
                match_alt = re.search(r'"lat":([\d\.]+),"lon":([\d\.]+)', content)
                if match_alt:
                    return float(match_alt.group(1)), float(match_alt.group(2))
                    
        return None, None
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None, None

# 4. ITERATE AND SCRAPE
# ---------------------------------------------------------
# We use lists to store data temporarily
lats = []
lons = []

for index, row in df.iterrows():
    url = row['Listing URL']
    print(f"Processing {index + 1}/{len(df)}: {url[:50]}...")
    
    lat, lon = fetch_coordinates(url)
    
    lats.append(lat)
    lons.append(lon)
    
    # IMPORTANT: Sleep to avoid getting blocked by Emlakjet
    time.sleep(random.uniform(2, 5)) 

# 5. SAVE RESULTS
# ---------------------------------------------------------
df['latitude'] = lats
df['longitude'] = lons

output_file = 'emlakjet_listings_with_coordinates.xlsx'
df.to_excel(output_file, index=False)

print(f"\nDone! Data saved to {output_file}")
print(df[['Listing URL', 'latitude', 'longitude']].head())    