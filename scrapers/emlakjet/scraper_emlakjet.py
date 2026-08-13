import asyncio
import os
import sys
import re
import random
from pathlib import Path
import pandas as pd
from geopy.distance import geodesic
from playwright.async_api import async_playwright

# ==========================================
# FIX (see CHANGELOG.md): the original version of this script used plain
# `requests.get()` to fetch each listing page, then regex-searched the raw
# HTML for embedded coordinates. Emlakjet injects its coordinate data
# client-side via JavaScript, so a non-JS-executing request never sees it -
# this is why every real run of the old script returned 100% NaN lat/lon
# (confirmed on the real emlakjet_listings_with_coordinates.xlsx, 19/19
# rows blank). This version uses Playwright to render the page first (same
# approach already used successfully in scraper_hepsiemlak.py), then
# runs the same coordinate patterns against the fully-rendered HTML.
#
# It also now computes Distance to Metro/University/Bus Station from the
# recovered coordinates, matching the output schema of the Sahibinden and
# Hepsiemlak scrapers, so this file can close Emlakjet's long-standing
# "no distance data" gap once real coordinates are available.
# ==========================================

# Reference coordinates (same constants used in scraper_sahibinden.py /
# scraper_hepsiemlak.py)
KURTKOY_METRO_COORDS = (40.909444, 29.296111)
SABANCI_UNIV_COORDS = (40.890547, 29.378386)
BUS_STATION_COORDS = (40.911000, 29.300000)

# Anchored to this script's own location, not the current working directory,
# so it reads/writes data/raw/emlakjet/ regardless of where you run
# `python scraper_emlakjet.py` from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "emlakjet"

# 1. LOAD YOUR FILE
# ---------------------------------------------------------
input_file = DATA_DIR / 'emlakjet_listings.xlsx'  # update this if your file is named differently

if not input_file.exists():
    csv_file = input_file.with_suffix('.csv')
    if csv_file.exists():
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"ERROR: Failed to read fallback CSV '{csv_file}': {e}")
            sys.exit(1)
    else:
        print(f"ERROR: Neither '{input_file}' nor '{csv_file}' were found.")
        print(f"Please place your Excel/CSV file in {DATA_DIR} or update 'input_file'.")
        sys.exit(1)
else:
    try:
        df = pd.read_excel(input_file, engine='openpyxl')
    except PermissionError:
        print(f"ERROR: Permission denied when trying to open '{input_file}'.")
        print(" - Close the file if it's open in Excel or another program, or run the script with sufficient privileges.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to read '{input_file}': {e}")
        csv_file = input_file.with_suffix('.csv')
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
            except Exception as e2:
                print(f"ERROR: Failed to read fallback CSV '{csv_file}': {e2}")
                sys.exit(1)
        else:
            sys.exit(1)

print(f"Loaded {len(df)} listings. Starting scrape...")

# 2. COORDINATE EXTRACTION (Playwright-rendered, JS-aware)
# ---------------------------------------------------------
COORD_PATTERNS = [
    re.compile(r"coordinate\{lat([\d.]+),lon([\d.]+)\}"),
    re.compile(r'"lat":\s*([\d.]+),\s*"lon(?:gitude)?":\s*([\d.]+)'),
    re.compile(r'"latitude":\s*([\d.]+),\s*"longitude":\s*([\d.]+)'),
]

USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')


async def fetch_coordinates(page, url):
    if pd.isna(url):
        return None, None
    try:
        await page.goto(url, timeout=45000, wait_until="networkidle")
        # small extra buffer for any late-firing JS/map widgets
        await asyncio.sleep(random.uniform(1, 2))
        content = await page.content()

        for pattern in COORD_PATTERNS:
            match = pattern.search(content)
            if match:
                return float(match.group(1)), float(match.group(2))

        return None, None
    except Exception as e:
        print(f"  [!] Error scraping {url}: {e}")
        return None, None


def compute_distances(lat, lon):
    if lat is None or lon is None:
        return None, None, None
    point = (lat, lon)
    return (
        round(geodesic(point, KURTKOY_METRO_COORDS).km, 2),
        round(geodesic(point, SABANCI_UNIV_COORDS).km, 2),
        round(geodesic(point, BUS_STATION_COORDS).km, 2),
    )


# 3. ITERATE AND SCRAPE
# ---------------------------------------------------------
async def main():
    lats, lons = [], []
    dist_metro, dist_univ, dist_bus = [], [], []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page(user_agent=USER_AGENT, locale='tr-TR')

        for index, row in df.iterrows():
            url = row['Listing URL']
            print(f"Processing {index + 1}/{len(df)}: {str(url)[:60]}...")

            lat, lon = await fetch_coordinates(page, url)
            lats.append(lat)
            lons.append(lon)

            m, u, b = compute_distances(lat, lon)
            dist_metro.append(m)
            dist_univ.append(u)
            dist_bus.append(b)

            if lat is None:
                print("    -> No coordinates found on this page.")

            # IMPORTANT: Sleep to avoid getting blocked by Emlakjet
            await asyncio.sleep(random.uniform(2, 5))

        await browser.close()

    df['latitude'] = lats
    df['longitude'] = lons
    df['Distance to Metro (km)'] = dist_metro
    df['Distance to University (km)'] = dist_univ
    df['Distance to Bus Station (km)'] = dist_bus

    output_file = DATA_DIR / 'emlakjet_listings_with_coordinates.xlsx'
    df.to_excel(output_file, index=False)

    found = df['latitude'].notna().sum()
    print(f"\nDone! Found coordinates for {found}/{len(df)} listings.")
    print(f"Data saved to {output_file}")
    print(df[['Listing URL', 'latitude', 'longitude', 'Distance to Metro (km)']].head())


if __name__ == "__main__":
    asyncio.run(main())
