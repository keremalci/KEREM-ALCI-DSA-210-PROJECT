# Emlakjet Scraper Guide

## Current Status

The old listing URLs (from `emlakjet_listings.xlsx`) return status 410 (Gone), meaning they're no longer available. We need to collect **fresh, current listings**.

## Challenge

Emlakjet appears to block automated scraping of their search pages. Here are the working approaches:

## Option 1: Manual URL Collection (Recommended)

### Step 1: Collect URLs Manually
1. Go to [Emlakjet.com](https://www.emlakjet.com)
2. Search for: **"Kurtköy Pendik kiralık daire"**
3. Browse through the search results
4. Right-click each listing → Copy link address
5. Save URLs to a text file (`emlakjet_urls.txt`), one URL per line

Example format:
```
https://www.emlakjet.com/ilan/some-listing-12345
https://www.emlakjet.com/ilan/another-listing-67890
...
```

### Step 2: Run the Scraper
```bash
cd scrapers/emlakjet
python scraper_emlakjet_selenium.py
```

Then provide the path to your `emlakjet_urls.txt` file.

## Option 2: Use Browser Console (Advanced)

1. Open Emlakjet search results in browser
2. Open Developer Console (F12)
3. Run this JavaScript to collect URLs:
```javascript
let urls = [];
document.querySelectorAll('a[href*="/ilan/"]').forEach(a => {
    if (a.href.includes('/ilan/') && !urls.includes(a.href)) {
        urls.push(a.href);
    }
});
console.log(urls.join('\\n'));
// Copy output and save to file
```

## Available Scrapers

### 1. `scraper_emlakjet_selenium.py` (Current - Best for fresh data)
- Uses Selenium/undetected-chrome
- Works with list of URLs
- Extracts full details + coordinates
- **Use this for new data collection**

### 2. `scraper_emlakjet.py` (Legacy - Coordinate enrichment)
- Adds coordinates to existing data
- Only works if you already have listing URLs

### 3. `scraper_emlakjet_full.py` (Experimental - BeautifulSoup)
- Attempts automated search scraping
- Currently not working due to Emlakjet blocking

## What Gets Scraped

Each listing provides:
- Price
- Area (m²)
- Room count
- Building age
- Floor
- Bathrooms
- Furnishment status
- Coordinates (lat/lon)
- Distance to metro/university/bus station

## Tips

- Collect 30-50 fresh URLs for a good dataset
- Space out your scraping (script has built-in delays)
- Use the Selenium scraper for reliability
- Check that listings are actually in Kurtköy area

## Troubleshooting

**URLs returning 404/410?**
- Listings are old/removed. Get fresh URLs.

**Can't access search page?**
- Use VPN or try from different network
- Manually collect URLs one by one

**Scraper crashes?**
- Check Chrome/ChromeDriver compatibility
- Reduce number of URLs per session
- Increase sleep delays in code
    