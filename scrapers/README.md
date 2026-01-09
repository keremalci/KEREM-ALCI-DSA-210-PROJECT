# Scrapers

This folder contains web scraping scripts for collecting rental listing data from Turkish real estate websites.
   
## Folder Structure

- **sahibinden/** - Scripts for scraping Sahibinden.com
- **emlakjet/** - Scripts for scraping Emlakjet.com
- **hepsiemlak/** - Scripts for scraping Hepsiemlak.com

## Usage

Each subfolder contains the scraper script(s) for the respective website. Some scrapers may have their own requirements.txt file for specific dependencies.

### Running Scrapers

Navigate to the specific scraper folder and run the script:

```bash
cd scrapers/sahibinden
python scraper_sahibinden.py
```

## Notes

- All scrapers use Selenium/Playwright for web scraping
- Some websites may require handling of CAPTCHA/Cloudflare challenges
- Data is saved in Excel (.xlsx) format
- Scraped data should be saved to the `data/raw/` folder
    