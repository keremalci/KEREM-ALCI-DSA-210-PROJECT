DSA-210 Project
===============

## Motivation

Having an insight into the features affecting rental prices helps renters choose homes more cost-efficiently. By collecting proper and generalized information, I aim to form an analogy between rental prices and the elements that determine them—structural features of the flat, building age, the existence of a middleman, and distance to transportation centers. The main purpose is to help a Sabancı University student find a suitable house in Kurtköy according to their budget by comparing and prioritizing these elements across different websites.

## Research Questions

- How do structural features (apartment size in m², room count, number of bathrooms, floor) influence rent in Kurtköy?
- How does proximity to transportation hubs (metro/bus stops, Sabiha Gökçen Airport) affect rental prices?
- Do newer buildings have a higher price compared to old ones?
- What is the effect of the furnishing on a flat?
- How do listing channels and agents affect the price?

## Dataset

### Time Frame

The dataset will cover only the current rental houses that are available on the market.

### Primary Data Collection

- Sahibinden.com, Emlakjet, Hepsiemlak for structural rental listings
- Google Maps for computing distances from transportation hubs

### Data Structure

- Collection Date: The date on which the data was collected.
- Listing Date: The date on which the listing was posted.
- Price: The price of the rental house.
- Area (m²): The area of the flat.
- Rooms: The number of rooms.
- Bathrooms: The number of bathrooms.
- Building Age: The age of the building in years.
- Furnishment: Whether the flat is furnished or not.
- Listing Type: Whether the house is rented from the owner or an agency.
- Distance to the nearest metro: Distance to Kurtköy Metro.
- Distance to the nearest bus station: Distance to the nearest bus station.
- Distance to the university: Distance to Sabancı University.

## Project Structure

```
KEREM-ALCI-DSA-210-PROJECT/
├── README.md                   # Project overview and documentation
├── FINAL_REPORT.md             # Detailed final report and insights
├── CHANGELOG.md                 # What changed and why, with before/after numbers
├── requirements.txt            # Python dependencies
├── sahibinden_urls.txt         # Seed list of Sahibinden URLs to scrape
│
├── main_pipeline.py             # Main entry point - runs the full workflow
├── test_results.py              # Statistical hypothesis testing (ANOVA, T-tests, Bonferroni)
│
├── scrapers/                   # Web scraping scripts (plain .py, no Jupyter required)
│   ├── sahibinden/
│   │   ├── find_sahibinden_urls.py  # Discovers new listing URLs, appends to sahibinden_urls.txt
│   │   └── scraper_sahibinden.py    # Scrapes each URL in sahibinden_urls.txt
│   ├── emlakjet/
│   │   └── scraper_emlakjet.py      # Enriches an existing Emlakjet listing file with coordinates/distances
│   └── hepsiemlak/
│       └── scraper_hepsiemlak.py    # Scrapes Hepsiemlak's Kurtkoy rental search results
│
├── data/                       # Data files
│   ├── raw/                   # Raw scraped data
│   │   ├── sahibinden/        # Sahibinden raw data
│   │   ├── emlakjet/          # Emlakjet raw data
│   │   └── hepsiemlak/        # Hepsiemlak raw data
│   └── outputs/                # Generated analysis outputs (Excel, charts)
│
├── analysis/                   # Analysis scripts and results
│   ├── data_cleaning.py        # Shared cleaners + multi-file loading - single source of truth
│   ├── ml_analysis.py          # Machine Learning & Deal Finder
│   └── ANALYSIS_SUMMARY.md    # Additional summary of findings
│
└── visualizations/             # Generated charts and plots
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/keremalci/KEREM-ALCI-DSA-210-PROJECT.git
cd KEREM-ALCI-DSA-210-PROJECT
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Analysis

Everything is a plain Python script - no Jupyter required. From the project root, with your virtual environment active:
```bash
python main_pipeline.py
```

**Key scripts:**

1.  **`main_pipeline.py`**:
    *   Runs the complete workflow.
    *   Loads data, trains models, finds deals, and generates visualizations.

2.  **`test_results.py`**:
    *   Performs detailed statistical hypothesis testing (ANOVA, T-tests, Bonferroni-corrected).

3.  **`analysis/ml_analysis.py`**:
    *   The core Machine Learning logic for price prediction.

### Running the Scrapers

Each scraper is a standalone script and can be run from anywhere (paths are anchored to the project root, not your current directory):
```bash
python scrapers/sahibinden/find_sahibinden_urls.py   # discover new listing URLs
python scrapers/sahibinden/scraper_sahibinden.py      # scrape them (up to 15 per run)
python scrapers/hepsiemlak/scraper_hepsiemlak.py      # scrape Hepsiemlak search results
python scrapers/emlakjet/scraper_emlakjet.py          # enrich an existing Emlakjet file with coordinates
```
Each opens a visible Chrome/Chromium window while it runs.

**Prerequisites:**
- A real Chrome browser installed (used by `undetected-chromedriver` for the Sahibinden scripts).
- Playwright's browser binary, installed once: `playwright install chromium` (used by the Hepsiemlak and Emlakjet scripts).
- `scraper_sahibinden.py` needs `sahibinden_urls.txt` (project root) populated first - either manually, or by running `find_sahibinden_urls.py`.
- `scraper_emlakjet.py` only enriches an *existing* listing file with coordinates - it doesn't scrape listings itself. It expects `data/raw/emlakjet/emlakjet_listings.xlsx` to already exist.

See `CHANGELOG.md` items 15-16 for why these scrapers were changed and what to check on your first run.

## Key Findings

See [FINAL_REPORT.md](FINAL_REPORT.md) for detailed findings.

## Technologies Used

- **Python 3.11+**
- **Web Scraping**: Selenium, Playwright, undetected-chromedriver
- **Data Analysis**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **Geolocation**: geopy

## License

This project is for educational purposes as part of DSA-210 course.
