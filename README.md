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
├── requirements.txt            # Python dependencies
│
├── main_pipeline.ipynb         # Main entry point (Jupyter Notebook)
├── test_results.ipynb          # Statistical testing (Jupyter Notebook)
│
├── scrapers/                   # Web scraping scripts
│   ├── sahibinden/            
│   │   └── scraper_sahibinden.ipynb  # Sahibinden scraper (Notebook)
│   ├── emlakjet/              
│   └── hepsiemlak/            
│
├── data/                       # Data files
│   ├── raw/                   # Raw scraped data
│   │   ├── sahibinden/        # Sahibinden raw data
│   │   ├── emlakjet/          # Emlakjet raw data
│   │   └── hepsiemlak/        # Hepsiemlak raw data
│   └── processed/             # Cleaned and merged data
│
├── analysis/                   # Analysis scripts and results
│   ├── ml_analysis.ipynb      # Machine Learning & Deal Finder (Notebook)
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

The project uses Jupyter Notebooks for analysis. You can run them using:
```bash
jupyter notebook
```

**Key Notebooks:**

1.  **`main_pipeline.ipynb`**:
    *   Runs the complete workflow.
    *   Loads data, trains models, finding deals, and generates visualizations.

2.  **`test_results.ipynb`**:
    *   Performs detailed statistical hypothesis testing (ANOVA, T-tests).

3.  **`analysis/ml_analysis.ipynb`**:
    *   The core Machine Learning logic for price prediction.

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
