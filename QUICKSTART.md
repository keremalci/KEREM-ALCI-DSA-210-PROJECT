# Quick Start Guide

## Running the Analysis

```bash
# From project root
cd analysis
python analyze_data.py
```

## Running Scrapers

### Sahibinden
```bash
cd scrapers/sahibinden
python scraper_sahibinden.py
# or
python scraper_sahibinden_uc.py  # Uses undetected-chrome
```

### Emlakjet
```bash
cd scrapers/emlakjet
python scraper_emlakjet.py
```

### Hepsiemlak
```bash
cd scrapers/hepsiemlak
python scraper_hepsiemlak.py
```

## File Locations

### Data Files
- **Raw data**: `data/raw/{source}/` - Original scraped data
- **Processed data**: `data/processed/` - Cleaned and merged datasets

### Results
- **Visualizations**: `visualizations/` - All generated charts (PNG)
- **Analysis summary**: `analysis/ANALYSIS_SUMMARY.md` - Key findings

### Scripts
- **Main analysis**: `analysis/analyze_data.py`
- **Scrapers**: `scrapers/{source}/scraper_*.py`

## Common Tasks

### 1. Collect New Data
```bash
# Run all scrapers to get latest listings
cd scrapers/sahibinden && python scraper_sahibinden.py
cd ../emlakjet && python scraper_emlakjet.py
cd ../hepsiemlak && python scraper_hepsiemlak.py
```

### 2. Analyze Data
```bash
cd analysis
python analyze_data.py
```

### 3. View Results
- Check console output for statistics
- Open `visualizations/*.png` for charts
- Read `analysis/ANALYSIS_SUMMARY.md` for insights

## Dependencies

Install all required packages:
```bash
pip install -r requirements.txt
```


## Documentation

- **Main README**: [README.md](README.md) - Project overview
- **Analysis Summary**: [analysis/ANALYSIS_SUMMARY.md](analysis/ANALYSIS_SUMMARY.md) - Findings
- **Folder READMEs**: Each major folder has its own README.md
    