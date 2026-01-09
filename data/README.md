# Data

This folder contains all data files for the project, organized by processing stage.

## Folder Structure

### raw/
Contains raw, unprocessed data scraped from websites, organized by source:
- **sahibinden/** - Raw data from Sahibinden.com
- **emlakjet/** - Raw data from Emlakjet.com
- **hepsiemlak/** - Raw data from Hepsiemlak.com

### processed/
Contains cleaned and processed data ready for analysis:
- `combined_rental_data.xlsx` - Merged and cleaned dataset from all sources

## Data Fields

The typical rental listing contains:
- Listing URL
- Collection Date
- Listing Date
- Price (₺)
- Area (m²)
- Number of Rooms
- Number of Bathrooms
- Building Age
- Furnishment Status
- Listing Type (Owner vs Agency)
- Distance to Metro (km)
- Distance to University (km)
- Distance to Bus Station (km)

## Notes

- Raw data may contain inconsistent formats and missing values
- Processed data has been cleaned and standardized
- All distance measurements use geodesic calculations
    