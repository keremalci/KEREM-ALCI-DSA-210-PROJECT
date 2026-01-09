# DSA-210 Project: Kurtköy Rental Price Analysis

## 1. Project Overview
This project analyzes the rental market in Kurtköy to help **Sabancı University students** identify fair prices and undervalued housing opportunities. By scraping and aggregating data from **Sahibinden**, **Emlakjet**, and **Hepsiemlak**, we developed a data-driven model to predict fair market value based on structural features.

**Core Objective**: Determine the primary drivers of rental prices in Kurtköy and build a "Deal Finder" tool to spot listings priced below their predicted value.

## 2. Key Market Insights
Analysis of the combined dataset points to the following conclusions:

*   **Size Dominates Price**: Apartment size ($m^2$) is the single strongest predictor of rent (**+0.39 correlation**).
*   **Bathrooms Matter**: The number of bathrooms has a strong positive impact on price, often correlating with larger, more premium units.
*   **Location is Secondary**: Contrary to expectations, distance to the university or metro has a minimal impact on price (**−0.10 correlation**). Students do not pay a significant premium for proximity.
    *   *Note: This insight relies primarily on Sahibinden data, as distance metrics were unavailable for other platforms.*
*   **Market "Sweet Spot"**: The median rent is **₺33,000**, with reliable 2+1 options clustering between **₺30,000 – ₺35,000**.
*   **Building Age Impact**: Newer buildings command only a slight premium, making older, well-maintained buildings a viable budget option.

## 3. Student Housing Strategy (Cost-Efficiency Guide)
Based on our data analysis, we recommend the following strategy for students to find the most cost-efficient rental houses:

### 1. Ignore the "Student Zone" Markup
*   **Data backing**: Distance has a low correlation with price.
    *   *Note: Distance analysis is based solely on data from Sahibinden, as other platforms did not provide geospatial information.*
*   **Strategy**: Expand your search radius by 1-2km. You will likely find higher quality apartments for the same price, without a significant increase in commute difficulty.

### 2. Prioritize Price-per-Square-Meter
Value is best measured by space. 
*   **Threshold**: Look for listings below **₺300/$m^2$**.
*   **Strategy**: Calculate this ratio for every listing. If a unit is 100$m^2$ and costs ₺25,000, that's ₺250/$m^2$ — a great deal.

### 3. Check Multiple Platforms
There is significant price variance between platforms for similar listings.
*   **Strategy**: Don't rely on just one site. Our aggregator showed that some platforms consistently had cheaper listings for the same specifications.

### 4. Consider Older Buildings
*   **Data backing**: New buildings command a premium that is often not justified by the utility provided.
*   **Strategy**: Filter for buildings aged 5-15 years. They are often structurally sound but priced lower than "0-year" new builds.

## 4. "Deal Finder" Results
We implemented a Machine Learning pipeline (Linear Regression) to predict the fair price of each listing. By comparing the **Predicted Price vs. Actual Price**, we identified "undervalued" opportunities.

### Top Detected Opportunities
| Listing Source | Actual Price | Predicted Fair Price | Potential Savings |
| :--- | :--- | :--- | :--- |
| **Emlakjet** | ₺26,000 | ₺37,646 | **₺11,646** |
| **Emlakjet** | ₺21,000 | ₺31,431 | **₺10,431** |
| **Hepsiemlak** | ₺25,500 | ₺32,500 | **₺7,000** |

> **Full deal list available in** `data/outputs/ml_analysis_results.xlsx`.

## 5. Technical Implementation

### Data Pipeline
*   **Ingestion**: Custom Python scrapers (`scrapers/`) collect raw data from major real estate platforms.
    *   *Limitation*: Emlakjet and Hepsiemlak scrapers could not retrieve geospatial data. Location analysis relies on Sahibinden.
*   **Processing**: Data is cleaned, standardized, and enriched using `geopy` where coordinates were available.

### Modeling
*   **Regression**: Trained Linear Regression ($R^2=0.44$), Random Forest, and Decision Tree models.
*   **Clustering**: Applied K-Means clustering to segment the market into **Budget**, **Mid-range**, and **Luxury** tiers.

### Repository Structure
```plaintext
DSA210Project/
├── main_pipeline.py          # Single entry point for the entire workflow
├── analysis/                 # Core logic for ML training and statistical analysis
├── data/                     # Organized storage for Raw, Processed, and Output data
├── scrapers/                 # Independent modules for each data source
└── visualizations/           # Generated dashboards and correlation plots
```
