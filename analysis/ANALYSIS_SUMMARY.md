# Kurtköy Rental Price Analysis - Key Findings

## Dataset Summary
- **Total Listings Analyzed**: 35 (after cleaning)
  - Sahibinden: 16 listings
  - Hepsiemlak: 12 listings
  - Emlakjet: 19 listings
- **Time Period**: Most recent 12 months

## Key Statistics

### Price Analysis
- **Average Rent**: ₺33,693/month
- **Median Rent**: ₺33,000/month
- **Price Range**: ₺21,000 - ₺52,000
- **Price per m²**: ₺373.31 average

### Property Features
- **Average Area**: 94.1 m²
- **Most Common**: 2+1 apartments (23 out of 35)
- **Average Building Age**: 11.5 years

### Location Factors
- **Distance to Metro**: 0.73 km average (400m - 1.16km)
- **Distance to University**: 7.23 km average
- **Distance to Bus Station**: Data available for some listings

## Correlation Analysis (Impact on Price)

### Positive Correlations (↑ Price increases with):
1. **Area (m²)**: +0.39 - Larger apartments cost more 
2. **Bathrooms**: +0.26 - More bathrooms = higher price
3. **Distance to Bus Station**: +0.04 (minimal effect)
4. **Distance to Metro**: +0.04 (minimal effect)

### Negative Correlations (↓ Price decreases with):
1. **Building Age**: -0.14 - Older buildings slightly cheaper
2. **Distance to University**: -0.10 - Farther from university = slightly cheaper

## Key Insights

### 1. **Size Matters Most**
The strongest predictor of rental price is apartment size (m²), with a correlation of 0.39.

### 2. **Proximity Doesn't Matter Much**
Surprisingly, distance to metro, bus station, and university have very weak correlations with price. This suggests that Kurtköy's compact size means location within the area doesn't significantly affect pricing.

### 3. **Building Age Has Limited Impact**
Newer buildings only command a slight premium (-0.14 correlation), suggesting renters prioritize other factors.

### 4. **Agency vs Owner**
Most listings (15 out of 16 identified) are from agencies rather than direct owners, which may affect prices.

### 5. **Price per m² Varies by Source**
- Emlakjet: ₺401.13/m² (higher-end listings)
- Sahibinden: ₺340.27/m² (more budget-friendly)

## Recommendations for Students

### Budget-Friendly Strategy:
1. **Focus on Size**: Look for smaller apartments (50-80 m²) if on a tight budget
2. **Don't Overpay for Location**: Since distance correlations are weak, you can save money by not prioritizing proximity
3. **Consider Older Buildings**: They're slightly cheaper without major quality differences
4. **Check Sahibinden.com**: Generally shows lower prices per m² than Emlakjet

### Optimal Range for Students:
- **Target Price**: ₺30,000 - ₺35,000/month (median range)
- **Recommended Size**: 80-95 m² (good value)
- **Property Type**: 2+1 apartment (most common and practical)

## Visualizations Generated
1. `analysis_overview.png` - Price distribution, price vs area, price by rooms, price by source
2. `distance_analysis.png` - Distance relationships with price
3. `combined_rental_data.xlsx` - Clean, merged dataset for further analysis

*Data Sources: Sahibinden.com, Hepsiemlak, Emlakjet*
