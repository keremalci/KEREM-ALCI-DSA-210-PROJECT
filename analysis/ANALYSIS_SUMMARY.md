# Kurtkoy Rental Price Analysis - Key Findings (Revised)

> See `../CHANGELOG.md` for what changed vs. the original analysis and why.
> Figures below are computed by the corrected `data_cleaning.py` pipeline
> against the real raw files.

## Dataset Summary
- **Total Listings Analyzed**: 47 (all rows retained; 4 rows across 2
  groups are flagged as likely cross-platform duplicates - see
  `Likely_Duplicate_Group` - but kept for transparency rather than
  silently dropped)
  - Sahibinden: 16 listings
  - Emlakjet: 19 listings
  - Hepsiemlak: 12 listings
- **Time Period**: Most recent 12 months (current listings only, no
  historical time series)

## Key Statistics

### Price Analysis
- **Average Rent**: ₺32,856/month
- **Median Rent**: ₺32,000/month
- **Price Range**: ₺21,000 - ₺52,000
- **Price per m²**: ₺363 average (Emlakjet ₺401, Sahibinden ₺340,
  Hepsiemlak ₺333 - Emlakjet consistently runs highest)

### Property Features
- **Average Area**: 94.0 m²
- **Most Common**: "2+1" apartments (28 of 47 listings - 3 total rooms
  counting the living room)
- **Building Age**: only reliably known for Sahibinden (15/16 rows);
  Emlakjet and Hepsiemlak never populate this field in the raw data, so a
  combined "average building age" isn't meaningful - reporting it as one
  number (as the original summary did) overstates how much is actually
  known about 2/3 of the dataset.

### Location Factors (Sahibinden + Hepsiemlak, n=28 - Emlakjet has no
distance data)
- **Distance to Metro**: 0.79 km average (0.40 - 1.16 km)
- **Distance to University**: 7.35 km average

## Correlation / Hypothesis-Test Results (Bonferroni-corrected, family of 6 tests, alpha=0.0083)

| Relationship | n | Statistic | p-value | Robust finding? |
|---|---|---|---|---|
| **Price vs Area** | 47 | Pearson r = **+0.431** | 0.0025 | **Yes - reject H0** |
| Price vs Bathrooms | 35 | ANOVA F = 4.158 | 0.0498 | No - fails after correction (was borderline "significant" only under the uncorrected threshold) |
| Price vs Building Age | 15 | ANOVA F = 0.390 | 0.763 | No relationship detected |
| Price vs Furnishment | 44 | Welch t = -0.700 | 0.490 | No relationship detected |
| Price vs Distance to Metro | 28 | Pearson r = -0.161 | 0.414 | No relationship detected |
| Price vs Distance to University | 28 | Pearson r = -0.229 | 0.240 | No relationship detected |
| Price vs Listing Type (Agent/Owner) | 15 vs 1 | - | - | Untestable - only 1 Owner listing |

## Key Insights

### 1. Size is the one statistically robust price driver
Area (m²) is the only relationship that survives correction for testing
multiple hypotheses on a small sample (r=0.43, p=0.0025).

### 2. Proximity still doesn't seem to matter - now checked on more data
Distance to metro and university show no meaningful correlation with
price, now verified across 28 listings from two sources (not 16 from one,
as in the original analysis) - because Hepsiemlak's raw data does include
distance fields; only Emlakjet lacks them entirely.

### 3. "Bathrooms matter" and "older buildings are cheaper" are unproven
Both looked plausible at raw p<0.05 in earlier analysis, but neither
survives correction for running several tests on ~35-47 rows. Treat these
as hypotheses for a larger future dataset, not established conclusions.

### 4. Agent vs Owner and Building Age are effectively Sahibinden-only comparisons
Emlakjet and Hepsiemlak never populate Listing Type or Building Age in the
raw data (0/19 and 0/12 rows respectively). Any claim about "most listings
are from agencies" is really a claim about Sahibinden's 16 listings, not
the full market.

### 5. Price per m² varies meaningfully by source
Emlakjet listings run about 20% higher per m² than Sahibinden or
Hepsiemlak (₺401 vs ₺333-340) in this sample.

## Recommendations for Students

### Budget-Friendly Strategy
1. **Focus on size** - it's the one factor with confirmed statistical
   backing in this dataset.
2. **Don't overpay for location** - distance to metro/university shows no
   meaningful price relationship, now confirmed on a larger sample.
3. **Check multiple platforms, but watch for duplicates** - the same unit
   can appear on two sites at the same price/area/room count.
4. **Treat "older buildings are cheaper" as a maybe, not a rule** - the
   statistical evidence for it is weak with the current sample size.

### Optimal Range for Students
- **Target Price**: ₺30,000 - ₺35,000/month (near the ₺32,000 median)
- **Recommended Size**: 80-95 m² (near the ₺94 m² average, where most
  listings cluster)
- **Property Type**: "2+1" (3 total rooms) - by far the most common and a
  reasonable default

## Visualizations Generated
1. `merged_analysis_plots.png` - 7-panel hypothesis test visuals (Price vs
   Area, Building Age, Listing Type, Bathrooms, Furnishment, Distance to
   Metro, Distance to University), each annotated with its Bonferroni
   verdict.
2. `data/outputs/deal_finder_results.png` - CV R2 by model/feature-set
   comparison, market segmentation (PCA + K-Means), and out-of-fold
   actual-vs-predicted price scatter.

*Data Sources: Sahibinden.com, Emlakjet, Hepsiemlak - collected 2025-11-30.*
