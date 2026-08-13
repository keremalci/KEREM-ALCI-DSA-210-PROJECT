# Kurtkoy Rental Price Analysis - Key Findings (Revised)

> See `../CHANGELOG.md` for what changed vs. the original analysis and why.
> Figures below are computed by the corrected `data_cleaning.py` pipeline
> against the expanded, real raw data (218 listings loaded, 216 used after
> excluding 2 confirmed data-error rows).

## Dataset Summary
- **Total Listings Used**: 216 (2 rows excluded as data errors - see
  `flag_data_errors()`; 52 rows flagged as possible cross-platform
  duplicates but kept in - see `flag_duplicates()`)
  - Sahibinden: 157 listings (16 from the original curated file + 141 from
    a previously-unused backup batch)
  - Emlakjet: 35 listings
  - Hepsiemlak: 24 listings
- **Time Period**: Most recent 12 months (current listings only, no
  historical time series)

## Key Statistics

### Price Analysis
- **Average Rent**: ₺33,672/month
- **Median Rent**: ₺32,000/month
- **Price Range**: ₺19,500 - ₺65,000
- **Price per m²**: ₺360 average (Emlakjet ₺395, Hepsiemlak ₺370,
  Sahibinden ₺351 - Emlakjet still runs highest per m²)

### Property Features
- **Average Area**: 97.3 m²
- **Most Common**: "2+1" apartments (122 of 216 listings - 3 total rooms
  counting the living room)
- **Building Age**: only reliably known for Sahibinden's original 16 rows;
  the 141-row backup batch and all of Emlakjet/Hepsiemlak leave this field
  blank, so a combined "average building age" still isn't meaningful.

### Location Factors (Sahibinden + Hepsiemlak, n=40 - Emlakjet has no
distance data)
- **Distance to Metro**: 0.77 km average
- **Distance to University**: 7.25 km average

## Correlation / Hypothesis-Test Results (Bonferroni-corrected, family of 7 tests, alpha=0.0071)

| Relationship | n | Statistic | p-value | Robust finding? |
|---|---|---|---|---|
| **Price vs Area** | 216 | Pearson r = **+0.500** | <0.00001 | **Yes - reject H0** |
| **Price vs Bathrooms** | 51 | ANOVA F = 13.84 | 0.00052 | **Yes - reject H0** |
| Price vs Building Age | 15 | ANOVA F = 0.390 | 0.763 | No relationship detected |
| Price vs Listing Type (Agent/Owner) | 32 | Welch t = -0.185 | 0.854 | No relationship detected |
| Price vs Furnishment | 71 | Welch t = 0.561 | 0.578 | No relationship detected |
| Price vs Distance to Metro | 40 | Pearson r = -0.270 | 0.092 | No relationship detected |
| Price vs Distance to University | 40 | Pearson r = -0.127 | 0.433 | No relationship detected |

## Key Insights

### 1. Size and bathroom count are the two statistically robust price drivers
Both survive Bonferroni correction, and both got stronger (not weaker)
after the dataset grew from 47 to 216 rows - area's correlation went from
r=0.43 to r=0.50, and bathrooms went from a borderline/failing result to a
clearly significant one (p=0.0005). Listings with 2 bathrooms average
₺43,900/month vs. ₺32,300 for 1 bathroom (n=5 vs. n=45 - still a small
sample for the 2-bathroom group, worth re-checking as more data comes in).

### 2. A data cleaning catch worth knowing about
Before excluding two clearly-erroneous rows (an Emlakjet listing showing
"770 m²" for a small apartment, price-per-m² ~10x below market), the
Area-vs-Price correlation on the expanded dataset briefly looked
non-significant. This is a reminder that a couple of bad rows can swing a
correlation meaningfully, especially before a dataset is very large.

### 3. Proximity still doesn't seem to matter
Distance to metro and university show no meaningful correlation with
price, now checked on 40 listings from two sources (up from 28).

### 4. Building Age and Furnishing show no proven effect
Neither survives testing, consistent with the smaller-dataset finding.

### 5. Possible duplicates are common in this market and hard to confirm
Even an exact (unrounded) match on price+area+rooms across sources hits 39
of 216 rows. Rather than guess, these are flagged in the exported Excel
(`Likely_Duplicate_Group`) for manual review, not auto-removed.

### 6. Price per m² still varies by source
Emlakjet listings run highest per m² (₺395), Sahibinden lowest (₺351).

## Recommendations for Students

### Budget-Friendly Strategy
1. **Focus on size and bathroom count** - both have confirmed statistical
   backing in this dataset.
2. **Don't overpay for location** - distance to metro/university shows no
   meaningful price relationship.
3. **Check multiple platforms, but don't assume every price/size match is
   a duplicate listing** - it may just be common market pricing.
4. **Building age and furnishing status aren't proven price factors** in
   this data - don't pay a premium or expect a discount based on either
   alone.

### Optimal Range for Students
- **Target Price**: ₺30,000 - ₺38,000/month (near the ₺32,000 median and
  ₺33,700 average)
- **Recommended Size**: 85-110 m² (near the ₺97 m² average)
- **Property Type**: "2+1" (3 total rooms) - by far the most common

## Visualizations Generated
1. `merged_analysis_plots.png` - 7-panel hypothesis test visuals, each
   annotated with its Bonferroni verdict.
2. `data/outputs/deal_finder_results.png` - CV R2 by model/feature-set
   comparison, market segmentation (PCA + K-Means), and out-of-fold
   actual-vs-predicted price scatter.

*Data Sources: Sahibinden.com, Emlakjet, Hepsiemlak.*
