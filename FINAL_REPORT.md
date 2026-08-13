# DSA-210 Project: Kurtkoy Rental Price Analysis

> **Revised report.** See `CHANGELOG.md` for the full list of code fixes
> this revision is based on. Numbers below come from the corrected
> pipeline run against the real 47-listing dataset (16 Sahibinden, 19
> Emlakjet, 12 Hepsiemlak).

## 1. Project Overview
This project analyzes the rental market in Kurtkoy to help **Sabanci
University students** identify fair prices and undervalued housing
opportunities, by scraping and aggregating data from **Sahibinden**,
**Emlakjet**, and **Hepsiemlak**.

**Core Objective**: Determine the primary drivers of rental prices in
Kurtkoy and build a "Deal Finder" tool to spot listings priced below their
predicted value.

## 2. Key Market Insights (Bonferroni-corrected)

Six hypothesis tests were run as one family (alpha = 0.05/6 = 0.0083 after
correction, since testing many hypotheses on 47 rows raises the chance of
a false positive):

| Finding | Statistic | p-value | Robust after correction? |
|---|---|---|---|
| **Size drives price** - Area vs Price | Pearson r = **0.431** | 0.0025 | **Yes** |
| Bathrooms vs Price | ANOVA F = 4.16 | 0.0498 | **No** - looked significant at raw alpha=0.05, but fails the corrected threshold. Treat as a weak/unproven signal, not an established driver. |
| Building Age vs Price | ANOVA F = 0.39 | 0.763 | No relationship detected |
| Furnished vs Unfurnished | Welch t = -0.70 | 0.490 | No relationship detected |
| Distance to Metro vs Price | Pearson r = -0.161 | 0.414 | No relationship detected |
| Distance to University vs Price | Pearson r = -0.229 | 0.240 | No relationship detected |
| Agent vs Owner | - | - | Untestable: 15 Agent vs. 1 Owner listing in the data |

**Size is the one statistically robust driver of price** in this dataset.
The "location doesn't matter much" finding from the original report holds
up, and now rests on a larger sample: Sahibinden (16 rows) **and**
Hepsiemlak (12 rows) both have real distance data (Emlakjet's raw data
never includes it), so this conclusion is drawn from 28 listings, not 16.

**Data completeness caveat:** Building Age and Listing Type are only
populated by Sahibinden (Emlakjet and Hepsiemlak leave both blank), so
those two tests are effectively Sahibinden-only comparisons on 15-16 rows -
worth re-testing once more data with those fields is collected.

## 3. Student Housing Strategy

Based on the corrected analysis, the practical advice from the original
report largely still holds, with one hedge added:

1. **Don't overpay for proximity.** Distance to metro/university/bus shows
   no statistically meaningful relationship to price (now checked on 28
   listings across two sources). Widening your search radius is unlikely
   to cost you meaningfully more.
2. **Prioritize price-per-square-meter.** Area is the one variable with a
   confirmed, statistically robust relationship to price.
3. **Check multiple platforms** - and watch for **cross-platform
   duplicates**: this dataset had 2 pairs of listings (4 rows) that appear
   to be the same unit posted on two different sites (matched by price,
   area, and room count). Don't double-count them when comparing "typical"
   prices across platforms.
4. **Treat the "bathrooms matter" and "older buildings are cheaper"
   findings as unproven**, not established, given this sample size -
   neither survives correction for multiple testing.

## 4. "Deal Finder" Results

The ML pipeline (Random Forest, selected automatically via cross-validated
comparison against Linear Regression, Decision Tree, and a leaner/fuller
feature-set comparison - see `CHANGELOG.md` item 5) flags listings whose
predicted fair price meaningfully exceeds their actual asking price, using
**out-of-fold predictions only** (no listing's own data is used to predict
its own price).

**Honest performance:** cross-validated R2 = **0.228** (Random Forest,
lean feature set: Area, Rooms, Bathrooms, Furnishing). This is
meaningfully lower than the original report's R2=0.44 - because that
number came from a Linear Regression fit that leaked training rows into
its own predictions. Re-checked honestly (out-of-fold), the *original*
feature set scores **below zero** R2 - worse than guessing the mean price.
0.228 means the current model explains roughly a fifth of price variation
out of sample: useful for flagging listings worth a second look, not a
precise valuation.

> Full deal list (with area, rooms, bathrooms, furnishing, and duplicate
> flags) available in `data/outputs/ml_analysis_results.xlsx`.

## 5. Technical Implementation

### Data Pipeline
- **Ingestion**: `scrapers/` - per-platform scrapers with conservative
  delays.
- **Cleaning**: `analysis/data_cleaning.py` is now the single shared
  source of truth for parsing all three sources' inconsistent formats
  (see `CHANGELOG.md` items 1, 8).
- **Limitation, stated explicitly**: Emlakjet never provides distance or
  Building Age/Listing Type data; Hepsiemlak never provides Bathrooms or
  Building Age/Listing Type. Only Sahibinden has all fields populated.

### Modeling
- **Evaluation**: 5-fold cross-validation (`KFold` + `cross_val_score` /
  `cross_val_predict`), not a single train/test split - the sample is too
  small (47 rows) for a single split to be a reliable estimate.
- **Models compared**: Linear Regression, Decision Tree, Random Forest,
  each on two feature sets (lean vs. extended), 6 combinations total. Best
  combination selected by mean cross-validated R2.
- **Clustering**: PCA + K-Means still segments the market into
  Budget/Mid-range/Luxury tiers, now built on the corrected feature
  cleaning.

### Repository Structure
```plaintext
DSA210Project/
├── main_pipeline.py / .ipynb   # Single entry point: runs test_results.py then ml_analysis.py
├── test_results.py / .ipynb    # Hypothesis testing (Bonferroni-corrected)
├── analysis/
│   ├── data_cleaning.py        # Shared cleaners - single source of truth
│   ├── ml_analysis.py / .ipynb # ML training, CV, Deal Finder
│   └── ANALYSIS_SUMMARY.md
├── data/                        # raw/, outputs/
├── scrapers/
├── visualizations/
├── CHANGELOG.md                 # What changed and why, with real before/after numbers
└── FINAL_REPORT.md              # This file
```

## 6. Recommendation for Future Work

With only 47 listings, several findings (bathrooms, building age,
duplicated model comparisons) are underpowered. The single highest-value
next step is **more data** - particularly Emlakjet listings with distance
data and Building Age/Listing Type coverage from Emlakjet and Hepsiemlak,
which are currently the two biggest data gaps. A scheduled scrape (e.g.
weekly, respecting the existing rate limits) would both grow the sample
and let price-over-time trends be studied, which this snapshot-based
design can't currently do.
