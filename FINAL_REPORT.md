# DSA-210 Project: Kurtkoy Rental Price Analysis

> **Revised report.** See `CHANGELOG.md` for the full list of code fixes
> and data-quality work this revision is based on. Numbers below come from
> the corrected pipeline run against the expanded, real 216-listing dataset
> (35 Emlakjet, 157 Sahibinden, 24 Hepsiemlak - see Section 5 for exactly
> where this data came from).

## 1. Project Overview
This project analyzes the rental market in Kurtkoy to help **Sabanci
University students** identify fair prices and undervalued housing
opportunities, by scraping and aggregating data from **Sahibinden**,
**Emlakjet**, and **Hepsiemlak**.

**Core Objective**: Determine the primary drivers of rental prices in
Kurtkoy and build a "Deal Finder" tool to spot listings priced below their
predicted value.

## 2. Key Market Insights (Bonferroni-corrected)

Seven hypothesis tests were run as one family (alpha = 0.05/7 = 0.0071
after correction, since testing several hypotheses on one dataset raises
the chance of a false positive):

| Finding | n | Statistic | p-value | Robust after correction? |
|---|---|---|---|---|
| **Size drives price** - Area vs Price | 216 | Pearson r = **0.500** | <0.00001 | **Yes** |
| **Bathrooms matter** - Bathrooms vs Price | 51 | ANOVA F = 13.84 | 0.00052 | **Yes** |
| Building Age vs Price | 15 | ANOVA F = 0.39 | 0.763 | No relationship detected |
| Agent vs Owner | 32 | Welch t = -0.19 | 0.854 | No relationship detected |
| Furnished vs Unfurnished | 71 | Welch t = 0.56 | 0.578 | No relationship detected |
| Distance to Metro vs Price | 40 | Pearson r = -0.270 | 0.092 | No relationship detected |
| Distance to University vs Price | 40 | Pearson r = -0.127 | 0.433 | No relationship detected |

**Size and bathroom count are the two statistically robust drivers of
price** in this dataset. Both survive correction for testing multiple
hypotheses, and both got *stronger*, not weaker, once the dataset grew
from 47 to 216 listings - a good sign they're real market effects, not
noise. Listings with 2 bathrooms average ₺43,900/month vs. ₺32,300 for 1
bathroom.

**A data-quality note worth being transparent about:** on the first pass
at the expanded 218-row dataset, the Area-vs-Price correlation actually
*flipped* to non-significant (r=0.08, p=0.23). The cause: two Emlakjet
listings had a scraped `Area(m2)` of 770 for 1-2 room apartments - clearly
a data error (confirmed by their price-per-m2 being ~10x below the market
average, whereas a genuinely large expensive property's price scales with
its size). Excluding just those 2 rows restored - and strengthened - the
correlation. See `CHANGELOG.md` item 12 for the detection method. This is
a useful reminder that a single-digit number of bad rows can meaningfully
distort a small dataset's conclusions.

**Data completeness caveat, unchanged from before:** Building Age is only
populated by Sahibinden (Emlakjet and Hepsiemlak leave it blank), so that
test is still effectively a Sahibinden-only comparison (n=15, all with
real age data).

## 3. Student Housing Strategy

1. **Prioritize price-per-square-meter and bathroom count.** These are the
   two variables with confirmed, statistically robust relationships to
   price in this dataset.
2. **Don't overpay for proximity.** Distance to metro/university/bus still
   shows no statistically meaningful relationship to price, now checked on
   40 listings across two sources (Sahibinden + Hepsiemlak; Emlakjet's raw
   data never includes distance fields).
3. **Check multiple platforms** - but treat "this exact price+area+room
   combo shows up on two sites" with caution rather than as confirmed
   duplication. At this dataset's size, common apartment sizes and round
   pricing make coincidental matches common - see `CHANGELOG.md` item 13.
4. **Building Age and Furnishing show no proven price effect** in this
   data - don't pay a premium (or expect a discount) based on either.

## 4. "Deal Finder" Results

The ML pipeline compares Linear Regression, Decision Tree, and Random
Forest on two feature sets (lean vs. extended) via 5-fold cross-validation,
and flags listings whose predicted fair price meaningfully exceeds their
actual asking price, using **out-of-fold predictions only** (no listing's
own data is used to predict its own price).

**Honest performance:** cross-validated R2 = **0.250** (Linear Regression,
lean feature set: Area, Rooms, Bathrooms, Furnishing) - the simplest model
now wins, and every model's fold-to-fold variance roughly halved compared
to the original 47-row run, meaning this estimate is meaningfully more
stable than before, not just a different number by chance. 0.25 means the
model explains about a quarter of price variation out of sample: useful
for flagging listings worth a second look, still not a precise valuation.

> Full deal list (with area, rooms, bathrooms, furnishing, and duplicate
> flags) available in `data/outputs/ml_analysis_results.xlsx`.

## 5. Technical Implementation

### Data Pipeline
- **Ingestion**: `scrapers/` - per-platform scrapers with conservative
  delays.
- **Cleaning**: `analysis/data_cleaning.py` is the single shared source of
  truth for parsing all sources' inconsistent formats.
- **Data sources**: `data_cleaning.SOURCE_FILES` merges every usable raw
  scrape batch per source (not just one file each) - a 141-row Sahibinden
  backup batch, an 18-listing Emlakjet batch, and a 12-listing Hepsiemlak
  batch were previously sitting unused. See `CHANGELOG.md` item 11 for the
  full accounting of which files were included/excluded and why.
- **Data quality**: `flag_data_errors()` excludes listings with an
  implausible price-per-m2 (Tukey's fences on the ratio); `flag_
  duplicates()` flags possible cross-platform duplicates for manual review
  without auto-excluding them (see item 13).
- **Limitation, stated explicitly**: Emlakjet never provides distance or
  Building Age/Listing Type data; Hepsiemlak never provides Bathrooms or
  Building Age/Listing Type. Only Sahibinden has all fields populated.

### Modeling
- **Evaluation**: 5-fold cross-validation (`KFold` + `cross_val_score` /
  `cross_val_predict`), not a single train/test split.
- **Models compared**: Linear Regression, Decision Tree, Random Forest,
  each on two feature sets (lean vs. extended), 6 combinations total. Best
  combination selected by mean cross-validated R2.
- **Clustering**: PCA + K-Means segments the market into Budget/Mid-range/
  Luxury tiers.

### Repository Structure
```plaintext
DSA210Project/
├── main_pipeline.py / .ipynb   # Single entry point: runs test_results.py then ml_analysis.py
├── test_results.py / .ipynb    # Hypothesis testing (Bonferroni-corrected)
├── analysis/
│   ├── data_cleaning.py        # Shared cleaners + multi-file loading - single source of truth
│   ├── ml_analysis.py / .ipynb # ML training, CV, Deal Finder
│   └── ANALYSIS_SUMMARY.md
├── data/                        # raw/, outputs/
├── scrapers/
├── visualizations/
├── CHANGELOG.md                 # What changed and why, with real before/after numbers
└── FINAL_REPORT.md              # This file
```

## 6. Recommendation for Future Work

The biggest remaining data gaps are Emlakjet distance data and Building
Age/Listing Type coverage for Emlakjet and Hepsiemlak - none of the extra
batches found this round filled those in. A scheduled scrape (e.g. weekly,
respecting the existing rate limits) would grow the sample further and
also let price-over-time trends be studied, which this snapshot-based
design can't currently do. Given how much two bad data points affected the
headline finding at n=218, it's also worth adding a lightweight sanity
check (e.g. the price-per-m2 outlier check now in `data_cleaning.py`) as a
standard step in the scrapers themselves, so errors get caught closer to
collection time.
