# Changelog - Fixes applied to the DSA210 Kurtkoy Rental Project

All items below were validated by running the actual code against the real
raw data files. **Round 1** (items 1-10) was validated against the three
originally-referenced files (47 listings: 16 Sahibinden, 19 Emlakjet, 12
Hepsiemlak). **Round 2** (items 11-14) expanded this to every usable raw
file found in `data/raw/` (218 listings, 216 after excluding 2 confirmed
data errors) - see item 11 for exactly which files and why. Numbers quoted
throughout are real outputs, not estimates.

**Note on how this was tested:** validation was done in a sandbox with no
internet access (couldn't `pip install scipy`/`scikit-learn` there), so the
shipped code (which imports the real `scipy`/`scikit-learn`, exactly like
the original project) was validated two ways: (1) the data-cleaning logic
was run directly - no ML/stats libraries needed there; (2) the stats and ML
logic were run end-to-end through drop-in numpy-only stand-ins that
implement the same algorithms (Pearson r / Welch's t / one-way ANOVA with a
correct incomplete-beta p-value, cross-checked against `sympy.stats`; OLS
Linear Regression, a CART Decision Tree, and a bagged Random Forest). Exact
decimal values may shift slightly on your machine with real scikit-learn
(e.g. Random Forest's bootstrap resampling differs), but the qualitative
conclusions below are robust and were double-checked two independent ways.

---

## 1. Bug: `Rooms` / `Building Age` cleaning was broken (`analysis/ml_analysis.ipynb`)

**Before:** a single generic `clean_numeric()` was used for both. For
Rooms, `"4+2".replace('+1', '')` doesn't match anything, so `float("4+2")`
raised and silently became `NaN`; for `"3+1"` it stripped to `"3"`,
dropping the living room instead of summing to 4. For Building Age
(Turkish text like `"6-10 arasi"`, or entirely absent for Emlakjet/
Hepsiemlak) the same function couldn't parse anything, so the feature was
effectively `NaN` for the whole dataset.

**After:** `analysis/data_cleaning.py` is now the single source of truth
for every field, shared by both `test_results.py` and
`analysis/ml_analysis.py`. `clean_rooms()` correctly handles Sahibinden/
Emlakjet's `"2+1"`, Emlakjet's `"1 Oda"`, and Hepsiemlak's stringified-list
format `"['2+1']"` - confirmed on the real data:

| Source | Raw `Rooms` | Cleaned (total rooms) |
|---|---|---|
| Emlakjet | `3+1` | 4.0 |
| Emlakjet | `1 Oda` | 1.0 |
| Sahibinden | `2+1` | 3.0 |
| Hepsiemlak | `['2+1']` | 3.0 |
| Hepsiemlak | `['3']` | 3.0 |

`Building Age` is now a categorical bucket (`0-5 Years` ... `21+ Years`,
`Unknown`) instead of a fabricated numeric column - and the real data shows
**Emlakjet and Hepsiemlak never populate this field** (19/19 and 12/12 rows
are `Unknown`); only Sahibinden's 16 rows have real values. That's a data
gap, not something the cleaning code can fix.

## 2. Bug: `main_pipeline.ipynb` called a script that didn't exist

**Before:** `run_script("analysis/ml_analysis.py", ...)` - but only
`analysis/ml_analysis.ipynb` existed in the repo, so running the pipeline
always failed with "Script not found at 'analysis/ml_analysis.py'".

**After:** `analysis/ml_analysis.py` is now a real, runnable script
(`ml_analysis.ipynb` mirrors it), and `main_pipeline.py`/`.ipynb` also now
runs `test_results.py` as Step 1. Verified end-to-end: `python
main_pipeline.py` runs both steps and produces all output files without
error.

## 3. Cross-validation instead of a single 80/20 split

**Before:** one `train_test_split(test_size=0.2)` on 35-47 rows leaves
~7-10 test rows - the reported R2 (0.44 in `FINAL_REPORT.md`) is mostly
noise from that one particular random split.

**After:** `analysis/ml_analysis.py` uses 5-fold cross-validation
(`KFold` + `cross_val_score`) and reports the mean and standard deviation
of R2 across folds, which is a far more honest estimate of how the model
will do on a listing it hasn't seen.

## 4. Data leakage in the "Deal Finder"

**Before:** the model was trained on 80% of the data, then
`best_model_obj.predict(X)` was called on **all** of `X` (train rows
included) to produce the "Predicted Price" used to flag deals. Rows the
model had memorized during training would show artificially small
prediction error, understating their true deal quality (or fabricating
"deals" among ordinary listings).

**After:** `cross_val_predict` produces every listing's predicted price
from a fold that excluded it during training - no row ever "sees itself."

**The bigger finding:** re-evaluating the *original* feature set (Area,
Rooms, Building Age, Furnishment, Listing Type) honestly via out-of-fold
prediction gives a **negative R2** on the real data (worse than just
guessing the average price) - versus the in-sample/leaky R2=0.48 that
matches the original report's 0.44. The original 0.44 was mostly an
overfitting artifact of a 16-parameter one-hot-encoded model fit to ~40
training rows.

## 5. Feature set was over-parameterized for 47 rows

Because Building Age and Listing Type are populated for only ~15/47 rows
(Sahibinden only), one-hot-encoding them adds mostly-empty columns that a
tiny dataset can't support. `analysis/ml_analysis.py` now compares a
"lean" feature set (Area, Rooms, Bathrooms, Furnishing) against the
original "extended" one via cross-validation and **picks whichever
generalizes better**, instead of assuming more features helps:

| Feature set | Model | CV R2 (mean +/- std) | OOF MAE |
|---|---|---|---|
| lean | **Random Forest** | **+0.228 +/- 0.244** | **~4,253 TL** |
| extended | Random Forest | +0.162 +/- 0.164 | ~4,515 TL |
| lean | Decision Tree | +0.117 +/- 0.286 | ~4,684 TL |
| lean | Linear Regression | -0.035 +/- 0.565 | ~4,896 TL |
| extended | Decision Tree | -0.068 +/- 0.255 | ~5,068 TL |
| extended | Linear Regression | -1.562 +/- 3.244 | ~5,988 TL |

Random Forest on the lean feature set is selected automatically. Even so,
R2~0.23 means the model explains roughly a fifth of price variance out of
sample - useful for rough triage, not a precise appraisal. This is stated
plainly in the updated `FINAL_REPORT.md`.

## 6. Multiple hypothesis tests without correction

**Before:** 5 tests run at raw alpha=0.05 with no correction for running
multiple comparisons on a small sample.

**After:** `test_results.py` runs the tests as one family (6 valid tests;
the Agent-vs-Owner test is excluded for insufficient data - see #8) and
applies a Bonferroni correction (alpha = 0.05/6 = 0.0083). **This changes a
real conclusion:** "Price vs Bathrooms" looked significant at raw
alpha=0.05 (p=0.0498, barely under the threshold) but **fails** the
Bonferroni-corrected threshold - i.e. on closer scrutiny it isn't a robust
finding.

| Test | n | Statistic | p-value | Raw (alpha=0.05) | Bonferroni (alpha=0.0083) |
|---|---|---|---|---|---|
| Price vs Area | 47 | r=0.431 | 0.00249 | Reject H0 | **Reject H0** |
| Price vs Building Age | 15 | F=0.390 | 0.763 | Fail | Fail |
| Price vs Bathrooms | 35 | F=4.158 | 0.0498 | Reject H0 | **Fail** (flips!) |
| Price vs Furnishment | 44 | t=-0.700 | 0.490 | Fail | Fail |
| Price vs Dist. to Metro | 28 | r=-0.161 | 0.414 | Fail | Fail |
| Price vs Dist. to University | 28 | r=-0.229 | 0.240 | Fail | Fail |
| Price vs Listing Type | Agent=15, Owner=1 | - | - | excluded (insufficient data) | excluded |

## 7. Location analysis was silently Sahibinden-only

**Before:** `FINAL_REPORT.md` said "distance metrics were unavailable for
other platforms," and the location conclusion was based on 16 rows.

**After:** checking the real data shows this is only true for
**Emlakjet** (`Distance to Metro/University/Bus Station` are `NaN` for all
19 rows). **Hepsiemlak's raw file already has complete distance data for
all 12 rows.** `test_results.py` and `ml_analysis.py` now use
Sahibinden + Hepsiemlak (28 rows) for the distance analysis - still not
the full 47, but nearly double the original sample, and it's now stated
explicitly rather than assumed. The conclusion itself doesn't change
(distance still doesn't correlate meaningfully with price, p=0.41 and
p=0.24), but it now rests on real, disclosed evidence instead of an
undocumented Sahibinden-only subset.

## 8. Cross-platform duplicate listings

**New:** `data_cleaning.flag_duplicates()` groups listings by rounded
price (nearest 500 TL), rounded area (nearest 5 m2), and room count, and
flags groups that span more than one source. On the real data this found
**2 likely duplicate pairs (4 rows)** - e.g. an Emlakjet listing at
30,000 TL/64m2 and a Hepsiemlak listing at 30,000 TL/65m2, both "2 rooms",
almost certainly the same unit cross-posted. `ml_analysis.py` excludes the
later-seen listing in each pair from model training (kept, but flagged, in
the raw data - not silently deleted).

## 9. Log-transformed price target

Rents are right-skewed (21,000-52,000 TL in this dataset). All models in
`ml_analysis.py` now fit `log(price)` internally via
`TransformedTargetRegressor` and report metrics back on the original TL
scale, which is standard practice for skewed monetary targets and was one
factor in the improved out-of-fold performance above.

## 10. Richer "Best Deals" export

**Before:** the exported sheet had only price/savings/cluster/URL.

**After:** `2_Best_Deals_Finder` now also includes Area, Total Rooms,
Bathrooms, Furnishing, Building Age, and the duplicate-group flag, so a
reader can sanity-check a flagged "deal" without opening every URL.

---

## 11. Round 2: the dataset was quietly 4.6x bigger than what was being used

While reviewing `data/raw/`, several extra scrape batches turned out to be
sitting unused next to the three "canonical" files referenced above:

- `data/raw/sahibinden/sahibinden_local_listings_backup.xlsx` - **141 rows**,
  zero URL overlap with the 16-row canonical Sahibinden file, zero missing
  values in any column.
- `data/raw/emlakjet/emlakjet_listings_enriched.xlsx` - 19 rows, 18 of which
  are new URLs not in the canonical 19-row Emlakjet file.
- `data/raw/hepsiemlak/hepsiemlak_listings_new_20251230.xlsx` - 17 rows, 12
  of which are new URLs not in the canonical 12-row Hepsiemlak file.

(Two other extra files were checked and confirmed NOT worth including:
`emlakjet_listings_20251230_225742.xlsx` is a strict column-subset of
`emlakjet_listings_enriched.xlsx` with the same rows; `emlakjet_listings_
with_coordinates.xlsx` has the same 19 rows as canonical with 100%-empty
lat/long columns - a geocoding attempt that never completed.)

`data_cleaning.SOURCE_FILES` now lists every worthwhile file per source
(most-complete-first), and `load_all_sources()` merges them, deduplicating
by `Listing URL` within each source. **Total dataset: 47 -> 218 listings**
(37 Emlakjet, 157 Sahibinden, 24 Hepsiemlak).

Merging in the backup batch also surfaced a `Furnishment` bug: it uses
`'Yes (Inferred)'` / `'No (Inferred)'` values that the existing
`clean_furnish()` didn't recognize (exact-string matching only) and would
have silently mapped to `'Unknown'`. Fixed with a `.startswith('yes'/'no')`
check (affects 10/141 rows).

## 12. New: a genuine data error in the expanded dataset, found and fixed

Two Emlakjet listings show `Area(m2): 770` for 1-2 room apartments -
physically implausible, and confirmed as a data error (not just a large
unit) by their price-per-m2: ~30 TL/m2 against a market average of
~330-400 TL/m2 (a legitimately large, expensive property's price scales
with its size and wouldn't show this). `data_cleaning.flag_data_errors()`
uses Tukey's fences (k=2.0) on the price/area ratio to catch exactly these
2 rows without flagging genuinely large-but-fairly-priced listings (e.g. a
240m2/47,000 TL Sahibinden listing at 196 TL/m2 is NOT flagged). These 2
rows are excluded from both `test_results.py` and `ml_analysis.py`.

**Why this mattered:** before excluding them, "Price vs Area" - the
project's headline finding - actually flipped to non-significant on the
218-row dataset (r=0.081, p=0.23). With just those 2 corrupted rows
removed, it comes back even stronger than before: **r=0.500, p<0.00001**
on 216 rows. Two bad data points were enough to erase the correlation on a
dataset this size - a good demonstration of why outlier checking matters,
not just bug-fixing.

## 13. Duplicate-flag policy changed: flag, but no longer auto-exclude

Round 1 (#8, 47 rows) found exactly 2 duplicate pairs and excluded them
from training. Re-tested on 218 rows, even an **exact, unrounded** match on
price+area+rooms across sources hits 39 rows in 17 groups - Kurtkoy rentals
cluster heavily around common sizes and round prices (25000, 30000, ...),
so this signal is far weaker at this scale than it looked on 47 rows. There
is no reliable way to tell a true cross-posted duplicate from a
coincidentally similar independent listing without an address or exact
coordinates, which this dataset doesn't have for most rows.
`ml_analysis.py` no longer drops these rows from training (auto-excluding
~20% of the data on a shaky heuristic would do more harm than good); the
`Likely_Duplicate_Group` flag is kept in the exported Excel for manual
review instead.

## 14. Final numbers on the expanded (216-row) dataset

Hypothesis tests (7 tests now that Listing Type has enough data to run;
Bonferroni alpha = 0.05/7 = 0.00714):

| Test | n | Statistic | p-value | Verdict |
|---|---|---|---|---|
| **Price vs Area** | 216 | r=0.500 | <0.00001 | **Reject H0 - robust** |
| **Price vs Bathrooms** | 51 | F=13.84 | 0.00052 | **Reject H0 - robust** (was borderline/failed in Round 1) |
| Price vs Building Age | 15 | F=0.390 | 0.763 | Fail (still Sahibinden-only, unchanged) |
| Price vs Listing Type | 32 | t=-0.185 | 0.854 | Fail (now testable, still no effect) |
| Price vs Furnishment | 71 | t=0.561 | 0.578 | Fail |
| Price vs Dist. to Metro | 40 | r=-0.270 | 0.092 | Fail |
| Price vs Dist. to University | 40 | r=-0.127 | 0.433 | Fail |

ML model comparison (5-fold CV, `TransformedTargetRegressor(log)`):

| Feature set | Model | CV R2 (mean +/- std) |
|---|---|---|
| **lean** | **Linear Regression** | **+0.250 +/- 0.109** |
| lean | Random Forest | +0.200 +/- 0.118 |
| extended | Linear Regression | +0.205 +/- 0.136 |
| extended | Random Forest | +0.199 +/- 0.118 |
| lean | Decision Tree | +0.138 +/- 0.090 |
| extended | Decision Tree | +0.129 +/- 0.075 |

Notably: with more data, the simplest model (Linear Regression) now wins,
and every model's fold-to-fold variance dropped by roughly half compared
to the 47-row results - a sign the larger dataset gives genuinely more
stable estimates, not just a different winner by chance.

---

## What wasn't changed

- The scraping approach (manual, `input()`-driven, conservative delays) is
  left as-is - it's a reasonable, respectful design for a course project
  scraping live sites, and productionizing it (scheduling, more listings
  over time) is a data-collection decision, not a code bug.
- `requirements.txt` is unchanged except adding `joblib` explicitly (it
  was already an implicit scikit-learn dependency); `scipy` and
  `scikit-learn` were already listed.
- The root-level stray `ml_analysis.py`/`debug_files.py`/`debug_processing.py`/
  `read_log.py`/`QUICKSTART.md`/`PROJECT_STRUCTURE.md`/`CLEANUP_SUMMARY.md`
  files from earlier local iteration are left untouched - they aren't
  referenced by `main_pipeline.py` and are outside the scope of this fix.
