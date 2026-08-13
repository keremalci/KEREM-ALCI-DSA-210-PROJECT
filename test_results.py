"""
DSA210 Kurtkoy Rental Project - Hypothesis Testing Report
=============================================================

Script version of test_results.ipynb (the notebook mirrors this file).

Fixes applied relative to the original test_results.ipynb:
  1. Uses the shared, corrected cleaners in `analysis/data_cleaning.py`
     instead of its own separate (and in one case buggy: total_rooms could
     silently become NaN for malformed strings without a fallback) copies.
  2. Bonferroni correction: 6 hypothesis tests are run as one "family", so
     the significance threshold is alpha/6 = 0.0083, not the raw 0.05. On
     the real data this changes the conclusion for "Price vs Bathrooms"
     from "significant" (raw p=0.0498) to "not significant after
     correction" - see CHANGELOG.md.
  3. Adds two tests that were missing: Price vs Distance to Metro and
     Price vs Distance to University. The original FINAL_REPORT.md quoted
     distance correlations, but this notebook never actually tested them
     with a p-value - they seem to have been computed ad hoc elsewhere.
  4. The distance tests now use BOTH Sahibinden and Hepsiemlak (28 rows),
     not just Sahibinden (16 rows). Hepsiemlak's raw data DOES include
     Distance to Metro/University/Bus Station columns - the original
     project's claim that "distance metrics were unavailable for other
     platforms" is only true for Emlakjet, not Hepsiemlak.
  5. Loads ALL available raw files per source (not just one each), nearly
     5x-ing the dataset from 47 to 218 listings - see data_cleaning.py's
     SOURCE_FILES for exactly which files and why.
  6. Excludes rows flagged by data_cleaning.flag_data_errors (e.g. two
     Emlakjet rows showing "Area(m2)": 770 for a 1-2 room apartment - a
     clear scraping error, confirmed by their price/m2 being ~10x off the
     rest of the market).
  7. Possible cross-platform duplicates (data_cleaning.flag_duplicates) are
     reported but NOT excluded - at 218 rows, even an exact price+area+room
     match is too common in this market to reliably distinguish a true
     duplicate from a coincidentally similar independent listing.
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, "analysis")
from data_cleaning import load_all_sources, clean_dataframe, flag_duplicates, flag_data_errors

ALPHA = 0.05


def run_test(test_name, p_val, stat_name, stat_val, null_hyp, n, bonferroni_alpha):
    print(f"\n TEST: {test_name}  (n={n})")
    print(f"   H0: {null_hyp}")
    print(f"   {stat_name}: {stat_val:.4f}")
    print(f"   p-value: {p_val:.5f}")
    raw_verdict = "REJECT H0" if p_val < ALPHA else "FAIL TO REJECT H0"
    bonf_verdict = "REJECT H0" if p_val < bonferroni_alpha else "FAIL TO REJECT H0"
    print(f"   RESULT (raw alpha=0.05):          {raw_verdict}")
    print(f"   RESULT (Bonferroni alpha={bonferroni_alpha:.4f}):  {bonf_verdict}")
    return {"test": test_name, "n": n, "stat_name": stat_name, "stat_val": stat_val,
            "p_val": p_val, "raw_verdict": raw_verdict, "bonferroni_verdict": bonf_verdict}


def main():
    print("Loading datasets...")
    raw = load_all_sources(".")
    df = clean_dataframe(raw)
    df = flag_duplicates(df)
    df = flag_data_errors(df)
    n_dupe = df["Likely_Duplicate_Group"].notna().sum()
    print(f"Merged {df['Source'].nunique()} sources. Total listings: {len(df)}"
          + (f"  ({n_dupe} rows flagged as POSSIBLE cross-platform duplicates - kept in, "
             f"see CHANGELOG.md)" if n_dupe else ""))

    n_errors = int(df["Likely_Data_Error"].sum())
    if n_errors:
        df = df[~df["Likely_Data_Error"]].copy()
        print(f"Excluded {n_errors} row(s) as likely data errors (price/m2 wildly off-market). "
              f"Testing on {len(df)} listings.")

    print("\n" + "=" * 70)
    print(" HYPOTHESIS TESTING REPORT (Emlakjet + Sahibinden + Hepsiemlak)")
    print("=" * 70)

    n_tests_planned = 6  # excludes the Listing Type test if it has insufficient data
    bonf_alpha = ALPHA / n_tests_planned
    results = []

    # 1. Price vs Area
    sub = df.dropna(subset=["clean_area", "clean_price"])
    r, p = stats.pearsonr(sub["clean_area"], sub["clean_price"])
    results.append(run_test("Price vs Area", p, "Pearson r", r, "Area does not affect Price", len(sub), bonf_alpha))

    # 2. Price vs Building Age bucket (effectively Sahibinden-only: Emlakjet/
    #    Hepsiemlak never populate Building Age in the real data)
    sub = df[df["clean_age_bucket"] != "Unknown"].dropna(subset=["clean_price"])
    groups = [g["clean_price"].values for _, g in sub.groupby("clean_age_bucket") if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, p = stats.f_oneway(*groups)
        results.append(run_test("Price vs Building Age", p, "ANOVA F", f_stat,
                                  "Age buckets have the same mean Price", len(sub), bonf_alpha))
    else:
        print("\n Test: Price vs Building Age - not enough data.")

    # 3. Price vs Listing Type (Sahibinden-only field in the real data)
    agent = df[df["clean_listing_type"] == "Agent"]["clean_price"].dropna()
    owner = df[df["clean_listing_type"] == "Owner"]["clean_price"].dropna()
    if len(agent) > 1 and len(owner) > 1:
        t_stat, p = stats.ttest_ind(agent, owner, equal_var=False)
        results.append(run_test("Price vs Listing Type", p, "Welch t", t_stat,
                                  "Agent/Owner prices are the same", len(agent) + len(owner), bonf_alpha))
    else:
        print(f"\n Test: Price vs Listing Type - not enough data (Agent={len(agent)}, Owner={len(owner)}). "
              f"Excluded from the Bonferroni family (only Sahibinden populates this field).")

    # 4. Price vs Bathrooms
    sub = df.dropna(subset=["clean_bathrooms", "clean_price"])
    groups = [g["clean_price"].values for _, g in sub.groupby("clean_bathrooms") if len(g) >= 2]
    if len(groups) >= 2:
        f_stat, p = stats.f_oneway(*groups)
        results.append(run_test("Price vs Bathrooms", p, "ANOVA F", f_stat,
                                  "Bathroom count does not affect Price", len(sub), bonf_alpha))

    # 5. Price vs Furnishment
    furn = df[df["clean_furnish"] == "Furnished"]["clean_price"].dropna()
    unfurn = df[df["clean_furnish"] == "Unfurnished"]["clean_price"].dropna()
    if len(furn) > 1 and len(unfurn) > 1:
        t_stat, p = stats.ttest_ind(furn, unfurn, equal_var=False)
        results.append(run_test("Price vs Furnishment", p, "Welch t", t_stat,
                                  "Furnished/Unfurnished prices are the same", len(furn) + len(unfurn), bonf_alpha))

    # 6. Price vs Distance to Metro (Sahibinden + Hepsiemlak; Emlakjet lacks this field)
    sub = df.dropna(subset=["Distance to Metro (km)", "clean_price"])
    r, p = stats.pearsonr(sub["Distance to Metro (km)"], sub["clean_price"])
    results.append(run_test("Price vs Distance to Metro", p, "Pearson r", r,
                              "Metro distance does not affect Price", len(sub), bonf_alpha))

    # 7. Price vs Distance to University (Sahibinden + Hepsiemlak)
    sub = df.dropna(subset=["Distance to University (km)", "clean_price"])
    r, p = stats.pearsonr(sub["Distance to University (km)"], sub["clean_price"])
    results.append(run_test("Price vs Distance to University", p, "Pearson r", r,
                              "University distance does not affect Price", len(sub), bonf_alpha))

    print("=" * 70 + "\n")
    n_tests_run = len(results)
    print(f"NOTE: {n_tests_run} tests were run as one family. Bonferroni-corrected "
          f"significance threshold = 0.05 / {n_tests_run} = {ALPHA / n_tests_run:.5f}.")

    # ==========================================
    # Visualization
    # ==========================================
    print("\nGenerating Plots...")
    plt.figure(figsize=(22, 12))
    plt.subplots_adjust(hspace=0.45, wspace=0.3)
    sns.set_style("whitegrid")

    def add_stats(ax, p_val, bonf_alpha):
        res = "Reject H0" if p_val < bonf_alpha else "Fail to Reject H0 (Bonferroni)"
        ax.text(0.05, 0.95, f"p={p_val:.4f}\n{res}", transform=ax.transAxes,
                verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    panels = [
        ("Price vs Area", "clean_area", None),
        ("Price vs Building Age", "clean_age_bucket", None),
        ("Price vs Listing Type", "clean_listing_type", None),
        ("Price vs Bathrooms", "clean_bathrooms", None),
        ("Price vs Furnishment", "clean_furnish", None),
        ("Price vs Distance to Metro", "Distance to Metro (km)", None),
        ("Price vs Distance to University", "Distance to University (km)", None),
    ]
    result_by_name = {r["test"]: r for r in results}

    for i, (title, col, _) in enumerate(panels, 1):
        ax = plt.subplot(3, 3, i)
        if col in ("clean_area", "Distance to Metro (km)", "Distance to University (km)"):
            sub = df.dropna(subset=[col, "clean_price"])
            sns.scatterplot(data=sub, x=col, y="clean_price", hue="Source", alpha=0.7, ax=ax)
            if len(sub) > 2:
                sns.regplot(data=sub, x=col, y="clean_price", scatter=False, color="red", ax=ax)
        elif col == "clean_age_bucket":
            sub = df[df["clean_age_bucket"] != "Unknown"].dropna(subset=["clean_price"])
            if not sub.empty:
                from data_cleaning import AGE_BUCKET_ORDER
                order = [o for o in AGE_BUCKET_ORDER if o in sub["clean_age_bucket"].unique()]
                sns.boxplot(data=sub, x="clean_age_bucket", y="clean_price", order=order, palette="Blues", ax=ax)
                ax.tick_params(axis="x", rotation=30)
        elif col == "clean_listing_type":
            sub = df[df["clean_listing_type"] != "Unknown"].dropna(subset=["clean_price"])
            if not sub.empty:
                sns.boxplot(data=sub, x="clean_listing_type", y="clean_price", palette="Set2", ax=ax)
        elif col == "clean_bathrooms":
            sub = df.dropna(subset=["clean_bathrooms", "clean_price"])
            sns.boxplot(data=sub, x="clean_bathrooms", y="clean_price", palette="Purples", ax=ax)
        elif col == "clean_furnish":
            sub = df[df["clean_furnish"] != "Unknown"].dropna(subset=["clean_price"])
            sns.boxplot(data=sub, x="clean_furnish", y="clean_price", palette="Pastel1", ax=ax)

        if title in result_by_name:
            add_stats(ax, result_by_name[title]["p_val"], ALPHA / n_tests_run)
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig("merged_analysis_plots.png", dpi=150)
    print("Done! Plots saved to 'merged_analysis_plots.png'")

    return pd.DataFrame(results)


if __name__ == "__main__":
    main()
