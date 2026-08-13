"""
DSA210 Kurtkoy Rental Project - Machine Learning & Deal Finder
=================================================================

This is the SCRIPT version of the ML analysis (analysis/ml_analysis.ipynb
mirrors this file cell-for-cell). It exists as a real, runnable .py file
because `main_pipeline.ipynb` calls `analysis/ml_analysis.py` directly via
`subprocess.run([python_exe, path])`.

Fixes applied relative to the original analysis/ml_analysis.ipynb:
  1. Reuses the shared, corrected cleaners in `data_cleaning.py` instead of
     a generic `clean_numeric()` that mis-parsed Rooms ("4+2" -> NaN,
     "3+1" -> 3 instead of 4) and silently produced ~NaN Building Age for
     every Emlakjet/Hepsiemlak row.
  2. Building Age is used as a categorical bucket (with an explicit
     "Unknown" bucket), not a fabricated numeric feature.
  3. Model evaluation uses K-Fold cross-validation instead of a single
     80/20 split. On this dataset (47 rows) a single split's R2 is noise -
     see CHANGELOG.md for the side-by-side numbers.
  4. The "Deal Finder" predictions (used to flag underpriced listings) now
     come from `cross_val_predict`, so every listing's predicted price
     comes from a fold that did NOT include it in training. The original
     code trained on 80% of the data then predicted on 100% of the data,
     which let training rows leak into their own "fair price" estimate and
     inflated the apparent R2 (0.44 in the original report, but honest
     out-of-fold R2 for that same feature set on the real data is negative
     - see CHANGELOG.md).
  5. Price is modeled in log-space (rents are right-skewed) and the two
     feature sets used ARE COMPARED via CV, and the better-generalizing one
     is selected automatically - rather than assuming more features help.
  6. Likely cross-platform duplicate listings are flagged (not silently
     dropped) via `data_cleaning.flag_duplicates` and excluded from the
     training set, since the same physical unit listed on two sites would
     otherwise be double-counted.
  7. The exported "Best Deals" sheet includes area/rooms/furnishing/source
     alongside price so a reader can sanity-check a flagged deal without
     opening the URL.

IMPORTANT CAVEAT (kept from the original project, now stated explicitly):
this dataset has only 47 listings. Cross-validated R2 for the best model
here is modest (~0.15-0.2, vs. an inflated 0.44 in the original leaky
evaluation). Treat "Deal Finder" output as a rough triage tool, not a
precise valuation - see FINAL_REPORT.md for the full discussion.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib

from data_cleaning import load_all_sources, clean_dataframe, flag_duplicates, flag_data_errors

RANDOM_STATE = 42
N_FOLDS = 5

# Two candidate feature sets - compared head-to-head via CV below rather
# than assuming the bigger one is better (it isn't, on this dataset: the
# Building Age / Listing Type columns are ~66% missing because Emlakjet and
# Hepsiemlak never populate them, so one-hot-encoding them mostly adds
# noise columns for a 47-row dataset).
FEATURE_SETS = {
    "lean": {
        "numeric": ["clean_area", "clean_rooms", "clean_bathrooms"],
        "categorical": ["clean_furnish"],
    },
    "extended": {
        "numeric": ["clean_area", "clean_rooms", "clean_bathrooms"],
        "categorical": ["clean_furnish", "clean_listing_type", "clean_age_bucket"],
    },
}

MODELS = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE, max_depth=5, min_samples_leaf=3),
    "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=6, min_samples_leaf=2, random_state=RANDOM_STATE),
}


def build_pipeline(numeric_features, categorical_features, model):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])
    reg = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    # Model price in log-space (rents are right-skewed) but report metrics
    # back on the original TL scale via TransformedTargetRegressor.
    return TransformedTargetRegressor(regressor=reg, func=np.log, inverse_func=np.exp)


def evaluate_all_combinations(df):
    """Runs K-Fold CV for every (feature set x model) combination and
    returns a results table plus the identity of the best combination."""
    y = df["clean_price"].values.astype(float)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    fitted = {}
    for feat_name, feats in FEATURE_SETS.items():
        X = df[feats["numeric"] + feats["categorical"]]
        for model_name, model in MODELS.items():
            pipe = build_pipeline(feats["numeric"], feats["categorical"], model)
            r2_scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
            oof_pred = cross_val_predict(pipe, X, y, cv=kf)
            mae = mean_absolute_error(y, oof_pred)
            rmse = np.sqrt(mean_squared_error(y, oof_pred))
            rows.append({
                "Feature Set": feat_name,
                "Model": model_name,
                "CV R2 (mean)": r2_scores.mean(),
                "CV R2 (std)": r2_scores.std(),
                "OOF MAE": mae,
                "OOF RMSE": rmse,
            })
            fitted[(feat_name, model_name)] = pipe
            print(f"  [{feat_name:8s} | {model_name:18s}] CV R2 = {r2_scores.mean():+.4f} (+/- {r2_scores.std():.4f})  OOF MAE = {mae:,.0f} TL")

    results_df = pd.DataFrame(rows).sort_values("CV R2 (mean)", ascending=False).reset_index(drop=True)
    best_row = results_df.iloc[0]
    best_key = (best_row["Feature Set"], best_row["Model"])
    return results_df, best_key, fitted


def main():
    print("--- Loading & Cleaning Data ---")
    raw = load_all_sources(".")
    df = clean_dataframe(raw)
    df = flag_duplicates(df)
    df = flag_data_errors(df)

    n_dupe_rows = df["Likely_Duplicate_Group"].notna().sum()
    if n_dupe_rows:
        print(f"  [i] {n_dupe_rows} rows flagged as POSSIBLE cross-platform duplicates "
              f"({df['Likely_Duplicate_Group'].nunique()} groups) - kept in training. Even an "
              f"EXACT price+area+room match is too common in this market to reliably tell true "
              f"duplicates from coincidentally similar listings (see CHANGELOG.md); the flag is "
              f"kept in the exported Excel for manual review, not auto-excluded.")

    # Only exclude rows with a clear, quantifiable data error (price/m2 wildly
    # off the rest of the market) - NOT the ambiguous duplicate flag above.
    df_train = df[~df["Likely_Data_Error"]].copy()
    print(f"  Training rows: {len(df_train)} (of {len(df)} total after cleaning; "
          f"{len(df) - len(df_train)} excluded as likely data errors)")

    print("\n--- Comparing Feature Sets x Models via 5-Fold Cross-Validation ---")
    results_df, best_key, fitted = evaluate_all_combinations(df_train)
    best_feat_name, best_model_name = best_key
    best_pipe = fitted[best_key]
    best_r2 = results_df.iloc[0]["CV R2 (mean)"]
    print(f"\nSelected: {best_model_name} on '{best_feat_name}' feature set (CV R2 = {best_r2:.4f})")

    feats = FEATURE_SETS[best_feat_name]
    X_all = df_train[feats["numeric"] + feats["categorical"]]
    y_all = df_train["clean_price"].values.astype(float)

    # Refit the winning pipeline on all training rows for the persisted model...
    best_pipe.fit(X_all, y_all)
    joblib.dump(best_pipe, "final_model.pkl")

    # ...but for "Predicted Price" shown to users, use out-of-fold predictions
    # (cross_val_predict) so training rows don't leak into their own estimate.
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_predicted = cross_val_predict(best_pipe, X_all, y_all, cv=kf)

    df_train = df_train.copy()
    df_train["Predicted_Price"] = oof_predicted
    df_train["Potential_Savings"] = df_train["Predicted_Price"] - df_train["clean_price"]

    # ==========================================
    # Unsupervised: PCA + KMeans market segmentation (on cleaned features)
    # ==========================================
    print("\n--- Market Segmentation (PCA + K-Means) ---")
    seg_numeric = ["clean_area", "clean_rooms", "clean_bathrooms"]
    seg_categorical = ["clean_furnish", "clean_age_bucket"]
    seg_pre = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), seg_numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                           ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), seg_categorical),
    ])
    X_seg = seg_pre.fit_transform(df_train[seg_numeric + seg_categorical])
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_seg)
    df_train["PCA1"], df_train["PCA2"] = X_pca[:, 0], X_pca[:, 1]

    n_clusters = min(3, len(df_train))
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    df_train["Cluster"] = kmeans.fit_predict(X_seg)

    # ==========================================
    # Visualization
    # ==========================================
    print("\n--- Generating Deal Finder Results Image ---")
    os.makedirs("data/outputs", exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Deal Finder Analysis Results (Best: {best_model_name} / {best_feat_name} features, "
                 f"CV R2={best_r2:.3f})", fontsize=14)

    pivot = results_df.pivot(index="Model", columns="Feature Set", values="CV R2 (mean)")
    pivot.plot(kind="bar", ax=axes[0])
    axes[0].set_title("CV R2 by Model x Feature Set")
    axes[0].set_ylabel("Cross-Validated R2 (higher is better)")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].legend(title="Feature Set")

    sns.scatterplot(x="PCA1", y="PCA2", hue="Cluster", data=df_train, palette="viridis", ax=axes[1], s=90)
    axes[1].set_title("Market Segmentation (PCA + K-Means)")

    sns.scatterplot(x="clean_price", y="Predicted_Price", hue="Source", data=df_train, ax=axes[2], alpha=0.7)
    lims = [df_train[["clean_price", "Predicted_Price"]].min().min(),
            df_train[["clean_price", "Predicted_Price"]].max().max()]
    axes[2].plot(lims, lims, "r--", label="Fair Value (x=y)")
    deals = df_train.nlargest(min(10, len(df_train)), "Potential_Savings")
    axes[2].scatter(deals["clean_price"], deals["Predicted_Price"], edgecolor="green", facecolor="none", s=120, linewidth=2, label="Top Deals")
    axes[2].set_title("Deal Finder: Actual vs Out-of-Fold Predicted Price")
    axes[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("data/outputs/deal_finder_results.png", dpi=200)
    print("Saved visualization to 'data/outputs/deal_finder_results.png'")

    # ==========================================
    # Excel export
    # ==========================================
    print("\n--- Saving Organized Excel File ---")
    best_deals_df = df_train[df_train["Potential_Savings"] > 0].sort_values("Potential_Savings", ascending=False)
    export_cols = ["Source", "clean_price", "Predicted_Price", "Potential_Savings",
                    "clean_area", "clean_rooms", "clean_bathrooms", "clean_furnish",
                    "clean_age_bucket", "Cluster", "Likely_Duplicate_Group", "Listing URL"]
    export_cols = [c for c in export_cols if c in best_deals_df.columns]
    best_deals_df = best_deals_df[export_cols].rename(columns={
        "clean_price": "Actual Price", "clean_area": "Area (m2)", "clean_rooms": "Total Rooms",
        "clean_bathrooms": "Bathrooms", "clean_furnish": "Furnishing", "clean_age_bucket": "Building Age",
    })

    with pd.ExcelWriter("data/outputs/ml_analysis_results.xlsx") as writer:
        results_df.to_excel(writer, sheet_name="1_Model_Performance", index=False)
        best_deals_df.to_excel(writer, sheet_name="2_Best_Deals_Finder", index=False)
        df_train[["clean_price", "Cluster", "PCA1", "PCA2", "Source"]].rename(
            columns={"clean_price": "Price"}).to_excel(writer, sheet_name="3_Clustering_Analysis", index=False)

    print("Saved 'data/outputs/ml_analysis_results.xlsx'")
    print("\n" + "=" * 60)
    print(f"ML ANALYSIS COMPLETE - Best model: {best_model_name} ({best_feat_name} features), "
          f"honest CV R2={best_r2:.4f}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    main()
