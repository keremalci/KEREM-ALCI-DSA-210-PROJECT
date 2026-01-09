import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import joblib
import os

# ==========================================
# 1. LOAD DATASETS
# ==========================================
print("--- Loading Data ---")
# Mapped directories containing data
data_dirs = {
    'Emlakjet': 'data/raw/emlakjet',
    'Sahibinden': 'data/raw/sahibinden',
    'Hepsiemlak': 'data/raw/hepsiemlak'
}

dfs = []
for source, directory in data_dirs.items():
    if os.path.exists(directory):
        # Find all Excel files in the directory
        files_in_dir = [f for f in os.listdir(directory) if f.endswith('.xlsx') and not f.startswith('~$')]
        
        if not files_in_dir:
            print(f"Warning: No Excel files found in {directory}")
            continue
            
        for filename in files_in_dir:
            filepath = os.path.join(directory, filename)
            try: 
                print(f"Reading {filepath}...")
                temp_df = pd.read_excel(filepath)
                temp_df['Source'] = source
                temp_df['SourceFile'] = filename # Track origin file
                dfs.append(temp_df)
                print(f"  -> Successfully loaded {len(temp_df)} rows")
            except Exception as e:
                print(f"Failed to read {filepath}: {e}")
    else:
        print(f"Directory not found: {directory}")

if not dfs:
    raise FileNotFoundError("No data files found. Please check the 'data/raw' directory.")

df = pd.concat(dfs, ignore_index=True)
print(f"Total rows loaded: {len(df)}")

# ==========================================
# 2. DATA CLEANING & PREPROCESSING
# ==========================================
print("--- Cleaning Data ---")
def clean_price(price):
    if pd.isna(price): return np.nan
    if isinstance(price, (int, float)): return price
    price = str(price).replace('TL', '').replace('.', '').replace(',', '').strip()
    try:
        return float(price)
    except:
        return np.nan

def clean_numeric(value):
    if pd.isna(value): return np.nan
    if isinstance(value, (int, float)): return value
    value = str(value).replace('m²', '').replace('m2', '').replace('+1', '').strip()
    try:
        return float(value.split()[0])
    except:
        return np.nan

if 'Price' in df.columns:
    df['clean_price'] = df['Price'].apply(clean_price)
elif 'price' in df.columns:
    df['clean_price'] = df['price'].apply(clean_price)
else:
    raise ValueError("Could not find a 'Price' or 'price' column.")

df = df.dropna(subset=['clean_price'])

# Clean specific numeric features if they exist
for col in ['Area(m2)', 'Rooms', 'Building Age']:
    if col in df.columns:
        df[col] = df[col].apply(clean_numeric)

# Clean specific categorical features if they exist
for col in ['Furnishment', 'Listing Type']:
    if col in df.columns:
        df[col] = df[col].fillna('Unknown').astype(str)

# Select Features
potential_numeric = ['Area(m2)', 'Rooms', 'Building Age']
potential_categorical = ['Furnishment', 'Listing Type']

numeric_features = [col for col in potential_numeric if col in df.columns]
categorical_features = [col for col in potential_categorical if col in df.columns]

print(f"Using Numeric Features: {numeric_features}")
print(f"Using Categorical Features: {categorical_features}")

X = df[numeric_features + categorical_features]
y = df['clean_price']

# Pipeline Setup
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# ==========================================
# 3. UNSUPERVISED LEARNING (Clustering)
# ==========================================
print("--- Running Unsupervised Learning ---")
try:
    X_processed = preprocessor.fit_transform(X)

    # PCA (Dimensionality Reduction for Visualization)
    n_components = min(2, X_processed.shape[1])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_processed)

    if n_components >= 1: df['PCA1'] = X_pca[:, 0]
    if n_components >= 2: df['PCA2'] = X_pca[:, 1]
    else: df['PCA2'] = 0

    # K-Means (Market Segmentation)
    kmeans = KMeans(n_clusters=min(3, len(df)), random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_processed)
except Exception as e:
    print(f"Skipping Unsupervised Learning: {e}")

# ==========================================
# 4. SUPERVISED LEARNING (Training)
# ==========================================
print("--- Running Supervised Learning ---")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42, max_depth=10),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model_score = -np.inf
best_model_name = ""
best_model_obj = None

for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('model', model)])
    
    # Train
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) 
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred) if len(y_test) > 1 else 0.0

    results[name] = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}
    
    if r2 > best_model_score:
        best_model_score = r2
        best_model_name = name
        best_model_obj = clf

print(f"Best Model Selected: {best_model_name} (R2: {best_model_score:.4f})")
joblib.dump(best_model_obj, 'final_model.pkl')

# Generate Predictions for entire dataset to find deals
df['Predicted_Price'] = best_model_obj.predict(X)
df['Potential_Savings'] = df['Predicted_Price'] - df['clean_price']

# ==========================================
# 5. VISUALIZATION (Deal Finder Dashboard)
# ==========================================
print("--- Generating Deal Finder Results Image ---")
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Deal Finder Analysis Results (Best Model: {best_model_name})', fontsize=16)

# Plot 1: Model Comparison
names = list(results.keys())
values = [results[m]['R2'] for m in names]
colors = ['green' if n == best_model_name else 'gray' for n in names]
axes[0].bar(names, values, color=colors)
axes[0].set_title('Model Performance (R² Score)')
axes[0].set_ylim(0, 1.0)
axes[0].set_ylabel('R² Score (Higher is Better)')

# Plot 2: Clustering (Market Segments)
if 'PCA1' in df.columns and 'Cluster' in df.columns:
    sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=df, palette='viridis', ax=axes[1], s=100)
    axes[1].set_title('Market Segmentation (PCA + K-Means)')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')

# Plot 3: Actual vs Predicted (Deal Spotter)
# Deals are points ABOVE the identity line (Predicted > Actual)
sns.scatterplot(x='clean_price', y='Predicted_Price', data=df, ax=axes[2], alpha=0.6, label='Listings')
min_val = min(df['clean_price'].min(), df['Predicted_Price'].min())
max_val = max(df['clean_price'].max(), df['Predicted_Price'].max())
axes[2].plot([min_val, max_val], [min_val, max_val], 'r--', label='Fair Value (x=y)')

# Highlight Deals (Top 50 savings)
deals = df.nlargest(50, 'Potential_Savings')
axes[2].scatter(deals['clean_price'], deals['Predicted_Price'], color='green', s=50, label='Best Deals')

axes[2].set_title('Deal Finder: Actual vs. Predicted Price')
axes[2].set_xlabel('Actual Price (TL)')
axes[2].set_ylabel('Predicted Fair Price (TL)')
axes[2].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('deal_finder_results.png', dpi=300)
print("Saved visualization to 'deal_finder_results.png'")

# ==========================================
# 6. EFFICIENT EXCEL ORGANIZATION
# ==========================================
print("--- Saving Organized Excel File ---")
results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

# Sort deals by highest savings first
# We want Positive Savings (Predicted > Actual)
best_deals_df = df[df['Potential_Savings'] > 0].sort_values(by='Potential_Savings', ascending=False)

# Identify Link Column dynamically if possible, or use known common names
link_col = next((col for col in df.columns if 'url' in col.lower() or 'link' in col.lower()), None)
cols_to_keep = ['Source', 'clean_price', 'Predicted_Price', 'Potential_Savings', 'Cluster']
if link_col:
    cols_to_keep.append(link_col)

# Filter columns that actually exist in the dataframe
cols_to_keep = [c for c in cols_to_keep if c in df.columns]

# Get Top 10
top_10_deals = best_deals_df[cols_to_keep].head(10)

# 1. Main Analysis Results (Existing)
with pd.ExcelWriter('ml_analysis_results.xlsx') as writer:
    results_df.to_excel(writer, sheet_name='1_Model_Performance', index=False)
    best_deals_df[cols_to_keep].to_excel(writer, sheet_name='2_All_Deals', index=False) 
    
    cluster_cols = ['clean_price', 'Cluster', 'PCA1', 'PCA2']
    # Filter cluster cols that exist
    cluster_cols = [c for c in cluster_cols if c in df.columns]
    df[cluster_cols].to_excel(writer, sheet_name='3_Clustering_Analysis', index=False)

print("Saved organized results to 'ml_analysis_results.xlsx'")

# 2. Top 10 Deals Separate File (Requested)
top_10_deals.to_excel('top_10_deals.xlsx', index=False)
print("Saved Top 10 Best Deals to 'top_10_deals.xlsx'")