import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
import joblib
import os
import re

# ==========================================
# 1. LOAD DATASETS
# ==========================================
files = {
    'Emlakjet': 'data/raw/emlakjet/emlakjet_listings.xlsx',
    'Sahibinden': 'data/raw/sahibinden/sahibinden_enriched_listings.xlsx',
    'Hepsiemlak': 'data/raw/hepsiemlak/hepsiemlak_listings.xlsx'
}

dfs = []
for source, filename in files.items():
    if os.path.exists(filename):
        try: 
            temp_df = pd.read_excel(filename)
            temp_df['Source'] = source
            dfs.append(temp_df)
        except:
            # Try CSV fallback
            try:
                temp_df = pd.read_csv(filename.replace('.xlsx', '.csv'))
                temp_df['Source'] = source
                dfs.append(temp_df)
            except: pass

if not dfs:
    print("❌ No data files found. Please make sure the Excel files are in the folder.")
    exit()

df = pd.concat(dfs, ignore_index=True)
print(f"✓ Loaded {len(df)} listings from {len(dfs)} sources.")

# ==========================================
# 2. ROBUST DATA CLEANING
# ==========================================
def clean_price(val):
    if pd.isna(val): return np.nan
    s = str(val).replace('TL', '').replace('.', '').strip()
    try: return float(s)
    except: return np.nan

def clean_area(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    # Handle Hepsiemlak dictionary format
    if s.startswith('{'):
        try:
            m = re.search(r"'netSqm':\s*(\d+)", s)
            if m: return float(m.group(1))
        except: pass
    s = s.replace('m²', '').replace('.', '').strip()
    try: return float(s)
    except: return np.nan

def clean_rooms(val):
    if pd.isna(val): return np.nan
    s = str(val).replace("'", "").replace("[", "").replace("]", "").lower().strip()
    if 'oda' in s: return 1.0
    if '+' in s:
        try:
            parts = s.split('+')
            return float(parts[0]) + float(parts[1])
        except: return np.nan
    try: return float(s)
    except: return np.nan

def clean_furnish(val):
    if pd.isna(val): return 'Unknown'
    s = str(val).lower()
    if s in ['true', 'eşyalı', 'evet', 'furnished']: return 'Furnished'
    if s in ['false', 'boş', 'hayır', 'unfurnished']: return 'Unfurnished'
    return 'Unknown'

# Apply Cleaning
df['clean_price'] = df['Price'].apply(clean_price)
df['clean_area'] = df['Area(m2)'].apply(clean_area)
df['clean_rooms'] = df['Rooms'].apply(clean_rooms)
df['clean_furnish'] = df['Furnishment'].apply(clean_furnish)
df['clean_bathrooms'] = pd.to_numeric(df['Bathrooms'], errors='coerce')
df['clean_metro_dist'] = pd.to_numeric(df['Distance to Metro (km)'], errors='coerce')

# Filter valid data for ML
ml_df = df.dropna(subset=['clean_price', 'clean_area', 'clean_rooms']).copy()

# ==========================================
# 3. MACHINE LEARNING PIPELINE
# ==========================================
# Features to use
features = ['clean_area', 'clean_rooms', 'clean_furnish', 'clean_bathrooms', 'clean_metro_dist', 'Source']
target = 'clean_price'

X = ml_df[features]
y = ml_df[target]

# Preprocessing: Handle missing values and text data automatically
numeric_features = ['clean_area', 'clean_rooms', 'clean_bathrooms', 'clean_metro_dist']
numeric_transformer = SimpleImputer(strategy='median')

categorical_features = ['clean_furnish', 'Source']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Define models to compare
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, random_state=42)
}

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

best_model = None
best_score = -np.inf
model_results = {}

print("\n🤖 Training and Comparing Models...")
for name, regressor in models.items():
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    model_results[name] = {
        'MAE': mae,
        'MSE': mse,
        'R2': r2,
        'MAPE': mape,
        'CV_R2_mean': cv_scores.mean(),
        'CV_R2_std': cv_scores.std(),
        'model': model,
        'y_pred': y_pred
    }
    
    print(f"{name}: MAE={mae:,.0f}, R²={r2:.3f}, MAPE={mape:.2f}%, CV R²={cv_scores.mean():.3f}±{cv_scores.std():.3f}")
    
    if r2 > best_score:
        best_score = r2
        best_model = model

print(f"\n🏆 Best Model: {max(model_results, key=lambda x: model_results[x]['R2'])}")

best_model_name = max(model_results, key=lambda x: model_results[x]['R2'])
best_y_pred = model_results[best_model_name]['y_pred']

print(f"\n📊 Sample Predictions for {best_model_name} (Actual vs Predicted):")
sample_size = min(10, len(y_test))
sample_df = pd.DataFrame({
    'Actual Price (TL)': y_test[:sample_size].values,
    'Predicted Price (TL)': best_y_pred[:sample_size],
    'Percentage Error (%)': np.abs((y_test[:sample_size].values - best_y_pred[:sample_size]) / y_test[:sample_size].values) * 100
})
print(sample_df.to_string(index=False, float_format='%.0f'))

# Use the best model for predictions
model = best_model

# Save the best model
joblib.dump(model, 'best_price_predictor.pkl')
print("\n💾 Model saved as 'best_price_predictor.pkl'")

# ==========================================
# 4. DEAL FINDER ALGORITHM
# ==========================================
# Predict "Fair Price" for ALL listings
ml_df['predicted_fair_price'] = model.predict(X)

# Calculate "Savings" (Predicted - Actual)
# Positive Savings = The house is cheaper than it "should" be (Good Deal)
ml_df['potential_savings'] = ml_df['predicted_fair_price'] - ml_df['clean_price']

print("\n💎 TOP 5 UNDERVALUED 'HIDDEN GEMS' (Best Deals):")
print("-" * 60)
best_deals = ml_df.sort_values(by='potential_savings', ascending=False).head(5)

for idx, row in best_deals.iterrows():
    print(f"Listing: {row['Listing URL']}")
    print(f"   Actual Price: {row['clean_price']:,.0f} TL")
    print(f"   Fair Price:   {row['predicted_fair_price']:,.0f} TL")
    print(f"   SAVINGS:      {row['potential_savings']:,.0f} TL (Undervalued)")
    print("-" * 60)

# ==========================================
# 5. VISUALIZATION
# ==========================================
plt.figure(figsize=(14, 6))

# Plot 1: Feature Importance
plt.subplot(1, 2, 1)
# Extract feature names
ohe = model.named_steps['preprocessor'].transformers_[1][1]['onehot']
feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))

regressor = model.named_steps['regressor']
if hasattr(regressor, 'feature_importances_'):
    importances = regressor.feature_importances_
    title = 'Feature Importances'
elif hasattr(regressor, 'coef_'):
    # For linear models, use absolute coefficients
    importances = np.abs(regressor.coef_)
    title = 'Feature Coefficients (Absolute)'
else:
    importances = np.ones(len(feature_names)) / len(feature_names)  # Equal importance
    title = 'Features'

# Sort
indices = np.argsort(importances)
plt.title(f'What drives the price in Kurtköy? ({title})')
plt.barh(range(len(indices)), importances[indices], color='#2ecc71', align='center')
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Importance')

# Plot 2: Actual vs Predicted
plt.subplot(1, 2, 2)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='#3498db')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted (Fair) Price')
plt.title('Accuracy of Predictions')

plt.tight_layout()
plt.savefig('ml_deal_finder_results.png')
print("\n📊 Graphs saved to 'ml_deal_finder_results.png'")
plt.show()

# ==========================================
# 6. MARKET SEGMENTATION (CLUSTERING)
# ==========================================
print("\n🏘️ Performing Market Segmentation...")

# Prepare data for clustering (use numeric features)
cluster_features = ['clean_area', 'clean_rooms', 'clean_bathrooms', 'clean_metro_dist', 'clean_price']
cluster_df = ml_df[cluster_features].dropna()

# Standardize
scaler = StandardScaler()
cluster_data = scaler.fit_transform(cluster_df)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(cluster_data)
cluster_df['Cluster'] = clusters

# Analyze clusters
cluster_summary = cluster_df.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

# Visualize clusters (using first 2 features for simplicity)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=cluster_df, x='clean_area', y='clean_price', hue='Cluster', palette='viridis', s=50)
plt.title('Market Segments in Kurtköy Real Estate')
plt.xlabel('Area (m²)')
plt.ylabel('Price (TL)')
plt.legend(title='Segment')
plt.savefig('market_segments.png')
print("📊 Market segments saved to 'market_segments.png'")
plt.show()

print("\n✅ ML Analysis Complete!")

# ==========================================
# 7. SAVE RESULTS TO EXCEL
# ==========================================
print("\n📊 Saving all results to Excel file...")

with pd.ExcelWriter('ml_analysis_results.xlsx') as writer:
    # Model Results
    results_df = pd.DataFrame({
        'Model': list(model_results.keys()),
        'MAE': [model_results[m]['MAE'] for m in model_results],
        'MSE': [model_results[m]['MSE'] for m in model_results],
        'R2': [model_results[m]['R2'] for m in model_results],
        'MAPE (%)': [model_results[m]['MAPE'] for m in model_results],
        'CV_R2_mean': [model_results[m]['CV_R2_mean'] for m in model_results],
        'CV_R2_std': [model_results[m]['CV_R2_std'] for m in model_results]
    })
    results_df.to_excel(writer, sheet_name='Model Results', index=False)
    
    # Sample Predictions
    sample_df.to_excel(writer, sheet_name='Sample Predictions', index=False)
    
    # Best Deals
    best_deals_df = best_deals[['Listing URL', 'clean_price', 'predicted_fair_price', 'potential_savings']].copy()
    best_deals_df.columns = ['Listing URL', 'Actual Price (TL)', 'Predicted Fair Price (TL)', 'Potential Savings (TL)']
    best_deals_df.to_excel(writer, sheet_name='Best Deals', index=False)
    
    # All Predictions
    all_predictions_df = ml_df[['Listing URL', 'clean_price', 'predicted_fair_price', 'potential_savings']].copy()
    all_predictions_df.columns = ['Listing URL', 'Actual Price (TL)', 'Predicted Fair Price (TL)', 'Potential Savings (TL)']
    all_predictions_df.to_excel(writer, sheet_name='All Predictions', index=False)

print("💾 Results saved to 'ml_analysis_results.xlsx'")    