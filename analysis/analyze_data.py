"""
Rental Price Analysis for Kurtköy Area
DSA-210 Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from pathlib import Path

# Determine if we're running from analysis/ folder or project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == 'analysis' else SCRIPT_DIR

# Set up paths
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
VIZ_DIR = PROJECT_ROOT / 'visualizations'

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def clean_price(price):
    """Clean price strings and convert to float"""
    if pd.isna(price):
        return np.nan
    if isinstance(price, (int, float)):
        return float(price)
    # Remove 'TL', spaces, and dots (thousands separator in Turkish format)
    price_str = str(price).replace('TL', '').replace('.', '').replace(',', '.').strip()
    try:
        return float(price_str)
    except:
        return np.nan

def clean_area(area):
    """Clean area values"""
    if pd.isna(area):
        return np.nan
    if isinstance(area, (int, float)):
        return float(area)
    try:
        return float(str(area).replace('m²', '').replace(',', '.').strip())
    except:
        return np.nan

def standardize_rooms(room_value):
    """Standardize room values"""
    if pd.isna(room_value):
        return np.nan
    room_str = str(room_value).strip("[]'\"")
    # Extract number before '+' if exists
    if '+' in room_str:
        return room_str
    return room_str

def clean_building_age(age):
    """Convert building age to numeric (midpoint of range or actual number)"""
    if pd.isna(age):
        return np.nan
    age_str = str(age).lower().strip()
    
    if '0-5' in age_str or '0-1' in age_str or '1-5' in age_str:
        return 3
    elif '6-10' in age_str or '5-10' in age_str:
        return 8
    elif '11-15' in age_str or '10-15' in age_str:
        return 13
    elif '16-20' in age_str or '15-20' in age_str:
        return 18
    elif '21-25' in age_str or '20-25' in age_str:
        return 23
    elif '26-30' in age_str or '25-30' in age_str:
        return 28
    elif '30+' in age_str or '31 ve üzeri' in age_str:
        return 35
    
    try:
        return float(age_str)
    except:
        return np.nan

def clean_bathrooms(bathrooms):
    """Clean bathroom values"""
    if pd.isna(bathrooms):
        return np.nan
    bath_str = str(bathrooms).lower().strip()
    
    if 'yok' in bath_str or 'none' in bath_str:
        return 0
    
    try:
        return float(bath_str)
    except:
        return np.nan

def load_all_data():
    """Load and combine data from all three sources"""
    print("Loading data from all sources...")
    
    # Load individual datasets
    df_sahibinden = pd.read_excel(DATA_RAW / 'sahibinden' / 'sahibinden_enriched_listings.xlsx')
    df_sahibinden['Source'] = 'Sahibinden'
    
    df_hepsiemlak = pd.read_excel(DATA_RAW / 'hepsiemlak' / 'hepsiemlak_listings.xlsx')
    df_hepsiemlak['Source'] = 'Hepsiemlak'
    
    df_emlakjet = pd.read_excel(DATA_RAW / 'emlakjet' / 'emlakjet_listings.xlsx')
    df_emlakjet['Source'] = 'Emlakjet'
    
    # Combine all datasets
    df_combined = pd.concat([df_sahibinden, df_hepsiemlak, df_emlakjet], ignore_index=True)
    
    print(f"\nTotal listings collected: {len(df_combined)}")
    print(f"  - Sahibinden: {len(df_sahibinden)}")
    print(f"  - Hepsiemlak: {len(df_hepsiemlak)}")
    print(f"  - Emlakjet: {len(df_emlakjet)}")
    
    # Clean data
    print("\nCleaning data...")
    df_combined['Price'] = df_combined['Price'].apply(clean_price)
    df_combined['Area(m2)'] = df_combined['Area(m2)'].apply(clean_area)
    df_combined['Rooms'] = df_combined['Rooms'].apply(standardize_rooms)
    df_combined['Building Age'] = df_combined['Building Age'].apply(clean_building_age)
    if 'Bathrooms' in df_combined.columns:
        df_combined['Bathrooms'] = df_combined['Bathrooms'].apply(clean_bathrooms)
    
    # Remove rows with missing critical data
    original_count = len(df_combined)
    df_combined = df_combined.dropna(subset=['Price', 'Area(m2)'])
    removed_count = original_count - len(df_combined)
    if removed_count > 0:
        print(f"Removed {removed_count} listings with missing price or area data")
    
    return df_combined

def basic_statistics(df):
    """Display basic statistics about the dataset"""
    print("\n" + "="*60)
    print("BASIC STATISTICS")
    print("="*60)
    
    print("\n--- Price Statistics ---")
    print(df['Price'].describe())
    
    print("\n--- Area Statistics (m²) ---")
    print(df['Area(m2)'].describe())
    
    print("\n--- Room Distribution ---")
    print(df['Rooms'].value_counts().sort_index())
    
    print("\n--- Building Age Statistics ---")
    print(df['Building Age'].describe())
    
    print("\n--- Listing Type Distribution ---")
    print(df['Listing Type'].value_counts())
    
    print("\n--- Furnishment Status ---")
    print(df['Furnishment'].value_counts())
    
    print("\n--- Distance to Metro (km) ---")
    print(df['Distance to Metro (km)'].describe())
    
    print("\n--- Distance to University (km) ---")
    print(df['Distance to University (km)'].describe())

def correlation_analysis(df):
    """Analyze correlations between features and price"""
    print("\n" + "="*60)
    print("CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numeric columns for correlation
    numeric_cols = ['Price', 'Area(m2)', 'Building Age', 'Bathrooms', 
                   'Distance to Metro (km)', 'Distance to University (km)', 
                   'Distance to Bus Station (km)']
    
    # Filter only existing columns
    available_cols = [col for col in numeric_cols if col in df.columns]
    
    corr_with_price = df[available_cols].corr()['Price'].sort_values(ascending=False)
    print("\nCorrelation with Price:")
    print(corr_with_price)
    
    return corr_with_price

def price_per_sqm_analysis(df):
    """Calculate and analyze price per square meter"""
    print("\n" + "="*60)
    print("PRICE PER SQUARE METER ANALYSIS")
    print("="*60)
    
    df['Price per m²'] = df['Price'] / df['Area(m2)']
    
    print(f"\nAverage Price per m²: ₺{df['Price per m²'].mean():.2f}")
    print(f"Median Price per m²: ₺{df['Price per m²'].median():.2f}")
    print(f"Min Price per m²: ₺{df['Price per m²'].min():.2f}")
    print(f"Max Price per m²: ₺{df['Price per m²'].max():.2f}")
    
    print("\n--- Price per m² by Source ---")
    print(df.groupby('Source')['Price per m²'].mean().sort_values(ascending=False))
    
    return df

def create_visualizations(df):
    """Create visualizations for the analysis"""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Price Distribution
    axes[0, 0].hist(df['Price'], bins=20, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('Price Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Price (₺)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['Price'].mean(), color='red', linestyle='--', label=f'Mean: ₺{df["Price"].mean():.0f}')
    axes[0, 0].legend()
    
    # 2. Price vs Area
    axes[0, 1].scatter(df['Area(m2)'], df['Price'], alpha=0.6, c=df['Price'], cmap='viridis')
    axes[0, 1].set_title('Price vs Area', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Area (m²)')
    axes[0, 1].set_ylabel('Price (₺)')
    
    # Add trend line
    z = np.polyfit(df['Area(m2)'].dropna(), df['Price'].dropna(), 1)
    p = np.poly1d(z)
    axes[0, 1].plot(df['Area(m2)'], p(df['Area(m2)']), "r--", alpha=0.8, label='Trend')
    axes[0, 1].legend()
    
    # 3. Price by Room Count
    room_prices = df.groupby('Rooms')['Price'].mean().sort_index()
    axes[1, 0].bar(room_prices.index.astype(str), room_prices.values, color='coral', edgecolor='black')
    axes[1, 0].set_title('Average Price by Room Count', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Rooms')
    axes[1, 0].set_ylabel('Average Price (₺)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 4. Price by Source
    source_prices = df.groupby('Source')['Price'].mean().sort_values(ascending=False)
    axes[1, 1].bar(source_prices.index, source_prices.values, color='lightgreen', edgecolor='black')
    axes[1, 1].set_title('Average Price by Source', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Source')
    axes[1, 1].set_ylabel('Average Price (₺)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'analysis_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'analysis_overview.png'}")
    
    # Create additional visualization for distance analysis
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    
    # Distance correlations
    if 'Distance to Metro (km)' in df.columns:
        axes2[0].scatter(df['Distance to Metro (km)'], df['Price'], alpha=0.6, color='blue')
        axes2[0].set_title('Price vs Distance to Metro', fontsize=12, fontweight='bold')
        axes2[0].set_xlabel('Distance to Metro (km)')
        axes2[0].set_ylabel('Price (₺)')
    
    if 'Distance to University (km)' in df.columns:
        axes2[1].scatter(df['Distance to University (km)'], df['Price'], alpha=0.6, color='green')
        axes2[1].set_title('Price vs Distance to University', fontsize=12, fontweight='bold')
        axes2[1].set_xlabel('Distance to University (km)')
        axes2[1].set_ylabel('Price (₺)')
    
    if 'Building Age' in df.columns:
        axes2[2].scatter(df['Building Age'], df['Price'], alpha=0.6, color='orange')
        axes2[2].set_title('Price vs Building Age', fontsize=12, fontweight='bold')
        axes2[2].set_xlabel('Building Age (years)')
        axes2[2].set_ylabel('Price (₺)')
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'distance_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {VIZ_DIR / 'distance_analysis.png'}")
    
    print("\nVisualization files created successfully!")

def main():
    """Main analysis function"""
    print("="*60)
    print("KURTKÖY RENTAL PRICE ANALYSIS")
    print("DSA-210 Project")
    print("="*60)
    
    # Load data
    df = load_all_data()
    
    # Basic statistics
    basic_statistics(df)
    
    # Correlation analysis
    correlation_analysis(df)
    
    # Price per square meter
    df = price_per_sqm_analysis(df)
    
    # Create visualizations
    create_visualizations(df)
    
    # Save combined dataset
    output_file = DATA_PROCESSED / 'combined_rental_data.xlsx'
    df.to_excel(output_file, index=False)
    print(f"\n✓ Combined dataset saved to: {output_file}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
    