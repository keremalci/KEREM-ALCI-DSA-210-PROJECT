import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import ast
import re

# ==========================================
# 1. LOAD DATA (3 SOURCES)
# ==========================================
file_emlakjet = 'emlakjet_listings.xlsx'
file_sahibinden = 'sahibinden_enriched_listings.xlsx'
file_hepsiemlak = 'hepsiemlak_listings.xlsx'

def load_file(filename):
    if os.path.exists(filename):
        try: return pd.read_excel(filename)
        except: 
            try: return pd.read_csv(filename.replace('.xlsx', '.csv'))
            except: pass
    return None

print("Loading datasets...")
df1 = load_file(file_emlakjet)
df2 = load_file(file_sahibinden)
df3 = load_file(file_hepsiemlak)

dfs = []
if df1 is not None:
    df1['Source'] = 'Emlakjet'
    dfs.append(df1)
if df2 is not None:
    df2['Source'] = 'Sahibinden'
    dfs.append(df2)
if df3 is not None:
    df3['Source'] = 'Hepsiemlak'
    dfs.append(df3)

if len(dfs) > 0:
    df = pd.concat(dfs, ignore_index=True)
    print(f"✓ Merged {len(dfs)} sources. Total listings: {len(df)}")
else:
    print("✗ No data files found. Please upload 'emlakjet_listings.xlsx', 'sahibinden_enriched_listings.xlsx', and 'hepsiemlak_listings.xlsx'.")
    exit()

# ==========================================
# 2. DATA CLEANING
# ==========================================

def clean_price(val):
    if pd.isna(val): return np.nan
    s = str(val).replace('TL', '').replace('.', '').strip()
    try: return float(s)
    except: return np.nan

def clean_area(val):
    if pd.isna(val): return np.nan
    s = str(val).strip()
    
    # Handle Hepsiemlak format: "{'netSqm': 80, ...}"
    if s.startswith('{') and 'Sqm' in s:
        try:
            # Try to extract grossSqm first, then netSqm
            # Regex is safer than eval for potentially malformed strings
            gross_match = re.search(r"'grossSqm':\s*\[(\d+)\]", s)
            if gross_match:
                return float(gross_match.group(1))
            
            net_match = re.search(r"'netSqm':\s*(\d+)", s)
            if net_match:
                return float(net_match.group(1))
        except:
            pass
            
    # Handle standard format: "135 m²"
    s = s.replace('m²', '').replace('.', '').strip()
    try: return float(s)
    except: return np.nan

def clean_rooms(val):
    if pd.isna(val): return np.nan
    s = str(val).replace("'", "").replace("[", "").replace("]", "").lower().strip()
    
    # Handle '1 oda'
    if 'oda' in s: return 1.0
    
    # Handle '3+1'
    if '+' in s:
        try:
            parts = s.split('+')
            return float(parts[0]) + float(parts[1])
        except: return np.nan
        
    try: return float(s)
    except: return np.nan

def clean_furnish(val):
    if pd.isna(val): return 'Unknown'
    
    # Handle Booleans (Hepsiemlak)
    if val is True: return 'Furnished'
    if val is False: return 'Unfurnished'
    
    s = str(val).lower()
    if s == 'true': return 'Furnished'
    if s == 'false': return 'Unfurnished'
    
    # Handle Text
    if any(x in s for x in ['boş', 'hayır']): return 'Unfurnished'
    if any(x in s for x in ['eşyalı', 'evet']): return 'Furnished'
    return 'Unknown'

def clean_listing_type(val):
    if pd.isna(val): return 'Unknown'
    s = str(val).lower()
    if 'ofis' in s or 'agent' in s: return 'Agent'
    if 'sahibinden' in s or 'owner' in s: return 'Owner'
    return 'Unknown'

def clean_bathrooms(val):
    if pd.isna(val): return np.nan
    try: return float(val)
    except: return np.nan

# Apply Cleaning
print("Cleaning data...")
df['cleaned_price'] = df['Price'].apply(clean_price)
df['cleaned_area'] = df['Area(m2)'].apply(clean_area)
df['cleaned_furnish'] = df['Furnishment'].apply(clean_furnish)
df['cleaned_type'] = df['Listing Type'].apply(clean_listing_type)
df['cleaned_bathrooms'] = df['Bathrooms'].apply(clean_bathrooms)
df['cleaned_age'] = df['Building Age'].fillna('Unknown')
df['total_rooms'] = df['Rooms'].apply(clean_rooms) # Use consistent naming for rooms

# ==========================================
# 3. HYPOTHESIS TESTING REPORT
# ==========================================
alpha = 0.05

def run_test(test_name, p_val, stat_name, stat_val, null_hyp):
    print(f"\n🔹 TEST: {test_name}")
    print(f"   H0: {null_hyp}")
    print(f"   {stat_name}: {stat_val:.4f}")
    print(f"   p-value: {p_val:.4f}")
    if p_val < alpha:
        print("   ✅ RESULT: REJECT H0 (Significant Relationship)")
    else:
        print("   ❌ RESULT: FAIL TO REJECT H0 (No Significant Relationship)")

print("\n" + "="*60)
print("📊 HYPOTHESIS TESTING REPORT (Emlakjet + Sahibinden + Hepsiemlak)")
print("="*60)

# 1. Price vs Area
sub = df.dropna(subset=['cleaned_area', 'cleaned_price'])
if len(sub) > 2:
    corr, p = stats.pearsonr(sub['cleaned_area'], sub['cleaned_price'])
    run_test("Price vs Area", p, "Pearson r", corr, "Area does not affect Price")
else: print("\n🔹 Test 1: Not enough data for Area.")

# 2. Price vs Age
sub = df[df['cleaned_age'] != 'Unknown'].dropna(subset=['cleaned_price'])
groups = [g['cleaned_price'].values for n, g in sub.groupby('cleaned_age') if len(g) >= 2]
if len(groups) >= 2:
    f_stat, p = stats.f_oneway(*groups)
    run_test("Price vs Building Age", p, "ANOVA F", f_stat, "Age groups have same Price")
else: print("\n🔹 Test 2: Not enough data for Building Age.")

# 3. Price vs Listing Type
agent = df[df['cleaned_type'] == 'Agent']['cleaned_price'].dropna()
owner = df[df['cleaned_type'] == 'Owner']['cleaned_price'].dropna()
if len(agent) > 1 and len(owner) > 1:
    t_stat, p = stats.ttest_ind(agent, owner, equal_var=False)
    run_test("Price vs Listing Type", p, "T-stat", t_stat, "Agent/Owner prices are same")
else: print(f"\n🔹 Test 3: Not enough data for Type (Agent={len(agent)}, Owner={len(owner)})")

# 4. Price vs Bathrooms
sub = df.dropna(subset=['cleaned_bathrooms', 'cleaned_price'])
groups = [g['cleaned_price'].values for n, g in sub.groupby('cleaned_bathrooms') if len(g) >= 2]
if len(groups) >= 2:
    f_stat, p = stats.f_oneway(*groups)
    run_test("Price vs Bathrooms", p, "ANOVA F", f_stat, "Bathrooms count doesn't affect Price")
else: print("\n🔹 Test 4: Not enough data for Bathrooms.")

# 5. Price vs Furnishment
furn = df[df['cleaned_furnish'] == 'Furnished']['cleaned_price'].dropna()
unfurn = df[df['cleaned_furnish'] == 'Unfurnished']['cleaned_price'].dropna()
if len(furn) > 1 and len(unfurn) > 1:
    t_stat, p = stats.ttest_ind(furn, unfurn, equal_var=False)
    run_test("Price vs Furnishment", p, "T-stat", t_stat, "Furnished/Unfurnished prices are same")
else: print(f"\n🔹 Test 5: Not enough data for Furnish (Furn={len(furn)}, Unfurn={len(unfurn)})")

print("="*60 + "\n")

# ==========================================
# 4. VISUALIZATION
# ==========================================
print("Generating Plots...")
plt.figure(figsize=(20, 15))
plt.subplots_adjust(hspace=0.4, wspace=0.3)
sns.set_style("whitegrid")

# Helper to add stats text
def add_stats(ax, p_val):
    res = "Reject H0" if p_val < alpha else "Fail to Reject H0"
    ax.text(0.05, 0.95, f"p={p_val:.4f}\n{res}", transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Plot 1: Area
ax1 = plt.subplot(2, 3, 1)
sub = df.dropna(subset=['cleaned_area', 'cleaned_price'])
sns.scatterplot(data=sub, x='cleaned_area', y='cleaned_price', hue='Source', alpha=0.7, ax=ax1)
if len(sub) > 2:
    sns.regplot(data=sub, x='cleaned_area', y='cleaned_price', scatter=False, color='red', ax=ax1)
    corr, p = stats.pearsonr(sub['cleaned_area'], sub['cleaned_price'])
    add_stats(ax1, p)
ax1.set_title('Price vs Area')

# Plot 2: Age
ax2 = plt.subplot(2, 3, 2)
sub = df[df['cleaned_age'] != 'Unknown'].dropna(subset=['cleaned_price'])
if not sub.empty:
    order = sorted(sub['cleaned_age'].unique())
    sns.boxplot(data=sub, x='cleaned_age', y='cleaned_price', order=order, palette="Blues", ax=ax2)
    # Re-calc p for plot annotation
    groups = [g['cleaned_price'].values for n, g in sub.groupby('cleaned_age') if len(g) >= 2]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        add_stats(ax2, p)
    ax2.set_title('Price vs Age')
    ax2.tick_params(axis='x', rotation=45)

# Plot 3: Type
ax3 = plt.subplot(2, 3, 3)
sub = df[df['cleaned_type'] != 'Unknown'].dropna(subset=['cleaned_price'])
if not sub.empty:
    sns.boxplot(data=sub, x='cleaned_type', y='cleaned_price', palette="Set2", ax=ax3)
    agent = sub[sub['cleaned_type'] == 'Agent']['cleaned_price']
    owner = sub[sub['cleaned_type'] == 'Owner']['cleaned_price']
    if len(agent) > 1 and len(owner) > 1:
        t, p = stats.ttest_ind(agent, owner, equal_var=False)
        add_stats(ax3, p)
    ax3.set_title('Price vs Type')

# Plot 4: Bathrooms
ax4 = plt.subplot(2, 3, 4)
sub = df.dropna(subset=['cleaned_bathrooms', 'cleaned_price'])
if not sub.empty:
    sns.boxplot(data=sub, x='cleaned_bathrooms', y='cleaned_price', palette="Purples", ax=ax4)
    groups = [g['cleaned_price'].values for n, g in sub.groupby('cleaned_bathrooms') if len(g) >= 2]
    if len(groups) >= 2:
        f, p = stats.f_oneway(*groups)
        add_stats(ax4, p)
    ax4.set_title('Price vs Bathrooms')

# Plot 5: Furnishment
ax5 = plt.subplot(2, 3, 5)
sub = df[df['cleaned_furnish'] != 'Unknown'].dropna(subset=['cleaned_price'])
if not sub.empty:
    sns.boxplot(data=sub, x='cleaned_furnish', y='cleaned_price', palette="Pastel1", ax=ax5)
    furn = sub[sub['cleaned_furnish'] == 'Furnished']['cleaned_price']
    unfurn = sub[sub['cleaned_furnish'] == 'Unfurnished']['cleaned_price']
    if len(furn) > 1 and len(unfurn) > 1:
        t, p = stats.ttest_ind(furn, unfurn, equal_var=False)
        add_stats(ax5, p)
    ax5.set_title('Price vs Furnishment')

plt.tight_layout()
plt.savefig('merged_analysis_plots.png')
print("Done! Plots saved to 'merged_analysis_plots.png'")
plt.show()    