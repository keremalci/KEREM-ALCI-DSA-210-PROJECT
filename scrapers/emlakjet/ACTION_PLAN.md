# Emlakjet Data Collection - Action Plan

## Current Status
- ❌ Old data (19 listings) has expired URLs
- ✅ 3 scrapers created and ready
- ✅ All dependencies installed
- 🎯 **Goal**: Collect 30-50 fresh Kurtköy rental listings

## 🚀 Recommended Workflow

### Option A: Semi-Automated (Easiest - 10 minutes)

**Step 1: Collect URLs with Helper Script**
```bash
cd scrapers/emlakjet
python collect_urls.py
```

What happens:
1. Browser opens automatically
2. You search for "Kurtköy Pendik kiralık daire" in the browser
3. Press Enter in terminal
4. Script extracts all URLs → saves to `emlakjet_urls.txt`

**Step 2: Scrape Listings**
```bash
python scraper_emlakjet_selenium.py
# When prompted, enter: emlakjet_urls.txt
```

**Step 3: Check Results**
- New file created: `../../data/raw/emlakjet/emlakjet_listings_TIMESTAMP.xlsx`
- Contains all listing details + distances

---

### Option B: Fully Manual (More Control - 15 minutes)

**Step 1: Collect URLs Manually**
1. Open https://www.emlakjet.com in your browser
2. Search: "Kurtköy Pendik kiralık daire"
3. For each listing:
   - Right-click → Copy link address
   - Paste into a text file (`emlakjet_urls.txt`)
4. Save file (one URL per line)

**Step 2: Scrape Listings**
```bash
cd scrapers/emlakjet
python scraper_emlakjet_selenium.py
# Provide path to your emlakjet_urls.txt
```

---

### Option C: Browser Console (Advanced - 5 minutes)

**Step 1: Use JavaScript Console**
1. Open Emlakjet search results in browser
2. Open DevTools (F12) → Console tab
3. Paste and run:
```javascript
let urls = [];
document.querySelectorAll('a[href*="/ilan/"]').forEach(a => {
    if (a.href && a.href.includes('/ilan/') && !urls.includes(a.href)) {
        urls.push(a.href);
    }
});
console.log(urls.join('\\n'));
```
4. Copy output → Save to `emlakjet_urls.txt`

**Step 2: Run Scraper**
```bash
python scraper_emlakjet_selenium.py
```

---

## 📋 What You'll Get

For each listing:
- ✅ Price (₺/month)
- ✅ Area (m²)
- ✅ Room count (2+1, 3+1, etc.)
- ✅ Building age
- ✅ Floor number
- ✅ Number of bathrooms
- ✅ Furnishment status
- ✅ Coordinates (latitude, longitude)
- ✅ Distance to Kurtköy Metro
- ✅ Distance to Sabancı University
- ✅ Distance to Bus Station

## 🎯 Success Metrics

**Target**: 30-50 fresh listings
- Minimum for good analysis: 20 listings
- Optimal for statistics: 40-50 listings
- Combined with Sahibinden & Hepsiemlak: 80-100 total

## ⚠️ Important Notes

1. **URL Quality**: Make sure URLs are actual listings (contain `/ilan/` and a listing ID)
2. **Active Listings**: Check that listings show prices (not "removed" or "sold")
3. **Location**: Verify listings are actually in Kurtköy area
4. **Delays**: Scraper has built-in 4-8 second delays between requests to be respectful

## 🔧 Troubleshooting

**Browser doesn't open?**
- Check Chrome is installed
- Try: `pip install --upgrade undetected-chromedriver`

**No URLs extracted?**
- Emlakjet may have changed their HTML structure
- Use manual method (Option B)
- Check browser console for JavaScript errors

**Scraper crashes mid-way?**
- Data is saved progressively (mostly)
- Check the output file for partial data
- Resume with remaining URLs

## 📊 After Collection

Once you have the new data:

```bash
# Go to analysis folder
cd ../../analysis

# Run analysis with new data
python analyze_data.py
```

The analysis script will automatically include the new Emlakjet data!

---

## ✅ Ready to Start?

**Recommended for you**: Use **Option A** (Semi-Automated)

Run this command:
```bash
cd /Users/mchalil/Documents/keremalc/KEREM-ALCI-DSA-210-PROJECT/scrapers/emlakjet
python collect_urls.py
```

Then follow the on-screen instructions! 🚀
    