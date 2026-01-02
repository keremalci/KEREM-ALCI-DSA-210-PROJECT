"""
Emlakjet URL Collector Helper
Opens browser for manual URL collection from Emlakjet search results
"""

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

def collect_urls_interactive():
    """
    Opens browser and helps collect URLs interactively
    """
    print("="*60)
    print("EMLAKJET URL COLLECTOR")
    print("="*60)
    
    # Setup browser
    print("\n1. Setting up browser...")
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = uc.Chrome(options=options)
    
    try:
        # Open Emlakjet
        print("2. Opening Emlakjet...")
        driver.get("https://www.emlakjet.com")
        time.sleep(3)
        
        print("\n" + "="*60)
        print("INSTRUCTIONS:")
        print("="*60)
        print("1. In the browser window that just opened:")
        print("   - Search for: 'Kurtköy Pendik kiralık daire'")
        print("   - Or navigate to the search results manually")
        print("2. Scroll through the listings")
        print("3. When you're ready, come back here and press Enter")
        print("4. The script will extract all listing URLs from the page")
        print("="*60)
        
        input("\nPress Enter when you're on the search results page...")
        
        # Extract URLs from current page
        print("\n3. Extracting URLs from current page...")
        
        urls = set()
        
        # Try multiple selectors
        selectors = [
            "a[href*='/ilan/']",
            "a[href*='emlakjet.com/ilan/']",
            ".listing-item a",
            "div[class*='listing'] a"
        ]
        
        for selector in selectors:
            try:
                links = driver.find_elements(By.CSS_SELECTOR, selector)
                for link in links:
                    href = link.get_attribute('href')
                    if href and '/ilan/' in href and 'emlakjet.com' in href:
                        # Only include if it looks like a valid listing
                        if len(href.split('/ilan/')[-1]) > 10:  # Has listing ID
                            urls.add(href)
            except:
                continue
        
        urls = sorted(list(urls))
        
        if urls:
            print(f"\n✓ Found {len(urls)} listing URLs!")
            print("\nFirst 5 URLs:")
            for url in urls[:5]:
                print(f"  {url}")
            
            # Save to file
            filename = "emlakjet_urls.txt"
            with open(filename, 'w') as f:
                f.write('\n'.join(urls))
            
            print(f"\n✓ Saved all URLs to: {filename}")
            print(f"\nNext steps:")
            print(f"1. Review {filename} to ensure URLs are correct")
            print(f"2. Run: python scraper_emlakjet_selenium.py")
            print(f"3. Provide the path to {filename}")
            
        else:
            print("\n⚠️ No URLs found automatically.")
            print("\nYou can:")
            print("1. Manually copy URLs from browser to a text file")
            print("2. Try scrolling more on the page and run this again")
            print("3. Use browser console to extract URLs (see README_SCRAPING.md)")
        
        print("\nBrowser will stay open for 30 seconds so you can copy more URLs manually if needed...")
        time.sleep(30)
        
    finally:
        driver.quit()
        print("\n✓ Browser closed")

if __name__ == "__main__":
    collect_urls_interactive()
    