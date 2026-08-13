"""
Sahibinden.com URL Discovery Helper - Kurtkoy Rentals
=======================================================

Companion tool to scraper_sahibinden.py. That script only VISITS listing
URLs you already have in `sahibinden_urls.txt` - it can't find new ones on
its own. This script automates that step: it browses Sahibinden's Kurtkoy
rental search results (same undetected-chromedriver approach as the main
scraper, since Sahibinden blocks plain HTTP requests the same way),
collects listing URLs, and appends any new ones to `sahibinden_urls.txt`
in the project root (deduplicated against what's already there).

Run this FIRST, then run scraper_sahibinden.py as usual.

NOTE: I could not test this against the live site from here (Sahibinden
blocks non-browser requests, same reason the main scraper needs a real
browser). The listing-URL pattern (`/ilan/...`) is stable and shared with
the main scraper's own URL filter, but the pagination ("next page") link
detection is a best-effort guess at Sahibinden's current markup and may
need a small tweak after your first run - see the comment in
`go_to_next_page()` if it stops after page 1 unexpectedly.

Usage:
    Run directly, from any directory: python find_sahibinden_urls.py
"""

import re
import time
import random
from pathlib import Path
from urllib.parse import urljoin

import undetected_chromedriver as uc
from bs4 import BeautifulSoup

# ==========================================
# Configuration
# ==========================================
SEARCH_URL = "https://www.sahibinden.com/kiralik-daire/istanbul-pendik-yenisehir-kurtkoy-mh."

# Anchored to this script's own location, not the current working directory
# (see scraper_sahibinden.py for why this matters now that these are plain
# .py files instead of Jupyter notebooks).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
URL_FILE = PROJECT_ROOT / "sahibinden_urls.txt"  # same file scraper_sahibinden.py reads

MIN_DELAY = 8              # seconds between page loads (search results are lighter than listing pages)
MAX_DELAY = 18
PAGE_LOAD_WAIT = 6
MAX_PAGES = 5               # safety cap on how many search-result pages to walk per run
MAX_NEW_URLS = 60           # safety cap on how many new URLs to add per run

LISTING_HREF_RE = re.compile(r"^/ilan/[\w-]+")


def setup_driver():
    print("Initializing browser session...")
    options = uc.ChromeOptions()
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--no-sandbox')
    options.add_argument(
        '--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
        'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
    driver = uc.Chrome(options=options, version_main=None)
    driver.maximize_window()
    return driver


def human_like_scroll(driver):
    scroll_pause = random.uniform(0.5, 1.5)
    scroll_height = random.randint(300, 600)
    driver.execute_script(f"window.scrollBy(0, {scroll_height});")
    time.sleep(scroll_pause)
    if random.random() > 0.7:
        driver.execute_script(f"window.scrollBy(0, -{random.randint(50, 150)});")
        time.sleep(random.uniform(0.3, 0.8))


def extract_listing_urls(driver):
    """Parse the current page's rendered HTML for listing links."""
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    found = set()
    for a in soup.find_all('a', href=True):
        href = a['href']
        if LISTING_HREF_RE.match(href):
            found.add(urljoin("https://www.sahibinden.com", href))
    return found


def go_to_next_page(driver):
    """
    Try to click Sahibinden's "next page" control. Returns True if it
    navigated, False if there's no next page (or it couldn't be found).

    NOTE: this looks for a link/button whose visible text contains
    "sonraki" (Turkish for "next") or whose rel/aria-label suggests
    pagination - a best-effort guess, since I couldn't inspect the live
    page. If discovery stops after page 1 every time, open the search
    page in a normal browser, inspect the pagination control at the
    bottom, and update the selector below to match it.
    """
    try:
        candidates = driver.find_elements(
            "xpath",
            "//a[contains(translate(., 'SONRAKİ', 'sonraki'), 'sonraki') "
            "or contains(@rel, 'next') or contains(@aria-label, 'next') "
            "or contains(@aria-label, 'Sonraki')]"
        )
        for el in candidates:
            if el.is_displayed() and el.is_enabled():
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", el)
                time.sleep(random.uniform(0.5, 1.0))
                el.click()
                return True
        return False
    except Exception as e:
        print(f"  [!] Could not navigate to next page: {e}")
        return False


def load_existing_urls():
    try:
        with open(URL_FILE, 'r') as f:
            return {line.strip() for line in f if line.strip()}
    except FileNotFoundError:
        return set()


def main():
    print("-" * 60)
    print("Sahibinden.com URL Discovery (Kurtkoy Rentals)")
    print("-" * 60)

    existing = load_existing_urls()
    print(f"{len(existing)} URL(s) already in {URL_FILE}")

    driver = setup_driver()
    all_found = set()

    try:
        print(f"\nNavigating to search results: {SEARCH_URL}")
        driver.get(SEARCH_URL)
        time.sleep(PAGE_LOAD_WAIT)

        if "Verify" in driver.title or "challenge" in driver.title.lower():
            print("\n  [!] CAPTCHA detected. Please solve it in the browser window.")
            input("  Press Enter once resolved...")

        for page_num in range(1, MAX_PAGES + 1):
            for _ in range(3):
                human_like_scroll(driver)

            page_urls = extract_listing_urls(driver)
            new_this_page = page_urls - existing - all_found
            all_found |= page_urls
            print(f"Page {page_num}: found {len(page_urls)} listing link(s), "
                  f"{len(new_this_page)} new.")

            if len(existing | all_found) - len(existing) >= MAX_NEW_URLS:
                print(f"Reached MAX_NEW_URLS cap ({MAX_NEW_URLS}). Stopping.")
                break

            if page_num < MAX_PAGES:
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                print(f"  Waiting {delay:.1f}s before next page...")
                time.sleep(delay)
                if not go_to_next_page(driver):
                    print("No further pages found (or pagination control not detected). Stopping.")
                    break
                time.sleep(PAGE_LOAD_WAIT)

    finally:
        print("\nShutting down browser...")
        time.sleep(2)
        driver.quit()

    new_urls = sorted((all_found - existing))[:MAX_NEW_URLS]

    if not new_urls:
        print("\nNo new URLs found. sahibinden_urls.txt left unchanged.")
        return

    with open(URL_FILE, 'a') as f:
        for url in new_urls:
            f.write(url + "\n")

    print(f"\nAppended {len(new_urls)} new URL(s) to {URL_FILE}.")
    print(f"Total URLs in file now: {len(existing) + len(new_urls)}")
    print("\nNext step: run scraper_sahibinden.py to scrape these listings "
          f"(it processes up to 15 per session - re-run it multiple times to work through all of them).")


if __name__ == "__main__":
    main()
