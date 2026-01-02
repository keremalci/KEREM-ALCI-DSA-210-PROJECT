"""
Extract Sahibinden listing URLs from saved HTML files
"""

from bs4 import BeautifulSoup
import os
import re

def extract_urls_from_html(html_file):
    """Extract all listing URLs from an HTML file"""
    print(f"\n📄 Processing: {os.path.basename(html_file)}")
    
    with open(html_file, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find all links to listing details
    urls = set()
    
    # Method 1: Find all <a> tags with href containing '/ilan/emlak-konut-kiralik'
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '/ilan/emlak-konut-kiralik' in href and '/detay' in href:
            # Make sure it's a full URL
            if href.startswith('http'):
                urls.add(href)
            elif href.startswith('/ilan/'):
                urls.add(f"https://www.sahibinden.com{href}")
    
    # Method 2: Find table rows with data-id (listing IDs)
    for tr in soup.find_all('tr', {'data-id': True}):
        data_id = tr.get('data-id')
        if data_id and data_id.isdigit():
            # Try to find the link within this row
            link = tr.find('a', href=re.compile(r'/ilan/.*' + data_id))
            if link and link.get('href'):
                href = link['href']
                if href.startswith('http'):
                    urls.add(href)
                elif href.startswith('/ilan/'):
                    urls.add(f"https://www.sahibinden.com{href}")
    
    print(f"  ✓ Found {len(urls)} unique URLs")
    return urls

def main():
    """Extract URLs from all HTML files in project root"""
    
    # Look for HTML files in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    
    html_files = []
    for file in os.listdir(project_root):
        if file.endswith('.html') and 'sahibinden' in file.lower():
            html_files.append(os.path.join(project_root, file))
    
    if not html_files:
        print("❌ No Sahibinden HTML files found in project root")
        return
    
    print(f"\n{'='*60}")
    print(f"EXTRACTING URLS FROM {len(html_files)} HTML FILES")
    print(f"{'='*60}")
    
    all_urls = set()
    
    for html_file in sorted(html_files):
        urls = extract_urls_from_html(html_file)
        all_urls.update(urls)
    
    print(f"\n{'='*60}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total unique URLs: {len(all_urls)}")
    
    # Save to file
    output_file = os.path.join(project_root, 'sahibinden_urls_extracted.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in sorted(all_urls):
            f.write(url + '\n')
    
    print(f"\n✓ Saved to: {output_file}")
    
    # Also show preview
    print(f"\nFirst 10 URLs:")
    for i, url in enumerate(sorted(all_urls)[:10], 1):
        print(f"  {i}. {url}")
    
    return all_urls

if __name__ == "__main__":
    main()
    