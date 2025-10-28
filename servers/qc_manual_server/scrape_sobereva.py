#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import sys
from datetime import datetime

# Set up logging
log_filename = f"scraper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Progress bar function
def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█', print_end="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        print_end   - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    sys.stdout.flush()
    # Print New Line on Complete
    if iteration == total: 
        print()

# Create output directory
OUTPUT_DIR = "sobereva_blogs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base URL
BASE_URL = "http://sobereva.com/"
LIST_URL = "http://sobereva.com/list.html"

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def get_page(url):
    """Fetch a web page with retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()  # Raise exception for 4XX/5XX responses
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt+1}/{max_retries} failed for {url}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 5 * (attempt + 1)  # Exponential backoff
                logging.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None

def extract_blog_urls(list_page_content):
    """Extract all blog post URLs from the list page"""
    soup = BeautifulSoup(list_page_content, 'html.parser')
    blog_urls = []
    
    # Based on the website structure, blog posts are likely listed in a specific section
    # Look for the blog directory section (博文的目录)
    blog_section = soup.find(string="博文的目录")
    if blog_section and blog_section.parent:
        # Start from this section and look for links
        section_parent = blog_section.parent
        
        # Find all links in this section and following sections
        for element in section_parent.find_all_next('a'):
            href = element.get('href')
            if not href:
                continue
                
            # Convert relative URLs to absolute
            if href.startswith('/'):
                full_url = urljoin(BASE_URL, href)
            else:
                full_url = href
                
            # Check if it's a sobereva.com URL
            if BASE_URL in full_url:
                # Blog posts typically have numeric IDs
                parts = full_url.rstrip('/').split('/')
                if parts[-1].isdigit():
                    blog_urls.append(full_url)
    else:
        # Fallback: look for all links that might be blog posts
        logging.warning("Could not find blog directory section, using fallback method")
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and href.startswith(('http://sobereva.com/', '/')) and 'list.html' not in href:
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    full_url = urljoin(BASE_URL, href)
                else:
                    full_url = href
                    
                # Check if it's likely a blog post URL (contains a number)
                if BASE_URL in full_url:
                    parts = full_url.rstrip('/').split('/')
                    if parts[-1].isdigit():
                        blog_urls.append(full_url)
    
    return list(set(blog_urls))  # Remove duplicates

def save_blog_post(url, content):
    """Save a blog post to a file"""
    # Extract the blog post ID from the URL
    blog_id = url.rstrip('/').split('/')[-1]
    
    # Create a blog-specific directory for this post and its resources
    blog_dir = os.path.join(OUTPUT_DIR, blog_id)
    os.makedirs(blog_dir, exist_ok=True)
    
    # Create a filename
    filename = f"index.html"
    filepath = os.path.join(blog_dir, filename)
    
    # Parse the content to find images and other resources
    soup = BeautifulSoup(content, 'html.parser')
    
    # Process images
    for img in soup.find_all('img'):
        src = img.get('src')
        if not src:
            continue
            
        # Convert relative URLs to absolute
        if src.startswith('/'):
            img_url = urljoin(BASE_URL, src)
        elif not (src.startswith('http://') or src.startswith('https://')):
            img_url = urljoin(url, src)
        else:
            img_url = src
            
        # Update the image source to point to the local file
        img_filename = os.path.basename(img_url.split('?')[0])  # Remove query parameters
        img_local_path = os.path.join('images', img_filename)
        img['src'] = img_local_path
        
        # Download the image
        try:
            img_dir = os.path.join(blog_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            img_filepath = os.path.join(img_dir, img_filename)
            
            # Skip if already downloaded
            if os.path.exists(img_filepath):
                continue
                
            logging.info(f"Downloading image: {img_url}")
            img_response = get_page(img_url)
            if img_response:
                with open(img_filepath, 'wb') as f:
                    f.write(img_response.content)
                time.sleep(random.uniform(0.5, 1.5))  # Be nice to the server
        except Exception as e:
            logging.error(f"Failed to download image {img_url}: {e}")
    
    # Process CSS and JS files
    for link in soup.find_all(['link', 'script']):
        # For link tags (CSS)
        if link.name == 'link' and link.get('rel') == ['stylesheet']:
            href = link.get('href')
            if href:
                process_resource(href, url, blog_dir, link, 'href')
        
        # For script tags (JS)
        elif link.name == 'script' and link.get('src'):
            src = link.get('src')
            process_resource(src, url, blog_dir, link, 'src')
    
    # Save the modified content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    logging.info(f"Saved blog post {blog_id} to {filepath}")
    return filepath

def process_resource(resource_url, base_url, blog_dir, tag, attr_name):
    """Process and download a resource (CSS, JS, etc.)"""
    if not resource_url:
        return
        
    # Convert relative URLs to absolute
    if resource_url.startswith('/'):
        full_url = urljoin(BASE_URL, resource_url)
    elif not (resource_url.startswith('http://') or resource_url.startswith('https://')):
        full_url = urljoin(base_url, resource_url)
    else:
        full_url = resource_url
        
    # Skip external resources
    if BASE_URL not in full_url:
        return
        
    # Update the tag to point to the local file
    resource_filename = os.path.basename(full_url.split('?')[0])  # Remove query parameters
    resource_type = resource_url.split('.')[-1] if '.' in resource_url else 'misc'
    resource_local_dir = resource_type
    resource_local_path = os.path.join(resource_local_dir, resource_filename)
    tag[attr_name] = resource_local_path
    
    # Download the resource
    try:
        resource_dir = os.path.join(blog_dir, resource_local_dir)
        os.makedirs(resource_dir, exist_ok=True)
        resource_filepath = os.path.join(resource_dir, resource_filename)
        
        # Skip if already downloaded
        if os.path.exists(resource_filepath):
            return
            
        logging.info(f"Downloading resource: {full_url}")
        resource_response = get_page(full_url)
        if resource_response:
            with open(resource_filepath, 'wb') as f:
                f.write(resource_response.content)
            time.sleep(random.uniform(0.5, 1.5))  # Be nice to the server
    except Exception as e:
        logging.error(f"Failed to download resource {full_url}: {e}")

def download_blog_posts(blog_urls):
    """Download all blog posts"""
    total = len(blog_urls)
    
    # Create a dictionary to store blog titles
    blog_titles = {}
    
    # Initialize progress bar
    print_progress(0, total, prefix='Downloading:', suffix=f'0/{total} Complete', length=40)
    
    for i, url in enumerate(blog_urls, 1):
        blog_id = url.rstrip('/').split('/')[-1]
        
        # Update progress bar
        print_progress(i-1, total, prefix='Downloading:', suffix=f'{i-1}/{total} Complete', length=40)
        
        logging.info(f"Processing {i}/{total}: {url} (ID: {blog_id})")
        
        # Check if we already downloaded this blog post
        blog_dir = os.path.join(OUTPUT_DIR, blog_id)
        blog_filepath = os.path.join(blog_dir, "index.html")
        if os.path.exists(blog_filepath):
            logging.info(f"Blog post {blog_id} already exists, skipping download")
            
            # Try to extract the title from the existing file
            try:
                with open(blog_filepath, 'r', encoding='utf-8') as f:
                    soup = BeautifulSoup(f.read(), 'html.parser')
                    title = extract_title(soup)
                    if title:
                        blog_titles[blog_id] = title
                        logging.info(f"Extracted title from existing file: {title}")
            except Exception as e:
                logging.error(f"Error extracting title from existing file: {e}")
            
            continue
        
        # Fetch the blog post
        response = get_page(url)
        if response:
            # Extract the title before saving
            soup = BeautifulSoup(response.text, 'html.parser')
            title = extract_title(soup)
            if title:
                blog_titles[blog_id] = title
                logging.info(f"Extracted title: {title}")
            
            save_blog_post(url, response.text)
            
            # Add a random delay between requests to be nice to the server
            if i < total:
                delay = random.uniform(1, 3)
                logging.info(f"Waiting {delay:.2f} seconds before next request...")
                time.sleep(delay)
        else:
            logging.error(f"Failed to download {url}")
    
    # Complete the progress bar
    print_progress(total, total, prefix='Downloading:', suffix=f'{total}/{total} Complete', length=40)
    
    # Save the blog titles for future reference
    titles_path = os.path.join(OUTPUT_DIR, "blog_titles.txt")
    with open(titles_path, 'w', encoding='utf-8') as f:
        for blog_id, title in sorted(blog_titles.items(), key=lambda x: int(x[0])):
            f.write(f"{blog_id}: {title}\n")
    
    return blog_titles

def extract_title(soup):
    """Extract the title of a blog post"""
    # Try different methods to find the title
    
    # Method 1: Look for the title tag
    if soup.title:
        title = soup.title.string
        if title and "sobereva" in title.lower():
            # Clean up the title
            title = title.strip()
            return title
    
    # Method 2: Look for the first h1 tag
    h1 = soup.find('h1')
    if h1:
        return h1.get_text().strip()
    
    # Method 3: Look for the first h2 tag
    h2 = soup.find('h2')
    if h2:
        return h2.get_text().strip()
    
    # Method 4: Look for a meta tag with name="description"
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        return meta_desc.get('content').strip()
    
    return None

def main():
    print("=" * 70)
    print(f"Sobereva Blog Scraper - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    logging.info("Starting to scrape sobereva.com blog posts")
    
    # Fetch the list page
    print("\nFetching list page...")
    logging.info(f"Fetching list page: {LIST_URL}")
    response = get_page(LIST_URL)
    if not response:
        logging.error("Failed to fetch the list page. Exiting.")
        print("Error: Failed to fetch the list page. Check the log file for details.")
        return
    
    # Save the list page for reference
    list_page_path = os.path.join(OUTPUT_DIR, "list.html")
    with open(list_page_path, 'w', encoding='utf-8') as f:
        f.write(response.text)
    logging.info(f"Saved list page to {list_page_path}")
    
    # Extract blog post URLs
    print("Extracting blog post URLs...")
    logging.info("Extracting blog post URLs")
    blog_urls = extract_blog_urls(response.text)
    logging.info(f"Found {len(blog_urls)} blog post URLs")
    print(f"Found {len(blog_urls)} blog post URLs")
    
    # Save the list of URLs for reference
    urls_path = os.path.join(OUTPUT_DIR, "blog_urls.txt")
    with open(urls_path, 'w', encoding='utf-8') as f:
        for url in blog_urls:
            f.write(f"{url}\n")
    print(f"Saved URLs list to {urls_path}")
    
    # Download all blog posts and get their titles
    print("\nStarting to download blog posts...")
    logging.info("Starting to download blog posts")
    blog_titles = download_blog_posts(blog_urls)
    
    # Create an index.html file that links to all downloaded blog posts
    print("\nCreating index page...")
    create_index_page(blog_urls, blog_titles)
    
    print("\n" + "=" * 70)
    print(f"Scraping completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total blog posts: {len(blog_urls)}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Log file: {os.path.abspath(log_filename)}")
    print("=" * 70)
    
    logging.info("Scraping completed")

def create_index_page(blog_urls, blog_titles=None):
    """Create an index.html file that links to all downloaded blog posts"""
    index_path = os.path.join(OUTPUT_DIR, "index.html")
    
    # Sort blog URLs by ID (numeric part of the URL)
    sorted_urls = sorted(blog_urls, key=lambda url: int(url.rstrip('/').split('/')[-1]))
    
    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sobereva Blog Archive</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            text-align: center;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f2f2f2;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        a {
            text-decoration: none;
            color: #0066cc;
        }
        a:hover {
            text-decoration: underline;
        }
        .blog-id {
            font-weight: bold;
            width: 60px;
        }
        .blog-title {
            width: 60%;
        }
        .blog-url {
            width: 30%;
        }
    </style>
</head>
<body>
    <h1>Sobereva Blog Archive</h1>
    <p>Total blog posts: """ + str(len(sorted_urls)) + """</p>
    <table>
        <tr>
            <th class="blog-id">ID</th>
            <th class="blog-title">Title</th>
            <th class="blog-url">URL</th>
        </tr>
"""
    
    for url in sorted_urls:
        blog_id = url.rstrip('/').split('/')[-1]
        title = blog_titles.get(blog_id, "") if blog_titles else ""
        
        html_content += f"""        <tr>
            <td class="blog-id">{blog_id}</td>
            <td class="blog-title"><a href="./{blog_id}/index.html">{title}</a></td>
            <td class="blog-url"><a href="{url}" target="_blank">{url}</a></td>
        </tr>
"""
    
    html_content += """    </table>
</body>
</html>
"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logging.info(f"Created index page at {index_path}")

if __name__ == "__main__":
    main()
