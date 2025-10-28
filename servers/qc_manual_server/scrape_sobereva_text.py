#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import logging
import sys
import re
import html2text
from datetime import datetime

# Set up logging
log_filename = f"scraper_text_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

# Create output directory
OUTPUT_DIR = "sobereva_blogs_text"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base URL
BASE_URL = "http://sobereva.com/"
LIST_URL = "http://sobereva.com/list.html"

# IDs of blog posts to skip (non-computational chemistry content)
SKIP_IDS = [
    # 动漫相关
    736, 735, 728, 723, 720, 718, 710, 707, 695, 694, 689, 686, 669, 668, 650, 633, 631, 629, 611, 544, 577, 561, 470, 520, 497, 496, 495, 494, 401, 446, 430, 358, 393, 386, 316, 349, 341, 326, 273, 306, 301, 300, 219, 254, 174, 194, 181, 118,
    # 音乐相关
    496, 459, 385, 360, 318, 299, 279, 249, 218, 187, 175, 154, 116, 95, 71, 72, 48, 49, 34, 35,
    # 漫展/二次元活动
    710, 424, 376, 288, 287, 283, 140, 226, 225, 135, 133, 123, 122, 120, 111, 110, 109, 84, 77, 89, 80,
    # 个人生活/杂谈
    653, 356, 342, 325, 324, 132, 124, 100, 99, 25, 19, 18, 17,
    # 其他无关内容
    508, 489, 450, 256
]

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Initialize HTML to Markdown converter
h2t = html2text.HTML2Text()
h2t.ignore_links = False
h2t.ignore_images = True
h2t.ignore_tables = False
h2t.ignore_emphasis = False
h2t.body_width = 0  # Don't wrap text

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

def extract_title(soup):
    """Extract the title of a blog post"""
    # Try different methods to find the title
    
    # Method 1: Look for the first h1 tag (most likely to be the actual blog post title)
    h1 = soup.find('h1')
    if h1:
        return h1.get_text().strip()
    
    # Method 2: Look for the first h2 tag
    h2 = soup.find('h2')
    if h2:
        return h2.get_text().strip()
    
    # Method 3: Look for a meta tag with name="description"
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        return meta_desc.get('content').strip()
    
    # Method 4: Look for the title tag (least preferred as it often contains site name)
    if soup.title:
        title = soup.title.string
        if title:
            # Clean up the title and try to extract just the blog post title
            title = title.strip()
            # If the title contains a separator like "-" or "|", take the first part
            if " - " in title:
                title = title.split(" - ")[0].strip()
            elif " | " in title:
                title = title.split(" | ")[0].strip()
            return title
    
    return None

def should_skip_blog(url):
    """Check if a blog post should be skipped based on its ID"""
    # Extract the blog post ID from the URL
    blog_id = url.rstrip('/').split('/')[-1]
    try:
        blog_id = int(blog_id)
        return blog_id in SKIP_IDS
    except ValueError:
        return False

def extract_blog_content(soup):
    """Extract the main content of a blog post and convert to Markdown"""
    # Try to find the main content area
    content_area = None
    
    # Method 1: Look for article tag
    article = soup.find('article')
    if article:
        content_area = article
    
    # Method 2: Look for div with class containing 'content' or 'article'
    if not content_area:
        for div in soup.find_all('div'):
            div_class = div.get('class', [])
            if isinstance(div_class, list):
                div_class = ' '.join(div_class)
            if 'content' in div_class or 'article' in div_class:
                content_area = div
                break
    
    # Method 3: Look for div with id containing 'content' or 'article'
    if not content_area:
        for div in soup.find_all('div'):
            div_id = div.get('id', '')
            if 'content' in div_id or 'article' in div_id:
                content_area = div
                break
    
    # If we still haven't found the content area, use the body
    if not content_area:
        content_area = soup.find('body')
    
    if content_area:
        # Remove script and style elements
        for element in content_area.find_all(['script', 'style', 'iframe', 'noscript']):
            element.decompose()
        
        # Convert to Markdown
        markdown_content = h2t.handle(str(content_area))
        
        # Clean up the Markdown
        markdown_content = re.sub(r'\n{3,}', '\n\n', markdown_content)  # Remove excessive newlines
        
        return markdown_content
    
    return ""

def save_blog_post_as_markdown(url, title, content):
    """Save a blog post as a Markdown file"""
    # Extract the blog post ID from the URL
    blog_id = url.rstrip('/').split('/')[-1]
    
    # Create a filename
    filename = f"{blog_id}.md"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    # Add title and metadata to the content
    full_content = f"# {title}\n\n"
    full_content += f"**URL:** {url}  \n"
    full_content += f"**ID:** {blog_id}  \n\n"
    full_content += "---\n\n"
    full_content += content
    
    # Save the content
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_content)
    
    logging.info(f"Saved blog post {blog_id} to {filepath}")
    return filepath

def process_single_blog(url):
    """Process a single blog post"""
    logging.info(f"Processing blog post: {url}")
    
    # Check if the blog post should be skipped
    if should_skip_blog(url):
        logging.info(f"Skipping blog post: {url}")
        return None
    
    # Fetch the blog post
    response = get_page(url)
    if not response:
        logging.error(f"Failed to fetch blog post: {url}")
        return None
    
    # Parse the HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the title
    title = extract_title(soup)
    if not title:
        logging.warning(f"Could not extract title for {url}")
        title = f"Blog Post {url.rstrip('/').split('/')[-1]}"
    
    logging.info(f"Title: {title}")
    
    # Extract the content
    content = extract_blog_content(soup)
    
    # Save the content as Markdown
    filepath = save_blog_post_as_markdown(url, title, content)
    
    return {
        'url': url,
        'title': title,
        'filepath': filepath
    }

def main():
    print("=" * 70)
    print(f"Sobereva Blog Text Scraper - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Fetch the list page
    print("\nFetching list page...")
    response = get_page(LIST_URL)
    if not response:
        print("Error: Failed to fetch the list page.")
        return
    
    # Extract blog post URLs
    print("Extracting blog post URLs...")
    blog_urls = extract_blog_urls(response.text)
    print(f"Found {len(blog_urls)} blog post URLs")
    
    # Save the list of URLs for reference
    urls_path = os.path.join(OUTPUT_DIR, "blog_urls.txt")
    with open(urls_path, 'w', encoding='utf-8') as f:
        for url in blog_urls:
            f.write(f"{url}\n")
    print(f"Saved URLs list to {urls_path}")
    
    # Process blog posts
    print("\nStarting to download blog posts...")
    print(f"Rate: 2 blog posts per minute (30 seconds between each post)")
    
    total_urls = len(blog_urls)
    processed_count = 0
    skipped_count = 0
    
    for i, url in enumerate(blog_urls, 1):
        print(f"\nProcessing {i}/{total_urls}: {url}")
        
        # Process the blog post
        result = process_single_blog(url)
        
        if result:
            processed_count += 1
            print(f"Successfully processed: {result['title']}")
            print(f"Saved to: {result['filepath']}")
        else:
            if should_skip_blog(url):
                skipped_count += 1
                print(f"Skipped (non-computational chemistry content)")
            else:
                print(f"Failed to process")
        
        # Add a delay between requests (30 seconds = 2 posts per minute)
        if i < total_urls:
            delay = 30
            print(f"Waiting {delay} seconds before next request...")
            time.sleep(delay)
    
    print("\n" + "=" * 70)
    print(f"Scraping completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total blog posts: {total_urls}")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
