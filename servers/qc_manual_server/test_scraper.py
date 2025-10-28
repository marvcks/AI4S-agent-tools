#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
from scrape_sobereva_text import process_single_blog, OUTPUT_DIR

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_scraper.py <blog_id>")
        print("Example: python test_scraper.py 164")
        return
    
    blog_id = sys.argv[1]
    url = f"http://sobereva.com/{blog_id}"
    
    print(f"Testing scraper with blog post: {url}")
    result = process_single_blog(url)
    
    if result:
        print(f"\nSuccessfully processed blog post:")
        print(f"Title: {result['title']}")
        print(f"Saved to: {result['filepath']}")
        
        # Print the first few lines of the content
        filepath = result['filepath']
        print("\nPreview of the content:")
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            preview_lines = lines[:20]  # First 20 lines
            for line in preview_lines:
                print(line.rstrip())
            if len(lines) > 20:
                print("...")
                print(f"(Total {len(lines)} lines)")
    else:
        print(f"\nFailed to process blog post: {url}")
        print("This could be because:")
        print("1. The blog post ID is invalid")
        print("2. The blog post is in the skip list (non-computational chemistry content)")
        print("3. There was an error fetching or processing the blog post")

if __name__ == "__main__":
    main()
