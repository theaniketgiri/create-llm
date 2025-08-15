#!/usr/bin/env python3
"""
Download and prepare datasets for LLM training.
"""

import argparse
import os
import requests
import zipfile
import tarfile
from pathlib import Path
from typing import Optional

def download_file(url: str, output_path: str):
    """Download a file from URL."""
    print(f"Downloading {url} to {output_path}...")
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Print progress
                if total_size > 0:
                    progress = (downloaded / total_size) * 100
                    print(f"\rProgress: {progress:.1f}%", end='', flush=True)
    
    print()  # New line after progress

def extract_archive(archive_path: str, extract_dir: str):
    """Extract archive file."""
    print(f"Extracting {archive_path} to {extract_dir}...")
    
    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar.gz') or archive_path.endswith('.tgz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

def download_wikitext(output_dir: str):
    """Download WikiText-103 dataset."""
    print("Downloading WikiText-103 dataset...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download URLs
    urls = {
        'train': 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    }
    
    for split, url in urls.items():
        output_path = os.path.join(output_dir, f'{split}.zip')
        
        # Download if not exists
        if not os.path.exists(output_path):
            download_file(url, output_path)
        
        # Extract
        extract_dir = os.path.join(output_dir, split)
        if not os.path.exists(extract_dir):
            extract_archive(output_path, output_dir)
    
    print("WikiText-103 download completed!")

def download_openwebtext(output_dir: str):
    """Download OpenWebText dataset (subset)."""
    print("Downloading OpenWebText dataset...")
    
    # Note: OpenWebText is very large, so we'll provide instructions
    print("OpenWebText is a large dataset (~40GB).")
    print("Please download it manually from:")
    print("https://github.com/jcpeterson/openwebtext")
    print(f"Then extract it to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for LLM training")
    parser.add_argument("--dataset", "-d", choices=["wikitext", "openwebtext"], 
                       default="wikitext", help="Dataset to download")
    parser.add_argument("--output", "-o", default="data/raw", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        if args.dataset == "wikitext":
            download_wikitext(args.output)
        elif args.dataset == "openwebtext":
            download_openwebtext(args.output)
        
        print("Download completed successfully!")
        return 0
    except Exception as e:
        print(f"Error during download: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
