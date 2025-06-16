#!/usr/bin/env python3
"""
Fake News Detection - Model Downloader

This script downloads and sets up the pre-trained DistilBERT model for the
Fake News Detection project. It handles downloading, verification, and
extraction of model files from a GitHub release.
"""

import os
import sys
import time
import hashlib
import zipfile
import argparse
import requests
from pathlib import Path
from tqdm import tqdm

# Constants
# GitHub release URL for the pre-trained model
MODEL_URL = "https://github.com/hitesh311/Fake-News-Detection/releases/download/v1.0.0/model_files.zip"
# MD5 checksum for verifying file integrity (GitHub uses MD5 for release assets)
MODEL_CHECKSUM = "b0318b147034d8af736743f5c7ff3186"  # MD5 checksum for v1.0.0
MODEL_DIR = "model_save"
ZIP_FILENAME = "model_files.zip"
CHUNK_SIZE = 8192  # 8KB chunks for download

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header():
    """Print the script header."""
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Fake News Detection - Model Downloader{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}\n")

def calculate_checksum(file_path, algorithm='md5', block_size=65536):
    """Calculate checksum of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm (default: 'md5')
        block_size: Size of blocks to read
        
    Returns:
        str: Hexadecimal digest of the file
    """
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for block in iter(lambda: f.read(block_size), b''):
            hash_obj.update(block)
    return hash_obj.hexdigest()

def download_file(url, destination, description="Downloading"):
    """Download a file with progress bar.
    
    Args:
        url: URL to download from
        destination: Path to save the file
        description: Description for progress bar
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Get file size for progress bar
        response = requests.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        file_size = int(response.headers.get('content-length', 0))
        
        # Check if file exists and has the same size
        if os.path.exists(destination):
            if os.path.getsize(destination) == file_size:
                print(f"{Colors.OKGREEN}File already exists: {destination}{Colors.ENDC}")
                return True
        
        # Download with progress bar
        print(f"{Colors.OKBLUE}{description} to {destination}...{Colors.ENDC}")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Create parent directories if they don't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Download with progress bar
        with open(destination, 'wb') as f, tqdm(
            desc=description,
            total=file_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
            ascii=True
        ) as pbar:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    size = f.write(chunk)
                    pbar.update(size)
        
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"{Colors.FAIL}Error downloading {url}: {e}{Colors.ENDC}")
        if os.path.exists(destination):
            os.remove(destination)
        return False

def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress.
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"{Colors.OKBLUE}Extracting files...{Colors.ENDC}")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get total size for progress bar
            total_size = sum(file.file_size for file in zip_ref.infolist())
            
            # Extract with progress
            with tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                ascii=True
            ) as pbar:
                for file in zip_ref.infolist():
                    zip_ref.extract(file, extract_to)
                    pbar.update(file.file_size)
        
        return True
    except zipfile.BadZipFile:
        print(f"{Colors.FAIL}Error: The downloaded file is not a valid zip archive{Colors.ENDC}")
        return False
    except Exception as e:
        print(f"{Colors.FAIL}Error extracting {zip_path}: {e}{Colors.ENDC}")
        return False

def verify_model_files(model_dir):
    """Verify that all required model files are present.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        bool: True if all files are present, False otherwise
    """
    required_files = [
        'config.json',
        'pytorch_model.bin',
        'special_tokens_map.json',
        'tokenizer_config.json',
        'vocab.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.isfile(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"{Colors.WARNING}Warning: Missing required model files: {', '.join(missing_files)}{Colors.ENDC}")
        return False
    
    print(f"{Colors.OKGREEN}✓ All required model files are present{Colors.ENDC}")
    return True

def main():
    """Main function to handle the download and setup of model files."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Download and setup model files for Fake News Detection')
    parser.add_argument('--force', action='store_true', help='Force download even if files exist')
    parser.add_argument('--clean', action='store_true', help='Remove existing model files before download')
    args = parser.parse_args()
    
    print_header()
    
    # Clean existing files if requested
    if args.clean and os.path.exists(MODEL_DIR):
        print(f"{Colors.WARNING}Removing existing model directory: {MODEL_DIR}{Colors.ENDC}")
        import shutil
        shutil.rmtree(MODEL_DIR)
    
    # Create model directory
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = os.path.join(MODEL_DIR, ZIP_FILENAME)
    
    # Download the model files
    if args.force or not os.path.exists(zip_path):
        if not download_file(MODEL_URL, zip_path, "Downloading model files"):
            sys.exit(1)
    
    # Verify checksum if available
    if MODEL_CHECKSUM:
        print(f"{Colors.OKBLUE}Verifying file integrity...{Colors.ENDC}")
        file_checksum = calculate_checksum(zip_path)
        if file_checksum != MODEL_CHECKSUM:
            print(f"{Colors.WARNING}Warning: File checksum does not match expected value{Colors.ENDC}")
            print(f"  Expected: {MODEL_CHECKSUM}")
            print(f"  Actual:   {file_checksum}")
            if input("Continue anyway? [y/N] ").lower() != 'y':
                print(f"{Colors.FAIL}Download aborted by user{Colors.ENDC}")
                sys.exit(1)
    
    # Extract the model files
    if not extract_zip(zip_path, MODEL_DIR):
        sys.exit(1)
    
    # Clean up the zip file
    try:
        os.remove(zip_path)
    except OSError as e:
        print(f"{Colors.WARNING}Warning: Could not remove {zip_path}: {e}{Colors.ENDC}")
    
    # Verify all required files are present
    if not verify_model_files(MODEL_DIR):
        print(f"{Colors.WARNING}Warning: Some model files may be missing or corrupted{Colors.ENDC}")
    
    print(f"\n{Colors.OKGREEN}{Colors.BOLD}✓ Model files downloaded and extracted successfully!{Colors.ENDC}")
    print(f"\nYou can now run the application with: {Colors.BOLD}python app.py{Colors.ENDC}")
    print(f"Model directory: {os.path.abspath(MODEL_DIR)}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n{Colors.FAIL}An error occurred: {e}{Colors.ENDC}")
        sys.exit(1)
