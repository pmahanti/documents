#!/usr/bin/env python3
"""
Helper script to download PSR shapefile data from NASA PGDA.

Note: The website may block automated downloads (403 error).
In that case, you'll need to download manually through a web browser.
"""

import requests
import sys
from pathlib import Path
from urllib.parse import urljoin


def download_file(url: str, output_path: Path, chunk_size: int = 8192):
    """
    Download a file with progress indication.

    Args:
        url: URL to download from
        output_path: Local path to save file
        chunk_size: Download chunk size in bytes
    """
    print(f"Downloading: {url}")
    print(f"Saving to: {output_path}")

    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"Progress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='\r')

        print(f"\nDownloaded successfully: {output_path.name}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"\nError downloading {url}: {e}")
        return False


def main():
    """Download PSR shapefile components."""
    # Base URL (may need to be updated)
    base_url = "https://pgda.gsfc.nasa.gov/data/document/90/"

    # File components to download
    base_name = "LPSR_80S_20MPP_ADJ"
    extensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.csv']

    # Output directory
    output_dir = Path('data')
    output_dir.mkdir(exist_ok=True)

    print("=" * 80)
    print("NASA PGDA PSR Data Downloader")
    print("=" * 80)
    print()
    print("Attempting to download LPSR_80S_20MPP_ADJ shapefile components...")
    print()

    success_count = 0
    failed_files = []

    for ext in extensions:
        filename = f"{base_name}{ext}"
        url = urljoin(base_url, filename)
        output_path = output_dir / filename

        if output_path.exists():
            print(f"Skipping (already exists): {filename}")
            success_count += 1
            continue

        if download_file(url, output_path):
            success_count += 1
        else:
            failed_files.append(filename)

        print()

    # Summary
    print("=" * 80)
    print("Download Summary")
    print("=" * 80)
    print(f"Successful: {success_count}/{len(extensions)}")

    if failed_files:
        print(f"\nFailed downloads ({len(failed_files)}):")
        for filename in failed_files:
            print(f"  - {filename}")

        print("\n" + "=" * 80)
        print("MANUAL DOWNLOAD REQUIRED")
        print("=" * 80)
        print("The website may be blocking automated downloads.")
        print("\nPlease download the files manually:")
        print("1. Visit: https://pgda.gsfc.nasa.gov/products/90")
        print("2. Download all LPSR_80S_20MPP_ADJ.* files")
        print(f"3. Place them in: {output_dir.absolute()}")
    else:
        print("\nAll files downloaded successfully!")
        print("\nNext steps:")
        print("1. Convert to parquet (optional): python psr_query.py --convert")
        print("2. Query PSRs: python psr_query.py --lat -89.5 --lon 45.0 --radius 50")

    return 0 if len(failed_files) == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
