#!/usr/bin/env python3
"""
Download Git LFS files from GitHub repository.

This script downloads the KPLO ephemeris kernels that are stored as
Git LFS objects in the repository.
"""

import os
import requests
import hashlib

def parse_lfs_pointer(filepath):
    """Parse a Git LFS pointer file to get the SHA256 OID and size."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    oid = None
    size = None
    for line in lines:
        if line.startswith('oid sha256:'):
            oid = line.split(':')[1].strip()
        elif line.startswith('size '):
            size = int(line.split()[1])

    return oid, size

def download_from_github_lfs(repo_user, repo_name, branch, filepath, output_path):
    """
    Download a file from GitHub LFS storage.

    Uses GitHub's media URL endpoint which serves LFS files.
    """
    # Try media.githubusercontent.com endpoint
    media_url = f"https://media.githubusercontent.com/media/{repo_user}/{repo_name}/{branch}/{filepath}"

    print(f"Attempting to download from: {media_url}")

    response = requests.get(media_url, stream=True, allow_redirects=True)

    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0

        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r  Progress: {downloaded}/{total_size} bytes ({percent:.1f}%)", end='', flush=True)

        print()  # New line after progress
        return True
    else:
        print(f"  Failed with status code: {response.status_code}")
        return False

def main():
    """Download KPLO ephemeris kernels."""
    print("=" * 70)
    print("GitHub LFS Kernel Downloader for KPLO")
    print("=" * 70)

    # Repository details
    repo_user = "pmahanti"
    repo_name = "documents"
    branch = "claude/psr-sdc1-work-01Xo7bsFPQ7aVi9trog3sdFj"

    # Files to download
    files_to_download = [
        "kernels/kplo_pm_20251113_20260108_v00.bsp",
        "kernels/kplo_pm_20251116_20260111_v00.bsp",
        "kernels/de430.bsp"
    ]

    base_dir = "/home/user/documents"

    for filepath in files_to_download:
        print(f"\nDownloading: {filepath}")

        pointer_file = os.path.join(base_dir, filepath)
        output_file = pointer_file

        # Check if it's an LFS pointer
        if os.path.exists(pointer_file):
            with open(pointer_file, 'r') as f:
                first_line = f.readline()
                if 'git-lfs' in first_line:
                    print("  Detected LFS pointer file")
                    oid, size = parse_lfs_pointer(pointer_file)
                    print(f"  OID: {oid}")
                    print(f"  Expected size: {size:,} bytes")
                else:
                    print("  File already exists and is not an LFS pointer - skipping")
                    continue

        # Try to download
        success = download_from_github_lfs(repo_user, repo_name, branch, filepath, output_file)

        if success:
            # Verify file
            actual_size = os.path.getsize(output_file)
            print(f"  ✓ Downloaded: {actual_size:,} bytes")

            # Check if it's still a pointer (failed download)
            with open(output_file, 'r', errors='ignore') as f:
                first_line = f.readline()
                if 'git-lfs' in first_line:
                    print("  ✗ Download failed - still got LFS pointer")
                else:
                    print("  ✓ Verified: Binary file downloaded successfully")
        else:
            print(f"  ✗ Download failed for {filepath}")

    print("\n" + "=" * 70)
    print("Download complete!")
    print("\nNote: If downloads failed, you may need to:")
    print("  1. Install git-lfs: sudo apt-get install git-lfs")
    print("  2. Run: git lfs pull")
    print("  3. Or manually download from:")
    print(f"     https://github.com/{repo_user}/{repo_name}/tree/{branch}/kernels")
    print("=" * 70)

if __name__ == "__main__":
    main()
