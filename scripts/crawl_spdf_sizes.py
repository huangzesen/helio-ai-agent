#!/usr/bin/env python3
"""
Crawl the SPDF archive to compute the total size of all CDF files.

Usage:
    python scripts/crawl_spdf_sizes.py
    python scripts/crawl_spdf_sizes.py --workers 20
    python scripts/crawl_spdf_sizes.py --output spdf_sizes.json

Crawls https://spdf.gsfc.nasa.gov/pub/data/ recursively, parsing Apache
directory listings to find .cdf files and sum their sizes.
Shows live progress as it goes.
"""

import argparse
import json
import re
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/"

# Regex to parse Apache HTML table directory listing rows like:
# <tr><td><a href="file.cdf">file.cdf</a></td><td align="right">2024-01-15 10:30  </td><td align="right">1.2M</td></tr>
# Size field can be: "1.2K", "34M", "5.6G", "  - " (directory)
ROW_PATTERN = re.compile(
    r'<tr><td><a\s+href="([^"]+)">[^<]+</a></td>'  # href
    r'<td[^>]*>[^<]*</td>'                           # date column
    r'<td[^>]*>\s*([\d.]+[KMG]?|-)\s*</td></tr>',   # size column
)


def parse_size(size_str: str) -> int:
    """Convert Apache size string like '1.2M' to bytes."""
    if size_str == "-":
        return 0
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if size_str[-1] in multipliers:
        return int(float(size_str[:-1]) * multipliers[size_str[-1]])
    return int(float(size_str))


class Progress:
    """Thread-safe progress tracker with live terminal output."""

    def __init__(self):
        self.lock = threading.Lock()
        self.dirs_crawled = 0
        self.dirs_queued = 0
        self.cdf_files = 0
        self.cdf_bytes = 0
        self.other_files = 0
        self.errors = 0
        self.mission_sizes = defaultdict(int)  # mission -> bytes
        self.mission_counts = defaultdict(int)  # mission -> count
        self.current_dirs = set()
        self.start_time = time.time()
        self._last_print_len = 0

    def add_directory(self, count=1):
        with self.lock:
            self.dirs_queued += count

    def finish_directory(self, url):
        with self.lock:
            self.dirs_crawled += 1
            self.current_dirs.discard(url)

    def start_directory(self, url):
        with self.lock:
            self.current_dirs.add(url)

    def add_cdf(self, size_bytes: int, mission: str):
        with self.lock:
            self.cdf_files += 1
            self.cdf_bytes += size_bytes
            self.mission_sizes[mission] += size_bytes
            self.mission_counts[mission] += 1

    def add_other(self):
        with self.lock:
            self.other_files += 1

    def add_error(self):
        with self.lock:
            self.errors += 1

    def format_size(self, nbytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
            if nbytes < 1024:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.2f} EB"

    def print_status(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            rate = self.dirs_crawled / elapsed if elapsed > 0 else 0

            # Current directory being crawled (show first one, truncated)
            current = ""
            if self.current_dirs:
                d = next(iter(self.current_dirs))
                # Strip base URL for display
                short = d.replace(BASE_URL, "")
                if len(short) > 50:
                    short = "..." + short[-47:]
                current = f"  crawling: {short}"

            line = (
                f"\r[{elapsed_str}] "
                f"dirs: {self.dirs_crawled}/{self.dirs_queued} ({rate:.1f}/s) | "
                f"CDF files: {self.cdf_files:,} ({self.format_size(self.cdf_bytes)}) | "
                f"errors: {self.errors}"
                f"{current}"
            )
            # Pad to overwrite previous line
            pad = max(0, self._last_print_len - len(line))
            sys.stderr.write(line + " " * pad)
            sys.stderr.flush()
            self._last_print_len = len(line)

    def print_summary(self):
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print("\n")
        print("=" * 70)
        print("SPDF CDF File Size Summary")
        print("=" * 70)
        print(f"Total CDF files:    {self.cdf_files:,}")
        print(f"Total CDF size:     {self.format_size(self.cdf_bytes)}")
        print(f"Other files:        {self.other_files:,}")
        print(f"Directories:        {self.dirs_crawled:,}")
        print(f"Errors:             {self.errors:,}")
        print(f"Time elapsed:       {elapsed_str}")
        print()
        print("Top 30 missions by CDF size:")
        print("-" * 50)
        sorted_missions = sorted(
            self.mission_sizes.items(), key=lambda x: x[1], reverse=True
        )
        for mission, size in sorted_missions[:30]:
            count = self.mission_counts[mission]
            print(f"  {mission:<30s} {self.format_size(size):>12s}  ({count:,} files)")
        print("=" * 70)


def fetch_directory(url: str, session: requests.Session, progress: Progress, retries=3):
    """Fetch a directory listing and return (subdirs, cdf_files_with_sizes)."""
    # Extract mission name (first path component after /pub/data/)
    rel = url.replace(BASE_URL, "")
    mission = rel.split("/")[0] if rel else "root"

    progress.start_directory(url)

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            break
        except (requests.RequestException, Exception) as e:
            if attempt == retries - 1:
                progress.add_error()
                progress.finish_directory(url)
                return [], mission
            time.sleep(1 * (attempt + 1))

    subdirs = []
    for match in ROW_PATTERN.finditer(resp.text):
        href, size_str = match.groups()
        # Skip parent directory and sorting links
        if href.startswith("?") or href.startswith("/"):
            continue
        full_url = urljoin(url, href)

        if href.endswith("/"):
            subdirs.append(full_url)
        elif href.lower().endswith(".cdf"):
            progress.add_cdf(parse_size(size_str), mission)
        else:
            progress.add_other()

    progress.finish_directory(url)
    return subdirs, mission


def crawl(base_url: str, max_workers: int, progress: Progress):
    """BFS crawl of the SPDF directory tree."""
    session = requests.Session()
    session.headers["User-Agent"] = "spdf-size-crawler/1.0 (research)"

    progress.add_directory(1)
    queue = [base_url]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while queue:
            # Submit a batch of directory fetches
            futures = {}
            for url in queue:
                f = executor.submit(fetch_directory, url, session, progress)
                futures[f] = url
            queue = []

            # Collect results and enqueue subdirectories
            for future in as_completed(futures):
                subdirs, mission = future.result()
                if subdirs:
                    progress.add_directory(len(subdirs))
                    queue.extend(subdirs)
                progress.print_status()


def main():
    parser = argparse.ArgumentParser(
        description="Crawl SPDF to compute total CDF file sizes"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of concurrent HTTP requests (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL,
        help=f"Base URL to crawl (default: {BASE_URL})"
    )
    args = parser.parse_args()

    progress = Progress()

    print(f"Crawling {args.url}")
    print(f"Using {args.workers} concurrent workers")
    print(f"Looking for .cdf files...\n")

    try:
        crawl(args.url, args.workers, progress)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Showing partial results...")

    progress.print_summary()

    if args.output:
        result = {
            "total_cdf_files": progress.cdf_files,
            "total_cdf_bytes": progress.cdf_bytes,
            "total_cdf_human": progress.format_size(progress.cdf_bytes),
            "directories_crawled": progress.dirs_crawled,
            "errors": progress.errors,
            "missions": {
                m: {
                    "bytes": progress.mission_sizes[m],
                    "human": progress.format_size(progress.mission_sizes[m]),
                    "files": progress.mission_counts[m],
                }
                for m in sorted(
                    progress.mission_sizes, key=progress.mission_sizes.get, reverse=True
                )
            },
        }
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
