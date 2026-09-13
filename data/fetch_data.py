"""Verify and inspect raw Customer Support on Twitter dataset.

The raw dataset was originally obtained via the Kaggle CLI:
    kaggle datasets download -d thoughtvector/customer-support-on-twitter
    unzip customer-support-on-twitter.zip -d data/raw/
    (produces data/raw/twcs.csv, ~493MB)

If running on a new machine without data/raw/twcs.csv, run the commands above
or configure Kaggle credentials (~/.kaggle/kaggle.json) and use the Kaggle API.

This script verifies that data/raw/twcs.csv is present, checks its file size,
and counts the total number of records using memory-efficient chunking.
"""

import os
import sys
from typing import Tuple


RAW_CSV_PATH = os.path.join(os.path.dirname(__file__), "raw", "twcs.csv")


def format_size(num_bytes: int) -> str:
    """Format byte size into human-readable units (MB, GB)."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def count_csv_records(csv_path: str, chunksize: int = 250000) -> int:
    """Count total data rows in CSV using memory-safe pandas chunking.

    Args:
        csv_path: Path to target CSV file.
        chunksize: Number of rows per chunk.

    Returns:
        int: Total number of data records (excluding header).
    """
    try:
        import pandas as pd
        total_records = 0
        for chunk in pd.read_csv(csv_path, chunksize=chunksize, usecols=["tweet_id"]):
            total_records += len(chunk)
        return total_records
    except ImportError:
        # Fallback to standard library csv reader if pandas is not yet installed
        import csv
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            return sum(1 for _ in reader)


def verify_raw_dataset(csv_path: str = RAW_CSV_PATH) -> Tuple[bool, int, int]:
    """Verify presence, size, and record count of raw twcs.csv.

    Args:
        csv_path: Path to verify.

    Returns:
        Tuple[bool, int, int]: (exists, size_in_bytes, row_count)
    """
    if not os.path.exists(csv_path):
        return False, 0, 0

    size_bytes = os.path.getsize(csv_path)
    record_count = count_csv_records(csv_path)
    return True, size_bytes, record_count


def main() -> None:
    """CLI entry point to verify twcs.csv."""
    print("=" * 60)
    print("Customer Support on Twitter — Dataset Verification")
    print("=" * 60)
    print(f"Target Path: {os.path.abspath(RAW_CSV_PATH)}")

    if not os.path.exists(RAW_CSV_PATH):
        print("\n[ERROR] twcs.csv not found at expected location.")
        print("\nTo download the dataset:")
        print("  1. Ensure kaggle CLI is installed: pip install kaggle")
        print("  2. Configure ~/.kaggle/kaggle.json with your API credentials")
        print("  3. Run:")
        print("     kaggle datasets download -d thoughtvector/customer-support-on-twitter")
        print("     unzip customer-support-on-twitter.zip -d data/raw/")
        sys.exit(1)

    print("\nFile detected! Calculating size and record count...")
    exists, size_bytes, row_count = verify_raw_dataset(RAW_CSV_PATH)

    print(f"\nStatus:       FOUND")
    print(f"File Size:    {format_size(size_bytes)} ({size_bytes:,} bytes)")
    print(f"Record Count: {row_count:,} rows")
    print("=" * 60)


if __name__ == "__main__":
    main()
