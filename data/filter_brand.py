"""Filter raw Customer Support tweets to AmazonHelp and reconstruct conversation threads.

This script processes `data/raw/twcs.csv` in memory-safe chunks to:
1. Extract all replies by 'AmazonHelp' (`author_id == 'AmazonHelp'`).
2. Identify the inbound customer tweet that triggered each reply (`in_response_to_tweet_id`).
3. Reconstruct (customer_text, brand_reply_text) pairs with thread root tracking.
4. Subsample to a reproducible set of 6,000 pairs.
5. Save to `data/processed/amazonhelp_subsample.csv` with columns:
   `tweet_id, customer_text, brand_reply_text, thread_id, created_at`
6. Output summary statistics:
   - Total AmazonHelp replies found
   - Total valid pairs before subsampling
   - Final subsample size
   - Average customer text length
   - Average brand reply text length
   - Date range covered
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd


DEFAULT_RAW_PATH = os.path.join(os.path.dirname(__file__), "raw", "twcs.csv")
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "processed", "amazonhelp_subsample.csv"
)
DEFAULT_TARGET_BRAND = "AmazonHelp"
DEFAULT_SUBSAMPLE_SIZE = 6000
DEFAULT_RANDOM_SEED = 42
DEFAULT_CHUNK_SIZE = 200000


def clean_id(series: pd.Series) -> pd.Series:
    """Normalize tweet ID strings by stripping float decimals and whitespace."""
    return series.dropna().astype(str).str.split(".").str[0].str.strip()


def extract_brand_replies(
    raw_csv_path: str,
    brand_handle: str = DEFAULT_TARGET_BRAND,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> Tuple[pd.DataFrame, Set[str], int]:
    """Pass 1: Extract all replies from the target brand and their target parent IDs.

    Args:
        raw_csv_path: Path to twcs.csv.
        brand_handle: Target brand author_id (e.g., 'AmazonHelp').
        chunksize: Number of rows to process per chunk.

    Returns:
        Tuple: (brand_replies_df, set_of_parent_tweet_ids, total_brand_replies_count)
    """
    print(f"--> Pass 1: Scanning for '{brand_handle}' replies in {raw_csv_path}...")
    brand_chunks = []
    parent_ids: Set[str] = set()
    total_brand_tweets = 0

    usecols = ["tweet_id", "author_id", "created_at", "text", "in_response_to_tweet_id"]

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, usecols=usecols, low_memory=False):
        matched_brand = chunk[chunk["author_id"] == brand_handle].copy()
        if not matched_brand.empty:
            total_brand_tweets += len(matched_brand)
            # Filter only tweets that are replies to another tweet
            replies = matched_brand[matched_brand["in_response_to_tweet_id"].notna()].copy()
            if not replies.empty:
                replies["in_resp_id_clean"] = clean_id(replies["in_response_to_tweet_id"])
                parent_ids.update(replies["in_resp_id_clean"].unique())
                replies["brand_tweet_id"] = clean_id(replies["tweet_id"])
                brand_chunks.append(
                    replies[["brand_tweet_id", "in_resp_id_clean", "text", "created_at"]]
                )

    if not brand_chunks:
        print(f"[ERROR] No tweets found for brand '{brand_handle}'")
        sys.exit(1)

    brand_df = pd.concat(brand_chunks, ignore_index=True)
    brand_df = brand_df.rename(
        columns={"text": "brand_reply_text", "created_at": "brand_created_at"}
    )
    print(
        f"    Found {total_brand_tweets:,} total '{brand_handle}' tweets "
        f"({len(brand_df):,} replies to {len(parent_ids):,} unique parent tweets)."
    )
    return brand_df, parent_ids, total_brand_tweets


def extract_customer_inbound(
    raw_csv_path: str,
    target_parent_ids: Set[str],
    brand_handle: str = DEFAULT_TARGET_BRAND,
    chunksize: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Pass 2: Extract inbound customer tweets corresponding to brand replies.

    Args:
        raw_csv_path: Path to twcs.csv.
        target_parent_ids: Set of tweet IDs that received a brand reply.
        brand_handle: Brand handle to exclude from customer authors.
        chunksize: Number of rows per chunk.

    Returns:
        pd.DataFrame: Matched customer inbound tweets.
    """
    print(f"--> Pass 2: Retrieving parent customer tweets from {raw_csv_path}...")
    customer_chunks = []
    usecols = [
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "in_response_to_tweet_id",
    ]

    for chunk in pd.read_csv(raw_csv_path, chunksize=chunksize, usecols=usecols, low_memory=False):
        chunk["tweet_id_clean"] = clean_id(chunk["tweet_id"])
        # Customer tweet must match a target parent ID, be inbound, and not authored by the brand
        matched = chunk[
            (chunk["tweet_id_clean"].isin(target_parent_ids))
            & (chunk["inbound"] == True)
            & (chunk["author_id"] != brand_handle)
        ].copy()

        if not matched.empty:
            if matched["in_response_to_tweet_id"].notna().any():
                matched["parent_id_clean"] = clean_id(matched["in_response_to_tweet_id"])
            else:
                matched["parent_id_clean"] = np.nan

            customer_chunks.append(
                matched[
                    [
                        "tweet_id_clean",
                        "text",
                        "created_at",
                        "parent_id_clean",
                    ]
                ]
            )

    if not customer_chunks:
        print("[ERROR] No matching customer tweets found.")
        sys.exit(1)

    customer_df = pd.concat(customer_chunks, ignore_index=True)
    customer_df = customer_df.rename(
        columns={
            "tweet_id_clean": "tweet_id",
            "text": "customer_text",
            "created_at": "created_at",
        }
    )
    print(f"    Retrieved {len(customer_df):,} inbound customer tweets.")
    return customer_df


def reconstruct_threads(customer_df: pd.DataFrame) -> pd.DataFrame:
    """Assign thread_id to customer tweets by tracing conversations back to thread root.

    Args:
        customer_df: Customer inbound tweets.

    Returns:
        pd.DataFrame: Customer dataframe with attached thread_id column.
    """
    parent_map: Dict[str, str] = {}
    valid_parents = customer_df[customer_df["parent_id_clean"].notna()]
    for _, row in valid_parents.iterrows():
        parent_map[str(row["tweet_id"])] = str(row["parent_id_clean"])

    def get_root_id(tid: str) -> str:
        curr = str(tid)
        visited: Set[str] = set()
        while curr in parent_map and parent_map[curr] not in visited:
            visited.add(curr)
            curr = parent_map[curr]
        return curr

    customer_df["thread_id"] = customer_df["tweet_id"].apply(get_root_id)
    return customer_df


def build_and_subsample_pairs(
    brand_df: pd.DataFrame,
    customer_df: pd.DataFrame,
    subsample_size: int = DEFAULT_SUBSAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[pd.DataFrame, int]:
    """Reconstruct pairs, drop invalid entries, and sample reproducible subset.

    Args:
        brand_df: Filtered brand replies dataframe.
        customer_df: Filtered customer inbound tweets dataframe.
        subsample_size: Number of pairs to sample.
        random_seed: Fixed seed for reproducibility.

    Returns:
        Tuple: (subsample_df, total_valid_pairs_before_subsample)
    """
    # Join on customer tweet_id == brand in_resp_id_clean
    pairs_df = pd.merge(
        customer_df,
        brand_df,
        left_on="tweet_id",
        right_on="in_resp_id_clean",
        how="inner",
    )

    # Normalize newlines and whitespace to avoid broken CSV tokens
    pairs_df["customer_text"] = (
        pairs_df["customer_text"]
        .astype(str)
        .str.replace("\r\n", "\n")
        .str.replace("\r", "\n")
        .str.strip()
    )
    pairs_df["brand_reply_text"] = (
        pairs_df["brand_reply_text"]
        .astype(str)
        .str.replace("\r\n", "\n")
        .str.replace("\r", "\n")
        .str.strip()
    )

    pairs_df = pairs_df[
        (pairs_df["customer_text"].str.len() > 0)
        & (pairs_df["brand_reply_text"].str.len() > 0)
    ]

    total_valid_pairs = len(pairs_df)

    # Subsample with fixed random seed
    sample_n = min(subsample_size, total_valid_pairs)
    subsample_df = pairs_df.sample(n=sample_n, random_state=random_seed).copy()

    # Retain strictly requested columns
    target_columns = [
        "tweet_id",
        "customer_text",
        "brand_reply_text",
        "thread_id",
        "created_at",
    ]
    subsample_df = subsample_df[target_columns].reset_index(drop=True)

    return subsample_df, total_valid_pairs


def print_summary_statistics(
    total_brand_replies: int,
    total_valid_pairs: int,
    subsample_df: pd.DataFrame,
) -> None:
    """Print formatted summary metrics for the extracted subsample."""
    avg_customer_len = subsample_df["customer_text"].str.len().mean()
    avg_brand_len = subsample_df["brand_reply_text"].str.len().mean()

    # Parse dates for min/max range
    parsed_dates = pd.to_datetime(
        subsample_df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )
    valid_dates = parsed_dates.dropna()
    if not valid_dates.empty:
        date_min_str = valid_dates.min().strftime("%Y-%m-%d")
        date_max_str = valid_dates.max().strftime("%Y-%m-%d")
        date_range_str = f"{date_min_str} to {date_max_str}"
    else:
        date_range_str = "Unknown"

    print("\n" + "=" * 65)
    print("           AMAZONHELP SUBSAMPLE SUMMARY STATISTICS           ")
    print("=" * 65)
    print(f"Total AmazonHelp replies found:       {total_brand_replies:,}")
    print(f"Total valid pairs before subsampling: {total_valid_pairs:,}")
    print(f"Final subsample size:                 {len(subsample_df):,}")
    print(f"Avg customer_text length:             {avg_customer_len:.1f} characters")
    print(f"Avg brand_reply_text length:          {avg_brand_len:.1f} characters")
    print(f"Date range covered:                   {date_range_str}")
    print("=" * 65 + "\n")


def main() -> None:
    """CLI entry point to execute brand filtering and pair reconstruction."""
    parser = argparse.ArgumentParser(
        description="Filter TWCS dataset to brand pairs and reconstruct threads."
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=DEFAULT_RAW_PATH,
        help="Path to raw twcs.csv",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to output processed CSV",
    )
    parser.add_argument(
        "--brand",
        type=str,
        default=DEFAULT_TARGET_BRAND,
        help="Brand author_id handle (default: AmazonHelp)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SUBSAMPLE_SIZE,
        help="Number of pairs to sample (default: 6000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.raw_path):
        print(f"[ERROR] Raw dataset not found at: {args.raw_path}")
        print("Please run `python data/fetch_data.py` for verification and instructions.")
        sys.exit(1)

    start_time = time.time()

    # Pass 1: Extract brand replies
    brand_df, target_parent_ids, total_brand_replies = extract_brand_replies(
        raw_csv_path=args.raw_path,
        brand_handle=args.brand,
    )

    # Pass 2: Extract matching customer tweets
    customer_df = extract_customer_inbound(
        raw_csv_path=args.raw_path,
        target_parent_ids=target_parent_ids,
        brand_handle=args.brand,
    )

    # Reconstruct threads
    customer_df = reconstruct_threads(customer_df)

    # Reconstruct pairs and subsample
    subsample_df, total_valid_pairs = build_and_subsample_pairs(
        brand_df=brand_df,
        customer_df=customer_df,
        subsample_size=args.sample_size,
        random_seed=args.seed,
    )

    # Ensure output directory exists and save CSV with robust quoting and standard line terminators
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    subsample_df.to_csv(
        args.output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    print(f"--> Saved subsample to: {os.path.abspath(args.output_path)}")

    # Print summary statistics
    print_summary_statistics(
        total_brand_replies=total_brand_replies,
        total_valid_pairs=total_valid_pairs,
        subsample_df=subsample_df,
    )
    print(f"Completed in {time.time() - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
