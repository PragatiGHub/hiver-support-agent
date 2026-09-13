"""Build golden set template for manual annotation.

Samples 200 rows from `data/processed/amazonhelp_subsample.csv` using stratified
sampling across date ranges, and outputs `golden_set/golden_set_template.csv`.

Features:
- Stratified sampling across 5 balanced date quantiles to ensure temporal diversity.
- Heuristic detection of edge case candidates (~15-20% of rows):
  * non_english: CJK scripts or common foreign language keywords
  * very_short_text: <35 chars or <=5 words
  * all_caps_or_sarcasm: >55% uppercase on >=15 chars or overt sarcasm phrases
  * multiple_issues: queries linking multiple disparate customer complaints
- Pre-fills `notes` column with `[EDGE CASE: <category>]` for flagged rows,
  leaving `notes` and `my_intent_label` blank for standard rows.
- Schema: `tweet_id, customer_text, brand_reply_text, my_intent_label, notes`
"""

import argparse
import csv
import os
import re
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "amazonhelp_subsample.csv"
)
DEFAULT_OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "golden_set", "golden_set_template.csv"
)
DEFAULT_SAMPLE_SIZE = 200
DEFAULT_RANDOM_SEED = 42
DEFAULT_DATE_BINS = 5


def detect_edge_case(text: str) -> str:
    """Classify whether a tweet exhibits characteristics of an edge case.

    Heuristics:
    1. non_english: CJK, Cyrillic, Arabic or common European customer care words.
    2. very_short_text: <35 characters or <= 5 words excluding mentions.
    3. all_caps_or_sarcasm: >55% uppercase on >=15 alpha chars or overt sarcasm.
    4. multiple_issues: Conjunctions linking multiple distinct issues.

    Args:
        text: Customer tweet text.

    Returns:
        str: Edge case category label if triggered, else empty string.
    """
    if not isinstance(text, str):
        return ""

    cleaned = re.sub(r"@\w+", "", text).strip()

    # 1. Non-English (CJK, Cyrillic, Arabic or common European customer care words)
    if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\u0600-\u06ff\u0400-\u04ff]", text):
        return "non_english"
    if re.search(
        r"\b(hola|ayuda|por favor|merci|bonjour|bitte|danke|grazie|ciao|pedido|hilfe|buenos dias)\b",
        cleaned,
        re.IGNORECASE,
    ):
        return "non_english"

    # 2. Very short text (<35 characters or <= 5 words)
    words = cleaned.split()
    if len(words) <= 5 or len(cleaned) < 35:
        return "very_short_text"

    # 3. All-caps rant or overt sarcasm
    alpha_chars = [c for c in cleaned if c.isalpha()]
    if len(alpha_chars) >= 15:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
        if upper_ratio >= 0.55:
            return "all_caps_or_sarcasm"

    if re.search(
        r"(thanks for nothing|great job amazon|wow thanks|what a joke|bravo amazon|slow claps?|worst customer service|ridiculous)",
        cleaned,
        re.IGNORECASE,
    ):
        return "all_caps_or_sarcasm"

    # 4. Multiple issues / multi-part inquiry
    multi_patterns = [
        r"\b(and also|in addition|plus my|as well as|not only|secondly)\b",
        r"\b(damaged|broken|wrong)\b.*\b(refund|charged|money|replace)\b",
        r"\b(delay|late|where is|not arrived)\b.*\b(cancel|refund|return|money)\b",
    ]
    for pat in multi_patterns:
        if re.search(pat, cleaned, re.IGNORECASE):
            return "multiple_issues"

    return ""


def sample_stratified_by_date(
    df: pd.DataFrame,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    num_bins: int = DEFAULT_DATE_BINS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    """Sample records evenly across date range quantiles.

    Args:
        df: Input processed subsample dataframe.
        sample_size: Total records to sample.
        num_bins: Number of quantile date strata.
        random_seed: Fixed seed for reproducibility.

    Returns:
        pd.DataFrame: Stratified sample of size `sample_size`.
    """
    df = df.copy()
    df["parsed_datetime"] = pd.to_datetime(
        df["created_at"],
        format="%a %b %d %H:%M:%S %z %Y",
        errors="coerce",
    )

    valid_df = df.dropna(subset=["parsed_datetime"]).copy()

    # Create balanced quantile strata
    valid_df["date_bin"] = pd.qcut(
        valid_df["parsed_datetime"],
        q=num_bins,
        labels=[f"Bin_{i+1}" for i in range(num_bins)],
    )

    per_bin_count = sample_size // num_bins
    remainder = sample_size % num_bins

    samples = []
    for i, (bin_name, group) in enumerate(valid_df.groupby("date_bin", observed=False)):
        n_to_sample = per_bin_count + (1 if i < remainder else 0)
        n_to_sample = min(n_to_sample, len(group))
        sample_group = group.sample(n=n_to_sample, random_state=random_seed + i)
        samples.append(sample_group)

    sampled_df = pd.concat(samples, ignore_index=True)
    return sampled_df


def build_golden_set_template(
    input_path: str = DEFAULT_INPUT_PATH,
    output_path: str = DEFAULT_OUTPUT_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Generate golden set annotation template with edge-case heuristics.

    Args:
        input_path: Path to amazonhelp_subsample.csv.
        output_path: Path to golden_set_template.csv.
        sample_size: Number of records to sample (default 200).
        random_seed: Random seed.

    Returns:
        Tuple: (template_df, edge_case_counts_dict)
    """
    if not os.path.exists(input_path):
        print(f"[ERROR] Input subsample not found at: {input_path}")
        print("Please run `python data/filter_brand.py` first.")
        sys.exit(1)

    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} candidate pairs from {input_path}")

    # Stratified sampling across date ranges
    sampled_df = sample_stratified_by_date(
        df,
        sample_size=sample_size,
        num_bins=DEFAULT_DATE_BINS,
        random_seed=random_seed,
    )

    # Detect edge cases
    edge_cases = sampled_df["customer_text"].apply(detect_edge_case)
    sampled_df["edge_case_type"] = edge_cases

    # Construct columns
    # Schema: tweet_id, customer_text, brand_reply_text, my_intent_label (blank), notes (blank or edge case flag)
    sampled_df["my_intent_label"] = ""
    sampled_df["notes"] = sampled_df["edge_case_type"].apply(
        lambda x: f"[EDGE CASE: {x}]" if x else ""
    )

    target_columns = [
        "tweet_id",
        "customer_text",
        "brand_reply_text",
        "my_intent_label",
        "notes",
    ]
    template_df = sampled_df[target_columns].copy()

    # Save to CSV with strict quoting
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    template_df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )

    edge_counts = edge_cases[edge_cases != ""].value_counts().to_dict()
    return template_df, edge_counts


def main() -> None:
    """CLI entry point for golden set template creation."""
    parser = argparse.ArgumentParser(
        description="Sample stratified golden set template with edge case candidate flags."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to source amazonhelp_subsample.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to destination golden_set_template.csv",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of rows to sample (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for sampling (default: 42)",
    )

    args = parser.parse_args()

    template_df, edge_counts = build_golden_set_template(
        input_path=args.input,
        output_path=args.output,
        sample_size=args.sample_size,
        random_seed=args.seed,
    )

    total_flagged = sum(edge_counts.values())
    pct_flagged = (total_flagged / len(template_df)) * 100

    print("\n" + "=" * 65)
    print("         GOLDEN SET TEMPLATE SAMPLING SUMMARY         ")
    print("=" * 65)
    print(f"Total Rows Sampled:        {len(template_df)}")
    print(f"Date Stratification:       {DEFAULT_DATE_BINS} balanced quantiles")
    print(f"Edge Case Candidates:      {total_flagged} / {len(template_df)} ({pct_flagged:.1f}%)")
    print("\nEdge Case Breakdown:")
    for category, count in sorted(edge_counts.items(), key=lambda x: -x[1]):
        print(f"  - {category:<25}: {count:>3} ({count/len(template_df):.1%})")
    print("-" * 65)
    print(f"Output Saved To:           {os.path.abspath(args.output)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
