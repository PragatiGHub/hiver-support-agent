"""Baseline models and silver-label training pipeline for intent classification.

This module provides:
1. MajorityClassBaseline: Evaluates the trivial majority-class predictor.
2. TFIDFBaseline: Scikit-learn TF-IDF Vectorizer + Logistic Regression classifier.
3. Silver-Label Pipeline: Generates weak training labels on strictly non-golden
   examples from `data/processed/amazonhelp_subsample.csv`.
4. Strict Leakage Prevention: Asserts that golden_set/golden_set.csv is held out.

SILVER LABELS DOCUMENTATION:
- What they are: Silver labels are automatically generated pseudo-labels produced
  by either Claude (claude-sonnet-4-6) or structured heuristics on non-golden data.
- Why they are weak: Silver labels contain machine noise, systemic classifier biases,
  and unverified edge cases. They are NOT ground truth.
- Why golden set is held out: golden_set/golden_set.csv contains 200 manually verified
  human labels. It must NEVER be used during vectorizer fitting, model training,
  silver-label generation, or hyperparameter selection.
"""

import argparse
import csv
import json
import logging
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.classify import (
    DEFAULT_TFIDF_MODEL_PATH,
    LLMIntentClassifier,
    MajorityClassifier,
    TFIDFClassifier,
)
from src.intents import FALLBACK_INTENT, INTENT_MAP, get_all_intents

logger = logging.getLogger(__name__)

# Default file paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_SUBSAMPLE_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "amazonhelp_subsample.csv"
)
DEFAULT_GOLDEN_PATH = os.path.join(PROJECT_ROOT, "golden_set", "golden_set.csv")
DEFAULT_SILVER_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "silver_labels.csv"
)


# =====================================================================
# DATA LEAKAGE SAFETY HELPER
# =====================================================================
def get_non_golden_data(
    subsample_path: str = DEFAULT_SUBSAMPLE_PATH,
    golden_path: str = DEFAULT_GOLDEN_PATH,
) -> pd.DataFrame:
    """Load subsample dataset and strictly filter out all golden set examples.

    Args:
        subsample_path: Path to data/processed/amazonhelp_subsample.csv (6,000 rows).
        golden_path: Path to golden_set/golden_set.csv (200 rows).

    Returns:
        pd.DataFrame: Non-golden customer messages guaranteed to have 0 overlap with golden set.
    """
    if not os.path.exists(subsample_path):
        raise FileNotFoundError(f"Subsample dataset not found at {subsample_path}")

    sub_df = pd.read_csv(subsample_path)
    golden_ids: Set[str] = set()

    if os.path.exists(golden_path):
        golden_df = pd.read_csv(golden_path)
        if "tweet_id" in golden_df.columns:
            golden_ids = set(golden_df["tweet_id"].dropna().astype(str).str.split(".").str[0])

    # Strict exclusion
    clean_sub_ids = sub_df["tweet_id"].astype(str).str.split(".").str[0]
    non_golden_df = sub_df[~clean_sub_ids.isin(golden_ids)].copy().reset_index(drop=True)

    # Verification assertion
    overlap = set(non_golden_df["tweet_id"].astype(str).str.split(".").str[0]).intersection(golden_ids)
    assert len(overlap) == 0, f"DATA LEAKAGE DETECTED! Overlapping IDs: {overlap}"

    logger.info(
        f"Filtered non-golden dataset: {len(non_golden_df):,} examples "
        f"(excluded {len(golden_ids)} held-out golden set IDs)."
    )
    return non_golden_df


# =====================================================================
# SILVER-LABEL GENERATION
# =====================================================================
def heuristic_silver_label(text: str) -> Tuple[str, str]:
    """Pattern-based bootstrap labeling rule for non-golden examples when LLM API is unavailable.

    Args:
        text: Customer tweet text.

    Returns:
        Tuple[str, str]: (predicted_intent, confidence)
    """
    txt = text.lower()

    # 2. Account access / security
    if re.search(
        r"\b(login|log in|password|passcode|otp|verification code|account on hold|locked out|compromised|hacked|2fa|authenticator|sign in)\b",
        txt,
    ):
        return "account_access_security", "high"

    # 3. Refund request
    if re.search(
        r"\b(refund|money back|refund status|reimburse|where is my refund|got no refund|return my money)\b",
        txt,
    ):
        return "refund_request", "high"

    # 4. Billing / charge dispute
    if re.search(
        r"\b(charged|unauthorized|subscription|prime membership|deducted|extra charge|charged twice|double charge|cashback|promo code|billing|overcharge)\b",
        txt,
    ):
        return "billing_charge_dispute", "high"

    # 5. Return / replacement status
    if re.search(
        r"\b(replacement|replace|exchange|return pickup|return label|courier pickup|pickup agent|pick up product|return request)\b",
        txt,
    ):
        return "return_replacement_status", "high"

    # 6. Wrong or damaged item
    if re.search(
        r"\b(damaged|arrived damaged|damaged on arrival|broken|arrived broken|cracked|shattered|smashed|wrong item|wrong product|defective|scratched|faulty|damaged product|used item|fake item|torn|missing item|opened box|different item|割れてた|割れた)\b",
        txt,
    ):
        return "wrong_or_damaged_item", "high"

    # 7. Delivery delay or non-delivery
    if re.search(
        r"\b(delivery|delayed|delay|late|never delivered|never received|not delivered|didn\'t arrive|did not arrive|not arrived|hasn\'t arrived|have not received|haven\'t received|still waiting|still waiting for|waiting for my order|one day shipping|paid for one day shipping|delivery didn\'t happen|delivery failed|delivered to another resident|delivered to wrong address|wrong building|package went to another|package missing|tracking|courier|out for delivery|supposed to arrive|missing package|package not delivered|delivery date)\b",
        txt,
    ):
        return "delivery_delay_or_non_delivery", "high"

    # 8. Order status inquiry
    if re.search(
        r"\b(where is my order|where is my package|when will.*ship|when will my order ship|when will it ship|shipped yet|dispatch|preparing for dispatch|digital code|gift card code|order status|tracking my order|track my order|track order|tracking number|has my order shipped|has it shipped|shipping status|is it dispatched)\b",
        txt,
    ):
        return "order_status_inquiry", "medium"

    # 9. General complaint
    if re.search(
        r"\b(worst|worst service|pathetic|terrible|terrible service|horrible|useless|disappointed|extremely disappointed|very disappointed|unacceptable|cheat|cheaters|disgusting|ridiculous|ridiculous service|rude|fraud|awful|scam)\b",
        txt,
    ):
        return "general_complaint", "medium"

    return "other", "low"


def generate_silver_labels(
    sample_size: int = 500,
    random_seed: int = 42,
    use_llm: bool = True,
    output_path: str = DEFAULT_SILVER_PATH,
    subsample_path: str = DEFAULT_SUBSAMPLE_PATH,
    golden_path: str = DEFAULT_GOLDEN_PATH,
) -> pd.DataFrame:
    """Generate silver-label training dataset strictly on non-golden customer messages.

    Args:
        sample_size: Number of non-golden examples to label (default 500).
        random_seed: Fixed seed for reproducible sampling.
        use_llm: If True and ANTHROPIC_API_KEY is available, uses Claude (claude-sonnet-4-6).
                 If False or API key missing, uses bootstrap heuristics.
        output_path: Destination path for data/processed/silver_labels.csv.
        subsample_path: Path to amazonhelp_subsample.csv.
        golden_path: Path to golden_set.csv.

    Returns:
        pd.DataFrame: Generated silver labels dataframe.
    """
    non_golden_df = get_non_golden_data(subsample_path, golden_path)
    sample_n = min(sample_size, len(non_golden_df))
    sampled = non_golden_df.sample(n=sample_n, random_state=random_seed).copy().reset_index(drop=True)

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    intents = []
    confidences = []
    source = "claude-sonnet-4-6" if (use_llm and has_api_key) else "heuristic_bootstrap"

    if use_llm and has_api_key:
        print(f"Generating {sample_n} silver labels using Claude (claude-sonnet-4-6)...")
        llm = LLMIntentClassifier(model="claude-sonnet-4-6")
        for i, text in enumerate(sampled["customer_text"]):
            res = llm.classify(text)
            intents.append(res["intent"])
            confidences.append(res["confidence"])
            if (i + 1) % 50 == 0 or (i + 1) == sample_n:
                print(f"  Processed {i + 1}/{sample_n} non-golden rows...")
    else:
        print(f"Generating {sample_n} silver labels using rule-based bootstrap heuristic...")
        for text in sampled["customer_text"]:
            intent, conf = heuristic_silver_label(str(text))
            intents.append(intent)
            confidences.append(conf)

    silver_df = pd.DataFrame(
        {
            "tweet_id": sampled["tweet_id"],
            "customer_text": sampled["customer_text"],
            "silver_intent": intents,
            "silver_confidence": confidences,
            "label_source": source,
        }
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    silver_df.to_csv(
        output_path,
        index=False,
        quoting=csv.QUOTE_ALL,
        lineterminator="\n",
    )
    print(f"--> Saved silver-label dataset ({len(silver_df)} rows) to: {output_path}")
    print(f"--> Silver label distribution:\n{silver_df['silver_intent'].value_counts()}")
    return silver_df


# =====================================================================
# BASELINE EVALUATION CLASSES
# =====================================================================
class MajorityClassBaseline:
    """Trivial baseline model that learns the majority class from non-golden labels."""

    def __init__(self, majority_class: Optional[str] = None):
        self.majority_class = majority_class or "other"
        self.class_distribution: Dict[str, float] = {}

    def fit(self, labels: List[str]) -> "MajorityClassBaseline":
        """Learn majority class and empirical distribution from training labels."""
        if not labels:
            self.majority_class = "other"
            return self

        counts = Counter(labels)
        total = sum(counts.values())
        self.class_distribution = {k: v / total for k, v in counts.items()}
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, texts: List[str]) -> List[str]:
        """Predict the constant majority class for all texts."""
        return [self.majority_class] * len(texts)

    def classify(self, text: str) -> Dict[str, str]:
        """Common callable interface."""
        return {"intent": self.majority_class, "confidence": "low"}


class TFIDFBaseline:
    """Scikit-learn TF-IDF + LogisticRegression baseline model."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_TFIDF_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None

        if self.model_path and os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                logger.warning(f"Could not load TF-IDF model from {self.model_path}: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFBaseline":
        """Fit TF-IDF vectorizer and Logistic Regression on non-golden silver labels."""
        self.pipeline = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=5000,
                        sublinear_tf=True,
                        stop_words="english",
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )
        self.pipeline.fit(texts, labels)
        return self

    def save(self, model_path: Optional[str] = None) -> None:
        """Save fitted model to disk."""
        path = model_path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(self.pipeline, path)

    def predict(self, texts: List[str]) -> List[str]:
        """Predict intent labels for a list of texts."""
        if self.pipeline is None:
            return [FALLBACK_INTENT] * len(texts)
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """Compute posterior class probabilities."""
        if self.pipeline is None:
            return np.zeros((len(texts), len(get_all_intents())))
        return self.pipeline.predict_proba(texts)

    def classify(self, text: str) -> Dict[str, str]:
        """Common callable interface."""
        if not text or not str(text).strip():
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        if self.pipeline is None:
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        probs = self.pipeline.predict_proba([str(text)])[0]
        best_idx = int(np.argmax(probs))
        predicted_class = str(self.pipeline.classes_[best_idx])
        max_prob = float(probs[best_idx])

        if max_prob >= 0.50:
            confidence = "high"
        elif max_prob >= 0.25:
            confidence = "medium"
        else:
            confidence = "low"

        if predicted_class not in INTENT_MAP:
            predicted_class = FALLBACK_INTENT
            confidence = "low"

        return {"intent": predicted_class, "confidence": confidence}


def train_and_save_tfidf(
    silver_csv_path: str = DEFAULT_SILVER_PATH,
    model_path: str = DEFAULT_TFIDF_MODEL_PATH,
) -> TFIDFBaseline:
    """Train TF-IDF Logistic Regression on non-golden silver labels and save artifact.

    Args:
        silver_csv_path: Path to silver_labels.csv.
        model_path: Path to save tfidf_model.joblib.

    Returns:
        TFIDFBaseline: Fitted baseline instance.
    """
    if not os.path.exists(silver_csv_path):
        print(f"Silver label file not found at {silver_csv_path}. Generating now...")
        generate_silver_labels(output_path=silver_csv_path)

    df = pd.read_csv(silver_csv_path)
    texts = df["customer_text"].astype(str).tolist()
    labels = df["silver_intent"].astype(str).tolist()

    baseline = TFIDFBaseline(model_path=model_path)
    baseline.fit(texts, labels)
    baseline.save(model_path)
    print(f"--> Trained TF-IDF baseline on {len(texts)} silver-labeled examples.")
    print(f"--> Saved fitted model artifact to: {model_path}")
    return baseline


def main() -> None:
    """CLI to generate silver labels and train TF-IDF model on non-golden data."""
    parser = argparse.ArgumentParser(
        description="Build silver labels and train baseline models (strictly non-golden)."
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of non-golden examples for silver labeling",
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Claude (claude-sonnet-4-6) if ANTHROPIC_API_KEY is available",
    )
    args = parser.parse_args()

    print("=" * 65)
    print("      BUILDING BASELINES & SILVER LABELS (NON-GOLDEN DATA)     ")
    print("=" * 65)

    # 1. Generate silver labels
    generate_silver_labels(sample_size=args.sample_size, use_llm=args.use_llm)

    # 2. Train and persist TF-IDF baseline
    train_and_save_tfidf()

    print("=" * 65)
    print("Baseline setup complete. All models trained strictly on non-golden data.")
    print("=" * 65)


if __name__ == "__main__":
    main()
