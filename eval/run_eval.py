"""Phase 8 evaluation harness for the held-out AmazonHelp golden set."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eval.baselines import MajorityClassBaseline, TFIDFBaseline
from eval.judge_agreement import create_human_scores_template, load_human_scores, run_agreement_analysis
from eval.llm_judge import LLMJudge
from eval.metrics import compute_classification_metrics, format_confusion_matrix_markdown, generate_confusion_matrix
from src.classify import LLMIntentClassifier, MajorityClassifier, TFIDFClassifier
from src.intents import get_all_intents
from src.pipeline import SupportAgentPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLDEN_SET_PATH = PROJECT_ROOT / "golden_set" / "golden_set.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "results" / "summary.md"
DEFAULT_JSON_PATH = PROJECT_ROOT / "eval" / "results" / "evaluation.json"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "eval" / "results" / "judge_cache.json"
SILVER_PATH = PROJECT_ROOT / "data" / "processed" / "silver_labels.csv"
TFIDF_MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "tfidf_model.joblib"


def _clean_id(value: Any) -> str:
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def load_golden_set(csv_path: str = str(DEFAULT_GOLDEN_SET_PATH)) -> pd.DataFrame:
    """Load exactly 200 labelled golden rows without mutating the source file."""
    golden = pd.read_csv(csv_path, usecols=["tweet_id", "customer_text", "my_intent_label"])
    required = {"tweet_id", "customer_text", "my_intent_label"}
    missing = required - set(golden.columns)
    if missing:
        raise ValueError(f"Golden set missing columns: {sorted(missing)}")
    if len(golden) != 200:
        raise ValueError(f"Expected 200 golden rows, found {len(golden)}")
    taxonomy = set(get_all_intents())
    labels = set(golden["my_intent_label"].dropna().astype(str))
    unknown = labels - taxonomy
    if unknown:
        raise ValueError(f"Golden set contains unknown intent labels: {sorted(unknown)}")
    if golden["my_intent_label"].isna().any():
        raise ValueError("Golden set contains unlabeled rows")
    return golden.copy()


def _classifier_predictions(classifier: Any, texts: List[str]) -> List[str]:
    predictions = []
    for text in texts:
        result = classifier.classify(text)
        predictions.append(str(result.get("intent", "other")))
    return predictions


def _load_classifiers() -> Dict[str, Any]:
    """Load existing classifier implementations and non-golden artifacts."""
    classifiers: Dict[str, Any] = {}
    if SILVER_PATH.exists():
        silver = pd.read_csv(SILVER_PATH)
        majority = MajorityClassBaseline().fit(silver["silver_intent"].astype(str).tolist())
        classifiers["Majority baseline"] = majority
    else:
        classifiers["Majority baseline"] = MajorityClassifier()

    if TFIDF_MODEL_PATH.exists():
        classifiers["TF-IDF + Logistic Regression"] = TFIDFBaseline(model_path=str(TFIDF_MODEL_PATH))
    else:
        classifiers["TF-IDF + Logistic Regression"] = TFIDFClassifier(model_path=str(TFIDF_MODEL_PATH))
    classifiers["Main LLM classifier"] = LLMIntentClassifier(model="claude-sonnet-4-6")
    return classifiers


def _judge_cache_key(record: Dict[str, Any]) -> str:
    payload = json.dumps(record, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_cache(path: Path, cache: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def run_full_evaluation(
    golden_df: pd.DataFrame,
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run classification and reply evaluation without using golden labels for fitting."""
    if len(golden_df) != 200 and sample_size is None:
        raise ValueError("Full evaluation requires the validated 200-row golden set")
    evaluation_df = golden_df.head(sample_size).copy() if sample_size else golden_df.copy()
    labels = get_all_intents()
    texts = evaluation_df["customer_text"].fillna("").astype(str).tolist()
    y_true = evaluation_df["my_intent_label"].astype(str).tolist()

    classification_results: Dict[str, Any] = {}
    for name, classifier in _load_classifiers().items():
        predictions = _classifier_predictions(classifier, texts)
        classification_results[name] = compute_classification_metrics(y_true, predictions, labels)

    pipeline = SupportAgentPipeline()
    pipeline_rows: List[Dict[str, Any]] = []
    golden_ids = {_clean_id(value) for value in golden_df["tweet_id"].tolist()}
    for _, row in evaluation_df.iterrows():
        result = pipeline.process(str(row["customer_text"]))
        retrieved_ids = {_clean_id(item.get("tweet_id")) for item in result["retrieved_examples"]}
        overlap = retrieved_ids.intersection(golden_ids)
        if overlap:
            raise AssertionError(f"DATA LEAKAGE DETECTED in retrieval results: {sorted(overlap)}")
        pipeline_rows.append({"tweet_id": _clean_id(row["tweet_id"]), **result})

    judge_results: List[Dict[str, Any]] = []
    judge_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    cache_path = Path(output_path).parent / "judge_cache.json"
    cache = _load_cache(cache_path)
    judge = LLMJudge(model="claude-sonnet-4-6") if judge_available else None
    for row in pipeline_rows:
        judge_input = {
            "customer_text": row["customer_text"],
            "intent": row["intent"],
            "retrieved_examples": row["retrieved_examples"],
            "reply": row["reply"],
        }
        key = _judge_cache_key(judge_input)
        if key in cache:
            score = cache[key]
        elif judge is not None:
            score = judge.evaluate_reply(
                row["customer_text"],
                row["reply"],
                classified_intent=row["intent"],
                retrieved_examples=row["retrieved_examples"],
            ).to_dict()
            cache[key] = score
        else:
            score = {
                "available": False,
                "relevance": None,
                "groundedness": None,
                "tone": None,
                "conciseness": None,
                "overall": None,
                "reason": "LLM judge unavailable: ANTHROPIC_API_KEY is not set.",
            }
        judge_results.append({"tweet_id": row["tweet_id"], **score})
    _save_cache(cache_path, cache)

    agreement_frame = pd.DataFrame(judge_results)
    if "overall" in agreement_frame:
        agreement_frame["judge_overall_score"] = agreement_frame["overall"]
    for dimension in ("relevance", "groundedness", "tone", "conciseness"):
        agreement_frame[f"judge_{dimension}"] = agreement_frame[dimension]
    human_scores_path = Path(output_path).parent / "human_scores.csv"
    human_template_path = Path(output_path).parent / "human_scores_template.csv"
    if human_scores_path.exists():
        human_scores = load_human_scores(str(human_scores_path))
        agreement_frame = agreement_frame.merge(human_scores, on="tweet_id", how="inner")
    elif not human_template_path.exists():
        template_source = pd.DataFrame(
            [
                {
                    "tweet_id": row["tweet_id"],
                    "customer_text": row["customer_text"],
                    "reply": row["reply"],
                }
                for row in pipeline_rows
            ]
        )
        create_human_scores_template(
            template_source,
            str(human_template_path),
            sample_size=min(40, len(template_source)),
        )
    agreement = run_agreement_analysis(agreement_frame)
    results: Dict[str, Any] = {
        "metadata": {
            "golden_set_path": str(DEFAULT_GOLDEN_SET_PATH),
            "rows_evaluated": len(evaluation_df),
            "taxonomy_labels": labels,
            "golden_set_held_out": True,
            "llm_judge_available": judge_available,
        },
        "classification": classification_results,
        "reply_quality": {
            "scores": judge_results,
            "available": judge_available,
            "message": "" if judge_available else "LLM judge results unavailable; no scores fabricated.",
        },
        "human_agreement": agreement,
        "pipeline_results": pipeline_rows,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(generate_results_markdown(results), encoding="utf-8")
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(_json_safe(results), indent=2), encoding="utf-8")
    return results


def generate_results_markdown(results: Dict[str, Any]) -> str:
    """Render a compact human-readable evaluation summary."""
    lines = ["# Phase 8 Evaluation Results", "", "## Classification Results", "", "| Classifier | Accuracy | Macro F1 | Weighted F1 |", "|---|---:|---:|---:|"]
    for name, metrics in results["classification"].items():
        lines.append(f"| {name} | {metrics['accuracy']:.4f} | {metrics['macro_f1']:.4f} | {metrics['weighted_f1']:.4f} |")
    lines.extend(["", "### Per-Intent Metrics", ""])
    for name, metrics in results["classification"].items():
        lines.append(f"#### {name}")
        lines.append("| Intent | Precision | Recall | F1 | Support |")
        lines.append("|---|---:|---:|---:|---:|")
        for intent, values in metrics["per_intent"].items():
            lines.append(f"| {intent} | {values['precision']:.4f} | {values['recall']:.4f} | {values['f1']:.4f} | {values['support']} |")
        cm = pd.DataFrame(metrics["confusion_matrix"], index=metrics["labels"], columns=metrics["labels"])
        lines.extend(["", "Confusion matrix:", "", format_confusion_matrix_markdown(cm), ""])
    lines.extend(["## Reply Quality Results", ""])
    reply_quality = results["reply_quality"]
    if not reply_quality["available"]:
        lines.append(reply_quality["message"])
    else:
        available_scores = [score for score in reply_quality["scores"] if score.get("available") and score.get("overall") is not None]
        for dimension in ("relevance", "groundedness", "tone", "conciseness", "overall"):
            values = [score[dimension] for score in available_scores]
            average = sum(values) / len(values) if values else float("nan")
            lines.append(f"- {dimension}: mean {average:.3f} (n={len(values)})")
    lines.extend(["", "## Human Agreement", ""])
    agreement = results["human_agreement"]
    if not agreement.get("available"):
        lines.append("human scores not available")
    else:
        for key, value in agreement.items():
            lines.append(f"- {key}: {value}")
    lines.extend(["", "## Safety", "", "The golden set was read for evaluation only. Its labels were not used for fitting, silver-label generation, or retrieval indexing.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 8 evaluation harness")
    parser.add_argument("--golden-set", default=str(DEFAULT_GOLDEN_SET_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()
    golden = load_golden_set(args.golden_set)
    results = run_full_evaluation(golden, args.output, args.sample_size)
    print(generate_results_markdown(results))
    print(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
