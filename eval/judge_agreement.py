"""Agreement utilities for LLM judge and manually entered human scores."""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, mean_squared_error

DIMENSIONS = ["relevance", "groundedness", "tone", "conciseness"]


def create_human_scores_template(
    evaluation_results: pd.DataFrame,
    output_path: str,
    sample_size: int = 40,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Write a blank template; no human values are generated automatically."""
    required = ["tweet_id", "customer_text", "reply"]
    missing = set(required) - set(evaluation_results.columns)
    if missing:
        raise ValueError(f"Evaluation results missing columns: {sorted(missing)}")
    sample = evaluation_results.head(sample_size).copy()
    template = sample[required].copy()
    for dimension in DIMENSIONS:
        template[f"human_{dimension}"] = pd.NA
    template.to_csv(output_path, index=False)
    return template


def load_human_scores(path: str) -> pd.DataFrame:
    """Load manually entered scores and validate the four 1-5 dimensions."""
    scores = pd.read_csv(path)
    required = {"tweet_id", *(f"human_{dimension}" for dimension in DIMENSIONS)}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"Human score file missing columns: {sorted(missing)}")
    for dimension in DIMENSIONS:
        values = pd.to_numeric(scores[f"human_{dimension}"], errors="coerce")
        invalid = values.notna() & ~values.between(1, 5)
        if invalid.any():
            raise ValueError(f"human_{dimension} contains ratings outside 1-5")
        scores[f"human_{dimension}"] = values
    return scores


def compute_correlation_metrics(
    human_scores: List[float], judge_scores: List[float]
) -> Dict[str, float]:
    """Compute ordinal-friendly rank correlation plus error metrics."""
    if len(human_scores) != len(judge_scores) or not human_scores:
        raise ValueError("Paired score lists must be non-empty and equal in length")
    if len(set(human_scores)) > 1 and len(set(judge_scores)) > 1:
        pearson = stats.pearsonr(human_scores, judge_scores)
        spearman = stats.spearmanr(human_scores, judge_scores)
        pearson_r, pearson_p = float(pearson.statistic), float(pearson.pvalue)
        spearman_rho, spearman_p = float(spearman.statistic), float(spearman.pvalue)
    else:
        pearson_r = pearson_p = spearman_rho = spearman_p = float("nan")
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "mae": float(mean_absolute_error(human_scores, judge_scores)),
        "rmse": float(np.sqrt(mean_squared_error(human_scores, judge_scores))),
        "n": float(len(human_scores)),
    }


def compute_discrete_agreement(
    human_scores: List[float], judge_scores: List[float], threshold: float = 3.5
) -> Dict[str, float]:
    """Compute ordinal exact agreement and binary acceptable/unacceptable kappa."""
    human_binary = [score >= threshold for score in human_scores]
    judge_binary = [score >= threshold for score in judge_scores]
    return {
        "cohen_kappa": float(cohen_kappa_score(human_binary, judge_binary)),
        "binary_accuracy": float(np.mean(np.array(human_binary) == np.array(judge_binary))),
        "exact_rating_agreement": float(np.mean(np.array(human_scores) == np.array(judge_scores))),
    }


def analyze_bias_and_calibration(
    human_scores: List[float], judge_scores: List[float]
) -> Dict[str, float]:
    """Measure whether the judge tends to score higher or lower than humans."""
    difference = np.asarray(judge_scores, dtype=float) - np.asarray(human_scores, dtype=float)
    return {"mean_delta_judge_minus_human": float(difference.mean()), "std_delta": float(difference.std(ddof=0))}


def run_agreement_analysis(
    evaluation_results_df: pd.DataFrame,
    human_col: str = "human_quality_score",
    judge_col: str = "judge_overall_score",
) -> Dict[str, Any]:
    """Run agreement on paired rows, or report that human scores are absent."""
    dimension_results: Dict[str, Any] = {}
    available_dimensions = []
    for dimension in DIMENSIONS:
        human_dimension = f"human_{dimension}"
        judge_dimension = f"judge_{dimension}"
        if human_dimension in evaluation_results_df and judge_dimension in evaluation_results_df:
            paired_dimension = evaluation_results_df[[human_dimension, judge_dimension]].dropna()
            if not paired_dimension.empty:
                human_values = paired_dimension[human_dimension].astype(float).tolist()
                judge_values = paired_dimension[judge_dimension].astype(float).tolist()
                dimension_results[dimension] = {
                    "n": len(paired_dimension),
                    **compute_correlation_metrics(human_values, judge_values),
                    **compute_discrete_agreement(human_values, judge_values),
                    **analyze_bias_and_calibration(human_values, judge_values),
                }
                available_dimensions.append(dimension)
    if available_dimensions:
        return {
            "available": True,
            "dimensions": dimension_results,
            "n_dimensions_available": len(available_dimensions),
        }

    if human_col not in evaluation_results_df or judge_col not in evaluation_results_df:
        return {"available": False, "message": "human scores not available"}
    paired = evaluation_results_df[[human_col, judge_col]].dropna()
    if paired.empty:
        return {"available": False, "message": "human scores not available"}
    human = paired[human_col].astype(float).tolist()
    judge = paired[judge_col].astype(float).tolist()
    result = {"available": True, "n": len(paired)}
    result.update(compute_correlation_metrics(human, judge))
    result.update(compute_discrete_agreement(human, judge))
    result.update(analyze_bias_and_calibration(human, judge))
    return result
