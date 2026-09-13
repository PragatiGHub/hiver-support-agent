"""Embedding-based retrieval of historical AmazonHelp support interactions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "processed" / "amazonhelp_subsample.csv"
DEFAULT_GOLDEN_PATH = PROJECT_ROOT / "golden_set" / "golden_set.csv"
DEFAULT_INDEX_PATH = PROJECT_ROOT / ".cache" / "amazonhelp_retrieval"


def _normalize_id(value: Any) -> Optional[str]:
    """Normalize CSV IDs so numeric-looking IDs compare consistently."""
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text or None


@dataclass
class RetrievedContext:
    """A historical customer message, reply, and similarity score."""

    customer_text: str
    brand_reply_text: str
    similarity: float
    tweet_id: Optional[str] = None
    thread_id: Optional[str] = None
    created_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def customer_query(self) -> str:
        return self.customer_text

    @property
    def brand_response(self) -> str:
        return self.brand_reply_text

    @property
    def similarity_score(self) -> float:
        return self.similarity

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "tweet_id": self.tweet_id,
            "customer_text": self.customer_text,
            "brand_reply_text": self.brand_reply_text,
            "similarity": float(self.similarity),
        }
        if self.thread_id is not None:
            result["thread_id"] = self.thread_id
        if self.created_at is not None:
            result["created_at"] = self.created_at
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class HistoricalTweetRetriever:
    """Dense retriever over non-golden historical AmazonHelp conversations."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        corpus_path: Union[str, Path] = DEFAULT_CORPUS_PATH,
        golden_path: Union[str, Path] = DEFAULT_GOLDEN_PATH,
        index_path: Union[str, Path] = DEFAULT_INDEX_PATH,
    ):
        self.model_name = model_name
        self.corpus_path = Path(corpus_path)
        self.golden_path = Path(golden_path)
        self.index_path = Path(index_path)
        self.model = SentenceTransformer(model_name)
        self.embeddings: Optional[np.ndarray] = None
        self.records = pd.DataFrame()

        if self.index_path.with_suffix(".npz").exists() and self.index_path.with_suffix(".pkl").exists():
            self.load_index(str(self.index_path))

    def index_historical_conversations(self, pairs_df: pd.DataFrame) -> None:
        """Filter leakage and encode the historical customer messages once."""
        required = {"tweet_id", "customer_text", "brand_reply_text"}
        missing = required - set(pairs_df.columns)
        if missing:
            raise ValueError(f"Historical data is missing columns: {sorted(missing)}")

        records = pairs_df.copy()
        records["tweet_id"] = records["tweet_id"].map(_normalize_id)
        records = records[records["tweet_id"].notna()].copy()
        records["customer_text"] = records["customer_text"].fillna("").astype(str)
        records["brand_reply_text"] = records["brand_reply_text"].fillna("").astype(str)
        records = records[records["customer_text"].str.strip().ne("")].reset_index(drop=True)
        records = records[~records["tweet_id"].isin(self._read_golden_ids())].reset_index(drop=True)

        self.records = records
        self.embeddings = self.model.encode(
            records["customer_text"].tolist(),
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def save_index(self, index_path: str) -> None:
        """Persist normalized embeddings and metadata using ``index_path`` as a prefix."""
        if self.embeddings is None:
            raise ValueError("Cannot save an empty retrieval index")
        prefix = Path(index_path)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(prefix.with_suffix(".npz"), embeddings=self.embeddings)
        self.records.to_pickle(prefix.with_suffix(".pkl"))

    def load_index(self, index_path: str) -> None:
        """Load a previously persisted embedding matrix and metadata."""
        prefix = Path(index_path)
        with np.load(prefix.with_suffix(".npz")) as data:
            self.embeddings = data["embeddings"].astype("float32")
        self.records = pd.read_pickle(prefix.with_suffix(".pkl"))
        if len(self.records) != len(self.embeddings):
            raise ValueError("Retrieval index metadata and embeddings have different lengths")

    def retrieve_similar(
        self, query: str, top_k: int = 3, threshold: float = -1.0
    ) -> List[RetrievedContext]:
        """Return the top-k cosine-similar historical interactions."""
        if top_k < 1:
            return []
        self._ensure_index()
        if self.embeddings is None or self.records.empty:
            return []

        query_embedding = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        with np.errstate(over="ignore", invalid="ignore"):
            scores = self.embeddings @ query_embedding
        results: List[RetrievedContext] = []
        for index in np.argsort(-scores):
            score = float(scores[index])
            if not np.isfinite(score) or score < threshold:
                continue
            row = self.records.iloc[int(index)]
            results.append(
                RetrievedContext(
                    customer_text=str(row["customer_text"]),
                    brand_reply_text=str(row["brand_reply_text"]),
                    similarity=score,
                    tweet_id=_normalize_id(row["tweet_id"]),
                    thread_id=_normalize_id(row["thread_id"]) if "thread_id" in row else None,
                    created_at=(
                        str(row["created_at"])
                        if "created_at" in row and pd.notna(row["created_at"])
                        else None
                    ),
                )
            )
            if len(results) == top_k:
                break
        return results

    def _read_golden_ids(self) -> set:
        if not self.golden_path.exists():
            return set()
        golden = pd.read_csv(self.golden_path, usecols=["tweet_id"])
        return {tweet_id for tweet_id in golden["tweet_id"].map(_normalize_id) if tweet_id}

    def _ensure_index(self) -> None:
        if self.embeddings is None:
            self.index_historical_conversations(pd.read_csv(self.corpus_path))
            self.save_index(str(self.index_path))


_DEFAULT_RETRIEVER: Optional[HistoricalTweetRetriever] = None


def retrieve_similar(customer_text: str, k: int = 3) -> List[Dict[str, Any]]:
    """Retrieve default-corpus examples as serializable dictionaries."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = HistoricalTweetRetriever()
    return [
        context.to_dict()
        for context in _DEFAULT_RETRIEVER.retrieve_similar(customer_text, top_k=k)
    ]
