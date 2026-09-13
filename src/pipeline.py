"""End-to-end Phase 7 pipeline for AmazonHelp customer support."""

import time
from typing import Any, Dict, List, Optional

from src.classify import BaseClassifier, LLMIntentClassifier
from src.draft_reply import DraftReply, ReplyDrafter
from src.escalate import decide_escalation
from src.retrieve import HistoricalTweetRetriever, RetrievedContext


SAFE_FALLBACK_REPLY = (
    "We couldn't generate an automatic reply. Please contact AmazonHelp support "
    "with more details so the team can assist."
)


class SupportAgentPipeline:
    """Run classification, retrieval, drafting, and escalation in order."""

    def __init__(
        self,
        classifier: Optional[BaseClassifier] = None,
        retriever: Optional[HistoricalTweetRetriever] = None,
        drafter: Optional[ReplyDrafter] = None,
    ):
        self.classifier = classifier or LLMIntentClassifier()
        self.retriever = retriever or HistoricalTweetRetriever()
        self.drafter = drafter or ReplyDrafter()

    def process(self, customer_text: str) -> Dict[str, Any]:
        """Execute the pipeline and return the public result dictionary."""
        started = time.time()
        errors: List[str] = []

        classification = self._classify(customer_text, errors)
        intent = classification["intent"]
        confidence = classification["confidence"]

        retrieved_contexts = self._retrieve(customer_text, errors)
        drafted_reply = self._draft(customer_text, intent, retrieved_contexts, errors)

        try:
            if any(
                error.startswith(("retrieval failed:", "reply generation failed:"))
                for error in errors
            ):
                escalation = {
                    "decision": "escalate",
                    "reason": "Escalate because retrieval or reply generation failed, so a grounded automated response is unavailable.",
                }
            else:
                escalation = decide_escalation(
                    customer_text,
                    intent,
                    confidence,
                    drafted_reply,
                )
        except Exception as exc:
            errors.append(f"escalation failed: {exc}")
            escalation = {
                "decision": "escalate",
                "reason": "Escalate because escalation checks failed and the case must be reviewed safely by a human.",
            }

        reply, influenced_by = self._reply_fields(drafted_reply, retrieved_contexts)
        result: Dict[str, Any] = {
            "customer_text": customer_text,
            "intent": intent,
            "confidence": confidence,
            "retrieved_examples": [self._context_to_dict(item) for item in retrieved_contexts],
            "reply": reply,
            "influenced_by": influenced_by,
            "escalation_decision": escalation["decision"],
            "escalation_reason": escalation["reason"],
            "latency_ms": round((time.time() - started) * 1000, 2),
        }
        if errors:
            result["errors"] = errors
        return result

    def process_batch(self, customer_texts: List[str]) -> List[Dict[str, Any]]:
        """Process messages sequentially, preserving input order."""
        return [self.process(customer_text) for customer_text in customer_texts]

    def _classify(self, customer_text: str, errors: List[str]) -> Dict[str, str]:
        try:
            result = self.classifier.classify(customer_text)
            if not isinstance(result, dict):
                raise TypeError("classifier returned a non-dictionary result")
            intent = result.get("intent")
            confidence = result.get("confidence")
            if not isinstance(intent, str) or not intent:
                raise ValueError("classifier result did not include an intent")
            if not isinstance(confidence, str) or not confidence:
                raise ValueError("classifier result did not include confidence")
            return {"intent": intent, "confidence": confidence}
        except Exception as exc:
            errors.append(f"classification failed: {exc}")
            return {"intent": "other", "confidence": "low"}

    def _retrieve(
        self, customer_text: str, errors: List[str]
    ) -> List[RetrievedContext]:
        try:
            contexts = self.retriever.retrieve_similar(customer_text, top_k=3)
            return list(contexts or [])[:3]
        except Exception as exc:
            errors.append(f"retrieval failed: {exc}")
            return []

    def _draft(
        self,
        customer_text: str,
        intent: str,
        retrieved_contexts: List[RetrievedContext],
        errors: List[str],
    ) -> Any:
        try:
            return self.drafter.draft(customer_text, intent, retrieved_contexts)
        except Exception as exc:
            errors.append(f"reply generation failed: {exc}")
            fallback = getattr(self.drafter, "_fallback", None)
            if callable(fallback):
                return fallback([])
            return SAFE_FALLBACK_REPLY

    @staticmethod
    def _context_to_dict(context: Any) -> Dict[str, Any]:
        if isinstance(context, dict):
            return context
        to_dict = getattr(context, "to_dict", None)
        if callable(to_dict):
            return to_dict()
        return {
            "tweet_id": getattr(context, "tweet_id", None),
            "customer_text": getattr(context, "customer_text", ""),
            "brand_reply_text": getattr(context, "brand_reply_text", ""),
            "similarity": float(getattr(context, "similarity", 0.0)),
        }

    @classmethod
    def _reply_fields(
        cls, drafted_reply: Any, retrieved_contexts: List[RetrievedContext]
    ) -> Any:
        if isinstance(drafted_reply, dict):
            reply = drafted_reply.get("reply", drafted_reply.get("reply_text"))
            influenced_by = drafted_reply.get("influenced_by", [])
        else:
            reply = getattr(drafted_reply, "reply_text", drafted_reply)
            influenced_by = getattr(drafted_reply, "influenced_by", None)
        if not isinstance(reply, str) or not reply.strip():
            reply = SAFE_FALLBACK_REPLY
        if not influenced_by:
            influenced_by = []
            for context in retrieved_contexts[:3]:
                item = cls._context_to_dict(context)
                influenced_by.append(
                    {
                        "tweet_id": str(item.get("tweet_id")),
                        "similarity": float(item.get("similarity", 0.0)),
                    }
                )
        return reply, influenced_by


_DEFAULT_PIPELINE: Optional[SupportAgentPipeline] = None


def run_pipeline(customer_text: str) -> Dict[str, Any]:
    """Run the default AmazonHelp support pipeline for one customer message."""
    global _DEFAULT_PIPELINE
    if _DEFAULT_PIPELINE is None:
        _DEFAULT_PIPELINE = SupportAgentPipeline()
    return _DEFAULT_PIPELINE.process(customer_text)


if __name__ == "__main__":
    print(run_pipeline("My package has not arrived yet."))
