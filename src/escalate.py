"""Deterministic Phase 6 escalation triage for AmazonHelp."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.intents import INTENT_MAP


HIGH_RISK_INTENTS = {"billing_charge_dispute", "account_access_security"}
LOW_CONFIDENCE = {"low", "very_low"}
MEDIUM_CONFIDENCE = {"medium", "moderate"}
HIGH_CONFIDENCE = {"high"}

LEGAL_OR_URGENT_PHRASES = (
    "lawyer",
    "legal action",
    "legal department",
    "fraud",
    "police",
    "lawsuit",
    "chargeback",
    "scam",
    "threaten",
    "threatening",
)
STRONG_NEGATIVE_PHRASES = (
    "never again",
    "worst",
    "terrible",
    "pathetic",
)


class EscalationReason(str, Enum):
    """Categorical reasons retained for compatibility with the earlier API."""

    NONE = "none"
    LOW_CLASSIFICATION_CONFIDENCE = "low_classification_confidence"
    LOW_RETRIEVAL_SIMILARITY = "low_retrieval_similarity"
    SENSITIVE_ACCOUNT_PII = "sensitive_account_pii"
    BILLING_OR_REFUND_DISPUTE = "billing_or_refund_dispute"
    PHYSICAL_SAFETY_OR_HARDWARE_HAZARD = "physical_safety_or_hardware_hazard"
    ANGRY_OR_HOSTILE_SENTIMENT = "angry_or_hostile_sentiment"
    LEGAL_OR_REGULATORY_THREAT = "legal_or_regulatory_threat"
    AMBIGUOUS_OR_MULTI_PART_QUERY = "ambiguous_or_multi_part_query"


@dataclass
class EscalationDecision:
    """Detailed compatibility result for callers using ``EscalationEngine``."""

    should_escalate: bool
    reason: EscalationReason
    explanation: str
    risk_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": "escalate" if self.should_escalate else "auto_handle",
            "reason": self.explanation,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def _confidence_level(confidence: Any) -> Optional[str]:
    if confidence is None:
        return None
    if isinstance(confidence, str):
        value = confidence.strip().lower()
        if value in LOW_CONFIDENCE:
            return "low"
        if value in MEDIUM_CONFIDENCE:
            return "medium"
        if value in HIGH_CONFIDENCE:
            return "high"
        try:
            confidence = float(value)
        except ValueError:
            return None
    if isinstance(confidence, (int, float)):
        if confidence < 0.5:
            return "low"
        if confidence < 0.75:
            return "medium"
        return "high"
    return None


def _present_phrases(text: str) -> List[str]:
    normalized = " ".join(str(text).lower().split())
    return [
        phrase
        for phrase in LEGAL_OR_URGENT_PHRASES + STRONG_NEGATIVE_PHRASES
        if phrase in normalized
    ]


def _has_reply(drafted_reply: Any) -> bool:
    if drafted_reply is None:
        return False
    if isinstance(drafted_reply, str):
        return bool(drafted_reply.strip())
    if isinstance(drafted_reply, dict):
        reply = drafted_reply.get("reply", drafted_reply.get("reply_text"))
        return isinstance(reply, str) and bool(reply.strip())
    reply = getattr(drafted_reply, "reply_text", None)
    return isinstance(reply, str) and bool(reply.strip())


def decide_escalation(
    text: Any,
    intent: Any,
    confidence: Any,
    drafted_reply: Any,
) -> Dict[str, str]:
    """Decide whether a customer message can be auto-handled.

    Confidence may be the classifier's ``high``/``medium``/``low`` label or a
    numeric score. Sentiment and urgency are deterministic phrase heuristics,
    not an ML sentiment model.
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "decision": "escalate",
            "reason": "Escalate because the customer message is missing or empty, so the issue cannot be assessed safely.",
        }

    if not _has_reply(drafted_reply):
        return {
            "decision": "escalate",
            "reason": "Escalate because no drafted reply is available for safe automated handling.",
        }

    if not isinstance(intent, str) or intent not in INTENT_MAP:
        return {
            "decision": "escalate",
            "reason": "Escalate because the intent is unknown, so the request cannot be routed safely.",
        }

    level = _confidence_level(confidence)
    phrases = _present_phrases(text)
    if level == "low" or level is None:
        confidence_reason = "classifier confidence is low or unavailable"
    else:
        confidence_reason = ""

    if intent == "billing_charge_dispute":
        reason = "Escalate because the issue involves a possible unauthorized billing transaction that requires account-specific investigation."
        if level in {"low", "medium"}:
            reason += " Classifier confidence is not high."
        if phrases:
            reason += f" The customer also uses high-risk language ({', '.join(repr(p) for p in phrases)})."
        return {"decision": "escalate", "reason": reason}

    if intent == "account_access_security":
        reason = "Escalate because the customer reports a potentially compromised account and account-level security action may be required."
        if level in {"low", "medium"}:
            reason += " Classifier confidence is not high."
        if phrases:
            reason += f" The customer also uses high-risk language ({', '.join(repr(p) for p in phrases)})."
        return {"decision": "escalate", "reason": reason}

    if confidence_reason:
        reason = f"Escalate because {confidence_reason} and the customer may require human assistance."
        if phrases:
            reason += f" The customer also uses high-risk language ({', '.join(repr(p) for p in phrases)})."
        return {"decision": "escalate", "reason": reason}

    if phrases:
        return {
            "decision": "escalate",
            "reason": f"Escalate because the customer uses legal/urgent or strongly negative language ({', '.join(repr(p) for p in phrases)}), indicating a higher-risk complaint.",
        }

    return {
        "decision": "auto_handle",
        "reason": "Auto-handle because the intent is a lower-risk support issue, classifier confidence is high, and no legal, fraud, threat, or strong-urgency language was detected.",
    }


class EscalationEngine:
    """Compatibility wrapper around the Phase 6 decision function."""

    def __init__(
        self,
        min_classification_confidence: float = 0.75,
        min_retrieval_similarity: float = 0.60,
    ):
        self.min_classification_confidence = min_classification_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

    def check_safety_rules(self, text: str) -> Optional[EscalationDecision]:
        if not isinstance(text, str) or not text.strip():
            return _detailed_decision(decide_escalation(text, "other", "high", "placeholder"))
        phrases = _present_phrases(text)
        if phrases:
            result = decide_escalation(text, "other", "high", "placeholder")
            return _detailed_decision(result)
        return None

    def evaluate(
        self,
        text: str,
        classification: Any,
        retrieved_contexts: List[Any],
    ) -> EscalationDecision:
        if isinstance(classification, dict):
            intent = classification.get("intent")
            confidence = classification.get("confidence")
        else:
            intent = getattr(classification, "intent", None)
            confidence = getattr(classification, "confidence", None)
        result = decide_escalation(text, intent, confidence, "available reply")
        return _detailed_decision(result)


def _detailed_decision(result: Dict[str, str]) -> EscalationDecision:
    if result["decision"] == "auto_handle":
        return EscalationDecision(False, EscalationReason.NONE, result["reason"], 0.0)
    reason = result["reason"].lower()
    if "billing" in reason or "unauthorized" in reason:
        category = EscalationReason.BILLING_OR_REFUND_DISPUTE
    elif "compromised" in reason or "security" in reason:
        category = EscalationReason.SENSITIVE_ACCOUNT_PII
    elif "confidence" in reason:
        category = EscalationReason.LOW_CLASSIFICATION_CONFIDENCE
    elif "legal" in reason or "urgent" in reason or "negative" in reason:
        category = EscalationReason.LEGAL_OR_REGULATORY_THREAT
    else:
        category = EscalationReason.AMBIGUOUS_OR_MULTI_PART_QUERY
    return EscalationDecision(True, category, result["reason"], 1.0)
