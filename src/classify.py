"""Intent classification engine: Baselines and LLM-based classification.

This module provides three intent classifiers for the Amazon customer support domain:
1. MajorityClassifier: Trivial baseline predicting the most frequent training class.
2. TFIDFClassifier: Classical ML baseline (TF-IDF + Logistic Regression) trained on non-golden silver labels.
3. LLMIntentClassifier: Zero-shot/few-shot classifier using Anthropic's Claude (claude-sonnet-4-6).

All classifiers adhere to a common callable interface:
    classify(text: str) -> {"intent": str, "confidence": "high" | "medium" | "low"}

EVALUATION SAFETY & DATA LEAKAGE:
The golden evaluation set (golden_set/golden_set.csv) is strictly held out.
No golden set examples are ever used for model fitting, vectorizer fitting,
silver-label generation, or prompt example construction.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Load environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
from src.intents import (
    FALLBACK_INTENT,
    INTENT_MAP,
    format_intent_taxonomy_for_prompt,
    get_all_intents,
)

logger = logging.getLogger(__name__)

# Default model artifact path for TF-IDF pipeline
DEFAULT_TFIDF_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "processed", "tfidf_model.joblib"
)

# Canonical confidence levels
CONFIDENCE_LEVELS: Set[str] = {"high", "medium", "low"}


class BaseClassifier:
    """Abstract base class for all customer support intent classifiers."""

    def classify(self, text: str) -> Dict[str, str]:
        """Classify inbound customer text into an intent and confidence category.

        Args:
            text: Inbound customer message text.

        Returns:
            Dict[str, str]: Dictionary containing:
                - 'intent': Canonical intent name from src.intents
                - 'confidence': One of 'high', 'medium', 'low'
        """
        raise NotImplementedError


def _heuristic_classify(text: str) -> Dict[str, str]:
    """Classify support text with deterministic, ordered keyword rules.

    Design goal: prefer the most specific support issue, apply explicit boundary rules,
    and keep general complaint and fallback as low-priority outcomes.
    """
    txt = str(text).lower()
    if not txt.strip():
        return {"intent": FALLBACK_INTENT, "confidence": "low"}

    def has_any(*patterns: str) -> bool:
        return any(re.search(pattern, txt) for pattern in patterns)

    account_access_patterns = (
        r"\b(login|log in|sign in|password|passcode|otp|verification code|2fa|authenticator)\b",
        r"\b(locked out|account on hold|account locked|account suspended|unable to access|cannot access|access denied|reset password)\b",
        r"\b(compromised|hacked|unauthorized access|security breach|account takeover)\b",
        r"\b(no sms|sms not arriving|verification failed|not receiving code)\b",
    )
    billing_patterns = (
        r"\b(charged|charge|charges|unauthorized charge|extra charge|double charge|charged twice|deducted|billing|bill|overcharge|wrong amount)\b",
        r"\b(subscription|prime membership|auto-renew|renewal|monthly fee|renewed|deducted)\b",
        r"\b(cashback|promo code|discount|voucher|coupon|offer not applied|promotional credit|billing dispute)\b",
        r"\b(payment pending|paid twice|money taken|take money|scam.*prime|prime.*charge|charged.*prime|cobro.*prime|cobr.*prime|sin.*autorizar|without.*permission)\b",
        r"\b(cobro|cobr|sin.*autorizar|cuenta.*carg|cargo.*no.*autoriz|cargo.*sin.*permiso|factura.*incorrect|cobrado)\b",
    )
    refund_patterns = (
        r"\b(refund|money back|reimburse|reimbursement|refund status|refund pending|where is my refund|got no refund|return my money)\b",
        r"\b(cancel.*order.*refund|refund.*without.*permission|refund.*not received|paid.*refund)\b",
        r"\b(reembolso|remboursement|rückerstattung|返金|払い戻し|restituzione|reembolso|reembolso.*sin.*autoriz)\b",
    )
    wrong_damage_patterns = (
        r"\b(damaged|arrived damaged|damaged on arrival|damaged product|broken|arrived broken|cracked|shattered|smashed|wrong item|wrong product|different item|incorrect item|faulty|defective|scratched|used item|fake item|torn|missing item|opened box|empty box|incomplete item|wrong size|wrong color|wrong model|expired stock)\b",
        r"\b(item arrived.*broken|box.*empty|received.*wrong|received.*used|product.*damaged|arrived.*defective)\b",
        r"\b(割れてた|割れた|壊れ|破損|間違った|違う商品|空箱|中身がない)\b",
    )
    return_replacement_patterns = (
        r"\b(replacement|replace|exchange|return pickup|return label|courier pickup|pickup agent|pick up product|return request|return label|replacement request|replacement unit|replacement status|pickup.*scheduled|pickup.*done|need.*return label)\b",
        r"\b(replacement.*dispatched|pickup.*scheduled|return.*pickup|need.*replacement|need.*return label|swap.*item|exchange.*item|return.*label|label.*pickup)\b",
        r"\b(remplacement|reemplazo|ersatz|交換|返品|交換依頼|返送|pickup|collect.*return|return.*label|etiqueta.*devoluci|etiqueta.*retorno|pedir.*recogida)\b",
    )
    delivery_failure_patterns = (
        r"\b(delivery|delayed|delay|late|never delivered|never received|not delivered|didn\'t arrive|did not arrive|not arrived|hasn\'t arrived|have not received|haven\'t received|still waiting|waiting for my order|delivery didn\'t happen|delivery failed|delivered to another resident|delivered to wrong address|wrong building|package went to another|package missing|missing package|package not delivered|delivery date|not yet delivered|item.*not.*deliver|arrive.*late|arriving.*late|never.*arrive)\b",
        r"\b(tracking.*not.*update|tracking says.*delivered.*not.*here|courier.*failed|warehouse.*delay|carrier.*delay|order.*missing|lost.*package)\b",
        r"\b(配達|未配達|届か|遅延|配送|追跡|配送予定|到着してない|届いてない|never.*arrive|not.*arrived|paquete.*nunca.*lleg|nunca.*lleg|no.*hay.*actualizaci)\b",
        r"\b(paquete.*nunca.*lleg|nunca.*lleg|no.*hay.*actualizaci|lleg.*nunca|nunca.*lleg\w*)\b",
    )
    order_status_patterns = (
        r"\b(where is my order|where is my package|when will.*ship|when will my order ship|when will it ship|shipped yet|dispatch|preparing for dispatch|digital code|gift card code|order status|tracking my order|track my order|track order|tracking number|has my order shipped|has it shipped|shipping status|is it dispatched|delivery estimate|delivery date pending|not yet dispatched|arriving.*monday|arriving.*tuesday|arriving.*friday)\b",
        r"\b(status.*order|order.*status|shipping.*status|track.*package|current.*shipment|estimado.*entrega|delivery.*estimate|tracking.*number|not yet shipped)\b",
        r"\b(発送|配送状況|追跡番号|発送済み|まだ発送|届く予定|配送予定日|注文状況|発送予定)\b",
    )
    complaint_patterns = (
        r"\b(worst|pathetic|terrible|horrible|useless|disappointed|extremely disappointed|very disappointed|unacceptable|ridiculous|rude|fraud|awful|scam|poor service|very poor|disappointing|problem|issue|not helpful|no response|no resolution|bad support)\b",
        r"\b(terrible service|poor feedback|service.*awful|customer service.*bad|not helpful at all|unhelpful|representatives.*not helpful|doesn't work)\b",
        r"\b(mal servicio|servicio terrible|muy decepcionado|problem.*service|no.*respuesta|not helpful|servicio.*10|poor.*service)\b",
    )

    account_access_strong = has_any(*account_access_patterns)
    billing_strong = has_any(*billing_patterns)
    refund_strong = has_any(*refund_patterns)
    wrong_strong = has_any(*wrong_damage_patterns)
    return_replacement_strong = has_any(*return_replacement_patterns)
    delivery_failure_strong = has_any(*delivery_failure_patterns)
    order_status_strong = has_any(*order_status_patterns)
    general_complaint_strong = has_any(*complaint_patterns)

    # Specific override order: account access and refund/billing are stronger than vague shipment or complaint language.
    if account_access_strong and not billing_strong:
        return {"intent": "account_access_security", "confidence": "high"}
    if refund_strong and not return_replacement_strong:
        return {"intent": "refund_request", "confidence": "high"}
    if billing_strong and not account_access_strong:
        return {"intent": "billing_charge_dispute", "confidence": "high"}
    if return_replacement_strong and not refund_strong:
        return {"intent": "return_replacement_status", "confidence": "high"}
    if wrong_strong:
        return {"intent": "wrong_or_damaged_item", "confidence": "high"}

    # Distinguish order-inquiry questions from real non-delivery or delayed shipments.
    if delivery_failure_strong and not wrong_strong:
        return {"intent": "delivery_delay_or_non_delivery", "confidence": "high"}
    if order_status_strong and not delivery_failure_strong:
        return {"intent": "order_status_inquiry", "confidence": "medium"}

    if general_complaint_strong and not (
        account_access_strong or billing_strong or refund_strong or wrong_strong or return_replacement_strong or delivery_failure_strong or order_status_strong
    ):
        return {"intent": "general_complaint", "confidence": "medium"}

    return {"intent": FALLBACK_INTENT, "confidence": "low"}


class LocalHeuristicClassifier(BaseClassifier):
    """Deterministic local fallback used when Claude cannot be reached."""

    def classify(self, text: str) -> Dict[str, str]:
        if not text or not str(text).strip():
            return {"intent": FALLBACK_INTENT, "confidence": "low"}
        return _heuristic_classify(text)


# =====================================================================
# 1. TRIVIAL BASELINE: Majority Class Predictor
# =====================================================================
class MajorityClassifier(BaseClassifier):
    """Trivial baseline that always predicts the most frequent class in training data.

    Why this baseline exists:
    A majority class classifier serves as the absolute zero-performance floor.
    In imbalanced distributions, predicting the majority class can yield deceptively
    high raw accuracy while having zero discriminatory power (0 recall on minority classes,
    macro F1 close to 0). Any useful model (TF-IDF or LLM) must significantly outperform
    this baseline in macro F1 and balanced accuracy.

    Data Leakage Prevention:
    The majority class is calculated strictly using non-golden training data (silver labels).
    The golden evaluation set is never touched.
    """

    def __init__(self, majority_intent: Optional[str] = None):
        """Initialize majority classifier.

        Args:
            majority_intent: The most frequent class in non-golden training data.
                             Defaults to 'other' (empirically dominant in Amazon non-golden subsample).
        """
        self.majority_intent = majority_intent or "other"
        if self.majority_intent not in INTENT_MAP:
            self.majority_intent = FALLBACK_INTENT

    def classify(self, text: str) -> Dict[str, str]:
        """Return the constant majority intent with conservative confidence.

        Args:
            text: Customer inquiry text.

        Returns:
            Dict[str, str]: Constant prediction dict {"intent": ..., "confidence": "low"}
        """
        if not text or not str(text).strip():
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        # A static majority prediction possesses no text-specific discriminative evidence,
        # so its confidence is inherently low.
        return {
            "intent": self.majority_intent,
            "confidence": "low",
        }


# =====================================================================
# 2. TF-IDF BASELINE: TfidfVectorizer + LogisticRegression
# =====================================================================
class TFIDFClassifier(BaseClassifier):
    """Classical ML baseline combining TfidfVectorizer and LogisticRegression.

    How it works:
    1. Extracts character/word n-grams (1-2 words) weighted by inverse document frequency.
    2. Uses multinomial Logistic Regression with balanced class weights to predict intent.
    3. Converts calibrated posterior probabilities (predict_proba) into categorical confidence:
       - Probability >= 0.50 -> 'high'
       - 0.25 <= Probability < 0.50 -> 'medium'
       - Probability < 0.25 -> 'low'
       (In a 9-class setting where random chance is ~0.11, 0.50 represents >4.5x uniform prior).

    What Silver Labels Are:
    Because Twitter Customer Support (TWCS) is unlabelled, we generate pseudo-labels
    ("silver labels") on a non-golden subset of customer tweets.
    Silver labels are weak labels: they are generated automatically (via LLM or heuristics)
    and contain noise, label bias, and errors compared to human-verified golden labels.
    However, they allow classical supervised models to learn feature representations without
    violating the strict separation of the 200 human-annotated golden evaluation examples.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        pipeline: Optional[Any] = None,
    ):
        """Initialize TF-IDF classifier, loading fitted model if available.

        Args:
            model_path: Filepath to saved joblib model artifact.
            pipeline: Pre-fitted sklearn Pipeline instance.
        """
        self.pipeline = pipeline
        self.model_path = model_path or DEFAULT_TFIDF_MODEL_PATH

        if self.pipeline is None and os.path.exists(self.model_path):
            try:
                self.pipeline = joblib.load(self.model_path)
            except Exception as e:
                logger.warning(f"Could not load TF-IDF model from {self.model_path}: {e}")

    def fit(self, texts: List[str], labels: List[str]) -> "TFIDFClassifier":
        """Fit vectorizer and logistic regression on training texts and silver labels.

        Args:
            texts: List of non-golden customer messages.
            labels: Corresponding silver intent labels.

        Returns:
            TFIDFClassifier: Fitted self.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

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
        """Persist fitted pipeline to disk."""
        target_path = model_path or self.model_path
        if self.pipeline is not None:
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            joblib.dump(self.pipeline, target_path)

    def classify(self, text: str) -> Dict[str, str]:
        """Classify customer text using trained TF-IDF model.

        Args:
            text: Customer tweet text.

        Returns:
            Dict[str, str]: {"intent": str, "confidence": "high"|"medium"|"low"}
        """
        if not text or not str(text).strip():
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        if self.pipeline is None:
            # Fallback if model has not been trained/loaded yet
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        try:
            probs = self.pipeline.predict_proba([str(text)])[0]
            best_idx = int(np.argmax(probs))
            predicted_class = str(self.pipeline.classes_[best_idx])
            max_prob = float(probs[best_idx])

            # Convert continuous posterior probability to discrete confidence
            if max_prob >= 0.50:
                confidence = "high"
            elif max_prob >= 0.25:
                confidence = "medium"
            else:
                confidence = "low"

            # Validate against taxonomy
            if predicted_class not in INTENT_MAP:
                predicted_class = FALLBACK_INTENT
                confidence = "low"

            return {
                "intent": predicted_class,
                "confidence": confidence,
            }
        except Exception as e:
            logger.warning(f"TF-IDF inference error: {e}")
            return {"intent": FALLBACK_INTENT, "confidence": "low"}


# =====================================================================
# 3. MAIN LLM CLASSIFIER: Anthropic Claude (claude-sonnet-4-6)
# =====================================================================
class LLMIntentClassifier(BaseClassifier):
    """Zero-shot/few-shot intent classifier using Anthropic Claude API.

    Model: claude-sonnet-4-6
    Reads ANTHROPIC_API_KEY from environment.
    Embeds the taxonomy descriptions and 2-3 examples from src/intents.py.
    Requires structured JSON output with exact intent name and high/medium/low confidence.
    Gracefully falls back to the deterministic LocalHeuristicClassifier on missing
    credentials or API failures.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM classifier with model configuration and API credentials.

        Args:
            model: Anthropic model identifier (default: claude-sonnet-4-6).
            temperature: Sampling temperature for deterministic classification.
            api_key: Optional explicit API key. If None, reads from os.environ.
        """
        self.model = model
        self.temperature = temperature
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.taxonomy_prompt = format_intent_taxonomy_for_prompt()
        self._client = None
        self.local_fallback = LocalHeuristicClassifier()

    def _get_client(self):
        """Lazy-initialize Anthropic client."""
        if self._client is None:
            key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                return None
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=key)
            except Exception as e:
                logger.warning(f"Failed to initialize Anthropic client: {e}")
                return None
        return self._client

    def _build_system_prompt(self) -> str:
        """Construct the prompt grounding Claude in the 9-intent taxonomy."""
        return (
            "You are an expert customer care triage AI for social media support (@AmazonHelp).\n"
            "Your task is to classify an inbound customer tweet into exactly ONE intent category "
            "from the provided taxonomy below.\n\n"
            f"{self.taxonomy_prompt}\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Output ONLY a valid JSON object with exactly two keys: 'intent' and 'confidence'.\n"
            "2. 'intent' MUST be an exact string from the taxonomy. Never invent or alter intent names.\n"
            "3. If the message does not clearly or confidently fit any specific category, or is ambiguous, "
            "non-English, sarcastic noise, or unrelated to support, choose 'other'.\n"
            "4. 'confidence' MUST be exactly one of: 'high', 'medium', 'low'.\n"
            "5. Do NOT include any markdown formatting, backticks, or explanatory text before or after the JSON."
        )

    def _parse_response(self, raw_content: str) -> Dict[str, str]:
        """Robustly parse JSON response from Claude and validate against taxonomy.

        Args:
            raw_content: Raw text returned by Claude API.

        Returns:
            Dict[str, str]: Validated {"intent": str, "confidence": str}
        """
        cleaned = raw_content.strip()
        # Strip markdown fences if the model included them despite instructions
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        intent = FALLBACK_INTENT
        confidence = "low"

        # Attempt 1: Direct JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                candidate_intent = str(data.get("intent", "")).strip().lower()
                candidate_conf = str(data.get("confidence", "")).strip().lower()

                if candidate_intent in INTENT_MAP:
                    intent = candidate_intent
                    if candidate_conf in CONFIDENCE_LEVELS:
                        confidence = candidate_conf
                    else:
                        confidence = "medium"
                else:
                    intent = FALLBACK_INTENT
                    confidence = "low"

                return {"intent": intent, "confidence": confidence}
        except Exception:
            pass

        # Attempt 2: Regex extraction fallback
        m_intent = re.search(r'"intent"\s*:\s*"([^"]+)"', raw_content)
        m_conf = re.search(r'"confidence"\s*:\s*"([^"]+)"', raw_content)

        if m_intent:
            candidate_intent = m_intent.group(1).strip().lower()
            if candidate_intent in INTENT_MAP:
                intent = candidate_intent
                if m_conf:
                    candidate_conf = m_conf.group(1).strip().lower()
                    if candidate_conf in CONFIDENCE_LEVELS:
                        confidence = candidate_conf
                    else:
                        confidence = "medium"
            else:
                intent = FALLBACK_INTENT
                confidence = "low"
        else:
            intent = FALLBACK_INTENT
            confidence = "low"

        return {"intent": intent, "confidence": confidence}

    def classify(self, text: str) -> Dict[str, str]:
        """Classify customer text using Claude LLM.

        Args:
            text: Customer tweet text.

        Returns:
            Dict[str, str]: {"intent": str, "confidence": "high"|"medium"|"low"}
        """
        # Guard: Empty or whitespace input
        if not text or not str(text).strip():
            return {"intent": FALLBACK_INTENT, "confidence": "low"}

        client = self._get_client()
        # Guard: Missing API key or uninitialized client
        if client is None:
            logger.warning("Anthropic client unavailable; using local heuristic classifier fallback.")
            return self.local_fallback.classify(text)

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=100,
                temperature=self.temperature,
                system=self._build_system_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": f'Customer Message:\n"{str(text).strip()}"',
                    }
                ],
            )
            raw_text = response.content[0].text if response.content else ""
            return self._parse_response(raw_text)

        except Exception as e:
            # Handle Anthropic API errors, timeouts, rate limits, connection issues
            logger.warning(
                "Anthropic API error in LLMIntentClassifier: %s; using local heuristic classifier fallback.",
                e,
            )
            return self.local_fallback.classify(text)


# =====================================================================
# COMMON CALLABLE INTERFACES
# =====================================================================

# Global singleton instances for direct function calls
_majority_instance: Optional[MajorityClassifier] = None
_tfidf_instance: Optional[TFIDFClassifier] = None
_llm_instance: Optional[LLMIntentClassifier] = None


def majority_classify(text: str) -> Dict[str, str]:
    """Classify text using the trivial Majority Classifier baseline."""
    global _majority_instance
    if _majority_instance is None:
        _majority_instance = MajorityClassifier()
    return _majority_instance.classify(text)


def tfidf_classify(text: str) -> Dict[str, str]:
    """Classify text using the classical TF-IDF + Logistic Regression baseline."""
    global _tfidf_instance
    if _tfidf_instance is None:
        _tfidf_instance = TFIDFClassifier()
    return _tfidf_instance.classify(text)


def llm_classify(text: str) -> Dict[str, str]:
    """Classify text using the Claude LLM classifier (claude-sonnet-4-6)."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMIntentClassifier()
    return _llm_instance.classify(text)


def classify(text: str, method: str = "llm") -> Dict[str, str]:
    """Unified common classification interface for all three classifiers.

    Args:
        text: Customer message text.
        method: One of 'llm', 'tfidf', 'majority'.

    Returns:
        Dict[str, str]: {"intent": str, "confidence": "high"|"medium"|"low"}
    """
    method_lower = str(method).strip().lower()
    if method_lower == "majority":
        return majority_classify(text)
    elif method_lower in {"tfidf", "tf-idf"}:
        return tfidf_classify(text)
    elif method_lower == "llm":
        return llm_classify(text)
    else:
        raise ValueError(
            f"Unknown classification method '{method}'. Choose from 'llm', 'tfidf', 'majority'."
        )
