"""Intent taxonomy definition for customer support tweet classification.

Defines the supported customer support intent categories for e-commerce
customer care (e.g. AmazonHelp).

Each intent includes:
- name: Unique string identifier
- description: Detailed definition of what customer queries belong here
- examples: 2-3 realistic customer message phrases

Fallback rule:
If a classifier cannot match an intent with sufficient confidence,
or if the text is ambiguous/unsupported, it defaults to "other".
"""

from enum import Enum
from typing import Any, Dict, List, Optional


FALLBACK_INTENT = "other"


INTENT_TAXONOMY: List[Dict[str, Any]] = [
    {
        "name": "delivery_delay_or_non_delivery",
        "description": "Package hasn't arrived, tracking not updating, or marked delivered but missing.",
        "examples": [
            "My package was supposed to arrive yesterday and tracking hasn't updated since Tuesday.",
            "Tracking says delivered to front porch but nothing is there. Where is my order?",
            "Why is my Prime one-day delivery delayed by 3 days with no explanation?",
        ],
    },
    {
        "name": "wrong_or_damaged_item",
        "description": "Received incorrect, damaged, or incomplete item.",
        "examples": [
            "Opened my box and received the wrong size shoes instead of what I ordered.",
            "The book arrived with a torn cover and crushed spine.",
            "I ordered a set of 3 filters but only 1 was inside the package.",
        ],
    },
    {
        "name": "refund_request",
        "description": "Customer asking for refund or refund status.",
        "examples": [
            "I returned the defective headphones a week ago, when will I get my refund?",
            "Please cancel this order immediately and refund the amount to my original card.",
            "Still haven't received my refund for order #112-4928104.",
        ],
    },
    {
        "name": "billing_charge_dispute",
        "description": "Unauthorized charge, unwanted subscription, or incorrect cashback/promo.",
        "examples": [
            "I was charged $14.99 for Prime which I never signed up for or authorized.",
            "My credit card shows two charges for the same single order.",
            "Did not receive the 15% promotional cashback that was promised during checkout.",
        ],
    },
    {
        "name": "account_access_security",
        "description": "Login issues, account compromised, or password/OTP trouble.",
        "examples": [
            "Can't log into my account because the OTP isn't sending to my phone.",
            "I think someone hacked my account, I just got an unauthorized password change email.",
            "My account was placed on hold and I can't access my digital orders.",
        ],
    },
    {
        "name": "order_status_inquiry",
        "description": "Pending order, digital code verification, or general order status question.",
        "examples": [
            "Can you tell me if order #402-9182391 has shipped yet?",
            "Purchased a gift card 2 hours ago, when will the digital code be emailed?",
            "Why is my order still showing 'Preparing for Dispatch' after 48 hours?",
        ],
    },
    {
        "name": "return_replacement_status",
        "description": "Return or replacement pickup/process delayed or unclear.",
        "examples": [
            "Scheduled a return pickup for today but the courier never showed up.",
            "How do I print a return shipping label without a printer at home?",
            "I submitted a replacement request 4 days ago, has the replacement unit dispatched?",
        ],
    },
    {
        "name": "general_complaint",
        "description": "Vague dissatisfaction not fitting other categories.",
        "examples": [
            "Your customer service has gone completely downhill, worst experience ever.",
            "Never shopping with you again, absolutely terrible service all around.",
            "Frustrated with how difficult it is to talk to a real human being.",
        ],
    },
    {
        "name": "other",
        "description": "Praise, off-topic, non-English noise, or unclear intent.",
        "examples": [
            "Shoutout to customer care for fixing my problem so fast! Great job!",
            "Hola necesito ayuda con mi cuenta por favor.",
            "lol what is this random tweet",
        ],
    },
]


class SupportIntent(str, Enum):
    """Canonical enum identifiers for supported customer intents."""

    DELIVERY_DELAY_OR_NON_DELIVERY = "delivery_delay_or_non_delivery"
    WRONG_OR_DAMAGED_ITEM = "wrong_or_damaged_item"
    REFUND_REQUEST = "refund_request"
    BILLING_CHARGE_DISPUTE = "billing_charge_dispute"
    ACCOUNT_ACCESS_SECURITY = "account_access_security"
    ORDER_STATUS_INQUIRY = "order_status_inquiry"
    RETURN_REPLACEMENT_STATUS = "return_replacement_status"
    GENERAL_COMPLAINT = "general_complaint"
    OTHER = "other"


# Quick-lookup dictionary keyed by canonical name
INTENT_MAP: Dict[str, Dict[str, Any]] = {item["name"]: item for item in INTENT_TAXONOMY}

# Descriptions mapping for fast access
INTENT_DESCRIPTIONS: Dict[str, str] = {
    item["name"]: item["description"] for item in INTENT_TAXONOMY
}


def get_all_intents() -> List[str]:
    """Return an ordered list of all canonical intent string names.

    Returns:
        List[str]: Canonical intent names.
    """
    return [item["name"] for item in INTENT_TAXONOMY]


def get_fallback_intent() -> str:
    """Return the default fallback intent name when matching is uncertain.

    Returns:
        str: 'other'
    """
    return FALLBACK_INTENT


def resolve_intent(
    predicted_intent: Optional[str],
    confidence: Optional[float] = None,
    min_confidence: float = 0.5,
) -> str:
    """Resolve a predicted intent against taxonomy and confidence threshold.

    Fallback rule:
    If predicted_intent is None, not in the taxonomy, or has confidence below
    min_confidence, defaults to 'other'.

    Args:
        predicted_intent: Candidate intent name.
        confidence: Optional classification probability/score [0.0 to 1.0].
        min_confidence: Threshold required to accept the prediction.

    Returns:
        str: Validated intent name or fallback 'other'.
    """
    if not predicted_intent or predicted_intent not in INTENT_MAP:
        return FALLBACK_INTENT

    if confidence is not None and confidence < min_confidence:
        return FALLBACK_INTENT

    return predicted_intent


def format_intent_taxonomy_for_prompt() -> str:
    """Format the intent taxonomy with descriptions and examples for LLM prompts.

    Returns:
        str: Prompt-ready markdown block detailing intents, descriptions, and examples.
    """
    lines = ["### Intent Taxonomy & Guidance:"]
    for intent in INTENT_TAXONOMY:
        lines.append(f"- **{intent['name']}**:")
        lines.append(f"  Description: {intent['description']}")
        lines.append("  Examples:")
        for ex in intent["examples"]:
            lines.append(f'    * "{ex}"')
    lines.append(
        f"\n**Fallback Rule**: If the customer message is ambiguous, non-English, "
        f"unrelated to support, or does not clearly match any category above with "
        f"high certainty, categorize it as '{FALLBACK_INTENT}'."
    )
    return "\n".join(lines)
