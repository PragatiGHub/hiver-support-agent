"""Grounded AmazonHelp reply generation using the Anthropic API."""

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic
from dotenv import load_dotenv

from src.retrieve import RetrievedContext


load_dotenv()


@dataclass
class DraftReply:
    """Generated reply plus context and fallback metadata."""

    reply_text: str
    is_grounded: bool
    used_contexts: List[RetrievedContext]
    model_name: str
    tokens_used: Optional[int] = None
    influenced_by: Optional[List[Dict[str, Any]]] = None
    fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reply": self.reply_text,
            "influenced_by": self.influenced_by or [],
            "fallback": self.fallback,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class ReplyDrafter:
    """Generate concise, evidence-grounded customer support replies."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.3,
        brand_name: str = "AmazonHelp",
        max_chars: int = 280,
    ):
        self.model = model
        self.temperature = temperature
        self.brand_name = brand_name
        self.max_chars = max_chars
        api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key) if api_key else None

    def build_prompt(
        self,
        customer_tweet: str,
        predicted_intent: str,
        retrieved_contexts: List[Union[RetrievedContext, Dict[str, Any]]],
    ) -> str:
        """Build a prompt containing the customer, intent, and top examples."""
        examples = []
        for number, context in enumerate(retrieved_contexts[:3], start=1):
            if isinstance(context, dict):
                tweet_id = context.get("tweet_id")
                customer_text = context.get("customer_text", "")
                reply_text = context.get("brand_reply_text", "")
                similarity = context.get("similarity", 0.0)
            else:
                tweet_id = context.tweet_id
                customer_text = context.customer_text
                reply_text = context.brand_reply_text
                similarity = context.similarity
            examples.append(
                f"Example {number} (tweet_id={tweet_id}, similarity={float(similarity):.4f})\n"
                f"Customer: {customer_text}\nBrand reply: {reply_text}"
            )
        reference_text = "\n\n".join(examples) or "No historical examples were retrieved."
        return f"""You draft customer support replies for {self.brand_name}.

Current customer message:
{customer_tweet}

Classified intent: {predicted_intent}

Historical examples (evidence for tone and approach, not text to copy blindly):
{reference_text}

Write one concise, empathetic reply in the general style of the examples, under {self.max_chars} characters.
Stay grounded in the evidence. Do not invent policies, refund guarantees, delivery timelines, URLs, or account/order access.
Do not claim an action was completed. Ask the customer to provide details or contact support when that is the only safe next step.
Return only the reply text, with no quotation marks or explanation."""

    def draft(
        self,
        customer_tweet: str,
        predicted_intent: str,
        retrieved_contexts: List[Union[RetrievedContext, Dict[str, Any]]],
    ) -> DraftReply:
        """Generate a reply or a clearly marked safe fallback."""
        contexts = retrieved_contexts[:3]
        influenced_by = []
        for context in contexts:
            if isinstance(context, dict):
                tweet_id = context.get("tweet_id")
                similarity = context.get("similarity", 0.0)
            else:
                tweet_id = context.tweet_id
                similarity = context.similarity
            influenced_by.append({"tweet_id": str(tweet_id), "similarity": float(similarity)})

        if self.client is None:
            return self._fallback(influenced_by)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=120,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": self.build_prompt(customer_tweet, predicted_intent, contexts),
                    }
                ],
            )
            reply = "".join(
                block.text
                for block in response.content
                if getattr(block, "type", None) == "text"
            ).strip().strip('"')[: self.max_chars].strip()
            if not reply:
                return self._fallback(influenced_by)
            usage = getattr(response, "usage", None)
            tokens_used = getattr(usage, "output_tokens", None)
            return DraftReply(
                reply,
                True,
                contexts,
                self.model,
                tokens_used,
                influenced_by,
                False,
            )
        except Exception:
            return self._fallback(influenced_by)

    def _fallback(self, influenced_by: List[Dict[str, Any]]) -> DraftReply:
        reply = (
            "We couldn't generate an automatic reply. Please contact AmazonHelp support "
            "with more details so the team can assist."
        )
        return DraftReply(
            reply[: self.max_chars],
            False,
            [],
            self.model,
            None,
            influenced_by,
            True,
        )
