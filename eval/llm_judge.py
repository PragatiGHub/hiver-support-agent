"""LLM-as-judge evaluation for generated AmazonHelp replies."""

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Union

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

RUBRIC_CRITERIA = """
Score each dimension from 1 to 5:
- relevance: directly addresses the customer's message and intent
- groundedness: supported by the retrieved examples; penalize invented policies,
  guarantees, refunds, delivery dates, links, or account actions
- tone: concise, professional, empathetic AmazonHelp-style support tone
- conciseness: focused, clear, and no more than 280 characters
Return JSON only with integer scores and a short reason.
"""


@dataclass
class JudgeScore:
    """Structured judge result; unavailable scores remain None."""

    overall_score: Optional[float]
    relevance_score: Optional[float]
    groundedness_score: Optional[float]
    tone_score: Optional[float]
    concision_score: Optional[float]
    critique: str
    pass_threshold: bool = False
    available: bool = True

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "relevance": result["relevance_score"],
                "groundedness": result["groundedness_score"],
                "tone": result["tone_score"],
                "conciseness": result["concision_score"],
                "overall": result["overall_score"],
                "reason": result["critique"],
            }
        )
        return result


class LLMJudge:
    """Evaluate replies with Claude without fabricating unavailable scores."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        temperature: float = 0.0,
        passing_grade: float = 3.5,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.passing_grade = passing_grade
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=self.api_key) if self.api_key else None

    def build_judge_prompt(
        self,
        customer_tweet: str,
        generated_reply: str,
        classified_intent: Optional[str] = None,
        retrieved_examples: Optional[Union[str, List[Any]]] = None,
        reference_reply: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> str:
        """Construct a prompt with all evidence available to the judge."""
        if retrieved_examples is None:
            retrieved_examples = retrieved_context or "No retrieved examples were available."
        if isinstance(retrieved_examples, list):
            examples_text = "\n\n".join(str(item) for item in retrieved_examples)
        else:
            examples_text = str(retrieved_examples)
        reference = reference_reply or "No reference reply should be used for scoring."
        return f"""You are an impartial evaluator of an AmazonHelp customer-support draft.

Customer message:
{customer_tweet}

Classified intent:
{classified_intent or "Not provided"}

Retrieved historical customer/reply examples:
{examples_text}

Generated reply:
{generated_reply}

Optional reference reply (context only, not ground truth):
{reference}

{RUBRIC_CRITERIA}
Groundedness is especially important: only credit claims supported by the historical examples or directly safe, generic support language. Penalize claims that invent Amazon policy, refund guarantees, delivery timelines, links, or completed account/order actions.
Output exactly one JSON object with keys: relevance, groundedness, tone, conciseness, overall, reason.
"""

    def evaluate_reply(
        self,
        customer_tweet: str,
        generated_reply: str,
        classified_intent: Optional[str] = None,
        retrieved_examples: Optional[Union[str, List[Any]]] = None,
        reference_reply: Optional[str] = None,
        retrieved_context: Optional[str] = None,
    ) -> JudgeScore:
        """Score one reply, returning unavailable values when judging cannot run."""
        if self.client is None:
            return self._unavailable("LLM judge unavailable: ANTHROPIC_API_KEY is not set.")
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=220,
                temperature=self.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": self.build_judge_prompt(
                            customer_tweet,
                            generated_reply,
                            classified_intent,
                            retrieved_examples,
                            reference_reply,
                            retrieved_context,
                        ),
                    }
                ],
            )
            raw = "".join(
                block.text for block in response.content if getattr(block, "type", None) == "text"
            ).strip()
            data = self._parse_json(raw)
            scores = {
                key: self._score(data.get(key))
                for key in ("relevance", "groundedness", "tone", "conciseness", "overall")
            }
            if any(value is None for value in scores.values()):
                return self._unavailable("LLM judge returned invalid or incomplete scores.")
            reason = str(data.get("reason", "")).strip()
            return JudgeScore(
                scores["overall"],
                scores["relevance"],
                scores["groundedness"],
                scores["tone"],
                scores["conciseness"],
                reason,
                scores["overall"] >= self.passing_grade,
                True,
            )
        except Exception as exc:
            return self._unavailable(f"LLM judge unavailable after API failure: {exc}")

    def evaluate_batch(self, records: List[Dict[str, Any]]) -> List[JudgeScore]:
        """Evaluate records sequentially; callers can cache these results."""
        scores = []
        for record in records:
            scores.append(
                self.evaluate_reply(
                    customer_tweet=str(record.get("customer_text", record.get("customer_tweet", ""))),
                    generated_reply=str(record.get("reply", record.get("generated_reply", ""))),
                    classified_intent=record.get("intent"),
                    retrieved_examples=record.get("retrieved_examples", record.get("retrieved_context")),
                    reference_reply=record.get("reference_reply"),
                )
            )
        return scores

    @staticmethod
    def _parse_json(raw: str) -> Dict[str, Any]:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(cleaned)
        if not isinstance(value, dict):
            raise ValueError("judge response was not a JSON object")
        return value

    @staticmethod
    def _score(value: Any) -> Optional[float]:
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        return score if 1.0 <= score <= 5.0 else None

    @staticmethod
    def _unavailable(reason: str) -> JudgeScore:
        return JudgeScore(None, None, None, None, None, reason, False, False)
