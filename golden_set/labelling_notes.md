# Golden Set Labelling Notes & Methodology

This document outlines the sampling strategy, schema definitions, intent annotation guidelines, and escalation decision rules for creating the 150–250 hand-labelled evaluation dataset in `golden_set.csv`.

---

## 1. Sampling Methodology

- **Source Corpus**: Inbound customer tweets directed at the selected brand (e.g., `@AppleSupport`) from `data/processed/`.
- **Target Size**: 150 to 250 diverse, high-fidelity customer inquiries.
- **Stratification Strategy**:
  - **Temporal balance**: Sample across multiple weeks/months to prevent clustering around single transient outages or release days.
  - **Length diversity**: Mix of short blunt complaints ("Battery dead again"), medium multi-sentence inquiries, and complex multi-issue complaints.
  - **Edge cases**: Deliberately include ambiguous tweets, sarcastic queries, non-English or heavily fragmented slang, and multi-intent messages.

---

## 2. Schema Definition (`golden_set.csv`)

| Column Name | Type | Description |
|---|---|---|
| `id` | integer | Sequential unique ID (1 to N) |
| `tweet_id` | string | Original Twitter tweet ID from Kaggle TWCS dataset |
| `author_id` | string | Anonymized user identifier |
| `created_at` | string | Raw timestamp of the inbound tweet |
| `text` | string | Raw customer message text |
| `true_intent` | string | Standardized intent class from `src/intents.py` |
| `escalate_needed` | boolean | `True` if human agent intervention is strictly required, `False` if safe to auto-respond |
| `escalation_reason` | string | Justification code or rationale if `escalate_needed=True` |
| `reference_reply` | string | Historical brand response or ideal ground-truth reference response |
| `human_quality_score` | float | 1–5 rating of reference reply (used for judge agreement validation) |
| `notes` | string | Annotator observations, ambiguity notes, or edge-case flags |

---

## 3. Annotation Guidelines

### Intent Classification Rules
1. **Single Predominant Intent**: If a customer mentions two issues (e.g., "Phone won't charge and bill is too high"), label with the primary actionable blocker.
2. **Standard Taxonomy Alignment**: Annotate strictly using labels declared in `src/intents.py`.
3. **Ambiguity Handling**:
   - If completely unintelligible or lacking sufficient context, assign `other_or_ambiguous`.

### Escalation Decision Rules (`escalate_needed = True`)
Flag for escalation when:
- High user distress, severe hostility, or legal threats.
- Request involves private account credentials, sensitive billing/credit card details, or authentication lockouts.
- Physical device hazards (e.g., swollen battery, smoke, heat risk).
- High complexity, repetitive failures after multiple troubleshooting attempts.

Mark `escalate_needed = False` when:
- Standard troubleshooting FAQ, publicly documented software features, routine how-to queries, or known public updates.

---

## 4. Human Quality Rating Rubric (1–5)
Used in `eval/judge_agreement.py` to benchmark LLM judge performance:
- **5 (Excellent)**: Fully addresses problem, correct tone, grounded in brand policy, provides exact next action or link.
- **4 (Good)**: Helpful and polite, minor detail missing or slightly generic.
- **3 (Adequate)**: Basic acknowledgment, requests more info or points to general support page.
- **2 (Poor)**: Misses user context, partially incorrect troubleshooting steps.
- **1 (Unacceptable)**: Hallucinated policy, rude, dangerous advice, or completely irrelevant.

---

## TODO Checklist for Labeller
- [ ] Export 300 random candidate inbound tweets from `data/processed/`.
- [ ] Deduplicate and filter out bot spam.
- [ ] Annotate first batch of 50 samples to calibrate edge cases.
- [ ] Complete full set of 150–250 annotations.
- [ ] Review class balance across intents to ensure representation of all core categories.
