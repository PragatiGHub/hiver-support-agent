# Automated Customer Support Agent: System Report

*Evaluation and design report for automated Twitter customer support triage, retrieval-augmented response generation, and escalation policy.*

---

## 1. Executive Summary & Problem Framing
- **Objective**: Design, benchmark, and evaluate an automated agent pipeline for high-volume customer support on Twitter.
- **Key Challenges**: Short unstructured text, high ambiguity, safety/brand risk, strict rate limits, and avoiding hallucinations.
- **Headline Results**:
  - *Intent Classification Accuracy / F1*: [TODO: Add benchmark numbers: Majority vs TF-IDF vs Claude]
  - *Escalation Accuracy / Safety Catch Rate*: [TODO: Add precision/recall for escalation]
  - *Reply Quality (LLM Judge & Human Agreement)*: [TODO: Add score distribution and Cohen's Kappa / Pearson r]

---

## 2. Intent Taxonomy & Data Methodology
- **Target Brand Selection**:
  - Selected Brand: [TODO: e.g., `@AppleSupport`]
  - Rationale: High query volume, diverse technical and billing issues, publicly observable resolution patterns.
- **Data Preprocessing & Thread Reconstruction**:
  - Pairing inbound customer tweets with brand responses via reply chain traversal.
- **Golden Set Creation & Annotation Methodology**:
  - Sampling technique (stratified by length, time, and keyword diversity).
  - Schema definition and labelling protocol (150–250 hand-annotated examples).
  - Intent definitions and boundary cases.

---

## 3. Intent Classification System & Baseline Comparison
- **Taxonomy Overview**:
  - [TODO: Detail canonical classes: Account Access, Billing, Hardware, Software Bug, How-To, Order/Shipping, Feedback, Other]
- **Baseline Models**:
  - Trivial Baseline (Majority Class)
  - Heuristic Baseline (Keyword matcher)
  - Classical ML Baseline (TF-IDF + Logistic Regression)
- **Proposed LLM Classifier**:
  - Prompt structure, few-shot prompting, structured JSON schema output.
- **Results & Confusion Matrix**:
  - [TODO: Insert comparative metrics table]
  - [TODO: Insert confusion matrix and error analysis on ambiguous queries]

---

## 4. Context Retrieval & Grounded Reply Generation
- **Historical Interaction Retrieval**:
  - Embedding model: `sentence-transformers/all-MiniLM-L6-v2`.
  - Vector search mechanics and cosine similarity thresholding.
- **Grounded Response Generation**:
  - System prompt design enforcing brand voice, empathy, conciseness (<= 280 characters).
  - Factual grounding: referencing retrieved resolutions to avoid fictitious URLs or unverified policies.
  - Handling multi-turn conversation context.

---

## 5. Escalation Policy & Safety Guardrails
- **Triage Logic**: Auto-handle vs Human Escalation.
- **Escalation Triggers**:
  - Deterministic safety gates: Physical hazard/battery swelling, legal threats, PII/credential exposure.
  - Probabilistic gates: Low classification confidence (< threshold), low retrieval similarity (< threshold).
  - Policy gates: Financial disputes and refund approvals requiring authorized human sign-off.
- **Stated Justification**:
  - Providing transparent rationale codes (`EscalationReason`) for human agent handoff.
- **Precision/Recall Evaluation on Escalation**:
  - [TODO: Trade-off curve between customer deflection rate and false auto-replies]

---

## 6. Evaluation Framework & LLM-as-a-Judge
- **Evaluation Dimensions**:
  - Relevance, Groundedness/Truthfulness, Brand Tone, Concision & Format.
- **LLM Judge Setup**:
  - Claude-based automated judge with structured rubric scoring (1 to 5).
- **Judge vs Human Agreement**:
  - Pearson $r$, Spearman $\rho$, Cohen's $\kappa$, MAE/RMSE against human ratings in `golden_set.csv`.
  - Calibration assessment: Evaluating judge leniency and systematic bias.

---

## 7. Operational Trade-offs, Failure Modes & Next Steps
- **Cost & Latency Analysis**:
  - Per-query token cost and round-trip latency across pipeline stages.
- **Observed Failure Modes & Limitations**:
  - [TODO: Analyze top 3 failure categories observed in golden set]
- **Production Readiness & Recommendations**:
  - Human-in-the-loop fallback workflows, queue management, active learning on uncertain samples.
