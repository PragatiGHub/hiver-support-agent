# Engineering Decision Log

A chronological bulleted log documenting key technical, algorithmic, and architectural decisions made throughout the project.

---

- **2026-09-11 | Project Inception & Structure**:
  - Selected a modular architecture separating data processing, core inference components (`src/`), and evaluation harness (`eval/`).
  - Standardized on Python 3.9 compatibility for broad local environment support.
  - Pinned core dependencies (`anthropic`, `pandas`, `scikit-learn`, `sentence-transformers`, `numpy`) with safe version constraints.

- **2026-09-11 | Brand Subsampling Choice**:
  - *Decision*: Target `@AppleSupport` (or `@AmazonHelp`) as the primary benchmark brand.
  - *Rationale*: Abundant high-volume conversational threads, distinct technical vs billing issues, and rich variation in customer urgency and sentiment.

- **2026-09-11 | Intent Taxonomy Granularity**:
  - *Decision*: Established 8 canonical intent classes (`account_access`, `billing_and_subscription`, `hardware_and_physical`, `software_bug_or_glitch`, `how_to_or_configuration`, `order_and_shipping`, `feedback_or_complaint`, `other_or_ambiguous`).
  - *Rationale*: Balances operational utility for routing while avoiding over-fragmentation that degrades classifier accuracy.

- **2026-09-11 | Golden Set Sizing & Labelling Protocol**:
  - *Decision*: Target 150–250 hand-labelled examples with stratified temporal and length sampling.
  - *Rationale*: Provides statistically meaningful sample size for macro-F1 and correlation calculations while remaining feasible for high-quality manual annotation.

- **2026-09-11 | Context Retrieval Architecture**:
  - *Decision*: Selected `sentence-transformers/all-MiniLM-L6-v2` for dense retrieval.
  - *Rationale*: Lightweight, fast CPU inference (~15ms per query), compact embedding dimension (384), and strong semantic matching performance on short texts.

- **2026-09-11 | Safety-First Escalation Policy**:
  - *Decision*: Multi-layered triage combining deterministic safety gates (PII, physical hazards, legal threats) with probabilistic thresholds (confidence and retrieval quality).
  - *Rationale*: Prioritizes customer safety and brand reputation; false escalations (agent reviews safe tweet) are significantly less harmful than false auto-replies (bot mishandles battery fire or billing dispute).

- **2026-09-11 | LLM-as-a-Judge Evaluation & Agreement Harness**:
  - *Decision*: Implement both continuous correlation (Pearson/Spearman) and discrete agreement (Cohen's Kappa) against manual human ratings.
  - *Rationale*: Validates that LLM judge scores reflect real human assessment before trusting automated grading for iterative development.

- **[TODO: Add subsequent experimental and modelling decisions here]**:
  - *[Date]*: [Decision and rationale]
