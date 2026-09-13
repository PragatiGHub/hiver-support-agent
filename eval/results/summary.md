# Phase 8 Evaluation Results

## Classification Results

| Classifier | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Majority baseline | 0.3500 | 0.0576 | 0.1815 |
| TF-IDF + Logistic Regression | 0.4800 | 0.2202 | 0.4121 |
| Main LLM classifier | 0.5500 | 0.4261 | 0.5226 |

### Per-Intent Metrics

#### Majority baseline
| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 0.0000 | 0.0000 | 0.0000 | 50 |
| wrong_or_damaged_item | 0.0000 | 0.0000 | 0.0000 | 12 |
| refund_request | 0.0000 | 0.0000 | 0.0000 | 7 |
| billing_charge_dispute | 0.0000 | 0.0000 | 0.0000 | 8 |
| account_access_security | 0.0000 | 0.0000 | 0.0000 | 5 |
| order_status_inquiry | 0.0000 | 0.0000 | 0.0000 | 12 |
| return_replacement_status | 0.0000 | 0.0000 | 0.0000 | 6 |
| general_complaint | 0.0000 | 0.0000 | 0.0000 | 30 |
| other | 0.3500 | 1.0000 | 0.5185 | 70 |

Confusion matrix:

| true \ predicted | delivery_delay_or_non_delivery | wrong_or_damaged_item | refund_request | billing_charge_dispute | account_access_security | order_status_inquiry | return_replacement_status | general_complaint | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 50 |
| wrong_or_damaged_item | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| refund_request | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7 |
| billing_charge_dispute | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 |
| account_access_security | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| order_status_inquiry | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 12 |
| return_replacement_status | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| general_complaint | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 30 |
| other | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 70 |

#### TF-IDF + Logistic Regression
| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 0.5686 | 0.5800 | 0.5743 | 50 |
| wrong_or_damaged_item | 0.0000 | 0.0000 | 0.0000 | 12 |
| refund_request | 0.7500 | 0.4286 | 0.5455 | 7 |
| billing_charge_dispute | 0.0000 | 0.0000 | 0.0000 | 8 |
| account_access_security | 0.0000 | 0.0000 | 0.0000 | 5 |
| order_status_inquiry | 0.0000 | 0.0000 | 0.0000 | 12 |
| return_replacement_status | 0.0000 | 0.0000 | 0.0000 | 6 |
| general_complaint | 0.3750 | 0.2000 | 0.2609 | 30 |
| other | 0.4715 | 0.8286 | 0.6010 | 70 |

Confusion matrix:

| true \ predicted | delivery_delay_or_non_delivery | wrong_or_damaged_item | refund_request | billing_charge_dispute | account_access_security | order_status_inquiry | return_replacement_status | general_complaint | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 29 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 16 |
| wrong_or_damaged_item | 2 | 0 | 0 | 1 | 0 | 0 | 0 | 3 | 6 |
| refund_request | 1 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 3 |
| billing_charge_dispute | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 6 |
| account_access_security | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 3 |
| order_status_inquiry | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6 |
| return_replacement_status | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 5 |
| general_complaint | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 6 | 20 |
| other | 7 | 0 | 0 | 2 | 1 | 0 | 0 | 2 | 58 |

#### Main LLM classifier
| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 0.7188 | 0.4600 | 0.5610 | 50 |
| wrong_or_damaged_item | 0.6000 | 0.2500 | 0.3529 | 12 |
| refund_request | 1.0000 | 0.7143 | 0.8333 | 7 |
| billing_charge_dispute | 0.2857 | 0.2500 | 0.2667 | 8 |
| account_access_security | 1.0000 | 0.2000 | 0.3333 | 5 |
| order_status_inquiry | 1.0000 | 0.2500 | 0.4000 | 12 |
| return_replacement_status | 0.0000 | 0.0000 | 0.0000 | 6 |
| general_complaint | 0.6111 | 0.3667 | 0.4583 | 30 |
| other | 0.4882 | 0.8857 | 0.6294 | 70 |

Confusion matrix:

| true \ predicted | delivery_delay_or_non_delivery | wrong_or_damaged_item | refund_request | billing_charge_dispute | account_access_security | order_status_inquiry | return_replacement_status | general_complaint | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| delivery_delay_or_non_delivery | 23 | 0 | 0 | 2 | 0 | 0 | 0 | 1 | 24 |
| wrong_or_damaged_item | 2 | 3 | 0 | 0 | 0 | 0 | 1 | 2 | 4 |
| refund_request | 0 | 0 | 5 | 1 | 0 | 0 | 0 | 0 | 1 |
| billing_charge_dispute | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 6 |
| account_access_security | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 4 |
| order_status_inquiry | 3 | 0 | 0 | 0 | 0 | 3 | 0 | 0 | 6 |
| return_replacement_status | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 5 |
| general_complaint | 1 | 2 | 0 | 0 | 0 | 0 | 1 | 11 | 15 |
| other | 3 | 0 | 0 | 2 | 0 | 0 | 0 | 3 | 62 |

## Reply Quality Results

LLM judge results unavailable; no scores fabricated.

## Human Agreement

human scores not available

## Safety

The golden set was read for evaluation only. Its labels were not used for fitting, silver-label generation, or retrieval indexing.
