# Hiver Support Agent

This system classifies AmazonHelp customer tweets into the repository's support-intent taxonomy. It retrieves similar historical customer/reply pairs from the processed AmazonHelp corpus. It drafts a concise reply grounded in those examples, then decides whether the case can be auto-handled or should be escalated to a human with a specific reason. The golden set remains held out from retrieval and is not used by the pipeline.

## Setup

The project targets Python 3.9+ and uses the virtual environment in `venv/` when it is available.

```bash
cd hiver-support-agent
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure Anthropic for classification and reply generation. Without a key, the classifier and drafter use their safe fallbacks and the pipeline will escalate low-confidence cases.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Alternatively, put `ANTHROPIC_API_KEY=sk-ant-...` in a local `.env` file. Do not commit credentials.

The repository expects these files for normal pipeline use:

- `data/processed/amazonhelp_subsample.csv`
- `golden_set/golden_set.csv`

The first retrieval run downloads or loads the `all-MiniLM-L6-v2` sentence-transformers model and creates a local cached index under `.cache/`. The model cache is normally stored by Hugging Face under `~/.cache/huggingface/`. Internet access is required the first time the model is not already cached.

### Rebuild Processed Data

The processed corpus is already included. To rebuild it from the Customer Support on Twitter dataset, install/configure Kaggle credentials at `~/.kaggle/kaggle.json`, obtain `twcs.csv`, and place it at `data/raw/twcs.csv`. Then run:

```bash
source venv/bin/activate
python data/fetch_data.py
python data/filter_brand.py --brand AmazonHelp \
  --raw-path data/raw/twcs.csv \
  --output-path data/processed/amazonhelp_subsample.csv \
  --sample-size 6000 \
  --seed 42
```

`data/fetch_data.py` verifies the raw dataset; downloading the Kaggle dataset requires valid Kaggle credentials and access to the dataset.

## Reproduce Phase 7

With the processed corpus present, activate the environment and run:

```bash
source venv/bin/activate
python -m py_compile src/pipeline.py src/retrieve.py src/draft_reply.py src/escalate.py
python -c 'from src.pipeline import run_pipeline; print(run_pipeline("My package has not arrived yet."))'
```

The pipeline creates or reuses `.cache/amazonhelp_retrieval.npz` and `.cache/amazonhelp_retrieval.pkl`. Delete those two cache files if the processed corpus is rebuilt and the index must be regenerated.

## Phase 8 Evaluation

Run a small held-out smoke evaluation without spending judge API credits:

```bash
source venv/bin/activate
python -m eval.run_eval --sample-size 5
```

Run the full 200-row classification and pipeline evaluation with:

```bash
python -m eval.run_eval
```

Reports are written to `eval/results/summary.md` and `eval/results/summary.json`; cached judge responses are stored in `eval/results/judge_cache.json`. Classification results include accuracy, macro F1, weighted F1, per-intent precision/recall/F1, and a nine-class confusion matrix for the Majority, TF-IDF, and main LLM classifiers. Reply judging scores relevance, groundedness, tone, conciseness, and overall quality on a 1-5 scale. LLM judging requires `ANTHROPIC_API_KEY`; without it, classification still runs and judge scores are recorded as unavailable rather than fabricated.

Human agreement requires manually entering ratings for approximately 40 replies using a score template and then running agreement analysis. Human scores are never generated automatically. The golden set is strictly held out from training, silver-label generation, retrieval indexing, and threshold tuning; it is read only for evaluation inputs and expected labels.

## Example Usage

```python
from src.pipeline import run_pipeline

result = run_pipeline("My package has not arrived yet.")
print(result)
```

The result includes the customer text, intent, confidence, retrieved examples, drafted reply, retrieval influence metadata, escalation decision, and escalation reason. Expected decisions are `auto_handle` or `escalate`.

## Data Safety and Limitations

- There is no real Amazon order, account, billing, or security API integration.
- Replies are grounded only in historical customer/reply examples and must not be treated as confirmed account actions.
- Anthropic classification and generation require `ANTHROPIC_API_KEY`; missing credentials trigger safe fallbacks.
- Retrieval uses semantic similarity between the new customer message and historical tweet text.
- `golden_set/golden_set.csv` is evaluation-only. Its labels are not used by the pipeline, and its tweet IDs are excluded from retrieval.
- Deterministic escalation phrase rules are lightweight heuristics, not an ML sentiment model.
