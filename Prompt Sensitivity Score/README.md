# PSS Pipeline — tiny-aya-global

Evaluates how sensitive [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global) is to prompt variations across four languages (en, es, hi, ar) using a **Prompt Sensitivity Score (PSS)**.

```
src/collect_data.py           →  data/raw_outputs.json
analysis/compute_pss.py       →  analysis/pss_results.csv
analysis/compute_pss_score.py →  analysis/pss_scores.csv, unstable_prompts.csv
plots/plot_pss_clean.py       →  plots/*.png
```

---

## Setup

```bash
git clone <repo-url> && cd PSS
cp .env.example .env          # add COHERE_API_KEY
uv sync
uv run python -m spacy download en_core_web_sm  # optional, improves NER
```

> Python 3.9+ required.

---

## Running the Pipeline

```bash
# Stage 1 — collect model outputs
uv run python src/collect_data.py

# Stage 2 — compute per-prompt metrics
uv run python analysis/compute_pss.py

# Stage 3 — compute composite PSS
uv run python analysis/compute_pss_score.py

# Stage 4 — generate plots
uv run python plots/plot_pss_clean.py
```

A sample dataset is available at `data/sample_outputs.jsonl` to test stages 2–4 without querying the API.

**Pause/resume:** create `data/PAUSE` to pause collection; delete to resume. Crashes are safe — completed records are skipped on restart.

---

## PSS Formula

```
PSS = (1 − semantic_similarity)
    × entity_change_rate
    × (1 − lexical_overlap)
    × (response_length_variance / mean(response_length_variance))
```

A high PSS means the model gives semantically different, factually inconsistent answers depending on how the question is phrased. Rows where `entity_change_rate` is null (ar/hi) produce a null PSS and are excluded from rankings.

---

## Configuration

All parameters are in `config/experiment_config.py`:

| Setting               | Default                                  |
|-----------------------|------------------------------------------|
| `MODEL_ID`            | `tiny-aya-global`                        |
| `LANGUAGES`           | `["en", "es", "hi", "ar"]`               |
| `NUM_PROMPTS`         | `100`                                    |
| `RANDOM_SEED`         | `42`                                     |
| `GENERATION_SETTINGS` | `{"temperature": 0, "max_tokens": 64}`   |
| `EMBED_MODEL_ID`      | `sentence-transformers/all-MiniLM-L6-v2` |

---

## Output Schema

Each line in `data/raw_outputs.json`:

```json
{
  "run_id":         "run_20260317T120000_abc12345",
  "prompt_id":      0,
  "language":       "en",
  "variant_type":   "base",
  "base_prompt":    "What is the capital of France?",
  "variant_prompt": "What is the capital of France?",
  "model":          "tiny-aya-global",
  "temperature":    0,
  "max_tokens":     64,
  "response":       "The capital of France is Paris.",
  "response_tokens": 7,
  "timestamp":      "2026-03-17T12:00:00+00:00"
}
```

`variant_type` ∈ `base | paraphrase | instruction | context | short`

---

## Project Structure

```
PSS/
├── config/
│   ├── __init__.py             # re-exports all params + COHERE_API_KEY
│   └── experiment_config.py    # all settings
├── src/
│   ├── collect_data.py         # Stage 1: data collection
│   └── helpers.py              # Cohere client + prompt variant generator
├── analysis/
│   ├── compute_pss.py          # Stage 2: 4 metrics → pss_results.csv
│   ├── compute_pss_score.py    # Stage 3: composite PSS → pss_scores.csv
│   └── utils.py                # shared analysis helpers
├── utils/
│   ├── embedding_utils.py      # semantic similarity (sentence-transformers)
│   ├── entity_utils.py         # NER (spaCy / regex fallback)
│   └── validate_dataset.py     # dataset validation
├── plots/
│   └── plot_pss_clean.py       # Stage 4: all plots
├── data/
│   └── sample_outputs.jsonl    # example data for offline testing
├── tests/
│   └── test_smoke.py
└── .env.example
```

---

## Troubleshooting

**spaCy model not found** — run `python -m spacy download en_core_web_sm`. Without it, `entity_change_rate` falls back to regex and is null for ar/hi.

**`ModuleNotFoundError: No module named 'analysis'`** — always run scripts from the project root.

**`JSONDecodeError` on raw_outputs.json** — the file is JSON Lines (one record per line), not a JSON array. Use `load_json_outputs()` from `analysis/utils.py`.

**Smoke tests** — `uv run python -m pytest tests/ -v`
