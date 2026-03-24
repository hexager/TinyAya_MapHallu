# TinyAya Global vs Water — Hallucination Rate

Modular experiments comparing **TinyAya Global** and **TinyAya Water** on MKQA. Output is **hallucination rate** (and accuracy) using MKQA gold answers: correct = model response contains a gold answer; hallucination rate = 1 − accuracy.

- **TinyAya Water**: strongest for Asia-Pacific and Europe.
- **Experiment 1**: Out-of-region language (e.g. Arabic) — does finetuning (Water) degrade performance on other languages?
- **Experiment 2**: Asia-Pacific / Europe — how do Global vs Water compare in-region?

Dataset: **MKQA** (500 samples by default). All runs log every request/response and write hallucination-rate summaries and one plot per experiment.

## Setup

From repo root (or from `hallucination-rate/`):

```bash
# Dependencies (parent project or install locally)
pip install cohere datasets matplotlib python-dotenv tqdm

# API key
export COHERE_API=your_key
# or .env in TinyAya_MapHallu or hallucination-rate with COHERE_API=...
```

## Run

From `hallucination-rate/`:

```bash
# Run both experiments (default: 500 MKQA samples)
python run_experiments.py

# Run only experiment 1 (out-of-region, default language: Arabic)
python run_experiments.py --experiment 1

# Run only experiment 2 (in-region)
python run_experiments.py --experiment 2

# Experiment 1 with Hebrew, 200 samples
python run_experiments.py --experiment 1 --language he --num-samples 200
```

Or run experiments independently:

```bash
python experiment_1_out_of_region.py --language ar --num-samples 500
python experiment_2_in_region.py --num-samples 500
```

## Output layout

- **`output/logs/`** — Per-run logs and JSONL (every query/response).
- **`output/results/`** — Per-experiment, per-model:
  - `results_<model>_<run_id>.json` — full results with `is_correct` per item.
  - `summary_<model>_<run_id>.json` — **hallucination_rate**, accuracy, n_correct, n_total.
- **`output/plots/`** — One plot per experiment: **hallucination rate** and accuracy (bar chart, Global vs Water).

## Metric

- **Correct**: normalized model response contains at least one MKQA gold answer (text or alias).
- **Hallucination rate** = 1 − (n_correct / n_total). Items with no gold answers are excluded from n_total.

## Config

Edit `config.py` to change:

- `MKQA_NUM_SAMPLES`, `DEFAULT_OUT_OF_REGION_LANG`
- `IN_REGION_LANGUAGES` / `OUT_OF_REGION_LANGUAGES`
- Paths under `OUTPUT_DIR`

## Module overview

| Module | Role |
|--------|------|
| `config.py` | Paths, model names, language lists, API key |
| `data/load_mkqa.py` | Load MKQA, sample N, get queries by language(s) |
| `model_client.py` | Cohere chat wrapper (query → response) |
| `metrics.py` | Correctness vs MKQA gold; hallucination rate |
| `evaluate.py` | Run model, add is_correct, compute & log hallucination rate |
| `plotting.py` | Hallucination-rate (and accuracy) bar chart |
| `experiment_1_out_of_region.py` | Exp 1: one out-of-region language |
| `experiment_2_in_region.py` | Exp 2: Asia-Pacific/Europe languages |
| `run_experiments.py` | CLI to run one or both experiments |
