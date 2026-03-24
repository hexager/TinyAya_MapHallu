# CMDR: Cross-Lingual Model Disagreement Runner

`CMDR.py` evaluates how stable a model is across languages on the same NLI examples.

It uses XNLI aligned data and asks the same task in each language:
- classify premise-hypothesis relation as `entailment`, `neutral`, or `contradiction`
- return JSON output with key `label`

Then it compares model behavior across language pairs.

## What `CMDR.py` does

1. Loads XNLI test split for selected languages (default: `en`, `hi`, `zh`, `fr`, `de`).
2. For each sample and each language:
   - sends a prompt to Cohere (`query_model`)
   - extracts predicted label from JSON (`get_text_from_response`)
   - extracts sequence confidence from token logprobs (`get_logprobs_from_response` + `calculate_sequence_probability`)
3. Builds language pairs:
   - only English-vs-other pairs if `english_only_pairs=True`
   - all combinations if `english_only_pairs=False`
4. Computes pairwise metrics:
   - `label_disagreement` = 1 if labels differ, else 0
   - `confidence_distance` = `abs(prob_a - prob_b)`
5. Writes detailed outputs for analysis/plotting.

## Output files (from `CMDR.py`)

By default in this project, outputs are written to `data/`:

- `data/cmdr_predictions.csv`  
  Per `(sample_id, language)` prediction with label and probability.
- `data/cmdr_sample_metrics.csv`  
  Per `(sample_id, lang_a, lang_b)` pair metrics, used by plotting script.
- `data/cmdr_summary.json`  
  Overall and per-pair aggregate statistics (mean/variance).

## Plotting (`Plot_metrics.py`)

`Plot_metrics.py` reads `cmdr_sample_metrics.csv` and saves plots + summaries.

Generated plots include:
- pairwise label disagreement bar chart
- pairwise confidence distance bar chart
- confidence distance box plot
- pairwise label disagreement heatmap
- English-vs-other label disagreement bar chart
- English-vs-other confidence distance bar chart

Also writes summary tables:
- `cmdr_pair_summary.csv`
- `cmdr_english_summary.csv`

## Complete run steps

### 1) Set API key

Create/update `.env` in repo root:

```env
COHERE_API=your_actual_api_key
```

### 2) Install dependencies

From project root:

```bash
uv sync
```

### 3) Run CMDR data generation

From project root:

```bash
uv run python CMDR/CMDR.py
```

This runs with the current defaults in `CMDR.py`:
- `num_samples=300`
- `languages=["en", "hi", "zh", "fr", "de"]`
- `english_only_pairs=True` (English vs each other language)
- `output_dir="data"`

If you want all pair combinations, set in `CMDR.py`:

```python
english_only_pairs=False
```

### 4) Generate plots

From project root:

```bash
uv run python CMDR/Plot_metrics.py --input-csv data/cmdr_sample_metrics.csv --output-dir plots
```

### 5) Check results

- Raw data: `data/`
- Figures: `plots/`
- Plot summaries: `plots/cmdr_pair_summary.csv`, `plots/cmdr_english_summary.csv`

## Notes

- Higher `Label Disagreement` means more cross-lingual inconsistency (potential hallucination sensitivity across languages).
- Higher `Confidence Disagreement` means confidence is unstable across languages even when labels may match.
- Increasing `num_samples` improves statistical reliability but increases API time/cost.
