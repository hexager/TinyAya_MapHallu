"""
config/experiment_config.py — Experiment-level parameters for the PSS pipeline.

All pipeline scripts read from this module.  Change a value here and it
propagates everywhere — no hunting through multiple files.

Sections
--------
1. Model
2. Embedding model (analysis stage)
3. Languages & instructions
4. Dataset / prompts
5. Prompt variants
6. Generation settings
7. Data paths
8. Experiment versioning
"""

# ---------------------------------------------------------------------------
# 1. Model
# ---------------------------------------------------------------------------
MODEL_ID = "tiny-aya-global"        # Cohere API model ID

# ---------------------------------------------------------------------------
# 2. Embedding model  (used in analysis/compute_pss.py)
# ---------------------------------------------------------------------------
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# 3. Languages & instructions
# ---------------------------------------------------------------------------
LANGUAGES = ["en", "es", "hi", "ar"]

LANGUAGE_NAMES = {
    "en": "English",
    "es": "Spanish",
    "hi": "Hindi",
    "ar": "Arabic",
}

# Suffix appended to every prompt so the model answers in the target language.
# MKQA queries are English-only; this bridges the gap for multilingual eval.
LANGUAGE_INSTRUCTIONS = {
    "en": "",
    "es": " Por favor responde en español.",
    "hi": " कृपया हिंदी में उत्तर दें।",
    "ar": " يرجى الإجابة باللغة العربية.",
}

# ---------------------------------------------------------------------------
# 4. Dataset / prompts
# ---------------------------------------------------------------------------
NUM_PROMPTS = 100       # base prompts sampled from MKQA
RANDOM_SEED = 42        # reproducible sampling

# ---------------------------------------------------------------------------
# 5. Prompt variants
# ---------------------------------------------------------------------------
PROMPT_VARIANTS = ["base", "paraphrase", "instruction", "context", "short"]

# ---------------------------------------------------------------------------
# 6. Generation settings
# ---------------------------------------------------------------------------
GENERATION_SETTINGS = {
    "temperature": 0,
    "max_tokens":  64,
}

# Flat aliases — used directly by model_client.py and collect_data.py.
TEMPERATURE = GENERATION_SETTINGS["temperature"]
MAX_TOKENS  = GENERATION_SETTINGS["max_tokens"]

# ---------------------------------------------------------------------------
# 7. Data paths
# ---------------------------------------------------------------------------
DATA_PATHS = {
    "data_dir":         "data",
    "raw_outputs":      "data/raw_outputs.jsonl",
    "run_summary":      "data/run_summary.csv",
    "errors_log":       "data/errors.log",
    "pause_file":       "data/PAUSE",
    "pss_results":      "analysis/pss_results.csv",
    "pss_scores":       "analysis/pss_scores.csv",
    "unstable_prompts": "analysis/unstable_prompts.csv",
    "run_metadata":     "analysis/run_metadata.json",
    "plots_dir":        "plots",
    "logs_dir":         "logs",
}

# Flat aliases — match the names used throughout the existing pipeline.
DATA_DIR         = DATA_PATHS["data_dir"]
RAW_OUTPUTS_FILE = DATA_PATHS["raw_outputs"]   # data/raw_outputs.jsonl
CSV_SUMMARY_FILE = DATA_PATHS["run_summary"]
ERRORS_LOG_FILE  = DATA_PATHS["errors_log"]
PAUSE_FILE       = DATA_PATHS["pause_file"]
PSS_RESULTS_FILE = DATA_PATHS["pss_results"]
PLOTS_DIR        = DATA_PATHS["plots_dir"]
LOGS_DIR         = DATA_PATHS["logs_dir"]

# Backward-compat alias (old config.py used N_PROMPTS).
N_PROMPTS = NUM_PROMPTS

# ---------------------------------------------------------------------------
# 8. Experiment versioning
# ---------------------------------------------------------------------------
EXPERIMENT_VERSION = "1.0.0"
