"""
config/__init__.py — Re-exports all experiment config + API key.

Existing pipeline code does ``from config import MODEL_ID`` etc.
This package keeps those imports working while the structured parameters
now live in config/experiment_config.py.

API key stays here (not in experiment_config.py) so it is never
accidentally committed alongside experiment parameters.
"""

from config.experiment_config import (  # noqa: F401
    DATA_DIR,
    DATA_PATHS,
    EMBED_MODEL_ID,
    ERRORS_LOG_FILE,
    EXPERIMENT_VERSION,
    GENERATION_SETTINGS,
    LANGUAGE_INSTRUCTIONS,
    LANGUAGE_NAMES,
    LANGUAGES,
    LOGS_DIR,
    MAX_TOKENS,
    MODEL_ID,
    N_PROMPTS,
    NUM_PROMPTS,
    PAUSE_FILE,
    PLOTS_DIR,
    PROMPT_VARIANTS,
    PSS_RESULTS_FILE,
    RANDOM_SEED,
    RAW_OUTPUTS_FILE,
    CSV_SUMMARY_FILE,
    TEMPERATURE,
)

# ---------------------------------------------------------------------------
# API credentials — loaded from .env (never hardcoded here)
# ---------------------------------------------------------------------------
import os as _os
from dotenv import load_dotenv as _load_dotenv

_load_dotenv()  # loads .env from project root if present

COHERE_API_KEY: str = _os.environ.get("COHERE_API_KEY", "")
if not COHERE_API_KEY:
    raise EnvironmentError(
        "COHERE_API_KEY is not set.\n"
        "Create a .env file in the project root with:\n"
        "    COHERE_API_KEY=your_key_here\n"
        "Or export it in your shell: export COHERE_API_KEY=your_key_here"
    )
