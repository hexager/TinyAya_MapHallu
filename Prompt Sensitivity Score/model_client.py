"""
model_client.py — Cohere API wrapper for the PSS pipeline.

Calls CohereLabs/tiny-aya-global via the Cohere Python SDK.
The API key is read from config.py — never hardcoded here.

Usage (standalone)
------------------
    from model_client import generate_response
    text = generate_response("What is the capital of France?")

Usage (class interface, used by collect_data.py)
-------------------------------------------------
    from model_client import ModelClient
    client = ModelClient()
    text, _ = client.generate("What is the capital of France?")
"""

import logging
from typing import Tuple

import cohere

from config import COHERE_API_KEY, MAX_TOKENS, MODEL_ID, TEMPERATURE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validate the key at import time so researchers get a clear error early.
# ---------------------------------------------------------------------------
if COHERE_API_KEY == "<INSERT_YOUR_API_KEY_HERE>" or not COHERE_API_KEY.strip():
    raise ValueError(
        "Cohere API key is not set.\n"
        "Open config.py and replace the placeholder:\n"
        '    COHERE_API_KEY = "<INSERT_YOUR_API_KEY_HERE>"\n'
        "with your real key from https://dashboard.cohere.com/api-keys"
    )


class ModelClient:
    """
    Thin wrapper around the Cohere chat API.

    Initialises a ``cohere.Client`` once and exposes a ``generate`` method
    with the same signature as the previous HuggingFace client so the rest
    of the pipeline requires no changes.
    """

    def __init__(self) -> None:
        logger.info("Initialising Cohere client (model: %s).", MODEL_ID)
        # The client authenticates using the key from config.py.
        self._co = cohere.ClientV2(api_key=COHERE_API_KEY)
        self.model_id = MODEL_ID

    def generate(self, prompt: str) -> Tuple[str, int]:
        """
        Send ``prompt`` to the Cohere chat API and return the response.

        Parameters
        ----------
        prompt : str
            Plain-text user prompt.

        Returns
        -------
        response_text : str
            The model's reply (stripped of leading/trailing whitespace).
        response_tokens : int
            Approximate token count (from Cohere's usage metadata if available,
            otherwise estimated from word count).
        """
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        resp = self._co.chat(
            model=MODEL_ID,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )

        # Extract text from the response content blocks
        text = ""
        for block in resp.message.content:
            if hasattr(block, "text"):
                text += block.text
        text = text.strip()

        # Token count from usage metadata (output tokens only)
        try:
            n_tokens = resp.usage.tokens.output_tokens
        except Exception:
            n_tokens = len(text.split())  # rough fallback

        return text, n_tokens


# ---------------------------------------------------------------------------
# Convenience function (as requested in the spec)
# ---------------------------------------------------------------------------

_default_client: ModelClient | None = None


def generate_response(prompt: str) -> str:
    """
    Call the Cohere API and return the generated text.

    Uses a process-level cached ``ModelClient`` so the SDK is initialised
    only once per Python process.

    Parameters
    ----------
    prompt : str

    Returns
    -------
    str — model response text
    """
    global _default_client
    if _default_client is None:
        _default_client = ModelClient()
    text, _ = _default_client.generate(prompt)
    return text
