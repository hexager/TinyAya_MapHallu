"""
src/helpers.py — Shared Cohere API helpers for the PSS pipeline.

All model queries in the pipeline must go through this module.
The API key is loaded from .env — never hardcoded.

Public API
----------
query_model(query, model, temp, logprobs)  Send a prompt, return raw response.
get_text_from_response(response)           Extract generated text as str.
get_logprobs_from_response(response)       Extract logprobs or None.
"""

import logging
import os
from typing import Optional

import cohere
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: Optional[cohere.ClientV2] = None


def _get_client() -> cohere.ClientV2:
    """Return a process-level cached Cohere ClientV2 instance."""
    global _client
    if _client is None:
        api_key = os.environ.get("COHERE_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "COHERE_API_KEY is not set.\n"
                "Create a .env file in the project root with:\n"
                "    COHERE_API_KEY=your_key_here"
            )
        _client = cohere.ClientV2(api_key=api_key)
        logger.info("Cohere ClientV2 initialised.")
    return _client


def query_model(
    query: str,
    model: str = "tiny-aya-global",
    temp: float = 0.3,
    logprobs: bool = False,
):
    """
    Send a query to the Cohere chat API and return the raw response object.

    Parameters
    ----------
    query    : str    The user prompt text.
    model    : str    Cohere model ID.
    temp     : float  Sampling temperature (0 = greedy / deterministic).
    logprobs : bool   Whether to request token-level log probabilities.

    Returns
    -------
    cohere ChatResponse object — pass to get_text_from_response() or
    get_logprobs_from_response() to extract specific fields.
    """
    co = _get_client()
    return co.chat(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": query}]}],
        temperature=temp,
        logprobs=logprobs,
    )


def get_text_from_response(response) -> str:
    """
    Extract the generated text from a Cohere ChatResponse.

    Parameters
    ----------
    response : Cohere ChatResponse (output of query_model)

    Returns
    -------
    str — stripped response text, empty string if nothing found
    """
    text = ""
    for block in response.message.content:
        if hasattr(block, "text"):
            text += block.text
    return text.strip()


def get_logprobs_from_response(response):
    """
    Extract log probabilities from a Cohere ChatResponse.

    Parameters
    ----------
    response : Cohere ChatResponse (output of query_model with logprobs=True)

    Returns
    -------
    logprobs object, or None if not available
    """
    try:
        return response.logprobs
    except AttributeError:
        return None
