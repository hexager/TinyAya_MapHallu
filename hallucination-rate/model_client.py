"""
Thin wrapper around Cohere chat API for TinyAya Global and Water.
"""

import os
from typing import Tuple

import cohere

from config import COHERE_API_KEY, MAX_TOKENS, TEMPERATURE


def get_client() -> cohere.ClientV2:
    if not COHERE_API_KEY:
        raise RuntimeError("Set COHERE_API or COHERE_API_KEY in .env")
    return cohere.ClientV2(api_key=COHERE_API_KEY)


def query_model(
    query: str,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> Tuple[str, int]:
    """
    Send a single prompt to the Cohere chat API and return (response_text, output_token_count).
    """
    co = get_client()
    resp = co.chat(
        model=model,
        messages=[{"role": "user", "content": [{"type": "text", "text": query}]}],
        temperature=temperature if temperature is not None else TEMPERATURE,
        max_tokens=max_tokens if max_tokens is not None else MAX_TOKENS,
    )
    text = ""
    for block in resp.message.content:
        if hasattr(block, "text"):
            text += block.text
    text = text.strip()
    try:
        n_tokens = resp.usage.tokens.output_tokens
    except Exception:
        n_tokens = len(text.split())
    return text, n_tokens
