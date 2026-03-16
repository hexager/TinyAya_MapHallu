"""
utils/embedding_utils.py — Embedding and cosine-similarity helpers.

Public API
----------
get_embed_model(model_id)       Cached SentenceTransformer instance.
get_embeddings(texts)           L2-normalised embeddings (neural or TF-IDF fallback).
avg_cosine_vs_base(embeddings)  Mean cosine sim of variants vs. base (row 0).
"""

import logging
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_EMBED_MODEL_CACHE: Optional[SentenceTransformer] = None


def get_embed_model(
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> SentenceTransformer:
    """
    Return a process-level cached SentenceTransformer instance.

    ``attn_implementation="eager"`` disables the SDPA kernel that causes
    crashes on some CPU + transformers>=4.36 combinations on Windows.
    """
    global _EMBED_MODEL_CACHE
    if _EMBED_MODEL_CACHE is None:
        logger.info("Loading embedding model: %s", model_id)
        _EMBED_MODEL_CACHE = SentenceTransformer(model_id)
    return _EMBED_MODEL_CACHE


def get_embeddings(
    texts: List[str],
    model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Encode ``texts`` into L2-normalised embeddings.

    Falls back to TF-IDF cosine vectors if sentence-transformers inference
    fails (e.g. PyTorch / Python 3.13 incompatibility on Windows).

    Returns
    -------
    np.ndarray of shape (len(texts), dim) — L2-normalised
    """
    try:
        model = get_embed_model(model_id)
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        logger.warning(
            "sentence-transformers encode failed (%s). "
            "Falling back to TF-IDF cosine embeddings.",
            exc,
        )
        return _tfidf_embeddings(texts)


def _tfidf_embeddings(texts: List[str]) -> np.ndarray:
    """
    L2-normalised TF-IDF vectors as a fallback embedding.

    Cosine similarity on these vectors measures lexical overlap, not deep
    semantics — treat results as approximate.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer(min_df=1, token_pattern=r"(?u)\b\w+\b")
    matrix = vec.fit_transform(texts).toarray().astype(np.float32)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def avg_cosine_vs_base(embeddings: np.ndarray) -> float:
    """
    Average cosine similarity of variant embeddings against the base.

    Row 0 is the base; rows 1..n-1 are the variants.
    Base-to-base similarity is excluded.

    Returns 1.0 when n < 2 (only base present).
    """
    if len(embeddings) < 2:
        return 1.0
    base     = embeddings[0]
    variants = embeddings[1:]
    sims     = variants @ base      # dot = cosine for L2-norm vectors
    return float(np.mean(sims))
