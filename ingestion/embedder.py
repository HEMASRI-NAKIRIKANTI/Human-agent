"""
embedder.py
-----------
Generates text embeddings via the configured provider (default: OpenAI).
Always uses the model specified in config.yaml → embeddings.model.
Batches requests to stay within API rate limits.

Rate-limit handling
-------------------
Transient 429 / RateLimitError  → exponential back-off, up to MAX_RETRIES attempts.
Quota exhaustion (insufficient_quota) → raises a clear, actionable error immediately
  because retrying won't help — the user must add credits to their OpenAI account.
"""

from __future__ import annotations

import os
import time
from typing import List

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

_MAX_RETRIES = 4          # max attempts per batch before giving up
_BACKOFF_BASE = 2.0       # seconds; doubled on each retry


def _get_openai_client():
    from openai import OpenAI

    config = get_config()
    api_key = os.environ.get(config.embeddings.api_key_env, "")
    if not api_key:
        raise EnvironmentError(
            f"Embedding API key not found. "
            f"Set the '{config.embeddings.api_key_env}' environment variable."
        )
    return OpenAI(api_key=api_key)


def _embed_batch_with_retry(
    client,
    batch: List[str],
    model: str,
    dimensions: int,
    batch_num: int,
) -> List[List[float]]:
    """Embed a single batch with exponential back-off on transient rate limits."""
    from openai import RateLimitError

    delay = _BACKOFF_BASE
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.embeddings.create(
                model=model,
                input=batch,
                dimensions=dimensions,
            )
            return [item.embedding for item in response.data]

        except RateLimitError as exc:
            error_body = getattr(exc, "body", {}) or {}
            error_code = error_body.get("error", {}).get("code", "")

            # Quota exhaustion is permanent — no point retrying
            if error_code == "insufficient_quota":
                raise RuntimeError(
                    "OpenAI quota exhausted. Please add credits at "
                    "https://platform.openai.com/account/billing and try again."
                ) from exc

            # Transient rate-limit — back off and retry
            if attempt < _MAX_RETRIES:
                logger.warning(
                    {
                        "event": "rate_limit_backoff",
                        "batch": batch_num,
                        "attempt": attempt,
                        "wait_seconds": delay,
                    }
                )
                time.sleep(delay)
                delay *= 2
            else:
                raise RuntimeError(
                    f"OpenAI rate limit persisted after {_MAX_RETRIES} retries. "
                    "Try again in a few minutes."
                ) from exc

    return []  # unreachable


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Embed a list of texts.  Automatically splits into API-safe batches.
    Returns a list of embedding vectors in the same order as the input.
    """
    config = get_config()
    provider = config.embeddings.provider
    model = config.embeddings.model
    batch_size = config.embeddings.batch_size
    dimensions = config.embeddings.dimensions

    all_embeddings: List[List[float]] = []

    if provider == "openai":
        client = _get_openai_client()

        for batch_start in range(0, len(texts), batch_size):
            batch = texts[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            logger.info(
                {"event": "embedding_batch", "batch": batch_num, "size": len(batch), "model": model}
            )
            embeddings = _embed_batch_with_retry(client, batch, model, dimensions, batch_num)
            all_embeddings.extend(embeddings)
    else:
        raise NotImplementedError(
            f"Embedding provider '{provider}' is not yet implemented. "
            "Supported: openai"
        )

    logger.info({"event": "embedding_complete", "total_vectors": len(all_embeddings)})
    return all_embeddings


def embed_query(query: str) -> List[float]:
    """Convenience wrapper — embed a single query string."""
    return embed_texts([query])[0]
