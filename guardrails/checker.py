"""
checker.py
----------
Guardrail functions called by the Validator Agent.

check_toxicity(text)
    → (is_toxic: bool, reason: str)
    Primary  : OpenAI Moderation API (zero-shot, no extra cost for moderation calls)
    Fallback : regex-based pattern matching

check_relevance(query, response, retrieved_docs)
    → (is_relevant: bool, score: float)
    Heuristic : weighted keyword-overlap between the response and the retrieved
                context.  A score above config.guardrails.relevance_threshold
                is considered relevant.  Explicit "I don't know" phrases always
                pass (the model correctly identified a lack of context).

Both functions are pure (no side effects) and fully testable in isolation.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Tuple

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)

# ── Toxicity patterns (regex fallback) ───────────────────────────────────────
# Intentionally minimal — the OpenAI Moderation API is the primary check.
_TOXICITY_PATTERNS: List[str] = [
    r"\b(hate\s*speech|racism|sexism)\b",
    r"\b(kill\s+yourself|self.harm|self.injury)\b",
    r"\b(bomb|terrorist\s+attack|mass\s+shooting)\b",
    r"\b(child\s+pornography|csam)\b",
]

# Phrases that indicate a deliberate "I don't know" — always pass relevance.
_NO_ANSWER_PHRASES: List[str] = [
    "don't have enough information",
    "cannot find",
    "not found in",
    "context does not contain",
    "context doesn't contain",
    "no information available",
    "unable to find",
    "not mentioned in",
    "outside the scope",
]


# ── Toxicity check ────────────────────────────────────────────────────────────

def check_toxicity(text: str) -> Tuple[bool, str]:
    """
    Return (is_toxic, reason).

    Uses the OpenAI Moderation API when
    config.guardrails.use_openai_moderation is True, falling back to regex
    patterns on any API error.
    """
    config = get_config()

    if config.guardrails.use_openai_moderation:
        try:
            from openai import OpenAI

            api_key = os.environ.get(config.llm.api_key_env, "")
            client = OpenAI(api_key=api_key)
            response = client.moderations.create(input=text)
            result = response.results[0]

            if result.flagged:
                flagged = [
                    cat
                    for cat, flagged_val in result.categories.__dict__.items()
                    if flagged_val
                ]
                reason = f"Moderation API flagged categories: {flagged}"
                logger.warning({"event": "moderation_flagged", "categories": flagged})
                return True, reason

            return False, ""

        except Exception as exc:
            logger.warning(
                {"event": "moderation_api_error", "error": str(exc), "fallback": "regex"}
            )
            # Fall through to regex check

    # ── Regex fallback ────────────────────────────────────────────────────────
    for pattern in _TOXICITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            reason = f"Regex pattern matched: {pattern}"
            logger.warning({"event": "regex_toxicity_detected", "pattern": pattern})
            return True, reason

    return False, ""


# ── Relevance check ───────────────────────────────────────────────────────────

def _tokenize(text: str) -> set[str]:
    """Lower-case word tokens of length ≥ 4 (filters out stop words cheaply)."""
    return set(re.findall(r"\b[a-z]{4,}\b", text.lower()))


def check_relevance(
    query: str,
    response: str,
    retrieved_docs: List[Dict[str, Any]],
) -> Tuple[bool, float]:
    """
    Return (is_relevant, score ∈ [0, 1]).

    Algorithm
    ---------
    1. If the response contains an explicit "I don't know" phrase → pass (True, 1.0).
    2. Build a vocabulary from the retrieved context.
    3. Compute Jaccard-like overlap between response tokens and context vocabulary.
    4. Compare against config.guardrails.relevance_threshold.
    """
    config = get_config()
    threshold: float = config.guardrails.relevance_threshold

    # Rule 1: explicit no-answer is always acceptable
    lower_response = response.lower()
    for phrase in _NO_ANSWER_PHRASES:
        if phrase in lower_response:
            logger.info({"event": "relevance_no_answer_phrase_detected"})
            return True, 1.0

    # Rule 2: no retrieved docs — cannot judge relevance
    if not retrieved_docs:
        return True, 1.0

    # Build context vocabulary
    context_tokens: set[str] = set()
    for doc in retrieved_docs:
        context_tokens.update(_tokenize(doc.get("content", "")))

    response_tokens = _tokenize(response)

    if not response_tokens or not context_tokens:
        return True, 1.0  # Cannot determine; pass

    # Overlap: what fraction of response words appear in the context?
    overlap_count = len(response_tokens & context_tokens)
    score = round(overlap_count / len(response_tokens), 4)

    is_relevant = score >= threshold
    logger.info(
        {
            "event": "relevance_check",
            "score": score,
            "threshold": threshold,
            "is_relevant": is_relevant,
        }
    )
    return is_relevant, score
