"""
validator_agent.py
------------------
LangGraph node: Validator Agent

Guardrail checks (all configurable via config.yaml):
  1. Empty / trivially short response
  2. Toxicity  — OpenAI Moderation API or regex fallback
  3. Relevance — keyword-overlap heuristic between response and retrieved context
  4. Max-retry exceeded — emit fallback if the generator keeps failing

On success  : sets is_valid=True, copies draft_response → final_response.
On failure  : sets is_valid=False so the graph can route back to the generator
              (up to guardrails.max_retries times), then substitutes the
              appropriate safe fallback string.
"""

from __future__ import annotations

from typing import Any, Dict

from guardrails.checker import check_relevance, check_toxicity
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validator Agent node.

    Reads  : state["draft_response"], state["query"], state["retrieved_docs"],
             state["retry_count"]
    Writes : state["is_valid"], state["validation_reason"], state["final_response"]
    """
    config = get_config()
    draft: str = state.get("draft_response", "")
    query: str = state["query"]
    retrieved_docs = state.get("retrieved_docs", [])
    retry_count: int = state.get("retry_count", 0)

    logger.info({"event": "validator_start", "retry_count": retry_count})

    # ── Guard 1: empty response ───────────────────────────────────────────────
    if not draft or len(draft.strip()) < 10:
        logger.warning({"event": "validation_failed", "reason": "empty_response"})
        return {
            "is_valid": False,
            "validation_reason": "Response is empty or too short.",
            "final_response": config.guardrails.off_topic_response,
        }

    # ── Guard 2: toxicity ─────────────────────────────────────────────────────
    if config.guardrails.check_toxicity:
        is_toxic, toxicity_detail = check_toxicity(draft)
        if is_toxic:
            logger.warning({"event": "toxicity_detected", "detail": toxicity_detail})
            return {
                "is_valid": False,
                "validation_reason": f"Toxicity detected: {toxicity_detail}",
                "final_response": config.guardrails.toxicity_response,
            }

    # ── Guard 3: relevance ────────────────────────────────────────────────────
    if config.guardrails.check_relevance and retrieved_docs:
        is_relevant, relevance_score = check_relevance(query, draft, retrieved_docs)
        if not is_relevant:
            logger.warning(
                {"event": "low_relevance", "score": relevance_score, "retry_count": retry_count}
            )
            if retry_count >= config.guardrails.max_retries:
                # Max retries exhausted — surface off-topic fallback
                return {
                    "is_valid": False,
                    "validation_reason": f"Low relevance after {retry_count} retries (score={relevance_score:.3f}).",
                    "final_response": config.guardrails.off_topic_response,
                }
            # Signal the graph to retry generation
            return {
                "is_valid": False,
                "validation_reason": f"Low relevance score: {relevance_score:.3f}",
            }

    # ── Guard 4: no retrieved documents ──────────────────────────────────────
    if not retrieved_docs:
        logger.info({"event": "no_context_available"})
        return {
            "is_valid": True,
            "validation_reason": "No context available; returning fallback.",
            "final_response": config.guardrails.off_topic_response,
        }

    # ── All checks passed ─────────────────────────────────────────────────────
    logger.info({"event": "validation_passed"})
    return {
        "is_valid": True,
        "validation_reason": "All guardrail checks passed.",
        "final_response": draft,
    }
