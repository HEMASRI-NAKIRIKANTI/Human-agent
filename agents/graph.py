"""
graph.py
--------
Builds and compiles the LangGraph StateGraph that orchestrates the three agents.

Flow
----

  ┌─────────┐     ┌───────────┐     ┌───────────┐
  │  START  │────▶│ retriever │────▶│ generator │
  └─────────┘     └───────────┘     └─────┬─────┘
                                          │
                                          ▼
                                    ┌───────────┐
                                    │ validator │
                                    └─────┬─────┘
                                          │
                          ┌───────────────┴────────────────┐
                          │  is_valid OR retries exhausted  │
                          ▼                                 ▼
                        END                           generator  (retry loop)

Config-driven knobs
-------------------
  guardrails.max_retries   — maximum generator→validator cycles before giving up
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import END, StateGraph

from agents.generator_agent import generator_node
from agents.retriever_agent import retriever_node
from agents.state import AgentState, initial_state
from agents.validator_agent import validator_node
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Conditional routing ───────────────────────────────────────────────────────

def _route_after_validator(state: Dict[str, Any]) -> Literal["generator", "end"]:
    """
    After the validator runs, decide whether to:
      • retry  → send back to the generator
      • end    → surface the final_response to the caller
    """
    config = get_config()
    is_valid: bool = state.get("is_valid", False)
    retry_count: int = state.get("retry_count", 0)

    if is_valid:
        logger.info({"event": "routing_end", "reason": "valid_response"})
        return "end"

    if retry_count < config.guardrails.max_retries:
        logger.info(
            {"event": "routing_retry", "attempt": retry_count, "max": config.guardrails.max_retries}
        )
        return "generator"

    logger.warning({"event": "routing_end", "reason": "max_retries_exceeded"})
    return "end"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_agent_graph():
    """Construct and compile the LangGraph workflow. Returns a CompiledGraph."""
    workflow = StateGraph(AgentState)

    # Register nodes
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("validator", validator_node)

    # Linear path: retriever → generator → validator
    workflow.set_entry_point("retriever")
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", "validator")

    # Conditional: validator either retries via generator or terminates
    workflow.add_conditional_edges(
        "validator",
        _route_after_validator,
        {
            "generator": "generator",
            "end": END,
        },
    )

    compiled = workflow.compile()
    logger.info({"event": "graph_compiled"})
    return compiled


# ── Singleton ─────────────────────────────────────────────────────────────────

_graph: Optional[Any] = None


def get_agent_graph():
    """Return the module-level singleton compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_agent_graph()
    return _graph


# ── Public entrypoint ─────────────────────────────────────────────────────────

def run_query(
    query: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Execute the full agent pipeline for a given query.

    Args:
        query        : The user's question.
        chat_history : Previous conversation turns as
                       [{"role": "user|assistant", "content": "..."}].
                       The generator uses the last 3 turns for context.

    Returns the final AgentState dict which includes:
        final_response  : str   — the response to show the user
        citations       : list  — [{filename, page_number, score}, …]
        is_valid        : bool
        validation_reason : str
        retrieved_docs  : list
        retry_count     : int
    """
    graph = get_agent_graph()
    state = initial_state(query, chat_history=chat_history)

    logger.info({"event": "query_start", "query": query})
    result: Dict[str, Any] = graph.invoke(state)
    logger.info(
        {
            "event": "query_complete",
            "is_valid": result.get("is_valid"),
            "citations": len(result.get("citations", [])),
            "retries": result.get("retry_count", 0),
        }
    )
    return result
