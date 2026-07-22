"""
state.py
--------
Defines the shared AgentState TypedDict that flows through every node in the
LangGraph workflow.  All fields must have default-compatible types so that the
initial state can be constructed cleanly in graph.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    Fields
    ------
    query               : The raw user question.
    retrieved_docs      : Chunks returned by the retriever, each as a dict
                          { content, metadata, score }.
    context             : Formatted retrieval context injected into the LLM prompt.
    citations           : Unique (filename, page_number, score) triples sourced
                          from retrieved_docs; rendered in the UI.
    draft_response      : Raw LLM output before validation.
    final_response      : Response surfaced to the user (may differ from draft
                          if guardrails reject and a fallback is used).
    is_valid            : True once the validator approves the draft.
    validation_reason   : Human-readable reason from the last validation check.
    retry_count         : Number of generator→validator cycles completed so far.
    error               : Non-None when an unrecoverable exception occurred.
    """

    query: str
    chat_history: List[Dict[str, str]]   # [{"role": "user|assistant", "content": "..."}]
    retrieved_docs: List[Dict[str, Any]]
    context: str
    citations: List[Dict[str, Any]]
    draft_response: str
    final_response: str
    is_valid: bool
    validation_reason: str
    retry_count: int
    error: Optional[str]


def initial_state(
    query: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> AgentState:
    """Return a fully-initialised state for a new query."""
    return AgentState(
        query=query,
        chat_history=chat_history or [],
        retrieved_docs=[],
        context="",
        citations=[],
        draft_response="",
        final_response="",
        is_valid=False,
        validation_reason="",
        retry_count=0,
        error=None,
    )
