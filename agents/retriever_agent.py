"""
retriever_agent.py
------------------
LangGraph node: Retriever Agent

Responsibilities:
  1. Embed the user query using the configured embedding model.
  2. Perform a similarity search against the ChromaDB vector store.
  3. Build a formatted context string that includes source attribution for
     every retrieved chunk.
  4. Populate `retrieved_docs` and `citations` in the shared state.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ingestion.embedder import embed_query
from ingestion.vector_store import get_vector_store
from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Retriever Agent node.

    Reads  : state["query"]
    Writes : state["retrieved_docs"], state["context"], state["citations"]
    """
    config = get_config()
    query: str = state["query"]

    logger.info({"event": "retriever_start", "query_preview": query[:120]})

    try:
        # ── Embed the query ───────────────────────────────────────────────────
        query_embedding = embed_query(query)

        # ── Vector similarity search ──────────────────────────────────────────
        store = get_vector_store()
        retrieved_docs: List[Dict[str, Any]] = store.similarity_search(
            query_embedding=query_embedding,
            top_k=config.retriever.top_k,
            score_threshold=config.retriever.score_threshold,
        )

        logger.info({"event": "retriever_results", "chunks_found": len(retrieved_docs)})

        # ── Build context & citations ─────────────────────────────────────────
        context_parts: List[str] = []
        seen_citations: set = set()
        citations: List[Dict[str, Any]] = []

        for i, doc in enumerate(retrieved_docs, start=1):
            meta = doc["metadata"]
            filename = meta.get("filename", "Unknown")
            page_num = int(meta.get("page_number", 0))
            score = doc["score"]

            context_parts.append(
                f"[Context {i}]\n"
                f"Source: {filename}, Page {page_num}\n"
                f"Relevance Score: {score:.2%}\n\n"
                f"{doc['content']}"
            )

            key = (filename, page_num)
            if key not in seen_citations:
                seen_citations.add(key)
                citations.append(
                    {"filename": filename, "page_number": page_num, "score": score}
                )

        context = "\n\n" + ("─" * 60) + "\n\n".join(context_parts)

        return {
            "retrieved_docs": retrieved_docs,
            "context": context,
            "citations": citations,
        }

    except Exception as exc:
        logger.error({"event": "retriever_error", "error": str(exc)}, exc_info=True)
        return {
            "retrieved_docs": [],
            "context": "",
            "citations": [],
            "error": str(exc),
        }
