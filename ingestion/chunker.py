"""
chunker.py
----------
Splits page-level text into smaller chunks while preserving rich metadata
(filename, page number, chunk index) so that every chunk is fully traceable
back to its source document and page.
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_loader import get_config
from utils.hash_utils import compute_chunk_id
from utils.logger import get_logger

logger = get_logger(__name__)


def _table_to_text(table: Dict[str, Any]) -> str:
    """Represent a structured table as readable plain text."""
    headers = " | ".join(table.get("headers", []))
    rows = "\n".join(" | ".join(row) for row in table.get("rows", []))
    return f"[Table]\n{headers}\n{rows}" if headers else f"[Table]\n{rows}"


def create_chunks(pages_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert a list of page-level dicts into a flat list of chunk dicts.

    Each chunk dict has the shape:
        {
            "doc_id":    <page-level doc_id>,
            "chunk_id":  <unique SHA-256 for this chunk>,
            "content":   <text content>,
            "metadata":  { filename, page_number, chunk_index, ... }
        }

    Tables on each page are converted to plain text and appended to the page
    text before splitting so they are embedded alongside the prose.
    """
    config = get_config()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.ingestion.chunk_size,
        chunk_overlap=config.ingestion.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks: List[Dict[str, Any]] = []

    for page in pages_data:
        # ── Build full page text ──────────────────────────────────────────────
        parts: List[str] = []

        text = page.get("text", "").strip()
        if text:
            parts.append(text)

        for table in page.get("tables", []):
            table_text = _table_to_text(table)
            if table_text.strip():
                parts.append(table_text)

        full_text = "\n\n".join(parts).strip()
        if not full_text:
            logger.warning(
                {"event": "empty_page_skipped", "filename": page["filename"], "page": page["page_number"]}
            )
            continue

        # ── Split ─────────────────────────────────────────────────────────────
        text_chunks = splitter.split_text(full_text)

        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_id = compute_chunk_id(page["doc_id"], chunk_idx)
            all_chunks.append(
                {
                    "doc_id": page["doc_id"],
                    "chunk_id": chunk_id,
                    "content": chunk_text,
                    "metadata": {
                        # ── Traceability ──────────────────────────────────────
                        "filename": page["filename"],
                        "page_number": page["page_number"],
                        "total_pages": page["total_pages"],
                        "chunk_index": chunk_idx,
                        "total_chunks_in_page": len(text_chunks),
                        # ── Provenance ────────────────────────────────────────
                        "file_hash": page["metadata"]["file_hash"],
                        "source_path": page["metadata"]["source_path"],
                        "ingested_at": page["metadata"]["ingested_at"],
                        # ── Content flags ─────────────────────────────────────
                        "has_tables": page["metadata"]["has_tables"],
                        "has_images": page["metadata"]["has_images"],
                        "ocr_applied": page["metadata"]["ocr_applied"],
                    },
                }
            )

    logger.info(
        {
            "event": "chunking_complete",
            "pages_processed": len(pages_data),
            "chunks_created": len(all_chunks),
        }
    )
    return all_chunks
