"""
pipeline.py
-----------
Orchestrates the full ingestion flow for a single PDF document:

  1. Dedup check  — skip if file hash matches an already-ingested record
  2. PDF extract  — page-level text / tables / images → list[dict]
  3. JSON persist — save extracted data to ./data/extracted_json/
  4. Chunk        — split text into overlapping chunks
  5. Embed        — generate OpenAI text-embedding-3-large vectors
  6. Store        — upsert into ChromaDB (chunk-level dedup via SHA-256 IDs)
  7. Register     — update the ingestion registry file

The optional `progress_callback(message: str, fraction: float)` is called at
each step so that the Streamlit UI can update a progress bar in real time.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ingestion.chunker import create_chunks
from ingestion.embedder import embed_texts
from ingestion.pdf_extractor import extract_pdf, save_extracted_json
from ingestion.vector_store import get_vector_store
from utils.config_loader import get_config
from utils.hash_utils import compute_file_hash
from utils.logger import get_logger

logger = get_logger(__name__)

ProgressCallback = Callable[[str, float], None]


# ── Ingestion Registry ────────────────────────────────────────────────────────

class IngestionRegistry:
    """
    Lightweight JSON file that records which documents have been ingested
    and with what file hash.  Used for fast file-level deduplication on upload.
    """

    def __init__(self, registry_path: str) -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _save(self) -> None:
        with open(self._path, "w", encoding="utf-8") as fh:
            json.dump(self._data, fh, indent=2)

    def is_registered(self, filename: str, file_hash: str) -> bool:
        """Return True if this exact file (by content hash) was already ingested."""
        record = self._data.get(filename)
        return record is not None and record.get("file_hash") == file_hash

    def register(
        self, filename: str, file_hash: str, pages: int, chunks_added: int
    ) -> None:
        self._data[filename] = {
            "file_hash": file_hash,
            "pages": pages,
            "chunks_added": chunks_added,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        self._save()
        logger.info({"event": "document_registered", "filename": filename})

    def unregister(self, filename: str) -> bool:
        """Remove a document record. Returns True if it existed."""
        if filename in self._data:
            del self._data[filename]
            self._save()
            logger.info({"event": "document_unregistered", "filename": filename})
            return True
        return False

    def get_all(self) -> Dict[str, Any]:
        return dict(self._data)


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_ingestion_pipeline(
    pdf_path: str,
    filename: str,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """
    Run the full ingestion pipeline for a single PDF.

    Returns a result dict with keys:
        status   : "success" | "skipped" | "warning" | "error"
        filename : str
        pages    : int          (on success)
        chunks_created : int   (on success)
        chunks_added   : int   (on success)
        json_path      : str   (on success)
        reason         : str   (on skipped/warning/error)
    """
    config = get_config()
    registry = IngestionRegistry(config.ingestion.registry_path)

    def _progress(msg: str, pct: float) -> None:
        logger.info({"event": "pipeline_progress", "step": msg, "pct": pct})
        if progress_callback:
            progress_callback(msg, pct)

    t_start = time.perf_counter()

    # ── Step 0: compute file hash ─────────────────────────────────────────────
    with open(pdf_path, "rb") as fh:
        file_bytes = fh.read()
    file_hash = compute_file_hash(file_bytes)

    # ── Step 1: file-level dedup check ────────────────────────────────────────
    if registry.is_registered(filename, file_hash):
        logger.info({"event": "document_skipped", "filename": filename, "reason": "duplicate"})
        return {"status": "skipped", "reason": "Document already ingested (identical content)", "filename": filename}

    try:
        # ── Step 2: extract PDF ───────────────────────────────────────────────
        _progress(f"Extracting content from '{filename}'…", 0.10)
        pages_data = extract_pdf(pdf_path, filename)

        if not pages_data:
            return {"status": "warning", "reason": "No pages could be extracted", "filename": filename}

        # ── Step 3: save extracted JSON ───────────────────────────────────────
        _progress("Saving extracted JSON…", 0.30)
        json_path = save_extracted_json(pages_data, config.ingestion.json_output_dir, filename)

        # ── Step 4: chunk ─────────────────────────────────────────────────────
        _progress("Splitting text into chunks…", 0.45)
        chunks = create_chunks(pages_data)

        if not chunks:
            return {"status": "warning", "reason": "No text content found in document", "filename": filename}

        # ── Step 5: embed ─────────────────────────────────────────────────────
        _progress(f"Generating embeddings for {len(chunks)} chunks…", 0.60)
        texts = [c["content"] for c in chunks]
        embeddings = embed_texts(texts)

        # ── Step 6: store in vector DB ────────────────────────────────────────
        _progress("Upserting into vector database…", 0.82)
        store = get_vector_store()
        chunks_added = store.add_chunks(chunks, embeddings)

        # ── Step 7: register ──────────────────────────────────────────────────
        registry.register(filename, file_hash, len(pages_data), chunks_added)
        elapsed = round(time.perf_counter() - t_start, 2)
        _progress("Ingestion complete ✓", 1.00)

        return {
            "status": "success",
            "filename": filename,
            "pages": len(pages_data),
            "chunks_created": len(chunks),
            "chunks_added": chunks_added,
            "json_path": json_path,
            "elapsed_seconds": elapsed,
        }

    except Exception as exc:
        logger.error({"event": "ingestion_error", "filename": filename, "error": str(exc)}, exc_info=True)
        return {"status": "error", "filename": filename, "reason": str(exc)}


# ── Delete pipeline ───────────────────────────────────────────────────────────

def delete_document_pipeline(filename: str) -> Dict[str, Any]:
    """
    Fully remove a document from the system:
      1. Delete all chunks from ChromaDB
      2. Remove from ingestion registry
      3. Delete the extracted JSON file (if it exists)

    Returns a result dict with status and chunk_count.
    """
    config = get_config()
    registry = IngestionRegistry(config.ingestion.registry_path)

    try:
        # Step 1: remove from vector store
        store = get_vector_store()
        chunks_removed = store.delete_document(filename)

        # Step 2: remove from registry
        registry.unregister(filename)

        # Step 3: remove extracted JSON
        stem = Path(filename).stem
        json_path = Path(config.ingestion.json_output_dir) / f"{stem}_extracted.json"
        if json_path.exists():
            json_path.unlink()
            logger.info({"event": "extracted_json_deleted", "path": str(json_path)})

        logger.info({"event": "delete_complete", "filename": filename, "chunks_removed": chunks_removed})
        return {"status": "success", "filename": filename, "chunks_removed": chunks_removed}

    except Exception as exc:
        logger.error({"event": "delete_error", "filename": filename, "error": str(exc)}, exc_info=True)
        return {"status": "error", "filename": filename, "reason": str(exc)}
