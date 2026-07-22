"""
vector_store.py
---------------
Adapter layer over ChromaDB.  All vector-store interactions go through this
module so that swapping to a different backend only requires changing this file
and updating config.yaml → vector_store.provider.

Key design decisions:
  • Cosine similarity distance space (1 − distance/2 → similarity score 0–1).
  • chunk_id is used as the ChromaDB document ID to guarantee idempotent inserts.
  • A module-level singleton is returned by `get_vector_store()`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings

from utils.config_loader import get_config
from utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    """Thin wrapper around a single ChromaDB collection."""

    def __init__(self) -> None:
        config = get_config()
        vs_cfg = config.vector_store

        # Persist on disk so data survives process restarts
        self._client = chromadb.PersistentClient(
            path=vs_cfg.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=vs_cfg.collection_name,
            metadata={"hnsw:space": vs_cfg.distance_metric},
        )
        logger.info(
            {
                "event": "vectorstore_ready",
                "provider": vs_cfg.provider,
                "collection": vs_cfg.collection_name,
                "chunk_count": self._collection.count(),
            }
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def chunk_exists(self, chunk_id: str) -> bool:
        """Return True if a chunk with this ID is already stored."""
        result = self._collection.get(ids=[chunk_id], include=[])
        return len(result["ids"]) > 0

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Upsert chunks into the collection.
        Skips any chunk whose chunk_id already exists (idempotent).

        Returns the number of *new* chunks inserted.
        """
        new_ids: List[str] = []
        new_embeddings: List[List[float]] = []
        new_documents: List[str] = []
        new_metadatas: List[Dict[str, Any]] = []

        for chunk, embedding in zip(chunks, embeddings):
            cid = chunk["chunk_id"]
            if not self.chunk_exists(cid):
                new_ids.append(cid)
                new_embeddings.append(embedding)
                new_documents.append(chunk["content"])
                # ChromaDB metadata values must be str | int | float | bool
                clean_meta = {
                    k: (str(v) if not isinstance(v, (str, int, float, bool)) else v)
                    for k, v in chunk["metadata"].items()
                }
                new_metadatas.append(clean_meta)

        if not new_ids:
            logger.info({"event": "no_new_chunks", "total_checked": len(chunks)})
            return 0

        self._collection.add(
            ids=new_ids,
            embeddings=new_embeddings,
            documents=new_documents,
            metadatas=new_metadatas,
        )
        logger.info({"event": "chunks_added", "count": len(new_ids)})
        return len(new_ids)

    # ── Read ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.30,
    ) -> List[Dict[str, Any]]:
        """
        Return the top-k most similar chunks above the score threshold.

        Each result dict:
            { "content": str, "metadata": dict, "score": float }
        """
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs: List[Dict[str, Any]] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [0, 1]
            similarity = round(1.0 - dist / 2.0, 4)
            if similarity >= score_threshold:
                docs.append({"content": doc, "metadata": meta, "score": similarity})

        logger.info(
            {"event": "retrieval_complete", "results_above_threshold": len(docs), "top_k": top_k}
        )
        return docs

    # ── Introspection ─────────────────────────────────────────────────────────

    def get_chunk_count(self) -> int:
        return self._collection.count()

    def get_all_documents_metadata(self) -> List[Dict[str, Any]]:
        """
        Return one metadata dict per unique ingested document (deduped by filename).
        """
        result = self._collection.get(include=["metadatas"])
        seen: Dict[str, Dict[str, Any]] = {}
        for meta in result["metadatas"]:
            fname = meta.get("filename", "unknown")
            if fname not in seen:
                seen[fname] = {
                    "filename": fname,
                    "total_pages": int(meta.get("total_pages", 0)),
                    "ingested_at": meta.get("ingested_at", ""),
                    "source_path": meta.get("source_path", ""),
                    "file_hash": meta.get("file_hash", ""),
                }
        return sorted(seen.values(), key=lambda d: d["ingested_at"], reverse=True)

    def delete_document(self, filename: str) -> int:
        """
        Remove all chunks that belong to *filename* from the collection.
        Returns the number of chunks deleted.
        """
        results = self._collection.get(
            where={"filename": filename}, include=[]
        )
        ids = results["ids"]
        if not ids:
            logger.info({"event": "delete_no_chunks_found", "filename": filename})
            return 0
        self._collection.delete(ids=ids)
        logger.info({"event": "document_deleted", "filename": filename, "chunks_removed": len(ids)})
        return len(ids)

    def document_ingested(self, filename: str) -> bool:
        """Quick check — has any chunk from this filename been stored?"""
        results = self._collection.get(
            where={"filename": filename}, include=[], limit=1
        )
        return len(results["ids"]) > 0


# ── Singleton factory ─────────────────────────────────────────────────────────

_store_instance: Optional[ChromaVectorStore] = None


def get_vector_store() -> ChromaVectorStore:
    """Return the module-level singleton ChromaVectorStore."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaVectorStore()
    return _store_instance
