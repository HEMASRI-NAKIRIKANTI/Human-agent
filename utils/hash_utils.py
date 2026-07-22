"""
hash_utils.py
-------------
SHA-256 helpers used for deduplication throughout the ingestion pipeline.
"""

import hashlib


def compute_page_hash(filename: str, page_number: int) -> str:
    """
    Stable, unique identifier for a (document, page) pair.
    Used as the primary key in the vector store and ingestion registry.
    """
    content = f"{filename}::page::{page_number}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def compute_file_hash(file_bytes: bytes) -> str:
    """
    SHA-256 hash of the raw file bytes.
    Used to detect when a document has been re-uploaded unchanged.
    """
    return hashlib.sha256(file_bytes).hexdigest()


def compute_chunk_id(page_hash: str, chunk_index: int) -> str:
    """
    Unique ID for an individual text chunk derived from its parent page hash.
    """
    content = f"{page_hash}::chunk::{chunk_index}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
