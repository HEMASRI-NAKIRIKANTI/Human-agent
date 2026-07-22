"""
pdf_extractor.py
----------------
Extracts text, tables, and images from PDF documents at the page level.

Strategy:
  1. Use pdfplumber for text and table extraction from digital PDFs.
  2. Detect scanned pages (low character density) and apply Tesseract OCR.
  3. Use PyMuPDF (fitz) to extract embedded raster images.
  4. Return a list of per-page dicts ready to be serialised as JSON.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pdfplumber

from utils.config_loader import get_config
from utils.hash_utils import compute_file_hash, compute_page_hash
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _is_scanned(text: str, min_chars: int) -> bool:
    """Return True when extracted text is too sparse to be reliable (scanned page)."""
    return len(text.strip()) < min_chars


def _ocr_page(pdf_path: str, page_number: int, language: str, dpi: int) -> str:
    """Convert a single PDF page to an image and run Tesseract OCR on it."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        images = convert_from_path(
            pdf_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
        )
        if images:
            text: str = pytesseract.image_to_string(images[0], lang=language)
            return text.strip()
    except Exception as exc:
        logger.warning({"event": "ocr_failed", "page": page_number, "error": str(exc)})
    return ""


def _extract_images(pdf_path: str, page_number: int) -> List[Dict[str, Any]]:
    """
    Extract embedded raster images from a page using PyMuPDF.
    Each image is returned as a base64-encoded string with its format.
    """
    images: List[Dict[str, Any]] = []
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]  # fitz is 0-indexed
        image_list = page.get_images(full=True)

        for idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes: bytes = base_image["image"]
                ext: str = base_image.get("ext", "png")
                b64 = base64.b64encode(image_bytes).decode("utf-8")
                images.append(
                    {
                        "image_id": f"page{page_number}_img{idx}",
                        "format": ext,
                        "base64": b64,
                        "size_bytes": len(image_bytes),
                    }
                )
            except Exception as img_exc:
                logger.warning(
                    {
                        "event": "image_extract_failed",
                        "page": page_number,
                        "xref": xref,
                        "error": str(img_exc),
                    }
                )
        doc.close()
    except ImportError:
        logger.warning({"event": "pymupdf_not_installed", "page": page_number})
    except Exception as exc:
        logger.warning({"event": "image_extraction_error", "page": page_number, "error": str(exc)})

    return images


def _parse_tables(raw_tables: list | None, page_number: int) -> List[Dict[str, Any]]:
    """Normalise pdfplumber table data into a structured list of dicts."""
    if not raw_tables:
        return []

    tables: List[Dict[str, Any]] = []
    for tbl_idx, table in enumerate(raw_tables):
        if not table:
            continue
        headers = [str(cell).strip() if cell is not None else "" for cell in (table[0] or [])]
        rows = [
            [str(cell).strip() if cell is not None else "" for cell in row]
            for row in table[1:]
            if row
        ]
        tables.append(
            {
                "table_id": f"page{page_number}_table{tbl_idx}",
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "col_count": len(headers),
            }
        )
    return tables


# ── Main extraction function ──────────────────────────────────────────────────

def extract_pdf(pdf_path: str, filename: str) -> List[Dict[str, Any]]:
    """
    Extract all pages from a PDF file.

    Args:
        pdf_path:  Absolute path to the PDF on disk.
        filename:  Original filename (used for hashing and metadata).

    Returns:
        A list of page-level dicts, one per page, containing:
            doc_id, filename, page_number, total_pages,
            text, tables, images, is_scanned, metadata
    """
    config = get_config()
    pages_data: List[Dict[str, Any]] = []

    # ── File-level hash ───────────────────────────────────────────────────────
    with open(pdf_path, "rb") as fh:
        file_bytes = fh.read()
    file_hash = compute_file_hash(file_bytes)
    ingested_at = datetime.now(timezone.utc).isoformat()

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        logger.info({"event": "extraction_start", "filename": filename, "total_pages": total_pages})

        for page_num, page in enumerate(pdf.pages, start=1):
            logger.info({"event": "extracting_page", "page": page_num, "total": total_pages})

            # ── Text ─────────────────────────────────────────────────────────
            raw_text: str = page.extract_text() or ""
            ocr_applied = False

            if config.ingestion.ocr_enabled and _is_scanned(
                raw_text, config.ingestion.min_text_chars_for_digital
            ):
                logger.info({"event": "ocr_triggered", "page": page_num})
                raw_text = _ocr_page(
                    pdf_path,
                    page_num,
                    config.ingestion.ocr_language,
                    config.ingestion.ocr_dpi,
                )
                ocr_applied = True

            text = raw_text.strip()

            # ── Tables ───────────────────────────────────────────────────────
            tables: List[Dict[str, Any]] = []
            try:
                tables = _parse_tables(page.extract_tables(), page_num)
            except Exception as exc:
                logger.warning({"event": "table_extract_failed", "page": page_num, "error": str(exc)})

            # ── Images ───────────────────────────────────────────────────────
            images: List[Dict[str, Any]] = []
            if config.ingestion.extract_images:
                images = _extract_images(pdf_path, page_num)

            # ── Assemble page record ─────────────────────────────────────────
            doc_id = compute_page_hash(filename, page_num)
            pages_data.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": page_num,
                    "total_pages": total_pages,
                    "text": text,
                    "tables": tables,
                    "images": images,
                    "is_scanned": ocr_applied,
                    "metadata": {
                        "file_hash": file_hash,
                        "source_path": str(pdf_path),
                        "ingested_at": ingested_at,
                        "page_bbox": list(page.bbox) if page.bbox else None,
                        "has_tables": len(tables) > 0,
                        "has_images": len(images) > 0,
                        "text_length": len(text),
                        "table_count": len(tables),
                        "image_count": len(images),
                        "ocr_applied": ocr_applied,
                    },
                }
            )

    logger.info(
        {"event": "extraction_complete", "filename": filename, "pages_extracted": len(pages_data)}
    )
    return pages_data


def save_extracted_json(
    pages_data: List[Dict[str, Any]], output_dir: str, filename: str
) -> str:
    """
    Persist the extracted page data as a JSON file.

    Returns the path of the saved file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(filename).stem
    json_path = out_dir / f"{stem}_extracted.json"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(pages_data, fh, indent=2, ensure_ascii=False)

    logger.info({"event": "json_saved", "path": str(json_path)})
    return str(json_path)
