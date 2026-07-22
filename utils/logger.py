"""
logger.py
---------
Structured JSON logger used across all modules.
Log level and output file are driven by config.yaml → logging section.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from utils.config_loader import get_config


class _JSONFormatter(logging.Formatter):
    """Emits log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if hasattr(record, "extra"):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)


# Track which loggers we have already configured to avoid duplicate handlers.
_configured: set[str] = set()


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with JSON formatting.
    Safe to call multiple times with the same name — handlers are added once.
    """
    logger = logging.getLogger(name)

    if name in _configured:
        return logger

    config = get_config()
    resolved_level = level or config.logging.level
    numeric_level = getattr(logging, resolved_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logger.propagate = False

    formatter = _JSONFormatter()

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── File handler ──────────────────────────────────────────────────────────
    log_path = Path(config.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    _configured.add(name)
    return logger
