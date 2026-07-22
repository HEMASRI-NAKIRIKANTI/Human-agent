"""
config_loader.py
----------------
Loads config.yaml once and exposes it as dot-notation accessible ConfigNode objects.
All modules should import `get_config()` — never read the YAML directly.
"""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any

# Root of the project (one level up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class ConfigNode:
    """
    Wraps a dict so nested keys are accessible via dot notation.
    e.g. config.llm.model  instead of config["llm"]["model"]
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


def load_config(config_path: str | Path = _CONFIG_PATH) -> ConfigNode:
    with open(config_path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return ConfigNode(raw)


# ── Module-level singleton ────────────────────────────────────────────────────
_config: ConfigNode | None = None


def get_config() -> ConfigNode:
    """Return the singleton config. Loads from disk on first call."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config() -> ConfigNode:
    """Force a reload from disk (useful for testing)."""
    global _config
    _config = load_config()
    return _config
