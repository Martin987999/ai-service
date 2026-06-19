"""Configuration loader.

Loads YAML config + environment variables into a single typed object.
集中加载 YAML 配置与环境变量。检索模式 / 重排开关 / 模型版本全部来自配置,改配置不改代码。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _get_bool(env_val: str | None, default: bool) -> bool:
    if env_val is None:
        return default
    return env_val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    raw: dict[str, Any]
    # secrets / runtime (from env)
    anthropic_api_key: str | None = None
    voyage_api_key: str | None = None
    allow_mock_fallback: bool = True
    log_level: str = "INFO"

    # ---- convenient typed accessors ----
    def get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in path:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    @property
    def has_anthropic(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def has_voyage(self) -> bool:
        return bool(self.voyage_api_key)


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no external dependency required)."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        # do not override an already-set real env var
        if key and key not in os.environ:
            os.environ[key] = val


@lru_cache(maxsize=1)
def load_settings(config_path: str | None = None) -> Settings:
    _load_dotenv()
    cfg_path = config_path or os.environ.get("RAG_CONFIG", "config/config.yaml")
    raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    return Settings(
        raw=raw,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") or None,
        voyage_api_key=os.environ.get("VOYAGE_API_KEY") or None,
        allow_mock_fallback=_get_bool(os.environ.get("ALLOW_MOCK_FALLBACK"), True),
        log_level=os.environ.get("LOG_LEVEL", raw.get("logging", {}).get("level", "INFO")),
    )
