"""Settings for the POC, read from `.env` next to this file (copy `.env.example`)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool) -> bool:
    raw = _get(name, "true" if default else "false").lower()
    return raw in ("1", "true", "yes", "on")


# Neo4j Aura
NEO4J_URI = _get("NEO4J_URI")
NEO4J_USERNAME = _get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _get("NEO4J_PASSWORD")
NEO4J_DATABASE = _get("NEO4J_DATABASE", "neo4j")

# TypeSafe — used when USE_SPACY=false, and by tests/test_typesafe.py.
TYPESAFE_API_KEY = _get("TYPESAFE_API_KEY")
TYPESAFE_DEFAULT_MODEL = _get("TYPESAFE_DEFAULT_MODEL") or None

# Extractor: USE_SPACY=true -> spaCy+GLiNER; false -> TypeSafeExtractor.
USE_SPACY = _get_bool("USE_SPACY", True)
SPACY_MODEL = _get("SPACY_MODEL", "en_core_web_trf")
GLINER_MODEL = _get("GLINER_MODEL", "gliner-community/gliner_medium-v2.5")
GLINER_THRESHOLD = float(_get("GLINER_THRESHOLD", "0.5"))
GLINER_DEVICE = _get("GLINER_DEVICE", "cpu")

# FastEmbed. Don't change after the first ingest without `make fix-vector-indexes`.
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# DeepSeek chat model for the agent; falls back to a fake model without a key.
DEEPSEEK_API_KEY = _get("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL = _get("DEEPSEEK_MODEL", "deepseek-chat")

INGEST_SESSION_ID = _get("INGEST_SESSION_ID", "typesafe-poc-ingest")
CHAT_SESSION_ID = _get("CHAT_SESSION_ID", "typesafe-poc-chat")


def require(*names: str) -> None:
    """Exit with a readable message if any of ``names`` is unset."""
    missing = [name for name in names if not globals().get(name)]
    if missing:
        raise SystemExit(
            f"Missing in .env: {', '.join(missing)}. Run `make env` and fill them in."
        )
