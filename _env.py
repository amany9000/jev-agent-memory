"""Settings for the POC, read from `.env` next to this file (copy `.env.example`)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
# Also exports TYPESAFE_* into os.environ, where the TypeSafe SDK reads them.
load_dotenv(ROOT / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# Neo4j Aura — values come from the credentials file Aura gives you on creation.
NEO4J_URI = _get("NEO4J_URI")
NEO4J_USERNAME = _get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = _get("NEO4J_PASSWORD")
NEO4J_DATABASE = _get("NEO4J_DATABASE", "neo4j")

# TypeSafe — the graph maker.
TYPESAFE_API_KEY = _get("TYPESAFE_API_KEY")
TYPESAFE_DEFAULT_MODEL = _get("TYPESAFE_DEFAULT_MODEL") or None

# GLiNER — candidate span finder.
GLINER_MODEL = _get("GLINER_MODEL", "gliner-community/gliner_medium-v2.5")
GLINER_THRESHOLD = float(_get("GLINER_THRESHOLD", "0.3"))
GLINER_DEVICE = _get("GLINER_DEVICE", "cpu")

# Embeddings for message/entity vectors. Do not change after the first ingest:
# the Neo4j vector index is created with this model's dimension.
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "openai/text-embedding-3-small")
USES_OPENAI_EMBEDDINGS = EMBEDDING_MODEL.startswith("openai/")

# Needed for OpenAI embeddings (the SDK reads it from the environment), and gives
# the agent a real chat model when langchain-openai is installed.
OPENAI_API_KEY = _get("OPENAI_API_KEY")
OPENAI_MODEL = _get("OPENAI_MODEL", "gpt-5-mini")

INGEST_SESSION_ID = _get("INGEST_SESSION_ID", "typesafe-poc-ingest")
CHAT_SESSION_ID = _get("CHAT_SESSION_ID", "typesafe-poc-chat")


def require(*names: str) -> None:
    """Exit with a readable message if any of ``names`` is unset."""
    missing = [name for name in names if not globals().get(name)]
    if missing:
        raise SystemExit(
            f"Missing in .env: {', '.join(missing)}. Run `make env` and fill them in."
        )
