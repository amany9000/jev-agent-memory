"""FastEmbed embeddings for neo4j-agent-memory.

neo4j-agent-memory ships OpenAI / sentence-transformers / Vertex / Bedrock
adapters but none for FastEmbed, so this implements its ``EmbeddingProvider``
protocol (``model``, ``dimensions``, async ``embed`` + ``embed_one``) and is
passed as ``BoltSettings(embedding=FastEmbedProvider(...))``.

FastEmbed runs ONNX models locally — no API key, no torch at inference time.
The model downloads on first use (~70 MB for the default).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dims, FastEmbed's default


def fastembed_dimensions(model: str) -> int:
    """Dimension of a FastEmbed model, from its supported-models table."""
    from fastembed import TextEmbedding

    for spec in TextEmbedding.list_supported_models():
        if spec["model"].lower() == model.lower():
            return int(spec["dim"])
    raise ValueError(
        f"{model!r} is not a FastEmbed text model. "
        "See TextEmbedding.list_supported_models() for the options."
    )


class FastEmbedProvider:
    """``EmbeddingProvider`` backed by ``fastembed.TextEmbedding``.

    The model loads lazily on the first ``embed`` call, and inference runs in a
    worker thread so it does not block the event loop.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        cache_dir: str | None = None,
        batch_size: int = 64,
        threads: int | None = 1,
    ) -> None:
        self.model = model
        # Known up front: Neo4j vector indexes are sized from this at connect time.
        self.dimensions = fastembed_dimensions(model)
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        # Single-threaded by default: ONNX Runtime's own intra-op thread pool can
        # still be tearing down OS threads when the interpreter starts destroying
        # C++ statics at exit, which aborts the process with a libc++abi
        # "recursive_mutex lock failed" error (seen on macOS). threads=1 avoids
        # spawning that pool; testing (8 baseline runs vs 15 with threads=1)
        # didn't reproduce the crash with this set.
        self._threads = threads
        self._engine: Any = None

    def close(self) -> None:
        """Release the ONNX session. The next ``embed`` call reloads it."""
        self._engine = None

    def _load(self) -> Any:
        if self._engine is None:
            from fastembed import TextEmbedding

            self._engine = TextEmbedding(
                model_name=self.model, cache_dir=self._cache_dir, threads=self._threads
            )
        return self._engine

    def _embed_sync(self, texts: list[str]) -> list[list[float]]:
        engine = self._load()
        return [vector.tolist() for vector in engine.embed(texts, batch_size=self._batch_size)]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._embed_sync, list(texts))

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]
