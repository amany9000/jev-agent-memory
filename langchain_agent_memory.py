#!/usr/bin/env python3
"""LangChain 1.x agent whose Neo4j memory graph is built by TypeSafe.

    ingest  — split documents into passages and store each one in memory.
              ``MemoryClient`` was handed a ``TypeSafeExtractor``, so every stored
              passage becomes (:Entity) nodes + relationships in Neo4j Aura.
    ask     — run the agent. ``Neo4jMemoryMiddleware`` injects memory into the
              prompt and persists both turns, and those turns go through the same
              TypeSafe extractor, so the conversation keeps growing the graph.
    demo    — ingest ``data/docs`` then ask one question.

Usage:
    uv run python langchain_agent_memory.py demo
    uv run python langchain_agent_memory.py ingest data/docs/acme.md
    uv run python langchain_agent_memory.py ask "Who runs Acme's Berlin office?"
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, tool
from langchain_core.tools.retriever import create_retriever_tool
from neo4j_agent_memory import (
    BoltSettings,
    ExtractionConfig,
    ExtractorType,
    MemoryClient,
    Neo4jConfig,
)
from neo4j_agent_memory.integrations.langchain import (
    Neo4jMemoryMiddleware,
    Neo4jMemoryRetriever,
)
from pydantic import SecretStr
from typesafe_sdk import AsyncTypeSafeClient

import _env
from typesafe_extractor import TypeSafeExtractor, gliner_candidate_extractor

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from neo4j_agent_memory.extraction import ExtractionResult

DOCS_DIR = _env.ROOT / "data" / "docs"
SYSTEM_PROMPT = (
    "You are a concise assistant. Answer from the user's memory graph; "
    "use the search_memory tool when the injected memory is not enough."
)


# =====================================================================
# Memory: Neo4j Aura + TypeSafe as the graph maker
# =====================================================================
def build_settings() -> BoltSettings:
    """Bolt settings for Aura. The library's own extraction is switched off —
    ``TypeSafeExtractor`` replaces it via ``MemoryClient(extractor=...)``."""
    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    if _env.USES_OPENAI_EMBEDDINGS:
        _env.require("OPENAI_API_KEY")
    return BoltSettings(
        neo4j=Neo4jConfig(
            uri=_env.NEO4J_URI,
            username=_env.NEO4J_USERNAME,
            password=SecretStr(_env.NEO4J_PASSWORD),
            database=_env.NEO4J_DATABASE,
        ),
        llm=None,
        embedding=_env.EMBEDDING_MODEL,
        extraction=ExtractionConfig(
            extractor_type=ExtractorType.PIPELINE,
            enable_spacy=False,
            enable_gliner=False,
            enable_llm_fallback=False,
        ),
    )


def build_extractor(typesafe: AsyncTypeSafeClient) -> TypeSafeExtractor:
    return TypeSafeExtractor(
        typesafe,
        candidate_extractor=gliner_candidate_extractor(
            model=_env.GLINER_MODEL,
            threshold=_env.GLINER_THRESHOLD,
            device=_env.GLINER_DEVICE,
        ),
        model=_env.TYPESAFE_DEFAULT_MODEL,
    )


@asynccontextmanager
async def open_memory() -> AsyncIterator[tuple[MemoryClient[Any, Any, Any], TypeSafeExtractor]]:
    """A connected ``MemoryClient`` whose extraction runs through TypeSafe."""
    _env.require("TYPESAFE_API_KEY")
    async with AsyncTypeSafeClient() as typesafe:
        extractor = build_extractor(typesafe)
        async with MemoryClient(build_settings(), extractor=extractor) as client:
            yield client, extractor


def print_extraction(result: ExtractionResult | None, indent: str = "   ") -> None:
    if result is None or not result.entities:
        print(f"{indent}(no entities)")
        return
    for entity in result.entities:
        print(f"{indent}{entity.type:<13} {entity.name}  ({entity.confidence:.2f})")
    for relation in result.relations:
        print(
            f"{indent}  {relation.source} -[{relation.relation_type}]-> {relation.target}"
            f"  ({relation.confidence:.2f})"
        )


# =====================================================================
# Ingestion
# =====================================================================
def load_passages(paths: list[Path]) -> list[tuple[str, str]]:
    """(source, passage) pairs. Blank lines split passages; small passages keep
    TypeSafe's per-call question count (and pairwise relation checks) down."""
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.glob("*.md")) + sorted(path.glob("*.txt")) if path.is_dir() else [path])
    passages = []
    for file in files:
        for block in file.read_text(encoding="utf-8").split("\n\n"):
            block = block.strip()
            if block and not block.startswith("#"):
                passages.append((file.name, block))
    return passages


async def ingest(
    client: MemoryClient[Any, Any, Any], extractor: TypeSafeExtractor, paths: list[Path]
) -> None:
    passages = load_passages(paths)
    if not passages:
        raise SystemExit(f"No passages found in: {', '.join(map(str, paths))}")
    print(f"Ingesting {len(passages)} passages into session {_env.INGEST_SESSION_ID!r}")
    for n, (source, passage) in enumerate(passages, 1):
        # `extract_entities=True` hands the passage to TypeSafeExtractor; memory
        # then writes the entities, MENTIONS links and relationships to Neo4j.
        await client.short_term.add_message(
            _env.INGEST_SESSION_ID,
            "user",
            passage,
            extract_entities=True,
            metadata={"source": source, "kind": "document"},
        )
        print(f"\n[{n}/{len(passages)}] {source}: {passage[:70]}...")
        print_extraction(extractor.last_result)


# =====================================================================
# The agent
# =====================================================================
def build_model() -> tuple[BaseChatModel, bool]:
    """(model, supports_tools). Falls back to a scripted fake model offline."""
    if _env.OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI

        print(f"Model: ChatOpenAI({_env.OPENAI_MODEL!r}) — tools enabled")
        return ChatOpenAI(model=_env.OPENAI_MODEL), True

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    print("Model: FakeListChatModel (no OPENAI_API_KEY) — memory path only, no tools")
    return FakeListChatModel(responses=["(offline model) See the memory context above."]), False


def build_tools(client: MemoryClient[Any, Any, Any]) -> list[BaseTool]:
    retriever = Neo4jMemoryRetriever(
        memory_client=client,
        session_id=None,  # bolt: search every session, so ingested docs are found
        search_short_term=True,
        search_long_term=True,
        search_reasoning=False,
        k=8,
    )
    search_memory = create_retriever_tool(
        retriever,
        "search_memory",
        "Search memory: ingested documents, past messages, known entities and preferences.",
    )

    @tool
    async def save_preference(category: str, preference: str) -> str:
        """Record a preference the user just expressed (category: food, tools, ...)."""
        await client.long_term.add_preference(category, preference)
        return f"Saved preference: {category} — {preference}"

    return [search_memory, save_preference]


async def ask(
    client: MemoryClient[Any, Any, Any], extractor: TypeSafeExtractor, question: str
) -> str:
    model, supports_tools = build_model()
    agent = create_agent(
        model,
        tools=build_tools(client) if supports_tools else [],
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            Neo4jMemoryMiddleware(
                client,
                session_id=_env.CHAT_SESSION_ID,
                include_reasoning=False,
                max_items=8,
                extract_entities=True,  # chat turns also go through TypeSafe
            )
        ],
    )
    # `ainvoke`, not `invoke`: the memory middleware only implements async hooks.
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].text
    print(f"\nUser:      {question}\nAssistant: {answer}")
    print("\nTypeSafe extraction from the last stored turn:")
    print_extraction(extractor.last_result)
    return answer


# =====================================================================
# CLI
# =====================================================================
async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_ingest = sub.add_parser("ingest", help="ingest documents into the memory graph")
    p_ingest.add_argument("paths", nargs="*", type=Path, default=[DOCS_DIR])
    p_ask = sub.add_parser("ask", help="ask the memory-backed agent a question")
    p_ask.add_argument("question")
    p_demo = sub.add_parser("demo", help="ingest data/docs, then ask one question")
    p_demo.add_argument("--question", default="Who leads Acme Robotics' Berlin office, and what do they use?")
    args = parser.parse_args()

    async with open_memory() as (client, extractor):
        if args.command in ("ingest", "demo"):
            await ingest(client, extractor, args.paths if args.command == "ingest" else [DOCS_DIR])
        if args.command == "ask":
            await ask(client, extractor, args.question)
        if args.command == "demo":
            await ask(client, extractor, args.question)


if __name__ == "__main__":
    asyncio.run(main())
