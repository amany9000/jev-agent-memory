#!/usr/bin/env python3
"""LangChain 1.x agent with a Neo4j memory graph.

    ingest  — split documents into passages and store each in memory.
    ask     — run the agent; both turns are stored the same way.
    demo    — ingest data/docs then ask one question.

Extractor: USE_SPACY=true (default) uses the standard spaCy+GLiNER pipeline;
USE_SPACY=false uses TypeSafeExtractor. See docs.md.

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

import _env
from fastembed_embedder import FastEmbedProvider
from typesafe_extractor import TypeSafeExtractor, gliner_candidate_extractor, typesafe_client

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from typesafe_sdk import AsyncTypeSafeClient

DOCS_DIR = _env.ROOT / "data" / "docs"
SYSTEM_PROMPT = (
    "You are a concise assistant. Answer from the user's memory graph; "
    "use the search_memory tool when the injected memory is not enough."
)


# =====================================================================
# Memory
# =====================================================================
def build_settings() -> BoltSettings:
    """Bolt settings for Aura. Extraction config only matters when USE_SPACY is
    true; otherwise open_memory() overrides it with TypeSafeExtractor."""
    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    return BoltSettings(
        neo4j=Neo4jConfig(
            uri=_env.NEO4J_URI,
            username=_env.NEO4J_USERNAME,
            password=SecretStr(_env.NEO4J_PASSWORD),
            database=_env.NEO4J_DATABASE,
        ),
        llm=None,
        embedding=FastEmbedProvider(_env.EMBEDDING_MODEL),
        extraction=ExtractionConfig(
            extractor_type=ExtractorType.PIPELINE,
            enable_spacy=_env.USE_SPACY,
            enable_gliner=True,
            enable_llm_fallback=False,
            spacy_model=_env.SPACY_MODEL,
            gliner_model=_env.GLINER_MODEL,
            gliner_threshold=_env.GLINER_THRESHOLD,
            gliner_device=_env.GLINER_DEVICE,
        ),
    )


def build_typesafe_extractor(typesafe: AsyncTypeSafeClient) -> TypeSafeExtractor:
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
async def open_memory() -> AsyncIterator[MemoryClient[Any, Any, Any]]:
    """A connected ``MemoryClient``: spaCy+GLiNER, or TypeSafeExtractor if USE_SPACY=false."""
    if _env.USE_SPACY:
        async with MemoryClient(build_settings()) as client:
            yield client
        return

    _env.require("TYPESAFE_API_KEY")
    async with typesafe_client() as typesafe:
        async with MemoryClient(build_settings(), extractor=build_typesafe_extractor(typesafe)) as client:
            yield client


async def entities_mentioned_by(client: MemoryClient[Any, Any, Any], message_id: Any) -> list[dict[str, Any]]:
    """Entities linked to ``message_id`` via MENTIONS (raw Cypher: no typed getter for this)."""
    return await client.query.cypher(
        "MATCH (m:Message {id: $message_id})-[:MENTIONS]->(e:Entity) "
        "RETURN e.name AS name, e.type AS type, e.confidence AS confidence",
        {"message_id": str(message_id)},
    )


def print_entities(entities: list[dict[str, Any]], indent: str = "   ") -> None:
    if not entities:
        print(f"{indent}(no entities)")
        return
    for row in entities:
        confidence = row.get("confidence")
        suffix = f"  ({confidence:.2f})" if confidence is not None else ""
        print(f"{indent}{row['type']:<13} {row['name']}{suffix}")
    print(f"{indent}(relationships aren't per-message — run `make graph` to see them)")


# =====================================================================
# Ingestion
# =====================================================================
def load_passages(paths: list[Path]) -> list[tuple[str, str]]:
    """(source, passage) pairs, split on blank lines."""
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


async def ingest(client: MemoryClient[Any, Any, Any], paths: list[Path]) -> None:
    passages = load_passages(paths)
    if not passages:
        raise SystemExit(f"No passages found in: {', '.join(map(str, paths))}")
    print(f"Ingesting {len(passages)} passages into session {_env.INGEST_SESSION_ID!r}")
    for n, (source, passage) in enumerate(passages, 1):
        message = await client.short_term.add_message(
            _env.INGEST_SESSION_ID,
            "user",
            passage,
            extract_entities=True,
            metadata={"source": source, "kind": "document"},
        )
        entities = await entities_mentioned_by(client, message.id)
        print(f"\n[{n}/{len(passages)}] {source}: {passage[:70]}...")
        print_entities(entities)


# =====================================================================
# The agent
# =====================================================================
def build_model() -> tuple[BaseChatModel, bool]:
    """(model, supports_tools). Falls back to a scripted fake model offline."""
    if _env.DEEPSEEK_API_KEY:
        from langchain_deepseek import ChatDeepSeek

        print(f"Model: ChatDeepSeek({_env.DEEPSEEK_MODEL!r}) — tools enabled")
        return ChatDeepSeek(model=_env.DEEPSEEK_MODEL, api_key=_env.DEEPSEEK_API_KEY), True

    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    print("Model: FakeListChatModel (no DEEPSEEK_API_KEY) — memory path only, no tools")
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


async def ask(client: MemoryClient[Any, Any, Any], question: str) -> str:
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
                extract_entities=True,
            )
        ],
    )
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
    answer = result["messages"][-1].text
    print(f"\nUser:      {question}\nAssistant: {answer}")

    conversation = await client.short_term.get_conversation(_env.CHAT_SESSION_ID)
    user_messages = [m for m in conversation.messages if m.role.value == "user"]
    print("\nEntities linked to your message:")
    entities = await entities_mentioned_by(client, user_messages[-1].id) if user_messages else []
    print_entities(entities)
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

    async with open_memory() as client:
        if args.command in ("ingest", "demo"):
            await ingest(client, args.paths if args.command == "ingest" else [DOCS_DIR])
        if args.command == "ask":
            await ask(client, args.question)
        if args.command == "demo":
            await ask(client, args.question)


if __name__ == "__main__":
    asyncio.run(main())
