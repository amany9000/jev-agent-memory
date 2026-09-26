#!/usr/bin/env python3
"""Health checks behind the `make check-*` targets. Each exits non-zero on failure.

    env        .env has the required values
    neo4j      Aura is reachable, credentials work, and the user can write
    memory     neo4j-agent-memory connects, sets up its schema, and round-trips a message
    typesafe   the TypeSafe API key works and answers a question
    extractor  GLiNER + TypeSafe extract typed entities and relations from a sample
               (no Neo4j involved)
    graph      what TypeSafe has written to the memory graph so far
"""

from __future__ import annotations

import asyncio
import sys
from urllib.parse import urlparse

import _env

OK, FAIL, WARN = "\033[32m✔\033[0m", "\033[31m✘\033[0m", "\033[33m!\033[0m"

SAMPLE = (
    "Maria Chen is the CTO of Acme Robotics. "
    "She opened the company's new office in Berlin in March and hired Tom Okafor to run it."
)
#: What a healthy extractor should find in SAMPLE (name -> expected type).
EXPECTED = {"Maria Chen": "PERSON", "Acme Robotics": "ORGANIZATION", "Berlin": "LOCATION"}


def fail(message: str, hint: str = "") -> None:
    print(f"{FAIL} {message}")
    if hint:
        print(f"  → {hint}")
    sys.exit(1)


# ---------------------------------------------------------------------------
def check_env() -> None:
    required = ["NEO4J_URI", "NEO4J_PASSWORD", "TYPESAFE_API_KEY"]
    if _env.USES_OPENAI_EMBEDDINGS:
        required.append("OPENAI_API_KEY")
    missing = [name for name in required if not getattr(_env, name)]
    for name in required:
        print(f"{FAIL if name in missing else OK} {name}")
    if not _env.USES_OPENAI_EMBEDDINGS:
        print(f"{OK if _env.OPENAI_API_KEY else WARN} OPENAI_API_KEY (optional: real chat model for the agent)")
    print(f"{OK} EMBEDDING_MODEL = {_env.EMBEDDING_MODEL}")
    if missing:
        fail("Missing required values in .env", "run `make env`, then fill them in")


async def check_neo4j() -> None:
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable

    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    uri = _env.NEO4J_URI
    parsed = urlparse(uri)
    print(f"URI:      {parsed.scheme}://{parsed.hostname}")
    print(f"User:     {_env.NEO4J_USERNAME}")
    print(f"Database: {_env.NEO4J_DATABASE}")
    if parsed.hostname and parsed.hostname.endswith("databases.neo4j.io") and parsed.scheme != "neo4j+s":
        print(f"{WARN} Aura needs an encrypted scheme — expected neo4j+s://, got {parsed.scheme}://")

    driver = AsyncGraphDatabase.driver(uri, auth=(_env.NEO4J_USERNAME, _env.NEO4J_PASSWORD))
    try:
        try:
            await driver.verify_connectivity()
        except AuthError:
            fail("Authentication failed", "check NEO4J_USERNAME / NEO4J_PASSWORD from the Aura credentials file")
        except ServiceUnavailable as error:
            fail(
                f"Cannot reach {parsed.hostname}: {error}",
                "is the Aura instance running (free instances pause after inactivity)? "
                "Is the URI neo4j+s://<id>.databases.neo4j.io?",
            )
        print(f"{OK} Connected and authenticated")

        async with driver.session(database=_env.NEO4J_DATABASE) as session:
            try:
                record = await (
                    await session.run(
                        "CALL dbms.components() YIELD name, versions, edition "
                        "RETURN name, versions[0] AS version, edition"
                    )
                ).single()
                print(f"{OK} Server: {record['name']} {record['version']} ({record['edition']})")

                # Write + delete in one transaction: proves the user can write.
                await (
                    await session.run(
                        "CREATE (n:PocHealthcheck {at: datetime()}) WITH n DELETE n RETURN 1"
                    )
                ).consume()
                print(f"{OK} Write permission (created and deleted a test node)")

                record = await (
                    await session.run(
                        "MATCH (n) RETURN count(n) AS nodes, "
                        "COUNT { MATCH (:Entity) } AS entities, COUNT { MATCH (:Message) } AS messages"
                    )
                ).single()
                print(
                    f"{OK} Database '{_env.NEO4J_DATABASE}': {record['nodes']} nodes "
                    f"({record['entities']} entities, {record['messages']} messages)"
                )
            except Neo4jError as error:
                fail(
                    f"Query failed: {error.code}: {error.message}",
                    "if the database is not found, set NEO4J_DATABASE from the Aura credentials file",
                )
    finally:
        await driver.close()
    print(f"\n{OK} Neo4j Aura connection is good")


async def check_memory() -> None:
    from neo4j_agent_memory import MemoryClient

    from langchain_agent_memory import build_settings

    session_id = "typesafe-poc-healthcheck"
    print(f"Embedder: {_env.EMBEDDING_MODEL}")
    # No TypeSafe here: this isolates the memory library + Aura + embedder.
    async with MemoryClient(build_settings()) as client:
        print(f"{OK} MemoryClient connected; schema and indexes ensured")
        await client.short_term.add_message(
            session_id, "user", "Healthcheck message from the TypeSafe POC.", extract_entities=False
        )
        conversation = await client.short_term.get_conversation(session_id)
        if not conversation.messages:
            fail("Stored a message but could not read it back")
        print(f"{OK} Message round-trip (with embedding) works")
        await client.short_term.clear_session(session_id)
        print(f"{OK} Cleaned up session {session_id!r}")
    print(f"\n{OK} Agent memory on Aura is working")


async def check_typesafe() -> None:
    from typesafe_sdk import AsyncTypeSafeClient, Choice, TypeSafeAuthenticationError, TypeSafeError

    _env.require("TYPESAFE_API_KEY")
    try:
        async with AsyncTypeSafeClient() as client:
            models = await client.models.list()
            names = ", ".join(m.name for m in models.models) or "(none listed)"
            print(f"{OK} Authenticated. Models: {names}")
            response = await client.system_one(
                state={"document": "Berlin is the capital of Germany."},
                questions={
                    "type": Choice(
                        instructions={"task": "Classify this mention.", "mention": "Berlin"},
                        criteria={"PERSON": None, "LOCATION": None, "ORGANIZATION": None},
                    )
                },
            )
    except TypeSafeAuthenticationError:
        fail("TypeSafe rejected the API key", "check TYPESAFE_API_KEY in .env")
    except TypeSafeError as error:
        fail(f"TypeSafe call failed: {error}")
    answer = response.choices["type"]
    print(f"{OK} system_one answered: Berlin -> {answer.choice} ({answer.confidence:.2f}) via {response.model}")
    if answer.choice != "LOCATION":
        print(f"{WARN} expected LOCATION")
    print(f"\n{OK} TypeSafe API is working")


async def check_extractor() -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    from langchain_agent_memory import build_extractor, print_extraction

    _env.require("TYPESAFE_API_KEY")
    print(f"Candidates: GLiNER {_env.GLINER_MODEL} @ {_env.GLINER_THRESHOLD} (first run downloads it)")
    print(f"Sample:     {SAMPLE}\n")
    async with AsyncTypeSafeClient() as typesafe:
        extractor = build_extractor(typesafe)
        candidates = await extractor.candidates.extract(
            SAMPLE, extract_relations=False, extract_preferences=False
        )
        print(f"GLiNER candidates ({len(candidates.entities)}):")
        for entity in candidates.entities:
            print(f"   {entity.type:<13} {entity.name}  ({entity.confidence:.2f})")
        if not candidates.entities:
            fail("GLiNER found no candidates", "lower GLINER_THRESHOLD or check the gliner install")

        result = await extractor.extract(SAMPLE)
    print("\nAfter TypeSafe:")
    print_extraction(result)

    found = {e.name: e.type for e in result.entities}
    print()
    misses = 0
    for name, expected in EXPECTED.items():
        got = found.get(name)
        if got == expected:
            print(f"{OK} {name} -> {expected}")
        else:
            misses += 1
            print(f"{FAIL} {name}: expected {expected}, got {got or 'nothing'}")
    if result.relations:
        print(f"{OK} {len(result.relations)} relation(s) extracted")
    else:
        print(f"{WARN} no relations extracted (entities are fine; relations are best-effort)")
    if misses:
        fail(f"{misses}/{len(EXPECTED)} expected entities missing or mistyped")
    print(f"\n{OK} Extractor is working")


async def show_graph() -> None:
    from neo4j import AsyncGraphDatabase

    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    driver = AsyncGraphDatabase.driver(_env.NEO4J_URI, auth=(_env.NEO4J_USERNAME, _env.NEO4J_PASSWORD))
    try:
        async with driver.session(database=_env.NEO4J_DATABASE) as session:
            print("Entities by type:")
            result = await session.run(
                "MATCH (e:Entity) RETURN e.type AS type, count(*) AS n ORDER BY n DESC"
            )
            async for record in result:
                print(f"   {record['type'] or '?':<13} {record['n']}")

            print("\nRelationships (RELATED_TO, newest first):")
            result = await session.run(
                "MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) "
                "RETURN a.name AS a, r.relation_type AS rel, b.name AS b, r.confidence AS c "
                "ORDER BY r.created_at DESC LIMIT 25"
            )
            rows = [record async for record in result]
            for record in rows:
                print(f"   {record['a']} -[{record['rel']}]-> {record['b']}  ({record['c'] or 0:.2f})")
            if not rows:
                print("   (none yet — run `make ingest`)")
    finally:
        await driver.close()
    print("\nExplore visually in Aura's Query tab: MATCH p=(:Entity)-[:RELATED_TO]->(:Entity) RETURN p")


COMMANDS = {
    "neo4j": check_neo4j,
    "memory": check_memory,
    "typesafe": check_typesafe,
    "extractor": check_extractor,
    "graph": show_graph,
}

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "env":
        check_env()
    elif command in COMMANDS:
        asyncio.run(COMMANDS[command]())
    else:
        sys.exit(f"usage: checks.py {{env|{'|'.join(COMMANDS)}}}")
