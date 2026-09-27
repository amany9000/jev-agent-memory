#!/usr/bin/env python3
"""Operational commands: graph, fix-vector-indexes, wipe-test-data, wipe-entities.
See docs.md for what each does and why entities aren't auto-deleted."""

from __future__ import annotations

import asyncio
import logging
import sys

import _env

DEBUG_SESSION_PREFIX = "debug-"

logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def _driver(uri: str):  # type: ignore[no-untyped-def]
    from neo4j import AsyncGraphDatabase

    return AsyncGraphDatabase.driver(
        uri,
        auth=(_env.NEO4J_USERNAME, _env.NEO4J_PASSWORD),
        notifications_min_severity="OFF",
    )


async def show_graph() -> None:
    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    driver = _driver(_env.NEO4J_URI)
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


async def fix_vector_indexes() -> None:
    """Drop vector indexes sized for a different embedding model; recreated on next connect."""
    from fastembed_embedder import fastembed_dimensions

    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    dims = fastembed_dimensions(_env.EMBEDDING_MODEL)
    driver = _driver(_env.NEO4J_URI)
    try:
        async with driver.session(database=_env.NEO4J_DATABASE) as session:
            result = await session.run("SHOW VECTOR INDEXES YIELD name, options RETURN name, options")
            rows = [record async for record in result]
            stale = [
                (row["name"], (row["options"] or {}).get("indexConfig", {}).get("vector.dimensions"))
                for row in rows
            ]
            stale = [(name, size) for name, size in stale if size != dims]
            if not stale:
                print(f"All {len(rows)} vector indexes already match {dims} dims — nothing to do")
                return
            for name, size in stale:
                await (await session.run(f"DROP INDEX `{name}` IF EXISTS")).consume()
                print(f"Dropped {name} ({size} dims)")
    finally:
        await driver.close()
    print(f"\nDone. `make test` (tests/test_memory.py) recreates them at {dims} dims.")


async def wipe_test_data(prefix: str = DEBUG_SESSION_PREFIX) -> None:
    """Delete conversations/messages whose session id starts with ``prefix``."""
    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    driver = _driver(_env.NEO4J_URI)
    try:
        async with driver.session(database=_env.NEO4J_DATABASE) as session:
            record = await (
                await session.run(
                    "MATCH (c:Conversation) WHERE c.session_id STARTS WITH $prefix "
                    "OPTIONAL MATCH (c)-[:HAS_MESSAGE]->(m:Message) "
                    "DETACH DELETE c, m "
                    "RETURN count(DISTINCT c) AS conversations, count(DISTINCT m) AS messages",
                    {"prefix": prefix},
                )
            ).single()
            print(
                f"Deleted {record['conversations']} debug conversation(s) "
                f"(session_id starts with {prefix!r}) and {record['messages']} message(s)"
            )
    finally:
        await driver.close()
    print(
        "\nEntities are untouched. If your debug run created some, delete them by name: "
        "`make wipe-entities NAMES=\"Name One,Name Two\"`"
    )


async def wipe_entities(names: list[str]) -> None:
    """Delete specific :Entity nodes by exact name."""
    if not names:
        sys.exit('usage: utils.py wipe-entities "Name One,Name Two"')
    _env.require("NEO4J_URI", "NEO4J_PASSWORD")
    driver = _driver(_env.NEO4J_URI)
    try:
        async with driver.session(database=_env.NEO4J_DATABASE) as session:
            record = await (
                await session.run(
                    "MATCH (e:Entity) WHERE e.name IN $names DETACH DELETE e RETURN count(e) AS n",
                    {"names": names},
                )
            ).single()
            print(f"Deleted {record['n']} of {len(names)} named entities")
    finally:
        await driver.close()


COMMANDS = {
    "graph": show_graph,
    "fix-vector-indexes": fix_vector_indexes,
    "wipe-test-data": wipe_test_data,
}
ARG_COMMANDS = {"wipe-entities": wipe_entities}

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in COMMANDS:
        asyncio.run(COMMANDS[command]())
    elif command in ARG_COMMANDS:
        raw = sys.argv[2] if len(sys.argv) > 2 else ""
        names = [name.strip() for name in raw.split(",") if name.strip()]
        asyncio.run(ARG_COMMANDS[command](names))
    else:
        usage = "|".join([*COMMANDS, *ARG_COMMANDS])
        sys.exit(f"usage: utils.py {{{usage}}}")
