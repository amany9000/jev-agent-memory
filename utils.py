#!/usr/bin/env python3
"""Operational utility commands behind `make graph` / `make fix-vector-indexes`.

These are operational tools, not tests — see tests/ (pytest) for the checks
that used to live in this project's predecessor script, checks.py.

    graph               show entities and relationships written to Aura so far
    fix-vector-indexes  drop Aura vector indexes sized for a different embedding model
"""

from __future__ import annotations

import asyncio
import logging
import sys

import _env

# The memory library's own driver logs "label/property does not exist" notices on a
# fresh database; expected until the first ingest.
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
    """Drop vector indexes whose dimension differs from the configured embedder.

    Indexes only — nodes and their stored embeddings are left alone. The memory
    library recreates the indexes at the right size on its next connect.
    """
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


COMMANDS = {"graph": show_graph, "fix-vector-indexes": fix_vector_indexes}

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command in COMMANDS:
        asyncio.run(COMMANDS[command]())
    else:
        sys.exit(f"usage: utils.py {{{'|'.join(COMMANDS)}}}")
