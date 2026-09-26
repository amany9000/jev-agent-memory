"""Neo4j Aura connectivity — skipped (not failed) without credentials or a reachable instance.

Run just this file to debug an Aura connection in isolation from the memory
library or the extractor: `uv run pytest tests/test_neo4j.py -v`.
"""

from __future__ import annotations

from urllib.parse import urlparse

import pytest

import _env

pytestmark = pytest.mark.neo4j


async def test_connects_and_authenticates(neo4j_driver) -> None:
    # neo4j_driver already called verify_connectivity(); reaching here is the assertion.
    assert neo4j_driver is not None


async def test_aura_uses_an_encrypted_scheme() -> None:
    parsed = urlparse(_env.NEO4J_URI)
    if parsed.hostname and parsed.hostname.endswith("databases.neo4j.io"):
        assert parsed.scheme == "neo4j+s", (
            f"Aura hosts need neo4j+s://, got {parsed.scheme}://"
        )


async def test_can_write_and_delete(neo4j_driver) -> None:
    """Proves write access, not just a read-only connection."""
    async with neo4j_driver.session(database=_env.NEO4J_DATABASE) as session:
        record = await (
            await session.run(
                "CREATE (n:PocHealthcheck {at: datetime()}) WITH n DELETE n RETURN 1 AS ok"
            )
        ).single()
        assert record["ok"] == 1


async def test_server_reports_a_version(neo4j_driver) -> None:
    async with neo4j_driver.session(database=_env.NEO4J_DATABASE) as session:
        # Aura can return more than one component row; the kernel is enough.
        result = await session.run(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name, versions[0] AS version, edition"
        )
        record = (await result.fetch(1))[0]
        await result.consume()
        assert record["name"]
        assert record["version"]
