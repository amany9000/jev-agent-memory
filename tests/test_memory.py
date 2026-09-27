"""neo4j-agent-memory + FastEmbed on Aura. Skipped without Neo4j credentials.

No TypeSafe here on purpose: entity extraction is off (`extract_entities=False`),
so this isolates the memory library + Aura + the local embedder. See
tests/test_extractor.py for the TypeSafe-driven extraction logic.
"""

from __future__ import annotations

import pytest

# "debug-" prefix: the convention any ad hoc/test session id follows so
# `make wipe-test-data` can find and delete it (see utils.py). Belt and braces
# here — this test also cleans up after itself via clear_session below.
SESSION_ID = "debug-pytest-memory"

pytestmark = pytest.mark.neo4j


@pytest.fixture
async def memory_client(neo4j_creds: None):
    from neo4j_agent_memory import MemoryClient
    from neo4j_agent_memory.llm.errors import EmbeddingDimensionMismatchError

    from langchain_agent_memory import build_settings

    try:
        async with MemoryClient(build_settings()) as client:
            yield client
    except EmbeddingDimensionMismatchError as error:
        pytest.fail(
            f"Aura's vector indexes were built for a different embedding size: "
            f"{str(error).splitlines()[0]} — run `make fix-vector-indexes`"
        )


async def test_message_round_trips_with_an_embedding(memory_client) -> None:
    await memory_client.short_term.clear_session(SESSION_ID)
    try:
        await memory_client.short_term.add_message(
            SESSION_ID, "user", "Healthcheck message from the TypeSafe POC.",
            extract_entities=False,
        )
        conversation = await memory_client.short_term.get_conversation(SESSION_ID)
        assert len(conversation.messages) == 1
        assert conversation.messages[0].content == "Healthcheck message from the TypeSafe POC."
    finally:
        await memory_client.short_term.clear_session(SESSION_ID)
