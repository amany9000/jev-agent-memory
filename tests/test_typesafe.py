"""The only test in this suite that makes a REAL, BILLED TypeSafe API call.

Skipped by default. Run it deliberately:

    make test-typesafe
    # or
    uv run pytest tests/test_typesafe.py --run-typesafe -v

It makes exactly one `system_one` call with one question. Do not add more
real-API tests here — extend tests/test_extractor.py's FakeTypeSafeClient
tests instead, which cover TypeSafeExtractor's actual logic for free.
"""

from __future__ import annotations

import os

import pytest

from conftest import require_env
from typesafe_extractor import typesafe_client

pytestmark = pytest.mark.typesafe


async def test_system_one_classifies_a_known_fact() -> None:
    require_env("TYPESAFE_API_KEY")
    from typesafe_sdk import Choice

    async with typesafe_client() as client:
        response = await client.system_one(
            state={"document": "Berlin is the capital of Germany."},
            questions={
                "type": Choice(
                    instructions={"task": "Classify this mention.", "mention": "Berlin"},
                    criteria={"PERSON": None, "LOCATION": None, "ORGANIZATION": None},
                )
            },
        )
    answer = response.choices["type"]
    print(
        f"\nsystem_one via {os.environ.get('TYPESAFE_BASE_URL') or 'default TypeSafe endpoint'} "
        f"answered Berlin -> {answer.choice} ({answer.confidence:.2f}) via {response.model}"
    )
    assert answer.choice == "LOCATION"
