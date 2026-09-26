"""Shared fixtures and marker plumbing for the test suite.

Markers (registered below, see ``pyproject.toml`` too):

    neo4j      needs NEO4J_URI / NEO4J_PASSWORD — skipped without them
    deepseek   needs DEEPSEEK_API_KEY — skipped without it
    typesafe   makes a REAL, BILLED TypeSafe call. Skipped unless you pass
               ``--run-typesafe`` (or run `make test-typesafe`). This is the
               only marker gated behind an explicit flag, on purpose: nothing
               in `make test` / a normal pytest run spends TypeSafe credits.

Everything else (GLiNER, the extractor's own logic against a fake TypeSafe
client, FastEmbed) runs unconditionally — it's local and free.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _env  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-typesafe",
        action="store_true",
        default=False,
        help="Run tests marked @pytest.mark.typesafe (makes real, billed TypeSafe API calls).",
    )


def pytest_configure(config: pytest.Config) -> None:
    for name, doc in [
        ("neo4j", "needs a live Neo4j Aura connection"),
        ("deepseek", "needs DEEPSEEK_API_KEY"),
        ("typesafe", "makes a real, billed TypeSafe API call — opt in with --run-typesafe"),
    ]:
        config.addinivalue_line("markers", f"{name}: {doc}")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-typesafe"):
        return
    skip = pytest.mark.skip(reason="real TypeSafe call — pass --run-typesafe (or `make test-typesafe`) to run it")
    for item in items:
        if "typesafe" in item.keywords:
            item.add_marker(skip)


def require_env(*names: str) -> None:
    """Skip the current test if any of ``names`` is unset in ``_env``."""
    missing = [name for name in names if not getattr(_env, name, None)]
    if missing:
        pytest.skip(f"not configured: {', '.join(missing)} (see .env.example)")


@pytest.fixture(scope="session")
def neo4j_creds() -> None:
    require_env("NEO4J_URI", "NEO4J_PASSWORD")


@pytest.fixture
async def neo4j_driver(neo4j_creds: None):
    """A connected async driver against the configured Aura instance.

    Skips the test (rather than failing) if the instance is unreachable —
    that's a environment/connectivity problem, not a code regression, and a
    free Aura instance pauses itself after inactivity.
    """
    from neo4j import AsyncGraphDatabase
    from neo4j.exceptions import AuthError, ServiceUnavailable

    driver = AsyncGraphDatabase.driver(
        _env.NEO4J_URI,
        auth=(_env.NEO4J_USERNAME, _env.NEO4J_PASSWORD),
        notifications_min_severity="OFF",  # fresh DB: expected "label does not exist" noise
    )
    try:
        await driver.verify_connectivity()
    except AuthError:
        await driver.close()
        pytest.fail("Neo4j authentication failed — check NEO4J_USERNAME / NEO4J_PASSWORD in .env")
    except ServiceUnavailable as error:
        await driver.close()
        pytest.skip(f"Neo4j Aura unreachable ({error}) — instance may be paused")
    try:
        yield driver
    finally:
        await driver.close()


class FakeTypeSafeClient:
    """A stand-in for ``AsyncTypeSafeClient`` that answers from a lookup table.

    Used to test ``TypeSafeExtractor``'s own logic (dedup, batching, direction,
    confidence filtering) without spending a single real API call. Give it a
    mapping of lowercase mention/pair -> answer label, and it answers every
    ``Choice`` question with that label (or ``"NONE"``) at a fixed confidence.
    """

    def __init__(self, truth: dict[str, str], *, confidence: float = 0.9) -> None:
        self.truth = truth
        self.confidence = confidence
        self.calls: list[dict[str, Any]] = []

    async def system_one(self, *, state: Any, questions: dict[str, Any], model: str | None = None) -> Any:
        from types import SimpleNamespace

        self.calls.append({"questions": len(questions), "names": list(questions), "model": model})
        answers = {}
        for name, question in questions.items():
            instructions = question.instructions
            if "mention" in instructions:
                key = instructions["mention"].lower()
            else:
                key = (instructions["source"].lower(), instructions["target"].lower())
            label = self.truth.get(key, "NONE")
            assert label in question.criteria, f"{label!r} not in criteria for {name!r}"
            answers[name] = SimpleNamespace(choice=label, confidence=self.confidence)
        return SimpleNamespace(choices=answers)

    async def __aenter__(self) -> "FakeTypeSafeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None
