"""Pure, offline sanity checks on configuration — no network, no API calls."""

from __future__ import annotations

import pytest

import _env


def test_required_settings_are_declared() -> None:
    """The names the rest of the suite (and the app) depends on actually exist."""
    for name in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD", "NEO4J_DATABASE"):
        assert hasattr(_env, name)
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_DEFAULT_MODEL"):
        assert hasattr(_env, name)
    for name in ("DEEPSEEK_API_KEY", "DEEPSEEK_MODEL"):
        assert hasattr(_env, name)
    assert hasattr(_env, "EMBEDDING_MODEL")


def test_embedding_model_is_a_known_fastembed_model() -> None:
    from fastembed_embedder import fastembed_dimensions

    # Raises ValueError for an unknown model name — this is what would catch a typo
    # in .env before it reaches Aura.
    dims = fastembed_dimensions(_env.EMBEDDING_MODEL)
    assert dims > 0


def test_require_raises_only_for_missing_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_env, "SOME_SETTING", "set", raising=False)
    monkeypatch.setattr(_env, "OTHER_SETTING", "", raising=False)
    _env.require("SOME_SETTING")  # present -> no error
    with pytest.raises(SystemExit, match="OTHER_SETTING"):
        _env.require("SOME_SETTING", "OTHER_SETTING")
