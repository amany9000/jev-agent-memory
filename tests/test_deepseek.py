"""DeepSeek chat model for the agent. Skipped without DEEPSEEK_API_KEY.

One real call — a single short round trip, not something worth gating behind
an opt-in flag the way TypeSafe is, but marked `deepseek` so it can be
excluded with `-m "not deepseek"` if you want a fully offline run.
"""

from __future__ import annotations

import pytest

import _env
from conftest import require_env

pytestmark = pytest.mark.deepseek


async def test_deepseek_replies() -> None:
    require_env("DEEPSEEK_API_KEY")
    from langchain_deepseek import ChatDeepSeek

    # No max_tokens cap: reasoning models spend output tokens thinking before
    # answering, so a small cap can return an empty reply.
    model = ChatDeepSeek(model=_env.DEEPSEEK_MODEL, api_key=_env.DEEPSEEK_API_KEY)
    reply = await model.ainvoke("Reply with the single word: pong")
    assert reply.text.strip(), f"{_env.DEEPSEEK_MODEL} returned an empty reply"
