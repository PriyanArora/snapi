"""The Chat page's saved conversations on the local JSONL log (Recent chats), and its reply text.

Runs offline: MONGODB_URI is unset and the log goes to a temporary folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import app
import books


def _chat(chat_id: str, updated_at: str, answer: str) -> dict:
    """Return a one-question chat as the Chat page saves it."""
    return {
        "chat_id": chat_id,
        "title": "What's my largest sector exposure?",
        "model_name": "claude-haiku-4-5-20251001",
        "created_at": "2026-09-27T14:00:00+00:00",
        "updated_at": updated_at,
        "turns": [
            {"role": "user", "content": "What's my largest sector exposure?"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_a_resaved_chat_lists_once_at_its_latest_version_newest_first(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Saving one chat_id twice keeps one entry, the latest; the most recently updated chat comes first."""
    monkeypatch.delenv(books.MONGODB_URI_ENV_VAR, raising=False)
    monkeypatch.setattr(books, "CHATS_PATH", tmp_path / "chats.jsonl")

    books.save_chat(_chat("first", "2026-09-27T14:00:00+00:00", "Technology."))
    books.save_chat(_chat("second", "2026-09-27T15:00:00+00:00", "Energy."))
    books.save_chat(_chat("first", "2026-09-27T16:00:00+00:00", "Still Technology."))

    chats = books.recent_chats()

    assert [chat["chat_id"] for chat in chats] == ["first", "second"]
    assert chats[0]["turns"][-1]["content"] == "Still Technology."
    assert [chat["chat_id"] for chat in books.recent_chats(limit=1)] == ["first"]


def test_reply_text_puts_a_blank_line_between_text_blocks() -> None:
    """Text on either side of an MCP tool call reads as two paragraphs, never "exposure.Let me"."""
    reply = AIMessage(
        content=[
            {"type": "text", "text": "Checking your exposure."},
            {"type": "mcp_tool_use", "id": "tool_1", "name": "get_holdings"},
            {"type": "text", "text": "Let me try another approach."},
        ]
    )

    assert app.reply_text(reply) == (
        "Checking your exposure.\n\nLet me try another approach."
    )
