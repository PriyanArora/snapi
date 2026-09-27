"""Seam 2: execute(proposal, persona) -> fills on a SimBroker persona (PRD Testing Decisions).

Checks the outcomes a user would notice at this seam: sells go before
buys, every fill is whole shares, trades under $200 are dropped, cash
never goes negative, and re-running the same approved targets places no
new orders. Marks are the committed price snapshot's closes, so the test
runs offline; fill logs and the run log go to a temporary folder.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import books
import config
from committee import Proposal
from execute import execute
from prices import latest_marks

PERSONA_KEY = "conservative"
STARTING_CASH = config.PERSONAS[PERSONA_KEY].starting_cash
MARKS = latest_marks()


@pytest.fixture(autouse=True)
def isolate_logs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep every fill log and the run log out of data/."""
    monkeypatch.setattr(books, "DATA_DIR", tmp_path)
    monkeypatch.setattr(books, "RUN_LOG_PATH", tmp_path / "run_log.jsonl")


def _proposal(target_weights: dict[str, float]) -> Proposal:
    """Wrap one persona's approved target weights as a Proposal."""
    return Proposal(
        target_weights={PERSONA_KEY: target_weights},
        orders={PERSONA_KEY: []},
    )


def test_fills_are_sequenced_whole_and_floored_and_cash_stays_positive() -> None:
    """A rotation sells before it buys, in whole shares, skips sub-$200 trades and keeps cash positive.

    Every target sits inside Conservative's limits, so no cap- or
    limit-restoring exemption from the $200 floor applies here.
    """
    # Precondition: one share of each ETF is worth less than the $200 floor.
    assert MARKS["XLE"] < config.MIN_TRADE_USD
    assert MARKS["XLY"] < config.MIN_TRADE_USD

    execute(
        _proposal({"NVDA": 0.08, "AAPL": 0.08, "MSFT": 0.04, "XLE": 0.05}), PERSONA_KEY
    )
    held_xle_shares = books.load_book(PERSONA_KEY).shares["XLE"]

    # Close AAPL and add to MSFT (both over $200), trim XLE by under $200
    # and ask for one share of XLY (under $200): the last two must be dropped.
    xle_trim_shares = int(config.MIN_TRADE_USD // MARKS["XLE"])
    rotated_targets = {
        "NVDA": 0.08,
        "MSFT": 0.08,
        "XLE": (held_xle_shares - xle_trim_shares) * MARKS["XLE"] / STARTING_CASH,
        "XLY": 1.5 * MARKS["XLY"] / STARTING_CASH,
    }
    fills = execute(_proposal(rotated_targets), PERSONA_KEY)

    sides = [fill.side for fill in fills]
    assert "sell" in sides
    assert "buy" in sides
    assert "sell" not in sides[sides.index("buy") :]

    for fill in fills:
        assert isinstance(fill.quantity, int)
        assert fill.quantity > 0
        assert fill.quantity * fill.price >= config.MIN_TRADE_USD

    filled_tickers = {fill.ticker for fill in fills}
    assert "XLE" not in filled_tickers
    assert "XLY" not in filled_tickers

    assert books.load_book(PERSONA_KEY).cash >= 0


def test_rerunning_the_same_targets_places_no_new_orders() -> None:
    """Approving the same targets twice trades once: the second run finds the book already there."""
    proposal = _proposal({"NVDA": 0.08, "JPM": 0.08, "XLY": 0.10})

    first_fills = execute(proposal, PERSONA_KEY)
    second_fills = execute(proposal, PERSONA_KEY)

    assert first_fills
    assert second_fills == []
