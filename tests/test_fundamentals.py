"""The fundamentals tilt: the finance teammate's forward P/E and debt rules, per stock."""

from __future__ import annotations

import pytest

from feeds import Fundamentals
from portfolio import get_fundamentals_tilt


def test_fundamentals_tilt_follows_the_sector_limits() -> None:
    """Technology's limit is 35 and Financials' is 15; high debt costs 0.05 more, except for banks."""
    fundamentals = {
        "AAPL": Fundamentals(forward_pe=40.0, debt_to_equity=1.0),
        "MSFT": Fundamentals(forward_pe=60.0, debt_to_equity=None),
        "NVDA": Fundamentals(forward_pe=40.0, debt_to_equity=3.0),
        "JPM": Fundamentals(forward_pe=10.0, debt_to_equity=5.0),
        "XOM": Fundamentals(forward_pe=None, debt_to_equity=None),
        "TSLA": Fundamentals(forward_pe=-5.0, debt_to_equity=0.0),
    }

    # Above the limit, then above 1.5x it (52.5).
    assert get_fundamentals_tilt("AAPL", fundamentals) == pytest.approx(-0.05)
    assert get_fundamentals_tilt("MSFT", fundamentals) == pytest.approx(-0.10)

    # Debt above 2x equity stacks on the P/E tilt, but never for Financials.
    assert get_fundamentals_tilt("NVDA", fundamentals) == pytest.approx(-0.10)
    assert get_fundamentals_tilt("JPM", fundamentals) == 0.0

    # Missing, non-positive or absent data skips the tilt.
    assert get_fundamentals_tilt("XOM", fundamentals) == 0.0
    assert get_fundamentals_tilt("TSLA", fundamentals) == 0.0
    assert get_fundamentals_tilt("KO", fundamentals) == 0.0
