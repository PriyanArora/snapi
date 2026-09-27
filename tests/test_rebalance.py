"""Seam 3: rebalance(book, persona) -> orders (PRD Testing Decisions).

Setup: a synthetic book of 70% semis under the Balanced persona's 30%
sector cap (test data only, not a demo book). Checks the outcome a user
would notice, at the SimBroker seam: caps hold after the orders fill, the
trim is pro-rata, and the freed weight lands in the most-underweight
bucket's ETF.
"""

from __future__ import annotations

import pytest

import config
from external.broker_models import Order
from external.sim import SimBroker
from portfolio import Book, rebalance, sector_weights, weights_by_ticker

BALANCED = config.PERSONAS["balanced"]

MARKS = {ticker: 100.0 for ticker in config.SECTOR_BY_TICKER}

# 70% semis (NVDA/AMD/AVGO), a few other small sectors, Consumer left at 0%
# (it has no stock per Q2) so it is unambiguously the most-underweight bucket.
SEED_SHARES = {
    "NVDA": 233,
    "AMD": 233,
    "AVGO": 234,
    "JPM": 50,  # Banks
    "XOM": 30,  # Energy
    "UNH": 20,  # Healthcare
    "AAPL": 40,  # Big Tech
    "CAT": 10,  # Industrials
}
SEED_CASH = 15_000.0


def _seeded_book() -> Book:
    return Book(cash=SEED_CASH, shares=dict(SEED_SHARES), avg_cost={})


def test_rebalance_trims_semis_and_refills_underweight_etf() -> None:
    """One rebalance run trims semis pro-rata, buys Consumer's ETF and stays under cap.

    Checks all of seam 3 in one pass: the 30% cap holds after the orders
    fill, the semis trim is pro-rata, and the weight it frees buys the
    most-underweight bucket's ETF (Consumer, left at 0% per Q2) (R10).
    """
    book = _seeded_book()
    before = sector_weights(book, MARKS)
    assert before["Semiconductors"] == pytest.approx(0.70)

    target = weights_by_ticker(book, MARKS)
    invested_pct = BALANCED.invested_pct_for_regime("risk_on")
    result = rebalance(book, BALANCED, target, MARKS, invested_pct)

    assert result.after_sector_weights["Semiconductors"] <= BALANCED.sector_cap + 1e-9

    scale = BALANCED.sector_cap / before["Semiconductors"]
    for ticker in ("NVDA", "AMD", "AVGO"):
        assert result.target_weights[ticker] == pytest.approx(target[ticker] * scale)

    consumer_etf = config.ETF_BY_SECTOR["Consumer"]
    assert result.target_weights[consumer_etf] > 0
    consumer_order = next(
        order for order in result.orders if order.ticker == consumer_etf
    )
    assert consumer_order.side == "buy"
    assert consumer_order.quantity > 0

    # Open the SimBroker at the book's starting equity, then buy the seed
    # positions, so it lands on the same cash the seeded Book already has
    # before applying the rebalance orders on top.
    opening_positions_cost = sum(
        shares * MARKS[ticker] for ticker, shares in book.shares.items()
    )
    broker = SimBroker(cash=book.cash + opening_positions_cost)
    for ticker, shares in book.shares.items():
        broker.place_order(
            Order(ticker=ticker, side="buy", quantity=shares, price=MARKS[ticker])
        )
    for order in result.orders:
        broker.place_order(order)

    filled_book = Book(
        cash=broker.cash(),
        shares={
            ticker: position.shares for ticker, position in broker.positions().items()
        },
        avg_cost={},
    )
    filled_sector_weights = sector_weights(filled_book, MARKS)
    for sector, weight in filled_sector_weights.items():
        assert weight <= BALANCED.sector_cap + 1e-9, f"{sector} ended above its cap"
