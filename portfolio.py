"""Portfolio state and the sector-cap rebalance layer.

`Book` is the plain state a persona holds; `rebalance` is the one pure
function that enforces the sector caps (F5, F7) and refills freed weight
into underweight buckets (F6, Q9). It is used both by the automatic sizing
chain (step 3, called with the sized target weights) and by the Rebalance
button (called with the book's own current weights, S8/user story 30) — the
caller decides what "target weights" means, this function only enforces
the caps and diversifies whatever it is handed.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from external.broker_models import Order, Position
from external.execution import build_orders


@dataclass(frozen=True)
class Book:
    """A persona's cash, whole-share positions and average cost per ticker."""

    cash: float
    shares: dict[str, int]
    avg_cost: dict[str, float]


@dataclass(frozen=True)
class RebalanceResult:
    """One rebalance proposal: the new targets, the orders and the before/after view."""

    target_weights: dict[str, float]
    orders: list[Order]
    before_sector_weights: dict[str, float]
    after_sector_weights: dict[str, float]


def _equity(book: Book, marks: dict[str, float]) -> float:
    """Total book value: cash plus every position marked at its current price."""
    positions_value = sum(
        shares * marks[ticker] for ticker, shares in book.shares.items()
    )
    return book.cash + positions_value


def weights_by_ticker(book: Book, marks: dict[str, float]) -> dict[str, float]:
    """Return each held ticker's weight, as a fraction of the book's equity."""
    equity = _equity(book, marks)
    return {
        ticker: shares * marks[ticker] / equity
        for ticker, shares in book.shares.items()
        if shares != 0
    }


def _sector_totals(weights: dict[str, float]) -> dict[str, float]:
    """Sum arbitrary ticker weights into their sector buckets (F9), all 7 present."""
    totals = {sector: 0.0 for sector in config.SECTORS}
    for ticker, weight in weights.items():
        sector = config.SECTOR_BY_TICKER.get(ticker)
        if sector is not None:
            totals[sector] += weight
    return totals


def sector_weights(book: Book, marks: dict[str, float]) -> dict[str, float]:
    """Return the book's weight in each of the 7 sector buckets, at `marks`."""
    return _sector_totals(weights_by_ticker(book, marks))


def _carry_forward_held_etfs(
    target_weights: dict[str, float],
    book: Book,
    marks: dict[str, float],
) -> dict[str, float]:
    """Add any bucket ETF the book still holds but that `target_weights` left out.

    The sized proposal only scores UNIVERSE stocks, so an ETF bought by an
    earlier refill would otherwise be missing from the targets — and
    `build_orders` closes any name missing from the targets.
    """
    equity = _equity(book, marks)
    bucket_etfs = set(config.ETF_BY_SECTOR.values())
    carried = dict(target_weights)

    for ticker, shares in book.shares.items():
        if ticker in carried or ticker not in bucket_etfs:
            continue
        carried[ticker] = shares * marks[ticker] / equity

    return carried


def _trim_over_cap_sectors(
    weights: dict[str, float],
    sector_cap: float,
) -> tuple[dict[str, float], float]:
    """Scale each over-cap bucket's tickers pro-rata down to the cap (F8).

    Returns the trimmed weights and the total weight freed by the trims.
    """
    trimmed = dict(weights)
    freed_weight = 0.0

    for sector in config.SECTORS:
        tickers_in_sector = [
            ticker
            for ticker in trimmed
            if config.SECTOR_BY_TICKER.get(ticker) == sector
        ]
        sector_weight = sum(trimmed[ticker] for ticker in tickers_in_sector)
        if sector_weight <= sector_cap:
            continue

        scale = sector_cap / sector_weight
        for ticker in tickers_in_sector:
            trimmed[ticker] *= scale
        freed_weight += sector_weight - sector_cap

    return trimmed, freed_weight


def _refill_underweight_etfs(
    weights: dict[str, float],
    refill_pool: float,
    equal_weight_target: float,
    sector_cap: float,
) -> dict[str, float]:
    """Spend `refill_pool` on the most-underweight buckets' ETFs (F6, Q9).

    A bucket is underweight when it sits below the equal-weight target.
    Each underweight bucket's ETF is topped up to the lesser of that target
    and the sector cap, most-underweight bucket first; whatever the pool
    can't cover stays in cash.
    """
    filled = dict(weights)
    remaining_pool = refill_pool

    sector_gaps = {
        sector: equal_weight_target - total
        for sector, total in _sector_totals(filled).items()
    }
    underweight_sectors = sorted(
        (sector for sector in config.SECTORS if sector_gaps[sector] > 0),
        key=lambda sector: sector_gaps[sector],
        reverse=True,
    )

    for sector in underweight_sectors:
        if remaining_pool <= 0:
            break

        etf = config.ETF_BY_SECTOR[sector]
        sector_weight = _sector_totals(filled)[sector]
        room = min(equal_weight_target, sector_cap) - sector_weight
        if room <= 0:
            continue

        top_up = min(room, remaining_pool)
        filled[etf] = filled.get(etf, 0.0) + top_up
        remaining_pool -= top_up

    return filled


def _positions(book: Book) -> dict[str, Position]:
    """Convert a Book's share counts into external.broker_models Positions."""
    return {
        ticker: Position(ticker=ticker, shares=shares)
        for ticker, shares in book.shares.items()
    }


def rebalance(
    book: Book,
    persona: config.Persona,
    target_weights: dict[str, float],
    marks: dict[str, float],
    caps_persona: config.Persona | None = None,
) -> RebalanceResult:
    """Trim over-cap sectors and refill underweight ETFs (F7, F8, F6, S8).

    `target_weights` is the proposal to enforce caps on: the sized weights
    for the automatic cycle, or `weights_by_ticker(book, marks)` for the
    Rebalance button. `caps_persona` lets a book be checked against a
    stricter persona's caps (R3); it defaults to `persona`.

    Builds whole-share orders with `build_orders` and drops any order
    under `config.MIN_TRADE_USD` (F11, sizing chain steps 4-5).
    """
    caps_persona = caps_persona or persona
    before_sector_weights = sector_weights(book, marks)

    working_weights = _carry_forward_held_etfs(target_weights, book, marks)
    equal_weight_target = sum(working_weights.values()) / len(config.SECTORS)

    trimmed_weights, freed_weight = _trim_over_cap_sectors(
        working_weights,
        caps_persona.sector_cap,
    )
    final_weights = _refill_underweight_etfs(
        trimmed_weights,
        freed_weight,
        equal_weight_target,
        caps_persona.sector_cap,
    )

    equity = _equity(book, marks)
    orders = build_orders(final_weights, _positions(book), marks, equity)
    sized_orders = [
        order
        for order in orders
        if order.price * order.quantity >= config.MIN_TRADE_USD
    ]

    return RebalanceResult(
        target_weights=final_weights,
        orders=sized_orders,
        before_sector_weights=before_sector_weights,
        after_sector_weights=_sector_totals(final_weights),
    )


if __name__ == "__main__":
    balanced = config.PERSONAS["balanced"]
    demo_marks = {ticker: 100.0 for ticker in config.SECTOR_BY_TICKER}
    demo_book = Book(
        cash=15_000.0,
        shares={
            "NVDA": 233,
            "AMD": 233,
            "AVGO": 234,
            "JPM": 50,
            "XOM": 30,
            "UNH": 20,
            "AAPL": 40,
            "CAT": 10,
        },
        avg_cost={},
    )
    demo_target = weights_by_ticker(demo_book, demo_marks)
    result = rebalance(demo_book, balanced, demo_target, demo_marks)

    assert result.after_sector_weights["Semiconductors"] <= balanced.sector_cap + 1e-9
    assert all(
        weight <= balanced.sector_cap + 1e-9
        for weight in result.after_sector_weights.values()
    )
    print("before:", {k: round(v, 3) for k, v in result.before_sector_weights.items()})
    print("after:", {k: round(v, 3) for k, v in result.after_sector_weights.items()})
    print(f"{len(result.orders)} orders")
