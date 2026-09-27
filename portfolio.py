"""Portfolio state and the sizing chain (PRD "Sizing chain").

`Book` is the plain state a persona holds. `rebalance` is the pure function
that enforces the sector caps (F5, F7) and refills freed weight into
underweight buckets (F6, Q9); it is used both by the automatic sizing
chain (step 3, called with the sized target weights) and by the Rebalance
button (called with the book's own current weights, S8/user story 30) — the
caller decides what "target weights" means, this function only enforces
the caps and diversifies whatever it is handed. `size_persona` runs the
full chain — blend, Fear & Greed tilt, PM drops, per-name max, then
`rebalance` — and is called by the committee's size node (preview) and
again by `execute` against a fresh snapshot (Phase3).
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from external.broker_models import Order, Position
from external.construction import blend_signals
from external.execution import build_orders
from external.limits import RiskLimits, apply_limits
from external.models import Signal

SHARE_ROUNDING_TOLERANCE = 1e-9

# F12 blend weights and the long-only permission for both analysts.
MODEL_WEIGHTS = {
    "technical": config.TECHNICAL_WEIGHT,
    "sentiment": config.SENTIMENT_WEIGHT,
}
LONG_ONLY_APPROACHES = {"technical": "long_only", "sentiment": "long_only"}


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


def book_equity(book: Book, marks: dict[str, float]) -> float:
    """Total book value: cash plus every position marked at its current price."""
    positions_value = sum(
        shares * marks[ticker] for ticker, shares in book.shares.items()
    )
    return book.cash + positions_value


def weights_by_ticker(book: Book, marks: dict[str, float]) -> dict[str, float]:
    """Return each held ticker's weight, as a fraction of the book's equity."""
    equity = book_equity(book, marks)
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
) -> dict[str, float]:
    """Add any bucket ETF the book still holds but that `target_weights` left out.

    Carried at weight 0.0, not at the ETF's current weight: `build_orders`
    closes any name missing from the targets, so the ETF still needs an
    entry here, but the refill pool (sized off invested %, not off this
    dict) re-derives its real weight every cycle. Carrying the current
    weight instead would stack a fresh refill on top of it each cycle and
    push gross past invested % (Sizing chain steps 3-4).
    """
    bucket_etfs = set(config.ETF_BY_SECTOR.values())
    carried = dict(target_weights)

    for ticker in book.shares:
        if ticker in carried or ticker not in bucket_etfs:
            continue
        carried[ticker] = 0.0

    return carried


def _trim_over_cap_sectors(
    weights: dict[str, float],
    sector_cap: float,
) -> dict[str, float]:
    """Scale each over-cap bucket's tickers pro-rata down to the cap (F8)."""
    trimmed = dict(weights)

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

    return trimmed


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


def _weights_after_orders(
    book: Book,
    orders: list[Order],
    marks: dict[str, float],
    equity: float,
) -> dict[str, float]:
    """Return each ticker's weight after `orders` fill, not the pre-floor target.

    A cap-restoring sell can fall under the $200 floor and get dropped
    (F7 vs F11), so the book's real post-trade weight can sit above the
    target the floor left on paper. The before/after view reports this,
    the book that actually results, rather than the untouched target.
    """
    shares_after = dict(book.shares)
    for order in orders:
        delta = order.quantity if order.side == "buy" else -order.quantity
        shares_after[order.ticker] = shares_after.get(order.ticker, 0) + delta

    return {
        ticker: shares * marks[ticker] / equity
        for ticker, shares in shares_after.items()
        if shares != 0
    }


def orders_for_targets(
    book: Book,
    target_weights: dict[str, float],
    marks: dict[str, float],
    sector_cap: float,
) -> list[Order]:
    """Return whole-share orders that move `book` to `target_weights` (sizing chain steps 4-5).

    Builds orders with `build_orders`, then drops any order under
    `config.MIN_TRADE_USD` (F11) — except a sell in a sector that is
    currently over `sector_cap`. That sell is kept even under $200,
    because the hard cap (F7) outranks the trade floor (F11); see PRD
    "What changed during build" row 7. Shared by `rebalance` (preview) and
    `execute.execute` (fresh re-derivation) so both size orders the same
    way.
    """
    equity = book_equity(book, marks)

    # Absorbs float round-trip error (e.g. weight*equity/mark landing a hair
    # under a held share count) before build_orders floors shares, so a
    # target equal to the current weight never emits a spurious 1-share sell.
    orders = build_orders(
        target_weights,
        _positions(book),
        marks,
        equity * (1 + SHARE_ROUNDING_TOLERANCE),
    )

    over_cap_sectors = {
        sector
        for sector, weight in sector_weights(book, marks).items()
        if weight > sector_cap
    }

    kept_orders = []
    for order in orders:
        is_large_enough = order.price * order.quantity >= config.MIN_TRADE_USD
        restores_cap = (
            order.side == "sell"
            and config.SECTOR_BY_TICKER.get(order.ticker) in over_cap_sectors
        )
        if is_large_enough or restores_cap:
            kept_orders.append(order)

    return kept_orders


def rebalance(
    book: Book,
    persona: config.Persona,
    target_weights: dict[str, float],
    marks: dict[str, float],
    invested_pct: float,
    caps_persona: config.Persona | None = None,
) -> RebalanceResult:
    """Trim over-cap sectors and refill underweight ETFs (F7, F8, F6, Q9, S8).

    `target_weights` is the proposal to enforce caps on: the sized weights
    for the automatic cycle, or `weights_by_ticker(book, marks)` for the
    Rebalance button. `invested_pct` is the persona's target invested % at
    the current regime (`persona.invested_pct_for_regime(regime)`): Q9's
    equal-weight target is `invested_pct / 7`, and the refill pool is
    whatever of `invested_pct` the sector trims and per-name clamps left
    unallocated — so gross can never grow past `invested_pct`. `caps_persona`
    lets a book be checked against a stricter persona's caps (R3); it
    defaults to `persona`.

    Builds whole-share orders through `orders_for_targets`, which drops
    trades under `config.MIN_TRADE_USD` except cap-restoring sells
    (F11, F7, sizing chain steps 4-5).
    """
    caps_persona = caps_persona or persona
    before_sector_weights = sector_weights(book, marks)

    working_weights = _carry_forward_held_etfs(target_weights, book)
    equal_weight_target = invested_pct / len(config.SECTORS)

    trimmed_weights = _trim_over_cap_sectors(working_weights, caps_persona.sector_cap)
    refill_pool = max(0.0, invested_pct - sum(trimmed_weights.values()))
    final_weights = _refill_underweight_etfs(
        trimmed_weights,
        refill_pool,
        equal_weight_target,
        caps_persona.sector_cap,
    )

    orders = orders_for_targets(book, final_weights, marks, caps_persona.sector_cap)
    equity = book_equity(book, marks)

    return RebalanceResult(
        target_weights=final_weights,
        orders=orders,
        before_sector_weights=before_sector_weights,
        after_sector_weights=_sector_totals(
            _weights_after_orders(book, orders, marks, equity)
        ),
    )


def fear_greed_tilt(fear_greed: float | None) -> float:
    """Return Q6's market-level conviction tilt: +0.10 below 25, -0.10 above 75, else 0."""
    if fear_greed is None:
        return 0.0  # feed unavailable -> tilt skipped ($0 Cost Guarantee)

    if fear_greed < config.FEAR_GREED_LOW:
        return config.FEAR_GREED_TILT

    if fear_greed > config.FEAR_GREED_HIGH:
        return -config.FEAR_GREED_TILT

    return 0.0


def _tilted(signal: Signal, tilt: float) -> Signal:
    """Shift one analyst signal by the tilt, kept inside blend_signals' [-1, 1] range."""
    tilted_value = max(-1.0, min(1.0, signal.value + tilt))
    return signal.model_copy(update={"value": tilted_value})


def blended_convictions(signals: list[Signal]) -> dict[str, float]:
    """Return F12's conviction per ticker (0.6 x Technical + 0.4 x Sentiment), for the PM's prompt."""
    blend = blend_signals(
        signals,
        MODEL_WEIGHTS,
        1.0,
        mode="long_only",
        investment_approaches=LONG_ONLY_APPROACHES,
    )
    return blend.convictions


def size_persona(
    persona: config.Persona,
    signals: list[Signal],
    regime: config.Regime,
    book: Book,
    marks: dict[str, float],
    fear_greed: float | None,
    dropped_tickers: set[str],
) -> RebalanceResult:
    """Run the PRD sizing chain for one persona and return its targets and orders.

    Blend (F12, gross = the regime's invested %), F&G tilt (Q6), PM drops
    (R1), per-name max (F10), then the sector-cap layer, whole-share orders
    and the $200 floor inside `rebalance` (F5-F8, F11, Q9).
    """
    invested_pct = persona.invested_pct_for_regime(regime)

    # Model weights sum to 1, so shifting every analyst's value by the tilt
    # shifts the blended conviction by the same amount (Q6).
    tilt = fear_greed_tilt(fear_greed)
    tilted_signals = [_tilted(signal, tilt) for signal in signals]
    blend = blend_signals(
        tilted_signals,
        MODEL_WEIGHTS,
        invested_pct,
        mode="long_only",
        investment_approaches=LONG_ONLY_APPROACHES,
    )

    # R1: a veto frees the name's weight for the Q9 refill; nothing is
    # renormalized, so no other name is upsized.
    kept_weights = {
        ticker: weight
        for ticker, weight in blend.weights.items()
        if ticker not in dropped_tickers
    }

    limits = RiskLimits(
        max_position_pct=persona.max_position_pct, max_gross_exposure=invested_pct
    )
    limited_weights = apply_limits(kept_weights, limits).weights

    return rebalance(book, persona, limited_weights, marks, invested_pct)
