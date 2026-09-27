"""Portfolio state and the sizing chain (PRD "Sizing chain").

`Book` is the plain state a persona holds. `rebalance` is the pure function
that enforces the sector caps (F5, F7) and refills freed weight into
underweight buckets (F6, Q9); it is used both by the automatic sizing
chain (step 3, called with the sized target weights) and by the Rebalance
button (called with the book's own current weights, S8/user story 30) — the
caller decides what "target weights" means, this function only enforces
the caps and diversifies whatever it is handed. `size_persona` runs the
full chain — blend, Fear & Greed and fundamentals tilts, PM drops, per-name max, then
`rebalance` — and is called by the committee's size node (preview) and
again by `execute` against a fresh snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import config
from external.broker_models import Order, Position
from external.construction import blend_signals
from external.execution import build_orders
from external.limits import RiskLimits, apply_limits
from external.models import Signal
from feeds import Fundamentals

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


def compute_book_equity(book: Book, marks: dict[str, float]) -> float:
    """Total book value: cash plus every position marked at its current price."""
    positions_value = sum(
        shares * marks[ticker] for ticker, shares in book.shares.items()
    )
    return book.cash + positions_value


def tickers_outside_universe(book: Book) -> list[str]:
    """Return the held tickers that are neither a universe stock nor a sector ETF, sorted.

    A universe change leaves them in a book until Reset books re-seeds it;
    they have no price mark or sector, so nothing can value or size them.
    """
    return sorted(
        ticker for ticker in book.shares if ticker not in config.SECTOR_BY_TICKER
    )


def weights_by_ticker(book: Book, marks: dict[str, float]) -> dict[str, float]:
    """Return each held ticker's weight, as a fraction of the book's equity."""
    equity = compute_book_equity(book, marks)
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
    """Return `target_weights` plus, at weight 0.0, any bucket ETF the book holds but the targets left out."""
    bucket_etfs = set(config.ETF_BY_SECTOR.values())
    carried = dict(target_weights)

    # `build_orders` closes any name missing from the targets, so a held ETF
    # needs an entry; the Q9 refill then re-derives its real weight. Carrying
    # its current weight instead would stack a fresh refill on top every
    # cycle and push gross past invested % (sizing chain steps 3-4).
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
    """Return `weights` with `refill_pool` spent on the most-underweight buckets' ETFs (F6, Q9).

    Each bucket below the equal-weight target has its ETF topped up to the
    lesser of that target and the cap; whatever the pool can't cover stays in cash.
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

    `orders` already reflects any sub-$200 order the floor dropped, so
    this is the book that actually results from a cycle, not the target
    left on paper before the floor and the cap/limit restores ran.
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


def _is_large_enough(order: Order) -> bool:
    """Return whether `order` clears the $200 trade floor (F11)."""
    return order.price * order.quantity >= config.MIN_TRADE_USD


def _cash_after_orders(book: Book, orders: list[Order]) -> float:
    """Return the book's cash once every order in `orders` has filled."""
    cash = book.cash
    for order in orders:
        cash += order.price * order.quantity * (1 if order.side == "sell" else -1)
    return cash


def _breaks_a_limit(
    book: Book,
    orders: list[Order],
    marks: dict[str, float],
    equity: float,
    caps_persona: config.Persona,
    invested_pct: float,
) -> bool:
    """Return whether the book that `orders` produce breaks a hard limit (F7, F10, F11).

    Checked against `caps_persona`: the per-name maximum on every stock
    (ETFs are exempt, Q9), every sector cap, and cash against
    `invested_pct`'s buffer.
    """
    weights_after = _weights_after_orders(book, orders, marks, equity)
    for ticker in config.UNIVERSE:
        if weights_after.get(ticker, 0.0) > caps_persona.max_position_pct:
            return True

    for sector_weight in _sector_totals(weights_after).values():
        if sector_weight > caps_persona.sector_cap:
            return True

    cash_floor = equity * (1 - invested_pct)
    return _cash_after_orders(book, orders) < cash_floor


def build_orders_for_targets(
    book: Book,
    target_weights: dict[str, float],
    marks: dict[str, float],
    caps_persona: config.Persona,
    invested_pct: float,
) -> list[Order]:
    """Return whole-share orders that move `book` to `target_weights`, minus trades under $200 (sizing chain steps 4-5).

    Shared by `rebalance` (preview) and `execute` (fresh re-derivation), so
    both size orders the same way.
    """
    equity = compute_book_equity(book, marks)

    # Absorbs float round-trip error (e.g. weight*equity/mark landing a hair
    # under a held share count) before build_orders floors shares, so a
    # target equal to the current weight never emits a spurious 1-share sell.
    orders = build_orders(
        target_weights,
        _positions(book),
        marks,
        equity * (1 + SHARE_ROUNDING_TOLERANCE),
    )

    # A sub-$200 sell stays when its sector is still over the cap once every
    # kept order fills: the hard cap (F7) outranks the floor (F11). Judged on
    # the post-trade book, so a same-sector buy can't push a capped sector
    # over while its own dropped sell sits on paper (build deviation #7).
    large_enough_orders = [order for order in orders if _is_large_enough(order)]
    post_sector_weights = _sector_totals(
        _weights_after_orders(book, large_enough_orders, marks, equity)
    )
    over_cap_sectors = {
        sector
        for sector, weight in post_sector_weights.items()
        if weight > caps_persona.sector_cap
    }

    floored_orders = []
    for order in orders:
        restores_cap = (
            order.side == "sell"
            and config.SECTOR_BY_TICKER.get(order.ticker) in over_cap_sectors
        )
        if _is_large_enough(order) or restores_cap:
            floored_orders.append(order)

    # Still breaking a per-name max, a cap or the cash buffer: restore every
    # dropped sell. A sell only moves the book toward targets that already
    # meet every limit, so it can't create a new breach (build deviation #14).
    if _breaks_a_limit(book, floored_orders, marks, equity, caps_persona, invested_pct):
        return [
            order for order in orders if order.side == "sell" or _is_large_enough(order)
        ]

    return floored_orders


def rebalance(
    book: Book,
    persona: config.Persona,
    target_weights: dict[str, float],
    marks: dict[str, float],
    invested_pct: float,
    caps_persona: config.Persona | None = None,
) -> RebalanceResult:
    """Return the capped targets, their whole-share orders and before/after sector weights (F5-F8, Q9, S8).

    `target_weights` is the sized proposal (cycle) or the book's own weights
    (Rebalance button); `caps_persona` checks against another persona's caps (R3).
    """
    caps_persona = caps_persona or persona
    before_sector_weights = sector_weights(book, marks)

    working_weights = _carry_forward_held_etfs(target_weights, book)
    equal_weight_target = invested_pct / len(config.SECTORS)

    # Q9: the pool is whatever of the regime's invested % the trims and
    # per-name clamps left unallocated, so gross never grows past it.
    trimmed_weights = _trim_over_cap_sectors(working_weights, caps_persona.sector_cap)
    refill_pool = max(0.0, invested_pct - sum(trimmed_weights.values()))
    final_weights = _refill_underweight_etfs(
        trimmed_weights,
        refill_pool,
        equal_weight_target,
        caps_persona.sector_cap,
    )

    orders = build_orders_for_targets(
        book, final_weights, marks, caps_persona, invested_pct
    )
    equity = compute_book_equity(book, marks)

    return RebalanceResult(
        target_weights=final_weights,
        orders=orders,
        before_sector_weights=before_sector_weights,
        after_sector_weights=_sector_totals(
            _weights_after_orders(book, orders, marks, equity)
        ),
    )


def get_fear_greed_tilt(fear_greed: float | None) -> float:
    """Return Q6's market-level conviction tilt: +0.10 below 25, -0.10 above 75, else 0."""
    if fear_greed is None:
        return 0.0  # feed unavailable -> tilt skipped ($0 Cost Guarantee)

    if fear_greed < config.FEAR_GREED_LOW:
        return config.FEAR_GREED_TILT

    if fear_greed > config.FEAR_GREED_HIGH:
        return -config.FEAR_GREED_TILT

    return 0.0


def _forward_pe_tilt(forward_pe: float | None, sector: str) -> float:
    """Return the forward P/E part of the fundamentals tilt: -0.05 above the sector's limit, -0.10 above 1.5x it."""
    # No number, or no forward earnings to price the stock on: no signal.
    if forward_pe is None or forward_pe <= 0:
        return 0.0

    limit = config.FORWARD_PE_LIMIT_BY_SECTOR[sector]
    if forward_pe > limit * config.FORWARD_PE_STRETCHED_MULTIPLE:
        return -config.FORWARD_PE_STRETCHED_TILT

    if forward_pe > limit:
        return -config.FORWARD_PE_TILT

    return 0.0


def _debt_to_equity_tilt(debt_to_equity: float | None, sector: str) -> float:
    """Return the debt part of the fundamentals tilt: -0.05 above the debt-to-equity limit."""
    # Bank leverage is structural, not a warning sign.
    if sector == config.DEBT_TO_EQUITY_SKIPPED_SECTOR:
        return 0.0

    if debt_to_equity is None or debt_to_equity <= 0:
        return 0.0

    if debt_to_equity > config.DEBT_TO_EQUITY_LIMIT:
        return -config.DEBT_TO_EQUITY_TILT

    return 0.0


def get_fundamentals_tilt(ticker: str, fundamentals: dict[str, Fundamentals]) -> float:
    """Return one stock's fundamentals tilt: 0 down to -0.15, for a forward P/E stretched for its sector and high debt.

    Missing data skips that part, like the Fear & Greed tilt when its feed is down.
    """
    ticker_fundamentals = fundamentals.get(ticker)
    if ticker_fundamentals is None:
        return 0.0

    sector = config.SECTOR_BY_TICKER[ticker]
    forward_pe_tilt = _forward_pe_tilt(ticker_fundamentals.forward_pe, sector)
    debt_tilt = _debt_to_equity_tilt(ticker_fundamentals.debt_to_equity, sector)
    return forward_pe_tilt + debt_tilt


def _tilted(signal: Signal, tilt: float) -> Signal:
    """Shift one analyst signal by the tilt, kept inside blend_signals' [-1, 1] range."""
    tilted_value = max(-1.0, min(1.0, signal.value + tilt))
    return signal.model_copy(update={"value": tilted_value})


def _tilted_signals(
    signals: list[Signal],
    fear_greed: float | None,
    fundamentals: dict[str, Fundamentals],
) -> list[Signal]:
    """Shift every analyst signal by the market's Fear & Greed tilt plus its own stock's fundamentals tilt."""
    market_tilt = get_fear_greed_tilt(fear_greed)

    tilted_signals = []
    for signal in signals:
        stock_tilt = get_fundamentals_tilt(signal.ticker, fundamentals)
        tilted_signals.append(_tilted(signal, market_tilt + stock_tilt))

    return tilted_signals


def blend_convictions(
    signals: list[Signal],
    fear_greed: float | None,
    fundamentals: dict[str, Fundamentals],
) -> dict[str, float]:
    """Return F12's conviction per ticker (0.6 x Technical + 0.4 x Sentiment), tilted by Q6 and fundamentals, for the PM's prompt.

    This is the same number `size_persona` sizes on, so the PM reasons over
    the number that is actually traded (Q6, F12).
    """
    tilted_signals = _tilted_signals(signals, fear_greed, fundamentals)
    blend = blend_signals(
        tilted_signals,
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
    fundamentals: dict[str, Fundamentals],
    dropped_tickers: set[str],
) -> RebalanceResult:
    """Run the PRD sizing chain for one persona and return its targets and orders.

    Blend (F12, gross = the regime's invested %), F&G and fundamentals
    tilts (Q6), PM drops (R1), per-name max (F10), then the sector-cap
    layer, whole-share orders and the $200 floor inside `rebalance`
    (F5-F8, F11, Q9).
    """
    invested_pct = persona.invested_pct_for_regime(regime)

    # Model weights sum to 1, so shifting every analyst's value by a stock's
    # tilt shifts its blended conviction by the same amount (Q6).
    tilted_signals = _tilted_signals(signals, fear_greed, fundamentals)
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
