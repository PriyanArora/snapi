"""Approve -> paper orders: `execute(proposal, persona)` and the S16 seed.

`execute` is the only path that trades (story 22, T18). It re-reads the
persona's book, re-derives whole-share orders from the approved target
weights through the same `build_orders_for_targets` the preview used, and
routes by account: Balanced to Alpaca Paper through SnapTrade (queued until
the next open, locked while any order is still open, R4, and paced under
SnapTrade's 10 requests a minute, Risk #8), Conservative and Aggressive to
their SimBroker at the snapshot mark. A rerun on the same targets finds the
book already there and places nothing (seam 2).

The seed lives here, not in books.py, because it calls `execute`.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from datetime import UTC, date, datetime

import books
import brokerage
import prices
from books import SEED_PERSONAS, ApprovalRecord, FillTag
from committee import Proposal
from config import PERSONAS, UNIVERSE, Regime
from external.broker_models import Fill, Order
from external.models import Signal
from portfolio import MODEL_WEIGHTS, Book, build_orders_for_targets, size_persona

GROSS_WEIGHT_TOLERANCE = 1e-9  # float round-off in the sizing chain's sums

# Called with (order_number, order_count) before each Balanced order waits its turn.
ProgressCallback = Callable[[int, int], None]

# One Balanced batch at a time. A batch takes minutes, Streamlit starts a new
# run on any click meanwhile, and a second Approve's lock check would pass
# until the first order is out (R4, Risk #17).
_alpaca_batch_lock = threading.Lock()


class ExecutionLockedError(Exception):
    """Raised when Balanced still has orders open at the broker (R4)."""


def _orders_for_book(
    book: Book,
    persona_key: str,
    target_weights: dict[str, float],
    marks: dict[str, float],
) -> list[Order]:
    """Return the orders that move `book` to the approved targets, under the persona's own limits."""
    persona = PERSONAS[persona_key]

    # The regime's invested % is already inside the approved targets; the
    # hard cash limit left to hold here is the persona's own buffer (F11).
    return build_orders_for_targets(
        book,
        target_weights,
        marks,
        persona,
        persona.max_invested_pct,
    )


def _log_approval(
    persona_key: str,
    tag: FillTag,
    orders: list[Order],
    acks: list[dict],
    fills: list[Fill],
) -> None:
    """Append one executed batch to the run log (Balanced step 5: the acks are its record)."""
    approval = ApprovalRecord(
        approved_at=datetime.now(UTC),
        persona=persona_key,
        tag=tag,
        orders=orders,
        acks=acks,
        fills=fills,
    )
    books.append_run_log(approval)


def _report_progress(
    report_progress: ProgressCallback | None,
    order_number: int,
    order_count: int,
) -> BaseException | None:
    """Call the progress callback and return what it raised instead of raising it.

    Streamlit stops a page on any click by raising inside its next call. A
    half-placed batch stays locked until it fills (R4), so the caller keeps
    placing and raises this once every order is out.
    """
    if report_progress is None:
        return None

    try:
        report_progress(order_number, order_count)
    except BaseException as progress_error:
        return progress_error

    return None


def _place_on_alpaca_paper(
    account_id: str,
    persona_key: str,
    orders: list[Order],
    tag: FillTag,
    report_progress: ProgressCallback | None,
) -> None:
    """Place each order once SnapTrade's rate budget allows and log every ack, even when the batch fails midway."""
    acks = []
    progress_error = None
    try:
        for order_number, order in enumerate(orders, start=1):
            if progress_error is None:
                progress_error = _report_progress(
                    report_progress, order_number, len(orders)
                )

            # The first order waits too: the reads just made share its minute (Risk #8).
            time.sleep(brokerage.ORDER_SPACING_SECONDS)
            time.sleep(brokerage.seconds_until_order_room(account_id))
            acks.append(brokerage.place_order(account_id, order))
    finally:
        # The run log, the lock and the order status must see every order
        # that did go out (Balanced steps 5-7).
        brokerage.clear_order_cache(account_id)
        _log_approval(persona_key, tag, orders, acks, fills=[])

    if progress_error is not None:
        raise progress_error


def _execute_on_alpaca_paper(
    persona_key: str,
    target_weights: dict[str, float],
    marks: dict[str, float],
    tag: FillTag,
    report_progress: ProgressCallback | None,
) -> list[Fill]:
    """Send Balanced's orders through SnapTrade; nothing fills before the next open, so no fills return."""
    if not _alpaca_batch_lock.acquire(blocking=False):
        raise ExecutionLockedError(
            "Balanced's orders from an earlier Approve are still going out; "
            "wait for that batch to finish."
        )

    try:
        account_id = brokerage.alpaca_paper_account_id()

        # Execution re-reads the book: a read cached from the preview could
        # miss an order or a fill.
        brokerage.clear_cache(account_id)

        # Queued orders aren't positions yet, so a book read alone would
        # double-order against them (R4, Risk #17).
        if brokerage.has_unfilled_orders(account_id):
            raise ExecutionLockedError(
                "Balanced has orders waiting for the next open; "
                "Approve unlocks once they fill."
            )

        orders = _orders_for_book(
            books.alpaca_paper_book(), persona_key, target_weights, marks
        )
        if orders:
            _place_on_alpaca_paper(
                account_id, persona_key, orders, tag, report_progress
            )
    finally:
        _alpaca_batch_lock.release()

    return []


def _execute_on_simulator(
    persona_key: str,
    target_weights: dict[str, float],
    marks: dict[str, float],
    tag: FillTag,
) -> list[Fill]:
    """Fill a simulator persona's orders at the snapshot mark and append them to its fill log."""
    orders = _orders_for_book(
        books.load_book(persona_key), persona_key, target_weights, marks
    )
    if not orders:
        return []

    broker = books.open_sim_broker(persona_key)
    fills = []
    for order in orders:
        fills.append(broker.place_order(order))

    books.append_fills(persona_key, fills, tag)
    _log_approval(persona_key, tag, orders, acks=[], fills=fills)
    return fills


def execute(
    proposal: Proposal,
    persona_key: str,
    tag: FillTag = "committee",
    report_progress: ProgressCallback | None = None,
) -> list[Fill]:
    """Execute one persona's approved targets against a fresh read of its book and return the fills.

    Balanced returns no fills (its orders queue at Alpaca Paper) and raises
    ExecutionLockedError while earlier orders are still open. `tag` marks
    where the batch came from: Approve, the Rebalance button or the seed.
    `report_progress` hears about each spaced-out Balanced order; simulator
    fills are instant.
    """
    persona = PERSONAS[persona_key]
    target_weights = proposal.target_weights[persona_key]
    marks = prices.latest_marks()

    # Approve executes a proposal after a JSON round trip through the
    # session. Targets above the persona's max invested % could only come
    # from a bug upstream, and would spend cash the book doesn't have.
    gross_weight = sum(target_weights.values())
    if gross_weight > persona.max_invested_pct + GROSS_WEIGHT_TOLERANCE:
        raise ValueError(
            f"{persona_key} targets sum to {gross_weight:.4f}, above its "
            f"max invested {persona.max_invested_pct:.4f}; refusing to trade"
        )

    if persona.account == "alpaca_paper":
        return _execute_on_alpaca_paper(
            persona_key, target_weights, marks, tag, report_progress
        )

    return _execute_on_simulator(persona_key, target_weights, marks, tag)


def _equal_conviction_signals() -> list[Signal]:
    """Return a full-conviction vote from both analysts on every name (Q14 default a)."""
    as_of = date.today().isoformat()
    equal_signals = []
    for model_name in MODEL_WEIGHTS:
        for ticker in UNIVERSE:
            equal_signals.append(
                Signal(model_name=model_name, ticker=ticker, date=as_of, value=1.0)
            )

    return equal_signals


def _seed_proposal(persona_key: str, regime: Regime) -> Proposal:
    """Return the equal-conviction opening proposal for the persona's empty book (S16).

    Equal conviction through the same sizing chain and cap layer as an
    approved run; no LLM call and no F&G tilt.
    """
    persona = PERSONAS[persona_key]
    empty_book = Book(cash=persona.starting_cash, shares={}, avg_cost={})
    sized = size_persona(
        persona,
        _equal_conviction_signals(),
        regime,
        empty_book,
        prices.latest_marks(),
        fear_greed=None,
        dropped_tickers=set(),
    )
    return Proposal(
        target_weights={persona_key: sized.target_weights},
        orders={persona_key: sized.orders},
    )


def _seed_if_empty(persona_key: str) -> list[Fill]:
    """Give one simulator persona its opening book at Friday's close, once, through `execute` (S16)."""
    # A restart must never double-seed (story 58).
    if books.has_fills(persona_key):
        return []

    proposal = _seed_proposal(persona_key, books.current_regime())
    return execute(proposal, persona_key, tag="seed")


def seed_books() -> None:
    """Seed Conservative and Aggressive when their fill logs are empty; Balanced is never seeded (S16)."""
    for persona_key in SEED_PERSONAS:
        _seed_if_empty(persona_key)


def reset_books() -> None:
    """Clear the simulated fill logs and seed them again, for a clean rehearsal (S16 Reset books)."""
    # Size the new opening books before deleting anything: if FRED is down,
    # the regime read raises here and the old books stay in place.
    regime = books.current_regime()
    seed_proposals = {}
    for persona_key in SEED_PERSONAS:
        seed_proposals[persona_key] = _seed_proposal(persona_key, regime)

    for persona_key, persona in PERSONAS.items():
        # Balanced only has a fill log under R5's simulator fallback.
        if persona.account == "simulator":
            books.clear_fills(persona_key)

    for persona_key, proposal in seed_proposals.items():
        execute(proposal, persona_key, tag="seed")
