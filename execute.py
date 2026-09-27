"""Approve -> paper orders: `execute(proposal, persona)` and the S16 seed.

`execute` is the only path that trades (story 22, T18). It re-reads the
persona's book, re-derives whole-share orders from the approved target
weights through the same `build_orders_for_targets` the preview used, and
routes by account: Balanced to Alpaca Paper through SnapTrade (queued until
the next open, locked while any order is still open, R4), Conservative and
Aggressive to their SimBroker at the snapshot mark. A rerun on the same
targets finds the book already there and places nothing (seam 2).

The seed lives here, not in books.py, because it calls `execute`.
"""

from __future__ import annotations

import time
from datetime import UTC, date, datetime

import books
import brokerage
import prices
from books import SEED_PERSONAS, ApprovalRecord, FillTag
from committee import Proposal
from config import PERSONAS, UNIVERSE
from external.broker_models import Fill, Order
from external.models import Signal
from portfolio import MODEL_WEIGHTS, Book, build_orders_for_targets, size_persona


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


def _execute_on_alpaca_paper(
    persona_key: str,
    target_weights: dict[str, float],
    marks: dict[str, float],
    tag: FillTag,
) -> list[Fill]:
    """Send Balanced's orders through SnapTrade and log the acks; nothing fills before the next open."""
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
    if not orders:
        return []

    acks = []
    try:
        for index, order in enumerate(orders):
            if index > 0:
                time.sleep(brokerage.ORDER_SPACING_SECONDS)
            acks.append(brokerage.place_order(account_id, order))
    finally:
        # Even after a mid-batch failure, the run log and the next lock
        # check must see every order that did go out (Balanced steps 5, 7).
        brokerage.clear_cache(account_id)
        approval = ApprovalRecord(
            approved_at=datetime.now(UTC),
            persona=persona_key,
            tag=tag,
            orders=orders,
            acks=acks,
            fills=[],
        )
        books.append_run_log(approval)

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
    approval = ApprovalRecord(
        approved_at=datetime.now(UTC),
        persona=persona_key,
        tag=tag,
        orders=orders,
        acks=[],
        fills=fills,
    )
    books.append_run_log(approval)
    return fills


def execute(
    proposal: Proposal,
    persona_key: str,
    tag: FillTag = "committee",
) -> list[Fill]:
    """Execute one persona's approved targets against a fresh read of its book and return the fills.

    Balanced returns no fills (its orders queue at Alpaca Paper) and raises
    ExecutionLockedError while earlier orders are still open. `tag` marks
    where the batch came from: Approve, the Rebalance button or the seed.
    """
    persona = PERSONAS[persona_key]
    target_weights = proposal.target_weights[persona_key]
    marks = prices.latest_marks()

    if persona.account == "alpaca_paper":
        return _execute_on_alpaca_paper(persona_key, target_weights, marks, tag)

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


def _seed_if_empty(persona_key: str) -> list[Fill]:
    """Give one simulator persona its opening book at Friday's close, once (S16).

    Equal conviction through the same sizing chain, cap layer and
    `execute` as an approved run; no LLM call and no F&G tilt.
    """
    # A restart must never double-seed (story 58).
    if books.has_fills(persona_key):
        return []

    sized = size_persona(
        PERSONAS[persona_key],
        _equal_conviction_signals(),
        books.current_regime(),
        books.load_book(persona_key),
        prices.latest_marks(),
        fear_greed=None,
        dropped_tickers=set(),
    )
    proposal = Proposal(
        target_weights={persona_key: sized.target_weights},
        orders={persona_key: sized.orders},
    )
    return execute(proposal, persona_key, tag="seed")


def seed_books() -> None:
    """Seed Conservative and Aggressive when their fill logs are empty; Balanced is never seeded (S16)."""
    for persona_key in SEED_PERSONAS:
        _seed_if_empty(persona_key)


def reset_books() -> None:
    """Clear the simulated fill logs and seed them again, for a clean rehearsal (S16 Reset books)."""
    for persona_key, persona in PERSONAS.items():
        # Balanced only has a fill log under R5's simulator fallback.
        if persona.account == "simulator":
            books.fills_path(persona_key).unlink(missing_ok=True)

    seed_books()
