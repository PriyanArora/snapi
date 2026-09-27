"""Approve -> paper orders: `execute(proposal, persona)`.

`execute` is the only path that trades (story 22, T18). It re-reads the
persona's book, re-derives whole-share orders from the approved target
weights through the same `build_orders_for_targets` the preview used, and
routes by account: Balanced to Alpaca Paper through SnapTrade (queued until
the next open, locked while any order is still open, R4), Conservative and
Aggressive to their SimBroker at the snapshot mark. A rerun on the same
targets finds the book already there and places nothing (seam 2).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime

import books
import brokerage
import prices
from books import ApprovalRecord, FillTag
from committee import Proposal
from config import PERSONAS
from external.broker_models import Fill, Order
from portfolio import Book, build_orders_for_targets


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
