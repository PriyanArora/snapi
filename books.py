"""Persona books, the per-cycle snapshot and the run log (PRD Execution, S14, S16, R9).

Conservative and Aggressive live in a fill log each (`data/fills_<persona>.jsonl`),
rebuilt at start-up by replaying the fills into a fresh SimBroker.
Balanced's book comes from its Alpaca Paper account through the cached
SnapTrade read. `take_snapshot` gathers one cycle's frozen inputs and
`log_run` records NAV per persona on every Run swarm (F23), so the forward
test builds up with no scheduler.

When MONGODB_URI is set, every log is instead a MongoDB collection named
after its file (`fills_conservative`, `run_log`, `sonnet_runs`, ...),
because Streamlit Cloud's disk is wiped on sleep and redeploy (Hosting).
Unset, the JSONL files stay, so the tests run offline.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

import brokerage
import feeds
import prices
import signals
from committee import Proposal, Snapshot
from config import (
    BENCHMARK,
    DATA_DIR,
    PERSONAS,
    RUN_LOG_PATH,
    SONNET_RUNS_PATH,
    UNIVERSE,
    Regime,
)
from external.broker_models import Fill, Order
from external.sim import SimBroker
from portfolio import Book, compute_book_equity, tickers_outside_universe

FillTag = Literal["committee", "rebalance", "seed"]

# S16: the two simulator personas that get an opening book; Balanced never does.
SEED_PERSONAS = ("conservative", "aggressive")

MONGODB_URI_ENV_VAR = "MONGODB_URI"
# Atlas's SRV connection strings name no database, so the logs go here.
DEFAULT_MONGODB_DATABASE = "snapi"


class FillRecord(BaseModel):
    """One simulator fill as stored in a persona's fill log; the file name carries the persona."""

    ticker: str
    side: Literal["buy", "sell"]
    quantity: int
    price: float
    filled_at: datetime
    tag: FillTag


class RunRecord(BaseModel):
    """One Run swarm: the proposal plus each persona's NAV and SPY's close (F23)."""

    kind: Literal["run"] = "run"
    taken_at: datetime
    proposal: Proposal
    nav: dict[str, float]
    spy_close: float


class SavedRunRecord(RunRecord):
    """A finished Sonnet run kept as the UI's fallback, stamped with when it was saved."""

    saved_at: datetime


class ApprovalRecord(BaseModel):
    """One executed batch for one persona: SnapTrade acks for Balanced, fills for the simulators."""

    kind: Literal["approval"] = "approval"
    approved_at: datetime
    persona: str
    tag: FillTag
    orders: list[Order]
    acks: list[dict]
    fills: list[Fill]


@functools.cache
def _mongodb_database(mongodb_uri: str) -> Database:
    """Return the URI's own database, else "snapi", on one MongoClient kept for the process."""
    return MongoClient(mongodb_uri).get_default_database(DEFAULT_MONGODB_DATABASE)


def _log_collection(path: Path) -> Collection | None:
    """Return the MongoDB collection named after a log file when MONGODB_URI is set, else None (R9)."""
    mongodb_uri = os.environ.get(MONGODB_URI_ENV_VAR)
    if not mongodb_uri:
        return None

    return _mongodb_database(mongodb_uri)[path.stem]


def _read_log(path: Path) -> list[dict]:
    """Return a log's records in the order they were appended, from MongoDB or the JSONL file."""
    collection = _log_collection(path)
    if collection is not None:
        # One client's ObjectIds grow with every insert, so _id order is
        # append order; SimBroker replay depends on it.
        # ponytail: two processes writing one log in the same second could
        # interleave; add a counter document if the laptop and Cloud ever do.
        return list(collection.find({}, {"_id": False}).sort("_id"))

    if not path.exists():
        return []

    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    return records


def _append_log(path: Path, records: Sequence[BaseModel]) -> None:
    """Append records, in order, to a log in MongoDB or its JSONL file."""
    # insert_many refuses an empty list.
    if not records:
        return

    collection = _log_collection(path)
    if collection is not None:
        documents = [record.model_dump(mode="json") for record in records]
        collection.insert_many(documents)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as log_file:
        for record in records:
            log_file.write(record.model_dump_json() + "\n")


def fills_path(persona_key: str) -> Path:
    """Return the persona's fill-log file, its whole simulated history (R9)."""
    return DATA_DIR / f"fills_{persona_key}.jsonl"


def _read_fills(persona_key: str) -> list[FillRecord]:
    """Return the persona's recorded fills in order; a corrupt record raises rather than replaying a wrong book."""
    records = []
    for document in _read_log(fills_path(persona_key)):
        records.append(FillRecord.model_validate(document))

    return records


def clear_fills(persona_key: str) -> None:
    """Delete the persona's whole fill log, from MongoDB or the disk (S16 Reset books)."""
    path = fills_path(persona_key)
    collection = _log_collection(path)
    if collection is not None:
        collection.drop()
        return

    path.unlink(missing_ok=True)


def has_fills(persona_key: str) -> bool:
    """Return True once the persona has any recorded fill (the seed guard, story 58)."""
    return bool(_read_fills(persona_key))


def _replay(records: list[FillRecord], starting_cash: float) -> SimBroker:
    """Return a fresh SimBroker with every recorded fill placed again, in order."""
    broker = SimBroker(cash=starting_cash)
    for record in records:
        broker.place_order(
            Order(
                ticker=record.ticker,
                side=record.side,
                quantity=record.quantity,
                price=record.price,
            )
        )

    return broker


def open_sim_broker(persona_key: str) -> SimBroker:
    """Return the persona's simulator, rebuilt from its fill log (story 24)."""
    return _replay(_read_fills(persona_key), PERSONAS[persona_key].starting_cash)


def _average_costs(records: list[FillRecord]) -> dict[str, float]:
    """Return each open position's average cost from its fills (Q10: the critic's cost basis).

    A buy averages in; a sell leaves the average alone; a position sold
    back to zero is forgotten, so a later re-buy starts a fresh average.
    """
    held_shares: dict[str, int] = {}
    avg_cost: dict[str, float] = {}
    for record in records:
        held = held_shares.get(record.ticker, 0)
        if record.side == "buy":
            cost_held = avg_cost.get(record.ticker, 0.0) * held
            cost_bought = record.price * record.quantity
            held_shares[record.ticker] = held + record.quantity
            avg_cost[record.ticker] = (cost_held + cost_bought) / (
                held + record.quantity
            )
            continue

        held_shares[record.ticker] = held - record.quantity
        if held_shares[record.ticker] <= 0:
            del held_shares[record.ticker]
            avg_cost.pop(record.ticker, None)

    return avg_cost


def load_book(persona_key: str) -> Book:
    """Return a simulator persona's book: cash and shares by replay, average cost from the fills."""
    records = _read_fills(persona_key)
    broker = _replay(records, PERSONAS[persona_key].starting_cash)

    shares = {}
    for ticker, position in broker.positions().items():
        shares[ticker] = position.shares

    return Book(cash=broker.cash(), shares=shares, avg_cost=_average_costs(records))


def alpaca_paper_book() -> Book:
    """Return Balanced's book from its cached Alpaca Paper read; cost basis from SnapTrade (Q10)."""
    account = brokerage.get_account_snapshot(brokerage.alpaca_paper_account_id())

    shares = {}
    avg_cost = {}
    for holding in account.holdings:
        # A name we hold no price for could be neither marked nor sized.
        if holding.ticker not in prices.ALL_TICKERS:
            raise ValueError(
                f"Alpaca Paper holds {holding.ticker!r}, which is outside the "
                "priced universe; close it in the paper account by hand"
            )

        whole_shares = int(holding.units)
        if whole_shares <= 0:
            continue

        shares[holding.ticker] = whole_shares
        if holding.cost_basis is not None:
            avg_cost[holding.ticker] = holding.cost_basis

    return Book(cash=account.cash, shares=shares, avg_cost=avg_cost)


def current_book(persona_key: str) -> Book:
    """Return the persona's book as it stands now, routed by the persona's account (F4)."""
    persona = PERSONAS[persona_key]
    if persona.account == "alpaca_paper" and brokerage.is_alpaca_paper_configured():
        return alpaca_paper_book()

    # Simulator personas, R5's fallback, and Balanced before its SnapTrade
    # keys exist (an untouched paper account: the starting cash, no shares).
    return load_book(persona_key)


def append_fills(persona_key: str, fills: list[Fill], tag: FillTag) -> None:
    """Append one batch of simulator fills to the persona's fill log."""
    filled_at = datetime.now(UTC)
    records = []
    for fill in fills:
        records.append(
            FillRecord(
                ticker=fill.ticker,
                side=fill.side,
                quantity=fill.quantity,
                price=fill.price,
                filled_at=filled_at,
                tag=tag,
            )
        )

    _append_log(fills_path(persona_key), records)


def current_regime() -> Regime:
    """Return today's macro regime from the five-flag checklist (F13)."""
    return signals.compute_macro_checklist(feeds.fetch_macro_series()).regime


def take_snapshot() -> Snapshot:
    """Gather one cycle's frozen inputs, shared by the agents and the dashboard (story 28).

    Balanced's book costs one cached SnapTrade read; the Sandbox's demo
    holdings are never mixed in (S16).
    """
    books_by_persona = {}
    for persona_key, persona in PERSONAS.items():
        book = current_book(persona_key)

        # Names with no mark or sector would crash sizing, and leaving them
        # out would log a false NAV into the forward test.
        stale_tickers = tickers_outside_universe(book)
        if stale_tickers:
            raise ValueError(
                f"{persona.display_name}'s book still holds "
                f"{', '.join(stale_tickers)} from before the stock universe "
                "changed. Press Reset books on the Portfolio page to re-seed "
                "the books."
            )

        books_by_persona[persona_key] = book

    technical_scores = {}
    for ticker in UNIVERSE:
        technical_scores[ticker] = signals.compute_technical_score(ticker)

    return Snapshot(
        taken_at=datetime.now(UTC),
        marks=prices.latest_marks(),
        technical=technical_scores,
        news=feeds.fetch_universe_news(),
        fear_greed=feeds.fetch_fear_and_greed(),
        macro=signals.compute_macro_checklist(feeds.fetch_macro_series()),
        books=books_by_persona,
        fundamentals=feeds.fetch_fundamentals(),
    )


def append_run_log(record: RunRecord | ApprovalRecord) -> None:
    """Append one record to the run log that the forward test reads."""
    _append_log(RUN_LOG_PATH, [record])


def read_run_log() -> list[dict]:
    """Return every run and approval record as plain JSON dicts, oldest first, for the dashboard."""
    return _read_log(RUN_LOG_PATH)


def log_run(snapshot: Snapshot, proposal: Proposal) -> RunRecord:
    """Record one Run swarm with each persona's NAV and SPY's close, then return it (F23, S14, story 44)."""
    nav = {}
    for persona_key, book in snapshot.books.items():
        nav[persona_key] = compute_book_equity(book, snapshot.marks)

    record = RunRecord(
        taken_at=snapshot.taken_at,
        proposal=proposal,
        nav=nav,
        spy_close=snapshot.marks[BENCHMARK],
    )
    append_run_log(record)
    return record


def save_sonnet_run(record: dict) -> None:
    """Keep a finished run record (a dumped RunRecord) as the fallback for a failed or slow live run.

    Validated first, so a record that wouldn't load back never gets saved.
    """
    saved_run = SavedRunRecord.model_validate(record | {"saved_at": datetime.now(UTC)})
    _append_log(SONNET_RUNS_PATH, [saved_run])


def latest_sonnet_run() -> dict | None:
    """Return the newest saved Sonnet run as a plain JSON dict, or None before one is saved."""
    collection = _log_collection(SONNET_RUNS_PATH)
    if collection is not None:
        # ObjectIds grow with every insert, so the highest _id is the newest.
        return collection.find_one({}, {"_id": False}, sort=[("_id", -1)])

    saved_runs = _read_log(SONNET_RUNS_PATH)
    if not saved_runs:
        return None

    return saved_runs[-1]
