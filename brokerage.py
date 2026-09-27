"""SnapTrade v13 calls as plain functions (PRD Execution > Balanced; no Broker adapter class).

Every account read goes through a 60-second in-process cache, so one cycle
costs one read per account, shared by the agents and the dashboard (T19,
story 28), and every request is noted against the account's 10-a-minute
budget (Risk #8). The one write is `place_order`, used only by `execute`
on Balanced's Approve. Run `python brokerage.py` once for the
go/no-go check and to find the two account ids for `.env`.
"""

from __future__ import annotations

import functools
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from dotenv import load_dotenv
from snaptrade_client import SnapTrade, SnapTradeAuth

from external.broker_models import Order

CONSUMER_KEY_ENV_VAR = "SNAPTRADE_CONSUMER_KEY"
CLIENT_ID_ENV_VAR = "SNAPTRADE_CLIENT_ID"
ALPACA_PAPER_ACCOUNT_ENV_VAR = "SNAPTRADE_ALPACA_PAPER_ACCOUNT_ID"
SANDBOX_ACCOUNT_ENV_VAR = "SNAPTRADE_SANDBOX_ACCOUNT_ID"

CACHE_SECONDS = 60  # T19: at least 60 s between reads of one account
ALPACA_PAPER_SLUG = "ALPACA-PAPER"

# Risk #8: treat SnapTrade's 10 requests/min per account as enforced.
ACCOUNT_REQUESTS_PER_MINUTE = 10
RATE_WINDOW_SECONDS = 60
# After a batch the lock caption and the order status re-read the two order lists.
POST_BATCH_READS = 2
# An unlocked Approve costs 4 + n + 2 requests on the account: 4 fresh reads
# (open orders, last-24h orders, positions, balance), the n orders, then the
# 2 order reads. Each order waits 12 s, the first one too, so the first
# minute holds 4 reads + 4 orders (12-48 s) + the 2 reads of a batch that ends
# there = 10, and a later minute at most 5 orders + those 2 reads + 2 expired
# book reads = 9. At 11 s a fifth order fits in the first minute (11).
# Balanced step 4's one order a second is then well kept.
ORDER_SPACING_SECONDS = 12
OPEN_ORDERS_LOOKBACK_DAYS = 90  # the SDK's maximum `days` for account orders

# R4: only these states mean a recent order is done. Anything else (queued,
# accepted, partial, unknown) has not filled yet and keeps Balanced locked.
FINISHED_ORDER_STATUSES = frozenset(
    {
        "EXECUTED",
        "CANCELED",
        "PARTIAL_CANCELED",
        "FAILED",
        "REJECTED",
        "EXPIRED",
        "REPLACED",
    }
)

_cache: dict[tuple[Callable, str], tuple[float, object]] = {}
# ponytail: never pruned; at 10 requests a minute a day adds ~14k floats per account.
_request_times: dict[str, list[float]] = {}


@dataclass(frozen=True)
class Holding:
    """One position as SnapTrade reports it; cost basis is the average price per share."""

    ticker: str
    units: float
    price: float | None
    cost_basis: float | None


@dataclass(frozen=True)
class AccountSnapshot:
    """One account's USD cash and holdings from a single cached read."""

    cash: float
    holdings: list[Holding]


@functools.cache
def get_client() -> SnapTrade:
    """Return the SnapTrade client built with v13's Personal-key auth helper (Balanced step 1).

    A Personal key needs no user id or secret; passing the key straight to
    the constructor, as pre-v13 tutorials do, raises TypeError.
    """
    auth = SnapTradeAuth.personal_api_key(
        consumer_key=os.environ[CONSUMER_KEY_ENV_VAR],
        client_id=os.environ[CLIENT_ID_ENV_VAR],
    )
    return SnapTrade(auth=auth)


def is_alpaca_paper_configured() -> bool:
    """Return True when the key and the Alpaca Paper account id are all set in the environment."""
    required_env_vars = (
        CONSUMER_KEY_ENV_VAR,
        CLIENT_ID_ENV_VAR,
        ALPACA_PAPER_ACCOUNT_ENV_VAR,
    )
    return all(os.environ.get(name) for name in required_env_vars)


def is_sandbox_configured() -> bool:
    """Return True when the key and the Sandbox account id are all set in the environment."""
    required_env_vars = (
        CONSUMER_KEY_ENV_VAR,
        CLIENT_ID_ENV_VAR,
        SANDBOX_ACCOUNT_ENV_VAR,
    )
    return all(os.environ.get(name) for name in required_env_vars)


def alpaca_paper_account_id() -> str:
    """Return Balanced's Alpaca Paper account id (found once with `python brokerage.py`)."""
    return os.environ[ALPACA_PAPER_ACCOUNT_ENV_VAR]


def sandbox_account_id() -> str:
    """Return the read-only SnapTrade Sandbox account id used for demo holdings and activity."""
    return os.environ[SANDBOX_ACCOUNT_ENV_VAR]


@functools.cache
def check_trade_gate() -> bool:
    """Return True when this key can trade and ALPACA-PAPER allows trading (Balanced step 2, go/no-go).

    Cached for the process: the PRD checks it once at start-up.
    """
    client = get_client()
    partner_info = client.reference_data.get_partner_info().body
    if not partner_info.get("can_access_trades"):
        return False

    # allows_trading lives on the brokerage record, not on partner info.
    for brokerage in client.reference_data.list_all_brokerages().body:
        if brokerage.get("slug") == ALPACA_PAPER_SLUG:
            return bool(brokerage.get("allows_trading"))

    return False


def _cached_read[ReadT](fetch: Callable[[str], ReadT], account_id: str) -> ReadT:
    """Return `fetch(account_id)` from the cache while it is under CACHE_SECONDS old (T19)."""
    cache_key = (fetch, account_id)
    cached = _cache.get(cache_key)
    if cached is not None:
        fetched_at, value = cached
        if time.monotonic() - fetched_at < CACHE_SECONDS:
            return value

    value = fetch(account_id)
    _cache[cache_key] = (time.monotonic(), value)
    return value


def clear_cache(account_id: str) -> None:
    """Forget every cached read of one account so the next read goes to SnapTrade."""
    stale_keys = [key for key in _cache if key[1] == account_id]
    for key in stale_keys:
        del _cache[key]


def clear_order_cache(account_id: str) -> None:
    """Forget the account's cached order lists, so the lock and order status show a batch just sent."""
    # Queued orders don't move positions or cash, so the book read stays cached.
    for fetch in (_fetch_open_orders, _fetch_recent_orders):
        _cache.pop((fetch, account_id), None)


def _note_request(account_id: str) -> None:
    """Record one SnapTrade request against the account's per-minute budget (Risk #8)."""
    _request_times.setdefault(account_id, []).append(time.monotonic())


def seconds_until_order_room(account_id: str) -> float:
    """Return how long the next order must wait so it, and the reads after the batch, stay within 10 a minute.

    Page renders just before Approve can already have spent part of the
    minute; ORDER_SPACING_SECONDS alone covers only Approve's own requests.
    """
    window_start = time.monotonic() - RATE_WINDOW_SECONDS
    window_times = []
    for sent_at in _request_times.get(account_id, []):
        if sent_at > window_start:
            window_times.append(sent_at)
    window_times.sort()  # two Streamlit runs can append out of order
    excess_requests = (
        len(window_times) + 1 + POST_BATCH_READS - ACCOUNT_REQUESTS_PER_MINUTE
    )
    if excess_requests <= 0:
        return 0.0

    # The oldest `excess_requests` requests must leave the window first.
    return window_times[excess_requests - 1] - window_start


def _to_float(value: str | float | None) -> float | None:
    """Convert SnapTrade's decimal strings to float, keeping None as None."""
    if value is None:
        return None

    return float(value)


def _holding_from_position(position: dict) -> Holding:
    """Turn one `get_all_account_positions` row into a Holding."""
    return Holding(
        ticker=position["instrument"]["symbol"],
        units=_to_float(position.get("units")) or 0.0,
        price=_to_float(position.get("price")),
        cost_basis=_to_float(position.get("cost_basis")),
    )


def _usd_cash(balances: list[dict]) -> float:
    """Return the USD row's cash from `get_user_account_balance` (F22: USD account)."""
    for balance in balances:
        is_usd = (balance.get("currency") or {}).get("code") == "USD"
        if is_usd and balance.get("cash") is not None:
            return float(balance["cash"])

    raise ValueError("SnapTrade returned no USD cash balance for this account")


def _fetch_account_snapshot(account_id: str) -> AccountSnapshot:
    """Read positions plus balance for one account (never `get_user_holdings`: 410 for new accounts)."""
    account_information = get_client().account_information
    _note_request(account_id)
    positions = account_information.get_all_account_positions(
        account_id=account_id
    ).body["results"]
    _note_request(account_id)
    balances = account_information.get_user_account_balance(account_id=account_id).body

    holdings = [_holding_from_position(position) for position in positions]
    return AccountSnapshot(cash=_usd_cash(balances), holdings=holdings)


def get_account_snapshot(account_id: str) -> AccountSnapshot:
    """Return one account's cash and holdings, read at most once per CACHE_SECONDS (Balanced step 3)."""
    return _cached_read(_fetch_account_snapshot, account_id)


def _order_row(order: dict) -> dict[str, str | float | None]:
    """Turn one recent-orders record into a plain, JSON-ready row."""
    universal_symbol = order.get("universal_symbol") or {}
    return {
        "brokerage_order_id": order.get("brokerage_order_id"),
        "status": order.get("status"),
        "ticker": universal_symbol.get("symbol"),
        "action": order.get("action"),
        "total_quantity": _to_float(order.get("total_quantity")),
        "filled_quantity": _to_float(order.get("filled_quantity")),
        "execution_price": _to_float(order.get("execution_price")),
        "time_placed": order.get("time_placed"),
    }


def _fetch_recent_orders(account_id: str) -> list[dict[str, str | float | None]]:
    """Read the last 24 h of orders in every status."""
    # The SDK default is executed orders only, which would hide queued
    # weekend orders and silently switch off the R4 lock.
    _note_request(account_id)
    response = get_client().account_information.get_user_account_recent_orders(
        account_id=account_id,
        only_executed=False,
    )
    return [_order_row(order) for order in response.body.get("orders", [])]


def get_recent_orders(account_id: str) -> list[dict[str, str | float | None]]:
    """Return the account's orders from the last 24 hours, cached; no fill polling (Balanced step 6)."""
    return _cached_read(_fetch_recent_orders, account_id)


def _fetch_open_orders(account_id: str) -> list[dict[str, str | float | None]]:
    """Read every order SnapTrade still lists as open, over the longest window it allows."""
    _note_request(account_id)
    response = get_client().account_information.get_user_account_orders(
        account_id=account_id,
        state="open",
        days=OPEN_ORDERS_LOOKBACK_DAYS,
    )
    return [_order_row(order) for order in response.body]


def get_open_orders(account_id: str) -> list[dict[str, str | float | None]]:
    """Return the account's open orders from the last 90 days, cached (R4)."""
    return _cached_read(_fetch_open_orders, account_id)


def has_unfilled_orders(account_id: str) -> bool:
    """Return True while any order is still open, which keeps Balanced locked (Balanced step 7, R4).

    Any order SnapTrade lists as open locks, as does any last-24h order
    that isn't in a finished status, unknown statuses included.
    """
    if get_open_orders(account_id):
        return True

    # Without real-time data SnapTrade refreshes the 90-day list once a day,
    # so an order sent minutes ago may be missing; the 24 h read is realtime.
    for order in get_recent_orders(account_id):
        if order["status"] not in FINISHED_ORDER_STATUSES:
            return True

    return False


def _activity_row(activity: dict) -> dict[str, str | float | None]:
    """Turn one account activity into a plain, JSON-ready row."""
    symbol = activity.get("symbol") or {}
    currency = activity.get("currency") or {}
    return {
        "date": activity.get("trade_date") or activity.get("settlement_date"),
        "type": activity.get("type"),
        "ticker": symbol.get("symbol"),
        "units": _to_float(activity.get("units")),
        "price": _to_float(activity.get("price")),
        "amount": _to_float(activity.get("amount")),
        "currency": currency.get("code"),
        "description": activity.get("description"),
    }


def _fetch_activities(account_id: str) -> list[dict[str, str | float | None]]:
    """Read every activity type in one call: dividends, splits, mergers, transfers, trades."""
    _note_request(account_id)
    response = get_client().account_information.get_account_activities(
        account_id=account_id
    )
    return [_activity_row(activity) for activity in response.body["data"]]


def get_activities(account_id: str) -> list[dict[str, str | float | None]]:
    """Return the account's activity rows, newest first as SnapTrade sends them, cached (S12)."""
    return _cached_read(_fetch_activities, account_id)


def place_order(account_id: str, order: Order) -> dict[str, str | int | None]:
    """Send one Market/Day whole-share order and return its acknowledgement row (Balanced steps 4-5).

    The caller waits ORDER_SPACING_SECONDS and `seconds_until_order_room`
    before each order and clears the order cache once the batch is out.
    """
    client_order_id = str(uuid.uuid4())
    _note_request(account_id)
    response = get_client().trading.place_force_order(
        account_id=account_id,
        action=order.side.upper(),
        order_type="Market",
        time_in_force="Day",
        symbol=order.ticker,
        # The SDK's units schema rejects an int; a whole float is still whole shares (F21).
        units=float(order.quantity),
        client_order_id=client_order_id,
    )
    return {
        "ticker": order.ticker,
        "side": order.side,
        "quantity": order.quantity,
        "client_order_id": client_order_id,
        "brokerage_order_id": response.body.get("brokerage_order_id"),
        "status": response.body.get("status"),
    }


def _account_row(account: dict) -> dict[str, str | bool | None]:
    """Turn one connected account into the row `python brokerage.py` prints."""
    return {
        "id": account["id"],
        "name": account.get("name"),
        "institution_name": account.get("institution_name"),
        "is_paper": account.get("is_paper"),
    }


def list_accounts() -> list[dict[str, str | bool | None]]:
    """Return every connected account's id, name and institution (setup only)."""
    accounts = get_client().account_information.list_user_accounts().body
    return [_account_row(account) for account in accounts]


if __name__ == "__main__":
    # Setup checklist go/no-go (story 49): one real partner-info call, then
    # the account ids to copy into .env.
    load_dotenv()
    print(f"trade gate open: {check_trade_gate()}")
    for account in list_accounts():
        print(account)
