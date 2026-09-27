"""snapi's Streamlit app: the password gate and the dashboard tabs (PRD Streamlit app).

Heavy modules (books, committee, execute, brokerage and everything they pull
in: langgraph, langchain, snaptrade, external/) are imported inside the live
functions that need them, not at the top. The PRD asks for lazy imports to
keep start-up fast; that overrides the usual imports-at-the-top rule.
Renderers take plain JSON-shaped dicts, so a run logged to
`data/run_log.jsonl` and a run made in this session draw the same way.
"""

from __future__ import annotations

import dataclasses
import hmac
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

import config

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from streamlit.delta_generator import DeltaGenerator

    from committee import Proposal, Snapshot
    from portfolio import Book, RebalanceResult

logger = logging.getLogger(__name__)

APP_PASSWORD_SECRET = "APP_PASSWORD"
ANTHROPIC_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
SNAPTRADE_SETUP_NOTE = (
    "SnapTrade isn't configured. Set SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID "
    "and SNAPTRADE_ALPACA_PAPER_ACCOUNT_ID in .env."
)
SANDBOX_SETUP_NOTE = (
    "Not configured. Set SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID and "
    "SNAPTRADE_SANDBOX_ACCOUNT_ID in .env to show the Sandbox's demo holdings."
)

ACTIVITY_SETUP_NOTE = (
    "The Activity tab needs SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID, "
    "SNAPTRADE_SANDBOX_ACCOUNT_ID and SNAPTRADE_ALPACA_PAPER_ACCOUNT_ID in .env."
)
CHAT_SETUP_NOTE = (
    "The chat needs ANTHROPIC_API_KEY, SNAPTRADE_MCP_URL and "
    "SNAPTRADE_MCP_OAUTH_TOKEN (minted by hand, R6) in .env."
)

# SnapTrade's hosted MCP: 18 read-only tools, so the chat can't trade and
# the SDK stays the only write path (S13, T21, story 42).
SNAPTRADE_MCP_URL_ENV_VAR = "SNAPTRADE_MCP_URL"
SNAPTRADE_MCP_TOKEN_ENV_VAR = "SNAPTRADE_MCP_OAUTH_TOKEN"
SNAPTRADE_MCP_SERVER_NAME = "snaptrade"
CHAT_MAX_TOKENS = 4000
CHAT_SYSTEM_PROMPT = (
    "You answer questions about the user's SnapTrade brokerage accounts, which "
    "hold paper and demo money only. Read them with the SnapTrade tools. You "
    "can't place or change trades. Keep answers short and plain."
)
CHAT_PLACEHOLDER = "Ask about your portfolio, e.g. What's my largest sector exposure?"

# Cash dividends. REI (reinvestment) pairs with a DIVIDEND row, so counting
# it too would double the income.
DIVIDEND_TYPES = frozenset({"DIVIDEND", "SUBSTITUTE_DIVIDEND"})
# Trades start hidden: Q11's table is dividends, splits, mergers and
# transfers. The type filter can add them back.
TRADE_TYPES = frozenset({"BUY", "SELL"})

NEW_YORK = ZoneInfo("America/New_York")
SATURDAY = 5  # datetime.weekday(): Monday is 0

ANALYSTS = ("sentiment", "technical", "macro")
CARD_TITLES = {
    "sentiment": "Sentiment analyst",
    "technical": "Technical analyst",
    "macro": "Macro analyst",
    "pm": "Portfolio Manager",
    "critique": "Devil's advocate",
}
# Graph node name -> the state key its update carries. The critic writes
# "critique"; the size node is code only and has no card.
CARD_BY_NODE = {
    "sentiment": "sentiment",
    "technical": "technical",
    "macro": "macro",
    "pm": "pm",
    "critic": "critique",
}


def read_jsonl(path: Path) -> list[dict]:
    """Return one dict per non-blank line of a JSON-lines file, or [] when it doesn't exist yet."""
    if not path.exists():
        return []

    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            records.append(json.loads(line))

    return records


def display_name(persona_key: str) -> str:
    """Return the persona's display name, for selector labels and notes."""
    return config.PERSONAS[persona_key].display_name


def has_anthropic_key() -> bool:
    """Return True when ANTHROPIC_API_KEY is set; Run swarm needs it."""
    return bool(os.environ.get(ANTHROPIC_KEY_ENV_VAR))


def show_live_error(action: str, error: Exception) -> None:
    """Explain a failed live call on the page instead of a traceback; the traceback goes to the terminal."""
    logger.error("%s failed", action, exc_info=error)
    st.error(f"{action} failed ({type(error).__name__}): {error}")


def escape_markdown_math(text: str) -> str:
    """Return LLM text with `$` escaped, so Streamlit never renders dollar amounts as LaTeX."""
    return text.replace("$", r"\$")


def read_app_password() -> str | None:
    """Return APP_PASSWORD from Streamlit secrets, or None when there is no secrets file or no such key."""
    try:
        return str(st.secrets[APP_PASSWORD_SECRET])
    except (FileNotFoundError, KeyError):
        # No .streamlit/secrets.toml raises a FileNotFoundError subclass.
        return None


def check_entered_password() -> None:
    """Unlock the session when the typed password matches; the typed text is never kept."""
    entered_password = st.session_state.pop("password")
    expected_password = read_app_password() or ""

    # Bytes, because compare_digest rejects non-ASCII str.
    st.session_state["password_ok"] = hmac.compare_digest(
        entered_password.encode(),
        expected_password.encode(),
    )


def check_password() -> None:
    """Show the disclaimer, then stop the script until the password matches APP_PASSWORD (S9, T17, Q13)."""
    st.title("snapi")
    st.caption(config.DISCLAIMER)
    if st.session_state.get("password_ok"):
        return

    if not read_app_password():
        st.error(
            f"No {APP_PASSWORD_SECRET} found. Add it to .streamlit/secrets.toml "
            "(or the Cloud app's Secrets) and reload."
        )
        st.stop()

    st.text_input(
        "Password",
        type="password",
        key="password",
        on_change=check_entered_password,
    )
    if st.session_state.get("password_ok") is False:
        st.error("Wrong password.")
    st.stop()


def find_trade_blocker(persona_key: str) -> str | None:
    """Return why a SnapTrade persona can't trade right now (no keys, or R5's closed gate), else None.

    The trade-gate call is cached for the process, so the PRD's start-up
    check costs one SnapTrade call.
    """
    if config.PERSONAS[persona_key].account != "alpaca_paper":
        return None

    import brokerage

    if not brokerage.is_alpaca_paper_configured():
        return SNAPTRADE_SETUP_NOTE

    if not brokerage.check_trade_gate():
        return (
            "SnapTrade's trade gate is closed. R5: route Balanced to the "
            "simulator in config.py."
        )

    return None


def prepare_live_books() -> None:
    """Seed the simulator personas' opening books once (S16) and check SnapTrade can trade (Balanced step 2)."""
    import execute

    try:
        execute.seed_books()
    except Exception as error:
        show_live_error("Seeding the opening books", error)

    for persona_key, persona in config.PERSONAS.items():
        try:
            blocker = find_trade_blocker(persona_key)
        except Exception as error:
            show_live_error(f"Checking {persona.display_name}'s trade gate", error)
            continue

        if blocker is not None:
            st.warning(f"{persona.display_name} can't trade: {blocker}")


def render_analyst_report(report: dict) -> None:
    """List each ticker's score with its one-line why; an abstained analyst shows its error first."""
    if report["abstained"]:
        st.warning(report["why"].get("error", "Abstained this cycle."))

    lines = []
    for ticker, score in report["score"].items():
        why = report["why"].get(ticker, "")
        lines.append(f"- **{ticker}** {score:+.2f}: {why}")

    st.markdown(escape_markdown_math("\n".join(lines)))


def render_macro_report(report: dict) -> None:
    """Show the regime and each derived flag, raised or clear, with its why (never the raw FRED series)."""
    if report["abstained"]:
        st.warning(report["why"].get("error", "Abstained this cycle."))

    regime = report["regime"].replace("_", "-")
    lines = [f"Regime: **{regime}**"]
    for flag, is_raised in report["flags"].items():
        status = "raised" if is_raised else "clear"
        why = report["why"].get(flag, "")
        lines.append(f"- **{flag}** {status}: {why}")

    st.markdown(escape_markdown_math("\n".join(lines)))


def render_pm_view(pm_view: dict) -> None:
    """Show the PM memo and every name it dropped, with the reason (R1: drop-only)."""
    st.markdown(escape_markdown_math(pm_view["memo"]))
    for drop in pm_view["drops"]:
        drop_line = f"Dropped **{drop['ticker']}**: {drop['reason']}"
        st.markdown(escape_markdown_math(drop_line))


def render_critique(critique: dict) -> None:
    """Show the critic's notes and, per persona, the code-computed holdings down more than 10% from cost (S5, Q10)."""
    st.markdown(escape_markdown_math(critique["notes"]))

    lines = ["Down more than 10% from cost:"]
    for persona_key, tickers in critique["down_more_than_10pct"].items():
        lines.append(f"- {display_name(persona_key)}: {', '.join(tickers) or 'none'}")

    st.markdown("\n".join(lines))


def render_card(slot: DeltaGenerator, card: str, report: dict) -> None:
    """Draw one agent's card into its slot, replacing whatever the slot showed before (S11)."""
    with slot.container(border=True):
        st.markdown(f"**{CARD_TITLES[card]}**")
        if card == "pm":
            render_pm_view(report)
        elif card == "critique":
            render_critique(report)
        elif card == "macro":
            render_macro_report(report)
        else:
            render_analyst_report(report)


def render_orders(orders: list[dict]) -> None:
    """Show whole-share orders as a table: ticker, side, shares, price and notional."""
    if not orders:
        st.caption(r"No trades (each change is under \$200 or already on target).")
        return

    rows = []
    for order in orders:
        rows.append(
            {
                "Ticker": order["ticker"],
                "Side": order["side"],
                "Shares": order["quantity"],
                "Price": round(order["price"], 2),
                "Notional": round(order["price"] * order["quantity"], 2),
            }
        )

    st.dataframe(rows, hide_index=True)


def render_persona_orders(orders_by_persona: dict[str, list[dict]]) -> None:
    """Show the three personas' order tables side by side: one view, three books (story 43)."""
    columns = st.columns(len(config.PERSONAS))
    for column, (persona_key, persona) in zip(
        columns, config.PERSONAS.items(), strict=True
    ):
        with column:
            st.markdown(f"**{persona.display_name}** orders")
            render_orders(orders_by_persona[persona_key])


def stream_committee(
    snapshot: Snapshot,
    card_slots: dict[str, DeltaGenerator],
) -> Proposal:
    """Stream the graph in "updates" mode, filling each card as its node finishes, and return the proposal.

    The loop runs in the main script thread: Streamlit is never called
    inside a graph node, which has no session context (R8).
    """
    import committee

    graph_updates = committee.build_graph().stream(
        {"snapshot": snapshot},
        stream_mode="updates",
    )
    state = {"snapshot": snapshot}
    for update in graph_updates:
        for node_name, node_update in update.items():
            state.update(node_update)
            card = CARD_BY_NODE.get(node_name)
            if card is not None:
                report = node_update[card].model_dump(mode="json")
                render_card(card_slots[card], card, report)

    return committee.build_proposal(state)


def run_swarm(card_slots: dict[str, DeltaGenerator]) -> None:
    """Run one committee cycle, log it with every persona's NAV (F23) and keep it in the session for Approve."""
    import books

    # A new run replaces the proposal on screen; nothing trades until Approve.
    st.session_state.pop("run_record", None)
    st.session_state.pop("approval", None)

    try:
        with st.spinner("The committee is meeting..."):
            snapshot = books.take_snapshot()
            proposal = stream_committee(snapshot, card_slots)
            run_record = books.log_run(snapshot, proposal)
    except Exception as error:
        show_live_error("Run swarm", error)
        return

    st.session_state["run_record"] = run_record.model_dump(mode="json")


def execute_for_persona(proposal: Proposal, persona_key: str, tag: str) -> str | None:
    """Execute one persona's leg of an approved proposal; return why it didn't run, or None when it did."""
    import execute

    persona_name = display_name(persona_key)
    try:
        blocker = find_trade_blocker(persona_key)
        if blocker is not None:
            return f"{persona_name} skipped: {blocker}"

        execute.execute(proposal, persona_key, tag=tag)
    except execute.ExecutionLockedError as error:
        return str(error)
    except Exception as error:
        logger.error("%s execution failed", persona_name, exc_info=error)
        return f"{persona_name} failed ({type(error).__name__}): {error}"

    return None


def execute_personas(proposal: Proposal, persona_keys: list[str], tag: str) -> dict:
    """Execute each persona's leg and return the skip notes plus the run-log records it wrote."""
    first_new_line = len(read_jsonl(config.RUN_LOG_PATH))

    notes = {}
    for persona_key in persona_keys:
        note = execute_for_persona(proposal, persona_key, tag)
        if note is not None:
            notes[persona_key] = note

    new_records = read_jsonl(config.RUN_LOG_PATH)[first_new_line:]
    return {"notes": notes, "records": new_records}


def next_open_phrase(approved_at: str) -> str:
    """Name the open that orders sent at `approved_at` wait for: Monday's on a weekend, else the next one."""
    approved_in_new_york = datetime.fromisoformat(approved_at).astimezone(NEW_YORK)
    if approved_in_new_york.weekday() >= SATURDAY:
        return "Monday's 09:30 ET open"

    return "the next market open"


def render_approval_record(record: dict) -> None:
    """Show one executed batch: SnapTrade's acknowledgements for Balanced, simulator fills for the others.

    A batch whose first SnapTrade order failed has neither; its failure
    note is shown above it.
    """
    if record["acks"]:
        st.success(
            "Sent through SnapTrade to Alpaca Paper: queued until "
            f"{next_open_phrase(record['approved_at'])}. Each order's "
            "acknowledgement is below."
        )
        ack_rows = []
        for ack in record["acks"]:
            ack_rows.append(
                {
                    "Brokerage order id": ack["brokerage_order_id"],
                    "Status": ack["status"],
                    "Ticker": ack["ticker"],
                    "Side": ack["side"],
                    "Shares": ack["quantity"],
                }
            )
        st.dataframe(ack_rows, hide_index=True)

    if record["fills"]:
        st.success(f"{len(record['fills'])} simulator fills at the snapshot mark.")
        render_orders(record["fills"])


def render_execution(persona_key: str, outcome: dict) -> None:
    """Show what one persona's execution did: its skip note, its records, or that nothing needed trading."""
    note = outcome["notes"].get(persona_key)
    if note is not None:
        st.warning(note)

    records = [
        record for record in outcome["records"] if record["persona"] == persona_key
    ]
    for record in records:
        render_approval_record(record)

    if note is None and not records:
        st.caption("Nothing to trade: the book already matches the approved targets.")


def caption_queued_lock() -> None:
    """Caption the SnapTrade persona while its earlier orders are still queued (R4); Approve skips it."""
    import brokerage

    snaptrade_names = [
        persona.display_name
        for persona in config.PERSONAS.values()
        if persona.account == "alpaca_paper"
    ]
    if not snaptrade_names or not brokerage.is_alpaca_paper_configured():
        return

    try:
        is_locked = brokerage.has_unfilled_orders(brokerage.alpaca_paper_account_id())
    except Exception as error:
        show_live_error("Checking for queued SnapTrade orders", error)
        return

    if is_locked:
        st.caption(
            f"{', '.join(snaptrade_names)} is locked: its orders are queued for the "
            "next open. Approve skips it until they fill; the proposal is still shown."
        )


def approve_controls(run_record: dict) -> None:
    """Approve executes the proposal on screen for all three personas; nothing trades before it (story 22, T18)."""
    caption_queued_lock()
    if not st.button("Approve", type="primary", key="approve"):
        return

    from committee import Proposal

    proposal = Proposal.model_validate(run_record["proposal"])
    st.session_state["approval"] = execute_personas(
        proposal,
        list(config.PERSONAS),
        tag="committee",
    )
    clear_rebalance_state()


def committee_tab() -> None:
    """Run swarm fills five agent cards live; then the PM memo, three order tables and the critic beside Approve."""
    run_clicked = st.button(
        "Run swarm",
        type="primary",
        disabled=not has_anthropic_key(),
    )
    if not has_anthropic_key():
        st.caption(f"Run swarm needs {ANTHROPIC_KEY_ENV_VAR} in .env.")

    # Lay out every slot first, so cards fill in place while the graph streams.
    card_slots = {}
    for analyst, column in zip(ANALYSTS, st.columns(len(ANALYSTS)), strict=True):
        card_slots[analyst] = column.empty()
    card_slots["pm"] = st.empty()
    orders_area = st.container()
    critic_column, approve_column = st.columns([2, 1])
    card_slots["critique"] = critic_column.empty()

    if run_clicked:
        run_swarm(card_slots)

    run_record = st.session_state.get("run_record")
    if run_record is None:
        st.info("Press Run swarm to convene the committee.")
        return

    proposal = run_record["proposal"]
    for card, slot in card_slots.items():
        render_card(slot, card, proposal[card])
    with orders_area:
        render_persona_orders(proposal["orders"])
    with approve_column:
        approve_controls(run_record)

    approval = st.session_state.get("approval")
    if approval is None:
        return

    st.subheader("Execution")
    columns = st.columns(len(config.PERSONAS))
    for column, (persona_key, persona) in zip(
        columns, config.PERSONAS.items(), strict=True
    ):
        with column:
            st.markdown(f"**{persona.display_name}**")
            render_execution(persona_key, approval)


def book_equity(book: dict, marks: dict[str, float]) -> float:
    """Return a JSON-shaped book's value: cash plus every position at its mark."""
    positions_value = 0.0
    for ticker, shares in book["shares"].items():
        positions_value += shares * marks[ticker]

    return book["cash"] + positions_value


def holdings_rows(book: dict, marks: dict[str, float]) -> list[dict]:
    """Return one table row per held ticker: sector, shares, mark, value, weight and average cost."""
    equity = book_equity(book, marks)

    rows = []
    for ticker, shares in sorted(book["shares"].items()):
        value = shares * marks[ticker]
        avg_cost = book["avg_cost"].get(ticker)
        rows.append(
            {
                "Ticker": ticker,
                "Sector": config.SECTOR_BY_TICKER.get(ticker, "none"),
                "Shares": shares,
                "Price": round(marks[ticker], 2),
                "Value": round(value, 2),
                "Weight %": round(100 * value / equity, 1),
                "Avg cost": None if avg_cost is None else round(avg_cost, 2),
            }
        )

    return rows


def sector_chart(
    weights_by_series: dict[str, dict[str, float]],
    caps_key: str,
) -> go.Figure:
    """Return grouped bars of sector weights (%) with a dashed line at the chosen persona's sector cap (S10)."""
    caps_persona = config.PERSONAS[caps_key]

    figure = go.Figure()
    for series_name, weights in weights_by_series.items():
        percents = [100 * weights[sector] for sector in config.SECTORS]
        figure.add_bar(x=list(config.SECTORS), y=percents, name=series_name)

    figure.add_hline(
        y=100 * caps_persona.sector_cap,
        line_dash="dash",
        annotation_text=f"{caps_persona.display_name} cap {caps_persona.sector_cap:.0%}",
    )
    figure.update_layout(
        barmode="group",
        yaxis_title="Weight (%)",
        height=320,
        showlegend=len(weights_by_series) > 1,
    )
    return figure


def render_book(
    persona_key: str,
    book: dict,
    marks: dict[str, float],
    sector_weights: dict[str, float],
) -> None:
    """Show one persona's limits, NAV, cash, holdings and sector weights against its own cap (S10, story 38)."""
    persona = config.PERSONAS[persona_key]
    st.caption(
        f"Sector cap {persona.sector_cap:.0%}, max per name "
        f"{persona.max_position_pct:.0%}, cash buffer {persona.cash_buffer:.0%}"
    )
    nav_column, cash_column = st.columns(2)
    nav_column.metric("NAV", f"${book_equity(book, marks):,.0f}")
    cash_column.metric("Cash", f"${book['cash']:,.0f}")

    holdings_column, chart_column = st.columns(2)
    with holdings_column:
        holdings = holdings_rows(book, marks)
        if holdings:
            st.dataframe(holdings, hide_index=True)
        else:
            st.caption("All cash: no positions yet.")
    with chart_column:
        figure = sector_chart({"Now": sector_weights}, persona_key)
        st.plotly_chart(figure, key=f"sectors_{persona_key}")


def preview_rebalance(
    persona_key: str,
    caps_key: str,
    book: Book,
    marks: dict[str, float],
) -> RebalanceResult:
    """Run the cycle's own sector-cap layer on the book's current weights; no LLM call (S8, stories 30-32)."""
    import books
    import portfolio

    persona = config.PERSONAS[persona_key]
    invested_pct = persona.invested_pct_for_regime(books.current_regime())
    return portfolio.rebalance(
        book,
        persona,
        portfolio.weights_by_ticker(book, marks),
        marks,
        invested_pct,
        caps_persona=config.PERSONAS[caps_key],
    )


def approve_rebalance(persona_key: str, result: RebalanceResult) -> None:
    """Send the previewed rebalance through the same `execute` as Approve, then redraw the fresh book (story 32)."""
    from committee import Proposal

    proposal = Proposal(
        target_weights={persona_key: result.target_weights},
        orders={persona_key: result.orders},
    )
    st.session_state[f"rebalance_outcome_{persona_key}"] = execute_personas(
        proposal,
        [persona_key],
        tag="rebalance",
    )
    del st.session_state[f"rebalance_preview_{persona_key}"]
    st.rerun()


def render_rebalance_preview(
    persona_key: str,
    caps_key: str,
    result: RebalanceResult,
) -> None:
    """Show before/after sector weights and the orders; only a preview on the persona's own caps is approvable (R3)."""
    st.markdown(f"**Rebalance preview against {display_name(caps_key)}'s caps**")
    before_and_after = {
        "Before": result.before_sector_weights,
        "After": result.after_sector_weights,
    }
    figure = sector_chart(before_and_after, caps_key)
    st.plotly_chart(figure, key=f"rebalance_chart_{persona_key}")
    render_orders([order.model_dump() for order in result.orders])

    # A stricter mandate is a what-if: the persona's own caps still apply
    # on the next committee cycle.
    if caps_key != persona_key:
        st.caption(
            f"Preview only: what moving to {display_name(caps_key)}'s caps would trade."
        )
        return

    if not result.orders:
        return

    if st.button("Approve rebalance", key=f"approve_rebalance_{persona_key}"):
        approve_rebalance(persona_key, result)


def rebalance_controls(persona_key: str, book: Book, marks: dict[str, float]) -> None:
    """Rebalance on demand, optionally against another persona's caps; the preview lives in the session (S8, R3)."""
    persona_keys = list(config.PERSONAS)
    caps_key = st.selectbox(
        "Check against caps of",
        persona_keys,
        index=persona_keys.index(persona_key),
        format_func=display_name,
        key=f"caps_{persona_key}",
    )
    preview_key = f"rebalance_preview_{persona_key}"
    outcome_key = f"rebalance_outcome_{persona_key}"

    # Kept in session_state: a button under another button's `if` never
    # fires, because the outer button is False again on that rerun.
    if st.button("Rebalance", key=f"rebalance_{persona_key}"):
        st.session_state.pop(preview_key, None)
        st.session_state.pop(outcome_key, None)
        try:
            result = preview_rebalance(persona_key, caps_key, book, marks)
        except Exception as error:
            show_live_error("Rebalance", error)
        else:
            st.session_state[preview_key] = (caps_key, result)

    outcome = st.session_state.get(outcome_key)
    if outcome is not None:
        render_execution(persona_key, outcome)

    preview = st.session_state.get(preview_key)
    if preview is not None:
        previewed_caps_key, result = preview
        render_rebalance_preview(persona_key, previewed_caps_key, result)


def clear_rebalance_state() -> None:
    """Forget every rebalance preview and outcome once the books change underneath them."""
    for persona_key in config.PERSONAS:
        st.session_state.pop(f"rebalance_preview_{persona_key}", None)
        st.session_state.pop(f"rebalance_outcome_{persona_key}", None)


def render_sandbox(rows: list[dict]) -> None:
    """Show the SnapTrade Sandbox's demo holdings on their own; they never join a persona's book (S16, story 59)."""
    st.caption(
        "Pre-filled demo holdings from SnapTrade's read-only Sandbox. "
        "They are not part of any persona's book."
    )
    if not rows:
        st.caption("The Sandbox account holds nothing.")
        return

    st.dataframe(rows, hide_index=True)


def sandbox_panel() -> None:
    """Read the Sandbox holdings through the shared 60 s SnapTrade cache and show them, or say why not."""
    import brokerage

    st.subheader("SnapTrade Sandbox (demo data)")
    if not brokerage.is_sandbox_configured():
        st.caption(SANDBOX_SETUP_NOTE)
        return

    try:
        account = brokerage.get_account_snapshot(brokerage.sandbox_account_id())
    except Exception as error:
        show_live_error("Reading the SnapTrade Sandbox", error)
        return

    render_sandbox([dataclasses.asdict(holding) for holding in account.holdings])


def reset_books_control() -> None:
    """Clear the simulator fill logs and re-seed the opening books, for a clean rehearsal (S16, story 58)."""
    import execute

    st.subheader("Reset books")
    if st.session_state.pop("books_reset", False):
        st.success("Simulator books cleared and re-seeded.")

    confirmed = st.checkbox(
        "Clear the simulator fill logs and re-seed the opening books",
        key="confirm_reset",
    )
    if not st.button("Reset books", disabled=not confirmed, key="reset_books"):
        return

    clear_rebalance_state()
    try:
        execute.reset_books()
    except Exception as error:
        show_live_error("Reset books", error)
        return

    st.session_state["books_reset"] = True
    st.rerun()


def portfolio_tab() -> None:
    """Per-persona holdings and sector charts with Rebalance, then the Sandbox panel and Reset books."""
    import books
    import portfolio
    import prices

    marks = prices.latest_marks()
    for persona_key, persona in config.PERSONAS.items():
        st.subheader(persona.display_name)
        try:
            book = books.current_book(persona_key)
        except Exception as error:
            show_live_error(f"Reading {persona.display_name}'s book", error)
            continue

        weights_by_sector = portfolio.sector_weights(book, marks)
        render_book(persona_key, dataclasses.asdict(book), marks, weights_by_sector)
        rebalance_controls(persona_key, book, marks)

    sandbox_panel()
    reset_books_control()


def normalized(values: list[float]) -> list[float]:
    """Return the series rescaled so its first value is 100."""
    return [100 * value / values[0] for value in values]


def forward_test_tab() -> None:
    """Plot NAV per persona and SPY from the run log, normalized to the first logged run (S14, F23, stories 44-45)."""
    run_records = read_jsonl(config.RUN_LOG_PATH)
    runs = [record for record in run_records if record["kind"] == "run"]
    if not runs:
        st.info(
            "The forward test starts with the first Run swarm: "
            "every run logs each persona's NAV."
        )
        return

    first_run_day = runs[0]["taken_at"][:10]
    st.caption(
        f"Forward test started today ({first_run_day}). Weekend runs are marked "
        "at Friday's close, so the lines start flat."
    )

    run_times = [run["taken_at"] for run in runs]
    figure = go.Figure()
    for persona_key, persona in config.PERSONAS.items():
        navs = [run["nav"][persona_key] for run in runs]
        figure.add_scatter(
            x=run_times,
            y=normalized(navs),
            name=persona.display_name,
            mode="lines+markers",
        )

    spy_closes = [run["spy_close"] for run in runs]
    figure.add_scatter(
        x=run_times,
        y=normalized(spy_closes),
        name=config.BENCHMARK,
        mode="lines+markers",
        line={"color": "gray", "dash": "dash"},
    )
    figure.update_layout(yaxis_title="Value (first run = 100)", height=420)
    st.plotly_chart(figure, key="forward_test")


def dividend_income(activities: list[dict]) -> float:
    """Return the cash dividends paid across the activity rows."""
    income = 0.0
    for activity in activities:
        if activity["type"] in DIVIDEND_TYPES:
            income += activity["amount"] or 0.0

    return income


def render_sandbox_activities(activities: list[dict]) -> None:
    """Show dividend income and the Sandbox's activity rows, newest first, filtered by type (S12, Q11)."""
    st.subheader("SnapTrade Sandbox activity (demo data)")
    st.metric("Dividend income", f"${dividend_income(activities):,.2f}")

    # SnapTrade's type is a best-effort string with no merger value, so
    # the filter offers whatever types the data holds.
    present_types = sorted({activity["type"] for activity in activities})
    default_types = [name for name in present_types if name not in TRADE_TYPES]
    chosen_types = st.multiselect(
        "Types",
        present_types,
        default=default_types,
        key="activity_types",
    )

    # SnapTrade already returns activities newest first.
    shown = [activity for activity in activities if activity["type"] in chosen_types]
    if not shown:
        st.caption("No activity of the chosen types.")
        return

    st.dataframe(shown, hide_index=True)


def render_alpaca_orders(orders: list[dict]) -> None:
    """Show Balanced's Alpaca Paper orders from the last 24 h with their status, newest first."""
    st.subheader("Alpaca Paper order history (last 24 h)")
    st.caption(
        "Alpaca Paper doesn't simulate dividends, so income shows for the Sandbox only."
    )
    if not orders:
        st.caption("No orders in the last 24 hours.")
        return

    newest_first = sorted(
        orders, key=lambda order: order["time_placed"] or "", reverse=True
    )
    st.dataframe(newest_first, hide_index=True)


def activity_tab() -> None:
    """Show SnapTrade Sandbox activity and Alpaca Paper order history, read-only (S12, story 40)."""
    import brokerage

    if not (
        brokerage.is_sandbox_configured() and brokerage.is_alpaca_paper_configured()
    ):
        st.info(ACTIVITY_SETUP_NOTE)
        return

    # Both reads go through the 60 s cache. Recent orders is the same
    # all-status read the R4 lock uses, so queued orders show their status (Q11).
    try:
        activities = brokerage.get_activities(brokerage.sandbox_account_id())
        orders = brokerage.get_recent_orders(brokerage.alpaca_paper_account_id())
    except Exception as error:
        show_live_error("Reading SnapTrade activity", error)
        return

    render_sandbox_activities(activities)
    render_alpaca_orders(orders)


def is_chat_configured() -> bool:
    """Return True when the Anthropic key, the MCP URL and the hand-minted OAuth token are all set (R6)."""
    required_env_vars = (
        ANTHROPIC_KEY_ENV_VAR,
        SNAPTRADE_MCP_URL_ENV_VAR,
        SNAPTRADE_MCP_TOKEN_ENV_VAR,
    )
    return all(os.environ.get(name) for name in required_env_vars)


def portfolio_chat_model() -> Runnable:
    """Return Claude wired to SnapTrade's hosted MCP with the hand-minted OAuth token (S13, R6).

    langchain-anthropic adds the mcp-client beta header itself once
    `mcp_servers` is set. The Claude API's MCP connector isn't ZDR-eligible,
    which is fine for paper data.
    """
    from langchain_anthropic import ChatAnthropic

    model = ChatAnthropic(
        model=config.get_model_name(),
        max_tokens=CHAT_MAX_TOKENS,
        mcp_servers=[
            {
                "type": "url",
                "url": os.environ[SNAPTRADE_MCP_URL_ENV_VAR],
                "name": SNAPTRADE_MCP_SERVER_NAME,
                "authorization_token": os.environ[SNAPTRADE_MCP_TOKEN_ENV_VAR],
            }
        ],
    )
    # The server entry only connects; the toolset entry offers its tools to Claude.
    toolset = {"type": "mcp_toolset", "mcp_server_name": SNAPTRADE_MCP_SERVER_NAME}
    return model.bind_tools([toolset])


def render_chat_turns(turns: list[dict]) -> None:
    """Show each question and answer as a chat bubble."""
    for turn in turns:
        with st.chat_message(turn["role"]):
            st.markdown(escape_markdown_math(turn["content"]))


def ask_portfolio_chat(history: list[dict], question: str) -> None:
    """Answer one question with the whole conversation so far, then keep both turns in the history.

    The MCP round trips run on Anthropic's side inside the one call; `.text`
    keeps only the answer's text blocks.
    """
    question_turn = {"role": "user", "content": question}
    messages = [("system", CHAT_SYSTEM_PROMPT)]
    for turn in [*history, question_turn]:
        messages.append((turn["role"], turn["content"]))

    try:
        with st.spinner("Reading your SnapTrade accounts..."):
            reply = portfolio_chat_model().invoke(messages)
    except Exception as error:
        # An expired token lands here (Risk #15): mint a new one (R6).
        show_live_error("The portfolio chat", error)
        return

    history.append(question_turn)
    history.append({"role": "assistant", "content": str(reply.text)})
    st.rerun()


def chat_tab() -> None:
    """Answer plain-English portfolio questions through SnapTrade's hosted MCP, live only (S13, stories 41-42)."""
    if not is_chat_configured():
        st.info(CHAT_SETUP_NOTE)
        return

    history = st.session_state.setdefault("chat_history", [])
    render_chat_turns(history)
    question = st.chat_input(CHAT_PLACEHOLDER)
    if question:
        ask_portfolio_chat(history, question)


def main() -> None:
    """Render the gated app; tabs follow the Q12 demo narrative."""
    st.set_page_config(page_title="snapi", layout="wide")
    load_dotenv()  # keys from .env
    check_password()
    prepare_live_books()

    tab_labels = ["Committee", "Portfolio", "Activity", "Chat", "Forward test"]
    committee_area, portfolio_area, activity_area, chat_area, forward_test_area = (
        st.tabs(tab_labels)
    )
    with committee_area:
        committee_tab()
    with portfolio_area:
        portfolio_tab()
    with activity_area:
        activity_tab()
    with chat_area:
        chat_tab()
    with forward_test_area:
        forward_test_tab()


if __name__ == "__main__":
    main()
