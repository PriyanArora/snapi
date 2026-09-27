"""snapi's Streamlit app: the password gate and the dashboard tabs (PRD Streamlit app).

Heavy modules (books, committee, execute, brokerage and everything they pull
in: langgraph, langchain, snaptrade, external/) are imported inside the live
functions that need them, not at the top. The PRD asks for lazy imports so
the public replay never loads them (S9); that overrides the usual
imports-at-the-top rule. Renderers take plain JSON-shaped dicts, so a run
logged to `data/run_log.jsonl` and a run made in this session draw the same
way.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st
from dotenv import load_dotenv

import config

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator

    from committee import Proposal, Snapshot

logger = logging.getLogger(__name__)

APP_PASSWORD_SECRET = "APP_PASSWORD"
ANTHROPIC_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
SNAPTRADE_SETUP_NOTE = (
    "SnapTrade isn't configured. Set SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID "
    "and SNAPTRADE_ALPACA_PAPER_ACCOUNT_ID in .env."
)

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


def is_cloud_mode() -> bool:
    """Return True without an Anthropic key: the public replay, or a laptop before its keys exist (S9)."""
    return not os.environ.get(ANTHROPIC_KEY_ENV_VAR)


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

    if read_app_password() is None:
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
        display_name = config.PERSONAS[persona_key].display_name
        lines.append(f"- {display_name}: {', '.join(tickers) or 'none'}")

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

    display_name = config.PERSONAS[persona_key].display_name
    try:
        blocker = find_trade_blocker(persona_key)
        if blocker is not None:
            return f"{display_name} skipped: {blocker}"

        execute.execute(proposal, persona_key, tag=tag)
    except execute.ExecutionLockedError as error:
        return str(error)
    except Exception as error:
        logger.error("%s execution failed", display_name, exc_info=error)
        return f"{display_name} failed ({type(error).__name__}): {error}"

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


def render_approval_record(record: dict) -> None:
    """Show one executed batch: SnapTrade's acknowledgements for Balanced, simulator fills for the others."""
    if record["acks"]:
        st.success("SnapTrade acknowledged: queued for Monday's 09:30 ET open.")
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
        return

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


def committee_tab() -> None:
    """Run swarm fills five agent cards live; then the PM memo, three order tables and the critic beside Approve."""
    run_clicked = st.button("Run swarm", type="primary", disabled=is_cloud_mode())
    if is_cloud_mode():
        st.caption(
            f"Run swarm needs {ANTHROPIC_KEY_ENV_VAR} in .env; "
            "the live committee runs on the laptop."
        )

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


def main() -> None:
    """Render the gated app; tabs follow the Q12 demo narrative."""
    st.set_page_config(page_title="snapi", layout="wide")
    load_dotenv()  # laptop keys from .env; the Cloud app has none
    check_password()
    prepare_live_books()

    (committee_area,) = st.tabs(["Committee"])
    with committee_area:
        committee_tab()


if __name__ == "__main__":
    main()
