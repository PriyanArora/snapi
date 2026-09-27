"""snapi's Streamlit app: the login page, the navbar and its pages (PRD Streamlit app).

Heavy modules (books, committee, execute, brokerage and everything they pull
in: langgraph, langchain, snaptrade, external/) are imported inside the live
functions that need them, not at the top. The PRD asks for lazy imports to
keep start-up fast; that overrides the usual imports-at-the-top rule.
Renderers take plain JSON-shaped dicts: the run kept in the session and
the approval records read back from the run log draw the same way.

Every setting is an env var, the same on localhost and Streamlit Cloud:
`.env` locally, the app's Secrets on Cloud, which Streamlit copies into
os.environ (Hosting, T17).
"""

from __future__ import annotations

import dataclasses
import functools
import hmac
import json
import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import quote
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from dotenv import load_dotenv

import config

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from streamlit.delta_generator import DeltaGenerator

    from brokerage import AccountSnapshot
    from committee import Proposal, Snapshot
    from portfolio import Book, RebalanceResult

logger = logging.getLogger(__name__)

APP_PASSWORD_ENV_VAR = "APP_PASSWORD"

USER_FIRST_NAME = "Priyan"
TAGLINE = "Turn your brokerage into your own AI hedge fund"
LOGIN_CARD_WIDTH = 400
HOME_PAGE = "Home"
# The navbar's links, in the Q12 demo order; the logo leads Home.
NAV_PAGES = ("Committee", "Portfolio", "Chat", "Forward test", "Backtest")
RECENT_ACTIVITY_ROWS = 5
# Fits the five links; a "content" width measures the shadcn root before
# its buttons render, so the links spilled out of the navbar.
NAV_LINKS_WIDTH = 540
# SnapTrade codes whose sentence-case form reads badly.
ACTIVITY_TYPE_LABELS = {"REI": "Dividend reinvestment"}
LOGO_FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@1,800"
    "&display=swap"
)
# Background grain. The SVG is URL-encoded because Streamlit's HTML sanitizer
# drops a <style> block that holds a raw <svg> tag.
GRAIN_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' width='240' height='240'>"
    "<filter id='g'><feTurbulence type='fractalNoise' baseFrequency='.8' "
    "numOctaves='2' stitchTiles='stitch'/>"
    "<feColorMatrix values='0 0 0 0 .25  0 0 0 0 .4  0 0 0 0 .45  1.5 0 0 0 -1'/>"
    "</filter><rect width='100%' height='100%' filter='url(#g)'/></svg>"
)
# Every page: soft grain over blue-teal glows on the off-white theme, and the
# logo font. The page font comes from .streamlit/config.toml; shadcn sets its
# own font on its component root, so the host inherits the page font instead.
SITE_CSS = f"""
<style>
@import url('{LOGO_FONT_URL}');
.stApp {{
  background-color: #F4F6F8;
  background-image:
    url("data:image/svg+xml,{quote(GRAIN_SVG)}"),
    radial-gradient(75% 70% at 0% 0%, rgba(159, 193, 193, 1), transparent 72%),
    radial-gradient(70% 65% at 100% 100%, rgba(159, 193, 193, 0.95), transparent 72%),
    radial-gradient(45% 45% at 100% 0%, rgba(130, 165, 195, 0.5), transparent 70%);
  background-size: 240px 240px, 100% 100%, 100% 100%, 100% 100%;
  background-attachment: fixed;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-ssui-v2-host] {{ font-family: inherit; }}
.snapi-logo {{
  font-family: 'Playfair Display', serif;
  font-style: italic;
  font-weight: 800;
  font-size: 5rem;
  line-height: 1;
  letter-spacing: -0.02em;
  text-align: center;
  color: #0F1B3D;
}}
</style>
"""
LOGIN_PAGE_CSS = (
    '<style>[data-testid="stMainBlockContainer"] { padding-top: 14vh; }</style>'
)
# After login: a glassy navbar that stays on top, the logo as a shadcn ghost
# button themed through its own tokens, and the Home page's big type.
APP_CSS = """
<style>
header[data-testid="stHeader"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 1rem; }
[data-testid="stLayoutWrapper"]:has(> .st-key-navbar) {
  position: sticky;
  top: 0.75rem;
  z-index: 999;
}
.st-key-navbar {
  padding: 0.7rem 1.1rem;
  background: rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(16px) saturate(160%);
  -webkit-backdrop-filter: blur(16px) saturate(160%);
  border: 1px solid rgba(15, 27, 61, 0.08);
  border-radius: 16px;
  box-shadow: 0 8px 24px rgba(15, 27, 61, 0.06);
}
/* Ghost buttons hover with --accent: a light yellow in the navbar. */
.st-key-navbar [data-ssui-v2-host] {
  --accent: #FFF1B8;
  --accent-foreground: #0F1B3D;
  --text-sm: 0.95rem;
}
.st-key-nav_logo [data-ssui-v2-host] {
  font-family: 'Playfair Display', serif;
  font-style: italic;
  --text-sm: 1.5rem;
  --text-sm--line-height: 1;
  --font-weight-medium: 800;
  --spacing: 0.26rem;
}
[class*="st-key-brokerage_card_"] {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 14px;
}
.snapi-hero { margin: 4rem 0 0; }
.snapi-hero h1 {
  margin: 0;
  padding: 0;
  font-size: clamp(3rem, 6.5vw, 6rem);
  font-weight: 800;
  line-height: 1.02;
  letter-spacing: -0.035em;
  color: #0F1B3D;
}
.snapi-hero h1 span {
  display: block;
  margin-top: 0.5rem;
  font-size: clamp(2.2rem, 4.2vw, 3.75rem);
  font-weight: 700;
  line-height: 1.08;
  letter-spacing: -0.03em;
  color: #56637F;
}
.snapi-total { margin: 2.5rem 0 2rem; }
.snapi-total p { margin: 0; color: #56637F; font-weight: 600; }
.snapi-total strong {
  display: block;
  font-size: clamp(2.5rem, 5vw, 4.25rem);
  font-weight: 800;
  line-height: 1.05;
  letter-spacing: -0.03em;
  color: #0F1B3D;
}
[class*="st-key-portfolio_card_"] {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-radius: 14px;
}
</style>
"""

ANTHROPIC_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
# The macro regime behind seeding, Run swarm, Rebalance and Reset books.
FRED_KEY_ENV_VAR = "FRED_API_KEY"
SNAPTRADE_SETUP_NOTE = (
    "SnapTrade isn't configured. Set SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID "
    "and SNAPTRADE_ALPACA_PAPER_ACCOUNT_ID in .env."
)
SANDBOX_SETUP_NOTE = (
    "Not configured. Set SNAPTRADE_CONSUMER_KEY, SNAPTRADE_CLIENT_ID and "
    "SNAPTRADE_SANDBOX_ACCOUNT_ID in .env to show the Sandbox's demo holdings."
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
UNIVERSE_CHANGED_NOTE = (
    "The stock universe changed. Press Reset books on the Portfolio page to "
    "re-seed the books."
)
BACKTEST_SETUP_NOTE = (
    "Run `uv run python backtest.py` (needs FRED_API_KEY) to generate the backtest."
)

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


def display_name(persona_key: str) -> str:
    """Return the persona's display name, for selector labels and notes."""
    return config.PERSONAS[persona_key].display_name


def missing_keys(*env_var_names: str) -> list[str]:
    """Return the named env vars that are unset or empty, for the plain "needs <KEY>" notes."""
    return [name for name in env_var_names if not os.environ.get(name)]


def show_live_error(action: str, error: Exception) -> None:
    """Explain a failed live call on the page instead of a traceback; the traceback goes to the terminal."""
    logger.error("%s failed", action, exc_info=error)
    st.error(f"{action} failed ({type(error).__name__}): {error}")


def escape_markdown_math(text: str) -> str:
    """Return LLM text with `$` escaped, so Streamlit never renders dollar amounts as LaTeX."""
    return text.replace("$", r"\$")


def read_app_password() -> str | None:
    """Return APP_PASSWORD from the environment, else from Streamlit secrets, else None (T17)."""
    env_password = os.environ.get(APP_PASSWORD_ENV_VAR)
    if env_password:
        return env_password

    try:
        return str(st.secrets[APP_PASSWORD_ENV_VAR])
    except (FileNotFoundError, KeyError):
        # No secrets file raises a FileNotFoundError subclass.
        return None


def is_password_correct(entered_password: str) -> bool:
    """Return whether the typed password matches APP_PASSWORD, compared in constant time."""
    expected_password = read_app_password() or ""

    # Bytes, because compare_digest rejects non-ASCII str.
    return hmac.compare_digest(
        entered_password.encode(),
        expected_password.encode(),
    )


def render_logo() -> None:
    """Draw the Snapi wordmark in its serif italic logo font."""
    st.html('<div class="snapi-logo">Snapi</div>')


def render_login_card() -> tuple[str, bool]:
    """Draw the shadcn tagline and password card; return the typed password and whether it was submitted."""
    is_password_shown = st.session_state.get("show_password", False)
    with ui.elements(key="login", width=LOGIN_CARD_WIDTH) as login:
        with login.stack(gap="lg"):
            with login.stack(align="center"):
                login.text(TAGLINE, variant="muted")
            with login.card():
                with login.card_content():
                    with login.stack(direction="horizontal", align="end", gap="sm"):
                        password_field = login.input(
                            "Password",
                            key="password",
                            type="text" if is_password_shown else "password",
                            placeholder="Enter the app password",
                        )
                        show_button = login.button(
                            "Hide" if is_password_shown else "Show",
                            key="show_password",
                            variant="outline",
                        )
                with login.card_footer():
                    with login.stack(gap="md"):
                        continue_button = login.button(
                            "Continue",
                            key="continue",
                            stretch=True,
                        )
                        login.text(config.DISCLAIMER, variant="caption")

    entered_password = password_field.value or ""
    if show_button.clicked:
        st.session_state["show_password"] = not is_password_shown
        st.rerun()

    # Only Continue logs in: Enter and Show/Hide also commit the field.
    return entered_password, continue_button.clicked


def check_password() -> None:
    """Show the login page and stop the script until the password matches APP_PASSWORD (S9, T17, Q13)."""
    if st.session_state.get("password_ok"):
        return

    st.html(LOGIN_PAGE_CSS)
    with st.container(horizontal_alignment="center"):
        render_logo()
        if not read_app_password():
            ui.alert(
                f"No {APP_PASSWORD_ENV_VAR} found.",
                "Set it in .env (or the app's Secrets on Streamlit Cloud) and restart the app.",
                variant="destructive",
                width=LOGIN_CARD_WIDTH,
            )
            st.stop()

        entered_password, is_submitted = render_login_card()
        if not is_submitted:
            st.stop()

        if is_password_correct(entered_password):
            st.session_state["password_ok"] = True
            st.rerun()

        ui.alert(
            "Wrong password.",
            "Check the password and try again.",
            variant="destructive",
            width=LOGIN_CARD_WIDTH,
        )
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


def seed_opening_books() -> None:
    """Seed Conservative and Aggressive once, while their fill logs are empty (S16); the seed's regime read needs FRED."""
    import books
    import execute

    try:
        is_seeded = all(
            books.has_fills(persona_key) for persona_key in books.SEED_PERSONAS
        )
    except Exception as error:
        # An unreachable MongoDB must not take down the tabs that don't need it.
        show_live_error("Reading the fill logs", error)
        return

    if is_seeded:
        return

    if missing_keys(FRED_KEY_ENV_VAR):
        st.info(f"Seeding the opening books needs {FRED_KEY_ENV_VAR} in .env.")
        return

    try:
        execute.seed_books()
    except Exception as error:
        show_live_error("Seeding the opening books", error)


def warn_if_universe_changed() -> None:
    """Warn once when a simulator book still holds tickers the stock universe no longer covers."""
    import books
    import portfolio

    for persona_key, persona in config.PERSONAS.items():
        # Reset books re-seeds only the simulator books.
        if persona.account != "simulator":
            continue

        try:
            book = books.load_book(persona_key)
        except Exception as error:
            show_live_error(f"Reading {persona.display_name}'s book", error)
            return

        if portfolio.tickers_outside_universe(book):
            st.warning(UNIVERSE_CHANGED_NOTE)
            return


def prepare_live_books() -> None:
    """Seed the simulator personas' opening books once (S16), flag stale books and check SnapTrade can trade (Balanced step 2)."""
    seed_opening_books()
    warn_if_universe_changed()

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


def order_time_note(order_count: int) -> str:
    """Say how far apart SnapTrade orders go out and about how long `order_count` of them take (Risk #8)."""
    import brokerage

    spacing = brokerage.ORDER_SPACING_SECONDS
    total_minutes = max(1, round(order_count * spacing / 60))
    return (
        f"spaced {spacing:g} s apart for SnapTrade's 10 requests/min limit, "
        f"about {total_minutes} min in total"
    )


def caption_snaptrade_order_time(orders_by_persona: dict[str, list]) -> None:
    """Estimate, beside an Approve button, how long the SnapTrade persona's spaced-out orders take."""
    import brokerage

    if not brokerage.is_alpaca_paper_configured():
        return

    for persona_key, orders in orders_by_persona.items():
        is_snaptrade = config.PERSONAS[persona_key].account == "alpaca_paper"
        if is_snaptrade and orders:
            st.caption(
                f"{display_name(persona_key)}'s {len(orders)} SnapTrade orders go out "
                f"{order_time_note(len(orders))} (up to a minute more right after "
                "other SnapTrade reads)."
            )


def show_order_progress(
    progress_slot: DeltaGenerator,
    persona_name: str,
    order_number: int,
    order_count: int,
) -> None:
    """Show which SnapTrade order is waiting its turn while the batch goes out."""
    progress_slot.progress(
        (order_number - 1) / order_count,
        text=(
            f"Placing {persona_name} order {order_number} of {order_count} "
            f"({order_time_note(order_count)})"
        ),
    )


def execute_for_persona(proposal: Proposal, persona_key: str, tag: str) -> str | None:
    """Execute one persona's leg of an approved proposal; return why it didn't run, or None when it did."""
    import execute

    persona_name = display_name(persona_key)
    # Only SnapTrade orders report progress; the simulator fills at once.
    progress_slot = st.empty()
    report_progress = functools.partial(
        show_order_progress, progress_slot, persona_name
    )
    try:
        blocker = find_trade_blocker(persona_key)
        if blocker is not None:
            return f"{persona_name} skipped: {blocker}"

        execute.execute(proposal, persona_key, tag=tag, report_progress=report_progress)
    except execute.ExecutionLockedError as error:
        return str(error)
    except Exception as error:
        logger.error("%s execution failed", persona_name, exc_info=error)
        return f"{persona_name} failed ({type(error).__name__}): {error}"
    finally:
        progress_slot.empty()

    return None


def execute_personas(proposal: Proposal, persona_keys: list[str], tag: str) -> dict:
    """Execute each persona's leg and return the skip notes plus the run-log records it wrote."""
    import books

    first_new_record = len(books.read_run_log())

    notes = {}
    for persona_key in persona_keys:
        note = execute_for_persona(proposal, persona_key, tag)
        if note is not None:
            notes[persona_key] = note

    new_records = books.read_run_log()[first_new_record:]
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
    approve_clicked = st.button("Approve", type="primary", key="approve")
    caption_snaptrade_order_time(run_record["proposal"]["orders"])
    if not approve_clicked:
        # Not on the click: execute re-reads the lock fresh, and a read here
        # would spend two of SnapTrade's 10 requests a minute (Risk #8).
        caption_queued_lock()
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
    missing_run_keys = missing_keys(ANTHROPIC_KEY_ENV_VAR, FRED_KEY_ENV_VAR)
    run_clicked = st.button(
        "Run swarm",
        type="primary",
        disabled=bool(missing_run_keys),
    )
    if missing_run_keys:
        st.caption(f"Run swarm needs {' and '.join(missing_run_keys)} in .env.")

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


CHART_FONT = "Plus Jakarta Sans, sans-serif"
CHART_TEXT_COLOR = "#0F1B3D"
CHART_MUTED_COLOR = "#56637F"
CHART_GRID_COLOR = "rgba(15, 27, 61, 0.08)"
# Blue, teal, amber, violet in a fixed order: brand-leaning hues with enough
# chroma to stay apart for colorblind readers on the off-white page.
CHART_SERIES_COLORS = ["#2F5DA8", "#1E9A96", "#D9822B", "#8A5CC2"]
ACCOUNT_LABELS = {
    "alpaca_paper": "trades on Alpaca Paper through SnapTrade",
    "simulator": "trades on the simulator",
}
REBALANCE_NOTE = (
    "Trims any sector over its cap pro-rata and buys ETFs for the most "
    "underweight sectors, with no AI calls. Nothing trades until you approve "
    "the preview."
)
SANDBOX_NOTE = (
    "Pre-filled demo holdings from SnapTrade's read-only Sandbox. "
    "They are not part of any persona's book."
)
HOLDINGS_COLUMNS = [
    ui.TableColumn(key="ticker", label="Ticker"),
    ui.TableColumn(key="sector", label="Sector"),
    ui.TableColumn(key="shares", label="Shares", align="right"),
    ui.TableColumn(key="price", label="Price", align="right"),
    ui.TableColumn(key="value", label="Value", align="right"),
    ui.TableColumn(key="weight", label="Weight", align="right"),
    ui.TableColumn(key="avg_cost", label="Avg cost", align="right"),
]
SANDBOX_COLUMNS = [
    ui.TableColumn(key="ticker", label="Ticker"),
    ui.TableColumn(key="units", label="Units", align="right"),
    ui.TableColumn(key="price", label="Price", align="right"),
    ui.TableColumn(key="cost_basis", label="Cost basis", align="right"),
]


def style_chart(figure: go.Figure, height: int) -> go.Figure:
    """Return the figure in the site's look: see-through background, page font, navy text and the series colors."""
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": CHART_FONT, "color": CHART_TEXT_COLOR},
        colorway=CHART_SERIES_COLORS,
        margin={"l": 8, "r": 8, "t": 36, "b": 8},
        legend={"orientation": "h", "x": 0, "y": 1.02, "yanchor": "bottom"},
    )
    figure.update_xaxes(showgrid=False, linecolor=CHART_GRID_COLOR)
    figure.update_yaxes(
        gridcolor=CHART_GRID_COLOR,
        zeroline=False,
        title_font={"color": CHART_MUTED_COLOR},
    )
    return figure


def render_section_heading(key: str, title: str, description: str) -> None:
    """Draw a shadcn heading with one muted line under it, the way Home heads its sections."""
    with ui.elements(key=key) as heading:
        heading.heading(title, level=4)
        heading.text(description, variant="muted")


def persona_limits_line(persona_key: str) -> str:
    """Return the persona's limits and trading venue as one short line."""
    persona = config.PERSONAS[persona_key]
    return (
        f"Sector cap {persona.sector_cap:.0%} · max {persona.max_position_pct:.0%} "
        f"per stock · {persona.cash_buffer:.0%} cash buffer · "
        f"{ACCOUNT_LABELS[persona.account]}"
    )


def holdings_rows(book: Book, marks: dict[str, float], equity: float) -> list[dict]:
    """Return one table row per held ticker: sector, shares, mark, value, weight and average cost."""
    rows = []
    for ticker, shares in sorted(book.shares.items()):
        value = shares * marks[ticker]
        rows.append(
            {
                "ticker": ticker,
                "sector": config.SECTOR_BY_TICKER.get(ticker, "none"),
                "shares": f"{shares:,}",
                "price": format_money(marks[ticker]),
                "value": format_money(value),
                "weight": f"{value / equity:.1%}",
                "avg_cost": format_money(book.avg_cost.get(ticker)),
            }
        )

    return rows


def sector_chart(
    weights_by_series: dict[str, dict[str, float]],
    caps_key: str,
) -> go.Figure:
    """Return grouped sideways bars of sector weights (%) with a dashed line at the chosen persona's sector cap (S10)."""
    caps_persona = config.PERSONAS[caps_key]

    # Sideways bars keep the sector names level in a half-width card.
    figure = go.Figure()
    for series_name, weights in weights_by_series.items():
        percents = [100 * weights[sector] for sector in config.SECTORS]
        figure.add_bar(
            x=percents,
            y=list(config.SECTORS),
            name=series_name,
            orientation="h",
        )

    figure.add_vline(
        x=100 * caps_persona.sector_cap,
        line_dash="dash",
        line_color=CHART_TEXT_COLOR,
        annotation_text=f"{caps_persona.display_name} cap {caps_persona.sector_cap:.0%}",
        annotation_position="top",
        annotation_font_color=CHART_MUTED_COLOR,
    )
    figure.update_layout(
        barmode="group",
        xaxis_title="Weight (%)",
        showlegend=len(weights_by_series) > 1,
    )
    style_chart(figure, height=380)
    figure.update_xaxes(showgrid=True, gridcolor=CHART_GRID_COLOR)
    figure.update_yaxes(showgrid=False, autorange="reversed")
    return figure


def render_book_metrics(persona_key: str, book: Book, equity: float) -> None:
    """Show the book's equity, cash, invested share and position count as shadcn metric cards."""
    persona = config.PERSONAS[persona_key]
    invested_pct = (equity - book.cash) / equity
    held_count = len(book.shares)

    equity_column, cash_column, invested_column, holdings_column = st.columns(4)
    with equity_column:
        ui.metric_card(
            "Equity",
            format_money(equity),
            description="Cash plus holdings at the latest close",
            key=f"equity_{persona_key}",
        )
    with cash_column:
        ui.metric_card(
            "Cash",
            format_money(book.cash),
            description=f"{book.cash / equity:.1%} of equity",
            key=f"cash_{persona_key}",
        )
    with invested_column:
        ui.metric_card(
            "Invested",
            f"{invested_pct:.1%}",
            description=f"Sized to at most {persona.max_invested_pct:.0%}",
            key=f"invested_{persona_key}",
        )
    with holdings_column:
        ui.metric_card(
            "Holdings",
            held_count,
            description="Stocks and sector ETFs",
            key=f"held_{persona_key}",
        )


def render_book(persona_key: str, book: Book, marks: dict[str, float]) -> None:
    """Show one persona's key numbers, sector weights against its own cap and holdings (S10, story 38)."""
    import portfolio

    persona = config.PERSONAS[persona_key]
    equity = portfolio.compute_book_equity(book, marks)
    render_book_metrics(persona_key, book, equity)

    if not book.shares:
        ui.alert(
            "All cash: no positions yet.",
            "Holdings and sector weights show here once orders fill.",
            key=f"all_cash_{persona_key}",
        )
        return

    chart_column, holdings_column = st.columns([5, 7])
    with chart_column:
        with st.container(border=True, key=f"portfolio_card_sectors_{persona_key}"):
            render_section_heading(
                f"sectors_head_{persona_key}",
                "Sector weights",
                f"Each bar is a sector's share of equity; the dashed line is "
                f"the {persona.sector_cap:.0%} cap.",
            )
            figure = sector_chart(
                {"Now": portfolio.sector_weights(book, marks)}, persona_key
            )
            st.plotly_chart(figure, key=f"sectors_{persona_key}")
    with holdings_column:
        with st.container(border=True, key=f"portfolio_card_holdings_{persona_key}"):
            render_section_heading(
                f"holdings_head_{persona_key}",
                "Holdings",
                "Priced at the latest close; weight is the share of equity.",
            )
            ui.table(
                holdings_rows(book, marks, equity),
                HOLDINGS_COLUMNS,
                key=f"holdings_{persona_key}",
            )


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

    approve_clicked = ui.button(
        "Approve rebalance", key=f"approve_rebalance_{persona_key}"
    )
    caption_snaptrade_order_time({persona_key: result.orders})
    if approve_clicked:
        approve_rebalance(persona_key, result)


def rebalance_controls(persona_key: str, book: Book, marks: dict[str, float]) -> None:
    """Rebalance on demand, optionally against another persona's caps; the preview lives in the session (S8, R3)."""
    persona_keys = list(config.PERSONAS)
    preview_key = f"rebalance_preview_{persona_key}"
    outcome_key = f"rebalance_outcome_{persona_key}"
    is_fred_missing = bool(missing_keys(FRED_KEY_ENV_VAR))

    with st.container(horizontal=True, vertical_alignment="bottom"):
        picked_caps_key = ui.select(
            "Check against caps of",
            persona_keys,
            index=persona_keys.index(persona_key),
            format_func=display_name,
            key=f"caps_{persona_key}",
            width=280,
        )
        # Kept in session_state: a button under another button's `if` never
        # fires, because the outer button is False again on that rerun.
        rebalance_clicked = ui.button(
            "Rebalance",
            key=f"rebalance_{persona_key}",
            disabled=is_fred_missing,
        )
    if is_fred_missing:
        st.caption(f"Rebalance needs {FRED_KEY_ENV_VAR} in .env.")

    # R3's selector defaults to the persona's own caps.
    caps_key = picked_caps_key or persona_key
    if rebalance_clicked:
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


def render_persona(persona_key: str, marks: dict[str, float]) -> None:
    """Show one persona's book and its Rebalance card, or say why its book can't be shown."""
    import books
    import portfolio

    persona = config.PERSONAS[persona_key]
    with ui.elements(key=f"persona_head_{persona_key}") as heading:
        heading.heading(persona.display_name, level=3)
        heading.text(persona_limits_line(persona_key), variant="muted")

    try:
        book = books.current_book(persona_key)
    except Exception as error:
        show_live_error(f"Reading {persona.display_name}'s book", error)
        return

    # Tickers from an older universe have no mark to value them by.
    stale_tickers = portfolio.tickers_outside_universe(book)
    if stale_tickers:
        st.caption(
            f"Holds {', '.join(stale_tickers)} from the old stock universe; "
            "Reset books below re-seeds it."
        )
        return

    render_book(persona_key, book, marks)
    with st.container(border=True, key=f"portfolio_card_rebalance_{persona_key}"):
        render_section_heading(
            f"rebalance_head_{persona_key}", "Rebalance", REBALANCE_NOTE
        )
        rebalance_controls(persona_key, book, marks)


def render_sandbox(rows: list[dict]) -> None:
    """Show the SnapTrade Sandbox's demo holdings on their own; they never join a persona's book (S16, story 59)."""
    if not rows:
        st.caption("The Sandbox account holds nothing.")
        return

    table_rows = []
    for row in rows:
        table_rows.append(
            {
                "ticker": row["ticker"],
                "units": format_quantity(row["units"]),
                "price": format_money(row["price"]),
                "cost_basis": format_money(row["cost_basis"]),
            }
        )
    ui.table(table_rows, SANDBOX_COLUMNS, key="sandbox_holdings")


def sandbox_panel() -> None:
    """Read the Sandbox holdings through the shared 60 s SnapTrade cache and show them, or say why not."""
    import brokerage

    render_section_heading("sandbox_head", "SnapTrade Sandbox", SANDBOX_NOTE)
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
    import books
    import execute

    seeded_names = " and ".join(display_name(key) for key in books.SEED_PERSONAS)
    render_section_heading(
        "reset_head",
        "Reset books",
        f"Re-seeds the {seeded_names} books at Friday's close, for a clean rehearsal.",
    )
    if st.session_state.pop("books_reset", False):
        ui.alert("Simulator books cleared and re-seeded.", key="books_reset_done")

    is_fred_missing = bool(missing_keys(FRED_KEY_ENV_VAR))
    if is_fred_missing:
        st.caption(f"Reset books needs {FRED_KEY_ENV_VAR} in .env to re-seed.")

    confirmed = ui.checkbox(
        "Clear the simulator fill logs and re-seed the opening books",
        key="confirm_reset",
    )
    reset_clicked = ui.button(
        "Reset books",
        variant="destructive",
        disabled=is_fred_missing or not confirmed,
        key="reset_books",
    )
    if not reset_clicked:
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
    """One persona at a time: its book, sector chart and Rebalance; then the Sandbox and Reset books."""
    import prices

    persona_key = ui.tabs(
        list(config.PERSONAS),
        format_func=display_name,
        key="portfolio_persona",
        width="content",
    )
    render_persona(persona_key, prices.latest_marks())

    with st.container(border=True, key="portfolio_card_sandbox"):
        sandbox_panel()
    with st.container(border=True, key="portfolio_card_reset"):
        reset_books_control()


def normalized(values: list[float]) -> list[float]:
    """Return the series rescaled so its first value is 100."""
    return [100 * value / values[0] for value in values]


def forward_test_tab() -> None:
    """Plot NAV per persona and SPY from the run log, normalized to the first logged run (S14, F23, stories 44-45)."""
    import books

    try:
        run_records = books.read_run_log()
    except Exception as error:
        show_live_error("Reading the run log", error)
        return

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


def backtest_metric_rows(metrics_by_curve: dict[str, dict[str, float]]) -> list[dict]:
    """Return one formatted table row per backtest curve (story 48)."""
    rows = []
    for curve_name, metrics in metrics_by_curve.items():
        rows.append(
            {
                "Curve": curve_name,
                "Total return": f"{metrics['total_return']:.1%}",
                "Annualized return": f"{metrics['annualized_return']:.1%}",
                "Sharpe": f"{metrics['sharpe']:.2f}",
                "Max drawdown": f"{metrics['max_drawdown']:.1%}",
                "Excess vs SPY": f"{metrics['excess_return_vs_spy']:+.1%}",
            }
        )

    return rows


def backtest_tab() -> None:
    """Plot the P1 backtest's four curves and metrics from the committed results file (S15, stories 46-48)."""
    if not config.BACKTEST_RESULTS_PATH.exists():
        st.info(BACKTEST_SETUP_NOTE)
        return

    results = json.loads(config.BACKTEST_RESULTS_PATH.read_text())
    max_name_weight = config.PERSONAS["balanced"].max_position_pct
    st.caption(
        "Deterministic, no AI: every Monday the names with a positive technical "
        "score at the previous close are held at equal weight, at most "
        f"{max_name_weight:.0%} each (Balanced's per-name max), with the rest in "
        "cash, less 5 bps of turnover. The overlay invests 100/75/50% of that by "
        "the macro regime known before each week, from point-in-time FRED data. "
        "The fundamentals tilt is left out: there is no free point-in-time "
        "forward P/E history."
    )
    # R11 / Risk #12: say plainly the window was benign, with the real count.
    st.caption(
        f"A benign year: SPY closed below its 200-day average on only "
        f"{results['spy_days_below_sma200']} of {results['window_days']} trading "
        "days. In a calm year the macro overlay has little to react to, so it can "
        "look like staying fully invested. FRED's high-yield spread history only "
        "starts in 2023, so there is no 2020 or 2022 stress test."
    )

    figure = go.Figure()
    for curve_name, growth in results["growth"].items():
        line_style = None
        if curve_name == config.BENCHMARK:
            line_style = {"color": "gray", "dash": "dash"}

        figure.add_scatter(
            x=results["dates"],
            y=normalized(growth),
            name=curve_name,
            mode="lines",
            line=line_style,
        )
    figure.update_layout(
        yaxis_title="Value (start = 100)",
        hovermode="x unified",
        height=420,
    )
    st.plotly_chart(figure, key="backtest")
    st.dataframe(backtest_metric_rows(results["metrics"]), hide_index=True)


def format_long_date(timestamp: str | None) -> str:
    """Return an ISO date or timestamp as e.g. "September 29, 2026", or "" when missing."""
    if not timestamp:
        return ""

    moment = datetime.fromisoformat(timestamp)
    return f"{moment:%B} {moment.day}, {moment.year}"


def format_money(amount: float | None) -> str:
    """Return a dollar amount as e.g. "$1,234.56" or "-$12.00", or "" when missing."""
    if amount is None:
        return ""

    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def format_label(value: str | None) -> str:
    """Return SnapTrade's upper-case codes as sentence case, e.g. PARTIAL_CANCELED -> Partial canceled."""
    return (value or "").replace("_", " ").capitalize()


def format_quantity(quantity: float | None) -> str:
    """Return a share count without a trailing .0, or "" when missing."""
    if quantity is None:
        return ""

    return f"{quantity:g}"


def account_value(snapshot: AccountSnapshot) -> float:
    """Return cash plus every holding at SnapTrade's last price; a holding without a price adds nothing."""
    holdings_value = 0.0
    for holding in snapshot.holdings:
        if holding.price is not None:
            holdings_value += holding.units * holding.price

    return snapshot.cash + holdings_value


ORDER_COLUMNS = [
    ui.TableColumn(key="date", label="Date"),
    ui.TableColumn(key="action", label="Action"),
    ui.TableColumn(key="ticker", label="Ticker"),
    ui.TableColumn(key="quantity", label="Shares", align="right"),
    ui.TableColumn(key="price", label="Price", align="right"),
    ui.TableColumn(key="status", label="Status"),
]
ACTIVITY_COLUMNS = [
    ui.TableColumn(key="date", label="Date"),
    ui.TableColumn(key="type", label="Type"),
    ui.TableColumn(key="ticker", label="Ticker"),
    ui.TableColumn(key="units", label="Units", align="right"),
    ui.TableColumn(key="amount", label="Amount", align="right"),
]


def order_table_row(order: dict) -> dict[str, str]:
    """Turn one SnapTrade order row into the Home table's display row."""
    return {
        "date": format_long_date(order["time_placed"]),
        "action": format_label(order["action"]),
        "ticker": order["ticker"] or "",
        "quantity": format_quantity(order["total_quantity"]),
        "price": format_money(order["execution_price"]),
        "status": format_label(order["status"]),
    }


def activity_table_row(activity: dict) -> dict[str, str]:
    """Turn one SnapTrade activity row into the Home table's display row."""
    return {
        "date": format_long_date(activity["date"]),
        "type": ACTIVITY_TYPE_LABELS.get(
            activity["type"], format_label(activity["type"])
        ),
        "ticker": activity["ticker"] or "",
        "units": format_quantity(activity["units"]),
        "amount": format_money(activity["amount"]),
    }


def read_alpaca_orders(account_id: str) -> list[dict[str, str]]:
    """Return Alpaca Paper's orders from the last 90 days as display rows, newest first.

    The 90-day list refreshes about daily, so the realtime last-24h read
    adds today's orders and wins on status.
    """
    import brokerage

    orders_by_id = {}
    recent_orders = brokerage.get_recent_orders(account_id)
    for order in [*brokerage.get_order_history(account_id), *recent_orders]:
        orders_by_id[order["brokerage_order_id"]] = order

    newest_first = sorted(
        orders_by_id.values(),
        key=lambda order: order["time_placed"] or "",
        reverse=True,
    )
    return [order_table_row(order) for order in newest_first]


def read_sandbox_activity(account_id: str) -> list[dict[str, str]]:
    """Return the Sandbox's activity (trades, dividends, splits, transfers) as display rows, newest first."""
    import brokerage

    # SnapTrade already returns activities newest first.
    activities = brokerage.get_activities(account_id)
    return [activity_table_row(activity) for activity in activities]


def configured_brokerages() -> list[tuple[str, str]]:
    """Return (kind, account id) for each connected account set in .env: "alpaca" and "sandbox"."""
    import brokerage

    brokerages = []
    if brokerage.is_alpaca_paper_configured():
        brokerages.append(("alpaca", brokerage.alpaca_paper_account_id()))
    if brokerage.is_sandbox_configured():
        brokerages.append(("sandbox", brokerage.sandbox_account_id()))
    return brokerages


def show_full_activity(brokerage_name: str, rows: list[dict], columns: list) -> None:
    """Open a popup with the brokerage's whole activity table."""

    def render_table() -> None:
        ui.table(rows, columns, key="full_activity", max_height=520)

    st.dialog(f"{brokerage_name} activity", width="large")(render_table)()


def render_brokerage_card(
    kind: str,
    account_id: str,
    connection: dict,
    value: float,
) -> None:
    """Draw one connected brokerage: logo, name, access, balance, its 5 latest rows and View all."""
    name = connection.get("name") or kind.title()
    access = "Trade" if connection.get("access") == "trade" else "Read-only"
    with st.container(border=True, key=f"brokerage_card_{kind}"):
        with st.container(horizontal=True, vertical_alignment="center"):
            ui.avatar(
                connection.get("logo_url"),
                fallback=name[:2].upper(),
                alt=name,
                size="lg",
                key=f"logo_{kind}",
            )
            with ui.elements(key=f"brokerage_head_{kind}") as head:
                with head.stack(
                    direction="horizontal", justify="between", align="center"
                ):
                    with head.stack(gap="xs"):
                        head.heading(name, level=4)
                        head.badge(access, variant="secondary")
                    head.heading(format_money(value), level=3)

        try:
            if kind == "alpaca":
                rows = read_alpaca_orders(account_id)
                columns = ORDER_COLUMNS
            else:
                rows = read_sandbox_activity(account_id)
                columns = ACTIVITY_COLUMNS
        except Exception as error:
            show_live_error(f"Reading {name} activity", error)
            return

        if not rows:
            ui.alert("No activity yet.", "Trades show here once they're placed.")
            return

        ui.table(rows[:RECENT_ACTIVITY_ROWS], columns, key=f"recent_{kind}")
        if ui.button("View all", key=f"view_all_{kind}", variant="outline"):
            show_full_activity(name, rows, columns)


def home_page() -> None:
    """Greet the user, then show the live total and each connected brokerage's balance and activity (S12)."""
    import brokerage

    st.html(
        f'<div class="snapi-hero"><h1>Hello {USER_FIRST_NAME},'
        "<span>Here is your overall portfolio across all brokerage accounts.</span>"
        "</h1></div>"
    )
    brokerages = configured_brokerages()
    if not brokerages:
        st.info(SNAPTRADE_SETUP_NOTE)
        return

    try:
        connections = brokerage.get_connections()
        values_by_account = {}
        for _, account_id in brokerages:
            snapshot = brokerage.get_account_snapshot(account_id)
            values_by_account[account_id] = account_value(snapshot)
    except Exception as error:
        show_live_error("Reading your brokerage accounts", error)
        return

    total_value = sum(values_by_account.values())
    st.html(
        '<div class="snapi-total"><p>Total balance, live</p>'
        f"<strong>{format_money(total_value)}</strong></div>"
    )
    with ui.elements(key="connected_heading") as heading:
        heading.heading("Connected brokerages", level=3)
        heading.text(f"{len(brokerages)} connections", variant="muted")

    for column, (kind, account_id) in zip(
        st.columns(len(brokerages)), brokerages, strict=True
    ):
        with column:
            render_brokerage_card(
                kind,
                account_id,
                connections.get(account_id, {}),
                values_by_account[account_id],
            )


def render_navbar(current_page: str) -> str:
    """Draw the glassy shadcn navbar; return the page the user picked, else the current one."""
    picked_page = current_page
    with st.container(
        key="navbar",
        horizontal=True,
        horizontal_alignment="distribute",
        vertical_alignment="center",
    ):
        with st.container(key="nav_logo", width="content"):
            if ui.button("Snapi", key="nav_logo_button", variant="ghost"):
                picked_page = HOME_PAGE

        link_buttons = {}
        with ui.elements(key="nav_links", width=NAV_LINKS_WIDTH) as nav:
            with nav.stack(direction="horizontal", gap="xs"):
                for page_name in NAV_PAGES:
                    variant = "secondary" if page_name == current_page else "ghost"
                    link_buttons[page_name] = nav.button(
                        page_name,
                        key=f"nav_{page_name.lower().replace(' ', '_')}",
                        variant=variant,
                    )

    for page_name, link_button in link_buttons.items():
        if link_button.clicked:
            picked_page = page_name
    return picked_page


def is_chat_configured() -> bool:
    """Return True when the Anthropic key, the MCP URL and the hand-minted OAuth token are all set (R6)."""
    return not missing_keys(
        ANTHROPIC_KEY_ENV_VAR,
        SNAPTRADE_MCP_URL_ENV_VAR,
        SNAPTRADE_MCP_TOKEN_ENV_VAR,
    )


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
    """Render the login page, then the navbar and one page at a time (Q12 demo order)."""
    st.set_page_config(page_title="snapi", layout="wide")
    st.html(SITE_CSS)
    # Cloud's root-level Secrets reach os.environ only once st.secrets is
    # parsed. `streamlit run` parses it at start-up; this covers any other
    # start (AppTest) and is a no-op without a secrets file.
    st.secrets.load_if_toml_exists()
    load_dotenv()  # .env fills in only what isn't set already
    check_password()
    st.html(APP_CSS)

    page_renderers = {
        HOME_PAGE: home_page,
        "Committee": committee_tab,
        "Portfolio": portfolio_tab,
        "Chat": chat_tab,
        "Forward test": forward_test_tab,
        "Backtest": backtest_tab,
    }
    current_page = st.session_state.get("page", HOME_PAGE)
    picked_page = render_navbar(current_page)
    if picked_page != current_page:
        st.session_state["page"] = picked_page
        st.rerun()

    prepare_live_books()
    page_renderers[current_page]()


if __name__ == "__main__":
    main()
