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
import queue
import textwrap
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING
from urllib.parse import quote
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st
import streamlit_shadcn_ui as ui
from dotenv import load_dotenv
from pydantic_core import to_jsonable_python

import config

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable
    from streamlit.delta_generator import DeltaGenerator

    from brokerage import AccountSnapshot
    from committee import Proposal
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
[class*="st-key-brokerage_card_"],
[class*="st-key-order_card_"],
[class*="st-key-pending_node_"] {
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

COMMITTEE_INTRO = (
    "A swarm of AI agents: three analysts read the market side by side, a "
    "portfolio manager weighs them, code sizes three portfolios, a devil's "
    "advocate pushes back, and a final judge recommends one."
)
ANALYSTS = ("sentiment", "technical", "macro")
# Agent card titles, keyed by the Proposal field each card shows.
CARD_TITLES = {
    "sentiment": "Sentiment analyst",
    "technical": "Technical analyst",
    "macro": "Macro analyst",
    "pm": "Portfolio Manager",
    "critique": "Devil's advocate",
    "verdict": "Final verdict",
}
MODEL_SELECT_WIDTH = 170
APPROVE_BAR_WIDTH = 460
CARD_PREVIEW_CHARS = 220
TOP_ORDER_ROWS = 3
# The graph's state keys that the Proposal names differently.
PROPOSAL_FIELD_BY_STATE_KEY = {
    "proposals": "orders",
    "recommended": "recommended_persona",
}
# The user's rule: a live run still going after 15 s shows the latest saved
# Sonnet run until it finishes. The ceiling stops the page waiting forever.
FALLBACK_AFTER_SECONDS = 15
RUN_CEILING_SECONDS = 300
POLL_SECONDS = 0.5
# Each persona card's opening words; the numbers come from config.PERSONAS.
PERSONA_STANCES = {
    "conservative": "Tight limits",
    "balanced": "Middle ground",
    "aggressive": "Loose limits",
}
ACCOUNT_NOTES = {
    "simulator": "Trades on the simulator.",
    "alpaca_paper": "Trades on Alpaca Paper through SnapTrade.",
}
APPROVE_CHOICES = ("recommended", "balanced", "conservative", "aggressive")
# The flow's connectors live in a 600-wide viewBox stretched to the page:
# x 100, 300 and 500 are the centers of three equal columns, 300 the middle.
CONNECTOR_COLOR = "#7FA3A6"
CONNECTOR_HEIGHT = 44
COLUMN_CENTERS = (100, 300, 500)
PAGE_CENTER = 300
CONVERGE_LINES = [(column_x, PAGE_CENTER) for column_x in COLUMN_CENTERS]
BRANCH_LINES = [(PAGE_CENTER, column_x) for column_x in COLUMN_CENTERS]
DOWN_LINE = [(PAGE_CENTER, PAGE_CENTER)]


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


def analyst_lines(report: dict) -> list[str]:
    """Return an analyst's full text: an abstain note first, then one "AAPL +0.42: why" line per ticker."""
    lines = []
    if report["abstained"]:
        lines.append(report["why"].get("error", "Abstained this cycle."))

    for ticker, score in report["score"].items():
        lines.append(f"{ticker} {score:+.2f}: {report['why'].get(ticker, '')}")

    return lines


def macro_lines(report: dict) -> list[str]:
    """Return the regime and each derived flag, raised or clear, with its why (never the raw FRED series)."""
    lines = []
    if report["abstained"]:
        lines.append(report["why"].get("error", "Abstained this cycle."))

    lines.append(f"Regime: {report['regime'].replace('_', '-')}")
    for flag, is_raised in report["flags"].items():
        status = "raised" if is_raised else "clear"
        lines.append(f"{flag} {status}: {report['why'].get(flag, '')}")

    return lines


def pm_lines(pm_view: dict) -> list[str]:
    """Return the PM memo and every name it dropped, with the reason (R1: drop-only)."""
    lines = [pm_view["memo"]]
    for drop in pm_view["drops"]:
        lines.append(f"Dropped {drop['ticker']}: {drop['reason']}")

    return lines


def critique_lines(critique: dict) -> list[str]:
    """Return the critic's notes and, per persona, the code-computed holdings down more than 10% from cost (S5, Q10)."""
    lines = [critique["notes"], "Down more than 10% from cost:"]
    for persona_key, tickers in critique["down_more_than_10pct"].items():
        lines.append(f"{display_name(persona_key)}: {', '.join(tickers) or 'none'}")

    return lines


def verdict_lines(verdict: dict) -> list[str]:
    """Return the final judge's summary and the persona it says to follow."""
    return [verdict["summary"], f"Follow: {display_name(verdict['persona'])}"]


def agent_lines(card: str, report: dict) -> list[str]:
    """Return one agent card's full text as lines, for its preview and its Show more popup."""
    if card == "pm":
        return pm_lines(report)

    if card == "critique":
        return critique_lines(report)

    if card == "verdict":
        return verdict_lines(report)

    if card == "macro":
        return macro_lines(report)

    return analyst_lines(report)


def show_full_text(title: str, text: str) -> None:
    """Open a popup with an agent's whole output, so its card never has to summarise."""

    def render_text() -> None:
        st.markdown(escape_markdown_math(text))

    st.dialog(title, width="large")(render_text)()


def render_pending_node(node_key: str, title: str) -> None:
    """Draw a node whose agent hasn't reported yet: its title over a shadcn skeleton."""
    with st.container(border=True, height="stretch", key=f"pending_node_{node_key}"):
        with ui.elements(key=f"pending_title_{node_key}") as heading:
            heading.heading(title, level=4)
        ui.skeleton(skeleton_height=72, key=f"skeleton_{node_key}")


def render_agent_node(card: str, report: dict | None) -> None:
    """Draw one agent as a shadcn card cut to about 220 characters, with Show more for the full text."""
    title = CARD_TITLES[card]
    if report is None:
        render_pending_node(card, title)
        return

    lines = agent_lines(card, report)
    preview = textwrap.shorten(" ".join(lines), CARD_PREVIEW_CHARS, placeholder="…")
    with ui.elements(key=f"agent_{card}") as agent_card:
        with agent_card.card():
            with agent_card.card_header():
                agent_card.heading(title, level=4)
            with agent_card.card_content():
                agent_card.text(preview)
            with agent_card.card_footer():
                show_more = agent_card.button(
                    "Show more",
                    key="show_more",
                    variant="outline",
                    size="sm",
                )

    if show_more.clicked:
        show_full_text(title, "\n\n".join(lines))


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


PROPOSED_ORDER_COLUMNS = [
    ui.TableColumn(key="side", label="Side"),
    ui.TableColumn(key="ticker", label="Ticker"),
    ui.TableColumn(key="shares", label="Shares", align="right"),
    ui.TableColumn(key="price", label="Price", align="right"),
    ui.TableColumn(key="notional", label="Notional", align="right"),
]
# Side, Ticker and Shares: the card's top-3 table fits three cards side by
# side at 1024 px; View all adds Price and Notional.
TOP_ORDER_COLUMNS = PROPOSED_ORDER_COLUMNS[:3]


def proposed_order_rows(orders: list[dict]) -> list[dict[str, str]]:
    """Return a proposal's orders as display rows, largest notional first."""
    largest_first = sorted(
        orders,
        key=lambda order: order["price"] * order["quantity"],
        reverse=True,
    )

    rows = []
    for order in largest_first:
        rows.append(
            {
                "side": order["side"].capitalize(),
                "ticker": order["ticker"],
                "shares": format_quantity(order["quantity"]),
                "price": format_money(order["price"]),
                "notional": format_money(order["price"] * order["quantity"]),
            }
        )

    return rows


def persona_summary(persona_key: str) -> str:
    """Return the persona's limits and account in one line, read from config so the numbers never drift."""
    persona = config.PERSONAS[persona_key]
    return (
        f"{PERSONA_STANCES[persona_key]}: {persona.sector_cap:.0%} per sector, "
        f"{persona.max_position_pct:.0%} per stock, {persona.cash_buffer:.0%} cash. "
        f"{ACCOUNT_NOTES[persona.account]}"
    )


def render_order_card(
    card_key: str,
    title: str,
    persona_key: str,
    orders: list[dict] | None,
) -> None:
    """Draw one persona's orders: title, its limits in a line, the top 3 orders and View all for the rest."""
    if orders is None:
        render_pending_node(card_key, title)
        return

    with st.container(border=True, height="stretch", key=f"order_card_{card_key}"):
        with ui.elements(key=f"order_head_{card_key}") as head:
            with head.stack(gap="xs"):
                head.heading(title, level=4)
                head.text(persona_summary(persona_key), variant="muted")

        if not orders:
            ui.alert(
                "No orders this cycle.",
                "Each change is under $200 or already on target.",
                key=f"no_orders_{card_key}",
            )
            return

        rows = proposed_order_rows(orders)
        ui.table(
            rows[:TOP_ORDER_ROWS],
            TOP_ORDER_COLUMNS,
            key=f"top_orders_{card_key}",
        )
        if ui.button("View all", key=f"view_all_orders_{card_key}", variant="outline"):
            show_full_table(title, rows, PROPOSED_ORDER_COLUMNS)


def render_connector(lines: list[tuple[int, int]], is_dotted: bool = True) -> None:
    """Draw curves from the row above (start x) down to the row below (end x), dotted or solid."""
    middle_y = CONNECTOR_HEIGHT / 2
    dash = 'stroke-dasharray="1 6" ' if is_dotted else ""

    paths = []
    for start_x, end_x in lines:
        curve = (
            f"M{start_x} 0 C{start_x} {middle_y} {end_x} {middle_y} "
            f"{end_x} {CONNECTOR_HEIGHT}"
        )
        paths.append(f'<path d="{curve}" {dash}vector-effect="non-scaling-stroke" />')

    # Stretched without keeping its aspect ratio, so x follows the columns.
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 600 {CONNECTOR_HEIGHT}" '
        f'preserveAspectRatio="none" fill="none" stroke="{CONNECTOR_COLOR}" '
        f'stroke-width="2" stroke-linecap="round">{"".join(paths)}</svg>'
    )
    # An <img>, because st.html's sanitizer strips inline <svg> tags.
    st.html(
        f'<img src="data:image/svg+xml,{quote(svg)}" alt="" '
        f'style="display: block; width: 100%; height: {CONNECTOR_HEIGHT}px">'
    )


def center_column() -> DeltaGenerator:
    """Return the middle half of the page, centered on the connectors' middle line."""
    return st.columns([1, 2, 1])[1]


def render_flow(fields: dict) -> None:
    """Draw the committee top to bottom from Proposal-shaped JSON; a node not in `fields` yet shows a skeleton.

    The three analysts converge on the PM, which branches into the three
    personas' orders; those converge on the devil's advocate, then the
    final verdict and, on a solid line, the recommended order.
    """
    orders_by_persona = fields.get("orders")
    recommended_key = fields.get("recommended_persona")

    # No gap, so each connector touches the cards above and below it.
    with st.container(key="committee_flow", gap=None):
        # Bottom-aligned, so all three cards end where the connector starts.
        analyst_columns = st.columns(len(ANALYSTS), vertical_alignment="bottom")
        for column, analyst in zip(analyst_columns, ANALYSTS, strict=True):
            with column:
                render_agent_node(analyst, fields.get(analyst))
        render_connector(CONVERGE_LINES)
        with center_column():
            render_agent_node("pm", fields.get("pm"))
        render_connector(BRANCH_LINES)

        persona_columns = st.columns(len(config.PERSONAS))
        for column, persona_key in zip(persona_columns, config.PERSONAS, strict=True):
            orders = None
            if orders_by_persona is not None:
                orders = orders_by_persona[persona_key]
            with column:
                title = f"{display_name(persona_key)} orders"
                render_order_card(persona_key, title, persona_key, orders)
        render_connector(CONVERGE_LINES)

        with center_column():
            render_agent_node("critique", fields.get("critique"))
        render_connector(DOWN_LINE)
        with center_column():
            render_agent_node("verdict", fields.get("verdict"))
        render_connector(DOWN_LINE, is_dotted=False)
        with center_column():
            if recommended_key is None:
                render_pending_node("recommended", "Recommended order")
            else:
                title = f"Recommended order · {display_name(recommended_key)}"
                orders = orders_by_persona[recommended_key]
                render_order_card("recommended", title, recommended_key, orders)


def proposal_fields(node_update: dict) -> dict:
    """Return one graph node's update as JSON-shaped Proposal fields, so a live run draws like a logged one."""
    fields = {}
    for state_key, value in node_update.items():
        field_name = PROPOSAL_FIELD_BY_STATE_KEY.get(state_key, state_key)
        fields[field_name] = to_jsonable_python(value)

    return fields


def stream_committee_run(question: str, model_name: str, updates: queue.Queue) -> None:
    """Run one committee cycle and log it, putting its progress on `updates`.

    Runs on a worker thread, so it makes no Streamlit call (R8): each node's
    fields go on the queue as ("update", fields), then ("done", run_record)
    or ("error", message). A Sonnet run is also saved as the fallback.
    """
    import books
    import committee

    try:
        snapshot = books.take_snapshot()
        # Seeded, or build_proposal would record the default question and model.
        state = {"snapshot": snapshot, "question": question, "model_name": model_name}
        graph_updates = committee.build_graph().stream(
            dict(state),
            stream_mode="updates",
        )
        for update in graph_updates:
            for node_update in update.values():
                state.update(node_update)
                updates.put(("update", proposal_fields(node_update)))

        proposal = committee.build_proposal(state)
        run_record = books.log_run(snapshot, proposal).model_dump(mode="json")
        if model_name.startswith(config.SONNET_DEMO_MODEL):
            books.save_sonnet_run(run_record)
    except Exception as error:
        logger.error("Run swarm failed", exc_info=error)
        updates.put(("error", f"{type(error).__name__}: {error}"))
        return

    updates.put(("done", run_record))


def start_live_run(question: str, model_name: str) -> None:
    """Start a committee run on a worker thread and keep its queue in the session, so every rerun keeps polling it."""
    # A new run replaces whatever the flow showed; nothing trades until Approve.
    for state_key in ("saved_run", "run_error", "approval"):
        st.session_state.pop(state_key, None)

    updates = queue.Queue()
    worker = threading.Thread(
        target=stream_committee_run,
        args=(question, model_name, updates),
        daemon=True,
    )
    worker.start()
    st.session_state["live_run"] = {
        "updates": updates,
        "started_at": time.monotonic(),
        "fields": {"question": question, "model_name": model_name},
        "is_past_fallback": False,
    }


def load_saved_run() -> dict | None:
    """Return the newest saved Sonnet run, or None when there is none or it can't be read (logged)."""
    import books

    try:
        return books.latest_sonnet_run()
    except Exception as error:
        logger.error("Reading the saved Sonnet run failed", exc_info=error)
        return None


def end_live_run(error_message: str) -> None:
    """Stop following a failed or stalled live run: its error shows, over the saved run when there is one."""
    live_run = st.session_state.pop("live_run")
    st.session_state["run_error"] = error_message
    if not live_run["is_past_fallback"]:
        st.session_state["saved_run"] = load_saved_run()


def apply_live_event(live_run: dict, event_kind: str, payload: dict | str) -> bool:
    """Apply one worker event to the session; return whether the page needs redrawing."""
    if event_kind == "update":
        live_run["fields"].update(payload)
        # A saved run on screen stays until the live run finishes.
        return st.session_state.get("saved_run") is None

    if event_kind == "done":
        del st.session_state["live_run"]
        st.session_state.pop("saved_run", None)
        st.session_state["committee_runs"].append(payload)
        return True

    end_live_run(payload)
    return True


def check_live_run_deadlines(live_run: dict, elapsed_seconds: float) -> bool:
    """Apply the 15 s fallback and the overall ceiling; return whether the page needs redrawing."""
    if elapsed_seconds > RUN_CEILING_SECONDS:
        end_live_run(
            f"No answer after {RUN_CEILING_SECONDS // 60} minutes, so the page "
            "stopped waiting."
        )
        return True

    if elapsed_seconds < FALLBACK_AFTER_SECONDS or live_run["is_past_fallback"]:
        return False

    live_run["is_past_fallback"] = True
    st.session_state["saved_run"] = load_saved_run()
    return st.session_state["saved_run"] is not None


def drain_live_run(live_run: dict) -> bool:
    """Apply every event the worker has queued so far; return whether the page needs redrawing."""
    needs_redraw = False
    while True:
        try:
            event_kind, payload = live_run["updates"].get_nowait()
        except queue.Empty:
            return needs_redraw

        if apply_live_event(live_run, event_kind, payload):
            needs_redraw = True


@st.fragment(run_every=POLL_SECONDS)
def follow_live_run() -> None:
    """Poll the live run's queue and deadlines, rerunning the whole page when the flow changes.

    A fragment on a timer rather than a loop, so every full run finishes:
    Streamlit clears a run's leftover elements only once it ends.
    """
    live_run = st.session_state.get("live_run")
    if live_run is None:
        return

    elapsed_seconds = time.monotonic() - live_run["started_at"]
    model_label = config.get_model_label(live_run["fields"]["model_name"])
    st.caption(f"The committee is meeting on {model_label}: {elapsed_seconds:.0f} s")

    # A run that just ended always redraws, so its deadlines are never checked.
    if drain_live_run(live_run) or check_live_run_deadlines(live_run, elapsed_seconds):
        st.rerun(scope="app")


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


def approve_choice_label(recommended_key: str, choice: str) -> str:
    """Return an Approve option's label: "Recommended (Balanced)" or a persona's name."""
    if choice == "recommended":
        return f"Recommended ({display_name(recommended_key)})"

    return display_name(choice)


def approve_controls(run_record: dict) -> None:
    """Approve executes one persona's orders from the run on screen, the recommended one by default (story 22, T18)."""
    proposal_fields = run_record["proposal"]
    recommended_key = proposal_fields["recommended_persona"]
    with ui.elements(key="approve_bar", width=APPROVE_BAR_WIDTH) as approve_bar:
        with approve_bar.stack(direction="horizontal", align="end", gap="sm"):
            choice_field = approve_bar.select(
                "Orders to approve",
                APPROVE_CHOICES,
                key="approve_choice",
                format_func=functools.partial(approve_choice_label, recommended_key),
            )
            approve_button = approve_bar.button("Approve", key="approve")

    persona_key = choice_field.value
    if persona_key == "recommended":
        persona_key = recommended_key

    caption_snaptrade_order_time({persona_key: proposal_fields["orders"][persona_key]})
    if not approve_button.clicked:
        # Not on the click: execute re-reads the lock fresh, and a read here
        # would spend two of SnapTrade's 10 requests a minute (Risk #8).
        caption_queued_lock()
        return

    from committee import Proposal

    proposal = Proposal.model_validate(proposal_fields)
    st.session_state["approval"] = {
        "taken_at": run_record["taken_at"],
        "persona": persona_key,
        "outcome": execute_personas(proposal, [persona_key], tag="committee"),
    }
    clear_rebalance_state()


def render_approval_section(run_record: dict, is_approvable: bool) -> None:
    """Show Approve for the latest or the saved run, then what it executed; an older run only says where Approve is."""
    if not is_approvable:
        st.caption("Approve works on the latest run: pick it under Previous runs.")
        return

    approve_controls(run_record)
    approval = st.session_state.get("approval")
    if approval is None or approval["taken_at"] != run_record["taken_at"]:
        return

    with ui.elements(key="execution_heading") as heading:
        heading.heading(f"Execution · {display_name(approval['persona'])}", level=4)
    render_execution(approval["persona"], approval["outcome"])


def render_prompt_bar(is_running: bool) -> tuple[str, str, bool]:
    """Draw the model, the question and Run swarm in one row; return the question, the model id and whether Run swarm was clicked.

    The question and Run swarm share one shadcn tree, so the click also
    commits a question typed just before it. The model sits apart because
    a tree shares its width equally between fields; a select commits as
    soon as it changes.
    """
    missing_run_keys = missing_keys(ANTHROPIC_KEY_ENV_VAR, FRED_KEY_ENV_VAR)
    model_labels = list(config.MODEL_CHOICES)
    default_label = config.get_model_label(config.get_model_name())
    if default_label not in model_labels:
        default_label = None

    with st.container(horizontal=True):
        model_label = ui.select(
            "Model",
            model_labels,
            value=default_label,
            key="model",
            width=MODEL_SELECT_WIDTH,
        )
        with ui.elements(key="prompt_bar") as prompt:
            with prompt.stack(gap="xs"):
                with prompt.stack(direction="horizontal", align="end", gap="sm"):
                    question_field = prompt.input(
                        "Question",
                        config.DEFAULT_QUESTION,
                        key="question",
                    )
                    run_button = prompt.button(
                        "Run swarm",
                        key="run_swarm",
                        disabled=bool(missing_run_keys) or is_running,
                        loading=is_running,
                    )
                if missing_run_keys:
                    prompt.text(
                        f"Run swarm needs {' and '.join(missing_run_keys)} in .env.",
                        variant="caption",
                    )

    # A question backspaced to nothing asks the default one.
    question = (question_field.value or "").strip() or config.DEFAULT_QUESTION
    return question, config.MODEL_CHOICES[model_label], run_button.clicked


def run_label(record: dict) -> str:
    """Return a run's picker label, e.g. "September 27, 2026, 10:42 · Sonnet 5 · <question>"; a saved Sonnet run says so."""
    proposal = record["proposal"]
    label = " · ".join(
        [
            format_long_time(record["taken_at"]),
            config.get_model_label(proposal["model_name"]),
            textwrap.shorten(proposal["question"], 60, placeholder="…"),
        ]
    )
    # Only a saved Sonnet run carries saved_at.
    if "saved_at" in record:
        return f"Saved Sonnet run · {label}"

    return label


def render_previous_runs(
    records: list[dict],
    is_disabled: bool,
    is_preselected: bool,
) -> dict | None:
    """Draw the Previous runs select over `records` (newest first); return the picked run, or None when none is."""
    if not records:
        return None

    choices = []
    for index, record in enumerate(records):
        choices.append(ui.Choice(value=index, label=run_label(record)))

    # A new key whenever a run or a saved run joins the top of the list (or
    # a failure clears the pick) resets the pick.
    list_state = f"{records[0]['taken_at']}_{len(records)}_{is_preselected}"
    picked_index = ui.select(
        "Previous runs",
        choices,
        index=0 if is_preselected else None,
        key=f"previous_runs_{list_state}",
        placeholder="Pick a previous run",
        disabled=is_disabled,
    )
    if picked_index is None:
        return None

    return records[picked_index]


def render_saved_run_label(saved_run: dict, is_running: bool) -> None:
    """Say plainly that the flow shows the saved Sonnet run, and whether the live run may still replace it."""
    title = (
        f"Saved Sonnet run from {format_long_time(saved_run['taken_at'])} · "
        f"question: {saved_run['proposal']['question']}"
    )
    description = None
    if is_running:
        description = (
            "The live run is still going; it replaces this one when it finishes."
        )

    ui.alert(title, description, key="saved_run_label")


def committee_tab() -> None:
    """Ask the committee a question on a chosen model, watch the swarm fill its flow live, then approve one persona (S11)."""
    with ui.elements(key="committee_intro") as intro:
        intro.text(COMMITTEE_INTRO, variant="muted")

    # This session's finished runs, oldest first; nothing outlives the session.
    runs = st.session_state.setdefault("committee_runs", [])
    live_run = st.session_state.get("live_run")
    question, model_name, run_clicked = render_prompt_bar(
        is_running=live_run is not None
    )
    if run_clicked:
        start_live_run(question, model_name)
        st.rerun()

    if live_run is not None:
        follow_live_run()

    run_error = st.session_state.get("run_error")
    if run_error is not None:
        ui.alert("Run swarm failed", run_error, variant="destructive", key="run_error")

    saved_run = st.session_state.get("saved_run")
    records = runs[::-1]
    if saved_run is not None:
        records.insert(0, saved_run)
    # After a failed run with nothing saved, the flow stays empty until a pick.
    is_preselected = run_error is None or saved_run is not None
    picked_run = render_previous_runs(records, live_run is not None, is_preselected)

    if live_run is not None and saved_run is None:
        render_flow(live_run["fields"])
    elif picked_run is None:
        if not records:
            ui.alert("Press Run swarm to convene the committee.", key="committee_empty")
        render_flow({})
    else:
        if picked_run is saved_run:
            render_saved_run_label(saved_run, is_running=live_run is not None)
        render_flow(picked_run["proposal"])
        is_approvable = picked_run is saved_run or picked_run is runs[-1]
        render_approval_section(picked_run, is_approvable)


def holdings_rows(book: Book, marks: dict[str, float], equity: float) -> list[dict]:
    """Return one table row per held ticker: sector, shares, mark, value, weight and average cost."""
    rows = []
    for ticker, shares in sorted(book.shares.items()):
        value = shares * marks[ticker]
        avg_cost = book.avg_cost.get(ticker)
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
    book: Book,
    marks: dict[str, float],
    sector_weights: dict[str, float],
) -> None:
    """Show one persona's limits, NAV, cash, holdings and sector weights against its own cap (S10, story 38)."""
    import portfolio

    persona = config.PERSONAS[persona_key]
    equity = portfolio.compute_book_equity(book, marks)
    st.caption(
        f"Sector cap {persona.sector_cap:.0%}, max per name "
        f"{persona.max_position_pct:.0%}, cash buffer {persona.cash_buffer:.0%}"
    )
    nav_column, cash_column = st.columns(2)
    nav_column.metric("NAV", f"${equity:,.0f}")
    cash_column.metric("Cash", f"${book.cash:,.0f}")

    holdings_column, chart_column = st.columns(2)
    with holdings_column:
        holdings = holdings_rows(book, marks, equity)
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

    approve_clicked = st.button(
        "Approve rebalance", key=f"approve_rebalance_{persona_key}"
    )
    caption_snaptrade_order_time({persona_key: result.orders})
    if approve_clicked:
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

    is_fred_missing = bool(missing_keys(FRED_KEY_ENV_VAR))
    if is_fred_missing:
        st.caption(f"Rebalance needs {FRED_KEY_ENV_VAR} in .env.")

    # Kept in session_state: a button under another button's `if` never
    # fires, because the outer button is False again on that rerun.
    rebalance_clicked = st.button(
        "Rebalance",
        key=f"rebalance_{persona_key}",
        disabled=is_fred_missing,
    )
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

    st.subheader("SnapTrade Sandbox")
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

    is_fred_missing = bool(missing_keys(FRED_KEY_ENV_VAR))
    if is_fred_missing:
        st.caption(f"Reset books needs {FRED_KEY_ENV_VAR} in .env to re-seed.")

    confirmed = st.checkbox(
        "Clear the simulator fill logs and re-seed the opening books",
        key="confirm_reset",
    )
    reset_clicked = st.button(
        "Reset books",
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

        # Tickers from an older universe have no mark to value them by.
        stale_tickers = portfolio.tickers_outside_universe(book)
        if stale_tickers:
            st.caption(
                f"Holds {', '.join(stale_tickers)} from the old stock universe; "
                "Reset books below re-seeds it."
            )
            continue

        weights_by_sector = portfolio.sector_weights(book, marks)
        render_book(persona_key, book, marks, weights_by_sector)
        rebalance_controls(persona_key, book, marks)

    sandbox_panel()
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


def format_long_time(timestamp: str) -> str:
    """Return an ISO timestamp in local time as e.g. "September 27, 2026, 10:42"."""
    moment = datetime.fromisoformat(timestamp).astimezone()
    return f"{format_long_date(moment.isoformat())}, {moment:%H:%M}"


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


def show_full_table(title: str, rows: list[dict], columns: list) -> None:
    """Open a popup with a whole table: a brokerage's activity or a persona's orders."""

    def render_table() -> None:
        ui.table(rows, columns, key="full_table", max_height=520)

    st.dialog(title, width="large")(render_table)()


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
            show_full_table(f"{name} activity", rows, columns)


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
