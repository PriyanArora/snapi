"""Fixed numbers from the PRD: universe, sectors, personas and finance rules.

Every constant here traces back to a Finance Decision (F1-F23) or an Open
Item default (Q1-Q9, Q13). Nothing in this file is computed from data — the
functions that read prices, news or macro series live in feeds.py and
signals.py, and use these constants as their fixed inputs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Universe and benchmark (Q1). Phase11 replaced Q1's list with household
# names across 7 SPDR sectors.
UNIVERSE = (
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AMZN",
    "TSLA",
    "KO",
    "JPM",
    "XOM",
    "JNJ",
)
BENCHMARK = "SPY"

# Sector buckets (F9) and their membership (Q2, Q3), one SPDR sector ETF each.
# ETFs map to their own bucket too, so SECTOR_BY_TICKER covers both stocks
# and the ETFs used for rebalance buys.
SECTORS = (
    "Technology",
    "Communication",
    "Consumer",
    "Staples",
    "Financials",
    "Energy",
    "Healthcare",
)

ETF_BY_SECTOR = {
    "Technology": "XLK",
    "Communication": "XLC",
    "Consumer": "XLY",
    "Staples": "XLP",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
}

_STOCK_SECTOR_BY_TICKER = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Communication",
    "META": "Communication",
    "AMZN": "Consumer",
    "TSLA": "Consumer",
    "KO": "Staples",
    "JPM": "Financials",
    "XOM": "Energy",
    "JNJ": "Healthcare",
}
SECTOR_BY_TICKER = _STOCK_SECTOR_BY_TICKER | {
    etf: sector for sector, etf in ETF_BY_SECTOR.items()
}

# Macro regime (F13) and its exposure fractions.
Regime = Literal["risk_on", "neutral", "risk_off"]

REGIME_FRACTIONS: dict[Regime, float] = {
    "risk_on": 1.0,
    "neutral": 0.75,
    "risk_off": 0.5,
}

# Macro checklist thresholds (Q5). SPY-vs-200-day-SMA is structural (no
# threshold number) and is evaluated directly in signals.py.
YIELD_CURVE_FLAG_BELOW = 0.0  # 10Y-3M below this triggers the flag
SAHM_FLAG_AT_OR_ABOVE = 0.50
HY_OAS_FLAG_ABOVE = 5.0
VIX_FLAG_ABOVE = 25.0

# Flag count -> regime (Q5): 0-1 risk-on, 2-3 neutral, 4-5 risk-off.
RISK_ON_MAX_FLAGS = 1
NEUTRAL_MAX_FLAGS = 3

# Persona configuration (F2-F5, F10, F11, Q4, Q8, persona table).
Account = Literal["simulator", "alpaca_paper"]


@dataclass(frozen=True)
class Persona:
    """One committee view's risk limits and account routing."""

    display_name: str
    account: Account
    starting_cash: float
    sector_cap: float
    max_position_pct: float
    cash_buffer: float

    @property
    def max_invested_pct(self) -> float:
        """Max % of equity this persona can ever hold in positions."""
        return 1 - self.cash_buffer

    def invested_pct_for_regime(self, regime: Regime) -> float:
        """Target invested % of equity at the given macro regime."""
        # Rounded so the persona table's values compare exactly (0.95 * 0.75 is 0.7124999...).
        return round(self.max_invested_pct * REGIME_FRACTIONS[regime], 6)


PERSONAS: dict[str, Persona] = {
    "conservative": Persona(
        display_name="Conservative",
        account="simulator",
        starting_cash=100_000,
        sector_cap=0.20,
        max_position_pct=0.08,
        cash_buffer=0.15,
    ),
    "balanced": Persona(
        display_name="Balanced",
        account="alpaca_paper",
        starting_cash=100_000,
        sector_cap=0.30,
        max_position_pct=0.12,
        cash_buffer=0.05,
    ),
    "aggressive": Persona(
        display_name="Aggressive",
        account="simulator",
        starting_cash=100_000,
        sector_cap=0.40,
        max_position_pct=0.20,
        cash_buffer=0.02,
    ),
}

# Conviction blend (F12).
TECHNICAL_WEIGHT = 0.6
SENTIMENT_WEIGHT = 0.4

# Cash and trade-size floor (F11).
MIN_TRADE_USD = 200

# Backtest trading cost (F19).
COST_BPS = 5

# Critic's "down more than 10%" flag (F20, Q10).
POSITION_DOWN_FLAG_PCT = 0.10

# Fear & Greed contrarian tilt (Q6).
FEAR_GREED_LOW = 25
FEAR_GREED_HIGH = 75
FEAR_GREED_TILT = 0.10

# Fundamentals tilt (the finance teammate, Phase11): a per-stock nudge in
# the same deterministic style as the Fear & Greed tilt, with no LLM call.
# What counts as an expensive forward P/E depends on the sector.
FORWARD_PE_LIMIT_BY_SECTOR = {
    "Technology": 35.0,
    "Communication": 25.0,
    "Consumer": 30.0,
    "Staples": 25.0,
    "Financials": 15.0,
    "Energy": 15.0,
    "Healthcare": 22.0,
}
FORWARD_PE_TILT = 0.05  # forward P/E above its sector's limit
FORWARD_PE_STRETCHED_MULTIPLE = 1.5
FORWARD_PE_STRETCHED_TILT = 0.10  # forward P/E above 1.5x the limit
DEBT_TO_EQUITY_LIMIT = 2.0  # a ratio: 2.0 means debt is twice equity
DEBT_TO_EQUITY_TILT = 0.05  # taken on top of the forward P/E tilt
# Bank leverage is structural, so the debt check skips this sector.
DEBT_TO_EQUITY_SKIPPED_SECTOR = "Financials"

# Technical score inputs (F14, Q7).
RSI_PERIOD = 14
RSI_MIDPOINT = 50
RSI_SCORE_DIVISOR = 20
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
MACD_MEAN_ABS_LOOKBACK_DAYS = 63
SMA_FAST = 50
SMA_SLOW = 200

# LLM model choice (T15). One env var picks Haiku 4.5 (dev) or Sonnet 5 (demo)
# as the default; the Committee page can pick any of MODEL_CHOICES per run.
SNAPI_MODEL_ENV_VAR = "SNAPI_MODEL"
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
SONNET_DEMO_MODEL = "claude-sonnet-5"
OPUS_MODEL = "claude-opus-5-5"

# Display label -> model id.
MODEL_CHOICES = {
    "Haiku 4.5": DEFAULT_MODEL,
    "Sonnet 5": SONNET_DEMO_MODEL,
    "Opus 5.5": OPUS_MODEL,
}

# What the committee answers when the user asks nothing more specific.
DEFAULT_QUESTION = (
    "Given today's market, what should each of my three portfolios buy or sell?"
)


def get_model_name() -> str:
    """Return the LLM model id, from SNAPI_MODEL or the Haiku 4.5 default."""
    # `or`, not a get() default: .env.example's empty SNAPI_MODEL= loads as "".
    return os.environ.get(SNAPI_MODEL_ENV_VAR) or DEFAULT_MODEL


def get_model_label(model_name: str) -> str:
    """Return a model id's display label ("Sonnet 5"), or the id itself when it isn't one of MODEL_CHOICES."""
    for label, choice_model_name in MODEL_CHOICES.items():
        if model_name == choice_model_name:
            return label

    return model_name


# Disclaimer (Q13), shown on the gate screen and the closing slide verbatim.
DISCLAIMER = (
    "Educational project built for Hack Atlantic. All trading shown is paper "
    "trading; no real money is used or at risk. Nothing here is investment "
    "advice."
)

# Local log files (R9): books.py reads and writes them, or their MongoDB
# collections of the same name when MONGODB_URI is set (Hosting).
DATA_DIR = Path(__file__).resolve().parent / "data"
RUN_LOG_PATH = DATA_DIR / "run_log.jsonl"
# The latest Sonnet run, kept so the UI can fall back to it when a live run
# fails or is too slow.
SONNET_RUNS_PATH = DATA_DIR / "sonnet_runs.jsonl"

# P1 backtest output (S15): backtest.py writes it, the Backtest tab reads it.
BACKTEST_RESULTS_PATH = DATA_DIR / "backtest_results.json"
