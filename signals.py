"""Signals: deterministic technical score and macro regime checklist.

No network and no LLM calls here (S2, S3; Who produces which number): the
Technical and Macro numbers are pure functions over the committed price
snapshot and FRED's latest macro series, so the live analysts and the P1
backtest can both call the same code. The LLM calls only narrate what these
functions already decided.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd
from stockstats import StockDataFrame

from config import (
    BENCHMARK,
    HY_OAS_FLAG_ABOVE,
    MACD_MEAN_ABS_LOOKBACK_DAYS,
    NEUTRAL_MAX_FLAGS,
    RISK_ON_MAX_FLAGS,
    RSI_MIDPOINT,
    RSI_PERIOD,
    RSI_SCORE_DIVISOR,
    SAHM_FLAG_AT_OR_ABOVE,
    SMA_FAST,
    SMA_SLOW,
    VIX_FLAG_ABOVE,
    YIELD_CURVE_FLAG_BELOW,
    Regime,
)
from prices import close_history


@dataclass(frozen=True)
class TechnicalScore:
    """A ticker's technical score in [-1, 1] plus the raw indicators behind it."""

    score: float
    rsi: float
    macd_histogram: float
    sma_fast: float
    sma_slow: float


@dataclass(frozen=True)
class MacroChecklist:
    """The 5-flag macro checklist (F13), its flag count and the regime it implies."""

    flags: dict[str, bool]
    flag_count: int
    regime: Regime


def _stock_frame(closes: pd.Series) -> StockDataFrame:
    """Wrap a close-price series so stockstats can derive RSI, MACD and SMA from it."""
    return StockDataFrame.retype(pd.DataFrame({"close": closes.values}))


def _clip_to_unit_range(sub_score: float) -> float:
    """Clip one technical sub-score to [-1, 1] (Q7)."""
    return max(-1.0, min(1.0, sub_score))


def compute_technical_score(ticker: str, as_of: date | None = None) -> TechnicalScore:
    """Return `ticker`'s technical score, plus the raw RSI/MACD/SMA behind it.

    The score is the average of three sub-scores, each clipped to [-1, 1] (F14, Q7):
    - RSI 14: (50 - RSI) / 20
    - MACD 12/26/9 histogram, divided by its own 63-day mean absolute value
    - 50/200-day SMA cross: +1 if the 50-day is above the 200-day, else -1

    Reads price history up to `as_of` (inclusive) so the P1 backtest can call
    this at any historical date, not just the latest one.
    """
    closes = close_history(ticker, as_of)
    frame = _stock_frame(closes)

    rsi = float(frame[f"rsi_{RSI_PERIOD}"].iloc[-1])
    rsi_score = _clip_to_unit_range((RSI_MIDPOINT - rsi) / RSI_SCORE_DIVISOR)

    # "macd"/"macds"/"macdh" use stockstats' own default windows (12, 26, 9),
    # which already match config's MACD_FAST/MACD_SLOW/MACD_SIGNAL.
    macd_histogram = float(frame["macdh"].iloc[-1])
    macd_mean_abs = frame["macdh"].tail(MACD_MEAN_ABS_LOOKBACK_DAYS).abs().mean()
    if macd_mean_abs == 0:
        macd_score = 0.0  # flat histogram history -> nothing to normalize against
    else:
        macd_score = _clip_to_unit_range(macd_histogram / macd_mean_abs)

    sma_fast = float(frame[f"close_{SMA_FAST}_sma"].iloc[-1])
    sma_slow = float(frame[f"close_{SMA_SLOW}_sma"].iloc[-1])
    ma_cross_score = 1.0 if sma_fast > sma_slow else -1.0

    score = (rsi_score + macd_score + ma_cross_score) / 3

    return TechnicalScore(
        score=score,
        rsi=rsi,
        macd_histogram=macd_histogram,
        sma_fast=sma_fast,
        sma_slow=sma_slow,
    )


def compute_macro_checklist(
    macro_series: dict[str, float], as_of: date | None = None
) -> MacroChecklist:
    """Return the 5-flag macro checklist (F13, Q5) and the regime it implies.

    `macro_series` is FRED's latest T10Y3M/SAHMREALTIME/BAMLH0A0HYM2/VIXCLS
    values (feeds.fetch_macro_series). The fifth flag, SPY against its
    200-day SMA, is read from the committed price snapshot instead, as of
    the same date. Flag count maps to regime: 0-1 risk_on, 2-3 neutral, 4-5
    risk_off. The UI shows these derived flags only, never the raw series.
    """
    spy_closes = close_history(BENCHMARK, as_of)
    spy_frame = _stock_frame(spy_closes)
    spy_close = float(spy_closes.iloc[-1])
    spy_sma_slow = float(spy_frame[f"close_{SMA_SLOW}_sma"].iloc[-1])

    flags = {
        "yield_curve": macro_series["T10Y3M"] < YIELD_CURVE_FLAG_BELOW,
        "sahm_rule": macro_series["SAHMREALTIME"] >= SAHM_FLAG_AT_OR_ABOVE,
        "credit_spread": macro_series["BAMLH0A0HYM2"] > HY_OAS_FLAG_ABOVE,
        "vix": macro_series["VIXCLS"] > VIX_FLAG_ABOVE,
        "spy_trend": spy_close < spy_sma_slow,
    }
    flag_count = sum(flags.values())

    if flag_count <= RISK_ON_MAX_FLAGS:
        regime: Regime = "risk_on"
    elif flag_count <= NEUTRAL_MAX_FLAGS:
        regime = "neutral"
    else:
        regime = "risk_off"

    return MacroChecklist(flags=flags, flag_count=flag_count, regime=regime)


if __name__ == "__main__":
    from config import PERSONAS, UNIVERSE

    for ticker in UNIVERSE:
        technical = compute_technical_score(ticker)
        assert -1.0 <= technical.score <= 1.0, f"{ticker} score out of range"
        print(f"{ticker}: score={technical.score:.3f} rsi={technical.rsi:.1f}")

    macro_series = {
        "T10Y3M": -0.2,
        "SAHMREALTIME": 0.1,
        "BAMLH0A0HYM2": 3.2,
        "VIXCLS": 14.0,
    }
    checklist = compute_macro_checklist(macro_series)
    assert checklist.flag_count == sum(checklist.flags.values())
    assert checklist.regime in ("risk_on", "neutral", "risk_off")
    print(f"macro: {checklist.flags} -> {checklist.regime}")

    for persona in PERSONAS.values():
        invested_pct = persona.invested_pct_for_regime(checklist.regime)
        print(f"{persona.display_name}: invested_pct={invested_pct}")
