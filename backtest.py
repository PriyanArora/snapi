"""Backtest (P1, S15): a deterministic 1-year weekly backtest of the technical picks vs SPY.

Our own pandas loop, not hedge_fund's backtest_fund, with no LLM or news
calls (F16, Out of Scope). Each Monday (or the week's first open day, F19)
it scores the 10 names with the same compute_technical_score the live
Technical analyst uses (story 4), holds the names with a positive score at
equal weight from that close to the next rebalance close, and pays 5 bps x
turnover. The macro overlay then scales each week's return by the invested
fraction (F13) of the regime that was knowable before the week, with the
rest in cash at 0% (R11: point-in-time FRED and ALFRED inputs).

`uv run python backtest.py` needs FRED_API_KEY and writes the committed
results file the Backtest tab reads. The file holds growth curves and
metrics only, never the raw ICE or CBOE series (redistribution terms).
"""

from __future__ import annotations

import json
import math
import os
import statistics
from datetime import date, timedelta
from itertools import pairwise

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

import config
from feeds import FeedUnavailableError
from prices import load_price_snapshot
from signals import compute_macro_checklist, compute_technical_score

BACKTEST_WEEKS = 52
WEEKS_PER_YEAR = 52

# R11: these daily series are shifted one day; SAHMREALTIME uses ALFRED vintages instead.
DAILY_SERIES_IDS = ("T10Y3M", "BAMLH0A0HYM2", "VIXCLS")
SAHM_SERIES_ID = "SAHMREALTIME"

TECHNICAL_CURVE = "Technical"
OVERLAY_CURVE = "Technical + macro overlay"
EQUAL_WEIGHT_CURVE = "Equal-weight universe"
CURVE_NAMES = (TECHNICAL_CURVE, OVERLAY_CURVE, config.BENCHMARK, EQUAL_WEIGHT_CURVE)

EQUAL_UNIVERSE_WEIGHTS = {
    ticker: 1 / len(config.UNIVERSE) for ticker in config.UNIVERSE
}


def weekly_rebalance_dates(trading_days: list[date]) -> list[date]:
    """Return the last 53 rebalance dates (52 weekly periods): each week's first trading day, normally Monday.

    `trading_days` must be sorted. Raises ValueError when the snapshot can't
    fit 52 weeks after the 200-day warm-up: stockstats would otherwise average
    a shorter window without saying so.
    """
    first_day_by_week: dict[tuple[int, int], date] = {}
    for trading_day in trading_days:
        iso_year, iso_week, _ = trading_day.isocalendar()
        first_day_by_week.setdefault((iso_year, iso_week), trading_day)

    rebalance_dates = list(first_day_by_week.values())[-(BACKTEST_WEEKS + 1) :]
    if len(rebalance_dates) < BACKTEST_WEEKS + 1:
        raise ValueError(
            f"The price snapshot covers {len(rebalance_dates) - 1} weeks; "
            f"the backtest needs {BACKTEST_WEEKS}"
        )

    warm_up_days = trading_days.index(rebalance_dates[0])
    if warm_up_days < config.SMA_SLOW:
        raise ValueError(
            f"Only {warm_up_days} trading days precede {rebalance_dates[0]}; "
            f"the {config.SMA_SLOW}-day SMA needs {config.SMA_SLOW}"
        )

    return rebalance_dates


def fetch_macro_history() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Download the daily FRED series and every SAHMREALTIME vintage once for the whole run.

    Raises FeedUnavailableError if FRED_API_KEY is not set.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise FeedUnavailableError(
            "FRED_API_KEY is not set; the backtest's macro overlay needs it"
        )

    fred = Fred(api_key=api_key)
    daily_series = {}
    for series_id in DAILY_SERIES_IDS:
        daily_series[series_id] = fred.get_series(series_id).dropna()

    # One row per (month, revision), all object columns. fredapi builds the
    # frame by transposing mixed rows, so under pandas 3 a missing value can
    # arrive as NaT; `where` turns every missing value back into NaN.
    all_releases = fred.get_series_all_releases(SAHM_SERIES_ID)
    values = all_releases["value"]
    sahm_releases = pd.DataFrame(
        {
            "date": pd.to_datetime(all_releases["date"]),
            "realtime_start": pd.to_datetime(all_releases["realtime_start"]),
            "value": pd.to_numeric(values.where(values.notna())),
        }
    )
    return daily_series, sahm_releases


def sahm_value_as_of(as_of: date, sahm_releases: pd.DataFrame) -> float:
    """Return the latest SAHMREALTIME reading as it was published on `as_of` (its ALFRED vintage)."""
    published = sahm_releases[sahm_releases["realtime_start"] <= pd.Timestamp(as_of)]

    # Every revision is its own row, so keep the latest real-time start per month:
    # that is the value readers saw on `as_of`.
    known_by_month = published.sort_values("realtime_start").groupby("date").tail(1)
    known_by_month = known_by_month.dropna(subset=["value"]).sort_values("date")
    if known_by_month.empty:
        raise ValueError(f"No {SAHM_SERIES_ID} vintage was published by {as_of}")

    return float(known_by_month["value"].iloc[-1])


def point_in_time_macro_series(
    as_of: date,
    daily_series: dict[str, pd.Series],
    sahm_releases: pd.DataFrame,
) -> dict[str, float]:
    """Return the four FRED checklist inputs as they were knowable on `as_of` (R11: no look-ahead).

    Daily series use their last value dated on or before the previous day;
    SAHMREALTIME uses the vintage published on or before `as_of`.
    """
    known_through = pd.Timestamp(as_of - timedelta(days=1))
    macro_series = {}
    for series_id, series in daily_series.items():
        known_values = series[series.index <= known_through]
        if known_values.empty:
            raise ValueError(f"FRED has no {series_id} value by {known_through.date()}")

        macro_series[series_id] = float(known_values.iloc[-1])

    macro_series[SAHM_SERIES_ID] = sahm_value_as_of(as_of, sahm_releases)
    return macro_series


def macro_exposure(
    week_start: date,
    daily_series: dict[str, pd.Series],
    sahm_releases: pd.DataFrame,
) -> float:
    """Return the invested fraction (F13: 100/75/50%) set by the regime known before `week_start`."""
    macro_series = point_in_time_macro_series(week_start, daily_series, sahm_releases)

    # The live checklist rules; SPY's 200-day trend is shifted one day too (R11).
    spy_known_through = week_start - timedelta(days=1)
    checklist = compute_macro_checklist(macro_series, as_of=spy_known_through)
    return config.REGIME_FRACTIONS[checklist.regime]


def technical_weights(as_of: date) -> dict[str, float]:
    """Return equal weights across the universe names with a positive technical score on `as_of`."""
    picks = []
    for ticker in config.UNIVERSE:
        if compute_technical_score(ticker, as_of).score > 0:
            picks.append(ticker)

    if not picks:
        return {}  # no positive score: all cash this week

    return {ticker: 1 / len(picks) for ticker in picks}


def weekly_price_returns(
    closes: pd.DataFrame,
    week_start: date,
    week_end: date,
) -> pd.Series:
    """Return each ticker's price return from one rebalance close to the next.

    Raises ValueError when a ticker has no close on either date, so NaN never
    reaches the results file.
    """
    start_closes = closes.loc[pd.Timestamp(week_start)]
    end_closes = closes.loc[pd.Timestamp(week_end)]
    period_returns = end_closes / start_closes - 1

    missing_tickers = period_returns.index[period_returns.isna()].tolist()
    if missing_tickers:
        raise ValueError(
            f"No close for {missing_tickers} on {week_start} or {week_end}"
        )

    return period_returns


def portfolio_return(weights: dict[str, float], period_returns: pd.Series) -> float:
    """Return a book's return over the week; weight not held is cash at 0%."""
    total_return = 0.0
    for ticker, weight in weights.items():
        total_return += weight * float(period_returns[ticker])

    return total_return


def turnover(target_weights: dict[str, float], held_weights: dict[str, float]) -> float:
    """Return the total traded weight that moves the held book onto the targets (F19 costs 5 bps of it)."""
    traded_weight = 0.0
    for ticker in target_weights.keys() | held_weights.keys():
        target_weight = target_weights.get(ticker, 0.0)
        held_weight = held_weights.get(ticker, 0.0)
        traded_weight += abs(target_weight - held_weight)

    return traded_weight


def drifted_weights(
    weights: dict[str, float],
    period_returns: pd.Series,
) -> dict[str, float]:
    """Return the weights after a week's price moves, so the next turnover counts the trim back to target."""
    cash = 1 - sum(weights.values())
    grown_values = {}
    for ticker, weight in weights.items():
        grown_values[ticker] = weight * (1 + float(period_returns[ticker]))

    total_value = cash + sum(grown_values.values())
    return {ticker: value / total_value for ticker, value in grown_values.items()}


def growth_of_one(weekly_returns: list[float]) -> list[float]:
    """Return the value of $1 at the start and after each week."""
    values = [1.0]
    for weekly_return in weekly_returns:
        values.append(values[-1] * (1 + weekly_return))

    return values


def max_drawdown(growth: list[float]) -> float:
    """Return the worst drop from a peak to a later trough, as a fraction <= 0."""
    peak = growth[0]
    worst_drawdown = 0.0
    for value in growth:
        peak = max(peak, value)
        worst_drawdown = min(worst_drawdown, value / peak - 1)

    return worst_drawdown


def curve_metrics(
    weekly_returns: list[float],
    benchmark_total_return: float,
) -> dict[str, float]:
    """Return a curve's total and annualized return, Sharpe, max drawdown and excess return over SPY (story 48).

    Sharpe is the weekly mean over the weekly standard deviation times
    sqrt(52), with a 0% risk-free rate to match the overlay's cash.
    """
    growth = growth_of_one(weekly_returns)
    total_return = growth[-1] - 1
    annualized_return = (1 + total_return) ** (WEEKS_PER_YEAR / len(weekly_returns)) - 1

    volatility = statistics.stdev(weekly_returns)
    if volatility == 0:
        sharpe = 0.0  # a flat curve, e.g. all cash, has no risk to divide by
    else:
        mean_return = statistics.mean(weekly_returns)
        sharpe = mean_return / volatility * math.sqrt(WEEKS_PER_YEAR)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(growth),
        "excess_return_vs_spy": total_return - benchmark_total_return,
    }


def count_days_below_sma(closes: pd.Series, start: date, end: date) -> tuple[int, int]:
    """Return how many trading days from `start` to `end` closed below their 200-day SMA, and the day count (R11)."""
    sma_slow = closes.rolling(config.SMA_SLOW).mean()
    window_closes = closes[pd.Timestamp(start) : pd.Timestamp(end)]
    is_below = window_closes < sma_slow[window_closes.index]
    return int(is_below.sum()), len(window_closes)


def run_backtest() -> dict:
    """Run the weekly technical backtest with its macro overlay and return the results file's contents.

    Raises ValueError when the snapshot can't cover the window, and
    FeedUnavailableError when FRED_API_KEY is not set.
    """
    closes = load_price_snapshot().pivot(index="date", columns="ticker", values="close")
    trading_days = [timestamp.date() for timestamp in closes.index]
    rebalance_dates = weekly_rebalance_dates(trading_days)
    daily_series, sahm_releases = fetch_macro_history()

    weekly_returns: dict[str, list[float]] = {name: [] for name in CURVE_NAMES}
    held_weights: dict[str, float] = {}
    for week_start, week_end in pairwise(rebalance_dates):
        period_returns = weekly_price_returns(closes, week_start, week_end)
        target_weights = technical_weights(week_start)
        cost = config.COST_BPS / 10_000 * turnover(target_weights, held_weights)
        technical_return = portfolio_return(target_weights, period_returns) - cost
        exposure = macro_exposure(week_start, daily_series, sahm_releases)

        weekly_returns[TECHNICAL_CURVE].append(technical_return)
        weekly_returns[OVERLAY_CURVE].append(exposure * technical_return)
        weekly_returns[config.BENCHMARK].append(float(period_returns[config.BENCHMARK]))
        weekly_returns[EQUAL_WEIGHT_CURVE].append(
            portfolio_return(EQUAL_UNIVERSE_WEIGHTS, period_returns)
        )
        held_weights = drifted_weights(target_weights, period_returns)

    benchmark_total_return = growth_of_one(weekly_returns[config.BENCHMARK])[-1] - 1
    growth_by_curve = {}
    metrics_by_curve = {}
    for curve_name, returns in weekly_returns.items():
        growth_by_curve[curve_name] = growth_of_one(returns)
        metrics_by_curve[curve_name] = curve_metrics(returns, benchmark_total_return)

    spy_days_below_sma200, window_days = count_days_below_sma(
        closes[config.BENCHMARK],
        rebalance_dates[0],
        rebalance_dates[-1],
    )
    return {
        "dates": [rebalance_date.isoformat() for rebalance_date in rebalance_dates],
        "growth": growth_by_curve,
        "metrics": metrics_by_curve,
        "spy_days_below_sma200": spy_days_below_sma200,
        "window_days": window_days,
    }


if __name__ == "__main__":
    load_dotenv()  # FRED_API_KEY from .env

    results = run_backtest()
    config.BACKTEST_RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n")

    for curve_name, metrics in results["metrics"].items():
        print(
            f"{curve_name}: total {metrics['total_return']:.1%}, "
            f"Sharpe {metrics['sharpe']:.2f}, max drawdown {metrics['max_drawdown']:.1%}"
        )
    print(f"wrote {config.BACKTEST_RESULTS_PATH}")
