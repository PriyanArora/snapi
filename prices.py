"""Price snapshot: download a committed CSV of daily bars and load it back.

The snapshot covers the 10-stock universe, the 7 sector ETFs and SPY (R9:
yfinance rate-limits per IP, so the demo runs off a committed file instead of
a live call). `download_price_snapshot` refreshes that file; the rest of the
app reads it through `load_price_snapshot` and the small helpers below.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import BENCHMARK, ETF_BY_SECTOR, UNIVERSE

ALL_TICKERS = UNIVERSE + tuple(ETF_BY_SECTOR.values()) + (BENCHMARK,)

SNAPSHOT_PATH = Path(__file__).resolve().parent / "data" / "prices.csv"

# ~2 years of history: a 1-year analysis window plus the 200-day SMA warm-up.
LOOKBACK_DAYS = 730

SNAPSHOT_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


class PriceDownloadError(Exception):
    """Raised when yfinance returns no usable bars for a ticker."""


def download_price_snapshot(
    tickers: tuple[str, ...] = ALL_TICKERS,
    end: date | None = None,
    out_path: Path = SNAPSHOT_PATH,
) -> None:
    """Download daily bars for `tickers` and write them to `out_path` as tidy long CSV.

    Raises PriceDownloadError if any ticker's download is empty or all-NaN.
    """
    end = end or date.today()
    start = end - timedelta(days=LOOKBACK_DAYS)
    end_exclusive = end + timedelta(days=1)  # yfinance's end date is exclusive

    ticker_frames = []
    for ticker in tickers:
        bars = yf.Ticker(ticker).history(
            start=start,
            end=end_exclusive,
            auto_adjust=True,
        )
        if bars.empty or bars["Close"].isna().all():
            raise PriceDownloadError(f"yfinance returned no usable data for {ticker!r}")

        tidy_bars = bars[["Open", "High", "Low", "Close", "Volume"]].reset_index()
        tidy_bars.insert(1, "ticker", ticker)
        tidy_bars.columns = SNAPSHOT_COLUMNS
        tidy_bars["date"] = tidy_bars["date"].dt.strftime("%Y-%m-%d")
        ticker_frames.append(tidy_bars)

    snapshot = pd.concat(ticker_frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_csv(out_path, index=False)


@lru_cache(maxsize=1)
def load_price_snapshot() -> pd.DataFrame:
    """Read the committed price snapshot once per process, cached in memory."""
    return pd.read_csv(SNAPSHOT_PATH, parse_dates=["date"])


def close_history(ticker: str, as_of: date | None = None) -> pd.Series:
    """Return `ticker`'s daily close prices indexed by date, optionally up to `as_of`."""
    snapshot = load_price_snapshot()
    closes = snapshot[snapshot["ticker"] == ticker].set_index("date")["close"]
    if as_of is not None:
        closes = closes[closes.index <= pd.Timestamp(as_of)]

    return closes.sort_index()


def latest_marks(as_of: date | None = None) -> dict[str, float]:
    """Return the last close price per ticker, optionally as of a given date."""
    snapshot = load_price_snapshot()
    if as_of is not None:
        snapshot = snapshot[snapshot["date"] <= pd.Timestamp(as_of)]

    last_rows = snapshot.sort_values("date").groupby("ticker").tail(1)
    return dict(zip(last_rows["ticker"], last_rows["close"], strict=True))


if __name__ == "__main__":
    download_price_snapshot()
    load_price_snapshot.cache_clear()

    marks = latest_marks()
    assert set(marks) == set(ALL_TICKERS), "snapshot is missing a ticker"
    assert close_history(BENCHMARK).iloc[-1] == marks[BENCHMARK], "stale cache"

    row_count = len(load_price_snapshot())
    print(f"wrote {SNAPSHOT_PATH} ({row_count} rows, {len(marks)} tickers)")
