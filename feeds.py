"""Feeds: network calls for news sentiment, market mood and macro data.

Every function here returns plain data, with no LLM calls (S1; Tooling Stack
rows: News sentiment, Sentiment fallback, Market mood, Macro). Alpha Vantage
NEWS_SENTIMENT answers are cached to disk once a day, since only 25 of its
free calls are available and a ticker-per-call universe fetch spends 10 of
them (Risk #16); the CNN Fear & Greed reading is cached the same way so a
demo cycle never depends on it being reachable live.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal
from urllib.error import URLError

import requests
from fredapi import Fred

from config import UNIVERSE

CACHE_DIR = Path(__file__).resolve().parent / ".cache"

ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"
ALPACA_NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
FEAR_AND_GREED_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

# CNN's endpoint 403s without a browser-looking request (PRD: Modules > Feeds).
FEAR_AND_GREED_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://www.cnn.com/markets/fear-and-greed",
}

REQUEST_TIMEOUT_SECONDS = 10

# The free tier also caps burst rate at 1 request/second (Risk #16); this is
# pacing, not a retry.
ALPHA_VANTAGE_PACE_SECONDS = 1

# The macro checklist's non-price inputs (F13); SPY-vs-200-day-SMA lives in signals.py.
MACRO_SERIES_IDS = ("T10Y3M", "SAHMREALTIME", "BAMLH0A0HYM2", "VIXCLS")

NewsSource = Literal["alpha_vantage", "alpaca", "unavailable"]


class FeedUnavailableError(Exception):
    """Raised when a feed's key is missing or its response can't be used."""


@dataclass(frozen=True)
class NewsArticle:
    """One headline about a ticker, with Alpha Vantage's sentiment when we have it."""

    title: str
    source: str
    published_at: str
    relevance: float | None = None
    sentiment_score: float | None = None
    sentiment_label: str | None = None


@dataclass(frozen=True)
class TickerNews:
    """A ticker's recent articles, labelled with which feed they came from."""

    source: NewsSource
    articles: list[NewsArticle]


def _cache_path(name: str) -> Path:
    """Disk path for today's cached copy of `name`, one file per calendar day."""
    return CACHE_DIR / f"{name}-{date.today().isoformat()}.json"


def _read_cache(name: str) -> dict | list | None:
    """Return today's cached payload for `name`, or None on a miss or a corrupt file."""
    path = _cache_path(name)
    if not path.exists():
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None  # corrupt cache entry -> treat as a miss, refetch and overwrite


def _write_cache(name: str, payload: dict | list) -> None:
    """Persist `payload` as today's cache for `name`."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(name).write_text(json.dumps(payload))


def _match_ticker_sentiment(ticker_sentiments: list[dict], ticker: str) -> dict | None:
    """Return the Alpha Vantage `ticker_sentiment` entry for `ticker`, or None if absent."""
    for entry in ticker_sentiments:
        if entry.get("ticker") == ticker:
            return entry

    return None


def _ticker_news_from_alpha_vantage(payload: dict, ticker: str) -> TickerNews:
    """Build a TickerNews for `ticker` from one Alpha Vantage NEWS_SENTIMENT response."""
    articles = []
    for item in payload.get("feed", []):
        ticker_sentiment = _match_ticker_sentiment(
            item.get("ticker_sentiment", []), ticker
        )
        if ticker_sentiment is None:
            continue  # article was returned but doesn't actually mention this ticker

        articles.append(
            NewsArticle(
                title=item["title"],
                source=item["source"],
                published_at=item["time_published"],
                relevance=float(ticker_sentiment["relevance_score"]),
                sentiment_score=float(ticker_sentiment["ticker_sentiment_score"]),
                sentiment_label=ticker_sentiment["ticker_sentiment_label"],
            )
        )

    return TickerNews(source="alpha_vantage", articles=articles)


def fetch_alpha_vantage_news(ticker: str) -> TickerNews:
    """Return `ticker`'s recent articles and per-ticker sentiment from Alpha Vantage.

    One call per ticker: a multi-ticker call would only return articles that
    mention every ticker. Cached to disk once a day, so a dev rerun spends no
    call (S1, Risk #16).

    Raises FeedUnavailableError if the key is missing, the request times out,
    fails to connect, comes back with an HTTP error status, or its body
    isn't valid JSON, or Alpha Vantage answers HTTP 200 with a
    Note/Information/Error Message body (its rate-limit and misconfiguration
    signal).
    """
    cache_name = f"alpha_vantage-{ticker}"
    cached_payload = _read_cache(cache_name)
    if cached_payload is not None:
        return _ticker_news_from_alpha_vantage(cached_payload, ticker)

    api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise FeedUnavailableError("ALPHAVANTAGE_API_KEY is not set")

    time.sleep(ALPHA_VANTAGE_PACE_SECONDS)
    try:
        response = requests.get(
            ALPHA_VANTAGE_URL,
            params={"function": "NEWS_SENTIMENT", "tickers": ticker, "apikey": api_key},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise FeedUnavailableError(
            f"Alpha Vantage request failed for {ticker!r}: {error}"
        ) from error

    if "Note" in payload or "Information" in payload or "Error Message" in payload:
        raise FeedUnavailableError(
            f"Alpha Vantage rejected the request for {ticker!r}: {payload}"
        )

    _write_cache(cache_name, payload)
    return _ticker_news_from_alpha_vantage(payload, ticker)


def fetch_alpaca_news(tickers: tuple[str, ...]) -> dict[str, TickerNews]:
    """Return headline-only news per ticker from Alpaca News (no sentiment field).

    One multi-symbol call; the Sentiment agent scores these headlines itself.
    Raises FeedUnavailableError if the Alpaca keys are missing, the request
    times out, fails to connect, comes back with an HTTP error status (e.g.
    401), or its body isn't valid JSON.
    """
    api_key_id = os.environ.get("ALPACA_API_KEY_ID")
    api_secret_key = os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key_id or not api_secret_key:
        raise FeedUnavailableError(
            "ALPACA_API_KEY_ID/ALPACA_API_SECRET_KEY are not set"
        )

    try:
        response = requests.get(
            ALPACA_NEWS_URL,
            params={"symbols": ",".join(tickers), "limit": 50},
            headers={
                "APCA-API-KEY-ID": api_key_id,
                "APCA-API-SECRET-KEY": api_secret_key,
            },
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as error:
        raise FeedUnavailableError(f"Alpaca News request failed: {error}") from error

    articles_by_ticker: dict[str, list[NewsArticle]] = {
        ticker: [] for ticker in tickers
    }
    for item in payload.get("news", []):
        article = NewsArticle(
            title=item["headline"],
            source=item["source"],
            published_at=item["created_at"],
        )
        for symbol in item.get("symbols", []):
            if symbol in articles_by_ticker:
                articles_by_ticker[symbol].append(article)

    return {
        ticker: TickerNews(source="alpaca", articles=articles)
        for ticker, articles in articles_by_ticker.items()
    }


def fetch_universe_news(tickers: tuple[str, ...] = UNIVERSE) -> dict[str, TickerNews]:
    """Return news per ticker: Alpha Vantage where it works, Alpaca News for the rest.

    Each TickerNews.source labels the Sentiment agent which feed it got, since
    Alpaca's headlines carry no ticker-specific sentiment score to read.
    """
    news_by_ticker: dict[str, TickerNews] = {}
    failed_tickers = []
    for ticker in tickers:
        try:
            news_by_ticker[ticker] = fetch_alpha_vantage_news(ticker)
        except FeedUnavailableError:
            failed_tickers.append(ticker)

    if not failed_tickers:
        return news_by_ticker

    try:
        news_by_ticker.update(fetch_alpaca_news(tuple(failed_tickers)))
    except FeedUnavailableError:
        for ticker in failed_tickers:
            news_by_ticker[ticker] = TickerNews(source="unavailable", articles=[])

    return news_by_ticker


def fetch_fear_and_greed() -> float | None:
    """Return today's CNN Fear & Greed score (0-100), or None if the feed is unavailable.

    Cached to disk once a day. The raw score is kept unrounded so the sizing
    chain's below-25/above-75 tilt thresholds (Q6, F15) see the real value;
    the sizing chain skips the market-level tilt (Q6) when this returns None.
    """
    cached_payload = _read_cache("fear_and_greed")
    if cached_payload is not None:
        return cached_payload["score"]

    try:
        response = requests.get(
            FEAR_AND_GREED_URL,
            headers=FEAR_AND_GREED_HEADERS,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        score = float(response.json()["fear_and_greed"]["score"])
    except (requests.RequestException, KeyError, ValueError):
        return None  # unavailable -> caller skips the tilt rather than guessing

    _write_cache("fear_and_greed", {"score": score})
    return score


def fetch_macro_series() -> dict[str, float]:
    """Return the latest value of each macro checklist series from FRED.

    Covers T10Y3M, SAHMREALTIME, BAMLH0A0HYM2 and VIXCLS (F13); SPY against
    its 200-day SMA comes from the price snapshot instead.

    Raises FeedUnavailableError if FRED_API_KEY is missing, a series has no
    data, fredapi rejects the key or request (ValueError), or the network
    call fails (URLError).
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise FeedUnavailableError("FRED_API_KEY is not set")

    fred = Fred(api_key=api_key)
    latest_value_by_series = {}
    try:
        for series_id in MACRO_SERIES_IDS:
            series = fred.get_series(series_id).dropna()
            if series.empty:
                raise FeedUnavailableError(f"FRED returned no data for {series_id!r}")

            latest_value_by_series[series_id] = float(series.iloc[-1])
    except (ValueError, URLError) as error:
        raise FeedUnavailableError(f"FRED request failed: {error}") from error

    return latest_value_by_series
