"""Seam 1: run_committee(snapshot) -> proposal, with a fake chat model (PRD Testing Decisions).

Setup: a fixed committed snapshot fixture and a fake chat model returning
canned structured outputs. Checks the outcomes a user would notice at this
seam: every persona's sector cap, per-name max and cash buffer hold on the
book the proposed orders actually produce, a raising analyst abstains
without sinking the run, the verdict's persona is the one recommended, and
a saved Sonnet run reads back intact.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

import books
import committee
import config
from committee import (
    CriticNotes,
    Drop,
    Narration,
    Note,
    PMView,
    Proposal,
    SentimentAnswer,
    SentimentView,
    Snapshot,
    Verdict,
    run_committee,
)
from external.broker_models import Order
from external.sim import SimBroker
from feeds import NewsArticle, TickerNews
from portfolio import Book
from signals import MacroChecklist, TechnicalScore

ALL_TICKERS = tuple(config.SECTOR_BY_TICKER)
FIXTURE_MARKS = {ticker: 100.0 for ticker in ALL_TICKERS}

# Semis and Big Tech carry a strong positive score so the per-name max and
# the sector caps actually bind; the rest sit in a narrow, mild-conviction
# range so they don't compete for the same limits.
_STRONG_TICKERS = {"NVDA", "AMD", "AVGO", "AAPL", "MSFT", "AMZN"}


def _technical_score(ticker: str) -> TechnicalScore:
    score = 0.9 if ticker in _STRONG_TICKERS else 0.2
    return TechnicalScore(
        score=score, rsi=50.0, macd_histogram=0.0, sma_fast=1.0, sma_slow=1.0
    )


FIXTURE_TECHNICAL = {ticker: _technical_score(ticker) for ticker in config.UNIVERSE}
FIXTURE_ARTICLE = NewsArticle(
    title="Quarterly results in line", source="Wire", published_at="20260925T120000"
)
FIXTURE_NEWS = {
    ticker: TickerNews(source="alpha_vantage", articles=[FIXTURE_ARTICLE])
    for ticker in config.UNIVERSE
}
FIXTURE_MACRO = MacroChecklist(
    flags={
        "yield_curve": False,
        "sahm_rule": False,
        "credit_spread": False,
        "vix": False,
        "spy_trend": False,
    },
    flag_count=0,
    regime="risk_on",
)
# Balanced already holds a within-sector mix that sits just under the 30%
# semis cap (NVDA 135, AMD 142, AVGO 143 at the $100 fixture mark): rotating
# toward this cycle's equal-weight semis targets sells 1 share of one name
# while buying several of another, so the $200 floor (F11) and the cap
# restore it can't override (F7, deviation #7) both fire for real.
FIXTURE_BOOKS = {
    "conservative": Book(cash=100_000.0, shares={}, avg_cost={}),
    "balanced": Book(
        cash=100_000.0,
        shares={"NVDA": 135, "AMD": 142, "AVGO": 143},
        avg_cost={},
    ),
    "aggressive": Book(cash=100_000.0, shares={}, avg_cost={}),
}


def _fixture_snapshot() -> Snapshot:
    return Snapshot(
        taken_at=datetime(2026, 9, 26, tzinfo=UTC),
        marks=FIXTURE_MARKS,
        technical=FIXTURE_TECHNICAL,
        news=FIXTURE_NEWS,
        fear_greed=20.0,  # below FEAR_GREED_LOW -> exercises the +0.10 tilt
        macro=FIXTURE_MACRO,
        books=FIXTURE_BOOKS,
    )


SENTIMENT_ANSWER = SentimentAnswer(
    views=[
        SentimentView(ticker=ticker, score=0.3, why="steady coverage")
        for ticker in config.UNIVERSE
    ]
)
NARRATION_ANSWER = Narration(
    notes=[Note(name=name, why="in line with the numbers") for name in config.UNIVERSE]
)
PM_ANSWER = PMView(
    memo="Leaning into semis and Big Tech; trimming energy on weak conviction.",
    drops=[Drop(ticker="XOM", reason="conviction too weak to size")],
)
CRITIC_ANSWER = CriticNotes(notes="Concentration risk in a handful of mega-cap names.")
VERDICT_ANSWER = Verdict(
    summary="Semis and Big Tech lead; follow the aggressive book.",
    persona="aggressive",
)


class FakeStructuredModel:
    """Returns one canned answer, or raises it, like a structured ChatAnthropic call."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def invoke(self, messages: list) -> dict:
        """Return the canned answer the way include_raw=True does, or raise it."""
        if isinstance(self._answer, Exception):
            raise self._answer
        return {
            "raw": AIMessage(content=""),
            "parsed": self._answer,
            "parsing_error": None,
        }


class FakeChatModel:
    """Stands in for ChatAnthropic: one canned answer per answer schema."""

    def __init__(self, answers: dict[type, object]) -> None:
        self._answers = answers

    def with_structured_output(
        self, schema: type, *, method: str, include_raw: bool
    ) -> FakeStructuredModel:
        """Return a structured model that answers with this schema's canned answer."""
        return FakeStructuredModel(self._answers[schema])


@pytest.fixture
def use_fake_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[dict[type, object]], None]:
    """Route committee.ask through a fake model and an isolated, throwaway cache dir."""

    def _install(answers: dict[type, object]) -> None:
        def build_fake_chat_model(model_name: str) -> FakeChatModel:
            return FakeChatModel(answers)

        monkeypatch.setattr(committee, "LLM_CACHE_DIR", tmp_path)
        monkeypatch.setattr(committee, "build_chat_model", build_fake_chat_model)

    return _install


def _book_after_orders(
    persona_key: str, orders: list[Order]
) -> tuple[dict[str, float], float]:
    """Fill `orders` against the persona's fixture book and return its resulting weights and cash."""
    starting_book = FIXTURE_BOOKS[persona_key]
    positions_value = sum(
        shares * FIXTURE_MARKS[ticker]
        for ticker, shares in starting_book.shares.items()
    )
    broker = SimBroker(cash=starting_book.cash + positions_value)
    for ticker, shares in starting_book.shares.items():
        broker.place_order(
            Order(
                ticker=ticker, side="buy", quantity=shares, price=FIXTURE_MARKS[ticker]
            )
        )
    for order in orders:
        broker.place_order(order)

    filled_equity = broker.cash() + sum(
        position.shares * FIXTURE_MARKS[position.ticker]
        for position in broker.positions().values()
    )
    weights = {
        ticker: position.shares * FIXTURE_MARKS[ticker] / filled_equity
        for ticker, position in broker.positions().items()
    }
    return weights, broker.cash()


def _assert_persona_limits_hold(proposal: Proposal) -> None:
    """Check the sector cap, per-name max and cash buffer on the book each persona's orders leave (Seam 1)."""
    for persona_key, persona in config.PERSONAS.items():
        weights, cash = _book_after_orders(persona_key, proposal.orders[persona_key])

        for ticker, weight in weights.items():
            if ticker in config.UNIVERSE:  # ETFs are exempt from the per-name max (Q9)
                assert weight <= persona.max_position_pct + 1e-9

        sector_totals = {sector: 0.0 for sector in config.SECTORS}
        for ticker, weight in weights.items():
            sector = config.SECTOR_BY_TICKER.get(ticker)
            if sector is not None:
                sector_totals[sector] += weight
        for sector_weight in sector_totals.values():
            assert sector_weight <= persona.sector_cap + 1e-9

        assert cash >= -1e-9


def test_every_persona_respects_its_limits(use_fake_model: Callable) -> None:
    """A normal run's three proposals all hold their sector cap, per-name max and cash buffer."""
    use_fake_model(
        {
            SentimentAnswer: SENTIMENT_ANSWER,
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
            Verdict: VERDICT_ANSWER,
        }
    )

    proposal = run_committee(_fixture_snapshot())

    _assert_persona_limits_hold(proposal)


def test_a_failing_analyst_abstains_and_the_run_completes(
    use_fake_model: Callable,
) -> None:
    """A raising sentiment call still lets the run finish, with sentiment marked abstained."""
    use_fake_model(
        {
            SentimentAnswer: TimeoutError("sentiment timed out"),
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
            Verdict: VERDICT_ANSWER,
        }
    )

    proposal = run_committee(_fixture_snapshot())

    assert proposal.sentiment.abstained is True
    assert proposal.technical.abstained is False
    _assert_persona_limits_hold(proposal)


@pytest.mark.parametrize(
    ("verdict_answer", "expected_persona"),
    [
        (VERDICT_ANSWER, "aggressive"),
        (TimeoutError("verdict timed out"), "balanced"),  # degrades like the PM
    ],
)
def test_recommend_follows_the_verdict_persona(
    use_fake_model: Callable,
    verdict_answer: object,
    expected_persona: str,
) -> None:
    """The recommended persona is the verdict's pick, and a failing verdict falls back to Balanced."""
    use_fake_model(
        {
            SentimentAnswer: SENTIMENT_ANSWER,
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
            Verdict: verdict_answer,
        }
    )

    proposal = run_committee(
        _fixture_snapshot(),
        question="Should I trim semis?",
        model_name=config.SONNET_DEMO_MODEL,
    )

    assert proposal.verdict.persona == expected_persona
    assert proposal.recommended_persona == expected_persona
    assert proposal.recommended_persona in proposal.orders
    assert proposal.question == "Should I trim semis?"
    assert proposal.model_name == config.SONNET_DEMO_MODEL


def test_a_saved_sonnet_run_reads_back(
    use_fake_model: Callable,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The newest saved run comes back from the local log with its proposal and verdict intact."""
    monkeypatch.delenv(books.MONGODB_URI_ENV_VAR, raising=False)
    monkeypatch.setattr(books, "SONNET_RUNS_PATH", tmp_path / "sonnet_runs.jsonl")
    use_fake_model(
        {
            SentimentAnswer: SENTIMENT_ANSWER,
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
            Verdict: VERDICT_ANSWER,
        }
    )
    proposal = run_committee(_fixture_snapshot(), model_name=config.SONNET_DEMO_MODEL)
    older_run = books.RunRecord(
        taken_at=datetime(2026, 9, 25, tzinfo=UTC),
        proposal=Proposal(target_weights={}, orders={}),
        nav={},
        spy_close=500.0,
    )
    newer_run = books.RunRecord(
        taken_at=datetime(2026, 9, 26, tzinfo=UTC),
        proposal=proposal,
        nav={persona_key: 100_000.0 for persona_key in config.PERSONAS},
        spy_close=500.0,
    )

    assert books.latest_sonnet_run() is None
    books.save_sonnet_run(older_run.model_dump(mode="json"))
    books.save_sonnet_run(newer_run.model_dump(mode="json"))
    saved_run = books.latest_sonnet_run()

    assert "saved_at" in saved_run
    assert Proposal.model_validate(saved_run["proposal"]) == proposal
