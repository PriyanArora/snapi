"""Seam 1: run_committee(snapshot) -> proposal, with a fake chat model (PRD Testing Decisions).

Setup: a fixed committed snapshot fixture and a fake chat model returning
canned structured outputs. Checks the outcomes a user would notice at this
seam: every persona's sector cap, per-name max and cash buffer hold, a
raising analyst abstains without sinking the run, and a raising PM/critic
degrades instead of crashing (D6 -- the only thing that exercises that
try/except branch).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from langchain_core.messages import AIMessage

import committee
import config
from committee import (
    CriticNotes,
    Drop,
    Narration,
    Note,
    PMView,
    SentimentAnswer,
    SentimentView,
    Snapshot,
    run_committee,
)
from feeds import TickerNews
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
FIXTURE_NEWS = {
    ticker: TickerNews(source="alpha_vantage", articles=[])
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
FIXTURE_BOOKS = {
    persona_key: Book(cash=100_000.0, shares={}, avg_cost={})
    for persona_key in config.PERSONAS
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


class FakeStructuredModel:
    """Returns one canned answer, or raises it, like a structured ChatAnthropic call."""

    def __init__(self, answer: object) -> None:
        self._answer = answer

    def invoke(self, messages: list) -> dict:
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
        return FakeStructuredModel(self._answers[schema])


@pytest.fixture
def use_fake_model(monkeypatch, tmp_path):
    """Route committee.ask through a fake model and an isolated, throwaway cache dir."""

    def _install(answers: dict[type, object]) -> None:
        monkeypatch.setattr(committee, "LLM_CACHE_DIR", tmp_path)
        monkeypatch.setattr(committee, "chat_model", lambda: FakeChatModel(answers))

    return _install


def _assert_persona_limits_hold(proposal) -> None:
    """Check the sector cap, per-name max and cash buffer for every persona (Seam 1)."""
    for persona_key, persona in config.PERSONAS.items():
        weights = proposal.target_weights[persona_key]

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

        assert sum(weights.values()) <= persona.max_invested_pct + 1e-9


def test_every_persona_respects_its_limits(use_fake_model) -> None:
    """A normal run's three proposals all hold their sector cap, per-name max and cash buffer."""
    use_fake_model(
        {
            SentimentAnswer: SENTIMENT_ANSWER,
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
        }
    )

    proposal = run_committee(_fixture_snapshot())

    _assert_persona_limits_hold(proposal)


def test_a_failing_analyst_abstains_and_the_run_completes(use_fake_model) -> None:
    """A raising sentiment call still lets the run finish, with sentiment marked abstained."""
    use_fake_model(
        {
            SentimentAnswer: TimeoutError("sentiment timed out"),
            Narration: NARRATION_ANSWER,
            PMView: PM_ANSWER,
            CriticNotes: CRITIC_ANSWER,
        }
    )

    proposal = run_committee(_fixture_snapshot())

    assert proposal.sentiment.abstained is True
    assert proposal.technical.abstained is False
    _assert_persona_limits_hold(proposal)


def test_pm_and_critic_degrade_instead_of_crashing(use_fake_model) -> None:
    """A raising PM/critic call degrades to a safe default (no veto) instead of crashing the run."""
    use_fake_model(
        {
            SentimentAnswer: SENTIMENT_ANSWER,
            Narration: NARRATION_ANSWER,
            PMView: RuntimeError("pm unavailable"),
            CriticNotes: RuntimeError("critic unavailable"),
        }
    )

    proposal = run_committee(_fixture_snapshot())

    assert proposal.pm.drops == []
    assert "critic unavailable" in proposal.critique.notes.lower()
    assert proposal.critique.down_more_than_10pct.keys() == config.PERSONAS.keys()
    _assert_persona_limits_hold(proposal)
