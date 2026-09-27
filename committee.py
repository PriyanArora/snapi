"""The investment committee: report schemas, the ask helper and the StateGraph.

One StateGraph, no memory or checkpointer (LangGraph shape): three analysts
run concurrently, then the Portfolio Manager, then `size` (code only, one
call per persona through `portfolio.size_persona`), then the critic, then
the verdict (which persona to follow) and `recommend` (code only). Each
analyst owns its own state key and is wrapped so a failure abstains instead
of discarding the whole run (story 9) -- a LangGraph superstep is
transactional, so one failing branch would otherwise cancel the others.

Every agent is shown the committee's question and runs on the model picked
for the run; both are optional graph inputs, defaulting to
`config.DEFAULT_QUESTION` and `config.get_model_name()`.

`run_committee(snapshot) -> proposal` is the PRD's entry point (seam 1);
`app.py` instead streams `build_graph()` in "updates" mode, so each card
fills as its node finishes, and calls `build_proposal` on the merged result.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

import config
from external.broker_models import Order
from external.cache import PromptCache, prompt_key
from external.models import Signal
from feeds import Fundamentals, TickerNews
from portfolio import (
    Book,
    blend_convictions,
    get_fear_greed_tilt,
    get_fundamentals_tilt,
    size_persona,
)
from signals import MacroChecklist, TechnicalScore

# R12: tracing off, pinned here rather than left to the shell default, since
# it would otherwise send prompts and portfolio data to a third-party cloud.
# LangSmith reads LANGSMITH_TRACING_V2 before LANGCHAIN_TRACING_V2 and
# LANGSMITH_TRACING, so this one name wins over any of them set in a shell.
os.environ["LANGSMITH_TRACING_V2"] = "false"

LLM_CACHE_DIR = Path(__file__).resolve().parent / ".cache" / "llm"
MAX_OUTPUT_TOKENS = 8000  # LLM call conventions: about 8k prevents truncation
NEWS_ITEMS_PER_TICKER = 10
ANALYSTS = ("sentiment", "technical", "macro")


@dataclass(frozen=True)
class Snapshot:
    """Frozen inputs for one committee cycle (PRD state shape)."""

    taken_at: datetime
    marks: dict[str, float]
    technical: dict[str, TechnicalScore]
    news: dict[str, TickerNews]
    fear_greed: float | None
    macro: MacroChecklist
    books: dict[str, Book]
    # Empty skips the per-stock fundamentals tilt, like a missing feed.
    fundamentals: dict[str, Fundamentals] = field(default_factory=dict)


class Report(BaseModel):
    """One analyst's per-ticker view: score, one-line why, and whether it abstained."""

    score: dict[str, float]
    why: dict[str, str]
    abstained: bool = False


class MacroReport(Report):
    """The macro analyst's report, plus the regime and flags its narrative explains."""

    regime: config.Regime
    flags: dict[str, bool]


class Drop(BaseModel):
    """The PM's veto of one ticker, with the stated reason (R1: drop-only)."""

    ticker: str
    reason: str


class PMView(BaseModel):
    """The Portfolio Manager's memo and drop-only vetoes; also its ask schema."""

    memo: str
    drops: list[Drop]


class Critique(BaseModel):
    """The critic's one-shot notes plus the code-computed >10%-down list (Q10)."""

    notes: str
    down_more_than_10pct: dict[str, list[str]]


class Verdict(BaseModel):
    """The final judge's short paragraph and the one persona to follow; also its ask schema."""

    summary: str
    persona: Literal["conservative", "balanced", "aggressive"]


class Proposal(BaseModel):
    """What Approve executes; also the run-log payload.

    The fields after `critique` default so run-log records written before
    them still parse.
    """

    target_weights: dict[str, dict[str, float]]
    orders: dict[str, list[Order]]
    sentiment: Report | None = None
    technical: Report | None = None
    macro: MacroReport | None = None
    pm: PMView | None = None
    critique: Critique | None = None
    question: str = config.DEFAULT_QUESTION
    model_name: str | None = None
    verdict: Verdict | None = None
    recommended_persona: str | None = None


class CommitteeState(TypedDict):
    """The graph's shared state for one run (PRD shape, plus `target_weights`).

    `target_weights` is not in the PRD's state shape, but `execute` needs
    the approved weights and an `Order` list alone can't rebuild them.
    `question` and `model_name` are optional inputs (see `_run_question`
    and `_run_model_name` for their defaults).
    """

    snapshot: Snapshot
    question: NotRequired[str]
    model_name: NotRequired[str]
    sentiment: Report
    technical: Report
    macro: MacroReport
    pm: PMView
    proposals: dict[str, list[Order]]
    target_weights: dict[str, dict[str, float]]
    critique: Critique
    verdict: Verdict
    recommended: str


# LLM answer schemas: lists of objects, never dict fields. The json_schema
# method forces additionalProperties: false and keeps only declared
# properties on every object, which turns a dict[str, float] field into an
# object the model can only ever fill with {}. List-of-object schemas survive
# intact; the nodes convert them into the Report state types above.
class SentimentView(BaseModel):
    """One ticker's sentiment score and why, as the LLM answers it."""

    ticker: str
    score: float
    why: str


class SentimentAnswer(BaseModel):
    """The sentiment analyst's full batched answer."""

    views: list[SentimentView]


class Note(BaseModel):
    """One narrated line: a ticker (technical) or a flag name (macro)."""

    name: str
    why: str


class Narration(BaseModel):
    """A narrate-only answer, shared by the technical and macro analysts."""

    notes: list[Note]


class CriticNotes(BaseModel):
    """The critic's one-shot counter-argument."""

    notes: str


# The six agent prompts, in graph order. The finance teammate owns the
# wording; these one-paragraph drafts stay here, in one obvious place.
SENTIMENT_PROMPT = (
    "You are the Sentiment analyst on an investment committee. For every "
    "ticker you are shown, read its recent headlines (and any Alpha Vantage "
    "sentiment score/label given) and score it from -1 (bearish) to +1 "
    "(bullish), with one plain-English line on why. If a ticker has no "
    "usable headlines, you may leave it out. Keep the committee's question, "
    "given first, in mind, but score each ticker from its own headlines and "
    "never add one."
)
TECHNICAL_PROMPT = (
    "You are the Technical analyst on an investment committee. You are given "
    "each ticker's already-computed score, RSI 14, MACD histogram and "
    "50/200-day SMA cross. Explain, in one plain-English line per ticker, "
    "what those numbers say about trend and momentum. Never invent a number "
    "or change the score. Keep the committee's question, given first, in "
    "mind, but still narrate every ticker and never add one."
)
MACRO_PROMPT = (
    "You are the Macro analyst on an investment committee. You are given the "
    "current regime, the flag count and which of the five macro flags are "
    "raised or clear. Explain, in one plain-English line per flag, what it "
    "means for risk appetite. Never invent a number or change the regime. "
    "Keep the committee's question, given first, in mind as you explain."
)
PM_PROMPT = (
    "You are the Portfolio Manager on an investment committee. You are given "
    "each ticker's technical score, sentiment score, blended conviction, the "
    "market-level Fear & Greed tilt and each stock's fundamentals tilt "
    "(forward P/E and debt), both already folded into that conviction, and "
    "the current regime. Write a short memo on the overall stance. You may "
    "drop a name the book should not hold, with a stated reason: a drop sets "
    "its target to zero and sells any shares held. You can never add a name "
    "or change a size. Keep the committee's question, given first, in mind "
    "as you write the memo."
)
CRITIC_PROMPT = (
    "You are the devil's-advocate critic on an investment committee. You are "
    "given the PM's memo, the orders sized for each persona, and which "
    "holdings are down more than 10% from cost. Give the strongest "
    "counter-argument to these orders in a short note. There is no "
    "automatic stop-loss here; a human decides. Keep the committee's "
    "question, given first, in mind as you argue."
)
VERDICT_PROMPT = (
    "You are the final judge of an investment committee. You are given each "
    "analyst's scores and one-line reasons, the macro regime, the PM's memo "
    "and drops, the orders sized for each persona (conservative, balanced "
    "and aggressive) and the critic's counter-argument. Write one short "
    "paragraph on the recommended direction that takes every agent's output "
    "into account, and pick exactly one persona to follow. You never write "
    "or change an order. Keep the committee's question, given first, in "
    "mind and answer it."
)


class AskError(Exception):
    """Raised when the model's structured answer can't be parsed; the node abstains."""


def build_chat_model(model_name: str) -> BaseChatModel:
    """Build the committee's Claude client for one model id.

    Effort goes to Sonnet 5 and Opus 5.5 only, because Haiku 4.5 rejects it;
    temperature, top-p and top-k are never set (LLM call conventions).
    Seam-1 tests monkeypatch this function to return a fake model.
    """
    accepts_effort = model_name.startswith(
        (config.SONNET_DEMO_MODEL, config.OPUS_MODEL)
    )
    if accepts_effort:
        return ChatAnthropic(
            model=model_name, max_tokens=MAX_OUTPUT_TOKENS, reasoning_effort="low"
        )

    return ChatAnthropic(model=model_name, max_tokens=MAX_OUTPUT_TOKENS)


def ask[AnswerT: BaseModel](
    agent: str,
    system: str,
    user: str,
    schema: type[AnswerT],
    model_name: str,
) -> AnswerT:
    """Return the model's structured answer, from the disk prompt cache when this exact prompt ran before.

    Identical (agent, model, system, user) hits the cache and costs $0 on
    rerun (story 16). Raises AskError on a parse failure, which the analyst
    wrapper or the PM/critic/verdict node turns into an abstain/degrade.
    """
    cache = PromptCache(LLM_CACHE_DIR)
    cache_key = prompt_key(agent, model_name, system, user)

    cached = cache.get(cache_key)
    if cached is not None:
        return schema.model_validate(cached["parsed"])

    structured_model = build_chat_model(model_name).with_structured_output(
        schema,
        method="json_schema",
        include_raw=True,
    )
    result = structured_model.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    if result["parsing_error"] is not None:
        raise AskError(
            f"{agent}: unparseable structured answer: {result['parsing_error']}"
        )

    answer = result["parsed"]
    # usage is kept beside the answer so a real cycle's cost can be measured
    cache.put(
        cache_key,
        {
            "parsed": answer.model_dump(mode="json"),
            "usage": result["raw"].usage_metadata,
        },
    )
    return answer


def _run_question(state: CommitteeState) -> str:
    """Return the question this run answers, or the default when the caller passed none."""
    return state.get("question") or config.DEFAULT_QUESTION


def _run_model_name(state: CommitteeState) -> str:
    """Return the model this run uses, or the SNAPI_MODEL/Haiku default when the caller passed none."""
    return state.get("model_name") or config.get_model_name()


def _ask_committee[AnswerT: BaseModel](
    state: CommitteeState,
    agent: str,
    system: str,
    facts: str,
    schema: type[AnswerT],
) -> AnswerT:
    """Ask one agent on the run's model, with the committee's question above its facts."""
    user = f"Committee question: {_run_question(state)}\n\n{facts}"
    return ask(agent, system, user, schema, _run_model_name(state))


def _news_prompt(news: dict[str, TickerNews]) -> str:
    """Render each universe ticker's headlines (and any AV sentiment) as plain text."""
    lines = []
    for ticker in config.UNIVERSE:
        ticker_news = news.get(ticker)
        if ticker_news is None:
            continue

        lines.append(f"{ticker} (source: {ticker_news.source}):")
        for article in ticker_news.articles[:NEWS_ITEMS_PER_TICKER]:
            article_line = f"- {article.title}"
            if article.sentiment_score is not None:
                article_line += (
                    f" [sentiment {article.sentiment_score:.2f} "
                    f"{article.sentiment_label}]"
                )
            lines.append(article_line)

    return "\n".join(lines)


def sentiment_node(state: CommitteeState) -> dict[str, Report]:
    """Sentiment analyst: the LLM scores each ticker's news; scores are clipped to [-1, 1] in code."""
    news = state["snapshot"].news
    answer = _ask_committee(
        state, "sentiment", SENTIMENT_PROMPT, _news_prompt(news), SentimentAnswer
    )
    tickers_with_headlines = {
        ticker for ticker, ticker_news in news.items() if ticker_news.articles
    }

    scores = {}
    reasons = {}
    for view in answer.views:
        if view.ticker not in config.UNIVERSE:
            continue  # only the universe it was shown can be scored
        if view.ticker not in tickers_with_headlines:
            continue  # a score with no headlines is invented; technical stands alone
        scores[view.ticker] = max(-1.0, min(1.0, view.score))
        reasons[view.ticker] = view.why

    return {"sentiment": Report(score=scores, why=reasons)}


def _technical_prompt(technical: dict[str, TechnicalScore]) -> str:
    """Render each universe ticker's code-computed score and raw indicators."""
    lines = []
    for ticker in config.UNIVERSE:
        indicators = technical[ticker]
        lines.append(
            f"{ticker}: score={indicators.score:.3f} rsi={indicators.rsi:.1f} "
            f"macd_histogram={indicators.macd_histogram:.3f} "
            f"sma50={indicators.sma_fast:.2f} sma200={indicators.sma_slow:.2f}"
        )

    return "\n".join(lines)


def technical_node(state: CommitteeState) -> dict[str, Report]:
    """Technical analyst: the score comes from code (R2); the LLM only narrates it."""
    technical = state["snapshot"].technical
    answer = _ask_committee(
        state, "technical", TECHNICAL_PROMPT, _technical_prompt(technical), Narration
    )

    reasons = {
        note.name: note.why for note in answer.notes if note.name in config.UNIVERSE
    }
    # Only the universe is scored, even if the snapshot also carries technical
    # scores for ETFs or the benchmark (F1, Q1).
    scores = {ticker: technical[ticker].score for ticker in config.UNIVERSE}
    return {"technical": Report(score=scores, why=reasons)}


def _macro_prompt(macro: MacroChecklist) -> str:
    """Render the regime, flag count and each flag as raised/clear -- derived flags only."""
    lines = [f"regime: {macro.regime}", f"flags raised: {macro.flag_count}"]
    for flag, is_raised in macro.flags.items():
        lines.append(f"{flag}: {'raised' if is_raised else 'clear'}")

    return "\n".join(lines)


def macro_node(state: CommitteeState) -> dict[str, MacroReport]:
    """Macro analyst: the regime and flags come from code (R2); the LLM only narrates them."""
    macro = state["snapshot"].macro
    answer = _ask_committee(
        state, "macro", MACRO_PROMPT, _macro_prompt(macro), Narration
    )

    reasons = {note.name: note.why for note in answer.notes if note.name in macro.flags}
    return {
        "macro": MacroReport(
            score={}, why=reasons, regime=macro.regime, flags=macro.flags
        )
    }


def _abstained_report(state_key: str, snapshot: Snapshot, error: Exception) -> Report:
    """Return the report an analyst falls back to on failure.

    Sentiment casts no vote. Technical and macro keep their code-computed
    numbers (R2) and lose only the narration, so the traded technical score
    stays the one the backtest uses (story 4).
    """
    why = {"error": f"abstained: {error}"}
    if state_key == "macro":
        macro = snapshot.macro
        return MacroReport(
            score={}, why=why, abstained=True, regime=macro.regime, flags=macro.flags
        )

    if state_key == "technical":
        # .items() rather than indexing UNIVERSE: this fallback must never raise.
        scores = {
            ticker: indicators.score
            for ticker, indicators in snapshot.technical.items()
            if ticker in config.UNIVERSE
        }
        return Report(score=scores, why=why, abstained=True)

    return Report(score={}, why=why, abstained=True)


def _abstain_on_error(
    state_key: str,
    node: Callable[[CommitteeState], dict],
) -> Callable[[CommitteeState], dict]:
    """Wrap an analyst so any failure returns a neutral abstained report instead of sinking the superstep.

    A LangGraph superstep is transactional: one raising task cancels its
    whole superstep, which would otherwise discard the other two analysts
    (story 9).
    """

    def run_or_abstain(state: CommitteeState) -> dict:
        """Return the analyst's update, or its abstained report when it raises."""
        try:
            return node(state)
        except Exception as error:
            logging.exception("%s analyst abstained", state_key)
            return {state_key: _abstained_report(state_key, state["snapshot"], error)}

    return run_or_abstain


def _analyst_signals(state: CommitteeState) -> list[Signal]:
    """Return one Signal per universe ticker each analyst report actually scored.

    An abstained sentiment report has an empty `score`, so it casts no vote
    and the technical score stands alone (story 9); an abstained technical
    report still carries its code-computed scores (R2).
    """
    as_of = state["snapshot"].taken_at.date().isoformat()
    signals = []
    for model_name in ("technical", "sentiment"):
        report = state[model_name]
        for ticker, score in report.score.items():
            signals.append(
                Signal(model_name=model_name, ticker=ticker, date=as_of, value=score)
            )

    return signals


def _pm_prompt(state: CommitteeState, convictions: dict[str, float]) -> str:
    """Render each ticker's technical/sentiment scores, blended conviction and both whys."""
    technical = state["technical"]
    sentiment = state["sentiment"]
    fear_greed = state["snapshot"].fear_greed
    tilt = get_fear_greed_tilt(fear_greed)
    lines = [
        f"regime: {state['snapshot'].macro.regime}",
        f"fear_greed: {fear_greed} (tilt {tilt:+.2f})",
    ]
    for ticker in config.UNIVERSE:
        fundamentals_tilt = get_fundamentals_tilt(
            ticker, state["snapshot"].fundamentals
        )
        lines.append(
            f"{ticker}: technical={technical.score.get(ticker)} "
            f"sentiment={sentiment.score.get(ticker)} "
            f"fundamentals_tilt={fundamentals_tilt:+.2f} "
            f"conviction={convictions.get(ticker, 0.0):.3f} "
            f"technical_why={technical.why.get(ticker, '')!r} "
            f"sentiment_why={sentiment.why.get(ticker, '')!r}"
        )

    return "\n".join(lines)


def pm_node(state: CommitteeState) -> dict[str, PMView]:
    """Portfolio Manager: a memo plus drop-only vetoes (R1); conviction is F12, in code."""
    convictions = blend_convictions(
        _analyst_signals(state),
        state["snapshot"].fear_greed,
        state["snapshot"].fundamentals,
    )
    try:
        answer = _ask_committee(
            state, "pm", PM_PROMPT, _pm_prompt(state, convictions), PMView
        )
    except Exception as error:
        logging.exception("PM view unavailable")
        return {"pm": PMView(memo=f"PM unavailable this cycle: {error}", drops=[])}

    kept_drops = [drop for drop in answer.drops if drop.ticker in config.UNIVERSE]
    return {"pm": PMView(memo=answer.memo, drops=kept_drops)}


def size_node(state: CommitteeState) -> dict:
    """Size all three personas from one committee view (S7); no LLM call here."""
    snapshot = state["snapshot"]
    signals = _analyst_signals(state)
    dropped_tickers = {drop.ticker for drop in state["pm"].drops}

    proposals = {}
    target_weights = {}
    for persona_key, persona in config.PERSONAS.items():
        result = size_persona(
            persona,
            signals,
            snapshot.macro.regime,
            snapshot.books[persona_key],
            snapshot.marks,
            snapshot.fear_greed,
            snapshot.fundamentals,
            dropped_tickers,
        )
        proposals[persona_key] = result.orders
        target_weights[persona_key] = result.target_weights

    return {"proposals": proposals, "target_weights": target_weights}


def _positions_down(book: Book, marks: dict[str, float]) -> list[str]:
    """Return the held tickers whose mark is more than 10% below their average cost (Q10)."""
    down_tickers = []
    for ticker, shares in book.shares.items():
        if shares <= 0:
            continue
        avg_cost = book.avg_cost.get(ticker)
        if not avg_cost:
            continue  # no cost basis on record -> nothing to compare against
        if marks[ticker] < avg_cost * (1 - config.POSITION_DOWN_FLAG_PCT):
            down_tickers.append(ticker)

    return sorted(down_tickers)


def _critic_prompt(
    memo: str,
    proposals: dict[str, list[Order]],
    down: dict[str, list[str]],
) -> str:
    """Render the PM memo, each persona's sized orders and the >10%-down list."""
    lines = [f"PM memo: {memo}"]
    for persona_key, orders in proposals.items():
        lines.append(f"{persona_key} orders:")
        for order in orders:
            lines.append(
                f"  {order.side} {order.quantity} {order.ticker} @ {order.price}"
            )
        lines.append(f"{persona_key} down more than 10%: {down[persona_key]}")

    return "\n".join(lines)


def critic_node(state: CommitteeState) -> dict[str, Critique]:
    """Critic: one-shot counter-argument; the >10%-down list is computed in code (Q10, F20)."""
    snapshot = state["snapshot"]
    down = {
        persona_key: _positions_down(book, snapshot.marks)
        for persona_key, book in snapshot.books.items()
    }

    try:
        answer = _ask_committee(
            state,
            "critic",
            CRITIC_PROMPT,
            _critic_prompt(state["pm"].memo, state["proposals"], down),
            CriticNotes,
        )
        notes = answer.notes
    except Exception as error:
        logging.exception("Critic notes unavailable")
        notes = f"Critic unavailable this cycle: {error}"

    return {"critique": Critique(notes=notes, down_more_than_10pct=down)}


def _verdict_prompt(state: CommitteeState) -> str:
    """Render every agent's output: analyst scores and whys, regime, PM memo and drops, sized orders, critique."""
    technical = state["technical"]
    sentiment = state["sentiment"]
    macro = state["macro"]
    pm = state["pm"]
    critique = state["critique"]

    lines = [f"regime: {macro.regime}"]
    for flag, why in macro.why.items():
        lines.append(f"macro {flag}: {why}")
    for ticker in config.UNIVERSE:
        lines.append(
            f"{ticker}: technical={technical.score.get(ticker)} "
            f"sentiment={sentiment.score.get(ticker)} "
            f"technical_why={technical.why.get(ticker, '')!r} "
            f"sentiment_why={sentiment.why.get(ticker, '')!r}"
        )
    for drop in pm.drops:
        lines.append(f"PM drops {drop.ticker}: {drop.reason}")

    # The memo, each persona's orders and its >10%-down list, as the critic saw them.
    lines.append(
        _critic_prompt(pm.memo, state["proposals"], critique.down_more_than_10pct)
    )
    lines.append(f"Critic: {critique.notes}")
    return "\n".join(lines)


def verdict_node(state: CommitteeState) -> dict[str, Verdict]:
    """Verdict: the final judge's paragraph and one persona to follow; it never writes or changes an order."""
    try:
        verdict = _ask_committee(
            state, "verdict", VERDICT_PROMPT, _verdict_prompt(state), Verdict
        )
    except Exception as error:
        logging.exception("Verdict unavailable")
        verdict = Verdict(
            summary=f"Verdict unavailable this cycle: {error}", persona="balanced"
        )

    return {"verdict": verdict}


def recommend_node(state: CommitteeState) -> dict[str, str]:
    """Recommend the verdict's persona, whose orders `size` already computed (no LLM, no new sizing)."""
    return {"recommended": state["verdict"].persona}


def build_graph() -> CompiledStateGraph:
    """Return START -> {sentiment, technical, macro} -> pm -> size -> critic -> verdict -> recommend -> END."""
    graph = StateGraph(CommitteeState)
    graph.add_node("sentiment", _abstain_on_error("sentiment", sentiment_node))
    graph.add_node("technical", _abstain_on_error("technical", technical_node))
    graph.add_node("macro", _abstain_on_error("macro", macro_node))
    graph.add_node("pm", pm_node)
    graph.add_node("size", size_node)
    graph.add_node("critic", critic_node)
    graph.add_node("verdict", verdict_node)
    graph.add_node("recommend", recommend_node)

    for analyst in ANALYSTS:
        graph.add_edge(START, analyst)

    # One waiting edge: pm runs once, after all three analysts finish.
    graph.add_edge(list(ANALYSTS), "pm")
    graph.add_edge("pm", "size")
    graph.add_edge("size", "critic")
    graph.add_edge("critic", "verdict")
    graph.add_edge("verdict", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()


def build_proposal(state: CommitteeState) -> Proposal:
    """Assemble the approvable Proposal from a finished run (also used after streaming in app.py)."""
    return Proposal(
        target_weights=state["target_weights"],
        orders=state["proposals"],
        sentiment=state["sentiment"],
        technical=state["technical"],
        macro=state["macro"],
        pm=state["pm"],
        critique=state["critique"],
        question=_run_question(state),
        model_name=_run_model_name(state),
        verdict=state["verdict"],
        recommended_persona=state["recommended"],
    )


def run_committee(
    snapshot: Snapshot,
    question: str = config.DEFAULT_QUESTION,
    model_name: str | None = None,
) -> Proposal:
    """Run one committee cycle for all three personas and return the proposal Approve will execute.

    `model_name` None means the SNAPI_MODEL/Haiku default.
    """
    graph_input = {
        "snapshot": snapshot,
        "question": question,
        "model_name": model_name or config.get_model_name(),
    }
    final_state = build_graph().invoke(graph_input)
    return build_proposal(final_state)
