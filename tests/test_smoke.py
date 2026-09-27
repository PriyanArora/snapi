"""Smoke checks: the reused external/ modules and stockstats import cleanly."""

from __future__ import annotations

import numpy


def test_external_modules_import() -> None:
    """All 7 reused ai-hedge-fund modules in external/ import without error."""
    from external import (
        broker_models,
        cache,
        construction,
        execution,
        limits,
        models,
        sim,
    )

    assert (
        broker_models
        and cache
        and construction
        and execution
        and limits
        and models
        and sim
    )


def test_stockstats_imports_on_numpy_one() -> None:
    """stockstats imports cleanly and numpy is pinned to major version 1 (PRD: numpy 1.26)."""
    import stockstats

    assert stockstats
    assert numpy.__version__.split(".")[0] == "1"
