"""Smoke checks: the reused external/ modules and stockstats import cleanly."""

from __future__ import annotations

import importlib

import numpy

EXTERNAL_MODULES = (
    "models",
    "broker_models",
    "sim",
    "execution",
    "limits",
    "construction",
    "cache",
)


def test_external_modules_import() -> None:
    """All 7 reused ai-hedge-fund modules in external/ import without error."""
    for module_name in EXTERNAL_MODULES:
        importlib.import_module(f"external.{module_name}")


def test_stockstats_imports_on_numpy_one() -> None:
    """stockstats imports cleanly on numpy 1.x (the PRD pins numpy 1.26)."""
    import stockstats

    assert stockstats
    assert numpy.__version__.split(".")[0] == "1"
