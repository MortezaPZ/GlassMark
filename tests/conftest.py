"""Test configuration: import path and the pass/fail table printed at the end."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (str(ROOT), str(ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)


def pytest_terminal_summary(terminalreporter) -> None:  # pragma: no cover - reporting only
    """Print one short table summarising every evaluated synthetic image."""
    try:
        from glassmark.checks import RESULT_REGISTRY, format_table
    except Exception:  # pragma: no cover - the suite failed before importing
        return
    if not RESULT_REGISTRY:
        return

    terminalreporter.write_sep("=", "phase 1 checks per image")
    for line in format_table(RESULT_REGISTRY).splitlines():
        terminalreporter.write_line(line)
