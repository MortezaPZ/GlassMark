"""Phase 1 tests, run on the synthetic plates.

The suite must fail when

* a known marker dot is missing from ``02_marker_mask.png``,
* a pure glare blob (no ink in the ground truth) is kept as a marker,
* ``03_cleaned.png`` still shows a large specular peak on a glare-only region.

The same checks also cover the colour read back from the cleaned image and the
readability of a dot that sat under a highlight.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from glassmark.cases import all_cases
from glassmark.checks import CaseCheck, evaluate_case, rows_of_kind
from glassmark.imgio import list_images
from glassmark.pipeline import run_folder

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "data" / "out" / "phase1"


@pytest.fixture(scope="session")
def synthetic_images() -> list[Path]:
    images = list_images(SYNTHETIC) if SYNTHETIC.exists() else []
    if not images:
        from scripts.make_synthetic import main as make_main

        assert make_main(["--output", str(SYNTHETIC), "--quiet"]) == 0
        images = list_images(SYNTHETIC)
    return images


@pytest.fixture(scope="session")
def processed(synthetic_images: list[Path]) -> list[CaseCheck]:
    run_folder(SYNTHETIC, OUTPUT, verbose=False)
    stems = [case.stem for case in all_cases()]
    return [evaluate_case(stem, OUTPUT, SYNTHETIC) for stem in stems]


def _failures(processed: list[CaseCheck], kind: str) -> list[str]:
    messages: list[str] = []
    for check, row in rows_of_kind(processed, kind):
        if row.failed:
            messages.append(
                f"{check.stem}: {row.check}\n"
                f"    expected {row.expected}\n"
                f"    actual   {row.actual}\n"
                f"    open {row.output}"
            )
    return messages


def test_known_dots_are_kept(processed: list[CaseCheck]) -> None:
    messages = _failures(processed, "dot_present")
    assert not messages, "marker dots missing from 02_marker_mask:\n" + "\n".join(messages)


def test_glare_only_blobs_are_not_kept(processed: list[CaseCheck]) -> None:
    messages = _failures(processed, "glare_not_marker")
    assert not messages, "glare kept as a marker in 02_marker_mask:\n" + "\n".join(messages)


def test_glare_peaks_are_suppressed(processed: list[CaseCheck]) -> None:
    messages = _failures(processed, "glare_suppressed")
    assert not messages, "specular peak survived in 03_cleaned:\n" + "\n".join(messages)


def test_dot_colours_are_readable(processed: list[CaseCheck]) -> None:
    messages = _failures(processed, "dot_color")
    assert not messages, "dot colour unreadable in 03_cleaned:\n" + "\n".join(messages)


def test_ink_survives_a_highlight(processed: list[CaseCheck]) -> None:
    messages = _failures(processed, "ink_under_glare")
    assert not messages, "a highlight deleted a marker:\n" + "\n".join(messages)


def test_every_case_has_rows(processed: list[CaseCheck]) -> None:
    assert len(processed) == len(all_cases())
    for check in processed:
        assert check.rows, f"no checks were produced for {check.stem}"
