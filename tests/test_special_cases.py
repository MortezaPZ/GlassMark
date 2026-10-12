"""Special cases A-H.

Every letter is processed and compared against its ground truth; a failure names
the output file to open.  The cases cover glare beside a dot, glare on a dot, a
full oil film, a warm gloss patch, a very large and a very small dot, a shiny
dot, clipped whites and two viewpoints of one plate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from glassmark.cases import special_case_map
from glassmark.checks import CaseCheck, evaluate_case
from glassmark.imgio import read_image
from glassmark.pipeline import run_folder

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "data" / "out" / "phase1_special"

LETTERS = special_case_map()
STEMS = [stem for letter in sorted(LETTERS) for stem in LETTERS[letter]]


@pytest.fixture(scope="module")
def processed() -> dict[str, CaseCheck]:
    if not any((SYNTHETIC / f"{stem}.png").exists() for stem in STEMS):
        from scripts.make_synthetic import main as make_main

        assert make_main(["--output", str(SYNTHETIC), "--quiet"]) == 0
    run_folder(SYNTHETIC, OUTPUT, stems=STEMS, verbose=False)
    return {stem: evaluate_case(stem, OUTPUT, SYNTHETIC) for stem in STEMS}


@pytest.mark.parametrize("stem", STEMS)
def test_special_case(processed: dict[str, CaseCheck], stem: str) -> None:
    check = processed[stem]
    if not check.passed:
        details = "\n".join(
            f"  {row.check}\n    expected {row.expected}\n    actual   {row.actual}\n"
            f"    open {OUTPUT / stem / row.output}"
            for row in check.failures
        )
        pytest.fail(f"{stem} ({check.case}):\n{details}")


def test_case_h_views_keep_the_same_dots(processed: dict[str, CaseCheck]) -> None:
    """Two angles of one plate must yield the same dot inventory."""
    inventories = []
    for stem in LETTERS["H"]:
        report = json.loads((OUTPUT / stem / "report.json").read_text(encoding="utf-8"))
        blobs = report["stage_d_markers"]["blobs"]
        inventories.append(sorted(str(blob["color"]) for blob in blobs))
    assert len(inventories) == 2, "case H needs two viewpoints"
    assert inventories[0] == inventories[1], (
        "the same plate produced different dots per viewpoint: "
        f"{inventories[0]} vs {inventories[1]}"
    )
    assert len(inventories[0]) >= 6, f"too few dots in case H: {inventories[0]}"


def test_masks_and_outputs_exist(processed: dict[str, CaseCheck]) -> None:
    for stem in STEMS:
        folder = OUTPUT / stem
        for name in (
            "00_input.png",
            "01_glare_mask.png",
            "02_marker_mask.png",
            "03_cleaned.png",
            "04_dots_only.png",
            "report.json",
        ):
            assert (folder / name).exists(), f"missing artefact {folder / name}"
        dots_only = read_image(folder / "04_dots_only.png")
        assert int(dots_only.max()) > 0, f"{folder / '04_dots_only.png'} is empty"
