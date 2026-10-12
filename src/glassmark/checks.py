"""Compare phase 1 outputs against the synthetic ground truth.

The same routines drive the pytest suite and the special case report, so both
answer exactly the same question: did the pipeline keep the dots that are there
and drop the glare that is not a dot?
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geom import annulus_mask, erode_mask, overlap_fraction, polygon_mask
from .glare import luma
from .imgio import read_image, read_json, sidecar_path

MIN_DOT_COVERAGE = 0.45
"""A ground-truth dot must be covered at least this much by ``02_marker_mask``."""

MAX_GLARE_COVERAGE = 0.05
"""A glare-only region may cover at most this much of ``02_marker_mask``."""

GLARE_PEAK_MARGIN = 30.0
"""Cleaned brightness allowed above the local background, in luminance units."""

MAX_INK_OVERLAP_FOR_PEAK_CHECK = 0.25
"""A highlight that mostly sits on a dot cannot be measured for peak contrast:
its surroundings are paint, not plate.  Such regions are covered by the ink
readability checks instead."""

GLARE_REDUCTION_RATIO = 0.5
"""Fraction of the original contrast over the background that must be removed."""

MIN_INK_CHROMA_UNDER_GLARE = 55.0
"""Mean chroma a marker under a highlight must keep in ``03_cleaned``."""

RESULT_REGISTRY: List["CaseCheck"] = []
"""Every evaluated case, so the pytest session can print one table at the end."""


@dataclass
class Row:
    check: str
    expected: str
    actual: str
    output: str
    kind: str = "other"
    status: str = "PASS"

    @property
    def failed(self) -> bool:
        return self.status != "PASS"


@dataclass
class CaseCheck:
    stem: str
    case: str
    title: str
    output_dir: str
    rows: List[Row] = field(default_factory=list)

    @property
    def failures(self) -> List[Row]:
        return [row for row in self.rows if row.failed]

    @property
    def passed(self) -> bool:
        return not self.failures

    def summary(self) -> str:
        failed = len(self.failures)
        return f"{'PASS' if failed == 0 else 'FAIL'} ({len(self.rows) - failed}/{len(self.rows)} checks)"


def register(check: CaseCheck) -> CaseCheck:
    for index, existing in enumerate(RESULT_REGISTRY):
        if existing.stem == check.stem:
            RESULT_REGISTRY[index] = check
            return check
    RESULT_REGISTRY.append(check)
    return check


def rows_of_kind(checks: Sequence[CaseCheck], kind: str) -> List[Tuple[CaseCheck, Row]]:
    """All rows of one kind, paired with the case they belong to."""
    return [(check, row) for check in checks for row in check.rows if row.kind == kind]


def _core_mask(polygon: Sequence[Sequence[float]], shape, radius: float) -> np.ndarray:
    mask = polygon_mask(shape, [list(polygon)])
    erode_radius = int(round(max(1.0, min(4.0, 0.3 * radius))))
    eroded = erode_mask(mask, erode_radius)
    return eroded if int(np.count_nonzero(eroded)) >= 4 else mask


def _percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def evaluate_case(
    stem: str,
    output_root: str | Path,
    synthetic_root: str | Path,
    case: str = "",
    title: str = "",
) -> CaseCheck:
    """Run every ground-truth check for one image and return the rows."""
    output_dir = Path(output_root) / stem
    gt = read_json(sidecar_path(Path(synthetic_root) / f"{stem}.png"))
    check = CaseCheck(
        stem=stem,
        case=case or str(gt.get("case", "")),
        title=title or str(gt.get("title", "")),
        output_dir=str(output_dir),
    )

    input_image = read_image(output_dir / "00_input.png")
    marker_mask = cv2.cvtColor(read_image(output_dir / "02_marker_mask.png"), cv2.COLOR_BGR2GRAY)
    cleaned = read_image(output_dir / "03_cleaned.png")
    shape = input_image.shape[:2]
    input_luma = luma(input_image)
    cleaned_luma = luma(cleaned)

    dots = list(gt.get("dots", []))
    glare_regions = list(gt.get("glare_regions", []))

    ink_mask = polygon_mask(shape, [list(d["polygon"]) for d in dots])
    dot_masks = {str(d["id"]): polygon_mask(shape, [list(d["polygon"])]) for d in dots}

    for dot in dots:
        dot_id = str(dot["id"])
        colour = str(dot["color"])
        mask = dot_masks[dot_id]
        coverage = overlap_fraction(mask, marker_mask)
        check.rows.append(
            Row(
                check=f"dot {dot_id} ({colour}) present in 02_marker_mask",
                expected=f"coverage >= {MIN_DOT_COVERAGE:.2f}",
                actual=f"coverage {coverage:.3f}",
                output="02_marker_mask.png",
                kind="dot_present",
                status="PASS" if coverage >= MIN_DOT_COVERAGE else "FAIL",
            )
        )

        selected = (marker_mask > 0) & (mask > 0)
        from .markers import classify_ink_color

        detected_colour, hue_distance = classify_ink_color(cleaned[selected])
        colour_ok = detected_colour == colour
        check.rows.append(
            Row(
                check=f"dot {dot_id} colour read back from 03_cleaned",
                expected=colour,
                actual=f"{detected_colour} (hue distance {hue_distance:.0f} deg)",
                output="03_cleaned.png",
                kind="dot_color",
                status="PASS" if colour_ok else "FAIL",
            )
        )

        if float(dot.get("glare_overlap_fraction", 0.0)) > 0.0:
            values = cleaned[selected]
            mean_chroma = 0.0
            if values.size:
                spread = values.astype(np.float32)
                mean_chroma = float((spread.max(axis=1) - spread.min(axis=1)).mean())
            check.rows.append(
                Row(
                    check=f"dot {dot_id} still coloured under a highlight",
                    expected=f"mean chroma >= {MIN_INK_CHROMA_UNDER_GLARE:.0f}",
                    actual=f"mean chroma {mean_chroma:.1f}",
                    output="03_cleaned.png",
                    kind="ink_under_glare",
                    status="PASS" if mean_chroma >= MIN_INK_CHROMA_UNDER_GLARE else "FAIL",
                )
            )

    for region in glare_regions:
        region_id = str(region["id"])
        radius = float(region["radius"])
        core = _core_mask(region["polygon"], shape, radius)
        core_non_ink = cv2.bitwise_and(core, cv2.bitwise_not(ink_mask))
        ink_overlap = float(region.get("ink_overlap_fraction", 0.0))

        if ink_overlap == 0.0:
            coverage = overlap_fraction(core, marker_mask)
            check.rows.append(
                Row(
                    check=f"glare {region_id} ({region['kind']}) not kept as a marker",
                    expected=f"coverage in 02 <= {MAX_GLARE_COVERAGE:.2f}",
                    actual=f"coverage {coverage:.3f}",
                    output="02_marker_mask.png",
                    kind="glare_not_marker",
                    status="PASS" if coverage <= MAX_GLARE_COVERAGE else "FAIL",
                )
            )

        if not bool(region.get("recoverable", True)):
            continue
        if ink_overlap > MAX_INK_OVERLAP_FOR_PEAK_CHECK:
            continue
        measure = core_non_ink if int(np.count_nonzero(core_non_ink)) >= 4 else core
        if int(np.count_nonzero(measure)) == 0:
            continue

        # Background taken from a ring well outside the core, so the soft halo of
        # the highlight itself cannot inflate the reference level.
        ring = cv2.bitwise_and(
            annulus_mask(core, int(1.2 * radius), int(2.6 * radius)),
            cv2.bitwise_not(ink_mask),
        )
        ring_values = input_luma[ring > 0]
        background = float(np.median(ring_values)) if ring_values.size else float("nan")
        if background != background:
            background = float(np.median(input_luma[measure]))

        input_peak = _percentile(input_luma[measure > 0], 95.0)
        cleaned_peak = _percentile(cleaned_luma[measure > 0], 95.0)
        contrast = max(1.0, input_peak - background)
        reduction = (input_peak - cleaned_peak) / contrast
        reduced_enough = (
            cleaned_peak - background <= GLARE_PEAK_MARGIN
            or reduction >= GLARE_REDUCTION_RATIO
        )
        check.rows.append(
            Row(
                check=f"glare {region_id} ({region['kind']}) suppressed in 03_cleaned",
                expected=(
                    f"p95 <= background+{GLARE_PEAK_MARGIN:.0f} or >= "
                    f"{GLARE_REDUCTION_RATIO:.0%} contrast removed"
                ),
                actual=(
                    f"input p95 {input_peak:.0f}, cleaned p95 {cleaned_peak:.0f}, "
                    f"background {background:.0f}, removed {reduction:.0%}"
                ),
                output="03_cleaned.png",
                kind="glare_suppressed",
                status="PASS" if reduced_enough else "FAIL",
            )
        )

    register(check)
    return check


def evaluate_many(
    stems: Iterable[str],
    output_root: str | Path,
    synthetic_root: str | Path,
) -> List[CaseCheck]:
    return [evaluate_case(stem, output_root, synthetic_root) for stem in stems]


def format_table(checks: Sequence[CaseCheck], max_rows: Optional[int] = None) -> str:
    """A short human readable pass/fail table."""
    lines: List[str] = []
    header = f"{'run':32s} {'case':5s} {'status':8s} checks"
    lines.append(header)
    lines.append("-" * len(header))
    for check in checks:
        lines.append(
            f"{check.stem:32s} {check.case:5s} "
            f"{'PASS' if check.passed else 'FAIL':8s} {check.summary()}"
        )
    failed_rows = [row for check in checks for row in check.failures]
    if failed_rows:
        lines.append("")
        lines.append("failing checks:")
        limit = max_rows if max_rows is not None else len(failed_rows)
        for row in failed_rows[:limit]:
            lines.append(
                f"  - {row.check}\n      expected: {row.expected}\n      actual:   {row.actual}"
                f"\n      look at: {row.output}"
            )
    return "\n".join(lines)


def write_markdown_report(path: str | Path, checks: Sequence[CaseCheck], heading: str) -> Path:
    """Write the expected-versus-actual tables for all cases to a markdown file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {heading}", ""]
    lines.append(
        "| run | case | status | checks passed |\n| --- | --- | --- | --- |"
    )
    for check in checks:
        total = len(check.rows)
        passed = total - len(check.failures)
        lines.append(
            f"| `{check.stem}` | {check.case} | {'PASS' if check.passed else 'FAIL'} | {passed}/{total} |"
        )
    for check in checks:
        lines.append("")
        lines.append(f"## {check.stem} — {check.title or check.case}")
        lines.append("")
        lines.append("| check | expected | actual | output file | status |")
        lines.append("| --- | --- | --- | --- | --- |")
        for row in check.rows:
            lines.append(
                f"| {row.check} | {row.expected} | {row.actual} | `{row.output}` | {row.status} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
