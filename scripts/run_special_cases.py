#!/usr/bin/env python
"""Generate and process the special cases A-H, then report expected vs actual.

Each letter stands for one physical situation:

===== ======================================================================
A     glare next to a red dot, not overlapping it
B     glare on top of a blue dot
C     oil film over the whole plate, markers still readable
D     yellow dot against a yellowish specular patch
E     tiny green dot next to a large red dot
F     two similarly sized dots, one with a shine on it
G     strong overexposure, clipped whites
H     the same plate photographed from two angles (angles are not classified)
===== ======================================================================

Outputs land in ``data/out/phase1_special/<stem>/`` and the written report is
``data/out/phase1_special/SPECIAL_CASES.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glassmark.cases import special_case_map  # noqa: E402
from glassmark.checks import CaseCheck, evaluate_case  # noqa: E402
from glassmark.pipeline import run_folder  # noqa: E402

SPECIAL_TITLES = {
    "A": "glare next to a red dot (not overlapping)",
    "B": "glare on top of a blue dot",
    "C": "oil film over the whole plate, markers readable",
    "D": "yellow dot against a yellowish specular patch",
    "E": "tiny green dot next to a large red dot",
    "F": "two similar dots, one with a shine on it",
    "G": "strong overexposure (clipped whites)",
    "H": "same plate, two camera angles",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GlassMark special cases A-H.")
    parser.add_argument("--synthetic", default="data/synthetic")
    parser.add_argument("--output", default="data/out/phase1_special")
    parser.add_argument("--report", default="data/out/phase1_special/SPECIAL_CASES.md")
    return parser


def _table_line(letter: str, stem: str, check: CaseCheck) -> str:
    status = "PASS" if check.passed else "FAIL"
    return f"| {letter} | `{stem}` | {SPECIAL_TITLES.get(letter, '')} | {status} |"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    synthetic = Path(args.synthetic)
    output = Path(args.output)
    report_path = Path(args.report)

    letters = special_case_map()
    stems = [stem for letter in sorted(letters) for stem in letters[letter]]
    if not stems:
        print("no special cases defined")
        return 1

    missing = [stem for stem in stems if not (synthetic / f"{stem}.png").exists()]
    if missing:
        from make_synthetic import main as make_main

        make_main(["--output", str(synthetic), "--quiet"])
    run_folder(synthetic, output, stems=stems, verbose=False)

    checks = [evaluate_case(stem, output, synthetic) for stem in stems]
    by_stem = {check.stem: check for check in checks}

    lines = ["# GlassMark phase 1 — special cases A–H", ""]
    lines.append("| letter | run | situation | result |")
    lines.append("| --- | --- | --- | --- |")
    for letter in sorted(letters):
        for stem in letters[letter]:
            lines.append(_table_line(letter, stem, by_stem[stem]))

    for letter in sorted(letters):
        lines.append("")
        lines.append(f"## {letter} — {SPECIAL_TITLES.get(letter, '')}")
        for stem in letters[letter]:
            check = by_stem[stem]
            lines.append("")
            lines.append(f"### `{stem}` — {'passed' if check.passed else 'FAILED'}")
            lines.append("")
            lines.append("| check | expected | actual | open | result |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in check.rows:
                lines.append(
                    f"| {row.check} | {row.expected} | {row.actual} | "
                    f"`{output / stem / row.output}` | {row.status} |"
                )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"{'letter':6s} {'run':34s} {'result':7s} open")
    print("-" * 92)
    failed = 0
    for letter in sorted(letters):
        for stem in letters[letter]:
            check = by_stem[stem]
            failed += 0 if check.passed else 1
            print(
                f"{letter:6s} {stem:34s} {'PASS' if check.passed else 'FAIL':7s} "
                f"{(output / stem / '03_cleaned.png')}"
            )
    print(f"\nreport: {report_path}")
    print(f"{len(checks) - failed}/{len(checks)} special case runs passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
