#!/usr/bin/env python
"""Single entry point for the whole phase 1 loop.

One path from nothing to a verified result::

    python scripts/glassmark.py all

which is exactly ``synth`` + ``run`` + ``test``.  The individual steps are also
available separately:

    python scripts/glassmark.py synth
    python scripts/glassmark.py run
    python scripts/glassmark.py test
    python scripts/glassmark.py special-cases
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_step(script: str, extra: list[str] | None = None) -> int:
    command = [sys.executable, str(ROOT / "scripts" / script)] + (extra or [])
    print(f"$ {' '.join(command)}")
    return subprocess.call(command, cwd=str(ROOT))


def run_tests(extra: list[str] | None = None) -> int:
    command = [sys.executable, "-m", "pytest", "-q"] + (extra or ["tests"])
    print(f"$ {' '.join(command)}")
    return subprocess.call(command, cwd=str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GlassMark phase 1: generate, run and test in one path."
    )
    parser.add_argument(
        "step",
        choices=("all", "synth", "run", "test", "special-cases"),
        help="what to do; 'all' runs synth, run and test in order",
    )
    parser.add_argument("--input", default="data/synthetic")
    parser.add_argument("--output", default="data/out/phase1")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    status = 0

    if args.step in {"all", "synth"}:
        status = run_step("make_synthetic.py", ["--output", args.input])
        if status != 0:
            return status
    if args.step in {"all", "run"}:
        status = run_step(
            "run_phase1.py", ["--input", args.input, "--output", args.output]
        )
        if status != 0:
            return status
    if args.step == "special-cases":
        status = run_step("run_special_cases.py")
        return status
    if args.step in {"all", "test"}:
        status = run_tests()

    return status


if __name__ == "__main__":
    raise SystemExit(main())
