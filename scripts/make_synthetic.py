#!/usr/bin/env python
"""Generate the synthetic plate photos and their ground truth sidecars.

Example:
    python scripts/make_synthetic.py
    python scripts/make_synthetic.py --output data/synthetic --only case_A_glare_next_to_red
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glassmark.cases import all_cases  # noqa: E402
from glassmark.imgio import write_image, write_json  # noqa: E402
from glassmark.synth import render  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate GlassMark synthetic plates.")
    parser.add_argument(
        "--output",
        default="data/synthetic",
        help="folder that receives <stem>.png and <stem>.json (default: data/synthetic)",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="restrict generation to these stems",
    )
    parser.add_argument("--quiet", action="store_true", help="do not print one line per image")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = set(args.only) if args.only else None

    written = 0
    for case in all_cases():
        if keep is not None and case.stem not in keep:
            continue
        rendered = render(case.stem, case.case, case.title, case.spec, case.camera)
        image_path = output_dir / f"{case.stem}.png"
        write_image(image_path, rendered.image)
        write_json(output_dir / f"{case.stem}.json", rendered.meta)
        written += 1
        if not args.quiet:
            dots = len(rendered.meta["dots"])  # type: ignore[arg-type]
            glare = len(rendered.meta["glare_regions"])  # type: ignore[arg-type]
            print(
                f"{case.stem:34s} case {case.case:4s} dots={dots} glare_regions={glare} "
                f"order={case.spec.order}"
            )

    if written == 0:
        print("nothing generated: check --only")
        return 1
    if not args.quiet:
        print(f"{written} synthetic image(s) written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
