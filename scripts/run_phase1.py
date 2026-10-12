#!/usr/bin/env python
"""Run the GlassMark phase 1 pipeline on one photo or a whole folder.

For every input image ``<stem>.png`` the pipeline writes::

    <output>/<stem>/00_input.png        copy of the input
    <output>/<stem>/01_glare_mask.png   white = glare
    <output>/<stem>/02_marker_mask.png  white = kept marker dots
    <output>/<stem>/03_cleaned.png      glare reduced, dots still visible
    <output>/<stem>/04_dots_only.png    black background, only dots
    <output>/<stem>/report.json         counts, bounding boxes, notes

Example:
    python scripts/run_phase1.py --input data/synthetic --output data/out/phase1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from glassmark.config import DEFAULT_CONFIG  # noqa: E402
from glassmark.imgio import list_images  # noqa: E402
from glassmark.pipeline import run_folder, run_image  # noqa: E402

OVERRIDES = (
    "bright_luma",
    "bright_rel",
    "ink_chroma_min",
    "glare_dilate",
    "inpaint_radius",
    "marker_chroma_min",
    "min_marker_area",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the GlassMark phase 1 glare suppression pipeline."
    )
    parser.add_argument(
        "--input",
        default="data/synthetic",
        help="image file or folder of images (default: data/synthetic)",
    )
    parser.add_argument(
        "--output",
        default="data/out/phase1",
        help="output root; each image gets its own <stem> folder (default: data/out/phase1)",
    )
    parser.add_argument("--only", nargs="*", default=None, help="restrict to these stems")
    for name in OVERRIDES:
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=float,
            default=None,
            help=f"override the '{name}' threshold",
        )
    parser.add_argument("--quiet", action="store_true", help="only print the summary line")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides = {
        name: getattr(args, name)
        for name in OVERRIDES
        if getattr(args, name) is not None
    }
    int_overrides = {
        key: int(value)
        for key, value in overrides.items()
        if key in {"glare_dilate", "inpaint_radius", "min_marker_area"}
    }
    float_overrides = {key: float(value) for key, value in overrides.items() if key not in int_overrides}
    cfg = DEFAULT_CONFIG.with_overrides(**float_overrides, **int_overrides)

    input_path = Path(args.input)
    output_root = Path(args.output)

    if input_path.is_file():
        result = run_image(input_path, output_root, cfg)
        stage_d = result.report["stage_d_markers"]  # type: ignore[index]
        print(
            f"{result.stem:34s} markers={stage_d['marker_count']:2d} -> {result.output_dir}"
        )
        return 0

    if not input_path.exists():
        print(f"input not found: {input_path}")
        print("generate the synthetic set first: python scripts/make_synthetic.py")
        return 2

    images = list_images(input_path)
    if not images:
        print(f"no images in {input_path}; run: python scripts/make_synthetic.py")
        return 2

    results = run_folder(input_path, output_root, cfg, stems=args.only, verbose=not args.quiet)
    if not args.quiet:
        print(f"{len(results)} image(s) processed into {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
