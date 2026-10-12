"""Phase 1 pipeline: one photo in, five artefacts plus a report out.

Stages (kept as separate functions so each can be tested on its own):

* ``glare.find_bright_regions``      - a) very bright / specular candidates
* ``glare.separate_glare_from_ink``  - b) glare versus coloured marker ink
* ``suppress.suppress_glare``        - c) rebuild glared ink, inpaint the rest
* ``markers.detect_markers``         - d) keep only marker-like blobs
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config
from .glare import find_bright_regions, separate_glare_from_ink
from .imgio import read_image, write_image, write_json
from .markers import detect_markers
from .report import build_report
from .suppress import suppress_glare

OUTPUT_NAMES = {
    "input": "00_input.png",
    "glare": "01_glare_mask.png",
    "markers": "02_marker_mask.png",
    "cleaned": "03_cleaned.png",
    "dots": "04_dots_only.png",
    "report": "report.json",
}


@dataclass
class Phase1Result:
    stem: str
    input_path: str
    output_dir: Path
    report: Dict[str, object] = field(default_factory=dict)
    files: Dict[str, str] = field(default_factory=dict)


def run_image(
    image_path: str | Path,
    output_root: str | Path,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> Phase1Result:
    """Process one photo and write every phase 1 artefact below ``output_root/stem``."""
    image_path = Path(image_path)
    stem = image_path.stem
    output_dir = Path(output_root) / stem
    output_dir.mkdir(parents=True, exist_ok=True)

    image = read_image(image_path)

    bright = find_bright_regions(image, cfg)
    separation = separate_glare_from_ink(image, bright, cfg)
    suppressed = suppress_glare(image, bright, separation, cfg)
    markers = detect_markers(suppressed.image, cfg)

    dots_only = np.zeros_like(suppressed.image)
    keep = markers.mask > 0
    dots_only[keep] = suppressed.image[keep]

    files = {
        "input": OUTPUT_NAMES["input"],
        "glare_mask": OUTPUT_NAMES["glare"],
        "marker_mask": OUTPUT_NAMES["markers"],
        "cleaned": OUTPUT_NAMES["cleaned"],
        "dots_only": OUTPUT_NAMES["dots"],
        "report": OUTPUT_NAMES["report"],
    }

    write_image(output_dir / OUTPUT_NAMES["input"], image)
    # 01 shows the glare that survives stage B, i.e. what the pipeline treats as
    # reflected light; bright pixels that are simply marker ink are not glare.
    write_image(output_dir / OUTPUT_NAMES["glare"], separation.glare_mask)
    write_image(output_dir / OUTPUT_NAMES["markers"], markers.mask)
    write_image(output_dir / OUTPUT_NAMES["cleaned"], suppressed.image)
    write_image(output_dir / OUTPUT_NAMES["dots"], dots_only)

    report = build_report(
        stem=stem,
        input_path=str(image_path),
        image=image,
        bright=bright,
        separation=separation,
        suppressed=suppressed,
        markers=markers,
        files=files,
        cfg=cfg,
    )
    for key, filename in files.items():
        report.setdefault("files_absolute", {})[key] = str((output_dir / filename).resolve())
    write_json(output_dir / OUTPUT_NAMES["report"], report)

    return Phase1Result(
        stem=stem, input_path=str(image_path), output_dir=output_dir,
        report=report, files=files,
    )


def run_folder(
    input_dir: str | Path,
    output_root: str | Path,
    cfg: Phase1Config = DEFAULT_CONFIG,
    stems: Optional[Sequence[str]] = None,
    verbose: bool = True,
) -> List[Phase1Result]:
    """Process every image in a folder (optionally restricted to given stems)."""
    from .imgio import list_images

    wanted = set(stems) if stems else None
    results: List[Phase1Result] = []
    for image_path in list_images(input_dir):
        if wanted is not None and image_path.stem not in wanted:
            continue
        result = run_image(image_path, output_root, cfg)
        stage_a = result.report["stage_a_bright_regions"]  # type: ignore[index]
        stage_d = result.report["stage_d_markers"]  # type: ignore[index]
        if verbose:
            print(
                f"{result.stem:34s} glare_regions={stage_a['region_count']:2d} "
                f"markers={stage_d['marker_count']:2d} "
                f"dropped={len(stage_d['dropped_candidates']):2d}"
            )
        results.append(result)
    return results
