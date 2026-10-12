"""Build the ``report.json`` written for every processed image.

The report is deliberately free of any product or vendor identity: it only
describes measurements taken from the image that was processed.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config
from .glare import BrightRegions, GlareSeparation, label_components
from .markers import MarkerBlobs
from .suppress import SuppressedImage


def _pixel_count(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def build_report(
    stem: str,
    input_path: str,
    image: np.ndarray,
    bright: BrightRegions,
    separation: GlareSeparation,
    suppressed: SuppressedImage,
    markers: MarkerBlobs,
    files: Dict[str, str],
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> Dict[str, object]:
    height, width = image.shape[:2]
    total = float(height * width)

    blob_masks: List[Optional[np.ndarray]] = [None] * len(markers.blobs)
    labels, table = label_components(markers.mask)
    for entry in table:
        box = [int(v) for v in entry["bbox"]]  # type: ignore[index]
        centroid = [float(v) for v in entry["centroid"]]  # type: ignore[index]
        best_index, best_distance = -1, 1e18
        for index, blob in enumerate(markers.blobs):
            blob_bbox = [int(v) for v in blob["bbox"]]  # type: ignore[index]
            distance = abs(blob_bbox[0] - box[0]) + abs(blob_bbox[1] - box[1])
            distance += 0.1 * abs(float(blob["centroid"][0]) - centroid[0])  # type: ignore[index]
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index >= 0:
            blob_masks[best_index] = labels == int(entry["label"])

    notes: List[str] = []
    blob_reports: List[Dict[str, object]] = []
    for index, blob in enumerate(markers.blobs):
        record = dict(blob)
        blob_mask = blob_masks[index]
        glare_pixels = 0
        if blob_mask is not None:
            glare_pixels = int(
                np.count_nonzero(blob_mask & (separation.glare_mask > 0))
            )
            coverage = glare_pixels / float(max(1, _pixel_count(blob_mask)))
        else:
            coverage = 0.0
        record["glare_pixels"] = glare_pixels
        record["glare_coverage"] = round(coverage, 4)
        record["glare_on_ink"] = bool(coverage >= cfg.glare_on_ink_note_min)
        if record["glare_on_ink"]:
            notes.append(
                f"marker {record['id']} ({record['color']}) sat under a highlight: "
                f"{round(100 * coverage, 1)}% of its area was bright; the ink was rebuilt "
                f"from the surrounding rim"
            )
            if record.get("filled_pixels"):
                notes.append(
                    f"marker {record['id']}: filled {record['filled_pixels']} washed out pixels"
                )
        blob_reports.append(record)

    for entry in bright.skipped:
        if entry.get("reason") == "too_large_exposure_not_recoverable":
            notes.append(
                "a bright region was skipped as exposure / backlight rather than recoverable "
                f"glare (area {entry['area']} px)"
            )
    for entry in separation.ink_dominated:
        notes.append(
            "a bright region was kept as marker ink, not glare "
            f"(ink fraction {entry['ink_fraction']})"
        )
    if suppressed.ink_regions:
        notes.append(
            f"{len(suppressed.ink_regions)} marker region(s) had their washed out core rebuilt"
        )
    if not markers.blobs:
        notes.append("no marker-like blob survived the filters")

    return {
        "tool": "GlassMark phase 1",
        "stage": "glare suppression and marker isolation",
        "input": input_path,
        "stem": stem,
        "image_size": [int(width), int(height)],
        "parameters": cfg.to_dict(),
        "stage_a_bright_regions": {
            "bright_pixel_count": _pixel_count(bright.mask),
            "bright_fraction": round(_pixel_count(bright.mask) / total, 6),
            "candidate_pixel_count": _pixel_count(bright.candidates),
            "region_count": len(bright.components),
            "regions": bright.components,
            "skipped_regions": bright.skipped,
        },
        "stage_b_glare_versus_ink": {
            "glare_pixel_count": _pixel_count(separation.glare_mask),
            "glare_fraction": round(_pixel_count(separation.glare_mask) / total, 6),
            "suppression_pixel_count": _pixel_count(separation.suppression_mask),
            "ink_evidence_pixel_count": _pixel_count(separation.ink_evidence),
            "ink_strong_pixel_count": _pixel_count(separation.ink_strong),
            "regions": separation.components,
            "bright_regions_kept_as_ink": separation.ink_dominated,
        },
        "stage_c_suppression": {
            "restored_ink_pixel_count": _pixel_count(suppressed.restored_mask),
            "inpainted_pixel_count": _pixel_count(suppressed.inpainted_mask),
            "rebuilt_ink_regions": suppressed.ink_regions,
        },
        "stage_d_markers": {
            "marker_count": len(markers.blobs),
            "mask_pixel_count": _pixel_count(markers.mask),
            "blobs": blob_reports,
            "dropped_candidates": markers.dropped,
        },
        "notes": notes,
        "files": files,
    }
