"""Stage C of phase 1: suppress the glare.

Two complementary repairs:

* marker ink that a highlight washed out is rebuilt from the ink that is still
  visible around the highlight, so a dot never disappears completely;
* the remaining glare (highlights, streaks, gloss patches) is replaced by the
  surrounding plate using an edge-aware inpainting pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config
from .glare import BrightRegions, GlareSeparation, chroma, dilate, label_components


@dataclass
class SuppressedImage:
    image: np.ndarray
    restored_mask: np.ndarray
    inpainted_mask: np.ndarray
    ink_regions: List[Dict[str, object]] = field(default_factory=list)


def _hull_mask(patch: np.ndarray) -> np.ndarray:
    """Filled convex hull of a boolean patch."""
    contours, _ = cv2.findContours(
        patch.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = [c for c in contours if c.shape[0] >= 3]
    hull = np.zeros(patch.shape, dtype=np.uint8)
    if not contour_list:
        return hull
    cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contour_list))], 1)
    return hull


def _restore_color(
    image_patch: np.ndarray, evidence: np.ndarray, cfg: Phase1Config
) -> np.ndarray:
    """Ink colour to paint a washed out core, taken from the visible rim."""
    if not np.any(evidence):
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    chroma_patch = chroma(image_patch)
    peak = float(chroma_patch[evidence].max())
    threshold = max(cfg.ink_chroma_min, 0.8 * peak)
    selected = evidence & (chroma_patch >= threshold)
    if int(np.count_nonzero(selected)) < 4:
        selected = evidence
    return np.median(image_patch[selected].reshape(-1, 3).astype(np.float32), axis=0)


def restore_ink_under_glare(
    image: np.ndarray,
    bright: BrightRegions,
    separation: GlareSeparation,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Rebuild the part of a marker dot that a highlight washed out.

    The repair starts from the highlight, not from the ink: every specular
    region is examined together with the ink that is still visible in and around
    it.  The convex hull of that visible ink is the shape the dot used to have,
    so the pixels of the highlight that fall inside the hull are painted with the
    ink colour of the rim.  Restricting the guess to the highlight means a dot
    can never be grown into the plate by accident.
    """
    restored = np.zeros(image.shape[:2], dtype=np.uint8)
    records: List[Dict[str, object]] = []
    if not cfg.restore_ink:
        return image.copy(), restored, records

    output = image.copy()
    height, width = image.shape[:2]
    labels, table = label_components(bright.candidates)

    for entry in table:
        component = labels == int(entry["label"])
        area = int(entry["area"])
        if area < int(cfg.min_glare_area):
            continue

        x, y, box_w, box_h = (int(v) for v in entry["bbox"])  # type: ignore[misc]
        pad = 10
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(width, x + box_w + pad), min(height, y + box_h + pad)
        window: Tuple[int, int, int, int] = (x0, y0, x1, y1)

        glare_patch = component[y0:y1, x0:x1]
        nearby = dilate(glare_patch.astype(np.uint8) * 255, 6) > 0
        evidence = (separation.ink_evidence[y0:y1, x0:x1] > 0) & nearby
        evidence_area = int(np.count_nonzero(evidence))
        if evidence_area < int(cfg.min_ink_area):
            continue

        bridged = np.zeros(glare_patch.shape, dtype=bool)
        hull = _hull_mask(evidence)
        hull_area = int(np.count_nonzero(hull))
        if hull_area and hull_area <= cfg.hull_fill_max_ratio * evidence_area:
            bridged |= (hull > 0) & ~evidence

        close_radius = int(
            round(min(max(2.0, 0.5 * np.sqrt(evidence_area / np.pi)), cfg.restore_close_cap))
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
        )
        closed = (
            cv2.morphologyEx(evidence.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
        )
        bridged |= closed & ~evidence

        bridged &= glare_patch
        if not np.any(bridged):
            continue

        color = _restore_color(image[y0:y1, x0:x1], evidence, cfg)
        target = output[y0:y1, x0:x1]
        target[bridged] = np.clip(color, 0, 255).astype(np.uint8)
        output[y0:y1, x0:x1] = target

        region = restored[y0:y1, x0:x1]
        region[bridged] = 255
        restored[y0:y1, x0:x1] = region

        records.append(
            {
                "bbox": [x0, y0, int(x1 - x0), int(y1 - y0)],
                "glare_area": area,
                "visible_ink_area": evidence_area,
                "rebuilt_pixels": int(np.count_nonzero(bridged)),
                "color_bgr": [int(round(float(v))) for v in color],
            }
        )
    return output, restored, records


def inpaint_glare(
    image: np.ndarray,
    separation: GlareSeparation,
    restored: np.ndarray,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace whatever is left of the glare with the surrounding plate."""
    ink_final = cv2.bitwise_or(separation.ink_evidence, restored)
    protect = dilate(ink_final, cfg.ink_protect_dilate)
    mask = cv2.bitwise_and(
        dilate(separation.glare_mask, cfg.glare_dilate), cv2.bitwise_not(protect)
    )
    if not np.any(mask):
        return image.copy(), mask

    repaired = cv2.inpaint(image, mask, cfg.inpaint_radius, cv2.INPAINT_TELEA)
    # Feather the seam so the repaired patch does not show a hard outline.
    feather = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), 1.2)
    alpha = np.clip(feather * 1.6, 0.0, 1.0)[..., None]
    blended = image.astype(np.float32) * (1.0 - alpha) + repaired.astype(np.float32) * alpha
    blended = _limit_injected_chroma(blended, image, mask, cfg)
    return np.clip(blended, 0, 255).astype(np.uint8), mask


def _limit_injected_chroma(
    repaired: np.ndarray, original: np.ndarray, mask: np.ndarray, cfg: Phase1Config
) -> np.ndarray:
    """Never let a repair invent colour the plate did not have.

    Inpainting draws from the boundary of the hole, so a highlight beside a dot
    gets tinted by that dot's ink.  Where the original pixel was achromatic the
    repaired one is pulled back towards grey, which removes the coloured smear
    while keeping the texture the repair produced.
    """
    repaired_values = repaired.astype(np.float32)
    original_values = original.astype(np.float32)
    chroma_before = original_values.max(axis=2) - original_values.min(axis=2)
    chroma_after = repaired_values.max(axis=2) - repaired_values.min(axis=2)
    limit = np.maximum(cfg.injected_chroma_floor, chroma_before + cfg.injected_chroma_margin)
    excess = (mask > 0) & (chroma_after > limit)
    if not np.any(excess):
        return repaired_values

    ratio = (limit / np.maximum(chroma_after, 1e-3))[..., None]
    grey = repaired_values.mean(axis=2, keepdims=True)
    desaturated = grey + (repaired_values - grey) * ratio
    output = repaired_values.copy()
    output[excess] = desaturated[excess]
    return output


def suppress_glare(
    image: np.ndarray,
    bright: BrightRegions,
    separation: GlareSeparation,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> SuppressedImage:
    """Stage C: rebuild glared ink, then inpaint the remaining glare."""
    restored_image, restored_mask, ink_records = restore_ink_under_glare(
        image, bright, separation, cfg
    )
    cleaned, inpainted_mask = inpaint_glare(restored_image, separation, restored_mask, cfg)
    return SuppressedImage(
        image=cleaned,
        restored_mask=restored_mask,
        inpainted_mask=inpainted_mask,
        ink_regions=ink_records,
    )
