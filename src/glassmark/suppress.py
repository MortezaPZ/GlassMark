"""Stage C of phase 1: suppress the glare.

Two complementary repairs:

* marker ink that a highlight washed out is rebuilt from the ink that is still
  visible around the highlight, so a dot never disappears completely;
* the remaining glare (highlights, streaks, gloss patches) is replaced by the
  surrounding plate using an edge-aware inpainting pass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config
from .glare import BrightRegions, GlareSeparation, chroma, dilate, label_components, luma


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


def _cap_chroma(values: np.ndarray, cap: float) -> np.ndarray:
    """Pull every pixel's colour back to at most *cap* chroma, keeping its level."""
    spread = values.max(axis=2) - values.min(axis=2)
    excess = spread > cap
    if not np.any(excess):
        return values
    ratio = (cap / np.maximum(spread, 1e-3))[..., None]
    grey = values.mean(axis=2, keepdims=True)
    limited = grey + (values - grey) * ratio
    return np.where(excess[..., None], limited, values)


def _light_shape(
    patch_luma: np.ndarray,
    shape_mask: np.ndarray,
    plate_luma: float,
    cfg: Phase1Config,
) -> Optional[np.ndarray]:
    """How strongly the highlight is added, relative to its peak, over a patch.

    The light a highlight adds is what can be measured - on the plate around the
    dot, where the base colour is known.  It falls off smoothly, and like any
    smooth glow a Gaussian is a good model for it, which makes its logarithm a
    quadratic in the patch coordinates.  Fitting that on the plate gives the
    shape of the light over the dot as well, including over the middle of the dot
    that the highlight hides.  ``None`` when nothing is measurable.
    """
    valid = ~(dilate(shape_mask.astype(np.uint8) * 255, 3) > 0)
    if int(np.count_nonzero(valid)) < 20:
        return None
    light = patch_luma - plate_luma
    usable = valid & (light > 1.0)
    if int(np.count_nonzero(usable)) < 15:
        return None

    rows, columns = np.nonzero(usable)
    span_x = max(1.0, 0.5 * (columns.max() - columns.min() + 1))
    span_y = max(1.0, 0.5 * (rows.max() - rows.min() + 1))
    x = (columns - columns.mean()) / span_x
    y = (rows - rows.mean()) / span_y
    design = np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=1).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(
        design, np.log(light[usable].astype(np.float64)), rcond=None
    )

    height, width = patch_luma.shape
    grid_y, grid_x = np.mgrid[0:height, 0:width].astype(np.float64)
    gx = (grid_x - columns.mean()) / span_x
    gy = (grid_y - rows.mean()) / span_y
    fitted = (
        coefficients[0]
        + coefficients[1] * gx
        + coefficients[2] * gy
        + coefficients[3] * gx * gx
        + coefficients[4] * gx * gy
        + coefficients[5] * gy * gy
    )
    light_fit = np.exp(np.clip(fitted, -20.0, 20.0))
    peak = float(light_fit.max())
    if peak <= 1.0:
        return None
    return np.clip(light_fit / peak, 0.0, 1.0)


def _recovered_ink(
    patch: np.ndarray,
    rebuild: np.ndarray,
    shape: np.ndarray,
    evidence: np.ndarray,
    plate_luma: float,
    cfg: Phase1Config,
) -> Optional[np.ndarray]:
    """Undo the wash of a highlight over a dot.

    A highlight behaves as a screen: ``observed = 255 - (255 - base) * t`` with
    one transmission ``t`` for every channel, and ``t = 1 - light / (255 - base)``.
    The light is known from the plate, but only up to a scale, and the ink is not
    known at all.  A marker holds one colour, so both are solved together from the
    dot itself: the light shape and the ink chroma that explain the wash of the
    dot's own pixels best.  That puts the strongest correction in the core of the
    washed out dot - where the highlight is strongest - instead of spreading it
    evenly and leaving a ring.

    ``None`` when the measurement is not trustworthy: a near-white plate leaves
    no range to undo, no plate is visible, or the highlight is flat over the dot
    and therefore carries no information about its own depth.
    """
    if plate_luma != plate_luma or 255.0 - plate_luma < cfg.ink_recovery_darkness_floor:
        return None
    smoothed = cv2.GaussianBlur(patch.astype(np.float32), (0, 0), 1.0)
    shape_strength = _light_shape(luma(smoothed), shape, plate_luma, cfg)
    if shape_strength is None:
        return None

    spread = chroma(smoothed)
    samples = shape_strength[rebuild]
    values = spread[rebuild]
    if samples.size < 12 or float(samples.std()) < 0.05:
        # The highlight is flat over the dot: its depth cannot be solved for.
        return None
    design = np.stack([np.ones_like(samples), -samples], axis=1).astype(np.float64)
    solution, *_ = np.linalg.lstsq(design, values.astype(np.float64), rcond=None)
    ink_chroma, scaled = float(solution[0]), float(solution[1])
    if ink_chroma <= 1.0 or scaled <= 0.0:
        return None
    gain = scaled / ink_chroma
    if not (0.2 <= gain <= 3.0):
        return None

    transmission = np.clip(1.0 - gain * shape_strength, cfg.ink_recovery_t_min, 1.0)
    recovered = 255.0 - (255.0 - smoothed) / np.maximum(transmission, 1e-6)[..., None]
    recovered = np.clip(recovered, 0.0, 255.0)

    visible = chroma(patch)[evidence] if np.any(evidence) else np.array([])
    rim = float(np.percentile(visible, 90)) if visible.size else 0.0
    cap = min(cfg.ink_recovery_ceiling, max(rim, rim * cfg.ink_recovery_factor))
    if cap <= 0.0:
        return None
    return _cap_chroma(recovered, cap)


def restore_ink_under_glare(
    image: np.ndarray,
    bright: BrightRegions,
    separation: GlareSeparation,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    """Rebuild the part of a marker dot that a highlight washed out.

    The repair starts from the highlight, not from the ink: every specular region
    is examined together with the ink that is still visible in and around it.  The
    convex hull of that visible ink is the shape the dot used to have, and the
    whole of that shape is painted again, so a ring of washed out ink comes back
    as a solid dot.  The colour is the ink the highlight washed out, recovered by
    dividing the light of the highlight back out of those pixels; only pixels of
    the shape that are bright (lit by the highlight) or coloured (ink) are
    touched, so the plate around the dot cannot be painted by accident.
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
            bridged |= hull > 0

        close_radius = int(
            round(min(max(2.0, 0.5 * np.sqrt(evidence_area / np.pi)), cfg.restore_close_cap))
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
        )
        closed = (
            cv2.morphologyEx(evidence.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel) > 0
        )
        bridged |= closed

        # The whole dot is rebuilt, not only the bright part of it: that is what
        # turns a washed out ring into a solid marker again.  Pixels of the shape
        # that carry no colour and are not lit by the highlight stay untouched, so
        # the plate cannot be painted by accident.
        patch_chroma = bright.chroma_map[y0:y1, x0:x1]
        rebuild = bridged & (glare_patch | (patch_chroma >= cfg.ink_chroma_min))
        if not np.any(rebuild):
            continue

        patch = image[y0:y1, x0:x1]
        centre_x = int(min(max(round(float(entry["centroid"][0])), 0), width - 1))  # type: ignore[index]
        centre_y = int(min(max(round(float(entry["centroid"][1])), 0), height - 1))  # type: ignore[index]
        local_level = float(bright.plate_level[centre_y, centre_x])
        level = local_level if local_level >= 0 else float("nan")
        color = _recovered_ink(patch, rebuild, bridged, evidence, level, cfg)
        target = output[y0:y1, x0:x1]
        if color is None:
            flat = np.clip(_restore_color(patch, evidence, cfg), 0, 255).astype(np.uint8)
            target[rebuild] = flat
        else:
            target[rebuild] = color[rebuild].astype(np.uint8)
        output[y0:y1, x0:x1] = target

        region = restored[y0:y1, x0:x1]
        region[rebuild] = 255
        restored[y0:y1, x0:x1] = region

        records.append(
            {
                "bbox": [x0, y0, int(x1 - x0), int(y1 - y0)],
            "glare_area": area,
            "visible_ink_area": evidence_area,
            "rebuilt_pixels": int(np.count_nonzero(rebuild)),
            "color_bgr": [
                int(round(float(v)))
                for v in (
                    np.median(target[rebuild], axis=0)
                    if np.any(rebuild)
                    else np.zeros(3, dtype=np.float32)
                )
            ],
            "plate_level": None if level != level else round(level, 2),
            "recovered_from_wash": bool(color is not None),
        }
        )
    return output, restored, records


def _plate_color_estimate(
    image: np.ndarray, mask: np.ndarray, protected: np.ndarray, cfg: Phase1Config
) -> np.ndarray:
    """Smooth estimate of the plate colour around a repair.

    Only pixels the repair does not touch and that are not marker ink take part,
    so no ink colour can ever reach the estimate.  The estimate is normalised by
    the same weights, which makes it fill the whole hole instead of fading away
    inside it.
    """
    valid = ((mask == 0) & (protected == 0)).astype(np.float32)
    sigma = cfg.plate_fill_sigma
    weights = cv2.GaussianBlur(valid, (0, 0), sigma)
    estimate = np.zeros(image.shape, dtype=np.float32)
    for channel in range(3):
        values = image[:, :, channel].astype(np.float32) * valid
        estimate[:, :, channel] = cv2.GaussianBlur(values, (0, 0), sigma)
    return estimate / np.maximum(weights, 1e-6)[..., None]


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

    # The inpainting draws from the boundary of the hole, and a marker next to the
    # hole is part of that boundary: its ink would be smeared across the plate.
    # The boundary is often still part of the glow as well, which leaves the patch
    # brighter than the plate around it.  Both are avoided by taking the repair
    # from the plate itself: a hole is filled with the level and the colour of the
    # plate around it, so a repair can neither invent ink colour nor leave a
    # bright or tinted patch behind.  Where no plate is visible at all the
    # inpainting result is kept.
    plate = _plate_color_estimate(image, mask, ink_final, cfg)
    weights = cv2.GaussianBlur(
        ((mask == 0) & (ink_final == 0)).astype(np.float32), (0, 0), cfg.plate_fill_sigma
    )
    filled = np.where((weights >= 0.05)[..., None], plate, blended)
    blended = blended * (1.0 - alpha) + filled * alpha
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
