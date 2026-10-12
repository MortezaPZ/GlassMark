"""Stage A and stage B of phase 1.

Stage A: find very bright / specular regions (glare candidates).
Stage B: separate oil glare from marker ink, i.e. decide which of those bright
pixels are reflected light and which are coloured marker paint.

The discriminator is chromaticity.  Reflected light is nearly achromatic (or
only faintly tinted, for a warm lamp), while marker ink keeps a strongly
saturated colour even when a highlight sits on top of it -- at least around the
rim of the dot, where the glare is weaker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config


def luma(bgr: np.ndarray) -> np.ndarray:
    """Perceptual luminance as float32."""
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def chroma(bgr: np.ndarray) -> np.ndarray:
    """max(B, G, R) - min(B, G, R) as float32: 0 for white light, high for ink."""
    values = bgr.astype(np.float32)
    return values.max(axis=2) - values.min(axis=2)


def local_background(luminance: np.ndarray, ksize: int, frac: float = 0.0) -> np.ndarray:
    """Estimate the illumination level under every pixel.

    A morphological opening with a structuring element wider than a highlight
    removes bright regions and keeps the surrounding brightness, which is
    exactly the reference a specular pixel has to stand out from.  Because an
    oily reflection can be tens of pixels wide, the element also grows with the
    image so that soft broad highlights are removed as well.
    """
    height, width = luminance.shape[:2]
    size = max(3, int(ksize), int(frac * min(height, width)))
    size = size if size % 2 else size + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(luminance, cv2.MORPH_OPEN, kernel)


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Dilate a 0/255 mask with an elliptical kernel."""
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def label_components(mask: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Label a 0/255 mask, returning the label image and a per-component table."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    table: List[Dict[str, object]] = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        table.append(
            {
                "label": int(label),
                "area": area,
                "bbox": [x, y, width, height],
                "centroid": [float(centroids[label][0]), float(centroids[label][1])],
            }
        )
    return labels, table


@dataclass
class BrightRegions:
    """Stage A output."""

    candidates: np.ndarray
    mask: np.ndarray
    luminance: np.ndarray
    chroma_map: np.ndarray
    background: np.ndarray
    components: List[Dict[str, object]] = field(default_factory=list)
    skipped: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class GlareSeparation:
    """Stage B output."""

    ink_evidence: np.ndarray
    ink_strong: np.ndarray
    ink_protect: np.ndarray
    glare_mask: np.ndarray
    suppression_mask: np.ndarray
    components: List[Dict[str, object]] = field(default_factory=list)
    ink_dominated: List[Dict[str, object]] = field(default_factory=list)
    ink_component_info: List[Dict[str, object]] = field(default_factory=list)


def find_bright_regions(bgr: np.ndarray, cfg: Phase1Config = DEFAULT_CONFIG) -> BrightRegions:
    """Stage A: threshold very bright pixels and keep plausible glare blobs.

    A pixel is a specular candidate when it is absolutely bright *and* clearly
    brighter than its surroundings, or when it is almost clipped while remaining
    achromatic.  Components that are far too large are exposure / backlight
    rather than a recoverable highlight, so they are set aside (and reported).
    """
    luminance = luma(bgr)
    chroma_map = chroma(bgr)
    background = local_background(luminance, cfg.bg_open_ksize, cfg.bg_open_frac)

    rise = luminance - background
    bright = (luminance >= cfg.bright_luma) & (
        (rise >= cfg.bright_rel) | (rise >= cfg.bright_rel_frac * background)
    )
    clipped = (luminance >= cfg.clip_luma) & (chroma_map <= cfg.clip_chroma_max)
    candidates = np.where(bright | clipped, np.uint8(255), np.uint8(0))

    height, width = candidates.shape
    max_area = cfg.max_glare_area_frac * float(height * width)
    labels, table = label_components(candidates)

    mask = np.zeros_like(candidates)
    components: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    for entry in table:
        area = int(entry["area"])
        if area < cfg.min_glare_area:
            entry["reason"] = "too_small"
            skipped.append(entry)
            continue
        if area > max_area:
            entry["reason"] = "too_large_exposure_not_recoverable"
            skipped.append(entry)
            continue
        component = labels == int(entry["label"])
        mask[component] = 255
        entry["id"] = f"g{len(components) + 1}"
        entry["mean_luma"] = round(float(luminance[component].mean()), 2)
        entry["peak_luma"] = round(float(luminance[component].max()), 2)
        entry["mean_chroma"] = round(float(chroma_map[component].mean()), 2)
        components.append(entry)

    for entry in skipped:
        entry.pop("label", None)
    for entry in components:
        entry.pop("label", None)

    return BrightRegions(
        candidates=candidates,
        mask=mask,
        luminance=luminance,
        chroma_map=chroma_map,
        background=background,
        components=components,
        skipped=skipped,
    )


def clean_ink_evidence(evidence: np.ndarray, min_area: int) -> np.ndarray:
    """Drop isolated chromatic pixels so they cannot punch holes in the glare mask."""
    if not np.any(evidence):
        return evidence.copy()
    opened = cv2.morphologyEx(
        evidence, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    labels, table = label_components(opened)
    cleaned = np.zeros_like(evidence)
    for entry in table:
        if int(entry["area"]) >= min_area:
            cleaned[labels == int(entry["label"])] = 255
    return cleaned


def ink_evidence(
    chroma_map: np.ndarray, cfg: Phase1Config = DEFAULT_CONFIG
) -> tuple[np.ndarray, List[Dict[str, object]]]:
    """Decide which chromatic pixels are marker paint rather than a warm gloss.

    A highlight over a dot can wash the ink down to a faint tint, so the chroma
    threshold alone is too crude.  The second criterion is the edge: paint stops
    abruptly, while a patch of light reflected off oil fades out smoothly.  Its
    chroma gradient at the border is therefore tiny, and it is treated as glare.
    """
    candidates = np.where(chroma_map >= cfg.ink_chroma_min, np.uint8(255), np.uint8(0))
    opened = cv2.morphologyEx(
        candidates, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    labels, table = label_components(opened)

    gradient_x = cv2.Sobel(chroma_map, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(chroma_map, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)

    evidence = np.zeros(chroma_map.shape, dtype=np.uint8)
    infos: List[Dict[str, object]] = []
    for entry in table:
        component = labels == int(entry["label"])
        area = int(entry["area"])
        peak_chroma = float(chroma_map[component].max())
        info: Dict[str, object] = {
            "bbox": entry["bbox"],
            "centroid": entry["centroid"],
            "area": area,
            "peak_chroma": round(peak_chroma, 2),
        }
        if area < int(cfg.min_ink_area):
            info["reason"] = "too_small"
            infos.append(info)
            continue

        border = cv2.bitwise_and(
            dilate(component.astype(np.uint8) * 255, 2),
            cv2.bitwise_not(component.astype(np.uint8) * 255),
        ) > 0
        edge = float(np.percentile(gradient[border], 75)) if np.any(border) else 0.0
        strength = edge / max(1.0, peak_chroma)
        info["edge_strength"] = round(strength, 4)

        if peak_chroma >= cfg.ink_chroma_strong or strength >= cfg.ink_edge_strength_min:
            info["reason"] = "marker_ink"
            evidence[component] = 255
        else:
            info["reason"] = "smooth_tinted_glare"
        infos.append(info)
    return evidence, infos


def separate_glare_from_ink(
    bgr: np.ndarray,
    bright: BrightRegions,
    cfg: Phase1Config = DEFAULT_CONFIG,
) -> GlareSeparation:
    """Stage B: label each bright pixel as reflected light or marker ink.

    Marker ink is the strongly chromatic part of the image.  Bright regions that
    are dominated by ink are not glare at all: they are the marker dots
    themselves, seen through a slightly glossy film, so they stay untouched.
    """
    del bgr  # the chroma / luminance maps of stage A carry everything needed
    chroma_map = bright.chroma_map
    raw_evidence = np.where(chroma_map >= cfg.ink_chroma_min, np.uint8(255), np.uint8(0))
    ink_mask, ink_components = ink_evidence(chroma_map, cfg)
    ink_strong = np.where(chroma_map >= cfg.ink_chroma_hard, np.uint8(255), np.uint8(0))
    ink_protect = dilate(ink_mask, cfg.ink_protect_dilate)

    # Pixel-wise separation: a bright pixel is glare unless marker ink is what
    # makes it bright.  Working per pixel keeps the halo of a highlight that
    # only partly covers a dot suppressible.
    keep = cv2.bitwise_and(bright.mask, cv2.bitwise_not(ink_mask))

    # A flare never ends abruptly: the glow around it is dimmer than the core but
    # still clearly above the plate.  Add that halo, as long as it is achromatic
    # and connected to a region that was already found.
    halo = (
        (bright.luminance >= cfg.glow_luma)
        & (ink_mask == 0)
        & (dilate(keep, cfg.glow_grow) > 0)
    )
    keep = np.where((keep > 0) | halo, np.uint8(255), np.uint8(0))

    labels, table = label_components(bright.mask)
    components: List[Dict[str, object]] = []
    ink_dominated: List[Dict[str, object]] = []
    for entry in table:
        component = labels == int(entry["label"])
        pixels = int(np.count_nonzero(component))
        if pixels == 0:
            continue
        ink_pixels = int(np.count_nonzero(component & (raw_evidence > 0)))
        ink_fraction = ink_pixels / float(pixels)
        record: Dict[str, object] = {
            "bbox": entry["bbox"],
            "centroid": entry["centroid"],
            "area": entry["area"],
            "ink_fraction": round(ink_fraction, 4),
            "mean_chroma": round(float(chroma_map[component].mean()), 2),
            "peak_luma": round(float(bright.luminance[component].max()), 2),
            "glare_pixels": int(np.count_nonzero(cv2.bitwise_and(
                component.astype(np.uint8) * 255, keep
            ))),
        }
        if ink_fraction > 0.5:
            record["reason"] = "bright_region_dominated_by_ink"
            ink_dominated.append(record)
        else:
            record["id"] = f"g{len(components) + 1}"
            components.append(record)

    suppression = cv2.bitwise_and(
        dilate(keep, cfg.glare_dilate), cv2.bitwise_not(ink_protect)
    )
    return GlareSeparation(
        ink_evidence=ink_mask,
        ink_component_info=ink_components,
        ink_strong=ink_strong,
        ink_protect=ink_protect,
        glare_mask=keep,
        suppression_mask=suppression,
        components=components,
        ink_dominated=ink_dominated,
    )
