"""Stage D of phase 1: keep only marker-like blobs.

Everything chromatic that survives stage C is a candidate.  A candidate is kept
as a marker dot when it is small, compact and strongly coloured; leftover shine,
gloss patches and wet smears are dropped (with the reason recorded, so a report
can explain every decision).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, Phase1Config
from .glare import chroma, label_components

# Reference hues in OpenCV degrees (0..360) for the four marker colours.
COLOR_REFERENCES: Dict[str, float] = {
    "red": 0.0,
    "yellow": 55.0,
    "green": 120.0,
    "blue": 225.0,
}


def classify_ink_color(bgr_pixels: np.ndarray) -> Tuple[str, float]:
    """Nearest marker colour of a set of pixels, plus the hue distance in degrees."""
    if bgr_pixels.size == 0:
        return "unknown", 999.0
    pixels = bgr_pixels.reshape(-1, 1, 3).astype(np.uint8)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    saturation = hsv[:, 1]
    if float(np.max(saturation)) < 60.0:
        return "unknown", 999.0
    keep = saturation >= max(60.0, 0.6 * float(np.max(saturation)))
    hues = hsv[keep, 0] * 2.0
    if hues.size == 0:
        return "unknown", 999.0
    hue = float(np.median(hues))
    best_name, best_distance = "unknown", 999.0
    for name, reference in COLOR_REFERENCES.items():
        distance = abs(hue - reference)
        distance = min(distance, 360.0 - distance)
        if distance < best_distance:
            best_name, best_distance = name, distance
    if best_distance > 50.0:
        return "unknown", best_distance
    return best_name, best_distance


@dataclass
class MarkerBlobs:
    """Stage D output."""

    mask: np.ndarray
    blobs: List[Dict[str, object]] = field(default_factory=list)
    dropped: List[Dict[str, object]] = field(default_factory=list)


def _fill_holes(
    component: np.ndarray,
    bbox: Tuple[int, int, int, int],
    shape: Tuple[int, int],
    close_radius: int,
    hull_max_ratio: float,
) -> np.ndarray:
    """Close holes / concavities of one component inside its own bounding box.

    A dot whose middle was washed out by a highlight comes back as a ring: its
    convex hull restores the round marker shape.  The hull is only trusted while
    it stays close to the visible ink area, so smears and trails are not turned
    into discs.
    """
    x, y, width, height = bbox
    height_img, width_img = shape
    pad = close_radius + 2
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(width_img, x + width + pad), min(height_img, y + height + pad)

    patch = component[y0:y1, x0:x1].astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_radius + 1, 2 * close_radius + 1)
    )
    closed = cv2.morphologyEx(patch * 255, cv2.MORPH_CLOSE, kernel) > 0
    filled = closed & (patch == 0)

    contours, _ = cv2.findContours(patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if c.shape[0] >= 3]
    if contours:
        hull = np.zeros(patch.shape, dtype=np.uint8)
        cv2.fillPoly(hull, [cv2.convexHull(np.vstack(contours))], 1)
        hull_bool = hull > 0
        visible = max(1, int(np.count_nonzero(patch)))
        if int(np.count_nonzero(hull_bool)) <= hull_max_ratio * visible:
            filled = filled | (hull_bool & (patch == 0))

    if not np.any(filled):
        return component.copy()

    result = component.copy()
    region = result[y0:y1, x0:x1]
    region[filled] = True
    result[y0:y1, x0:x1] = region
    return result


def detect_markers(
    image: np.ndarray, cfg: Phase1Config = DEFAULT_CONFIG
) -> MarkerBlobs:
    """Stage D: isolate marker-like blobs on an already cleaned image."""
    chroma_map = chroma(image)
    ink = np.where(chroma_map >= cfg.marker_chroma_min, np.uint8(255), np.uint8(0))
    if not np.any(ink):
        return MarkerBlobs(mask=np.zeros(image.shape[:2], dtype=np.uint8))

    opened = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )
    labels, table = label_components(opened)
    height, width = image.shape[:2]
    max_area = cfg.max_marker_area_frac * float(height * width)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    blobs: List[Dict[str, object]] = []
    dropped: List[Dict[str, object]] = []

    candidates: List[Dict[str, object]] = []
    for entry in table:
        component = labels == int(entry["label"])
        area = int(entry["area"])
        x, y, box_w, box_h = (int(v) for v in entry["bbox"])  # type: ignore[misc]
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        perimeter = float(sum(cv2.arcLength(c, True) for c in contours))
        circularity = (
            4.0 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0
        )
        fill_ratio = area / float(max(1, box_w * box_h))
        aspect = box_w / float(max(1, box_h))
        pixel_chroma = chroma_map[component]
        peak_chroma = float(pixel_chroma.max())
        strong = pixel_chroma >= np.percentile(pixel_chroma, 60.0)
        mean_chroma = float(pixel_chroma[strong].mean()) if np.any(strong) else 0.0

        record: Dict[str, object] = {
            "bbox": [x, y, box_w, box_h],
            "area": area,
            "centroid": [float(entry["centroid"][0]), float(entry["centroid"][1])],
            "circularity": round(circularity, 3),
            "fill_ratio": round(fill_ratio, 3),
            "aspect": round(aspect, 3),
            "mean_chroma": round(mean_chroma, 2),
            "peak_chroma": round(peak_chroma, 2),
            "equivalent_radius": round(float(np.sqrt(area / np.pi)), 2),
        }

        if area < cfg.min_marker_area:
            record["reason"] = "too_small"
            dropped.append(record)
            continue
        if area > max_area:
            record["reason"] = "too_large_uniform_colour_region"
            dropped.append(record)
            continue
        if peak_chroma < cfg.marker_chroma_peak_min or mean_chroma < cfg.marker_chroma_min:
            record["reason"] = "weak_colour_not_marker_ink"
            dropped.append(record)
            continue
        if (
            circularity < cfg.min_circularity
            or fill_ratio < cfg.min_fill_ratio
            or aspect > cfg.max_aspect
            or aspect < 1.0 / cfg.max_aspect
        ):
            record["reason"] = "not_dot_shaped"
            dropped.append(record)
            continue

        candidates.append({"record": record, "component": component})

    for entry in candidates:
        record = entry["record"]  # type: ignore[assignment]
        component: np.ndarray = entry["component"]  # type: ignore[assignment]
        bbox = tuple(int(v) for v in record["bbox"])  # type: ignore[index]
        close_radius = int(
            round(min(max(2.0, 0.5 * float(record["equivalent_radius"])), cfg.restore_close_cap))
        )
        filled = _fill_holes(
            component,
            bbox,
            image.shape[:2],
            close_radius,
            cfg.marker_hull_fill_max_ratio,
        )
        grew = filled & ~component
        mask[filled] = 255

        selected = filled & (chroma_map >= cfg.marker_chroma_min)
        colour, hue_distance = classify_ink_color(image[selected])
        record = dict(record)
        record["id"] = f"m{len(blobs) + 1}"
        record["color"] = colour
        record["hue_distance_deg"] = round(hue_distance, 1)
        record["color_bgr"] = [
            int(round(float(v))) for v in np.median(image[selected].reshape(-1, 3), axis=0)
        ]
        record["filled_pixels"] = int(np.count_nonzero(grew))
        blobs.append(record)

    return MarkerBlobs(mask=mask, blobs=blobs, dropped=dropped)
