"""Polygon / mask geometry shared by the synthetic generator and the tests.

Keeping one implementation means the ground truth and the checks rasterise
regions in exactly the same way.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

Point = Sequence[float]
Polygon = List[List[float]]


def circle_polygon(cx: float, cy: float, radius: float, samples: int = 64) -> Polygon:
    """A closed polygon approximating a circle."""
    angles = np.linspace(0.0, 2.0 * np.pi, samples, endpoint=False)
    return [
        [float(cx + radius * np.cos(a)), float(cy + radius * np.sin(a))] for a in angles
    ]


def rotated_rect_polygon(
    cx: float, cy: float, length: float, width: float, angle_deg: float
) -> Polygon:
    """A closed polygon for a rotated rectangle."""
    half_l, half_w = length / 2.0, width / 2.0
    corners = np.array(
        [[-half_l, -half_w], [half_l, -half_w], [half_l, half_w], [-half_l, half_w]],
        dtype=np.float64,
    )
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    corners = corners @ rot.T + np.array([cx, cy])
    return corners.tolist()


def transform_polygon(polygon: Polygon, homography: np.ndarray) -> Polygon:
    """Warp a polygon with a 3x3 homography."""
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(points, np.asarray(homography, dtype=np.float64))
    return warped.reshape(-1, 2).tolist()


def polygon_mask(shape: Tuple[int, int], polygons: Iterable[Polygon]) -> np.ndarray:
    """Rasterise polygons into a uint8 0/255 mask of *shape* = (height, width)."""
    mask = np.zeros(shape, dtype=np.uint8)
    polys = [np.rint(np.asarray(p, dtype=np.float64)).astype(np.int32) for p in polygons]
    polys = [p for p in polys if p.shape[0] >= 3]
    if polys:
        cv2.fillPoly(mask, polys, 255)
    return mask


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def overlap_fraction(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """|a AND b| / |a| (0.0 when a is empty)."""
    area = mask_area(mask_a)
    if area == 0:
        return 0.0
    return float(np.count_nonzero(cv2.bitwise_and(mask_a, mask_b))) / float(area)


def coverage(mask: np.ndarray, region: np.ndarray) -> float:
    """Fraction of *region* pixels that are set in *mask*."""
    return overlap_fraction(region, mask)


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask, kernel)


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.erode(mask, kernel)


def annulus_mask(mask: np.ndarray, inner: int, outer: int) -> np.ndarray:
    """Pixels within *outer* dilation of *mask* but outside its *inner* dilation."""
    grown = dilate_mask(mask, outer)
    core = dilate_mask(mask, inner)
    return cv2.bitwise_and(grown, cv2.bitwise_not(core))
