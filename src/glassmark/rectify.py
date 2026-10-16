"""Rectification and geometric calibration of the glass plate (Phase 2).

This module handles:
1. Detecting the four corners of the rectangular glass plate in perspective.
2. Computing the homography matrix to transform the perspective view into
   an orthogonal, normalized top-down view (canonical plate space).
3. Warping both the image and detected marker coordinates into plate space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Canonical plate aspect ratio / dimensions in standard space
CANONICAL_PLATE_WIDTH = 720.0
CANONICAL_PLATE_HEIGHT = 530.0


@dataclass
class PlateCorners:
    """Four ordered corners of the glass plate in image coordinates.

    Order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """

    corners: np.ndarray  # shape (4, 2), float32
    confidence: float
    method: str

    @property
    def top_left(self) -> Tuple[float, float]:
        return float(self.corners[0, 0]), float(self.corners[0, 1])

    @property
    def top_right(self) -> Tuple[float, float]:
        return float(self.corners[1, 0]), float(self.corners[1, 1])

    @property
    def bottom_right(self) -> Tuple[float, float]:
        return float(self.corners[2, 0]), float(self.corners[2, 1])

    @property
    def bottom_left(self) -> Tuple[float, float]:
        return float(self.corners[3, 0]), float(self.corners[3, 1])

    def to_list(self) -> List[List[float]]:
        return [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in self.corners]


@dataclass
class RectificationResult:
    """Output of plate rectification."""

    corners: PlateCorners
    homography: np.ndarray  # 3x3 matrix from image space to canonical plate space
    inverse_homography: np.ndarray  # 3x3 matrix from canonical plate space to image space
    canonical_size: Tuple[int, int]  # (width, height)
    rectified_image: Optional[np.ndarray] = None


def order_quad_corners(pts: np.ndarray) -> np.ndarray:
    """Order/extract 4 points as: Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_plate_corners(image: np.ndarray) -> PlateCorners:
    """Detect the 4 corners of the glass plate against the background table.

    The plate sits on a darker table surface. This detector:
    1. Measures the peripheral background table luminance.
    2. Measures the interior plate luminance.
    3. Adapts the segmentation threshold dynamically.
    4. Extracts the largest convex contour and its 4 extreme boundary vertices.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    # Sample table border (outer 25 pixels)
    border_samples = np.concatenate([
        gray[:25, :].ravel(),
        gray[-25:, :].ravel(),
        gray[:, :25].ravel(),
        gray[:, -25:].ravel(),
    ])
    table_p95 = float(np.percentile(border_samples, 95))

    # Sample central plate interior
    interior = gray[int(0.2 * h) : int(0.8 * h), int(0.2 * w) : int(0.8 * w)]
    plate_median = float(np.median(interior))

    # Adaptive contrast threshold
    thresh_val = max(table_p95 + 0.35 * (plate_median - table_p95), table_p95 + 6.0)

    _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
    closed = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
    )
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback to image boundaries if no contour found
        fallback = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return PlateCorners(corners=fallback, confidence=0.0, method="fallback_bounds")

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    total_area = float(h * w)

    # Check if contour is plausible for a plate (at least 20% of image)
    if area < 0.20 * total_area:
        fallback = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        return PlateCorners(corners=fallback, confidence=0.1, method="fallback_small_contour")

    hull = cv2.convexHull(largest).reshape(-1, 2)
    ordered = order_quad_corners(hull)
    confidence = min(1.0, area / (0.45 * total_area))

    return PlateCorners(corners=ordered, confidence=confidence, method="adaptive_hull_extremes")


def compute_rectification(
    image: np.ndarray,
    corners: Optional[PlateCorners] = None,
    canonical_w: float = CANONICAL_PLATE_WIDTH,
    canonical_h: float = CANONICAL_PLATE_HEIGHT,
    warp_image: bool = True,
) -> RectificationResult:
    """Compute the homography matrix and rectify the image to canonical plate space."""
    if corners is None:
        corners = detect_plate_corners(image)

    dest_quad = np.float32([
        [0.0, 0.0],
        [canonical_w, 0.0],
        [canonical_w, canonical_h],
        [0.0, canonical_h],
    ])

    H = cv2.getPerspectiveTransform(corners.corners, dest_quad)
    H_inv = cv2.getPerspectiveTransform(dest_quad, corners.corners)

    rectified_img = None
    if warp_image:
        rectified_img = cv2.warpPerspective(
            image,
            H,
            (int(round(canonical_w)), int(round(canonical_h))),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

    return RectificationResult(
        corners=corners,
        homography=H,
        inverse_homography=H_inv,
        canonical_size=(int(round(canonical_w)), int(round(canonical_h))),
        rectified_image=rectified_img,
    )


def warp_points(points: np.ndarray | Sequence[Tuple[float, float]], homography: np.ndarray) -> np.ndarray:
    """Transform an array of 2D points using a 3x3 homography matrix."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(pts, np.asarray(homography, dtype=np.float64))
    return warped.reshape(-1, 2)
