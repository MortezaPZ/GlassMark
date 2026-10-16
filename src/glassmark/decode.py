"""Orientation alignment and semantic pattern decoding (Phase 3).

This module handles:
1. Identifying the reference anchor dot (e.g. largest dot, distinct marker).
2. Aligning the plate orientation so the anchor aligns to the canonical
   reference direction (12 o'clock, 0 degrees).
3. Mapping dots into polar coordinate bins (clock angle + normalized radius)
   and decoding the dot layout into a structured ID, symbol sequence, and hash.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Canonical color ordering for symbols/tokens
COLOR_TO_CODE = {
    "red": "R",
    "blue": "B",
    "green": "G",
    "yellow": "Y",
    "unknown": "?",
}


@dataclass
class NormalizedDot:
    """A marker dot positioned in canonical rectified plate coordinates."""

    id: str
    color: str
    raw_pos: Tuple[float, float]  # (x, y) in image space
    plate_pos: Tuple[float, float]  # (x, y) in canonical plate space (0..W, 0..H)
    plate_uv: Tuple[float, float]  # normalized (u, v) in 0..1
    clock_deg: float  # angle from plate center (0° = 12 o'clock, 90° = 3 o'clock)
    distance_norm: float  # distance from center normalized by half plate diagonal
    area: float
    is_anchor: bool = False
    symbol: str = "?"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "color": self.color,
            "symbol": self.symbol,
            "raw_pos": [round(self.raw_pos[0], 2), round(self.raw_pos[1], 2)],
            "plate_pos": [round(self.plate_pos[0], 2), round(self.plate_pos[1], 2)],
            "plate_uv": [round(self.plate_uv[0], 4), round(self.plate_uv[1], 4)],
            "clock_deg": round(self.clock_deg, 2),
            "distance_norm": round(self.distance_norm, 4),
            "area": round(self.area, 2),
            "is_anchor": self.is_anchor,
        }


@dataclass
class DecodedPlate:
    """Complete semantic decode of a plate."""

    anchor_id: Optional[str]
    anchor_color: Optional[str]
    rotation_offset_deg: float  # Rotation applied to align anchor to 12 o'clock
    dots: List[NormalizedDot]
    symbol_sequence: str  # Sequence of dots ordered clockwise from 12 o'clock
    signature_hash: str  # Deterministic SHA256 signature of the layout
    layout_type: str = "polar_clock"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor_id": self.anchor_id,
            "anchor_color": self.anchor_color,
            "rotation_offset_deg": round(self.rotation_offset_deg, 2),
            "symbol_sequence": self.symbol_sequence,
            "signature_hash": self.signature_hash,
            "layout_type": self.layout_type,
            "dot_count": len(self.dots),
            "dots": [d.to_dict() for d in self.dots],
        }


def find_anchor_dot(blobs: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Identify the anchor dot among detected blobs.

    Rule: The largest marker dot (maximum area) is the primary anchor.
    In case of ties, the most chromatic / red marker is favored.
    """
    if not blobs:
        return None

    # Score based on area and color prominence
    def score(b: Dict[str, Any]) -> float:
        area = float(b.get("area", 0.0))
        color = str(b.get("color", ""))
        color_bonus = 1.15 if color == "red" else (1.05 if color == "yellow" else 1.0)
        return area * color_bonus

    return max(blobs, key=score)


def decode_layout(
    blobs: Sequence[Dict[str, Any]],
    homography: np.ndarray,
    canonical_w: float = 720.0,
    canonical_h: float = 530.0,
    align_to_12_oclock: bool = True,
) -> DecodedPlate:
    """Decode marker blobs into canonical normalized coordinates, orient, and extract symbols.

    1. Maps each blob centroid from image space to canonical plate space via homography.
    2. Identifies the primary anchor dot (largest/key dot).
    3. If align_to_12_oclock is True, rotates the angular frame so the anchor is at 0° (12 o'clock).
    4. Sorts dots clockwise from 12 o'clock to create a stable, viewpoint-invariant symbol sequence.
    5. Computes a cryptographic layout signature hash.
    """
    if not blobs:
        return DecodedPlate(
            anchor_id=None,
            anchor_color=None,
            rotation_offset_deg=0.0,
            dots=[],
            symbol_sequence="",
            signature_hash="",
        )

    # Plate center and normalization diagonal
    center_x = canonical_w / 2.0
    center_y = canonical_h / 2.0
    max_radius = 0.5 * np.hypot(canonical_w, canonical_h)

    # 1. Transform points to canonical plate space
    import cv2

    raw_centroids = np.float32([[b["centroid"][0], b["centroid"][1]] for b in blobs]).reshape(-1, 1, 2)
    warped_centroids = cv2.perspectiveTransform(raw_centroids, np.asarray(homography, dtype=np.float64)).reshape(-1, 2)

    # 2. Find anchor
    anchor_blob = find_anchor_dot(blobs)
    anchor_id = str(anchor_blob["id"]) if anchor_blob else None

    # Temporary dot list to find anchor angle
    temp_dots: List[Dict[str, Any]] = []
    anchor_angle = 0.0

    for idx, b in enumerate(blobs):
        wx, wy = float(warped_centroids[idx, 0]), float(warped_centroids[idx, 1])
        dx = wx - center_x
        dy = wy - center_y
        # Standard clock angle: 0° is North (12 o'clock, negative y in image coords), clockwise
        clock_deg = (float(np.degrees(np.arctan2(dx, -dy)))) % 360.0
        dist_norm = float(np.hypot(dx, dy) / max_radius)
        u = float(wx / canonical_w)
        v = float(wy / canonical_h)

        is_anchor = (str(b.get("id")) == anchor_id)
        if is_anchor:
            anchor_angle = clock_deg

        temp_dots.append({
            "blob": b,
            "raw_pos": (float(b["centroid"][0]), float(b["centroid"][1])),
            "plate_pos": (wx, wy),
            "plate_uv": (u, v),
            "clock_deg": clock_deg,
            "dist_norm": dist_norm,
            "is_anchor": is_anchor,
        })

    rotation_offset = anchor_angle if align_to_12_oclock else 0.0

    # 3. Apply rotation alignment
    normalized_dots: List[NormalizedDot] = []
    for td in temp_dots:
        b = td["blob"]
        color = str(b.get("color", "unknown"))
        symbol = COLOR_TO_CODE.get(color, "?")

        aligned_clock = (td["clock_deg"] - rotation_offset) % 360.0

        nd = NormalizedDot(
            id=str(b.get("id", "")),
            color=color,
            raw_pos=td["raw_pos"],
            plate_pos=td["plate_pos"],
            plate_uv=td["plate_uv"],
            clock_deg=aligned_clock,
            distance_norm=td["dist_norm"],
            area=float(b.get("area", 0.0)),
            is_anchor=td["is_anchor"],
            symbol=symbol,
        )
        normalized_dots.append(nd)

    # 4. Sort dots clockwise starting from 12 o'clock (0°)
    normalized_dots.sort(key=lambda d: d.clock_deg)

    # 5. Build symbol sequence and hash
    # Format: [ColorSymbol:ClockHour:RadialBin]
    seq_tokens: List[str] = []
    for d in normalized_dots:
        # 12-hour clock representation: 0° -> 12, 30° -> 1, etc.
        hour = int(round(d.clock_deg / 30.0))
        if hour == 0:
            hour = 12
        radial_bin = int(round(d.distance_norm * 10.0))
        seq_tokens.append(f"{d.symbol}{hour}r{radial_bin}")

    symbol_sequence = "-".join(seq_tokens)
    sig_content = f"{symbol_sequence}:{len(normalized_dots)}"
    sig_hash = hashlib.sha256(sig_content.encode("utf-8")).hexdigest()[:16]

    anchor_color = str(anchor_blob.get("color")) if anchor_blob else None

    return DecodedPlate(
        anchor_id=anchor_id,
        anchor_color=anchor_color,
        rotation_offset_deg=rotation_offset,
        dots=normalized_dots,
        symbol_sequence=symbol_sequence,
        signature_hash=sig_hash,
    )
