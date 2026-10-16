"""Unit and regression tests for Phase 2 (Rectification) and Phase 3 (Decoding)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from glassmark.cases import all_cases, case_by_stem
from glassmark.decode import decode_layout, find_anchor_dot
from glassmark.imgio import read_image, read_json
from glassmark.rectify import CANONICAL_PLATE_HEIGHT, CANONICAL_PLATE_WIDTH, compute_rectification, detect_plate_corners

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
OUTPUT = ROOT / "data" / "out" / "phase1_special"


def test_plate_corner_detection_all_cases() -> None:
    """Test that plate corners are detected with high confidence across all synthetic cases."""
    for case in all_cases():
        img_path = SYNTHETIC / f"{case.stem}.png"
        if not img_path.exists():
            continue
        img = read_image(img_path)
        corners = detect_plate_corners(img)

        assert corners.corners.shape == (4, 2)
        assert corners.confidence >= 0.70, f"Low confidence corner detection on {case.stem}"

        # Corners must form a convex quadrilateral with positive area
        vec_a = corners.corners[1] - corners.corners[0]
        vec_b = corners.corners[2] - corners.corners[0]
        area = float(np.abs(vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]))
        assert area > 10000.0, f"Degenerate corner quad on {case.stem}"


def test_rectification_dimensions_and_warp() -> None:
    """Test that rectified images match canonical dimensions."""
    img_path = SYNTHETIC / "base_01_oil_before.png"
    img = read_image(img_path)
    res = compute_rectification(img, warp_image=True)

    assert res.rectified_image is not None
    h, w = res.rectified_image.shape[:2]
    assert w == int(CANONICAL_PLATE_WIDTH)
    assert h == int(CANONICAL_PLATE_HEIGHT)
    assert res.homography.shape == (3, 3)


def test_phase3_viewpoint_invariance_case_h() -> None:
    """The two viewpoints of Case H must decode to the IDENTICAL symbol sequence and hash."""
    h1_report = read_json(OUTPUT / "case_H_view1" / "report.json")
    h2_report = read_json(OUTPUT / "case_H_view2" / "report.json")

    dec1 = h1_report.get("stage_f_decoding", {})
    dec2 = h2_report.get("stage_f_decoding", {})

    assert dec1, "Missing stage_f_decoding in case_H_view1 report"
    assert dec2, "Missing stage_f_decoding in case_H_view2 report"

    assert dec1["symbol_sequence"] == dec2["symbol_sequence"], (
        f"Symbol sequence mismatch between Case H views:\n"
        f"View 1: {dec1['symbol_sequence']}\n"
        f"View 2: {dec2['symbol_sequence']}"
    )
    assert dec1["signature_hash"] == dec2["signature_hash"], (
        f"Signature hash mismatch between Case H views:\n"
        f"View 1: {dec1['signature_hash']}\n"
        f"View 2: {dec2['signature_hash']}"
    )
    assert dec1["dot_count"] == dec2["dot_count"] >= 6
