"""Tests for client-specified 4-step pipeline."""

import cv2
import numpy as np
from pathlib import Path

from glassmark.client_steps import run_step_by_step_pipeline, CLIENT_SYMBOLS
from glassmark.server import process_image_buffer

ROOT = Path(__file__).resolve().parents[1]


def test_client_steps_pipeline_synthetic():
    image_path = ROOT / "data" / "synthetic" / "base_01_oil_before.png"
    data = image_path.read_bytes()
    res = process_image_buffer(data)
    
    assert res["status"] == "success"
    assert "client_steps" in res
    
    cs = res["client_steps"]
    assert cs["step1_cleaned_glare_removal"] == "PASS"
    assert cs["step2_symbol_substitution"]["total_substituted"] > 0
    assert cs["step2_symbol_substitution"]["symbol_map"]["red"] == "+"
    assert cs["step2_symbol_substitution"]["symbol_map"]["blue"] == "*"
    assert cs["step2_symbol_substitution"]["symbol_map"]["green"] == "#"
    assert cs["step2_symbol_substitution"]["symbol_map"]["yellow"] == "O"
    
    s3 = cs["step3_anchor_12_oclock"]
    assert s3["anchor_id"] is not None
    assert s3["anchor_area"] > 0
    
    s4 = cs["step4_hierarchical_encoding"]
    assert s4["marks_count"] > 0
    assert len(s4["symbol_sequence"]) > 0
    assert len(s4["signature_hash"]) == 16
    
    # Images present
    images = res["images"]
    assert images["step2_symbols"].startswith("data:image/png;base64,")
    assert images["step3_aligned"].startswith("data:image/png;base64,")
