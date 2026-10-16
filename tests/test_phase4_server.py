"""Integration and end-to-end tests for Phase 4 Web UI and REST API server."""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from glassmark.server import create_server

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "synthetic" / "base_01_oil_before.png"


@pytest.fixture(scope="module")
def live_server():
    """Start local test server on an ephemeral port."""
    server = create_server(host="127.0.0.1", port=8899)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    yield "http://127.0.0.1:8899"
    server.shutdown()


def test_web_ui_html_endpoint(live_server: str) -> None:
    """GET / must return 200 and the HTML GUI."""
    req = urllib.request.Request(f"{live_server}/")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        html = resp.read().decode("utf-8")
        assert "<!DOCTYPE html>" in html
        assert "GlassMark" in html


def test_api_health_endpoint(live_server: str) -> None:
    """GET /api/health must return status healthy."""
    req = urllib.request.Request(f"{live_server}/api/health")
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        data = json.loads(resp.read().decode("utf-8"))
        assert data.get("status") == "healthy"
        assert data.get("service") == "GlassMark"


def test_api_process_raw_image(live_server: str) -> None:
    """POST /api/process with raw image bytes must execute phases 1-3 and return results."""
    image_bytes = IMAGE_PATH.read_bytes()
    req = urllib.request.Request(
        f"{live_server}/api/process",
        data=image_bytes,
        headers={"Content-Type": "image/png"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        result = json.loads(resp.read().decode("utf-8"))

        assert result.get("status") == "success"
        decoding = result.get("decoding", {})
        assert decoding.get("symbol_sequence")
        assert decoding.get("signature_hash")
        assert decoding.get("dot_count", 0) >= 6

        rectification = result.get("rectification", {})
        assert len(rectification.get("corners", [])) == 4

        images = result.get("images", {})
        for key in ("input", "glare", "markers", "cleaned", "dots", "rectified"):
            assert key in images
            assert images[key].startswith("data:image/png;base64,")
