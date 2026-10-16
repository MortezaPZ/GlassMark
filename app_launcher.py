"""Standalone desktop entrypoint for GlassMark (used for PyInstaller EXE build).

Starts the local backend server, automatically opens the user's default web browser
to the graphical user interface, and handles graceful shutdown.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Add src to sys.path so frozen / unfrozen both find glassmark
base_dir = Path(__file__).resolve().parent
src_dir = base_dir / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from glassmark.server import create_server


def find_free_port(preferred: int = 8080) -> int:
    """Find a free port starting with the preferred one."""
    for port in range(preferred, preferred + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


def main() -> int:
    port = find_free_port(8080)
    server = create_server("127.0.0.1", port)
    url = f"http://127.0.0.1:{port}/"

    print("=" * 65)
    print("  [GlassMark] Glass Plate Calibration & Decoding System")
    print(f"  URL: {url}")
    print("=" * 65)
    print("Opening browser...")

    def open_browser():
        time.sleep(1.2)
        try:
            webbrowser.open(url)
        except Exception as e:
            print(f"Could not open browser automatically: {e}")

    threading.Thread(target=open_browser, daemon=True).start()

    print("\nApplication is ready. Close this window to exit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down GlassMark...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
