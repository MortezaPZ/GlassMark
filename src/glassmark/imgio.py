"""Image and JSON helpers.

Reads/writes go through numpy buffers so that non-ASCII paths work on every
platform (``cv2.imread``/``cv2.imwrite`` fail on such paths under Windows).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import cv2
import numpy as np

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def read_image(path: str | Path) -> np.ndarray:
    """Read a colour image as uint8 BGR."""
    path = Path(path)
    buffer = np.fromfile(str(path), dtype=np.uint8)
    if buffer.size == 0:
        raise FileNotFoundError(f"empty or missing image: {path}")
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return image


def encode_png(image: np.ndarray) -> bytes:
    """Encode an image as PNG bytes in memory."""
    ok, buffer = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode image to PNG")
    return bytes(buffer)


def write_image(path: str | Path, image: np.ndarray) -> None:
    """Write a uint8 image, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix if path.suffix else ".png"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"cannot encode image: {path}")
    buffer.tofile(str(path))


def list_images(folder: str | Path) -> List[Path]:
    """All decodable image files in *folder*, sorted by name."""
    folder = Path(folder)
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def sidecar_path(image_path: str | Path) -> Path:
    """Ground-truth JSON that belongs to an image."""
    return Path(image_path).with_suffix(".json")


def iter_image_sidecars(folder: str | Path) -> Iterable[tuple[Path, Path]]:
    """Yield ``(image_path, sidecar_path)`` pairs that really have a sidecar."""
    for image_path in list_images(folder):
        sidecar = sidecar_path(image_path)
        if sidecar.exists():
            yield image_path, sidecar
