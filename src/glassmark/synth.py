"""Synthetic oily-glass scenes with a JSON ground truth.

The scene is a rectangular glass plate on a dark table.  Coloured marker dots
are painted on the glass, oil adds a broad gloss film, and specular highlights
(hard reflections plus bright streaks) sit on top of everything.  The oil may be
applied before or after the dots.

Every scene is rendered in a canonical *scene space* and then warped with a mild
rotation / perspective homography, so the same physical plate can be rendered
from several viewpoints without any knowledge of the viewpoint being needed
later: the ground truth is transformed together with the pixels.

The ground truth is written next to each image as ``<stem>.json`` and contains,
per dot: colour, centre, approximate radius, polygon and glare overlap; and per
glare region: kind, centre, approximate radius, polygon, ink overlap and whether
its local background is already saturated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geom import (
    circle_polygon,
    dilate_mask,
    overlap_fraction,
    polygon_mask,
    rotated_rect_polygon,
    transform_polygon,
)

SCENE_W, SCENE_H = 860, 640
PLATE = (70, 55, 790, 585)  # x0, y0, x1, y1 in scene space
MARGIN = 60
OUT_W, OUT_H = SCENE_W + 2 * MARGIN, SCENE_H + 2 * MARGIN

# Marker ink colours, BGR, strongly saturated.
INK = {
    "blue": (205.0, 45.0, 20.0),
    "red": (30.0, 25.0, 215.0),
    "green": (40.0, 190.0, 35.0),
    "yellow": (10.0, 210.0, 245.0),
}

GLARE_KINDS = ("specular_blob", "bright_streak", "warm_blob", "broad_wash")


@dataclass
class Dot:
    """A marker dot in scene space."""

    id: str
    color: str
    center: Tuple[float, float]
    radius: float


@dataclass
class Glare:
    """A bright region in scene space.

    ``sigma`` is the (major, minor) Gaussian radius of the highlight; ``length``
    is used by streaks; ``color`` is the BGR colour of the reflected light.
    """

    id: str
    kind: str
    center: Tuple[float, float]
    sigma: Tuple[float, float]
    alpha: float
    angle_deg: float = 0.0
    color: Tuple[float, float, float] = (250.0, 250.0, 246.0)
    length: float = 0.0

    @property
    def gt_radius(self) -> float:
        """Radius of the ground-truth core polygon.

        The core is the bright middle of the highlight, not its soft halo: it is
        the part that has to disappear from the cleaned image, and the part a
        marker must never be reported on.
        """
        if self.kind == "bright_streak":
            return max(3.0, 0.85 * min(self.sigma))
        return max(5.0, 1.0 * max(self.sigma))


@dataclass
class SceneSpec:
    """Everything that defines one physical plate state."""

    seed: int
    dots: List[Dot]
    glare: List[Glare]
    order: str = "oil_before_dots"
    oil_strength: float = 0.0
    oil_whole_plate: bool = False
    dot_sheen: float = 0.0
    sheen_dot_ids: Tuple[str, ...] = ()
    plate_gain: float = 1.0
    exposure: float = 1.0


@dataclass
class Camera:
    """Viewpoint / exposure of one photo of a plate."""

    rotation_deg: float = 0.0
    perspective_px: float = 0.0
    scale: float = 0.95
    offset: Tuple[float, float] = (0.0, 0.0)
    exposure: float = 1.0
    gamma: float = 1.0
    blur_sigma: float = 0.8
    noise_sigma: float = 3.0


@dataclass
class RenderedScene:
    image: np.ndarray
    homography: np.ndarray
    meta: Dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# scene compositing helpers
# --------------------------------------------------------------------------- #
def _screen(image: np.ndarray, color: Sequence[float], alpha: np.ndarray) -> np.ndarray:
    """Screen-blend a light colour onto the image (additive, saturating)."""
    colour = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    screened = 255.0 - (255.0 - image) * (255.0 - colour) / 255.0
    a = alpha[..., None].astype(np.float32)
    return image * (1.0 - a) + screened * a


def _paint(image: np.ndarray, color: Sequence[float], alpha: np.ndarray) -> np.ndarray:
    """Alpha-composite opaque paint (marker ink) onto the image."""
    colour = np.asarray(color, dtype=np.float32).reshape(1, 1, 3)
    a = alpha[..., None].astype(np.float32)
    return image * (1.0 - a) + colour * a


def _gaussian_map(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    sigma: Tuple[float, float],
    angle_deg: float,
    peak: float = 1.0,
    irregularity: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Rotated elliptical Gaussian falling off from ``center``."""
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = np.deg2rad(angle_deg)
    dx, dy = xx - center[0], yy - center[1]
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    sigma_major = max(1.0, float(sigma[0]))
    sigma_minor = max(1.0, float(sigma[1]))
    field = np.exp(-0.5 * ((u / sigma_major) ** 2 + (v / sigma_minor) ** 2))

    if irregularity > 0.0 and rng is not None:
        # Oil gloss is never a perfect ellipse: add one or two weaker lobes.
        lobes = rng.integers(1, 3)
        for _ in range(int(lobes)):
            offset = rng.normal(0.0, 0.6, size=2) * np.array([sigma_major, sigma_minor])
            direction = rng.uniform(0, 2 * np.pi)
            lobe_center = (
                center[0] + offset[0] * np.cos(direction),
                center[1] + offset[1] * np.sin(direction),
            )
            lobe = _gaussian_map(
                shape,
                lobe_center,
                (sigma_major * 1.25, sigma_minor * 1.15),
                angle_deg + rng.uniform(-25, 25),
            )
            field = np.maximum(field, 0.35 * lobe)
    return (peak * np.clip(field, 0.0, 1.0)).astype(np.float32)


def _low_frequency_field(
    shape: Tuple[int, int], rng: np.random.Generator, cells: Tuple[int, int] = (7, 9)
) -> np.ndarray:
    """Smooth 0..1 noise, used for oil films and table texture."""
    noise = rng.random((cells[0], cells[1])).astype(np.float32)
    field = cv2.resize(noise, (shape[1], shape[0]), interpolation=cv2.INTER_CUBIC)
    field -= field.min()
    peak = field.max()
    return field / peak if peak > 0 else field


def _plate_mask(blur: float = 3.0) -> np.ndarray:
    x0, y0, x1, y1 = PLATE
    mask = np.zeros((SCENE_H, SCENE_W), dtype=np.float32)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 1.0, -1, lineType=cv2.LINE_AA)
    return cv2.GaussianBlur(mask, (0, 0), blur)


# --------------------------------------------------------------------------- #
# rendering stages
# --------------------------------------------------------------------------- #
def _background(rng: np.random.Generator) -> np.ndarray:
    texture = _low_frequency_field((SCENE_H, SCENE_W), rng, cells=(5, 6))
    base = 34.0 + 10.0 * texture
    image = np.dstack([base * 0.92, base * 0.96, base]).astype(np.float32)

    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    radius = np.sqrt((xx - SCENE_W / 2) ** 2 + (yy - SCENE_H / 2) ** 2)
    vignette = 1.0 - 0.35 * np.clip(radius / (0.75 * SCENE_W), 0.0, 1.0)
    return image * vignette[..., None]


def _plate(image: np.ndarray, spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    texture = _low_frequency_field((SCENE_H, SCENE_W), rng, cells=(4, 8))
    x0, y0, x1, y1 = PLATE
    height, width = y1 - y0, x1 - x0
    yy, xx = np.mgrid[0:SCENE_H, 0:SCENE_W].astype(np.float32)
    light = 1.0 - 0.30 * np.clip(
        np.sqrt(((xx - (x0 + 0.3 * width)) / width) ** 2 + ((yy - (y0 + 0.25 * height)) / height) ** 2),
        0.0,
        1.0,
    )
    slate = 62.0 * spec.plate_gain
    plate = np.dstack(
        [slate * 0.94 * light, slate * 0.98 * light, slate * 1.04 * light]
    ).astype(np.float32)
    plate += (18.0 * texture - 9.0)[..., None] * np.array([1.0, 1.0, 1.04], dtype=np.float32)
    alpha = _plate_mask()[..., None]
    return image * (1.0 - alpha) + plate * alpha


def _oil(
    image: np.ndarray,
    spec: SceneSpec,
    rng: np.random.Generator,
    over_ink: bool = False,
) -> np.ndarray:
    """Broad low-frequency gloss film."""
    if spec.oil_strength <= 0.0:
        return image
    field = _low_frequency_field((SCENE_H, SCENE_W), rng, cells=(3, 5))
    if spec.oil_whole_plate:
        alpha = spec.oil_strength * (0.62 + 0.38 * field)
    else:
        alpha = spec.oil_strength * (0.18 + 0.82 * field ** 1.6)
    plate = _plate_mask(blur=8.0)
    # A little oil always wraps around the plate edge.
    alpha = alpha * (0.35 + 0.65 * plate)
    tint = (238.0, 238.0, 230.0) if over_ink else (240.0, 240.0, 234.0)
    image = _screen(image, tint, alpha.astype(np.float32))

    if over_ink:
        # Oil laid on top of the dots slightly smears and dulls them.
        image = cv2.GaussianBlur(image, (0, 0), 0.9)
        grey = np.repeat(image.mean(axis=2, keepdims=True), 3, axis=2)
        image = image + (grey - image) * 0.06
    return image


def _dots(image: np.ndarray, spec: SceneSpec, rng: np.random.Generator) -> np.ndarray:
    for dot in spec.dots:
        ink = np.asarray(INK[dot.color], dtype=np.float32)
        radius = float(dot.radius)

        disc = np.zeros((SCENE_H, SCENE_W), dtype=np.float32)
        cv2.circle(
            disc,
            (int(round(dot.center[0])), int(round(dot.center[1]))),
            int(round(radius)),
            1.0,
            -1,
            lineType=cv2.LINE_AA,
        )
        image = _paint(image, ink, disc)

        # Marker pens leave a darker rim where the felt edge pressed hardest.
        rim = np.zeros((SCENE_H, SCENE_W), dtype=np.float32)
        cv2.circle(
            rim,
            (int(round(dot.center[0])), int(round(dot.center[1]))),
            int(round(radius)),
            0.45,
            max(1, int(round(radius * 0.35))),
            lineType=cv2.LINE_AA,
        )
        image = _paint(image, ink * 0.65, rim)

        sheen = spec.dot_sheen
        if spec.sheen_dot_ids:
            sheen = sheen if dot.id in spec.sheen_dot_ids else 0.0
        if sheen > 0.0:
            offset = radius * 0.32
            gloss = _gaussian_map(
                (SCENE_H, SCENE_W),
                (dot.center[0] - offset, dot.center[1] - offset),
                (radius * 0.55, radius * 0.45),
                -35.0,
                peak=sheen,
            )
            gloss *= disc > 0.35
            image = _screen(image, (255.0, 255.0, 250.0), gloss.astype(np.float32))
    return image


def _glare(
    image: np.ndarray, spec: SceneSpec, rng: np.random.Generator
) -> np.ndarray:
    for glare in spec.glare:
        if glare.kind == "bright_streak":
            sigma = (max(1.0, glare.length / 3.0), max(0.8, glare.sigma[1]))
            field = _gaussian_map(
                (SCENE_H, SCENE_W),
                glare.center,
                sigma,
                glare.angle_deg,
                peak=glare.alpha,
            )
        else:
            irregularity = 0.55 if glare.kind == "specular_blob" else 0.0
            field = _gaussian_map(
                (SCENE_H, SCENE_W),
                glare.center,
                glare.sigma,
                glare.angle_deg,
                peak=glare.alpha,
                irregularity=irregularity,
                rng=rng,
            )
        image = _screen(image, glare.color, field.astype(np.float32))
    return image


def _camera_matrix(camera: Camera, rng: np.random.Generator) -> np.ndarray:
    center = (SCENE_W / 2.0, SCENE_H / 2.0)
    matrix = cv2.getRotationMatrix2D(center, camera.rotation_deg, camera.scale)
    matrix[0, 2] += MARGIN + camera.offset[0]
    matrix[1, 2] += MARGIN + camera.offset[1]
    homography = np.vstack([matrix, [0.0, 0.0, 1.0]]).astype(np.float64)

    if camera.perspective_px > 0.0:
        source = np.float32(
            [[0, 0], [SCENE_W, 0], [SCENE_W, SCENE_H], [0, SCENE_H]]
        )
        dest = cv2.perspectiveTransform(source.reshape(-1, 1, 2), homography).reshape(-1, 2)
        jitter = rng.uniform(
            -camera.perspective_px, camera.perspective_px, size=(4, 2)
        ).astype(np.float32)
        homography = cv2.getPerspectiveTransform(source, dest + jitter)
    return homography


def _apply_camera(image: np.ndarray, camera: Camera, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    homography = _camera_matrix(camera, rng)
    warped = cv2.warpPerspective(
        image,
        homography,
        (OUT_W, OUT_H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if camera.blur_sigma > 0:
        warped = cv2.GaussianBlur(warped, (0, 0), camera.blur_sigma)
    if camera.exposure != 1.0:
        warped = warped * camera.exposure
    if camera.gamma != 1.0:
        table = np.clip(np.arange(256, dtype=np.float32) / 255.0, 0, 1) ** camera.gamma
        warped = np.interp(np.clip(warped, 0, 255), np.arange(256), table * 255.0)
    if camera.noise_sigma > 0:
        warped = warped + rng.normal(0.0, camera.noise_sigma, warped.shape).astype(np.float32)
    return np.clip(warped, 0, 255).astype(np.uint8), homography


# --------------------------------------------------------------------------- #
# ground truth
# --------------------------------------------------------------------------- #
def _polygon_center_radius(polygon: List[List[float]]) -> Tuple[List[float], float]:
    points = np.asarray(polygon, dtype=np.float64)
    center = points.mean(axis=0)
    radius = float(np.mean(np.linalg.norm(points - center, axis=1)))
    return [float(center[0]), float(center[1])], radius


def _region_polygon(glare: Glare) -> List[List[float]]:
    if glare.kind == "bright_streak":
        return rotated_rect_polygon(
            glare.center[0],
            glare.center[1],
            glare.length * 0.5,
            1.7 * max(1.0, glare.sigma[1]),
            glare.angle_deg,
        )
    return circle_polygon(glare.center[0], glare.center[1], glare.gt_radius)


def build_meta(
    stem: str,
    case: str,
    title: str,
    spec: SceneSpec,
    camera: Camera,
    image: np.ndarray,
    homography: np.ndarray,
) -> Dict[str, object]:
    """Ground truth in final image coordinates."""
    shape = image.shape[:2]
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    dot_entries: List[Dict[str, object]] = []
    dot_masks: Dict[str, np.ndarray] = {}
    dot_polygons: List[List[List[float]]] = []
    for dot in spec.dots:
        polygon = transform_polygon(circle_polygon(*dot.center, dot.radius), homography)
        mask = polygon_mask(shape, [polygon])
        dot_masks[dot.id] = mask
        dot_polygons.append(polygon)
        center, radius = _polygon_center_radius(polygon)
        dot_entries.append(
            {
                "id": dot.id,
                "color": dot.color,
                "center": [round(center[0], 2), round(center[1], 2)],
                "radius": round(radius, 2),
                "scene_center": [dot.center[0], dot.center[1]],
                "scene_radius": dot.radius,
                "polygon": [[round(x, 2), round(y, 2)] for x, y in polygon],
            }
        )

    all_ink = polygon_mask(shape, dot_polygons)

    glare_entries: List[Dict[str, object]] = []
    glare_masks: Dict[str, np.ndarray] = {}
    for glare in spec.glare:
        polygon = transform_polygon(_region_polygon(glare), homography)
        mask = polygon_mask(shape, [polygon])
        glare_masks[glare.id] = mask
        center, radius = _polygon_center_radius(polygon)
        area = int(np.count_nonzero(mask))
        ink_overlap = overlap_fraction(mask, all_ink)
        # Local background: an annulus around the region, ignoring marker ink.
        from .geom import annulus_mask  # local import keeps the public API tidy

        ring = annulus_mask(mask, 0, max(4, int(radius)))
        ring = cv2.bitwise_and(ring, cv2.bitwise_not(all_ink))
        ring_values = luma[ring > 0]
        background_p99 = (
            float(np.percentile(ring_values, 99)) if ring_values.size > 0 else float("nan")
        )
        glare_entries.append(
            {
                "id": glare.id,
                "kind": glare.kind,
                "center": [round(center[0], 2), round(center[1], 2)],
                "radius": round(radius, 2),
                "scene_center": [glare.center[0], glare.center[1]],
                "peak_alpha": glare.alpha,
                "color_bgr": list(glare.color),
                "area_px": area,
                "contains_ink": bool(ink_overlap > 0.0),
                "ink_overlap_fraction": round(float(ink_overlap), 4),
                "background_luma_p99": round(background_p99, 2)
                if background_p99 == background_p99
                else None,
                "saturated_background": bool(
                    background_p99 == background_p99 and background_p99 >= 238.0
                ),
                "recoverable": bool(
                    glare.kind != "broad_wash"
                    and area > 0
                    and area <= 0.22 * shape[0] * shape[1]
                ),
                "polygon": [[round(x, 2), round(y, 2)] for x, y in polygon],
            }
        )

    for entry in dot_entries:
        ids = [
            glare_entry["id"]
            for glare_entry in glare_entries
            if float(glare_entry["ink_overlap_fraction"]) > 0.0
            and overlap_fraction(dot_masks[str(entry["id"])], glare_masks[str(glare_entry["id"])]) > 0.0
        ]
        entry["glare_ids"] = ids
        dot_pixels = dot_masks[str(entry["id"])]
        if ids:
            glare_union = np.zeros(shape, dtype=np.uint8)
            for glare_id in ids:
                glare_union = cv2.bitwise_or(glare_union, glare_masks[str(glare_id)])
            entry["glare_overlap_fraction"] = round(
                float(overlap_fraction(dot_pixels, glare_union)), 4
            )
        else:
            entry["glare_overlap_fraction"] = 0.0

    return {
        "image": f"{stem}.png",
        "case": case,
        "title": title,
        "seed": spec.seed,
        "dot_order": spec.order,
        "image_size": [int(image.shape[1]), int(image.shape[0])],
        "scene_size": [SCENE_W, SCENE_H],
        "camera": {
            "rotation_deg": camera.rotation_deg,
            "perspective_px": camera.perspective_px,
            "scale": camera.scale,
            "offset": [camera.offset[0], camera.offset[1]],
            "exposure": camera.exposure,
            "gamma": camera.gamma,
        },
        "scene": {
            "oil_strength": spec.oil_strength,
            "oil_whole_plate": spec.oil_whole_plate,
            "dot_sheen": spec.dot_sheen,
            "plate_gain": spec.plate_gain,
            "exposure": spec.exposure,
        },
        "dots": dot_entries,
        "glare_regions": glare_entries,
    }


def render(stem: str, case: str, title: str, spec: SceneSpec, camera: Camera) -> RenderedScene:
    """Render one photo plus its ground truth."""
    rng = np.random.default_rng(spec.seed)
    image = _background(rng)
    image = _plate(image, spec, rng)
    if spec.oil_strength > 0.0 and spec.order == "oil_before_dots":
        image = _oil(image, spec, rng, over_ink=False)
    image = _dots(image, spec, rng)
    image = _glare(image, spec, rng)
    if spec.oil_strength > 0.0 and spec.order == "oil_after_dots":
        image = _oil(image, spec, rng, over_ink=True)
    if spec.exposure != 1.0:
        image = np.clip(image * spec.exposure, 0, 255)

    post_rng = np.random.default_rng(spec.seed + 991)
    final, homography = _apply_camera(image, camera, post_rng)
    meta = build_meta(stem, case, title, spec, camera, final, homography)
    return RenderedScene(image=final, homography=homography, meta=meta)
