"""Tunable parameters for the phase 1 (glare suppression + marker isolation) pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Dict


@dataclass(frozen=True)
class Phase1Config:
    """All thresholds of the pipeline in one place.

    Luminance and chroma are expressed in 0..255 units.  ``chroma`` is
    ``max(B, G, R) - min(B, G, R)``: near zero for white/specular light and
    large for marker ink.
    """

    # --- stage A: specular candidates -------------------------------------
    bright_luma: float = 155.0
    """Absolute luminance above which a pixel may be specular."""
    bright_rel: float = 38.0
    """How far above the local background luminance a bright pixel must sit."""
    bright_rel_frac: float = 0.12
    """Relative rise over the background, so a bright scene stays detectable."""
    bg_open_ksize: int = 61
    """Minimum structuring element for the local background estimate."""
    bg_open_frac: float = 0.16
    """The element grows to this fraction of the short image side, so that even
    a broad soft highlight is removed by the opening."""
    clip_luma: float = 244.0
    """Luminance of near-clipped pixels; accepted on their own (no contrast)."""
    clip_chroma_max: float = 70.0
    """Near-clipped pixels must be this achromatic to count as specular."""
    min_glare_area: int = 16
    max_glare_area_frac: float = 0.22
    """Larger bright areas are exposure/backlight, not recoverable glare."""
    glow_luma: float = 136.0
    """Soft halo around a detected highlight: bright, achromatic and touching an
    already detected specular region, so the glow around a flare is removed too."""
    glow_grow: int = 18
    """How far the soft halo may reach from a detected specular region."""

    # --- stage B: glare versus marker ink ---------------------------------
    ink_chroma_min: float = 55.0
    """Chroma a pixel needs before it can be marker ink at all."""
    ink_chroma_strong: float = 110.0
    """A component this chromatic is ink no matter how soft its edge is."""
    ink_edge_strength_min: float = 0.20
    """Chroma gradient at the border of a component, divided by its peak chroma.
    Paint has a hard edge, a warm gloss patch fades away smoothly."""
    ink_chroma_hard: float = 150.0
    """Chroma that survives even a strong specular wash."""
    min_ink_area: int = 6
    """Ink blobs smaller than this are noise and must not hole the glare mask."""
    glare_dilate: int = 3
    """Glare mask growth before inpainting (catches soft halos)."""
    ink_protect_dilate: int = 2
    """Ink mask growth that keeps marker edges out of the inpaint mask."""

    # --- stage C: suppression ---------------------------------------------
    inpaint_radius: int = 5
    restore_ink: bool = True
    restore_close_cap: int = 26
    injected_chroma_margin: float = 12.0
    """A repair may add at most this much chroma over the original pixel."""
    injected_chroma_floor: float = 22.0
    """Absolute chroma a repair may add on an achromatic pixel."""
    """Upper bound for the morphological closing used to bridge a glared core."""
    hull_fill_max_ratio: float = 3.2
    """Only bridge an ink region over a highlight while its convex hull stays
    below this multiple of the visible ink area (protects smeared trails)."""

    # --- stage D: marker-like blobs only ----------------------------------
    marker_chroma_min: float = 50.0
    marker_chroma_peak_min: float = 60.0
    min_marker_area: int = 14
    max_marker_area_frac: float = 0.05
    min_circularity: float = 0.30
    min_fill_ratio: float = 0.38
    max_aspect: float = 4.5
    marker_ink_dilate: int = 1
    marker_hull_fill_max_ratio: float = 2.6
    """Same convex-hull guard as above, used when filling marker holes."""

    # --- reporting --------------------------------------------------------
    glare_on_ink_note_min: float = 0.02
    """Glare coverage of a marker blob above which the report adds a note."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **kwargs: float) -> "Phase1Config":
        return replace(self, **kwargs)


DEFAULT_CONFIG = Phase1Config()
