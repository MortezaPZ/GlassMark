"""Catalogue of synthetic plates: the base series plus special cases A-H.

Every entry is a full description of one photo (plate state + camera).  The
generator script renders the catalogue; the tests read the same catalogue to
know what to expect.  Case H is two photos of the *same* plate from two
viewpoints: both entries share one :class:`SceneSpec`, only the camera differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .synth import Camera, Dot, Glare, SceneSpec

WHITE = (250.0, 250.0, 246.0)
COOL = (252.0, 248.0, 238.0)
WARM = (150.0, 185.0, 215.0)


@dataclass
class CaseDefinition:
    stem: str
    case: str
    title: str
    spec: SceneSpec
    camera: Camera

    @property
    def letter(self) -> str:
        return self.case


def base_dots() -> List[Dot]:
    """The one physical plate used by the base series and by case H."""
    return [
        Dot("d_blue_1", "blue", (235.0, 180.0), 13.0),
        Dot("d_red_1", "red", (520.0, 155.0), 16.0),
        Dot("d_green_1", "green", (250.0, 430.0), 9.0),
        Dot("d_yellow_1", "yellow", (590.0, 415.0), 14.0),
        Dot("d_blue_2", "blue", (420.0, 300.0), 8.0),
        Dot("d_green_2", "green", (690.0, 255.0), 11.0),
    ]


def _glare(id_: str, center: Tuple[float, float], sigma: Tuple[float, float], alpha: float,
           angle: float = 0.0, kind: str = "specular_blob",
           color: Tuple[float, float, float] = WHITE, length: float = 0.0) -> Glare:
    return Glare(
        id=id_,
        kind=kind,
        center=center,
        sigma=sigma,
        alpha=alpha,
        angle_deg=angle,
        color=color,
        length=length,
    )


def base_scene(seed: int, order: str, glare: List[Glare], oil: float = 0.16,
               whole_plate: bool = False, plate_gain: float = 1.0,
               sheen: float = 0.0, sheen_ids: Tuple[str, ...] = ()) -> SceneSpec:
    return SceneSpec(
        seed=seed,
        dots=base_dots(),
        glare=glare,
        order=order,
        oil_strength=oil,
        oil_whole_plate=whole_plate,
        dot_sheen=sheen,
        sheen_dot_ids=sheen_ids,
        plate_gain=plate_gain,
    )


def all_cases() -> List[CaseDefinition]:
    """All synthetic plates in a deterministic order."""
    base_glare_1 = [
        _glare("g1", (700.0, 470.0), (26.0, 16.0), 0.85, 20.0),
        _glare("g2", (430.0, 120.0), (6.0, 6.0), 0.85, 12.0, "bright_streak", COOL, 190.0),
    ]
    base_glare_2 = [
        _glare("g1", (330.0, 250.0), (18.0, 13.0), 0.80, -15.0),
        _glare("g2", (560.0, 520.0), (7.0, 7.0), 0.82, 100.0, "bright_streak", COOL, 220.0),
    ]
    base_glare_3 = [
        _glare("g1", (520.0, 155.0), (17.0, 12.0), 0.82, 0.0),
        _glare("g2", (250.0, 520.0), (24.0, 16.0), 0.70, 30.0),
    ]
    base_glare_4 = [
        _glare("g1", (590.0, 415.0), (15.0, 11.0), 0.80, 10.0),
        _glare("g2", (420.0, 300.0), (12.0, 9.0), 0.70, -20.0),
    ]

    cases: List[CaseDefinition] = [
        CaseDefinition(
            "base_01_oil_before",
            "base",
            "oil applied before the dots, two highlights",
            base_scene(101, "oil_before_dots", base_glare_1),
            Camera(rotation_deg=1.5, perspective_px=6.0, scale=0.95),
        ),
        CaseDefinition(
            "base_02_oil_after",
            "base",
            "oil applied after the dots, one highlight near the ruler edge",
            base_scene(102, "oil_after_dots", base_glare_2, oil=0.20),
            Camera(rotation_deg=-2.2, perspective_px=8.0, scale=0.94, exposure=1.03),
        ),
        CaseDefinition(
            "base_03_glare_on_red",
            "base",
            "hard case: highlight sitting on the red dot",
            base_scene(103, "oil_before_dots", base_glare_3),
            Camera(rotation_deg=2.6, perspective_px=10.0, scale=0.93),
        ),
        CaseDefinition(
            "base_04_glare_on_yellow_and_blue",
            "base",
            "oil after the dots, highlights on the yellow and the small blue dot",
            base_scene(104, "oil_after_dots", base_glare_4, oil=0.18),
            Camera(rotation_deg=-3.4, perspective_px=12.0, scale=0.92, exposure=0.97),
        ),
        # ---------------------------------------------------------------- A
        CaseDefinition(
            "case_A_glare_next_to_red",
            "A",
            "highlight next to the red dot without touching it",
            base_scene(
                201,
                "oil_before_dots",
                [
                    _glare("g1", (566.0, 155.0), (13.0, 10.0), 0.80, 0.0),
                    _glare("g2", (240.0, 520.0), (28.0, 18.0), 0.75, 25.0),
                    _glare("g3", (620.0, 300.0), (6.0, 6.0), 0.80, 70.0, "bright_streak", COOL, 160.0),
                ],
            ),
            Camera(rotation_deg=1.0, perspective_px=5.0, scale=0.95),
        ),
        # ---------------------------------------------------------------- B
        CaseDefinition(
            "case_B_glare_on_top_of_blue",
            "B",
            "highlight centred on the blue dot",
            base_scene(
                202,
                "oil_before_dots",
                [
                    _glare("g1", (235.0, 180.0), (14.0, 11.0), 0.85, 0.0),
                    _glare("g2", (650.0, 520.0), (20.0, 14.0), 0.70, -10.0),
                ],
            ),
            Camera(rotation_deg=-1.8, perspective_px=7.0, scale=0.95),
        ),
        # ---------------------------------------------------------------- C
        CaseDefinition(
            "case_C_oil_film_whole_plate",
            "C",
            "oil film over the whole plate, markers must stay readable",
            base_scene(
                203,
                "oil_after_dots",
                [
                    _glare("g1", (760.0, 250.0), (30.0, 18.0), 0.72, 40.0),
                    _glare("g2", (300.0, 250.0), (6.0, 6.0), 0.82, 95.0, "bright_streak", COOL, 200.0),
                ],
                oil=0.26,
                whole_plate=True,
            ),
            Camera(rotation_deg=2.0, perspective_px=9.0, scale=0.93, exposure=1.04),
        ),
        # ---------------------------------------------------------------- D
        CaseDefinition(
            "case_D_yellow_vs_warm_glare",
            "D",
            "yellow dot against a warm tinted specular patch",
            base_scene(
                204,
                "oil_before_dots",
                [
                    _glare("g1", (450.0, 200.0), (34.0, 22.0), 0.85, 15.0, "warm_blob", WARM),
                    _glare("g2", (760.0, 500.0), (20.0, 14.0), 0.70, -30.0),
                ],
            ),
            Camera(rotation_deg=-1.2, perspective_px=6.0, scale=0.95),
        ),
        # ---------------------------------------------------------------- E
        CaseDefinition(
            "case_E_tiny_green_big_red",
            "E",
            "tiny green dot next to a large red dot",
            SceneSpec(
                seed=205,
                dots=[
                    Dot("d_tiny_green", "green", (295.0, 300.0), 5.0),
                    Dot("d_big_red", "red", (330.0, 330.0), 18.0),
                    Dot("d_blue_1", "blue", (560.0, 200.0), 12.0),
                    Dot("d_yellow_1", "yellow", (600.0, 470.0), 14.0),
                ],
                glare=[
                    _glare("g1", (470.0, 520.0), (22.0, 15.0), 0.75, 20.0),
                    _glare("g2", (400.0, 110.0), (6.0, 6.0), 0.80, 15.0, "bright_streak", COOL, 180.0),
                ],
                order="oil_after_dots",
                oil_strength=0.18,
            ),
            Camera(rotation_deg=1.7, perspective_px=7.0, scale=0.95),
        ),
        # ---------------------------------------------------------------- F
        CaseDefinition(
            "case_F_two_similar_dots_one_shine",
            "F",
            "two equally sized dots, one carries a gloss spot",
            SceneSpec(
                seed=206,
                dots=[
                    Dot("d_blue_shiny", "blue", (300.0, 330.0), 12.0),
                    Dot("d_blue_matte", "blue", (580.0, 330.0), 12.0),
                    Dot("d_red_1", "red", (440.0, 520.0), 10.0),
                ],
                glare=[
                    _glare("g1", (440.0, 180.0), (25.0, 17.0), 0.70, -25.0),
                    _glare("g2", (720.0, 430.0), (6.0, 6.0), 0.80, 110.0, "bright_streak", COOL, 170.0),
                ],
                order="oil_before_dots",
                oil_strength=0.14,
                dot_sheen=0.50,
                sheen_dot_ids=("d_blue_shiny",),
            ),
            Camera(rotation_deg=-2.5, perspective_px=8.0, scale=0.94),
        ),
        # ---------------------------------------------------------------- G
        CaseDefinition(
            "case_G_overexposed",
            "G",
            "clipped whites: bright plate, highlight still above the background",
            base_scene(
                207,
                "oil_before_dots",
                [
                    _glare("g1", (300.0, 200.0), (26.0, 17.0), 0.80, 20.0),
                    _glare("g2", (620.0, 430.0), (20.0, 14.0), 0.70, -15.0),
                    _glare("g3", (480.0, 550.0), (6.0, 6.0), 0.82, 90.0, "bright_streak", COOL, 200.0),
                ],
                oil=0.10,
                plate_gain=3.8,
            ),
            Camera(rotation_deg=1.2, perspective_px=6.0, scale=0.95, exposure=1.02),
        ),
    ]

    # -------------------------------------------------------------------- H
    h_scene = base_scene(
        208,
        "oil_before_dots",
        [
            _glare("g1", (700.0, 470.0), (26.0, 16.0), 0.85, 20.0),
            _glare("g2", (520.0, 155.0), (17.0, 12.0), 0.78, 0.0),
            _glare("g3", (300.0, 120.0), (6.0, 6.0), 0.82, 30.0, "bright_streak", COOL, 180.0),
        ],
    )
    cases.extend(
        [
            CaseDefinition(
                "case_H_view1",
                "H",
                "same plate as case H view 2, first viewpoint",
                h_scene,
                Camera(rotation_deg=4.5, perspective_px=12.0, scale=0.90,
                       offset=(-14.0, 6.0), exposure=0.95),
            ),
            CaseDefinition(
                "case_H_view2",
                "H",
                "same plate as case H view 1, second viewpoint",
                h_scene,
                Camera(rotation_deg=-5.0, perspective_px=14.0, scale=0.97,
                       offset=(18.0, -8.0), exposure=1.06, gamma=0.96),
            ),
        ]
    )
    return cases


def case_by_stem(stem: str) -> CaseDefinition:
    for case in all_cases():
        if case.stem == stem:
            return case
    raise KeyError(f"unknown synthetic case: {stem}")


def special_case_map() -> Dict[str, List[str]]:
    """Letter -> stems, for the A-H report."""
    result: Dict[str, List[str]] = {}
    for case in all_cases():
        if case.case in {"A", "B", "C", "D", "E", "F", "G", "H"}:
            result.setdefault(case.case, []).append(case.stem)
    return result
