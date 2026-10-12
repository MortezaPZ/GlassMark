# GlassMark — phase 1

GlassMark reads a photo of a glass plate that carries an oily film and coloured
marker dots, reduces the specular highlights and oil glare, and returns the
marker dots as blobs.

**Phase 1 only.** It does not classify the camera angle, does not place plates in
a layout, does not translate colours into symbols, and does not pick a "largest"
mark or rotate anything to 12 o'clock. No 3D reconstruction, no web interface, no
GPU: OpenCV and NumPy on the CPU.

## Scope and physical assumptions

* The scene is a rectangular glass plate on a table.
* Marker ink is blue, red, green or yellow, in dots of different sizes.
* Oil (olive or ordinary) may be applied **before or after** the dots; the
  pipeline assumes no particular order.
* Photos may come from several viewpoints.
* Until real client photos arrive, the synthetic set is the source of truth for
  the tests. The results here say nothing yet about real oily glass.

The separation of glare from ink relies on one physical fact: reflected light is
nearly achromatic and fades away smoothly, while marker paint is strongly
saturated and stops abruptly. Chroma (`max(B,G,R) - min(B,G,R)`) and the chroma
gradient at the border of a region are therefore the two main measurements.

## Install

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# Linux / macOS: .venv/bin/python -m pip install -r requirements.txt
```

`opencv-python-headless` is listed because the pipeline only reads and writes
image files. An already installed `opencv-python` works just as well.

## Run

One path from nothing to a verified result:

```bash
python scripts/glassmark.py all
```

which is exactly:

```bash
python scripts/make_synthetic.py
python scripts/run_phase1.py --input data/synthetic --output data/out/phase1
python -m pytest tests/test_phase1.py -q
```

Other entry points:

```bash
python scripts/glassmark.py synth            # only generate synthetic plates
python scripts/glassmark.py run              # only run the pipeline
python scripts/glassmark.py test             # only run the tests
python scripts/glassmark.py special-cases    # cases A-H with expected vs actual
python scripts/run_phase1.py --input path/to/one/photo.png --output data/out/phase1
```

Threshold overrides for experiments (all optional):
`--bright-luma`, `--bright-rel`, `--ink-chroma-min`, `--glare-dilate`,
`--inpaint-radius`, `--marker-chroma-min`, `--min-marker-area`.

## Outputs

For every input `<stem>.png`, the pipeline writes `<output>/<stem>/`:

| file | meaning |
| --- | --- |
| `00_input.png` | copy of the input, so the folder is self-contained |
| `01_glare_mask.png` | white = glare, i.e. the bright pixels the pipeline treats as reflected light. Bright pixels that are only marker ink are **not** glare and stay black. |
| `02_marker_mask.png` | white = the marker dots that were kept |
| `03_cleaned.png` | glare reduced, marker dots still visible |
| `04_dots_only.png` | black background, only the kept dot pixels |
| `report.json` | counts, one bounding box per blob, the estimated colour per blob, why candidates were dropped, and a note for every blob that sat under a highlight |

`report.json` contains no product or vendor names — only measurements of the
image that was processed.

## Pipeline stages

The stages are separate functions so each can be called and tested alone.

| stage | function | what it does |
| --- | --- | --- |
| A | `glare.find_bright_regions` | finds very bright / specular regions: absolute brightness, brightness above a wide local background, near-clipped achromatic pixels, plus the soft glow around a found highlight. Oversized bright areas are reported as exposure rather than recoverable glare. |
| B | `glare.separate_glare_from_ink` | separates oil glare from marker ink, per pixel: strongly chromatic pixels, or chromatic pixels with a hard edge, are ink and are protected; smooth tinted patches are glare. |
| C | `suppress.suppress_glare` | rebuilds the part of a dot a highlight washed out (convex hull of the visible ink, painted with the ink colour of the rim), then inpaints the remaining glare and refuses to invent colour that was not there. |
| D | `markers.detect_markers` | keeps only marker-like blobs: chromatic, small, compact, dot-shaped; drops leftover shine and records the reason for every dropped candidate. |

## Synthetic data

`scripts/make_synthetic.py` writes `data/synthetic/<stem>.png` plus a
`<stem>.json` sidecar with the ground truth: per dot the colour, centre,
approximate radius, polygon and how much glare covered it; per glare region the
kind, centre, radius, polygon, ink overlap and whether its surroundings are
already saturated. The base series repeats one physical plate (same dots, same
sizes) with oil before or after the dots and different viewpoints. The special
cases A–H each add one situation:

| case | situation |
| --- | --- |
| A | glare next to a red dot, not overlapping |
| B | glare on top of a blue dot |
| C | oil film over the whole plate, markers still readable |
| D | yellow dot against a yellowish specular patch |
| E | tiny green dot next to a large red dot |
| F | two similarly sized dots, one with a shine on it |
| G | strong overexposure, clipped whites |
| H | one plate from two viewpoints (angles are not classified) |

## Tests

```bash
python -m pytest tests -q
```

`tests/test_phase1.py` fails when a known dot is missing from
`02_marker_mask.png`, when a glare-only blob is kept as a marker, or when
`03_cleaned.png` still shows a large specular peak on a known glare-only region;
it also checks the colour read back from the cleaned image and that a dot under a
highlight stays coloured. `tests/test_special_cases.py` runs A–H and checks the
artefacts exist. Both print a short pass/fail table at the end, and
`scripts/run_special_cases.py` writes the same information as
`data/out/phase1_special/SPECIAL_CASES.md`.

## Known limits

* Verified on synthetic images only; real oily glass is phase 2.
* A **yellow** dot and a warm-tinted specular patch can share a hue. They are
  separated by saturation and by edge sharpness, which needs a reasonably sharp,
  non-blurred photo.
* A very large bloom (a highlight that covers a big part of the frame) is treated
  as exposure and skipped rather than repaired.
* A dot that is **completely** washed out, with no hue left anywhere in the
  pixels, cannot be recovered from a single photo. Such a dot comes back
  desaturated (the rebuilt core keeps roughly the chroma of the visible rim).
* The rebuilt core of a washed out dot is flat: it inherits the ink colour of the
  rim, not the original marker texture.
* One photo at a time; nothing here uses several views of the same plate together.
