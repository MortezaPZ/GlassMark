# GlassMark — Complete Pipeline (Phases 1 – 4)

GlassMark reads a photo of a glass plate carrying an oily film and coloured
marker dots, eliminates specular highlights and oil glare, calibrates and rectifies
perspective geometry, and semantically decodes the marker dot arrangement.

**Phases 1 through 4 are fully implemented, verified, and test-covered.**

---

## Architecture and Pipeline Stages

| Phase | Stage | Function / Module | Description |
| --- | --- | --- | --- |
| **Phase 1** | A | `glare.find_bright_regions` | Detects specular candidates, local background rise, near-clipped regions, and highlight halos. |
| **Phase 1** | B | `glare.separate_glare_from_ink` | Separates oil glare from marker ink using chroma and edge gradient dynamics. |
| **Phase 1** | C | `suppress.suppress_glare` | Rebuilds washed-out ink cores and inpaints glare regions using surrounding plate slate color. |
| **Phase 1** | D | `markers.detect_markers` | Keeps marker-like circular blobs, drops shine/smears, and classifies BGR colors. |
| **Phase 2** | E | `rectify.detect_plate_corners` & `compute_rectification` | Detects the 4 plate corners and computes the homography to warp perspective into a canonical orthogonal top-down view (720x530). |
| **Phase 3** | F | `decode.decode_layout` | Identifies the primary anchor dot (largest dot), aligns orientation to 12 o'clock, and extracts a viewpoint-invariant symbol sequence + SHA256 hash. |
| **Phase 4** | G | `server.create_server` / Web GUI & REST API | Lightweight server with zero external web dependencies providing interactive browser UI and REST endpoints. |

---

## Outputs

For every input `<stem>.png`, the pipeline produces in `<output>/<stem>/`:

| File | Meaning |
| --- | --- |
| `00_input.png` | Copy of the input photo |
| `01_glare_mask.png` | White = glare treated as reflected light |
| `02_marker_mask.png` | White = kept marker dots |
| `03_cleaned.png` | Glare reduced, marker dots visible and restored |
| `04_dots_only.png` | Black background, only kept marker dot pixels |
| `05_rectified.png` | Orthogonal canonical top-down view of the plate (720x530) |
| `report.json` | Detailed metrics across all stages: bright regions, ink separation, marker blobs, rectification homography, anchor, 12 o'clock clock degrees, symbol sequence, and hash |

---

## Quickstart & CLI Usage

### 1. Installation
```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt   # Windows
# Linux / macOS: .venv/bin/python -m pip install -r requirements.txt
```

### 2. Run Everything & Verify Tests
```bash
python scripts/glassmark.py all
```
Which runs in order:
1. `python scripts/make_synthetic.py`
2. `python scripts/run_phase1.py --input data/synthetic --output data/out/phase1`
3. `python -m pytest tests -q` (all 25 tests)

### 3. Start Web UI & REST API Server
```bash
python scripts/glassmark.py serve --port 8080
```
Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/) in your web browser.

#### REST API Endpoints:
* `POST /api/process`: Upload raw image bytes or multipart form. Returns JSON with stages E-F data and base64-encoded output images.
* `GET /api/health`: Healthcheck endpoint.

---

## Tests and Verification
```bash
python -m pytest tests
```
The test suite covers:
* `tests/test_phase1.py`: Dot coverage, glare suppression, chromatic stability, core restoration.
* `tests/test_special_cases.py`: Special boundary cases A through H.
* `tests/test_phases_2_and_3.py`: Quadrilateral plate detection, canonical 720x530 warping, viewpoint-invariance (identical sequence and hash on multi-view Case H).
* `tests/test_phase4_server.py`: Live HTTP server endpoints, web GUI delivery, REST API processing.
