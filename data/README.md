# Acoustic-camera dataset for CMF-UOT

Recordings used in *On Sensor Array and Camera Fusion via Unbalanced Optimal Transport* (I. Jaouedi, G. Chardon, J. Picheral, submitted to *Signal Processing*).

- **DOI:** https://doi.org/10.5281/zenodo.24512049
- **License:** CC BY 4.0
- **Size:** 2.5 GB to download (44 recordings stored as compressed integer ADC counts, plus camera and calibration files)

With the code repository (https://github.com/ilyes-jaouedi/cmf-uot-fusion), run `poetry run python -m cmf_uot.download` (add `--paper-only` for the 0.2 GB used in the article). Zenodo stores the files without folders; the download command puts them back in `waves/`, `camera_detections/` and `images/` (mapping in `folders.json`), checks them against `checksums.sha256`, and rebuilds each recording as the float64 `.npy` file used in the article (see below). To keep the data elsewhere, set `CMF_UOT_DATA=/path/to/folder`.

Recorded on 23 July 2025. The published images have their GPS metadata removed, and a passer-by seen through a window in `ref_3.jpg` is blurred.

## Experimental setup

Recorded in a gymnasium at CentraleSupélec (Gif-sur-Yvette, France).

| Item | Description |
|---|---|
| Microphone array | 32 MEMS microphones on four horizontal lines, about 1 m × 0.6 m, irregular spacing |
| Sources | 4 omnidirectional broadband loudspeakers (Visaton BF32, 150 Hz–20 kHz) emitting white noise |
| Camera | smartphone camera, images 4080 × 3072 px |
| Sampling rate | 50 kHz (acquisition setting `srinv = 9`: fs = 500 kHz / (srinv + 1)) |
| Recording length | 612,000 samples per channel (597 segments of 1024 samples) |

## Coordinate frame

Right-handed frame, in metres in the code and centimetres in the JSON files: X horizontal along the scene, Y the depth axis seen by the array, Z vertical. The array is placed at x = −3 m (`get_array_from_npy(x_position=-3)`), the camera at (0.07, −3.00, 0.90) m. The axes are drawn on the setup photo, Fig. 8a of the article.

## Files

### `waves/` — acoustic recordings

Each recording has shape `(612000, 32)` (samples × microphones). The acquisition system delivers 20-bit samples, so every value is an integer multiple of 7.08 × 10⁻⁶. On Zenodo each recording is a compressed NumPy archive `mesure_*.npz` holding

- `counts`: the integer ADC counts, `int32`, shape `(612000, 32)`;
- `scale`: `7.08e-6`;

and the signal is `counts * scale` (float64). This is lossless: it reproduces the float64 arrays used in the article bit for bit, and the download command writes them back as `mesure_*.npy` files whose SHA-256 match `original_npy_checksums.sha256`. Without the code:

```python
import numpy as np
with np.load("mesure_manip1_all.npz") as z:
    x = z["counts"] * z["scale"]        # float64, shape (612000, 32)
```

The file names below are those of the reconstructed `.npy` files (`.npz` on Zenodo).

| File | Content |
|---|---|
| `array.npy` | microphone positions in the array plane, shape `(32, 3)`, metres (converted to the scene frame by `cmf_uot.geometry.get_array_from_npy`) |
| `mesure_manip{n}_all.npy` | configuration *n*, all four loudspeakers active |
| `mesure_manip{n}_{k}.npy` | configuration *n*, loudspeaker *k* alone (k = 0…3) |
| `mesure_manip_{d}m_{all,k}.npy` | distance series: array moved to x = −d m, d = 3, 4, 5, 6 (source positions in `mesure_manip_array.json`); used in the ICASSP 2026 paper, not in this article. `mesure_manip_5m_0` is missing: the stored file was damaged (its first 5 MiB were overwritten with zeros) |
| `mesure_reference_{all,k}.npy` | additional reference recording, not used in the article |

Configurations:

| n | Use in the article | Sources (x, y, z) in cm, from `camera_detections/` |
|---|---|---|
| 1 | Standard case, Section 6.1 | S1 (93, 65, 112), S2 (−22, 0, 131), S3 (31, −44.5, 158), S4 (−5, −99.5, 123) |
| 2 | Acoustic ambiguity, Section 6.2.1 | S1 moved to (90, 0, 129): same y as S2 |
| 3 | not used (S1 at (90, 20, 129)) | |
| 4 | Visual occlusion, Section 6.2.2 | S1 at (107, 72, 135), on the camera's line of sight to S3 |

### `camera_detections/` — camera detections (JSON)

Lists of detections `{"u", "v", "x", "y", "z"}`: pixel coordinates (u, v) of each loudspeaker in the image, and its measured position (x, y, z) in cm, used as ground truth. The detections were annotated by hand (see Section 6 of the article).

| File | Use |
|---|---|
| `mesure_manip_1.json`, `_2.json`, `_3.json` | detections of configurations 1–3 |
| `mesure_manip_4.json` | configuration 4 as seen by the camera: three detections, S1 occluded |
| `mesure_manip_4_all4.json` | configuration 4 with all four positions (ground truth) |
| `mesure_manip_array.json` | detections for the distance series |

### `images/`

Camera images of each configuration (`mesure_manip_{n}.jpg`, `mesure_manip_array.jpg`) and calibration images (`ref_{1..4}.jpg`).

### Calibration

| File | Content |
|---|---|
| `references.json` | 12 calibration points: pixel (u, v) ↔ position (x, y, z) in cm |
| `transformation_matrix_A.npy` | 2 × 3 least-squares map from (u, v, 1) to (x, z), fitted on `references.json` (`cmf_uot.camera.pixel_to_world`) |

## Loading

```python
from cmf_uot.data import load_sigs, load_camera_detections
from cmf_uot.paths import DATA_DIR

sigs = load_sigs(manip=1)                                   # (612000, 32), configuration 1
dets = load_camera_detections(DATA_DIR / "camera_detections" / "mesure_manip_1.json")
```

## Citation

Please cite the article above when using this dataset.
