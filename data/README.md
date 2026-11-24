# Acoustic-camera dataset for CMF-UOT

Recordings used in *On Sensor Array and Camera Fusion via Unbalanced Optimal Transport* (I. Jaouedi, G. Chardon, J. Picheral, submitted to *Signal Processing*).

- **DOI:** https://doi.org/10.5281/zenodo.XXXXXXX
- **License:** CC BY 4.0
- **Size:** 7.1 GB (45 recordings of 157 MB each, plus camera and calibration files)

Download the record and extract it here, so that this folder contains `waves/`, `camera_detections/`, `images/`, `transformation_matrix_A.npy` and `references.json`. Alternatively, keep it anywhere and set `CMF_UOT_DATA=/path/to/folder`.

## Experimental setup

Recorded in a gymnasium at CentraleSupélec (Gif-sur-Yvette, France).

| Item | Description |
|---|---|
| Microphone array | 32 MEMS microphones on four horizontal lines, about 1 m × 0.6 m, irregular spacing |
| Sources | 4 omnidirectional broadband loudspeakers (Visaton BF32, 150 Hz–20 kHz) emitting white noise |
| Camera | smartphone camera, images 4080 × 3072 px |
| Sampling rate | **TO CONFIRM** (50 kHz assumed by the code) |
| Recording length | 612,000 samples per channel (597 segments of 1024 samples) |

## Coordinate frame

Right-handed frame, in metres in the code and centimetres in the JSON files: X horizontal along the scene, Y the depth axis seen by the array, Z vertical. The array is placed at x = −3 m (`get_array_from_npy(x_position=-3)`), the camera at (0.07, −3.00, 0.90) m. The axes are drawn on the setup photo, Fig. 8a of the article.

## Files

### `waves/` — acoustic recordings

`numpy` arrays, `float64`, shape `(612000, 32)` (samples × microphones).

| File | Content |
|---|---|
| `array.npy` | microphone positions in the array plane, shape `(32, 3)`, metres (converted to the scene frame by `cmf_uot.geometry.get_array_from_npy`) |
| `mesure_manip{n}_all.npy` | configuration *n*, all four loudspeakers active |
| `mesure_manip{n}_{k}.npy` | configuration *n*, loudspeaker *k* alone (k = 0…3) **TO CONFIRM** |
| `mesure_manip_{d}m_*.npy` | additional series with the array at d = 3, 4, 5, 6 m (not used in the article) **TO CONFIRM** |
| `mesure_reference_*.npy` | reference recording **TO CONFIRM** |

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
