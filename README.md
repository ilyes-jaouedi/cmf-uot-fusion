# CMF-UOT: sensor array and camera fusion via unbalanced optimal transport

Reference implementation of

> I. Jaouedi, G. Chardon, J. Picheral, **On Sensor Array and Camera Fusion via Unbalanced Optimal Transport**, submitted to *Signal Processing* (Elsevier), 2026.

CMF-UOT fuses a microphone array with a camera *inside* the source estimation: the covariance matrix fitting (CMF) criterion of the array is regularized by an unbalanced optimal transport (UOT) term that couples the acoustic power map with the camera detections. The array brings power and coarse position, the camera brings accurate bearings; no training data is needed for the fusion itself.

This repository contains the solvers, the Cramér–Rao bound computations, and one script per figure of the article. The acoustic-camera recordings are published separately on Zenodo (see [Data](#data)).

---

## Contents

```
cmf-uot-fusion/
├── src/cmf_uot/            Python package
│   ├── geometry.py         search grid, microphone array, free-field steering vectors
│   ├── covariance.py       sample spatial covariance at one frequency bin
│   ├── camera.py           pixel-to-world mapping, angular cost matrix C
│   ├── bounds.py           Fisher information of the camera measurement (CRB)
│   ├── data.py             dataset loaders
│   ├── paths.py            default data / results folders
│   └── solvers/
│       ├── cmf.py          CMF (acoustic only), greedy coordinate descent
│       ├── cmf_uot1.py     CMF-UOT1: power b decoupled from the plan P, block coordinate descent
│       └── cmf_uot2.py     CMF-UOT2: hard coupling b = P1, greedy coordinate descent
├── experiments/
│   ├── simulations/        Monte Carlo study and CRB (Section 5), Figs. 1, 2, 4-7
│   └── real_data/          experiments on recorded data (Section 6), Figs. 3, 8-15
├── configs/grid.cfg        reconstruction grid of the real-data experiments
├── results/                outputs; ships with the CSV results used in the article
├── tests/                  unit tests (pytest)
└── data/                   dataset location (downloaded separately)
```

## Installation

Requires Python ≥ 3.10 and [Poetry](https://python-poetry.org/docs/#installation).

```bash
git clone https://github.com/OWNER/cmf-uot-fusion.git
cd cmf-uot-fusion
poetry install          # creates .venv and installs the package and its dependencies
poetry run pytest       # optional: runs the unit tests
```

## Data

The real-data experiments use the acoustic-camera dataset recorded for the article (32-channel MEMS microphone array, four loudspeakers, one camera), available under CC BY 4.0:

> **Dataset DOI:** https://doi.org/10.5281/zenodo.XXXXXXX

```bash
poetry run python -m cmf_uot.download --paper-only   # recordings used in the article (0.5 GB)
poetry run python -m cmf_uot.download                # complete dataset (7.1 GB)
```

Files are placed in `data/` and checked against their SHA-256 checksums (layout and file description in [`data/README.md`](data/README.md)). To keep the data elsewhere, set `CMF_UOT_DATA=/path/to/data`.

The simulation experiments do not need the dataset.

## Reproducing the figures

All commands are run from the repository root; figures and CSV files are written to `results/` (or `$CMF_UOT_RESULTS`).

| Figure | Content | Command | Needs dataset |
|---|---|---|---|
| 1 | Simulation geometry | `poetry run python experiments/simulations/fig01_simulation_geometry.py` | no |
| 2 | Cramér–Rao bounds vs. SNR | `poetry run python experiments/simulations/fig02_crb_vs_snr.py` | no |
| 3 | Spatial configuration of the experiment | `poetry run python experiments/real_data/fig03_configuration.py` | yes |
| 4–6 | RMSE vs. SNR, snapshots, camera noise | `poetry run python experiments/simulations/fig04_06_plot_sweeps.py` | no |
| 7 | Sensitivity to λ, β, α | `poetry run python experiments/simulations/fig07_plot_sensitivity.py` | no |
| 8b | Microphone array layout | `poetry run python experiments/real_data/fig08_array_layout.py` | yes |
| 9–11 | Standard case | `poetry run python experiments/real_data/standard_case.py` | yes |
| 12–13 | Acoustic ambiguity | `poetry run python experiments/real_data/acoustic_ambiguity.py` | yes |
| 14–15 | Visual occlusion | `poetry run python experiments/real_data/visual_occlusion.py` | yes |

Figures 4–7 are plotted from the CSV files in `results/`, which are the Monte Carlo results reported in the article. To recompute them (100 trials per point, long), run first:

```bash
poetry run python experiments/simulations/run_snr_sweep.py         # sim_results_snr.csv       (Fig. 4)
poetry run python experiments/simulations/run_snapshot_sweep.py    # sim_results_snapshots.csv (Fig. 5)
poetry run python experiments/simulations/run_camera_noise_sweep.py  # sim_results_camera.csv  (Fig. 6)
poetry run python experiments/simulations/run_sensitivity.py       # sensitivity_*.csv         (Fig. 7)
```

Each sweep accepts `--trials N` for a quick run (written with a `_test` suffix). The random seed is fixed (`SEED = 0`), so a full run reproduces the CSV files.

Each real-data script prints the localization error of every source, as reported in Section 6 of the article.

## Minimal example

```python
import numpy as np
from cmf_uot.geometry import get_grid_from_config, get_array_from_npy, freefield
from cmf_uot.covariance import calculate_sigma
from cmf_uot.data import load_sigs, load_camera_detections
from cmf_uot.camera import compute_C
from cmf_uot.paths import DATA_DIR
from cmf_uot.solvers.cmf_uot2 import TransportOptimizer_numpy

Grid, xg, yg, zg = get_grid_from_config()                  # 462,000 candidate points
Array = get_array_from_npy()                               # 32 microphones
Sigma = calculate_sigma(load_sigs(manip=1), Lseg=1024, freq=4000)
G = freefield(Array, Grid, k=2 * np.pi * 4000 / 343.0)     # steering matrix

detections = load_camera_detections(DATA_DIR / "camera_detections" / "mesure_manip_1.json")
A = np.load(DATA_DIR / "transformation_matrix_A.npy")
C = compute_C(Grid, detections, A)                         # camera cost matrix

a = np.ones(len(detections)) / len(detections)
opt = TransportOptimizer_numpy(C, a, Sigma, G, lambda_=..., mu=...)   # see standard_case.py
opt.optimize_greedy_coordinate_descent(num_iterations=200, verbose=False)
power_map = opt.P.sum(axis=1)                              # fused acoustic power on the grid
```

`experiments/real_data/standard_case.py` shows the full pipeline, including the heuristic choice of λ and μ (Section 4.4 of the article).

## Naming

The two fusion variants are called as in the article:

| Article | Module | Class | Coupling between power b and plan P |
|---|---|---|---|
| CMF-UOT1 | `solvers/cmf_uot1.py` | `TransportOptimizerUnifiedGCD` | soft, quadratic penalty (b and P separate variables) |
| CMF-UOT2 | `solvers/cmf_uot2.py` | `TransportOptimizer_numpy` | hard, b = P1 |

## Citation

If you use this code or the dataset, please cite the article (a `CITATION.cff` file is provided):

```bibtex
@article{jaouedi2026cmfuot,
  title   = {On Sensor Array and Camera Fusion via Unbalanced Optimal Transport},
  author  = {Jaouedi, Ilyes and Chardon, Gilles and Picheral, Jos{\'e}},
  journal = {Signal Processing},
  note    = {Submitted},
  year    = {2026}
}
```

## License

Code: [MIT](LICENSE). Dataset: CC BY 4.0 (see the Zenodo record).

## Acknowledgements

This work was supported by the industrial chair "Massive and Heterogeneous Data Processing for Smart Vehicles", operated by CentraleSupélec and sponsored by FORVIA.
