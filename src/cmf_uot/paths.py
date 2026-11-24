"""Default folders. Override with the environment variables CMF_UOT_DATA and
CMF_UOT_RESULTS (useful when the 6.6 GB dataset lives on another disk)."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CMF_UOT_DATA", REPO_ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("CMF_UOT_RESULTS", REPO_ROOT / "results"))
CONFIG_FILE = REPO_ROOT / "configs" / "grid.cfg"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
