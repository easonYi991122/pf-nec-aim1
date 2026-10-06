"""The only external-path configuration for the code-only handoff."""
import os
from pathlib import Path
import tempfile

EXPORT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("PF_DATA_ROOT", EXPORT_ROOT / "data")).resolve()
CACHE_ROOT = Path(os.environ.get("PF_CACHE_ROOT", EXPORT_ROOT / ".private")).resolve()
RUN_ROOT = Path(os.environ.get("PF_RUN_DIR", CACHE_ROOT / "runs")).resolve()


def require_data():
    if "PF_DATA_ROOT" not in os.environ:
        raise ValueError("Set PF_DATA_ROOT to your authorized read-only data directory")
    for path in (DATA_ROOT / "NEC Cleaned Data.csv", DATA_ROOT / "Raw CSV Files"):
        if not path.exists():
            raise FileNotFoundError("Required authorized data input is missing: " + path.name)
    return DATA_ROOT


def writable(path):
    path = Path(path).resolve()
    if path.is_relative_to(DATA_ROOT) or not any(path.is_relative_to(root) for root in (CACHE_ROOT, RUN_ROOT)):
        raise ValueError("Output must be inside PF_CACHE_ROOT or PF_RUN_DIR, disjoint from PF_DATA_ROOT")
    return path


def initialize():
    for root in (CACHE_ROOT, RUN_ROOT, CACHE_ROOT / "tmp", CACHE_ROOT / "mpl"):
        writable(root).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = tempfile.tempdir = str(CACHE_ROOT / "tmp")
    os.environ["MPLCONFIGDIR"] = str(CACHE_ROOT / "mpl")
    return CACHE_ROOT
