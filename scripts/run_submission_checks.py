"""Run focused, data-free checks of the submitted processing and modeling code."""
from pathlib import Path
import os
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "4"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))
sys.path.insert(0, str(ROOT / "notebooks" / "hemodynamics"))

TESTS = [
    "test_canonical",
    "test_windows",
    "test_mover_signals",
    "test_unified",
    "test_labels",
    "test_beats",
    "test_bshi",
    "test_bshi_variants",
    "test_independent_features",
    "test_bshi_model",
    "test_bshi_train",
    "test_bshi_data",
    "test_bshi_aug",
    "test_bshi_store",
    "test_eda_metrics",
    "test_vitaldb_pipeline.NotebookStructureTests",
]


def main():
    """Load a cross-pipeline test subset without starting acquisition or a cohort run."""
    suite = unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
