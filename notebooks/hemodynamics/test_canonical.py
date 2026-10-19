# Check canonical using controlled inputs and isolated files.
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hemodynamics.canonical import (CheckpointStore, DiskSpaceStop, assert_patient_independence,
                                   ensure_disk_space, load_case, patient_splits, verify_case, write_case)


class CanonicalTests(unittest.TestCase):
    def setUp(self):
        # Collect the per-record values into a table for comparison.
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.wave = pd.DataFrame({"time_sec": np.arange(100) / 100,
                                  "ART": np.linspace(65, 95, 100), "ECG": np.sin(np.arange(100)),
                                  "PPG": np.cos(np.arange(100))})
        self.numeric = pd.DataFrame({"time_sec": [0., .5], "MAP": [70., 72.]})

    def tearDown(self):
        self.temp.cleanup()

    def commit(self):
        return write_case(self.root, "mover", "case1", self.wave, self.numeric,
                          {"patient_id": "patient1", "split": "train", "min_free_space_bytes": 0, "art_original_fs": 180, "art_final_fs": 100})

    def test_real_parquet_readback_and_corruption(self):
        # Verify real parquet readback and corruption.
        metadata = self.commit()
        self.assertTrue(verify_case(self.root, metadata))
        loaded = load_case(self.root, "mover", "case1")
        np.testing.assert_allclose(loaded["ART"], self.wave.ART, rtol=1e-6)
        self.assertEqual(loaded["patient_id"], "patient1")
        with (self.root / metadata["map_output_path"]).open("ab") as f:
            f.write(b"corruption")
        self.assertFalse(verify_case(self.root, metadata))

    def test_checkpoint_reopen_merge(self):
        # Verify checkpoint reopen merge.
        store = CheckpointStore(self.root, "mover")
        store.update("case1", status="processing", retry_count=1)
        restarted = CheckpointStore(self.root, "mover")
        restarted.update("case1", status="completed", output_bytes=12)
        self.assertEqual(restarted.get("case1")["retry_count"], 1)
        self.assertEqual(restarted.rows()[0]["status"], "completed")

    def test_disk_guard(self):
        # Verify disk guard.
        with patch("hemodynamics.canonical.shutil.disk_usage") as usage:
            usage.return_value.free = 110
            self.assertEqual(ensure_disk_space(self.root, 100, 10), 110)
            with self.assertRaises(DiskSpaceStop):
                ensure_disk_space(self.root, 100, 11)

    def test_missing_channel_not_fabricated(self):
        self.wave.PPG = np.nan
        with self.assertRaises(ValueError):
            self.commit()

    def test_nonmonotonic_times_rejected(self):
        self.wave.loc[2, "time_sec"] = 0
        with self.assertRaises(ValueError):
            self.commit()

    def test_patient_splits_and_leakage(self):
        # Verify patient splits and leakage.
        splits = patient_splits([f"p{i}" for i in range(100)], seed=5052)
        self.assertEqual(splits.split.value_counts().to_dict(), {"train": 70, "validation": 15, "test": 15})
        pd.testing.assert_frame_equal(splits, patient_splits([f"p{i}" for i in reversed(range(100))], seed=5052))
        self.assertTrue(all(v == 0 for v in assert_patient_independence(splits).values()))
        bad = pd.concat([splits, pd.DataFrame({"patient_id": [splits.iloc[0].patient_id], "split": ["test" if splits.iloc[0].split != "test" else "train"]})])
        with self.assertRaises(AssertionError):
            assert_patient_independence(bad)

    def test_atomic_failure_no_committed_case(self):
        # Verify atomic failure no committed case.
        with patch("hemodynamics.canonical.os.replace", side_effect=OSError("interrupted rename")):
            with self.assertRaises(OSError):
                self.commit()
        self.assertFalse(any(self.root.rglob("*.tmp")))
        self.assertFalse((self.root / "data/parquet/mover/case_id=case1/metadata.json").exists())

    def test_compressed_output_budget_stops_before_commit(self):
        # Verify compressed output budget stops before commit.
        with self.assertRaises(DiskSpaceStop):
            write_case(self.root, "mover", "case1", self.wave, self.numeric,
                       {"min_free_space_bytes": 0, "max_case_output_bytes": 1})
        self.assertFalse((self.root / "data/parquet/mover/case_id=case1/metadata.json").exists())
        self.assertFalse(any(self.root.rglob("*.tmp")))

    def test_nested_manifest_metadata_is_portable(self):
        # Verify nested manifest metadata is portable.
        from hemodynamics.canonical import write_manifest
        write_manifest(self.root, "example", pd.DataFrame({"case_id": ["case1"], "metadata": [{"empty": {}}]}))
        result = pd.read_parquet(self.root / "data/manifests/example.parquet")
        self.assertEqual(json.loads(result.metadata.iloc[0]), {"empty": {}})

    def test_path_traversal_rejected(self):
        with self.assertRaises(ValueError):
            write_case(self.root, "mover", "../outside", self.wave, self.numeric, {})


if __name__ == "__main__":
    unittest.main()
