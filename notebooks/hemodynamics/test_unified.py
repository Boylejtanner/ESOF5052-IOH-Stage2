# Check unified using controlled inputs and isolated files.
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hemodynamics import unified
from hemodynamics.canonical import CheckpointStore, write_case

FS = 100.0
SETTINGS = Path(__file__).resolve().parents[2] / "configs" / "hemodynamics.json"


def synthetic_case(root, dataset, case_id, patient, split, status, seconds=700, bad_art=False):
    n = int(seconds * FS)
    t = np.arange(n) / FS
    art = (80 + 20 * np.sin(2 * np.pi * 1.2 * t)).astype(np.float32)
    if bad_art:
        art[:] = 0.0
        art[0] = 1.0
    # Collect the per-record values into a table for comparison.
    wave = pd.DataFrame({"time_sec": t, "ART": art, "ECG": np.sin(t).astype(np.float32), "PPG": np.cos(t).astype(np.float32)})
    mt = np.arange(0, seconds, 2.0)
    numeric = pd.DataFrame({"time_sec": mt, "MAP": np.full(len(mt), 80.0, dtype=np.float32)})
    written = write_case(root, dataset, case_id, wave, numeric, {"patient_id": patient, "split": split, "min_free_space_bytes": 0,
                                                                 "duration_sec": seconds, "time_reference": f"{dataset}_clock"})
    skip = {"case_id", "status"}
    CheckpointStore(root, dataset).update(case_id, **{k: v for k, v in written.items() if k not in skip}, status=status)


class UnifiedTests(unittest.TestCase):
    def setUp(self):
        # Create the destination directory before writing any outputs.
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs" / "hemodynamics.json").write_text(SETTINGS.read_text())
        for i, split in enumerate(["train", "train", "train", "validation", "test"]):
            synthetic_case(self.root, "vitaldb", str(i + 1), 100 + i, split, "complete")
        for i, split in enumerate(["train", "train", "validation", "test"]):
            synthetic_case(self.root, "mover", f"m{i}", f"patient{i}", split, "completed", bad_art=(i == 3))
        CheckpointStore(self.root, "mover").update("excluded1", status="excluded")

    def tearDown(self):
        self.temp.cleanup()

    def test_index_is_incremental_and_rule_aware(self):
        # Verify index is incremental and rule aware.
        first = unified.index_ready_windows(self.root, "vitaldb", workers=1, progress=False)
        self.assertEqual((first["completed_cases"], first["indexed_now"], first["skipped_current"]), (5, 5, 0))
        windows = pd.read_parquet(self.root / "data/manifests/ready_windows_vitaldb.parquet")
        # Compare the result with the expected value or tolerance.
        self.assertEqual(sorted(windows.case_id.unique()), ["1", "2", "3", "4", "5"])
        self.assertTrue(windows.is_ready.any())
        second = unified.index_ready_windows(self.root, "vitaldb", workers=1, progress=False)
        self.assertEqual((second["indexed_now"], second["skipped_current"]), (0, 5))
        # Check indexed now against the expected result.
        config_path = self.root / "configs" / "hemodynamics.json"
        config = json.loads(config_path.read_text())
        config["window_rule"] = {**config["window_rule"], "art_map_tolerance_mmHg": 25}
        config_path.write_text(json.dumps(config))
        third = unified.index_ready_windows(self.root, "vitaldb", workers=1, progress=False)
        self.assertEqual(third["indexed_now"], 5, "a changed rule must recompute")

    def test_new_case_only_indexes_that_case_and_keeps_old_rows(self):
        # Verify new case only indexes that case and keeps old rows.
        unified.index_ready_windows(self.root, "vitaldb", workers=1, progress=False)
        before = pd.read_parquet(self.root / "data/manifests/ready_windows_vitaldb.parquet")
        synthetic_case(self.root, "vitaldb", "6", 106, "train", "complete")
        result = unified.index_ready_windows(self.root, "vitaldb", workers=1, progress=False)
        self.assertEqual((result["indexed_now"], result["skipped_current"]), (1, 5))
        after = pd.read_parquet(self.root / "data/manifests/ready_windows_vitaldb.parquet")
        self.assertEqual(len(after), len(before) + int((after.case_id == "6").sum()))
        pd.testing.assert_frame_equal(before.reset_index(drop=True), after.loc[after.case_id != "6"].reset_index(drop=True))

    def test_parallel_matches_serial(self):
        # Verify parallel matches serial.
        unified.index_ready_windows(self.root, "mover", workers=1, progress=False)
        serial = pd.read_parquet(self.root / "data/manifests/ready_windows_mover.parquet")
        (self.root / "data/manifests/ready_windows_mover.parquet").unlink()
        unified.index_ready_windows(self.root, "mover", workers=2, progress=False)
        parallel = pd.read_parquet(self.root / "data/manifests/ready_windows_mover.parquet")
        pd.testing.assert_frame_equal(serial.reset_index(drop=True), parallel.reset_index(drop=True))

    def test_flat_art_case_has_no_ready_windows(self):
        # Verify flat ART case has no ready windows.
        unified.index_ready_windows(self.root, "mover", workers=1, progress=False)
        stats = pd.read_parquet(self.root / "data/manifests/mover_case_window_stats.parquet").set_index("case_id")
        self.assertEqual(stats.loc["m3", "n_ready_windows"], 0)
        self.assertGreater(stats.loc["m0", "n_ready_windows"], 0)

    def test_unified_index_schema_splits_and_experiments(self):
        # Verify unified index schema splits and experiments.
        with self.assertRaises(FileNotFoundError):
            unified.build_unified_index(self.root, progress=False, auto_index=False)
        unified.index_ready_windows(self.root, "mover", workers=1, progress=False)
        # Compare the result with the expected value or tolerance.
        index = unified.build_unified_index(self.root, progress=False)
        self.assertTrue((self.root / "data/manifests/vitaldb_case_window_stats.parquet").exists(), "missing dataset is indexed automatically")
        self.assertEqual(index.groupby("dataset").size().to_dict(), {"mover": 4, "vitaldb": 5})
        self.assertTrue(index.patient_key.str.contains(":").all())
        self.assertTrue(index.has_window_stats.all())
        self.assertTrue(set(index.split_pooled) <= {"train", "validation", "test"})
        for column in ("waveform_path", "map_path", "waveform_sha256", "ready_span_min_theta_70", "time_reference", "valid_art"):
            self.assertIn(column, index)
        # Compare the result with the expected value or tolerance.
        self.assertEqual(set(index.loc[index.dataset == "mover", "time_reference"]), {"mover_clock"})
        self.assertTrue((self.root / "data/manifests/unified_case_index.parquet").exists())
        pooled = unified.experiment_split(index, "E1_pooled")
        self.assertEqual(sum(len(v) for v in pooled.values()), len(index))
        # Compare the result with the expected value or tolerance.
        e2 = unified.experiment_split(index, "E2_mover_to_vitaldb")
        self.assertEqual(set(e2["train"].dataset), {"mover"})
        self.assertEqual(set(e2["test"].dataset), {"vitaldb"})
        # Compare the result with the expected value or tolerance.
        e3x = unified.experiment_split(index, "E3x_vitaldb_to_mover_external")
        self.assertEqual(len(e3x["test"]), 4)
        paths = unified.parquet_paths(self.root, e2["train"])
        self.assertTrue(all(Path(p).is_file() for p in paths))
        with self.assertRaises(KeyError):
            unified.experiment_split(index, "nope")

    def test_vitaldb_status_word_and_mover_status_word_are_both_completed(self):
        self.assertEqual(len(unified._completed_rows(self.root, "vitaldb")), 5)
        self.assertEqual(len(unified._completed_rows(self.root, "mover")), 4)


if __name__ == "__main__":
    unittest.main()
