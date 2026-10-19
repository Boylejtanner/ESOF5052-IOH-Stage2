# Check snapshot using controlled inputs and isolated files.
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from hemodynamics import snapshot as S


def make_root(base):
    root = Path(base)
    for sub in ("data/manifests", "configs", "notebooks/hemodynamics", "docs", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    # Write the verified values in the pipeline's Parquet table format.
    index = pd.DataFrame({"dataset": ["mover", "mover", "vitaldb", "vitaldb"], "case_id": ["a", "b", "1", "2"],
                          "patient_key": ["mover:p1", "mover:p2", "vitaldb:9", "vitaldb:9"],
                          "split_within_dataset": ["train", "test", "train", "train"], "split_pooled": ["train", "test", "validation", "validation"],
                          "waveform_sha256": ["w1", "w2", "w3", "w4"], "map_sha256": ["m1", "m2", "m3", "m4"], "extra": [1, 2, 3, 4]})
    # Write the verified values in the pipeline's Parquet table format.
    index.to_parquet(root / "data/manifests/unified_case_index.parquet", index=False)
    index.to_parquet(root / "data/manifests/vitaldb_patient_splits.parquet", index=False)
    (root / "data/manifests/mover_cohort_frozen.json").write_text(json.dumps({"patient_list_sha256": "abc", "rule_sha256": "def", "rung": "R1"}))
    # Store the settings and results as JSON for later inspection.
    (root / "configs/hemodynamics.json").write_text(json.dumps({"random_seed": 5052, "patient_split_seed": 5052}))
    (root / "notebooks/n.ipynb").write_text("{}")
    (root / "notebooks/hemodynamics/m.py").write_text("x = 1\n")
    (root / "notebooks/hemodynamics/__pycache__").mkdir()
    (root / "notebooks/hemodynamics/__pycache__/m.pyc").write_bytes(b"junk")
    (root / "docs/d.md").write_text("doc")
    (root / "results/r.json").write_text("{}")
    return root, index


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root, self.index = make_root(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_tag_and_verify(self):
        # Verify round trip tag and verify.
        tag = S.snapshot_dataset_v1(self.root)
        dest = self.root / tag["snapshot_dir"]
        self.assertEqual(S.verify_snapshot(dest), [])
        # Check paths against the expected result.
        manifest = json.loads((dest / "SNAPSHOT_MANIFEST.json").read_text())
        paths = {e["path"] for e in manifest["files"]}
        self.assertIn("notebooks/hemodynamics/m.py", paths)
        self.assertIn("data/manifests/unified_case_index.parquet", paths)
        # Compare the result with the expected value or tolerance.
        self.assertFalse(any("__pycache__" in p for p in paths))
        self.assertTrue((dest / "dataset_v1.json").exists())
        self.assertEqual((tag["n_cases"], tag["n_patients"], tag["mover_rung"]), (4, 3, "R1"))
        self.assertEqual(tag["counts_by_split"]["vitaldb/train"], {"cases": 2, "patients": 1})
        self.assertEqual(tag["mover_patient_list_sha256"], "abc")
        self.assertTrue((self.root / S.TAG_FILE).exists())

    def test_tampering_and_missing_files_are_detected(self):
        # Verify tampering and missing files are detected.
        dest = self.root / S.snapshot_dataset_v1(self.root)["snapshot_dir"]
        (dest / "docs/d.md").write_text("changed")
        (dest / "results/r.json").unlink()
        problems = S.verify_snapshot(dest)
        self.assertEqual(sorted(p.split(":")[0] for p in problems), ["changed", "missing"])

    def test_content_hash_ignores_row_order_and_extra_columns_but_not_splits(self):
        # Verify content hash ignores row order and extra columns but not splits.
        base = S.index_content_hash(self.index)
        self.assertEqual(base, S.index_content_hash(self.index.sample(frac=1, random_state=3).assign(extra=0)))
        moved = self.index.copy()
        moved.loc[0, "split_within_dataset"] = "validation"
        self.assertNotEqual(base, S.index_content_hash(moved))

    def test_idempotent_until_forced_or_changed(self):
        # Verify idempotent until forced or changed.
        first = S.snapshot_dataset_v1(self.root)
        again = S.snapshot_dataset_v1(self.root)
        self.assertTrue(again["reused"])
        self.assertEqual(first["snapshot_dir"], again["snapshot_dir"])
        self.assertEqual(len(list((self.root / "backups").glob("dataset_v1_*"))), 1)
        forced = S.snapshot_dataset_v1(self.root, force=True)
        self.assertNotEqual(first["snapshot_dir"], forced["snapshot_dir"])
        self.assertEqual(len(list((self.root / "backups").glob("dataset_v1_*"))), 2)

    def test_guard_raises_on_changed_index_and_missing_tag(self):
        # Verify guard raises on changed index and missing tag.
        with self.assertRaises(FileNotFoundError):
            S.assert_dataset_v1(self.root)
        S.snapshot_dataset_v1(self.root)
        self.assertEqual(S.assert_dataset_v1(self.root)["version"], "v1")
        changed = self.index.copy()
        changed.loc[1, "split_pooled"] = "train"
        changed.to_parquet(self.root / "data/manifests/unified_case_index.parquet", index=False)
        with self.assertRaises(RuntimeError):
            S.assert_dataset_v1(self.root)
        self.assertEqual(S.assert_dataset_v1(self.root, allow_changed=True)["version"], "v1")

    def test_extra_destination_is_copied_and_verified(self):
        with tempfile.TemporaryDirectory() as other:
            tag = S.snapshot_dataset_v1(self.root, extra_dest=other)
            self.assertEqual(S.verify_snapshot(tag["extra_dest"]), [])

    def test_extra_destination_added_after_the_first_run(self):
        # Verify extra destination added after the first run.
        first = S.snapshot_dataset_v1(self.root)
        self.assertIsNone(first["extra_dest"])
        with tempfile.TemporaryDirectory() as other:
            # Check reused against the expected result.
            again = S.snapshot_dataset_v1(self.root, extra_dest=other)
            self.assertTrue(again["reused"])
            self.assertEqual(S.verify_snapshot(again["extra_dest"]), [])
            self.assertEqual(json.loads((self.root / S.TAG_FILE).read_text())["extra_dest"], again["extra_dest"])


if __name__ == "__main__":
    unittest.main()
