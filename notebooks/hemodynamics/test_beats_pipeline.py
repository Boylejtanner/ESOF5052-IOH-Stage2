# Check beats pipeline using controlled inputs and isolated files.
import json
from pathlib import Path
import _thread
import tempfile
import threading
import unittest

import numpy as np
import pandas as pd

from hemodynamics import beats_pipeline as P
from hemodynamics import snapshot as S
from hemodynamics.test_beats import synth

FS = 100.0


def build_root(base, per_dataset=3, duration=100.0, config_beats=None):
    root = Path(base)
    for sub in ("data/manifests", "configs", "notebooks/hemodynamics", "docs", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows, wins = [], {"mover": [], "vitaldb": []}
    for ds in ("mover", "vitaldb"):
        for i in range(per_dataset):
            # Write the verified values in the pipeline's Parquet table format.
            cid = f"{ds[0]}{i}"
            t, r, ecg, art, ppg = synth(duration=duration, seed=hash((ds, i)) % 1000)
            d = root / "data/parquet" / ds / f"case_id={cid}"
            # Write the verified values in the pipeline's Parquet table format.
            d.mkdir(parents=True)
            pd.DataFrame({"time_sec": t, "ART": art.astype("float32"), "ECG": ecg.astype("float32"), "PPG": ppg.astype("float32")}).to_parquet(d / "waveform.parquet", index=False)
            rows.append({"dataset": ds, "case_id": cid, "patient_key": f"{ds}:{i}", "split_within_dataset": "train", "split_pooled": "train",
                         "waveform_sha256": f"w{cid}", "map_sha256": f"m{cid}", "waveform_path": f"data/parquet/{ds}/case_id={cid}/waveform.parquet",
                         "n_ready_windows": 3 + i, "ready_span_min_theta_50": duration / 60 * 0.9})
            wins[ds].append(pd.DataFrame({"case_id": cid, "t_start_sec": [5.0, 35.0], "t_end_sec": [65.0, 95.0], "min_valid_frac": 0.9, "is_ready": True}))
    pd.DataFrame(rows).to_parquet(root / "data/manifests/unified_case_index.parquet", index=False)
    for ds, frames in wins.items():
        pd.concat(frames).to_parquet(root / f"data/manifests/ready_windows_{ds}.parquet", index=False)
    cfg = {"random_seed": 5052, "patient_split_seed": 5052, "min_free_space_gb": 0, "beats": {"workers": 2, "min_free_after_gb": 0, **(config_beats or {})}}
    (root / "configs/hemodynamics.json").write_text(json.dumps(cfg))
    S.snapshot_dataset_v1(root)
    return root


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Keep test files in an isolated directory, away from real dataset outputs.
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = build_root(cls.temp.name, per_dataset=6)
        cls.first = P.run(cls.root, workers=3, progress=False)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_outputs_manifest_and_roundtrip(self):
        # Verify outputs manifest and roundtrip.
        self.assertEqual(self.first["status"], "COMPLETE", self.first)
        self.assertEqual((self.first["processed"], self.first["failed"]), (12, 0))
        manifest = pd.read_parquet(self.root / "data/manifests/beats_manifest.parquet")
        # Compare the result with the expected value or tolerance.
        self.assertEqual(len(manifest), 12)
        self.assertTrue((manifest.n_beats > 80).all() and (manifest.valid_frac > 0.9).all())
        df = P.load_beats(self.root, "mover", "m0")
        self.assertEqual(list(df.columns), P.B.BEAT_COLUMNS)
        # Compare the result with the expected value or tolerance.
        self.assertEqual(df.beat_idx.tolist(), list(range(len(df))))
        self.assertTrue(df.t_r.between(5.0, 95.0).all())
        self.assertAlmostEqual(float(df.ptt_ecg_art.median()), 0.15 + 0.18169 * 0.12, delta=0.02)
        waves, beat_idx = P.load_beat_waveforms(self.root, "mover", "m0")
        self.assertEqual(waves["art"].dtype, np.float16)
        self.assertEqual(waves["art"].shape, (len(df), 200))
        self.assertEqual(beat_idx.tolist(), df.beat_idx.tolist())
        self.assertTrue(np.isfinite(waves["ecg"].astype(float)).any())

    def test_rerun_skips_finished_cases_and_config_change_invalidates(self):
        # Verify rerun skips finished cases and config change invalidates.
        again = P.run(self.root, workers=2, progress=False)
        self.assertEqual((again["processed"], again["skipped_finished"]), (0, 12))
        self.assertIn("nothing to do", again["reason"])
        # Check processed against the expected result.
        meta = P.case_dir(self.root, "vitaldb", "v0") / "beats_meta.json"
        meta.unlink()
        self.assertEqual(P.run(self.root, workers=2, progress=False)["processed"], 1)
        self.assertTrue(meta.exists())
        cfg_path = self.root / "configs/hemodynamics.json"
        original = cfg_path.read_text()
        try:
            # Check processed against the expected result.
            cfg = json.loads(original)
            cfg["beats"]["phase_samples"] = 100
            cfg_path.write_text(json.dumps(cfg))
            redo = P.run(self.root, workers=3, progress=False, case_ids=[("mover", "m0")])
            self.assertEqual(redo["processed"], 1)
            waves, _ = P.load_beat_waveforms(self.root, "mover", "m0")
            self.assertEqual(waves["art"].shape[1], 100)
        finally:
            cfg_path.write_text(original)
            P.run(self.root, workers=3, progress=False, case_ids=[("mover", "m0")])

    def test_projection_and_storage_budget(self):
        # Verify projection and storage budget.
        proj = P.project(self.root)
        self.assertEqual(proj["sample_cases"], 12)
        self.assertGreater(proj["projected_total_gb"], 0)
        self.assertGreater(proj["bytes_per_beat_waveforms"], 100)
        self.assertTrue((self.root / "results/beats_projection.json").exists())
        cfg_path = self.root / "configs/hemodynamics.json"
        original = cfg_path.read_text()
        try:
            # Check changed against the expected result.
            cfg = json.loads(original)
            cfg["beats"]["max_projected_waveform_gb"] = proj["projected_waveform_gb"] / 3
            cfg_path.write_text(json.dumps(cfg))
            result = P.apply_storage_budget(self.root, proj)
            self.assertTrue(result["changed"])
            self.assertEqual(json.loads(cfg_path.read_text())["beats"]["phase_samples"], 100)
        finally:
            cfg_path.write_text(original)


class RobustnessTests(unittest.TestCase):
    def test_guard_requires_the_dataset_tag(self):
        # Verify guard requires the dataset tag.
        with tempfile.TemporaryDirectory() as base:
            root = build_root(base, per_dataset=1)
            (root / "data/manifests/dataset_v1.json").unlink()
            with self.assertRaises(FileNotFoundError):
                P.run(root, progress=False)

    def test_a_broken_case_is_isolated_and_logged(self):
        # Verify a broken case is isolated and logged.
        with tempfile.TemporaryDirectory() as base:
            root = build_root(base, per_dataset=2)
            (root / "data/parquet/mover/case_id=m1/waveform.parquet").write_bytes(b"not parquet")
            result = P.run(root, workers=2, progress=False)
            self.assertEqual(result["status"], "INCOMPLETE")
            self.assertEqual((result["processed"], result["failed"]), (3, 1))
            self.assertIn("m1", (root / "data/manifests/beats_errors.csv").read_text())
            self.assertFalse((P.case_dir(root, "mover", "m1") / "beats_meta.json").exists())

    def test_interrupt_pauses_cleanly_and_resume_finishes(self):
        # Verify interrupt pauses cleanly and resume finishes.
        with tempfile.TemporaryDirectory() as base:
            root = build_root(base, per_dataset=4)
            timer = threading.Timer(3.0, _thread.interrupt_main)
            timer.start()
            try:
                first = P.run(root, workers=1, progress=False)
            finally:
                timer.cancel()
            self.assertIn(first["status"], ("PAUSED", "COMPLETE"))
            self.assertFalse((root / "data/checkpoints/beats/run.lock").exists(), "the lock must be released")
            second = P.run(root, workers=2, progress=False)
            self.assertEqual(second["status"], "COMPLETE")
            self.assertEqual(len(pd.read_parquet(root / "data/manifests/beats_manifest.parquet")), 8)

    def test_second_concurrent_run_is_refused(self):
        # Verify second concurrent run is refused.
        import os, psutil
        with tempfile.TemporaryDirectory() as base:
            root = build_root(base, per_dataset=1)
            lock = root / "data/checkpoints/beats/run.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            parent = psutil.Process(os.getppid())                   # a different, live process holds the lock
            lock.write_text(json.dumps({"pid": parent.pid, "create_time": parent.create_time()}))
            with self.assertRaises(RuntimeError):
                P.run(root, progress=False)
            self.assertTrue(lock.exists(), "a live owner's lock must not be removed")
            lock.write_text(json.dumps({"pid": 99999999, "create_time": 0}))      # a dead owner's lock is taken over
            self.assertEqual(P.run(root, workers=2, progress=False)["status"], "COMPLETE")


if __name__ == "__main__":
    unittest.main()
