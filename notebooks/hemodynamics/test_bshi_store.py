import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hemodynamics import bshi as S
from hemodynamics import bshi_data as D
from hemodynamics import bshi_store as ST
from hemodynamics.test_bshi import World
from hemodynamics.test_bshi_data import FAKE


def prepared_rows(w, n=64, seed=3):
    rows = w.sample(n, seed=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    rows["patient_key"] = rows.dataset + "-" + rows.case_id.astype(str)
    rows["aut65_low_coverage"] = rng.random(len(rows)) < 0.2
    rows["time_to_event_sec"] = np.where(rng.random(len(rows)) < 0.3, rng.random(len(rows)) * 240, np.nan).astype(np.float32)
    rows["sampling_weight"] = (rng.random(len(rows)) * 3 + 0.5).astype(np.float32)
    return rows


class TestStore(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.w = World(cls.tmp.name)
        cls.rows = prepared_rows(cls.w)
        cases = sorted(cls.rows.patient_key.unique())
        cls.train_p = set(cases[: max(1, len(cases) * 2 // 3)])
        cls.frames = {"train": cls.rows[cls.rows.patient_key.isin(cls.train_p)], "validation": cls.rows[~cls.rows.patient_key.isin(cls.train_p)]}
        cls.out = Path(cls.tmp.name) / "store"
        cls.summary = ST.build_store(cls.w.root, cls.frames, cls.out, shard_windows=12, workers=2, progress=None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_summary_and_layout(self):
        self.assertEqual(self.summary["windows"], len(self.rows))
        self.assertTrue((self.out / "MANIFEST.sha256").exists() and (self.out / "summary.json").exists())
        for split in ("train", "validation"):
            self.assertTrue(list((self.out / split).glob("shard_*/waves_f16.npy")))

    def test_shards_hold_whole_patients_and_splits_are_patient_disjoint(self):
        seen = {}
        for split in ("train", "validation"):
            st = ST.Store(self.out, split)
            for d in st.dirs:
                for p in pd.read_parquet(d / "meta.parquet").patient_key.unique():
                    self.assertNotIn(p, seen, "a patient appears in two shards")
                    seen[p] = split
        self.assertEqual(set(seen), set(self.rows.patient_key))
        self.assertTrue(all(seen[p] == "train" for p in self.train_p))

    def test_round_trip_equals_a_fresh_assembly(self):
        st = ST.Store(self.out, "validation")
        a = st.gather(np.arange(len(st)))
        rows = st.meta.merge(self.rows.drop(columns=["split"], errors="ignore"), on="window_id", suffixes=("", "_r"))
        b = S.assemble(self.w.root, rows, "eap", cache=S.CaseCache(), return_times=True)
        self.assertEqual(a["window_id"].tolist(), rows.window_id.tolist())
        self.assertLess(float(np.abs(a["waves"].astype(np.float32) - b["x"][:, :3]).max()), 2e-2)          # float16 storage
        self.assertTrue(np.array_equal(a["scal"], ST._scal_matrix(b["x"])))
        self.assertTrue(np.array_equal(a["masks"].astype(bool), b["beat_mask"]))
        self.assertTrue(np.array_equal(a["ctx"], b["context"]))
        self.assertTrue(np.array_equal(a["t_rel"], b["t_rel"]) and np.array_equal(a["rr"], b["rr"]))      # beat times for gap augmentation are kept
        self.assertTrue(np.array_equal(a["reg_ok"].astype(bool), ~rows.aut65_low_coverage.to_numpy()))
        self.assertTrue(np.allclose(a["sampling_weight"], rows.sampling_weight.to_numpy()))
        self.assertTrue(np.array_equal(a["y_event"], b["y_event"]))
        self.assertTrue(np.allclose(a["time_to_event_sec"], rows.time_to_event_sec.to_numpy(), equal_nan=True))

    def test_manifest_detects_a_changed_byte(self):
        ok, bad = ST.verify_manifest(self.out, workers=2)
        self.assertTrue(ok and not bad)
        victim = next((self.out / "validation").glob("shard_*/ctx_f32.npy"))
        raw = bytearray(victim.read_bytes())
        raw[-1] ^= 0xFF
        victim.write_bytes(bytes(raw))
        try:
            ok, bad = ST.verify_manifest(self.out, workers=2)
            self.assertFalse(ok)
            self.assertEqual(bad, [victim.relative_to(self.out).as_posix()])
        finally:
            raw[-1] ^= 0xFF
            victim.write_bytes(bytes(raw))
        self.assertTrue(ST.verify_manifest(self.out, workers=2)[0])

    def test_batches_cover_every_window_once_and_are_deterministic(self):
        st = ST.Store(self.out, "train")
        ids = np.concatenate([b["window_id"].numpy() for b in st.batches(8, shuffle=True, seed=1)])
        self.assertEqual(sorted(ids.tolist()), sorted(self.frames["train"].window_id.tolist()))
        again = np.concatenate([b["window_id"].numpy() for b in st.batches(8, shuffle=True, seed=1)])
        other = np.concatenate([b["window_id"].numpy() for b in st.batches(8, shuffle=True, seed=2)])
        self.assertTrue(np.array_equal(ids, again))
        self.assertFalse(np.array_equal(ids, other))
        b = next(iter(st.batches(5, shuffle=False, limit=5)))
        self.assertEqual(tuple(b["waves"].shape), (5, 3, 60, 200))
        self.assertEqual(tuple(b["scal"].shape), (5, 60, 17))
        self.assertEqual(b["beat_mask"].dtype, torch.bool)
        self.assertEqual(len(list(st.batches(8, shuffle=False, limit=10))), 2)

    def test_preprocessing_a_store_batch_equals_preprocessing_the_assembled_batch(self):
        st = ST.Store(self.out, "validation")
        sb = next(iter(st.batches(len(st), shuffle=False)))
        rows = st.meta.merge(self.rows.drop(columns=["split"], errors="ignore"), on="window_id", suffixes=("", "_r"))
        for variant in ("eap", "ea"):
            assembled = S.assemble(self.w.root, rows, variant, cache=S.CaseCache())
            ab = {k: torch.from_numpy(v) for k, v in assembled.items()}
            ab["reg_mask"] = torch.from_numpy(~rows.aut65_low_coverage.to_numpy())
            pre = D.Preprocessor(variant, FAKE, device="cpu", channels_last=False)
            p1, p2 = pre(sb), pre(ab)
            self.assertTrue(torch.equal(p1["x"][:, 3:], p2["x"][:, 3:]) or variant == "ea")
            mask = torch.ones(1, 7, 1, 200)
            from hemodynamics.bshi_model import input_masks
            mask, _ = input_masks(variant)
            self.assertLess(float(((p1["x"] - p2["x"]) * mask).abs().max()), 2e-2)             # what the model can see is the same (float16 on planes 0-2)
            self.assertTrue(torch.allclose(p1["context"], p2["context"]))
            self.assertTrue(torch.equal(p1["beat_mask"], p2["beat_mask"]), variant)
            for k in ("y_event", "delta", "aut_pos", "m_delta", "m_aut", "m_mag"):
                self.assertTrue(torch.equal(p1[k], p2[k]), (variant, k))


class TestSelection(unittest.TestCase):
    def frame(self, n=30000, seed=0):
        rng = np.random.default_rng(seed)
        ds = np.where(rng.random(n) < 0.04, "mover", "vitaldb")
        key = rng.integers(0, 500, n)
        pk = np.array([f"{d}-{k}" for d, k in zip(ds, key)])
        split = np.where(key % 10 < 7, "train", np.where(key % 10 < 8, "validation", "test"))
        return pd.DataFrame({"window_id": np.arange(n), "dataset": ds, "patient_key": pk, "split_pooled": split, "y_event": rng.random(n) < 0.03, "primary_eligible": True, "in_full": True})

    def test_composition_disjointness_and_weights(self):
        idx = self.frame()
        sel = ST.sweep_selection(idx, n_train=3000, n_val=1000, mover_share=0.10, seed=1)
        tr, va = sel[sel.split == "train"], sel[sel.split == "validation"]
        self.assertEqual((len(tr), len(va)), (3000, 1000))
        self.assertAlmostEqual(float((tr.dataset == "mover").mean()), 0.10, delta=0.002)
        self.assertFalse(set(tr.patient_key) & set(va.patient_key))
        self.assertTrue(sel.window_id.is_unique)
        for split, g in sel.groupby("split"):
            pool = idx[idx.split_pooled == split]
            for (ds, ev), gg in g.assign(ev=g.y_event).groupby(["dataset", "ev"]):
                natural = int(((pool.dataset == ds) & (pool.y_event == ev)).sum())
                self.assertAlmostEqual(float(gg.sampling_weight.sum()), natural, delta=1e-3 * natural + 1e-6)         # weights recover the natural stratum counts
        again = ST.sweep_selection(idx, n_train=3000, n_val=1000, mover_share=0.10, seed=1)
        self.assertTrue(sel.window_id.equals(again.window_id))


class TestStratifiedPositions(unittest.TestCase):
    def test_positions_have_the_requested_composition(self):
        rng = np.random.default_rng(0)
        ds = np.where(rng.random(20000) < 0.05, "mover", "vitaldb")
        y = rng.random(20000) < 0.03
        pos = D.resample_positions(ds, y, 5000, 0.15, 0.30, 7)
        self.assertEqual(len(pos), 5000)
        self.assertAlmostEqual(float((ds[pos] == "mover").mean()), 0.15, delta=0.005)
        self.assertAlmostEqual(float(y[pos][ds[pos] == "vitaldb"].mean()), 0.30, delta=0.01)
        self.assertTrue(np.array_equal(pos, D.resample_positions(ds, y, 5000, 0.15, 0.30, 7)))


if __name__ == "__main__":
    unittest.main()
