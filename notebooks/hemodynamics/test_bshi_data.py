import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hemodynamics import bshi as S
from hemodynamics import bshi_data as D
from hemodynamics.bshi_train import aut65_prediction
from hemodynamics.test_bshi import World

N = len(S.CONTEXT_NAMES)
FAKE = {"context": {k: {"center": [0.0] * N, "scale": [2.0] * N} for k in ("eap_full", "ea_core")},
        "target": {"delta_map": {"scale": 10.0}, "aut65": {"log1p_mean_given_positive": 1.0, "log1p_std_given_positive": 0.5}}}
ROOT = Path(__file__).resolve().parents[2]


def raw_batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    aut = torch.tensor([0.0, 0.0, 3.0, 12.0, 0.0, 1.5, 0.0, 40.0])[:n]
    return {"x": torch.full((n, 7, 60, 200), 9.0), "context": (torch.arange(N).float() - 8).repeat(n, 1) * 5, "beat_mask": torch.rand((n, 3, 60), generator=g) > 0.2, "window_id": torch.arange(n),
            "y_event": (torch.arange(n) % 3 == 0).float(), "delta_map": torch.tensor([0.0, 5.0, 10.0, float("nan"), 20.0, 8.0, 0.0, 30.0])[:n], "aut65": aut,
            "dataset_code": torch.tensor([0, 1] * (n // 2), dtype=torch.int8)}


class TestPreprocessor(unittest.TestCase):
    def test_clamp_standardization_and_variant_mask(self):
        pre = D.Preprocessor("ea", FAKE, clamp=6.0, device="cpu", channels_last=False)
        b = pre(raw_batch())
        self.assertTrue((b["x"][:, 3:6] == 6.0).all() and (b["x"][:, 0] == 9.0).all())
        self.assertLessEqual(float(b["context"].abs().max()), 6.0)
        j = S.CONTEXT_NAMES.index("map_current")
        self.assertAlmostEqual(float(b["context"][0, j]), float(np.clip((j - 8) * 5 / 2.0, -6, 6)))
        for name in S.context_off(S.VARIANTS["ea"]):
            self.assertTrue((b["context"][:, S.CONTEXT_NAMES.index(name)] == 0).all(), name)    # variant-zeroed entries stay exactly 0

    def test_ppg_variants_use_the_eap_statistics_and_ppg_free_ones_the_ea_statistics(self):
        stats = {"context": {"eap_full": {"center": [1.0] * N, "scale": [1.0] * N}, "ea_core": {"center": [3.0] * N, "scale": [1.0] * N}}, "target": FAKE["target"]}
        a = D.Preprocessor("eap", stats, device="cpu")
        b = D.Preprocessor("ea", stats, device="cpu")
        self.assertEqual(float(a.center[0]), 1.0)
        self.assertEqual(float(b.center[0]), 3.0)
        self.assertEqual(float(D.Preprocessor("wave", stats, device="cpu").center[0]), 1.0)
        self.assertEqual(float(D.Preprocessor("art_only", stats, device="cpu").center[0]), 3.0)

    def test_targets_and_loss_masks(self):
        pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
        raw = raw_batch()
        raw["reg_mask"] = torch.tensor([True, True, True, True, False, True, True, True])
        b = pre(raw)
        self.assertTrue(torch.equal(b["m_delta"], torch.tensor([True, True, True, False, False, True, True, True])))     # NaN delta and the low-coverage window are masked
        self.assertTrue(torch.equal(b["m_aut"], raw["reg_mask"]))
        self.assertTrue(torch.equal(b["m_mag"], torch.tensor([False, False, True, True, False, True, False, True])))        # magnitude only where AUT65 > 0 and observed
        self.assertAlmostEqual(float(b["delta"][2]), 1.0)                         # 10 mmHg / scale 10
        self.assertTrue(torch.isfinite(b["delta"]).all() and torch.isfinite(b["aut_mag"]).all())
        self.assertEqual(b["aut_pos"].tolist(), [0, 0, 1, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(float(b["aut_mag"][2]), (np.log1p(3.0) - 1.0) / 0.5, places=5)

    def test_two_part_aut65_round_trip_and_prediction(self):
        pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
        b = pre(raw_batch())
        back = torch.expm1(b["aut_mag"] * pre.aut_std + pre.aut_mean) * b["aut_pos"]
        self.assertTrue(torch.allclose(back, raw_batch()["aut65"], atol=1e-4))
        out = {"aut_logit": torch.tensor([30.0, -30.0]), "aut_mag": (torch.log1p(torch.tensor([7.0, 7.0])) - pre.aut_mean) / pre.aut_std}
        pred = aut65_prediction(out, pre)
        self.assertAlmostEqual(float(pred[0]), 7.0, places=3)                    # certain and 7 mmHg*min
        self.assertAlmostEqual(float(pred[1]), 0.0, places=3)                    # certainly zero


def idx_frame(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ds = np.where(rng.random(n) < 0.05, "mover", "vitaldb")
    ev = rng.random(n) < np.where(ds == "mover", 0.04, 0.03)
    pk = np.array([f"{d}-{i}" for d, i in zip(ds, rng.integers(0, 300, n))])
    key = np.array([int(p.split("-")[1]) % 10 for p in pk])
    split = np.where(key < 7, "train", np.where(key < 8, "validation", "test"))
    return pd.DataFrame({"window_id": np.arange(n), "dataset": ds, "y_event": ev, "patient_key": pk, "split_pooled": split, "split_within_dataset": split, "aut65_low_coverage": rng.random(n) < 0.02})


class TestSampling(unittest.TestCase):
    def test_resample_composition(self):
        idx = idx_frame(20000)
        r = D.resample_index(idx, 4000, mover_share=0.15, event_share=0.30, seed=1)
        self.assertEqual(len(r), 4000)
        self.assertAlmostEqual(float((r.dataset == "mover").mean()), 0.15, delta=0.005)
        for ds in ("mover", "vitaldb"):
            self.assertAlmostEqual(float(r[r.dataset == ds].y_event.mean()), 0.30, delta=0.01)
        self.assertTrue(r.equals(D.resample_index(idx, 4000, 0.15, 0.30, 1)))        # deterministic

    def test_single_dataset_ignores_the_mover_share(self):
        idx = idx_frame(5000)
        r = D.resample_index(idx[idx.dataset == "vitaldb"], 1000, 0.15, 0.3, 2)
        self.assertEqual(set(r.dataset), {"vitaldb"})
        self.assertEqual(len(r), 1000)

    def test_prior_shift_recovers_natural_prevalence(self):
        idx = idx_frame(20000)
        shift = D.prior_shift_by_dataset(idx, 0.30)
        for ds, s in shift.items():
            self.assertLess(s, 0)
            p_nat = float(idx[idx.dataset == ds].y_event.mean())
            logit_sampled = np.log(0.3 / 0.7)                                         # a model that outputs the sampled prior
            self.assertAlmostEqual(1 / (1 + np.exp(-(logit_sampled + s))), p_nat, places=6)


class TestFrames(unittest.TestCase):
    def test_experiment_frames_are_patient_disjoint_and_follow_the_definitions(self):
        idx = idx_frame(8000)
        f = D.frames_for(idx, "pooled")
        pt = lambda d: set(d.patient_key)
        self.assertFalse(pt(f["train"]) & pt(f["val"]) or pt(f["train"]) & pt(f["tests"]["pooled_test"]) or pt(f["val"]) & pt(f["tests"]["pooled_test"]))
        f = D.frames_for(idx, "vitaldb")
        self.assertEqual(set(f["train"].dataset), {"vitaldb"})
        self.assertEqual(len(f["tests"]["mover_all"]), int((idx.dataset == "mover").sum()))          # E3x: ALL MOVER windows
        f = D.frames_for(idx, "mover")
        self.assertEqual(set(f["val"].dataset), {"mover"})
        self.assertEqual(set(f["tests"]["vitaldb_test"].split_within_dataset), {"test"})              # reverse: VitalDB TEST split
        with self.assertRaises(ValueError):
            D.frames_for(idx, "nope")

    def test_window_flags_attach_the_low_coverage_mask_by_window_id(self):
        idx = idx_frame(500)
        fl = D.WindowFlags(idx)
        b = fl.attach({"window_id": torch.tensor([7, 3, 499, 0])})
        want = ~idx.set_index("window_id").loc[[7, 3, 499, 0], "aut65_low_coverage"].to_numpy()
        self.assertEqual(b["reg_mask"].tolist(), want.tolist())


class TestCompactDecode(unittest.TestCase):
    def test_gpu_decode_matches_the_cpu_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(12, seed=5).reset_index(drop=True)
            b = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
        x = b["x"]
        waves = torch.from_numpy(x[:, :3].astype(np.float16))
        scal = torch.from_numpy(np.stack([x[:, 3, :, 0]] + [x[:, p, :, s[0]] for name, (p, s) in S.SCALAR_LAYERS.items() if name != "map_beat"], axis=2).astype(np.float32))
        y = D.decode_compact_gpu(waves, scal).numpy()
        self.assertTrue(np.array_equal(y[:, 3:], x[:, 3:]))                               # planes 3-6: exact broadcasts
        self.assertLess(float(np.abs(y[:, :3] - x[:, :3]).max()), 2e-2)                     # planes 0-2: float16 storage


@unittest.skipUnless((ROOT / D.DEV_DIR).exists(), "dev subset not present")
class TestDevSubset(unittest.TestCase):
    def test_batches_carry_weights_and_the_variant_zeroes_ppg(self):
        for variant in ("eap", "ea"):
            ds = D.DevSubsetBatches(ROOT, "validation", variant, batch_size=64, shuffle=False, seed=0, files_per_chunk=1)
            b = next(iter(ds))
            self.assertEqual(tuple(b["x"].shape), (64, 7, 60, 200))
            self.assertIn("sampling_weight", b)
            self.assertTrue(torch.isfinite(b["x"]).all())
            if variant == "ea":
                self.assertEqual(float(b["x"][:, 2].abs().max()), 0.0)
            else:
                self.assertGreater(float(b["x"][:, 2].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
