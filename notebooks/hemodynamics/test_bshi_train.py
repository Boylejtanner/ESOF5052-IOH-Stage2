import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from hemodynamics import bshi as S
from hemodynamics import bshi_data as D
from hemodynamics import bshi_train as T
from hemodynamics.bshi_model import BshiNet, ModelConfig
from hemodynamics.test_bshi_data import FAKE

CUDA = torch.cuda.is_available()


def fake_batch(n, seed):
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randn((n, len(S.CONTEXT_NAMES)), generator=g)
    y = (ctx[:, 0] + 0.3 * torch.randn(n, generator=g) > 0.7).float()
    return {"x": torch.randn((n, 7, 60, 200), generator=g), "context": ctx, "beat_mask": torch.rand((n, 3, 60), generator=g) > 0.1, "window_id": torch.arange(n) + seed * 1000,
            "y_event": y, "delta_map": torch.relu(ctx[:, 1] * 8 + 5), "aut65": torch.relu(ctx[:, 2]) * (y > 0).float() * 4, "dataset_code": torch.zeros(n, dtype=torch.int8)}


class Fixed:
    """Train loader: fixed batches, order shuffled by epoch (deterministic)."""
    def __init__(self, batches): self.batches = batches
    def __call__(self, epoch):
        order = np.random.default_rng(100 + epoch).permutation(len(self.batches))
        return [self.batches[i] for i in order]


def small_trainer(out, epochs=4, **over):
    cfg = T.TrainConfig.from_dict({"epochs": epochs, "patience": 50, "amp": "none", "num_workers": 0, "warmup_epochs": 1.0, "lr": 2e-3, "deterministic": True, "dropout": 0.1, "channels_last": False}, **over)
    pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
    tr = Fixed([fake_batch(16, s) for s in range(3)])
    val = [fake_batch(16, 50), fake_batch(16, 51)]
    return T.Trainer(cfg, ModelConfig(dropout=0.1), pre, tr, lambda: val, out, device="cpu", steps_per_epoch=3, log=lambda *a: None)


class TestSchedule(unittest.TestCase):
    def test_warmup_then_cosine(self):
        cfg = T.TrainConfig(lr=1e-3, warmup_epochs=1.0, min_lr_frac=0.01, schedule="cosine")
        spe, total = 10, 100
        lrs = [T.lr_at(cfg, s, total, spe) for s in range(total)]
        self.assertAlmostEqual(lrs[0], 1e-4)
        self.assertAlmostEqual(lrs[9], 1e-3)                                   # peak at the end of the warmup
        self.assertTrue(all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:])))
        self.assertAlmostEqual(lrs[-1], 1e-3 * 0.01, delta=2e-5)
        self.assertAlmostEqual(lrs[55], 1e-3 * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * 45 / 90))), places=9)

    def test_onecycle_peaks_at_thirty_percent_and_constant_is_flat(self):
        cfg = T.TrainConfig(lr=1e-3, schedule="onecycle")
        lrs = [T.lr_at(cfg, s, 100, 10) for s in range(100)]
        self.assertEqual(int(np.argmax(lrs)), 30)
        self.assertLess(lrs[0], lrs[30] / 10)
        self.assertLess(lrs[-1], lrs[30] / 1000)
        self.assertEqual(T.lr_at(T.TrainConfig(lr=5e-4, schedule="constant"), 7, 100, 10), 5e-4)


class TestOptimizers(unittest.TestCase):
    def test_no_weight_decay_on_biases_and_norms(self):
        m = BshiNet(ModelConfig())
        o = T.build_optimizer(m, T.TrainConfig(weight_decay=0.05), "cpu")
        decay, plain = o.param_groups
        self.assertEqual(decay["weight_decay"], 0.05)
        self.assertEqual(plain["weight_decay"], 0.0)
        self.assertTrue(all(p.ndim > 1 for p in decay["params"]) and all(p.ndim <= 1 for p in plain["params"]))
        self.assertEqual(len(decay["params"]) + len(plain["params"]), len(list(m.parameters())))

    def test_sgd_is_nesterov_and_adamw_takes_beta2(self):
        m = BshiNet(ModelConfig(variant="map_only"))
        s = T.build_optimizer(m, T.TrainConfig(optimizer="sgd", momentum=0.9), "cpu")
        self.assertTrue(s.defaults["nesterov"])
        self.assertEqual(s.defaults["momentum"], 0.9)
        a = T.build_optimizer(m, T.TrainConfig(optimizer="adamw", beta2=0.95), "cpu")
        self.assertEqual(a.defaults["betas"], (0.9, 0.95))
        with self.assertRaises(ValueError):
            T.build_optimizer(m, T.TrainConfig(optimizer="adam"), "cpu")

    def test_config_from_dict_ignores_unknown_keys(self):
        c = T.TrainConfig.from_dict({"lr": 0.1, "store_train_stride": 2, "nonsense": 1}, epochs=3)
        self.assertEqual((c.lr, c.epochs), (0.1, 3))


class TestEarlyStopping(unittest.TestCase):
    def test_scripted_sequence(self):
        s = T.EarlyStopper(patience=3, min_delta=5e-4)
        flags = [s.update(v, i) for i, v in enumerate([0.50, 0.60, 0.6004, 0.59, 0.58, 0.57])]
        self.assertEqual([f[0] for f in flags], [True, True, False, False, False, False])      # 0.6004 does not beat 0.60 by min_delta
        self.assertEqual([f[1] for f in flags], [False, False, False, False, True, True])      # three bad epochs in a row
        self.assertEqual((s.best, s.best_epoch), (0.60, 1))

    def test_nan_never_improves_and_state_round_trips(self):
        s = T.EarlyStopper(2, 0.0)
        self.assertTrue(s.update(0.5, 0)[0])
        self.assertFalse(s.update(float("nan"), 1)[0])
        t = T.EarlyStopper(2, 0.0)
        t.load(s.state())
        self.assertEqual(t.state(), s.state())

    def test_trainer_stops_and_restores_the_best_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            tr = small_trainer(tmp, epochs=8, patience=2)
            seq = iter([0.5, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.5])
            tr.validate = lambda: {"auprc": next(seq), "auroc": 0.7}
            res = tr.fit(resume=False)
            self.assertTrue(res["stopped_early"])
            self.assertEqual(res["best_epoch"], 1)
            self.assertEqual(len(res["history"]), 4)                          # epochs 0,1 then two bad epochs
            best = torch.load(Path(tmp) / "best.pt", weights_only=False)
            for k, v in best["model"].items():
                self.assertTrue(torch.equal(tr.model.state_dict()[k].cpu(), v), k)      # restored


class TestLoss(unittest.TestCase):
    def test_masks_remove_unobserved_targets_and_all_false_is_zero(self):
        pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
        n = 6
        out = {k: torch.randn(n) for k in ("event", "delta_map", "aut_logit", "aut_mag")}
        t = pre({**fake_batch(n, 1), "delta_map": torch.tensor([1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]), "aut65": torch.tensor([0, 0, 3.0, float("nan"), 2.0, 0]), "reg_mask": torch.tensor([1, 1, 1, 1, 0, 1], dtype=torch.bool)})
        total, parts = T.multitask_loss(out, t, {"event": 1.0, "delta_map": 0.3, "aut65": 0.3})
        self.assertTrue(torch.isfinite(total))
        keep = t["m_delta"]
        manual = torch.nn.functional.huber_loss(out["delta_map"][keep], t["delta"][keep], delta=1.0)
        self.assertAlmostEqual(float(parts["delta_map"]), float(manual), places=6)
        mag_keep = t["m_mag"]
        manual_mag = torch.nn.functional.huber_loss(out["aut_mag"][mag_keep], t["aut_mag"][mag_keep], delta=1.0)
        self.assertAlmostEqual(float(parts["aut_mag"]), float(manual_mag), places=6)
        t0 = {**t, "m_delta": torch.zeros(n, dtype=torch.bool), "m_aut": torch.zeros(n, dtype=torch.bool), "m_mag": torch.zeros(n, dtype=torch.bool)}
        _, p0 = T.multitask_loss(out, t0, {"event": 1.0, "delta_map": 0.3, "aut65": 0.3})
        self.assertEqual((float(p0["delta_map"]), float(p0["aut_pos"]), float(p0["aut_mag"])), (0.0, 0.0, 0.0))

    def test_pos_weight_raises_the_loss_on_positives(self):
        pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
        out = {k: torch.zeros(8) for k in ("event", "delta_map", "aut_logit", "aut_mag")}
        t = pre(fake_batch(8, 3))
        a = T.multitask_loss(out, t, {"event": 1.0, "delta_map": 0, "aut65": 0})[0]
        b = T.multitask_loss(out, t, {"event": 1.0, "delta_map": 0, "aut65": 0}, pos_weight=5.0)[0]
        self.assertGreater(float(b), float(a))


class TestMetrics(unittest.TestCase):
    def frame(self, n=400, seed=0, weights=False):
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.2).astype(np.float32)
        logit = rng.normal(size=n) + 1.5 * y
        df = pd.DataFrame({"window_id": np.arange(n), "y": y, "logit": logit, "delta_pred": rng.random(n) * 10, "delta_true": rng.random(n) * 10, "m_delta": np.ones(n, bool),
                           "aut_pred": rng.random(n), "aut_true": rng.random(n), "m_aut": np.ones(n, bool), "dataset_code": rng.integers(0, 2, n).astype(np.int8),
                           "w": (rng.random(n) * 5 + 0.5).astype(np.float32) if weights else np.ones(n, np.float32)})
        df.attrs["loss"] = 0.5
        return df

    def test_equal_to_sklearn(self):
        for weighted in (False, True):
            df = self.frame(weights=weighted)
            m = T.metrics_from_predictions(df)
            p = 1 / (1 + np.exp(-df.logit.to_numpy()))
            w = df.w.to_numpy()
            self.assertAlmostEqual(m["auroc"], roc_auc_score(df.y > 0.5, p, sample_weight=w), places=9)
            self.assertAlmostEqual(m["auprc"], average_precision_score(df.y > 0.5, p, sample_weight=w), places=9)
            self.assertAlmostEqual(m["brier"], brier_score_loss(df.y > 0.5, p, sample_weight=w), places=9)

    def test_prior_shift_changes_calibration_not_the_ranking_within_a_dataset(self):
        df = self.frame()
        a, b = T.metrics_from_predictions(df), T.metrics_from_predictions(df, {"mover": -2.4, "vitaldb": -2.4})
        self.assertAlmostEqual(a["auroc"], b["auroc"], places=9)
        self.assertNotAlmostEqual(a["brier"], b["brier"], places=3)

    def test_single_class_gives_nan_not_an_error(self):
        df = self.frame()
        df["y"] = 0.0
        m = T.metrics_from_predictions(df)
        self.assertTrue(math.isnan(m["auroc"]) and math.isnan(m["auprc"]))

    def test_mae_against_the_median_reference(self):
        df = self.frame()
        df["delta_pred"] = df.delta_true
        m = T.metrics_from_predictions(df)
        self.assertAlmostEqual(m["mae_delta_mmhg"], 0.0, places=9)
        self.assertGreater(m["mae_delta_ref"], 0.5)


class TestCheckpointResume(unittest.TestCase):
    def test_a_resumed_run_continues_exactly(self):
        with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
            a = small_trainer(ta, epochs=4)
            ra = a.fit(resume=False)["history"]
            b1 = small_trainer(tb, epochs=4)
            b1.fit(resume=False, until_epoch=2)                                   # "interrupted" after two epochs
            b2 = small_trainer(tb, epochs=4)                                      # a new process: new model, new optimizer
            rb = b2.fit(resume=True)["history"]
            self.assertEqual(len(rb), 4)
            for e in (2, 3):
                self.assertAlmostEqual(float(ra.loc[e, "total"]), float(rb.loc[e, "total"]), places=4)
                self.assertAlmostEqual(float(ra.loc[e, "lr"]), float(rb.loc[e, "lr"]), places=12)
                self.assertAlmostEqual(float(ra.loc[e, "val_auprc"]), float(rb.loc[e, "val_auprc"]), places=4)
            for k, v in a.model.state_dict().items():
                self.assertTrue(torch.allclose(v, b2.model.state_dict()[k], atol=1e-4), k)

    def test_log_and_checkpoints_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            small_trainer(tmp, epochs=2).fit(resume=False)
            self.assertTrue((Path(tmp) / "epochs.csv").exists() and (Path(tmp) / "last.pt").exists() and (Path(tmp) / "best.pt").exists())
            log = pd.read_csv(Path(tmp) / "epochs.csv")
            self.assertEqual(len(log), 2)
            for c in ("epoch", "total", "lr", "val_auprc", "val_auroc", "train_windows_per_s", "grad_norm"):
                self.assertIn(c, log.columns)


class TestEMA(unittest.TestCase):
    def test_shadow_follows_the_weights_and_swaps_back(self):
        m = BshiNet(ModelConfig(variant="map_only"))
        e = T.EMA(m, 0.5)
        before = {k: v.clone() for k, v in m.state_dict().items() if v.dtype.is_floating_point}
        with torch.no_grad():
            for p in m.parameters():
                p.add_(1.0)
        e.update(m)
        k = "h_event.bias"
        self.assertTrue(torch.allclose(e.shadow[k], before[k] + 0.5))
        e.swap_in(m)
        self.assertTrue(torch.allclose(m.state_dict()[k], before[k] + 0.5))
        e.swap_out(m)
        self.assertTrue(torch.allclose(m.state_dict()[k], before[k] + 1.0))


@unittest.skipUnless(CUDA, "needs a GPU (256 windows through the full network)")
class TestOverfit(unittest.TestCase):
    def test_a_fixed_batch_of_256_windows_is_memorized(self):
        torch.manual_seed(5052)
        pre = D.Preprocessor("eap", FAKE, device="cuda", channels_last=True)
        raw = fake_batch(256, 7)
        raw["y_event"] = (torch.rand(256) < 0.3).float()                    # random labels: only memorization can reduce the loss
        b = pre(raw)
        model = BshiNet(ModelConfig(dropout=0.0)).cuda().to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
        w = {"event": 1.0, "delta_map": 0.3, "aut65": 0.3}
        first = None
        for step in range(250):
            out = {k: v.float() for k, v in model(b["x"], b["context"], b["beat_mask"]).items()}
            loss, _ = T.multitask_loss(out, b, w)
            if first is None:
                first = float(loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.assertLess(float(loss), 0.1 * first, (first, float(loss)))


@unittest.skipUnless(CUDA, "needs a GPU")
class TestBenchmark(unittest.TestCase):
    def test_step_benchmark_reports_throughput_and_memory(self):
        r = T.benchmark_step(ModelConfig(), 64, bf16=True, channels_last=True, iters=3, warmup=2)
        self.assertFalse(r["oom"])
        self.assertGreater(r["windows_per_s"], 50)
        self.assertGreater(r["peak_vram_gb"], 0.1)


if __name__ == "__main__":
    unittest.main()
