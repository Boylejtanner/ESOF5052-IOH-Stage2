import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import bshi as S
from hemodynamics import eda_metrics as E
from hemodynamics import eda_models as M
from hemodynamics.test_bshi import CFG, make_case, window_rows

ROOT = Path(__file__).resolve().parents[2]


def loop_stats(values, valid):
    vals = [v for v, ok in zip(values, valid) if ok and np.isfinite(v)]
    if not vals:
        return np.nan, np.nan, np.nan
    first = [v for v, ok in zip(values[:10], valid[:10]) if ok and np.isfinite(v)]
    last = [v for v, ok in zip(values[-10:], valid[-10:]) if ok and np.isfinite(v)]
    trend = (np.mean(last) - np.mean(first)) if first and last else np.nan
    return float(np.mean(vals)), float(np.std(vals)), float(trend)


class TestWindowFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.df, _ = make_case(Path(cls.tmp.name), "vitaldb", "x", n=900, seed=4)
        cls.cols = {c: cls.df[c].to_numpy() for c in cls.df.columns}
        cls.masks = S.beat_masks(cls.df.valid_beat.to_numpy(), cls.df.sqi_ecg.to_numpy(), cls.df.art_ok.to_numpy(), cls.df.ppg_ok.to_numpy(), cls.df.ptt_ecg_ppg.to_numpy(), CFG)
        cls.lo = np.array([0, 17, 120, 333, 780])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_masked_stats_match_a_loop_reference(self):
        rng = np.random.default_rng(0)
        v, valid = rng.normal(5, 2, (30, 60)), rng.random((30, 60)) > 0.2
        valid[3] = False
        valid[4, :10] = False
        mean, std, trend = M.masked_stats(v, valid)
        for i in range(30):
            ref = loop_stats(v[i], valid[i])
            for got, want in zip((mean[i], std[i], trend[i]), ref):
                self.assertTrue((np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-9)

    def test_window_features_match_a_loop_reference(self):
        f = M.window_features(self.cols, self.masks, self.lo)
        mask_of = {"ecg": self.masks[0], "art": self.masks[1], "ppg": self.masks[2]}
        for i, lo in enumerate(self.lo):
            sl = slice(lo, lo + 60)
            for name in M.ART_MORPH + ["ptt_ecg_art"] + M.PPG_FEATS:
                valid = mask_of[M.MASK_OF[name]][sl]
                mean, std, trend = loop_stats(self.cols[name][sl].astype(float), valid)
                if name in M.PPG_DEVICE_UNIT:
                    self.assertAlmostEqual(f.loc[i, f"{name}_cv"], std / abs(mean), places=4)
                    self.assertAlmostEqual(f.loc[i, f"{name}_rtrend"], trend / abs(mean), places=4)
                    self.assertAlmostEqual(f.loc[i, f"{name}_mean"], mean, places=3)
                else:
                    for suffix, want in (("mean", mean), ("std", std), ("trend", trend)):
                        got = f.loc[i, f"{name}_{suffix}"]
                        self.assertTrue((np.isnan(got) and np.isnan(want)) or abs(got - want) < 1e-3 * max(1.0, abs(want)), f"{name}_{suffix}")
            valid = mask_of["ecg"][sl]
            rr = self.cols["rr_sec"][sl].astype(float)
            pair = valid[1:] & valid[:-1]
            self.assertAlmostEqual(f.loc[i, "rmssd"], float(np.sqrt(np.mean(np.diff(rr)[pair] ** 2))), places=4)
            self.assertAlmostEqual(f.loc[i, "sdnn"], float(np.std(rr[valid])), places=4)
            self.assertEqual(f.loc[i, "n_art_ok"], float(mask_of["art"][sl].sum()))

    def test_ppg_unit_scaling_does_not_change_any_ladder_feature(self):
        scaled = dict(self.cols)
        for c in ("ppg_amp", "ppg_ac_dc"):
            scaled[c] = self.cols[c] * 20.0
        a, b = M.window_features(self.cols, self.masks, self.lo), M.window_features(scaled, self.masks, self.lo)
        for c in [c for c in M.RUNGS["G4"] if c not in M.L.FEATURES]:
            np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=1e-4, atol=1e-6, err_msg=c)
        np.testing.assert_allclose(b["ppg_amp_mean"].to_numpy(), 20.0 * a["ppg_amp_mean"].to_numpy(), rtol=1e-4)           # the raw diagnostic columns do scale


class TestLadderDefinition(unittest.TestCase):
    def test_rungs_are_nested_unique_and_free_of_raw_device_units(self):
        names = list(M.RUNGS)
        for small, big in zip(names[:-1], names[1:]):
            self.assertTrue(set(M.RUNGS[small]) < set(M.RUNGS[big]), (small, big))
        for r, cols in M.RUNGS.items():
            self.assertEqual(len(cols), len(set(cols)), r)
            self.assertFalse(set(cols) & set(M.RAW_PPG_UNITS), r)
        self.assertEqual([len(M.RUNGS[r]) for r in names], [5, 29, 35, 51, 58])

    def test_fit_predict_is_deterministic_and_learns(self):
        rng = np.random.default_rng(1)
        n = 4000
        x = rng.normal(size=(n, 3))
        y = (x[:, 0] + 0.5 * rng.normal(size=n) > 1.2).astype(int)
        df = pd.DataFrame(x, columns=["a", "b", "c"]).assign(y=y, patient_key=rng.integers(0, 200, n).astype(str))
        tr, te = df.iloc[:3000], df.iloc[3000:]
        for learner in ("lr", "hgb"):
            (p1,), _ = M.fit_predict(tr, [te], ["a", "b", "c"], learner, seed=3, subsample=1)
            (p2,), _ = M.fit_predict(tr, [te], ["a", "b", "c"], learner, seed=3, subsample=1)
            self.assertTrue(np.array_equal(p1, p2))
            res = M.evaluate(te, p1, reps=20)
            self.assertGreater(res["auroc"], 0.85)
            self.assertLessEqual(res["auroc_lo"], res["auroc"] + 0.05)
            self.assertEqual(res["windows"], len(te))


class TestAnalysisTables(unittest.TestCase):
    def preds(self):
        rng = np.random.default_rng(2)
        n = 6000
        tte = np.where(rng.random(n) < 0.1, rng.uniform(0, 240, n), np.nan)
        y = (~np.isnan(tte)).astype(int)
        p = np.clip(0.05 + 0.5 * y + rng.normal(0, 0.15, n), 0, 1)
        return {("pooled_test", "G0", "lr"): pd.DataFrame({"patient_key": rng.integers(0, 300, n).astype(str), "y": y, "gray_zone": rng.random(n) < 0.3, "time_to_event_sec": tte,
                                                           "dataset": "vitaldb", "p": p})}

    def test_lead_time_bins_partition_the_event_windows(self):
        preds = self.preds()
        t = M.auroc_by_lead_time(preds, reps=5)
        d = preds[("pooled_test", "G0", "lr")]
        self.assertEqual(int(t.event_windows.sum()), int(((d.y == 1) & (d.time_to_event_sec > 0) & (d.time_to_event_sec <= 240)).sum()))
        self.assertEqual(list(t.lead_bin_sec), ["0-60", "60-120", "120-180", "180-240"])
        self.assertTrue((t.auroc > 0.8).all())

    def test_roc_curve_is_monotone_and_calibration_has_ten_bins(self):
        roc, cal = M.curve_tables(self.preds())
        self.assertTrue((np.diff(roc.fpr) >= 0).all() and (np.diff(roc.tpr) >= 0).all())
        self.assertEqual(len(cal), 10)

    def test_event_power_rule(self):
        self.assertTrue(E.score_event_power(300, 50, 300, 50)["pass"])
        self.assertFalse(E.score_event_power(299, 50, 300, 50)["pass"])
        self.assertFalse(E.score_event_power(500, 49, 300, 50)["pass"])


class TestSavedOutputs(unittest.TestCase):
    def test_summary_document_is_short_and_scorecard_is_consistent(self):
        doc = ROOT / "docs" / "eda_summary.md"
        score = ROOT / "results" / "dataset_readiness.json"
        if not (doc.exists() and score.exists()):
            self.skipTest("run eda_metrics.build_all first")
        self.assertLessEqual(len(doc.read_text(encoding="utf-8").splitlines()), 60)
        import json
        s = json.loads(score.read_text(encoding="utf-8"))
        self.assertEqual(s["n_pass"] + s["n_fail"], len(s["checks"]))
        self.assertEqual(s["n_fail"], sum(c["status"] == "FAIL" for c in s["checks"]))


if __name__ == "__main__":
    unittest.main()
