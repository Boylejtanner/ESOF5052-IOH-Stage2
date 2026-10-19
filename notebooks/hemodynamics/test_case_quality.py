import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import case_quality as C
from hemodynamics.test_beats import synth

ROOT = Path(__file__).resolve().parents[2]


def row(**over):
    base = {"dataset": "vitaldb", "case_id": "1", "n_beats": 5000, "assoc_failed_share": 0.05, "art_ok_frac": 0.96, "ppg_ok_frac": 0.9, "sqi_ecg_median": 0.93,
            "sqi_ppg_median": 0.98, "hr_median": 72.0, "n_m_ecg": 4800, "n_m_art": 4700, "n_m_ppg": 4400, "ppg_late_share": 0.0, "ecg_amp": 0.9, "art_rate_bpm": 71.0,
            "art_pp_mmhg": 50.0, "ecg_amp_ratio": 1.0}
    base.update(over)
    return base


class TestFlags(unittest.TestCase):
    def flags(self, **cases):
        df = pd.DataFrame([row(case_id=name, **kw) for name, kw in cases.items()])
        return C.flag_cases(df).set_index("case_id")

    def test_healthy_case_is_clean_and_usable(self):
        f = self.flags(ok={})
        self.assertFalse(f.loc["ok", ["ecg_dead", "art_dead", "ppg_dead", "ppg_suspect_case"]].any())
        self.assertTrue(f.loc["ok", "bshi_core_possible"] and f.loc["ok", "bshi_full_possible"])

    def test_dead_ecg_variants_are_flagged_with_their_kind(self):
        f = self.flags(noise=dict(sqi_ecg_median=0.13, art_ok_frac=0.0, hr_median=140.0, ecg_amp_ratio=0.004),
                       contradiction=dict(sqi_ecg_median=0.19, art_ok_frac=0.0, hr_median=113.0, art_rate_bpm=65.0, ecg_amp_ratio=5.0),
                       weak=dict(sqi_ecg_median=0.3, art_ok_frac=0.0, hr_median=65.0, art_rate_bpm=67.0, ecg_amp_ratio=0.19))
        self.assertTrue(f.ecg_dead.all())
        self.assertEqual(f.ecg_dead_kind.to_dict(), {"noise": "noise_floor", "contradiction": "rate_contradiction", "weak": "weak_ecg"})
        self.assertFalse(f.bshi_core_possible.any())

    def test_ppg_association_failure_alone_does_not_make_the_ecg_dead(self):
        f = self.flags(case=dict(assoc_failed_share=1.0, art_ok_frac=0.3, ppg_ok_frac=0.0, sqi_ecg_median=0.28, ecg_amp_ratio=0.2, sqi_ppg_median=0.0, n_m_ppg=0))
        self.assertFalse(f.loc["case", "ecg_dead"])
        self.assertTrue(f.loc["case", "ppg_dead"])
        self.assertFalse(f.loc["case", "bshi_full_possible"])

    def test_flat_arterial_line_is_art_dead_not_ecg_dead(self):
        f = self.flags(case=dict(art_ok_frac=0.0, art_rate_bpm=np.nan, art_pp_mmhg=np.nan, sqi_ecg_median=0.9, ecg_amp_ratio=1.0, n_m_art=0, n_m_ppg=0))
        self.assertTrue(f.loc["case", "art_dead"])
        self.assertFalse(f.loc["case", "ecg_dead"])
        self.assertFalse(f.loc["case", "bshi_core_possible"])

    def test_ecg_amplitude_alone_cannot_flag_a_case_with_a_good_arterial_link(self):
        f = self.flags(case=dict(ecg_amp_ratio=0.1))
        self.assertFalse(f.loc["case", "ecg_dead"])

    def test_late_ppg_marks_the_case_suspect_and_limits_full_set(self):
        f = self.flags(case=dict(ppg_late_share=0.7, n_m_ppg=10))
        self.assertTrue(f.loc["case", "ppg_suspect_case"])
        self.assertTrue(f.loc["case", "bshi_core_possible"])
        self.assertFalse(f.loc["case", "bshi_full_possible"])

    def test_every_documented_flag_column_exists(self):
        f = C.flag_cases(pd.DataFrame([row()]))
        for c in ("ecg_dead", "art_dead", "ppg_dead", "ppg_late_share", "ppg_suspect_case", "bshi_core_possible", "bshi_full_possible"):
            self.assertIn(c, f.columns)


class TestBeatAndRawFeatures(unittest.TestCase):
    CFG = {"sqi_ecg_min": 0.5, "ppg_delay_max_sec": 0.9, "min_usable_beats": 54}

    def test_beat_features_apply_the_bshi_masks(self):
        n = 100
        b = pd.DataFrame({"t_r": np.arange(n, dtype=float), "rr_sec": np.ones(n), "hr_bpm": 60.0, "sqi_ecg": 0.9, "sqi_ppg": 0.9, "flags": 0, "valid_beat": True,
                          "art_ok": True, "ppg_ok": True, "ptt_ecg_ppg": 0.5})
        b.loc[:9, "valid_beat"] = False
        b.loc[10:19, "art_ok"] = False
        b.loc[20:29, "ptt_ecg_ppg"] = 1.1
        b.loc[30:39, "sqi_ecg"] = 0.2
        f = C.beat_features(b, self.CFG)
        self.assertEqual((f["n_m_ecg"], f["n_m_art"], f["n_m_ppg"]), (80, 70, 60))
        self.assertAlmostEqual(f["ppg_late_share"], 10 / 90)

    def test_chunk_starts_fall_inside_ready_spans(self):
        starts = C._chunk_starts(np.array([0.0, 1000.0]), np.array([300.0, 1400.0]), 6, 60.0)
        self.assertEqual(len(starts), 6)
        for s in starts:
            self.assertTrue((0 <= s <= 240) or (1000 <= s <= 1340))

    def test_raw_features_measure_amplitude_and_independent_art_rate(self):
        t, r, ecg, art, ppg = synth(duration=400.0, hr=72.0, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "waveform.parquet"
            pd.DataFrame({"time_sec": t, "ART": art.astype("float32"), "ECG": ecg.astype("float32"), "PPG": ppg.astype("float32")}).to_parquet(path, index=False)
            normal = C.raw_features(path, np.array([0.0]), np.array([400.0]))
            pd.DataFrame({"time_sec": t, "ART": art.astype("float32"), "ECG": (0.01 * ecg).astype("float32"), "PPG": ppg.astype("float32")}).to_parquet(path, index=False)
            weak = C.raw_features(path, np.array([0.0]), np.array([400.0]))
        self.assertAlmostEqual(normal["art_rate_bpm"], 72.0, delta=4.0)
        self.assertGreater(normal["art_pp_mmhg"], 30.0)
        self.assertAlmostEqual(weak["ecg_amp"] / normal["ecg_amp"], 0.01, delta=0.002)


class TestSavedDecision(unittest.TestCase):
    def test_decision_json_matches_the_case_quality_table(self):
        path, parquet = ROOT / "results" / "dead_ecg_decision.json", C.quality_path(ROOT)
        if not (path.exists() and parquet.exists()):
            self.skipTest("run the dead-ECG decision first")
        out = json.loads(path.read_text(encoding="utf-8"))
        q = pd.read_parquet(parquet)
        self.assertEqual(out["evidence"]["flagged_ecg_dead_cases"], int(q.ecg_dead.sum()))
        self.assertEqual(sum(out["evidence"]["split_distribution"].values()), int(q.ecg_dead.sum()))
        self.assertTrue(out["decision"]["default_kept"])
        self.assertEqual(out["evidence"]["recovered"], out["evidence"]["reference_set_art_ok_frac_below_0.1_vitaldb"])
        self.assertTrue(q.groupby(["dataset", "case_id"]).size().eq(1).all())


if __name__ == "__main__":
    unittest.main()
