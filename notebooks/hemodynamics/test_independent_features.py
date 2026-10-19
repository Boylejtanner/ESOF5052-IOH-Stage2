"""Tests for independent_features (synthetic signals only, no real data)."""
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import pyarrow as pa

from hemodynamics import independent_features as F

FS = 100


def synth_art(n_beats=80, T=0.8, tau=1.5, S=120.0, N=95.0, seed=0, noise=0.05):
    """ART beats: sin^2 upstroke to S at 0.1 s, cos^2 fall to the notch N at 0.3 s, then N exp(-(t-0.3)/tau) plus a small dicrotic wave."""
    t = np.arange(int(round(T * FS))) / FS
    D = N * np.exp(-(T - 0.3) / tau)
    y = np.empty_like(t)
    a, b, c = t < 0.1, (t >= 0.1) & (t < 0.3), t >= 0.3
    y[a] = D + (S - D) * np.sin(np.pi * t[a] / 0.2) ** 2
    y[b] = N + (S - N) * np.cos(np.pi * (t[b] - 0.1) / 0.4) ** 2
    y[c] = N * np.exp(-(t[c] - 0.3) / tau) + np.where(t[c] < 0.42, 4.0 * np.sin(np.pi * (t[c] - 0.3) / 0.12), 0.0)
    x = np.tile(y, n_beats)
    return x + np.random.default_rng(seed).normal(0, noise, len(x)), D


def synth_ppg(n_beats=80, T=0.8, amp=1.0, dc=10.0):
    t = np.arange(int(round(T * FS))) / FS
    e_end = np.exp(-(T - 0.12) / 0.25)
    y = np.where(t < 0.12, np.sin(np.pi * t / 0.24) ** 2, (np.exp(-(t - 0.12) / 0.25) - e_end) / (1 - e_end))
    return dc + amp * np.tile(y, n_beats)


def synth_ecg(rr, seed=0):
    r_t = np.concatenate(([1.0], 1.0 + np.cumsum(rr)))
    n = int((r_t[-1] + 1.0) * FS)
    t = np.arange(n) / FS
    x = np.random.default_rng(seed).normal(0, 0.01, n)
    for r in r_t:
        lo, hi = int((r - 0.05) * FS), int((r + 0.05) * FS)
        x[lo:hi] += np.exp(-0.5 * ((t[lo:hi] - r) / 0.008) ** 2) - 0.2 * np.exp(-0.5 * ((t[lo:hi] - r - 0.03) / 0.01) ** 2)
    return t, x


def one_chunk(n):
    return np.array([-1.0]), np.array([n / FS + 2.0])


class Helpers(unittest.TestCase):
    def test_ppv_of_known_modulation(self):
        k = np.arange(200)
        self.assertAlmostEqual(F.span_variation(50 + 5 * np.sin(2 * np.pi * k / 5)), 0.190, delta=0.01)

    def test_sdnn_rmssd(self):
        iv = np.tile([0.80, 0.85], 50)
        t = np.cumsum(iv)
        sdnn, rmssd, pnn50, _ = F.hrv(t, iv)
        self.assertAlmostEqual(sdnn, 25.0, delta=0.01)
        self.assertAlmostEqual(rmssd, 50.0, delta=0.01)
        self.assertEqual(pnn50, 0.0)

    def test_ac1_of_ar1(self):
        rng = np.random.default_rng(5052)
        for phi in (0.3, 0.8):
            e = rng.normal(size=20000)
            x = np.empty_like(e)
            x[0] = e[0]
            for i in range(1, len(e)):
                x[i] = phi * x[i - 1] + e[i]
            self.assertAlmostEqual(F.ac1(x), phi, delta=0.05)

    def test_dfa_white_noise(self):
        self.assertAlmostEqual(F.dfa_alpha1(np.random.default_rng(1).normal(size=3000)), 0.5, delta=0.1)

    def test_kendall_and_sampen(self):
        self.assertEqual(F.kendall_trend(np.arange(10.0)), 1.0)
        self.assertEqual(F.kendall_trend(-np.arange(10.0)), -1.0)
        se_noise = F.sample_entropy(np.random.default_rng(2).normal(size=300))
        se_sine = F.sample_entropy(np.sin(np.arange(300) * 0.3))
        self.assertGreater(se_noise, 1.5)
        self.assertLess(se_sine, 0.5)

    def test_lf_hf_sign(self):
        t = np.cumsum(np.full(400, 0.8))
        self.assertGreater(F.lf_hf(t, np.sin(2 * np.pi * 0.1 * t)), 2.0)
        self.assertLess(F.lf_hf(t, np.sin(2 * np.pi * 0.25 * t)), -2.0)

    def test_chunk_plan_is_causal(self):
        s, e = F.chunk_plan([400, 430, 460, 1000])
        np.testing.assert_array_equal(e, [400, 430, 460, 1000])
        np.testing.assert_array_equal(s, [95, 400, 430, 695])
        self.assertTrue((s < e).all())


class NativeArt(unittest.TestCase):
    def test_decay_tau_and_notch(self):
        x, D = synth_art()
        s, e = one_chunk(len(x))
        df = F.native_pulses(np.arange(len(x)) / FS, x, s, e, kind="art")
        ok = df[df.ok]
        self.assertGreater(len(ok), 60)
        self.assertAlmostEqual(ok.tau.median(), 1.5, delta=0.15)
        self.assertAlmostEqual(ok.notch_rel.median(), (95.0 - D) / (120.0 - D), delta=0.05)
        self.assertAlmostEqual(ok.interval.median(), 0.8, delta=0.01)
        self.assertAlmostEqual(ok.sbp.median(), 120.0, delta=1.0)
        self.assertAlmostEqual(ok.dbp.median(), D, delta=1.0)
        self.assertGreater(ok.q.dropna().median(), 0.99)

    def test_ppg_polarity_recovered(self):
        x = synth_ppg()
        self.assertEqual(F.ppg_polarity(x), 1.0)
        self.assertEqual(F.ppg_polarity(-x), -1.0)
        s, e = one_chunk(len(x))
        ts = np.arange(len(x)) / FS
        a, b = F.native_pulses(ts, x, s, e, kind="ppg"), F.native_pulses(ts, -x, s, e, kind="ppg")
        self.assertTrue((b.pol == -1).all() and (a.pol == 1).all())
        self.assertAlmostEqual(a[a.ok].amp.median(), b[b.ok].amp.median(), delta=0.05)
        self.assertAlmostEqual(b[b.ok].interval.median(), 0.8, delta=0.01)

    def test_ecg_rr(self):
        rr = np.tile([0.80, 0.85], 60)
        t, x = synth_ecg(rr)
        s, e = one_chunk(len(x))
        df = F.native_ecg(t, x, s, e)
        v = df[df.valid]
        self.assertGreater(len(v), 100)
        self.assertAlmostEqual(v.rr.median(), 0.825, delta=0.03)
        self.assertLess(v.ectopic.mean(), 0.02)


def synth_case(seconds=900, seed=0):
    n_beats = int(seconds / 0.8)
    art, _ = synth_art(n_beats, seed=seed)
    ppg = synth_ppg(n_beats)
    rng = np.random.default_rng(seed)
    rr = 0.8 + rng.normal(0, 0.01, n_beats - 3)
    _, ecg = synth_ecg(rr, seed)
    n = min(len(art), len(ppg), len(ecg))
    return np.arange(n) / FS, ecg[:n], art[:n], ppg[:n]


def params_for(ts, ecg, art, ppg, t_end):
    s, e = F.chunk_plan(t_end)
    return F.window_params(F.native_ecg(ts, ecg, s, e), F.native_pulses(ts, art, s, e, "art"), F.native_pulses(ts, ppg, s, e, "ppg"), t_end)


class Windows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ts, cls.ecg, cls.art, cls.ppg = synth_case()
        cls.t_end = np.arange(330.0, 871.0, 30.0)
        cls.base = params_for(cls.ts, cls.ecg, cls.art, cls.ppg, cls.t_end)

    def test_columns_match_catalogue(self):
        self.assertEqual(list(self.base.columns), ["t_end_sec"] + F.column_names())
        self.assertEqual(len(F.candidate_names()), 147)
        self.assertEqual(len(self.base), len(self.t_end))

    def test_values_finite_when_covered(self):
        last = self.base.iloc[-1]
        for c in ("a_sbp_l60", "a_tau_l60", "a_ppv_l300", "a_map_ac1_l300", "a_map_dfa_l300", "e_hr_l60", "e_sdnn_l300", "e_rr_sampen_l300",
                  "p_hr_l60", "p_pvi_l60", "a_sbp_l60_sn", "x_hr_map_corr_l60"):
            self.assertTrue(np.isfinite(last[c]), c)
        self.assertAlmostEqual(last.a_hr_l60, 75.0, delta=1.0)
        self.assertAlmostEqual(last.e_hr_l60, 75.0, delta=2.0)
        self.assertAlmostEqual(last.a_sbp_l60_sn, 1.0, delta=0.02)

    def test_causality(self):
        cut = 600.0
        k = int(cut * FS) + 1
        ecg, art, ppg = self.ecg.copy(), self.art.copy(), self.ppg.copy()
        rng = np.random.default_rng(9)
        art[k:] = art[k:] * 0.6 + rng.normal(0, 3, len(art) - k)
        ecg[k:] = -ecg[k:] * 3 + rng.normal(0, 0.2, len(ecg) - k)
        ppg[k:] = -ppg[k:] * 2
        pert = params_for(self.ts, ecg, art, ppg, self.t_end)
        early = self.base.t_end_sec <= cut
        pd.testing.assert_frame_equal(self.base[early], pert[early])
        self.assertFalse(self.base[~early].equals(pert[~early]))

    def test_self_normalization_is_causal(self):
        df = self.base.copy()
        df.loc[df.t_end_sec > 700, "a_sbp_l60"] = 500.0
        a, b = F.self_normalize(self.base.copy()), F.self_normalize(df)
        early = a.t_end_sec <= 700
        np.testing.assert_array_equal(a.loc[early, "a_sbp_l60_sn"].to_numpy(), b.loc[early, "a_sbp_l60_sn"].to_numpy())
        self.assertTrue(a.a_sbp_l60_sn.iloc[:F.SN_MIN_HISTORY - 1].isna().all())

    def test_insufficient_coverage_is_nan(self):
        ts, ecg, art, ppg = self.ts, self.ecg.copy(), self.art.copy(), np.full(len(self.ppg), np.nan)
        art[int(500 * FS):int(560 * FS)] = np.nan                       # window ending at 570: only ~9 s of ART in the last 60 s
        p = params_for(ts, ecg, art, ppg, self.t_end)
        row = p.loc[p.t_end_sec == 570].iloc[0]
        self.assertTrue(np.isnan(row.a_sbp_l60) and np.isnan(row.a_ppv_l60))
        p_cols = [c for c in F.column_names() if c.startswith("p_") and c != "p_cover_l60"]
        self.assertTrue(p[p_cols].isna().all().all())
        self.assertFalse((p[p_cols] == 0).any().any())


class ColumnIsolation(unittest.TestCase):
    def test_read_signal_requests_one_column(self):
        seen = []

        def reader(path, columns=None, filters=None):
            seen.append(list(columns))
            return pa.table({c: np.zeros(3) for c in columns})

        for sig, col in (("ecg", "ECG"), ("art", "ART"), ("ppg", "PPG")):
            F.read_signal("x", sig, 0, 10, reader=reader)
            self.assertEqual(seen[-1], ["time_sec", col])

    def test_case_job_reads_each_signal_alone(self):
        ts, ecg, art, ppg = synth_case(seconds=420)
        full = {"time_sec": ts, "ECG": ecg, "ART": art, "PPG": ppg}
        seen = []

        def fake(path, columns=None, filters=None):
            seen.append(tuple(columns))
            return pa.table({c: full[c] for c in columns})

        with mock.patch.object(F.pq, "read_table", fake):
            params, info = F.case_job((".", "vitaldb", 1, np.array([360.0, 390.0]), False))
        self.assertEqual(info["error"], "")
        self.assertEqual(seen, [("time_sec", "ECG"), ("time_sec", "ART"), ("time_sec", "PPG")])
        self.assertEqual(len(params), 2)


if __name__ == "__main__":
    unittest.main()
