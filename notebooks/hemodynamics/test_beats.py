import unittest

import numpy as np

from hemodynamics import beats as B

FS = 100.0
ART_ONSET, ART_TP = 0.15, 0.12          # intersecting-tangent foot of a sin^2 upstroke sits at onset + 0.1817 * tau_p
PPG_ONSET, PPG_TP = 0.20, 0.18
TRUE_PTT_ART = ART_ONSET + 0.18169 * ART_TP
TRUE_PTT_PPG = PPG_ONSET + 0.18169 * PPG_TP


def pulse(t, onset, tau_p, tau_d=0.25):
    tau = t - onset
    out = np.zeros_like(t)
    rise = (tau >= 0) & (tau < tau_p)
    out[rise] = np.sin(np.pi * tau[rise] / (2 * tau_p)) ** 2
    dec = tau >= tau_p
    out[dec] = np.exp(-(tau[dec] - tau_p) / tau_d)
    return out


def synth(duration=120.0, hr=72.0, seed=0, noise=0.015, art_pp=45.0, art_dbp=70.0, ppg_onset=PPG_ONSET, qrs_sigma=0.012, s_amp=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * FS)) / FS
    r, now, k = [], 1.0, 0
    while now < duration - 1.5:
        r.append(now)
        now += 60.0 / hr + 0.03 * np.sin(2 * np.pi * k / 17) + rng.normal(0, 0.01)
        k += 1
    r = np.array(r)
    ecg = 0.1 * np.sin(2 * np.pi * 0.25 * t) + rng.normal(0, noise, len(t))
    art = np.full(len(t), art_dbp)
    ppg = np.full(len(t), 5.0)
    for rk in r:
        ecg += (np.exp(-0.5 * ((t - rk) / qrs_sigma) ** 2) - s_amp * np.exp(-0.5 * ((t - rk - 0.06) / 0.045) ** 2)
                + 0.25 * np.exp(-0.5 * ((t - rk - 0.26) / 0.05) ** 2) + 0.1 * np.exp(-0.5 * ((t - rk + 0.16) / 0.02) ** 2))
        art += art_pp * pulse(t, rk + ART_ONSET, ART_TP)
        ppg += pulse(t, rk + ppg_onset, PPG_TP)
    art += rng.normal(0, 0.3, len(t))
    ppg += rng.normal(0, 0.01, len(t))
    return t, r, ecg, art, ppg


def gate(x, t, on=8.0, off=2.0):
    y = x.copy()
    y[(t % (on + off)) >= on] = np.nan
    return y


class DetectionTests(unittest.TestCase):
    def test_clean_signal_recall_precision_delays_and_quality(self):
        t, r, ecg, art, ppg = synth()
        df, waves, meta = B.analyze(ecg, art, ppg, 0.0)
        got = df.t_r.to_numpy()
        inner = r[(r > 3) & (r < t[-1] - 3)]
        hit = np.array([np.abs(got - x).min() < 0.02 for x in inner])
        spurious = np.array([np.abs(inner - x).min() > 0.05 for x in got if 3 < x < t[-1] - 3])
        self.assertGreaterEqual(hit.mean(), 0.98)
        self.assertGreaterEqual(1 - spurious.mean(), 0.98)
        ok = df.loc[df.art_ok & df.ppg_ok]
        self.assertGreater(len(ok), 0.9 * len(df))
        self.assertAlmostEqual(float(ok.ptt_ecg_art.median()), TRUE_PTT_ART, delta=0.015)
        self.assertAlmostEqual(float(ok.ptt_ecg_ppg.median()), TRUE_PTT_PPG, delta=0.015)
        self.assertAlmostEqual(float(ok.ptt_art_ppg.median()), TRUE_PTT_PPG - TRUE_PTT_ART, delta=0.015)
        self.assertAlmostEqual(float(df.hr_bpm.median()), 72.0, delta=2.0)
        self.assertAlmostEqual(float(df.hr_art_bpm.median()), 72.0, delta=2.0)
        self.assertAlmostEqual(float(ok.pp.median()), 45.0, delta=3.0)
        for col, floor in (("sqi_ecg", 0.7), ("sqi_art", 0.7), ("sqi_ppg", 0.5), ("sqi_cross", 0.9)):
            self.assertGreater(float(df[col].median()), floor, col)
        self.assertEqual(waves["ecg"].shape, (len(df), 200))
        self.assertTrue(((df[["sqi_ecg", "sqi_art", "sqi_ppg", "sqi_cross"]] >= 0) & (df[["sqi_ecg", "sqi_art", "sqi_ppg", "sqi_cross"]] <= 1)).all().all())

    def test_polarity_is_handled(self):
        t, r, ecg, art, ppg = synth(seed=3)
        df, waves, meta = B.analyze(-ecg, art, -ppg, 0.0)
        self.assertEqual(meta["ecg_polarity"], -1.0)
        self.assertEqual(meta["ppg_polarity"], -1.0)
        ok = df.loc[df.art_ok & df.ppg_ok]
        self.assertAlmostEqual(float(ok.ptt_ecg_ppg.median()), TRUE_PTT_PPG, delta=0.015)
        self.assertGreater(float(np.nanmedian(waves["ecg"][:, 0])), 0.5)        # stored R-positive: z-scored R sits at phase 0

    def test_high_heart_rate_association(self):
        t, r, ecg, art, ppg = synth(hr=130.0, seed=5)
        df, _, _ = B.analyze(ecg, art, ppg, 0.0)
        ok = df.loc[df.art_ok & df.ppg_ok]
        self.assertGreater(len(ok), 0.85 * len(df))
        self.assertAlmostEqual(float(ok.ptt_ecg_art.median()), TRUE_PTT_ART, delta=0.015)
        self.assertAlmostEqual(float(df.hr_bpm.median()), 130.0, delta=3.0)

    def test_oximeter_latency_longer_than_the_rr_interval(self):
        t, r, ecg, art, ppg = synth(hr=120.0, seed=31, ppg_onset=0.55)         # PPG foot ~0.58 s after R, RR = 0.5 s
        df, _, _ = B.analyze(ecg, art, ppg, 0.0)
        ok = df.loc[df.art_ok & df.ppg_ok]
        self.assertGreater(len(ok), 0.8 * len(df))
        self.assertAlmostEqual(float(ok.ptt_ecg_ppg.median()), 0.55 + 0.18169 * PPG_TP, delta=0.02)
        self.assertAlmostEqual(float(ok.ptt_ecg_art.median()), TRUE_PTT_ART, delta=0.015)

    def test_mover_style_gating_never_fabricates_beats_across_a_gap(self):
        t, r, ecg, art, ppg = synth(duration=200.0, seed=7)
        df, waves, _ = B.analyze(gate(ecg, t), gate(art, t), gate(ppg, t), 0.0)
        valid = df.loc[df.valid_beat]
        phase = valid.t_r % 10.0
        self.assertTrue(((phase + valid.rr_sec) <= 8.0 + 0.02).all(), "a valid cycle must lie entirely inside an 8 s block")
        crossing = df.loc[(df.t_r % 10.0 + df.rr_sec) > 8.0 + 0.02]
        self.assertTrue(((crossing["flags"] & B.FLAGS["gap_overlap"]) > 0).all())
        self.assertFalse(crossing.valid_beat.any())
        share = len(valid) / len(df)
        self.assertTrue(0.5 < share < 0.9, share)
        self.assertGreater((valid.art_ok & valid.ppg_ok).mean(), 0.6)
        self.assertTrue(np.isnan(waves["art"][~np.isfinite(waves["art"]).all(axis=1)]).any(axis=1).all())

    def test_flatline_art_is_flagged_and_timing_masked(self):
        t, r, ecg, art, ppg = synth(seed=11)
        df, _, _ = B.analyze(ecg, np.full(len(t), 3.0) + np.random.default_rng(0).normal(0, 0.05, len(t)), ppg, 0.0)
        self.assertFalse(df.art_ok.any())
        self.assertGreater(((df["flags"] & B.FLAGS["art_flatline"]) > 0).mean(), 0.9)
        self.assertTrue(df.ptt_ecg_art.isna().all() and df.sqi_art.eq(0).all())
        self.assertGreater(df.valid_beat.mean(), 0.9)                          # the ECG-based beat is still valid

    def test_cvp_like_low_pulse_pressure_is_implausible(self):
        t, r, ecg, art, ppg = synth(seed=13, art_pp=6.0, art_dbp=4.0)
        df, _, _ = B.analyze(ecg, art, ppg, 0.0)
        self.assertFalse(df.art_ok.any())
        self.assertTrue(((df["flags"] & (B.FLAGS["art_implausible"] | B.FLAGS["art_flatline"])) > 0).mean() > 0.8)

    def test_missing_ppg_keeps_art_and_ecg_beats(self):
        t, r, ecg, art, ppg = synth(seed=17)
        df, _, _ = B.analyze(ecg, art, np.full(len(t), np.nan), 0.0)
        self.assertFalse(df.ppg_ok.any())
        self.assertTrue(((df["flags"] & B.FLAGS["ppg_missing"]) > 0).all())
        self.assertGreater(df.art_ok.mean(), 0.9)
        self.assertTrue(df.ptt_ecg_ppg.isna().all())

    def test_ecg_noise_burst_lowers_confidence_not_crashes(self):
        t, r, ecg, art, ppg = synth(seed=19)
        noisy = ecg.copy()
        noisy[(t > 40) & (t < 50)] += np.random.default_rng(1).normal(0, 0.8, ((t > 40) & (t < 50)).sum())
        df, _, _ = B.analyze(noisy, art, ppg, 0.0)
        quiet = df.loc[(df.t_r < 35) | (df.t_r > 55)]
        burst = df.loc[(df.t_r > 41) & (df.t_r < 49)]
        self.assertGreater(float(quiet.sqi_ecg.median()), float(burst.sqi_ecg.median()) if len(burst) else 0.0)

    def test_one_corrupted_segment_cannot_flip_the_polarity(self):
        # Regression (real MOVER): the max of per-segment extremes let a lead-off/artefact segment outvote the clean recording.
        t, r, ecg, art, ppg = synth(duration=120.0, seed=41, qrs_sigma=0.007, s_amp=0.7)
        rng = np.random.default_rng(3)
        bad = np.concatenate([np.full(300, np.nan), -40.0 + rng.normal(0, 25.0, 700) * (rng.random(700) < 0.3)])
        ecg_bad = np.concatenate([ecg, bad])
        self.assertEqual(B.ecg_orientation(ecg_bad), 1.0)
        # what the old "max over segments" rule would have said:
        high = B._sos("high", FS, 0.5, 0.0)
        segs = B._sample_segments(ecg_bad, FS)
        pos = max(np.percentile(B.sosfiltfilt(high, x - np.median(x)), 99.5) for x in segs)
        neg = max(-np.percentile(B.sosfiltfilt(high, x - np.median(x)), 0.5) for x in segs)
        self.assertGreater(neg, pos, "the corrupted segment must dominate the old rule, otherwise this test proves nothing")

    def test_narrow_tall_r_with_a_broad_negative_lobe_keeps_positive_polarity(self):
        # Regression: markers must sit on the tall R, not on the opposite lobe (polarity is decided on the unfiltered signal).
        t, r, ecg, art, ppg = synth(seed=37, qrs_sigma=0.007, s_amp=0.7)
        self.assertEqual(B.ecg_orientation(ecg), 1.0)
        df, _, meta = B.analyze(ecg, art, ppg, 0.0)
        self.assertEqual(meta["ecg_polarity"], 1.0)
        got = df.t_r.to_numpy()
        inner = r[(r > 3) & (r < t[-1] - 3)]
        offsets = np.array([got[np.abs(got - x).argmin()] - x for x in inner])
        self.assertLess(float(np.abs(np.median(offsets))), 0.015, "R markers must sit on the tall R, not on the opposite lobe")
        self.assertGreater(float((np.abs(offsets) < 0.03).mean()), 0.97)
        self.assertEqual(B.ecg_orientation(-ecg), -1.0)

    def test_too_little_signal_returns_empty(self):
        df, waves, _ = B.analyze(np.zeros(500), np.zeros(500), np.zeros(500), 0.0)
        self.assertEqual(len(df), 0)
        self.assertEqual(list(df.columns), B.BEAT_COLUMNS)
        self.assertEqual(waves["art"].shape, (0, 200))


class HelperTests(unittest.TestCase):
    def test_associate_estimates_the_delay_and_masks_missing_pulses(self):
        r = 1.0 + 0.8 * np.arange(40)
        feet = r + 0.45
        feet = np.delete(feet, 10)                              # beat 10 has no pulse
        idx, delay = B.associate(r, feet)
        self.assertEqual(int((idx < 0).sum()), 1)
        self.assertEqual(int(idx[10]), -1)
        self.assertTrue(np.allclose(feet[idx[idx >= 0]], r[idx >= 0] + 0.45))
        self.assertAlmostEqual(float(np.nanmedian(delay)), 0.45, places=2)
        self.assertEqual(B.associate(r, np.array([]))[0].tolist(), [-1] * 40)

    def test_associate_handles_a_delay_longer_than_the_rr_interval(self):
        rng = np.random.default_rng(2)
        r = np.cumsum(0.45 + rng.normal(0, 0.02, 60)) + 1.0     # 133 bpm
        feet = r + 0.62 + rng.normal(0, 0.005, 60)              # each pulse arrives after the NEXT R wave
        idx, delay = B.associate(r, feet, min_delay=0.30)       # without a minimum delay the one-beat-shifted fit (0.17 s) is equally good
        self.assertGreater(int((idx >= 0).sum()), 55)
        self.assertEqual(idx[idx >= 0].tolist(), np.flatnonzero(idx >= 0).tolist(), "pulse k must belong to beat k, not k+1")
        self.assertAlmostEqual(float(np.nanmedian(delay)), 0.62, delta=0.01)

    def test_phase_matrix_maps_cycles_monotonically(self):
        x = np.arange(1000, dtype=float)
        m = B.phase_matrix(x, np.array([10.0, 100.0]), np.array([60.0, 200.0]), 200)
        self.assertAlmostEqual(float(m[0, 0]), 10.0, places=4)
        self.assertAlmostEqual(float(m[0, 100]), 35.0, places=4)
        self.assertAlmostEqual(float(m[0, -1]), 59.75, places=3)
        self.assertTrue((np.diff(m, axis=1) > 0).all())
        g = x.copy()
        g[40:45] = np.nan
        self.assertTrue(np.isnan(B.phase_matrix(g, np.array([10.0]), np.array([60.0]), 50)).any())

    def test_merge_spans(self):
        s, e = B.merge_spans([0, 50, 200], [100, 120, 300])
        self.assertEqual((s.tolist(), e.tolist()), ([0.0, 200.0], [120.0, 300.0]))

    def test_finite_segments_min_length(self):
        x = np.ones(100)
        x[30:40] = np.nan
        x[90:92] = np.nan
        s, e = B.finite_segments(x, 20)
        self.assertEqual((s.tolist(), e.tolist()), ([0, 40], [30, 90]))


class CaseTests(unittest.TestCase):
    def test_process_case_keeps_only_span_beats_and_float16_waveforms(self):
        t, r, ecg, art, ppg = synth(duration=300.0, seed=23)
        df, waves, meta = B.process_case(t + 1000.0, art, ecg, ppg, [1060.0, 1140.0], [1150.0, 1250.0], pad_sec=10.0)
        self.assertEqual(meta["regions"], 1)                       # the two windows overlap and merge
        self.assertTrue(df.t_r.between(1060.0, 1250.0).all())
        self.assertEqual(df.beat_idx.tolist(), list(range(len(df))))
        self.assertEqual(waves["art"].dtype, np.float16)
        self.assertEqual(waves["ecg"].shape, (len(df), 200))
        self.assertAlmostEqual(float(df.ptt_ecg_art.median()), TRUE_PTT_ART, delta=0.015)
        self.assertGreater(len(df), 0.9 * (250 - 60) * 72 / 60)
        none, w2, _ = B.process_case(t + 1000.0, art, ecg, ppg, [5000.0], [5100.0])
        self.assertEqual(len(none), 0)

    def test_irregular_row_placement_is_densified(self):
        t, r, ecg, art, ppg = synth(duration=120.0, seed=29)
        keep = ~((t > 60) & (t < 70))                              # rows physically missing from the file (not NaN)
        df, _, _ = B.process_case(t[keep], art[keep], ecg[keep], ppg[keep], [5.0], [115.0], store_waveforms=False)
        self.assertTrue(((df.t_r > 61) & (df.t_r < 69) & df.valid_beat).sum() == 0)
        self.assertGreater(df.valid_beat.mean(), 0.8)


if __name__ == "__main__":
    unittest.main()
