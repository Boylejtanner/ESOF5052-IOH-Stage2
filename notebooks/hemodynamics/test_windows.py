# Check windows using controlled inputs and isolated files.
import unittest

import numpy as np
import pandas as pd

from hemodynamics import windows as W

FS = 100.0


def _art(n, seed=0):
    t = np.arange(n) / FS
    return (90 + 20 * np.sin(2 * np.pi * 1.2 * t) + np.random.default_rng(seed).normal(0, .5, n)).astype(np.float32)


def _holes(values, on=8.0, off=2.0):
    out = values.copy()
    period = int((on + off) * FS)
    for start in range(int(on * FS), len(out), period):
        out[start:start + int(off * FS)] = np.nan
    return out


def _map(duration, value=90.0, cadence=2.0):
    t = np.arange(0, duration, cadence)
    return t, np.full(len(t), value)


class ReadyWindowTests(unittest.TestCase):
    def setUp(self):
        self.duration = 900
        self.n = int(self.duration * FS)
        self.rule = W.load_rule()
        self.art = _art(self.n)
        self.ecg = np.sin(np.arange(self.n) / 7.0).astype(np.float32)
        self.ppg = np.cos(np.arange(self.n) / 9.0).astype(np.float32)
        self.mt, self.mv = _map(self.duration)

    def windows(self, art=None, mt=None, mv=None, rule=None):
        return W.ready_windows(self.art if art is None else art, self.ecg, self.ppg, FS,
                               self.mt if mt is None else mt, self.mv if mv is None else mv, rule or self.rule, case_id="c1")

    def test_structural_gaps_pass_at_075_but_not_at_09(self):
        # Verify structural gaps pass at 075 but not at 09.
        art, ecg, ppg = (_holes(x) for x in (self.art, self.ecg, self.ppg))
        frame = W.ready_windows(art, ecg, ppg, FS, self.mt, self.mv, self.rule, case_id="c1")
        self.assertTrue(frame.is_ready.sum() > 0)
        self.assertTrue(frame.min_valid_frac.between(0.75, 0.85).all())
        summary = W.case_summary(frame, self.rule)
        self.assertGreater(summary["ready_span_min_theta_75"], 8)
        self.assertEqual(summary["n_ready_theta_90"], 0)
        self.assertEqual(summary["ready_span_min_theta_90"], 0.0)

    def test_flat_line_art_is_rejected(self):
        # Verify flat line ART is rejected.
        flat = np.zeros(self.n, dtype=np.float32)
        frame = self.windows(art=flat)
        self.assertEqual(int(frame.is_ready.sum()), 0)
        self.assertFalse(frame.art_plausible.any())

    def test_art_map_mismatch_is_rejected(self):
        # Compare the result with the expected value or tolerance.
        frame = self.windows(mv=np.full(len(self.mt), 40.0))
        self.assertEqual(int(frame.is_ready.sum()), 0)
        self.assertTrue((frame.art_map_diff.dropna() > 20).all())

    def test_horizon_gap_rule(self):
        # Verify horizon gap rule.
        keep = ~((self.mt > 400) & (self.mt < 600))
        frame = self.windows(mt=self.mt[keep], mv=self.mv[keep])
        early = frame.loc[frame.t_end_sec <= 210]
        straddle = frame.loc[(frame.t_end_sec >= 240) & (frame.t_end_sec <= 590)]
        self.assertTrue(len(early) and early.is_ready.all())
        self.assertTrue(len(straddle) and not straddle.is_ready.any())
        self.assertTrue((straddle.map_horizon_max_gap_sec > self.rule["horizon_max_gap_sec"]).any())

    def test_windows_without_enough_future_map_are_not_ready(self):
        # Verify windows without enough future MAP are not ready.
        frame = self.windows()
        last_map = float(self.mt[-1])
        late = frame.loc[frame.t_end_sec + self.rule["horizon_sec"] - last_map > self.rule["horizon_max_gap_sec"] + 1]
        early = frame.loc[frame.t_end_sec + self.rule["horizon_sec"] <= last_map]
        self.assertTrue(len(late) and not late.is_ready.any())
        self.assertTrue(len(early) and early.is_ready.all())

    def test_no_map_means_no_ready_windows(self):
        frame = self.windows(mt=np.array([]), mv=np.array([]))
        self.assertEqual(int(frame.is_ready.sum()), 0)

    def test_eligibility_does_not_depend_on_map_level(self):
        # Outcome blindness: low-but-covered MAP must not change readiness when ART agrees with MAP.
        art = (self.art - 90 + 55).astype(np.float32)
        frame = self.windows(art=art, mv=np.full(len(self.mt), 55.0))
        baseline = self.windows()
        self.assertEqual(int(frame.is_ready.sum()), int(baseline.is_ready.sum()))

    def test_union_span_merges_overlaps(self):
        # Compare the result with the expected value or tolerance.
        self.assertAlmostEqual(W.union_span_minutes([0, 30, 60], 90), 2.5)
        self.assertAlmostEqual(W.union_span_minutes([0, 300], 90), 3.0)
        self.assertEqual(W.union_span_minutes([], 90), 0.0)

    def test_deterministic(self):
        a, b = self.windows(), self.windows()
        pd.testing.assert_frame_equal(a, b)


class LadderTests(unittest.TestCase):
    def setUp(self):
        self.ladder, self.targets = W.load_ladder({})

    def _stats(self, spans):
        rows = []
        for i, (s75, s70, s60, s50) in enumerate(spans):
            rows.append({"patient_id": f"p{i}", "case_id": f"c{i}", "ready_span_min_theta_75": s75, "ready_span_min_theta_70": s70,
                         "ready_span_min_theta_60": s60, "ready_span_min_theta_50": s50})
        return pd.DataFrame(rows)

    def test_strictest_rung_meeting_target_is_chosen(self):
        # Verify strictest rung meeting target is chosen.
        spans = [(35, 40, 45, 50)] * 150 + [(5, 25, 30, 40)] * 160 + [(0, 0, 0, 12)] * 100
        rung, funnel, _ = W.choose_rung(self._stats(spans), self.ladder, {"min_cases": 300, "min_patients": 250, "floor_cases": 150})
        self.assertEqual(rung["name"], "R2")
        self.assertEqual(funnel.set_index("rung").loc["R1", "cases"], 150)
        self.assertEqual(funnel.set_index("rung").loc["R4", "cases"], 410)

    def test_falls_back_to_loosest_and_reports(self):
        # Verify falls back to loosest and reports.
        spans = [(0, 0, 0, 12)] * 200
        rung, _, reason = W.choose_rung(self._stats(spans), self.ladder, {"min_cases": 300, "min_patients": 250, "floor_cases": 150})
        self.assertEqual(rung["name"], "R4")
        self.assertIn("loosest", reason)

    def test_stops_below_floor(self):
        # Check rung against the expected result.
        rung, _, reason = W.choose_rung(self._stats([(0, 0, 0, 12)] * 40), self.ladder, self.targets)
        self.assertIsNone(rung)
        self.assertIn("not creating a cohort", reason)

    def test_patient_target_counts_distinct_patients(self):
        # Verify patient target counts distinct patients.
        stats = self._stats([(35, 40, 45, 50)] * 320)
        stats["patient_id"] = [f"p{i % 100}" for i in range(320)]
        rung, _, _ = W.choose_rung(stats, self.ladder, {"min_cases": 300, "min_patients": 250, "floor_cases": 150})
        self.assertEqual(rung["name"], "R4")

    def test_configured_theta_must_exist(self):
        with self.assertRaises(ValueError):
            W.load_ladder({"eligibility_ladder": [{"name": "X", "theta": 0.55, "min_ready_span_min": 5}]})


if __name__ == "__main__":
    unittest.main()
