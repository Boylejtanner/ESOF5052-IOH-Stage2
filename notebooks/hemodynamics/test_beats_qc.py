import unittest

import numpy as np
import pandas as pd

from hemodynamics import beats_qc as Q


def spiky_ecg(n=3000, fs=100.0, period=0.8, sign=1.0):
    t = np.arange(n) / fs
    x = np.zeros(n)
    r = np.arange(50, n - 50, int(period * fs))
    x[r] = sign * 1000.0
    return t, x, r / fs


class TestRPeakAlignment(unittest.TestCase):
    def test_markers_on_r_peaks_pass_for_either_polarity(self):
        for sign in (1.0, -1.0):
            t, x, t_r = spiky_ecg(sign=sign)
            self.assertGreater(Q.r_peak_alignment(t_r, t, x), 0.99)

    def test_constant_offset_onto_the_t_wave_fails(self):
        t, x, t_r = spiky_ecg()
        x[(np.rint((t_r + 0.25) * 100)).astype(int)] = -300.0     # undershoot where a late marker would land
        self.assertLess(Q.r_peak_alignment(t_r + 0.25, t, x), 0.05)

    def test_no_usable_markers_gives_nan(self):
        t, x, _ = spiky_ecg()
        self.assertTrue(np.isnan(Q.r_peak_alignment(np.array([0.0, 1000.0]), t, x)))


class TestGates(unittest.TestCase):
    @staticmethod
    def case(ds, **over):
        row = {"dataset": ds, "art_align_rate": 0.95, "hr_diff_bpm": 0.5, "ptt_ecg_art": 0.18, "ptt_ecg_ppg": 0.5, "ptt_art_ppg": 0.3, "ptt_ecg_art_iqr": 0.02,
               "r_peak_alignment": 0.98}
        row.update(over)
        return row

    def test_all_good_passes(self):
        table, ok = Q.evaluate_gates(pd.DataFrame([self.case("vitaldb"), self.case("mover")]))
        self.assertTrue(ok)
        self.assertTrue(table["pass"].all())

    def test_misplaced_markers_fail_even_when_other_gates_pass(self):
        table, ok = Q.evaluate_gates(pd.DataFrame([self.case("vitaldb"), self.case("mover", r_peak_alignment=0.2)]))
        self.assertFalse(ok)
        failed = table.loc[~table["pass"]]
        self.assertEqual(list(failed.check), ["r_peak_alignment"])
        self.assertEqual(list(failed.dataset), ["mover"])

    def test_mover_art_gate_is_looser_than_vitaldb(self):
        table, ok = Q.evaluate_gates(pd.DataFrame([self.case("vitaldb", art_align_rate=0.85), self.case("mover", art_align_rate=0.85)]))
        self.assertFalse(ok)
        failed = table.loc[~table["pass"]]
        self.assertEqual(list(failed.dataset), ["vitaldb"])


class TestSelection(unittest.TestCase):
    def test_gate_json_is_serialisable(self):
        import json
        json.dumps(Q.GATES_JSON())


if __name__ == "__main__":
    unittest.main()
