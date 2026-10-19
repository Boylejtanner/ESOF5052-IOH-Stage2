# Check bshi bench using controlled inputs and isolated files.
import unittest

import numpy as np
import pandas as pd

from hemodynamics import bshi_bench as K

CFG = {"dev_subset": {"max_gb": 3, "per_split_max": {"train": 600, "validation": 200, "test": 200}, "max_per_case": 5, "min_event_share": 0.25, "min_mover_share": 0.2, "seed": 5052}}


def fake_index(seed=0):
    # Seed the generated values so this check can be repeated.
    rng = np.random.default_rng(seed)
    rows = []
    wid = 0
    for ds, n_cases, per_case, p_event in (("vitaldb", 120, 40, 0.04), ("mover", 24, 30, 0.07)):
        for c in range(n_cases):
            split = ("train", "train", "train", "train", "validation", "test")[c % 6]
            for k in range(per_case):
                rows.append((wid, ds, f"{ds[0]}{c}", f"{ds}:{c}", split, bool(rng.random() < p_event)))
                wid += 1
    f = pd.DataFrame(rows, columns=["window_id", "dataset", "case_id", "patient_key", "split_pooled", "y_event"])
    f["primary_eligible"], f["in_full"], f["has60"] = True, True, True
    f["t_end_sec"], f["time_to_event_sec"] = np.arange(len(f), dtype=float), np.nan
    return f


class TestDevSelection(unittest.TestCase):
    def test_caps_shares_and_weights(self):
        # Verify caps shares and weights.
        idx = fake_index()
        sel, notes = K.select_dev_windows(idx, CFG)
        self.assertLessEqual(int(sel.groupby(["dataset", "case_id"]).size().max()), CFG["dev_subset"]["max_per_case"])
        for split, n in notes.items():
            # Check mover share against the expected result.
            self.assertGreaterEqual(n["mover_share"], 0.2, split)
            self.assertGreaterEqual(n["event_share"], 0.25 - 0.02, split)
            self.assertLessEqual(n["selected"], n["requested"])
        pop = idx.assign(y=idx.y_event).groupby(["split_pooled", "dataset", "y"]).size()
        w = sel.groupby(["split", "dataset", "y"]).sampling_weight.sum()
        for key, value in w.items():
            self.assertAlmostEqual(value, pop[key], delta=1e-3 * pop[key])                    # weighted counts recover the natural stratum sizes

    def test_selection_is_deterministic_and_keeps_patients_in_one_split(self):
        # Verify selection is deterministic and keeps patients in one split.
        idx = fake_index()
        a, _ = K.select_dev_windows(idx, CFG)
        b, _ = K.select_dev_windows(idx, CFG)
        pd.testing.assert_frame_equal(a, b)
        self.assertTrue((a.groupby("patient_key").split.nunique() == 1).all())
        self.assertTrue(a.window_id.is_unique)


if __name__ == "__main__":
    unittest.main()
