# Check bshi audit using controlled inputs and isolated files.
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import bshi as S
from hemodynamics import bshi_audit as AU

ROOT = Path(__file__).resolve().parents[2]


def frame(n=40):
    # Collect the per-record values into a table for comparison.
    rng = np.random.default_rng(0)
    return pd.DataFrame({"dataset": ["vitaldb"] * n, "case_id": [f"{i // 10:04d}" for i in range(n)], "patient_key": [f"p{i // 10}" for i in range(n)], "window_id": np.arange(n),
                         "delta_map": rng.random(n).astype("float32"), "delta_map_robust": rng.random(n).astype("float32"), "aut65": rng.random(n).astype("float32"),
                         "aut65_low_coverage": np.arange(n) % 10 == 0, "y_event": np.arange(n) % 5 == 0, "time_to_event_sec": np.where(np.arange(n) % 5 == 0, 100.0, np.nan).astype("float32")})


class TestHelpers(unittest.TestCase):
    def test_loss_masks_drop_low_coverage_and_nan(self):
        # Verify loss masks drop low coverage and nan.
        df = frame()
        df.loc[3, "delta_map"] = np.nan
        m = AU.loss_masks(df)
        self.assertTrue(m["event"].all())
        self.assertFalse(m["delta_map"][0] or m["delta_map"][3])          # low coverage and NaN are masked
        self.assertTrue(m["aut65"][1])
        self.assertEqual(int(m["time_to_event"].sum()), 8)

    def test_clean_makes_json_safe(self):
        out = AU.clean({"a": np.float32(1.5), "b": [np.int64(2), float("nan")], "c": np.bool_(True), "d": float("inf")})
        self.assertEqual(json.loads(json.dumps(out, allow_nan=False)), {"a": 1.5, "b": [2, None], "c": True, "d": None})

    def test_sample_by_case_caps_windows_per_case_and_is_deterministic(self):
        # Verify sample by case caps windows per case and is deterministic.
        df = frame(100)
        a = AU._sample_by_case(df, 30, 1, 5)
        self.assertEqual(len(a), 30)
        self.assertLessEqual(int(a.groupby("case_id").size().max()), 5)
        self.assertTrue(a.equals(AU._sample_by_case(df, 30, 1, 5)))

    def test_smd_is_signed_and_zero_for_constant_features(self):
        # Verify smd is signed and zero for constant features.
        a = np.array([[1.0, 5.0], [3.0, 5.0], [2.0, 5.0]])
        b = a + np.array([2.0, 0.0])
        s = AU._smd(a, b)
        self.assertLess(s[0], 0)
        self.assertEqual(s[1], 0.0)

    def test_patient_means_have_one_row_per_patient(self):
        # Compare the result with the expected value or tolerance.
        df = frame(40)
        pm = AU._patient_means(df, np.arange(40 * 2, dtype=float).reshape(40, 2), ["x", "y"])
        self.assertEqual(len(pm), 4)

    def test_protocol_rows_use_the_right_flag(self):
        df = pd.DataFrame({"primary_eligible": [True, True, False], "in_core": [True, True, True], "in_full": [True, False, True]})
        self.assertEqual(len(AU.protocol_rows(df, "ea_core")), 2)
        self.assertEqual(len(AU.protocol_rows(df, "eap_full")), 1)


@unittest.skipUnless((ROOT / AU.RESULT).exists() and (ROOT / S.OUT_DIR / "context_stats_v1.json").exists(), "audit has not been run")
class TestSavedAudit(unittest.TestCase):
    def test_all_gates_pass_and_stats_are_consistent(self):
        # Verify all gates pass and stats are consistent.
        res = json.loads((ROOT / AU.RESULT).read_text(encoding="utf-8"))
        self.assertTrue(all(res["gates"].values()), [k for k, v in res["gates"].items() if not v])
        st = json.loads((ROOT / S.OUT_DIR / "context_stats_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(st["names"], S.CONTEXT_NAMES)
        for proto in ("eap_full", "ea_core"):
            self.assertEqual(len(st[proto]["center"]), len(S.CONTEXT_NAMES))
            self.assertTrue(all(s > 0 for s in st[proto]["scale"]))
        zeroed = set(st["ea_core"]["zeroed_by_variant"])
        self.assertEqual(zeroed, set(S.context_off(S.VARIANTS["ea"])))
        self.assertEqual(st["source_split"], "train")

    def test_readiness_document_is_short_and_has_a_verdict(self):
        # Compare the result with the expected value or tolerance.
        text = (ROOT / AU.DOC).read_text(encoding="utf-8")
        self.assertLessEqual(len(text.splitlines()), 90)
        self.assertIn("**Verdict: SUFFICIENT AFTER THE FIXES BELOW.**", text)

    def test_case_meta_has_departments_for_every_vitaldb_case(self):
        # Verify case meta has departments for every VitalDB case.
        meta = pd.read_parquet(ROOT / AU.CASE_META)
        v = meta[meta.dataset == "vitaldb"]
        self.assertTrue(v.department.notna().all())
        self.assertTrue(meta[meta.dataset == "mover"].department.isna().all())
        self.assertTrue(meta.patient_key.notna().all())


if __name__ == "__main__":
    unittest.main()
