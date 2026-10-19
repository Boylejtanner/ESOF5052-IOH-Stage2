import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from hemodynamics import labels as L
from hemodynamics import snapshot as S

RULE = L.load_label_rule({})


def track(duration=1200, level=80.0, cadence=2.0, dips=(), drop_times=()):
    """MAP track at 2 s cadence; dips = (start, end, value) overrides; drop_times removes samples in [a, b)."""
    t = np.arange(0.0, duration, cadence)
    v = np.full(len(t), float(level))
    for a, b, value in dips:
        v[(t >= a) & (t <= b)] = value
    keep = np.ones(len(t), dtype=bool)
    for a, b in drop_times:
        keep &= ~((t >= a) & (t < b))
    return t[keep], v[keep]


def windows(*t0s):
    t0 = np.array(t0s, dtype=float)
    return pd.DataFrame({"t_start_sec": t0 - 90, "t_end_sec": t0, "min_valid_frac": 1.0})


def one(t0, t, v, rule=RULE):
    return L.label_case(windows(t0), t, v, rule).iloc[0]


class DefinitionTests(unittest.TestCase):
    def test_sustained_needs_60_seconds_of_consecutive_samples(self):
        t, v = track(dips=[(400, 460, 55)])                      # samples 400..460 -> last-first = 60 s
        self.assertTrue(one(300, t, v).y_event)
        t, v = track(dips=[(400, 458, 55)])                      # 58 s
        self.assertFalse(one(300, t, v).y_event)

    def test_gap_longer_than_10_seconds_breaks_a_run(self):
        t, v = track(dips=[(400, 470, 55)], drop_times=[(430, 442)])   # 14 s hole inside the dip
        self.assertFalse(one(300, t, v).y_event)
        t, v = track(dips=[(400, 470, 55)], drop_times=[(430, 438)])   # 10 s step still continuous: 70 s run
        self.assertTrue(one(300, t, v).y_event)

    def test_qualifying_minute_must_come_from_samples_inside_the_horizon(self):
        t, v = track(dips=[(550, 640, 55)])                      # starts +250 s: only 50 s fit inside H=300
        self.assertFalse(one(300, t, v).y_event)
        t, v = track(dips=[(500, 570, 55)])                      # +200 s .. +270 s
        row = one(300, t, v)
        self.assertTrue(row.y_event)
        self.assertAlmostEqual(float(row.time_to_event_sec), 200.0)
        t, v = track(dips=[(400, 900, 55)])                      # long episode starting +100 s keeps going past the horizon: still an event
        row = one(300, t, v)
        self.assertTrue(row.y_event)
        self.assertAlmostEqual(float(row.time_to_event_sec), 100.0)
        t, v = track(dips=[(540, 900, 55)])                      # starts +240 s: exactly 60 s inside the horizon
        self.assertTrue(one(300, t, v).y_event)
        t, v = track(dips=[(542, 900, 55)])                      # starts +242 s: 58 s inside
        self.assertFalse(one(300, t, v).y_event)

    def test_already_hypotensive_lookback_and_current_flag(self):
        t, v = track(dips=[(270, 290, 55)])                      # dip ended 10 s before t0=300
        row = one(300, t, v)
        self.assertTrue(row.already_hypotensive and not row.eligible_onset)
        self.assertFalse(row.map_current_lt65)                   # but MAP is back above 65 now
        self.assertTrue(one(400, t, v).eligible_onset)           # 110 s later the lookback is clear
        t, v = track(dips=[(290, 300, 60)])
        self.assertTrue(one(300, t, v).map_current_lt65)

    def test_gray_zone_tag(self):
        for level, expected in ((70.0, True), (80.0, False), (64.0, False), (75.0, False)):
            t, v = track(level=level)
            self.assertEqual(bool(one(300, t, v).gray_zone), expected, level)

    def test_delta_map_and_robust_variant(self):
        t, v = track(dips=[(450, 470, 50)])
        self.assertAlmostEqual(float(one(300, t, v).delta_map), 30.0)
        t, v = track()
        self.assertEqual(float(one(300, t, v).delta_map), 0.0)
        t, v = track()
        v[np.searchsorted(t, 500)] = 22.0                        # isolated artefact inside the valid range
        row = one(300, t, v)
        self.assertAlmostEqual(float(row.delta_map), 58.0)
        self.assertLess(float(row.delta_map_robust), 5.0)

    def test_slope_is_mmhg_per_minute(self):
        t = np.arange(0.0, 600, 2.0)
        v = 70 + t / 10.0                                        # +1 mmHg per 10 s = 6 mmHg/min
        self.assertAlmostEqual(float(one(300, t, v).map_slope_60), 6.0, places=3)

    def test_aut65_constant_and_against_brute_force(self):
        t, v = track(level=55.0)
        row = one(300, t, v)
        self.assertAlmostEqual(float(row.aut65), 50.0, places=3)             # 10 mmHg x 5 min
        self.assertAlmostEqual(float(row.aut65_observed_fraction), 1.0, places=2)
        rng = np.random.default_rng(5)
        t = np.arange(0.0, 900, 2.0)
        v = np.clip(65 + 18 * np.sin(t / 37.0) + rng.normal(0, 4, len(t)), 21, 190)
        keep = ~((t > 520) & (t < 560))                          # a 42 s hole: unobserved, contributes 0
        t, v = t[keep], v[keep]
        got = one(200, t, v)
        grid = np.arange(200.0, 500.0, 0.005)
        interp = np.interp(grid, t, v)
        k = np.clip(np.searchsorted(t, grid, side="right") - 1, 0, len(t) - 2)
        observed = (t[k + 1] - t[k]) <= RULE["max_gap_sec"]
        brute = float(np.sum(np.where(observed, np.maximum(0.0, 65 - interp), 0.0)) * 0.005 / 60)
        self.assertAlmostEqual(float(got.aut65), brute, delta=0.02)
        self.assertAlmostEqual(float(got.aut65_observed_fraction), float(np.mean(observed)), delta=0.002)

    def test_gap_flags_ambiguous_negative_and_low_coverage(self):
        t, v = track(drop_times=[(400, 500)])                    # 100 s hole in the horizon of t0=300
        row = one(300, t, v)
        self.assertTrue(row.label_ambiguous and row.aut65_low_coverage)
        self.assertGreater(float(row.horizon_gap_max_sec), 60.0)
        t, v = track(dips=[(500, 570, 55)], drop_times=[(380, 450)])
        row = one(300, t, v)
        self.assertTrue(row.y_event and not row.label_ambiguous)           # an event is never "ambiguous"

    def test_no_map_is_handled_without_error(self):
        row = one(300, np.array([]), np.array([]))
        self.assertFalse(row.eligible_onset)
        self.assertFalse(row.y_event)
        self.assertTrue(np.isnan(row.map_current))

    def test_values_outside_valid_range_are_ignored_and_output_is_deterministic(self):
        t, v = track()
        v[10] = 5.0
        v[11] = 300.0
        a, b = L.label_case(windows(200, 300, 400), t, v, RULE), L.label_case(windows(200, 300, 400), t, v, RULE)
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(float(a.map_min_60.min()), 80.0)

    def test_rule_hash_changes_with_the_rule(self):
        self.assertNotEqual(L.label_rule_sha256(RULE), L.label_rule_sha256({**RULE, "sustained_min_sec": 30}))

    def test_metric_helpers(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(L.auroc(y, np.array([.1, .2, .8, .9])), 1.0)
        self.assertEqual(L.auroc(y, np.array([.5, .5, .5, .5])), 0.5)
        self.assertAlmostEqual(L.auprc(y, np.array([.1, .2, .8, .9])), 1.0)
        self.assertTrue(np.isnan(L.auroc(np.zeros(3), np.ones(3))))


def build_root(base, patients=12):
    root = Path(base)
    for sub in ("data/manifests", "configs", "notebooks/hemodynamics", "docs", "results"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    rows, wins = [], {"mover": [], "vitaldb": []}
    rng = np.random.default_rng(1)
    for ds in ("mover", "vitaldb"):
        for i in range(patients):
            cid = f"{ds[0]}{i}"
            d = root / "data/parquet" / ds / f"case_id={cid}"
            d.mkdir(parents=True)
            t, v = track(duration=1500, dips=[(600 + 40 * i, 700 + 40 * i, 55)])
            v = v + rng.normal(0, 1, len(v))
            pd.DataFrame({"time_sec": t, "MAP": v.astype("float32")}).to_parquet(d / "map.parquet", index=False)
            rows.append({"dataset": ds, "case_id": cid, "patient_key": f"{ds}:{i}", "split_within_dataset": ["train", "train", "validation", "test"][i % 4],
                         "split_pooled": ["train", "train", "validation", "test"][i % 4], "waveform_sha256": f"w{cid}", "map_sha256": f"m{cid}",
                         "map_path": f"data/parquet/{ds}/case_id={cid}/map.parquet"})
            t0 = np.arange(100.0, 1100.0, 30.0)
            wins[ds].append(pd.DataFrame({"case_id": cid, "t_start_sec": t0 - 90, "t_end_sec": t0, "min_valid_frac": 0.9, "is_ready": True}))
    pd.DataFrame(rows).to_parquet(root / "data/manifests/unified_case_index.parquet", index=False)
    pd.DataFrame(rows).to_parquet(root / "data/manifests/vitaldb_patient_splits.parquet", index=False)
    for ds, frames in wins.items():
        pd.concat(frames).to_parquet(root / f"data/manifests/ready_windows_{ds}.parquet", index=False)
    (root / "configs/hemodynamics.json").write_text(json.dumps({"random_seed": 5052, "patient_split_seed": 5052, "label_rule": {"sustained_min_sec": 60}}))
    return root


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = build_root(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_guard_then_incremental_build_and_rule_change(self):
        with self.assertRaises(FileNotFoundError):
            L.build_labels(self.root, workers=1, progress=False)
        S.snapshot_dataset_v1(self.root)
        first = L.build_labels(self.root, workers=1, progress=False)
        self.assertEqual({k: v["computed"] for k, v in first["datasets"].items()}, {"mover": 12, "vitaldb": 12})
        lab = L.load_labels(self.root)
        self.assertEqual(list(lab.columns), L.LABEL_COLUMNS)
        self.assertEqual(len(lab), 24 * 34)
        self.assertTrue(lab.y_event.any() and lab.eligible_onset.any())
        second = L.build_labels(self.root, workers=1, progress=False)
        self.assertEqual({k: v["skipped_current"] for k, v in second["datasets"].items()}, {"mover": 12, "vitaldb": 12})
        config = self.root / "configs/hemodynamics.json"
        config.write_text(json.dumps({"label_rule": {"sustained_min_sec": 30}}))
        third = L.build_labels(self.root, workers=1, progress=False)
        self.assertEqual(third["datasets"]["mover"]["computed"], 12)
        self.assertNotEqual(set(L.load_labels(self.root).label_rule_sha256), {first["rule_sha256"]})
        by_split, pooled = L.summarize_labels(self.root, write=True)
        self.assertEqual(int(by_split.windows.sum()), len(L.load_labels(self.root)))
        self.assertTrue((self.root / "results/label_summary.json").exists())

    def test_baseline_a_runs_on_a_labelled_set(self):
        S.snapshot_dataset_v1(self.root)
        L.build_labels(self.root, workers=1, progress=False)
        out = L.baseline_a(self.root, reps=5, write=True)
        self.assertEqual(set(out["settings"]), {"pooled_train->pooled_test", "vitaldb_train->vitaldb_test", "E3x vitaldb_train->all_mover"})
        e3x = out["settings"]["E3x vitaldb_train->all_mover"]
        self.assertGreater(e3x["test_windows"], 0)
        self.assertTrue(0.0 <= e3x["model"]["auroc"] <= 1.0)
        self.assertIn("auroc", e3x["model_ci95"])
        self.assertTrue((self.root / "results/baseline_A.json").exists())


if __name__ == "__main__":
    unittest.main()
