# Check bshi using controlled inputs and isolated files.
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import beats_pipeline as P
from hemodynamics import bshi as S

CFG = {"version": "v1", "window_sec": 90, "n_beats": 60, "phase": 200, "min_usable_beats": 54, "sqi_ecg_min": 0.5, "ppg_delay_max_sec": 0.9, "ecg_robust_clip": 8.0,
       "art_center_mmhg": 80, "art_scale_mmhg": 40, "workers": 2}


def make_case(root, dataset, case_id, n=1500, seed=0):
    """A synthetic per-case beat table + float16 waveforms written exactly where the real pipeline stores them."""
    # Seed the generated values so this check can be repeated.
    rng = np.random.default_rng(seed)
    rr = np.clip(0.8 + 0.05 * rng.standard_normal(n), 0.5, 1.3)
    t_r = 5.0 + np.cumsum(rr) - rr
    # Calculate valid-sample flags for the following step.
    valid, art_ok, ppg_ok = rng.random(n) > 0.02, rng.random(n) > 0.02, rng.random(n) > 0.04
    sqi_ecg = np.clip(rng.beta(8, 1.5, n), 0.5, 1)
    pick = rng.random(n)
    sqi_ecg = np.where(pick < 0.02, rng.uniform(0.1, 0.45, n), np.where(pick < 0.04, 0.5, np.where(pick < 0.06, 0.49999, sqi_ecg)))   # boundary values on both sides of 0.5
    pick = rng.random(n)
    ptt_ppg = np.where(rng.random(n) < 0.03, rng.uniform(0.95, 1.2, n), rng.normal(0.5, 0.03, n))
    ptt_ppg = np.where(pick < 0.02, 0.9, np.where(pick < 0.04, 0.9001, ptt_ppg))                                                       # boundary values around 0.9
    art_cols = ["sbp", "dbp", "pp", "map_beat", "dpdt_max", "sys_time", "art_area", "ptt_ecg_art", "hr_art_bpm"]
    ppg_cols = ["ppg_amp", "ppg_ac_dc", "ppg_upstroke", "ptt_art_ppg", "art_ppg_corr"]
    df = pd.DataFrame({"beat_idx": np.arange(n, dtype=np.int64), "t_r": t_r, "rr_sec": rr.astype(np.float32), "hr_bpm": (60 / rr).astype(np.float32), "ecg_conf": np.float32(0.9)})
    for c in art_cols + ppg_cols:
        df[c] = (rng.normal(50, 10, n) + 100 * rng.random() * (c == "sbp")).astype(np.float32)
    df["sbp"] = (df["dbp"] + np.abs(df["pp"]) + 20).astype(np.float32)
    df["ptt_ecg_art"] = rng.normal(0.18, 0.02, n).astype(np.float32)
    df["ptt_ecg_ppg"] = ptt_ppg.astype(np.float32)
    df["art_ppg_corr"] = rng.uniform(-1, 1, n).astype(np.float32)
    for c in ("art_foot_t", "art_sys_t", "ppg_foot_t", "ppg_peak_t"):
        df[c] = t_r
    # Calculate the observation count for the following step.
    df["art_int_sec"] = rr.astype(np.float32)
    df["sqi_ecg"], df["sqi_art"] = sqi_ecg.astype(np.float32), rng.beta(8, 1.5, n).astype(np.float32)
    df["sqi_ppg"], df["sqi_cross"] = rng.beta(8, 1.5, n).astype(np.float32), rng.beta(8, 1.5, n).astype(np.float32)
    df["flags"] = np.int64(0)
    df["valid_beat"], df["art_ok"], df["ppg_ok"] = valid, art_ok, ppg_ok
    df.loc[rng.choice(n, 3, replace=False), "dpdt_max"] = np.nan                      # a masked-in beat with a missing feature value
    waves = {"ecg": rng.standard_normal((n, 200)), "art": 90 + 25 * np.sin(np.linspace(0, 6.28, 200))[None, :] + rng.normal(0, 3, (n, 200)), "ppg": rng.standard_normal((n, 200))}
    waves["ecg"][~valid] = np.nan
    waves["art"][~art_ok] = np.nan
    # Create the destination directory before writing any outputs.
    waves["ppg"][~ppg_ok] = np.nan
    waves = {k: v.astype(np.float16) for k, v in waves.items()}
    d = P.case_dir(root, dataset, case_id)
    d.mkdir(parents=True, exist_ok=True)
    P._atomic_parquet(df, d / "beats.parquet")
    P._atomic_parquet(P.waveform_table(df, waves, 200), d / "beat_waveforms.parquet")
    return df, waves


def window_rows(dataset, case_id, df, stride=9.0, seed=0, start_id=0):
    # Collect the per-record values into a table for comparison.
    rng = np.random.default_rng(seed + 1)
    t_end = np.arange(100.0, float(df.t_r.iloc[-1]) - 1.5, stride)
    r = S.index_windows(df.t_r.to_numpy(), df.rr_sec.to_numpy(), df.valid_beat.to_numpy(), df.sqi_ecg.to_numpy(), df.art_ok.to_numpy(), df.ppg_ok.to_numpy(),
                        df.ptt_ecg_ppg.to_numpy(), t_end, CFG)
    rows = pd.DataFrame({"window_id": np.arange(start_id, start_id + len(t_end), dtype=np.int64), "dataset": dataset, "case_id": case_id, "t_end_sec": t_end, **r})
    for c in S.LABEL_CONTEXT:
        rows[c] = rng.normal(80, 10, len(rows))
    rows["y_event"], rows["gray_zone"] = rng.random(len(rows)) < 0.1, rng.random(len(rows)) < 0.2
    rows["delta_map"], rows["delta_map_robust"], rows["aut65"] = rng.normal(0, 5, len(rows)), rng.normal(0, 5, len(rows)), np.abs(rng.normal(0, 5, len(rows)))
    return rows


class World:
    def __init__(self, tmp):
        # Store the settings and results as JSON for later inspection.
        self.root = Path(tmp)
        (self.root / "configs").mkdir(parents=True)
        (self.root / "configs" / "hemodynamics.json").write_text(json.dumps({"bshi": CFG}))
        self.cases, frames, nid = {}, [], 0
        for k, (ds, cid) in enumerate((("mover", "m0"), ("vitaldb", "v0"), ("vitaldb", "v1"))):
            df, waves = make_case(self.root, ds, cid, seed=k)
            self.cases[(ds, cid)] = (df, waves)
            rows = window_rows(ds, cid, df, seed=k, start_id=nid)
            nid += len(rows)
            frames.append(rows)
        self.index = pd.concat(frames, ignore_index=True)
        self.usable = self.index.loc[self.index.has60 & (self.index.n_art_ok >= 54)].reset_index(drop=True)

    def sample(self, n, seed=0, ppg=False):
        pool = self.index.loc[self.index.has60 & (self.index.n_ppg_ok >= 54)] if ppg else self.usable
        return pool.sample(min(n, len(pool)), random_state=seed).reset_index(drop=True)


def nr(sorted_vals, q):
    return sorted_vals[int(math.floor((len(sorted_vals) - 1) * q + 0.5))]


REF_MASK_OF = {"ptt_ecg_art": "art", "ptt_ecg_ppg": "ppg", "ptt_art_ppg": "ppg", "rr_sec": "ecg", "pp": "art", "dpdt_max": "art", "sys_time": "art", "art_area": "art",
               "ppg_amp": "ppg", "ppg_ac_dc": "ppg", "ppg_upstroke": "ppg", "art_ppg_corr": "ppg", "sqi_ecg": "ecg", "sqi_art": "art", "sqi_ppg": "ppg", "sqi_cross": "art"}


def reference_window(df, waves, lo, ctx_label_values, variant):
    """Independent, loop-based implementation of the documented spec (not sharing code with bshi.py)."""
    # Reserve output arrays and masks with explicit initial values.
    off = S.VARIANTS[variant]
    x = np.zeros((7, 60, 200), dtype=np.float64)
    beats = [df.iloc[lo + k] for k in range(60)]
    m_ecg = [bool(b.valid_beat) and float(b.sqi_ecg) >= 0.5 for b in beats]
    m_art = [m_ecg[k] and bool(beats[k].art_ok) for k in range(60)]
    m_ppg = [m_art[k] and bool(beats[k].ppg_ok) and float(beats[k].ptt_ecg_ppg) <= 0.9 for k in range(60)]
    if "ppg" in off:
        m_ppg = [False] * 60
    masks = {"ecg": m_ecg, "art": m_art, "ppg": m_ppg}

    def robust(name, mk):
        vals = []
        for k in range(60):
            if mk[k]:
                w = waves[name][lo + k].astype(np.float64)
                vals += [v for v in w if np.isfinite(v)]
        if not vals:
            return
        s = sorted(vals)
        med, scale = nr(s, .5), nr(s, .75) - nr(s, .25)
        scale = scale if scale > 1e-6 else 1.0
        for k in range(60):
            if mk[k]:
                # Exclude or flag missing values before using these measurements.
                w = waves[name][lo + k].astype(np.float64)
                z = np.clip((w - med) / scale, -8, 8)
                x[{"ecg": 0, "ppg": 2}[name], k] = np.where(np.isfinite(w), z, 0.0)

    robust("ecg", m_ecg)
    if "ppg" not in off:
        robust("ppg", m_ppg)
    for k in range(60):
        if m_art[k]:
            w = waves["art"][lo + k].astype(np.float64)
            x[1, k] = np.where(np.isfinite(w), np.clip((w - 80) / 40, -8, 8), 0.0)

    def zs(name, mk):
        vals = [(k, float(beats[k][name])) for k in range(60) if mk[k] and np.isfinite(float(beats[k][name]))]
        out = np.zeros(60)
        if vals:
            # Summarize the observed levels and variability for this set of measurements.
            v = np.array([a for _, a in vals])
            mean, std = v.mean(), v.std()
            den = std if std > 1e-9 else 1.0
            for k, a in vals:
                out[k] = (a - mean) / den
        return out

    x[3] = zs("map_beat", m_art)[:, None]
    layout = [(4, n, k, 50) for k, n in enumerate(S.TIMING_BANDS)] + [(5, n, k, 25) for k, n in enumerate(S.MORPH_BANDS)] + [(6, n, k, 50) for k, n in enumerate(S.SQI_BANDS)]
    for plane, name, k, width in layout:
        if name in off:
            continue
        mk = masks[REF_MASK_OF[name]]
        if name.startswith("sqi_"):
            col = np.array([min(1.0, max(0.0, float(beats[j][name]))) if (mk[j] and np.isfinite(float(beats[j][name]))) else 0.0 for j in range(60)])
        else:
            col = zs(name, mk)
        x[plane, :, k * width:(k + 1) * width] = col[:, None]
    return x


class TestIndexWindows(unittest.TestCase):
    def test_hand_checked_window_selection(self):
        # Verify hand checked window selection.
        n = 200
        t_r, rr = np.arange(n, dtype=float), np.ones(n)
        art_ok = np.ones(n, bool)
        art_ok[45:50] = False
        r = S.index_windows(t_r, rr, np.ones(n, bool), np.full(n, 0.9), art_ok, np.ones(n, bool), np.full(n, 0.5), [100.0, 50.0, 100.5], CFG)
        self.assertEqual((r["beat_hi"][0], r["beat_lo"][0], bool(r["has60"][0])), (100, 40, True))   # beats 0..99 end by t=100
        self.assertEqual(int(r["n_art_ok"][0]), 55)
        self.assertFalse(r["has60"][1])                                                              # only 50 beats exist before t=50
        self.assertEqual(int(r["beat_hi"][2]), 100)                                                  # the cycle ending at 101 touches the future

    def test_no_beat_ends_after_the_forecast_origin(self):
        # Verify no beat ends after the forecast origin.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            for (ds, cid), (df, _) in w.cases.items():
                rows = w.index[(w.index.dataset == ds) & (w.index.case_id == cid) & w.index.has60]
                ce, tr = df.t_r.to_numpy() + df.rr_sec.to_numpy().astype(np.float64), df.t_r.to_numpy()
                for r in rows.itertuples():
                    # Check t end sec against the expected result.
                    self.assertLessEqual(ce[r.beat_lo:r.beat_hi].max(), r.t_end_sec)
                    self.assertGreater(tr[r.beat_lo], r.t_end_sec - 90)
                    self.assertEqual(r.beat_hi - r.beat_lo, 60)

    def test_check_index_rejects_patient_leakage_and_dead_ecg_in_core(self):
        # Verify check index rejects patient leakage and dead ECG in core.
        base = pd.DataFrame({"window_id": [0, 1], "patient_key": ["p", "p"], "split_pooled": ["train", "test"], "split_within_dataset": ["train", "test"], "has60": [True, True],
                             "in_core": [True, False], "in_full": [False, False], "primary_eligible": [True, True], "ecg_dead_case": [False, False], "beat_lo": [0, 0], "beat_hi": [60, 60]})
        with self.assertRaises(AssertionError):
            S.check_index(base)
        ok = base.assign(split_pooled="train", split_within_dataset="train")
        self.assertTrue(S.check_index(ok))
        with self.assertRaises(AssertionError):
            S.check_index(ok.assign(ecg_dead_case=True))
        with self.assertRaises(AssertionError):
            S.check_index(ok.assign(in_full=[True, True], in_core=[False, False]))


class TestAssembler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.w = World(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fast_assembler_equals_the_loop_reference_on_200_windows(self):
        # Verify fast assembler equals the loop reference on 200 windows.
        for variant in ("eap", "ea"):
            rows = self.w.sample(200, seed=3, ppg=False)
            out = S.assemble(self.w.root, rows, variant)
            worst = 0.0
            for i, r in enumerate(rows.itertuples()):
                df, waves = self.w.cases[(r.dataset, r.case_id)]
                ref = reference_window(df, waves, r.beat_lo, None, variant)
                worst = max(worst, float(np.abs(out["x"][i] - ref).max()))
            self.assertLess(worst, 2e-4, f"{variant}: max difference {worst}")

    def test_shape_dtype_and_finiteness(self):
        # Verify shape dtype and finiteness.
        rows = self.w.sample(64, seed=1)
        out = S.assemble(self.w.root, rows, "eap")
        self.assertEqual(out["x"].shape, (64, 7, 60, 200))
        self.assertEqual(out["x"].dtype, np.float32)
        self.assertTrue(np.isfinite(out["x"]).all() and np.isfinite(out["context"]).all())
        self.assertLessEqual(float(np.abs(out["x"][:, [0, 2]]).max()), 8.0)
        self.assertEqual(out["context"].shape[1], len(S.CONTEXT_NAMES))
        self.assertEqual(out["variant_mask"].shape, (64, len(S.GROUPS)))

    def test_masked_beats_are_exactly_zero_and_their_sqi_is_zero(self):
        # Verify masked beats are exactly zero and their SQI is zero.
        rows = self.w.sample(64, seed=2)
        out = S.assemble(self.w.root, rows, "eap")
        m_ecg, m_art, m_ppg = out["beat_mask"][:, 0], out["beat_mask"][:, 1], out["beat_mask"][:, 2]
        # Compare the result with the expected value or tolerance.
        self.assertTrue((out["x"][:, 1].transpose(0, 2, 1)[~np.broadcast_to(m_art[:, None, :], (64, 200, 60))] == 0).all())
        self.assertTrue((out["x"][:, 0].transpose(0, 2, 1)[~np.broadcast_to(m_ecg[:, None, :], (64, 200, 60))] == 0).all())
        sc = S.to_planes(out["x"])["scalars"]
        self.assertTrue((sc["sqi_art"][~m_art] == 0).all() and (sc["sqi_ppg"][~m_ppg] == 0).all() and (sc["sqi_ecg"][~m_ecg] == 0).all())
        self.assertTrue((sc["ptt_ecg_ppg"][~m_ppg] == 0).all() and (sc["ppg_amp"][~m_ppg] == 0).all() and (sc["sbp" if "sbp" in sc else "pp"][~m_art] == 0).all())
        self.assertGreater(int((~m_ppg).sum()), int((~m_art).sum()))                                  # PPG is masked more often than ART in the synthetic world

    def test_future_beats_cannot_change_the_tensor(self):
        # Verify future beats cannot change the tensor.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(80, seed=4)
            before = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            # Apply the same operation to each named field in this group.
            for (ds, cid), (df, waves) in w.cases.items():
                hi_max = int(rows.loc[(rows.dataset == ds) & (rows.case_id == cid), "beat_hi"].max()) if ((rows.dataset == ds) & (rows.case_id == cid)).any() else None
                if hi_max is None or hi_max >= len(df):
                    continue
                df2, waves2 = df.copy(), {k: v.copy() for k, v in waves.items()}
                rng = np.random.default_rng(9)
                for c in df2.columns:
                    if c not in ("beat_idx", "t_r", "valid_beat", "art_ok", "ppg_ok", "flags") and df2[c].dtype.kind == "f":
                        df2.loc[df2.index >= hi_max, c] = rng.normal(0, 99, len(df2) - hi_max)
                df2.loc[df2.index >= hi_max, "rr_sec"] = np.float32(0.2)
                for k in waves2:
                    waves2[k][hi_max:] = rng.normal(0, 50, (len(df2) - hi_max, 200)).astype(np.float16)
                d = P.case_dir(w.root, ds, cid)
                P._atomic_parquet(df2, d / "beats.parquet")
                P._atomic_parquet(P.waveform_table(df2, waves2, 200), d / "beat_waveforms.parquet")
            after = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            self.assertTrue(np.array_equal(before["x"], after["x"]))
            self.assertTrue(np.array_equal(before["context"], after["context"]))

    def test_normalization_does_not_depend_on_other_windows(self):
        # Verify normalization does not depend on other windows.
        rows = self.w.sample(40, seed=5)
        batch = S.assemble(self.w.root, rows, "eap")
        for i in (0, 17, 39):
            # Compare the result with the expected value or tolerance.
            alone = S.assemble(self.w.root, rows.iloc[[i]], "eap")
            self.assertTrue(np.array_equal(batch["x"][i], alone["x"][0]))
            self.assertTrue(np.array_equal(batch["context"][i], alone["context"][0]))

    def test_ea_is_eap_with_the_ppg_groups_zeroed(self):
        # Verify ea is eap with the PPG groups zeroed.
        rows = self.w.sample(48, seed=6)
        eap, ea = S.assemble(self.w.root, rows, "eap")["x"], S.assemble(self.w.root, rows, "ea")["x"]
        zeroed = eap.copy()
        zeroed[:, 2] = 0
        for k, name in enumerate(S.TIMING_BANDS):
            if name in S.PPG_GROUPS:
                zeroed[:, 4, :, k * 50:(k + 1) * 50] = 0
        for k, name in enumerate(S.MORPH_BANDS):
            if name in S.PPG_GROUPS:
                zeroed[:, 5, :, k * 25:(k + 1) * 25] = 0
        for k, name in enumerate(S.SQI_BANDS):
            if name in S.PPG_GROUPS:
                zeroed[:, 6, :, k * 50:(k + 1) * 50] = 0
        # the PPG-dependent masks also change which beats count, but the shared groups use the ART mask only, so they must match exactly
        self.assertTrue(np.array_equal(ea, zeroed))
        self.assertEqual(float(S.variant_mask("ea").sum()), len(S.GROUPS) - len(S.PPG_GROUPS))

    def test_ppg_free_variant_ignores_every_ppg_input(self):
        # Verify PPG free variant ignores every PPG input.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(60, seed=7)
            ea0, eap0 = S.assemble(w.root, rows, "ea", cache=S.CaseCache()), S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            # Apply the same operation to each named field in this group.
            for (ds, cid), (df, waves) in w.cases.items():
                df2, waves2 = df.copy(), {k: v.copy() for k, v in waves.items()}
                rng = np.random.default_rng(11)
                for c in ("ppg_amp", "ppg_ac_dc", "ppg_upstroke", "ptt_ecg_ppg", "ptt_art_ppg", "art_ppg_corr", "sqi_ppg", "sqi_cross"):
                    df2[c] = rng.normal(5, 3, len(df2)).astype(np.float32)
                df2["ppg_ok"] = rng.random(len(df2)) > 0.5
                waves2["ppg"] = rng.normal(0, 3, waves2["ppg"].shape).astype(np.float16)
                d = P.case_dir(w.root, ds, cid)
                P._atomic_parquet(df2, d / "beats.parquet")
                P._atomic_parquet(P.waveform_table(df2, waves2, 200), d / "beat_waveforms.parquet")
            ea1, eap1 = S.assemble(w.root, rows, "ea", cache=S.CaseCache()), S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            self.assertTrue(np.array_equal(ea0["x"], ea1["x"]))
            self.assertTrue(np.array_equal(ea0["context"], ea1["context"]))
            self.assertFalse(np.array_equal(eap0["x"], eap1["x"]))

    def test_deterministic_and_order_independent(self):
        # Verify deterministic and order independent.
        rows = self.w.sample(50, seed=8)
        a, b = S.assemble(self.w.root, rows, "eap"), S.assemble(self.w.root, rows, "eap", cache=S.CaseCache(maxsize=1))
        self.assertTrue(np.array_equal(a["x"], b["x"]))
        perm = np.random.default_rng(0).permutation(len(rows))
        c = S.assemble(self.w.root, rows.iloc[perm], "eap")
        self.assertTrue(np.array_equal(a["x"][perm], c["x"]))

    def test_split_and_dataset_are_never_inputs(self):
        # Verify split and dataset are never inputs.
        for name in S.CONTEXT_NAMES:
            self.assertFalse(any(s in name for s in ("split", "dataset", "patient", "case")), name)
        rows = self.w.sample(20, seed=9)
        # Compare the result with the expected value or tolerance.
        base = S.assemble(self.w.root, rows, "eap")
        changed = rows.assign(split_pooled="test", patient_key="x")
        self.assertTrue(np.array_equal(base["x"], S.assemble(self.w.root, changed, "eap")["x"]))
        with_flag = S.assemble(self.w.root, rows, "eap", include_dataset_flag=True)
        self.assertEqual(with_flag["context"].shape[1], len(S.CONTEXT_NAMES) + 1)
        self.assertEqual(base["context"].shape[1], len(S.CONTEXT_NAMES))

    def test_ppg_device_units_cannot_identify_the_dataset(self):
        # Verify PPG device units cannot identify the dataset.
        self.assertNotIn("mean_ppg_amp", S.CONTEXT_NAMES)
        self.assertNotIn("mean_ppg_ac_dc", S.CONTEXT_NAMES)
        with tempfile.TemporaryDirectory() as tmp:
            # Build the requested beat images, numerical context, and beat masks.
            w = World(tmp)
            rows = w.sample(50, seed=13)
            before = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            for (ds, cid), (df, waves) in w.cases.items():
                df2, waves2 = df.copy(), {k: v.copy() for k, v in waves.items()}
                for c in ("ppg_amp", "ppg_ac_dc"):
                    df2[c] = (df2[c] * 16).astype(np.float32)
                waves2["ppg"] = (waves2["ppg"].astype(np.float32) * 16).astype(np.float16)                 # a PPG recorded in 16x larger device units
                d = P.case_dir(w.root, ds, cid)
                P._atomic_parquet(df2, d / "beats.parquet")
                P._atomic_parquet(P.waveform_table(df2, waves2, 200), d / "beat_waveforms.parquet")
            after = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            self.assertLess(float(np.abs(before["x"] - after["x"]).max()), 1e-4)
            self.assertTrue(np.array_equal(before["context"], after["context"]))

    def test_apply_variant_equals_assembling_the_variant_directly(self):
        # Verify apply variant equals assembling the variant directly.
        rows = self.w.sample(60, seed=12)
        eap = S.assemble(self.w.root, rows, "eap")
        derived, direct = S.apply_variant(eap, "ea"), S.assemble(self.w.root, rows, "ea")
        for k in ("x", "context", "beat_mask", "variant_mask"):
            self.assertTrue(np.array_equal(derived[k], direct[k]), k)
        self.assertTrue(np.array_equal(S.apply_variant(eap, "eap")["x"], eap["x"]))

    def test_rejects_windows_without_60_beats(self):
        # Verify rejects windows without 60 beats.
        bad = self.w.index.loc[~self.w.index.has60].head(2)
        if len(bad):
            with self.assertRaises(ValueError):
                S.assemble(self.w.root, bad, "eap")

    def test_batch_iterator_covers_every_window_once_and_is_seeded(self):
        # Verify batch iterator covers every window once and is seeded.
        rows = self.w.usable
        ids = lambda seed: np.concatenate([b["window_id"] for b in S.window_batch_iter(self.w.root, rows, "eap", batch_size=32, seed=seed, cases_per_chunk=2)])
        first, again, other = ids(1), ids(1), ids(2)
        self.assertEqual(sorted(first.tolist()), sorted(rows.window_id.tolist()))
        self.assertTrue(np.array_equal(first, again))
        self.assertFalse(np.array_equal(first, other))

    def test_torch_dataset_yields_tensors_also_through_spawned_workers(self):
        # Verify torch dataset yields tensors also through spawned workers.
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")
        # Check x and Tensor against the expected result.
        ds = S.make_torch_dataset(self.w.root, self.w.usable.head(70), "ea", batch_size=32)
        batch = next(iter(ds))
        self.assertIsInstance(batch["x"], torch.Tensor)
        self.assertEqual(tuple(batch["x"].shape[1:]), (7, 60, 200))
        loader = torch.utils.data.DataLoader(S.make_torch_dataset(self.w.root, self.w.usable, "ea", batch_size=32), batch_size=None, num_workers=2)
        ids = np.concatenate([b["window_id"].numpy() for b in loader])
        self.assertEqual(sorted(ids.tolist()), sorted(self.w.usable.window_id.tolist()))                  # two workers: every window exactly once


class TestCompactCodec(unittest.TestCase):
    def test_round_trip_through_parquet_binary_columns(self):
        # Verify round trip through parquet binary columns.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            out = S.assemble(w.root, w.sample(24, seed=1), "eap")
            # Compare the result with the expected value or tolerance.
            frame = pd.DataFrame(S.encode_compact(out))
            path = Path(tmp) / "part.parquet"
            frame.to_parquet(path, index=False)
            back = S.decode_compact(pd.read_parquet(path).to_dict("list"))
            self.assertTrue(np.array_equal(back["x"][:, 3:], out["x"][:, 3:]))                  # planes 3-6 are exact broadcasts of the stored scalars
            self.assertLess(float(np.abs(back["x"][:, :3] - out["x"][:, :3]).max()), 5e-3)      # waveform planes are stored as float16
            for k in ("context", "beat_mask", "variant_mask", "window_id", "y_event", "aut65", "gray_zone"):
                self.assertTrue(np.array_equal(back[k], out[k]), k)
            self.assertLess(len(frame.waveforms_f16[0]) + len(frame.scalars_f32[0]), 90_000)


if __name__ == "__main__":
    unittest.main()
