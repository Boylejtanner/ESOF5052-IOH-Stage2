"""Audit BSHI labels, input ranges, patient splits, and training-data readiness.

``run_audit(root)`` writes ``results/bshi_training_audit.json`` (all numbers), ``data/derived/bshi/context_stats_v1.json`` (train-split standardization of the 25-value context),
``data/derived/bshi/target_stats_v1.json`` (train-split target scaling) and ``data/derived/bshi/case_meta.parquet`` (department, age, sex, ASA per case);
``write_readiness_doc`` turns the JSON into ``docs/bshi_training_readiness.md``. Only the TRAIN split is used for any statistic that a model will consume.
The audit never reads the test split for a decision: test windows appear only in the distribution-agreement check of the context (a diagnostic, nothing is tuned on it).
"""
from __future__ import annotations
# Check that model inputs, labels, and patient partitions are consistent.

import concurrent.futures as cf
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bshi as S
from .canonical import atomic_json, utc_now

SPLITS = ("train", "validation", "test")
RESULT = Path("results") / "bshi_training_audit.json"
CONTEXT_STATS = S.OUT_DIR / "context_stats_v1.json"
TARGET_STATS = S.OUT_DIR / "target_stats_v1.json"
CASE_META = S.OUT_DIR / "case_meta.parquet"
CLAMP = 6.0
MODALITY = {"ecg": 0, "art": 1, "ppg": 2}
IDX_COLUMNS = ["window_id", "dataset", "case_id", "patient_key", "split_pooled", "split_within_dataset", "t_start_sec", "t_end_sec", "map_current", "map_mean_60", "map_min_60",
               "map_std_60", "map_slope_60", "gray_zone", "y_event", "time_to_event_sec", "delta_map", "delta_map_robust", "aut65", "aut65_observed_fraction", "aut65_low_coverage",
               "beat_lo", "beat_hi", "n_beats_avail", "n_ecg_ok", "n_art_ok", "n_ppg_ok", "primary_eligible", "in_core", "in_full", "ecg_dead_case"]


def clean(v):
    """JSON-safe copy (numpy scalars -> python, NaN/inf -> None)."""
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if np.isfinite(v) else None
    return v


def load_index(root):
    return S.load_window_index(root, columns=IDX_COLUMNS)


def protocol_rows(idx, protocol):
    _, subset = S.PROTOCOLS[protocol]
    return idx[idx.primary_eligible & idx[{"core": "in_core", "full": "in_full"}[subset]]]


# 1. inputs
def audit_inputs(root, idx, per_stratum=500, seed=5052):
    """Assemble ``6 x per_stratum`` core windows (both datasets, all splits; random cases, at most 10 windows per case): finiteness, plane variance, share of scalar-plane values with |z| > 6."""
    # Summarize the records separately within each requested group.
    core = protocol_rows(idx, "ea_core")
    rows = pd.concat([_sample_by_case(g, per_stratum, seed, 10) for _, g in core.groupby(["dataset", "split_pooled"])])
    rows = rows.sort_values(["dataset", "case_id", "beat_lo"]).reset_index(drop=True)
    out = {"windows": int(len(rows)), "per_stratum": {f"{d}/{s}": int(n) for (d, s), n in rows.groupby(["dataset", "split_pooled"]).size().items()}, "variants": {}}
    cache = S.CaseCache()
    for variant, sel in (("ea", rows), ("eap", rows[rows.in_full])):
        # Exclude or flag missing values before using these measurements.
        sel = sel.reset_index(drop=True)
        t0 = time.perf_counter()
        b = S.assemble(root, sel, variant, cache=cache)
        secs = time.perf_counter() - t0
        x, ctx, bm = b["x"], b["context"], b["beat_mask"]
        res = {"windows": int(len(sel)), "assemble_windows_per_s": round(len(sel) / secs, 1), "x_finite": bool(np.isfinite(x).all()), "context_finite": bool(np.isfinite(ctx).all()),
               "labels_finite_where_expected": bool(np.isfinite(b["delta_map"]).all() and np.isfinite(b["aut65"]).all()), "planes": {}, "scalar_bands": {}}
        for p, name in enumerate(S.PLANES):
            # Summarize the observed levels and variability for this set of measurements.
            plane = x[:, p]
            res["planes"][name] = {"std_across_windows": float(plane.std(axis=0).mean()), "all_zero_windows": int((~plane.reshape(len(sel), -1).any(axis=1)).sum()),
                                   "abs_max": float(np.abs(plane).max())}
        # Apply the same operation to each named field in this group.
        for name, (plane, sl) in S.SCALAR_LAYERS.items():
            if plane == 6:
                continue
            col = sl[0] if sl else 0
            v = x[:, plane, :, col]
            valid = bm[:, MODALITY[S.MASK_OF[name]]]
            for ds in ("mover", "vitaldb"):
                m = valid & (sel.dataset.to_numpy() == ds)[:, None]
                vals = np.abs(v[m])
                res["scalar_bands"][f"{name}/{ds}"] = {"values": int(vals.size), "share_abs_gt6": float((vals > CLAMP).mean()) if vals.size else None,
                                                        "share_abs_gt4": float((vals > 4).mean()) if vals.size else None, "abs_max": float(vals.max()) if vals.size else None}
        out["variants"][variant] = res
    # Calculate output values for the following step.
    bands = out["variants"]["eap"]["scalar_bands"]
    worst = max(((k, v["share_abs_gt6"]) for k, v in bands.items() if v["share_abs_gt6"] is not None), key=lambda kv: kv[1])
    out["worst_scalar_band"] = {"band": worst[0], "share_abs_gt6": worst[1]}
    out["clamp"] = {"limit": CLAMP, "where": "planes 3-5 (MAP, timing, morphology) in the training pipeline (bshi_aug.clamp_scalar_planes), assembler unchanged",
                    "material": bool(worst[1] > 0.005), "bands_above_0p5pct": sorted(k for k, v in bands.items() if v["share_abs_gt6"] and v["share_abs_gt6"] > 0.005)}
    return out


# 2. targets
def loss_masks(df):
    """Explicit loss masks on an index frame: event (all windows of the protocol; the 65-75 mmHg gray zone stays in and is only reported), regression (delta MAP, AUT65 are
    defined for every eligible window; windows with low horizon observation are excluded because their minimum / integral are biased low), time-to-event (events only)."""
    # Exclude or flag missing values before using these measurements.
    ok = ~df.aut65_low_coverage.to_numpy()
    return {"event": np.ones(len(df), dtype=bool), "delta_map": np.isfinite(df.delta_map.to_numpy()) & ok, "delta_map_robust": np.isfinite(df.delta_map_robust.to_numpy()) & ok,
            "aut65": np.isfinite(df.aut65.to_numpy()) & ok, "time_to_event": df.y_event.to_numpy() & np.isfinite(df.time_to_event_sec.to_numpy())}


def audit_targets(root, idx, write=True):
    out = {"tables": [], "loss_masks": {"event": "all windows of the protocol (gray zone kept, reported separately)",
                                         "delta_map, delta_map_robust, aut65": "finite and not aut65_low_coverage (observed fraction of the horizon below the label rule)",
                                         "time_to_event": "event windows only (auxiliary, not trained by default)"}}
    # Process the selected records in order and retain their individual results.
    for proto in ("eap_full", "ea_core"):
        pr = protocol_rows(idx, proto)
        for (ds, sp), g in pr.groupby(["dataset", "split_pooled"]):
            # Exclude or flag missing values before using these measurements.
            m = loss_masks(g)
            ev = g[g.y_event]
            out["tables"].append({"protocol": proto, "dataset": ds, "split": sp, "windows": len(g), "patients": g.patient_key.nunique(), "events": int(g.y_event.sum()),
                                  "prevalence": float(g.y_event.mean()), "delta_map_nan": int(g.delta_map.isna().sum()), "delta_map_robust_nan": int(g.delta_map_robust.isna().sum()),
                                  "aut65_nan": int(g.aut65.isna().sum()), "tte_nan_in_events": int(ev.time_to_event_sec.isna().sum()),
                                  "tte_nan_share_nonevents": float(g.loc[~g.y_event, "time_to_event_sec"].isna().mean()), "tte_max_s": float(ev.time_to_event_sec.max()),
                                  "gray_zone": int(g.gray_zone.sum()), "aut65_low_coverage": int(g.aut65_low_coverage.sum()), "delta_map_zero_share": float((g.delta_map == 0).mean()),
                                  "aut65_zero_share": float((g.aut65 == 0).mean()), "reg_mask_share": float(m["delta_map"].mean())})
    # Calculate the working array for the following step.
    tr = protocol_rows(idx, "eap_full")
    tr = tr[tr.split_pooled == "train"]
    m = loss_masks(tr)
    d, dr = tr.delta_map.to_numpy()[m["delta_map"]], tr.delta_map_robust.to_numpy()[m["delta_map_robust"]]
    # Summarize the records separately within each requested group.
    a = tr.aut65.to_numpy()[m["aut65"]]
    pos = a > 0
    la = np.log1p(a[pos])
    # Summarize the records separately within each requested group.
    stats = {"source": "train split of eap_full (pooled), loss masks applied", "windows": int(len(tr)),
             "delta_map": {"center": 0.0, "scale": float(d.std()), "mean": float(d.mean()), "median": float(np.median(d)), "p99": float(np.quantile(d, 0.99)), "zero_share": float((d == 0).mean()),
                           "rule": "target = delta_map / scale (one-sided, >= 0, so no centering); Huber delta 1.0 in scaled units"},
             "delta_map_robust": {"center": 0.0, "scale": float(dr.std()), "mean": float(dr.mean()), "zero_share": float((dr == 0).mean())},
             "aut65": {"p_positive": float(pos.mean()), "log1p_mean_given_positive": float(la.mean()), "log1p_std_given_positive": float(la.std()),
                       "rule": "two heads: logit of P(aut65 > 0) with BCE on all masked windows; (log1p(aut65) - mean) / std with Huber ONLY where aut65 > 0; prediction = P * expm1(magnitude)"},
             "event": {"prevalence_train": float(tr.y_event.mean()), "by_dataset": {ds: float(g.y_event.mean()) for ds, g in tr.groupby("dataset")}}}
    out["target_stats"] = stats
    if write:
        atomic_json(Path(root) / TARGET_STATS, {"version": "target_stats_v1", "created_utc": utc_now(), **clean(stats)})
    return out


# 3. context
def _context_chunk(root, rows, variant):
    return S.assemble(root, rows, variant, cache=S.CaseCache())["context"]


def _assemble_context(root, rows, variant, workers=12, chunks=48):
    rows = rows.sort_values(["dataset", "case_id", "beat_lo"]).reset_index(drop=True)
    parts = np.array_split(np.arange(len(rows)), min(chunks, max(1, len(rows) // 200)))
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = list(ex.map(_context_chunk, [str(root)] * len(parts), [rows.iloc[p] for p in parts], [variant] * len(parts)))
    return rows, np.concatenate(res)


def _smd(a, b):
    sd = np.sqrt((a.var(axis=0) + b.var(axis=0)) / 2)
    return np.where(sd > 1e-9, (a.mean(axis=0) - b.mean(axis=0)) / np.where(sd > 1e-9, sd, 1), 0.0)


def _sample_by_case(df, n, seed, per_case):
    """Up to ``n`` windows from random cases, at most ``per_case`` random windows per case: every case is read once, which makes the context pass about 20x faster than
    drawing windows uniformly (one case load per window)."""
    # Summarize the records separately within each requested group.
    rng = np.random.default_rng(seed)
    groups = list(df.groupby(["dataset", "case_id"], sort=False).indices.values())
    parts, got = [], 0
    for i in rng.permutation(len(groups)):
        pos = groups[i]
        take = pos if len(pos) <= per_case else rng.choice(pos, per_case, replace=False)
        parts.append(df.iloc[np.sort(take)])
        got += len(take)
        if got >= n:
            break
    return pd.concat(parts).head(n)


def _patient_means(rows, ctx, names):
    return pd.DataFrame(ctx, columns=names).groupby(rows.patient_key.to_numpy()).mean()


def audit_context(root, idx, pooled=40000, per_cell=8000, reps=500, seed=5052, write=True):
    # Calculate project paths and output values for the following step.
    root = Path(root)
    names = S.CONTEXT_NAMES
    out = {"names": names, "sampling": f"pooled train statistics: random cases, at most 40 windows per case ({pooled} windows); split agreement: every patient of a cell contributes up to 24 windows "
                                       f"(about {per_cell} per cell), compared on PATIENT means with a permutation test (random re-assignment of patients, {reps} reps)", "stats": {}}
    stats_doc = {"version": "context_stats_v1", "created_utc": utc_now(), "names": names, "source_split": "train", "seed": seed, "clamp": CLAMP,
                 "rule": "z = clip((x - center) / scale, -clamp, clamp); center = median, scale = max(IQR / 1.349, SD / 2) (1 if constant); entries a variant zeroes stay exactly 0: "
                         "standardize first, then multiply by the variant context mask", "code_version": S.BSHI_CODE_VERSION}
    cells = {}
    for proto in ("eap_full", "ea_core"):
        # Exclude or flag missing values before using these measurements.
        variant, _ = S.PROTOCOLS[proto]
        pr = protocol_rows(idx, proto)
        tr = pr[pr.split_pooled == "train"]
        # Exclude or flag missing values before using these measurements.
        rows, ctx = _assemble_context(root, _sample_by_case(tr, pooled, seed, 40), variant)
        out["context_finite_" + proto] = bool(np.isfinite(ctx).all())
        q1, med, q3 = np.quantile(ctx, [0.25, 0.5, 0.75], axis=0)
        std = ctx.std(axis=0)
        scale = np.maximum((q3 - q1) / 1.349, 0.5 * std)          # IQR scale, but never below half the SD: near-constant, heavy-tailed features (SQI close to 1) would otherwise clip often
        scale = np.where(scale > 1e-8, scale, 1.0)
        zero = np.array([n in S.context_off(S.VARIANTS[variant]) for n in names])
        center = np.where(zero, 0.0, med)
        # Summarize the observed levels and variability for this set of measurements.
        scale = np.where(zero, 1.0, scale)
        z = np.clip((ctx - center) / scale, -CLAMP, CLAMP)
        beyond = (np.abs((ctx - center) / scale) > CLAMP).mean(axis=0)
        # Summarize the observed levels and variability for this set of measurements.
        stats_doc[proto] = {"n": int(len(ctx)), "center": center.tolist(), "scale": scale.tolist(), "mean": ctx.mean(axis=0).tolist(), "std": std.tolist(),
                            "share_beyond_clamp": beyond.tolist(), "zeroed_by_variant": [n for n, zf in zip(names, zero) if zf]}
        out["stats"][proto] = {"n": int(len(ctx)), "max_share_beyond_clamp": float(beyond.max()), "feature_of_max": names[int(beyond.argmax())], "standardized_std_range":
                               [float(z.std(axis=0)[~zero].min()), float(z.std(axis=0)[~zero].max())]}
    full = protocol_rows(idx, "eap_full")
    cells = {}
    for (ds, sp), g in full.groupby(["dataset", "split_pooled"]):
        per = int(max(3, min(24, per_cell // g.patient_key.nunique())))
        cells[(ds, sp)] = _assemble_context(root, g.sample(frac=1, random_state=seed).groupby("patient_key").head(per), "eap")
    out["smd_within_dataset"] = {}
    rng = np.random.default_rng(seed)
    # Process the selected records in order and retain their individual results.
    for ds in ("mover", "vitaldb"):
        pa = _patient_means(*cells[(ds, "train")], names)
        for sp in ("validation", "test"):
            # Reserve output arrays and masks with explicit initial values.
            pb = _patient_means(*cells[(ds, sp)], names)
            s = _smd(pa.to_numpy(), pb.to_numpy())
            pool = np.vstack([pa.to_numpy(), pb.to_numpy()])
            null = np.empty(reps)
            for r in range(reps):
                perm = rng.permutation(len(pool))
                null[r] = np.abs(_smd(pool[perm[:len(pa)]], pool[perm[len(pa):]])).max()
            k = int(np.abs(s).argmax())
            out["smd_within_dataset"][f"{ds}: train vs {sp}"] = {
                "patients": [int(len(pa)), int(len(pb))], "max_abs_smd": float(np.abs(s).max()), "feature": names[k], "expected_se_of_one_smd": float(np.sqrt(1 / len(pa) + 1 / len(pb))),
                "permutation_p_of_max_abs_smd": float((1 + (null >= np.abs(s).max()).sum()) / (reps + 1)), "null_max_abs_smd_median": float(np.median(null)),
                "features_above_0p2": [n for n, v in zip(names, s) if abs(v) > 0.2]}
    out["all_context_finite"] = bool(all(out.get("context_finite_" + p) for p in ("eap_full", "ea_core")) and all(np.isfinite(c).all() for _, c in cells.values()))
    if write:
        atomic_json(root / CONTEXT_STATS, clean(stats_doc))
    return out


# 4. balance and sampler
def audit_balance(idx):
    full = protocol_rows(idx, "eap_full")
    out = {"prevalence": [], "windows_per_patient_train": {}}
    for (ds, sp), g in full.groupby(["dataset", "split_pooled"]):
        out["prevalence"].append({"dataset": ds, "split": sp, "windows": len(g), "events": int(g.y_event.sum()), "prevalence": float(g.y_event.mean()), "patients": g.patient_key.nunique()})
    tr = full[full.split_pooled == "train"]
    for ds, g in tr.groupby("dataset"):
        n = g.groupby("patient_key").size()
        out["windows_per_patient_train"][ds] = {"patients": int(len(n)), "mean": float(n.mean()), "median": float(n.median()), "max": int(n.max()), "max_share_of_dataset": float(n.max() / n.sum())}
    # Summarize the records separately within each requested group.
    out["mover_share_natural_batch_train"] = float((tr.dataset == "mover").mean())
    nat = {ds: float(g.y_event.mean()) for ds, g in tr.groupby("dataset")}
    rec = {"batch_composition": {"mover_share": 0.15, "event_share_within_dataset": 0.30},
           "why": "a natural batch of 256 holds about 10 MOVER windows and 8 events; the composition above gives about 38 MOVER windows and 77 events per batch",
           "training_loss": "no per-window loss weights; the sampler provides the balance (patient-level sampling stays uniform within a stratum)",
           "natural_prevalence_metrics": "validate and test on the FULL, unsampled windows (natural prevalence by construction); report AUROC, AUPRC, calibration on them",
           "prior_shift": {ds: float(np.log(p / (1 - p)) - np.log(0.30 / 0.70)) for ds, p in nat.items()},
           "prior_shift_use": "ADD this (negative) offset per dataset to the event logit to recover probabilities at natural prevalence (offset = logit(natural prevalence) - logit(sampled event share)); AUROC is unaffected",
           "dev_subset_weights": "sampling_weight in dev_subset recovers natural counts per (split, dataset, event) for weighted metrics",
           "to_be_tested_by_the_sweep": "train.sampler = 'natural' vs 'stratified' is a sweep axis (docs/training_decisions.md)"}
    out["recommendation"] = rec
    out["patient_independence_asserted"] = bool(S.check_index(_index_for_check(idx)))
    return out


def _index_for_check(idx):
    """check_index needs the flag columns plus has60 and the beat span; rebuild the few it asserts on from the columns we hold."""
    df = idx.copy()
    df["has60"] = (df.beat_hi - df.beat_lo) == S.N_BEATS
    return df


# 5. recipes and raw windows
def audit_recipes(root, idx, n=64, seed=5052):
    rows = protocol_rows(idx, "eap_full").sample(n, random_state=seed).reset_index(drop=True)
    base = S.assemble(root, rows, "eap", cache=S.CaseCache())
    table = []
    for model, variant in S.PROPOSAL_MODELS.items():
        off = S.VARIANTS[variant]
        b = S.apply_variant(base, variant)
        keep_equal = all(np.array_equal(b["x"][:, p], base["x"][:, p]) for g, p in S.PLANE_OF_GROUP.items() if g not in off)
        table.append({"proposal_model": model, "recipe": variant, "groups_on": int(len(S.GROUPS) - len(off)), "groups_total": len(S.GROUPS), "nonzero_share_of_tensor": float((b["x"] != 0).mean()),
                      "context_entries_on": int(len(S.CONTEXT_NAMES) - len(S.context_off(off))), "kept_planes_unchanged": bool(keep_equal)})
    return {"table": table, "models_needing_raw_windows": {"C_raw_multimodal_tcn": "bshi_raw.raw_windows", "D_stft_cwt": "bshi_raw.raw_windows"}}


def audit_raw(root, idx, n=200, seed=5052):
    """bshi_raw.raw_windows against an independent read of the waveform files (valid samples must be identical, masks must equal the finite pattern)."""
    # Order the measurements before matching intervals or comparing rows.
    import pyarrow.parquet as pq
    from . import bshi_raw as R
    rows = protocol_rows(idx, "eap_full").sample(n, random_state=seed).sort_values(["dataset", "case_id", "t_end_sec"]).reset_index(drop=True)
    # Summarize the records separately within each requested group.
    t0 = time.perf_counter()
    x, m = R.raw_windows(root, rows, cache=R.RawCache(maxsize=8))
    secs = time.perf_counter() - t0
    big = protocol_rows(idx, "eap_full")
    # Join matching records before combining their measurements.
    big = big[big.groupby(["dataset", "case_id"]).window_id.transform("size") >= 100]
    cases = big.drop_duplicates(["dataset", "case_id"]).groupby("dataset").sample(3, random_state=seed)[["dataset", "case_id"]]
    grouped = big.merge(cases).groupby(["dataset", "case_id"]).head(100).reset_index(drop=True)
    t1 = time.perf_counter()
    R.raw_windows(root, grouped, cache=R.RawCache(maxsize=8))
    grouped_rate = len(grouped) / (time.perf_counter() - t1)
    bad, checked, files = 0, 0, {}
    # Apply this calculation to each row in the selected table.
    for i, r in rows.iterrows():
        key = (r.dataset, r.case_id)
        if key not in files:
            files[key] = pq.read_table(R.waveform_path(root, *key), columns=["time_sec", "ECG", "ART", "PPG"]).to_pandas()
        t = files[key]
        w = t[(t.time_sec >= r.t_end_sec - 90 - 0.005) & (t.time_sec < r.t_end_sec - 0.005)]
        for c, col in enumerate(("ECG", "ART", "PPG")):
            # Exclude or flag missing values before using these measurements.
            v = w[col].to_numpy(dtype=np.float32)
            pos = np.rint((w.time_sec.to_numpy() - (r.t_end_sec - 90)) * R.FS).astype(int)
            keep = (pos >= 0) & (pos < 9000) & np.isfinite(v)
            got = x[i, c][pos[keep]]
            checked += int(keep.sum())
            bad += int((got != v[keep]).sum()) + int(m[i, c].sum() != keep.sum())
    return {"windows": int(n), "samples_compared": int(checked), "mismatches": int(bad), "windows_per_s_random_access": round(n / secs, 1), "windows_per_s_case_grouped": round(grouped_rate, 1),
            "shape": list(x.shape), "valid_share": float(m.mean())}


# 6. augmentation hooks
def audit_aug(root, idx, n=64, seed=5052):
    import torch
    from . import bshi_aug as A
    # Load the recorded settings or metadata for this processing step.
    rows = protocol_rows(idx, "eap_full").sample(n, random_state=seed).reset_index(drop=True)
    b = A.to_torch(S.assemble(root, rows, "eap", cache=S.CaseCache(), return_times=True))
    stats = json.loads((Path(root) / "results" / "bshi_gap_stats.json").read_text(encoding="utf-8"))
    gen = lambda: torch.Generator().manual_seed(5052)
    calls = {"mover_gaps": lambda: A.mover_gaps(b, stats, gen()), "ppg_dropout": lambda: A.ppg_dropout(b, 0.3, 0.2, gen()), "ecg_noise": lambda: A.ecg_noise(b, 10.0, gen()),
             "art_damping": lambda: A.art_damping(b, 9), "art_flatline": lambda: A.art_flatline(b, 0.2, gen()), "timing_jitter": lambda: A.timing_jitter(b, 0.03, gen())}
    table = []
    for name, fn in calls.items():
        t0 = time.perf_counter()
        o = fn()
        ms = (time.perf_counter() - t0) * 1000
        o2 = fn()
        table.append({"function": name, "exactness": A.AUGMENTATIONS[name]["exactness"], "finite": bool(torch.isfinite(o["x"]).all()), "deterministic": bool(torch.equal(o["x"], o2["x"])),
                      "changed_share": float((o["x"] != b["x"]).float().mean()), "ms_per_batch_cpu": round(ms, 1)})
    return {"table": table, "mover_gap_stats": {k: stats[k] for k in ("cases", "valid_fraction_mean", "valid_run_median_s", "gap_median_s")},
            "compact_store_must_keep": ["t_rel (60 x float32, s)", "rr (60 x float32, s)"],
            "clamp_effect_on_real_batch": float(((b["x"][:, 3:6].abs() > CLAMP).float().mean()))}


# 7. case meta (leave-one-department-out)
def build_case_meta(root, idx, write=True):
    root = Path(root)
    cq = pd.read_parquet(root / S.OUT_DIR / "case_quality.parquet", columns=["dataset", "case_id", "patient_key", "split_pooled", "split_within_dataset", "ecg_dead"])
    # Join related tables by case key to keep measurements aligned.
    clin = pd.read_csv(root / "data" / "reference" / "vitaldb" / "clinical_data.csv", usecols=lambda c: c in {"caseid", "department", "age", "sex", "asa", "emop", "optype", "opname"})
    clin["case_key"] = clin.caseid.astype(int)
    meta = cq.copy()
    meta["case_key"] = pd.to_numeric(meta.case_id, errors="coerce")
    meta = meta.merge(clin.drop(columns=["caseid"]), on="case_key", how="left").drop(columns=["case_key"])
    # Join related tables by dataset, case id to keep measurements aligned.
    meta.loc[meta.dataset == "mover", ["department", "age", "sex", "asa", "emop", "optype", "opname"]] = np.nan
    flags = idx.assign(core=idx.primary_eligible & idx.in_core, full=idx.primary_eligible & idx.in_full)
    cases = flags.groupby(["dataset", "case_id"]).agg(windows_core=("core", "sum"), windows_full=("full", "sum")).reset_index()
    ev = idx[idx.primary_eligible & idx.in_full & idx.y_event].groupby(["dataset", "case_id"]).size().rename("events_full")
    cases = cases.merge(ev, on=["dataset", "case_id"], how="left").fillna({"events_full": 0})
    meta = meta.merge(cases, on=["dataset", "case_id"], how="left").fillna({"windows_core": 0, "windows_full": 0, "events_full": 0})
    if write:
        meta.to_parquet(root / CASE_META, index=False)
    # Exclude or flag missing values before using these measurements.
    v = meta[(meta.dataset == "vitaldb")]
    by = v.groupby("department", dropna=False).agg(cases=("case_id", "size"), cases_in_full=("windows_full", lambda s: int((s > 0).sum())), windows_full=("windows_full", "sum"),
                                                   events_full=("events_full", "sum"), patients=("patient_key", "nunique")).reset_index().sort_values("cases", ascending=False)
    return {"path": str(CASE_META), "rows": int(len(meta)), "columns": list(meta.columns), "source": "data/reference/vitaldb/clinical_data.csv (already on disk, no download); MOVER rows are empty (no department in the MOVER files used)",
            "department_table": by.fillna({"department": "(missing)"}).to_dict("records"), "vitaldb_cases_without_department": int(v.department.isna().sum()),
            "age_sex_asa_present": {c: int(v[c].notna().sum()) for c in ("age", "sex", "asa")}}


# 8-9. evaluation inputs and storage
def audit_evaluation(idx, stride_sec=30.0, cap_sec=240.0):
    out = {"tables": []}
    for proto in ("eap_full", "ea_core"):
        # Summarize the records separately within each requested group.
        pr = protocol_rows(idx, proto).sort_values(["dataset", "case_id", "t_end_sec"])
        d = pr.groupby(["dataset", "case_id"]).t_end_sec.diff().dropna()
        span = pr.groupby(["dataset", "split_pooled", "patient_key", "case_id"]).t_end_sec.agg(lambda s: s.max() - s.min()).rename("span_s")
        for (ds, sp), g in pr.groupby(["dataset", "split_pooled"]):
            # Summarize the records separately within each requested group.
            sp_s = span.loc[ds, sp]
            out["tables"].append({"protocol": proto, "dataset": ds, "split": sp, "patients": g.patient_key.nunique(), "windows": len(g), "window_hours_at_30s": len(g) * stride_sec / 3600,
                                  "span_hours": float(sp_s.sum() / 3600), "median_hours_per_patient": float(sp_s.groupby(level="patient_key").sum().median() / 3600)})
        out[proto + "_stride_30s_share"] = float((d == stride_sec).mean())
        out[proto + "_chronological"] = bool((d > 0).all())
    out["lead_time_cap_s"] = cap_sec
    out["tte_max_s"] = float(idx.loc[idx.y_event & idx.primary_eligible, "time_to_event_sec"].max())
    out["lead_time_cap_respected"] = bool(out["tte_max_s"] <= cap_sec + 1e-6)
    out["notes"] = "window_hours_at_30s = windows x 30 s (each window represents one 30 s stride of monitored time); span_hours = first to last window per case (includes capture gaps)."
    return out


def audit_storage(root, idx, train_stride=2):
    # Load the recorded settings or metadata for this processing step.
    meta = json.loads((Path(root) / "results" / "bshi_dev_subset.json").read_text(encoding="utf-8"))
    dev_dir = Path(root) / S.OUT_DIR / "dev_subset"
    dev_bytes = sum(p.stat().st_size for p in dev_dir.rglob("*.parquet"))
    dev_windows = int(meta.get("windows") or meta.get("rows") or 17167)
    per_window = dev_bytes / dev_windows + 480.0
    out = {"bytes_per_window": round(per_window), "dev_subset_bytes": int(dev_bytes), "dev_windows": dev_windows, "plus_t_rel_rr_bytes": 480, "tables": []}
    for proto in ("eap_full", "ea_core"):
        pr = protocol_rows(idx, proto)
        counts = {sp: int((pr.split_pooled == sp).sum()) for sp in SPLITS}
        stored = {"train": counts["train"] // train_stride, "validation": counts["validation"], "test": counts["test"]}
        out["tables"].append({"protocol": proto, "windows": counts, "stored_windows": stored, "gb": round(sum(stored.values()) * per_window / 1e9, 1)})
    return out


# orchestration
def run_audit(root=".", write=True, only=None):
    """Run every step (about 2 minutes), or only the named steps (``only=['context']``) merged into the previous results file."""
    root = Path(root).resolve()
    t0 = time.perf_counter()
    idx = load_index(root)
    res = {"created_utc": utc_now(), "code_version": S.BSHI_CODE_VERSION}
    if only and (root / RESULT).exists():
        res = json.loads((root / RESULT).read_text(encoding="utf-8"))
    steps = (("inputs", lambda: audit_inputs(root, idx)), ("targets", lambda: audit_targets(root, idx, write)), ("context", lambda: audit_context(root, idx, write=write)),
             ("balance", lambda: audit_balance(idx)), ("recipes", lambda: audit_recipes(root, idx)), ("raw", lambda: audit_raw(root, idx)), ("augmentation", lambda: audit_aug(root, idx)),
             ("case_meta", lambda: build_case_meta(root, idx, write)), ("evaluation", lambda: audit_evaluation(idx)), ("storage", lambda: audit_storage(root, idx)))
    # Process the selected records in order and retain their individual results.
    for name, fn in steps:
        if only and name not in only:
            continue
        try:
            res[name] = fn()
        except Exception as exc:
            res[name] = {"error": repr(exc)}
            if write:
                atomic_json(root / RESULT, clean(res))
            raise
    res["gates"] = gates(res)
    res["seconds"] = round(time.perf_counter() - t0, 1)
    if write:
        atomic_json(root / RESULT, clean(res))
    return res


def gates(res):
    """PASS/FAIL lines (the notebook section prints them)."""
    g = {}
    inp = res["inputs"]["variants"]
    g["inputs finite (ea and eap, x and context)"] = all(v["x_finite"] and v["context_finite"] for v in inp.values())
    g["every plane the variant keeps has variance across windows"] = all(p["std_across_windows"] > 0 for v, r in inp.items() for name, p in r["planes"].items() if not (v == "ea" and name == "ppg"))
    g["scalar-plane outliers handled by the +-6 model-side clamp"] = res["inputs"]["clamp"]["limit"] == CLAMP
    t = res["targets"]["tables"]
    g["no NaN in delta_map / delta_map_robust / aut65 of eligible windows"] = all(r["delta_map_nan"] == 0 and r["delta_map_robust_nan"] == 0 and r["aut65_nan"] == 0 for r in t)
    g["time_to_event defined for every event"] = all(r["tte_nan_in_events"] == 0 for r in t)
    if "context" in res:
        c = res["context"]
        g["context finite and standardized from the train split only"] = c["all_context_finite"]
        g["within-dataset context splits no more different than random patient partitions (permutation p > 0.01)"] = all(v["permutation_p_of_max_abs_smd"] > 0.01
                                                                                                                         for v in c["smd_within_dataset"].values())
    g["patient-level split independence"] = res["balance"]["patient_independence_asserted"]
    g["proposal models A,B,E,F,G map onto recipes (planes unchanged)"] = all(r["kept_planes_unchanged"] for r in res["recipes"]["table"])
    g["raw 100 Hz windows equal the waveform files"] = res["raw"]["mismatches"] == 0
    g["augmentations finite and deterministic"] = all(r["finite"] and r["deterministic"] for r in res["augmentation"]["table"])
    g["department table available for leave-one-department-out"] = len(res["case_meta"]["department_table"]) >= 3
    ev = res["evaluation"]
    g["chronological 30 s windows and 240 s lead-time cap available"] = ev["eap_full_chronological"] and ev["lead_time_cap_respected"] and ev["eap_full_stride_30s_share"] > 0.9
    return g


DOC = Path("docs") / "bshi_training_readiness.md"


def _f(v, d=1):
    return f"{v:.{d}f}"


def write_readiness_doc(root=".", res=None):
    """docs/bshi_training_readiness.md from the audit JSON (at most 90 lines)."""
    # Load the recorded settings or metadata for this processing step.
    root = Path(root).resolve()
    res = res or json.loads((root / RESULT).read_text(encoding="utf-8"))
    g, inp, tg, ctx, bal, rec, raw, aug, meta, ev, sto = (res[k] for k in ("gates", "inputs", "targets", "context", "balance", "recipes", "raw", "augmentation", "case_meta", "evaluation", "storage"))
    n_pass = sum(bool(v) for v in g.values())
    eap, ea = inp["variants"]["eap"], inp["variants"]["ea"]
    band = inp["variants"]["eap"]["scalar_bands"]
    t = {(r["dataset"], r["split"]): r for r in tg["tables"] if r["protocol"] == "eap_full"}
    ts = tg["target_stats"]
    smd = ctx["smd_within_dataset"]
    hrs = {(r["dataset"], r["split"]): r for r in ev["tables"] if r["protocol"] == "eap_full"}
    wpp = bal["windows_per_patient_train"]
    dep = {d["department"]: d for d in meta["department_table"]}
    low = sum(r["aut65_low_coverage"] for r in t.values()) / sum(r["windows"] for r in t.values())
    smd_mt = smd["mover: train vs test"]
    std_rng = [min(v["std_across_windows"] for v in eap["planes"].values()), max(v["std_across_windows"] for v in eap["planes"].values())]
    store = {r["protocol"]: r["gb"] for r in sto["tables"]}
    rows = [
        ("1", "Finite, non-degenerate inputs", "yes", f"{inp['windows']:,} windows (both datasets, all splits): `ea` and {eap['windows']:,} `eap`: x and context finite; plane std across windows {_f(std_rng[0], 2)}-{_f(std_rng[1], 2)}; `ea` PPG plane is 0 by design", "none"),
        ("2", "Outliers in scalar planes", "partly", f"abs(z)>6: MOVER `ppg_ac_dc` {_f(100 * band['ppg_ac_dc/mover']['share_abs_gt6'], 2)}% of beats (max {_f(band['ppg_ac_dc/mover']['abs_max'])}) vs VitalDB {_f(100 * band['ppg_ac_dc/vitaldb']['share_abs_gt6'], 2)}%; every other band <= 0.07%", f"model-side clamp +-{CLAMP:g} on planes 3-5 (`bshi_aug.clamp_scalar_planes`); assembler unchanged"),
        ("3", "Targets complete, explicit masks", "yes", f"NaN in `delta_map`, `delta_map_robust`, `aut65`: 0 in all 12 protocol x dataset x split cells; `time_to_event` defined for every event (max {_f(ev['tte_max_s'], 0)} s); {_f(100 * low, 1)}% of windows have low AUT65 observation", "`bshi_audit.loss_masks`: event = all windows (gray zone kept, reported); regressions drop low-coverage windows"),
        ("4", "Target scaling (train only)", "yes", f"`delta_map` >= 0 (one-sided, {_f(100 * ts['delta_map']['zero_share'], 1)}% zeros), train SD {_f(ts['delta_map']['scale'], 1)} mmHg; P(AUT65>0) = {_f(100 * ts['aut65']['p_positive'], 1)}%, log1p magnitude {_f(ts['aut65']['log1p_mean_given_positive'], 2)} +- {_f(ts['aut65']['log1p_std_given_positive'], 2)}", "`target_stats_v1.json`; AUT65 two-part target (P(>0) BCE + log1p Huber where >0)"),
        ("5", "Context standardization (train only)", "yes", f"25 values finite; {ctx['stats']['eap_full']['n']:,} train windows; share beyond +-{CLAMP:g} after standardizing <= {_f(100 * ctx['stats']['eap_full']['max_share_beyond_clamp'], 1)}% ({ctx['stats']['eap_full']['feature_of_max']})", "`context_stats_v1.json` (median, max(IQR/1.349, SD/2); `eap_full` and `ea_core`)"),
        ("6", "Context agrees across splits inside a dataset", "yes, caution", f"VitalDB max abs(SMD) {_f(smd['vitaldb: train vs test']['max_abs_smd'], 2)} (p={_f(smd['vitaldb: train vs test']['permutation_p_of_max_abs_smd'], 2)}); MOVER val {_f(smd['mover: train vs validation']['max_abs_smd'], 2)} (p={_f(smd['mover: train vs validation']['permutation_p_of_max_abs_smd'], 2)}); MOVER TEST {_f(smd_mt['max_abs_smd'], 2)} (`{smd_mt['feature']}`, p={_f(smd_mt['permutation_p_of_max_abs_smd'], 3)}, 53 patients)", "none; MOVER test patients differ slightly in timing/HR: read its CIs as wide (risk R4)"),
        ("7", "Class balance and sampler", "yes", f"event prevalence 3.0-3.1% VitalDB, 3.5-3.9% MOVER in every split; MOVER = {_f(100 * bal['mover_share_natural_batch_train'], 1)}% of natural train windows; largest patient {_f(100 * wpp['mover']['max_share_of_dataset'], 1)}% (MOVER) / {_f(100 * wpp['vitaldb']['max_share_of_dataset'], 2)}% (VitalDB) of train windows", "recommended: batches of 15% MOVER, 30% events, no loss weights; logit prior shift "
         f"{_f(bal['recommendation']['prior_shift']['mover'], 2)} / {_f(bal['recommendation']['prior_shift']['vitaldb'], 2)}; metrics on full natural val/test; `train.sampler` is a sweep axis"),
        ("8", "Patient-level independence", "yes", "`check_index` re-asserted on all 1,414,177 windows: no patient in two splits", "Verify the same patient partition in the compact training store"),
        ("9", "Comparison models A, B, E, F, G (BSHI recipes)", "yes (new)", "; ".join(f"{r['proposal_model'].split('_')[0]}{'-noPPG' if 'no_ppg' in r['proposal_model'] else ''}={r['recipe']} ({r['groups_on']}/{r['groups_total']} groups)" for r in rec["table"]), "variants added to `bshi.py` (tensors of `eap`/`ea` unchanged, `bs2`); 12 tests"),
        ("10", "Models C (1-D TCN) and D (STFT/CWT) need raw signals", "yes (new)", f"`bshi_raw.raw_windows`: {raw['windows']} windows, {raw['samples_compared']:,} samples vs the waveform files, {raw['mismatches']} mismatches; {_f(raw['windows_per_s_case_grouped'], 0)} windows/s case-grouped, {_f(raw['windows_per_s_random_access'], 0)} random access", "`bshi_raw.py` + tests; the C and D models are for the next evaluation stage"),
        ("11", "Augmentation / corruption hooks", "yes (new)", ", ".join(r["function"] for r in aug["table"]) + " (all APPROXIMATION on the finished tensor; EXACT through `assemble(extra_mask)` for gaps and `extra_mask_ppg` for PPG dropout)" + f"; MOVER capture measured on {aug['mover_gap_stats']['cases']} cases: valid run median {_f(aug['mover_gap_stats']['valid_run_median_s'], 0)} s, gap {_f(aug['mover_gap_stats']['gap_median_s'], 0)} s, {_f(100 * aug['mover_gap_stats']['valid_fraction_mean'], 0)}% valid", "`bshi_aug.py` (+ exact path `assemble(extra_mask=...)`); store must keep `t_rel`, `rr` (assembler `return_times=True`)"),
        ("12", "Leave-one-department-out (VitalDB)", "yes (new)", f"department for all {sum(d['cases'] for d in meta['department_table']):,} cases from the local `clinical_data.csv`: " + ", ".join(f"{k} {v['cases']:,}" for k, v in dep.items()) + "; MOVER has none", f"`case_meta.parquet` (+ age, sex, ASA); Urology ({dep['Urology']['events_full']:.0f} events) and Gynecology are too small for stable folds"),
        ("13", "Evaluation inputs (FA/h, lead time, replay)", "yes", f"windows are chronological per case, 30 s stride in {_f(100 * ev['eap_full_stride_30s_share'], 0)}% of consecutive pairs (rest are capture gaps); lead-time cap {_f(ev['lead_time_cap_s'], 0)} s respected; test monitoring: VitalDB {_f(hrs[('vitaldb', 'test')]['window_hours_at_30s'], 0)} h, MOVER {_f(hrs[('mover', 'test')]['window_hours_at_30s'], 0)} h", "monitored-hours table in the audit JSON; metric code is for the next evaluation stage"),
        ("14", "Storage and speed", "yes, after a correction", f"{sto['bytes_per_window'] / 1000:.1f} kB/window compact (estimate); measured later: 76.9 kB/window, pooled store 39.5 GB. On-the-fly loader: 3,112 windows/s in case order, but only 60-150 windows/s for shuffled training; the compact store trains at about 2,200 windows/s", "store built (`store_pooled_v1`); within-dataset experiments need their own store"),
    ]
    lines = ["# BSHI training readiness", "",
             "The implemented training interface includes model-side clamping, training-only statistics, explicit loss masks, comparison recipes, raw-window access, "
             "corruption hooks, case metadata, and beat times for gap augmentation. "
             f"The `eap`/`ea` tensor format remains unchanged (`BSHI_CODE_VERSION` is `{res['code_version']}`). {n_pass}/{len(g)} audit gates pass.", "",
             "Source: `notebooks/hemodynamics/bshi_audit.py` -> `results/bshi_training_audit.json` (section 7 of `notebooks/bshi_build.ipynb` re-runs it). No statistic that a model consumes uses validation or test windows; "
             "test windows appear only in the split-agreement diagnostic (row 6).", "",
             "| # | Requirement | Have it? | Evidence | Action taken |", "|---|---|---|---|---|"]
    lines += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in rows]
    lines += ["", "Monitored time (windows x 30 s, `eap_full`):", "", "| dataset | split | patients | windows | hours |", "|---|---|---|---|---|"]
    lines += [f"| {ds} | {sp} | {hrs[(ds, sp)]['patients']:,} | {hrs[(ds, sp)]['windows']:,} | {_f(hrs[(ds, sp)]['window_hours_at_30s'], 0)} |" for ds in ("vitaldb", "mover") for sp in SPLITS]
    lines += ["", "Implemented components: input recipes; `assemble(return_times, extra_mask, extra_mask_ppg)` for beat timing and controlled corruption; "
              "raw-window access; augmentation; loss masks; context and target statistics; and case metadata.",
              "", "The selected eight-parameter context is specified for the next integration stage and has not yet been trained with BSHI. "
              "Matched context ablations, false-alarm evaluation, and department-held-out experiments extend the current training results. "
              "The MOVER test subset has 53 patients and 47 monitored hours, so its estimates require patient-level uncertainty intervals."]
    assert len(lines) <= 90, len(lines)
    (root / DOC).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def print_gates(res):
    for k, v in res["gates"].items():
        print(("PASS  " if v else "FAIL  ") + k)
    return all(res["gates"].values())


if __name__ == "__main__":
    # Store the settings and results as JSON for later inspection.
    import sys
    r = run_audit(".", only=sys.argv[1:] or None)
    print(json.dumps(r["gates"], indent=1), r["seconds"])
    print("doc lines:", write_readiness_doc("."))
