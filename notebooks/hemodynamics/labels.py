"""Forecast labels for every ready window, from the arterial MAP track only (no waveform is read here).

Definitions (proposal Sec. V-A; the proposal leaves the "sustained" duration open, it is fixed here and hashed into every output):
* t0 = ``t_end_sec`` of a ready window; MAP is valid when finite and inside ``map_valid_range``.
* A *sustained episode* is a run of consecutive valid MAP samples all below ``hypo_threshold`` whose gaps are <= ``max_gap_sec``
  and whose last-minus-first time is >= ``sustained_min_sec``.
* ``y_event`` = a sustained episode STARTS in (t0, t0+H] and stays below the threshold for >= ``sustained_min_sec`` counting only
  samples inside the horizon (a longer episode that continues past t0+H still counts; no data beyond the horizon is needed);
  ``time_to_event_sec`` = start of that run minus t0.
* ``map_current`` = median valid MAP in the last ``map0_window_sec``; ``already_hypotensive`` = any valid MAP < threshold in the last
  ``already_hypo_lookback_sec`` (this also drops ongoing episodes and the first minute after recovery; ``map_current_lt65`` is kept
  so downstream can relax it); ``eligible_onset`` = not already hypotensive and ``map_current`` finite.
* ``delta_map`` = max(0, map_current - min valid MAP in (t0, t0+H]); ``delta_map_robust`` uses the minimum of a trailing median over
  ``robust_min_window_sec`` taken only from samples whose whole median window lies inside the horizon.
* ``aut65`` = integral over (t0, t0+H] of max(0, 65 - MAP(t)) dt in mmHg*min, MAP piecewise linear between consecutive valid samples
  (exact, including crossings of the threshold); intervals with a gap > ``max_gap_sec`` are unobserved and contribute 0, reported by
  ``aut65_observed_fraction``.
* ``horizon_gap_max_sec`` is the largest MAP gap in the horizon (edges included); ``label_ambiguous`` = no event but that gap exceeds
  ``sustained_min_sec`` (an unobserved episode could have fitted in it). ``gray_zone`` = gray_zone[0] <= map_current < gray_zone[1].
"""
from __future__ import annotations
# Create future-pressure outcomes while keeping future values out of the inputs.

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import rankdata

from .canonical import atomic_json, utc_now
from .snapshot import assert_dataset_v1

DEFAULT_RULE = {"horizon_sec": 300, "hypo_threshold": 65, "sustained_min_sec": 60, "max_gap_sec": 10, "already_hypo_lookback_sec": 60,
                "gray_zone": [65, 75], "map_valid_range": [20, 200], "map0_window_sec": 10, "robust_min_window_sec": 30, "aut_low_coverage": 0.9}
LABEL_COLUMNS = ["dataset", "case_id", "patient_key", "split_within_dataset", "split_pooled", "t_start_sec", "t_end_sec", "min_valid_frac",
                 "map_current", "map_mean_60", "map_min_60", "map_std_60", "map_slope_60", "n_map_60", "already_hypotensive", "map_current_lt65",
                 "eligible_onset", "gray_zone", "y_event", "time_to_event_sec", "delta_map", "delta_map_robust", "aut65", "aut65_observed_fraction",
                 "aut65_low_coverage", "horizon_gap_max_sec", "horizon_n_map", "label_ambiguous", "label_rule_sha256", "dataset_version"]
_LABELS_DIR = Path("data") / "derived" / "labels"
LABEL_CODE_VERSION = "lbl1"      # bump when label semantics change without a rule change, so cached labels are recomputed
FEATURES = ["map_current", "map_mean_60", "map_slope_60", "map_min_60", "map_std_60"]


def load_label_rule(settings=None):
    rule = {**DEFAULT_RULE, **((settings or {}).get("label_rule") or {})}
    return rule


def label_rule_sha256(rule):
    return hashlib.sha256(json.dumps(rule, sort_keys=True).encode()).hexdigest()


# primitives
def _gather(t, v, lo, hi):
    """Ragged slices t[lo:hi], v[lo:hi] as NaN-padded matrices (rows = windows)."""
    count = (hi - lo).astype(int)
    k = int(count.max()) if len(count) else 0
    if k <= 0 or len(t) == 0:
        return np.full((len(lo), 0), np.nan), np.full((len(lo), 0), np.nan)
    pos = np.arange(k)[None, :]
    ok = pos < count[:, None]
    safe = np.minimum(lo[:, None] + pos, len(t) - 1)
    return np.where(ok, t[safe], np.nan), np.where(ok, v[safe], np.nan)


def sustained_runs(t, v, thr, max_gap, min_sec):
    """(starts, ends) of runs of consecutive valid samples < thr (gaps <= max_gap) lasting >= min_sec."""
    if len(t) == 0:
        return np.array([]), np.array([])
    # Reserve output arrays and masks with explicit initial values.
    below = v < thr
    link = np.zeros(len(t), dtype=bool)
    link[1:] = below[1:] & below[:-1] & (np.diff(t) <= max_gap)
    start_idx = np.flatnonzero(below & ~link)
    end_idx = np.flatnonzero(below & ~np.append(link[1:], False))
    keep = (t[end_idx] - t[start_idx]) >= min_sec
    return t[start_idx[keep]], t[end_idx[keep]]


def _partial_area(a0, a1, dt, s):
    """Integral over [0, s] of max(0, a) where a is linear from a0 (at 0) to a1 (at dt); s <= dt."""
    a_s = a0 + (a1 - a0) * s / np.where(dt > 0, dt, 1.0)
    both_pos = (a0 >= 0) & (a_s >= 0)
    down = (a0 > 0) & (a_s < 0)
    up = (a0 < 0) & (a_s > 0)
    denom_down = np.where(down, a0 - a_s, 1.0)
    denom_up = np.where(up, a_s - a0, 1.0)
    return np.where(both_pos, s * (a0 + a_s) / 2,
                    np.where(down, (s * a0 / denom_down) * a0 / 2, np.where(up, (s - s * (-a0) / denom_up) * a_s / 2, 0.0)))


class _Integral:
    """Cumulative below-threshold area (mmHg*s) and observed time, evaluated at arbitrary times."""

    def __init__(self, t, v, thr, max_gap):
        self.t = t
        if len(t) < 2:
            # Reserve output arrays and masks with explicit initial values.
            self.dt = self.obs = self.a0 = self.a1 = np.array([])
            self.C = self.O = np.zeros(max(1, len(t)))
            return
        self.dt = np.diff(t)
        self.obs = self.dt <= max_gap
        self.a0, self.a1 = thr - v[:-1], thr - v[1:]
        area = np.where(self.obs, _partial_area(self.a0, self.a1, self.dt, self.dt), 0.0)
        self.C = np.concatenate([[0.0], np.cumsum(area)])
        self.O = np.concatenate([[0.0], np.cumsum(np.where(self.obs, self.dt, 0.0))])

    def at(self, tau):
        tau = np.asarray(tau, dtype=float)
        area, seen = np.zeros_like(tau), np.zeros_like(tau)
        if len(self.t) < 2:
            return area, seen
        # Locate the requested interval within the sorted sample times.
        k = np.searchsorted(self.t, tau, side="right") - 1
        after = k >= len(self.t) - 1
        area[after], seen[after] = self.C[-1], self.O[-1]
        inside = (k >= 0) & ~after
        kk = k[inside]
        s = tau[inside] - self.t[kk]
        o = self.obs[kk]
        area[inside] = self.C[kk] + np.where(o, _partial_area(self.a0[kk], self.a1[kk], self.dt[kk], s), 0.0)
        seen[inside] = self.O[kk] + np.where(o, s, 0.0)
        return area, seen


def _horizon_gap(tm, t0, horizon):
    """Largest gap between valid MAP samples in (t0, t0+H] counting both edges; H when there are none."""
    if tm.shape[1] == 0:
        return np.full(len(t0), float(horizon))
    # Exclude or flag missing values before using these measurements.
    valid = np.isfinite(tm)
    first = np.where(valid.any(1), np.nanmin(np.where(valid, tm, np.inf), axis=1), np.nan)
    last = np.where(valid.any(1), np.nanmax(np.where(valid, tm, -np.inf), axis=1), np.nan)
    inner = np.diff(tm, axis=1)
    inner = np.where(valid[:, 1:] & valid[:, :-1], inner, 0.0)
    best = np.maximum(inner.max(axis=1) if inner.size else 0.0, np.maximum(first - t0, t0 + horizon - last))
    return np.where(valid.any(1), best, float(horizon))


def label_case(win, t, v, rule):
    """Label one case. ``win`` has t_start_sec, t_end_sec, min_valid_frac (ready windows only); (t, v) is its raw MAP track."""
    # Exclude or flag missing values before using these measurements.
    thr, horizon = float(rule["hypo_threshold"]), float(rule["horizon_sec"])
    lo_v, hi_v = rule["map_valid_range"]
    t, v = np.asarray(t, dtype=float), np.asarray(v, dtype=float)
    # Exclude or flag missing values before using these measurements.
    good = np.isfinite(t) & np.isfinite(v) & (v >= lo_v) & (v <= hi_v)
    t, v = t[good], v[good]
    order = np.argsort(t, kind="stable")
    t, v = t[order], v[order]
    # Locate the requested interval within the sorted sample times.
    t0 = win["t_end_sec"].to_numpy(dtype=float)
    n = len(t0)
    nan = np.full(n, np.nan)
    hi = np.searchsorted(t, t0, side="right")
    lo10 = np.searchsorted(t, t0 - float(rule["map0_window_sec"]), side="right")
    lo60 = np.searchsorted(t, t0 - float(rule["already_hypo_lookback_sec"]), side="right")
    jhi = np.searchsorted(t, t0 + horizon, side="right")
    with warnings.catch_warnings():
        # Exclude or flag missing values before using these measurements.
        warnings.simplefilter("ignore", RuntimeWarning)
        _, v10 = _gather(t, v, lo10, hi)
        map_current = np.nanmedian(v10, axis=1) if v10.shape[1] else nan.copy()
        # Exclude or flag missing values before using these measurements.
        t60, v60 = _gather(t, v, lo60, hi)
        has60 = np.isfinite(v60)
        n60 = has60.sum(1)
        mean60 = np.where(n60 > 0, np.nanmean(v60, axis=1), np.nan) if v60.shape[1] else nan.copy()
        min60 = np.where(n60 > 0, np.nanmin(v60, axis=1), np.nan) if v60.shape[1] else nan.copy()
        std60 = np.where(n60 > 1, np.nanstd(v60, axis=1), np.nan) if v60.shape[1] else nan.copy()
        slope = nan.copy()
        if v60.shape[1]:
            x = np.where(has60, t60 - t0[:, None], 0.0)
            y = np.where(has60, v60, 0.0)
            sx, sy, sxx, sxy = x.sum(1), y.sum(1), (x * x).sum(1), (x * y).sum(1)
            den = n60 * sxx - sx * sx
            slope = np.where((n60 >= 3) & (den > 0), (n60 * sxy - sx * sy) / np.where(den > 0, den, 1.0) * 60.0, np.nan)
        # Exclude or flag missing values before using these measurements.
        already = (v60 < thr).any(axis=1) if v60.shape[1] else np.zeros(n, dtype=bool)
        th, vh = _gather(t, v, hi, jhi)
        min_future = np.nanmin(vh, axis=1) if vh.shape[1] else nan.copy()
        horizon_n = np.isfinite(vh).sum(1) if vh.shape[1] else np.zeros(n, dtype=int)
        delta = np.where(np.isfinite(map_current) & (horizon_n > 0), np.maximum(0.0, map_current - min_future), np.nan)
        delta_robust = nan.copy()
        if len(t) and vh.shape[1]:
            # Locate the requested interval within the sorted sample times.
            rw = float(rule["robust_min_window_sec"])
            rlo = np.searchsorted(t, t - rw, side="left")
            _, vr = _gather(t, v, rlo, np.arange(len(t)) + 1)
            roll = np.where(np.isfinite(vr).sum(1) >= 5, np.nanmedian(vr, axis=1), np.nan)
            _, rh = _gather(t, roll, hi, jhi)                                    # trailing medians of the horizon samples only
            usable = np.where(th >= (t0 + rw)[:, None], rh, np.inf)             # whole median window must lie inside the horizon
            rmin = np.where(np.isfinite(usable), usable, np.inf).min(axis=1)
            delta_robust = np.where(np.isfinite(map_current) & np.isfinite(rmin), np.maximum(0.0, map_current - rmin), np.nan)
    starts, ends = sustained_runs(t, v, thr, float(rule["max_gap_sec"]), float(rule["sustained_min_sec"]))
    j = np.searchsorted(starts, t0, side="right")
    jc = np.minimum(j, max(len(starts) - 1, 0))
    if len(starts):
        # The qualifying minute must come from samples inside the horizon; a longer episode that continues past t0+H still counts.
        inside_end = np.where(ends[jc] <= t0 + horizon, ends[jc], t[np.maximum(jhi - 1, 0)])
        event = (j < len(starts)) & (starts[jc] <= t0 + horizon) & (inside_end - starts[jc] >= float(rule["sustained_min_sec"]))
        tte = np.where(event, starts[jc] - t0, np.nan)
    else:
        event, tte = np.zeros(n, dtype=bool), nan.copy()
    integral = _Integral(t, v, thr, float(rule["max_gap_sec"]))
    area_hi, seen_hi = integral.at(t0 + horizon)
    area_lo, seen_lo = integral.at(t0)
    aut65 = (area_hi - area_lo) / 60.0
    # Exclude or flag missing values before using these measurements.
    observed = np.clip((seen_hi - seen_lo) / horizon, 0.0, 1.0)
    gap = _horizon_gap(th, t0, horizon)
    gray = (map_current >= rule["gray_zone"][0]) & (map_current < rule["gray_zone"][1])
    # Exclude or flag missing values before using these measurements.
    lt65 = map_current < thr
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return pd.DataFrame({
        "t_start_sec": win["t_start_sec"].to_numpy(dtype=float), "t_end_sec": t0, "min_valid_frac": f32(win["min_valid_frac"].to_numpy()),
        "map_current": f32(map_current), "map_mean_60": f32(mean60), "map_min_60": f32(min60), "map_std_60": f32(std60), "map_slope_60": f32(slope),
        "n_map_60": n60.astype(np.int16), "already_hypotensive": already, "map_current_lt65": lt65 & np.isfinite(map_current),
        "eligible_onset": ~already & np.isfinite(map_current), "gray_zone": gray & np.isfinite(map_current), "y_event": event,
        "time_to_event_sec": f32(tte), "delta_map": f32(delta), "delta_map_robust": f32(delta_robust), "aut65": f32(aut65),
        "aut65_observed_fraction": f32(observed), "aut65_low_coverage": observed < float(rule["aut_low_coverage"]),
        "horizon_gap_max_sec": f32(gap), "horizon_n_map": horizon_n.astype(np.int16),
        "label_ambiguous": (~event) & (gap > float(rule["sustained_min_sec"]))})


# build
def _job(root, row, win, rule):
    root = Path(root)
    m = pq.read_table(root / row["map_path"]).to_pandas()
    frame = label_case(win, m["time_sec"].to_numpy(), m["MAP"].to_numpy(), rule)
    return row["dataset"], row["case_id"], frame


def _signature(map_sha, rule_hash, win):
    # Hash the content so later runs can detect changed inputs.
    h = hashlib.sha256(f"{map_sha}|{rule_hash}|{LABEL_CODE_VERSION}".encode())
    h.update(np.ascontiguousarray(win["t_end_sec"].to_numpy(dtype=float)).tobytes())
    return h.hexdigest()


def _write(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd", row_group_size=131072)
        if pq.ParquetFile(temp).metadata.num_rows != len(frame):
            raise ValueError("labels readback row count mismatch")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def labels_path(root, dataset):
    return Path(root) / _LABELS_DIR / f"labels_{dataset}.parquet"


def build_labels(root, datasets=("mover", "vitaldb"), workers=None, progress=True, allow_changed=False):
    """Compute labels for every ready window of every case; incremental (unchanged cases are skipped)."""
    # Keep the label definition and patient partitions fixed while processing cases.
    started = time.perf_counter()
    root = Path(root)
    tag = assert_dataset_v1(root, allow_changed=allow_changed)
    # Load the recorded settings or metadata for this processing step.
    settings = json.loads((root / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))
    rule = load_label_rule(settings)
    rule_hash = label_rule_sha256(rule)
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet")
    # Read the stored table required for this calculation.
    sig_path = root / _LABELS_DIR / "case_signatures.parquet"
    old_sig = pd.read_parquet(sig_path) if sig_path.exists() else pd.DataFrame(columns=["dataset", "case_id", "signature"])
    known = {(d, c): s for d, c, s in old_sig[["dataset", "case_id", "signature"]].itertuples(index=False, name=None)}
    summary = {"rule_sha256": rule_hash, "datasets": {}}
    new_sigs = []
    for dataset in datasets:
        # Summarize the records separately within each requested group.
        windows = pd.read_parquet(root / "data" / "manifests" / f"ready_windows_{dataset}.parquet", columns=["case_id", "t_start_sec", "t_end_sec", "min_valid_frac", "is_ready"])
        windows = windows.loc[windows.is_ready].astype({"case_id": str})
        by_case = {k: g.reset_index(drop=True) for k, g in windows.groupby("case_id")}
        cases = index.loc[index.dataset == dataset]
        path = labels_path(root, dataset)
        existing = pd.read_parquet(path) if path.exists() else None
        have = set(existing.case_id.astype(str)) if existing is not None else set()
        todo, skipped = [], 0
        # Apply this calculation to each row in the selected table.
        for r in cases.itertuples():
            cid = str(r.case_id)
            win = by_case.get(cid)
            if win is None or not len(win):
                continue
            sig = _signature(r.map_sha256, rule_hash, win)
            new_sigs.append({"dataset": dataset, "case_id": cid, "signature": sig})
            if known.get((dataset, cid)) == sig and cid in have:
                skipped += 1
            else:
                todo.append((r, win))
        frames = []
        if todo:
            workers_n = max(1, min(int(workers or max(1, (os.cpu_count() or 8) - 8)), len(todo), 16))
            bar = None
            if progress:
                from tqdm.auto import tqdm
                bar = tqdm(total=len(todo), desc=f"labels [{dataset}]", unit="case")
            meta = {str(r.case_id): (r.patient_key, r.split_within_dataset, r.split_pooled) for r, _ in todo}
            # Ensure the associated files, locks, or resources are released after this operation.
            try:
                with cf.ProcessPoolExecutor(max_workers=workers_n, mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = [pool.submit(_job, str(root), {"dataset": dataset, "case_id": str(r.case_id), "map_path": r.map_path}, win, rule) for r, win in todo]
                    for future in cf.as_completed(futures):
                        _, cid, frame = future.result()
                        pk, sw, sp = meta[cid]
                        frame.insert(0, "split_pooled", sp)
                        frame.insert(0, "split_within_dataset", sw)
                        frame.insert(0, "patient_key", pk)
                        frame.insert(0, "case_id", cid)
                        frame.insert(0, "dataset", dataset)
                        frames.append(frame)
                        bar and bar.update(1)
            finally:
                bar and bar.close()
        redo = {str(r.case_id) for r, _ in todo}
        keep = existing.loc[~existing.case_id.astype(str).isin(redo) & existing.case_id.astype(str).isin({str(c) for c in cases.case_id})] if existing is not None else None
        parts = [p for p in (keep, *frames) if p is not None and len(p)]
        if parts and (todo or existing is None):
            # Collect the per-record values into a table for comparison.
            out = pd.concat(parts, ignore_index=True)
            out["label_rule_sha256"] = rule_hash
            out["dataset_version"] = tag["version"]
            out = out.sort_values(["case_id", "t_end_sec"], kind="stable").reset_index(drop=True)[LABEL_COLUMNS]
            _write(path, out)
        summary["datasets"][dataset] = {"cases": len(cases), "computed": len(todo), "skipped_current": skipped,
                                        "rows": int(len(pd.read_parquet(path, columns=["case_id"]))) if path.exists() else 0}
    sigs = pd.DataFrame(new_sigs)
    if len(sigs):
        _write(sig_path, sigs)
    summary.update(seconds=round(time.perf_counter() - started, 1), dataset_index_sha256=tag["index_content_sha256"], at=utc_now())
    atomic_json(root / "data" / "manifests" / "labels_manifest.json", summary)
    if progress:
        print("LABELS:", json.dumps({k: v for k, v in summary.items() if k != "rule_sha256"}), flush=True)
    return summary


def load_labels(root, datasets=("mover", "vitaldb"), columns=None):
    frames = [pd.read_parquet(labels_path(root, d), columns=columns) for d in datasets if labels_path(root, d).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns or LABEL_COLUMNS)


def summarize_labels(root, min_valid_frac=0.7, write=True):
    """Per dataset x split table (and pooled split) of window counts, prevalence and label quality; saved to results/label_summary.json."""
    lab = load_labels(root)
    lab = lab.loc[lab.min_valid_frac >= min_valid_frac - 1e-6]

    def table(keys):
        rows = []
        for key, g in lab.groupby(keys):
            e = g.loc[g.eligible_onset]
            rows.append({**(dict(zip(keys, key if isinstance(key, tuple) else (key,)))), "windows": len(g), "eligible": len(e),
                         "event_prevalence_pct": round(100 * e.y_event.mean(), 2) if len(e) else None,
                         "ambiguous_neg_pct": round(100 * e.label_ambiguous.mean(), 2) if len(e) else None,
                         "aut65_low_coverage_pct": round(100 * e.aut65_low_coverage.mean(), 2) if len(e) else None,
                         "gray_zone_pct": round(100 * e.gray_zone.mean(), 2) if len(e) else None,
                         "aut65_zero_pct": round(100 * (e.aut65 == 0).mean(), 2) if len(e) else None,
                         "delta_map_zero_pct": round(100 * (e.delta_map == 0).mean(), 2) if len(e) else None})
        return pd.DataFrame(rows)

    by_split = table(["dataset", "split_within_dataset"])
    pooled = table(["split_pooled"])
    if write:
        atomic_json(Path(root) / "results" / "label_summary.json", {"min_valid_frac": min_valid_frac, "by_dataset_split": by_split.to_dict("records"), "pooled": pooled.to_dict("records"), "at": utc_now()})
    return by_split, pooled


# baseline A
def auroc(y, s):
    y = np.asarray(y, dtype=bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, s):
    y = np.asarray(y, dtype=bool)
    if not y.any():
        return float("nan")
    order = np.argsort(-np.asarray(s), kind="stable")
    yy = y[order]
    tp = np.cumsum(yy)
    return float((tp / np.arange(1, len(yy) + 1))[yy].sum() / yy.sum())


def _calibration(y, p):
    # Fit the model or transform using the supplied fitting partition.
    from sklearn.linear_model import LogisticRegression
    p = np.clip(p, 1e-6, 1 - 1e-6)
    z = np.log(p / (1 - p)).reshape(-1, 1)
    fit = LogisticRegression(C=1e6, max_iter=200).fit(z, np.asarray(y, dtype=int))
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def _metrics(y, p):
    slope, intercept = _calibration(y, p)
    return {"auroc": auroc(y, p), "auprc": auprc(y, p), "brier": float(np.mean((p - y) ** 2)), "cal_slope": slope, "cal_intercept": intercept}


def _cluster_ci(df, y_col, p_col, reps, seed=5052):
    # Summarize the records separately within each requested group.
    groups = {k: (g[y_col].to_numpy(), g[p_col].to_numpy()) for k, g in df.groupby("patient_key")}
    keys = np.array(list(groups))
    rng = np.random.default_rng(seed)
    acc = {"auroc": [], "auprc": [], "brier": []}
    for _ in range(reps):
        pick = rng.choice(keys, len(keys), replace=True)
        y = np.concatenate([groups[k][0] for k in pick])
        p = np.concatenate([groups[k][1] for k in pick])
        if not y.any():
            continue
        acc["auroc"].append(auroc(y, p)); acc["auprc"].append(auprc(y, p)); acc["brier"].append(float(np.mean((p - y) ** 2)))
    return {k: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] for k, a in acc.items() if a}


def _baseline_frame(root, min_valid_frac):
    lab = load_labels(root)
    lab = lab.loc[lab.eligible_onset & ~lab.label_ambiguous & (lab.min_valid_frac >= min_valid_frac - 1e-6)].dropna(subset=FEATURES).copy()
    lab["y"] = lab.y_event.astype(int)
    return lab


def _baseline_settings(lab):
    return {
        "pooled_train->pooled_test": (lab.split_pooled == "train", lab.split_pooled == "test"),
        "vitaldb_train->vitaldb_test": ((lab.dataset == "vitaldb") & (lab.split_within_dataset == "train"), (lab.dataset == "vitaldb") & (lab.split_within_dataset == "test")),
        "E3x vitaldb_train->all_mover": ((lab.dataset == "vitaldb") & (lab.split_within_dataset == "train"), lab.dataset == "mover"),
    }


def _fit_predict(train, test):
    # Calculate predicted risk without refitting on evaluation windows.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(train[FEATURES])
    clf = LogisticRegression(max_iter=500).fit(scaler.transform(train[FEATURES]), train.y)
    return clf.predict_proba(scaler.transform(test[FEATURES]))[:, 1]


def calibration_points(root, setting="E3x vitaldb_train->all_mover", bins=10, min_valid_frac=0.7):
    """Reliability-curve points (equal-count bins) of baseline A for one setting."""
    # Summarize the records separately within each requested group.
    lab = _baseline_frame(root, min_valid_frac)
    tr_mask, te_mask = _baseline_settings(lab)[setting]
    test = lab.loc[te_mask].copy()
    test["p"] = _fit_predict(lab.loc[tr_mask], test)
    test["bin"] = pd.qcut(test.p.rank(method="first"), bins, labels=False)
    return test.groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), windows=("y", "size")).reset_index()


def baseline_a(root, reps=200, min_valid_frac=0.7, write=True):
    """Current-MAP + trend baseline (logistic on MAP features) with patient-level bootstrap CIs; also zero-heavy regression references."""
    # Load the recorded settings or metadata for this processing step.
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    lab = _baseline_frame(root, min_valid_frac)
    settings = _baseline_settings(lab)
    out = {"rule_sha256": label_rule_sha256(load_label_rule(json.loads((Path(root) / "configs" / "hemodynamics.json").read_text()))),
           "min_valid_frac": min_valid_frac, "features": FEATURES, "settings": {}}
    for name, (tr_mask, te_mask) in settings.items():
        # Calculate prediction metrics from these labels and model scores.
        train, test = lab.loc[tr_mask], lab.loc[te_mask].copy()
        test["p"] = _fit_predict(train, test)
        test["naive"] = -test.map_current.to_numpy()
        entry = {"train_windows": int(len(train)), "test_windows": int(len(test)), "test_patients": int(test.patient_key.nunique()),
                 "prevalence": float(test.y.mean()), "model": _metrics(test.y.to_numpy(), test.p.to_numpy()),
                 "model_ci95": _cluster_ci(test, "y", "p", reps), "negative_current_map_auroc": auroc(test.y.to_numpy(), test.naive.to_numpy()),
                 "negative_current_map_auroc_ci95": _cluster_ci(test.assign(naive=test.naive), "y", "naive", reps).get("auroc")}
        gray = test.loc[test.gray_zone]
        if len(gray) and gray.y.any():
            entry["gray_zone"] = {"windows": int(len(gray)), "prevalence": float(gray.y.mean()), **{k: v for k, v in _metrics(gray.y.to_numpy(), gray.p.to_numpy()).items() if k in ("auroc", "auprc")}}
        if name.startswith("pooled"):
            entry["by_dataset"] = {ds: {"patients": int(g.patient_key.nunique()), **{k: v for k, v in _metrics(g.y.to_numpy(), g.p.to_numpy()).items() if k in ("auroc", "auprc")},
                                        "auroc_ci95": _cluster_ci(g, "y", "p", reps).get("auroc")} for ds, g in test.groupby("dataset")}
        out["settings"][name] = entry
    train, test = lab.loc[lab.split_pooled == "train"].dropna(subset=["delta_map", "aut65"]), lab.loc[lab.split_pooled == "test"].dropna(subset=["delta_map", "aut65"])
    scaler = StandardScaler().fit(train[FEATURES])
    reg = {"test_windows": int(len(test))}
    for target in ("delta_map", "aut65"):
        # Fit the model or transform using the supplied fitting partition.
        ridge = Ridge(alpha=1.0).fit(scaler.transform(train[FEATURES]), train[target])
        reg[target] = {"zero_fraction_test": float((test[target] == 0).mean()), "mean_test": float(test[target].mean()),
                       "mae_predict_train_median": float(np.mean(np.abs(test[target] - train[target].median()))),
                       "mae_ridge_on_map_features": float(np.mean(np.abs(test[target] - ridge.predict(scaler.transform(test[FEATURES])))))}
    out["regression_references_pooled_test"] = reg
    if write:
        atomic_json(Path(root) / "results" / "baseline_A.json", out)
    return out
