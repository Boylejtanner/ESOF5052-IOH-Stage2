"""Dataset-agnostic "ready window" definition shared by MOVER and VitalDB.

A candidate window is a fixed history span (default 90 s, 30 s stride) whose ART/ECG/PPG validity and arterial-MAP
coverage allow a beat-synchronous image and a label to be built later. Eligibility depends on signal validity and MAP
coverage only; it never looks at whether MAP is low, so no outcome-based selection can happen here.
"""
from __future__ import annotations
# Define when a signal window has enough usable data for forecasting.

import numpy as np
import pandas as pd

DEFAULT_RULE = {
    "window_sec": 90, "stride_sec": 30, "horizon_sec": 300, "theta_floor": 0.50,
    "map_valid_range": [20, 200], "map_hist_min_points": 15, "map_current_max_age_sec": 10,
    "horizon_map_min_fraction": 0.5, "horizon_max_gap_sec": 120,
    "art_plausible": {"pp_range": [15, 150], "mean_range": [30, 180], "min_std": 3},
    "art_map_tolerance_mmHg": 20, "thetas": [0.5, 0.6, 0.7, 0.75, 0.8, 0.9],
}
DEFAULT_LADDER = [
    {"name": "R1", "theta": 0.75, "min_ready_span_min": 30},
    {"name": "R2", "theta": 0.70, "min_ready_span_min": 20},
    {"name": "R3", "theta": 0.60, "min_ready_span_min": 15},
    {"name": "R4", "theta": 0.50, "min_ready_span_min": 10},
]
DEFAULT_TARGETS = {"min_cases": 300, "min_patients": 250, "floor_cases": 150}
WINDOW_COLUMNS = ["case_id", "t_start_sec", "t_end_sec", "min_valid_frac", "valid_art", "valid_ecg", "valid_ppg",
                  "map_hist_points", "map_horizon_points", "map_horizon_max_gap_sec", "art_plausible", "art_map_diff", "is_ready"]
_CHUNK = 192


def load_rule(settings=None):
    rule = {**DEFAULT_RULE, "art_plausible": dict(DEFAULT_RULE["art_plausible"])}
    given = (settings or {}).get("window_rule", {})
    for key, value in given.items():
        if key == "art_plausible":
            rule[key].update(value)
        else:
            rule[key] = value
    return rule


def load_ladder(settings=None):
    settings = settings or {}
    ladder = [dict(r) for r in settings.get("eligibility_ladder", DEFAULT_LADDER)]
    targets = {**DEFAULT_TARGETS, **settings.get("eligibility_targets", {})}
    rule = load_rule(settings)
    for rung in ladder:
        if not any(abs(rung["theta"] - t) < 1e-9 for t in rule["thetas"]):
            raise ValueError(f"Ladder theta {rung['theta']} must be one of window_rule.thetas")
    return ladder, targets


def theta_key(theta):
    return int(round(float(theta) * 100))


def _empty(case_id):
    # Collect the per-record values into a table for comparison.
    frame = pd.DataFrame({
        "case_id": pd.Series(dtype="object"), "t_start_sec": pd.Series(dtype="float64"), "t_end_sec": pd.Series(dtype="float64"),
        "min_valid_frac": pd.Series(dtype="float32"), "valid_art": pd.Series(dtype="float32"), "valid_ecg": pd.Series(dtype="float32"),
        "valid_ppg": pd.Series(dtype="float32"), "map_hist_points": pd.Series(dtype="int32"), "map_horizon_points": pd.Series(dtype="int32"),
        "map_horizon_max_gap_sec": pd.Series(dtype="float32"), "art_plausible": pd.Series(dtype="bool"),
        "art_map_diff": pd.Series(dtype="float32"), "is_ready": pd.Series(dtype="bool")})
    return frame[WINDOW_COLUMNS]


def _valid_cumsum(values):
    return np.concatenate(([0], np.cumsum(np.isfinite(values), dtype=np.int64)))


def _art_plausibility(art, starts, width, rule):
    """Per-window p95-p5 / mean / std test on the finite ART samples (blocks flat lines that are 'valid' non-NaN)."""
    # Reserve output arrays and masks with explicit initial values.
    cfg = rule["art_plausible"]
    ok = np.zeros(len(starts), dtype=bool)
    mean = np.full(len(starts), np.nan)
    view = np.lib.stride_tricks.sliding_window_view(art, width)
    for lo in range(0, len(starts), _CHUNK):
        # Exclude or flag missing values before using these measurements.
        idx = starts[lo:lo + _CHUNK]
        block = np.sort(view[idx], axis=1)
        count = np.isfinite(block).sum(axis=1)
        good = count > 1
        rows = np.arange(len(idx))
        p5 = block[rows, np.clip(np.floor(0.05 * (count - 1)).astype(int), 0, width - 1)]
        p95 = block[rows, np.clip(np.floor(0.95 * (count - 1)).astype(int), 0, width - 1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            m = np.nanmean(block, axis=1, dtype=np.float64)
            s = np.nanstd(block, axis=1, dtype=np.float64)
        pp = p95.astype(np.float64) - p5.astype(np.float64)
        passed = (good & (pp >= cfg["pp_range"][0]) & (pp <= cfg["pp_range"][1])
                  & (m >= cfg["mean_range"][0]) & (m <= cfg["mean_range"][1]) & (s > cfg["min_std"]))
        ok[lo:lo + _CHUNK] = passed
        mean[lo:lo + _CHUNK] = m
    return ok, mean


def ready_windows(art, ecg, ppg, fs, map_time, map_val, rule=None, t0=0.0, case_id=None):
    """Candidate windows (min valid fraction >= theta_floor) with MAP/plausibility flags and ``is_ready``."""
    rule = rule or load_rule()
    fs = float(fs)
    art, ecg, ppg = (np.asarray(x, dtype=np.float32) for x in (art, ecg, ppg))
    if not (len(art) == len(ecg) == len(ppg)):
        raise ValueError("ART/ECG/PPG must share one grid")
    width, step = int(round(rule["window_sec"] * fs)), int(round(rule["stride_sec"] * fs))
    if len(art) < width:
        return _empty(case_id)
    starts = np.arange(0, len(art) - width + 1, step)
    frac = []
    for channel in (art, ecg, ppg):
        cs = _valid_cumsum(channel)
        frac.append((cs[starts + width] - cs[starts]) / width)
    min_frac = np.minimum(np.minimum(frac[0], frac[1]), frac[2])
    keep = min_frac >= rule["theta_floor"] - 1e-12
    if not keep.any():
        return _empty(case_id)
    starts, min_frac = starts[keep], min_frac[keep]
    frac = [f[keep] for f in frac]
    # Calculate the window boundary for the following step.
    t_start = t0 + starts / fs
    t_end = t_start + rule["window_sec"]

    mt = np.asarray(map_time, dtype=np.float64)
    mv = np.asarray(map_val, dtype=np.float64)
    # Exclude or flag missing values before using these measurements.
    lo, hi = rule["map_valid_range"]
    good = np.isfinite(mt) & np.isfinite(mv) & (mv >= lo) & (mv <= hi)
    mt, mv = mt[good], mv[good]
    # Reserve output arrays and masks with explicit initial values.
    order = np.argsort(mt, kind="stable")
    mt, mv = mt[order], mv[order]
    count = len(starts)
    hist_points = np.zeros(count, dtype=np.int32)
    # Reserve output arrays and masks with explicit initial values.
    horizon_points = np.zeros(count, dtype=np.int32)
    max_gap = np.full(count, float(rule["horizon_sec"]), dtype=np.float32)
    hist_mean = np.full(count, np.nan)
    current = np.zeros(count, dtype=bool)
    expected = None
    if len(mt):
        # Locate the requested interval within the sorted sample times.
        cadence = float(np.clip(np.median(np.diff(mt)) if len(mt) > 1 else 2.0, 1.0, 30.0))
        expected = rule["horizon_sec"] / cadence
        cmv = np.concatenate(([0.0], np.cumsum(mv)))
        i0 = np.searchsorted(mt, t_start, side="left")
        i1 = np.searchsorted(mt, t_end, side="left")
        j1 = np.searchsorted(mt, t_end + rule["horizon_sec"], side="right")
        hist_points = (i1 - i0).astype(np.int32)
        horizon_points = (j1 - i1).astype(np.int32)
        with np.errstate(invalid="ignore", divide="ignore"):
            hist_mean = np.where(hist_points > 0, (cmv[i1] - cmv[i0]) / np.maximum(hist_points, 1), np.nan)
        has_last = i1 > i0
        last_time = mt[np.clip(i1 - 1, 0, len(mt) - 1)]
        current = has_last & (last_time >= t_end - rule["map_current_max_age_sec"])
        for k in range(count):
            a, b = i1[k], j1[k]
            if b <= a:
                continue
            seg = mt[a:b]
            gap = max(seg[0] - t_end[k], t_end[k] + rule["horizon_sec"] - seg[-1])
            if len(seg) > 1:
                gap = max(gap, float(np.max(np.diff(seg))))
            max_gap[k] = gap

    # Reserve output arrays and masks with explicit initial values.
    map_ok = ((hist_points >= rule["map_hist_min_points"]) & current & (expected is not None)
              & (horizon_points >= rule["horizon_map_min_fraction"] * (expected or np.inf))
              & (max_gap <= rule["horizon_max_gap_sec"]))
    plausible = np.zeros(count, dtype=bool)
    art_mean = np.full(count, np.nan)
    candidates = np.flatnonzero(map_ok)
    if len(candidates):
        p, m = _art_plausibility(art, starts[candidates], width, rule)
        plausible[candidates], art_mean[candidates] = p, m
    with np.errstate(invalid="ignore"):
        diff = np.abs(art_mean - hist_mean)
    # Exclude or flag missing values before using these measurements.
    agree = np.isfinite(diff) & (diff <= rule["art_map_tolerance_mmHg"])
    ready = map_ok & plausible & agree
    return pd.DataFrame({
        "case_id": case_id if case_id is not None else "", "t_start_sec": t_start, "t_end_sec": t_end,
        "min_valid_frac": min_frac.astype(np.float32), "valid_art": frac[0].astype(np.float32), "valid_ecg": frac[1].astype(np.float32),
        "valid_ppg": frac[2].astype(np.float32), "map_hist_points": hist_points, "map_horizon_points": horizon_points,
        "map_horizon_max_gap_sec": max_gap, "art_plausible": plausible, "art_map_diff": diff.astype(np.float32), "is_ready": ready})[WINDOW_COLUMNS]


def union_span_minutes(t_start, window_sec):
    """Length of the union of [t, t+window) intervals, in minutes (t_start sorted)."""
    t = np.sort(np.asarray(t_start, dtype=np.float64))
    if not len(t):
        return 0.0
    ends = t + window_sec
    previous_end = np.concatenate(([-np.inf], np.maximum.accumulate(ends)[:-1]))
    return float(np.sum(ends - np.maximum(t, previous_end)) / 60.0)


def case_summary(windows, rule=None):
    """One-row case statistics: counts and ready span (minutes) at every theta."""
    rule = rule or load_rule()
    out = {"n_candidate_windows": int(len(windows)), "n_ready_windows": int(windows.is_ready.sum()) if len(windows) else 0,
           "art_plausible_fraction": float(windows.art_plausible.mean()) if len(windows) else 0.0}
    ready = windows.loc[windows.is_ready] if len(windows) else windows
    for theta in rule["thetas"]:
        sel = ready.loc[ready.min_valid_frac >= theta - 1e-6] if len(ready) else ready
        key = theta_key(theta)
        out[f"n_ready_theta_{key}"] = int(len(sel))
        out[f"ready_span_min_theta_{key}"] = union_span_minutes(sel.t_start_sec.to_numpy(), rule["window_sec"]) if len(sel) else 0.0
    return out


def rung_passes(stats, rung):
    return stats[f"ready_span_min_theta_{theta_key(rung['theta'])}"] >= rung["min_ready_span_min"]


def choose_rung(stats, ladder, targets):
    """Strictest rung with >= min_cases cases and >= min_patients patients; else the loosest rung.

    ``stats`` needs patient_id plus the ready_span_min_theta_* columns. Returns (rung|None, funnel DataFrame, reason).
    ``rung`` is None when even the loosest rung has fewer than floor_cases cases.
    """
    rows = []
    for rung in ladder:
        passed = stats.loc[rung_passes(stats, rung)]
        rows.append({"rung": rung["name"], "theta": rung["theta"], "min_ready_span_min": rung["min_ready_span_min"],
                     "cases": int(len(passed)), "patients": int(passed.patient_id.nunique())})
    funnel = pd.DataFrame(rows)
    for rung, row in zip(ladder, rows):
        if row["cases"] >= targets["min_cases"] and row["patients"] >= targets["min_patients"]:
            return rung, funnel, f"{rung['name']} is the strictest rung meeting {targets['min_cases']} cases / {targets['min_patients']} patients"
    loosest = ladder[-1]
    last = rows[-1]
    if last["cases"] < targets["floor_cases"]:
        return None, funnel, (f"even the loosest rung {loosest['name']} gives only {last['cases']} cases "
                              f"(< floor {targets['floor_cases']}); not creating a cohort")
    return loosest, funnel, (f"no rung reached {targets['min_cases']} cases / {targets['min_patients']} patients; "
                             f"using the loosest rung {loosest['name']} ({last['cases']} cases, {last['patients']} patients)")
