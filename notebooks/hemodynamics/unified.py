"""Unified case index and incremental ready-window pass for MOVER and VitalDB.

No model, label or BSHI logic: this only (1) computes the shared ready-window table with ``windows.py`` for every completed
case of a dataset (incrementally; a case is recomputed only if its waveform hash or the rule changed) and (2) joins both
datasets into one manifest that training/Spark code can consume.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .canonical import CheckpointStore, assert_patient_independence, patient_splits, utc_now, write_manifest
from .windows import case_summary, load_rule, ready_windows

COMPLETED = ("complete", "completed")
SPLITS = ("train", "validation", "test")
EXPERIMENTS = {
    "E1_pooled": "train/validation/test = split_pooled (patient-level 70/15/15 over both datasets pooled)",
    "E2_mover_to_vitaldb": "train = MOVER train; validation/test = VitalDB validation/test (institutional generalization)",
    "E3_vitaldb_to_mover": "train = VitalDB train; validation/test = MOVER validation/test",
    "E3x_vitaldb_to_mover_external": "train/validation = VitalDB train/validation; test = ALL MOVER cases",
}
_STAT_COLUMNS = ["valid_art", "valid_ecg", "valid_ppg", "map_median", "n_candidate_windows", "n_ready_windows", "art_plausible_fraction"]
_THETAS = (50, 60, 70, 75, 80, 90)


def _settings(root):
    return json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))


def rule_sha256(settings):
    return hashlib.sha256(json.dumps(load_rule(settings), sort_keys=True).encode()).hexdigest()


def _completed_rows(root, dataset):
    return [r for r in CheckpointStore(root, dataset).rows() if r.get("status") in COMPLETED and r.get("waveform_output_path") and r.get("map_output_path")]


def _write_parquet(path, frame):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _index_case(root, row, rule, fs, rule_hash):
    root = Path(root)
    wave = pq.read_table(root / row["waveform_output_path"], columns=["time_sec", "ART", "ECG", "PPG"]).to_pandas()
    numeric = pq.read_table(root / row["map_output_path"]).to_pandas()
    case_id = str(row["case_id"])
    frame = ready_windows(wave["ART"].to_numpy(), wave["ECG"].to_numpy(), wave["PPG"].to_numpy(), fs,
                          numeric["time_sec"].to_numpy(), numeric["MAP"].to_numpy(), rule,
                          t0=float(wave["time_sec"].iloc[0]), case_id=case_id)
    values = numeric["MAP"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    stats = {"case_id": case_id, "patient_id": str(row.get("patient_id")), "waveform_sha256": row["waveform_sha256"], "rule_sha256": rule_hash,
             "duration_sec": float(len(wave) / fs), "map_points": int(values.size), "map_median": float(np.median(values)) if values.size else None,
             **{f"valid_{c.lower()}": float(np.isfinite(wave[c].to_numpy()).mean()) for c in ("ART", "ECG", "PPG")},
             **case_summary(frame, rule)}
    return stats, frame


def index_ready_windows(root, dataset="vitaldb", workers=None, progress=True):
    """Incremental ready-window pass over every completed case of ``dataset``.

    Writes data/manifests/ready_windows_<dataset>.parquet and <dataset>_case_window_stats.parquet. Cases whose
    waveform hash and rule hash are unchanged are skipped; windows/stats of other cases are kept.
    """
    started = time.perf_counter()
    root = Path(root)
    settings = _settings(root)
    rule, fs, rule_hash = load_rule(settings), float(settings["target_waveform_fs"]), rule_sha256(settings)
    manifests = root / "data" / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    stats_path, windows_path = manifests / f"{dataset}_case_window_stats.parquet", manifests / f"ready_windows_{dataset}.parquet"
    rows = _completed_rows(root, dataset)
    old_stats = pd.read_parquet(stats_path) if stats_path.exists() else pd.DataFrame()
    old_windows = pd.read_parquet(windows_path) if windows_path.exists() else None
    current = set()
    if old_windows is not None and {"waveform_sha256", "rule_sha256"}.issubset(old_stats.columns):
        current = set(old_stats.loc[old_stats.rule_sha256.eq(rule_hash), ["case_id", "waveform_sha256"]].astype(str).itertuples(index=False, name=None))
    todo = [r for r in rows if (str(r["case_id"]), str(r["waveform_sha256"])) not in current]
    new_stats, new_frames = [], []
    if todo:
        workers = max(1, min(int(workers or max(1, (os.cpu_count() or 8) - 4)), len(todo), 24))
        bar = None
        if progress:
            from tqdm.auto import tqdm
            bar = tqdm(total=len(todo), desc=f"ready windows [{dataset}]", unit="case")
        try:
            if workers == 1:
                results = (_index_case(root, r, rule, fs, rule_hash) for r in todo)
                for stats, frame in results:
                    new_stats.append(stats)
                    new_frames.append(frame)
                    bar and bar.update(1)
            else:
                with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = [pool.submit(_index_case, root, r, rule, fs, rule_hash) for r in todo]
                    for future in cf.as_completed(futures):
                        stats, frame = future.result()
                        new_stats.append(stats)
                        new_frames.append(frame)
                        bar and bar.update(1)
        finally:
            bar and bar.close()
    redo = {s["case_id"] for s in new_stats}
    kept_stats = old_stats.loc[~old_stats.case_id.astype(str).isin(redo)] if len(old_stats) else old_stats
    stats = pd.concat([f for f in (kept_stats, pd.DataFrame(new_stats)) if len(f)], ignore_index=True) if (len(kept_stats) or new_stats) else pd.DataFrame()
    kept_windows = old_windows.loc[~old_windows.case_id.astype(str).isin(redo)] if old_windows is not None else None
    parts = [f for f in (kept_windows, *new_frames) if f is not None and len(f)]
    windows = pd.concat(parts, ignore_index=True) if parts else None
    if todo:
        if len(stats):
            write_manifest(root, f"{dataset}_case_window_stats", stats.sort_values("case_id").reset_index(drop=True))
        if windows is not None:
            _write_parquet(windows_path, windows.sort_values(["case_id", "t_start_sec"]).reset_index(drop=True))
    own = stats.loc[stats.case_id.astype(str).isin({str(r["case_id"]) for r in rows})] if len(stats) else stats
    ready_col = "ready_span_min_theta_70"
    result = {"dataset": dataset, "completed_cases": len(rows), "indexed_now": len(todo), "skipped_current": len(rows) - len(todo),
              "ready_window_rows": 0 if windows is None else int(len(windows)),
              "ready_hours_theta70": round(float(own[ready_col].sum() / 60), 1) if ready_col in own else 0.0,
              "seconds": round(time.perf_counter() - started, 1), "stats_path": str(stats_path), "windows_path": str(windows_path)}
    if progress:
        print(f"READY WINDOWS [{dataset}]: {result['indexed_now']} cases indexed, {result['skipped_current']} already current; "
              f"{result['ready_window_rows']:,} window rows; {result['ready_hours_theta70']} ready hours at theta 0.70 ({result['seconds']} s).", flush=True)
    return result


def _pick(stats, names):
    for name in names:
        if stats is not None and name in stats.index and pd.notna(stats[name]):
            return stats[name]
    return np.nan


def build_unified_index(root, progress=True, datasets=("mover", "vitaldb"), auto_index=True):
    """One row per completed case of both datasets; writes data/manifests/unified_case_index.{parquet,csv}.

    ``auto_index`` runs the (incremental) ready-window pass first for any dataset whose stats file is missing.
    """
    root = Path(root)
    settings = _settings(root)
    manifests = root / "data" / "manifests"
    records = []
    for dataset in datasets:
        rows = _completed_rows(root, dataset)
        if not rows:
            continue
        path = manifests / f"{dataset}_case_window_stats.parquet"
        if not path.exists():
            if not auto_index:
                raise FileNotFoundError(f"{path.name} is missing; run index_ready_windows(ROOT, '{dataset}') first")
            index_ready_windows(root, dataset, progress=progress)
        stats = pd.read_parquet(path).drop_duplicates("case_id", keep="last").assign(case_id=lambda d: d.case_id.astype(str)).set_index("case_id")
        for row in rows:
            case_id = str(row["case_id"])
            s = stats.loc[case_id] if case_id in stats.index else None
            record = {"dataset": dataset, "case_id": case_id, "patient_id": str(row.get("patient_id")), "patient_key": f"{dataset}:{row.get('patient_id')}",
                      "split_within_dataset": row.get("split") if row.get("split") in SPLITS else "unassigned",
                      "time_reference": row.get("time_reference"), "duration_sec": row.get("duration_sec"),
                      "waveform_rows": row.get("waveform_rows"), "map_rows": row.get("map_rows"), "has_window_stats": s is not None,
                      "waveform_path": row["waveform_output_path"], "map_path": row["map_output_path"],
                      "waveform_sha256": row.get("waveform_sha256"), "map_sha256": row.get("map_sha256"), "processed_bytes": row.get("processed_bytes")}
            for name in _STAT_COLUMNS:
                record[name] = _pick(s, [name])
            for channel in ("art", "ecg", "ppg"):
                if np.isnan(record[f"valid_{channel}"]):
                    miss = _pick(s, [f"{channel}_missingness"])
                    record[f"valid_{channel}"] = 1 - miss if not np.isnan(miss) else np.nan
            for theta in _THETAS:
                record[f"ready_span_min_theta_{theta}"] = _pick(s, [f"ready_span_min_theta_{theta}"])
            records.append(record)
    if not records:
        raise ValueError("No completed cases found for the requested datasets")
    index = pd.DataFrame(records)
    if index.duplicated(["dataset", "case_id"]).any():
        raise ValueError("Duplicate dataset/case_id rows in the unified index")
    pooled = patient_splits(index.patient_key.unique(), settings["patient_split_seed"])
    index = index.merge(pooled.rename(columns={"patient_id": "patient_key", "split": "split_pooled"}), on="patient_key", how="left", validate="many_to_one")
    assert_patient_independence(index[["patient_key", "split_pooled"]].drop_duplicates().rename(columns={"patient_key": "patient_id", "split_pooled": "split"}))
    for dataset, group in index.groupby("dataset"):
        labelled = group.loc[group.split_within_dataset.isin(SPLITS), ["patient_key", "split_within_dataset"]].drop_duplicates()
        assert_patient_independence(labelled.rename(columns={"patient_key": "patient_id", "split_within_dataset": "split"}))
    index = index.sort_values(["dataset", "case_id"]).reset_index(drop=True)
    write_manifest(root, "unified_case_index", index)
    if progress:
        summary = index.groupby(["dataset", "split_within_dataset"]).agg(cases=("case_id", "size"), patients=("patient_key", "nunique"),
                                                                          ready_hours_theta70=("ready_span_min_theta_70", lambda x: round(x.sum() / 60, 1)))
        print(f"UNIFIED INDEX ({utc_now()}): {len(index):,} cases, {index.patient_key.nunique():,} patients\n{summary}", flush=True)
    return index


def experiment_split(index, name):
    """Train/validation/test case tables for a named experiment (see EXPERIMENTS)."""
    if name not in EXPERIMENTS:
        raise KeyError(f"Unknown experiment {name!r}; choose from {sorted(EXPERIMENTS)}")
    within = index.split_within_dataset
    m, v = index.dataset.eq("mover"), index.dataset.eq("vitaldb")
    if name == "E1_pooled":
        return {s: index.loc[index.split_pooled.eq(s)] for s in SPLITS}
    if name == "E2_mover_to_vitaldb":
        return {"train": index.loc[m & within.eq("train")], "validation": index.loc[v & within.eq("validation")], "test": index.loc[v & within.eq("test")]}
    if name == "E3_vitaldb_to_mover":
        return {"train": index.loc[v & within.eq("train")], "validation": index.loc[m & within.eq("validation")], "test": index.loc[m & within.eq("test")]}
    return {"train": index.loc[v & within.eq("train")], "validation": index.loc[v & within.eq("validation")], "test": index.loc[m]}


def parquet_paths(root, frame, which="waveform"):
    """Absolute parquet paths for a case table, ready for spark.read.parquet(*paths)."""
    column = {"waveform": "waveform_path", "map": "map_path"}[which]
    return [str((Path(root) / p).resolve()) for p in frame[column]]
