"""Parallel, resumable beat/SQI extraction over every case of the unified index.

One task per case (largest first) in a spawn process pool; each worker reads the case waveform, runs ``beats.process_case`` on the union of the
case's ready windows and writes ``data/derived/beats/<dataset>/case_id=<id>/{beats,beat_waveforms}.parquet`` atomically, then a
``beats_meta.json`` sidecar LAST. A case is finished only when that sidecar matches the current config hash and source waveform hash, so
an interrupted run loses at most the in-flight cases and a rerun skips everything finished. Status banners match the other pipelines:
COMPLETE / PAUSED / INCOMPLETE / FAILED.
"""
from __future__ import annotations
# Extract and store beat measurements one case at a time, with restart support.

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from . import beats as B
from .canonical import DiskSpaceStop, atomic_json, ensure_disk_space, safe_id, utc_now, write_manifest
from .snapshot import assert_dataset_v1

BEATS_CODE_VERSION = "bt3"          # bump when detector/feature semantics change so finished cases are recomputed
OUT_DIR = Path("data") / "derived" / "beats"
DEFAULTS = {"version": "v1", "min_valid_frac": 0.5, "span_pad_sec": 10, "phase_samples": 200, "workers": 20, "store_waveforms": True,
            "waveform_dtype": "float16", "max_projected_waveform_gb": 60, "min_free_after_gb": 150}


def load_beats_config(settings=None):
    """Merge beat-extraction defaults with the central configuration."""
    return {**DEFAULTS, **((settings or {}).get("beats") or {})}


def config_hash(cfg):
    """Hash detector and storage settings that determine reusable case outputs."""
    keep = {k: cfg[k] for k in ("version", "min_valid_frac", "span_pad_sec", "phase_samples", "store_waveforms", "waveform_dtype")}
    return hashlib.sha256(json.dumps({**keep, "code": BEATS_CODE_VERSION}, sort_keys=True).encode()).hexdigest()


def case_dir(root, dataset, case_id):
    """Resolve the validated dataset and case directory for beat-level outputs."""
    return Path(root) / OUT_DIR / safe_id(dataset) / f"case_id={safe_id(case_id)}"


def _read_settings(root):
    return json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))


def _atomic_parquet(table_or_frame, path):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        if isinstance(table_or_frame, pd.DataFrame):
            table_or_frame = pa.Table.from_pandas(table_or_frame, preserve_index=False)
        pq.write_table(table_or_frame, temp, compression="zstd", row_group_size=65536)
        rows = pq.ParquetFile(temp).metadata.num_rows
        if rows != table_or_frame.num_rows:
            raise ValueError("parquet readback row count mismatch")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def waveform_table(df, waves, n_phase):
    """Encode phase-normalized waveform arrays as fixed-size Arrow list columns."""
    cols = {"beat_idx": pa.array(df["beat_idx"].to_numpy().astype(np.int32))}
    for name in ("ecg", "art", "ppg"):
        flat = np.ascontiguousarray(waves[name], dtype=np.float16).reshape(-1)
        cols[name] = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float16()), n_phase)
    return pa.table(cols)


def _meta_ok(meta_path, chash, source_sha):
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return meta.get("config_hash") == chash and meta.get("waveform_sha256") == source_sha
    except (OSError, ValueError):
        return False


def _job(root, row, win_start, win_end, cfg, chash):
    """Process one case; never raises (errors are returned)."""
    started = time.perf_counter()
    try:
        # Read the stored table required for this calculation.
        root = Path(root)
        w = pq.read_table(root / row["waveform_path"], columns=["time_sec", "ART", "ECG", "PPG"]).to_pandas()
        df, waves, meta = B.process_case(w["time_sec"].to_numpy(), w["ART"].to_numpy(), w["ECG"].to_numpy(), w["PPG"].to_numpy(), win_start, win_end,
                                         n_phase=int(cfg["phase_samples"]), pad_sec=float(cfg["span_pad_sec"]), store_waveforms=bool(cfg["store_waveforms"]))
        directory = case_dir(root, row["dataset"], row["case_id"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "beats_meta.json").unlink(missing_ok=True)                 # invalidate before rewriting
        if len(df):
            _atomic_parquet(df, directory / "beats.parquet")
            if cfg["store_waveforms"]:
                _atomic_parquet(waveform_table(df, waves, int(cfg["phase_samples"])), directory / "beat_waveforms.parquet")
        else:
            for name in ("beats.parquet", "beat_waveforms.parquet"):
                (directory / name).unlink(missing_ok=True)
        # Calculate output values for the following step.
        sizes = {n: (directory / n).stat().st_size if (directory / n).exists() else 0 for n in ("beats.parquet", "beat_waveforms.parquet")}
        v = df.loc[df.valid_beat] if len(df) else df
        out = {"dataset": row["dataset"], "case_id": str(row["case_id"]), "n_beats": int(len(df)), "n_valid": int(len(v)),
               "valid_frac": float(len(v) / len(df)) if len(df) else 0.0, "art_ok_frac": float(df.art_ok.mean()) if len(df) else 0.0,
               "ppg_ok_frac": float(df.ppg_ok.mean()) if len(df) else 0.0,
               "sqi_ecg_median": float(df.sqi_ecg.median()) if len(df) else None, "sqi_art_median": float(df.sqi_art.median()) if len(df) else None,
               "sqi_ppg_median": float(df.sqi_ppg.median()) if len(df) else None, "hr_median": float(df.hr_bpm.median()) if len(df) else None,
               "ptt_ecg_art_median": float(df.ptt_ecg_art.median()) if len(df) else None, "ptt_ecg_ppg_median": float(df.ptt_ecg_ppg.median()) if len(df) else None,
               "bytes_scalar": int(sizes["beats.parquet"]), "bytes_waveforms": int(sizes["beat_waveforms.parquet"]),
               "ecg_polarity": meta.get("ecg_polarity"), "ppg_polarity": meta.get("ppg_polarity"), "regions": int(meta.get("regions", 0)),
               "seconds": round(time.perf_counter() - started, 2), "beats_version": BEATS_CODE_VERSION, "config_hash": chash,
               "waveform_sha256": row["waveform_sha256"], "written_at": utc_now()}
        atomic_json(directory / "beats_meta.json", out)
        return {"status": "ok", "meta": out}
    except DiskSpaceStop as exc:
        return {"status": "disk_stop", "case_id": str(row["case_id"]), "dataset": row["dataset"], "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "case_id": str(row["case_id"]), "dataset": row["dataset"], "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-600:]}"}


class _Lock:
    def __init__(self, root):
        self.path = Path(root) / "data" / "checkpoints" / "beats" / "run.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Accumulate the calculation over the required samples or iterations.
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    # Load the recorded settings or metadata for this processing step.
                    info = json.loads(self.path.read_text())
                    alive = (info["pid"] != os.getpid() and psutil.pid_exists(info["pid"])
                             and abs(psutil.Process(info["pid"]).create_time() - info["create_time"]) < 2)
                except Exception:
                    alive = False
                if alive:
                    raise RuntimeError(f"Another beat-extraction run is active (PID {info['pid']}). Interrupt that cell first; never run two at once.")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as handle:
                json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time()}, handle)
            return self
        raise RuntimeError("Could not acquire the beat-extraction lock")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _banner(status, text, enabled=True):
    if enabled:
        print(f"\n===== {status} =====\n{text}\n", flush=True)


def collect_manifest(root, datasets=("mover", "vitaldb"), write=True):
    """Rebuild data/manifests/beats_manifest.parquet from the per-case sidecars."""
    rows = []
    for ds in datasets:
        for p in sorted((Path(root) / OUT_DIR / ds).glob("case_id=*/beats_meta.json")):
            try:
                rows.append(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
    frame = pd.DataFrame(rows)
    if write and len(frame):
        write_manifest(root, "beats_manifest", frame.sort_values(["dataset", "case_id"]).reset_index(drop=True))
    return frame


def _selection(root, datasets, case_ids):
    index = pd.read_parquet(Path(root) / "data" / "manifests" / "unified_case_index.parquet")
    index = index.loc[index.dataset.isin(datasets)]
    if case_ids is not None:
        keep = {(d, str(c)) for d, c in case_ids}
        index = index.loc[[(d, str(c)) in keep for d, c in zip(index.dataset, index.case_id)]]
    return index.sort_values("n_ready_windows", ascending=False, kind="stable")


def run(root, datasets=("mover", "vitaldb"), workers=None, progress=True, case_ids=None, max_cases=None, allow_changed=False, check_projection=True):
    """Extract beats for every selected case that is not already finished. ``case_ids`` = iterable of (dataset, case_id)."""
    # Calculate project paths and the run settings for the following step.
    started = time.perf_counter()
    root = Path(root)
    assert_dataset_v1(root, allow_changed=allow_changed)
    settings = _read_settings(root)
    cfg = load_beats_config(settings)
    chash = config_hash(cfg)
    index = _selection(root, datasets, case_ids)
    windows = {}
    for ds in datasets:
        # Summarize the records separately within each requested group.
        w = pd.read_parquet(root / "data" / "manifests" / f"ready_windows_{ds}.parquet", columns=["case_id", "t_start_sec", "t_end_sec", "min_valid_frac", "is_ready"])
        w = w.loc[w.is_ready & (w.min_valid_frac >= cfg["min_valid_frac"] - 1e-6)].astype({"case_id": str})
        windows[ds] = {k: (g.t_start_sec.to_numpy(), g.t_end_sec.to_numpy()) for k, g in w.groupby("case_id")}
    todo, skipped, no_windows = [], 0, 0
    for r in index.itertuples():
        win = windows[r.dataset].get(str(r.case_id))
        if win is None:
            no_windows += 1
            continue
        if _meta_ok(case_dir(root, r.dataset, r.case_id) / "beats_meta.json", chash, r.waveform_sha256):
            skipped += 1
            continue
        todo.append(({"dataset": r.dataset, "case_id": str(r.case_id), "waveform_path": r.waveform_path, "waveform_sha256": r.waveform_sha256}, win))
    if max_cases is not None:
        todo = todo[:max_cases]
    if progress:
        print(f"BEATS: {len(todo)} case(s) to process, {skipped} already finished, {no_windows} without ready windows "
              f"(phase_samples={cfg['phase_samples']}, waveforms={cfg['store_waveforms']}, config {chash[:8]}).", flush=True)
    workers_n = max(1, min(int(workers or cfg["workers"]), max(1, (os.cpu_count() or 8) - 4), max(1, len(todo))))
    reserve = int(float(settings.get("min_free_space_gb", 100)) * 1e9)
    if check_projection and todo:
        proj_path = root / "results" / "beats_projection.json"
        if proj_path.exists():
            # Load the recorded settings or metadata for this processing step.
            proj = json.loads(proj_path.read_text(encoding="utf-8"))
            need = float(proj.get("remaining_gb_estimate", 0)) * 1e9
            free = ensure_disk_space(root, 0)
            if free - need < float(cfg["min_free_after_gb"]) * 1e9:
                msg = (f"Projected {need / 1e9:.0f} GB more output would leave {(free - need) / 1e9:.0f} GB free (< {cfg['min_free_after_gb']} GB). "
                       "Free space or lower beats.phase_samples / set store_waveforms=false in configs/hemodynamics.json.")
                _banner("FAILED", msg, progress)
                return {"status": "FAILED", "reason": msg}
    done, failed, bytes_out, beats_out = 0, [], 0, 0
    status, reason = "COMPLETE", "all selected cases finished"
    bar = None
    if progress and todo:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(todo), desc="beats", unit="case")
    pending, iterator, interrupted = {}, iter(todo), 0
    stop = False
    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        with _Lock(root), cf.ProcessPoolExecutor(max_workers=workers_n, mp_context=multiprocessing.get_context("spawn")) as pool:
            while True:
                try:
                    while not stop and len(pending) < workers_n * 2:
                        try:
                            row, win = next(iterator)
                        except StopIteration:
                            break
                        ensure_disk_space(root, reserve, additional_bytes=(len(pending) + 1) * 200_000_000)
                        pending[pool.submit(_job, str(root), row, win[0], win[1], cfg, chash)] = row
                    if not pending:
                        break
                    finished, _ = cf.wait(pending, timeout=0.5, return_when=cf.FIRST_COMPLETED)
                    # Process the selected records in order and retain their individual results.
                    for fut in finished:
                        row = pending.pop(fut)
                        try:
                            res = fut.result()
                        except Exception as exc:                          # e.g. BrokenProcessPool: record, do not abort the run
                            res = {"status": "failed", "dataset": row["dataset"], "case_id": row["case_id"], "error": f"{type(exc).__name__}: {exc}"}
                        if res["status"] == "ok":
                            done += 1
                            bytes_out += res["meta"]["bytes_scalar"] + res["meta"]["bytes_waveforms"]
                            beats_out += res["meta"]["n_beats"]
                        elif res["status"] == "disk_stop":
                            stop, status, reason = True, "PAUSED", res["error"]
                        else:
                            failed.append(res)
                        if bar:
                            bar.update(1)
                            bar.set_postfix(beats=f"{beats_out:,}", GB=f"{bytes_out / 1e9:.2f}", failed=len(failed), refresh=False)
                except DiskSpaceStop as exc:
                    stop, status, reason = True, "PAUSED", str(exc)
                except KeyboardInterrupt:
                    interrupted += 1
                    stop, status, reason = True, "PAUSED", "Interrupted by user"
                    if interrupted == 1:
                        print("Stop requested: finishing the cases in flight (interrupt again to abandon them) ...", flush=True)
                    else:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
    except KeyboardInterrupt:
        status, reason = "PAUSED", "Interrupted by user"
    finally:
        bar and bar.close()
    if failed:
        # Save the comparison table so the same measurements can be inspected later.
        errors = pd.DataFrame([{"timestamp": utc_now(), "dataset": f["dataset"], "case_id": f["case_id"], "error": f["error"][:900]} for f in failed])
        path = root / "data" / "manifests" / "beats_errors.csv"
        errors.to_csv(path, mode="a", header=not path.exists(), index=False)
        if status == "COMPLETE":
            status, reason = "INCOMPLETE", f"{len(failed)} case(s) failed (see data/manifests/beats_errors.csv); rerun to retry them"
    manifest = collect_manifest(root, datasets)
    seconds = time.perf_counter() - started
    result = {"status": status, "reason": reason, "processed": done, "skipped_finished": skipped, "failed": len(failed), "beats_written": beats_out,
              "gb_written": round(bytes_out / 1e9, 3), "seconds": round(seconds, 1), "cases_per_min": round(done / seconds * 60, 1) if seconds else None,
              "workers": workers_n, "manifest_rows": int(len(manifest)), "config_hash": chash}
    if status == "COMPLETE" and not len(todo):
        result["reason"] = "nothing to do: every selected case is already finished"
    atomic_json(root / "results" / "beats_run.json", result)
    _banner(status, f"{reason}\n{done} processed, {skipped} already finished, {len(failed)} failed | {beats_out:,} beats, {bytes_out / 1e9:.2f} GB in {seconds / 60:.1f} min "
                    f"({result['cases_per_min']} cases/min with {workers_n} workers).", progress)
    return result


def load_beats(root, dataset, case_id, columns=None):
    """Load the stored beat-feature table for one case."""
    return pq.read_table(case_dir(root, dataset, case_id) / "beats.parquet", columns=columns).to_pandas()


def load_beat_waveforms(root, dataset, case_id):
    """Load stored phase-normalized beat waveforms for one case."""
    # Read the stored table required for this calculation.
    t = pq.read_table(case_dir(root, dataset, case_id) / "beat_waveforms.parquet")
    n = t.num_rows
    return {k: np.asarray(t[k].combine_chunks().values.to_numpy(zero_copy_only=False), dtype=np.float16).reshape(n, -1) for k in ("ecg", "art", "ppg")}, t["beat_idx"].to_numpy()


def project(root, write=True):
    """Extrapolate disk and wall time for ALL cases from the finished sample (clearly an estimate)."""
    # Read the stored table required for this calculation.
    root = Path(root)
    settings = _read_settings(root)
    cfg = load_beats_config(settings)
    manifest = collect_manifest(root, write=False)
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet")
    if len(manifest) < 10:
        raise ValueError("Need at least 10 finished cases for a projection (run the dry run first)")
    # Join related tables by dataset, case id to keep measurements aligned.
    m = manifest.merge(index[["dataset", "case_id", "ready_span_min_theta_50"]].astype({"case_id": str}), on=["dataset", "case_id"], how="left")
    m = m.loc[m.ready_span_min_theta_50 > 0]
    hours = m.ready_span_min_theta_50.sum() / 60
    beats_per_hour = m.n_beats.sum() / hours
    total_hours = index.ready_span_min_theta_50.sum() / 60
    # Calculate output values for the following step.
    total_beats = beats_per_hour * total_hours
    scalar_per_beat = m.bytes_scalar.sum() / max(1, m.n_beats.sum())
    wave_per_beat = m.bytes_waveforms.sum() / max(1, m.n_beats.sum())
    finished_gb = float((manifest.bytes_scalar.sum() + manifest.bytes_waveforms.sum()) / 1e9)
    # Calculate output values for the following step.
    remaining = max(0.0, total_beats * (scalar_per_beat + wave_per_beat) / 1e9 - finished_gb)
    sec_per_case = float(manifest.seconds.mean())
    out = {"sample_cases": int(len(manifest)), "sample_ready_hours": round(hours, 1), "beats_per_ready_hour": round(beats_per_hour),
           "total_ready_hours": round(total_hours, 1), "projected_total_beats": int(total_beats),
           "bytes_per_beat_scalar": round(scalar_per_beat, 1), "bytes_per_beat_waveforms": round(wave_per_beat, 1),
           "projected_scalar_gb": round(total_beats * scalar_per_beat / 1e9, 4), "projected_waveform_gb": round(total_beats * wave_per_beat / 1e9, 4),
           "projected_total_gb": round(total_beats * (scalar_per_beat + wave_per_beat) / 1e9, 4), "remaining_gb_estimate": round(remaining, 4),
           "mean_seconds_per_case_in_worker": round(sec_per_case, 2), "workers": int(cfg["workers"]),
           "projected_wall_minutes_extrapolation": round(len(index) * sec_per_case / int(cfg["workers"]) / 60, 1),
           "phase_samples": int(cfg["phase_samples"]), "store_waveforms": bool(cfg["store_waveforms"]),
           "note": "Extrapolation from a sample; real time also depends on disk and the other processes running."}
    if write:
        atomic_json(root / "results" / "beats_projection.json", out)
    return out


def apply_storage_budget(root, projection=None):
    """If the projected waveform output exceeds beats.max_projected_waveform_gb: halve the phase samples (min 100), then drop waveforms.

    Rewrites only the additive 'beats' keys in configs/hemodynamics.json and returns what it did (finished cases are recomputed on the next run).
    """
    # Load the recorded settings or metadata for this processing step.
    root = Path(root)
    path = root / "configs" / "hemodynamics.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    # Calculate the run settings for the following step.
    cfg = load_beats_config(settings)
    proj = projection or project(root)
    limit = float(cfg["max_projected_waveform_gb"])
    actions = []
    waveform_gb = proj["projected_waveform_gb"]
    phase = int(cfg["phase_samples"])
    if cfg["store_waveforms"] and waveform_gb > limit:
        if phase > 100:
            waveform_gb *= 100 / phase
            phase = 100
            actions.append(f"phase_samples {cfg['phase_samples']} -> 100 (the loader can upsample to 200)")
        if waveform_gb > limit:
            actions.append("store_waveforms -> false (beats must be resampled on the fly from waveform.parquet)")
    if actions:
        # Store the settings and results as JSON for later inspection.
        settings.setdefault("beats", {})
        settings["beats"]["phase_samples"] = phase
        settings["beats"]["store_waveforms"] = not any("store_waveforms" in a for a in actions)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return {"changed": bool(actions), "actions": actions, "projected_waveform_gb_after": round(float(waveform_gb) if cfg["store_waveforms"] else 0.0, 2), "limit_gb": limit}
