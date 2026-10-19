"""One-pass, parallel MOVER ingestion.

archive readers (one sequential gzip/tar stream per archive) -> bounded pool of
decode worker processes -> verified Parquet fragments + ONE checkpoint
coordinator -> case assembly -> patient-level selection/splits -> fast EDA.

Why this is fast without changing the science
* Arterial MAP exists only in GE_S5 members, and every inspected S5 member has a
  PLETH group; eligibility needs PPG and MAP coverage, so a member without a
  PLETH* wave group can contribute neither and is skipped by a raw byte search
  (no XML parse). Its ART/ECG-only samples are therefore not part of the
  canonical streams (see docs). Members of patients with no usable surgery and
  members outside every surgery window are skipped too.
* Each retained member is parsed/decoded ONCE by the trusted ``decode_xml`` and
  its verified fragments are reused for quality assessment and the final case
  files, so no archive is reopened for a pilot, selection or conversion pass.
* Decoding/finalization run in a bounded multiprocessing pool; gzip inflate
  (inherently sequential per archive) is the remaining floor, run for the three
  archives concurrently.

Windows uses ``spawn``: call ``run``/``ingest`` from a notebook or from a script
guarded by ``if __name__ == "__main__":`` (an unguarded script would re-run itself
in every worker).
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import gzip
import hashlib
import json
import multiprocessing
import os
import queue
import re
import shutil
import signal
import sqlite3
import tarfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .canonical import (MAP_COLUMNS, WAVE_COLUMNS, CheckpointStore, DiskSpaceStop, _json_value,
                        assert_patient_independence, atomic_json, ensure_disk_space, load_case,
                        patient_splits, safe_id, sha256_file, utc_now, verify_case, write_case,
                        write_manifest)
from .mover_pipeline import (_assemble, _assert_observed_pressure_schema, _cases, _error,
                             _processing_signature, _reserve, load_config, quality)
from .mover_signals import decode_xml
from .windows import case_summary, choose_rung, load_ladder, load_rule, ready_windows, rung_passes, theta_key

FAST_DEFAULTS = {
    "decode_workers": 16, "finalize_workers": 20, "archive_readers": 3,
    "max_inflight_payload_mb": 6144, "max_inflight_tasks": 64, "min_available_ram_gb": 16,
    "tar_buffer_kb": 256, "member_window_pad_sec": 120, "commit_batch_members": 100,
    "commit_interval_sec": 2.0, "progress_interval_sec": 1.0, "staging_cap_gb": 150,
    "worker_recycle_tasks": 1000, "canary_min_fragments": 20, "canary_max_violation_rate": 0.05,
    "canary_give_up_fragments": 300, "canary_min_failures": 25, "canary_max_failure_rate": 0.5,
    "canary_art_map_tolerance_mmhg": 30.0,
    "keep_staging": False, "concurrent_vitaldb_worker_scale": 0.5,
    "debug_task_delay_sec": 0.0, "stage_requires_all_waveforms": True,
}
STATE_DB = "fast_ingest.sqlite"
INGEST_PROFILE = "window_v1"
DONE_STATES = ("done", "no_signal")
_PLETH = b'name="PLETH'
_DATETIME = re.compile(rb'datetime="([^"]+)"')
_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
_KEEP_METADATA = ("group_names", "groups", "original_fs", "numeric_channel_counts", "waveform_counts",
                  "selected_tracks", "target_fs", "overlap_conflicting_samples", "duplicate_wave_packets",
                  "conflicting_wave_timestamps", "duplicate_map_samples", "conflicting_map_timestamps",
                  "waveform_valid_samples", "map_valid_samples", "map_channel_identity",
                  "map_mapping_evidence", "timestamp_source", "scaling_validation", "native_unit_evidence")


def _fast(config):
    merged = dict(FAST_DEFAULTS)
    merged.update(config.get("fast_ingest", {}))
    return merged


def _window_mode(config):
    return config.get("eligibility_mode", "legacy_case") == "window_v1"


def _profile(config):
    """Ingest profile stamped on archive markers so legacy 'completed' markers never short-circuit the window pass."""
    return INGEST_PROFILE if _window_mode(config) else None


def _short(name):
    return name.replace("epic_", "").replace("_v2.tar.gz", "")


def _say(enabled, text):
    if enabled:
        print(text, flush=True)


def _banner(status, text, enabled=True):
    _say(enabled, f"\n===== {status} =====\n{text}\n")



# cheap member screening (no XML parse)

def _iso_epoch(raw):
    stamp = datetime.fromisoformat(raw.decode("ascii").replace("Z", "+00:00"))
    if stamp.utcoffset() is None:
        raise ValueError("naive timestamp")
    return stamp.timestamp()


def member_span(payload):
    """(start_utc, end_utc) from the first/last cpc timestamp, or None when unsure."""
    first = _DATETIME.search(payload)
    index = payload.rfind(b'datetime="')
    if first is None or index < 0:
        return None
    last = _DATETIME.match(payload, index)
    if last is None:
        return None
    try:
        a, b = _iso_epoch(first.group(1)), _iso_epoch(last.group(1))
    except (ValueError, UnicodeDecodeError):
        return None
    start, end = min(a, b), max(a, b) + 1.0
    return None if end - start > 3600.0 else (start, end)


def match_cases(entries, span, pad):
    """Cases whose surgery window overlaps the (padded) member span; all if span unknown."""
    if span is None:
        return list(entries)
    low, high = span[0] - pad, span[1] + pad
    return [e for e in entries if e[1] < high and e[2] > low]



# worker side

_W = {}


def _worker_init(root, config, signature):
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # the coordinator owns shutdown
    except (ValueError, OSError):
        pass
    _W.update(root=Path(root), config=config, fast=_fast(config), signature=signature)
    try:
        pa.set_cpu_count(2)
        pa.set_io_thread_count(2)
    except Exception:
        pass


def _trim_metadata(metadata):
    return {k: metadata[k] for k in _KEEP_METADATA if k in metadata}


def _frame32(frame, columns):
    out = frame.copy()
    out["time_sec"] = out["time_sec"].astype("float64")
    for column in columns[1:]:
        out[column] = out[column].astype("float32")
    return out


def _write_parquet_atomic(frame, path):
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("wb") as handle:
            frame.to_parquet(handle, index=False, compression="zstd")
        with pq.ParquetFile(temp) as parquet:
            if parquet.metadata.num_rows != len(frame):
                raise ValueError("Fragment readback row count mismatch")
        digest = sha256_file(temp)
        os.replace(temp, path)
        return digest
    finally:
        temp.unlink(missing_ok=True)


def _stats(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return 0, None, None
    return int(finite.size), float(np.median(finite)), float(finite.std())


def _sanity(waveform, numeric):
    out = {}
    for name in ("ART", "ECG", "PPG"):
        n, median, std = _stats(waveform[name].to_numpy(dtype=float) if len(waveform) else [])
        out[f"{name.lower()}_n"], out[f"{name.lower()}_median"], out[f"{name.lower()}_std"] = n, median, std
    n, median, _ = _stats(numeric["MAP"].to_numpy(dtype=float) if len(numeric) else [])
    out["map_n"], out["map_median"] = n, median
    out["all_four"] = all(out[k] > 0 for k in ("art_n", "ecg_n", "ppg_n", "map_n"))
    return out


def _has_signal(sanity):
    return any(sanity[k] > 0 for k in ("art_n", "ecg_n", "ppg_n", "map_n"))


def _stage_worthy(sanity, config, fast):
    """Window mode stages only members that can contribute a ready window (all three waveforms present)."""
    if _window_mode(config) and fast.get("stage_requires_all_waveforms", True):
        return all(sanity[k] > 0 for k in ("art_n", "ecg_n", "ppg_n"))
    return _has_signal(sanity)


def _write_fragment(root, config, archive, member, case_id, patient_id, waveform, numeric, metadata, sanity, source_bytes, signature):
    digest = hashlib.sha256(f"{archive}\0{member}".encode()).hexdigest()[:24]
    directory = root / "data" / "parquet" / "mover" / "staging" / f"case_id={safe_id(case_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    wave, numeric = _frame32(waveform, WAVE_COLUMNS), _frame32(numeric, MAP_COLUMNS)
    ensure_disk_space(root, _reserve(config), int(wave.memory_usage(index=False).sum() + numeric.memory_usage(index=False).sum()))
    info = {"case_id": case_id, "patient_id": patient_id, "archive": archive, "member": member,
            "source_bytes": source_bytes, "waveform_rows": len(wave), "map_rows": len(numeric),
            "processing_signature": signature, "sanity": sanity, "metadata": _trim_metadata(metadata)}
    for label, frame in (("waveform", wave), ("map", numeric)):
        path = directory / f"{digest}_{label}.parquet"
        info[f"{label}_sha256"] = _write_parquet_atomic(frame, path)
        info[f"{label}_path"] = str(path.relative_to(root))
    info["output_bytes"] = sum((root / info[f"{label}_path"]).stat().st_size for label in ("waveform", "map"))
    return info


def _ingest_task(archive, member, payload, cases):
    started = time.perf_counter()
    written = []
    try:
        config, root, fast = _W["config"], _W["root"], _W["fast"]
        if fast["debug_task_delay_sec"]:
            time.sleep(float(fast["debug_task_delay_sec"]))
        fragments = []
        for case_id, patient_id, start, end in cases:
            waveform, numeric, metadata = decode_xml(payload, float(start), float(end), config["target_waveform_fs"],
                                                     track_map=config.get("mover_track_map"))
            _assert_observed_pressure_schema(metadata)
            sanity = _sanity(waveform, numeric)
            if not _stage_worthy(sanity, config, fast):
                continue
            info = _write_fragment(root, config, archive, member, case_id, patient_id, waveform, numeric, metadata,
                                   sanity, len(payload), _W["signature"])
            written += [root / info["waveform_path"], root / info["map_path"]]
            fragments.append(info)
        return {"status": "done" if fragments else "no_signal", "fragments": fragments,
                "seconds": round(time.perf_counter() - started, 4)}
    except DiskSpaceStop as exc:
        status, error = "disk_stop", str(exc)
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
    for path in written:
        path.unlink(missing_ok=True)
    return {"status": status, "error": error, "fragments": [], "seconds": round(time.perf_counter() - started, 4)}


def _py(value):
    return _json_value(value)


def _assess_task(case, fragments):
    try:
        row = pd.Series(case)
        wave, numeric, extra = _assemble(_W["root"], row, fragments, _W["config"])
        assessment = quality(row, wave, numeric, _W["config"])
        return {"status": "assessed", "case_id": case["case_id"], "assessment": _py(assessment),
                "estimated_bytes": int(sum(f["output_bytes"] for f in fragments)),
                "duplicate_conflict_count": extra["duplicate_conflict_count"]}
    except DiskSpaceStop as exc:
        return {"status": "disk_stop", "case_id": case["case_id"], "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}


def _window_assessment(row, wave, numeric, config):
    """Legacy whole-case metrics plus window-level readiness; the cohort rung is decided later, case by case."""
    assessment = quality(row, wave, numeric, config)
    rule = load_rule(config)
    windows = ready_windows(wave["ART"].to_numpy(), wave["ECG"].to_numpy(), wave["PPG"].to_numpy(), config["target_waveform_fs"],
                            numeric["time_sec"].to_numpy(), numeric["MAP"].to_numpy(), rule,
                            t0=float(wave["time_sec"].iloc[0]) if len(wave) else 0.0, case_id=str(row["case_id"]))
    assessment = {**assessment, **case_summary(windows, rule), "legacy_whole_case_pass": bool(assessment["final_usable"]),
                  "final_usable": False, "provisional_usable": False}
    return assessment, windows


def _assess_window_task(case, fragments):
    try:
        row = pd.Series(case)
        wave, numeric, extra = _assemble(_W["root"], row, fragments, _W["config"])
        assessment, windows = _window_assessment(row, wave, numeric, _W["config"])
        return {"status": "assessed", "case_id": case["case_id"], "assessment": _py(assessment), "windows": windows,
                "estimated_bytes": int(sum(f["output_bytes"] for f in fragments)),
                "duplicate_conflict_count": extra["duplicate_conflict_count"]}
    except DiskSpaceStop as exc:
        return {"status": "disk_stop", "case_id": case["case_id"], "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}


def _assess_stored_task(case, stored):
    """A verified case file already on disk is authoritative: assess it directly, no archive data needed."""
    try:
        root = _W["root"]
        wave = pd.read_parquet(root / stored["waveform_output_path"])
        numeric = pd.read_parquet(root / stored["map_output_path"])
        assessment, windows = _window_assessment(pd.Series(case), wave, numeric, _W["config"])
        return {"status": "assessed", "case_id": case["case_id"], "assessment": _py(assessment), "windows": windows,
                "estimated_bytes": int(stored.get("processed_bytes", 0)),
                "duplicate_conflict_count": int(stored.get("duplicate_conflict_count", 0) or 0), "reused": True}
    except Exception as exc:
        return {"status": "failed", "case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}


def _assess_any(case, payload):
    if isinstance(payload, dict):
        return _assess_stored_task(case, payload)
    return _assess_window_task(case, payload) if _window_mode(_W["config"]) else _assess_task(case, payload)


def _unique_metadata(fragments):
    seen, unique = set(), []
    for fragment in fragments:
        key = json.dumps(fragment["metadata"], sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(fragment["metadata"])
    return unique


def _write_task(case, fragments):
    try:
        config, root = _W["config"], _W["root"]
        row = pd.Series(case)
        wave, numeric, extra = _assemble(root, row, fragments, config)
        assessment = quality(row, wave, numeric, config)
        if _window_mode(config):
            assessment = {**assessment, "provisional_usable": True, "final_usable": True, "exclusion_reason": "",
                          "eligibility_mode": "window_v1"}
        elif not assessment["final_usable"]:
            return {"status": "excluded", "case_id": case["case_id"], "assessment": _py(assessment)}
        fs = config["target_waveform_fs"]
        metadata = {**extra, **assessment, "processing_signature": _W["signature"], "patient_id": case["patient_id"],
                    "split": case["split"], "duration_sec": case["duration_sec"], "start_utc": case["start_utc"],
                    "end_utc": case["end_utc"], "ingest_mode": "fast_one_pass",
                    "source_archive": sorted({f["archive"] for f in fragments}),
                    "source_member": sorted(f["member"] for f in fragments),
                    "source_bytes": int(sum(f["source_bytes"] for f in fragments)),
                    "native_metadata": _unique_metadata(fragments), "art_final_fs": fs, "ecg_final_fs": fs,
                    "ppg_final_fs": fs, "min_free_space_bytes": _reserve(config)}
        committed = write_case(root, "mover", case["case_id"], wave, numeric, _py(metadata))
        return {"status": "completed", "case_id": case["case_id"], "committed": _py(committed)}
    except DiskSpaceStop as exc:
        return {"status": "disk_stop", "case_id": case["case_id"], "error": str(exc)}
    except Exception as exc:
        return {"status": "failed", "case_id": case["case_id"], "error": f"{type(exc).__name__}: {exc}"}



# durable state (touched by the coordinator thread only)

class _State:
    def __init__(self, root):
        directory = Path(root) / "data" / "checkpoints" / "mover"
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / STATE_DB
        self.db = sqlite3.connect(self.path, timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS archives (archive TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        self.db.execute("CREATE TABLE IF NOT EXISTS members (archive TEXT, member TEXT, status TEXT NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(archive, member))")
        self.db.execute("CREATE TABLE IF NOT EXISTS fragments (archive TEXT, member TEXT, case_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(archive, member, case_id))")
        self.db.execute("CREATE INDEX IF NOT EXISTS fragments_case ON fragments(case_id)")
        self.db.commit()

    def close(self):
        try:
            self.db.commit()
        finally:
            self.db.close()

    def commit(self):
        self.db.commit()

    def meta(self, key):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)", (key, json.dumps(value, default=str)))
        self.db.commit()

    def archives(self):
        return {r[0]: json.loads(r[1]) for r in self.db.execute("SELECT archive, payload FROM archives")}

    def put_archive(self, name, payload):
        self.db.execute("INSERT OR REPLACE INTO archives VALUES (?, ?)", (name, json.dumps(payload, default=str)))
        self.db.commit()

    def known_members(self):
        query = ("SELECT archive, member, status, json_extract(payload,'$.processing_signature'), "
                 "COALESCE(json_extract(payload,'$.retry_count'),0) FROM members")
        return {(a, m): (s, sig, int(r)) for a, m, s, sig, r in self.db.execute(query)}

    def put_member(self, archive, member, status, payload):
        self.db.execute("INSERT OR REPLACE INTO members VALUES (?, ?, ?, ?)",
                        (archive, member, status, json.dumps(payload, default=str)))

    def put_fragment(self, info):
        self.db.execute("INSERT OR REPLACE INTO fragments VALUES (?, ?, ?, ?)",
                        (info["archive"], info["member"], info["case_id"], json.dumps(info, default=str)))

    def fragments_by_case(self):
        grouped = defaultdict(list)
        for (payload,) in self.db.execute("SELECT payload FROM fragments"):
            info = json.loads(payload)
            grouped[info["case_id"]].append(info)
        for infos in grouped.values():
            infos.sort(key=lambda f: (f["archive"], f["member"]))
        return grouped

    def failed_members(self):
        return [json.loads(p) for (p,) in self.db.execute("SELECT payload FROM members WHERE status='failed'")]

    def totals(self):
        counts = dict(self.db.execute("SELECT status, COUNT(*) FROM members GROUP BY status").fetchall())
        fragments, size = self.db.execute("SELECT COUNT(*), COALESCE(SUM(json_extract(payload,'$.output_bytes')),0) FROM fragments").fetchone()
        return {"members_by_status": counts, "fragments": int(fragments), "staging_bytes": int(size)}

    def clear_staging(self):
        self.db.execute("DELETE FROM fragments")
        self.db.execute("DELETE FROM members")
        self.db.commit()

    def clear_all(self):
        for table in ("meta", "archives", "members", "fragments"):
            self.db.execute(f"DELETE FROM {table}")
        self.db.commit()


class _RunLock:
    """One fast run at a time; a stale lock (dead PID) is taken over."""

    def __init__(self, root):
        self.path = Path(root) / "data" / "checkpoints" / "mover" / "fast_ingest.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                info = {}
                try:
                    info = json.loads(self.path.read_text())
                except Exception:
                    pass
                pid = info.get("pid")
                alive = False
                if pid and pid != os.getpid() and psutil.pid_exists(pid):
                    try:
                        alive = abs(psutil.Process(pid).create_time() - float(info.get("create_time", -1))) < 2
                    except (psutil.Error, ValueError):
                        alive = False
                if alive:
                    raise RuntimeError(f"Another MOVER fast run is active (PID {pid}). Interrupt that cell first; never run two at once.")
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as handle:
                json.dump({"pid": os.getpid(), "create_time": psutil.Process().create_time(), "created": utc_now()}, handle)
            return self
        raise RuntimeError("Could not acquire the MOVER fast-run lock")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def reset_fast_state(root):
    """Explicit opt-in: forget the fast pass (own staging + state DB only; never raw archives or final cases)."""
    root = Path(root)
    with _RunLock(root):
        state = _State(root)
        try:
            state.clear_all()
        finally:
            state.close()
        shutil.rmtree(root / "data" / "parquet" / "mover" / "staging", ignore_errors=True)



# ingestion

class _Context:
    def __init__(self, root, config, fast, by_patient, known, signature, stop_after):
        self.root, self.config, self.fast = Path(root), config, fast
        self.by_patient, self.candidates = by_patient, set(by_patient)
        self.known, self.signature, self.stop_after = known, signature, stop_after
        self.stop, self.cv, self.results = threading.Event(), threading.Condition(), queue.Queue()
        self.inflight_bytes = self.inflight_count = self.submitted_total = 0
        self.max_bytes = int(fast["max_inflight_payload_mb"]) * 1024 * 1024
        self.max_tasks = int(fast["max_inflight_tasks"])
        self.min_ram = int(float(fast["min_available_ram_gb"]) * 1024 ** 3)
        self.pad = float(fast["member_window_pad_sec"])
        self.counters = defaultdict(Counter)
        self.errors = {}
        self.pending = Counter()
        self.stop_reason, self.fatal, self.pool = None, False, None
        self.staging_bytes, self.staging_cap = 0, int(float(fast["staging_cap_gb"]) * 1e9)
        self.ram_low = False
        self._mem_t = 0.0
        self.canary = {"status": "pending", "checked": 0, "violations": 0, "all_four": 0}

    def request_stop(self, reason, fatal=False):
        with self.cv:
            if self.stop_reason is None:
                self.stop_reason = reason
            self.fatal = self.fatal or fatal
            self.stop.set()
            self.cv.notify_all()

    def mem_ok(self):
        now = time.monotonic()
        if now - self._mem_t > 0.5:
            self._mem_t = now
            self.ram_low = psutil.virtual_memory().available < self.min_ram
        return not self.ram_low

    def acquire(self, nbytes):
        with self.cv:
            while True:
                if self.stop.is_set():
                    return False
                too_big = self.inflight_count > 0 and self.inflight_bytes + nbytes > self.max_bytes
                if not too_big and self.inflight_count < self.max_tasks and self.mem_ok():
                    break
                self.cv.wait(0.5)
            self.inflight_bytes += nbytes
            self.inflight_count += 1
            return True

    def release(self, nbytes):
        with self.cv:
            self.inflight_bytes -= nbytes
            self.inflight_count -= 1
            self.cv.notify_all()

    def bump_pending(self, archive, delta):
        with self.cv:
            self.pending[archive] += delta


def _is_final(known, signature, max_retries):
    status, sig, retries = known
    if sig != signature:
        return False
    return status in DONE_STATES or (status == "failed" and retries >= max_retries)


def _on_done(ctx, meta, size, future):
    ctx.results.put(("result", meta, future))
    ctx.release(size)


def _reader(ctx, name):
    counters = ctx.counters[name]
    source = ctx.root / "data" / "reference" / "mover" / name
    max_member = int(ctx.config["max_member_bytes"])
    max_retries = int(ctx.config["max_retries"])
    try:
        with source.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as compressed, \
                tarfile.open(fileobj=compressed, mode="r|", bufsize=int(ctx.fast["tar_buffer_kb"]) * 1024) as archive:
            for member in archive:
                archive.members.clear()
                if ctx.stop.is_set():
                    counters["stopped"] = 1
                    return
                counters["raw"] = raw.tell()
                if not member.isfile():
                    continue
                counters["files_seen"] += 1
                if counters["files_seen"] % 256 == 1:
                    ensure_disk_space(ctx.root, _reserve(ctx.config))
                    if ctx.staging_bytes > ctx.staging_cap:
                        ctx.request_stop("Staging cap reached (fast_ingest.staging_cap_gb); finalize the cases already ingested or raise the cap")
                        counters["stopped"] = 1
                        return
                base = member.name.rsplit("/", 1)[-1]
                patient = base[:16]
                if patient not in ctx.candidates:
                    counters["skipped_patient"] += 1
                    continue
                if member.size > max_member or not base.endswith(".xml"):
                    ctx.results.put(("event", name, member.name, f"Unexpected member (size {member.size}); skipped, not decoded"))
                    counters["skipped_unexpected"] += 1
                    continue
                payload = archive.fileobj.read(member.size)
                if len(payload) != member.size:
                    raise EOFError(f"Truncated archive member {member.name}")
                counters["bytes_read"] += member.size
                if _PLETH not in payload:
                    counters["skipped_no_pleth"] += 1
                    continue
                counters["pleth_members"] += 1
                key = (name, member.name)
                prior = ctx.known.get(key)
                if prior is not None and _is_final(prior, ctx.signature, max_retries):
                    counters["resume_skipped"] += 1
                    continue
                matches = match_cases(ctx.by_patient[patient], member_span(payload), ctx.pad)
                if not matches:
                    counters["skipped_no_overlap"] += 1
                    continue
                if not ctx.acquire(len(payload)):
                    counters["stopped"] = 1
                    return
                retry = (prior[2] if prior and prior[1] == ctx.signature else 0)
                meta = (name, member.name, patient, member.size, [m[0] for m in matches], retry)
                ctx.bump_pending(name, 1)
                try:
                    future = ctx.pool.submit(_ingest_task, name, member.name, payload,
                                             [(c, patient, s, e) for c, s, e in matches])
                except Exception as exc:  # broken pool / shutdown
                    ctx.release(len(payload))
                    ctx.bump_pending(name, -1)
                    ctx.request_stop(f"Worker pool unavailable: {type(exc).__name__}: {exc}", fatal=True)
                    counters["stopped"] = 1
                    return
                counters["submitted"] += 1
                future.add_done_callback(functools.partial(_on_done, ctx, meta, len(payload)))
                with ctx.cv:
                    ctx.submitted_total += 1
                    reached = ctx.stop_after is not None and ctx.submitted_total >= ctx.stop_after
                if reached:
                    ctx.request_stop("Stopped by test hook")
            counters["raw"] = raw.tell()
            if not ctx.stop.is_set():
                for _ in iter(lambda: compressed.read(1 << 20), b""):
                    counters["raw"] = raw.tell()
                counters["eof"] = 1
    except DiskSpaceStop as exc:
        ctx.request_stop(str(exc))
        counters["stopped"] = 1
    except Exception as exc:
        ctx.errors[name] = f"{type(exc).__name__}: {exc}"
    finally:
        ctx.results.put(("reader_done", name))


class _Bars:
    def __init__(self, ctx, names, sizes, enabled):
        self.ctx, self.enabled = ctx, enabled
        self.bars = {n: tqdm(total=sizes[n], desc=f"{_short(n)} compressed", unit="B", unit_scale=True,
                             unit_divisor=1000, mininterval=0.5, disable=not enabled) for n in names}
        self.overall = tqdm(total=sum(sizes[n] for n in names), desc="overall", unit="B", unit_scale=True,
                            unit_divisor=1000, mininterval=0.5, disable=not enabled)

    def refresh(self, totals):
        if not self.enabled:
            return
        raw_total = 0
        for name, bar in self.bars.items():
            c = self.ctx.counters[name]
            raw = min(int(c["raw"]), bar.total)
            raw_total += raw
            bar.set_postfix(seen=c["files_seen"], pleth=c["pleth_members"], dec=c["decoded"], fail=c["failed"],
                            refresh=False)
            bar.update(raw - bar.n)
        ctx = self.ctx
        self.overall.set_postfix(
            pleth=sum(c["pleth_members"] for c in ctx.counters.values()), decoded=totals["decoded"],
            frags=totals["fragments"], fail=totals["failed"], q_MB=round(ctx.inflight_bytes / 1e6),
            tasks=ctx.inflight_count, RAM_free_GB=round(psutil.virtual_memory().available / 1024 ** 3, 1),
            disk_GB=round(shutil.disk_usage(ctx.root).free / 1e9, 1), stage_GB=round(ctx.staging_bytes / 1e9, 2),
            canary=ctx.canary["status"] + (" (WAITING FOR RAM)" if ctx.ram_low else ""), refresh=False)
        self.overall.update(raw_total - self.overall.n)

    def close(self):
        for bar in self.bars.values():
            bar.close()
        self.overall.close()


_PRESSURE_ABS_RANGE = (-50.0, 400.0)


def _scale_violation(sanity, fast):
    """True only for evidence of a decode scale/schema fault, not for an unusual physiological channel.

    INVP1 channel 1 is not always an arterial line: real members carry a CVP-like trace (~7 mmHg), a zeroed or
    disconnected line, or a flat line, and there the waveform still agrees with the numeric IBP_MEAN. The window rule
    rejects such windows later, so they must not abort the pass. A gain/offset fault instead appears as an impossible
    magnitude or as the waveform disagreeing with the independent numeric MAP.
    """
    low, high = _PRESSURE_ABS_RANGE
    bad = bool(sanity["art_n"] and not low <= sanity["art_median"] <= high)
    bad = bad or bool(sanity["map_n"] and not low <= sanity["map_median"] <= high)
    if sanity["art_n"] and sanity["map_n"]:
        bad = bad or abs(sanity["art_median"] - sanity["map_median"]) > float(fast["canary_art_map_tolerance_mmhg"])
    return bad or bool(sanity["ppg_n"] and (sanity["ppg_std"] is None or not np.isfinite(sanity["ppg_std"])))


def _canary(ctx, sanity, progress):
    c, fast = ctx.canary, ctx.fast
    if c["status"] != "pending":
        return
    bad = _scale_violation(sanity, fast)
    c["checked"] += 1
    c["violations"] += int(bad)
    c["all_four"] += int(sanity["all_four"])
    rate = c["violations"] / c["checked"]
    if c["checked"] >= int(fast["canary_min_fragments"]) and rate > float(fast["canary_max_violation_rate"]):
        c["status"] = "FAILED"
        ctx.request_stop(f"Canary failed: {c['violations']}/{c['checked']} real fragments violate pressure sanity ranges (schema/scale suspicion). Fragments kept for inspection.", fatal=True)
    elif c["checked"] >= int(fast["canary_min_fragments"]) and c["all_four"] > 0:
        c["status"] = "passed"
        _say(progress, f"Canary passed on {c['checked']} real fragments ({c['all_four']} with ART+ECG+PPG+MAP); ingestion continues.")
    elif c["checked"] >= int(fast["canary_give_up_fragments"]):
        c["status"] = "FAILED"
        ctx.request_stop("Canary failed: no real fragment with ART+ECG+PPG+MAP after the first fragments; stopping before wasting the pass.", fatal=True)


def _resolve_workers(fast, workers, concurrent_vitaldb, key):
    count = int(workers or fast[key])
    if concurrent_vitaldb and workers is None:
        count = max(2, int(count * float(fast["concurrent_vitaldb_worker_scale"])))
    return max(1, min(count, max(1, (os.cpu_count() or 8) - 4)))


class _ThreadEnv:
    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in _THREAD_ENV}
        for k in _THREAD_ENV:
            os.environ[k] = "1"
        return self

    def __exit__(self, *exc):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _make_pool(root, config, signature, workers, fast):
    return cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                  initializer=_worker_init, initargs=(str(root), config, signature),
                                  max_tasks_per_child=int(fast["worker_recycle_tasks"]))


def _plan(root):
    cases = _cases(root)
    usable = cases.loc[cases["linkage_usable"].astype(bool) & cases["patient_id"].notna()]
    by_patient = defaultdict(list)
    for case_id, patient, start, end in zip(usable.case_id, usable.patient_id, usable.start_utc, usable.end_utc):
        by_patient[str(patient)].append((str(case_id), float(start), float(end)))
    for entries in by_patient.values():
        entries.sort(key=lambda e: (e[1], e[0]))
    return cases, dict(by_patient)


def _archive_ok(state, signature, profile=None):
    return ((state or {}).get("status") == "completed" and state.get("gzip_crc_verified") and state.get("processing_signature") == signature
            and (profile is None or state.get("ingest_profile") == profile))


def ingest(root, *, workers=None, progress=True, concurrent_vitaldb=False, _stop_after=None):
    """Single parallel pass over the MOVER archives; returns a status dict (never raises on interrupt)."""
    root = Path(root)
    config = load_config(root)
    fast = _fast(config)
    signature = _processing_signature(config)
    started = time.perf_counter()
    workers = _resolve_workers(fast, workers, concurrent_vitaldb, "decode_workers")
    with _RunLock(root):
        _say(progress, f"PLAN: loading surgery/patient linkage ... (decode workers={workers}, readers={min(len(config['mover_archives']), int(fast['archive_readers']))}, "
                       f"payload queue<= {fast['max_inflight_payload_mb']} MB, keep >= {fast['min_available_ram_gb']} GB RAM free, disk reserve {config['min_free_space_gb']} GB)")
        _, by_patient = _plan(root)
        state = _State(root)
        try:
            archives = state.archives()
            todo = []
            for name in config["mover_archives"]:
                source = root / "data" / "reference" / "mover" / name
                fingerprint = {"size_bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
                old = archives.get(name)
                if old and old.get("fingerprint") != fingerprint:
                    raise ValueError(f"Source archive {name} changed; the fast checkpoint cannot be reused")
                if _archive_ok(old, signature, _profile(config)):
                    _say(progress, f"{name}: verified complete in an earlier run - not read again.")
                else:
                    todo.append((name, fingerprint))
            if not todo:
                return {"status": "INGEST_COMPLETE", "reason": "all archives already verified", "seconds": time.perf_counter() - started,
                        **state.totals()}
            known = state.known_members()
            ctx = _Context(root, config, fast, by_patient, known, signature, _stop_after)
            ctx.staging_bytes = state.totals()["staging_bytes"]
            sizes = {n: f["size_bytes"] for n, f in todo}
            for name, fingerprint in todo:
                state.put_archive(name, {"archive": name, "fingerprint": fingerprint, "processing_signature": signature,
                                         "ingest_profile": _profile(config),
                                         "status": "processing", "started_at": utc_now()})
            _say(progress, f"INGEST: {len(todo)} archive(s) in parallel, {len(by_patient):,} candidate patients, "
                           f"{len(known):,} members already checkpointed. Compressed-byte bars show real progress; the cell ends with COMPLETE/PAUSED/INCOMPLETE/FAILED.")
            totals = {"decoded": 0, "fragments": 0, "failed": 0}
            bars = _Bars(ctx, [n for n, _ in todo], sizes, progress)
            threads = []
            batch_items, batch_t = 0, time.monotonic()
            with _ThreadEnv():
                pool = _make_pool(root, config, signature, workers, fast)
                ctx.pool = pool
                try:
                    threads = [threading.Thread(target=_reader, args=(ctx, name), name=f"reader-{_short(name)}", daemon=True) for name, _ in todo]
                    for t in threads:
                        t.start()
                    last_ui, interrupts = 0.0, 0
                    while True:
                        try:
                            try:
                                item = ctx.results.get(timeout=0.25)
                            except queue.Empty:
                                item = None
                            if item is not None:
                                batch_items += _handle(ctx, state, item, totals, progress)
                            now = time.monotonic()
                            if batch_items and (batch_items >= int(fast["commit_batch_members"]) or now - batch_t >= float(fast["commit_interval_sec"])):
                                state.commit()
                                batch_items, batch_t = 0, now
                            if now - last_ui >= float(fast["progress_interval_sec"]):
                                bars.refresh(totals)
                                last_ui = now
                            if not any(t.is_alive() for t in threads) and ctx.inflight_count == 0 and ctx.results.empty():
                                break
                        except KeyboardInterrupt:
                            interrupts += 1
                            ctx.request_stop("Interrupted by user")
                            if interrupts == 1:
                                _say(progress, "Stop requested: finishing in-flight members and saving checkpoints (interrupt again to abandon in-flight members) ...")
                            else:
                                break
                    deadline = time.monotonic() + 300
                    for t in threads:
                        while t.is_alive() and time.monotonic() < deadline:
                            try:
                                t.join(timeout=0.5)
                            except KeyboardInterrupt:
                                ctx.request_stop("Interrupted by user")
                except BaseException as exc:
                    ctx.request_stop(f"Internal error: {type(exc).__name__}: {exc}", fatal=True)
                    raise
                finally:
                    pool.shutdown(wait=True, cancel_futures=True)
                    while not ctx.results.empty():
                        batch_items += _handle(ctx, state, ctx.results.get_nowait(), totals, progress)
                    state.commit()
            bars.refresh(totals)
            bars.close()
            return _close_ingest(ctx, state, todo, signature, totals, progress, started)
        finally:
            state.close()


def _handle(ctx, state, item, totals, progress):
    kind = item[0]
    if kind == "event":
        _error(ctx.root, "fast_ingest", item[1], item[2], None, item[3])
        return 0
    if kind == "reader_done":
        return 0
    _, meta, future = item
    archive, member, patient, member_bytes, case_ids, retry = meta
    counters = ctx.counters[archive]
    ctx.bump_pending(archive, -1)
    try:
        result = future.result()
    except cf.CancelledError:
        counters["unpersisted"] += 1
        return 0
    except Exception as exc:  # BrokenProcessPool etc.
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "fragments": []}
        if "BrokenProcessPool" in type(exc).__name__:
            counters["unpersisted"] += 1
            ctx.request_stop("A decode worker process crashed (often memory). Rerun to resume; completed members are kept.", fatal=True)
            return 0
    status = result["status"]
    record = {"archive": archive, "member": member, "patient_id": patient, "member_bytes": member_bytes,
              "case_ids": case_ids, "processing_signature": ctx.signature, "retry_count": retry,
              "seconds": result.get("seconds"), "updated_at": utc_now()}
    if status in DONE_STATES:
        for info in result["fragments"]:
            state.put_fragment(info)
            ctx.staging_bytes += info["output_bytes"]
            totals["fragments"] += 1
            _canary(ctx, info["sanity"], progress)
        record["fragment_count"] = len(result["fragments"])
        state.put_member(archive, member, status, record)
        counters["decoded"] += 1
        totals["decoded"] += 1
        return 1
    if status == "disk_stop":
        counters["unpersisted"] += 1
        ctx.request_stop(result["error"])
        return 0
    record.update(error=result.get("error"), retry_count=retry + 1)
    state.put_member(archive, member, "failed", record)
    _error(ctx.root, "fast_ingest", archive, member, None, result.get("error"))
    counters["failed"] += 1
    totals["failed"] += 1
    attempts = totals["decoded"] + totals["failed"]
    if (totals["failed"] >= int(ctx.fast["canary_min_failures"])
            and totals["failed"] > float(ctx.fast["canary_max_failure_rate"]) * attempts):
        ctx.request_stop(f"Systematic decode failures: {totals['failed']} of {attempts} real PLETH members failed "
                         f"(latest: {result.get('error')}). Stopping instead of wasting the pass; checkpoints kept.", fatal=True)
    return 1


def _close_ingest(ctx, state, todo, signature, totals, progress, started):
    problems, paused = [], False
    for name, fingerprint in todo:
        c = ctx.counters[name]
        payload = {"archive": name, "fingerprint": fingerprint, "processing_signature": signature, "ingest_profile": _profile(ctx.config),
                   "counters": {k: int(v) for k, v in c.items()}, "updated_at": utc_now()}
        if ctx.errors.get(name):
            payload.update(status="failed", error=ctx.errors[name], gzip_crc_verified=False)
            problems.append(f"{name}: {ctx.errors[name]}")
        elif c["eof"] and ctx.pending[name] == 0 and not c["stopped"] and not c["unpersisted"]:
            payload.update(status="completed", gzip_crc_verified=True, completed_at=utc_now())
        else:
            payload.update(status="paused", gzip_crc_verified=False)
            paused = True
        state.put_archive(name, payload)
    elapsed = time.perf_counter() - started
    summary = {"seconds": elapsed, "stop_reason": ctx.stop_reason, "canary": dict(ctx.canary),
               "counters": {n: {k: int(v) for k, v in ctx.counters[n].items()} for n, _ in todo}, **state.totals()}
    if ctx.fatal:
        summary.update(status="FAILED", reason=ctx.stop_reason)
        _banner("FAILED", f"{ctx.stop_reason}\nCheckpoints and fragments were kept; fix the cause and rerun the same cell to resume.", progress)
    elif problems:
        summary.update(status="INCOMPLETE", reason="; ".join(problems))
        _banner("INCOMPLETE", "Archive pass did not finish cleanly: " + "; ".join(problems) + "\nRaw archives are untouched and still required.", progress)
    elif paused or ctx.stop.is_set():
        summary.update(status="PAUSED", reason=ctx.stop_reason or "stopped")
        _banner("PAUSED", f"{ctx.stop_reason or 'stopped'}.\nAll finished members are checkpointed. Rerun the same cell to resume (completed members are skipped; gzip must replay the compressed prefix).", progress)
    else:
        failed = int(state.totals()["members_by_status"].get("failed", 0))
        summary.update(status="INGEST_COMPLETE", reason=f"{failed} PLETH members failed and quarantine their surgeries" if failed else "all archives read to the end and gzip-CRC verified")
        _say(progress, f"INGEST_COMPLETE in {elapsed / 60:.1f} min: {summary['reason']}. Next: assessing cases and writing the patient-level cohort ...")
    return summary



# finalization: assess -> select -> write -> manifests -> EDA -> readiness

def select_patients(eligible, config):
    """Deterministic, outcome-blind patient-level selection on measured upper-bound bytes."""
    max_bytes, target = config["max_mover_size_gb"] * 1e9, config["target_mover_size_gb"] * 1e9
    sizes = eligible.groupby("patient_id").estimated_bytes.sum()
    if float(sizes.sum()) <= max_bytes:
        return set(sizes.index), "all_eligible_patients"
    order = np.random.default_rng(config["random_seed"]).permutation(sorted(sizes.index))
    chosen, total = set(), 0.0
    for patient in order:
        estimate = float(sizes[patient])
        if total + estimate <= target:
            chosen.add(patient)
            total += estimate
    return chosen, "seeded_patient_subset_to_target_size"


def _run_jobs(pool, fn, jobs, window, handle, bar):
    pending, iterator, exhausted = {}, iter(jobs), False
    try:
        while True:
            while not exhausted and len(pending) < window:
                try:
                    key, args = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending[pool.submit(fn, *args)] = key
            if not pending:
                return
            done, _ = cf.wait(pending, timeout=0.5, return_when=cf.FIRST_COMPLETED)
            for future in done:
                key = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "failed", "case_id": key, "error": f"{type(exc).__name__}: {exc}"}
                handle(key, result)
                bar.update(1)
    except KeyboardInterrupt:
        for future in pending:
            future.cancel()
        raise


def finalize(root, *, workers=None, progress=True, concurrent_vitaldb=False):
    root = Path(root)
    config = load_config(root)
    fast = _fast(config)
    signature = _processing_signature(config)
    started = time.perf_counter()
    workers = _resolve_workers(fast, workers, concurrent_vitaldb, "finalize_workers")
    with _RunLock(root):
        state = _State(root)
        try:
            archives = state.archives()
            profile = _profile(config)
            window = _window_mode(config)
            missing = [n for n in config["mover_archives"] if not _archive_ok(archives.get(n), signature, profile)]
            if missing:
                _banner("INCOMPLETE", "Cannot finalize: archive pass not complete/verified for " + ", ".join(missing) +
                        ".\nPartial fragments are NOT promoted to a cohort or splits, and raw-deletion readiness stays false.", progress)
                return {"status": "INCOMPLETE", "reason": "archives not complete: " + ", ".join(missing), **state.totals()}
            fragments = state.fragments_by_case()
            done_meta = state.meta("finalized")
            if done_meta and done_meta.get("signature") == signature and done_meta.get("profile") == profile and not fragments:
                _banner("COMPLETE", f"Already finalized at {done_meta['at']}: {done_meta['selected_cases']} selected surgeries. "
                                    "Use reset_fast_state(ROOT) to re-ingest from the archives.", progress)
                return {"status": "COMPLETE", "reason": "already finalized", **done_meta}
            failed_members = state.failed_members()
            quarantined = defaultdict(list)
            for member in failed_members:
                for case_id in member["case_ids"]:
                    quarantined[case_id].append(f"{member['archive']}:{member['member']}: {member.get('error')}")
            cases = _cases(root)
            usable = cases.loc[cases["linkage_usable"].astype(bool) & cases["patient_id"].notna()].copy()
            usable["case_id"] = usable["case_id"].astype(str)
            case_rows = {r["case_id"]: r for r in usable.to_dict("records")}
            default_reason = "no_art_ecg_ppg_member_in_surgery_window" if window else "no_pleth_members_in_surgery_window"
            updates = {cid: {"eligibility_status": "complete", "final_usable": False,
                             "exclusion_reason": default_reason, "source_member_count": 0} for cid in case_rows}
            for cid, notes in quarantined.items():
                if cid in updates:
                    updates[cid].update(exclusion_reason="quarantined_unreadable_pleth_member;uncertain_case_linkage_excluded_conservatively")
            store = CheckpointStore(root, "mover")
            reusable = {}
            if window:
                stored_rows = {str(r["case_id"]): r for r in store.rows()}
                for cid in case_rows:
                    prior = stored_rows.get(cid)
                    if (prior and prior.get("status") == "completed" and prior.get("processing_signature") == signature
                            and prior.get("ingest_mode") == "fast_one_pass" and verify_case(root, prior)):
                        reusable[cid] = prior
            assess_jobs = [(cid, (_case_args(case_rows[cid]), reusable[cid] if cid in reusable else fragments[cid]))
                           for cid in sorted(set(fragments) | set(reusable)) if cid in case_rows and cid not in quarantined]
            _say(progress, f"FINALIZE: {len(assess_jobs):,} surgeries have candidate fragments or verified case files ({len(reusable):,} reused as-is); "
                           f"assessing {'window-level' if window else 'case-level'} quality with {workers} processes ...")
            assessed = {}
            with _ThreadEnv(), _make_pool(root, config, signature, workers, fast) as pool:
                try:
                    bar = tqdm(total=len(assess_jobs), desc="assess surgeries", unit="case", disable=not progress)
                    try:
                        _run_jobs(pool, _assess_any, assess_jobs, workers * 3, lambda k, r: assessed.__setitem__(k, r), bar)
                    finally:
                        bar.close()
                except KeyboardInterrupt:
                    pool.shutdown(wait=True, cancel_futures=True)
                    _banner("PAUSED", "Finalization interrupted during assessment; nothing was promoted. Rerun to continue.", progress)
                    return {"status": "PAUSED", "reason": "interrupted during assessment"}
                except cf.process.BrokenProcessPool as exc:
                    _banner("FAILED", f"A worker crashed during assessment: {exc}", progress)
                    return {"status": "FAILED", "reason": str(exc)}
                for cid, result in assessed.items():
                    if result["status"] == "assessed":
                        updates[cid].update(result["assessment"])
                        frags = fragments.get(cid, [])
                        archives_used = sorted({f["archive"] for f in frags}) or list(reusable.get(cid, {}).get("source_archive") or [])
                        updates[cid].update(estimated_bytes=result["estimated_bytes"], eligibility_status="complete",
                                            source_archive=";".join(archives_used),
                                            source_member_count=len(frags) or len(reusable.get(cid, {}).get("source_member") or []),
                                            has_ppg_observed=True, duplicate_conflict_count=result["duplicate_conflict_count"])
                    elif result["status"] == "disk_stop":
                        _banner("PAUSED", f"Disk guard: {result['error']}", progress)
                        pool.shutdown(wait=True, cancel_futures=True)
                        return {"status": "PAUSED", "reason": result["error"]}
                    else:
                        updates[cid].update(exclusion_reason="assessment_failed", assessment_error=result.get("error"))
                        _error(root, "fast_assess", None, None, cid, result.get("error"))
                window_info = None
                if window:
                    window_info = _apply_ladder(config, assessed, updates, case_rows, reusable)
                    if window_info["rung"] is None:
                        return _finish_without_cohort(root, cases, updates, state, fragments, window_info["reason"], progress, signature, started,
                                                      window_info=window_info)
                eligible = pd.DataFrame([{"case_id": cid, "patient_id": case_rows[cid]["patient_id"], "estimated_bytes": u["estimated_bytes"]}
                                         for cid, u in sorted(updates.items()) if u.get("final_usable")])
                if eligible.empty:
                    return _finish_without_cohort(root, cases, updates, state, fragments, "No eligible surgeries met the unchanged ART+ECG+PPG+arterial-MAP criteria", progress, signature, started,
                                                  window_info=window_info)
                chosen, selection_mode = select_patients(eligible, config)
                if len(chosen) < 3:
                    return _finish_without_cohort(root, cases, updates, state, fragments, f"Only {len(chosen)} eligible patients; at least 3 are required for patient-level train/validation/test splits", progress, signature, started,
                                                  window_info=window_info)
                splits = patient_splits(sorted(chosen), config["patient_split_seed"])
                split_of = dict(zip(splits.patient_id.astype(str), splits.split))
                write_jobs, results = [], {}
                for cid in sorted(updates):
                    u = updates[cid]
                    if not u.get("final_usable"):
                        continue
                    patient = str(case_rows[cid]["patient_id"])
                    if patient not in chosen:
                        u.update(final_usable=True, selected=False, split="unassigned", exclusion_reason=u.get("exclusion_reason", "") or "eligible_not_selected_size_cap")
                        continue
                    u.update(selected=True, split=split_of[patient])
                    prior = store.get(cid)
                    valid = bool(prior and prior.get("status") == "completed" and prior.get("processing_signature") == signature
                                 and prior.get("ingest_mode") == "fast_one_pass" and verify_case(root, prior))
                    if valid and prior.get("split") == u["split"]:
                        results[cid] = {"status": "completed", "committed": prior}
                        continue
                    if valid and window:
                        results[cid] = {"status": "completed", "committed": _resplit_case(root, store, cid, prior, u["split"])}
                        continue
                    write_jobs.append((cid, (_case_args(case_rows[cid], u["split"]), fragments[cid])))
                _say(progress, f"SELECT: {len(chosen):,} patients ({selection_mode}); writing {len(write_jobs):,} verified case files "
                               f"({len(results):,} already valid) with {workers} processes ...")
                bar = tqdm(total=len(write_jobs), desc="write cases", unit="case", disable=not progress)
                try:
                    _run_jobs(pool, _write_task, write_jobs, workers * 2, lambda k, r: results.__setitem__(k, r), bar)
                except KeyboardInterrupt:
                    pool.shutdown(wait=True, cancel_futures=True)
                    _banner("PAUSED", "Finalization interrupted while writing; verified cases are kept. Rerun to continue.", progress)
                    return {"status": "PAUSED", "reason": "interrupted while writing"}
                except cf.process.BrokenProcessPool as exc:
                    _banner("FAILED", f"A worker crashed while writing: {exc}", progress)
                    return {"status": "FAILED", "reason": str(exc)}
                finally:
                    bar.close()
                    pool.shutdown(wait=True, cancel_futures=True)
            write_failures, disk_stops = [], []
            for cid, result in sorted(results.items()):
                if result["status"] == "completed":
                    committed = _py(result["committed"])
                    store.update(cid, **_store_fields(committed), status="completed", completed_at=utc_now())
                    updates[cid].update(waveform_output_path=committed["waveform_output_path"], map_output_path=committed["map_output_path"],
                                        processed_bytes=committed["processed_bytes"])
                elif result["status"] == "disk_stop":
                    disk_stops.append(result["error"])
                else:
                    write_failures.append(cid)
                    store.update(cid, status="failed", error=result.get("error") or result.get("status"))
                    updates[cid].update(final_usable=False, exclusion_reason="write_failed", selected=False)
                    _error(root, "fast_write", None, None, cid, result.get("error"))
            if disk_stops:
                _banner("PAUSED", f"Disk guard while writing: {disk_stops[0]}\nVerified cases are kept; free space and rerun to continue.", progress)
                return {"status": "PAUSED", "reason": disk_stops[0]}
            return _publish(root, config, cases, updates, case_rows, selection_mode, splits, write_failures, state, fragments,
                            failed_members, signature, progress, started, fast, window_info=window_info, reused=len(reusable))
        finally:
            state.close()


def _resplit_case(root, store, cid, prior, split):
    """A patient-level split change is a metadata-only edit: waveform/map files and their hashes stay untouched."""
    path = Path(root) / "data" / "parquet" / "mover" / f"case_id={safe_id(cid)}" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata["split"] = split
    atomic_json(path, metadata)
    store.update(cid, split=split)
    return {**prior, "split": split}


def _apply_ladder(config, assessed, updates, case_rows, reusable):
    """Outcome-blind rung selection on window-level readiness; sets final_usable/exclusion_reason per assessed case."""
    ladder, targets = load_ladder(config)
    rows = []
    for cid, result in assessed.items():
        if result["status"] == "assessed":
            rows.append({"case_id": cid, "patient_id": str(case_rows[cid]["patient_id"]), **updates[cid]})
    stats = pd.DataFrame(rows)
    if stats.empty:
        return {"rung": None, "reason": "no case could be assessed", "funnel": pd.DataFrame(), "stats": stats, "windows": {},
                "ladder": ladder, "targets": targets, "existing_failing": []}
    rung, funnel, reason = choose_rung(stats, ladder, targets)
    passed_rung = []
    for _, row in stats.iterrows():
        first = next((r["name"] for r in ladder if rung_passes(row, r)), "none")
        passed_rung.append(first)
        u = updates[row["case_id"]]
        u["rung_passed"] = first
        ok = rung is not None and rung_passes(row, rung)
        u["final_usable"], u["provisional_usable"] = bool(ok), bool(ok)
        if ok:
            u["exclusion_reason"] = ""
        elif not row["has_art"]:
            u["exclusion_reason"] = "no_arterial_signal"
        elif not row["has_map"]:
            u["exclusion_reason"] = "no_arterial_map_numerics"
        else:
            u["exclusion_reason"] = "ready_span_below_chosen_rung"
    stats["rung_passed"] = passed_rung
    stats["final_usable"] = [updates[c]["final_usable"] for c in stats.case_id]
    stats["exclusion_reason"] = [updates[c]["exclusion_reason"] for c in stats.case_id]
    existing_failing = sorted(cid for cid in reusable if rung is not None and cid in updates and not updates[cid].get("final_usable"))
    return {"rung": rung, "reason": reason, "funnel": funnel, "stats": stats, "windows": {c: r["windows"] for c, r in assessed.items() if r["status"] == "assessed"},
            "ladder": ladder, "targets": targets, "existing_failing": existing_failing, "existing_total": len(reusable)}


def _funnel_text(window_info):
    if not window_info:
        return ""
    funnel = window_info["funnel"]
    parts = " | ".join(f"{r.rung} (theta {r.theta:.2f}, >={r.min_ready_span_min:g} ready min): {r.cases} cases / {r.patients} patients" for r in funnel.itertuples()) \
        if len(funnel) else "no assessable cases"
    rung = window_info["rung"]
    text = (f"Window-level eligibility ladder: {parts}.\n"
            f"Chosen rung: {rung['name'] if rung else 'none'} - {window_info['reason']}.")
    if window_info.get("existing_failing"):
        text += f"\nWARNING: {len(window_info['existing_failing'])} previously written case(s) do not pass the chosen rung: {window_info['existing_failing'][:10]}"
    return text


def _write_window_manifests(root, window_info, selected_case_ids):
    """Per-case statistics for EVERY assessed case (any future threshold is a recompute) + ready windows of selected cases."""
    stats = window_info["stats"]
    if len(stats):
        keep = [c for c in stats.columns if not isinstance(stats[c].iloc[0], (dict, list))]
        write_manifest(root, "mover_case_window_stats", stats[keep])
    frames = [window_info["windows"][c] for c in selected_case_ids if c in window_info["windows"] and len(window_info["windows"][c])]
    directory = Path(root) / "data" / "manifests"
    path = directory / "ready_windows_mover.parquet"
    temp = path.with_name(path.name + ".tmp")
    frame = pd.concat(frames, ignore_index=True) if frames else None
    if frame is not None:
        try:
            with temp.open("wb") as handle:
                frame.to_parquet(handle, index=False, compression="zstd")
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)
    return 0 if frame is None else int(len(frame))


def _store_fields(committed):
    skip = {"case_id", "status", "updated_at", "completed_at", "error"}
    return {k: v for k, v in committed.items() if k not in skip}


def _case_args(row, split=None):
    out = {k: row[k] for k in ("case_id", "patient_id", "start_utc", "end_utc", "duration_sec")}
    out["case_id"], out["patient_id"] = str(out["case_id"]), str(out["patient_id"])
    out["split"] = split or "unassigned"
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


def _finish_without_cohort(root, cases, updates, state, fragments, reason, progress, signature, started, window_info=None):
    records = []
    for record in cases.to_dict("records"):
        record.update(updates.get(str(record["case_id"]), {}))
        records.append(record)
    frame = pd.DataFrame(records)
    write_manifest(root, "mover_all_cases", frame)
    write_manifest(root, "mover_eligible_cohort", frame.loc[frame["final_usable"].fillna(False).astype(bool)])
    if window_info is not None:
        _write_window_manifests(root, window_info, [])
    _banner("INCOMPLETE", f"{reason}.\n{_funnel_text(window_info)}\nManifests of the assessed surgeries were written (including mover_case_window_stats); fragments are kept for inspection; no splits/selection were created and raw archives must be kept.", progress)
    return {"status": "INCOMPLETE", "reason": reason, "assessed_cases": int(len(fragments)), "seconds": time.perf_counter() - started}


def _publish(root, config, cases, updates, case_rows, selection_mode, splits, write_failures, state, fragments, failed_members,
             signature, progress, started, fast, window_info=None, reused=0):
    records = []
    for record in cases.to_dict("records"):
        record.update(updates.get(str(record["case_id"]), {}))
        records.append(record)
    frame = pd.DataFrame(records)
    frame["final_usable"] = frame["final_usable"].fillna(False).astype(bool)
    eligible = frame.loc[frame["final_usable"] & frame["linkage_usable"].astype(bool)].copy()
    selected = eligible.loc[eligible["selected"].fillna(False).astype(bool)].copy()
    selected_split = selected.drop(columns=["split"]).merge(splits, on="patient_id", how="left", validate="many_to_one")
    overlaps = assert_patient_independence(selected_split)
    write_manifest(root, "mover_all_cases", frame)
    write_manifest(root, "mover_eligible_cohort", eligible)
    write_manifest(root, "mover_selected_cohort", selected_split)
    write_manifest(root, "mover_patient_splits", splits)
    store = CheckpointStore(root, "mover")
    write_manifest(root, "mover_processed_cases", pd.DataFrame(store.rows()))
    window_text, window_summary = "", {}
    if window_info is not None:
        window_rows = _write_window_manifests(root, window_info, list(selected_split.case_id.astype(str)))
        rung = window_info["rung"]
        key = theta_key(rung["theta"])
        ready_hours = float(selected_split.get(f"ready_span_min_theta_{key}", pd.Series(dtype=float)).fillna(0).sum() / 60)
        window_summary = {"rung": rung["name"], "rung_theta": rung["theta"], "rung_min_ready_span_min": rung["min_ready_span_min"],
                          "reason": window_info["reason"], "funnel": window_info["funnel"].to_dict("records"),
                          "ready_hours_selected": round(ready_hours, 1), "ready_window_rows": window_rows,
                          "existing_cases_reused": reused, "existing_cases_failing_rung": window_info.get("existing_failing", [])}
        rule_blob = json.dumps({"window_rule": load_rule(config), "ladder": window_info["ladder"], "targets": window_info["targets"]}, sort_keys=True)
        atomic_json(root / "data" / "manifests" / "mover_cohort_frozen.json", {
            "frozen_at": utc_now(), "note": "Patient-level split frozen for this cohort; do not re-split without recording a new freeze.",
            **window_summary, "selected_cases": int(len(selected_split)), "selected_patients": int(selected_split.patient_id.nunique()),
            "patients_per_split": splits.split.value_counts().to_dict(), "split_seed": config["patient_split_seed"],
            "patient_list_sha256": hashlib.sha256("\n".join(sorted(selected_split.patient_id.astype(str))).encode()).hexdigest(),
            "rule_sha256": hashlib.sha256(rule_blob.encode()).hexdigest()})
        window_text = (_funnel_text(window_info) + f"\nReady hours at the chosen rung (selected cases): {ready_hours:.1f} h; ready-window rows: {window_rows:,}; "
                       f"{reused} already-written case files reused without rewriting.\n")
    summary = fast_eda(root)
    readiness = fast_deletion_readiness(root)
    ok = not write_failures and not failed_members
    status = "COMPLETE" if ok else "INCOMPLETE"
    done = {"signature": signature, "profile": _profile(config), "at": utc_now(), "selected_cases": int(len(selected_split)),
            "selected_patients": int(selected_split.patient_id.nunique()), "selection": selection_mode,
            "write_failures": write_failures, "quarantined_members": len(failed_members)}
    if ok and not fast["keep_staging"]:
        shutil.rmtree(root / "data" / "parquet" / "mover" / "staging", ignore_errors=True)
        state.clear_staging()
    state.set_meta("finalized", done)
    atomic_json(root / "results" / "mover_fast_finalize.json",
                {**done, "status": status, "patient_intersections": overlaps, "seconds": time.perf_counter() - started,
                 "eligible_cases": int(len(eligible)), "eligible_patients": int(eligible.patient_id.nunique()),
                 "window_eligibility": window_summary,
                 "eda": summary, "raw_archive_readiness": readiness.to_dict("records")})
    text = (f"{len(selected_split):,} surgeries / {selected_split.patient_id.nunique():,} patients selected "
            f"({len(eligible):,} eligible surgeries / {eligible.patient_id.nunique():,} patients; selection: {selection_mode}).\n"
            f"{window_text}"
            f"Splits (patients): {splits.split.value_counts().to_dict()}; patient overlaps: {overlaps}.\n"
            f"Processed output: {summary.get('processed_gb', 0):.2f} GB, {summary.get('waveform_hours', 0):.1f} waveform hours.\n"
            f"Raw-archive deletion readiness: {readiness[['archive', 'safe_to_delete']].values.tolist()} (verify before any manual deletion).")
    if not ok:
        text += f"\nUnresolved: {len(write_failures)} write failures, {len(failed_members)} unreadable PLETH members (see data/manifests/mover_processing_errors.csv)."
    _banner(status, text, progress)
    return {"status": status, "reason": text, **done, "eda": summary, "window_eligibility": window_summary}



# fast EDA and deletion readiness (manifests/checkpoints only; never the archives)

def fast_eda(root):
    root = Path(root)
    rows = pd.DataFrame(CheckpointStore(root, "mover").rows())
    if rows.empty or "status" not in rows:
        return {"completed_cases": 0}
    done = rows.loc[(rows["status"] == "completed") & (rows.get("ingest_mode") == "fast_one_pass")].copy()
    all_cases = pd.read_parquet(root / "data" / "manifests" / "mover_all_cases.parquet")
    summary = {"completed_cases": int(len(done)), "completed_patients": int(done.patient_id.nunique()) if len(done) else 0,
               "waveform_hours": float(done.duration_sec.sum() / 3600) if len(done) else 0.0,
               "processed_gb": float(done.processed_bytes.sum() / 1e9) if len(done) else 0.0,
               "source_member_gb": float(done.source_bytes.sum() / 1e9) if len(done) else 0.0}
    if len(done):
        summary["compression_source_to_output"] = float(done.source_bytes.sum() / max(done.processed_bytes.sum(), 1))
        summary["bytes_per_waveform_hour"] = float(done.processed_bytes.sum() / max(done.duration_sec.sum() / 3600, 1e-9))
        summary["patients_per_split"] = done.groupby("split").patient_id.nunique().to_dict()
        summary["surgeries_per_split"] = done.split.value_counts().to_dict()
        summary["max_surgeries_per_patient"] = int(done.groupby("patient_id").size().max())
        columns = [c for c in ("duration_sec", "art_missingness", "ecg_missingness", "ppg_missingness", "map_valid_points",
                               "map_coverage", "map_median_gap", "map_median", "map_points_per_min") if c in done]
        summary["distributions"] = done[columns].describe().to_dict()
        by_archive = Counter(a for value in done.source_archive for a in value)
        summary["cases_per_source_archive"] = dict(by_archive)
        if done.split.isin(["train", "validation", "test"]).all():
            summary["patient_intersections"] = assert_patient_independence(done[["patient_id", "split"]].drop_duplicates())
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plot_columns = [c for c in ("duration_sec", "art_missingness", "ecg_missingness", "ppg_missingness", "map_coverage", "map_median_gap") if c in done]
            done[plot_columns].hist(figsize=(12, 8), bins=30)
            plt.tight_layout()
            (root / "figures").mkdir(exist_ok=True)
            plt.savefig(root / "figures" / "mover_fast_quality.png", dpi=120)
            plt.close("all")
        except Exception as exc:  # plotting must never fail a finished cohort
            summary["plot_error"] = str(exc)
    reasons = all_cases.exclusion_reason.fillna("").replace("", "eligible_or_selected").value_counts()
    summary["case_status_counts"] = reasons.head(25).to_dict()
    (root / "results").mkdir(exist_ok=True)
    reasons.rename("cases").rename_axis("reason").reset_index().to_csv(root / "results" / "mover_fast_exclusions.csv", index=False)
    atomic_json(root / "results" / "mover_fast_eda.json", summary)
    return summary


def fast_deletion_readiness(root):
    root = Path(root)
    config = load_config(root)
    signature = _processing_signature(config)
    state = _State(root)
    try:
        archives = state.archives()
        failed = state.failed_members()
    finally:
        state.close()
    selected_path = root / "data" / "manifests" / "mover_selected_cohort.parquet"
    selected = pd.read_parquet(selected_path) if selected_path.exists() else pd.DataFrame()
    records = {r["case_id"]: r for r in CheckpointStore(root, "mover").rows()}
    split_ok, loader_ok = False, False
    if len(selected):
        try:
            split_ok = not any(assert_patient_independence(selected[["patient_id", "split"]].drop_duplicates()).values())
        except Exception:
            split_ok = False
    rows = []
    for name in config["mover_archives"]:
        state_row = archives.get(name)
        scan_ok = bool(_archive_ok(state_row, signature, _profile(config)))
        expected = [c for c, a in zip(selected.case_id, selected.source_archive) if name in str(a).split(";")] if len(selected) else []
        verified = [c for c in expected if records.get(c, {}).get("status") == "completed" and verify_case(root, records[c])]
        failed_cases = [c for c in expected if records.get(c, {}).get("status") == "failed"]
        manifest_ok = bool(expected) and all(
            selected.loc[selected.case_id == c, "waveform_output_path"].iloc[0] == records.get(c, {}).get("waveform_output_path") for c in expected)
        loader_ok = False
        for c in verified[:: max(1, len(verified) // 3)][:3]:
            try:
                loader_ok = bool(np.isfinite(load_case(root, "mover", c)["MAP"]).any())
            except Exception:
                loader_ok = False
            if not loader_ok:
                break
        member_failures = sum(1 for m in failed if m["archive"] == name)
        checks = ((scan_ok, "complete gzip-verified archive pass missing"), (bool(expected), "no final selected cases from this archive"),
                  (len(verified) == len(expected), "selected outputs missing/corrupt"), (not failed_cases, "unresolved selected-case failures"),
                  (member_failures == 0, f"{member_failures} unreadable PLETH members in this archive"),
                  (split_ok, "patient split audit not verified"), (manifest_ok, "manifests do not match verified outputs"),
                  (loader_ok, "canonical loader check failed"))
        reasons = [text for ok, text in checks if not ok]
        safe = not reasons
        rows.append({"archive": name, "archive_size": (root / "data" / "reference" / "mover" / name).stat().st_size,
                     "archive_scan_completed": scan_ok, "selected_cases_expected": len(expected),
                     "selected_cases_processed": sum(records.get(c, {}).get("status") == "completed" for c in expected),
                     "selected_cases_verified": len(verified), "selected_cases_failed": len(failed_cases),
                     "selected_cases_missing": len(expected) - len(verified),
                     "processed_output_bytes": int(sum(records[c].get("processed_bytes", 0) for c in verified)), "safe_to_delete": safe,
                     "reason": "; ".join(reasons) if reasons else
                     "Independently verified. Deleting still discards non-selected surgeries and ART/ECG-only members; deletion is your manual decision."})
    frame = pd.DataFrame(rows)
    write_manifest(root, "mover_raw_archive_deletion_readiness", frame)
    return frame


def run(root, *, workers=None, progress=True, concurrent_vitaldb=False, _stop_after=None):
    """Ingest then finalize. Returns a dict whose ``status`` is COMPLETE/PAUSED/INCOMPLETE/FAILED."""
    first = ingest(root, workers=workers, progress=progress, concurrent_vitaldb=concurrent_vitaldb, _stop_after=_stop_after)
    if first["status"] != "INGEST_COMPLETE":
        return first
    result = finalize(root, workers=workers, progress=progress, concurrent_vitaldb=concurrent_vitaldb)
    result["ingest"] = first
    return result
