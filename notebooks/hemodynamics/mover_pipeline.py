"""Sequential, fail-closed MOVER discovery and patient-level ingestion."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import psutil
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from .canonical import (CheckpointStore, DiskSpaceStop, assert_patient_independence, atomic_json,
                        ensure_disk_space, environment_record, load_case, patient_splits, safe_id,
                        sha256_file, utc_now, verify_case, write_case, write_manifest)


class DiscoveryIncomplete(RuntimeError):
    """Reject selection or processing when archive discovery is incomplete."""
    pass


def load_config(root):
    """Load the central settings and validate the MOVER storage budget."""
    config = json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))
    if not 0 < config["target_mover_size_gb"] <= config["max_mover_size_gb"]:
        raise ValueError("Invalid MOVER size limits")
    return config


def _reserve(config):
    return int(config["min_free_space_gb"] * 1e9)


def _processing_signature(config):
    parameters = {k: config.get(k) for k in ("target_waveform_fs", "clinical_timezone", "mover_track_map", "map_min_mmhg", "map_max_mmhg")}
    parameters["decoder_sha256"] = sha256_file(Path(__file__).with_name("mover_signals.py"))
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


@contextmanager
def _database(root):
    directory = Path(root) / "data" / "checkpoints" / "mover"
    directory.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(directory / "inventory.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=FULL")
        db.execute("CREATE TABLE IF NOT EXISTS archives (archive TEXT PRIMARY KEY, payload TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS members (archive TEXT, member TEXT, payload TEXT NOT NULL, PRIMARY KEY(archive, member))")
        db.execute("CREATE TABLE IF NOT EXISTS fragments (archive TEXT, member TEXT, case_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(archive, member, case_id))")
        db.execute("CREATE TABLE IF NOT EXISTS case_stats (archive TEXT, member TEXT, case_id TEXT, payload TEXT NOT NULL, PRIMARY KEY(archive, member, case_id))")
        db.execute("CREATE INDEX IF NOT EXISTS case_stats_case ON case_stats(case_id)")
        db.execute("CREATE INDEX IF NOT EXISTS fragments_case ON fragments(case_id)")
        with db:
            yield db
    finally:
        db.close()


def _error(root, phase, archive, member, case_id, error):
    path = Path(root) / "data" / "manifests" / "mover_processing_errors.csv"
    frame = pd.DataFrame([{"timestamp": utc_now(), "phase": phase, "archive": archive,
                           "archive_member": member, "case_id": case_id, "error": str(error)}])
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def audit_linkage(root):
    """Check the authorized patient crosswalk and case linkage before cohort selection."""
    root = Path(root)
    config = load_config(root)
    ensure_disk_space(root, _reserve(config))
    source = root / "data" / "reference" / "mover"
    error_log = root / "data" / "manifests" / "mover_processing_errors.csv"
    if not error_log.exists():
        pd.DataFrame(columns=["timestamp", "phase", "archive", "archive_member", "case_id", "error"]).to_csv(error_log, index=False)
    linkage = pd.read_csv(source / "EPIC_MRN_PAT_ID.csv", dtype=str)
    unique = linkage.drop_duplicates()
    ambiguous = set(unique.groupby("LOG_ID").PAT_ID.nunique().loc[lambda s: s > 1].index)
    if unique.groupby("MRN").PAT_ID.nunique().gt(1).any():
        raise ValueError("MRN-to-patient linkage is ambiguous; stop before cohort construction")
    crosswalk = unique.loc[~unique.LOG_ID.isin(ambiguous)].drop_duplicates(["LOG_ID", "MRN"])
    patient_table = None
    reference_inventory = []
    with tarfile.open(source / "EPIC_EMR.tar.gz", "r|gz") as archive:
        for member in archive:
            archive.members.clear()
            reference_inventory.append({"archive": "EPIC_EMR.tar.gz", "member": member.name, "bytes": member.size, "is_file": member.isfile()})
            if member.isfile() and Path(member.name).name == "patient_information.csv":
                if member.size > 50_000_000:
                    raise ValueError("Unexpectedly large patient information table")
                patient_table = pd.read_csv(io.BytesIO(archive.extractfile(member).read()), dtype=str)
                break
    if patient_table is None:
        raise ValueError("Observed patient_information.csv missing")
    columns = ["LOG_ID", "MRN", "IN_OR_DTTM", "OUT_OR_DTTM"]
    if not set(columns).issubset(patient_table):
        raise ValueError("Clinical linkage schema changed")
    rows = patient_table[columns].drop_duplicates().merge(crosswalk, on=["LOG_ID", "MRN"], how="left", validate="many_to_one")
    for field in ("IN_OR_DTTM", "OUT_OR_DTTM"):
        local = pd.to_datetime(rows[field], format="%m/%d/%y %H:%M", errors="coerce")
        rows[field + "_utc"] = local.dt.tz_localize(config["clinical_timezone"], ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    inconsistent = set(rows.groupby("LOG_ID")[columns[1:] + ["PAT_ID"]].nunique(dropna=False).max(axis=1).loc[lambda s: s > 1].index)
    records = []
    for case_id, group in rows.groupby("LOG_ID", sort=True):
        row = group.iloc[0]
        reasons = []
        if case_id in ambiguous or case_id in inconsistent:
            reasons.append("ambiguous_patient_or_surgery_window")
        if pd.isna(row.PAT_ID):
            reasons.append("patient_linkage_unavailable")
        start, end = row.IN_OR_DTTM_utc, row.OUT_OR_DTTM_utc
        duration = (end - start).total_seconds() if pd.notna(start) and pd.notna(end) else None
        if duration is None or duration <= 0:
            reasons.append("invalid_or_ambiguous_surgery_time")
        elif duration < config["minimum_duration_sec"]:
            reasons.append("duration_below_minimum")
        elif duration > config["max_case_hours"] * 3600:
            reasons.append("duration_exceeds_memory_guard")
        records.append({"dataset": "mover", "case_id": case_id, "patient_id": None if pd.isna(row.PAT_ID) or case_id in ambiguous else row.PAT_ID,
                        "start_utc": start.timestamp() if pd.notna(start) else None,
                        "end_utc": end.timestamp() if pd.notna(end) else None,
                        "duration_sec": duration, "linkage_usable": not bool(reasons),
                        "provisional_usable": False, "final_usable": False, "eligibility_status": "not_scanned",
                        "exclusion_reason": ";".join(reasons), "split": "unassigned"})
    cases = pd.DataFrame(records)
    write_manifest(root, "mover_all_cases", cases)
    report = {"crosswalk_rows": len(linkage), "duplicate_crosswalk_rows": int(linkage.duplicated().sum()),
              "unique_patients_crosswalk": int(linkage.PAT_ID.nunique()), "unique_cases_crosswalk": int(linkage.LOG_ID.nunique()),
              "null_identifiers": linkage.isna().sum().to_dict(), "ambiguous_LOG_ID_count": len(ambiguous),
              "unique_clinical_cases": len(cases), "clinical_rows": len(patient_table),
              "cases_with_reliable_linkage_and_window": int(cases.linkage_usable.sum()),
              "patients_with_multiple_surgeries": int(unique.groupby("PAT_ID").LOG_ID.nunique().gt(1).sum()),
              "maximum_surgeries_per_patient": int(unique.groupby("PAT_ID").LOG_ID.nunique().max()),
              "case_identifier": "LOG_ID", "patient_identifier": "PAT_ID", "alternate_patient_identifier": "MRN",
              "waveform_filename_identifier": "PAT_ID (first 16 characters)",
              "clinical_timezone": config["clinical_timezone"], "timezone_basis": "UTC-to-Los-Angeles overlap validated against 300 real members; clinical DST ambiguities excluded",
              "emr_inventory_complete": False, "reference_members_observed": reference_inventory}
    atomic_json(root / "data" / "manifests" / "mover_patient_linkage_audit.json", report)
    return cases, report


def _cases(root):
    path = Path(root) / "data" / "manifests" / "mover_all_cases.parquet"
    if not path.exists():
        return audit_linkage(root)[0]
    return pd.read_parquet(path)


def _packet_schema(payload):
    tree = ET.fromstring(payload)
    groups = tree.findall(".//mg")
    rates = {}
    fields = set()
    for group in groups:
        entries = {item.get("name"): item.text for item in group.findall("m")}
        if "Hz" in entries:
            rates.setdefault(group.get("name"), set()).add(float(entries["Hz"]))
    for element in tree.findall(".//m"):
        if element.get("name") != "Wave":
            fields.add(element.get("name"))
    packets = tree.findall("cpc")
    times = [pd.Timestamp(packet.get("datetime")).timestamp() for packet in packets if packet.get("datetime")]
    return {"groups": sorted({group.get("name") for group in groups}),
            "original_fs": {k: sorted(v) for k, v in rates.items()}, "numeric_names": sorted(fields - {"Points", "PointsBytes", "Min", "Max", "Offset", "Gain", "Hz"}),
            "start_utc": min(times) if times else None, "end_utc": max(times) + 1 if times else None,
            "packet_count": len(packets), "nominal_duration_minutes": tree.get("duration")}


def _case_matches(schema, patient_cases):
    if schema["start_utc"] is None or schema["end_utc"] is None:
        return patient_cases.iloc[:0]
    return patient_cases.loc[patient_cases.start_utc.lt(schema["end_utc"]) & patient_cases.end_utc.gt(schema["start_utc"])]


class _ArchiveProgress:
    def __init__(self, label, total, enabled):
        self.enabled = enabled
        self.label = label
        self.bar = tqdm(total=total, desc=label, unit="B", unit_scale=True,
                        unit_divisor=1000, mininterval=1, disable=not enabled, leave=True)

    def update(self, position, **counters):
        if not self.enabled:
            return
        position = min(self.bar.total, max(self.bar.n, position))
        self.bar.set_postfix(counters, refresh=False)
        self.bar.update(position - self.bar.n)

    def close(self, status):
        if self.enabled:
            self.bar.set_description(f"{self.label} [{status}]", refresh=True)
        self.bar.close()


def _announce(enabled, text):
    if enabled:
        print(text, flush=True)


def discover(root, members_per_archive=None, complete_scan=False, save_fragments=True, retry_failed=False, *, progress=False):
    """Bounded discovery by default. Optional progress measures compressed bytes read."""
    root = Path(root)
    config = load_config(root)
    if complete_scan and not config["allow_full_archive_scan"]:
        raise DiscoveryIncomplete("Set allow_full_archive_scan only after inspecting the bounded discovery results; this decompresses all three archives sequentially.")
    limit = None if complete_scan else (members_per_archive or config["discovery_members_per_archive"])
    _announce(progress, f"STARTING {'COMPLETE' if complete_scan else 'BOUNDED'} MOVER DISCOVERY: {len(config['mover_archives'])} archives. Progress is compressed bytes read, not eligible cases. Do not start a second scan in another cell.")
    _announce(progress, "Loading patient/case metadata and resume checkpoints...")
    processing_signature = _processing_signature(config)
    cases = _cases(root)
    by_patient = {p: g for p, g in cases.loc[cases.linkage_usable].groupby("patient_id")}
    pilot_patient_ids = set()
    started = time.perf_counter()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    total_source_bytes = 0
    with _database(root) as db:
        prior = db.execute("SELECT payload FROM fragments").fetchall()
        for item in prior:
            info = json.loads(item[0])
            if info.get("map_rows", 0) and info.get("metadata", {}).get("waveform_counts", {}).get("ART", {}).get("valid_samples", 0):
                pilot_patient_ids.add(info["patient_id"])
        for archive_number, name in enumerate(config["mover_archives"], start=1):
            source = root / "data" / "reference" / "mover" / name
            fingerprint = {"size_bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}
            row = db.execute("SELECT payload FROM archives WHERE archive=?", (name,)).fetchone()
            old = json.loads(row[0]) if row else {}
            if old and old.get("fingerprint") != fingerprint:
                raise ValueError("Source archive changed; checkpoint cannot be reused")
            if not retry_failed and old.get("scan_completed") and old.get("gzip_crc_verified") and old.get("processing_signature") == processing_signature:
                _announce(progress, f"[{archive_number}/{len(config['mover_archives'])}] {name}: SKIPPED — verified complete scan already cached.")
                continue
            _announce(progress, f"[{archive_number}/{len(config['mover_archives'])}] SCANNING {name} ({fingerprint['size_bytes'] / 1e9:.2f} GB compressed).")
            archive_progress = _ArchiveProgress(f"Archive {archive_number}/{len(config['mover_archives'])}: {name}", fingerprint["size_bytes"], progress)
            state = {"archive": name, "fingerprint": fingerprint, "processing_signature": processing_signature, "scan_completed": False, "sample_limit": limit,
                     "started_at": utc_now(), "status": "processing"}
            db.execute("INSERT OR REPLACE INTO archives VALUES (?, ?)", (name, json.dumps(state)))
            db.commit()
            count, directory_count, failed, inspected, skipped = 0, 0, 0, 0, 0
            exhausted = True
            try:
                with source.open("rb") as raw, gzip.GzipFile(fileobj=raw, mode="rb") as compressed, tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        archive.members.clear()
                        if not member.isfile():
                            directory_count += 1
                            continue
                        free_bytes = ensure_disk_space(root, _reserve(config), member.size if save_fragments else 0)
                        count += 1
                        archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed, free_GB=round(free_bytes / 1e9, 1))
                        previous = db.execute("SELECT payload FROM members WHERE archive=? AND member=?", (name, member.name)).fetchone()
                        if previous and json.loads(previous[0]).get("status") == "completed" and json.loads(previous[0]).get("processing_signature") == processing_signature and (not complete_scan or json.loads(previous[0]).get("quality_scanned")):
                            skipped += 1
                            archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed, free_GB=round(free_bytes / 1e9, 1))
                            if limit is not None and count >= limit:
                                exhausted = False
                                break
                            continue
                        previous_record = json.loads(previous[0]) if previous else {}
                        if previous_record.get("status") == "failed" and previous_record.get("processing_signature") == processing_signature and previous_record.get("retry_count", 0) >= config["max_retries"]:
                            failed += 1
                            skipped += 1
                            archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed, free_GB=round(free_bytes / 1e9, 1))
                            if limit is not None and count >= limit:
                                exhausted = False
                                break
                            continue
                        inspected += 1
                        payload = None
                        record = {"archive": name, "member": member.name, "member_bytes": member.size,
                                  "patient_id": Path(member.name).name[:16], "status": "processing",
                                  "processing_signature": processing_signature, "retry_count": previous_record.get("retry_count", 0) if previous_record.get("processing_signature") == processing_signature else 0}
                        try:
                            if member.size > config["max_member_bytes"] or not member.name.endswith(".xml"):
                                raise ValueError("Unexpected member type/size; no arbitrary extraction")
                            payload = archive.extractfile(member).read()
                            total_source_bytes += len(payload)
                            schema = _packet_schema(payload)
                            patient_cases = by_patient.get(record["patient_id"], cases.iloc[:0])
                            matches = _case_matches(schema, patient_cases)
                            # A member may straddle surgeries; overlapping clinical windows are never guessed.
                            if len(matches) > 1:
                                windows = matches.sort_values("start_utc")
                                if (windows.start_utc.to_numpy()[1:] < windows.end_utc.to_numpy()[:-1]).any():
                                    raise ValueError("Ambiguous overlapping surgery windows")
                            record.update(schema, case_ids=matches.case_id.tolist(), status="completed", processing_signature=processing_signature, quality_scanned=complete_scan or "PLETH" in schema["groups"])
                            db.execute("DELETE FROM case_stats WHERE archive=? AND member=?", (name, member.name))
                            for _, matched_case in matches.iterrows():
                                stats = {"archive": name, "member": member.name, "case_id": matched_case.case_id,
                                         "source_bytes": member.size, "groups": schema["groups"], "original_fs": schema["original_fs"],
                                         "valid_intervals": {c: [] for c in ("ART", "ECG", "PPG")}, "map_times": [], "map_values": []}
                                if "PLETH" in schema["groups"] or complete_scan:
                                    stats.update(_summarize_member(payload, matched_case, config))
                                db.execute("INSERT OR REPLACE INTO case_stats VALUES (?, ?, ?, ?)", (name, member.name, matched_case.case_id, json.dumps(stats, default=str)))
                            core = {"ECG1", "INVP1", "PLETH"}.issubset(schema["groups"]) and "IBP_MEAN" in schema["numeric_names"]
                            if save_fragments and core and len(matches):
                                native_core = bool(stats.get("map_times")) and all(stats["valid_intervals"][c] for c in ("ART", "ECG", "PPG"))
                                if record["patient_id"] in pilot_patient_ids or (native_core and len(pilot_patient_ids) < min(10, config["pilot_cases"])):
                                    pilot_patient_ids.add(record["patient_id"])
                                    for _, case in matches.iterrows():
                                        _save_fragment(root, db, name, member.name, payload, case, config)
                        except DiskSpaceStop:
                            raise
                        except Exception as error:
                            failed += 1
                            affected = by_patient.get(record["patient_id"], cases.iloc[:0])
                            record.update(status="failed", error=str(error), case_ids=affected.case_id.tolist(), retry_count=record["retry_count"] + 1)
                            db.execute("DELETE FROM case_stats WHERE archive=? AND member=?", (name, member.name))
                            for case_id in affected.case_id:
                                failed_stats = {"archive": name, "member": member.name, "case_id": case_id, "decoding_error": str(error),
                                                "groups": [], "original_fs": {}, "source_bytes": member.size,
                                                "valid_intervals": {c: [] for c in ("ART", "ECG", "PPG")}, "map_times": [], "map_values": []}
                                db.execute("INSERT OR REPLACE INTO case_stats VALUES (?, ?, ?, ?)", (name, member.name, case_id, json.dumps(failed_stats)))
                            _error(root, "discovery", name, member.name, None, error)
                        db.execute("INSERT OR REPLACE INTO members VALUES (?, ?, ?)", (name, member.name, json.dumps(record)))
                        db.commit()
                        peak_rss = max(peak_rss, process.memory_info().rss)
                        archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed, free_GB=round(free_bytes / 1e9, 1))
                        if limit is not None and count >= limit:
                            exhausted = False
                            break
                    if exhausted:
                        _announce(progress, f"{name}: checking end-of-stream and gzip checksum...")
                        for _ in iter(lambda: compressed.read(1 << 20), b""):
                            archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed)
                        archive_progress.update(raw.tell(), seen=count, inspected=inspected, skipped=skipped, errors=failed)
                state.update(scan_completed=exhausted, enumeration_reached_eof=exhausted, gzip_crc_verified=exhausted,
                             files_seen_this_pass=count, directories_seen_this_pass=directory_count, errors_this_pass=failed,
                             status="completed_with_case_errors" if exhausted and failed else "completed" if exhausted else "sampled", completed_at=utc_now())
            except (DiskSpaceStop, KeyboardInterrupt) as error:
                state.update(status="paused", scan_completed=False, error=str(error), last_file_count=count)
                db.execute("INSERT OR REPLACE INTO archives VALUES (?, ?)", (name, json.dumps(state)))
                db.commit()
                raise
            except Exception as error:
                state.update(status="failed", scan_completed=False, error=str(error))
                _error(root, "archive", name, None, None, error)
            finally:
                try:
                    db.execute("INSERT OR REPLACE INTO archives VALUES (?, ?)", (name, json.dumps(state)))
                    db.commit()
                finally:
                    archive_progress.close(state["status"].upper())
                    _announce(progress, f"{name}: {state['status'].upper()} — {count:,} files seen, {inspected:,} inspected, {skipped:,} skipped, {failed:,} errors.")
    _announce(progress, "Archive passes finished. Saving inventory and building case eligibility...")
    inventory = export_inventory(root)
    metrics = {"phase": "bounded_discovery" if limit else "complete_discovery", "wall_seconds": time.perf_counter() - started,
               "new_member_input_bytes": total_source_bytes, "peak_rss_bytes": peak_rss,
               "input_MB_per_second": total_source_bytes / 1e6 / max(time.perf_counter() - started, 1e-9),
               "free_disk_bytes": ensure_disk_space(root, _reserve(config)), "archive_count": len(config["mover_archives"]),
               "full_scan_complete": bool(inventory.scan_completed.all()), "environment": environment_record(), "config": config}
    atomic_json(root / "results" / "mover_discovery_metrics.json", metrics)
    eligible = build_eligibility(root, progress=progress)
    _announce(progress, "Saving conservative raw-archive deletion-readiness report...")
    deletion_readiness(root)
    result = "DISCOVERY COMPLETE" if metrics["full_scan_complete"] else "DISCOVERY FINISHED — FULL SCAN NOT COMPLETE"
    eligibility_note = f"{len(eligible):,} eligible surgeries" if metrics["full_scan_complete"] else "eligibility pending complete inventory"
    _announce(progress, f"{result}: {int(inventory.scan_completed.sum())}/{len(config['mover_archives'])} archives verified; {eligibility_note}. Elapsed {time.perf_counter() - started:.1f} seconds. Reports and checkpoints saved.")
    return inventory, metrics


def _assert_observed_pressure_schema(metadata):
    observed = {"INVP1": (100., .01, "GE_S5", "GE_S5"), "GE_ART": (180., .25, "AnalogWaves", "GE_Solar_8000i")}
    for name, expected in observed.items():
        group = metadata.get("groups", {}).get(name)
        if not group:
            continue
        rate, gain, make, model = expected
        if any(float(fs) != rate for fs in group.get("sample_rates", [])):
            raise ValueError("Unobserved arterial sampling rate; schema review required")
        for variant in group.get("variants", []):
            device = variant.get("device", {})
            if float(variant["Gain"]) != gain or float(variant["Offset"]) != 0 or device.get("make") != make or device.get("model") != model:
                raise ValueError("Unobserved arterial device/scaling; schema review required before mmHg conversion")
    counts = metadata.get("numeric_channel_counts", {})
    if counts.get("IBP_MEAN", 0) > 1 and (counts.get("IBP_MEAN") != 6 or counts.get("IBP_SYS") != 6 or counts.get("IBP_DIA") != 6 or "INVP1" not in metadata.get("group_names", [])):
        raise ValueError("Unvalidated multiple invasive-pressure layout; do not guess MAP channel")


def _summarize_member(payload, case, config):
    from .mover_signals import decode_xml
    wave, numeric, metadata = decode_xml(payload, float(case.start_utc), float(case.end_utc), config["target_waveform_fs"], track_map=config.get("mover_track_map"))
    _assert_observed_pressure_schema(metadata)
    intervals = {}
    for channel in ("ART", "ECG", "PPG"):
        indexes = np.rint(wave.loc[np.isfinite(wave[channel]), "time_sec"].to_numpy() * config["target_waveform_fs"]).astype(np.int64)
        indexes = np.unique(indexes)
        if len(indexes):
            cuts = np.flatnonzero(np.diff(indexes) != 1) + 1
            intervals[channel] = [[int(part[0]), int(part[-1]) + 1] for part in np.split(indexes, cuts)]
        else:
            intervals[channel] = []
    valid = numeric.loc[np.isfinite(numeric.MAP) & numeric.MAP.between(config["map_min_mmhg"], config["map_max_mmhg"])].drop_duplicates("time_sec").sort_values("time_sec")
    return {"valid_intervals": intervals, "map_times": valid.time_sec.tolist(), "map_values": valid.MAP.tolist(), "decoding_metadata": metadata}


def _union_count(intervals):
    if not intervals:
        return 0
    intervals = sorted(intervals)
    start, end = intervals[0]
    count = 0
    for a, b in intervals[1:]:
        if a <= end:
            end = max(end, b)
        else:
            count += end - start
            start, end = a, b
    return count + end - start


def _save_fragment(root, db, archive, member, payload, case, config):
    from .mover_signals import decode_xml
    existing = db.execute("SELECT payload FROM fragments WHERE archive=? AND member=? AND case_id=?", (archive, member, case.case_id)).fetchone()
    if existing:
        old = json.loads(existing[0])
        if old.get("processing_signature") == _processing_signature(config) and all((Path(root) / old[k]).is_file() and sha256_file(Path(root) / old[k]) == old[k.replace("_path", "_sha256")] for k in ("waveform_path", "map_path")):
            return old
    waveform, numeric, metadata = decode_xml(payload, float(case.start_utc), float(case.end_utc), config["target_waveform_fs"], track_map=config.get("mover_track_map"))
    _assert_observed_pressure_schema(metadata)
    if waveform.empty:
        return None
    digest = hashlib.sha256((archive + member).encode()).hexdigest()[:24]
    directory = Path(root) / "data" / "parquet" / "mover" / "pilot_fragments" / f"case_id={safe_id(case.case_id)}"
    directory.mkdir(parents=True, exist_ok=True)
    info = {"case_id": case.case_id, "patient_id": case.patient_id, "archive": archive, "member": member,
            "source_bytes": len(payload), "waveform_rows": len(waveform), "map_rows": len(numeric),
            "diagnostic_only": True, "processing_signature": _processing_signature(config), "metadata": metadata}
    for label, frame in (("waveform", waveform), ("map", numeric)):
        ensure_disk_space(root, _reserve(config), int(frame.memory_usage(index=False, deep=True).sum()) * 2)
        path = directory / f"{digest}_{label}.parquet"
        temporary = path.with_name(path.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                frame.to_parquet(handle, index=False, compression="zstd")
            if pq.ParquetFile(temporary).metadata.num_rows != len(frame):
                raise ValueError("Fragment readback row count mismatch")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        info[f"{label}_path"] = str(path.relative_to(root))
        info[f"{label}_sha256"] = sha256_file(path)
    info["output_bytes"] = sum((Path(root) / info[f"{label}_path"]).stat().st_size for label in ("waveform", "map"))
    db.execute("INSERT OR REPLACE INTO fragments VALUES (?, ?, ?, ?)", (archive, member, case.case_id, json.dumps(info, default=str)))
    db.commit()
    return info


def export_inventory(root):
    """Summarize checkpointed archive members, channels, sampling rates, and errors."""
    rows = []
    schema = {"observed_groups": {}, "observed_rates": {}, "observed_numeric_names": set()}
    with _database(root) as db:
        for row in db.execute("SELECT payload FROM archives"):
            state = json.loads(row[0])
            groups, rates, patients = set(), {}, set()
            files, source_bytes, errors = 0, 0, 0
            for item_row in db.execute("SELECT payload FROM members WHERE archive=?", (state["archive"],)):
                item = json.loads(item_row[0])
                files += 1
                source_bytes += item["member_bytes"]
                errors += item.get("status") == "failed"
                patients.add(item["patient_id"])
                groups.update(item.get("groups", []))
                for group, values in item.get("original_fs", {}).items():
                    rates.setdefault(group, set()).update(values)
                schema["observed_numeric_names"].update(item.get("numeric_names", []))
            schema["observed_groups"][state["archive"]] = sorted(groups)
            schema["observed_rates"][state["archive"]] = {k: sorted(v) for k, v in rates.items()}
            rows.append({"archive": state["archive"], "archive_size": state["fingerprint"]["size_bytes"],
                         "sampled_file_count": files, "sampled_patient_count": len(patients),
                         "sampled_source_bytes": source_bytes, "sampled_errors": errors,
                         "scan_completed": bool(state.get("scan_completed")), "total_member_count": files if state.get("scan_completed") else None,
                         "status": state["status"], "observed_groups": ";".join(sorted(groups))})
    frame = pd.DataFrame(rows)
    write_manifest(root, "mover_archive_inventory", frame)
    schema["observed_numeric_names"] = sorted(schema["observed_numeric_names"])
    schema["inventory_complete"] = bool(len(frame) == len(load_config(root)["mover_archives"]) and frame.scan_completed.all())
    atomic_json(Path(root) / "data" / "manifests" / "mover_schema_inventory.json", schema)
    return frame


def _assemble(root, case, fragments, config):
    n = int(np.ceil(case.duration_sec * config["target_waveform_fs"]))
    if case.duration_sec > config["max_case_hours"] * 3600:
        raise ValueError("Case exceeds configured bounded-memory duration")
    wave = pd.DataFrame({"time_sec": np.arange(n) / config["target_waveform_fs"], **{c: np.full(n, np.nan, dtype=np.float32) for c in ("ART", "ECG", "PPG")}})
    numeric_parts = []
    conflicts = 0
    signature = _processing_signature(config)
    for fragment in sorted(fragments, key=lambda f: (f["archive"], f["member"])):
        if fragment.get("processing_signature") != signature:
            raise ValueError("Fragment configuration/decoder changed; rerun selected-member ingestion")
        _assert_observed_pressure_schema(fragment["metadata"])
        for label in ("waveform", "map"):
            path = Path(root) / fragment[f"{label}_path"]
            if sha256_file(path) != fragment[f"{label}_sha256"]:
                raise ValueError("Corrupt fragment; rerun matching archive pass")
        part = pd.read_parquet(Path(root) / fragment["waveform_path"])
        indexes = np.rint(part.time_sec.to_numpy() * config["target_waveform_fs"]).astype(int)
        if np.any(indexes < 0) or np.any(indexes >= n) or len(np.unique(indexes)) != len(indexes):
            raise ValueError("Fragment grid is invalid")
        for channel in ("ART", "ECG", "PPG"):
            old = wave[channel].to_numpy()[indexes]
            new = part[channel].to_numpy()
            differing = np.isfinite(old) & np.isfinite(new) & ~np.isclose(old, new, rtol=1e-5, atol=1e-5)
            conflicts += int(differing.sum())
            if differing.any():
                raise ValueError("Conflicting overlapping waveform packets; manual schema review required")
            finite = np.isfinite(new)
            wave.loc[indexes[finite], channel] = new[finite].astype(np.float32)
        numeric_parts.append(pd.read_parquet(Path(root) / fragment["map_path"]))
    numeric = pd.concat(numeric_parts, ignore_index=True) if numeric_parts else pd.DataFrame(columns=["time_sec", "MAP"])
    if len(numeric):
        if numeric.groupby("time_sec").MAP.nunique().gt(1).any():
            raise ValueError("Conflicting overlapping MAP observations")
        numeric = numeric.drop_duplicates("time_sec").sort_values("time_sec").reset_index(drop=True)
    return wave, numeric, {"duplicate_conflict_count": conflicts}


def quality(case, waveform, numeric, config):
    """Measure waveform missingness and numeric-MAP coverage for one decoded case."""
    missing = {c.lower() + "_missingness": float(1 - np.isfinite(waveform[c]).mean()) for c in ("ART", "ECG", "PPG")}
    return _quality_from_missing(case, missing, numeric, config)


def _quality_from_missing(case, missing, numeric, config):
    valid = numeric.loc[np.isfinite(numeric.MAP) & numeric.MAP.between(config["map_min_mmhg"], config["map_max_mmhg"])].drop_duplicates("time_sec").sort_values("time_sec")
    gaps = np.diff(valid.time_sec.to_numpy())
    # Long unobserved spans do not count as MAP coverage, and are not forward-filled.
    covered = float(np.minimum(gaps, config["map_coverage_gap_cap_sec"]).sum()) if len(gaps) else 0.
    coverage = min(1., covered / case.duration_sec)
    median = float(np.median(gaps)) if len(gaps) else None
    points_per_min = len(valid) / (case.duration_sec / 60)
    reasons = [f"{channel}_missingness" for channel in ("art", "ecg", "ppg") if missing[channel + "_missingness"] > config["maximum_waveform_missingness"]]
    if len(valid) < config["map_min_valid_points"]:
        reasons.append("insufficient_valid_map_points")
    if points_per_min < config["map_min_points_per_min"]:
        reasons.append("low_map_point_frequency")
    if coverage < config["map_min_coverage"]:
        reasons.append("low_map_temporal_coverage")
    if median is None or median > config["map_max_median_gap_sec"]:
        reasons.append("large_map_median_gap")
    return {**missing, "map_valid_points": len(valid), "map_points_per_min": points_per_min,
            "map_coverage": coverage, "map_median_gap": median, "map_max_gap": float(gaps.max()) if len(gaps) else None,
            "map_median": float(valid.MAP.median()) if len(valid) else None,
            "has_art": missing["art_missingness"] < 1, "has_ecg": missing["ecg_missingness"] < 1,
            "has_ppg": missing["ppg_missingness"] < 1, "has_map": bool(len(valid)),
            "provisional_usable": not reasons, "final_usable": not reasons, "exclusion_reason": ";".join(reasons)}


def build_eligibility(root, *, progress=False):
    """Union actual valid-sample intervals across every member and storage shard."""
    config = load_config(root)
    cases = _cases(root)
    records = []
    signature = _processing_signature(config)
    with _database(root) as db:
        states = [json.loads(row[0]) for row in db.execute("SELECT payload FROM archives")]
        complete = len(states) == len(config["mover_archives"]) and all(s.get("scan_completed") and s.get("gzip_crc_verified") and s.get("processing_signature") == signature for s in states)
        case_rows = tqdm(cases.iterrows(), total=len(cases), desc="Checking surgery eligibility", unit="case", mininterval=1, disable=not progress, leave=True)
        for _, case in case_rows:
            record = case.to_dict()
            record["processing_signature"] = signature
            found = [json.loads(r[0]) for r in db.execute("SELECT payload FROM case_stats WHERE case_id=?", (case.case_id,))]
            record.update(source_archive=";".join(sorted({r["archive"] for r in found})), source_member_count=len(found),
                          has_ppg_observed=any("PLETH" in r.get("groups", []) for r in found),
                          eligibility_status="complete" if complete else "pending_complete_scan", final_usable=False)
            if any(r.get("decoding_error") for r in found):
                record.update(final_usable=False, provisional_usable=False, exclusion_reason="corrupt_or_unknown_member_for_patient;uncertain_case_linkage_excluded_conservatively")
            elif case.linkage_usable and found:
                try:
                    for stats in found:
                        if "decoding_metadata" in stats:
                            _assert_observed_pressure_schema(stats["decoding_metadata"])
                    count = int(np.ceil(case.duration_sec * config["target_waveform_fs"]))
                    missing = {c.lower() + "_missingness": max(0., 1 - _union_count([interval for r in found for interval in r["valid_intervals"][c]]) / count) for c in ("ART", "ECG", "PPG")}
                    numeric = pd.DataFrame({"time_sec": [t for r in found for t in r["map_times"]], "MAP": [v for r in found for v in r["map_values"]]})
                    if len(numeric) and numeric.groupby("time_sec").MAP.nunique().gt(1).any():
                        raise ValueError("Conflicting MAP observations across archive members")
                    assessment = _quality_from_missing(case, missing, numeric, config)
                    record.update(assessment)
                    record["final_usable"] = bool(complete and assessment["final_usable"])
                    if not complete:
                        record["exclusion_reason"] = "sample_incomplete;" + assessment["exclusion_reason"]
                    for channel, names in (("art", ("INVP1", "GE_ART")), ("ecg", ("ECG1", "GE_ECG")), ("ppg", ("PLETH",))):
                        rates = sorted({rate for r in found for name in names for rate in r.get("original_fs", {}).get(name, [])})
                        record[channel + "_original_fs"] = ";".join(str(rate) for rate in rates)
                        record[channel + "_final_fs"] = config["target_waveform_fs"]
                except Exception as error:
                    record.update(final_usable=False, exclusion_reason="signal_quality_summary_failed")
                    _error(root, "quality", None, None, case.case_id, error)
            elif case.linkage_usable and complete:
                record["exclusion_reason"] = "no_waveform_members_linked"
            records.append(record)
    _announce(progress, "Eligibility checks finished. Writing CSV/Parquet manifests...")
    cases = pd.DataFrame(records)
    write_manifest(root, "mover_all_cases", cases)
    eligible = cases.loc[cases.final_usable.fillna(False) & cases.linkage_usable].copy()
    eligible["complete_archive_discovery"] = complete
    write_manifest(root, "mover_eligible_cohort", eligible)
    return eligible


def ingest_members(root, case_ids):
    """One sequential archive pass, caching only the requested cases' members."""
    root = Path(root)
    config = load_config(root)
    if not config["allow_full_archive_scan"]:
        raise DiscoveryIncomplete("Full selected-member retrieval decompresses source archives; enable only after bounded discovery and safety tests")
    cases = _cases(root)
    requested_ids = set(case_ids)
    requested = cases.loc[cases.case_id.isin(requested_ids) & cases.linkage_usable]
    by_patient = {p: group for p, group in requested.groupby("patient_id")}
    store = CheckpointStore(root, "mover")
    started, source_bytes, processed_members = time.perf_counter(), 0, 0
    with _database(root) as db:
        required_archives = set(a for value in requested.source_archive for a in str(value).split(";") if a)
        for name in config["mover_archives"]:
            if name not in required_archives:
                continue
            source = root / "data" / "reference" / "mover" / name
            state_row = db.execute("SELECT payload FROM archives WHERE archive=?", (name,)).fetchone()
            if not state_row or not json.loads(state_row[0]).get("scan_completed"):
                raise DiscoveryIncomplete("Complete metadata enumeration is required before selected-case retrieval")
            state = json.loads(state_row[0])
            state["selected_ingestion_completed"] = False
            try:
                with gzip.open(source, "rb") as compressed, tarfile.open(fileobj=compressed, mode="r|") as archive:
                    for member in archive:
                        archive.members.clear()
                        if not member.isfile():
                            continue
                        patient = Path(member.name).name[:16]
                        if patient not in by_patient:
                            continue
                        matched_stats = db.execute("SELECT case_id FROM case_stats WHERE archive=? AND member=?", (name, member.name)).fetchall()
                        matching_ids = {r[0] for r in matched_stats} & requested_ids
                        if not matching_ids:
                            continue
                        ensure_disk_space(root, _reserve(config), member.size * 2)
                        if member.size > config["max_member_bytes"]:
                            _error(root, "ingest", name, member.name, None, "Member exceeds size guard")
                            continue
                        payload = archive.extractfile(member).read()
                        source_bytes += len(payload)
                        for _, case in by_patient[patient].loc[by_patient[patient].case_id.isin(matching_ids)].iterrows():
                            previous = store.get(case.case_id) or {}
                            if previous.get("status") == "completed" and previous.get("processing_signature") == _processing_signature(config) and verify_case(root, previous):
                                continue
                            try:
                                _save_fragment(root, db, name, member.name, payload, case, config)
                                processed_members += 1
                            except DiskSpaceStop:
                                raise
                            except Exception as error:
                                store.update(case.case_id, status="failed", archive=name, archive_member=member.name, error=str(error), retry_count=previous.get("retry_count", 0) + 1)
                                _error(root, "ingest", name, member.name, case.case_id, error)
                    for _ in iter(lambda: compressed.read(1 << 20), b""):
                        pass
                state["selected_ingestion_completed"] = True
                state["selected_ingestion_completed_at"] = utc_now()
            except (DiskSpaceStop, KeyboardInterrupt) as error:
                state["ingestion_pause_reason"] = str(error)
                raise
            finally:
                db.execute("INSERT OR REPLACE INTO archives VALUES (?, ?)", (name, json.dumps(state)))
                db.commit()
    metrics = {"phase": "selected_member_ingestion", "wall_seconds": time.perf_counter() - started, "member_input_bytes": source_bytes,
               "members_processed": processed_members, "requested_cases": len(requested), "requested_patients": int(requested.patient_id.nunique()),
               "peak_rss_bytes_at_finish": psutil.Process().memory_info().rss, "free_disk_bytes": ensure_disk_space(root, _reserve(config))}
    atomic_json(root / "results" / "mover_ingestion_metrics.json", metrics)
    return metrics


def _pilot_case_ids(eligible, config, limit):
    order = eligible.assign(_key=eligible.case_id.map(lambda c: hashlib.sha256(f"{config['random_seed']}:{c}".encode()).hexdigest())).sort_values("_key")
    selected = []
    for archive in config["mover_archives"]:
        group = order.loc[order.source_archive.str.contains(archive, regex=False, na=False)].sort_values("duration_sec")
        if len(group):
            for position in np.linspace(0, len(group) - 1, min(3, len(group)), dtype=int):
                case_id = group.iloc[position].case_id
                if case_id not in selected:
                    selected.append(case_id)
    for case_id in order.case_id:
        if len(selected) >= limit:
            break
        if case_id not in selected:
            selected.append(case_id)
    return selected[:limit]


def pilot(root, limit=None):
    """Validate a bounded set of discovered cases before enabling full processing."""
    root = Path(root)
    config = load_config(root)
    cases = _cases(root)
    store = CheckpointStore(root, "mover")
    inventory_path = root / "data" / "manifests" / "mover_archive_inventory.parquet"
    inventory = pd.read_parquet(inventory_path) if inventory_path.exists() else pd.DataFrame()
    pilot_ids = None
    eligible_count = None
    if len(inventory) == len(config["mover_archives"]) and inventory.scan_completed.all():
        eligible = pd.read_parquet(root / "data" / "manifests" / "mover_eligible_cohort.parquet")
        eligible_count = len(eligible)
        pilot_ids = _pilot_case_ids(eligible, config, limit or config["pilot_cases"])
        ingest_members(root, pilot_ids)
    with _database(root) as db:
        fragments = [json.loads(row[0]) for row in db.execute("SELECT payload FROM fragments")]
    grouped = {}
    for fragment in fragments:
        grouped.setdefault(fragment["case_id"], []).append(fragment)
    candidates = pilot_ids if pilot_ids is not None else sorted(grouped, key=lambda c: hashlib.sha256(f"{config['random_seed']}:{c}".encode()).hexdigest())[:limit or config["pilot_cases"]]
    rows = []
    started = time.perf_counter()
    for case_id in candidates:
        case = cases.loc[cases.case_id.eq(case_id)].iloc[0]
        case_fragments = grouped.get(case_id, [])
        input_signature = hashlib.sha256(json.dumps({"processing": _processing_signature(config), "quality_config": {k: v for k, v in config.items() if k.startswith("map_") or k in ("maximum_waveform_missingness", "minimum_duration_sec")}, "case_start": case.start_utc, "case_end": case.end_utc,
                "fragments": sorted((f["archive"], f["member"], f["waveform_sha256"], f["map_sha256"]) for f in case_fragments)}, sort_keys=True).encode()).hexdigest()
        previous = store.get(case_id)
        if previous and previous.get("input_fragment_signature") == input_signature and ((previous.get("status") == "completed" and verify_case(root, previous)) or previous.get("status") == "excluded"):
            rows.append(previous)
            continue
        store.update(case_id, status="processing", input_fragment_signature=input_signature, patient_id=case.patient_id, started_at=utc_now(), archive=";".join(sorted({f["archive"] for f in case_fragments})), archive_member=";".join(f["member"] for f in case_fragments), retry_count=(previous or {}).get("retry_count", 0))
        try:
            wave, numeric, extra = _assemble(root, case, case_fragments, config)
            assessment = quality(case, wave, numeric, config)
            if not assessment["final_usable"]:
                row = store.update(case_id, status="excluded", **assessment, patient_id=case.patient_id,
                                   diagnostic_only=True, completed_at=utc_now(), duration_sec=float(case.duration_sec))
            else:
                metadata = {**assessment, **extra, "processing_signature": _processing_signature(config), "patient_id": case.patient_id, "split": "unassigned", "pilot_only": True,
                            "duration_sec": float(case.duration_sec), "start_utc": float(case.start_utc), "end_utc": float(case.end_utc),
                            "art_final_fs": config["target_waveform_fs"], "ecg_final_fs": config["target_waveform_fs"], "ppg_final_fs": config["target_waveform_fs"],
                            "source_archive": sorted({f["archive"] for f in grouped[case_id]}),
                            "source_member": [f["member"] for f in grouped[case_id]], "source_bytes": sum(f["source_bytes"] for f in grouped[case_id]),
                            "native_metadata": [f["metadata"] for f in grouped[case_id]], "min_free_space_bytes": _reserve(config)}
                committed = write_case(root, "mover", case_id, wave, numeric, metadata)
                loaded = load_case(root, "mover", case_id)
                if any(not np.isfinite(loaded[c]).any() for c in ("ART", "ECG", "PPG", "MAP")):
                    raise ValueError("Canonical loader failed core signal check")
                row = store.update(case_id, **{k: v for k, v in committed.items() if k != "case_id"}, status="completed", completed_at=utc_now())
            rows.append(row)
        except (DiskSpaceStop, KeyboardInterrupt):
            store.update(case_id, status="pending", error="Interrupted/disk guard; fragments retained for resume")
            raise
        except Exception as error:
            failed_row = store.update(case_id, status="failed", error=str(error), retry_count=(previous or {}).get("retry_count", 0) + 1)
            rows.append(failed_row)
            _error(root, "pilot", None, None, case_id, error)
    frame = pd.DataFrame(rows)
    write_manifest(root, "mover_pilot_cases", frame)
    complete = [r for r in rows if r.get("status") == "completed"]
    hours = sum(r["duration_sec"] for r in complete) / 3600
    output = sum(r.get("processed_bytes", 0) for r in complete)
    input_bytes = sum(r.get("source_bytes", 0) for r in complete)
    metrics = {"cases_tested": len(candidates), "eligible_cases_processed": len(complete), "excluded_cases": int(frame.status.eq("excluded").sum()) if len(frame) else 0,
               "waveform_hours": hours, "input_member_bytes": input_bytes, "processed_bytes": output,
               "diagnostic_fragment_bytes": sum(f["output_bytes"] for f in fragments), "source_to_output_ratio": input_bytes / output if output else None,
               "bytes_per_waveform_hour": output / hours if hours else None, "wall_seconds": time.perf_counter() - started,
               "required_eligible_pilot_cases": min(10, eligible_count) if eligible_count is not None else None,
               "full_case_size_estimate_valid": bool(eligible_count and len(complete) >= min(10, eligible_count) and len(inventory) == len(config["mover_archives"]) and inventory.scan_completed.all()),
               "processing_failures": int(frame.status.eq("failed").sum()) if len(frame) else 0,
               "note": "Sampled/partial fragments are not a complete cohort; do not extrapolate GB or patient counts from missing/excluded cases."}
    atomic_json(root / "results" / "mover_pilot_metrics.json", metrics)
    deletion_readiness(root)
    return metrics


def select_cohort(root):
    """Freeze eligible surgeries and patient-level partitions after discovery and validation."""
    root = Path(root)
    config = load_config(root)
    inventory = pd.read_parquet(root / "data" / "manifests" / "mover_archive_inventory.parquet")
    metrics = json.loads((root / "results" / "mover_pilot_metrics.json").read_text())
    if not inventory.scan_completed.all() or len(inventory) != len(config["mover_archives"]):
        raise DiscoveryIncomplete("Complete all archive discovery passes before final patient selection")
    eligible = build_eligibility(root)
    if eligible.empty:
        raise DiscoveryIncomplete("No verified eligible surgeries. ART+ECG+PPG+arterial MAP criteria have NOT been relaxed.")
    if not metrics["full_case_size_estimate_valid"] or not metrics["bytes_per_waveform_hour"]:
        raise DiscoveryIncomplete("Need ten complete eligible pilot cases, or every eligible case when fewer than ten exist, for a defensible size estimate")
    measured = {r["case_id"]: r["processed_bytes"] for r in CheckpointStore(root, "mover").rows() if r.get("status") == "completed" and verify_case(root, r)}
    eligible["estimated_bytes"] = eligible.apply(lambda r: measured.get(r.case_id, r.duration_sec / 3600 * metrics["bytes_per_waveform_hour"]), axis=1)
    patient_sizes = eligible.groupby("patient_id").estimated_bytes.sum()
    total = float(patient_sizes.sum())
    if total <= config["max_mover_size_gb"] * 1e9:
        selected_patients = set(patient_sizes.index)
    else:
        order = np.random.default_rng(config["random_seed"]).permutation(sorted(patient_sizes.index))
        selected_patients, size = set(), 0.
        for patient in order:
            estimate = float(patient_sizes.loc[patient])
            if size + estimate <= config["target_mover_size_gb"] * 1e9:
                selected_patients.add(patient)
                size += estimate
    selected = eligible.loc[eligible.patient_id.isin(selected_patients)].copy()
    splits = patient_splits(selected.patient_id, config["patient_split_seed"])
    selected = selected.drop(columns="split").merge(splits, on="patient_id", how="left", validate="many_to_one")
    overlaps = assert_patient_independence(selected)
    write_manifest(root, "mover_selected_cohort", selected)
    write_manifest(root, "mover_patient_splits", splits)
    atomic_json(root / "results" / "mover_selection_summary.json", {"candidate_patients": int(_cases(root).patient_id.nunique()),
                "candidate_surgeries": len(_cases(root)), "eligible_patients": int(eligible.patient_id.nunique()), "eligible_surgeries": len(eligible),
                "selected_patients": len(selected_patients), "selected_surgeries": len(selected), "estimated_bytes": float(selected.estimated_bytes.sum()),
                "patients_per_split": splits.split.value_counts().to_dict(), "surgeries_per_split": selected.split.value_counts().to_dict(), "patient_intersections": overlaps,
                "selection_unit": "patient; all eligible surgeries retained", "seed": config["random_seed"]})
    deletion_readiness(root)
    return selected


def process_selected(root):
    """Process the selected cohort only after the full-processing gate is enabled."""
    root = Path(root)
    config = load_config(root)
    if not config["allow_full_process"]:
        raise DiscoveryIncomplete("FULL_PROCESS remains disabled until complete discovery, pilot validation, selection, and safety tests pass")
    selected_path = root / "data" / "manifests" / "mover_selected_cohort.parquet"
    if not selected_path.exists():
        raise DiscoveryIncomplete("No verified patient-selected cohort")
    selected = pd.read_parquet(selected_path)
    assert_patient_independence(selected)
    store = CheckpointStore(root, "mover")
    signature = _processing_signature(config)
    if not selected.processing_signature.eq(signature).all():
        raise DiscoveryIncomplete("Selected cohort processing configuration changed; refresh complete discovery and selection")
    previously_verified = {r["case_id"]: r for r in store.rows() if r.get("status") == "completed" and r.get("processing_signature") == signature and r.get("case_id") in set(selected.case_id) and verify_case(root, r)}
    total_bytes = sum(r["processed_bytes"] for r in previously_verified.values())
    ingest_members(root, selected.case_id)
    with _database(root) as db:
        for index, case in selected.iterrows():
            previous = store.get(case.case_id)
            if previous and previous.get("status") == "completed" and previous.get("processing_signature") == signature and previous.get("split") == case.split and verify_case(root, previous):
                for key in ("waveform_output_path", "map_output_path", "processed_bytes"):
                    selected.loc[index, key] = previous[key]
                continue
            try:
                fragments = [json.loads(r[0]) for r in db.execute("SELECT payload FROM fragments WHERE case_id=?", (case.case_id,))]
                if not fragments:
                    raise DiscoveryIncomplete("Selected case fragments not ingested; complete a selected-members archive pass first")
                wave, numeric, extra = _assemble(root, case, fragments, config)
                assessment = quality(case, wave, numeric, config)
                if not assessment["final_usable"]:
                    raise ValueError("Selected case no longer satisfies quality criteria")
                previous_bytes = previously_verified.get(case.case_id, {}).get("processed_bytes", 0)
                remaining = int(config["max_mover_size_gb"] * 1e9) - total_bytes + previous_bytes
                if remaining <= 0:
                    raise DiskSpaceStop("Maximum MOVER output limit reached; preserve checkpoint and revise patient-level size estimate")
                metadata = {**extra, **assessment, "processing_signature": _processing_signature(config), "max_case_output_bytes": remaining, "patient_id": case.patient_id, "split": case.split,
                            "duration_sec": case.duration_sec, "start_utc": case.start_utc, "end_utc": case.end_utc,
                            "source_archive": sorted({f["archive"] for f in fragments}), "source_member": [f["member"] for f in fragments],
                            "native_metadata": [f["metadata"] for f in fragments], "art_final_fs": config["target_waveform_fs"],
                            "ecg_final_fs": config["target_waveform_fs"], "ppg_final_fs": config["target_waveform_fs"], "min_free_space_bytes": _reserve(config)}
                committed = write_case(root, "mover", case.case_id, wave, numeric, metadata)
                total_bytes += committed["processed_bytes"] - previous_bytes
                store.update(case.case_id, **{k: v for k, v in committed.items() if k != "case_id"}, status="completed", completed_at=utc_now())
                for key in ("waveform_output_path", "map_output_path", "processed_bytes"):
                    selected.loc[index, key] = committed[key]
                for fragment in fragments:
                    for label in ("waveform", "map"):
                        (root / fragment[f"{label}_path"]).unlink(missing_ok=True)
                db.execute("DELETE FROM fragments WHERE case_id=?", (case.case_id,))
                db.commit()
            except (DiskSpaceStop, KeyboardInterrupt):
                store.update(case.case_id, status="pending", error="Clean pause; raw sources and fragments retained")
                raise
            except Exception as error:
                store.update(case.case_id, status="failed", error=str(error), retry_count=(previous or {}).get("retry_count", 0) + 1)
                _error(root, "full_process", None, None, case.case_id, error)
    write_manifest(root, "mover_processed_cases", pd.DataFrame(store.rows()))
    write_manifest(root, "mover_selected_cohort", selected)
    deletion_readiness(root)
    return eda(root)


def deletion_readiness(root):
    """Report whether verified case outputs cover each archive; do not delete raw data."""
    root = Path(root)
    config = load_config(root)
    selected_path = root / "data" / "manifests" / "mover_selected_cohort.parquet"
    selected = pd.read_parquet(selected_path) if selected_path.exists() else pd.DataFrame()
    store = CheckpointStore(root, "mover")
    records = {r["case_id"]: r for r in store.rows()}
    expected_by_archive = {name: set() for name in config["mover_archives"]}
    selected_ids = set(selected.case_id) if len(selected) else set()
    with _database(root) as db:
        states = {r["archive"]: json.loads(r["payload"]) for r in db.execute("SELECT archive,payload FROM archives")}
        for name in config["mover_archives"]:
            for row in db.execute("SELECT DISTINCT case_id FROM case_stats WHERE archive=?", (name,)):
                if row[0] in selected_ids:
                    expected_by_archive[name].add(row[0])
    splits_pass = bool(len(selected)) and not any(assert_patient_independence(selected).values())
    rows = []
    for name in config["mover_archives"]:
        expected = expected_by_archive[name]
        completed = [records[c] for c in expected if records.get(c, {}).get("status") == "completed"]
        verified = [r for r in completed if verify_case(root, r)]
        failed = sum(records.get(c, {}).get("status") == "failed" for c in expected)
        missing = len(expected) - len(verified)
        scan_complete = bool(states.get(name, {}).get("scan_completed") and states.get(name, {}).get("gzip_crc_verified"))
        manifests_valid = bool(len(selected)) and all(all(key in selected and selected.loc[selected.case_id.eq(c), key].iloc[0] == records.get(c, {}).get(key) for key in ("waveform_output_path", "map_output_path")) for c in expected)
        loader_valid = False
        if verified:
            try:
                load_case(root, "mover", verified[0]["case_id"])
                loader_valid = True
            except Exception:
                pass
        safe = bool(scan_complete and expected and missing == 0 and failed == 0 and splits_pass and manifests_valid and loader_valid)
        reasons = []
        for condition, reason in ((scan_complete, "complete archive scan not verified"), (bool(expected), "final selected cases not established"),
                                  (missing == 0, "selected outputs missing/corrupt"), (failed == 0, "unresolved selected failures"),
                                  (splits_pass, "final patient split not verified"), (manifests_valid, "final manifests do not point to verified outputs"),
                                  (loader_valid, "canonical loader not verified for this archive")):
            if not condition:
                reasons.append(reason)
        rows.append({"archive": name, "archive_size": (root / "data" / "reference" / "mover" / name).stat().st_size,
                     "archive_scan_completed": scan_complete, "selected_cases_expected": len(expected), "selected_cases_processed": len(completed),
                     "selected_cases_verified": len(verified), "selected_cases_failed": failed, "selected_cases_missing": missing,
                     "processed_output_bytes": sum(r.get("processed_bytes", 0) for r in verified), "safe_to_delete": safe,
                     "reason": ";".join(reasons) if reasons else "All independently verified; user must decide deletion and backup policy"})
    frame = pd.DataFrame(rows)
    write_manifest(root, "mover_raw_archive_deletion_readiness", frame)
    return frame


def write_schema_audit(root):
    """Document observed linkage, waveform schemas, and processing evidence."""
    root = Path(root)
    config = load_config(root)
    linkage = json.loads((root / "data/manifests/mover_patient_linkage_audit.json").read_text())
    inventory = pd.read_parquet(root / "data/manifests/mover_archive_inventory.parquet")
    schema = json.loads((root / "data/manifests/mover_schema_inventory.json").read_text())
    pilot_path = root / "results/mover_pilot_metrics.json"
    pilot_metrics = json.loads(pilot_path.read_text()) if pilot_path.exists() else {}
    text = "# MOVER schema audit\n\n## OBSERVED FACTS\n\n"
    text += f"- Crosswalk: {linkage['crosswalk_rows']:,} rows; {linkage['unique_patients_crosswalk']:,} PAT_ID patients; {linkage['unique_cases_crosswalk']:,} LOG_ID surgeries. MRN is a second patient identifier. Null identifier counts: {linkage['null_identifiers']}.\n"
    text += f"- Exact duplicate rows: {linkage['duplicate_crosswalk_rows']:,}. {linkage['ambiguous_LOG_ID_count']} LOG_IDs map to multiple patients and are excluded, not guessed. {linkage['patients_with_multiple_surgeries']:,} patients have multiple surgeries; maximum {linkage['maximum_surgeries_per_patient']}.\n"
    text += f"- EMR patient_information.csv: {linkage['clinical_rows']:,} rows / {linkage['unique_clinical_cases']:,} unique LOG_ID; {linkage['cases_with_reliable_linkage_and_window']:,} have reliable linkage, nonambiguous OR timestamps and configured duration. Clinical times have month/day/two-digit-year hour:minute formatting. IN_OR_DTTM/OUT_OR_DTTM are local Los Angeles times; cpc.datetime is UTC. 300 members across three shards validated surgery overlap (64/72/72 matches with timezone conversion versus 7/8/6 with naive UTC). DST ambiguity is excluded.\n"
    text += "- Archive format is gzip-compressed TAR, not RAR. File hierarchy observed: shard directory / UCI_deidentified_part*_EPIC_* / Waveforms / two-hex-prefix / PAT_ID{CB or IP}-UTC-timestamp.xml. Filename patient prefix is NOT LOG_ID. XML is cpcArchive with nominal duration=30 minutes, multiple timestamped cpc/device records; each surgery can use multiple members.\n"
    text += "- Observed GE_ART and GE_ECG: 180 Hz. GE_S5 INVP1: 100 Hz, gain .01; ECG1: 300 Hz; PLETH: 100 Hz. Samples are base64 signed little-endian int16 with per-channel gain metadata. -32767 is an observed missing marker. WaveDataGap is retained. Anti-alias polyphase resampling operates on contiguous valid segments; absent samples remain NaN.\n"
    text += "- ART: INVP1 or GE_ART. ECG: ECG1 or GE_ECG. PPG: PLETH, not numeric SPO2. Older AnalogWaves members contain ART/ECG only; do not fabricate PPG for them.\n"
    text += "- S5 arterial numeric MAP: IBP_MEAN occurrence 1, explicitly configured as IBP_MEAN:1 (not NIBP_MEAN). Six ordered SYS/MEAN/DIA occurrences have no channel-label attribute. INVP1 pressure agreement was checked across three different patients: 2,173 SYS>=MAP>=DIA triplets; waveform-mean/MAP MAE 1.961, 6.143, 1.389 mmHg. This supports the observed S5 first channel only. Unknown layouts must fail closed. ECG/PPG physical units remain unverified; native amplitude is retained.\n"
    text += "- Auxiliary PacketTime/POLLTIME examples are stale relative to deidentified cpc.datetime. They are not used to align surgeries.\n"
    text += "- EMR tables observed without extraction: patient_information.csv (18,662,515 bytes), patient_medications.csv (7,713,625,486), patient_lda.csv (129,572,495), patient_labs.csv (3,568,653,054). EMR member enumeration is incomplete. No full archive extraction occurred.\n"
    text += "- Flowsheet first table observed: flowsheets_cleaned/flowsheet_part7.csv, 5,883,521,632 bytes; columns LOG_ID, MRN, FLO_NAME, FLO_DISPLAY_NAME, RECORD_TYPE, RECORDED_TIME, MEAS_VALUE, UNITS (plus saved index). Timestamps resemble 2022-06-19 22:00:00. First100,000 rows contain MAP (mmHg), Arterial Line MAP (ART), MAP-ART A-line and NIBP - MAP. This mixed perioperative table has NOT been substituted for selected native arterial MAP. Full flowsheet inventory/MAP extraction is pending if required.\n\n"
    text += "### Bounded waveform inventory\n\n| Archive | Compressed bytes | XML files observed | Patients observed | Uncompressed member bytes inspected | Complete scan |\n|---|---:|---:|---:|---:|---|\n"
    for _, row in inventory.iterrows():
        text += f"| {row.archive} | {int(row.archive_size):,} | {int(row.sampled_file_count):,} | {int(row.sampled_patient_count):,} | {int(row.sampled_source_bytes):,} | {bool(row.scan_completed)} |\n"
    text += "\n## INFERENCES / ASSUMPTIONS AND UNRESOLVED QUESTIONS\n\n"
    text += "- A bounded prefix is schema discovery, not a representative final patient cohort or a complete archive census. Total archive member counts and whether patients/surgeries span archives remain unknown until complete enumeration. No estimate is presented as an observed total.\n"
    text += "- Native arterial MAP is preferred to avoid flowsheet timing/coverage ambiguity. Six-pressure-channel ordering must be validated for any newly observed device/schema. Nonzero gain offsets and unknown encodings are not inferred.\n"
    text += "- Complete eligibility requires >=30min, all four modalities, waveform missingness<=20%, >=300 valid arterial MAP points, sufficient natural MAP cadence, gap-capped temporal coverage>=80%, median MAP gap<=10s. Coverage uses actual valid time intervals, not merely first-to-last span. Physiological MAP range is a quality check only; no outcome-based cohort selection.\n"
    text += "- Final MOVER patient selection/splitting is withheld until complete inventory and a validated full-case pilot support the GB estimate. Every selected patient's eligible surgeries remain together. No forecast windows, BSHI or model training.\n"
    text += "- Full scan entails sequential gzip decompression, including when resuming. Verified member processing/output is skipped, but seeking directly to a compressed member is not promised. Temporary selected fragments are removed only after canonical case verification; raw archives are never deleted.\n\n"
    text += "## PILOT / PROCESSING STATUS\n\n```json\n" + json.dumps(pilot_metrics, indent=2) + "\n```\n\n"
    diagnostic_path = root / "results/mover_diagnostic_metrics.json"
    if diagnostic_path.exists():
        text += "### Actual diagnostic physiological data (not eligible full cases)\n\n```json\n" + diagnostic_path.read_text() + "\n```\n\n"
        text += "The 13 bounded diagnostic surgeries did not meet full-case waveform coverage. This is NOT an estimate of the global eligible count. 59 member fragments from 8 surgeries nevertheless contain actual finite ART/ECG/PPG/arterial MAP; every fragment was read back and hash-verified. Stored grid-hours can overlap and include missing rows, so they are not unique valid four-signal hours. Local PPG flat periods were observed; no advanced SQI or imputation was applied. A 120 GB patient-count estimate and real complete-case cross-dataset loader verification remain blocked until eligible full surgeries are processed.\n\n"
    text += f"Configuration: seed {config['random_seed']}; patient split seed {config['patient_split_seed']}; target {config['target_mover_size_gb']} decimal GB, maximum {config['max_mover_size_gb']} GB, reserve {config['min_free_space_gb']} GB. FULL_PROCESS is disabled until real-case pilot and complete discovery pass.\n"
    path = root / "data/manifests/mover_schema_audit.md"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def eda(root):
    """Summarize completed MOVER cases, output sizes, patient counts, and signal coverage."""
    root = Path(root)
    records = pd.DataFrame(CheckpointStore(root, "mover").rows())
    if records.empty:
        return {"processed_cases": 0, "status": "no completed cohort"}
    completed = records.loc[records.status.eq("completed")].copy()
    write_manifest(root, "mover_processing_summary", records)
    result = {"selected_patients_processed": int(completed.patient_id.nunique()) if len(completed) else 0,
              "selected_surgeries_processed": len(completed), "total_waveform_hours": float(completed.duration_sec.sum() / 3600) if len(completed) else 0,
              "processed_bytes": int(completed.processed_bytes.sum()) if len(completed) else 0,
              "failures": int(records.status.eq("failed").sum()), "exclusions": int(records.status.eq("excluded").sum())}
    if len(completed):
        result["patients_per_split"] = completed.groupby("split").patient_id.nunique().to_dict()
        result["surgeries_per_split"] = completed.split.value_counts().to_dict()
        result["maximum_surgeries_per_patient"] = int(completed.groupby("patient_id").size().max())
        result["quality_distributions"] = completed[[c for c in ("duration_sec", "art_missingness", "ecg_missingness", "ppg_missingness", "map_valid_points", "map_coverage", "map_median_gap", "map_median") if c in completed]].describe().to_dict()
        if completed.split.isin(["train", "validation", "test"]).all():
            result["patient_intersections"] = assert_patient_independence(completed)
        import matplotlib.pyplot as plt
        plot_columns = [c for c in ("duration_sec", "art_missingness", "ecg_missingness", "ppg_missingness", "map_coverage") if c in completed]
        if plot_columns:
            completed[plot_columns].hist(figsize=(12, 8))
            plt.tight_layout()
            (root / "figures").mkdir(exist_ok=True)
            plt.savefig(root / "figures" / "mover_processed_quality.png", dpi=120)
            plt.close("all")
    atomic_json(root / "results" / "mover_processed_eda.json", result)
    return result
