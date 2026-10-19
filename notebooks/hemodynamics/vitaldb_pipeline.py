"""Audited, case-bounded VitalDB retrieval; original references/cohorts are read-only."""
from __future__ import annotations
# Retrieve and verify the selected VitalDB tracks with bounded, resumable work.

from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import gc
import hashlib
import heapq
import multiprocessing as mp
import pickle
import queue
import shutil
from fractions import Fraction
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
from typing import Any
import urllib.request

import numpy as np
import pandas as pd
import requests
from scipy.signal import resample_poly
import vitaldb
from vitaldb import dataset as vital_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESAMPLING_VERSION = "polyphase_contiguous_v1"
MODES = {"PILOT", "FULL_DOWNLOAD", "RESUME"}
MAX_DOWNLOAD_WORKERS = 12
SPLIT_ORDER = ("train", "validation", "test")
PHASES = ("network", "resampling", "writing", "readback")


def load_settings(root: Path | None = None) -> dict[str, Any]:
    # Load the recorded settings or metadata for this processing step.
    path = Path(root or PROJECT_ROOT) / "configs" / "hemodynamics.json"
    settings = json.loads(path.read_text(encoding="utf-8"))
    required = ("random_seed", "patient_split_seed", "min_free_space_gb", "target_waveform_fs", "max_retries", "allow_full_vitaldb_download", "vitaldb")
    if not all(key in settings for key in required):
        raise ValueError("Missing shared VitalDB configuration")
    if float(settings["target_waveform_fs"]) <= 0 or float(settings["min_free_space_gb"]) < 0 or int(settings["max_retries"]) < 1:
        raise ValueError("Invalid target sampling rate, disk reserve or retry count")
    fractions = settings["vitaldb"]["patient_split_fractions"]
    if any(float(fractions[name]) <= 0 for name in ("train", "validation", "test")) or not math.isclose(sum(fractions.values()), 1.0):
        raise ValueError("Patient split fractions must sum to one")
    if len(settings["vitaldb"]["pilot_duration_quantiles"]) != 3:
        raise ValueError("This pipeline requires exactly three deterministic pilot quantiles")
    return settings


def processing_signature(settings: dict[str, Any] | None = None) -> str:
    # Store the settings and results as JSON for later inspection.
    settings = settings or load_settings()
    vital = settings["vitaldb"]
    payload = {"resampling_version": RESAMPLING_VERSION, "time_reference": "seconds_since_recording_start", "dataset_version": vital_dataset.DATASET_VERSION,
               "patient_split_seed": settings["patient_split_seed"], "target_waveform_fs": settings["target_waveform_fs"],
               **{key: vital[key] for key in ("waveform_tracks", "map_track_priority", "patient_split_fractions", "resample_window", "resample_padtype", "quality_thresholds")}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# Public inspection constants are sourced from the central configuration, never independent defaults.
_INITIAL_SETTINGS = load_settings()
SEED = int(_INITIAL_SETTINGS["patient_split_seed"])
RESERVE_BYTES = int(float(_INITIAL_SETTINGS["min_free_space_gb"]) * 1_000_000_000)
WAVEFORM_INTERVAL = 1.0 / float(_INITIAL_SETTINGS["target_waveform_fs"])
TRACKS = _INITIAL_SETTINGS["vitaldb"]["waveform_tracks"]
MAP_PRIORITY = tuple(_INITIAL_SETTINGS["vitaldb"]["map_track_priority"])
QUALITY = _INITIAL_SETTINGS["vitaldb"]["quality_thresholds"]


def _canonical():
    from hemodynamics import canonical
    return canonical


def _atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".part")
    try:
        pending.write_text(json.dumps(data, indent=2, allow_nan=False, default=str), encoding="utf-8")
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _atomic_frame(path: Path, data: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".part")
    try:
        if path.suffix == ".parquet":
            data.to_parquet(pending, index=False)
        else:
            data.to_csv(pending, index=False)
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _retry(operation, attempts: int | None = None, backoff_seconds: float | None = None):
    settings = load_settings()
    attempts = int(settings["max_retries"]) if attempts is None else attempts
    backoff_seconds = float(settings["vitaldb"]["retry_backoff_seconds"]) if backoff_seconds is None else backoff_seconds
    if attempts < 1:
        raise ValueError("attempts must be positive")
    for attempt in range(attempts):
        try:
            return operation()
        except (requests.RequestException, OSError, EOFError, RuntimeError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"Operation failed after {attempts} attempts: {exc}") from exc
            time.sleep(backoff_seconds * 2 ** attempt)


_TRACK_CACHE: dict[str, Any] = {}


def load_track_index(root: Path, refresh: bool = False) -> pd.DataFrame:
    """Cache only public track identifiers, not patient data or complete raw files."""
    settings = load_settings(root)
    options = settings["vitaldb"]
    path = Path(root) / "data" / "checkpoints" / "vitaldb" / "track_index.csv"
    if refresh or not path.exists():
        def fetch():
            # Read the saved table and retain its named columns.
            response = requests.get(f"{vital_dataset.api_url}/trks", timeout=(options["request_connect_timeout_seconds"], options["index_read_timeout_seconds"]))
            response.raise_for_status()
            from io import StringIO
            frame = pd.read_csv(StringIO(response.text))
            if not {"caseid", "tid", "tname"}.issubset(frame.columns):
                raise RuntimeError("Unexpected public track-index schema")
            return frame
        _atomic_frame(path, _retry(fetch, int(settings["max_retries"]), float(options["retry_backoff_seconds"])))
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _TRACK_CACHE.get("key") == key:
        return _TRACK_CACHE["frame"]
    frame = pd.read_csv(path)
    if not {"caseid", "tid", "tname"}.issubset(frame.columns):
        raise ValueError("Invalid cached VitalDB track index")
    _TRACK_CACHE.clear()
    _TRACK_CACHE.update(key=key, frame=frame)
    return frame


def track_lookup(index: pd.DataFrame) -> dict[int, set]:
    """caseid -> track names; memoized for the cached index so audit and dispatch group it once."""
    if _TRACK_CACHE.get("frame") is index and "lookup" in _TRACK_CACHE:
        return _TRACK_CACHE["lookup"]
    lookup = {int(case): set(group.tname) for case, group in index.groupby("caseid", sort=False)}
    if _TRACK_CACHE.get("frame") is index:
        _TRACK_CACHE["lookup"] = lookup
    return lookup


def resolve_tracks(index: pd.DataFrame, case_id: int, settings: dict | None = None) -> dict[str, str]:
    settings = settings or load_settings()
    if isinstance(index, dict):
        available = index.get(int(case_id), set())
    else:
        available = set(index.loc[index.caseid.eq(case_id), "tname"])
    result = dict(settings["vitaldb"]["waveform_tracks"])
    absent = set(result.values()) - available
    if absent:
        raise ValueError(f"Case {case_id} lacks audited waveform tracks: {sorted(absent)}")
    result["MAP"] = next((name for name in settings["vitaldb"]["map_track_priority"] if name in available), "")
    if not result["MAP"]:
        raise ValueError(f"Case {case_id} lacks audited invasive arterial MAP")
    return result


def patient_splits(cohort: pd.DataFrame, clinical: pd.DataFrame, seed: int | None = None, settings: dict | None = None) -> pd.DataFrame:
    """Use the documented EMR subjectid, never the surgery ID, as patient linkage."""
    settings = settings or load_settings()
    seed = int(settings["patient_split_seed"]) if seed is None else seed
    fractions = settings["vitaldb"]["patient_split_fractions"]
    if clinical.caseid.duplicated().any() or cohort.caseid.duplicated().any():
        raise ValueError("Non-unique case IDs")
    linkage = cohort[["caseid"]].merge(clinical[["caseid", "subjectid"]], on="caseid", how="left", validate="one_to_one")
    if linkage.subjectid.isna().any():
        raise ValueError("Missing documented patient linkage: splits cannot be assigned")
    # Seed the generated values so this check can be repeated.
    patients = np.sort(linkage.subjectid.unique())
    np.random.default_rng(seed).shuffle(patients)
    train_end = int(len(patients) * fractions["train"])
    validation_end = train_end + int(len(patients) * fractions["validation"])
    mapping = {int(pid): ("train" if i < train_end else "validation" if i < validation_end else "test") for i, pid in enumerate(patients)}
    linkage = linkage.rename(columns={"caseid": "case_id", "subjectid": "patient_id"})
    linkage["patient_id"] = linkage.patient_id.astype("int64")
    # Summarize the records separately within each requested group.
    linkage["split"] = linkage.patient_id.map(mapping)
    linkage["patient_id_source"] = "clinical_data.csv:subjectid (EMR deidentified hospital patient ID)"
    linkage["seed"] = seed
    groups = {s: set(g.patient_id) for s, g in linkage.groupby("split")}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if groups[a] & groups[b]:
            raise AssertionError("Patient leakage between splits")
    return linkage.sort_values("case_id").reset_index(drop=True)


def pilot_case_ids(cohort: pd.DataFrame, settings: dict | None = None) -> list[int]:
    """Three outcome-blind duration representatives: 10th, 50th and 90th percentiles."""
    ordered = cohort.sort_values(["duration_seconds", "caseid"]).reset_index(drop=True)
    if len(ordered) < 3:
        raise ValueError("At least three usable cases are required")
    settings = settings or load_settings()
    positions = [int((len(ordered) - 1) * q) for q in settings["vitaldb"]["pilot_duration_quantiles"]]
    ids = [int(ordered.iloc[i].caseid) for i in positions]
    if len(set(ids)) != 3:
        raise ValueError("Pilot selection is not three unique cases")
    return ids


def representative_order(links: pd.DataFrame, seed: int, done=(), cases=None) -> list[int]:
    """Deterministic dispatch order for a full run.

    Every prefix mirrors the split mix (about 70/15/15 patients), and each patient's
    first not-yet-covered surgery precedes all repeat surgeries. ``links`` needs
    case_id/patient_id/split. ``done`` cases are never returned but count as covering
    their patient, so repeat surgeries of an already-covered patient go last.
    ``cases`` restricts what is returned; the shuffle is over the whole linkage table,
    so each case keeps the same place in its split queue however many cases are finished.
    """
    frame = links.loc[:, ["case_id", "patient_id", "split"]].copy()
    frame["case_id"] = frame.case_id.astype("int64")
    if frame.case_id.duplicated().any():
        raise ValueError("Duplicate case IDs in patient linkage")
    # Seed the generated values so this check can be repeated.
    wanted = None if cases is None else {int(case) for case in cases}
    done = {int(case) for case in done}
    frame = frame.sort_values("case_id", kind="stable").reset_index(drop=True)
    shuffled = frame.iloc[np.random.default_rng(int(seed)).permutation(len(frame))]
    rank = Counter(int(patient) for case, patient in zip(frame.case_id, frame.patient_id) if int(case) in done)
    tiers: dict[int, dict[str, list[int]]] = {}
    for case, patient, split in zip(shuffled.case_id, shuffled.patient_id, shuffled.split):
        if int(case) in done or (wanted is not None and int(case) not in wanted):
            continue
        tier = rank[int(patient)]
        rank[int(patient)] += 1
        tiers.setdefault(tier, {}).setdefault(str(split), []).append(int(case))
    order: list[int] = []
    # Process the selected records in order and retain their individual results.
    for tier in sorted(tiers):
        splits = sorted(tiers[tier], key=lambda name: (SPLIT_ORDER.index(name) if name in SPLIT_ORDER else len(SPLIT_ORDER), name))
        queues = {name: deque(tiers[tier][name]) for name in splits}
        total = sum(len(group) for group in queues.values())
        share = {name: len(queues[name]) / total for name in splits}
        emitted = dict.fromkeys(splits, 0)
        for step in range(1, total + 1):
            # Largest-deficit scheduling keeps each split within about one case of its share at every prefix.
            pick = max((name for name in splits if queues[name]), key=lambda name: (share[name] * step - emitted[name], -splits.index(name)))
            order.append(queues[pick].popleft())
            emitted[pick] += 1
    return order


def audit_project(root: Path, refresh_index: bool = False, phase=None, strict_tracks: bool = True) -> dict[str, Any]:
    # Check cohort and track evidence before any selected recording is downloaded.
    root = Path(root)
    settings = load_settings(root)
    options = settings["vitaldb"]
    # Read the saved table and retain its named columns.
    quality = options["quality_thresholds"]
    signature = processing_signature(settings)
    manifest_dir = root / "data" / "manifests"
    reference = root / "data" / "reference" / "vitaldb"
    # Read the saved table and retain its named columns.
    cohort = pd.read_csv(manifest_dir / "usable_cohort.csv")
    full = pd.read_csv(manifest_dir / "stage_d_full_cohort.csv")
    clinical = pd.read_csv(reference / "clinical_data.csv")
    parameters = pd.read_csv(reference / "clinical_parameters.csv")
    tracks = pd.read_csv(reference / "track_names.csv")
    definition = parameters.loc[parameters.Parameter.eq("subjectid")]
    if len(definition) != 1 or "patient" not in str(definition.iloc[0]["Description"]).lower():
        raise ValueError("The patient identity field lacks reliable documentation")
    if not cohort.usable.all() or not cohort.processing_status.eq("checked").all():
        raise ValueError("The final usable cohort is not fully checked")
    pd.testing.assert_frame_equal(
        full.loc[full.usable].reset_index(drop=True), cohort.reset_index(drop=True), check_dtype=False,
    )
    # Process the selected records in order and retain their individual results.
    for name, csv in (("usable_cohort", cohort), ("stage_d_full_cohort", full)):
        parquet = pd.read_parquet(manifest_dir / f"{name}.parquet")
        for column in csv:
            if column == "split":
                if not csv[column].fillna("").eq(parquet[column].fillna("")).all():
                    raise ValueError("Original CSV/Parquet splits differ")
            elif pd.api.types.is_numeric_dtype(csv[column]):
                if not np.allclose(csv[column], parquet[column], equal_nan=True, atol=1e-10, rtol=1e-10):
                    raise ValueError(f"Original CSV/Parquet differ: {column}")
            elif not csv[column].fillna("").equals(parquet[column].fillna("")):
                raise ValueError(f"Original CSV/Parquet differ: {column}")
    criteria_mask = (
        cohort.duration_minutes.ge(quality["duration_minutes_min"]) & cohort.has_art & cohort.has_ecg & cohort.has_ppg & cohort.has_map
        & cohort.missing_art_pct.lt(quality["waveform_missing_pct_strict_max"])
        & cohort.missing_ecg_pct.lt(quality["waveform_missing_pct_strict_max"])
        & cohort.missing_ppg_pct.lt(quality["waveform_missing_pct_strict_max"])
        & cohort.map_valid_count.ge(quality["map_valid_count_min"])
        & cohort.map_measurements_per_minute.ge(quality["map_measurements_per_minute_min"])
        & cohort.map_coverage_span_pct.ge(quality["map_coverage_span_pct_min"])
        & cohort.map_median_gap_sec.le(quality["map_median_gap_sec_max"])
    )
    if not criteria_mask.all():
        raise ValueError("Existing cohort violates its notebook eligibility criteria")
    for name in (*options["waveform_tracks"].values(), *options["map_track_priority"]):
        if not tracks.Parameter.eq(name).any():
            raise ValueError(f"Track missing from reference dictionary: {name}")
    splits = patient_splits(cohort, clinical, settings=settings)
    for extension in ("csv", "parquet"):
        _atomic_frame(manifest_dir / f"vitaldb_patient_splits.{extension}", splits)
    if phase:
        phase("Resolving tracks: loading cached/public VitalDB index")
    index = load_track_index(root, refresh=refresh_index)
    lookup = track_lookup(index)
    resolved, track_failures = [], {}
    for caseid in cohort.caseid:
        try:
            resolved.append(resolve_tracks(lookup, int(caseid), settings=settings))
        except ValueError as error:
            if strict_tracks:
                raise
            track_failures[str(int(caseid))] = str(error)
    # Summarize the records separately within each requested group.
    protected = [root / "notebooks" / "00_environment_check.ipynb", root / "notebooks" / "01-vitaldbaccess.ipynb"]
    protected += [manifest_dir / f"{stem}.{ext}" for stem in ("stage_d_full_cohort", "usable_cohort") for ext in ("csv", "parquet")]
    protected += [reference / f for f in ("clinical_data.csv", "clinical_parameters.csv", "track_names.csv")]
    split_summary = {
        split: {"patients": int(group.patient_id.nunique()), "surgeries": len(group)}
        for split, group in splits.groupby("split")
    }
    # Exclude or flag missing values before using these measurements.
    audit = {
        "dataset": "vitaldb", "audit_version": 2, "cohort_path": "data/manifests/usable_cohort.csv",
        "configuration_path": "configs/hemodynamics.json", "configuration_sha256": _digest(root / "configs" / "hemodynamics.json"),
        "configuration": settings, "processing_signature": signature, "patient_split_seed": int(settings["patient_split_seed"]),
        "original_cohort_rows": len(full), "final_usable_cases": len(cohort), "excluded_cases": int((~full.usable).sum()),
        "row_unit": "one surgery/recording identified by caseid, NOT one unique patient",
        "cohort_columns": list(cohort.columns), "original_split_assigned_count": int(cohort.split.notna().sum()),
        "csv_parquet_equivalent": True, "csv_parquet_notes": "CSV float rounding <= 5.7e-14; empty split CSV NaN versus Parquet empty string",
        "clinical_cases": len(clinical), "clinical_patients": int(clinical.subjectid.nunique()),
        "clinical_patients_with_repeated_surgeries": int(clinical.subjectid.value_counts().gt(1).sum()),
        "clinical_max_surgeries_per_patient": int(clinical.subjectid.value_counts().max()),
        "usable_patients": int(splits.patient_id.nunique()),
        "usable_patients_with_repeated_surgeries": int(splits.patient_id.value_counts().gt(1).sum()),
        "usable_max_surgeries_per_patient": int(splits.patient_id.value_counts().max()),
        "patient_id_field": "subjectid", "patient_id_documentation": definition.astype(object).where(pd.notna(definition), None).iloc[0].to_dict(),
        "patient_linkage_complete": True, "patient_splits": split_summary,
        "patient_split_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "split_policy": f"Configured fractions {options['patient_split_fractions']}; sorted subject IDs shuffled by NumPy default_rng seed{settings['patient_split_seed']}; all cohort surgeries retained",
        "exact_waveform_tracks": options["waveform_tracks"], "waveform_native_hz": 500, "waveform_output_hz": float(settings["target_waveform_fs"]),
        "map_alias_priority": options["map_track_priority"], "cohort_map_source_counts": pd.Series([r["MAP"] for r in resolved]).value_counts().to_dict(),
        "map_alias_note": "Original unqualified ART_MBP matches more than one device; download explicitly prefers Solar8000, otherwise EV1000",
        "quality_criteria": quality, "cohort_track_resolution_failures": track_failures,
        "quality_limitations": "Provisional notebook criteria count non-NaN values, not physiological plausibility or advanced SQI; MAP coverage is span, not continuously valid coverage",
        "original_waveform_aliases": ["ART", "ECG_II", "PLETH"], "original_map_alias": "ART_MBP",
        "original_map_interval_sec": 1, "original_missingness_source": "NaN percentage across 100Hz case-aligned arrays",
        "original_map_source": "non-NaN observations on a1s grid; gaps and density from valid grid indices; no declared physiologic range",
        "pipeline_map_source": "native numeric record timestamps on same authoritative packed vital timeline; no resampling/fill",
        "pipeline_waveform_resampling": "scipy.signal.resample_poly anti-alias FIR per contiguous finite native run; contiguous records merge; native sample placement rounds to nearest native tick (<=1ms at500Hz); global downsample phase preserved; exact observed spans mask output; no gap interpolation/fill",
        "resampling_version": RESAMPLING_VERSION, "resampling_window": options["resample_window"], "resampling_padtype": options["resample_padtype"],
        "timestamp_definition": "time_sec = source record timestamp minus packed VitalFile.dtstart; explicitly seconds_since_recording_start, NOT operation/OR start or wall clock",
        "clinical_alignment_provenance": "clinical_parameters.csv documents casestart as recording start0 and anestart/aneend/opstart/opend as offsets from casestart. Each case retains these reference offsets and packed source origin/end; no surgery-start re-zeroing. Clinical caseend is whole seconds while waveform duration retains subsecond precision.",
        "sdk_version": version("vitaldb"), "public_dataset_version": vital_dataset.DATASET_VERSION,
        "sdk_transport": "VitalFile.load_vital with exact four-track inclusion; packed stream skips unrelated track records and stops after last selected track",
        "legacy_timestamp_warning": "Installed SDK warns deprecated per-track CSV API can differ by per-case constant; authoritative packed timeline used instead",
        "pilot_selection": f"Duration quantiles {options['pilot_duration_quantiles']}, ties by caseid, outcome-blind", "pilot_case_ids": pilot_case_ids(cohort, settings),
        "reserve_bytes": int(float(settings["min_free_space_gb"]) * 1_000_000_000), "protected_sha256": {str(p.relative_to(root)): _digest(p) for p in protected},
        "pilot_passed": False, "pilot_results": [],
    }
    path = manifest_dir / "vitaldb_project_audit.json"
    # Handle an existing output or checkpoint before starting new work.
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("protected_sha256") != audit["protected_sha256"]:
            raise ValueError("Protected source files changed since the persisted audit; review before proceeding")
        if previous.get("processing_signature") == signature:
            audit["pilot_passed"] = previous.get("pilot_passed", False)
            audit["pilot_results"] = previous.get("pilot_results", [])
            for key in ("pilot_resume_evidence", "validation_results", "pilot_download_elapsed_sec", "anti_alias_validation", "created_or_updated_files", "download_speed_smoke", "last_download_run"):
                if key in previous:
                    audit[key] = previous[key]
    _atomic_json(path, audit)
    write_audit_markdown(root, audit)
    return audit


def write_audit_markdown(root: Path, audit: dict[str, Any]) -> None:
    # Store the settings and results as JSON for later inspection.
    lines = [
        "# VitalDB project audit", "", "## Existing evidence and cohort provenance",
        f"- Original notebook `notebooks/01-vitaldbaccess.ipynb` StageD resumes checkpointed inspections and saves all {audit['original_cohort_rows']:,} cases, then filters `usable` into `usable_cohort`.",
        f"- Final cohort: **{audit['final_usable_cases']:,} checked usable surgeries**, {audit['excluded_cases']} excluded; one row per `caseid`, not patient. Existing CSV/Parquet values agree within rounding, and all original splits are empty.",
        "- Environment check notebook is unexecuted. Notebook outputs reference a historic GoogleDrive path; all new operations discover the current project root.",
        "- Original cohorts/notebooks/reference files are read-only; their SHA256 hashes are recorded in the JSON audit.",
        "- Columns: " + ", ".join(f"`{x}`" for x in audit["cohort_columns"]), "",
        "## Patient identity: legitimate linkage exists",
        "- `clinical_parameters.csv` explicitly describes `subjectid` as **EMR: Subject ID; Deidentified hospital ID of patient**. It is independently distinct from randomized surgery `caseid`.",
        f"- Clinical reference: {audit['clinical_cases']:,} surgeries / {audit['clinical_patients']:,} subjects; {audit['clinical_patients_with_repeated_surgeries']} repeated subjects; maximum {audit['clinical_max_surgeries_per_patient']} surgeries per subject.",
        f"- Usable cohort: {audit['usable_patients']:,} patients; {audit['usable_patients_with_repeated_surgeries']} repeated patients; maximum {audit['usable_max_surgeries_per_patient']} surgeries per patient. Complete one-to-one surgery linkage verified.",
        f"- New non-destructive `vitaldb_patient_splits.csv/.parquet` uses central patient_split_seed={audit['patient_split_seed']} and configured patient-level fractions. All surgeries per patient share a split. Intersections are zero; independence is claimed only using documented linkage, never case IDs.",
        "- Patients/surgeries per split: " + json.dumps(audit["patient_splits"]), "",
        "## Exact tracks and eligibility",
        "- Original notebook aliases: `ART`, `ECG_II`, `PLETH` at100Hz; `ART_MBP` on1s grid.",
        "- Reference dictionary confirms `SNUADC/ART` (mmHg), `SNUADC/ECG_II` (mV), `SNUADC/PLETH` (unitless), all native500Hz.",
        "- Both `Solar8000/ART_MBP` and `EV1000/ART_MBP` are invasive MAP (mmHg). Actual public track index validated for every usable case. Explicit Solar-first policy resolves the original unqualified alias ambiguity; fallback EV1000 is recorded, not silently assumed.",
        "- MAP source counts: " + json.dumps(audit["cohort_map_source_counts"]),
        "- Original requirements: checked; all four modalities; >=30min; each waveform NaN missingness strictly<20%; MAP>=300 non-NaN grid observations; >=10observations/min; coverage span>=80%; median gap<=10s.",
        "- No notebook physiologic range exclusion, advanced SQI, or final-model eligibility was present. A coverage *span* threshold is not a promise of gap-free MAP. Cohort is provisional data-engineering eligibility.", "",
        "## Retrieval, timestamps, gaps and storage",
        f"- Installed VitalDB{audit['sdk_version']} reads version{audit['public_dataset_version']} packed `.vital` streams, the authoritative shared timeline. Deprecated CSV track endpoints can be shifted relative to recordings; they are not used for signal retrieval.",
        "- Only four exact required tracks are included; skipped unrelated blocks are not materialized. Each spawn worker holds one complete case at a time; worker count and aggregate estimated RAM are bounded by VitalDB download settings. Native records are assembled onto their shared500Hz clock (<=1ms timestamp quantization); contiguous finite runs merge across packet boundaries, while NaNs/time gaps split filtering runs.",
        "- `scipy.signal.resample_poly` applies its anti-alias low-pass FIR before500->100Hz decimation, with configured Kaiser window and reflect boundary padding. Each finite run has an explicit phase pad to anchor output to the recording-wide100Hz grid; filter delay is compensated. Padding is internal to the filter, never written into unobserved spans; a separate exact source-span mask keeps gaps NaN. Single-sample runs are discarded rather than fabricated.",
        "- `time_sec` is explicitly `seconds_since_recording_start`, computed as source_dt-dtstart, NOT seconds since operation/OR start. All three final sampling-rate metadata fields are explicit100Hz. Case metadata retains packed source origin/end plus clinical casestart/caseend/anestart/aneend/opstart/opend offsets, whose recording-relative definitions are verified in clinical_parameters.csv. Negative anesthesia-start offsets remain negative; operation start is not silently treated as recording start. Clinical end values are whole-second reference precision.",
        "- Adjacent observed native records are filtered as one run to avoid packet-boundary artifacts. MAP keeps native timestamps/values, duplicate timestamps use last record, no MAP resampling or long-gap fill.",
        "- Canonical Parquet waveform columns: `time_sec,ART,ECG,PPG`; numeric columns: `time_sec,MAP`. Shared canonical writer supplies atomic output, hashes, verified resume and case metadata.",
        f"- Central `configs/hemodynamics.json` controls random/split seeds, target sampling rate, disk reserve ({audit['reserve_bytes']:,}bytes), retry/backoff/timeouts, exact tracks, pilot selection, split fractions, anti-alias filter and VitalDB-specific provisional thresholds. All parent keys remain intact; no MOVER thresholds are silently reused for VitalDB.",
        "- Modes: PILOT (exact three outcome-blind duration representatives), RESUME (pilot by default), FULL_DOWNLOAD. Full mode requires both allow_full_vitaldb_download=true and passed pilot readback/resume. Configuration defaults to false; this task never starts full download. Processing-signature checks invalidate old unfiltered or differently seeded outputs rather than skipping them.", "",
        "## Real pilot evidence",
        f"- Deterministic pilot IDs: {audit['pilot_case_ids']}; pilot passed: **{audit['pilot_passed']}**.",
    ]
    if audit.get("pilot_results"):
        lines += ["", "```json", json.dumps(audit["pilot_results"], indent=2, allow_nan=False), "```"]
    if audit.get("pilot_resume_evidence"):
        lines += ["", "Resume evidence: " + json.dumps(audit["pilot_resume_evidence"])]
    if audit.get("validation_results"):
        lines += ["", "Validation: " + json.dumps(audit["validation_results"])]
    if audit.get("anti_alias_validation"):
        lines += ["", "Anti-alias regression evidence: " + json.dumps(audit["anti_alias_validation"])]
    if audit.get("download_speed_smoke"):
        lines += ["", "## Concurrent download diagnosis and bounded real smoke", "```json", json.dumps(audit["download_speed_smoke"], indent=2), "```", "Full ingestion was not run as part of this diagnostic. The scientific processing signature and three original pilot outputs are unchanged. The smoke used 4 spawn workers; the current default (configs/hemodynamics.json vitaldb.download_workers) and hard maximum (vitaldb_pipeline.MAX_DOWNLOAD_WORKERS) are measured separately in results/vitaldb_download_report.json. At most worker-count cases are in flight under an estimated-RAM cap; CPU/math/Arrow pools are limited. Progress heartbeats remain visible during idle network waits; Interrupt returns PAUSED and checkpoints remain resumable."]
    if audit.get("last_download_run"):
        run = audit["last_download_run"]
        lines += ["", "Latest verified run: " + json.dumps({key: run.get(key) for key in ("summary_status", "verified_cases", "cohort_cases", "remaining_cases", "download_workers", "elapsed_sec", "audit_seconds", "startup_to_dispatch_sec")})]
    if audit.get("created_or_updated_files"):
        lines += ["", "## Exact file inventory", *["- `" + path + "`" for path in audit["created_or_updated_files"]]]
    lines += ["", "## Limits and next steps", "- No training, IOH outcome selection, BSHI generation or full cohort waveform download performed. Actual bytes/timing/missingness and native numeric gaps are recorded per case. Download manifest rows distinguish complete/failed cases and original cohort quality from measured current packed-file quality.", "- Compare future research splits using the linked patient manifest, not source blank splits. Public source files can evolve; cached track index, dataset version and original hashes make this run auditable.", ""]
    destination = Path(root) / "data" / "manifests" / "vitaldb_project_audit.md"
    pending = destination.with_name(destination.name + ".part")
    try:
        pending.write_text("\n".join(lines), encoding="utf-8")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


_NET = threading.local()


class _ByteCounter:
    """Compressed HTTP body bytes the SDK actually reads; flushes live deltas to the coordinator."""

    def __init__(self, case_id: int, flush_bytes: int = 1 << 20):
        self.case_id, self.flush_bytes, self.total, self.unsent = int(case_id), flush_bytes, 0, 0

    def add(self, count: int) -> None:
        self.total += count
        self.unsent += count
        if self.unsent >= self.flush_bytes:
            self.flush()

    def flush(self) -> None:
        if self.unsent:
            _worker_event(self.case_id, "bytes", bytes=self.unsent)
            self.unsent = 0


class _CountingResponse:
    """Read-through proxy over the urlopen response consumed by the SDK's gzip reader."""

    def __init__(self, raw, counter: _ByteCounter):
        self._raw, self._counter = raw, counter

    def read(self, size=-1):
        data = self._raw.read(size)
        self._counter.add(len(data))
        return data

    def read1(self, size=-1):
        data = self._raw.read1(size)
        self._counter.add(len(data))
        return data

    def __getattr__(self, name):
        return getattr(self._raw, name)


class _UrllibShim:
    """Replaces urllib.request inside vitaldb.utils; only urlopen() gains byte counting."""

    def urlopen(self, *args, **kwargs):
        response = urllib.request.urlopen(*args, **kwargs)
        counter = getattr(_NET, "counter", None)
        return response if counter is None else _CountingResponse(response, counter)

    def __getattr__(self, name):
        return getattr(urllib.request, name)


def _install_byte_counting() -> bool:
    module = sys.modules.get(vitaldb.VitalFile.load_vital.__module__)
    if module is None or not hasattr(getattr(module, "request", None), "urlopen"):
        return False
    if not isinstance(module.request, _UrllibShim):
        module.request = _UrllibShim()
    return True


def last_network_bytes() -> int:
    """Bytes read by this thread's most recent load_vital_case (0 when counting is unavailable)."""
    return int(getattr(_NET, "last_bytes", 0))


def load_vital_case(case_id: int, tracks: dict[str, str], timeout_seconds: float | None = None):
    """SDK streaming parser; explicit success check (constructor ignores parser failure)."""
    timeout_seconds = float(load_settings()["vitaldb"]["track_timeout_seconds"]) if timeout_seconds is None else timeout_seconds
    vf = vitaldb.VitalFile()
    old_timeout = socket.getdefaulttimeout()
    counter = _ByteCounter(case_id)
    _NET.counter, _NET.last_bytes = (counter if _install_byte_counting() else None), 0
    try:
        socket.setdefaulttimeout(timeout_seconds)
        url = f"{vital_dataset.api_url}/{vital_dataset.DATASET_VERSION}/{int(case_id)}.vital"
        if not vf.load_vital(url, track_names=list(tracks.values())):
            raise RuntimeError(f"VitalDB parser failed for {case_id}")
    finally:
        socket.setdefaulttimeout(old_timeout)
        counter.flush()
        _NET.counter, _NET.last_bytes = None, counter.total
    if vf.dtend <= vf.dtstart:
        raise RuntimeError("Invalid packed-file timeline")
    for name in tracks.values():
        if name not in vf.trks or not vf.trks[name].recs:
            raise RuntimeError(f"Incomplete required track {name}")
    return vf


def _valid_runs(values: np.ndarray):
    # Exclude or flag missing values before using these measurements.
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    return zip(edges[::2], edges[1::2])


def _resample_track(track, origin: float, duration: float, interval: float, options: dict) -> np.ndarray:
    rate = float(track.srate)
    target_rate = 1.0 / interval
    if target_rate > rate:
        raise ValueError("Waveform upsampling is not permitted")
    ratio = (Fraction(str(target_rate)) / Fraction(str(rate))).limit_denominator(100_000)
    up, down = ratio.numerator, ratio.denominator
    if not math.isclose(up / down, target_rate / rate, rel_tol=1e-10):
        raise ValueError("Unsupported non-rational sampling-rate conversion")
    native = np.full(math.ceil(duration * rate), np.nan, dtype=np.float32)
    count = math.ceil(duration / interval)
    support = np.zeros(count, dtype=bool)
    # Process the selected records in order and retain their individual results.
    for record in track.recs:
        values = np.asarray(record["val"], dtype=np.float32)
        if track.fmt > 2:
            values = values * track.gain + track.offset
        # Exclude or flag missing values before using these measurements.
        values = values.copy()
        values[~np.isfinite(values) | (values > 4e9)] = np.nan
        start = float(record["dt"] - origin)
        native_start = int(np.rint(start * rate))
        lo = max(0, native_start)
        hi = min(len(native), native_start + len(values))
        if hi > lo:
            native[lo:hi] = values[lo - native_start:hi - native_start]
        # Exact observed spans, not rounded native slots, determine output gap masks.
        for run_start, run_end in _valid_runs(values):
            first = max(0, math.ceil((start + run_start / rate) / interval - 1e-8))
            last = min(count, math.ceil((start + run_end / rate) / interval - 1e-8))
            if last > first:
                support[first:last] = True
    output = np.full(count, np.nan, dtype=np.float32)
    # Process the selected records in order and retain their individual results.
    for start, end in _valid_runs(native):
        if end - start < 2:
            continue
        phase = int(start % down)
        run = native[start:end]
        # Filter phase is anchored to the global recording grid; padding is internal
        # to this finite run only. No padded values are emitted into missing spans.
        phased = np.pad(run, (phase, 0), mode=options["resample_padtype"]) if phase else run
        filtered = resample_poly(phased, up, down, window=tuple(options["resample_window"]), padtype=options["resample_padtype"])
        first_filtered = (int(start) - phase) * up // down
        first = max(0, math.ceil(int(start) * up / down))
        last = min(count, math.ceil(int(end) * up / down), first_filtered + len(filtered))
        if last > first:
            output[first:last] = filtered[first - first_filtered:last - first_filtered]
    output[~support] = np.nan
    return output


def native_to_frames(vf, tracks: dict[str, str], interval: float | None = None, settings: dict | None = None):
    """Anti-aliased finite-run downsampling; no filtering across NaNs or time gaps."""
    # Collect the per-record values into a table for comparison.
    settings = settings or load_settings()
    interval = 1.0 / float(settings["target_waveform_fs"]) if interval is None else interval
    duration = float(vf.dtend - vf.dtstart)
    count = math.ceil(duration / interval)
    wave = pd.DataFrame({"time_sec": np.arange(count, dtype=np.float64) * interval})
    source_rates = {}
    for channel in ("ART", "ECG", "PPG"):
        track = vf.trks[tracks[channel]]
        if track.type != 1 or track.srate <= 0:
            raise ValueError(f"{channel} is not a waveform")
        source_rates[channel] = float(track.srate)
        wave[channel] = _resample_track(track, float(vf.dtstart), duration, interval, settings["vitaldb"])
    numeric_track = vf.trks[tracks["MAP"]]
    if numeric_track.type != 2:
        raise ValueError("MAP is not numeric")
    # Exclude or flag missing values before using these measurements.
    numeric = pd.DataFrame({
        "time_sec": [float(rec["dt"] - vf.dtstart) for rec in numeric_track.recs],
        "MAP": [float(rec["val"]) for rec in numeric_track.recs],
    })
    numeric = numeric.loc[numeric.time_sec.ge(0) & numeric.time_sec.le(duration)]
    numeric = numeric.sort_values("time_sec", kind="stable").drop_duplicates("time_sec", keep="last").reset_index(drop=True)
    numeric.loc[~np.isfinite(numeric.MAP), "MAP"] = np.nan
    return wave, numeric, source_rates


def frame_metrics(wave: pd.DataFrame, numeric: pd.DataFrame, interval: float | None = None, quality: dict | None = None) -> dict[str, Any]:
    settings = load_settings()
    interval = 1.0 / float(settings["target_waveform_fs"]) if interval is None else interval
    quality = settings["vitaldb"]["quality_thresholds"] if quality is None else quality
    if len(wave) < 2 or len(numeric) < 2:
        raise ValueError("Insufficient physiological data")
    times = wave.time_sec.to_numpy()
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Waveform timestamps must increase")
    steps = np.diff(times)
    if not np.allclose(steps, interval, atol=1e-8):
        raise ValueError("Waveform grid differs from the configured target rate")
    duration = float(times[-1] + interval)
    valid = numeric.loc[np.isfinite(numeric.MAP), "time_sec"].to_numpy()
    if len(valid) < 2 or np.any(np.diff(numeric.time_sec) <= 0):
        raise ValueError("Native MAP has insufficient observations or invalid timestamps")
    # Exclude or flag missing values before using these measurements.
    gaps = np.diff(valid)
    metrics = {
        "waveform_rows": len(wave), "map_rows": len(numeric), "duration_sec": duration,
        "waveform_interval_sec": interval, "waveform_time_min_sec": float(times[0]), "waveform_time_max_sec": float(times[-1]),
        "map_valid_count": len(valid), "map_missing_pct": float(numeric.MAP.isna().mean() * 100),
        "map_time_min_sec": float(numeric.time_sec.iloc[0]), "map_time_max_sec": float(numeric.time_sec.iloc[-1]),
        "map_median_gap_sec": float(np.median(gaps)), "map_p95_gap_sec": float(np.percentile(gaps, 95)),
        "map_max_gap_sec": float(gaps.max()), "map_measurements_per_minute": float(len(valid) * 60 / duration),
        "map_coverage_span_pct": float((valid[-1] - valid[0]) / duration * 100),
    }
    # Exclude or flag missing values before using these measurements.
    metrics["missingness"] = {name: float(wave[name].isna().mean() * 100) for name in ("ART", "ECG", "PPG")}
    metrics["measured_quality_passed"] = bool(
        duration >= quality["duration_minutes_min"] * 60 and all(p < quality["waveform_missing_pct_strict_max"] for p in metrics["missingness"].values())
        and len(valid) >= quality["map_valid_count_min"] and metrics["map_measurements_per_minute"] >= quality["map_measurements_per_minute_min"]
        and metrics["map_coverage_span_pct"] >= quality["map_coverage_span_pct_min"] and metrics["map_median_gap_sec"] <= quality["map_median_gap_sec_max"]
    )
    return metrics


def _completed(root: Path, state: dict | None, settings: dict | None = None) -> bool:
    if not state or state.get("status") != "complete":
        return False
    settings = settings or load_settings(root)
    if state.get("processing_signature") != processing_signature(settings):
        return False
    return bool(_canonical().verify_case(root, state))


def export_output_manifest(root: Path, store) -> pd.DataFrame:
    rows = store.rows()
    if isinstance(rows, pd.DataFrame):
        output = rows.copy()
    else:
        if isinstance(rows, dict):
            rows = list(rows.values())
        output = pd.DataFrame(rows)
    for column in output.columns:
        if output[column].map(lambda x: isinstance(x, (dict, list))).any():
            output[column] = output[column].map(lambda x: json.dumps(x, sort_keys=True, default=str) if isinstance(x, (dict, list)) else x)
    for ext in ("csv", "parquet"):
        _atomic_frame(Path(root) / "data" / "manifests" / f"vitaldb_download_manifest.{ext}", output)
    return output


_WORKER_EVENTS = None


def _worker_initialize(events, arrow_cpu_threads=1, arrow_io_threads=1):
    global _WORKER_EVENTS
    import pyarrow as pa
    _WORKER_EVENTS = events
    events.cancel_join_thread()
    pa.set_cpu_count(max(1, min(2, int(arrow_cpu_threads))))
    pa.set_io_thread_count(max(1, min(2, int(arrow_io_threads))))


def _worker_event(case_id, phase, **extra):
    if _WORKER_EVENTS is not None:
        try:
            _WORKER_EVENTS.put({"case_id": int(case_id), "phase": phase, "at": time.monotonic(), **extra})
        except (OSError, ValueError):
            pass


def _create_executor(workers, events, settings):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    options = settings["vitaldb"]
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                               initializer=_worker_initialize,
                               initargs=(events, options.get("arrow_cpu_threads", 1), options.get("arrow_io_threads", 1)))


class _PhaseClock:
    """Per-attempt phase stopwatch; each entry also tells the coordinator which phase is live."""

    def __init__(self, case_id):
        self.case_id, self.phase, self.began, self.seconds = int(case_id), None, 0.0, {}

    def enter(self, phase):
        self.stop()
        self.phase, self.began = phase, time.perf_counter()
        _worker_event(self.case_id, phase)

    def stop(self):
        if self.phase:
            self.seconds[self.phase] = self.seconds.get(self.phase, 0.0) + time.perf_counter() - self.began
            self.phase = None

    def info(self, failed_phase=None):
        return {"phase_seconds": {name: round(value, 4) for name, value in self.seconds.items()},
                "network_bytes": last_network_bytes(), "failed_phase": failed_phase}


def _portable(error, info):
    """Attach attempt telemetry; unpicklable errors (e.g. HTTPError holding a socket) become RuntimeError."""
    status = getattr(error, "code", None)
    info["http_status"] = int(status) if isinstance(status, int) else None
    try:
        pickle.loads(pickle.dumps(error))
    except Exception:
        error = RuntimeError(f"{type(error).__name__}: {error}")
    try:
        error.attempt_info = info
    except AttributeError:
        pass
    return error


def _process_case(job):
    """One attempt, one case, one process; never touch coordinator checkpoints."""
    clock = _PhaseClock(job["case_id"])
    _NET.last_bytes = 0
    try:
        written = _process_case_attempt(job, clock)
    except BaseException as error:
        failed_phase = clock.phase
        clock.stop()
        portable = _portable(error, clock.info(failed_phase))
        if portable is error:
            raise
        raise portable from error
    clock.stop()
    written["attempt_info"] = clock.info()
    return written


def _process_case_attempt(job, clock):
    # Keep waveform timing, quality checks, and provenance together for this case attempt.
    from types import SimpleNamespace
    root, case_id = Path(job["root"]), job["case_id"]
    settings = job["settings"]
    options = settings["vitaldb"]
    interval = 1.0 / float(settings["target_waveform_fs"])
    signature = processing_signature(settings)
    reserve_bytes = job["reserve_bytes"]
    exact = job["tracks"]
    patient = SimpleNamespace(**job["patient"])
    vf = wave = numeric = loaded = reloaded_wave = reloaded_map = None
    try:
        # Decode the measurements while retaining their signal timing.
        _canonical().ensure_disk_space(root, reserve_bytes=reserve_bytes, additional_bytes=job["projected_bytes"])
        clock.enter("network")
        vf = load_vital_case(case_id, exact, timeout_seconds=float(options["track_timeout_seconds"]))
        clock.enter("resampling")
        # Resample waveforms while preserving the native numeric-MAP timestamps.
        wave, numeric, rates = native_to_frames(vf, exact, interval, settings)
        source_origin = float(vf.dtstart)
        source_end = float(vf.dtend)
        del vf
        vf = None
        metrics = frame_metrics(wave, numeric, interval, options["quality_thresholds"])
        if not metrics["measured_quality_passed"]:
            raise ValueError(f"Current packed-file quality fails audited provisional thresholds: {metrics}")
        # Exclude or flag missing values before using these measurements.
        metadata = {
            "case_id": str(case_id), "patient_id": int(patient.patient_id), "split": str(patient.split),
            "min_free_space_bytes": reserve_bytes,
            "patient_id_source": str(patient.patient_id_source), "track_names": exact,
            "sampling_rates": {**rates, "waveform_output": float(settings["target_waveform_fs"]), "MAP": "native_numeric"},
            "processing_signature": signature, "patient_split_seed": int(settings["patient_split_seed"]),
            "art_final_fs": float(settings["target_waveform_fs"]), "ecg_final_fs": float(settings["target_waveform_fs"]),
            "ppg_final_fs": float(settings["target_waveform_fs"]), "time_reference": "seconds_since_recording_start",
            "clinical_alignment_provenance": {
                "source": "data/reference/vitaldb/clinical_data.csv", "case_id": int(case_id),
                "reference_definition": "clinical_parameters.csv: casestart is anonymized recording start0; other casefile times are seconds from casestart, not offsets from operation start",
                "reference_offsets_sec": {name: (float(value) if pd.notna(value) else None) for name, value in job["clinical_offsets"].items()},
                "clinical_caseend_precision": "whole seconds in reference; packed waveform duration can retain subsecond precision",
            },
            "source_dataset_version": vital_dataset.DATASET_VERSION,
            "source_url": f"{vital_dataset.api_url}/{vital_dataset.DATASET_VERSION}/{case_id}.vital",
            "time_origin": "anonymized_recording_start", "source_time_origin_sec": source_origin,
            "source_time_end_sec": source_end, "map_gap_fill": "none", "waveform_gap_fill": "none",
            "resampling": RESAMPLING_VERSION, "resampling_window": options["resample_window"],
            "resampling_padtype": options["resample_padtype"], "source_cohort_manifest": "data/manifests/usable_cohort.csv",
            "original_cohort_metrics": job["original_cohort_metrics"],
            **metrics,
        }
        _canonical().ensure_disk_space(root, reserve_bytes=reserve_bytes, additional_bytes=int(wave.memory_usage(index=False).sum() + numeric.memory_usage(index=False).sum()))
        clock.enter("writing")
        written = _canonical().write_case(root, "vitaldb", case_id, wave, numeric, metadata)
        if not _canonical().verify_case(root, written):
            raise RuntimeError("Canonical case hash/schema verification failed")
        del wave, numeric
        wave = numeric = None
        # Read committed outputs back and compare them with the expected signal content.
        clock.enter("readback")
        loaded = _canonical().load_case(root, "vitaldb", case_id)
        reloaded_wave = pd.DataFrame({"time_sec": loaded["waveform_time_sec"], **{name: loaded[name] for name in ("ART", "ECG", "PPG")}})
        reloaded_map = pd.DataFrame({"time_sec": loaded["map_time_sec"], "MAP": loaded["MAP"]})
        reloaded_metrics = frame_metrics(reloaded_wave, reloaded_map, interval, options["quality_thresholds"])
        del reloaded_wave, reloaded_map
        if reloaded_metrics != metrics:
            raise RuntimeError("Parquet readback changed waveform/MAP metrics")
        del loaded
        written["output_bytes"] = int(written["processed_bytes"])
        written["readback_verified"] = True
        written["status"] = "complete"
        written["attempts"] = job["attempts"]
        written["error"] = None
        return written
    finally:
        vf = wave = numeric = loaded = reloaded_wave = reloaded_map = None
        gc.collect()


def _stop_executor(executor, force=False):
    # Python3.12 has no public terminate_workers API. Only this executor's own
    # process handles are terminated/joined; never unrelated notebook processes.
    processes = list((getattr(executor, "_processes", None) or {}).values())
    if force:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 2
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
            if process.is_alive():
                process.kill()
                process.join(timeout=0.2)
    executor.shutdown(wait=not force, cancel_futures=True)


def _clean_case_temps(root, case_id):
    directory = Path(root) / "data" / "parquet" / "vitaldb" / f"case_id={int(case_id)}"
    for pattern in ("*.tmp", "*.part"):
        for path in directory.glob(pattern):
            path.unlink(missing_ok=True)


def _recover_case(root, case_id, settings):
    path = Path(root) / "data" / "parquet" / "vitaldb" / f"case_id={int(case_id)}" / "metadata.json"
    try:
        # Load the recorded settings or metadata for this processing step.
        written = json.loads(path.read_text(encoding="utf-8"))
        written.update(status="complete", error=None, readback_verified=True)
        return written if _completed(root, written, settings) else None
    except (OSError, ValueError, KeyError):
        return None


def _hms(seconds):
    if seconds is None:
        return "unknown"
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


class _LiveProgress:
    def __init__(self, enabled, total, existing_count, existing_bytes, interval, root, cohort_total=None):
        self.enabled, self.root = enabled, Path(root)
        self.started = time.monotonic()
        self.last_tick = -float("inf")
        self.interval = max(0.05, float(interval))
        self.existing_count, self.output_bytes, self.written_bytes = existing_count, existing_bytes, 0
        self.cohort_total = cohort_total
        self.stats = {"retries": 0, "net_bytes": 0}
        self.last_durable = "none"
        self.last_durable_at = None
        self.recent = deque()
        self.net_samples = deque()
        self.bar = None
        if enabled:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=total, desc="VitalDB remaining cases", unit="case", mininterval=self.interval)

    def phase(self, text):
        if self.enabled:
            print(text, flush=True)

    def completion(self, written):
        self.output_bytes += int(written.get("processed_bytes", 0))
        self.written_bytes += int(written.get("processed_bytes", 0))
        self.last_durable = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.last_durable_at = time.monotonic()
        self.recent.append(time.monotonic())
        if self.bar:
            self.bar.update(1)

    def tick(self, processed, skipped, failed, retries, inflight, force=False):
        now = time.monotonic()
        if not self.enabled or (not force and now - self.last_tick < self.interval):
            return
        self.last_tick = now
        while self.recent and now - self.recent[0] > 60:
            self.recent.popleft()
        # Calculate elapsed processing time for the following step.
        elapsed = now - self.started
        rate = len(self.recent) * 60 / max(0.001, min(60, elapsed))
        remaining = self.bar.total - self.bar.n if self.bar else 0
        eta = remaining * 60 / rate if rate else None
        # Calculate the group counts for the following step.
        ages = [now - entry["started"] for entry in inflight.values()]
        request_ages = [now - entry.get("phase_started", entry["started"]) for entry in inflight.values() if entry.get("phase") == "network"]
        ids = sorted(entry["job"]["case_id"] for entry in inflight.values())
        counts = Counter(entry.get("phase", "starting") for entry in inflight.values())
        phase_text = "/".join(str(counts.get(name, 0)) for name in PHASES)
        retrying = len(retries) + sum(entry["job"]["run_attempts"] > 1 for entry in inflight.values())
        net_total = self.stats["net_bytes"] + sum(entry.get("net_bytes", 0) for entry in inflight.values())
        self.net_samples.append((now, net_total))
        while len(self.net_samples) > 2 and now - self.net_samples[0][0] > 10:
            self.net_samples.popleft()
        then, before = self.net_samples[0]
        net_rate = (net_total - before) / (now - then) if now > then else 0.0
        verified = self.existing_count + len(processed)
        total = self.cohort_total or verified
        ago = "" if self.last_durable_at is None else f" ({now - self.last_durable_at:.0f}s ago)"
        line = (f"verified={verified}/{total} ({verified / max(1, total) * 100:.1f}%) skipped={len(skipped)} failed={len(failed)} "
                f"retrying={retrying} retries={self.stats['retries']} in_flight={ids} phases[net/resample/write/readback]={phase_text} "
                f"oldest_age={max(ages, default=0):.1f}s oldest_request_age={max(request_ages, default=0):.1f}s "
                f"last_durable={self.last_durable}{ago} cases/min={rate:.2f} elapsed={elapsed:.1f}s "
                f"ETA={_hms(eta)} net_MB/s={net_rate / 1e6:.2f} net_GB={net_total / 1e9:.3f} "
                f"written_GB={self.written_bytes / 1e9:.3f} output_GB={self.output_bytes / 1e9:.3f} "
                f"free_disk_GB={shutil.disk_usage(self.root).free / 1e9:.2f}")
        print("\r" + line + " " * 8, end="", flush=True)

    def close(self):
        if self.enabled:
            print(flush=True)
        if self.bar:
            self.bar.close()


class _AttemptLog:
    """One JSON line per finished attempt; the file and logs folder appear on the first attempt."""

    def __init__(self, root):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = Path(root) / "logs"
        self.path, serial = directory / f"vitaldb_download_{stamp}.jsonl", 1
        while self.path.exists():
            serial += 1
            self.path = directory / f"vitaldb_download_{stamp}-{serial}.jsonl"
        self.count = 0

    def write(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=str, allow_nan=False) + "\n")
        self.count += 1


def _verify_cases(root, states, cases, settings, workers=None, report=None):
    """Hash/readback-verify checkpointed cases with threads (~35 cases/s on this machine; spawned processes measured only 1.3x faster)."""
    cases = [case for case in cases if states.get(case)]
    if not cases:
        return set()
    workers = max(1, min(len(cases), workers or min(16, os.cpu_count() or 4)))
    verified, checked, last_report = set(), 0, time.monotonic()
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(_completed, root, states[case], settings): case for case in cases}
        for future in as_completed(futures):
            checked += 1
            if future.result():
                verified.add(futures[future])
            if report and time.monotonic() - last_report >= 5:
                report(checked, len(cases))
                last_report = time.monotonic()
    except BaseException:
        pool.shutdown(wait=False, cancel_futures=True)  # an Interrupt must not wait for thousands of queued checks
        raise
    pool.shutdown()
    return verified


def _run_pipeline_impl(root: Path, mode: str = "PILOT", resume_target: str = "PILOT", max_attempts: int | None = None,
                 backoff_seconds: float | None = None, reserve_bytes: int | None = None,
                 download_workers: int | None = None, progress: bool = False, max_cases: int | None = None,
                 case_ids: list[int] | None = None) -> dict[str, Any]:
    """Spawn-safe bounded ingestion. max_cases caps unfinished dispatches, not full readiness.

    For a repeat of a bounded smoke batch, pass its selected_cases as case_ids;
    otherwise max_cases deliberately selects the next unfinished sorted batch.
    """
    # Calculate project paths for the following step.
    started = time.monotonic()
    root = Path(root).resolve()
    settings = load_settings(root)
    options = settings["vitaldb"]
    interval = 1.0 / float(settings["target_waveform_fs"])
    max_attempts = int(settings["max_retries"]) if max_attempts is None else max_attempts
    backoff_seconds = float(options["retry_backoff_seconds"]) if backoff_seconds is None else backoff_seconds
    reserve_bytes = int(float(settings["min_free_space_gb"]) * 1e9) if reserve_bytes is None else reserve_bytes
    workers = int(options.get("download_workers", 4)) if download_workers is None else int(download_workers)
    if not 1 <= workers <= MAX_DOWNLOAD_WORKERS:
        raise ValueError(f"download_workers must be between1 and the hard maximum{MAX_DOWNLOAD_WORKERS}")
    if max_attempts < 1 or backoff_seconds < 0 or (max_cases is not None and max_cases < 0):
        raise ValueError("Retry count must be positive; backoff/max_cases nonnegative")
    if mode not in MODES or resume_target not in ("PILOT", "FULL_DOWNLOAD"):
        raise ValueError("Mode must be PILOT/FULL_DOWNLOAD/RESUME and a valid resume_target")
    target = resume_target if mode == "RESUME" else mode
    if target == "FULL_DOWNLOAD" and not settings["allow_full_vitaldb_download"]:
        raise RuntimeError(f"FULL_DOWNLOAD blocked: no download was started. After reviewing the verified pilot, "
                           f"set allow_full_vitaldb_download=true (the top-level JSON key) in {root / 'configs' / 'hemodynamics.json'}, "
                           "save the file, and rerun RUN VITALDB DOWNLOAD - START HERE. PILOT remains available without this opt-in.")
    run_started_utc = datetime.now(timezone.utc)

    def phase(message):
        if progress:
            print(f"[{datetime.now():%H:%M:%S} +{time.monotonic() - started:5.1f}s] {message}", flush=True)

    # Read the saved table and retain its named columns.
    phase(f"Start: mode={mode} target={target} workers={workers} max_cases={max_cases}")
    phase("Loading cohort")
    cohort = pd.read_csv(root / "data" / "manifests" / "usable_cohort.csv")
    phase("Auditing cohort, patient linkage and protected sources")
    audit_started = time.monotonic()
    audit = audit_project(root, phase=phase, strict_tracks=False)
    audit_seconds = time.monotonic() - audit_started
    phase(f"Audit finished in {audit_seconds:.1f}s")
    store = _canonical().CheckpointStore(root, "vitaldb")
    states = {int(row["case_id"]): row for row in store.rows()}
    if target == "FULL_DOWNLOAD" and (not audit.get("pilot_passed") or not all(_completed(root, states.get(int(case)), settings) for case in audit["pilot_case_ids"])):
        raise RuntimeError("FULL_DOWNLOAD blocked until three real pilots pass validated readback and resume")
    all_cases = sorted(int(case) for case in cohort.caseid)
    scope = sorted(int(case) for case in audit["pilot_case_ids"]) if target == "PILOT" else all_cases
    if case_ids is not None:
        requested = sorted(set(int(case) for case in case_ids))
        if not set(requested).issubset(scope):
            raise ValueError("case_ids must belong to the selected existing cohort")
        scope = requested
    phase("Verifying completed checkpoints (hash/readback/signature), not redownloading them")
    verify_started = time.monotonic()
    verified = _verify_cases(root, states, all_cases, settings,
                             report=lambda done, total: phase(f"Checkpoint verification: {done}/{total} recorded cases checked"))
    for case in all_cases:
        previous = states.get(case)
        if case not in verified and previous and previous.get("status") != "complete":
            recovered = _recover_case(root, case, settings)
            if recovered:
                recovered["attempts"] = int(previous.get("attempts", 0))
                states[case] = store.update(case, **{key: value for key, value in recovered.items() if key != "case_id"})
                verified.add(case)
    phase(f"Checkpoint verification finished: {len(verified)} verified of {len(all_cases)} in {time.monotonic() - verify_started:.1f}s")
    skipped = [case for case in scope if case in verified]
    pending = [case for case in scope if case not in verified]
    links_frame = pd.read_csv(root / "data" / "manifests" / "vitaldb_patient_splits.csv")
    if target == "FULL_DOWNLOAD" and case_ids is None:
        phase(f"Ordering {len(pending)} unfinished cases: seed {settings['patient_split_seed']}, representative 70/15/15 prefixes, one surgery per patient first")
        pending = representative_order(links_frame, int(settings["patient_split_seed"]), done=verified, cases=pending)
    if max_cases is not None:
        pending = pending[:max_cases]
    selected = sorted(set(skipped) | set(pending))
    phase(f"Resolving tracks for {len(pending)} cases and preparing case-bounded worker jobs")
    # Read the saved table and retain its named columns.
    prepare_started = time.monotonic()
    index = load_track_index(root)
    lookup = track_lookup(index) if {"caseid", "tname"}.issubset(index.columns) else index
    phase(f"Track index ready in {time.monotonic() - prepare_started:.1f}s")
    links = links_frame.set_index("case_id")
    # Read the saved table and retain its named columns.
    clinical = pd.read_csv(root / "data" / "reference" / "vitaldb" / "clinical_data.csv",
                          usecols=["caseid", "casestart", "caseend", "anestart", "aneend", "opstart", "opend"]).set_index("caseid")
    cohort_by_case = cohort.set_index("caseid", drop=False)
    # Calculate the summary statistics for the following step.
    ready, retry_heap = deque(), []
    processed, failed, dispatch_order = [], set(), []
    wall_times, first_dispatch = {}, {}
    stopped_reason, paused, pool_broken = None, False, False
    attempt_log = _AttemptLog(root)
    stats = {"retries": 0, "attempts": 0, "net_bytes": 0, "http_429": 0, "phase_seconds": dict.fromkeys(PHASES, 0.0),
             "attempt_seconds": 0.0, "failures": {}, "net_bytes_counted": False}
    resolved = {}
    for case in pending:
        try:
            resolved[case] = resolve_tracks(lookup, case, settings)
        except Exception as error:
            failed.add(case)
            states[case] = store.update(case, status="failed", error=str(error))
            continue
        ready.append(case)

    def make_job(case):
        # Calculate the observed duration for the following step.
        duration = float(cohort_by_case.loc[case, "duration_seconds"])
        original = cohort_by_case.loc[case].drop(labels=["split"], errors="ignore").to_dict()
        return {"root": str(root), "case_id": case, "settings": settings, "tracks": resolved[case],
                "patient": links.loc[case].to_dict(), "clinical_offsets": clinical.loc[case].to_dict(),
                "original_cohort_metrics": original, "reserve_bytes": reserve_bytes,
                "projected_bytes": math.ceil(duration / interval * 40),
                "estimated_ram_bytes": math.ceil(duration * (8000 + 60 / interval) * 2 + 256_000_000),
                "attempts": int(states.get(case, {}).get("attempts", 0)), "run_attempts": 0}

    phase(f"Resolved tracks for {len(ready)} cases in {time.monotonic() - prepare_started:.1f}s; worker jobs are built lazily at dispatch")
    live = _LiveProgress(progress, len(pending), len(verified), sum(int(states[c].get("processed_bytes", 0)) for c in verified),
                         options.get("progress_interval_seconds", 1), root, cohort_total=len(all_cases))
    live.stats = stats
    live.started = started
    times = [states[c].get("updated_at", "none") for c in verified]
    live.last_durable = max(times, default="none")
    executor, events, inflight = None, None, {}
    peak_inflight = 0
    memory_budget = int(float(options.get("max_inflight_ram_gb", 8)) * 1e9)
    startup_seconds = time.monotonic() - started
    signature = processing_signature(settings)

    def log_attempt(entry, status, info=None, error=None):
        # Calculate the summary statistics for the following step.
        job, case = entry["job"], entry["job"]["case_id"]
        info = info or {}
        phases = info.get("phase_seconds", {})
        net_bytes = int(info["network_bytes"]) if "network_bytes" in info else int(entry.get("net_bytes", 0))
        stats["attempts"] += 1
        stats["net_bytes"] += net_bytes
        stats["net_bytes_counted"] = stats["net_bytes_counted"] or net_bytes > 0
        stats["attempt_seconds"] += time.monotonic() - entry["started"]
        for name, seconds in phases.items():
            stats["phase_seconds"][name] = stats["phase_seconds"].get(name, 0.0) + seconds
        if info.get("http_status") == 429:
            stats["http_429"] += 1
        attempt_log.write({"at_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "case_id": case,
                           "split": str(job["patient"].get("split")), "patient_id": int(job["patient"]["patient_id"]),
                           "attempt": job["run_attempts"], "status": status,
                           "seconds": round(time.monotonic() - entry["started"], 3),
                           "phase_seconds": {name: phases.get(name, 0.0) for name in PHASES},
                           "network_bytes": net_bytes, "output_bytes": int(entry.get("output_bytes", 0)),
                           "http_status": info.get("http_status"), "failed_phase": info.get("failed_phase"), "error": error})

    def commit(job, written):
        case = job["case_id"]
        written.update(status="complete", attempts=job["attempts"], error=None)
        if not _completed(root, written, settings):
            raise RuntimeError("Worker output failed coordinator hash/readback/signature verification")
        states[case] = store.update(case, **{key: value for key, value in written.items() if key != "case_id"})
        processed.append(case)
        failed.discard(case)
        wall_times[str(case)] = time.monotonic() - first_dispatch.get(case, started)
        live.completion(written)
        if len(processed) % 10 == 0:
            export_output_manifest(root, store)

    def handle(future):
        nonlocal stopped_reason, paused, pool_broken
        entry = inflight.pop(future)
        job, case = entry["job"], entry["job"]["case_id"]
        status, info, error_text = "complete", {}, None
        # Ensure the associated files, locks, or resources are released after this operation.
        try:
            written = future.result()
            info = written.pop("attempt_info", None) or {}
            entry["output_bytes"] = int(written.get("processed_bytes", 0))
            commit(job, written)
        except BrokenProcessPool as error:
            status, error_text = "pool_broken", str(error)
            pool_broken, stopped_reason = True, f"BrokenProcessPool: {error}; rerun RESUME"
            failed.add(case)
            states[case] = store.update(case, status="failed", error=stopped_reason)
        except (KeyboardInterrupt, SystemExit) as error:
            status, info = "interrupted", getattr(error, "attempt_info", {})
            paused, stopped_reason = True, "Interrupted; rerun RESUME"
            states[case] = store.update(case, status="interrupted", error=stopped_reason)
        except _canonical().DiskSpaceStop as error:
            status, error_text, info = "disk_stopped", str(error), getattr(error, "attempt_info", {})
            paused, stopped_reason = True, str(error)
            states[case] = store.update(case, status="disk_stopped", error=str(error))
        except Exception as error:
            info = getattr(error, "attempt_info", info)
            error_text = f"{type(error).__name__}: {error}"
            if isinstance(error, OSError) and getattr(error, "errno", None) == 28:
                status = "disk_stopped"
                paused, stopped_reason = True, f"Disk exhausted: {error}"
                states[case] = store.update(case, status="disk_stopped", error=stopped_reason)
            elif job["run_attempts"] < max_attempts and not paused and not pool_broken:
                status = "retry"
                stats["retries"] += 1
                delay = backoff_seconds * 2 ** (job["run_attempts"] - 1)
                heapq.heappush(retry_heap, (time.monotonic() + delay, case, job))
                states[case] = store.update(case, status="retrying", error=error_text)
            else:
                status = "failed"
                failed.add(case)
                stats["failures"][str(case)] = error_text
                states[case] = store.update(case, status="failed", error=error_text)
                if live.bar:
                    live.bar.update(1)
        finally:
            _clean_case_temps(root, case)
            log_attempt(entry, status, info, error_text)

    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        if ready:
            phase(f"Starting {workers} workers (Windows spawn; math threads1; bounded in-flight RAM {memory_budget / 1e9:.1f}GB)")
            events = mp.get_context("spawn").Queue()
            executor = _create_executor(workers, events, settings)
        # Repeat this step while the loop's completion condition is unmet.
        while ready or retry_heap or inflight:
            if paused or pool_broken:
                break
            while len(inflight) < workers:
                now = time.monotonic()
                retry = bool(retry_heap and retry_heap[0][0] <= now)
                if retry:
                    job = heapq.heappop(retry_heap)[2]
                elif ready:
                    job = ready.popleft()
                    if not isinstance(job, dict):
                        job = make_job(job)
                else:
                    break
                reserved_ram = sum(entry["job"]["estimated_ram_bytes"] for entry in inflight.values())
                if job["estimated_ram_bytes"] + reserved_ram > memory_budget:
                    if not inflight:
                        paused, stopped_reason = True, "Case exceeds max_inflight_ram_gb; increase the explicit RAM budget before RESUME"
                        states[job["case_id"]] = store.update(job["case_id"], status="interrupted", error=stopped_reason)
                    else:
                        ready.appendleft(job)
                    break
                reserved_disk = sum(entry["job"]["projected_bytes"] for entry in inflight.values())
                try:
                    _canonical().ensure_disk_space(root, reserve_bytes=reserve_bytes,
                                                  additional_bytes=reserved_disk + job["projected_bytes"])
                except (OSError, _canonical().DiskSpaceStop) as error:
                    paused, stopped_reason = True, str(error)
                    states[job["case_id"]] = store.update(job["case_id"], status="disk_stopped", error=stopped_reason)
                    break
                job["attempts"] += 1
                job["run_attempts"] += 1
                if job["case_id"] not in first_dispatch:
                    if not first_dispatch:
                        startup_seconds = now - started
                        phase(f"Dispatching first case {job['case_id']} (start-up {startup_seconds:.1f}s)")
                    first_dispatch[job["case_id"]] = now
                    dispatch_order.append(job["case_id"])
                states[job["case_id"]] = store.update(job["case_id"], status="downloading", attempts=job["attempts"],
                                                     patient_id=int(job["patient"]["patient_id"]), split=str(job["patient"]["split"]), error=None)
                try:
                    future = executor.submit(_process_case, job)
                except BrokenProcessPool as error:
                    pool_broken, stopped_reason = True, f"BrokenProcessPool: {error}; rerun RESUME"
                    failed.add(job["case_id"])
                    states[job["case_id"]] = store.update(job["case_id"], status="failed", error=stopped_reason)
                    break
                inflight[future] = {"job": job, "started": now, "phase": "starting"}
                peak_inflight = max(peak_inflight, len(inflight))
            if paused or pool_broken:
                break
            if events is not None:
                # Repeat this step while the loop's completion condition is unmet.
                while True:
                    try:
                        event = events.get_nowait()
                    except (queue.Empty, OSError, ValueError):
                        break
                    for entry in inflight.values():
                        if entry["job"]["case_id"] == event["case_id"]:
                            if event["phase"] == "bytes":
                                entry["net_bytes"] = entry.get("net_bytes", 0) + int(event["bytes"])
                            else:
                                entry["phase"] = event["phase"]
                                entry["phase_started"] = event.get("at", time.monotonic())
            live.tick(processed, skipped, failed, retry_heap, inflight)
            timeout = live.interval
            if retry_heap:
                timeout = min(timeout, max(0.01, retry_heap[0][0] - time.monotonic()))
            if inflight:
                done, _ = wait(tuple(inflight), timeout=timeout, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: inflight[item]["job"]["case_id"]):
                    handle(future)
            elif retry_heap:
                time.sleep(timeout)
            elif ready and not paused:
                continue
    except (KeyboardInterrupt, SystemExit):
        paused, stopped_reason = True, "Interrupted; rerun RESUME to continue from the first unfinished case"
    except BrokenProcessPool as error:
        pool_broken, stopped_reason = True, f"BrokenProcessPool: {error}; rerun RESUME"
    finally:
        if executor is not None:
            if paused or pool_broken:
                deadline = time.monotonic() + max(0, float(options.get("stop_grace_seconds", 5)))
                while inflight and time.monotonic() < deadline and not pool_broken:
                    try:
                        done, _ = wait(tuple(inflight), timeout=min(live.interval, max(0, deadline - time.monotonic())), return_when=FIRST_COMPLETED)
                        for future in done:
                            handle(future)
                        live.tick(processed, skipped, failed, retry_heap, inflight)
                    except (KeyboardInterrupt, SystemExit):
                        break
                _stop_executor(executor, force=bool(inflight) or pool_broken)
                # Process the selected records in order and retain their individual results.
                for entry in list(inflight.values()):
                    job = entry["job"]
                    written = _recover_case(root, job["case_id"], settings)
                    if written:
                        commit(job, written)
                        log_attempt(entry, "complete", {"phase_seconds": {}}, None)
                    else:
                        status = "failed" if pool_broken else "interrupted"
                        if pool_broken:
                            failed.add(job["case_id"])
                        states[job["case_id"]] = store.update(job["case_id"], status=status, error=stopped_reason)
                        log_attempt(entry, "pool_broken" if pool_broken else "interrupted", {}, stopped_reason)
                    _clean_case_temps(root, job["case_id"])
                inflight.clear()
                for _, case, _ in retry_heap:
                    states[case] = store.update(case, status="interrupted", error=stopped_reason)
                retry_heap.clear()
            else:
                _stop_executor(executor)
        if paused or pool_broken:
            for case in first_dispatch:
                if states.get(case, {}).get("status") in ("downloading", "retrying"):
                    states[case] = store.update(case, status="failed" if pool_broken else "interrupted", error=stopped_reason)
                    _clean_case_temps(root, case)
        if events is not None:
            events.close()
            events.cancel_join_thread()
        live.tick(processed, skipped, failed, retry_heap, inflight, force=True)
        export_output_manifest(root, store)

    phase("Confirming whole-cohort completion (pilot/capped runs are not full completion)")
    final_states = {int(row["case_id"]): row for row in store.rows()}
    # Cases verified at start or committed this run were hash/readback-verified moments ago; re-hashing them would only delay the stop/exit.
    valid = {case for case in (verified | set(processed))
             if final_states.get(case, {}).get("status") == "complete" and final_states[case].get("processing_signature") == signature}
    summary = "PAUSED" if paused else "COMPLETE" if len(valid) == len(all_cases) else "FAILED" if failed and not processed and not valid else "INCOMPLETE"
    # Calculate the output summary for the following step.
    dispatch_wall = max(time.monotonic() - started - startup_seconds, 1e-9)
    elapsed_total = time.monotonic() - started
    result = {"mode": mode, "target": target, "selected_cases": selected, "processed": sorted(processed),
              "skipped": skipped, "failed": sorted(failed), "stopped_reason": stopped_reason,
              "summary_status": summary, "cohort_cases": len(all_cases), "verified_cases": len(valid),
              "remaining_cases": len(all_cases) - len(valid), "download_workers": workers,
              "peak_inflight": peak_inflight, "dispatch_order": dispatch_order,
              "case_wall_times_sec": wall_times, "elapsed_sec": elapsed_total,
              "startup_to_dispatch_sec": startup_seconds, "audit_seconds": audit_seconds,
              "processed_bytes": sum(int(final_states[case].get("processed_bytes", 0)) for case in valid),
              "run_started_utc": run_started_utc.isoformat(timespec="seconds"), "verified_at_start": len(verified),
              "attempts": stats["attempts"], "retries": stats["retries"], "http_429": stats["http_429"],
              "failures": dict(stats["failures"]), "network_bytes": stats["net_bytes"],
              "network_bytes_counted": stats["net_bytes_counted"],
              "written_bytes_this_run": sum(int(final_states[case].get("processed_bytes", 0)) for case in processed),
              "phase_seconds_total": {name: round(value, 3) for name, value in stats["phase_seconds"].items()},
              "dispatch_wall_sec": dispatch_wall, "attempt_log": str(attempt_log.path) if attempt_log.count else None,
              "max_cases": max_cases}
    pilot_states = [final_states.get(int(case)) for case in audit["pilot_case_ids"]]
    if target == "PILOT" and all(_completed(root, state, settings) and state.get("readback_verified") and state.get("measured_quality_passed") for state in pilot_states):
        audit["pilot_results"] = [
            {key: state[key] for key in ("case_id", "patient_id", "split", "patient_split_seed", "processing_signature", "resampling", "time_reference", "art_final_fs", "ecg_final_fs", "ppg_final_fs", "clinical_alignment_provenance", "waveform_rows", "map_rows", "duration_sec", "missingness", "map_median_gap_sec", "map_max_gap_sec", "map_coverage_span_pct", "processed_bytes", "readback_verified", "track_names")}
            for state in pilot_states]
        if set(skipped) == set(selected) and not processed:
            audit["pilot_passed"] = True
            audit["pilot_resume_evidence"] = result
    audit["last_download_run"] = result
    inventory = list(audit.get("created_or_updated_files", []))
    for case in sorted(valid):
        for name in ("waveform.parquet", "map.parquet", "metadata.json"):
            relative = str(Path("data") / "parquet" / "vitaldb" / f"case_id={case}" / name)
            if relative not in inventory:
                inventory.append(relative)
    audit["created_or_updated_files"] = inventory
    _atomic_json(root / "data" / "manifests" / "vitaldb_project_audit.json", audit)
    write_audit_markdown(root, audit)
    live.close()
    return result


REPORT_RELATIVE = Path("results") / "vitaldb_download_report.json"
LOCK_RELATIVE = Path("data") / "checkpoints" / "vitaldb" / "run.lock"


def _lock_holder_alive(lock: dict[str, Any]) -> bool:
    try:
        import psutil
        process = psutil.Process(int(lock["pid"]))
        return process.is_running() and abs(process.create_time() - float(lock.get("create_time", process.create_time()))) < 2
    except Exception:
        return False


def _acquire_run_lock(root: Path, mode: str) -> Path:
    """Advisory marker so notebooks can see an active run; a second live run only triggers a warning."""
    path = Path(root).resolve() / LOCK_RELATIVE
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
        if int(previous["pid"]) != os.getpid() and _lock_holder_alive(previous):
            print(f"WARNING: another VitalDB run (pid {previous['pid']}, {previous.get('mode')}, started {previous.get('started_utc')}) appears active; "
                  "two coordinators on one checkpoint store duplicate downloads. Interrupt one of them.", flush=True)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    create_time = None
    try:
        import psutil
        create_time = psutil.Process().create_time()
    except Exception:
        pass
    _atomic_json(path, {"pid": os.getpid(), "create_time": create_time, "mode": mode,
                        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return path


def _release_run_lock(path: Path) -> None:
    try:
        if int(json.loads(path.read_text(encoding="utf-8"))["pid"]) == os.getpid():
            path.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, TypeError):
        pass


def _write_report(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Run totals/rates/failures; written after every run, including PAUSED ones."""
    done = len(result["processed"])
    wall = float(result.get("dispatch_wall_sec") or 0.0)
    per_case = wall / done if done and wall else None
    counted = bool(result.get("network_bytes_counted"))
    report = {
        "report_version": 1, "run_started_utc": result.get("run_started_utc"),
        "run_finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "mode": result["mode"], "target": result["target"], "summary_status": result["summary_status"],
        "stopped_reason": result["stopped_reason"], "download_workers": result["download_workers"], "max_cases": result.get("max_cases"),
        "cohort_cases": result["cohort_cases"], "verified_cases": result["verified_cases"], "verified_at_start": result.get("verified_at_start"),
        "remaining_cases": result["remaining_cases"], "completed_this_run": done, "skipped_verified": len(result["skipped"]),
        "failed_cases": result["failed"], "failures": result.get("failures", {}),
        "attempts": result.get("attempts", 0), "retries": result.get("retries", 0), "http_429": result.get("http_429", 0),
        "elapsed_sec": result["elapsed_sec"], "startup_to_dispatch_sec": result["startup_to_dispatch_sec"],
        "audit_seconds": result["audit_seconds"], "dispatch_wall_sec": wall or None,
        "seconds_per_case_wall": per_case, "cases_per_min": done * 60 / wall if done and wall else None,
        "eta_remaining_sec_extrapolated": result["remaining_cases"] * per_case if per_case else None,
        "eta_note": "Extrapolation: remaining cases x this run's wall seconds per case; not a measurement of the remaining cases.",
        "network_bytes": result.get("network_bytes", 0), "network_bytes_exact": counted,
        "network_MB_per_sec": result.get("network_bytes", 0) / wall / 1e6 if counted and wall else None,
        "network_bytes_note": ("Compressed HTTP body bytes read through the SDK urlopen stream (excludes TLS/HTTP headers); failed attempts included."
                               if counted else "Network bytes unavailable for this run; use written_bytes_this_run (output Parquet bytes) for throughput."),
        "written_bytes_this_run": result.get("written_bytes_this_run", 0),
        "written_MB_per_sec": result.get("written_bytes_this_run", 0) / wall / 1e6 if wall else None,
        "total_output_GB": result["processed_bytes"] / 1e9, "phase_seconds_total": result.get("phase_seconds_total", {}),
        "peak_inflight": result["peak_inflight"], "attempt_log": result.get("attempt_log"), "unified_index": result.get("unified_index"),
    }
    _atomic_json(Path(root) / REPORT_RELATIVE, report)
    return report


def _build_unified_index_after_download(root: Path, progress: bool = False) -> dict[str, Any]:
    """Best-effort ready-window index and unified manifest; never turns a finished download into a failure."""
    started = time.monotonic()
    try:
        from hemodynamics.unified import build_unified_index, index_ready_windows
    except ImportError as error:
        print(f"WARNING: download complete; unified index not built (hemodynamics.unified unavailable: {error}).", flush=True)
        return {"status": "skipped", "reason": f"ImportError: {error}"}
    try:
        index_ready_windows(root, "vitaldb", progress=progress)
        rows = len(build_unified_index(root, progress=progress))
    except Exception as error:
        print(f"WARNING: download complete; unified indexing failed ({type(error).__name__}: {error}). Rerun notebook section 5.", flush=True)
        return {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    return {"status": "built", "rows": rows, "seconds": round(time.monotonic() - started, 1)}


def run_pipeline(root: Path, mode: str = "PILOT", resume_target: str = "PILOT", max_attempts: int | None = None,
                 backoff_seconds: float | None = None, reserve_bytes: int | None = None,
                 download_workers: int | None = None, progress: bool = False, max_cases: int | None = None,
                 case_ids: list[int] | None = None) -> dict[str, Any]:
    """Run a bounded, visible, resumable download; Interrupt always returns PAUSED."""
    started = time.monotonic()
    lock = _acquire_run_lock(Path(root), mode)
    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        result = _run_pipeline_impl(root, mode, resume_target, max_attempts, backoff_seconds, reserve_bytes,
                                    download_workers, progress, max_cases, case_ids)
    except (KeyboardInterrupt, SystemExit):
        # Also cover interrupts during the pre-dispatch audit/public-index fetch.
        root = Path(root).resolve()
        settings = load_settings(root)
        store = _canonical().CheckpointStore(root, "vitaldb")
        rows = store.rows()
        for row in rows:
            if row.get("status") in ("downloading", "retrying"):
                store.update(row["case_id"], status="interrupted", error="Interrupted before coordinator completion; RESUME")
        cohort = pd.read_csv(root / "data" / "manifests" / "usable_cohort.csv")
        signature = processing_signature(settings)
        # No hashing here: an Interrupt must return at once. The next run re-verifies every recorded case by hash/readback.
        valid = [row for row in rows if row.get("status") == "complete" and row.get("processing_signature") == signature]
        export_output_manifest(root, store)
        result = {"mode": mode, "target": resume_target if mode == "RESUME" else mode,
                  "selected_cases": [], "processed": [], "skipped": [], "failed": [],
                  "stopped_reason": "Interrupted during startup/finalization; rerun RESUME", "summary_status": "PAUSED",
                  "cohort_cases": len(cohort), "verified_cases": len(valid), "remaining_cases": len(cohort) - len(valid),
                  "download_workers": download_workers or settings["vitaldb"].get("download_workers", 4),
                  "peak_inflight": 0, "dispatch_order": [], "case_wall_times_sec": {}, "elapsed_sec": time.monotonic() - started,
                  "startup_to_dispatch_sec": None, "audit_seconds": None,
                  "processed_bytes": sum(int(row.get("processed_bytes", 0)) for row in valid), "max_cases": max_cases}
    finally:
        _release_run_lock(lock)
    print(f"{result['summary_status']}: verified={result['verified_cases']}/{result['cohort_cases']} "
          f"completed_now={len(result['processed'])} skipped={len(result['skipped'])} failed={len(result['failed'])} "
          f"remaining={result['remaining_cases']}; resumable_failures={result['failed']}", flush=True)
    _write_report(Path(root).resolve(), result)
    if progress:
        print(f"Report: {Path(root).resolve() / REPORT_RELATIVE}" + (f" | attempt log: {result['attempt_log']}" if result.get("attempt_log") else ""), flush=True)
    if result["summary_status"] == "COMPLETE" and result["target"] == "FULL_DOWNLOAD":
        result["unified_index"] = _build_unified_index_after_download(Path(root).resolve(), progress)
        _write_report(Path(root).resolve(), result)
    return result


def assert_originals_unchanged(root: Path) -> None:
    audit = json.loads((Path(root) / "data" / "manifests" / "vitaldb_project_audit.json").read_text(encoding="utf-8"))
    for relative, expected in audit["protected_sha256"].items():
        if _digest(Path(root) / relative) != expected:
            raise AssertionError(f"Protected original changed: {relative}")
