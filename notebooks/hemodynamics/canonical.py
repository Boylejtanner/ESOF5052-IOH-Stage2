"""Case-level physiological Parquet storage; no windows, imputation or training."""
from __future__ import annotations
# Validate case files and preserve signal timing, patient identity, and checksums.

from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WAVE_COLUMNS = ["time_sec", "ART", "ECG", "PPG"]
MAP_COLUMNS = ["time_sec", "MAP"]
DEFAULT_RESERVE_BYTES = 100_000_000_000


class DiskSpaceStop(RuntimeError):
    """Signal that a write would violate the configured free-space reserve."""
    pass


def utc_now():
    """Return a timezone-aware UTC timestamp for provenance records."""
    return datetime.now(timezone.utc).isoformat()


def ensure_disk_space(root, reserve_bytes=DEFAULT_RESERVE_BYTES, additional_bytes=0):
    """Check that the projected write preserves the disk reserve; return free bytes."""
    free = shutil.disk_usage(Path(root)).free
    if free - max(0, additional_bytes) < reserve_bytes:
        raise DiskSpaceStop(f"Disk guard: {free / 1e9:.2f} GB free; reserve {reserve_bytes / 1e9:.2f} GB, projected write {additional_bytes / 1e9:.2f} GB. Checkpoint retained; raw sources untouched.")
    return free


def safe_id(value):
    """Validate a dataset or case identifier before using it in a filesystem path."""
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise ValueError("Unsafe dataset/case identifier")
    return value


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def atomic_json(path, data):
    """Write strict JSON through an adjacent file and replace the destination atomically."""
    # Create the destination directory before writing any outputs.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8") as f:
            # Store the settings and results as JSON for later inspection.
            json.dump(_json_value(data), f, indent=2, default=str, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        # Replace the destination only after the new file has been written successfully.
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def sha256_file(path):
    """Stream a file into a SHA-256 digest without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(root, name, frame):
    """Write matching CSV and Parquet manifests with serialized nested metadata."""
    # Create the destination directory before writing any outputs.
    directory = Path(root) / "data" / "manifests"
    directory.mkdir(parents=True, exist_ok=True)
    frame = frame.copy()
    for column in frame.select_dtypes(include="object"):
        if frame[column].map(lambda value: isinstance(value, (dict, list, tuple))).any():
            frame[column] = frame[column].map(lambda value: json.dumps(value, default=str, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value)
    # Process the selected records in order and retain their individual results.
    for extension in ("parquet", "csv"):
        path = directory / f"{safe_id(name)}.{extension}"
        temp = path.with_name(path.name + ".tmp")
        try:
            if extension == "csv":
                frame.to_csv(temp, index=False)
            else:
                with temp.open("wb") as handle:
                    frame.to_parquet(handle, index=False, compression="zstd")
                if pq.ParquetFile(temp).metadata.num_rows != len(frame):
                    raise ValueError("Manifest row count mismatch")
            # Replace the destination only after the new file has been written successfully.
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def _validate_frame(frame, columns):
    if list(frame.columns) != columns or frame.empty:
        raise ValueError(f"Expected nonempty {columns}")
    values = frame["time_sec"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(np.diff(values) <= 0):
        raise ValueError("Case times must be finite, nonnegative and strictly increasing")
    for column in columns[1:]:
        if not np.isfinite(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"No finite {column} samples; missing channels cannot be fabricated")


def write_case(root, dataset, case_id, waveform, map_numeric, metadata):
    """Validate and commit waveform, numeric MAP, and provenance files for one case."""
    # Check signal schemas and the disk reserve before committing any case output.
    root = Path(root)
    dataset, case_id = safe_id(dataset), safe_id(case_id)
    _validate_frame(waveform, WAVE_COLUMNS)
    # Create the destination directory before writing any outputs.
    _validate_frame(map_numeric, MAP_COLUMNS)
    estimate = int(waveform.memory_usage(index=False, deep=True).sum() + map_numeric.memory_usage(index=False, deep=True).sum()) * 2
    ensure_disk_space(root, int(metadata.get("min_free_space_bytes", DEFAULT_RESERVE_BYTES)), estimate)
    directory = root / "data" / "parquet" / dataset / f"case_id={case_id}"
    # Create the destination directory before writing any outputs.
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    hashes = {}
    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        for label, frame in (("waveform", waveform), ("map", map_numeric)):
            frame = frame.astype({c: ("float64" if c == "time_sec" else "float32") for c in frame.columns})
            path = directory / f"{label}.parquet"
            temp = path.with_name(path.name + ".tmp")
            paths.append(temp)
            with temp.open("wb") as handle:
                frame.to_parquet(handle, index=False, compression="zstd", row_group_size=65536)
            # Read the new Parquet file back before accepting its row count and hash.
            readback = pq.ParquetFile(temp).read().to_pandas()
            _validate_frame(readback, list(frame.columns))
            if len(readback) != len(frame):
                raise ValueError("Parquet readback row count mismatch")
            # Record the file hash and check the total case-size budget before promotion.
            hashes[label] = sha256_file(temp)
            budget = metadata.get("max_case_output_bytes")
            previous_bytes = (directory / "waveform.parquet").stat().st_size if label == "map" else 0
            if budget is not None and previous_bytes + temp.stat().st_size > int(budget):
                raise DiskSpaceStop("Compressed case would exceed configured total output-size ceiling; checkpoint retained")
            # Replace the destination only after the new file has been written successfully.
            os.replace(temp, path)
        result = dict(metadata)
        result.update(dataset=dataset, case_id=case_id, storage_version=1,
                      waveform_output_path=str((directory / "waveform.parquet").relative_to(root)),
                      map_output_path=str((directory / "map.parquet").relative_to(root)),
                      waveform_rows=len(waveform), map_rows=len(map_numeric),
                      waveform_sha256=hashes["waveform"], map_sha256=hashes["map"],
                      time_reference=metadata.get("time_reference", "seconds_since_case_start"), written_at=utc_now(),
                      processed_bytes=sum((directory / f"{label}.parquet").stat().st_size for label in ("waveform", "map")))
        atomic_json(directory / "metadata.json", result)
        if not verify_case(root, result):
            raise ValueError("Committed case failed verification")
        return result
    finally:
        for temp in paths:
            temp.unlink(missing_ok=True)


def _inside_root(root, relative):
    # Reject a manifest path that would read or write outside the project directory.
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Manifest output escapes project root")
    return path


def verify_case(root, metadata):
    """Check committed metadata, file hashes, and expected signal schemas."""
    # Accept a stored case only when metadata, hashes, schemas, and time ordering agree.
    try:
        directory = Path(root) / "data" / "parquet" / safe_id(metadata["dataset"]) / f"case_id={safe_id(metadata['case_id'])}"
        committed = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        for key in ("dataset", "case_id", "waveform_output_path", "map_output_path", "waveform_sha256", "map_sha256"):
            if committed.get(key) != metadata.get(key):
                return False
        # Process the selected records in order and retain their individual results.
        for label, columns in (("waveform", WAVE_COLUMNS), ("map", MAP_COLUMNS)):
            path = _inside_root(root, metadata[f"{label}_output_path"])
            if not path.is_file() or sha256_file(path) != metadata[f"{label}_sha256"]:
                return False
            parquet = pq.ParquetFile(path)
            if parquet.schema_arrow.names != columns or parquet.metadata.num_rows != metadata[f"{label}_rows"]:
                return False
            previous = -1.0
            finite = {c: False for c in columns[1:]}
            for batch in parquet.iter_batches(batch_size=65536):
                frame = batch.to_pandas()
                times = frame.time_sec.to_numpy()
                if not np.isfinite(times).all() or times[0] <= previous or np.any(np.diff(times) <= 0):
                    return False
                previous = times[-1]
                for c in finite:
                    finite[c] |= bool(np.isfinite(frame[c].to_numpy()).any())
            if not all(finite.values()):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def load_case(root, dataset, case_id):
    """Load a verified case, preserving separate waveform and numeric-MAP time axes."""
    directory = Path(root) / "data" / "parquet" / safe_id(dataset) / f"case_id={safe_id(case_id)}"
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if not verify_case(root, metadata):
        raise ValueError("Case is incomplete/corrupt; resume ingestion before loading")
    # Read the stored table required for this calculation.
    waveform = pq.ParquetFile(_inside_root(root, metadata["waveform_output_path"])).read().to_pandas()
    numeric = pq.ParquetFile(_inside_root(root, metadata["map_output_path"])).read().to_pandas()
    return {"dataset": dataset, "case_id": str(case_id), "patient_id": metadata.get("patient_id"),
            "split": metadata.get("split", "unassigned"), "metadata": metadata,
            "sampling_rate_metadata": {k: v for k, v in metadata.items() if "fs" in k or "interval" in k},
            "waveform_time_sec": waveform.time_sec.to_numpy(), "map_time_sec": numeric.time_sec.to_numpy(),
            **{c: waveform[c].to_numpy() for c in ("ART", "ECG", "PPG")}, "MAP": numeric.MAP.to_numpy()}


class CheckpointStore:
    """Persist case-level progress in SQLite so interrupted ingestion can resume."""
    def __init__(self, root, dataset):
        # Create the destination directory before writing any outputs.
        self.directory = Path(root) / "data" / "checkpoints" / safe_id(dataset)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "state.sqlite"
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL)")

    @contextmanager
    def _connect(self):
        # Use a transactional SQLite connection and close it even when an operation fails.
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def get(self, case_id):
        with self._connect() as db:
            row = db.execute("SELECT payload FROM cases WHERE case_id=?", (str(case_id),)).fetchone()
        return json.loads(row[0]) if row else None

    def update(self, case_id, **fields):
        with self._connect() as db:
            # Lock the checkpoint update so competing writers cannot overwrite each other.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT payload FROM cases WHERE case_id=?", (str(case_id),)).fetchone()
            record = json.loads(row[0]) if row else {"case_id": str(case_id), "status": "pending"}
            record.update(fields, updated_at=utc_now())
            db.execute("INSERT OR REPLACE INTO cases VALUES (?, ?)", (str(case_id), json.dumps(record, default=str, allow_nan=False)))
        return record

    def rows(self):
        with self._connect() as db:
            return [json.loads(row[0]) for row in db.execute("SELECT payload FROM cases ORDER BY case_id")]


def patient_splits(patient_ids, seed=5052):
    """Assign unique patients deterministically to approximately 70/15/15 partitions."""
    patients = sorted(set(str(p) for p in patient_ids))
    if any(p in ("", "nan", "None", "<NA>") for p in patients):
        raise ValueError("True patient identifiers are required for splitting")
    if len(patients) < 3:
        raise ValueError("At least three patients required for nonempty train/validation/test")
    # Shuffle unique patients once, keeping every surgery from one patient in the same split.
    patients = np.random.default_rng(seed).permutation(patients)
    n = len(patients)
    validation = max(1, int(round(n * .15)))
    test = max(1, int(round(n * .15)))
    train = n - validation - test
    frame = pd.DataFrame({"patient_id": patients, "split": ["train"] * train + ["validation"] * validation + ["test"] * test})
    assert_patient_independence(frame)
    return frame.sort_values("patient_id").reset_index(drop=True)


def assert_patient_independence(frame):
    """Reject missing identities, invalid split names, or patients shared across splits."""
    if frame.patient_id.isna().any() or not frame.split.isin(["train", "validation", "test"]).all():
        raise ValueError("Missing patient identity or invalid split")
    groups = {s: set(frame.loc[frame.split.eq(s), "patient_id"].astype(str)) for s in ("train", "validation", "test")}
    intersections = {f"{a}_{b}": len(groups[a] & groups[b]) for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    if any(intersections.values()):
        raise AssertionError(f"Patient leakage: {intersections}")
    return intersections


def environment_record():
    """Record installed library versions for data-processing provenance."""
    versions = {}
    for package in ("vitaldb", "pandas", "numpy", "pyarrow", "scipy", "wfdb", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "versions": versions, "recorded_at": utc_now()}
