"""Create a checksummed snapshot of the cohort definition and dataset metadata.

Copies only small, irreplaceable project state (manifests, config, notebooks, code, docs, results) - never the case
parquet data or raw archives - and records a content hash of the unified index so downstream stages (labels, beats)
can refuse to run against a dataset that changed after it was tagged.
"""
from __future__ import annotations
# Record the dataset definition so later processing uses a consistent cohort.

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from pathlib import Path

import pandas as pd

from .canonical import atomic_json, sha256_file, utc_now

VERSION = "v1"
TAG_FILE = Path("data") / "manifests" / "dataset_v1.json"
_SOURCES = (("data/manifests", "**/*"), ("configs", "**/*"), ("notebooks", "*.ipynb"), ("notebooks/hemodynamics", "*.py"),
            ("docs", "**/*"), ("results", "**/*"))
_SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints"}
_INDEX_COLUMNS = ["dataset", "case_id", "patient_key", "split_within_dataset", "split_pooled", "waveform_sha256", "map_sha256"]


def index_content_hash(index):
    """sha256 over the sorted identity/split/data-hash rows; independent of row order, parquet bytes and extra columns."""
    rows = sorted("\x1f".join(str(v) for v in row) for row in index[_INDEX_COLUMNS].itertuples(index=False, name=None))
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def _read_index(root):
    return pd.read_parquet(Path(root) / "data" / "manifests" / "unified_case_index.parquet")


def _versions():
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def dataset_state(root):
    """Everything that identifies the dataset right now (no writes)."""
    # Summarize the records separately within each requested group.
    root = Path(root)
    index = _read_index(root)
    counts = {f"{ds}/{split}": {"cases": int(len(g)), "patients": int(g.patient_key.nunique())}
              for (ds, split), g in index.groupby(["dataset", "split_within_dataset"])}
    # Summarize the records separately within each requested group.
    frozen = root / "data" / "manifests" / "mover_cohort_frozen.json"
    frozen_info = json.loads(frozen.read_text(encoding="utf-8")) if frozen.exists() else {}
    splits = root / "data" / "manifests" / "vitaldb_patient_splits.parquet"
    # Summarize the records separately within each requested group.
    config = root / "configs" / "hemodynamics.json"
    settings = json.loads(config.read_text(encoding="utf-8"))
    return {
        "version": VERSION, "n_cases": int(len(index)), "n_patients": int(index.patient_key.nunique()),
        "cases_by_dataset": {k: int(v) for k, v in index.groupby("dataset").size().items()}, "counts_by_split": counts,
        "index_content_sha256": index_content_hash(index),
        "mover_patient_list_sha256": frozen_info.get("patient_list_sha256"), "mover_rule_sha256": frozen_info.get("rule_sha256"),
        "mover_rung": frozen_info.get("rung"),
        "vitaldb_patient_splits_sha256": sha256_file(splits) if splits.exists() else None,
        "config_sha256_at_tag": sha256_file(config), "seed": settings.get("random_seed"), "patient_split_seed": settings.get("patient_split_seed"),
        "versions": _versions(),
    }


def _sources(root):
    root = Path(root)
    seen = set()
    for folder, pattern in _SOURCES:
        for path in sorted((root / folder).glob(pattern)):
            if path.is_file() and not (_SKIP_PARTS & set(path.parts)) and path.suffix not in (".tmp", ".part") and path not in seen:
                seen.add(path)
                yield path


def _new_dir(parent, stamp):
    target = parent / f"dataset_v1_{stamp}"
    n = 1
    while target.exists():
        n += 1
        target = parent / f"dataset_v1_{stamp}_{n}"
    return target


def _copy_tree(root, files, dest):
    entries = []
    for path in files:
        # Create the destination directory before writing any outputs.
        rel = path.relative_to(root)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        entries.append({"path": rel.as_posix(), "size": target.stat().st_size, "sha256": sha256_file(target)})
    return entries


def verify_snapshot(snapshot_dir):
    """Re-hash a snapshot against its SNAPSHOT_MANIFEST.json; returns a list of problems (empty = intact)."""
    # Load the recorded settings or metadata for this processing step.
    snapshot_dir = Path(snapshot_dir)
    manifest = json.loads((snapshot_dir / "SNAPSHOT_MANIFEST.json").read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = snapshot_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            problems.append(f"changed: {entry['path']}")
    return problems


def _copy_extra(snapshot_dir, extra_dest):
    copy = Path(extra_dest) / Path(snapshot_dir).name
    shutil.copytree(snapshot_dir, copy, dirs_exist_ok=True)
    problems = verify_snapshot(copy)
    if problems:
        shutil.rmtree(copy, ignore_errors=True)
        raise RuntimeError(f"Extra copy verification failed: {problems[:3]}")
    return str(copy)


def snapshot_dataset_v1(root, extra_dest=None, force=False):
    """Create the backup + tag file; idempotent while the dataset content hash is unchanged (unless ``force``)."""
    root = Path(root)
    state = dataset_state(root)
    tag_path = root / TAG_FILE
    if tag_path.exists() and not force:
        tag = json.loads(tag_path.read_text(encoding="utf-8"))
        existing = root / tag.get("snapshot_dir", "")
        if tag.get("index_content_sha256") == state["index_content_sha256"] and (existing / "SNAPSHOT_MANIFEST.json").exists():
            if not verify_snapshot(existing):
                if extra_dest and not (Path(extra_dest) / existing.name / "SNAPSHOT_MANIFEST.json").exists():
                    tag["extra_dest"] = _copy_extra(existing, extra_dest)
                    atomic_json(tag_path, tag)
                return {**tag, "reused": True}
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    dest = _new_dir(root / "backups", stamp)
    try:
        # Create the destination directory before writing any outputs.
        dest.mkdir(parents=True)
        entries = _copy_tree(root, _sources(root), dest)
        atomic_json(dest / "SNAPSHOT_MANIFEST.json", {"created_at": utc_now(), "version": VERSION, "source_root": str(root), "files": entries})
        problems = verify_snapshot(dest)
        if problems:
            raise RuntimeError(f"Snapshot verification failed: {problems[:3]}")
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    tag = {**state, "created_at": utc_now(), "snapshot_dir": dest.relative_to(root).as_posix(), "snapshot_files": len(entries),
           "snapshot_bytes": sum(e["size"] for e in entries), "extra_dest": None, "reused": False}
    if extra_dest:
        tag["extra_dest"] = _copy_extra(dest, extra_dest)
    atomic_json(tag_path, tag)
    shutil.copy2(tag_path, dest / TAG_FILE.name)     # the tag travels with the snapshot (not part of the hashed manifest)
    return tag


def assert_dataset_v1(root, allow_changed=False):
    """Raise unless the unified index content still matches the v1 tag (guards labels/beats against a moved dataset)."""
    root = Path(root)
    tag_path = root / TAG_FILE
    if not tag_path.exists():
        raise FileNotFoundError("dataset_v1.json is missing: run snapshot_dataset_v1(ROOT) first (labels_and_baseline.ipynb step 2).")
    tag = json.loads(tag_path.read_text(encoding="utf-8"))
    now = index_content_hash(_read_index(root))
    if now != tag["index_content_sha256"] and not allow_changed:
        raise RuntimeError("The unified index no longer matches the dataset v1 tag (cases, splits or data hashes changed). "
                           "Restore the dataset, or re-tag deliberately with snapshot_dataset_v1(ROOT, force=True) and rerun downstream stages.")
    return tag
