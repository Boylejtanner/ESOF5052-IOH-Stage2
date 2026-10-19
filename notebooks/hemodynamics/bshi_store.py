"""Compact window store: patient-disjoint shards of memory-mapped ``.npy`` arrays (the training format; the laptop gets a share of it, never the beat store).

Per window (EAP representation; every other recipe is derived by masks, see ``bshi_data.Preprocessor``):
  waves_f16 (N,3,60,200) ECG/ART/PPG planes | scal_f32 (N,60,17) the per-beat scalar matrix (planes 3-6 are exact broadcasts of it, decoded on the GPU) | masks_u8 (N,3,60) |
  ctx_f32 (N,25) | t_rel_f32 and rr_f32 (N,60) beat start relative to the window end and R-R (s), needed for MOVER-style gap augmentation | labels (y_event, delta_map,
  delta_map_robust, aut65, time_to_event_sec, gray_zone) | reg_ok_u8 (not aut65_low_coverage) | dataset_code | window_id | sampling_weight, plus ``meta.parquet`` (dataset, case_id,
  patient_key, split, t_end_sec, ...). Layout: ``<store>/<split>/shard_NNN/*.npy`` and ``<store>/MANIFEST.sha256`` (sha256 of every file).
A shard holds whole patients only, so shards are patient-disjoint and a machine can take any set of them.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bshi as S

ARRAYS = {"waves": ("waves_f16", np.float16, (3, S.N_BEATS, S.PHASE)), "scal": ("scal_f32", np.float32, (S.N_BEATS, len(S.SCALAR_ORDER))), "masks": ("masks_u8", np.uint8, (3, S.N_BEATS)),
          "ctx": ("ctx_f32", np.float32, (len(S.CONTEXT_NAMES),)), "t_rel": ("t_rel_f32", np.float32, (S.N_BEATS,)), "rr": ("rr_f32", np.float32, (S.N_BEATS,))}
SCALARS = {"y_event": np.float32, "delta_map": np.float32, "delta_map_robust": np.float32, "aut65": np.float32, "time_to_event_sec": np.float32, "gray_zone": np.float32,
           "reg_ok": np.uint8, "dataset_code": np.int8, "window_id": np.int64, "sampling_weight": np.float32}
META_COLUMNS = ["window_id", "dataset", "case_id", "patient_key", "split", "t_end_sec", "sampling_weight"]
SAMPLE_COLUMNS = list(S.TORCH_INDEX_COLUMNS) + ["patient_key", "t_end_sec", "time_to_event_sec", "aut65_low_coverage", "sampling_weight", "split"]


def plan_shards(rows, shard_windows=20000):
    """Split ``rows`` into shards of about ``shard_windows`` windows, never splitting a patient."""
    rows = rows.sort_values(["patient_key", "case_id", "beat_lo"]).reset_index(drop=True)
    shards, start, current = [], 0, 0
    sizes = rows.groupby("patient_key", sort=False).size().to_numpy()
    for k, s in enumerate(sizes):
        current += int(s)
        if current >= shard_windows:
            shards.append((start, start + current))
            start += current
            current = 0
    if current:
        shards.append((start, start + current))
    return [rows.iloc[a:b].reset_index(drop=True) for a, b in shards]


def _scal_matrix(x):
    return np.stack([x[:, 3, :, 0]] + [x[:, p, :, s[0]] for name, (p, s) in S.SCALAR_LAYERS.items() if name != "map_beat"], axis=2).astype(np.float32)


def write_shard(root, rows, shard_dir, chunk=512):
    """Assemble ``rows`` (any order) and write one shard. Returns {windows, bytes, seconds}."""
    t0 = time.perf_counter()
    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    mm = {k: np.lib.format.open_memmap(shard_dir / f"{fname}.npy", mode="w+", dtype=dt, shape=(n,) + shp) for k, (fname, dt, shp) in ARRAYS.items()}
    sc = {k: np.zeros(n, dtype=dt) for k, dt in SCALARS.items()}
    cache = S.CaseCache()
    for i in range(0, n, chunk):
        part = rows.iloc[i:i + chunk].reset_index(drop=True)
        b = S.assemble(root, part, "eap", cache=cache, return_times=True)
        j = slice(i, i + len(part))
        mm["waves"][j] = b["x"][:, :3].astype(np.float16)
        mm["scal"][j] = _scal_matrix(b["x"])
        mm["masks"][j] = b["beat_mask"].astype(np.uint8)
        mm["ctx"][j] = b["context"]
        mm["t_rel"][j] = b["t_rel"]
        mm["rr"][j] = b["rr"]
        for c in ("y_event", "delta_map", "delta_map_robust", "aut65", "gray_zone"):
            sc[c][j] = b[c]
        sc["dataset_code"][j] = b["dataset_code"]
        sc["window_id"][j] = b["window_id"]
    sc["time_to_event_sec"][:] = rows.time_to_event_sec.to_numpy(dtype=np.float32) if "time_to_event_sec" in rows else np.nan
    sc["reg_ok"][:] = (~rows.aut65_low_coverage.to_numpy()).astype(np.uint8) if "aut65_low_coverage" in rows else 1
    sc["sampling_weight"][:] = rows.sampling_weight.to_numpy(dtype=np.float32) if "sampling_weight" in rows else 1.0
    for m in mm.values():
        m.flush()
    del mm
    for k, v in sc.items():
        np.save(shard_dir / f"{k}.npy", v)
    rows[[c for c in META_COLUMNS if c in rows]].to_parquet(shard_dir / "meta.parquet", index=False)
    return {"windows": int(n), "bytes": int(sum(p.stat().st_size for p in shard_dir.iterdir())), "seconds": round(time.perf_counter() - t0, 1)}


def _shard_task(args):
    root, rows, shard_dir = args
    return write_shard(root, rows, shard_dir)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(store):
    store = Path(store)
    files = sorted(p for p in store.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    with cf.ThreadPoolExecutor(8) as ex:
        digests = list(ex.map(sha256_file, files))
    (store / "MANIFEST.sha256").write_text("".join(f"{d}  {p.relative_to(store).as_posix()}\n" for d, p in zip(digests, files)), encoding="utf-8")
    return len(files)


def verify_manifest(store, workers=8):
    """Recompute every hash; returns (ok, list of bad files)."""
    store = Path(store)
    lines = [l.split("  ", 1) for l in (store / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines() if l.strip()]
    with cf.ThreadPoolExecutor(workers) as ex:
        got = list(ex.map(lambda l: sha256_file(store / l[1]) if (store / l[1]).exists() else None, lines))
    bad = [l[1] for l, g in zip(lines, got) if g != l[0]]
    return (not bad), bad


def build_store(root, frames, out_dir, shard_windows=20000, workers=12, max_gb=None, progress=print):
    """``frames``: {split name: rows} (rows need dataset, case_id, beat_lo/hi, patient_key, label columns, t_end_sec; optional sampling_weight, aut65_low_coverage, time_to_event_sec).
    Shards of whole patients are written in parallel. Returns the summary dict (also saved as ``summary.json``)."""
    out = Path(out_dir)
    jobs, plan = [], {}
    for split, rows in frames.items():
        shards = plan_shards(rows.assign(split=split), shard_windows)
        plan[split] = len(shards)
        for k, sh in enumerate(shards):
            jobs.append((str(root), sh, str(out / split / f"shard_{k:03d}")))
    t0 = time.perf_counter()
    res = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for r in ex.map(_shard_task, jobs):
            res.append(r)
            if progress:
                progress(f"  shard {len(res)}/{len(jobs)}: {r['windows']} windows, {r['bytes'] / 1e9:.2f} GB, {r['seconds']} s")
            if max_gb and sum(x["bytes"] for x in res) / 1e9 > max_gb:
                raise RuntimeError(f"store exceeds {max_gb} GB")
    n_files = write_manifest(out)
    summary = {"windows": int(sum(r["windows"] for r in res)), "gb": round(sum(r["bytes"] for r in res) / 1e9, 3), "shards": plan, "seconds": round(time.perf_counter() - t0, 1), "files": n_files,
               "windows_per_s": round(sum(r["windows"] for r in res) / (time.perf_counter() - t0), 1), "code_version": S.BSHI_CODE_VERSION, "format": __doc__.split("\n\n")[0]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


class Store:
    """Reader of one split of a store: memory-mapped shards, ``meta`` frame, batch iterator (CPU tensors; the GPU decodes planes 3-6)."""

    def __init__(self, path, split, shards=None):
        import torch  # noqa: F401  (imported lazily: the module is also used for building without torch)
        self.path, self.split = Path(path), split
        dirs = sorted((self.path / split).glob("shard_*"))
        self.dirs = [d for i, d in enumerate(dirs) if shards is None or i in shards]
        self.mm = [{k: np.load(d / f"{fname}.npy", mmap_mode="r") for k, (fname, _, _) in ARRAYS.items()} for d in self.dirs]
        self.sc = [{k: np.load(d / f"{k}.npy") for k in SCALARS} for d in self.dirs]
        self.meta = pd.concat([pd.read_parquet(d / "meta.parquet") for d in self.dirs], ignore_index=True)
        self.sizes = [len(s["window_id"]) for s in self.sc]
        self.offsets = np.concatenate([[0], np.cumsum(self.sizes)])

    def __len__(self):
        return int(self.offsets[-1])

    def gather(self, idx):
        """Arrays for global indices ``idx`` (any order, returned in that order)."""
        idx = np.asarray(idx)
        sh = np.searchsorted(self.offsets, idx, side="right") - 1
        out = None
        for s in np.unique(sh):
            sel = np.flatnonzero(sh == s)
            loc = idx[sel] - self.offsets[s]
            order = np.argsort(loc)
            part = {k: np.asarray(v[loc[order]]) for k, v in self.mm[s].items()}
            part.update({k: v[loc[order]] for k, v in self.sc[s].items()})
            if out is None:
                out = {k: np.empty((len(idx),) + v.shape[1:], dtype=v.dtype) for k, v in part.items()}
            inv = np.empty_like(order)
            inv[order] = np.arange(len(order))
            for k, v in part.items():
                out[k][sel] = v[inv]
        return out

    def batches(self, batch_size=256, shuffle=True, seed=0, indices=None, limit=None):
        import torch
        order = np.arange(len(self)) if indices is None else np.asarray(indices)
        if shuffle:
            order = order[np.random.default_rng(seed).permutation(len(order))]
        if limit:
            order = order[:limit]
        for i in range(0, len(order), batch_size):
            a = self.gather(order[i:i + batch_size])
            yield {"waves": torch.from_numpy(a["waves"]), "scal": torch.from_numpy(a["scal"]), "beat_mask": torch.from_numpy(a["masks"].astype(bool)), "context": torch.from_numpy(a["ctx"]),
                   "t_rel": torch.from_numpy(a["t_rel"]), "rr": torch.from_numpy(a["rr"]), "window_id": torch.from_numpy(a["window_id"]), "dataset_code": torch.from_numpy(a["dataset_code"]),
                   "reg_mask": torch.from_numpy(a["reg_ok"].astype(bool)), "sampling_weight": torch.from_numpy(a["sampling_weight"]),
                   **{c: torch.from_numpy(a[c]) for c in ("y_event", "delta_map", "delta_map_robust", "aut65", "gray_zone", "time_to_event_sec")}}


# selection of the sweep subset
def sweep_selection(idx, n_train=60000, n_val=20000, mover_share=0.10, seed=5052):
    """Deterministic stratified sample of the FULL set: ``mover_share`` of the windows from MOVER, the rest from VitalDB, uniformly at random inside each dataset (so events keep their natural
    share inside a dataset), train windows from train patients, validation windows from validation patients (patient-disjoint by construction). ``sampling_weight`` of a window =
    (windows of its stratum split x dataset x event in the full set) / (windows of that stratum in the subset): weighted metrics recover natural counts."""
    rng = np.random.default_rng(seed)
    full = idx[idx.primary_eligible & idx.in_full]
    out = []
    for split, n in (("train", n_train), ("validation", n_val)):
        pool = full[full.split_pooled == split]
        parts = []
        for ds, share in (("mover", mover_share), ("vitaldb", 1 - mover_share)):
            p = pool[pool.dataset == ds]
            k = min(int(round(n * share)), len(p))
            parts.append(p.iloc[np.sort(rng.choice(len(p), k, replace=False))])
        sel = pd.concat(parts).assign(split=split)
        sel["event"] = sel.y_event.astype(bool)
        full_n = pool.assign(event=pool.y_event.astype(bool)).groupby(["dataset", "event"]).size()
        sub_n = sel.groupby(["dataset", "event"]).size()
        w = (full_n / sub_n).rename("sampling_weight")
        sel = sel.join(w, on=["dataset", "event"]).drop(columns="event")
        out.append(sel)
    return pd.concat(out, ignore_index=True)


# the store of one experiment
STORE_DIRS = {"pooled": "store_pooled_v1", "vitaldb": "store_vitaldb_v1", "mover": "store_mover_v1"}
MAIN_TEST = {"pooled": "pooled_test", "vitaldb": "vitaldb_test", "mover": "mover_test"}


def store_path(root, train_on="pooled"):
    return Path(root) / "data" / "derived" / "bshi" / STORE_DIRS[train_on]


def experiment_frames(idx, train_on="pooled", stride=2, with_test=True):
    """Frames for ``build_store`` of one experiment. The pooled split and the within-dataset split are DIFFERENT patient partitions, so every ``train_on`` has its own store (a window that is
    training data in one can be test data in the other). Training windows: every ``stride``-th window of each case in time order (neighbours 30 s apart are near-duplicates), weight ``stride``
    so weighted sums recover natural counts; validation and test: ALL windows, weight 1."""
    from . import bshi_data as D
    fr = D.frames_for(idx, train_on)
    tr = fr["train"].sort_values(["patient_key", "case_id", "t_end_sec"])
    tr = tr[tr.groupby("case_id").cumcount().to_numpy() % int(stride) == 0].assign(sampling_weight=np.float32(stride))
    out = {"train": tr, "validation": fr["val"].assign(sampling_weight=np.float32(1.0))}
    if with_test:
        out["test"] = fr["tests"][MAIN_TEST[train_on]].assign(sampling_weight=np.float32(1.0))
    return out


def build_experiment_store(root, train_on="pooled", stride=2, workers=12, max_gb=45.0, with_test=True, progress=print):
    """Build ``data/derived/bshi/<store dir of train_on>`` from the real windows (resumable: an existing, hash-verified store is returned as it is)."""
    from . import bshi_data as D
    out = store_path(root, train_on)
    if (out / "MANIFEST.sha256").exists():
        return json.loads((out / "summary.json").read_text(encoding="utf-8"))
    idx = D.protocol_frame(root, "eap_full")
    frames = experiment_frames(idx, train_on, stride, with_test)
    if progress:
        progress("windows to store: " + ", ".join(f"{k} {len(v):,}" for k, v in frames.items()))
    return build_store(root, frames, out, workers=workers, max_gb=max_gb, progress=progress)
