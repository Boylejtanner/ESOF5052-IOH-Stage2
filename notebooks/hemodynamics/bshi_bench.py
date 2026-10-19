"""Benchmark BSHI loading and build a bounded development subset."""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import bshi as S
from .canonical import atomic_json, utc_now


# throughput
def _bench_rows(idx, n_windows, seed):
    pool = idx[idx.primary_eligible & idx.in_full & idx.has60]
    rng = np.random.default_rng(seed)
    counts = pool.groupby(["dataset", "case_id"]).size()
    order = rng.permutation(len(counts))
    keep, total = [], 0
    for i in order:
        keep.append(counts.index[i])
        total += counts.iloc[i]
        if total >= n_windows:
            break
    keys = set(keep)
    rows = pool[[(d, c) in keys for d, c in zip(pool.dataset, pool.case_id)]]
    return rows.iloc[:n_windows].reset_index(drop=True)


def _bench_worker(root, rows, variant, batch_size, shard):
    t0 = time.perf_counter()
    n = 0
    for b in S.window_batch_iter(root, rows, variant, batch_size, True, 0, 4, case_subset=shard):
        n += len(b["x"])
    return n, time.perf_counter() - t0


def benchmark_throughput(root, n_windows=20000, workers=(1, 8, 16), variant="eap", batch_size=256, seed=5052, torch_workers=16, write=True):
    root = Path(root)
    idx = S.load_window_index(root, columns=["window_id", "dataset", "case_id", "beat_lo", "beat_hi", "primary_eligible", "in_full", "has60"] + S.LABEL_CONTEXT + S.LABEL_OUT)
    rows = _bench_rows(idx, n_windows, seed)
    out = {"windows": int(len(rows)), "cases": int(rows.groupby(["dataset", "case_id"]).ngroups), "variant": variant, "batch_size": batch_size, "target_windows_per_s": 3000,
           "cpu_count": os.cpu_count(), "assemble_only": {}, "at": utc_now()}
    for w in workers:
        if w == 1:
            n, sec = _bench_worker(str(root), rows, variant, batch_size, None)
            out["assemble_only"]["1"] = {"windows_per_s": round(n / sec, 1), "seconds": round(sec, 1)}
            continue
        t0 = time.perf_counter()
        with cf.ProcessPoolExecutor(max_workers=w, mp_context=multiprocessing.get_context("spawn")) as pool:
            futs = [pool.submit(_bench_worker, str(root), rows, variant, batch_size, (i, w)) for i in range(w)]
            res = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        total = sum(n for n, _ in res)
        steady = max(s for _, s in res)
        out["assemble_only"][str(w)] = {"windows_per_s_steady": round(total / steady, 1), "windows_per_s_incl_startup": round(total / wall, 1), "seconds_slowest_worker": round(steady, 1),
                                        "wall_seconds": round(wall, 1)}
    if torch_workers:
        try:
            import torch
            ds = S.make_torch_dataset(root, rows, variant, batch_size=batch_size, shuffle=True, seed=seed)
            loader = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=torch_workers, prefetch_factor=4, persistent_workers=True)
            epochs = []
            for epoch in range(2):                                 # epoch 1 includes the one-off worker start-up; epoch 2 is the steady state
                t0, seen, first_ids = time.perf_counter(), 0, []
                for b in loader:
                    seen += len(b["x"])
                    if epoch == 0 and len(first_ids) < 4:
                        first_ids.append(time.perf_counter() - t0)
                epochs.append({"epoch": epoch + 1, "windows": seen, "seconds": round(time.perf_counter() - t0, 1), "windows_per_s": round(seen / (time.perf_counter() - t0), 1),
                               "seconds_to_first_batches": [round(v, 1) for v in first_ids] if first_ids else None})
            out["torch_dataloader"] = {"workers": torch_workers, "persistent_workers": True, "epochs": epochs, "windows_per_s_steady": epochs[-1]["windows_per_s"],
                                       "note": "epoch 1 includes spawning the workers (sequential on Windows); training pays it once"}
        except Exception as exc:                                   # torch missing or a worker failure: report, do not hide
            out["torch_dataloader"] = {"error": f"{type(exc).__name__}: {exc}"}
    best = max([v.get("windows_per_s_steady", v.get("windows_per_s", 0)) for v in out["assemble_only"].values()] + [out.get("torch_dataloader", {}).get("windows_per_s_steady") or 0])
    sixteen = out["assemble_only"].get("16", {}).get("windows_per_s_steady", 0)
    out["target_met_at_16_workers"] = bool(sixteen >= 3000)
    out["decision"] = ("memory-mappable per-case store NOT built: the parquet loader already exceeds the 1,500 windows/s fallback threshold" if sixteen >= 1500
                       else "below 1,500 windows/s at 16 workers: build the memory-mappable store")
    out["best_windows_per_s"] = best
    if write:
        atomic_json(root / "results" / "bshi_throughput.json", out)
    return out


# dev subset
def _cap_take(frame, k, cap, rng, used):
    """Random ``k`` windows with at most ``cap`` per case in total: ``used`` (case -> windows already taken) is shared across calls and updated."""
    if k <= 0 or not len(frame):
        return frame.iloc[0:0]
    f = frame.assign(_r=rng.random(len(frame))).sort_values("_r", kind="stable")
    f = f.assign(_used=[used.get(c, 0) for c in zip(f.dataset, f.case_id)])
    f = f[(f.groupby(["dataset", "case_id"]).cumcount() + f["_used"]) < cap]
    out = f.head(k)
    for c, n in out.groupby(["dataset", "case_id"]).size().items():
        used[c] = used.get(c, 0) + int(n)
    return out.drop(columns=["_r", "_used"])


def select_dev_windows(idx, cfg):
    """Deterministic stratified selection of full-set windows. Per split the MOVER supply (at most ``max_per_case`` per case) fixes the size so that the MOVER share stays
    >= min_mover_share; events are over-sampled to >= min_event_share when availability allows. The per-case cap holds across events and non-events."""
    d = cfg["dev_subset"]
    rng = np.random.default_rng(int(d["seed"]))
    cap, big = int(d["max_per_case"]), 10 ** 9
    pool = idx[idx.primary_eligible & idx.in_full & idx.has60].copy()
    pool["y"] = pool.y_event.astype(bool)
    parts, notes = [], {}
    for split, n in d["per_split_max"].items():
        p = pool[pool.split_pooled == split]
        m, v = p[p.dataset == "mover"], p[p.dataset == "vitaldb"]
        probe = {}
        m_ev_all = len(_cap_take(m[m.y], big, cap, rng, probe))
        m_avail = m_ev_all + len(_cap_take(m[~m.y], big, cap, rng, probe))
        n_mover = min(m_avail, int(np.ceil((d["min_mover_share"] + 0.05) * n)))
        n_eff = min(int(n), int(n_mover / (d["min_mover_share"] + 0.02)))
        ev_target = int(np.ceil((d["min_event_share"] + 0.02) * n_eff))
        used = {}
        m_ev = _cap_take(m[m.y], min(m_ev_all, int(round(0.3 * n_mover))), cap, rng, used)
        m_ne = _cap_take(m[~m.y], n_mover - len(m_ev), cap, rng, used)
        v_ev = _cap_take(v[v.y], max(0, ev_target - len(m_ev)), cap, rng, used)
        v_ne = _cap_take(v[~v.y], (n_eff - len(m_ev) - len(m_ne)) - len(v_ev), cap, rng, used)
        sel = pd.concat([m_ev, m_ne, v_ev, v_ne])
        pop = p.groupby(["dataset", "y"]).size()
        cnt = sel.groupby(["dataset", "y"]).size()
        w = {k: float(pop[k] / cnt[k]) for k in cnt.index}
        sel = sel.assign(sampling_weight=[w[(a, b)] for a, b in zip(sel.dataset, sel.y)], split=split)
        parts.append(sel)
        notes[split] = {"requested": int(n), "selected": int(len(sel)), "mover": int((sel.dataset == "mover").sum()), "events": int(sel.y.sum()),
                        "mover_events": int(sel[sel.dataset == "mover"].y.sum()), "event_share": round(float(sel.y.mean()), 4), "mover_share": round(float((sel.dataset == "mover").mean()), 4),
                        "mover_supply_under_cap": int(m_avail), "max_per_case": int(sel.groupby(["dataset", "case_id"]).size().max()),
                        "size_limited_by": "MOVER supply under the per-case cap" if n_eff < n else "requested size"}
    return pd.concat(parts).reset_index(drop=True), notes


def _dev_worker(root, rows, out_path, variant):
    parts = []
    for _, g in rows.groupby(["dataset", "case_id"], sort=False):
        for j in range(0, len(g), 128):
            sub = g.iloc[j:j + 128]
            b = S.assemble(root, sub, variant, cache=S.CaseCache(2))
            enc = S.encode_compact(b)
            enc.update(dataset=sub.dataset.to_numpy(), case_id=sub.case_id.astype(str).to_numpy(), patient_key=sub.patient_key.to_numpy(), split=sub.split.to_numpy(),
                       t_end_sec=sub.t_end_sec.to_numpy(), time_to_event_sec=sub.time_to_event_sec.to_numpy(dtype=np.float32),
                       sampling_weight=sub.sampling_weight.to_numpy(dtype=np.float32))
            parts.append(pd.DataFrame(enc))
    frame = pd.concat(parts, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".tmp")
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), tmp, compression="zstd")
    os.replace(tmp, out_path)
    return len(frame), Path(out_path).stat().st_size


def build_dev_subset(root, workers=None, force=False, write=True):
    """data/derived/bshi/dev_subset/split=<s>/part-*.parquet (EAP, compact binary) + results/bshi_dev_subset.json."""
    import shutil
    root = Path(root)
    cfg = S.bshi_config(root)
    out_dir = root / S.OUT_DIR / "dev_subset"
    sig = S.index_signature(root)
    meta_path = root / "results" / "bshi_dev_subset.json"
    if not force and meta_path.exists() and out_dir.exists():
        import json
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if old.get("index_signature") == sig and old.get("config") == cfg["dev_subset"]:
            return old
    if out_dir.exists():
        shutil.rmtree(out_dir)
    idx = S.load_window_index(root)
    sel, notes = select_dev_windows(idx, cfg)
    n_workers = max(1, min(int(workers or cfg["workers"]), (os.cpu_count() or 8) - 2))
    started = time.perf_counter()
    jobs = []
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        for split, g in sel.groupby("split"):
            keys = list(g.groupby(["dataset", "case_id"], sort=False).groups)
            for shard in range(n_workers):
                mine = set(keys[shard::n_workers])
                sub = g[[(a, b) in mine for a, b in zip(g.dataset, g.case_id)]]
                if len(sub):
                    jobs.append((split, shard, pool.submit(_dev_worker, str(root), sub.reset_index(drop=True), str(out_dir / f"split={split}" / f"part-{shard:04d}.parquet"), "eap")))
        results = [(s, k, f.result()) for s, k, f in jobs]
    total_bytes = sum(r[2][1] for r in results)
    if total_bytes > cfg["dev_subset"]["max_gb"] * 1e9:
        raise RuntimeError(f"dev subset is {total_bytes / 1e9:.2f} GB, above max_gb={cfg['dev_subset']['max_gb']}: lower per_split_max in configs/hemodynamics.json")
    by = sel.groupby(["split", "dataset", "y"]).agg(windows=("window_id", "size"), weight_mean=("sampling_weight", "mean"), patients=("patient_key", "nunique")).reset_index()
    summary = {"index_signature": sig, "config": cfg["dev_subset"], "bytes": int(total_bytes), "gb": round(total_bytes / 1e9, 3), "windows": int(len(sel)), "parts": len(results),
               "seconds": round(time.perf_counter() - started, 1), "per_split": notes, "strata": by.to_dict("records"),
               "patients": int(sel.patient_key.nunique()), "cases": int(sel.groupby(["dataset", "case_id"]).ngroups),
               "max_windows_per_case": int(sel.groupby(["dataset", "case_id"]).size().max()),
               "format": "parquet, one row per window: waveforms_f16 (BINARY, 3x60x200 float16), scalars_f32 (BINARY, 60x17 float32), beat_mask_u8, context_f32, variant_mask_f32, "
                         "labels, sampling_weight; EAP; derive EA with bshi.apply_variant. Decode with bshi.decode_compact.",
               "weighting": "sum of sampling_weight over a stratum (split, dataset, event) equals the number of full-set windows in that stratum, so a weighted metric recovers natural prevalence",
               "at": utc_now()}
    if write:
        atomic_json(meta_path, summary)
    return summary


def verify_dev_subset(root, n=64, seed=5052):
    """Decode some stored windows and compare with a fresh assembly of the same windows."""
    root = Path(root)
    files = sorted((root / S.OUT_DIR / "dev_subset").rglob("part-*.parquet"))
    rng = np.random.default_rng(seed)
    f = files[int(rng.integers(len(files)))]
    t = pq.read_table(f).to_pandas()
    t = t.iloc[rng.choice(len(t), min(n, len(t)), replace=False)].reset_index(drop=True)
    back = S.decode_compact(t.to_dict("list"))
    idx = S.load_window_index(root).set_index("window_id").loc[t.window_id.to_numpy()].reset_index()
    fresh = S.assemble(root, idx, "eap", cache=S.CaseCache(4))
    return {"windows_checked": int(len(t)), "file": str(f.relative_to(root)), "planes_0_2_max_abs_diff": float(np.abs(back["x"][:, :3] - fresh["x"][:, :3]).max()),
            "planes_3_6_exactly_equal": bool(np.array_equal(back["x"][:, 3:], fresh["x"][:, 3:])), "context_equal": bool(np.array_equal(back["context"], fresh["context"])),
            "labels_equal": bool(all(np.array_equal(back[c], fresh[c]) for c in S.LABEL_OUT))}
