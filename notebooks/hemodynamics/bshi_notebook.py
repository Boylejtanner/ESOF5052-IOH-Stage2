"""Support training notebooks with run identifiers, progress paths, and experiment summaries."""
from __future__ import annotations
# Keep experiment setup, run paths, and summaries out of notebook training cells.

import dataclasses
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bshi_run as R
from . import bshi_train as T


def run_id(cfg, smoke=False):
    h = hashlib.sha1(json.dumps(cfg.to_dict(), sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{'smoke_' if smoke else ''}{cfg.variant}_{cfg.train_on}_{cfg.optimizer}_{h}"


def loader_throughput(loader_fn, batches=30, warmup=3):
    """Windows/s a loader delivers to the CPU side (the GPU decode and the model are not included)."""
    it = iter(loader_fn(0) if callable(loader_fn) else loader_fn)
    n, t0 = 0, None
    for i, b in enumerate(it):
        if i == warmup:
            t0 = time.perf_counter()
        elif i > warmup:
            n += len(b["window_id"])
        if i >= warmup + batches:
            break
    return round(n / max(time.perf_counter() - t0, 1e-9), 1) if t0 else float("nan")


def sanity_run(root, cfg, smoke, steps=50, device="cuda"):
    """50 optimizer steps from scratch in a throw-away folder: the loss must fall and be finite; returns first / last loss and the speed."""
    c = dataclasses.replace(cfg, epochs=1, max_steps_per_epoch=steps, patience=1, num_workers=min(cfg.num_workers, 4), warmup_epochs=0.2)
    with tempfile.TemporaryDirectory() as tmp:
        tr, data = R.build_trainer(root, c, tmp, smoke=smoke, epoch_windows=steps * c.batch_size, device=device, log=lambda *a: None, progress=False)
        res = tr.fit(resume=False)
    h = res["history"].iloc[0]
    return {"steps": int(h.steps), "mean_loss": round(float(h.total), 4), "windows_per_s": float(h.train_windows_per_s), "finite": bool(np.isfinite(h.total))}


def cluster_status(root):
    """What the Spark master (if it runs) knows: alive workers, and which of them are on ANOTHER machine (the laptop). Never starts anything; a missing master is a normal answer."""
    from . import cluster as C
    out = {"master_running": False, "master": None, "workers": [], "remote": []}
    try:
        cfg = C.load_config(root)
        host = C.master_host(cfg)
        out["master"] = host
        view = C.master_view(host, cfg["master_ui"], timeout=3)
    except Exception as exc:
        out["error"] = repr(exc)[:150]
        return out
    if view:
        out["master_running"] = True
        out["workers"] = [{"id": w["id"], "host": w["host"], "cores": w["cores"], "memory_mb": w["memory"]} for w in C.alive_workers(view)]
        out["remote"] = [w for w in out["workers"] if w["host"] != host]
    return out


def resolve_mode(root, mode, status=None):
    """The mode that will really run and why. ``single_gpu`` (desktop only) never needs a cluster. ``trial_parallel`` uses the laptop when it is registered and otherwise runs the trials one
    after another on the desktop. ``ddp_2node`` needs the laptop; without it the run falls back to ``single_gpu`` instead of failing."""
    if mode not in ("single_gpu", "trial_parallel", "ddp_2node"):
        raise ValueError("MODE must be 'single_gpu', 'trial_parallel' or 'ddp_2node'")
    if mode == "single_gpu":
        return "single_gpu", "desktop only: no cluster, no laptop, no Java needed"
    st = status or cluster_status(root)
    n = len(st["remote"])
    if mode == "ddp_2node":
        if n:
            return "ddp_2node", f"laptop registered ({st['remote'][0]['host']}): two-machine training is possible (check the scaling estimate in section 8 first)"
        return "single_gpu", "no laptop worker is registered with the Spark master: FALLING BACK to training on the desktop only"
    if n:
        return "trial_parallel", f"{1 + n} machines: the trials run in parallel, one per GPU"
    return "trial_sequential", "no laptop worker is registered: the trials run one after another on the desktop (no Spark needed)"


def run_trials_local(root, cfgs, log=print):
    """The trial harness without Spark: runs ``bshi_sweep.run_trial`` for every config one after another on this machine's GPU (one retry per failure). Same result rows as ``cluster.run_trials``."""
    import os
    import socket
    from . import bshi_sweep as W
    os.environ["BSHI_ROOT"] = str(root)
    out = []
    # Process the selected records in order and retain their individual results.
    for i, c in enumerate(cfgs):
        t0, m, err = time.time(), None, None
        for attempt in (1, 2):
            try:
                m, err = W.run_trial(c), None
                break
            except Exception as exc:
                err = repr(exc)[:200]
        out.append({"trial_id": i, "ok": m is not None, "attempts": attempt, "worker": socket.gethostname(), "seconds": round(time.time() - t0, 1), "metrics": m, "error": err})
        log(f"trial {i + 1}/{len(cfgs)} {c.get('name')}: " + (f"validation AUPRC {m['val_auprc']:.4f}, {out[-1]['seconds']:.0f} s" if m else f"FAILED ({err})"))
    return out


def store_status(root, train_on="pooled"):
    """Does the compact store of this experiment exist (the fast training path)? Returns a dict with what the notebook prints."""
    from . import bshi_store as ST
    p = ST.store_path(root, train_on)
    ok = (p / "MANIFEST.sha256").exists()
    out = {"path": str(p), "exists": ok}
    if ok and (p / "summary.json").exists():
        s = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        out.update(windows=s["windows"], gb=s["gb"])
    return out


def store_estimate(root, train_on, stride=2, windows="eap_full"):
    """Windows and disk size a store of this experiment would take (77 kB per window), and the free disk: look before building one."""
    # Calculate the observation count for the following step.
    import shutil
    from . import bshi_data as D
    from . import bshi_store as ST
    fr = ST.experiment_frames(D.protocol_frame(root, windows), train_on, stride)
    n = sum(len(v) for v in fr.values())
    gb = n * 76.9e3 / 1e9
    return {"windows": {k: len(v) for k, v in fr.items()}, "estimated_gb": round(gb, 1), "free_gb": round(shutil.disk_usage(root).free / 1e9), "ok_to_build": bool(shutil.disk_usage(root).free / 1e9 - gb > 150)}


def paths_table(root, rid, train_on="pooled"):
    """Where everything of a run is written (relative to the project folder)."""
    # Calculate the result array for the following step.
    from . import bshi_store as ST
    r = Path(root)
    rows = [("checkpoints: best.pt (best validation AUPRC), last.pt (to resume), epochs.csv", f"models/{rid}/"),
            ("live progress: progress.json / .jsonl / .png / .html", f"models/{rid}/live/"),
            ("final metrics and configuration", f"results/train/run_{rid}.json"),
            ("model card", f"models/{rid}/MODEL_CARD.md"),
            ("fast training data (compact store of this experiment)", str(ST.store_path(r, train_on).relative_to(r)).replace("\\", "/") + "/"),
            ("BSHI images: NOT saved, built on the fly from", "data/derived/beats/ (beat store) + data/derived/bshi/window_index_v1.parquet"),
            ("look at BSHI pictures", "notebooks/bshi_build.ipynb -> figures/bshi/")]
    return pd.DataFrame(rows, columns=["what", "where"])


def launch_commands(root, cfg, master_ip, speeds="2500,900", accum=4, compress="fp16", global_batch=256, store="data/derived/bshi/sweep_subset"):
    """The exact commands for both machines (rank 0 on the desktop, rank 1 on the laptop)."""
    base = (f"python -m hemodynamics.bshi_ddp --world-size 2 --master tcp://{master_ip}:29500 --store {store} --global-batch {global_batch} --speeds {speeds} --accum {accum} "
            f"--compress {compress} --epochs {cfg.epochs} --set optimizer={cfg.optimizer} lr={cfg.lr} weight_decay={cfg.weight_decay} variant={cfg.variant}")
    return {"desktop (rank 0)": f"cd {root}; .\\.venv\\Scripts\\Activate.ps1; {base} --rank 0 --out runs\\ddp_run", "laptop (rank 1)": f"cd C:\\bshi_worker; .\\.venv\\Scripts\\Activate.ps1; {base} --rank 1 --out runs\\ddp_run"}


def save_results(root, rid, cfg, res, extra=None):
    """``results/train/run_<id>.json`` and a model card next to the checkpoints in ``models/<id>/``."""
    # Store the settings and results as JSON for later inspection.
    out = {"run_id": rid, "config": cfg.to_dict(), "created": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": {k: v for k, v in res.items() if not k.startswith("_")}, **(extra or {})}
    p = Path(root) / "results" / "train" / f"run_{rid}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    # Store the settings and results as JSON for later inspection.
    p.write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    card = Path(root) / "models" / rid / "MODEL_CARD.md"
    card.parent.mkdir(parents=True, exist_ok=True)
    m = res.get("validation", {})
    card.write_text("\n".join([f"# BSHI model {rid}", "", f"* input recipe: `{cfg.variant}`, trained on: `{cfg.train_on}`, optimizer `{cfg.optimizer}` (lr {cfg.lr}, weight decay {cfg.weight_decay}, batch {cfg.batch_size}, {cfg.schedule})",
                               f"* validation AUROC {m.get('auroc', float('nan')):.4f}, AUPRC {m.get('auprc', float('nan')):.4f}, Brier {m.get('brier', float('nan')):.4f} (natural prevalence)",
                               "* research prototype, not a clinical device; trained on VitalDB and MOVER windows, evaluated patient-independently",
                               f"* config and metrics: `results/train/run_{rid}.json`; checkpoints `best.pt`, `last.pt` and `epochs.csv` in this folder", ""]), encoding="utf-8")
    return p
