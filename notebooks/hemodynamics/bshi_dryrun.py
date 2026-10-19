"""Run a bounded training experiment and measure GPU, CPU, and data-loader throughput.
projected 30-epoch time, a checkpoint reload check and the validation AUROC / AUPRC next to the G0 and G4 logistic baselines on the SAME validation windows.

    python -m hemodynamics.bshi_dryrun --epochs 3 --out models/dry_run          (cwd = project root, PYTHONPATH = notebooks)

The test split is never read. ``--smoke`` uses the compact sweep subset and tiny epochs (for the unit test).
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bshi_data as D
from . import bshi_run as R
from . import bshi_train as T
from .bshi_model import BshiNet


def summarize_monitor(samples, t_train):
    """GPU utilization / VRAM / CPU statistics over the samples taken while training ran (``t_train``: list of (start, end) wall-clock intervals)."""
    if not samples:
        return {}
    s = pd.DataFrame(samples)
    m = np.zeros(len(s), dtype=bool)
    for a, b in t_train:
        m |= ((s.t >= a) & (s.t <= b)).to_numpy()
    s = s[m] if m.any() else s
    return {"samples": int(len(s)), "gpu_util_mean_%": round(float(s.gpu_util.mean()), 1), "gpu_util_median_%": float(s.gpu_util.median()), "share_samples_gpu_util_ge_80_%": round(100 * float((s.gpu_util >= 80).mean()), 1),
            "vram_used_max_gb": round(float(s.vram_mb.max()) / 1024, 2), "cpu_util_mean_%": round(float(s.cpu_util.mean()), 1), "cpu_util_max_%": round(float(s.cpu_util.max()), 1)}


def bottleneck(mon):
    if not mon:
        return "unknown (no samples)"
    if mon["share_samples_gpu_util_ge_80_%"] >= 80:
        return "none: the GPU is busy (>= 80% utilization in at least 80% of the samples)"
    if mon["cpu_util_mean_%"] >= 60:
        return "the data loader (CPU assembly): GPU waits for batches while the CPU is busy"
    return "neither side is saturated (small kernels / launch latency or input transfer); the GPU is not the limit but the loader is not either"


def reload_check(trainer, d, out_dir, cfg, n_batches=4):
    """Load best.pt into a fresh network and compare its predictions on the first validation batches with the trained model's."""
    ck = torch.load(Path(out_dir) / "best.pt", map_location="cpu", weights_only=False)
    m2 = BshiNet(trainer.model_cfg).to(trainer.device)
    m2.load_state_dict(ck["model"])
    a = T.predict(trainer.model, d["pre"], d["val_loader"](), cfg, trainer.device, max_batches=n_batches)
    b = T.predict(m2, d["pre"], d["val_loader"](), cfg, trainer.device, max_batches=n_batches)
    diff = float(np.max(np.abs(a.logit.to_numpy() - b.logit.to_numpy())))
    return {"best_epoch": int(ck["epoch"]), "windows_compared": int(len(a)), "max_abs_logit_difference": diff, "ok": bool(diff < 1e-2)}


def monitor_from_progress(out_dir):
    """GPU / CPU samples taken by the live progress display while the run trained (so a finished run can be summarized without re-running it)."""
    from . import bshi_progress as P
    rows = [r for r in P.read_jsonl(Path(out_dir) / "live" / "progress.jsonl") if "event" not in r and r.get("phase") == "train" and r.get("gpu_util") is not None]
    return [{"t": i, "gpu_util": r["gpu_util"], "vram_mb": r.get("vram_gb", 0.0) * 1024, "cpu_util": r.get("cpu_util", float("nan"))} for i, r in enumerate(rows)]


def run_dry(root, out_dir, epochs=3, smoke=False, epoch_windows=None, overrides=None, log=print, finalize_only=False):
    """Train a few epochs and write ``dry_run.json``. ``finalize_only``: the training already happened in ``out_dir`` (epochs.csv, best.pt, live/progress.jsonl): redo only the checks and the report."""
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = R.load_cfg(root, epochs=epochs, patience=max(epochs, 2), **(overrides or {}))
    ew = epoch_windows if smoke else (epoch_windows or None)
    samples, stop, t_train = [], threading.Event(), []
    mon = threading.Thread(target=R.gpu_monitor, args=(stop, samples, 2.0), daemon=True)
    times = []

    def mark(msg):
        times.append(time.time())
        log(msg)

    t_start = time.time()
    trainer, d = R.build_trainer(root, cfg, out_dir, smoke=smoke, epoch_windows=ew, device="cuda", log=mark, progress=not finalize_only)
    t_built = time.time()
    if finalize_only:
        hist = pd.read_csv(out_dir / "epochs.csv")
        trainer.model.load_state_dict(torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)["model"])
        samples = monitor_from_progress(out_dir)
        total = json.loads((out_dir / "live" / "progress.json").read_text(encoding="utf-8"))["elapsed_s"]
        val_s = (total - float(hist.train_seconds.sum())) / max(1, len(hist))          # validation + checkpoint writing per epoch
        epoch_wall = (hist.train_seconds + val_s).tolist()
    else:
        mon.start()
        t_fit0 = time.time()
        times.clear()
        times.append(t_fit0)
        res = trainer.fit(resume=False)
        t_train.append((t_fit0, time.time()))
        stop.set()
        mon.join(timeout=5)
        hist = res["history"]
        epoch_wall = np.diff(np.array(times)).tolist()
        val_s = float(np.mean(trainer.progress.val_seconds)) if trainer.progress is not None and trainer.progress.val_seconds else float("nan")
    mon_stats = summarize_monitor(samples, t_train)
    rc = reload_check(trainer, d, out_dir, cfg)
    wps = float(np.median(hist.train_windows_per_s)) if len(hist) else float("nan")
    full_epoch_s = d["n_train"] / wps + val_s if wps == wps and val_s == val_s else float("nan")
    out = {"config": cfg.to_dict(), "data_source": d["source"], "train_windows_full_epoch": d["n_train"], "train_windows_run_per_epoch": int(hist.windows.iloc[0]) if len(hist) else None,
           "validation_windows": d["n_val"], "steps_per_epoch_run": d["steps_per_epoch"],
           "build_seconds": round(t_built - t_start, 1), "epoch_wall_seconds": [round(x, 1) for x in epoch_wall], "epoch_train_seconds": hist.train_seconds.round(1).tolist(),
           "epoch_windows_per_s": hist.train_windows_per_s.tolist(), "validation_seconds_mean": round(val_s, 1) if val_s == val_s else None,
           "val_auroc": hist.val_auroc.tolist(), "val_auprc": hist.val_auprc.tolist(), "report_from_saved_run": bool(finalize_only),
           "first_epoch_vs_next_seconds": [epoch_wall[0], epoch_wall[1]] if len(epoch_wall) > 1 else None,
           "full_epoch_minutes_projected": round(full_epoch_s / 60, 1) if full_epoch_s == full_epoch_s else None,
           "projected_30_epoch_hours": round(30 * full_epoch_s / 3600, 2) if full_epoch_s == full_epoch_s else None, "monitor": mon_stats, "bottleneck": bottleneck(mon_stats),
           "checkpoint_reload": rc, "total_minutes": round((time.time() - t_start) / 60, 1)}
    if not smoke:
        fr = d["frames"]
        base = R.baselines_on(root, {"validation": fr["val"]})
        if d.get("stores"):
            va = d["stores"]["val"]
            m_full, _ = R.evaluate_loader(trainer, d["pre"], D.store_eval_loader(va, 512)(), cfg, va.meta.set_index("window_id").patient_key, 100, d["shift"])
        else:
            m_full, _ = R.evaluate_frame(root, trainer, d["pre"], cfg, fr["val"], d["flags"], reps=100)
        out["validation_same_windows"] = {"network_auroc": m_full.get("auroc"), "network_auprc": m_full.get("auprc"), "network_ci": m_full.get("ci"),
                                          **{f"{k[0]}_logistic_{k[1]}": v for k, v in base.items()}}
    (out_dir / "dry_run.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("BSHI_ROOT", "."))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--out", default="models/dry_run")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--finalize-only", action="store_true", help="the training already ran in --out: redo only the checks and the report")
    ap.add_argument("--epoch-windows", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="optimizer steps per epoch (0 = the whole training set); the projection to full epochs is computed from windows/s")
    a = ap.parse_args(argv)
    out = run_dry(a.root, a.out, a.epochs, a.smoke, a.epoch_windows or None, overrides={"max_steps_per_epoch": a.max_steps} if a.max_steps else None, finalize_only=a.finalize_only)
    print(json.dumps({k: out[k] for k in ("epoch_wall_seconds", "epoch_windows_per_s", "val_auroc", "val_auprc", "projected_30_epoch_hours", "monitor", "bottleneck", "checkpoint_reload", "total_minutes")}, indent=1, default=str))


if __name__ == "__main__":
    main()
