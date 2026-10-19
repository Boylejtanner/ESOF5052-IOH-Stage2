"""Measure GPU throughput and learning-rate behavior for optimizer selection."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import bshi_train as T
from .bshi_model import ModelConfig


def nvidia_smi_memory():
    """(used GB, total GB, free GB) as seen by the driver (includes Windows and other programs)."""
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.split(",")
    used, total, free = (float(v) / 1024 for v in out)
    return used, total, free


def m1_throughput(root=".", batches=(128, 256, 384, 512, 768, 1024), iters=12, write=True):
    """Training-step windows/s and peak VRAM for batch size x {bf16, fp32} x {channels_last on, off} on synthetic batches. A combination stops at its first out-of-memory batch size."""
    used, total, free = nvidia_smi_memory()
    rows = []
    for bf16 in (True, False):
        for cl in (True, False):
            for bs in batches:
                r = T.benchmark_step(ModelConfig(), bs, bf16=bf16, channels_last=cl, iters=iters)
                rows.append(r)
                if r["oom"] or r["peak_vram_gb"] > free:
                    break
    ok = [r for r in rows if not r["oom"]]
    best = max(ok, key=lambda r: r["windows_per_s"])
    fit = [r for r in ok if r["peak_vram_gb"] <= free - 1.0]
    largest = max((r for r in fit if r["bf16"] and r["channels_last"]), key=lambda r: r["batch_size"])
    res = {"vram_used_by_other_programs_gb": round(used, 2), "vram_free_gb": round(free, 2), "headroom_gb": 1.0, "rows": rows, "best": best, "largest_bf16_channels_last_batch_with_1gb_headroom": largest["batch_size"],
           "bf16_speedup_over_fp32_at_256": round(next(r["windows_per_s"] for r in ok if r["bf16"] and r["channels_last"] and r["batch_size"] == 256) /
                                                  next(r["windows_per_s"] for r in ok if not r["bf16"] and r["channels_last"] and r["batch_size"] == 256), 2),
           "channels_last_speedup_at_256_bf16": round(next(r["windows_per_s"] for r in ok if r["bf16"] and r["channels_last"] and r["batch_size"] == 256) /
                                                      next(r["windows_per_s"] for r in ok if r["bf16"] and not r["channels_last"] and r["batch_size"] == 256), 2)}
    if write:
        p = Path(root) / "results" / "train" / "gpu_throughput.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return res


def m2_lr_range(loader_fn, pre, optimizer, lr_lo=1e-5, lr_hi=1.0, steps=300, model_cfg=None, beta2=0.999, weight_decay=0.01, device="cuda", amp="bf16", seed=5052, smooth=0.9, channels_last=False):
    """Exponential learning-rate sweep over ``steps`` optimizer steps with a fresh model; returns arrays of lr and smoothed loss (and the raw loss). The curve's steepest descent / minimum
    guide the peak-LR candidates; it stops early when the loss explodes (4x the best smoothed loss)."""
    T.set_seed(seed)
    cfg = T.TrainConfig(optimizer=optimizer, beta2=beta2, weight_decay=weight_decay, lr=lr_lo, amp=amp, schedule="constant")
    dev = torch.device(device)
    model = T.BshiNet(model_cfg or ModelConfig()).to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = T.build_optimizer(model, cfg, dev)
    lrs, raw, sm, best, avg = [], [], [], float("inf"), None
    it = iter(loader_fn(0))
    for step in range(steps):
        lr = lr_lo * (lr_hi / lr_lo) ** (step / max(1, steps - 1))
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader_fn(step))
            batch = next(it)
        b = pre(batch)
        model.train()
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
            out = model(b["x"], b["context"], b["beat_mask"])
        loss, _ = T.multitask_loss({k: v.float() for k, v in out.items()}, b, cfg.loss_weights)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        v = float(loss)
        if not np.isfinite(v):
            break
        avg = v if avg is None else smooth * avg + (1 - smooth) * v
        lrs.append(lr), raw.append(v), sm.append(avg)
        best = min(best, avg)
        if step > 20 and avg > 4 * best:
            break
    return {"optimizer": optimizer, "beta2": beta2, "lr": np.array(lrs), "loss": np.array(raw), "smoothed": np.array(sm)}


def suggest_lrs(curve, n=3):
    """Peak-LR candidates from a range-test curve. The smoothed-loss minimum sits at the edge of divergence, so the candidates bracket BELOW it: min/30, min/10 and min/3. Also returns the LR
    of the steepest descent for reference."""
    lr, s = curve["lr"], curve["smoothed"]
    k_min = int(np.argmin(s))
    grad = np.gradient(s, np.log(lr))
    steep = float(lr[int(np.argmin(grad))])
    x = float(lr[k_min])
    return {"lr_at_min_loss": x, "lr_steepest_descent": steep, "candidates": [float(f"{c:.2g}") for c in (x / 30.0, x / 10.0, x / 3.0)][:n]}
