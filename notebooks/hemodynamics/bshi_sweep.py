"""Compare optimizer, learning-rate, batch-size, and schedule configurations.

``run_trial(cfg)`` executes inside a Spark task on one machine and trains on that machine's GPU from the memory-mapped sweep subset (``data/derived/bshi/sweep_subset``): TRAIN windows
for fitting, VALIDATION windows (weighted back to natural prevalence with ``sampling_weight``) for early stopping and for the objective. The test split is never read here.
Stages (see ``run_sweep``): B1 optimizer x peak-LR grid (candidates from the LR range test), B2 weight decay / batch size / one-cycle around the best, B3 top-3 with extra seeds.
"""
from __future__ import annotations

import math
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

OPTIMIZERS = {"adamw_b999": {"optimizer": "adamw", "beta2": 0.999}, "adamw_b95": {"optimizer": "adamw", "beta2": 0.95}, "sgd_nesterov": {"optimizer": "sgd"}}
STORE_REL = Path("data") / "derived" / "bshi" / "sweep_subset"


def run_trial(cfg):
    """One training run (6 short epochs by default) -> metrics dict. ``cfg`` keys: name, opt (a key of OPTIMIZERS), lr, weight_decay, batch_size, schedule, epochs, patience, epoch_windows,
    seed, variant, gate, dropout, sampler, store (optional path relative to BSHI_ROOT)."""
    import torch
    from . import bshi_data as D
    from . import bshi_store as ST
    from . import bshi_train as T
    from .bshi_model import ModelConfig
    root = Path(os.environ.get("BSHI_ROOT", "."))
    store = root / cfg.get("store", str(STORE_REL))
    tr, va = ST.Store(store, "train"), ST.Store(store, "validation")
    o = OPTIMIZERS[cfg["opt"]]
    tcfg = T.TrainConfig.from_dict({"optimizer": o["optimizer"], "beta2": o.get("beta2", 0.999), "lr": cfg["lr"], "weight_decay": cfg.get("weight_decay", 0.01), "batch_size": cfg.get("batch_size", 256),
                                    "epochs": cfg.get("epochs", 6), "patience": cfg.get("patience", 2), "schedule": cfg.get("schedule", "cosine"), "seed": cfg.get("seed", 5052),
                                    "variant": cfg.get("variant", "eap"), "sampler": cfg.get("sampler", "natural"), "warmup_epochs": cfg.get("warmup_epochs", 1.0), "num_workers": 0, "channels_last": False})
    mcfg = ModelConfig(variant=tcfg.variant, gate=cfg.get("gate", True), dropout=cfg.get("dropout", 0.1))
    epoch_windows = int(cfg.get("epoch_windows", len(tr)))
    spe = math.ceil(epoch_windows / tcfg.batch_size)
    pre = D.Preprocessor(tcfg.variant, D.load_stats(root), clamp=tcfg.clamp, device="cuda", channels_last=tcfg.channels_last)
    out_dir = Path(tempfile.mkdtemp(prefix=f"bshi_trial_{cfg.get('name', 'x')}_"))
    torch.cuda.reset_peak_memory_stats()
    try:
        trainer = T.Trainer(tcfg, mcfg, pre, D.store_train_loader(tr, tcfg.batch_size, epoch_windows, tcfg.seed, tcfg.sampler), D.store_eval_loader(va, 512), out_dir, device="cuda",
                            steps_per_epoch=spe, log=lambda *a: None)
        res = trainer.fit(resume=False)
        h = res["history"]
        best = h.loc[h.val_auprc.idxmax()] if h.val_auprc.notna().any() else h.iloc[-1]
        return {"name": cfg.get("name"), "opt": cfg["opt"], "lr": cfg["lr"], "weight_decay": tcfg.weight_decay, "batch_size": tcfg.batch_size, "schedule": tcfg.schedule, "seed": tcfg.seed,
                "val_auprc": float(best.val_auprc), "val_auroc": float(best.val_auroc), "val_brier": float(best.val_brier), "best_epoch": int(res["best_epoch"]), "epochs_run": int(len(h)),
                "stopped_early": bool(res["stopped_early"]), "seconds": float(res["seconds"]), "train_windows_per_s": float(h.train_windows_per_s.median()),
                "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "final_train_loss": float(h.total.iloc[-1]), "diverged": bool(not np.isfinite(h.total).all()),
                "curve": [[int(e), float(l), None if a != a else float(a)] for e, l, a in zip(h.epoch, h.total, h.val_auprc)]}
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def plot_lr_range(curves, sugg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import figstyle as F
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for (name, c), col in zip(curves.items(), ("#1f77b4", "#2ca02c", "#d62728")):
        ax.plot(c["lr"], c["smoothed"], color=col, label=name)
        for v in sugg[name]["candidates"]:
            ax.axvline(v, color=col, lw=0.6, ls=":")
    ax.set_xscale("log")
    ax.legend(title="optimizer (dotted = sweep candidates)", fontsize=8)
    F.finish_axes(ax, "learning rate (log scale)", "training loss (smoothed, multi-task)", "learning-rate range test")
    fig.tight_layout()
    F.audit_figure(fig, "fig_lr_range")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_sweep(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from . import figstyle as F
    d = df[df.ok & df.val_auprc.notna()]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for (name, g), col in zip(d[d.stage.str.startswith("B1")].groupby("opt"), ("#1f77b4", "#2ca02c", "#d62728")):
        g = g.sort_values("lr")
        ax[0].plot(g.lr, g.val_auprc, marker="o", color=col, label=name)
    ax[0].set_xscale("log")
    ax[0].legend(title="optimizer", fontsize=8)
    F.finish_axes(ax[0], "peak learning rate (log scale)", "validation AUPRC (natural prevalence, weighted)", "B1: optimizer x peak LR (batch 256, 6 short epochs)")
    top = d.sort_values("val_auprc", ascending=False).head(12).iloc[::-1]
    ax[1].barh(top.name, top.val_auprc, color="#2ca02c")
    F.finish_axes(ax[1], "validation AUPRC (natural prevalence, weighted)", "trial", "best 12 trials of all stages")
    fig.tight_layout()
    F.audit_figure(fig, "fig_sweep")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def lr_range_all(root=".", steps=300, batch_size=256, write=True):
    """M2: one LR range-test curve per optimizer on the TRAIN part of the sweep subset; returns the curves and the suggested peak-LR candidates."""
    import torch
    from . import bshi_data as D
    from . import bshi_measure as M
    from . import bshi_store as ST
    root = Path(root)
    tr = ST.Store(root / STORE_REL, "train")
    pre = D.Preprocessor("eap", D.load_stats(root), device="cuda", channels_last=False)
    loader = D.store_train_loader(tr, batch_size, None, 5052, "natural")
    curves, sugg = {}, {}
    for name, o in OPTIMIZERS.items():
        hi = 1.0 if o["optimizer"] == "adamw" else 3.0
        lo = 1e-5 if o["optimizer"] == "adamw" else 1e-4
        c = M.m2_lr_range(loader, pre, o["optimizer"], lr_lo=lo, lr_hi=hi, steps=steps, beta2=o.get("beta2", 0.999), channels_last=False)
        curves[name] = c
        sugg[name] = M.suggest_lrs(c)
        torch.cuda.empty_cache()
    if write:
        import json
        p = root / "results" / "train" / "lr_range.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"suggestions": sugg, "curves": {k: {"lr": v["lr"].tolist(), "smoothed": v["smoothed"].tolist()} for k, v in curves.items()}, "steps": steps, "batch_size": batch_size}), encoding="utf-8")
    return curves, sugg


def trials_frame(results):
    rows = []
    for r in results:
        m = r["metrics"] or {}
        rows.append({"trial_id": r["trial_id"], "ok": r["ok"], "attempts": r["attempts"], "worker": r["worker"], "error": r["error"], **{k: v for k, v in m.items() if k != "curve"}, "curve": m.get("curve")})
    return pd.DataFrame(rows)


def rank(df):
    """Objective: validation AUPRC (weighted to natural prevalence), tie-break validation AUROC then shorter wall time."""
    d = df[df.ok & df.val_auprc.notna() & ~df.diverged.fillna(False)]
    return d.sort_values(["val_auprc", "val_auroc", "seconds"], ascending=[False, False, True])


def run_sweep(spark, root, suggestions, epoch_windows=29000, epochs=6, deadline_min=40.0, log=print, partial_path=None):
    """B1-B3 through ``cluster.run_trials``; stops starting new stages when the wall-time budget is used. Returns the trials DataFrame (all stages) and the final ranking."""
    from . import cluster as C
    t0 = time.time()
    base = {"epochs": epochs, "patience": 2, "epoch_windows": epoch_windows, "batch_size": 256, "weight_decay": 0.01, "schedule": "cosine", "seed": 5052}
    stages = []

    def stage(label, cfgs):
        if (time.time() - t0) / 60 > deadline_min:
            log(f"[sweep] budget used, skipping {label}")
            return pd.DataFrame()
        t1 = time.time()
        df = trials_frame(C.run_trials(spark, cfgs, fn="hemodynamics.bshi_sweep:run_trial", retries=1))
        df["stage"] = label
        log(f"[sweep] {label}: {len(cfgs)} trials in {(time.time() - t1) / 60:.1f} min; best val AUPRC {df.val_auprc.max():.4f}")
        stages.append(df)
        if partial_path:
            pd.concat(stages, ignore_index=True).drop(columns=["curve"]).to_csv(partial_path, index=False)
        return df

    b1_cfgs = [{**base, "name": f"{o}_lr{lr:g}", "opt": o, "lr": lr} for o, s in suggestions.items() for lr in s["candidates"]]
    b1 = stage("B1 optimizer x peak LR", b1_cfgs)
    best = rank(b1).iloc[0]
    b2_cfgs = [{**base, "name": f"{best.opt}_wd0.05", "opt": best.opt, "lr": best.lr, "weight_decay": 0.05},
               {**base, "name": f"{best.opt}_bs128", "opt": best.opt, "lr": best.lr, "batch_size": 128},
               {**base, "name": f"{best.opt}_bs384", "opt": best.opt, "lr": best.lr, "batch_size": 384},
               {**base, "name": f"{best.opt}_onecycle", "opt": best.opt, "lr": best.lr, "schedule": "onecycle"}]
    b2 = stage("B2 weight decay / batch size / one-cycle", b2_cfgs)
    allr = pd.concat([x for x in stages if len(x)], ignore_index=True)
    top = rank(allr).head(3)
    cfg_of = {c["name"]: c for c in b1_cfgs + b2_cfgs}
    b3_cfgs = [{**cfg_of[r["name"]], "name": f"{r['name']}_seed{s}", "seed": s} for _, r in top.iterrows() for s in (5053, 5054)]
    stage("B3 top-3 x 2 more seeds", b3_cfgs)
    df = pd.concat([x for x in stages if len(x)], ignore_index=True)
    df["minutes_elapsed"] = round((time.time() - t0) / 60, 1)
    return df
