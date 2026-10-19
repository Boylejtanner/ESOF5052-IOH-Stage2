"""Orchestration for ``notebooks/train_bshi.ipynb`` and the dry run: data wiring (SMOKE = the compact sweep subset, FULL = the on-the-fly assembler on the real windows), training with the
resumable ``Trainer``, evaluation with patient-bootstrap CIs, G0/G4 baselines on the SAME windows, and the plots (every axis labelled with units).
The test split is only read by ``evaluate_test`` and nothing in training or selection calls it.
"""
from __future__ import annotations
# Connect input recipes, data loaders, training, and saved evaluation results.

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bshi as S
from . import bshi_data as D
from . import bshi_store as ST
from . import bshi_train as T
from .bshi_model import ModelConfig

SWEEP_REL = Path("data") / "derived" / "bshi" / "sweep_subset"


def load_cfg(root, **over):
    train = json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8")).get("train", {})
    return T.TrainConfig.from_dict(train, **over)


def model_cfg(cfg):
    return ModelConfig(variant=cfg.variant, gate=cfg.gate, dropout=cfg.dropout)


def data_summary(root, cfg):
    """Counts per split and dataset for the experiment (full windows), the sampler and the expected epoch time."""
    idx = D.protocol_frame(root, cfg.windows)
    fr = D.frames_for(idx, cfg.train_on)
    rows = []
    for name, f in [("train", fr["train"]), ("validation", fr["val"])] + list(fr["tests"].items()):
        rows.append({"set": name, "windows": len(f), "events": int(f.y_event.sum()), "prevalence_%": round(100 * float(f.y_event.mean()), 2) if len(f) else float("nan"), "patients": f.patient_key.nunique(),
                     "mover_share_%": round(100 * float((f.dataset == "mover").mean()), 1) if len(f) else float("nan")})
    out = pd.DataFrame(rows)
    gpu_rate = 2500.0
    return out, {"steps_per_epoch": math.ceil(len(fr["train"]) / cfg.batch_size), "expected_epoch_minutes_at_2500_windows_per_s": round(len(fr["train"]) / gpu_rate / 60, 1)}


def make_data(root, cfg, smoke=False, epoch_windows=None, device="cuda", use_store=True):
    """Everything ``Trainer`` needs: preprocessor, train / validation loaders, steps per epoch, the prior shift (stratified sampler) and bookkeeping.
    FULL mode reads the compact store of the experiment (``data/derived/bshi/store_<train_on>_v1``; built by ``bshi_store.build_experiment_store``) when it exists: shuffled training from the
    on-the-fly assembler reads a different surgery's beat file for almost every window and runs at about 60-150 windows/s (measured, the analysis workflow), the store at the GPU's speed."""
    # Choose the correct store and partition before creating training and evaluation loaders.
    root = Path(root)
    stats = D.load_stats(root)
    pre = D.Preprocessor(cfg.variant, stats, clamp=cfg.clamp, device=device, channels_last=cfg.channels_last)
    if smoke:
        tr, va = ST.Store(root / SWEEP_REL, "train"), ST.Store(root / SWEEP_REL, "validation")
        ew = int(epoch_windows or 4096)
        shift = D.prior_shift_by_dataset(D.store_frame(tr), cfg.event_share) if cfg.sampler == "stratified" else None
        return {"pre": pre, "train_loader": D.store_train_loader(tr, cfg.batch_size, ew, cfg.seed, cfg.sampler, cfg.mover_share, cfg.event_share), "val_loader": D.store_eval_loader(va, 512),
                "steps_per_epoch": math.ceil(ew / cfg.batch_size), "shift": shift, "source": "sweep subset (compact store)", "n_train": len(tr), "n_val": len(va), "meta": {"val": va.meta}}
    idx = D.protocol_frame(root, cfg.windows)
    fr = D.frames_for(idx, cfg.train_on)
    flags = D.WindowFlags(idx)
    spath = ST.store_path(root, cfg.train_on)
    if use_store and cfg.windows == "eap_full" and (spath / "MANIFEST.sha256").exists():
        tr, va = ST.Store(spath, "train"), ST.Store(spath, "validation")
        te = ST.Store(spath, "test") if (spath / "test").exists() else None
        ew = int(epoch_windows or len(tr))
        shift = D.prior_shift_by_dataset(D.store_frame(tr), cfg.event_share) if cfg.sampler == "stratified" else None
        return {"pre": pre, "train_loader": D.store_train_loader(tr, cfg.batch_size, ew, cfg.seed, cfg.sampler, cfg.mover_share, cfg.event_share), "val_loader": D.store_eval_loader(va, 512),
                "steps_per_epoch": math.ceil(ew / cfg.batch_size), "shift": shift, "source": f"compact store {spath.name}", "n_train": len(tr), "n_val": len(va),
                "stores": {"train": tr, "val": va, "test": te}, "frames": fr, "flags": flags, "index": idx}
    train = fr["train"]
    shift = None
    if cfg.sampler == "stratified":                                   # one resampled index per run (a new one per epoch would restart the persistent workers)
        shift = D.prior_shift_by_dataset(train, cfg.event_share)
        train = D.resample_index(train, int(epoch_windows or len(train)), cfg.mover_share, cfg.event_share, cfg.seed)
    tl = D.make_loader(root, train, cfg.variant, cfg.batch_size, True, cfg.seed, cfg.num_workers, flags)
    vl = D.make_loader(root, fr["val"], cfg.variant, 512, False, 0, max(2, cfg.num_workers // 2), flags)
    return {"pre": pre, "train_loader": lambda epoch: tl, "val_loader": lambda: vl, "steps_per_epoch": math.ceil(len(train) / cfg.batch_size), "shift": shift,
            "source": "on-the-fly assembler (real windows; SLOW for training, build the compact store)", "n_train": len(train), "n_val": len(fr["val"]), "frames": fr, "flags": flags, "index": idx}


def build_trainer(root, cfg, out_dir, smoke=False, epoch_windows=None, device="cuda", log=print, progress=True, label="run", progress_kw=None, use_store=True):
    """Trainer + data. ``progress`` attaches a live ``bshi_progress.Progress`` writing ``<out_dir>/live/progress.{json,jsonl,png,html}`` (see that module for how to watch it)."""
    d = make_data(root, cfg, smoke, epoch_windows, device, use_store)
    tr = T.Trainer(cfg, model_cfg(cfg), d["pre"], d["train_loader"], d["val_loader"], out_dir, device=device, steps_per_epoch=d["steps_per_epoch"], shift=d["shift"], log=log)
    tr.val_batches = math.ceil(d["n_val"] / 512)                      # for the validation percentage in the progress display
    if progress:
        from . import bshi_progress as P
        tr.progress = P.Progress(Path(out_dir) / "live", label=label, **(progress_kw or {}))
    return tr, d


def evaluate_loader(trainer, pre, loader, cfg, patient_key=None, reps=200, shift=None):
    df = T.predict(trainer.model, pre, loader, cfg, trainer.device)
    m = T.metrics_from_predictions(df, shift)
    if patient_key is not None:
        pk = pd.Series(patient_key).reindex(df.window_id).to_numpy()
        m["ci"] = T.bootstrap_ci(df, pk, reps=reps, shift=shift)
    return m, df


def evaluate_frame(root, trainer, pre, cfg, frame, flags, shift=None, reps=200, workers=8):
    """Metrics of the trained model on the windows of ``frame`` (real windows through the on-the-fly loader), with patient-bootstrap CIs."""
    loader = D.make_loader(root, frame, cfg.variant, 512, False, 0, workers, flags)
    return evaluate_loader(trainer, pre, loader, cfg, frame.set_index("window_id").patient_key, reps, shift)


def baselines_on(root, frames, rungs=("G0", "G4")):
    """Logistic-regression ladder rungs trained on the POOLED train windows and evaluated on the given frames {name: frame}: the numbers to compare the network with, on the same windows."""
    # Calculate the required feature columns for the following step.
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from . import eda_models as M
    from . import labels as L
    cols = {r: M.RUNGS[r] for r in rungs}
    need = sorted({c for v in cols.values() for c in v})
    in_index = [c for c in need if c in L.FEATURES]                  # the MAP features live in the window index, the others in window_features_v1
    base = S.load_window_index(root, columns=["window_id"] + in_index)
    feats = pd.read_parquet(Path(root) / S.OUT_DIR / M.FEATURES_NAME, columns=["window_id"] + [c for c in need if c not in in_index])
    df = base.merge(feats, on="window_id").dropna(subset=need).set_index("window_id")
    idx = D.protocol_frame(root, "eap_full")
    train = idx[idx.split_pooled == "train"]
    train = train[train.window_id.isin(df.index)]
    out = {}
    for r, c in cols.items():
        sc = StandardScaler().fit(df.loc[train.window_id, c])
        clf = LogisticRegression(max_iter=500).fit(sc.transform(df.loc[train.window_id, c]), train.y_event.astype(int))
        for name, f in frames.items():
            # Calculate predicted risk without refitting on evaluation windows.
            f = f[f.window_id.isin(df.index)]
            p = clf.predict_proba(sc.transform(df.loc[f.window_id, c]))[:, 1]
            out[(r, name)] = {"auroc": L.auroc(f.y_event.to_numpy(), p), "auprc": L.auprc(f.y_event.to_numpy(), p), "n": int(len(f))}
    return out


# evaluation
def gray_zone_table(df, gray):
    """AUROC inside the 65-75 mmHg gray zone and outside it. ``gray``: bool array aligned with ``df``."""
    from . import labels as L
    rows = []
    for name, m in (("gray zone 65-75 mmHg", gray), ("outside the gray zone", ~gray)):
        y = df.y.to_numpy()[m] > 0.5
        rows.append({"windows": name, "n": int(m.sum()), "events": int(y.sum()), "auroc": L.auroc(y, df.logit.to_numpy()[m]) if y.any() and (~y).any() else float("nan")})
    return pd.DataFrame(rows)


def baseline_table(root, setting, set_name="full"):
    """G0 and G4 AUROC / AUPRC of the project's baseline ladder (``results/eda/table_baseline_ladder_all.csv``) for a setting, e.g. 'pooled_train->pooled_test'."""
    # Read the saved table and retain its named columns.
    t = pd.read_csv(Path(root) / "results" / "eda" / "table_baseline_ladder_all.csv")
    t = t[(t.set == set_name) & (t.setting == setting) & (t.rung.isin(["G0", "G4"]))]
    return t[["rung", "learner", "auroc", "auroc_lo", "auroc_hi"]].rename(columns={"learner": "learner (lr = logistic, hgb = boosting)"}).reset_index(drop=True)


BASELINE_SETTING = {("pooled", "pooled_test"): "pooled_test", ("pooled", "vitaldb_test"): "vitaldb_test_only", ("pooled", "mover_test"): "mover_test_only",
                    ("vitaldb", "vitaldb_test"): "vitaldb_train->vitaldb_test", ("vitaldb", "mover_all"): "E3x vitaldb_train->all_mover",
                    ("mover", "vitaldb_test"): "reverse mover_train->vitaldb_test", ("pooled", "dev_test"): "pooled_test"}


def compare_with_baselines(root, cfg, res):
    """One row per evaluated set: the network's AUROC (with CI) next to the G0 (MAP only) and G4 (all features) logistic / boosting AUROCs of ``results/eda`` (same full-set windows; the dev-subset
    smoke test is only a rough context)."""
    # Read the saved table and retain its named columns.
    t = pd.read_csv(Path(root) / "results" / "eda" / "table_baseline_ladder_all.csv")
    t = t[t.set == "full"]
    rows = []
    # Apply the same operation to each named field in this group.
    for name, m in res.items():
        key = BASELINE_SETTING.get((cfg.train_on, name))
        if name.startswith("_") or key is None or name == "validation":
            continue
        row = {"set": name, "baseline setting": key, "network AUROC": round(m["auroc"], 4)}
        for rung in ("G0", "G4"):
            for learner in ("lr", "hgb"):
                v = t[(t.setting == key) & (t.rung == rung) & (t.learner == learner)].auroc
                row[f"{rung} {learner}"] = round(float(v.iloc[0]), 4) if len(v) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_run(root, trainer, data, cfg, smoke, reps=200):
    """Validation and TEST metrics with patient-bootstrap CIs plus the calibration / lead-time / gray-zone inputs. SMOKE: validation from the sweep subset, test from the dev subset's test split.
    FULL: the real validation windows and every test frame of the experiment. Returns (results dict, predictions per set)."""
    # Load the saved run and evaluate its fixed model on the requested partitions.
    pre, shift = data["pre"], data["shift"]
    res, preds = {}, {}
    if smoke:
        # Read the stored table required for this calculation.
        res["validation"], preds["validation"] = evaluate_loader(trainer, pre, data["val_loader"](), cfg, data["meta"]["val"].set_index("window_id").patient_key, reps, shift)
        files = sorted((Path(root) / D.DEV_DIR / "split=test").glob("part-*.parquet"))
        meta = pd.concat([pd.read_parquet(f, columns=["window_id", "patient_key", "time_to_event_sec", "gray_zone"]) for f in files]).set_index("window_id")
        flags = D.WindowFlags(D.protocol_frame(root, "eap_full"))
        res["dev_test"], preds["dev_test"] = evaluate_loader(trainer, pre, D.dev_loader(root, "test", cfg.variant, 512, False, 0, 0, flags), cfg, meta.patient_key, reps, shift)
        extra = {"dev_test": meta}
    else:
        fr, flags = data["frames"], data["flags"]
        stores = data.get("stores")

        def positions(store, frame):
            """Positions of the frame's windows in a store, or None when the store does not hold all of them (then the on-the-fly loader is used)."""
            if store is None:
                return None
            p = pd.Series(np.arange(len(store)), index=store.meta.window_id.to_numpy()).reindex(frame.window_id.to_numpy())
            return None if p.isna().any() else p.to_numpy().astype(np.int64)

        def evaluate_any(store, frame):
            pos = positions(store, frame)
            if pos is None:
                return evaluate_frame(root, trainer, pre, cfg, frame, flags, shift, reps)
            return evaluate_loader(trainer, pre, D.store_eval_loader(store, 512, pos)(), cfg, frame.set_index("window_id").patient_key, reps, shift)
        res["validation"], preds["validation"] = evaluate_any(stores["val"] if stores else None, fr["val"])
        extra = {"validation": fr["val"].set_index("window_id")}
        for name, f in fr["tests"].items():
            res[name], preds[name] = evaluate_any(stores["test"] if stores else None, f)
            extra[name] = f.set_index("window_id")
    res["_extra"] = extra
    return res, preds


def metrics_table(res):
    rows = []
    for name, m in res.items():
        if name.startswith("_"):
            continue
        ci = m.get("ci", {})
        rows.append({"set": name, "windows": m["n"], "events": m["events"], "AUROC": round(m["auroc"], 4), "AUROC 95% CI": "-" if "auroc" not in ci else f"{ci['auroc'][0]:.3f}-{ci['auroc'][1]:.3f}",
                     "AUPRC": round(m["auprc"], 4), "Brier": round(m["brier"], 4), "calibration slope": round(m["cal_slope"], 3), "MAE delta MAP (mmHg)": round(m.get("mae_delta_mmhg", float("nan")), 2),
                     "MAE AUT65 (mmHg*min)": round(m.get("mae_aut65", float("nan")), 3)})
    return pd.DataFrame(rows)
def plot_history(history, path=None):
    # Collect the per-record values into a table for comparison.
    import matplotlib.pyplot as plt
    from . import figstyle as F
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    ax[0].plot(h.epoch, h.total, marker="o", label="training loss")
    if "val_loss" in h:
        ax[0].plot(h.epoch, h.val_loss, marker="s", label="validation loss")
    # Label the panel with the measurement names and their units.
    ax[0].legend(fontsize=8)
    F.finish_axes(ax[0], "epoch", "multi-task loss (dimensionless)", "loss")
    ax[1].plot(h.epoch, h.val_auprc, marker="o", color="#2ca02c", label="AUPRC")
    ax[1].plot(h.epoch, h.val_auroc, marker="s", color="#1f77b4", label="AUROC")
    # Label the panel with the measurement names and their units.
    ax[1].legend(fontsize=8)
    F.finish_axes(ax[1], "epoch", "validation metric (0-1, natural prevalence)", "validation")
    ax[2].plot(h.epoch, h.train_windows_per_s, marker="o", color="#ff7f0e")
    # Label the panel with the measurement names and their units.
    F.finish_axes(ax[2], "epoch", "training speed (windows/s)", "speed")
    fig.tight_layout()
    F.audit_figure(fig, "fig_training_history")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=90)
    return fig


def plot_calibration(df, shift=None, bins=10, path=None):
    import matplotlib.pyplot as plt
    from . import figstyle as F
    logit = df.logit.to_numpy()
    if shift:
        logit = logit + np.where(df.dataset_code.to_numpy() == 1, shift.get("vitaldb", 0.0), shift.get("mover", 0.0))
    # Summarize the records separately within each requested group.
    p = 1 / (1 + np.exp(-logit))
    q = pd.qcut(p, bins, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": df.y.to_numpy(), "q": q}).groupby("q", observed=True).agg(p=("p", "mean"), y=("y", "mean"))
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.plot(g.p, g.y, marker="o")
    ax.plot([0, g.p.max()], [0, g.p.max()], ls="--", color="grey")
    F.finish_axes(ax, "mean predicted event probability (decile bins)", "observed event rate (fraction of windows)", "calibration")
    fig.tight_layout()
    F.audit_figure(fig, "fig_calibration")
    if path:
        fig.savefig(path, dpi=90)
    return fig


def auroc_by_lead_time(df, tte, edges=(0, 60, 120, 180, 240)):
    """AUROC of the event score against non-events for events whose onset is within each lead-time bin. ``tte``: Series time_to_event_sec indexed by window_id."""
    from . import labels as L
    t = tte.reindex(df.window_id).to_numpy()
    y = df.y.to_numpy() > 0.5
    rows = []
    for a, b in zip(edges[:-1], edges[1:]):
        pos = y & (t > a) & (t <= b)
        keep = pos | ~y
        rows.append({"lead_time_bin_s": f"{a}-{b}", "events": int(pos.sum()), "auroc": L.auroc(pos[keep], df.logit.to_numpy()[keep]) if pos.any() else float("nan")})
    return pd.DataFrame(rows)


def gpu_monitor(stop_event, samples, period=2.0):
    """Background sampler of nvidia-smi (GPU utilization %, memory used MB) and CPU percent, appended to ``samples``."""
    import subprocess

    import psutil
    psutil.cpu_percent(None)
    while not stop_event.is_set():
        try:
            u, m = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(",")
            samples.append({"t": time.time(), "gpu_util": float(u), "vram_mb": float(m), "cpu_util": psutil.cpu_percent(None)})
        except Exception:
            pass
        stop_event.wait(period)
