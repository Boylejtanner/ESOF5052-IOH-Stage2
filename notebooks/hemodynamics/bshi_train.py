"""Trainer for the BSHI network.

AdamW or SGD-Nesterov behind one config; linear warmup + cosine decay or one-cycle; bf16 autocast (no GradScaler), channels_last, cudnn.benchmark; gradient clipping; the multi-task loss with
EXPLICIT masks (event on every window; delta MAP and AUT65 only where the target is observed; the AUT65 magnitude only where AUT65 > 0); early stopping on validation AUPRC with restore-best;
optional EMA of the weights; checkpoints with the optimizer, the step, the early-stopping state and the RNG states, so a resumed run continues exactly; a CSV log per epoch.
The trainer only sees TRAIN data for fitting and VALIDATION data for selection: the test frames never enter this module.
"""
from __future__ import annotations
# Train the multi-task network and retain the best validation checkpoint.

import copy
import math
import random
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import labels as L
from .bshi_model import BshiNet, ModelConfig


@dataclass
class TrainConfig:
    """Collect optimizer, batching, loss, and stopping settings for a reproducible run."""
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.01
    beta2: float = 0.999
    momentum: float = 0.9
    batch_size: int = 256
    epochs: int = 30
    patience: int = 5
    min_delta: float = 5e-4
    seed: int = 5052
    num_workers: int = 16
    amp: str = "bf16"
    variant: str = "eap"
    train_on: str = "pooled"
    schedule: str = "cosine"
    warmup_epochs: float = 1.0
    min_lr_frac: float = 0.01
    grad_clip: float = 1.0
    loss_weights: dict = field(default_factory=lambda: {"event": 1.0, "delta_map": 0.3, "aut65": 0.3})
    clamp: float = 6.0
    pos_weight: float | None = None
    sampler: str = "natural"
    mover_share: float = 0.15
    event_share: float = 0.30
    ema: float = 0.0
    gate: bool = True
    dropout: float = 0.1
    channels_last: bool = False                    # measured: 7% slower than channels_last=False with GroupNorm (results/train/gpu_throughput.json)
    cudnn_benchmark: bool = True
    deterministic: bool = False
    windows: str = "eap_full"
    max_steps_per_epoch: int | None = None

    @classmethod
    def from_dict(cls, d=None, **over):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in {**(d or {}), **over}.items() if k in known})

    def to_dict(self):
        return asdict(self)


def set_seed(seed, deterministic=False):
    """Seed Python, NumPy, and PyTorch, with optional deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# optimizer, schedule, early stopping, EMA
def build_optimizer(model, cfg, device):
    """Weight decay only on matrices and kernels (not on biases and GroupNorm parameters)."""
    # Apply weight decay to learned matrices and kernels, not biases or normalization terms.
    decay = [p for p in model.parameters() if p.ndim > 1]
    plain = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": plain, "weight_decay": 0.0}]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(0.9, cfg.beta2), eps=1e-8, fused=torch.device(device).type == "cuda")
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, nesterov=True)
    raise ValueError(f"optimizer must be adamw or sgd, not {cfg.optimizer!r}")


def lr_at(cfg, step, total, steps_per_epoch):
    """Learning rate of optimizer step ``step`` (0-based) out of ``total``: warmup + cosine, one-cycle, or constant."""
    # Express the schedule in optimizer steps so warm-up and decay follow the same clock.
    total = max(1, total)
    if cfg.schedule == "constant":
        return cfg.lr
    if cfg.schedule == "onecycle":
        up = max(1, int(0.3 * total))
        start, end = cfg.lr / 25.0, cfg.lr / 1e4
        if step < up:
            return start + (cfg.lr - start) * 0.5 * (1 - math.cos(math.pi * step / up))
        t = min(1.0, (step - up) / max(1, total - up))
        return end + (cfg.lr - end) * 0.5 * (1 + math.cos(math.pi * t))
    warm = int(round(cfg.warmup_epochs * steps_per_epoch))
    if step < warm:
        return cfg.lr * (step + 1) / max(1, warm)
    t = min(1.0, (step - warm) / max(1, total - warm))
    lo = cfg.lr * cfg.min_lr_frac
    return lo + (cfg.lr - lo) * 0.5 * (1 + math.cos(math.pi * t))


class EarlyStopper:
    """Maximize a validation metric: improved when it beats the best by more than ``min_delta``; stop after ``patience`` epochs without improvement. NaN never improves."""

    def __init__(self, patience, min_delta):
        self.patience, self.min_delta, self.best, self.best_epoch, self.bad = patience, min_delta, None, -1, 0

    def update(self, value, epoch):
        ok = value == value and (self.best is None or value > self.best + self.min_delta)
        if ok:
            self.best, self.best_epoch, self.bad = float(value), epoch, 0
        else:
            self.bad += 1
        return ok, self.bad >= self.patience

    def state(self):
        return {"best": self.best, "best_epoch": self.best_epoch, "bad": self.bad}

    def load(self, s):
        self.best, self.best_epoch, self.bad = s["best"], s["best_epoch"], s["bad"]


class EMA:
    """Maintain an exponential moving average of model weights for optional evaluation."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def swap_in(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items() if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)

    def swap_out(self, model):
        model.load_state_dict(self.backup, strict=False)


# loss
def masked_mean(values, mask):
    """Average only observed target values while handling an empty mask safely."""
    m = mask.float()
    return (values * m).sum() / m.sum().clamp(min=1.0)


def multitask_loss(out, t, weights, pos_weight=None):
    """Total loss and its parts. ``t`` is the Preprocessor's target dict. Event: every window. Delta MAP: Huber on the train-standardized target where observed. AUT65: BCE for
    AUT65 > 0 where observed + Huber on the standardized log1p magnitude ONLY where AUT65 > 0."""
    # Mask unavailable regression targets while retaining every valid event label.
    pw = None if pos_weight is None else torch.tensor(float(pos_weight), device=out["event"].device)
    l_event = F.binary_cross_entropy_with_logits(out["event"], t["y_event"], pos_weight=pw)
    l_delta = masked_mean(F.huber_loss(out["delta_map"], t["delta"], delta=1.0, reduction="none"), t["m_delta"])
    l_pos = masked_mean(F.binary_cross_entropy_with_logits(out["aut_logit"], t["aut_pos"], reduction="none"), t["m_aut"])
    l_mag = masked_mean(F.huber_loss(out["aut_mag"], t["aut_mag"], delta=1.0, reduction="none"), t["m_mag"])
    total = weights["event"] * l_event + weights["delta_map"] * l_delta + weights["aut65"] * (l_pos + l_mag)
    return total, {"event": l_event.detach(), "delta_map": l_delta.detach(), "aut_pos": l_pos.detach(), "aut_mag": l_mag.detach()}


def aut65_prediction(out, pre):
    """Expected AUT65 (mmHg*min) from the two parts: P(AUT65 > 0) * expm1(magnitude)."""
    return torch.sigmoid(out["aut_logit"]) * torch.expm1(out["aut_mag"] * pre.aut_std + pre.aut_mean).clamp(min=0)


# prediction and metrics
@torch.no_grad()
def predict(model, pre, loader, cfg, device, weights=None, max_batches=None, on_batch=None):
    """Predictions of every window of ``loader`` (a DataFrame: window_id, y, logit, delta_pred/true, aut_pred/true, masks, dataset_code, w) and the mean validation loss. ``on_batch(i)`` is a progress hook."""
    model.eval()
    dev = torch.device(device)
    cols, loss_sum, n = {k: [] for k in ("window_id", "y", "logit", "delta_pred", "delta_true", "m_delta", "aut_pred", "aut_true", "m_aut", "dataset_code", "w")}, 0.0, 0
    # Process the selected records in order and retain their individual results.
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        if on_batch is not None:
            on_batch(i)
        b = pre(batch)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=cfg.amp == "bf16"):
            out = model(b["x"], b["context"], b["beat_mask"])
        out = {k: v.float() for k, v in out.items()}
        total, _ = multitask_loss(out, b, weights or cfg.loss_weights, cfg.pos_weight)
        loss_sum += float(total) * len(b["y_event"])
        n += len(b["y_event"])
        aut_true = torch.expm1(b["aut_mag"] * pre.aut_std + pre.aut_mean) * b["aut_pos"]
        for k, v in (("window_id", b["window_id"]), ("y", b["y_event"]), ("logit", out["event"]), ("delta_pred", out["delta_map"] * pre.delta_scale), ("delta_true", b["delta"] * pre.delta_scale),
                     ("m_delta", b["m_delta"]), ("aut_pred", aut65_prediction(out, pre)), ("aut_true", aut_true), ("m_aut", b["m_aut"])):
            cols[k].append(v.detach().cpu().numpy())
        cols["dataset_code"].append(b["dataset_code"].cpu().numpy() if b["dataset_code"] is not None else np.zeros(len(b["y_event"]), dtype=np.int8))
        cols["w"].append(b["w"].cpu().numpy() if b["w"] is not None else np.ones(len(b["y_event"]), dtype=np.float32))
    df = pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})
    df.attrs["loss"] = loss_sum / max(1, n)
    return df


def metrics_from_predictions(df, shift=None):
    """AUROC, AUPRC (weighted by ``w``: natural prevalence for sampled sets), Brier, calibration slope/intercept, MAE of delta MAP and AUT65 against the 'predict the train median' reference.
    ``shift``: per-dataset logit offsets (``bshi_data.prior_shift_by_dataset``) when training used a stratified sampler. AUROC is not affected by it."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    logit = df.logit.to_numpy(dtype=np.float64)
    if shift:
        logit = logit + np.where(df.dataset_code.to_numpy() == 1, shift.get("vitaldb", 0.0), shift.get("mover", 0.0))
    p = 1.0 / (1.0 + np.exp(-logit))
    y = df.y.to_numpy() > 0.5
    w = df.w.to_numpy(dtype=np.float64)
    out = {"n": int(len(df)), "events": int(y.sum()), "loss": float(df.attrs.get("loss", float("nan")))}
    if y.any() and (~y).any():
        # Calculate prediction metrics from these labels and model scores.
        out["auroc"] = float(roc_auc_score(y, p, sample_weight=w))
        out["auprc"] = float(average_precision_score(y, p, sample_weight=w))
        slope, intercept = L._calibration(y, p)
        out.update(cal_slope=slope, cal_intercept=intercept)
    else:
        out.update(auroc=float("nan"), auprc=float("nan"), cal_slope=float("nan"), cal_intercept=float("nan"))
    out["brier"] = float(np.average((p - y) ** 2, weights=w))
    md, ma = df.m_delta.to_numpy(bool), df.m_aut.to_numpy(bool)
    if md.any():
        out["mae_delta_mmhg"] = float(np.average(np.abs(df.delta_pred[md] - df.delta_true[md]), weights=w[md]))
        out["mae_delta_ref"] = float(np.average(np.abs(df.delta_true[md] - np.median(df.delta_true[md])), weights=w[md]))
    if ma.any():
        out["mae_aut65"] = float(np.average(np.abs(df.aut_pred[ma] - df.aut_true[ma]), weights=w[ma]))
        out["mae_aut65_ref"] = float(np.average(np.abs(df.aut_true[ma] - np.median(df.aut_true[ma])), weights=w[ma]))
    return out


def bootstrap_ci(df, patient_key, reps=200, seed=5052, shift=None):
    """Patient-level bootstrap CIs (the project's ``labels._cluster_ci``) of AUROC / AUPRC / Brier. ``patient_key``: Series indexed like df (one entry per window)."""
    logit = df.logit.to_numpy(dtype=np.float64)
    if shift:
        logit = logit + np.where(df.dataset_code.to_numpy() == 1, shift.get("vitaldb", 0.0), shift.get("mover", 0.0))
    frame = pd.DataFrame({"y": (df.y.to_numpy() > 0.5).astype(int), "p": 1.0 / (1.0 + np.exp(-logit)), "patient_key": np.asarray(patient_key)})
    return L._cluster_ci(frame, "y", "p", reps, seed)


# trainer
class Trainer:
    """Train and validate BSHI with checkpointing, early stopping, and epoch metrics."""
    def __init__(self, cfg: TrainConfig, model_cfg: ModelConfig, pre, train_loader, val_loader, out_dir, device="cuda", steps_per_epoch=None, shift=None, log=print):
        # Create the destination directory before writing any outputs.
        self.cfg, self.model_cfg, self.pre, self.train_loader, self.val_loader = cfg, model_cfg, pre, train_loader, val_loader
        self.out_dir, self.device, self.shift, self.log = Path(out_dir), torch.device(device), shift, log
        self.out_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed, cfg.deterministic)
        torch.backends.cudnn.benchmark = bool(cfg.cudnn_benchmark and self.device.type == "cuda" and not cfg.deterministic)
        self.model = BshiNet(model_cfg).to(self.device)
        if cfg.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        self.opt = build_optimizer(self.model, cfg, self.device)
        self.ema = EMA(self.model, cfg.ema) if cfg.ema > 0 else None
        self.stopper = EarlyStopper(cfg.patience, cfg.min_delta)
        self.steps_per_epoch = int(steps_per_epoch or cfg.max_steps_per_epoch or 100)
        if cfg.max_steps_per_epoch:
            self.steps_per_epoch = min(self.steps_per_epoch, cfg.max_steps_per_epoch)
        self.step, self.epoch0, self.history, self.best_state = 0, 0, [], None
        self.progress = None                                          # a bshi_progress.Progress; set by the caller (bshi_run.build_trainer does)

    # -- checkpoints
    def _rng(self):
        return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None, "numpy": np.random.get_state(), "python": random.getstate()}

    def save_checkpoint(self, name="last.pt"):
        # Retain optimizer and random-generator state as well as the model weights.
        state = {"model": self.model.state_dict(), "opt": self.opt.state_dict(), "ema": None if self.ema is None else self.ema.shadow, "epoch": self.epoch0, "step": self.step,
                 "stopper": self.stopper.state(), "best_state": self.best_state, "history": self.history, "cfg": self.cfg.to_dict(), "model_cfg": self.model_cfg.to_dict(), "rng": self._rng()}
        tmp = self.out_dir / (name + ".tmp")
        torch.save(state, tmp)
        tmp.replace(self.out_dir / name)

    def load_checkpoint(self, path=None):
        # Restore the same run state rather than restarting the optimizer from scratch.
        state = torch.load(path or self.out_dir / "last.pt", map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"])
        self.opt.load_state_dict(state["opt"])
        if self.ema is not None and state["ema"] is not None:
            self.ema.shadow = {k: v.to(self.device) for k, v in state["ema"].items()}
        self.epoch0, self.step, self.history, self.best_state = state["epoch"], state["step"], state["history"], state["best_state"]
        self.stopper.load(state["stopper"])
        r = state["rng"]
        torch.set_rng_state(r["torch"])
        if r["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(r["cuda"])
        np.random.set_state(r["numpy"])
        random.setstate(r["python"])
        return state

    # -- one epoch / validation
    def _train_epoch(self, epoch):
        cfg, total = self.cfg, self.cfg.epochs * self.steps_per_epoch
        self.model.train()
        # Reserve output arrays and masks with explicit initial values.
        it = self.train_loader(epoch) if callable(self.train_loader) else self.train_loader
        sums, n_win, steps = {k: torch.zeros((), device=self.device) for k in ("total", "event", "delta_map", "aut_pos", "aut_mag")}, 0, 0
        gnorm = torch.zeros((), device=self.device)
        lr = float("nan")
        t0 = time.perf_counter()
        prog = self.progress
        if prog is not None:
            prog.begin_epoch(epoch)
        # Process the selected records in order and retain their individual results.
        for i, batch in enumerate(it):
            if i >= self.steps_per_epoch:
                break
            lr = lr_at(cfg, self.step, total, self.steps_per_epoch)
            for g in self.opt.param_groups:
                g["lr"] = lr
            b = self.pre(batch)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=cfg.amp == "bf16"):
                out = self.model(b["x"], b["context"], b["beat_mask"])
            out = {k: v.float() for k, v in out.items()}
            loss, parts = multitask_loss(out, b, cfg.loss_weights, cfg.pos_weight)
            # Backpropagate the combined loss and limit large gradient updates before stepping.
            loss.backward()
            gnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.ema is not None:
                self.ema.update(self.model)
            sums["total"] += loss.detach()
            for k, v in parts.items():
                sums[k] += v
            n_win += len(b["y_event"])
            steps += 1
            self.step += 1
            if prog is not None and prog.due():
                prog.update(steps, n_win, {k: float(v) / steps for k, v in sums.items()}, lr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        # Convert accumulated loss sums into epoch averages and record processing speed.
        row = {k: float(v) / max(1, steps) for k, v in sums.items()}
        row.update(lr=lr, grad_norm=float(gnorm) / max(1, steps), steps=steps, windows=n_win, train_seconds=round(secs, 2), train_windows_per_s=round(n_win / max(secs, 1e-9), 1))
        return row

    def validate(self):
        # Evaluate validation batches, temporarily using averaged weights when EMA is enabled.
        if self.ema is not None:
            self.ema.swap_in(self.model)
        try:
            vl = self.val_loader() if callable(self.val_loader) else self.val_loader
            prog = self.progress
            if prog is not None:
                prog.begin_validation(getattr(self, "val_batches", None) or (len(vl) if hasattr(vl, "__len__") else None))
            df = predict(self.model, self.pre, vl, self.cfg, self.device, on_batch=None if prog is None else prog.validation_batch)
        finally:
            if self.ema is not None:
                self.ema.swap_out(self.model)
        return metrics_from_predictions(df, self.shift)

    def fit(self, resume=True, until_epoch=None):
        """Train until ``cfg.epochs`` (or ``until_epoch``, to stop early without changing the schedule), early stopping, or the end; returns the history and restores the best weights."""
        # Resume a saved run when requested and select the best epoch using validation AUPRC.
        cfg = self.cfg
        if resume and (self.out_dir / "last.pt").exists():
            self.load_checkpoint()
            self.log(f"resumed from epoch {self.epoch0}")
        stopped, t_all = False, time.perf_counter()
        prog = self.progress
        if prog is not None:
            prog.start_run(cfg.epochs, self.steps_per_epoch, cfg.batch_size, self.epoch0, {"optimizer": cfg.optimizer, "lr": cfg.lr, "batch_size": cfg.batch_size, "variant": cfg.variant})
        # Repeat training and validation for each epoch until the stopping rule is met.
        for epoch in range(self.epoch0, min(cfg.epochs, until_epoch or cfg.epochs)):
            row = {"epoch": epoch, **self._train_epoch(epoch)}
            val = self.validate()
            row.update({"val_" + k: v for k, v in val.items()})
            # Choose checkpoints using validation AUPRC, not training loss or test outcomes.
            improved, stop = self.stopper.update(val.get("auprc", float("nan")), epoch)
            if improved:
                state = self.model.state_dict() if self.ema is None else {**self.model.state_dict(), **self.ema.shadow}
                self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
                torch.save({"model": self.best_state, "epoch": epoch, "metric": self.stopper.best, "model_cfg": self.model_cfg.to_dict(), "cfg": cfg.to_dict()}, self.out_dir / "best.pt")
            row["improved"], row["best_auprc"] = bool(improved), self.stopper.best
            self.history.append(row)
            if prog is not None:
                prog.epoch_done(row)
            self.epoch0 = epoch + 1
            pd.DataFrame(self.history).to_csv(self.out_dir / "epochs.csv", index=False)
            self.save_checkpoint()
            self.log(f"epoch {epoch}: loss {row['total']:.4f} | val AUPRC {val.get('auprc', float('nan')):.4f} AUROC {val.get('auroc', float('nan')):.4f} | {row['train_windows_per_s']:.0f} windows/s")
            if stop:
                stopped = True
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        if prog is not None:
            prog.finish({"best_epoch": self.stopper.best_epoch, "best_auprc": self.stopper.best, "stopped_early": stopped})
        return {"history": pd.DataFrame(self.history), "best_epoch": self.stopper.best_epoch, "best_auprc": self.stopper.best, "stopped_early": stopped,
                "seconds": round(time.perf_counter() - t_all, 1)}


# M1: step throughput and memory on synthetic batches
def synthetic_batch(n, device, seed=0):
    """Construct a synthetic batch for checking the training interface without patient data."""
    # Seed the generated values so this check can be repeated.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 7, 60, 200), generator=g)
    b = {"x": x, "context": torch.randn((n, 25), generator=g), "beat_mask": torch.rand((n, 3, 60), generator=g) > 0.1, "window_id": torch.arange(n),
         "y_event": (torch.rand(n, generator=g) < 0.1).float(), "delta_map": torch.rand(n, generator=g) * 20, "aut65": torch.rand(n, generator=g) * (torch.rand(n, generator=g) < 0.2), "dataset_code": torch.zeros(n, dtype=torch.int8)}
    return b


def benchmark_step(model_cfg, batch_size, bf16=True, channels_last=True, device="cuda", iters=12, warmup=4):
    """Training-step windows/s and peak VRAM for one configuration on a synthetic batch; ``{'oom': True}`` when it does not fit."""
    # Measure a complete forward/backward update after GPU warm-up.
    dev = torch.device(device)
    try:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        model = BshiNet(model_cfg).to(dev)
        if channels_last:
            model = model.to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        x = torch.randn((batch_size, 7, 60, 200), device=dev)
        if channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        ctx = torch.randn((batch_size, 25), device=dev)
        bm = torch.ones((batch_size, 3, 60), dtype=torch.bool, device=dev)
        # Accumulate the calculation over the required samples or iterations.
        for i in range(warmup + iters):
            if i == warmup:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
                out = model(x, ctx, bm)
            loss = sum(v.float().mean() for v in out.values())
            # Backpropagate the combined loss and limit large gradient updates before stepping.
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        secs = time.perf_counter() - t0
        return {"batch_size": batch_size, "bf16": bf16, "channels_last": channels_last, "windows_per_s": round(batch_size * iters / secs, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2), "oom": False}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"batch_size": batch_size, "bf16": bf16, "channels_last": channels_last, "oom": True}
