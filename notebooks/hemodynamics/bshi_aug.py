"""Augmentation and corruption hooks for BSHI batches (torch tensors, CPU or GPU, deterministic given a ``torch.Generator``).

Each function is registered in ``AUGMENTATIONS`` with its exactness:
  EXACT           the result equals what ``bshi.assemble`` produces for the corrupted input (use ``assemble(..., extra_mask=...)`` with the SAME keep mask; see ``gap_keep_mask``)
  APPROXIMATION   the corruption is applied to the finished tensor (planes are not re-normalized over the remaining beats), the right choice for training-time augmentation
                  from the compact store. Mask semantics (zero values, SQI 0, mask False, counts updated) follow the real masks.
Beat times (``t_rel``, ``rr``) come from ``bshi.assemble(..., return_times=True)`` or the compact store.

Batch keys used: x (B,7,60,200), beat_mask (B,3,60) bool [ecg, art, ppg], context (B,25), t_rel and rr (B,60) seconds.
"""
from __future__ import annotations
# Apply controlled signal corruption to test how the model handles poor inputs.

import json
from pathlib import Path

import numpy as np
import torch

from . import bshi as S

PPG_BANDS = [("timing", "ptt_ecg_ppg"), ("timing", "ptt_art_ppg"), ("morphology", "ppg_amp"), ("morphology", "ppg_ac_dc"), ("morphology", "ppg_upstroke"),
             ("morphology", "art_ppg_corr"), ("sqi", "sqi_ppg"), ("sqi", "sqi_cross")]
_BANDS = {"timing": (4, S.TIMING_BANDS, 50), "morphology": (5, S.MORPH_BANDS, 25), "sqi": (6, S.SQI_BANDS, 50)}
ART_BANDS = [("timing", "ptt_ecg_art"), ("morphology", "pp"), ("morphology", "dpdt_max"), ("morphology", "sys_time"), ("morphology", "art_area"), ("morphology", "art_ppg_corr"),
             ("sqi", "sqi_art"), ("sqi", "sqi_cross")]


def band_slice(plane_name, band):
    plane, names, width = _BANDS[plane_name]
    k = names.index(band)
    return plane, slice(k * width, (k + 1) * width)


# MOVER gap statistics
def measure_mover_gaps(root, n_cases=40, seed=5052, write=True):
    """Valid-run and gap-length quantiles (s) of MOVER capture (all of ART, ECG, PPG finite) from a direct measurement on ``n_cases`` MOVER waveforms."""
    # Read the stored table required for this calculation.
    import pandas as pd
    import pyarrow.parquet as pq
    root = Path(root)
    idx = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet", columns=["dataset", "case_id", "waveform_path"])
    idx = idx[idx.dataset == "mover"]
    rng = np.random.default_rng(seed)
    runs, gaps, fracs = [], [], []
    for _, r in idx.iloc[rng.choice(len(idx), min(n_cases, len(idx)), replace=False)].iterrows():
        # Exclude or flag missing values before using these measurements.
        t = pq.read_table(root / r.waveform_path, columns=["time_sec", "ART", "ECG", "PPG"]).to_pandas()
        ts = t.time_sec.to_numpy()
        n = int(round((ts[-1] - ts[0]) * 100)) + 1
        # Exclude or flag missing values before using these measurements.
        ok = np.zeros(n, dtype=bool)
        ok[np.rint((ts - ts[0]) * 100).astype(np.int64)] = np.isfinite(t[["ART", "ECG", "PPG"]].to_numpy()).all(axis=1)
        edges = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(np.int8)))
        s, e = edges[0::2], edges[1::2]
        runs.append((e - s) / 100.0)
        gaps.append((s[1:] - e[:-1]) / 100.0)
        fracs.append(float(ok.mean()))
    runs, gaps = np.concatenate(runs), np.concatenate(gaps)
    q = np.linspace(0, 1, 101)
    out = {"cases": int(min(n_cases, len(idx))), "valid_fraction_mean": float(np.mean(fracs)), "valid_run_quantiles_s": np.quantile(runs, q).round(3).tolist(),
           "gap_quantiles_s": np.quantile(gaps, q).round(3).tolist(), "valid_run_median_s": float(np.median(runs)), "gap_median_s": float(np.median(gaps)), "runs": int(len(runs))}
    if write:
        (root / "results" / "bshi_gap_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _quantile_sampler(q, shape, gen, device):
    table = torch.as_tensor(q, dtype=torch.float32, device=device)
    u = torch.rand(shape, generator=gen, device=device) * (len(table) - 1)
    lo = u.floor().long().clamp(0, len(table) - 2)
    frac = u - lo
    return table[lo] * (1 - frac) + table[lo + 1] * frac


def gap_keep_mask(rr, gap_stats, gen, pairs=48):
    """(B,60) bool of beats that survive a MOVER-style on/off capture.

    In real MOVER windows beats exist only inside valid runs and the one beat whose cycle spans a gap is invalid (MOVER valid-beat share 88.8%). The simulation therefore walks
    along the beats on a gap-free clock (cumulative R-R), draws valid-run lengths from the measured MOVER distribution with a random phase, and marks the beat that contains each
    run boundary as lost. Pass the SAME mask to ``assemble(extra_mask=...)`` for the exact version (the remaining beats are re-normalized like a real window)."""
    dev = rr.device
    b = rr.shape[0]
    runs = _quantile_sampler(gap_stats["valid_run_quantiles_s"], (b, pairs), gen, dev).clamp(min=0.5)
    bounds = torch.cumsum(runs, dim=1) - torch.rand((b, 1), generator=gen, device=dev) * runs[:, :1]      # random phase into the first run
    end = torch.cumsum(rr, dim=1)
    start = end - rr
    inside = (bounds.unsqueeze(1) > start.unsqueeze(2)) & (bounds.unsqueeze(1) <= end.unsqueeze(2))
    return ~inside.any(dim=2)


# tensor-level functions
def _copy(batch):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def drop_beats(batch, keep):
    """Beats with keep=False are not captured: all planes zero at those beats, masks False, per-window counts updated (APPROXIMATION: remaining beats are not re-normalized)."""
    out = _copy(batch)
    out["x"] = out["x"] * keep.unsqueeze(1).unsqueeze(3).to(out["x"].dtype)
    out["beat_mask"] = out["beat_mask"] & keep.unsqueeze(1)
    ctx = out["context"]
    for j, name in enumerate(("n_ecg_ok", "n_art_ok", "n_ppg_ok")):
        ctx[:, S.CONTEXT_NAMES.index(name)] = out["beat_mask"][:, j].sum(dim=1).to(ctx.dtype)
    return out


def mover_gaps(batch, gap_stats, gen):
    """Training-time MOVER-style gap augmentation from a batch that carries ``rr`` (APPROXIMATION of the exact assemble(extra_mask) path)."""
    keep = gap_keep_mask(batch["rr"], gap_stats, gen)
    out = drop_beats(batch, keep)
    out["keep"] = keep
    return out


def ppg_dropout(batch, p_window, p_beat, gen):
    """With probability ``p_window`` a window loses all PPG, otherwise each beat loses PPG with probability ``p_beat``: PPG plane and PPG bands zero, mask False, count updated."""
    # Calculate output values and the input array for the following step.
    dev = batch["x"].device
    b = batch["x"].shape[0]
    window = torch.rand((b, 1), generator=gen, device=dev) < p_window
    beat = torch.rand((b, S.N_BEATS), generator=gen, device=dev) < p_beat
    drop = window | beat                                                                  # (B,60)
    out = _copy(batch)
    x = out["x"]
    keep = (~drop).to(x.dtype)
    x[:, 2] = x[:, 2] * keep.unsqueeze(2)
    for plane_name, band in PPG_BANDS:
        plane, sl = band_slice(plane_name, band)
        x[:, plane, :, sl] = x[:, plane, :, sl] * keep.unsqueeze(2)
    out["beat_mask"][:, 2] = out["beat_mask"][:, 2] & ~drop
    out["context"][:, S.CONTEXT_NAMES.index("n_ppg_ok")] = out["beat_mask"][:, 2].sum(dim=1).to(out["context"].dtype)
    out["dropped"] = drop
    return out


def ecg_noise(batch, snr_db, gen):
    """Add white Gaussian noise to the ECG plane at the given SNR (dB) relative to the window's ECG power over unmasked beats (APPROXIMATION)."""
    # Calculate output values and the input array for the following step.
    out = _copy(batch)
    x = out["x"]
    live = out["beat_mask"][:, 0].unsqueeze(2).to(x.dtype)
    power = (x[:, 0] ** 2 * live).sum(dim=(1, 2)) / (live.sum(dim=(1, 2)) * x.shape[3]).clamp(min=1.0)
    sigma = torch.sqrt(power * 10 ** (-float(snr_db) / 10)).view(-1, 1, 1)
    x[:, 0] = x[:, 0] + torch.randn(x[:, 0].shape, generator=gen, device=x.device) * sigma * live
    return out


def art_damping(batch, width):
    """Over-damped arterial line: moving average of ``width`` phase samples on the ART plane (lower pulse pressure, rounder waveform) (APPROXIMATION)."""
    # Calculate output values and the input array for the following step.
    out = _copy(batch)
    x = out["x"]
    b, n = x.shape[0], x.shape[2]
    sig = x[:, 1].reshape(b * n, 1, -1)
    pad = width // 2
    sig = torch.nn.functional.pad(sig, (pad, width - 1 - pad), mode="replicate")
    kernel = torch.ones(1, 1, width, device=x.device, dtype=x.dtype) / width
    x[:, 1] = torch.nn.functional.conv1d(sig, kernel).reshape(b, n, -1)
    return out


def art_flatline(batch, p_beat, gen):
    """A random share of beats show a flat arterial trace (the beat mean) while the masks stay as they were: robustness to an undetected failure (APPROXIMATION)."""
    # Calculate output values and the input array for the following step.
    dev = batch["x"].device
    out = _copy(batch)
    x = out["x"]
    flat = torch.rand((x.shape[0], S.N_BEATS), generator=gen, device=dev) < p_beat
    mean = x[:, 1].mean(dim=2, keepdim=True)
    x[:, 1] = torch.where(flat.unsqueeze(2), mean.expand_as(x[:, 1]), x[:, 1])
    out["flat"] = flat
    return out


def timing_jitter(batch, sigma_s, gen):
    """R-wave anchoring jitter: every beat's ECG, ART and PPG waveform is circularly shifted along the phase axis by N(0, sigma_s) seconds converted to phase samples with the beat's
    own R-R interval (needs ``rr``) (APPROXIMATION: the stored delay bands are not recomputed)."""
    # Calculate output values and the input array for the following step.
    out = _copy(batch)
    x = out["x"]
    b, n, p = x.shape[0], x.shape[2], x.shape[3]
    shift = torch.round(torch.randn((b, n), generator=gen, device=x.device) * sigma_s / batch["rr"].clamp(min=0.3) * p).long()
    base = torch.arange(p, device=x.device).view(1, 1, p)
    gather = ((base - shift.unsqueeze(2)) % p).unsqueeze(1).expand(b, 3, n, p)
    x[:, :3] = torch.gather(x[:, :3], 3, gather)
    return out


AUGMENTATIONS = {
    "mover_gaps": {"fn": mover_gaps, "exactness": "APPROXIMATION (exact path: bshi.assemble(extra_mask=gap_keep_mask(...)))", "needs": ["rr"]},
    "drop_beats": {"fn": drop_beats, "exactness": "APPROXIMATION (exact path: assemble(extra_mask=keep))", "needs": []},
    "ppg_dropout": {"fn": ppg_dropout, "exactness": "APPROXIMATION in values, exact mask semantics (exact path: assemble(extra_mask_ppg=keep))", "needs": []},
    "ecg_noise": {"fn": ecg_noise, "exactness": "APPROXIMATION", "needs": []},
    "art_damping": {"fn": art_damping, "exactness": "APPROXIMATION", "needs": []},
    "art_flatline": {"fn": art_flatline, "exactness": "APPROXIMATION", "needs": []},
    "timing_jitter": {"fn": timing_jitter, "exactness": "APPROXIMATION", "needs": ["rr"]},
}


def to_torch(batch, device="cpu"):
    """numpy batch dict from ``bshi.assemble`` -> torch tensors."""
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items() if isinstance(v, np.ndarray)}


def clamp_scalar_planes(x, limit=6.0):
    """MODEL-SIDE clamp of the per-window z-scored scalar planes (MAP, timing, morphology: planes 3-5) to +-limit; waveform planes (already clipped to +-8) and the SQI plane are
    untouched. The assembler is not changed: a near-zero PPG DC level makes the MOVER ``ppg_ac_dc`` band unstable (see docs/bshi_training_readiness.md)."""
    out = x.clone()
    out[:, 3:6] = out[:, 3:6].clamp(-limit, limit)
    return out
