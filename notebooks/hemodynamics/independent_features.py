"""Extract causal ECG, ART, and PPG beat parameters using each signal independently.

Every parameter uses ONE signal: its own beat detection, segmentation, polarity and normalization. Nothing is paired with another signal's beats
(the beat store's ``map_beat``, ``art_int_sec`` and ``hr_art_bpm`` are built on ECG R-R cycles, so they are NOT used here).

Strict causality (online-computable): for each case the window ends e_1 < e_2 < ... define chunks; chunk j runs the detectors only on samples in
[s_j - PREROLL, e_j] and keeps the beats whose cycle completes in (s_j - LATENCY, e_j - LATENCY]. A window ending at e_j therefore uses only beats
detected from samples <= e_j, polarity comes from a running vote over the signal's own past, and the self-normalized (``_sn``) parameters divide by the
expanding median of the same case's earlier windows. Changing any sample after e_j changes no parameter of window j (tested).

Native series (one row per beat, ``t`` = time the cycle is complete):
* ECG: R time, RR (s), R confidence, R amplitude (oriented ECG units, for ECG-derived respiration), validity, ectopy flag.
* ART: foot-to-foot pulses: SBP, DBP, PP, foot-to-foot mean pressure, dP/dt max/min, rise time, interval, dicrotic notch (relative height and
  ejection fraction of the cycle), systolic/diastolic area ratio, diastolic decay time constant tau (log-linear fit with zero asymptote), form factor,
  skewness, own template quality, validity.
* PPG: polarity from the PPG alone (steeper upstroke), amplitude, AC/DC, rise time, width at 50% (fraction of the cycle), interval, quality, validity.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import gaussian_filter1d
from scipy.signal import lombscargle, sosfiltfilt

from . import beats as B

FS = 100.0
COLUMN = {"ecg": "ECG", "art": "ART", "ppg": "PPG"}
PREROLL = 10.0             # s of signal before a chunk that the detector may use (context only)
LATENCY = 1.0              # s: a beat is used once its cycle is complete and 1 s has passed (lets the detector see the next upstroke)
CHUNK_LOOKBACK = 305.0     # s covered before the first window end of a run of windows (300 s lookback + margin)
LOOKBACKS = {"l60": 60.0, "l300": 300.0}
MIN_COVER = {"l60": 0.5, "l300": 0.7}
MIN_BEATS = {"l60": 20, "l300": 100}
SN_MIN_HISTORY = 11        # windows (current included): about 5 min of earlier windows at the 30 s stride
DETREND_SIGMA = 30.0       # beats (Gaussian kernel detrend for the temporal-correlation family)
ROLL_WIN, ROLL_STEP = 50, 10
VERSION = "if1"
OUT_DIR = Path("data") / "derived" / "independence"
PARAMS_NAME = "window_params_v1.parquet"


# reading
def read_signal(path, signal, t_lo=None, t_hi=None, reader=None):
    """(time_sec, values) of ONE signal; only ``time_sec`` and that signal's column are requested from the file."""
    reader = reader or pq.read_table
    cols = ["time_sec", COLUMN[signal]]
    filters = None
    if t_lo is not None and t_hi is not None:
        filters = [("time_sec", ">=", float(t_lo)), ("time_sec", "<=", float(t_hi))]
    t = reader(path, columns=cols, filters=filters)
    if hasattr(t, "column"):
        ts = np.asarray(t.column("time_sec").to_numpy(), dtype=np.float64)
        x = np.asarray(t.column(COLUMN[signal]).to_numpy(zero_copy_only=False), dtype=np.float64)
    else:
        ts, x = np.asarray(t["time_sec"], dtype=np.float64), np.asarray(t[COLUMN[signal]], dtype=np.float64)
    return ts, x


def dense(ts, x):
    """(t0, NaN-filled 100 Hz array) on the case clock."""
    if len(ts) == 0:
        return 0.0, np.zeros(0)
    t0 = float(ts[0])
    n = int(round((ts[-1] - t0) * FS)) + 1
    out = np.full(n, np.nan)
    out[np.rint((ts - t0) * FS).astype(np.int64)] = x
    return t0, out


def chunk_plan(t_end, lookback=CHUNK_LOOKBACK):
    """Chunk starts/ends for sorted unique window ends: chunk j covers (s_j, e_j] with s_j = max(e_{j-1}, e_j - lookback)."""
    e = np.unique(np.asarray(t_end, dtype=np.float64))
    s = np.empty_like(e)
    if len(e):
        s[0] = e[0] - lookback
        s[1:] = np.maximum(e[:-1], e[1:] - lookback)
    return s, e


def _iter_chunks(t0, x, starts, ends):
    """Yield (j, offset_time, chunk array, new-data slice for polarity votes)."""
    n = len(x)
    prev_end = -np.inf
    for j, (s, e) in enumerate(zip(starts, ends)):
        i0 = max(0, int(np.ceil((s - PREROLL - t0) * FS)))
        i1 = min(n, int(np.floor((e - t0) * FS)) + 1)
        if i1 - i0 < int(B.MIN_SEG_SEC * FS):
            prev_end = e
            continue
        v0 = max(i0, int(np.floor((prev_end - t0) * FS)) + 1) if np.isfinite(prev_end) else i0
        prev_end = e
        yield j, t0 + i0 / FS, x[i0:i1], x[max(v0, i0):i1]


def _keep(avail, s, e):
    return (avail > s - LATENCY) & (avail <= e - LATENCY)


# polarity (own signal only)
def ecg_vote(x):
    """Length-weighted vote (+: the larger QRS deflection is positive), same rule as ``beats.ecg_orientation`` but returned as a score."""
    high = B._sos("high", FS, 0.5, 0.0)
    score = 0.0
    for a, b in zip(*B.finite_segments(x, int(B.MIN_SEG_SEC * FS))):
        seg = x[a:b].astype(np.float64)
        f = sosfiltfilt(high, seg - np.median(seg))
        score += len(seg) * (1.0 if np.percentile(f, 99.5) >= -np.percentile(f, 0.5) else -1.0)
    return score


def ppg_vote(x):
    """Length-weighted vote (+: the systolic upstroke, the steeper edge, is positive). Uses the PPG alone (``beats.orientation`` also needs the ECG)."""
    low = B._sos("low", FS, 0.0, 12.0)
    score = 0.0
    for a, b in zip(*B.finite_segments(x, int(B.MIN_SEG_SEC * FS))):
        seg = x[a:b].astype(np.float64)
        d = np.gradient(sosfiltfilt(low, seg - np.median(seg))) * FS
        score += len(seg) * (1.0 if np.percentile(d, 99.5) >= -np.percentile(d, 0.5) else -1.0)
    return score


def ppg_polarity(x):
    """Choose PPG orientation from the waveform alone using its polarity vote."""
    return 1.0 if ppg_vote(x) >= 0 else -1.0


# ECG native
ECG_COLS = ["t", "rr", "conf", "ramp", "valid", "ectopic"]


def _nan_count(x):
    return np.concatenate(([0], np.cumsum(~np.isfinite(x))))


def native_ecg(ts, x, starts, ends):
    """ECG-only beat series (one row per R-R interval, ``t`` = time of the closing R)."""
    t0, x = dense(ts, x)
    parts, vote = [], 0.0
    for j, toff, xc, xnew in _iter_chunks(t0, x, starts, ends):
        vote += ecg_vote(xnew)
        sign = 1.0 if vote >= 0 else -1.0
        r, conf, _ = B.detect_r_peaks(xc, FS, sign=sign)
        if len(r) < 2:
            continue
        g = sign * xc
        ri = np.clip(np.rint(r).astype(np.int64), 0, len(xc) - 1)
        half = int(0.5 * FS)
        ramp = np.array([g[k] - np.nanmedian(g[max(0, k - half):k + half]) for k in ri])
        nanc = _nan_count(xc)
        rr = np.diff(r) / FS
        gap = (nanc[ri[1:] + 1] - nanc[ri[:-1]]) > 0
        valid = (conf[1:] >= 0.3) & (conf[:-1] >= 0.3) & (rr >= 0.286) & (rr <= 2.0) & ~gap
        t = toff + r[1:] / FS
        keep = _keep(t, starts[j], ends[j])
        parts.append(pd.DataFrame({"t": t[keep], "rr": rr[keep], "conf": conf[1:][keep], "ramp": ramp[1:][keep], "valid": valid[keep]}))
    if not parts:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c != "valid" and c != "ectopic" else "bool") for c in ECG_COLS})
    df = pd.concat(parts, ignore_index=True).sort_values("t", kind="stable").reset_index(drop=True)
    med = df.rr.where(df.valid).rolling(9, min_periods=3).median().shift(1)        # previous beats only (causal)
    df["ectopic"] = (df.valid & med.notna() & ((df.rr / med - 1.0).abs() > 0.2)).to_numpy()
    return df[ECG_COLS]


# ART / PPG native
ART_COLS = ["t", "sbp", "dbp", "pp", "mapp", "dpdt_max", "dpdt_min", "rise", "interval", "notch_rel", "ejec", "area_ratio", "tau", "form",
            "skew", "q", "ok"]
PPG_COLS = ["t", "amp", "acdc", "rise", "width50", "interval", "q", "ok", "pol"]
SHAPE_N, QBLOCK = 50, 30
_GRID = np.linspace(0.0, 1.0, SHAPE_N, endpoint=False)


def _smooth_segments(x, hi=12.0):
    """Low-pass of each finite segment (NaN elsewhere) and its first and second derivatives (units/s, units/s^2)."""
    sos = B._sos("low", FS, 0.0, hi)
    y = np.full(len(x), np.nan)
    for a, b in zip(*B.finite_segments(x, int(B.MIN_SEG_SEC * FS))):
        seg = x[a:b].astype(np.float64)
        m = np.median(seg)
        y[a:b] = sosfiltfilt(sos, seg - m) + m
    d1 = np.gradient(y) * FS if len(y) > 1 else np.full(len(y), np.nan)
    d2 = np.gradient(d1) * FS if len(y) > 1 else np.full(len(y), np.nan)
    return y, d1, d2


def find_notch(d1, d2, P, end):
    """Dicrotic notch sample: first local minimum (d1 from - to +) in [P + 0.05 s, P + min(0.5 s, 0.7 (end - P))], else the d2 maximum if positive; -1 if none."""
    lo = P + int(0.05 * FS)
    hi = P + int(min(0.5 * FS, 0.7 * (end - P)))
    if hi - lo < 3 or hi >= len(d1):
        return -1
    seg = d1[lo:hi + 1]
    if not np.isfinite(seg).all():
        return -1
    cross = np.flatnonzero((seg[:-1] < 0) & (seg[1:] >= 0))
    if len(cross):
        return lo + int(cross[0]) + 1
    s2 = d2[lo:hi + 1]
    k = int(np.argmax(s2))
    return lo + k if np.isfinite(s2[k]) and s2[k] > 0 else -1


def decay_tau(y, a, b):
    """Time constant (s) of the exponential decay y ~ exp(-t / tau) on samples [a, b] (zero asymptote; a Windkessel R*C surrogate); NaN if unusable."""
    if b - a < 8:
        return np.nan
    seg = y[a:b + 1]
    if not (np.isfinite(seg).all() and (seg > 0).all()):
        return np.nan
    t = np.arange(len(seg)) / FS
    ly = np.log(seg)
    tc = t - t.mean()
    slope = float(tc @ (ly - ly.mean()) / (tc @ tc))
    if slope >= 0:
        return np.nan
    tau = -1.0 / slope
    return tau if tau <= 20.0 else np.nan


def _shape(cyc):
    s = np.interp(_GRID * (len(cyc) - 1), np.arange(len(cyc)), cyc)
    sd = s.std()
    return (s - s.mean()) / sd if sd > 0 else np.full(SHAPE_N, np.nan)


def _art_cycles(xc, y, d1, d2, nanc, det):
    """Per-cycle ART quantities for cycles foot[k] -> foot[k+1]."""
    foot, peak, top, base, dpdt = det["foot"], det["peak"], det["top"], det["base"], det["dpdt"]
    n = len(foot) - 1
    cols = [c for c in ART_COLS if c not in ("t", "q", "ok")]
    out = {c: np.full(n, np.nan) for c in cols}
    ok = np.zeros(n, dtype=bool)
    shapes = np.full((n, SHAPE_N), np.nan)
    for k in range(n):
        f0, f1, P = foot[k], foot[k + 1], int(peak[k])
        i0, i1 = int(np.ceil(f0)), int(np.floor(f1))
        if i1 - i0 < 10 or not (f0 < P < f1) or nanc[i1 + 1] - nanc[i0] > 0:
            continue
        cyc = xc[i0:i1 + 1]
        T = (f1 - f0) / FS
        sbp, dbp = float(top[k]), float(base[k])
        pp = sbp - dbp
        mapp = float(cyc.mean())
        sd = float(cyc.std())
        out["sbp"][k], out["dbp"][k], out["pp"][k], out["mapp"][k] = sbp, dbp, pp, mapp
        out["dpdt_max"][k], out["dpdt_min"][k] = dpdt[k], float(np.nanmin(d1[P:i1 + 1]))
        out["rise"][k], out["interval"][k] = (P - f0) / FS, T
        if pp > 0:
            out["form"][k] = (mapp - dbp) / pp
        if sd > 0:
            out["skew"][k] = float(((cyc - mapp) ** 3).mean() / sd ** 3)
        notch = find_notch(d1, d2, P, i1)
        if notch > 0 and notch < i1:
            if pp > 0:
                out["notch_rel"][k] = (y[notch] - dbp) / pp
            out["ejec"][k] = (notch - f0) / (f1 - f0)
            sys_a, dia_a = float(np.sum(xc[i0:notch] - dbp)), float(np.sum(xc[notch:i1 + 1] - dbp))
            if dia_a > 0:
                out["area_ratio"][k] = sys_a / dia_a
            out["tau"][k] = decay_tau(y, notch + int(0.4 * (i1 - notch)), i1 - 2)
        clipped = max(np.mean(cyc >= cyc.max() - 0.1), np.mean(cyc <= cyc.min() + 0.1)) >= 0.15
        ok[k] = (15 <= pp <= 150) and sbp <= 300 and dbp >= -20 and (20 <= mapp <= 200) and sd >= 0.5 and not clipped and 0.286 <= T <= 2.0
        shapes[k] = _shape(cyc)
    return out, ok, shapes


def _ppg_cycles(xc, pol, nanc, det):
    foot, peak, top, base = det["foot"], det["peak"], det["top"], det["base"]
    n = len(foot) - 1
    out = {c: np.full(n, np.nan) for c in ("amp", "acdc", "rise", "width50", "interval")}
    ok = np.zeros(n, dtype=bool)
    shapes = np.full((n, SHAPE_N), np.nan)
    for k in range(n):
        f0, f1, P = foot[k], foot[k + 1], int(peak[k])
        i0, i1 = int(np.ceil(f0)), int(np.floor(f1))
        if i1 - i0 < 10 or not (f0 < P < f1) or nanc[i1 + 1] - nanc[i0] > 0:
            continue
        cyc = pol * xc[i0:i1 + 1]
        T = (f1 - f0) / FS
        amp = float(top[k] - base[k])
        dc = abs(float(cyc.mean()))
        out["amp"][k], out["rise"][k], out["interval"][k] = amp, (P - f0) / FS, T
        if dc > 0:
            out["acdc"][k] = amp / dc
        if amp > 0:
            out["width50"][k] = float(np.mean(cyc - base[k] >= 0.5 * amp))
        ok[k] = amp > 0 and cyc.std() > 0 and 0.286 <= T <= 2.0
        shapes[k] = _shape(cyc)
    return out, ok, shapes


def block_quality(shapes, ok):
    """Correlation of each valid pulse's shape with the median shape of the PREVIOUS block of QBLOCK valid pulses (causal; first block NaN)."""
    q = np.full(len(shapes), np.nan)
    idx = np.flatnonzero(ok & np.isfinite(shapes).all(axis=1))
    for b in range(1, int(np.ceil(len(idx) / QBLOCK))):
        prev, cur = idx[(b - 1) * QBLOCK:b * QBLOCK], idx[b * QBLOCK:(b + 1) * QBLOCK]
        tmpl = np.median(shapes[prev], axis=0)
        sd = tmpl.std()
        if sd > 0:
            q[cur] = shapes[cur] @ ((tmpl - tmpl.mean()) / sd) / SHAPE_N
    return q


def _empty(cols):
    return pd.DataFrame({c: pd.Series(dtype="bool" if c == "ok" else "float64") for c in cols})


def native_pulses(ts, x, starts, ends, kind):
    """ART-only (``kind='art'``) or PPG-only (``kind='ppg'``) pulse series; one row per foot-to-foot cycle, ``t`` = time of the closing foot."""
    t0, x = dense(ts, x)
    parts, vote = [], 0.0
    for j, toff, xc, xnew in _iter_chunks(t0, x, starts, ends):
        pol = 1.0
        if kind == "ppg":
            vote += ppg_vote(xnew)
            pol = 1.0 if vote >= 0 else -1.0
        det = B.detect_pulses(xc, FS, kind=kind, polarity=pol)
        if len(det["foot"]) < 2:
            continue
        nanc = _nan_count(xc)
        if kind == "art":
            y, d1, d2 = _smooth_segments(xc, 12.0)
            out, ok, shapes = _art_cycles(xc, y, d1, d2, nanc, det)
        else:
            out, ok, shapes = _ppg_cycles(xc, pol, nanc, det)
            out["pol"] = np.full(len(ok), pol)
        t = toff + det["foot"][1:] / FS
        keep = _keep(t, starts[j], ends[j])
        df = pd.DataFrame({"t": t[keep], **{c: v[keep] for c, v in out.items()}, "ok": ok[keep]})
        parts.append((df, shapes[keep]))
    cols = ART_COLS if kind == "art" else PPG_COLS
    if not parts:
        return _empty(cols)
    df = pd.concat([p[0] for p in parts], ignore_index=True)
    shapes = np.concatenate([p[1] for p in parts])
    order = np.argsort(df.t.to_numpy(), kind="stable")
    df, shapes = df.iloc[order].reset_index(drop=True), shapes[order]
    df["q"] = block_quality(shapes, df.ok.to_numpy())
    return df[cols]


# window helpers
_LS_F = np.arange(0.04, 0.4001, 0.005)          # Hz: LF 0.04-0.15 (Mayer waves), HF 0.15-0.40 (respiration)
_EDR_F = np.arange(0.10, 0.6001, 0.005)
NAN = float("nan")


def detrend(v, sigma=DETREND_SIGMA):
    """Residual after a Gaussian-kernel smoother (bandwidth ``sigma`` beats), the standard critical-slowing-down detrend."""
    v = np.asarray(v, dtype=np.float64)
    return v - gaussian_filter1d(v, sigma, mode="nearest") if len(v) else v


def ac1(v):
    """Compute lag-one autocorrelation; return NaN for short or constant sequences."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 4:
        return NAN
    v = v - v.mean()
    d = float(v @ v)
    return float(v[:-1] @ v[1:] / d) if d > 0 else NAN


def skewness(v):
    """Compute the standardized third moment of a sufficiently long beat sequence."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 4:
        return NAN
    sd = v.std()
    return float(((v - v.mean()) ** 3).mean() / sd ** 3) if sd > 0 else NAN


def dfa_alpha1(v, scales=(4, 5, 6, 8, 10, 12, 14, 16)):
    """Short-term detrended-fluctuation exponent (box sizes 4-16 beats, linear detrend per box); white noise 0.5, 1/f noise 1.0.

    F(n) is divided by sqrt(1 - 4/n^2), the exact small-box shrinkage of linear detrending for white noise (without it white noise gives about
    0.59-0.61). The factor is fixed per box size, so it shifts alpha1 by a constant and leaves every rank-based result unchanged."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < 64:
        return NAN
    y = np.cumsum(v - v.mean())
    logn, logf = [], []
    for n in scales:
        m = len(y) // n
        seg = y[:m * n].reshape(m, n)
        tc = np.arange(n) - (n - 1) / 2.0
        c = seg - seg.mean(axis=1, keepdims=True)
        b = c @ tc / (tc @ tc)
        f = np.sqrt(np.mean((c - b[:, None] * tc) ** 2) / (1.0 - 4.0 / n ** 2))
        if f > 0:
            logn.append(np.log(n))
            logf.append(np.log(f))
    if len(logn) < 4:
        return NAN
    return float(np.polyfit(logn, logf, 1)[0])


def sample_entropy(v, m=2, r_frac=0.2, max_n=300):
    """SampEn(m, r = r_frac * SD) of the last ``max_n`` values (self-matches excluded, n - m templates for both lengths)."""
    v = np.asarray(v, dtype=np.float64)
    v = v[np.isfinite(v)][-max_n:]
    n = len(v)
    if n < 50:
        return NAN
    r = r_frac * v.std()
    if r <= 0:
        return NAN

    def pairs(k):
        emb = sliding_window_view(v, k)[:n - m]
        d = np.abs(emb[:, None, :] - emb[None, :, :]).max(axis=2)
        return (int(np.count_nonzero(d <= r)) - len(emb)) / 2.0

    b, a = pairs(m), pairs(m + 1)
    return float(-np.log(a / b)) if a > 0 and b > 0 else NAN


def _lin_resid(t, v):
    tc = t - t.mean()
    d = tc @ tc
    return v - v.mean() - (tc * (tc @ (v - v.mean())) / d if d > 0 else 0.0)


def lf_hf(t, v):
    """log(LF / HF) of an unevenly sampled beat series (Lomb-Scargle periodogram, LF 0.04-0.15 Hz, HF 0.15-0.40 Hz)."""
    t, v = np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if len(v) < 100 or t[-1] - t[0] < 120:
        return NAN
    p = lombscargle(t - t[0], _lin_resid(t, v), 2 * np.pi * _LS_F)
    lf, hf = p[_LS_F < 0.15].sum(), p[_LS_F >= 0.15].sum()
    return float(np.log(lf / hf)) if lf > 0 and hf > 0 else NAN


def peak_rate(t, v):
    """Dominant rate (per minute) in 0.1-0.6 Hz of a beat-sampled series (ECG-derived respiration from R amplitude)."""
    t, v = np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)
    if len(v) < 20 or t[-1] - t[0] < 30:
        return NAN
    p = lombscargle(t - t[0], _lin_resid(t, v), 2 * np.pi * _EDR_F)
    return float(60.0 * _EDR_F[int(np.argmax(p))]) if np.isfinite(p).all() and p.max() > 0 else NAN


def kendall_trend(x):
    """Kendall tau-a of x against its index (+1 = steadily rising)."""
    x = np.asarray(x, dtype=np.float64)
    k = len(x)
    if k < 5:
        return NAN
    iu = np.triu_indices(k, 1)
    return float(np.sign(x[None, :] - x[:, None])[iu].sum() / (k * (k - 1) / 2))


def rolling_trends(r):
    """(Kendall tau of rolling lag-1 autocorrelation, of rolling variance) over windows of ROLL_WIN beats every ROLL_STEP beats."""
    r = np.asarray(r, dtype=np.float64)
    if len(r) < ROLL_WIN + 4 * ROLL_STEP:
        return NAN, NAN
    w = sliding_window_view(r, ROLL_WIN)[::ROLL_STEP]
    wc = w - w.mean(axis=1, keepdims=True)
    ss = (wc ** 2).sum(axis=1)
    ac = np.where(ss > 0, (wc[:, :-1] * wc[:, 1:]).sum(axis=1) / np.where(ss > 0, ss, 1.0), np.nan)
    return kendall_trend(ac), kendall_trend(ss / ROLL_WIN)


def span_variation(v, k=8, relative=True):
    """Median over sliding k-beat spans of (max - min) [/ span mean]: PPV / SPV / PVI-like respiratory variation (k = 8 beats ~ 1-2 breaths)."""
    v = np.asarray(v, dtype=np.float64)
    if len(v) < k:
        return NAN
    w = sliding_window_view(v, k)
    r = w.max(axis=1) - w.min(axis=1)
    if relative:
        m = w.mean(axis=1)
        r = np.where(m != 0, r / np.where(m != 0, m, 1.0), np.nan)
    return float(np.nanmedian(r))


def slope_per_min(t, v):
    """Fit a linear trend against time in seconds and express its slope per minute."""
    t, v = np.asarray(t, dtype=np.float64), np.asarray(v, dtype=np.float64)
    ok = np.isfinite(v)
    t, v = t[ok], v[ok]
    if len(v) < 3:
        return NAN
    tc = t - t.mean()
    d = tc @ tc
    return float(tc @ (v - v.mean()) / d * 60.0) if d > 0 else NAN


def successive(t, iv, tol=0.05):
    """Mask of consecutive beat pairs that are truly adjacent (the next beat starts where the previous one ends)."""
    return np.abs(np.diff(t) - iv[1:]) < tol


def hrv(t, iv):
    """SDNN, RMSSD (ms), pNN50, SD1/SD2 of a normal-to-normal interval series (s)."""
    if len(iv) < 5:
        return NAN, NAN, NAN, NAN
    d = np.diff(iv)[successive(t, iv)]
    sdnn = float(iv.std() * 1000.0)
    if len(d) < 5:
        return sdnn, NAN, NAN, NAN
    rmssd = float(np.sqrt(np.mean(d ** 2)) * 1000.0)
    pnn50 = float(np.mean(np.abs(d) > 0.05))
    sd1 = np.sqrt(np.var(d) / 2.0)
    sd2 = np.sqrt(max(2.0 * np.var(iv) - np.var(d) / 2.0, 0.0))
    return sdnn, rmssd, pnn50, float(sd1 / sd2) if sd2 > 0 else NAN


def spearman(a, b):
    """Compute a rank-based correlation for at least ten finite paired measurements."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return NAN
    ra = np.argsort(np.argsort(a[ok])).astype(np.float64)
    rb = np.argsort(np.argsort(b[ok])).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra @ ra) * (rb @ rb))
    return float(ra @ rb / d) if d > 0 else NAN


def _med(v):
    v = v[np.isfinite(v)]
    return float(np.median(v)) if len(v) else NAN


# window parameters
ART_LEVEL = ["sbp", "dbp", "pp", "dpdt_max", "dpdt_min", "rise", "notch_rel", "ejec", "area_ratio", "tau", "form", "skew"]
ART_TC = {"map": "mapp", "sbp": "sbp", "pp": "pp", "int": "interval"}
ART_TREND = {"map": "mapp", "pp": "pp", "hr": "hr", "dpdt": "dpdt_max", "tau": "tau"}
SN_RULES = {**{f"a_{c}_l60": "ratio" for c in ("sbp", "dbp", "pp", "dpdt_max", "dpdt_min", "rise", "tau", "area_ratio", "hr", "form")},
            "a_notch_rel_l60": "diff",
            **{c: "ratio" for c in ("e_hr_l60", "e_sdnn_l60", "e_rmssd_l60", "e_sdnn_l300", "e_rmssd_l300")},
            **{f"p_{c}_l60": "ratio" for c in ("amp", "acdc", "rise", "width", "hr")}}


def _slices(t, e, iv):
    """{lookback: (slice or None when coverage is insufficient, covered fraction)} of beats completed in (e - LATENCY - L, e - LATENCY]."""
    hi = int(np.searchsorted(t, e - LATENCY, "right"))
    out = {}
    for lb, L in LOOKBACKS.items():
        lo = int(np.searchsorted(t, e - LATENCY - L, "right"))
        n = hi - lo
        cov = min(float(np.nansum(iv[lo:hi])) / L, 1.0) if n else 0.0
        out[lb] = (slice(lo, hi) if n >= MIN_BEATS[lb] and cov >= MIN_COVER[lb] else None, cov)
    return out


def art_window(A, e):
    """Summarize native ART pulses over causal 60-second and 300-second lookbacks."""
    o = {}
    sl = _slices(A["t"], e, A["interval"])
    s60, s300 = sl["l60"][0], sl["l300"][0]
    o["a_cover_l60"] = sl["l60"][1]
    if s60 is not None:
        for c in ART_LEVEL:
            o[f"a_{c}_l60"] = _med(A[c][s60])
        o["a_hr_l60"] = _med(A["hr"][s60])
        o["a_map_l60"] = _med(A["mapp"][s60])
        o["a_q_l60"] = _med(A["q"][s60])
        o["a_int_sbp_corr_l60"] = spearman(A["interval"][s60], A["sbp"][s60])
        o["a_pp_dbp_corr_l60"] = spearman(A["pp"][s60], A["dbp"][s60])
    for lb, s in (("l60", s60), ("l300", s300)):
        if s is None:
            continue
        t, mp, pp, iv = A["t"][s], A["mapp"][s], A["pp"][s], A["interval"][s]
        o[f"a_ppv_{lb}"] = span_variation(pp)
        o[f"a_spv_{lb}"] = span_variation(A["sbp"][s], relative=False)
        o[f"a_map_sd_{lb}"] = float(mp.std())
        o[f"a_pp_cv_{lb}"] = float(pp.std() / pp.mean()) if pp.mean() > 0 else NAN
        o[f"a_int_sdnn_{lb}"], o[f"a_int_rmssd_{lb}"] = hrv(t, iv)[:2]
        o[f"a_map_range_{lb}"] = float(np.percentile(mp, 95) - np.percentile(mp, 5))
        for nm, col in ART_TREND.items():
            o[f"a_{nm}_slope_{lb}"] = slope_per_min(t, A[col][s])
        for nm, col in ART_TC.items():
            o[f"a_{nm}_ac1_{lb}"] = ac1(detrend(A[col][s]))
    if s60 is not None and s300 is not None:
        for nm, col in {**ART_TREND, "notch_rel": "notch_rel"}.items():
            o[f"a_{nm}_d60_300"] = _med(A[col][s60]) - _med(A[col][s300])
    if s300 is not None:
        t = A["t"][s300]
        for nm, col in ART_TC.items():
            v = A[col][s300]
            r = detrend(v)
            o[f"a_{nm}_rsd_l300"] = float(r.std())
            o[f"a_{nm}_rskew_l300"] = skewness(r)
            o[f"a_{nm}_dfa_l300"] = dfa_alpha1(v)
            o[f"a_{nm}_ktau_ac1_l300"], o[f"a_{nm}_ktau_var_l300"] = rolling_trends(r)
        for nm, col in (("map", "mapp"), ("int", "interval")):
            v = A[col][s300]
            o[f"a_{nm}_sampen_l300"] = sample_entropy(detrend(v))
            o[f"a_{nm}_lfhf_l300"] = lf_hf(t, v)
    return o


def ecg_window(E, e):
    """Summarize ECG intervals, variability, and dynamics while excluding ectopic intervals."""
    o = {}
    sl = _slices(E["t"], e, E["rr"])
    s60, s300 = sl["l60"][0], sl["l300"][0]
    o["e_cover_l60"] = sl["l60"][1]
    nn = {}
    for lb, s in (("l60", s60), ("l300", s300)):
        if s is None:
            continue
        t, rr, ect = E["t"][s], E["rr"][s], E["ectopic"][s]
        tn, rn = t[~ect], rr[~ect]
        if len(rn) < MIN_BEATS[lb] // 2:
            continue
        nn[lb] = rn
        if lb == "l60":
            o["e_hr_l60"] = 60.0 / _med(rn)
            o["e_q_l60"] = _med(E["conf"][s])
        sdnn, rmssd, pnn50, sd12 = hrv(tn, rn)
        o[f"e_sdnn_{lb}"], o[f"e_rmssd_{lb}"], o[f"e_pnn50_{lb}"], o[f"e_sd1sd2_{lb}"] = sdnn, rmssd, pnn50, sd12
        o[f"e_ectopy_{lb}"] = float(ect.mean())
        o[f"e_rsa_{lb}"] = span_variation(rn, relative=False) * 1000.0
        o[f"e_edr_rate_{lb}"] = peak_rate(t, E["ramp"][s])
        o[f"e_hr_slope_{lb}"] = slope_per_min(tn, 60.0 / rn)
        o[f"e_rr_ac1_{lb}"] = ac1(detrend(rn))
        if lb == "l300":
            r = detrend(rn)
            o["e_rr_rsd_l300"] = float(r.std() * 1000.0)
            o["e_rr_rskew_l300"] = skewness(r)
            o["e_rr_dfa_l300"] = dfa_alpha1(rn)
            o["e_rr_ktau_ac1_l300"], o["e_rr_ktau_var_l300"] = rolling_trends(r)
            o["e_rr_sampen_l300"] = sample_entropy(r)
            o["e_rr_lfhf_l300"] = lf_hf(tn, rn)
    if "l60" in nn and "l300" in nn:
        o["e_hr_d60_300"] = 60.0 / _med(nn["l60"]) - 60.0 / _med(nn["l300"])
    return o


def ppg_window(P, e):
    """Summarize PPG morphology and interval dynamics using PPG-only detections."""
    o = {}
    sl = _slices(P["t"], e, P["interval"])
    s60, s300 = sl["l60"][0], sl["l300"][0]
    o["p_cover_l60"] = sl["l60"][1]
    med = {}
    if s60 is not None:
        o["p_rise_l60"] = _med(P["rise"][s60])
        o["p_width_l60"] = _med(P["width50"][s60])
        o["p_hr_l60"] = _med(P["hr"][s60])
        o["p_amp_l60"] = _med(P["amp"][s60])
        o["p_acdc_l60"] = _med(P["acdc"][s60])
        o["p_q_l60"] = _med(P["q"][s60])
        o["p_hr_slope_l60"] = slope_per_min(P["t"][s60], P["hr"][s60])
        o["p_int_amp_corr_l60"] = spearman(P["interval"][s60], P["amp"][s60])
    for lb, s in (("l60", s60), ("l300", s300)):
        if s is None:
            continue
        t, amp, iv = P["t"][s], P["amp"][s], P["interval"][s]
        med[lb] = _med(amp)
        o[f"p_amp_cv_{lb}"] = float(amp.std() / amp.mean()) if amp.mean() > 0 else NAN
        o[f"p_pvi_{lb}"] = span_variation(amp)
        o[f"p_int_sdnn_{lb}"], o[f"p_int_rmssd_{lb}"] = hrv(t, iv)[:2]
        o[f"p_amp_rslope_{lb}"] = slope_per_min(t, amp) / med[lb] if med[lb] > 0 else NAN
        o[f"p_amp_ac1_{lb}"] = ac1(detrend(amp))
    if "l60" in med and "l300" in med and med["l300"] > 0:
        o["p_amp_d60_300"] = med["l60"] / med["l300"] - 1.0
    if s300 is not None:
        t, amp = P["t"][s300], P["amp"][s300]
        o["p_int_ac1_l300"] = ac1(detrend(P["interval"][s300]))
        o["p_amp_dfa_l300"] = dfa_alpha1(amp)
        o["p_amp_lfhf_l300"] = lf_hf(t, amp)
        o["p_amp_ktau_var_l300"] = rolling_trends(detrend(amp))[1]
    return o


def cross_window(E, A, e):
    """Comparison arm only (CROSS-SIGNAL): Spearman of ECG heart rate with ART pulse MAP interpolated at the ECG beats over the last 60 s."""
    se = _slices(E["t"], e, E["rr"])["l60"][0]
    sa = _slices(A["t"], e, A["interval"])["l60"][0]
    if se is None or sa is None:
        return {}
    t, rr = E["t"][se], E["rr"][se]
    m = np.interp(t, A["t"][sa], A["mapp"][sa])
    return {"x_hr_map_corr_l60": spearman(60.0 / rr, m)}


def _arrays(df, kind):
    if df is None or len(df) == 0:
        return None
    if kind == "ecg":
        d = df.loc[df.valid.astype(bool)]
        return {"t": d.t.to_numpy(np.float64), "rr": d.rr.to_numpy(np.float64), "conf": d.conf.to_numpy(np.float64),
                "ramp": d.ramp.to_numpy(np.float64), "ectopic": d.ectopic.to_numpy(bool)} if len(d) else None
    d = df.loc[df.ok.astype(bool)]
    if not len(d):
        return None
    out = {c: d[c].to_numpy(np.float64) for c in d.columns if c not in ("ok",)}
    out["hr"] = 60.0 / out["interval"]
    return out


def self_normalize(df):
    """Add ``<param>_sn``: the value relative to the expanding median of the same case's earlier windows (current included; needs >= SN_MIN_HISTORY)."""
    df = df.sort_values("t_end_sec", kind="stable")
    for c, rule in SN_RULES.items():
        v = df[c].astype(np.float64)
        base = v.expanding(min_periods=SN_MIN_HISTORY).median()
        if rule == "ratio":
            sn = v / base.where(base != 0)
        else:
            sn = v - base
        df[c + "_sn"] = sn.replace([np.inf, -np.inf], np.nan)
    return df


def window_params(ecg, art, ppg, t_end):
    """One row per unique window end with every catalogue column (NaN when a signal or lookback is insufficient, never 0)."""
    e = np.unique(np.asarray(t_end, dtype=np.float64))
    A, E, P = _arrays(art, "art"), _arrays(ecg, "ecg"), _arrays(ppg, "ppg")
    rows = []
    for ei in e:
        o = {"t_end_sec": ei}
        if A is not None:
            o.update(art_window(A, ei))
        if E is not None:
            o.update(ecg_window(E, ei))
        if P is not None:
            o.update(ppg_window(P, ei))
        if A is not None and E is not None:
            o.update(cross_window(E, A, ei))
        rows.append(o)
    raw_cols = [c for c in column_names() if not c.endswith("_sn")]
    df = pd.DataFrame(rows).reindex(columns=["t_end_sec"] + raw_cols)
    df = self_normalize(df)
    return df[["t_end_sec"] + column_names()].reset_index(drop=True)


# catalogue
_ART_DEF = {"sbp": ("systolic pressure (pulse maximum)", "mmHg"), "dbp": ("diastolic pressure (minimum before the upstroke)", "mmHg"),
            "pp": ("pulse pressure SBP - DBP", "mmHg"), "dpdt_max": ("maximum upstroke dP/dt", "mmHg/s"),
            "dpdt_min": ("minimum dP/dt after the peak", "mmHg/s"), "rise": ("foot-to-peak (systolic rise) time", "s"),
            "notch_rel": ("dicrotic notch height (P_notch - DBP)/PP", "ratio"), "ejec": ("foot-to-notch time / pulse interval", "fraction"),
            "area_ratio": ("systolic / diastolic area above DBP (split at the notch)", "ratio"),
            "tau": ("diastolic decay time constant (exponential fit, notch to next foot; R*C surrogate)", "s"),
            "form": ("form factor (foot-to-foot mean - DBP)/PP", "ratio"), "skew": ("skewness of the pulse samples", "unitless"),
            "hr": ("ART pulse rate 60 / foot-to-foot interval", "bpm"), "map": ("foot-to-foot mean pressure", "mmHg"),
            "int": ("foot-to-foot pulse interval", "s")}
_TC_DEF = {"ac1": ("lag-1 autocorrelation of the Gaussian-detrended (30 beats) {q} series", "unitless"),
           "rsd": ("SD of the detrended {q} series", "{u}"), "rskew": ("skewness of the detrended {q} series", "unitless"),
           "dfa": ("DFA alpha1 (4-16 beats) of the {q} series", "unitless"),
           "ktau_ac1": ("Kendall tau of rolling (50 beats, step 10) lag-1 autocorrelation of the detrended {q} series", "unitless"),
           "ktau_var": ("Kendall tau of rolling (50 beats, step 10) variance of the detrended {q} series", "unitless"),
           "sampen": ("sample entropy (m = 2, r = 0.2 SD, last 300 beats) of the detrended {q} series", "unitless"),
           "lfhf": ("log LF(0.04-0.15 Hz)/HF(0.15-0.40 Hz) Lomb-Scargle power of the {q} series", "log ratio")}


def catalogue():
    """List of parameter descriptors in column order (``candidate`` False = quality/meta column or cross-signal comparison arm)."""
    rows = []

    def add(name, signal, family, definition, unit, lookback, norm="raw", independent=True, candidate=True):
        rows.append({"name": name, "signal": signal, "family": family, "definition": definition, "unit": unit, "lookback": lookback,
                     "normalization": norm, "independent": "yes" if independent else "no", "candidate": candidate})

    # ART
    for c in ART_LEVEL + ["hr"]:
        add(f"a_{c}_l60", "A", "level_shape", "median " + _ART_DEF[c][0], _ART_DEF[c][1], "L60")
    add("a_map_l60", "A", "meta", "median foot-to-foot mean pressure (QC vs current MAP)", "mmHg", "L60", candidate=False)
    add("a_q_l60", "A", "meta", "median template correlation of ART pulses (own quality)", "unitless", "L60", candidate=False)
    add("a_cover_l60", "A", "meta", "fraction of the last 60 s covered by valid ART pulses", "fraction", "L60", candidate=False)
    add("a_int_sbp_corr_l60", "A", "within_coupling", "Spearman of pulse interval vs SBP over valid pulses", "unitless", "L60")
    add("a_pp_dbp_corr_l60", "A", "within_coupling", "Spearman of PP vs DBP over valid pulses", "unitless", "L60")
    for lb in ("l60", "l300"):
        L = lb.upper()
        add(f"a_ppv_{lb}", "A", "variability", "PPV: median over 8-beat spans of (PPmax - PPmin)/mean PP", "ratio", L)
        add(f"a_spv_{lb}", "A", "variability", "SPV: median over 8-beat spans of SBPmax - SBPmin", "mmHg", L)
        add(f"a_map_sd_{lb}", "A", "variability", "SD of pulse mean pressure", "mmHg", L)
        add(f"a_pp_cv_{lb}", "A", "variability", "coefficient of variation of PP", "ratio", L)
        add(f"a_int_sdnn_{lb}", "A", "variability", "SD of ART pulse intervals", "ms", L)
        add(f"a_int_rmssd_{lb}", "A", "variability", "RMSSD of adjacent ART pulse intervals", "ms", L)
        add(f"a_map_range_{lb}", "A", "variability", "robust range (p95 - p5) of pulse mean pressure", "mmHg", L)
        for nm in ART_TREND:
            u = {"map": "mmHg/min", "pp": "mmHg/min", "hr": "bpm/min", "dpdt": "mmHg/s/min", "tau": "s/min"}[nm]
            add(f"a_{nm}_slope_{lb}", "A", "trend", f"linear slope of per-pulse {_ART_DEF[nm if nm != 'dpdt' else 'dpdt_max'][0]}", u, L)
        for nm in ART_TC:
            add(f"a_{nm}_ac1_{lb}", "A", "temporal", _TC_DEF["ac1"][0].format(q=nm.upper()), "unitless", L)
    for nm in list(ART_TREND) + ["notch_rel"]:
        u = {"map": "mmHg", "pp": "mmHg", "hr": "bpm", "dpdt": "mmHg/s", "tau": "s", "notch_rel": "ratio"}[nm]
        add(f"a_{nm}_d60_300", "A", "trend", f"median L60 - median L300 of {nm.upper()}", u, "L60-L300")
    for nm in ART_TC:
        unit = {"map": "mmHg", "sbp": "mmHg", "pp": "mmHg", "int": "s"}[nm]
        for k in ("rsd", "rskew", "dfa", "ktau_ac1", "ktau_var"):
            add(f"a_{nm}_{k}_l300", "A", "temporal", _TC_DEF[k][0].format(q=nm.upper()), _TC_DEF[k][1].format(u=unit), "L300")
    for nm in ("map", "int"):
        for k in ("sampen", "lfhf"):
            add(f"a_{nm}_{k}_l300", "A", "temporal", _TC_DEF[k][0].format(q=nm.upper()), _TC_DEF[k][1], "L300")
    # ECG
    add("e_hr_l60", "E", "level_shape", "ECG heart rate 60 / median NN interval", "bpm", "L60")
    add("e_q_l60", "E", "meta", "median R-peak confidence (own quality)", "unitless", "L60", candidate=False)
    add("e_cover_l60", "E", "meta", "fraction of the last 60 s covered by valid R-R intervals", "fraction", "L60", candidate=False)
    for lb in ("l60", "l300"):
        L = lb.upper()
        add(f"e_sdnn_{lb}", "E", "variability", "SDNN of NN intervals (ectopy excluded)", "ms", L)
        add(f"e_rmssd_{lb}", "E", "variability", "RMSSD of adjacent NN intervals", "ms", L)
        add(f"e_pnn50_{lb}", "E", "variability", "share of adjacent NN differences > 50 ms", "fraction", L)
        add(f"e_sd1sd2_{lb}", "E", "variability", "Poincare SD1/SD2", "ratio", L)
        add(f"e_ectopy_{lb}", "E", "variability", "share of R-R intervals outside 0.8-1.2 x causal median of the previous 9", "fraction", L)
        add(f"e_rsa_{lb}", "E", "variability", "respiratory sinus arrhythmia: median over 8-beat spans of NNmax - NNmin", "ms", L)
        add(f"e_edr_rate_{lb}", "E", "variability", "ECG-derived respiration rate (Lomb-Scargle peak 0.1-0.6 Hz of R amplitude)", "breaths/min", L)
        add(f"e_hr_slope_{lb}", "E", "trend", "linear slope of beat-to-beat ECG heart rate", "bpm/min", L)
        add(f"e_rr_ac1_{lb}", "E", "temporal", _TC_DEF["ac1"][0].format(q="RR"), "unitless", L)
    add("e_hr_d60_300", "E", "trend", "ECG heart rate L60 - L300", "bpm", "L60-L300")
    for k in ("rsd", "rskew", "dfa", "ktau_ac1", "ktau_var", "sampen", "lfhf"):
        add(f"e_rr_{k}_l300", "E", "temporal", _TC_DEF[k][0].format(q="RR"), _TC_DEF[k][1].format(u="ms"), "L300")
    # PPG
    add("p_rise_l60", "P", "level_shape", "median PPG foot-to-peak time", "s", "L60")
    add("p_width_l60", "P", "level_shape", "median share of the PPG cycle above 50% amplitude", "fraction", "L60")
    add("p_hr_l60", "P", "level_shape", "median PPG pulse rate", "bpm", "L60")
    add("p_amp_l60", "P", "meta", "median PPG amplitude (device units; only the _sn form is a candidate)", "device units", "L60", candidate=False)
    add("p_acdc_l60", "P", "meta", "median PPG AC/DC (device dependent; only the _sn form is a candidate)", "ratio", "L60", candidate=False)
    add("p_q_l60", "P", "meta", "median template correlation of PPG pulses (own quality)", "unitless", "L60", candidate=False)
    add("p_cover_l60", "P", "meta", "fraction of the last 60 s covered by valid PPG pulses", "fraction", "L60", candidate=False)
    add("p_hr_slope_l60", "P", "trend", "linear slope of PPG pulse rate", "bpm/min", "L60")
    add("p_int_amp_corr_l60", "P", "within_coupling", "Spearman of PPG pulse interval vs amplitude", "unitless", "L60")
    for lb in ("l60", "l300"):
        L = lb.upper()
        add(f"p_amp_cv_{lb}", "P", "variability", "coefficient of variation of PPG amplitude", "ratio", L)
        add(f"p_pvi_{lb}", "P", "variability", "PVI-like: median over 8-beat spans of (AMPmax - AMPmin)/mean", "ratio", L)
        add(f"p_int_sdnn_{lb}", "P", "variability", "SD of PPG pulse intervals", "ms", L)
        add(f"p_int_rmssd_{lb}", "P", "variability", "RMSSD of adjacent PPG pulse intervals", "ms", L)
        add(f"p_amp_rslope_{lb}", "P", "trend", "slope of PPG amplitude / median amplitude", "1/min", L)
        add(f"p_amp_ac1_{lb}", "P", "temporal", _TC_DEF["ac1"][0].format(q="PPG amplitude"), "unitless", L)
    add("p_amp_d60_300", "P", "trend", "PPG amplitude median L60 / median L300 - 1", "ratio", "L60-L300")
    add("p_int_ac1_l300", "P", "temporal", _TC_DEF["ac1"][0].format(q="PPG interval"), "unitless", "L300")
    add("p_amp_dfa_l300", "P", "temporal", _TC_DEF["dfa"][0].format(q="PPG amplitude"), "unitless", "L300")
    add("p_amp_lfhf_l300", "P", "temporal", _TC_DEF["lfhf"][0].format(q="PPG amplitude"), "log ratio", "L300")
    add("p_amp_ktau_var_l300", "P", "temporal", _TC_DEF["ktau_var"][0].format(q="PPG amplitude"), "unitless", "L300")
    # self-normalized versions
    by_name = {r["name"]: r for r in rows}
    for c, rule in SN_RULES.items():
        base = by_name[c]
        how = "ratio to" if rule == "ratio" else "difference from"
        add(c + "_sn", base["signal"], "level_shape_sn", f"{base['definition']}, {how} the case's expanding median (causal, >= 5 min)",
            "ratio" if rule == "ratio" else base["unit"], base["lookback"], norm=f"self_{rule}")
    # comparison arm (cross-signal)
    add("x_hr_map_corr_l60", "E+A", "cross", "Spearman of ECG heart rate vs ART pulse MAP interpolated at the R peaks", "unitless", "L60",
        independent=False, candidate=False)
    return rows


def column_names():
    """Return the ordered names of all parameter and quality fields in the catalogue."""
    return [r["name"] for r in catalogue()]


def candidate_names(signal=None):
    """Return forecasting candidates, optionally restricted to one signal."""
    return [r["name"] for r in catalogue() if r["candidate"] and (signal is None or r["signal"] == signal)]


# build
def native_dir(root, dataset, case_id):
    """Resolve the directory containing native detections for one dataset and case."""
    return Path(root) / OUT_DIR / "native" / str(dataset) / f"case_id={case_id}"


def _write_native(df, path):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == np.float64 and c != "t":
            out[c] = out[c].astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, compression="zstd", index=False)


def case_job(args):
    """Read ONE case once (time span of its windows plus the lookback), build the three native series and the window parameters."""
    root, dataset, case_id, t_end, write_native = args
    from .bshi_raw import waveform_path
    t0 = time.perf_counter()
    info = {"dataset": dataset, "case_id": case_id, "windows": int(len(np.unique(t_end))), "error": ""}
    try:
        starts, ends = chunk_plan(t_end)
        lo, hi = float(starts.min() - PREROLL - 1.0), float(ends.max())
        path = waveform_path(root, dataset, case_id)
        nat = {}
        for sig in ("ecg", "art", "ppg"):
            ts, x = read_signal(path, sig, lo, hi)
            nat[sig] = native_ecg(ts, x, starts, ends) if sig == "ecg" else native_pulses(ts, x, starts, ends, kind=sig)
            info[f"n_{sig}"] = int(len(nat[sig]))
            info[f"n_{sig}_ok"] = int(nat[sig]["valid" if sig == "ecg" else "ok"].sum()) if len(nat[sig]) else 0
        info["t_native"] = time.perf_counter() - t0
        if write_native:
            d = native_dir(root, dataset, case_id)
            for sig, df in nat.items():
                _write_native(df, d / f"{sig}.parquet")
        params = window_params(nat["ecg"], nat["art"], nat["ppg"], t_end)
        params.insert(0, "case_id", case_id)
        params.insert(0, "dataset", dataset)
        info["t_total"] = time.perf_counter() - t0
        return params, info
    except Exception as exc:                                             # recorded per case; the build continues
        info["error"] = f"{type(exc).__name__}: {exc}"[:300]
        info["t_total"] = time.perf_counter() - t0
        return None, info


def _sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, workers=28, cases=None, write_native=True, out_name=PARAMS_NAME, log=None):
    """Window parameter table for every ea_core window (``cases`` = optional list of (dataset, case_id) for timing runs)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from .bshi import load_window_index

    root = Path(root)
    t_start = time.time()
    idx = load_window_index(root, "ea_core", columns=["window_id", "dataset", "case_id", "t_end_sec"])
    if cases is not None:
        want = pd.DataFrame(list(cases), columns=["dataset", "case_id"]).astype({"case_id": idx.case_id.dtype})
        idx = idx.merge(want, on=["dataset", "case_id"])
    groups = [(ds, cid, g.t_end_sec.to_numpy(np.float64)) for (ds, cid), g in idx.groupby(["dataset", "case_id"], sort=False)]
    groups.sort(key=lambda g: -(g[2].max() - g[2].min()))                   # longest first for load balance
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = "1"
    say = (lambda m: print(m, file=log, flush=True)) if log else print
    parts, infos = [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(case_job, (str(root), ds, cid, te, write_native)) for ds, cid, te in groups]
        for i, f in enumerate(as_completed(futs), 1):
            p, info = f.result()
            infos.append(info)
            if p is not None:
                parts.append(p)
            if i % 100 == 0 or i == len(futs):
                say(f"[{time.time() - t_start:7.0f}s] cases {i}/{len(futs)} failed {sum(bool(x['error']) for x in infos)}")
    info_df = pd.DataFrame(infos)
    params = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["dataset", "case_id", "t_end_sec"] + column_names())
    params["case_id"] = params.case_id.astype(idx.case_id.dtype)
    table = idx.merge(params, on=["dataset", "case_id", "t_end_sec"], how="left", validate="many_to_one")
    for c in column_names():
        table[c] = table[c].astype(np.float32)
    out_dir = root / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / out_name, compression="zstd", index=False)
    info_df.to_csv(out_dir / (Path(out_name).stem + "_cases.csv"), index=False)
    manifest = {"version": VERSION, "rows": int(len(table)), "cases": int(len(groups)), "failed_cases": int((info_df.error != "").sum()),
                "build_sec": round(time.time() - t_start, 1), "code_sha256": _sha(Path(__file__)),
                "config_sha256": _sha(root / "configs" / "hemodynamics.json"), "columns": column_names(),
                "params": {"preroll_s": PREROLL, "latency_s": LATENCY, "lookbacks_s": LOOKBACKS, "min_cover": MIN_COVER, "min_beats": MIN_BEATS,
                           "sn_min_history_windows": SN_MIN_HISTORY, "detrend_sigma_beats": DETREND_SIGMA},
                "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if out_name == PARAMS_NAME:
        (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return table, info_df, manifest


def write_catalogue(root):
    """Save parameter definitions, units, lookbacks, and candidate status as a CSV table."""
    out = Path(root) / "results" / "independence"
    out.mkdir(parents=True, exist_ok=True)
    cat = pd.DataFrame(catalogue())
    cat.to_csv(out / "table_param_catalogue.csv", index=False)
    return cat
