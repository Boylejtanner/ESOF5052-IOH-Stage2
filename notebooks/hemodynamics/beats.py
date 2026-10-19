"""Beat detection, cross-signal timing, morphology and signal-quality indices for ECG / ART / PPG (pure numpy/scipy, array in -> arrays out).

Resolution limit: the case files are on a 100 Hz grid (10 ms). Timing features use sub-sample interpolation (parabolic R refinement,
intersecting-tangent foot with linear sub-sample position), which in the synthetic tests recovers delays to a few ms, but absolute
accuracy on real data is bounded by the 100 Hz resampling (VitalDB was 500 Hz natively; MOVER ECG was 300 Hz).

Method
* ECG R peaks: per contiguous finite segment, 5-25 Hz band-pass (zero-phase), squared-derivative envelope with a block-adaptive threshold
  and 0.28 s refractory period, R located on the filtered signal and refined by a parabola; the ECG is stored R-positive.
  Confidence = 1 - exp(-amp / (6 * local noise MAD)).
* ART / PPG pulses are detected independently of the ECG (a pulse can arrive after the next R at high heart rate): low-pass, peaks gated by a
  LOCAL amplitude, diastolic minimum between peaks, foot = intersection of the maximum-upslope tangent with the diastolic level. PPG polarity is
  decided once per case so that its systolic upstroke is the steeper edge; ECG polarity so that the larger QRS deflection is positive.
* Association: pulses are matched to R waves around a per-channel delay ESTIMATED from the data (the delay, not the R-R interval, is stable
  from beat to beat; oximeter latency of 0.4-0.6 s is common and can exceed the R-R interval); each R takes the foot nearest to R + delay
  within +-0.12 s; an R without one is MASKED (timing NaN, flag ``assoc_failed``), never forced.
* One row per R-R cycle. Waveforms are resampled to a normalized cardiac phase over the R-R cycle (ECG and PPG z-scored per beat, ART in mmHg).
* SQIs in [0, 1]: template correlation against the median template of a block of ~30 beats, combined with detection confidence, plausibility
  and cross-signal consistency.
"""
from __future__ import annotations
# Detect cardiac landmarks and describe pulse shape and signal quality.

from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt

FS = 100.0
FLAGS = {"gap_overlap": 1, "art_missing": 2, "ppg_missing": 4, "ecg_low_conf": 8, "art_flatline": 16, "art_clipped": 32,
         "art_implausible": 64, "assoc_failed": 128}
BEAT_COLUMNS = ["beat_idx", "t_r", "rr_sec", "hr_bpm", "ecg_conf", "art_foot_t", "art_sys_t", "sbp", "dbp", "pp", "map_beat", "dpdt_max", "sys_time",
                "art_area", "ppg_foot_t", "ppg_peak_t", "ppg_amp", "ppg_ac_dc", "ppg_upstroke", "ptt_ecg_art", "ptt_ecg_ppg", "ptt_art_ppg",
                "art_ppg_corr", "art_int_sec", "hr_art_bpm", "sqi_ecg", "sqi_art", "sqi_ppg", "sqi_cross", "flags", "valid_beat", "art_ok", "ppg_ok"]
_F32 = [c for c in BEAT_COLUMNS if c not in ("beat_idx", "t_r", "art_foot_t", "art_sys_t", "ppg_foot_t", "ppg_peak_t", "flags", "valid_beat", "art_ok", "ppg_ok")]
_F64 = ["t_r", "art_foot_t", "art_sys_t", "ppg_foot_t", "ppg_peak_t"]
MIN_SEG_SEC = 2.5


@lru_cache(maxsize=16)
def _sos(kind, fs, lo, hi):
    if kind == "band":
        return butter(2, [lo, hi], btype="bandpass", fs=fs, output="sos")
    if kind == "high":
        return butter(2, lo, btype="highpass", fs=fs, output="sos")
    return butter(2, hi, btype="lowpass", fs=fs, output="sos")


def finite_segments(x, min_len):
    # Exclude or flag missing values before using these measurements.
    f = np.isfinite(x)
    d = np.diff(np.concatenate(([0], f.astype(np.int8), [0])))
    s, e = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    keep = (e - s) >= min_len
    return s[keep], e[keep]


def _parabolic(y, i):
    if i <= 0 or i >= len(y) - 1:
        return float(i)
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    den = y0 - 2 * y1 + y2
    return float(i) if den == 0 else float(i + np.clip(0.5 * (y0 - y2) / den, -1, 1))


def _block_threshold(env, fs, frac=0.3, block_sec=8.0):
    n, block = len(env), int(block_sec * fs)
    if n <= block:
        return np.full(n, frac * np.percentile(env, 97))
    centers, values = [], []
    for a in range(0, n, block):
        seg = env[a:a + block]
        if len(seg) < block // 2 and centers:
            break
        centers.append(a + len(seg) / 2)
        values.append(frac * np.percentile(seg, 97))
    return np.interp(np.arange(n), centers, values)


# orientation
def _sample_segments(x, fs, max_samples=400000):
    starts, ends = finite_segments(x, int(MIN_SEG_SEC * fs))
    total, out = 0, []
    stride = max(1, len(starts) // 400)
    for a, b in list(zip(starts, ends))[::stride]:
        out.append(x[a:b].astype(np.float64))
        total += b - a
        if total >= max_samples:
            break
    return out


def ecg_orientation(ecg, fs=FS):
    """+1 when the largest QRS deflection of the (baseline-removed, otherwise UNFILTERED) ECG is positive, else -1.

    Two traps this avoids (both seen on real MOVER recordings): (1) deciding on the 5-25 Hz band-passed signal, which attenuates a narrow
    tall R more than a broad opposite lobe; (2) taking the maximum of per-segment extremes across the case, so one corrupted segment
    (lead-off / artefact burst) outvotes hundreds of clean ones. Each segment votes, weighted by its length."""
    high = _sos("high", fs, 0.5, 0.0)
    score = 0.0
    for x in _sample_segments(ecg, fs):
        f = sosfiltfilt(high, x - np.median(x))
        score += len(x) * (1.0 if np.percentile(f, 99.5) >= -np.percentile(f, 0.5) else -1.0)
    return 1.0 if score >= 0 else -1.0


def orientation(ecg, ppg, fs=FS):
    """(ecg_sign, ppg_polarity) decided ONCE per case: ECG so that the larger QRS deflection is positive, PPG so that the systolic
    upstroke (the steeper edge) is positive. Per-region decisions can flip within a case, so callers must pass these on."""
    low = _sos("low", fs, 0.0, 12.0)
    score = 0.0
    for x in _sample_segments(ppg, fs):
        d = np.gradient(sosfiltfilt(low, x - np.median(x))) * fs
        score += len(x) * (1.0 if np.percentile(d, 99.5) >= -np.percentile(d, 0.5) else -1.0)
    return ecg_orientation(ecg, fs), (1.0 if score >= 0 else -1.0)


# ECG
def detect_r_peaks(ecg, fs=FS, sign=None):
    """Return (r_idx float sample positions, confidence in [0,1], polarity +1/-1)."""
    # Detect peaks within continuous ECG segments and avoid joining across capture gaps.
    starts, ends = finite_segments(ecg, int(MIN_SEG_SEC * fs))
    sos = _sos("band", fs, 5.0, 25.0)
    filt = []
    for a, b in zip(starts, ends):
        x = ecg[a:b].astype(np.float64)
        filt.append((a, sosfiltfilt(sos, x - np.median(x))))
    if not filt:
        return np.array([]), np.array([]), 1.0 if sign is None else sign
    if sign is None:
        sign = ecg_orientation(ecg, fs)
    idx, conf = [], []
    for a, f in filt:
        # Measure the signal derivative to locate its rapid changes.
        g = sign * f
        d = np.gradient(g)
        env = uniform_filter1d(d * d, size=max(3, int(0.08 * fs)), mode="nearest")
        if not np.isfinite(env).all() or env.max() <= 0:
            continue
        peaks, _ = find_peaks(env, height=_block_threshold(env, fs), distance=int(0.28 * fs))
        last = -10 ** 9
        # Process the selected records in order and retain their individual results.
        for p in peaks:
            lo, hi = max(0, p - int(0.08 * fs)), min(len(g), p + int(0.05 * fs) + 1)
            r = lo + int(np.argmax(g[lo:hi]))
            if r == last:
                continue
            last = r
            w0, w1 = max(0, r - int(2 * fs)), min(len(g), r + int(2 * fs))
            # Use the local median absolute deviation to judge a peak against nearby noise.
            noise = 1.4826 * np.median(np.abs(g[w0:w1] - np.median(g[w0:w1]))) + 1e-9
            amp = g[r] - np.median(g[max(0, r - int(0.5 * fs)):min(len(g), r + int(0.5 * fs))])
            c = 1.0 - np.exp(-max(amp, 0.0) / (6.0 * noise))
            if r < int(0.15 * fs) or r > len(g) - int(0.15 * fs):
                c = 0.0                                                  # QRS truncated by the segment edge: cannot be validated
            idx.append(a + _parabolic(g, r))
            conf.append(c)
    order = np.argsort(idx)
    return np.asarray(idx)[order], np.asarray(conf)[order], sign


# ART / PPG
def detect_pulses(x, fs=FS, kind="art", polarity=1.0):
    """Pulse feet/peaks. Returns dict of arrays (sample positions float; ``top``/``base`` raw levels; ``dpdt`` units/s).

    ``polarity`` orients the signal (PPG: see ``orientation``). Pulse acceptance uses a LOCAL amplitude (2 s max-min, smoothed over 4 s),
    never the amplitude of a whole contiguous segment: hours-long VitalDB segments include slow pressure swings that would otherwise
    reject genuine small pulses.
    """
    # Find pulse landmarks using local amplitude and noise rather than whole-case scaling.
    empty = {k: np.array([]) for k in ("foot", "peak", "top", "base", "dpdt")}
    if int(np.isfinite(x).sum()) < int(MIN_SEG_SEC * fs):
        return empty
    sos = _sos("low", fs, 0.0, 20.0 if kind == "art" else 12.0)
    rows = {k: [] for k in empty}
    starts, ends = finite_segments(x, int(MIN_SEG_SEC * fs))
    win, smooth = max(3, int(2.0 * fs)), max(3, int(4.0 * fs))
    for a, b in zip(starts, ends):
        # Filter the continuous signal before estimating its peak positions.
        raw = polarity * x[a:b].astype(np.float64)
        y = sosfiltfilt(sos, raw - np.median(raw))
        resid = raw - np.median(raw) - y
        # Use the local median absolute deviation to judge a peak against nearby noise.
        noise = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
        min_amp = 3.0 if kind == "art" else 6.0 * noise
        local = uniform_filter1d(maximum_filter1d(y, win) - minimum_filter1d(y, win), smooth, mode="nearest")
        peaks, props = find_peaks(y, distance=int(0.28 * fs), prominence=min_amp * 0.5)
        keep = (props["prominences"] >= 0.4 * local[peaks]) & (local[peaks] >= min_amp)
        peaks = peaks[keep]
        d = np.gradient(y) * fs
        # Process the selected records in order and retain their individual results.
        for k, P in enumerate(peaks):
            w0 = peaks[k - 1] if k > 0 else max(0, P - int(0.7 * fs))
            if P - w0 < 5:
                continue
            m = w0 + int(np.argmin(y[w0:P + 1]))
            if k == 0 and m == w0 == 0:
                continue                                              # onset lies before the segment
            u = m + int(np.argmax(d[m:P + 1]))
            if d[u] <= 0 or y[P] - y[m] < min_amp * 0.5:
                continue
            u_f = _parabolic(d, u)
            # Project the steepest upstroke back to the pulse baseline to estimate its foot.
            foot = float(np.clip(u_f - (y[u] - y[m]) * fs / d[u], m, u_f))
            rows["foot"].append(a + foot)
            rows["peak"].append(a + P)
            rows["top"].append(raw[max(0, P - 2):P + 3].max())
            rows["base"].append(raw[max(0, m - 2):m + 3].min())
            rows["dpdt"].append(d[u])
    out = {k: np.asarray(v, dtype=float) for k, v in rows.items()}
    order = np.argsort(out["foot"]) if len(out["foot"]) else np.array([], dtype=int)
    return {k: v[order] for k, v in out.items()}


def _delay_support(r_s, foot_s, taus, tol):
    out = np.empty(len(taus), dtype=int)
    for i, tau in enumerate(taus):
        # Locate the requested interval within the sorted sample times.
        target = r_s + tau
        j = np.searchsorted(foot_s, target - tol)
        jj = np.minimum(j, len(foot_s) - 1)
        out[i] = np.count_nonzero((j < len(foot_s)) & (foot_s[jj] <= target + tol))
    return out


def estimate_delay(r_s, foot_s, min_delay=0.04, hi=1.4, step=0.01, tol=0.06, min_share=0.4):
    """Dominant R -> pulse-foot delay (s): the delay at which most R waves have a foot within +-tol (smallest delay >= min_delay among near-ties).

    Pulse-oximeter and some arterial outputs carry device latency (0.4-0.6 s after the R wave is common), which can exceed the R-R
    interval, so a fixed 'first foot after R' window is wrong; the delay is stable from beat to beat while the R-R interval varies.
    When the delay exceeds the R-R interval, 'pulse k follows R k' and 'pulse k follows R k+1 with a shorter delay' fit almost equally;
    ``min_delay`` (e.g. the ART delay + 0.02 s for the PPG) resolves it. Estimates at the search limit are treated as 'no relationship'.
    """
    if len(r_s) < 5 or len(foot_s) < 5:
        return None
    taus = np.arange(min_delay, hi, step)
    support = _delay_support(r_s, foot_s, taus, tol)
    best = support.max()
    if best < min_share * len(r_s):
        return None
    tau = taus[int(np.flatnonzero(support >= 0.95 * best)[0])]
    if tau >= hi - 0.1:
        return None
    target = r_s + tau
    j = np.searchsorted(foot_s, target - tol)
    jj = np.minimum(j, len(foot_s) - 1)
    hit = (j < len(foot_s)) & (np.abs(foot_s[jj] - target) <= tol)
    return float(np.median(foot_s[jj][hit] - r_s[hit])) if hit.any() else float(tau)


def associate(r_s, foot_s, tol=0.12, block_sec=300.0, min_delay=0.04):
    """Assign to each R (seconds) the pulse foot nearest to R + delay (delay re-estimated per ``block_sec`` block, global fallback).

    Returns (index of the foot or -1, delay used). An R without a foot within +-tol of its expected arrival is MASKED, never forced;
    a foot serves at most one R.
    """
    # Match each ECG beat to the nearest plausible pulse without reusing a pulse.
    out = np.full(len(r_s), -1)
    delay = np.full(len(r_s), np.nan)
    if len(foot_s) == 0 or len(r_s) == 0:
        return out, delay
    tau_global = estimate_delay(r_s, foot_s, min_delay)
    if tau_global is None:
        return out, delay
    blocks = np.floor((r_s - r_s[0]) / block_sec).astype(int)
    for b in np.unique(blocks):
        # Locate the requested interval within the sorted sample times.
        sel = np.flatnonzero(blocks == b)
        near = foot_s[(foot_s >= r_s[sel[0]] - 2.0) & (foot_s <= r_s[sel[-1]] + 3.5)]
        tau = estimate_delay(r_s[sel], near, min_delay) if len(sel) >= 30 else None
        # Locate the requested interval within the sorted sample times.
        tau = tau_global if tau is None else tau
        target = r_s[sel] + tau
        j = np.searchsorted(foot_s, target)
        lo, hi = np.clip(j - 1, 0, len(foot_s) - 1), np.clip(j, 0, len(foot_s) - 1)
        pick = np.where(np.abs(foot_s[lo] - target) <= np.abs(foot_s[hi] - target), lo, hi)
        ok = np.abs(foot_s[pick] - target) <= tol
        out[sel[ok]] = pick[ok]
        delay[sel] = tau
    taken = out >= 0
    if taken.any():
        order = np.flatnonzero(taken)
        _, first = np.unique(out[order], return_index=True)
        dup = np.setdiff1d(np.arange(len(order)), first)
        for i in dup:                                                  # keep the R whose expected arrival is closest to the shared foot
            others = order[out[order] == out[order[i]]]
            best = others[np.argmin(np.abs(foot_s[out[others]] - (r_s[others] + delay[others])))]
            out[others[others != best]] = -1
    return out, delay


def phase_matrix(x, r0, r1, n_phase, chunk=8192):
    """Linear resampling of each cycle [r0, r1) (sample positions) to ``n_phase`` points; NaN if any neighbour sample is missing."""
    out = np.full((len(r0), n_phase), np.nan, dtype=np.float32)
    grid = np.arange(n_phase) / n_phase
    for a in range(0, len(r0), chunk):
        # Calculate the acceptance flags and output values for the following step.
        pos = r0[a:a + chunk, None] + (r1[a:a + chunk] - r0[a:a + chunk])[:, None] * grid[None, :]
        i0 = np.floor(pos).astype(np.int64)
        ok = (i0 >= 0) & (i0 + 1 < len(x))
        i0c = np.clip(i0, 0, len(x) - 2)
        w = pos - i0c
        out[a:a + chunk] = np.where(ok, x[i0c] * (1 - w) + x[i0c + 1] * w, np.nan)
    return out


def _zscore(m):
    mu = np.nanmean(m, axis=1, keepdims=True)
    sd = np.nanstd(m, axis=1, keepdims=True)
    return (m - mu) / np.where(sd > 1e-9, sd, np.nan)


def template_corr(z, block=30):
    """Pearson correlation of each z-scored beat with the median template of its block of beats (NaN when too few valid beats)."""
    out = np.full(len(z), np.nan, dtype=np.float32)
    for a in range(0, len(z), block):
        zb = z[a:a + block]
        good = np.isfinite(zb).all(axis=1)
        if good.sum() < 5:
            continue
        templ = np.median(zb[good], axis=0)
        templ = (templ - templ.mean()) / (templ.std() + 1e-9)
        out[a:a + block] = np.where(good, np.nanmean(zb * templ[None, :], axis=1), np.nan)
    return out


def _empty_frame():
    return pd.DataFrame({c: pd.Series(dtype="float64" if c in _F64 else "float32" if c in _F32 else "int64" if c in ("beat_idx", "flags") else "bool") for c in BEAT_COLUMNS})


def analyze(ecg, art, ppg, t_origin=0.0, fs=FS, n_phase=200, ecg_sign=None, ppg_pol=None):
    """Detect and describe every R-R cycle of one region. Arrays share one uniform grid; sample 0 is at clock time ``t_origin``.

    ``ecg_sign`` / ``ppg_pol`` come from ``orientation`` on the whole case (decided from the region when omitted).
    Returns (beats DataFrame with BEAT_COLUMNS, waveforms dict of float32 (n, n_phase) arrays for ecg/art/ppg, meta dict).
    """
    ecg, art, ppg = (np.asarray(x, dtype=np.float64) for x in (ecg, art, ppg))
    if ecg_sign is None or ppg_pol is None:
        o_ecg, o_ppg = orientation(ecg, ppg, fs)
        ecg_sign = o_ecg if ecg_sign is None else ecg_sign
        ppg_pol = o_ppg if ppg_pol is None else ppg_pol
    r_idx, r_conf, ecg_sign = detect_r_peaks(ecg, fs, sign=ecg_sign)
    meta = {"ecg_polarity": ecg_sign, "ppg_polarity": ppg_pol}
    if len(r_idx) < 3:
        return _empty_frame(), {c: np.zeros((0, n_phase), dtype=np.float32) for c in ("ecg", "art", "ppg")}, meta
    # Calculate pulse pressure and the observation count for the following step.
    ap = detect_pulses(art, fs, "art")
    pp = detect_pulses(ppg, fs, "ppg", polarity=ppg_pol)
    n = len(r_idx) - 1
    r0, r1 = r_idx[:-1], r_idx[1:]
    # Exclude or flag missing values before using these measurements.
    rr = (r1 - r0) / fs
    r_s = r_idx / fs
    ja_all, delay_a = associate(r_s, ap["foot"] / fs)
    # Exclude or flag missing values before using these measurements.
    art_delay = float(np.nanmedian(delay_a)) if np.isfinite(delay_a).any() else 0.04
    jp_all, _ = associate(r_s, pp["foot"] / fs, min_delay=max(0.04, art_delay + 0.02))   # the oximeter foot follows the arterial foot of the same beat
    ja, jp = ja_all[:n], jp_all[:n]
    wave_e_raw = phase_matrix(ecg_sign * ecg, r0, r1, n_phase)
    wave_a = phase_matrix(art, r0, r1, n_phase)
    wave_p_raw = phase_matrix(ppg_pol * ppg, r0, r1, n_phase)
    fin_e, fin_a, fin_p = (np.isfinite(w).all(axis=1) for w in (wave_e_raw, wave_a, wave_p_raw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _describe(n, r_idx, r_conf, r0, r1, rr, ja, jp, ap, pp, ppg, ppg_pol, wave_e_raw, wave_a, wave_p_raw, fin_e, fin_a, fin_p,
                         t_origin, fs, meta)


def _describe(n, r_idx, r_conf, r0, r1, rr, ja, jp, ap, pp, ppg, ppg_pol, wave_e_raw, wave_a, wave_p_raw, fin_e, fin_a, fin_p, t_origin, fs, meta):
    # Calculate beat-level physiology and quality only where the relevant samples are usable.
    has_a, has_p = ja >= 0, jp >= 0
    take = lambda arr, j, has: np.where(has, arr[np.maximum(j, 0)] if len(arr) else np.nan, np.nan)
    a_foot, a_peak = take(ap["foot"], ja, has_a), take(ap["peak"], ja, has_a)
    # Calculate systolic pressure and diastolic pressure for the following step.
    sbp, dbp, dpdt = take(ap["top"], ja, has_a), take(ap["base"], ja, has_a), take(ap["dpdt"], ja, has_a)
    p_foot, p_peak = take(pp["foot"], jp, has_p), take(pp["peak"], jp, has_p)
    p_top, p_base, p_dpdt = take(pp["top"], jp, has_p), take(pp["base"], jp, has_p), take(pp["dpdt"], jp, has_p)
    ze, za, zp = _zscore(wave_e_raw), _zscore(wave_a), _zscore(wave_p_raw)
    c_e, c_a, c_p = template_corr(ze), template_corr(za), template_corr(zp)
    wave_e, wave_p = ze.astype(np.float32), zp.astype(np.float32)

    if True:
        rr_med = pd.Series(rr).rolling(9, center=True, min_periods=3).median().to_numpy()
        ecg_low = (r_conf[:-1] < 0.3) | (rr < 0.286) | (rr > 2.0) | (rr > 1.6 * rr_med)
        pp_art = sbp - dbp
        art_std = np.nanstd(wave_a, axis=1)
        flat = fin_a & (art_std < 0.5)
        near_top = (np.abs(wave_a - np.nanmax(wave_a, axis=1, keepdims=True)) < 0.1).mean(axis=1)
        near_bot = (np.abs(wave_a - np.nanmin(wave_a, axis=1, keepdims=True)) < 0.1).mean(axis=1)
        # Exclude or flag missing values before using these measurements.
        clipped = fin_a & ~flat & ((near_top >= 0.15) | (near_bot >= 0.15))
        implausible = has_a & ((pp_art < 15) | (pp_art > 150) | (sbp > 300) | (dbp < -20))
        ppg_amp = p_top - p_base
        # Exclude or flag missing values before using these measurements.
        ppg_mean = np.nanmean(wave_p_raw, axis=1)
        resid = ppg_pol * ppg
        resid = resid[np.isfinite(resid)]
        noise_p = 1.4826 * np.median(np.abs(np.diff(resid))) / np.sqrt(2) + 1e-9 if len(resid) > 1 else 1e-9
        # Reserve output arrays and masks with explicit initial values.
        ppg_factor = np.clip(ppg_amp / (6.0 * noise_p), 0.0, 1.0)
        ptt_a, ptt_p = (a_foot - r0) / fs, (p_foot - r0) / fs
        art_int = np.full(n, np.nan)
        consecutive = has_a[:-1] & has_a[1:] & (ja[1:] == ja[:-1] + 1)
        art_int[:-1] = np.where(consecutive, (a_foot[1:] - a_foot[:-1]) / fs, np.nan)
        agree = np.clip(1 - (np.abs(art_int - rr) / rr - 0.05) / 0.10, 0, 1)
        comps = np.vstack([np.where(has_a, ((ptt_a >= 0.05) & (ptt_a <= 0.60)).astype(float), np.nan),
                           np.where(has_p, ((ptt_p >= 0.05) & (ptt_p <= 1.20)).astype(float), np.nan), agree])
        n_comp = (~np.isnan(comps)).sum(axis=0)
        sqi_cross = np.where(n_comp > 0, np.nansum(comps, axis=0) / np.maximum(n_comp, 1), 0.0)
        art_ok = fin_a & has_a & ~flat & ~clipped & ~implausible
        sqi_art = np.where(art_ok, np.clip(np.nan_to_num(c_a), 0, 1), 0.0)
        sqi_ppg = np.where(fin_p & has_p, np.clip(np.nan_to_num(c_p), 0, 1) * ppg_factor, 0.0)
        sqi_ecg = np.where(fin_e, np.clip(np.nan_to_num(c_e), 0, 1) * r_conf[:-1], 0.0)
        ppg_ok = fin_p & has_p & (sqi_ppg > 0)
        # Collect the per-record values into a table for comparison.
        flags = (np.where(~fin_e, FLAGS["gap_overlap"], 0) | np.where(~fin_a, FLAGS["art_missing"], 0) | np.where(~fin_p, FLAGS["ppg_missing"], 0)
                 | np.where(ecg_low, FLAGS["ecg_low_conf"], 0) | np.where(flat, FLAGS["art_flatline"], 0) | np.where(clipped, FLAGS["art_clipped"], 0)
                 | np.where(implausible, FLAGS["art_implausible"], 0) | np.where((fin_a & ~has_a) | (fin_p & ~has_p), FLAGS["assoc_failed"], 0))
        # Collect the per-record values into a table for comparison.
        valid = fin_e & ~ecg_low
        keep_a, keep_p = art_ok, has_p & fin_p
        map_beat = np.where(fin_a, np.nanmean(wave_a, axis=1), np.nan)
        df = pd.DataFrame({
            "beat_idx": np.arange(n), "t_r": t_origin + r0 / fs, "rr_sec": rr, "hr_bpm": 60.0 / rr, "ecg_conf": r_conf[:-1],
            "art_foot_t": np.where(keep_a, t_origin + a_foot / fs, np.nan), "art_sys_t": np.where(keep_a, t_origin + a_peak / fs, np.nan),
            "sbp": np.where(keep_a, sbp, np.nan), "dbp": np.where(keep_a, dbp, np.nan), "pp": np.where(keep_a, pp_art, np.nan),
            "map_beat": np.where(fin_a & ~flat, map_beat, np.nan), "dpdt_max": np.where(keep_a, dpdt, np.nan),
            "sys_time": np.where(keep_a, (a_peak - a_foot) / fs, np.nan), "art_area": np.where(keep_a, (map_beat - dbp) * rr, np.nan),
            "ppg_foot_t": np.where(keep_p, t_origin + p_foot / fs, np.nan), "ppg_peak_t": np.where(keep_p, t_origin + p_peak / fs, np.nan),
            "ppg_amp": np.where(keep_p, ppg_amp, np.nan),
            "ppg_ac_dc": np.where(keep_p, ppg_amp / np.where(np.abs(ppg_mean) > 1e-9, np.abs(ppg_mean), np.nan), np.nan),
            "ppg_upstroke": np.where(keep_p, p_dpdt / np.where(ppg_amp > 0, ppg_amp, np.nan), np.nan),
            "ptt_ecg_art": np.where(keep_a, ptt_a, np.nan), "ptt_ecg_ppg": np.where(keep_p, ptt_p, np.nan),
            "ptt_art_ppg": np.where(keep_a & keep_p, (p_foot - a_foot) / fs, np.nan),
            "art_ppg_corr": np.where(fin_a & fin_p, np.nanmean(za * zp, axis=1), np.nan),
            "art_int_sec": np.where(keep_a, art_int, np.nan), "hr_art_bpm": np.where(keep_a, 60.0 / art_int, np.nan),
            "sqi_ecg": sqi_ecg, "sqi_art": sqi_art, "sqi_ppg": sqi_ppg, "sqi_cross": sqi_cross,
            "flags": flags.astype(np.int64), "valid_beat": valid, "art_ok": art_ok, "ppg_ok": ppg_ok})
    for c in _F32:
        df[c] = df[c].astype(np.float32)
    return df[BEAT_COLUMNS], {"ecg": wave_e, "art": wave_a, "ppg": wave_p}, meta


def merge_spans(t_start, t_end, pad=0.0):
    """Union of [t_start, t_end] intervals as (start, end) arrays; touching/overlapping intervals merge."""
    # Order the measurements before matching intervals or comparing rows.
    order = np.argsort(t_start)
    s, e = np.asarray(t_start, dtype=float)[order], np.asarray(t_end, dtype=float)[order]
    out_s, out_e = [], []
    for a, b in zip(s, e):
        if out_e and a <= out_e[-1] + pad:
            out_e[-1] = max(out_e[-1], b)
        else:
            out_s.append(a)
            out_e.append(b)
    return np.asarray(out_s), np.asarray(out_e)


def process_case(time_sec, art, ecg, ppg, win_start, win_end, fs=FS, n_phase=200, pad_sec=10.0, store_waveforms=True):
    """Beats of every ready span of one case. ``time_sec`` is the case clock of the (uniform) grid; windows are (t_start, t_end) pairs.

    Detection runs on each merged span padded by ``pad_sec``; only beats whose R lies inside the unpadded span are kept.
    Returns (beats DataFrame incl. global beat_idx, waveforms dict or None, meta).
    """
    time_sec = np.asarray(time_sec, dtype=float)
    t_first = float(time_sec[0])
    n_total = int(round((time_sec[-1] - t_first) * fs)) + 1
    idx = np.rint((time_sec - t_first) * fs).astype(np.int64)
    dense = {}
    for name, x in (("art", art), ("ecg", ecg), ("ppg", ppg)):
        # Reserve output arrays and masks with explicit initial values.
        d = np.full(n_total, np.nan)
        d[idx] = np.asarray(x, dtype=np.float64)
        dense[name] = d
    s_arr, e_arr = merge_spans(win_start, win_end)
    ecg_sign, ppg_pol = orientation(dense["ecg"], dense["ppg"], fs)              # once per case: regions must not flip polarity
    frames, waves, metas = [], {"ecg": [], "art": [], "ppg": []}, []
    for a, b in zip(s_arr, e_arr):
        i0 = max(0, int(np.floor((a - pad_sec - t_first) * fs)))
        i1 = min(n_total, int(np.ceil((b + pad_sec - t_first) * fs)) + 1)
        if i1 - i0 < int(MIN_SEG_SEC * fs) * 2:
            continue
        df, w, meta = analyze(dense["ecg"][i0:i1], dense["art"][i0:i1], dense["ppg"][i0:i1], t_first + i0 / fs, fs, n_phase, ecg_sign, ppg_pol)
        if not len(df):
            continue
        keep = ((df.t_r >= a) & (df.t_r <= b)).to_numpy()
        if keep.any():
            frames.append(df.loc[keep])
            for k in waves:
                waves[k].append(w[k][keep])
            metas.append(meta)
    if not frames:
        return _empty_frame(), (None if not store_waveforms else {k: np.zeros((0, n_phase), dtype=np.float16) for k in waves}), {"regions": 0}
    out = pd.concat(frames, ignore_index=True)
    out["beat_idx"] = np.arange(len(out), dtype=np.int64)
    wave_out = {k: np.concatenate(v).astype(np.float16) for k, v in waves.items()} if store_waveforms else None
    return out[BEAT_COLUMNS], wave_out, {"regions": len(frames), "ecg_polarity": float(np.mean([m["ecg_polarity"] for m in metas])),
                                         "ppg_polarity": float(np.mean([m["ppg_polarity"] for m in metas]))}
