"""Quality control of the beat extraction on a handful of real cases: deterministic case choice, metrics, pass/fail gates, overlay figures.

QC reads the WRITTEN outputs (beats.parquet), so it verifies what downstream stages will actually consume. Gates (dataset-level medians of the
per-case values) and why they differ from a naive reading of the proposal:
* ART alignment rate: share of beats with usable ART (finite, not flat/clipped) that have an associated, plausible ART pulse; >= 0.90 VitalDB, >= 0.80 MOVER.
* |HR(ECG) - HR(ART)| median <= 3 bpm: an independent check that R peaks and ART pulses describe the same beats.
* ECG->ART delay median in 0.05-0.40 s; ECG->PPG delay in 0.10-1.00 s and |ART->PPG| <= 0.80 s. The PPG bounds are wider than the usual 0.35-0.50 s
  because pulse-oximeter outputs carry device latency: the first real-data check (overlays + 0.4-0.7 s delays, consistent across cases and datasets)
  showed ECG->PPG ~0.65 s on VitalDB and ~0.40 s on MOVER, which is a real device property and also a cross-dataset shift to keep in mind.
* ECG->ART delay IQR <= 0.08 s: a consistent association has a tight delay distribution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import beats_pipeline as P
from .beats import FLAGS
from .canonical import atomic_json, utc_now

GATES = {"art_align_rate": {"vitaldb": 0.90, "mover": 0.80}, "hr_diff_bpm_max": 3.0, "ptt_ecg_art": (0.05, 0.40), "ptt_ecg_ppg": (0.10, 1.00),
         "ptt_art_ppg_abs_max": 0.80, "ptt_ecg_art_iqr_max": 0.08, "r_peak_alignment_min": 0.90}


def r_peak_alignment(t_r, time_sec, ecg, half_sec=0.4, fs=100.0, ratio=0.5):
    """Share of R markers that sit on the dominant deflection of the raw ECG within +-half_sec (polarity-agnostic).

    Independent of the detector's own orientation logic: a marker placed on a trough or T wave 0.25 s after a tall R fails, even though
    HR agreement and delay ranges would still look self-consistent. NaN-neighbourhoods are skipped.
    """
    ecg = np.asarray(ecg, dtype=float)
    x = np.abs(ecg - np.nanmedian(ecg))
    idx = np.rint((np.asarray(t_r) - float(time_sec[0])) * fs).astype(int)
    k = int(half_sec * fs)
    ok = (idx >= k) & (idx < len(x) - k)
    idx = idx[ok]
    if not len(idx):
        return float("nan")
    win = x[idx[:, None] + np.arange(-k, k + 1)[None, :]]
    centre = x[idx]
    keep = np.isfinite(win).all(axis=1) & np.isfinite(centre)
    return float((centre[keep] >= ratio * win[keep].max(axis=1)).mean()) if keep.any() else float("nan")


def pick_cases(root, seed=5052, per_dataset=6):
    """Deterministic QC set: per dataset one low-plausibility case (lowest art_plausible_fraction of a 60-case sample) plus random others."""
    index = pd.read_parquet(Path(root) / "data" / "manifests" / "unified_case_index.parquet")
    out = []
    for ds in ("vitaldb", "mover"):
        d = index.loc[index.dataset == ds]
        pool = d.sample(n=min(len(d), 60), random_state=seed)
        low = pool.sort_values("art_plausible_fraction", kind="stable").iloc[0]
        rest = pool.drop(low.name).sample(n=per_dataset - 1, random_state=seed + 1)
        out += [(ds, str(c)) for c in [low.case_id, *rest.case_id]]
    return out


def pick_dry_run(root, qc_cases, per_dataset=20, seed=5052):
    index = pd.read_parquet(Path(root) / "data" / "manifests" / "unified_case_index.parquet")
    chosen = list(qc_cases)
    for ds in ("vitaldb", "mover"):
        have = {c for d, c in chosen if d == ds}
        pool = index.loc[(index.dataset == ds) & ~index.case_id.astype(str).isin(have) & (index.n_ready_windows > 0)]
        extra = pool.sample(n=max(0, per_dataset - len(have)), random_state=seed + 2)
        chosen += [(ds, str(c)) for c in extra.case_id]
    return chosen


def case_metrics(beats):
    """Detector-reliability metrics from one case's beat table."""
    f = beats["flags"].to_numpy()
    valid = beats["valid_beat"].to_numpy()
    art_usable = valid & ((f & (FLAGS["art_missing"] | FLAGS["art_flatline"] | FLAGS["art_clipped"])) == 0)
    ppg_usable = valid & ((f & FLAGS["ppg_missing"]) == 0)
    both = beats[["hr_bpm", "hr_art_bpm"]].dropna()
    q = beats["ptt_ecg_art"].dropna()
    return {
        "beats": int(len(beats)), "valid_frac": float(valid.mean()) if len(beats) else 0.0,
        "art_align_rate": float(beats.loc[art_usable, "art_ok"].mean()) if art_usable.any() else float("nan"),
        "ppg_align_rate": float(beats.loc[ppg_usable, "ppg_ok"].mean()) if ppg_usable.any() else float("nan"),
        "art_flat_or_clipped_frac": float((((f & (FLAGS["art_flatline"] | FLAGS["art_clipped"])) > 0) & valid).sum() / max(1, valid.sum())),
        "hr_diff_bpm": float((both.hr_bpm - both.hr_art_bpm).abs().median()) if len(both) else float("nan"),
        "hr_median": float(beats.hr_bpm.median()),
        "ptt_ecg_art": float(q.median()) if len(q) else float("nan"), "ptt_ecg_art_iqr": float(q.quantile(.75) - q.quantile(.25)) if len(q) else float("nan"),
        "ptt_ecg_ppg": float(beats.ptt_ecg_ppg.median()), "ptt_art_ppg": float(beats.ptt_art_ppg.median()),
        "sqi_ecg": float(beats.sqi_ecg.median()), "sqi_art": float(beats.sqi_art.median()), "sqi_ppg": float(beats.sqi_ppg.median()),
        "sqi_cross": float(beats.sqi_cross.median())}


def evaluate_gates(per_case):
    """Dataset-level gate table from the per-case frame (columns: dataset + case_metrics keys)."""
    rows, passed = [], True
    for ds, g in per_case.groupby("dataset"):
        med = g.median(numeric_only=True)
        checks = {
            "art_align_rate": (med.art_align_rate, f">= {GATES['art_align_rate'][ds]}", med.art_align_rate >= GATES["art_align_rate"][ds]),
            "hr_diff_bpm": (med.hr_diff_bpm, f"<= {GATES['hr_diff_bpm_max']}", med.hr_diff_bpm <= GATES["hr_diff_bpm_max"]),
            "ptt_ecg_art_s": (med.ptt_ecg_art, f"{GATES['ptt_ecg_art']}", GATES["ptt_ecg_art"][0] <= med.ptt_ecg_art <= GATES["ptt_ecg_art"][1]),
            "ptt_ecg_ppg_s": (med.ptt_ecg_ppg, f"{GATES['ptt_ecg_ppg']}", GATES["ptt_ecg_ppg"][0] <= med.ptt_ecg_ppg <= GATES["ptt_ecg_ppg"][1]),
            "abs_ptt_art_ppg_s": (abs(med.ptt_art_ppg), f"<= {GATES['ptt_art_ppg_abs_max']}", abs(med.ptt_art_ppg) <= GATES["ptt_art_ppg_abs_max"]),
            "ptt_ecg_art_iqr_s": (med.ptt_ecg_art_iqr, f"<= {GATES['ptt_ecg_art_iqr_max']}", med.ptt_ecg_art_iqr <= GATES["ptt_ecg_art_iqr_max"]),
            "r_peak_alignment": (med.r_peak_alignment, f">= {GATES['r_peak_alignment_min']}", med.r_peak_alignment >= GATES["r_peak_alignment_min"])}
        for name, (value, rule, ok) in checks.items():
            rows.append({"dataset": ds, "check": name, "median_of_cases": round(float(value), 4), "gate": rule, "pass": bool(ok)})
            passed &= bool(ok)
    return pd.DataFrame(rows), passed


def plot_overlay(root, dataset, case_id, center, path, half=7.5):
    """ECG / ART / PPG with the STORED detections and an SQI strip for ``center +- half`` seconds."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pyarrow.parquet as pq
    root = Path(root)
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet")
    row = index.loc[(index.dataset == dataset) & (index.case_id.astype(str) == str(case_id))].iloc[0]
    w = pq.read_table(root / row.waveform_path, columns=["time_sec", "ART", "ECG", "PPG"]).to_pandas()
    t0 = float(w.time_sec.iloc[0])
    a, b = int((center - half - t0) * 100), int((center + half - t0) * 100)
    w = w.iloc[max(0, a):b]
    beats = P.load_beats(root, dataset, case_id)
    b = beats.loc[(beats.t_r >= center - half) & (beats.t_r <= center + half)]
    fig, ax = plt.subplots(4, 1, figsize=(11, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2, 1, .6]})
    ax[0].plot(w.time_sec, w.ECG, lw=.7)
    ax[0].plot(b.t_r, np.interp(b.t_r, w.time_sec, np.nan_to_num(w.ECG.to_numpy())), "rv", ms=5)
    ax[0].set_ylabel("ECG, R peaks")

    def on_signal(times, col):
        times = times.dropna()
        return times, np.interp(times, w.time_sec, np.nan_to_num(w[col].to_numpy()))

    ax[1].plot(w.time_sec, w.ART, lw=.7)
    ax[1].plot(*on_signal(b.art_foot_t, "ART"), "g^", ms=5)
    ax[1].plot(b.art_sys_t, b.sbp, "rv", ms=4)
    ax[1].set_ylabel("ART mmHg, foot/sys")
    ax[2].plot(w.time_sec, w.PPG, lw=.7)
    ax[2].plot(*on_signal(b.ppg_foot_t, "PPG"), "g^", ms=5)
    ax[2].set_ylabel("PPG, foot")
    for col, colr in (("sqi_ecg", "C0"), ("sqi_art", "C1"), ("sqi_ppg", "C2"), ("sqi_cross", "C3")):
        ax[3].step(b.t_r, b[col], where="post", color=colr, label=col, lw=1)
    ax[3].set_ylim(-0.05, 1.05)
    ax[3].legend(ncol=4, fontsize=7, loc="lower right")
    ax[3].set_ylabel("SQI")
    ax[3].set_xlabel("case clock, s")
    for x in ax:
        x.set_xlim(center - half, center + half)
        x.grid(alpha=.3)
    fig.suptitle(f"{dataset} case {case_id}  t={center:.0f}s  (markers are the STORED beat outputs; gaps/flat segments are masked, not forced)", fontsize=9)
    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=70)
    plt.close(fig)


def run_qc(root, cases=None, figures=True, figures_per_case=3, seed=5052, write=True, progress=True):
    """Run beats on the QC cases (skipped when already finished), compute metrics + gates, draw overlays, write results/beats_qc.json."""
    root = Path(root)
    cases = cases or pick_cases(root, seed)
    run = P.run(root, case_ids=cases, progress=progress)
    rows = []
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet")
    for ds, cid in cases:
        beats = P.load_beats(root, ds, cid)
        if len(beats):
            row = index.loc[(index.dataset == ds) & (index.case_id.astype(str) == str(cid))].iloc[0]
            import pyarrow.parquet as pq
            w = pq.read_table(root / row.waveform_path, columns=["time_sec", "ECG"]).to_pandas()
            align = r_peak_alignment(beats.loc[beats.valid_beat, "t_r"].to_numpy(), w.time_sec.to_numpy(), w.ECG.to_numpy())
            rows.append({"dataset": ds, "case_id": cid, **case_metrics(beats), "r_peak_alignment": align})
    per_case = pd.DataFrame(rows)
    table, passed = evaluate_gates(per_case)
    rng = np.random.default_rng(seed)
    pngs = []
    if figures:
        for ds, cid in cases:
            beats = P.load_beats(root, ds, cid, columns=["t_r", "valid_beat"])
            candidates = beats.loc[beats.valid_beat, "t_r"].to_numpy()
            if len(candidates) < 50:
                continue
            for k, c in enumerate(rng.choice(candidates[20:-20], size=figures_per_case, replace=False)):
                out = root / "figures" / "beats_qc" / f"{ds}_{cid}_{k}.png"
                plot_overlay(root, ds, cid, float(c), out)
                pngs.append(str(out.relative_to(root)))
    report = {"qc_passed": bool(passed), "at": utc_now(), "cases": [list(c) for c in cases], "gates": GATES_JSON(), "gate_table": table.to_dict("records"),
              "per_case": per_case.round(4).to_dict("records"), "figures": pngs, "run": {k: run.get(k) for k in ("status", "processed", "skipped_finished", "failed")}}
    if write:
        atomic_json(root / "results" / "beats_qc.json", report)
    return report, per_case, table


def GATES_JSON():
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in GATES.items()}
