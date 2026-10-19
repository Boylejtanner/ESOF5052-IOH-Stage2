"""Per-case signal-quality flags (dead ECG / ART / PPG, late PPG) and the evidence for the dead-ECG decision.

A case is ``ecg_dead`` ("ECG unusable") when all of the following hold (every threshold is in ``RULE`` and is written to results/dead_ecg_decision.json):
  A  the ECG SQI is low                                  (``sqi_ecg_median`` < 0.5)
  B  the arterial pulse cannot be tied to the ECG beats  (``art_ok_frac`` < 0.1)
  C  the ECG contradicts or cannot explain the arterial signal: its heart rate differs from the arterial pulse rate measured independently
     from the raw ART by more than 25 bpm, or the raw ECG amplitude is below 0.25 x the dataset median (noise floor / very weak lead).
``assoc_failed`` is NOT used: that flag bit is shared with PPG association failures, which flagged a case whose ART was fine (first draft).
Nothing here edits beats or labels; it reads them.
"""
from __future__ import annotations
# Identify unusable channels before constructing beat-based model inputs.

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import beats as B
from . import beats_pipeline as P
from . import labels as L
from .canonical import atomic_json, utc_now
from .snapshot import assert_dataset_v1

QUALITY_VERSION = "cq2"
RULE = {"sqi_ecg_max": 0.5, "art_ok_frac_max": 0.1, "hr_disagree_bpm": 25.0, "tiny_ecg_amp_ratio": 0.25, "noise_floor_amp_ratio": 0.05,
        "art_pp_min_mmhg": 15.0, "ppg_dead_ok_frac_max": 0.1, "ppg_suspect_late_share": 0.5, "late_delay_sec": 0.9, "chunks": 6, "chunk_sec": 60.0}
OUT_DIR = Path("data") / "derived" / "bshi"
BEAT_COLS = ["t_r", "rr_sec", "hr_bpm", "sqi_ecg", "sqi_ppg", "flags", "valid_beat", "art_ok", "ppg_ok", "ptt_ecg_ppg"]


def quality_path(root):
    return Path(root) / OUT_DIR / "case_quality.parquet"


def _sig_path(root):
    return Path(root) / OUT_DIR / "case_quality.meta.json"


def _bshi_cfg(root):
    return json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))["bshi"]


def signature(root):
    # Store the settings and results as JSON for later inspection.
    cfg = _bshi_cfg(root)
    man = pd.read_parquet(Path(root) / "data" / "manifests" / "beats_manifest.parquet", columns=["dataset", "case_id", "config_hash", "waveform_sha256"])
    h = hashlib.sha256(json.dumps({"v": QUALITY_VERSION, "rule": RULE, "min_usable": cfg["min_usable_beats"], "sqi": cfg["sqi_ecg_min"], "delay": cfg["ppg_delay_max_sec"],
                                   "beats": sorted(set(man.config_hash))}, sort_keys=True).encode())
    h.update(pd.util.hash_pandas_object(man[["dataset", "case_id", "waveform_sha256"]].astype(str), index=False).to_numpy().tobytes())
    return h.hexdigest()


def _chunk_starts(ws, we, n, length):
    s, e = B.merge_spans(ws, we)
    lens = np.maximum(e - s - length, 0.0)
    total = float(lens.sum())
    if total <= 0 or not len(s):
        return []
    cum = np.cumsum(lens)
    out = []
    for k in range(n):
        # Locate the requested interval within the sorted sample times.
        pos = (k + 0.5) / n * total
        i = int(np.searchsorted(cum, pos, side="right"))
        i = min(i, len(s) - 1)
        out.append(float(s[i] + (pos - (cum[i - 1] if i else 0.0))))
    return out


def raw_features(waveform_path, ws, we, rule=RULE, fs=100.0):
    """ECG amplitude (p99-p1, median over chunks), independent ART pulse rate and pulse pressure from a few ready chunks of the raw signal."""
    # Read the stored table required for this calculation.
    t = pq.read_table(waveform_path, columns=["time_sec", "ART", "ECG"]).to_pandas()
    ts = t.time_sec.to_numpy(dtype=float)
    amps, rates, pps = [], [], []
    # Process the selected records in order and retain their individual results.
    for c0 in _chunk_starts(ws, we, int(rule["chunks"]), float(rule["chunk_sec"])):
        i0, i1 = np.searchsorted(ts, [c0, c0 + rule["chunk_sec"]])
        if i1 - i0 < int(0.5 * rule["chunk_sec"] * fs):
            continue
        n = int(round(rule["chunk_sec"] * fs))
        pos = np.clip(np.rint((ts[i0:i1] - c0) * fs).astype(int), 0, n - 1)
        dense = {}
        for name in ("ECG", "ART"):
            # Reserve output arrays and masks with explicit initial values.
            d = np.full(n, np.nan)
            d[pos] = t[name].to_numpy(dtype=float)[i0:i1]
            dense[name] = d
        e = dense["ECG"][np.isfinite(dense["ECG"])]
        if len(e) > fs * 5:
            amps.append(float(np.percentile(e, 99) - np.percentile(e, 1)))
        pulses = B.detect_pulses(dense["ART"], fs, "art")
        if len(pulses["foot"]) >= 8:
            # Summarize the observed levels and variability for this set of measurements.
            span = (pulses["foot"][-1] - pulses["foot"][0]) / fs
            rates.append(60.0 * (len(pulses["foot"]) - 1) / span if span > 0 else np.nan)
            pps.append(float(np.median(pulses["top"] - pulses["base"])))
    return {"ecg_amp": float(np.median(amps)) if amps else np.nan, "art_rate_bpm": float(np.nanmedian(rates)) if rates else np.nan,
            "art_pp_mmhg": float(np.median(pps)) if pps else np.nan, "n_chunks": len(amps)}


def beat_features(beats, cfg, rule=RULE):
    """Case-level summary of the beat table under the BSHI masks (``cfg`` = config bshi block)."""
    n = len(beats)
    if not n:
        return {"n_beats": 0}
    # Calculate valid-sample flags for the following step.
    f = beats["flags"].to_numpy()
    valid = beats["valid_beat"].to_numpy()
    m_ecg = valid & (beats["sqi_ecg"].to_numpy() >= cfg["sqi_ecg_min"])
    m_art = m_ecg & beats["art_ok"].to_numpy()
    ppg_ok = beats["ppg_ok"].to_numpy()
    m_ppg = m_art & ppg_ok & (beats["ptt_ecg_ppg"].to_numpy() <= cfg["ppg_delay_max_sec"])
    pv = valid & ppg_ok
    return {"n_beats": int(n), "assoc_failed_share": float(((f & B.FLAGS["assoc_failed"]) > 0).mean()), "art_ok_frac": float(beats["art_ok"].mean()),
            "ppg_ok_frac": float(ppg_ok.mean()), "sqi_ecg_median": float(beats["sqi_ecg"].median()), "sqi_ppg_median": float(beats["sqi_ppg"].median()),
            "hr_median": float(beats["hr_bpm"].median()), "n_m_ecg": int(m_ecg.sum()), "n_m_art": int(m_art.sum()), "n_m_ppg": int(m_ppg.sum()),
            "ppg_late_share": float((beats["ptt_ecg_ppg"].to_numpy()[pv] > rule["late_delay_sec"]).mean()) if pv.any() else np.nan}


def flag_cases(df, rule=RULE, min_usable=54):
    """Pure classification. ``df`` needs the columns produced by ``beat_features`` + ``raw_features`` + ``ecg_amp_ratio``; returns df with flag columns."""
    # Exclude or flag missing values before using these measurements.
    d = df.copy()
    a = d["sqi_ecg_median"] < rule["sqi_ecg_max"]
    b = d["art_ok_frac"] < rule["art_ok_frac_max"]
    # Exclude or flag missing values before using these measurements.
    disagree = d["art_rate_bpm"].notna() & ((d["hr_median"] - d["art_rate_bpm"]).abs() > rule["hr_disagree_bpm"])
    tiny = d["ecg_amp_ratio"] < rule["tiny_ecg_amp_ratio"]
    d["ecg_dead"] = (a & b & (disagree | tiny)).fillna(False).astype(bool)
    d["ecg_dead_kind"] = np.where(~d["ecg_dead"], "", np.where(d["ecg_amp_ratio"] < rule["noise_floor_amp_ratio"], "noise_floor",
                                                               np.where(disagree, "rate_contradiction", "weak_ecg")))
    # Exclude or flag missing values before using these measurements.
    d["art_dead"] = ((d["art_rate_bpm"].isna() | (d["art_pp_mmhg"] < rule["art_pp_min_mmhg"])) & (d["art_ok_frac"] < rule["art_ok_frac_max"]) & ~d["ecg_dead"]).astype(bool)
    d["art_unlinked"] = ((d["art_ok_frac"] < rule["art_ok_frac_max"]) & ~d["ecg_dead"] & ~d["art_dead"]).astype(bool)
    d["ppg_dead"] = ((d["ppg_ok_frac"] < rule["ppg_dead_ok_frac_max"]) | (d["sqi_ppg_median"] < 0.05)).fillna(False).astype(bool)
    d["ppg_suspect_case"] = (d["ppg_late_share"] > rule["ppg_suspect_late_share"]).fillna(False).astype(bool)
    d["bshi_core_possible"] = (~d["ecg_dead"] & ~d["art_dead"] & (d["n_m_art"] >= min_usable)).astype(bool)
    d["bshi_full_possible"] = (d["bshi_core_possible"] & ~d["ppg_dead"] & (d["n_m_ppg"] >= min_usable)).astype(bool)
    return d


def _job(root, row, ws, we, cfg):
    started = time.perf_counter()
    try:
        beats = P.load_beats(root, row["dataset"], row["case_id"], columns=BEAT_COLS)
        out = {"dataset": row["dataset"], "case_id": str(row["case_id"]), **beat_features(beats, cfg)}
        out.update(raw_features(Path(root) / row["waveform_path"], ws, we))
        out["seconds"] = round(time.perf_counter() - started, 2)
        return {"status": "ok", "row": out}
    except Exception as exc:
        return {"status": "failed", "dataset": row["dataset"], "case_id": str(row["case_id"]), "error": f"{type(exc).__name__}: {exc}"}


def build(root, workers=None, progress=True, force=False):
    """case_quality.parquet: one row per case. Skipped when the signature (rule, beats config hash, case hashes) is unchanged."""
    # Use settings and input fingerprints to decide which case outputs need rebuilding.
    root = Path(root)
    assert_dataset_v1(root)
    sig = signature(root)
    if not force and quality_path(root).exists() and _sig_path(root).exists() and json.loads(_sig_path(root).read_text())["signature"] == sig:
        return pd.read_parquet(quality_path(root))
    cfg = _bshi_cfg(root)
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet").astype({"case_id": str})
    windows = {}
    for ds in ("mover", "vitaldb"):
        # Summarize the records separately within each requested group.
        w = pd.read_parquet(root / "data" / "manifests" / f"ready_windows_{ds}.parquet", columns=["case_id", "t_start_sec", "t_end_sec", "min_valid_frac", "is_ready"])
        w = w.loc[w.is_ready & (w.min_valid_frac >= 0.5 - 1e-6)].astype({"case_id": str})
        windows[ds] = {k: (g.t_start_sec.to_numpy(), g.t_end_sec.to_numpy()) for k, g in w.groupby("case_id")}
    jobs = [(r._asdict(), *windows[r.dataset].get(r.case_id, (np.array([]), np.array([])))) for r in index.itertuples(index=False)]
    n_workers = max(1, min(int(workers or cfg["workers"]), (os.cpu_count() or 8) - 2))
    rows, failed = [], []
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(jobs), desc="case quality", unit="case")
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(_job, str(root), {k: r[k] for k in ("dataset", "case_id", "waveform_path")}, ws, we, cfg) for r, ws, we in jobs]
        for fut in cf.as_completed(futures):
            res = fut.result()
            (rows if res["status"] == "ok" else failed).append(res["row"] if res["status"] == "ok" else res)
            bar and bar.update(1)
    bar and bar.close()
    if failed:
        raise RuntimeError(f"{len(failed)} case(s) failed, first: {failed[0]}")
    # Join related tables by dataset, case id to keep measurements aligned.
    feat = pd.DataFrame(rows)
    feat["ecg_amp_ratio"] = feat.ecg_amp / feat.groupby("dataset").ecg_amp.transform("median")
    keys = index[["dataset", "case_id", "patient_key", "split_pooled", "split_within_dataset"]]
    out = keys.merge(feat, on=["dataset", "case_id"], how="left")
    out = flag_cases(out, RULE, int(cfg["min_usable_beats"])).sort_values(["dataset", "case_id"]).reset_index(drop=True)
    quality_path(root).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(quality_path(root), index=False)
    atomic_json(_sig_path(root), {"signature": sig, "rule": RULE, "version": QUALITY_VERSION, "at": utc_now(), "cases": int(len(out))})
    return out


def load_case_quality(root):
    return pd.read_parquet(quality_path(root))


# the dead-ECG decision
def _score_block(frame, y="y", p="p", reps=200):
    if not len(frame) or not frame[y].any():
        return {"windows": int(len(frame)), "events": int(frame[y].sum()) if len(frame) else 0, "auroc": None, "auroc_ci95": None}
    return {"windows": int(len(frame)), "events": int(frame[y].sum()), "patients": int(frame.patient_key.nunique()), "prevalence": float(frame[y].mean()),
            "auroc": L.auroc(frame[y], frame[p]), "auprc": L.auprc(frame[y], frame[p]), "auroc_ci95": L._cluster_ci(frame, y, p, reps).get("auroc")}


def build_dead_ecg_decision(root, reps=None, write=True):
    """Evidence + decision for the dead-ECG cases -> results/dead_ecg_decision.json and results/eda/dead_ecg_cases.csv."""
    # Load the recorded settings or metadata for this processing step.
    root = Path(root)
    settings = json.loads((root / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))
    reps = int(reps or settings["eda"]["bootstrap_reps"])
    # Read the stored table required for this calculation.
    q = load_case_quality(root)
    beats_man = pd.read_parquet(root / "data" / "manifests" / "beats_manifest.parquet", columns=["dataset", "case_id", "art_ok_frac"]).astype({"case_id": str})
    ref68 = set(beats_man.loc[(beats_man.dataset == "vitaldb") & (beats_man.art_ok_frac < 0.1), "case_id"])
    flagged = set(q.loc[q.ecg_dead & (q.dataset == "vitaldb"), "case_id"])
    # Read the stored table required for this calculation.
    flagged_all = q.loc[q.ecg_dead]
    index = pd.read_parquet(root / "data" / "manifests" / "unified_case_index.parquet").astype({"case_id": str})
    lab = L.load_labels(root)
    lab["case_id"] = lab.case_id.astype(str)
    dead_keys = set(zip(flagged_all.dataset, flagged_all.case_id))
    lab["dead"] = [(d, c) in dead_keys for d, c in zip(lab.dataset, lab.case_id)]
    prim = lab.loc[lab.eligible_onset & ~lab.label_ambiguous & (lab.min_valid_frac >= 0.7 - 1e-6)].dropna(subset=L.FEATURES).copy()
    prim["y"] = prim.y_event.astype(int)
    train = prim.loc[~prim.dead & (prim.split_pooled == "train")]
    rest_test, dead = prim.loc[~prim.dead & (prim.split_pooled == "test")].copy(), prim.loc[prim.dead].copy()
    rest_test["p"] = L._fit_predict(train, rest_test)
    if len(dead):
        dead["p"] = L._fit_predict(train, dead)
    # Summarize the records separately within each requested group.
    ready_h = index.assign(dead=[(d, c) in dead_keys for d, c in zip(index.dataset, index.case_id)]).groupby("dead").ready_span_min_theta_50.sum() / 60
    own = q.groupby("patient_key").agg(n=("case_id", "size"), n_dead=("ecg_dead", "sum"), splits=("split_pooled", "nunique"))
    pats = own.loc[own.n_dead > 0]
    desc = lambda g: {"map_current_median": float(g.map_current.median()), "map_current_p10": float(g.map_current.quantile(.1)), "map_current_p90": float(g.map_current.quantile(.9)),
                      "windows": int(len(g)), "events": int(g.y.sum()), "event_prevalence": float(g.y.mean()) if len(g) else None}
    all_windows, all_dead = len(lab), int(lab.dead.sum())
    checks = {"every_flagged_case_has_art_ok_frac_below_0.1": bool((flagged_all.art_ok_frac < 0.1).all()) if len(flagged_all) else True,
              "excluded_window_share_below_5pct": bool(all_dead / max(1, all_windows) < 0.05),
              "no_patient_spans_two_splits": bool((pats.splits == 1).all()) if len(pats) else True,
              "recovers_reference_68": flagged == ref68}
    evidence = {
        "rule": RULE, "reference_set_art_ok_frac_below_0.1_vitaldb": len(ref68), "flagged_ecg_dead_cases": int(len(flagged_all)),
        "flagged_by_dataset": flagged_all.dataset.value_counts().to_dict(), "flagged_by_kind": flagged_all.ecg_dead_kind.value_counts().to_dict(),
        "art_dead_cases": int(q.art_dead.sum()), "art_unlinked_cases": int(q.art_unlinked.sum()),
        "rule_revision_note": "First draft (SQI<0.3, art_ok<0.1 or assoc_failed>0.9, HR>120 with ART disagreement or amplitude<0.2) recovered 61 of the 68 reference cases and "
                              "flagged 2 extra: the assoc_failed flag bit is shared with PPG association failures (case 1320 has a usable ART), and 7 reference cases had an SQI "
                              "just above 0.3, an HR below 120 or an amplitude ratio just above 0.2. The revised rule above recovers the reference set.",
        "recovered": len(flagged & ref68), "missed_from_reference": sorted(ref68 - flagged)[:20],
        "extra_beyond_reference": sorted(flagged - ref68)[:20], "patients": int(flagged_all.patient_key.nunique()),
        "split_distribution": flagged_all.split_pooled.value_counts().to_dict(), "ready_hours_dead": round(float(ready_h.get(True, 0.0)), 1),
        "ready_hours_rest": round(float(ready_h.get(False, 0.0)), 1), "labeled_windows_dead": all_dead, "labeled_windows_all": int(all_windows),
        "primary_windows_and_events_dead": {"windows": int(len(dead)), "events": int(dead.y.sum())},
        "map_features_dead": desc(dead) if len(dead) else None, "map_features_rest": desc(prim.loc[~prim.dead]),
        "baseline_A_style_auroc": {"model": "logistic on MAP features trained on non-dead pooled-train windows",
                                   "dead_cases_all_splits": _score_block(dead, reps=reps), "rest_test": _score_block(rest_test, reps=reps)},
        "patients_with_other_healthy_cases": int((pats.n > pats.n_dead).sum()), "patients_flagged": int(len(pats)),
        "ecg_amp_ratio_dead_median": float(flagged_all.ecg_amp_ratio.median()) if len(flagged_all) else None,
        "hr_median_dead": float(flagged_all.hr_median.median()) if len(flagged_all) else None,
        "art_rate_median_dead": float(flagged_all.art_rate_bpm.median()) if len(flagged_all) else None}
    # Calculate output values for the following step.
    decision = {"decision": "EXCLUDE from every beat-synchronous (BSHI) set; KEEP in MAP-only baselines and in the EDA 'all windows' tables (always labelled)",
                "default_kept": all(v for k, v in checks.items() if k != "recovers_reference_68"),
                "checks": checks,
                "rejected_alternative": "ART-anchored beats for ECG-less cases: they are 2.1% of cases, have no ECG->ART timing (the key BSHI feature) and would form a different input distribution; recorded as a future option, not built.",
                "usage_rule": "BSHI-vs-baseline comparisons must use the identical BSHI-eligible window set; baseline A is additionally reported on all windows.",
                "at": utc_now()}
    out = {"evidence": evidence, "decision": decision}
    if write:
        # Join related tables by dataset, case id to keep measurements aligned.
        atomic_json(root / "results" / "dead_ecg_decision.json", out)
        (root / "results" / "eda").mkdir(parents=True, exist_ok=True)
        win = prim.groupby(["dataset", "case_id"]).agg(primary_windows=("y", "size"), primary_events=("y", "sum")).reset_index()
        tab = flagged_all.merge(win, on=["dataset", "case_id"], how="left").merge(index[["dataset", "case_id", "ready_span_min_theta_50"]], on=["dataset", "case_id"], how="left")
        tab[["dataset", "case_id", "patient_key", "split_pooled", "n_beats", "sqi_ecg_median", "art_ok_frac", "assoc_failed_share", "hr_median", "art_rate_bpm", "ecg_amp_ratio",
             "ready_span_min_theta_50", "primary_windows", "primary_events"]].to_csv(root / "results" / "eda" / "dead_ecg_cases.csv", index=False)
    return out
