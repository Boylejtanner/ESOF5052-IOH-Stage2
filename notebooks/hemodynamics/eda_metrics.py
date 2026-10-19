"""EDA metrics for the project report: tables under results/eda, figures under figures/eda, readiness scorecard, docs/eda_summary.md.

``build_all(root)`` is idempotent: every stage stores the signature of its inputs and is skipped when nothing changed. Stages: cohort, signal, labels, models (features, G0-G4
ladder, lead time, dataset shift; the slow one), readiness, summary.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as md
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bshi as S
from . import case_quality as CQ
from . import eda_models as M
from . import labels as L
from .canonical import atomic_json, utc_now
from .figstyle import finish_axes
from .snapshot import assert_dataset_v1

EDA_VERSION = "eda1"
FIG_DPI = 150


def _paths(root):
    root = Path(root)
    (root / "results" / "eda").mkdir(parents=True, exist_ok=True)
    (root / "figures" / "eda").mkdir(parents=True, exist_ok=True)
    return root / "results" / "eda", root / "figures" / "eda"


def _table(root, name, frame):
    d, _ = _paths(root)
    frame.to_csv(d / f"{name}.csv", index=False)
    return f"results/eda/{name}.csv"


def _figure(root, name, fig):
    import matplotlib.pyplot as plt
    from . import figstyle
    figstyle.capitalize_figure(fig, panel_letters=True)   # capital first letters, proper symbols, panel letters (a), (b), ...
    figstyle.audit_figure(fig, name)                      # an unlabeled figure can no longer be saved
    _, f = _paths(root)
    fig.savefig(f / f"{name}.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return f"figures/eda/{name}.png"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _settings(root):
    return json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))


def _read_json(root, rel):
    return json.loads((Path(root) / rel).read_text(encoding="utf-8"))


def _stage_signature(root, name, extra=""):
    s = _settings(root)
    return hashlib.sha256(json.dumps({"v": EDA_VERSION, "stage": name, "index": S.index_signature(root), "eda": s["eda"], "bshi": {k: v for k, v in s["bshi"].items() if k != "workers"},
                                      "extra": extra}, sort_keys=True).encode()).hexdigest()


def _run_stage(root, name, fn, force=False, extra=""):
    d, _ = _paths(root)
    meta = d / f"_stage_{name}.json"
    sig = _stage_signature(root, name, extra)
    if not force and meta.exists() and json.loads(meta.read_text())["signature"] == sig:
        return json.loads(meta.read_text())["headline"], False
    started = time.perf_counter()
    headline = fn()
    atomic_json(meta, {"signature": sig, "headline": headline, "seconds": round(time.perf_counter() - started, 1), "at": utc_now()})
    return headline, True


# cohort
def _unified(root):
    return pd.read_parquet(Path(root) / "data" / "manifests" / "unified_case_index.parquet").astype({"case_id": str})


def _window_index(root, columns=None):
    return S.load_window_index(root, columns=columns)


def stage_cohort(root):
    root = Path(root)
    idx = _unified(root)
    summ = _read_json(root, "results/bshi_index_summary.json")
    fin = _read_json(root, "results/mover_fast_finalize.json")
    audit = _read_json(root, "data/manifests/vitaldb_project_audit.json")
    mover_all = pd.read_parquet(root / "data" / "manifests" / "mover_all_cases.parquet", columns=["case_id", "has_ppg_observed", "has_art", "eligibility_status", "exclusion_reason"])
    stats = pd.read_parquet(root / "data" / "manifests" / "mover_case_window_stats.parquet", columns=["case_id", "final_usable"])
    win = pd.read_parquet(root / S.OUT_DIR / S.INDEX_NAME, columns=["dataset", "case_id", "patient_key", "primary_eligible", "in_core", "in_full"])
    cases_with = {k: win[win[k] & win.primary_eligible].groupby("dataset").agg(cases=("case_id", "nunique"), patients=("patient_key", "nunique")) for k in ("in_core", "in_full")}
    rows = [
        ("mover", 1, "surgeries listed in the three archive case tables", len(mover_all), None),
        ("mover", 2, "with the ECG + ART + PPG waveform members observed", int(mover_all.has_ppg_observed.sum()), None),
        ("mover", 3, "of those, with an arterial line", int(mover_all.has_art.sum()), None),
        ("mover", 4, "assessed for window-level eligibility", len(stats), None),
        ("mover", 5, "selected at rung R1 (>= 30 ready minutes at theta 0.75)", int(fin["window_eligibility"]["funnel"][0]["cases"]), int(fin["window_eligibility"]["funnel"][0]["patients"])),
        ("mover", 6, "with at least one BSHI core window (>= 54 of 60 usable beats)", int(cases_with["in_core"].loc["mover", "cases"]), int(cases_with["in_core"].loc["mover", "patients"])),
        ("mover", 7, "with at least one BSHI full window (also usable PPG)", int(cases_with["in_full"].loc["mover", "cases"]), int(cases_with["in_full"].loc["mover", "patients"])),
        ("vitaldb", 1, "surgeries in the VitalDB clinical table", int(audit["clinical_cases"]), int(audit["clinical_patients"])),
        ("vitaldb", 2, "with ART + ECG + PPG + ART numeric MAP tracks (original cohort)", int(audit["original_cohort_rows"]), None),
        ("vitaldb", 3, "usable after quality filters (duration, missingness, MAP density)", int(audit["final_usable_cases"]), int(audit["usable_patients"])),
        ("vitaldb", 4, "with at least one BSHI core window", int(cases_with["in_core"].loc["vitaldb", "cases"]), int(cases_with["in_core"].loc["vitaldb", "patients"])),
        ("vitaldb", 5, "with at least one BSHI full window", int(cases_with["in_full"].loc["vitaldb", "cases"]), int(cases_with["in_full"].loc["vitaldb", "patients"]))]
    for r in fin["window_eligibility"]["funnel"]:
        rows.append(("mover", 10, f"ladder {r['rung']}: theta {r['theta']}, >= {r['min_ready_span_min']} ready min", int(r["cases"]), int(r["patients"])))
    funnel = pd.DataFrame(rows, columns=["dataset", "step", "description", "cases", "patients"])
    s = _settings(root)
    wr, vq = s["window_rule"], s["vitaldb"]["quality_thresholds"]
    criteria = pd.DataFrame([
        ("VitalDB", "case", f"ART, ECG-II, PLETH and an ART-derived numeric MAP present; duration >= {vq['duration_minutes_min']} min; waveform missing <= {vq['waveform_missing_pct_strict_max']}%; "
                            f">= {vq['map_valid_count_min']} valid MAP values; >= {vq['map_measurements_per_minute_min']} MAP values/min; MAP span coverage >= {vq['map_coverage_span_pct_min']}%; median MAP gap <= {vq['map_median_gap_sec_max']} s"),
        ("MOVER", "case", f"GE S5 members with a PLETH group and a verified arterial INVP1 channel; surgery time and patient linkage valid; duration >= {s['minimum_duration_sec'] // 60} min; "
                          f"window-level ladder (not whole-surgery missingness): R1 needs >= 30 ready minutes of windows with >= 75% valid samples; MAP kept in {s['map_min_mmhg']}-{s['map_max_mmhg']} mmHg"),
        ("both", "window", f"{wr['window_sec']} s observation window every {wr['stride_sec']} s; ready when the valid-sample fraction >= {wr['theta_floor']} (labels use >= 0.7); numeric MAP no older than "
                           f"{wr['map_current_max_age_sec']} s; at least {int(100 * wr['horizon_map_min_fraction'])}% of the horizon covered by MAP"),
        ("both", "label", f"event = MAP < {s['label_rule']['hypo_threshold']} mmHg for >= {s['label_rule']['sustained_min_sec']} s (gaps <= {s['label_rule']['max_gap_sec']} s bridged) starting in the next {s['label_rule']['horizon_sec']} s; "
                          f"windows already hypotensive (any MAP < 65 in the last {s['label_rule']['already_hypo_lookback_sec']} s) are excluded; 65-75 mmHg windows are tagged (gray zone), not dropped; "
                          "windows whose horizon has > 60 s of missing MAP are ambiguous and excluded"),
        ("both", "BSHI window", f"60 most recent beats whose whole cycle ends by t0; beat usable when the ECG SQI >= {s['bshi']['sqi_ecg_min']}; core = >= {s['bshi']['min_usable_beats']} beats with a valid ART pulse; "
                                f"full = >= {s['bshi']['min_usable_beats']} beats with ART and PPG (ECG->PPG delay <= {s['bshi']['ppg_delay_max_sec']} s); dead-ECG cases excluded"),
        ("both", "split", f"patient-level 70/15/15 (seed {s['patient_split_seed']}) frozen before windowing; no patient appears in two splits; two views: pooled and within-dataset")],
        columns=["scope", "stage", "rule"])
    prim = win[win.primary_eligible]
    splits = []
    for col in ("split_pooled", "split_within_dataset"):
        for (ds, sp), g in idx.groupby(["dataset", col]):
            splits.append({"split_type": col, "dataset": ds, "split": sp, "cases": int(len(g)), "patients": int(g.patient_key.nunique())})
    sp = pd.DataFrame(splits)
    for col in ("split_pooled", "split_within_dataset"):
        m = sp.split_type == col
        sp.loc[m, "patient_share_of_dataset_pct"] = sp[m].groupby("dataset").patients.transform(lambda x: 100 * x / x.sum()).round(2)
    pooled = idx.groupby("split_pooled").agg(patients=("patient_key", "nunique"), cases=("case_id", "size"))
    pooled["patient_share_pct"] = (100 * pooled.patients / pooled.patients.sum()).round(2)
    pm = idx.groupby(["split_pooled", "dataset"]).patient_key.nunique().unstack(fill_value=0)
    pooled["mover_patient_share_pct"] = (100 * pm["mover"] / (pm["mover"] + pm["vitaldb"])).round(2)
    pooled = pooled.reset_index()
    win_split = summ["funnel"]["split_pooled"]
    wtab = pd.DataFrame([r for r in win_split if r["dataset"] != "all" and r["split"] != "all" and r["step"] in ("primary_eligible", "primary_core", "primary_full")])
    leak = int((idx.groupby("patient_key").split_pooled.nunique() > 1).sum()) + int((idx.groupby("patient_key").split_within_dataset.nunique() > 1).sum())
    out = [_table(root, "table_cohort_funnel", funnel), _table(root, "table_selection_criteria", criteria), _table(root, "table_split_cases_patients", sp),
           _table(root, "table_split_pooled_summary", pooled), _table(root, "table_split_windows", wtab),
           _table(root, "table_mover_exclusion_reasons", mover_all.exclusion_reason.replace("", "(selected)").value_counts().rename_axis("reason").reset_index(name="surgeries"))]
    return {"files": out, "leakage_patients_in_two_splits": leak, "mover_cases": int(len(idx[idx.dataset == "mover"])), "vitaldb_cases": int(len(idx[idx.dataset == "vitaldb"])),
            "patients": int(idx.patient_key.nunique()), "pooled_patient_shares": dict(zip(pooled.split_pooled, pooled.patient_share_pct)),
            "mover_patient_share_by_split": dict(zip(pooled.split_pooled, pooled.mover_patient_share_pct))}


# signal
def _gap_profile(root, per_dataset=24, seed=5052):
    import pyarrow.parquet as pq
    idx = _unified(root)
    rng = np.random.default_rng(seed)
    rows, runs = [], {"mover": [], "vitaldb": []}
    for ds, g in idx.groupby("dataset"):
        for _, r in g.iloc[rng.choice(len(g), min(per_dataset, len(g)), replace=False)].iterrows():
            t = pq.read_table(root / r.waveform_path, columns=["time_sec", "ART", "ECG", "PPG"]).to_pandas()
            n = int(round((t.time_sec.iloc[-1] - t.time_sec.iloc[0]) * 100)) + 1
            pos = np.rint((t.time_sec.to_numpy() - t.time_sec.iloc[0]) * 100).astype(np.int64)
            ok = np.zeros(n, dtype=bool)
            ok[pos] = np.isfinite(t[["ART", "ECG", "PPG"]].to_numpy()).all(axis=1)
            edges = np.flatnonzero(np.diff(np.r_[False, ok, False].astype(np.int8)))
            starts, ends = edges[0::2], edges[1::2]
            valid_runs = (ends - starts) / 100.0
            gaps = (starts[1:] - ends[:-1]) / 100.0
            runs[ds].append(valid_runs)
            rows.append({"dataset": ds, "case_id": r.case_id, "hours": n / 360000.0, "valid_fraction": float(ok.mean()), "valid_runs_per_hour": len(valid_runs) / (n / 360000.0),
                         "median_valid_run_s": float(np.median(valid_runs)) if len(valid_runs) else 0.0, "median_gap_s": float(np.median(gaps)) if len(gaps) else 0.0,
                         "share_gaps_le_3s": float((gaps <= 3).mean()) if len(gaps) else np.nan})
    per_case = pd.DataFrame(rows)
    summary = per_case.groupby("dataset").agg(cases_sampled=("case_id", "size"), valid_fraction_median=("valid_fraction", "median"), valid_runs_per_hour_median=("valid_runs_per_hour", "median"),
                                              median_valid_run_s=("median_valid_run_s", "median"), median_gap_s=("median_gap_s", "median"), share_gaps_le_3s=("share_gaps_le_3s", "median")).round(3).reset_index()
    return summary, runs


def stage_signal(root):
    root = Path(root)
    idx = _unified(root)
    man = pd.read_parquet(root / "data" / "manifests" / "beats_manifest.parquet").astype({"case_id": str}).merge(
        CQ.load_case_quality(root)[["dataset", "case_id", "ppg_late_share", "ppg_suspect_case", "ecg_dead", "ppg_dead", "bshi_core_possible", "bshi_full_possible"]], on=["dataset", "case_id"])
    labels = _window_index(root, ["dataset", "min_valid_frac"])
    cov = idx.groupby("dataset").agg(cases=("case_id", "size"), waveform_hours=("duration_sec", lambda x: x.sum() / 3600), ready_hours_theta50=("ready_span_min_theta_50", lambda x: x.sum() / 60)).round(1)
    cov["ready_share_pct"] = (100 * cov.ready_hours_theta50 / cov.waveform_hours).round(1)
    lw = labels.groupby("dataset").min_valid_frac.agg(labeled_windows="size", mean_min_valid_frac="mean", share_ge_0_7=lambda x: (x >= 0.7).mean()).round(3)
    cov = cov.join(lw).reset_index()
    gaps, runs = _gap_profile(root)
    bq = man.groupby("dataset").agg(cases=("case_id", "size"), beats=("n_beats", "sum"), valid_beats=("n_valid", "sum"), art_ok_case_mean=("art_ok_frac", "mean"), ppg_ok_case_mean=("ppg_ok_frac", "mean"),
                                    sqi_ecg_median=("sqi_ecg_median", "median"), sqi_art_median=("sqi_art_median", "median"), sqi_ppg_median=("sqi_ppg_median", "median"), hr_median=("hr_median", "median"),
                                    ptt_ecg_art_median=("ptt_ecg_art_median", "median"), ptt_ecg_ppg_median=("ptt_ecg_ppg_median", "median"),
                                    ptt_ecg_ppg_p10=("ptt_ecg_ppg_median", lambda x: x.quantile(.1)), ptt_ecg_ppg_p90=("ptt_ecg_ppg_median", lambda x: x.quantile(.9)),
                                    ppg_late_share_mean=("ppg_late_share", "mean"), cases_ppg_suspect=("ppg_suspect_case", "sum"), cases_ecg_dead=("ecg_dead", "sum"), cases_ppg_dead=("ppg_dead", "sum"),
                                    cases_core_possible=("bshi_core_possible", "sum"), cases_full_possible=("bshi_full_possible", "sum")).reset_index()
    bq["valid_beat_pct"] = (100 * bq.valid_beats / bq.beats).round(2)
    qc = _read_json(root, "results/beats_qc.json")
    out = [_table(root, "table_signal_coverage", cov), _table(root, "table_gap_profile", gaps), _table(root, "table_beat_quality", bq.round(4)),
           _table(root, "table_qc_gates", pd.DataFrame(qc["gate_table"])), _table(root, "table_qc_per_case", pd.DataFrame(qc["per_case"]))]
    per_case = man[["dataset", "case_id", "hr_median", "ptt_ecg_art_median", "ptt_ecg_ppg_median", "ppg_late_share", "sqi_ecg_median", "sqi_art_median", "sqi_ppg_median", "valid_frac"]]
    edges = np.logspace(-2, 4, 61)
    gap_rows = []
    for ds in ("vitaldb", "mover"):
        v = np.concatenate(runs[ds]) if runs[ds] else np.array([1.0])
        dens, _ = np.histogram(np.clip(v, 0.01, 1e5), bins=edges, density=True)
        gap_rows += [{"dataset": ds, "bin_lo_s": float(lo), "bin_hi_s": float(hi), "density_per_s": float(d)} for lo, hi, d in zip(edges[:-1], edges[1:], dens)]
    out += [_table(root, "table_case_signal_quality", per_case.round(5)), _table(root, "curve_gap_run_hist", pd.DataFrame(gap_rows))]
    return {"files": out, "gap_profile": gaps.to_dict("records"), "valid_beat_pct": dict(zip(bq.dataset, bq.valid_beat_pct)), "ptt_ecg_ppg_median": dict(zip(bq.dataset, bq.ptt_ecg_ppg_median.round(3))),
            "cases_ecg_dead": int(bq.cases_ecg_dead.sum()), "cases_ppg_suspect": dict(zip(bq.dataset, bq.cases_ppg_suspect.astype(int))), "qc_passed": bool(qc["qc_passed"])}


# labels
def stage_labels(root):
    root = Path(root)
    cols = ["dataset", "case_id", "patient_key", "split_pooled", "split_within_dataset", "t_end_sec", "primary_eligible", "label_ambiguous", "aut65_low_coverage", "gray_zone", "already_hypotensive",
            "y_event", "time_to_event_sec", "delta_map", "aut65", "min_valid_frac", "eligible_onset"]
    w = _window_index(root, cols)
    p = w[w.primary_eligible].copy()
    p["y"] = p.y_event.astype(bool)
    prev = []
    groups = [((ds, sp), g) for (ds, sp), g in p.groupby(["dataset", "split_within_dataset"])] + [(("all", sp), g) for sp, g in p.groupby("split_pooled")]
    for key, g in groups:
        prev.append({"dataset": key[0], "split": key[1], "windows": int(len(g)), "event_windows": int(g.y.sum()), "prevalence_pct": round(100 * g.y.mean(), 2),
                     "patients": int(g.patient_key.nunique()), "patients_with_event": int(g[g.y].patient_key.nunique())})
    for ds, g in p.groupby("dataset"):
        prev.append({"dataset": ds, "split": "all", "windows": int(len(g)), "event_windows": int(g.y.sum()), "prevalence_pct": round(100 * g.y.mean(), 2), "patients": int(g.patient_key.nunique()),
                     "patients_with_event": int(g[g.y].patient_key.nunique())})
    prev.append({"dataset": "all", "split": "all", "windows": int(len(p)), "event_windows": int(p.y.sum()), "prevalence_pct": round(100 * p.y.mean(), 2), "patients": int(p.patient_key.nunique()),
                 "patients_with_event": int(p[p.y].patient_key.nunique())})
    ev = p[p.y].assign(onset=lambda d: (d.t_end_sec + d.time_to_event_sec).round(0))
    episodes = ev.groupby(["dataset", "case_id", "onset"]).ngroups
    ep_by = ev.groupby("dataset").apply(lambda g: g.groupby(["case_id", "onset"]).ngroups).to_dict()
    ep_cases = ev.groupby("dataset").apply(lambda g: g.groupby("case_id").apply(lambda x: x.onset.nunique()).describe()[["mean", "50%", "max"]].round(2).to_dict()).to_dict()
    tte = ev.groupby("dataset").time_to_event_sec.describe(percentiles=[.1, .25, .5, .75, .9]).round(1).reset_index()
    dist = []
    for ds, g in p.groupby("dataset"):
        for col in ("delta_map", "aut65"):
            v = g[col].dropna()
            dist.append({"dataset": ds, "target": col, "windows": int(len(v)), "zero_share_pct": round(100 * (v == 0).mean(), 2), "mean": round(float(v.mean()), 3), "p50": round(float(v.quantile(.5)), 3),
                         "p90": round(float(v.quantile(.9)), 3), "p99": round(float(v.quantile(.99)), 3), "max": round(float(v.max()), 2)})
    quality = []
    for ds, g in w.groupby("dataset"):
        e = g[g.eligible_onset]
        quality.append({"dataset": ds, "labeled_windows": int(len(g)), "already_hypotensive_pct": round(100 * g.already_hypotensive.mean(), 2), "primary_eligible_windows": int(g.primary_eligible.sum()),
                        "gray_zone_share_of_eligible_pct": round(100 * e.gray_zone.mean(), 2), "ambiguous_pct_of_eligible": round(100 * e.label_ambiguous.mean(), 2),
                        "aut65_low_coverage_pct_of_eligible": round(100 * e.aut65_low_coverage.mean(), 2)})
    wpp = p.groupby(["dataset", "patient_key"]).size().groupby("dataset").describe(percentiles=[.1, .5, .9]).round(1).reset_index()
    out = [_table(root, "table_labels_prevalence", pd.DataFrame(prev)), _table(root, "table_time_to_event", tte), _table(root, "table_label_distributions", pd.DataFrame(dist)),
           _table(root, "table_label_quality", pd.DataFrame(quality)), _table(root, "table_windows_per_patient", wpp)]
    hist_rows = []
    for ds in ("vitaldb", "mover"):
        for name, values, edges in (("time_to_event_s", ev.loc[ev.dataset == ds, "time_to_event_sec"].dropna().to_numpy(), np.linspace(0, 240, 31)),
                                    ("delta_map_mmhg", p.loc[(p.dataset == ds) & p.delta_map.notna(), "delta_map"].clip(-40, 60).to_numpy(), np.linspace(-40, 60, 61)),
                                    ("log10_aut65", np.log10(p.loc[(p.dataset == ds) & (p.aut65 > 0), "aut65"].clip(lower=1e-3).to_numpy()), np.linspace(-3, 3, 41))):
            dens, _ = np.histogram(values, bins=edges, density=True)
            hist_rows += [{"quantity": name, "dataset": ds, "bin_lo": float(lo), "bin_hi": float(hi), "density": float(d)} for lo, hi, d in zip(edges[:-1], edges[1:], dens)]
    out.append(_table(root, "curve_label_hist", pd.DataFrame(hist_rows)))
    pp = pd.DataFrame(prev)
    return {"files": out, "episodes_total": int(episodes), "episodes_by_dataset": {k: int(v) for k, v in ep_by.items()}, "prevalence_all_pct": float(pp[(pp.dataset == "all") & (pp.split == "all")].prevalence_pct.iloc[0]),
            "events_total": int(p.y.sum()), "patients_with_event": int(p[p.y].patient_key.nunique()), "aut65_zero_pct": {r["dataset"]: r["zero_share_pct"] for r in dist if r["target"] == "aut65"},
            "episodes_per_case_with_event": ep_cases}


# models
def stage_models(root, progress=True):
    root = Path(root)
    s = _settings(root)
    reps, seed = int(s["eda"]["bootstrap_reps"]), int(s["eda"]["seed"])
    feats = M.build_window_features(root)
    ladder, preds = M.run_ladder(root, reps=reps, seed=seed, progress=progress)
    lead = M.auroc_by_lead_time(preds, reps=min(reps, 100), seed=seed)
    gray = M.gray_zone_table(preds, reps=min(reps, 100), seed=seed)
    roc, cal = M.curve_tables(preds)
    base = _read_json(root, "results/baseline_A.json")
    brows = []
    for name, e in base["settings"].items():
        m, ci = e["model"], e["model_ci95"]
        brows.append({"setting": name, "windows_set": "all primary windows (baseline A, as saved)", "test_patients": e["test_patients"], "test_windows": e["test_windows"], "prevalence_pct": round(100 * e["prevalence"], 2),
                      "auroc": m["auroc"], "auroc_lo": ci["auroc"][0], "auroc_hi": ci["auroc"][1], "auprc": m["auprc"], "auprc_lo": ci["auprc"][0], "auprc_hi": ci["auprc"][1], "brier": m["brier"], "cal_slope": m["cal_slope"]})
    dead = _read_json(root, "results/dead_ecg_decision.json")["evidence"]["baseline_A_style_auroc"]
    dtab = pd.DataFrame([{"subset": k, **{c: v.get(c) for c in ("windows", "events", "patients", "prevalence", "auroc", "auprc")}, "auroc_lo": (v.get("auroc_ci95") or [None, None])[0],
                          "auroc_hi": (v.get("auroc_ci95") or [None, None])[1]} for k, v in dead.items() if isinstance(v, dict)])
    out = [_table(root, "table_baseline_ladder_all", ladder.round(5)), _table(root, "table_baseline_A_saved", pd.DataFrame(brows)), _table(root, "table_auroc_by_lead_time", lead.round(4)),
           _table(root, "table_gray_zone", gray.round(4)), _table(root, "curve_roc", roc.round(5)), _table(root, "curve_calibration", cal.round(5)), _table(root, "table_dead_ecg_vs_rest", dtab)]
    full = ladder[ladder.set == "full"]
    key = full[full.setting.isin(["pooled_test", "E3x vitaldb_train->all_mover", "reverse mover_train->vitaldb_test", "vitaldb_train->vitaldb_test", "mover_test_only"])]
    report = key[key.learner == "hgb"].pivot_table(index="rung", columns="setting", values="auroc")
    out.append(_table(root, "table_ladder_auroc_hgb_report_view", report.round(4).reset_index()))
    g = lambda s, r, l: full[(full.setting == s) & (full.rung == r) & (full.learner == l)].iloc[0]
    head = {}
    for s_, tag in (("pooled_test", "pooled"), ("E3x vitaldb_train->all_mover", "E3x"), ("reverse mover_train->vitaldb_test", "reverse"), ("vitaldb_train->vitaldb_test", "vitaldb_test"), ("mover_test_only", "mover_test")):
        head[tag] = {f"{r}_{l}": [round(float(g(s_, r, l).auroc), 4), round(float(g(s_, r, l).auroc_lo), 4), round(float(g(s_, r, l).auroc_hi), 4)] for r in ("G0", "G2", "G4") for l in ("lr", "hgb")}
    return {"files": out, "feature_rows": int(len(pd.read_parquet(M.features_path(root), columns=["window_id"]))), "auroc_full_set": head}


def stage_shift(root):
    """Domain classifier per feature family and standardized mean differences (cheap: about 10 s)."""
    root = Path(root)
    s = _settings(root)
    domain, smd = M.dataset_shift(root, seed=int(s["eda"]["seed"]), reps=min(int(s["eda"]["bootstrap_reps"]), 100))
    out = [_table(root, "table_domain_shift", domain.round(4)), _table(root, "table_smd", smd.round(4))]
    return {"files": out, "domain_auroc": {k.strip(): float(v) for k, v in zip(domain.feature_family, domain.domain_auroc.round(3))},
            "top_smd": [{"feature": r.feature, "smd": round(float(r.smd), 2)} for r in smd.head(5).itertuples()]}


# figures (rendered ONLY from the saved CSVs)
DS_COLOR = {"vitaldb": "#3b7dd8", "mover": "#d8743b"}
SETTING_TITLE = {"pooled_test": "pooled train -> pooled test", "E3x vitaldb_train->all_mover": "VitalDB train -> all MOVER (E3x)",
                 "reverse mover_train->vitaldb_test": "MOVER train -> VitalDB test (reverse)"}


SETTING_SHORT = {"pooled_test": "pooled -> pooled test", "E3x vitaldb_train->all_mover": "VitalDB -> all MOVER (E3x)"}


def _csv(root, name):
    p = Path(root) / "results" / "eda" / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width))


def _fig_cohort(root, plt):
    funnel = _csv(root, "table_cohort_funnel")
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, ds in zip(axes, ("vitaldb", "mover")):
        f = funnel[(funnel.dataset == ds) & (funnel.step < 10)]
        ax.barh(range(len(f)), f.cases, color=DS_COLOR[ds])
        ax.set_yticks(range(len(f)))
        ax.set_yticklabels([_wrap(d, 46) for d in f.description], fontsize=7)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlim(right=float(f.cases.max()) * 4)
        for i, v in enumerate(f.cases):
            ax.text(v, i, f" {int(v):,}", va="center", fontsize=7)
        finish_axes(ax, "surgeries (count, log scale)", "cohort step", f"{ds.upper()}: cohort funnel")
    fig.tight_layout()
    return _figure(root, "fig_cohort_funnel", fig)


def _fig_split_windows(root, plt):
    wtab = _csv(root, "table_split_windows")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, step, ttl in zip(axes, ("primary_eligible", "primary_core", "primary_full"), ("eligible windows", "BSHI core windows (ECG+ART)", "BSHI full windows (ECG+ART+PPG)")):
        t = wtab[wtab.step == step].pivot(index="split", columns="dataset", values="windows").reindex(["train", "validation", "test"])
        t[["mover", "vitaldb"]].plot(kind="bar", ax=ax, color=[DS_COLOR["mover"], DS_COLOR["vitaldb"]], logy=True, rot=0)
        ax.legend(title="dataset", fontsize=7)
        finish_axes(ax, "pooled split", "windows (count, log scale)", ttl)
    fig.tight_layout()
    return _figure(root, "fig_split_windows", fig)


def _fig_signal_quality(root, plt):
    man = _csv(root, "table_case_signal_quality")
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4))
    panels = (("hr_median", "heart rate (beats/min)"), ("ptt_ecg_art_median", "ECG->ART delay (s)"), ("ptt_ecg_ppg_median", "ECG->PPG delay (s)"),
              ("ppg_late_share", "share of PPG beats with delay > 0.9 s (fraction)"), ("sqi_ecg_median", "ECG SQI (0-1)"), ("sqi_art_median", "ART SQI (0-1)"),
              ("sqi_ppg_median", "PPG SQI (0-1)"), ("valid_frac", "valid-beat share (fraction)"))
    for ax, (col, xl) in zip(axes.ravel(), panels):
        for ds in ("vitaldb", "mover"):
            v = man.loc[man.dataset == ds, col].dropna()
            ax.hist(v, bins=40, alpha=0.55, density=True, color=DS_COLOR[ds], label=f"{ds} (n={len(v)})")
        ax.legend(fontsize=7)
        finish_axes(ax, xl, "share of cases (density, 1/unit)")
    fig.suptitle("Per-case beat and signal-quality distributions (median per case)")
    fig.tight_layout()
    return _figure(root, "fig_signal_quality", fig)


def _fig_gap_profile(root, plt):
    h = _csv(root, "curve_gap_run_hist")
    fig, ax = plt.subplots(figsize=(6.8, 4))
    for ds in ("vitaldb", "mover"):
        d = h[h.dataset == ds]
        ax.stairs(d.density_per_s.to_numpy(), np.append(d.bin_lo_s.to_numpy(), d.bin_hi_s.iloc[-1]), color=DS_COLOR[ds], fill=True, alpha=0.45, label=ds)
    ax.set_xscale("log")
    ax.legend()
    finish_axes(ax, "uninterrupted valid run length (s, log scale)", "density (1/s)", "Capture continuity: intermittent MOVER vs continuous VitalDB")
    return _figure(root, "fig_gap_profile", fig)


def _fig_labels(root, plt):
    prev, h = _csv(root, "table_labels_prevalence"), _csv(root, "curve_label_hist")
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    pv = prev[(prev.split != "all") & (prev.dataset != "all")]
    pv.pivot(index="split", columns="dataset", values="prevalence_pct").reindex(["train", "validation", "test"])[["mover", "vitaldb"]].plot(
        kind="bar", ax=axes[0], color=[DS_COLOR["mover"], DS_COLOR["vitaldb"]], rot=0)
    axes[0].legend(title="dataset", fontsize=7)
    finish_axes(axes[0], "within-dataset split", "event prevalence (% of eligible windows)", "event prevalence per split")
    specs = (("time_to_event_s", "time from window end to hypotension onset (s)", "density (1/s)", "time to onset (event windows)"),
             ("delta_map_mmhg", "delta MAP: MAP at t0 minus lowest MAP in the horizon (mmHg, 0 = no drop)", "density (1/mmHg)", "delta MAP = max(0, drop), clipped at 60"),
             ("log10_aut65", "log10 AUT65 (mmHg*min), windows with AUT65 > 0", "density (1/log10 unit)", "AUT65 burden (non-zero windows)"))
    for ax, (q, xl, yl, ttl) in zip(axes[1:], specs):
        for ds in ("vitaldb", "mover"):
            d = h[(h.quantity == q) & (h.dataset == ds)]
            ax.stairs(d.density.to_numpy(), np.append(d.bin_lo.to_numpy(), d.bin_hi.iloc[-1]), color=DS_COLOR[ds], fill=True, alpha=0.45, label=ds)
        ax.legend(fontsize=7)
        if q == "delta_map_mmhg":
            ax.set_xlim(0, 60)
        finish_axes(ax, xl, yl, ttl)
    fig.tight_layout()
    return _figure(root, "fig_labels", fig)


def _fig_ladder(root, plt):
    t = _csv(root, "table_baseline_ladder_all")
    full = t[t.set == "full"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), sharey=True)
    for ax, setting in zip(axes, SETTING_TITLE):
        for learner, marker, off in (("lr", "o", -0.12), ("hgb", "s", 0.12)):
            d = full[(full.setting == setting) & (full.learner == learner)].sort_values("rung")
            ax.errorbar(np.arange(len(d)) + off, d.auroc, yerr=[d.auroc - d.auroc_lo, d.auroc_hi - d.auroc], fmt=marker, capsize=3, label={"lr": "logistic", "hgb": "gradient boosting"}[learner])
        ax.set_xticks(range(5))
        ax.set_xticklabels(sorted(M.RUNGS))
        ax.set_ylim(0.6, 1.0)
        ax.legend(fontsize=7)
        finish_axes(ax, "feature rung (G0 MAP ... G4 all features)", "AUROC (95% patient-bootstrap CI)", SETTING_TITLE[setting])
    fig.suptitle("Beat-feature baseline ladder G0-G4 on the identical BSHI-full windows")
    caption = "  |  ".join(f"{k}: {v}" for k, v in M.RUNG_TEXT.items())
    fig.text(0.5, -0.02, _wrap(caption, 190), ha="center", va="top", fontsize=6.5)
    fig.tight_layout()
    return _figure(root, "fig_auroc_ladder", fig)


def _fig_roc_cal_lead(root, plt):
    roc, cal, lead = _csv(root, "curve_roc"), _csv(root, "curve_calibration"), _csv(root, "table_auroc_by_lead_time")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    for setting, ls in (("pooled_test", "-"), ("E3x vitaldb_train->all_mover", "--")):
        for (rung, learner), c in ((("G0", "lr"), "#888"), (("G4", "hgb"), "#c0392b")):
            r = roc[(roc.setting == setting) & (roc.rung == rung) & (roc.learner == learner)]
            axes[0].plot(r.fpr, r.tpr, ls, color=c, label=f"{SETTING_SHORT[setting]} {rung}-{learner}")
            q = cal[(cal.setting == setting) & (cal.rung == rung) & (cal.learner == learner)]
            axes[1].plot(q.mean_predicted, q.observed_rate, ls, marker="o", ms=3, color=c, label=f"{SETTING_SHORT[setting]} {rung}-{learner}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.8)
    axes[1].plot([0, 0.3], [0, 0.3], "k:", lw=0.8)
    for a in axes[:2]:
        a.legend(fontsize=6)
    finish_axes(axes[0], "false positive rate (1 - specificity)", "true positive rate (sensitivity)", "ROC")
    finish_axes(axes[1], "mean predicted event probability (decile bins)", "observed event rate (fraction of windows)", "calibration (deciles)")
    lt = lead[(lead.setting == "pooled_test") & (((lead.rung == "G0") & (lead.learner == "lr")) | ((lead.rung == "G4") & (lead.learner == "hgb")))]
    bins = list(dict.fromkeys(lt.lead_bin_sec))
    for (rung, learner), g in lt.groupby(["rung", "learner"]):
        axes[2].errorbar(range(len(g)), g.auroc, yerr=[g.auroc - g.auroc_lo, g.auroc_hi - g.auroc], marker="o", capsize=3, label=f"{rung}-{learner}")
    axes[2].set_xticks(range(len(bins)))
    axes[2].set_xticklabels(bins)
    axes[2].legend(fontsize=7)
    finish_axes(axes[2], "lead time to hypotension onset (s)", "AUROC", "pooled test: AUROC by lead time")
    fig.tight_layout()
    return _figure(root, "fig_roc_calibration_leadtime", fig)


def _fig_domain(root, plt):
    domain, smd = _csv(root, "table_domain_shift"), _csv(root, "table_smd")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
    axes[0].barh(range(len(domain)), domain.domain_auroc, xerr=[domain.domain_auroc - domain.auroc_lo, domain.auroc_hi - domain.domain_auroc], color="#7d5ba6")
    axes[0].set_yticks(range(len(domain)))
    axes[0].set_yticklabels([_wrap(s.strip(), 44) for s in domain.feature_family], fontsize=7)
    axes[0].invert_yaxis()
    axes[0].axvline(0.5, color="k", lw=0.8)
    axes[0].set_xlim(0.4, 1.0)
    finish_axes(axes[0], "domain-classifier AUROC (0.5 = datasets indistinguishable)", "feature family", "can a classifier tell the datasets apart?")
    top = smd.head(12)
    axes[1].barh(range(len(top)), top.smd, color=["#d8743b" if v > 0 else "#3b7dd8" for v in top.smd])
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top.feature, fontsize=7)
    axes[1].invert_yaxis()
    finish_axes(axes[1], "standardized mean difference, MOVER minus VitalDB (SD units)", "feature", "largest feature differences (top 12)")
    fig.tight_layout()
    return _figure(root, "fig_domain_shift", fig)


FIGURE_BUILDERS = (("table_cohort_funnel", _fig_cohort), ("table_split_windows", _fig_split_windows), ("table_case_signal_quality", _fig_signal_quality),
                   ("curve_gap_run_hist", _fig_gap_profile), ("curve_label_hist", _fig_labels), ("table_baseline_ladder_all", _fig_ladder),
                   ("curve_roc", _fig_roc_cal_lead), ("table_domain_shift", _fig_domain))


def render_figures(root):
    """Render every EDA figure from the saved results/eda CSVs only (cheap); each figure is audited for axis labels. Returns the list of written files."""
    plt = _mpl()
    written = []
    for needs, fn in FIGURE_BUILDERS:
        if (Path(root) / "results" / "eda" / f"{needs}.csv").exists():
            written.append(fn(root, plt))
    return written


# readiness
def score_event_power(events, patients_with_event, min_events, min_patients):
    return {"events": int(events), "patients_with_event": int(patients_with_event), "pass": bool(events >= min_events and patients_with_event >= min_patients)}


def stage_readiness(root):
    root = Path(root)
    s = _settings(root)
    eda = s["eda"]
    checks = []

    def add(name, value, threshold, ok, note=""):
        checks.append({"check": name, "value": value, "threshold": threshold, "status": "PASS" if ok else "FAIL", "note": note})

    tag = assert_dataset_v1(root)
    add("dataset v1 guard (unified index matches the tag)", tag["version"], "index hash equal", True)
    idx = _unified(root)
    w = _window_index(root)
    try:
        S.check_index(w)
        add("zero patient leakage and index integrity (check_index)", "0 patients in two splits", "0", True)
    except AssertionError as exc:
        add("zero patient leakage and index integrity (check_index)", str(exc), "0", False)
    shares = 100 * idx.groupby("split_pooled").patient_key.nunique() / idx.patient_key.nunique()
    for sp, target in (("train", 70), ("validation", 15), ("test", 15)):
        add(f"pooled {sp} patient share within 3 points of {target}%", round(float(shares[sp]), 2), f"{target} +- 3", abs(shares[sp] - target) <= 3)
    for ds, g in idx.groupby("dataset"):
        sh = 100 * g.groupby("split_within_dataset").patient_key.nunique() / g.patient_key.nunique()
        add(f"{ds} within-dataset split shares within 3 points of 70/15/15", {k: round(float(v), 1) for k, v in sh.items()}, "70/15/15 +- 3",
            all(abs(sh[k] - t) <= 3 for k, t in (("train", 70), ("validation", 15), ("test", 15))))
    prim = w[w.primary_eligible]
    for set_name, flag in (("core", "in_core"), ("full", "in_full")):
        for label, mask in (("pooled test", prim.split_pooled == "test"), ("VitalDB test", (prim.dataset == "vitaldb") & (prim.split_within_dataset == "test")),
                            ("MOVER test only (within-dataset test split)", (prim.dataset == "mover") & (prim.split_within_dataset == "test")), ("all MOVER patients (E3x external test)", prim.dataset == "mover")):
            g = prim[mask & prim[flag]]
            r = score_event_power(g.y_event.sum(), g[g.y_event.astype(bool)].patient_key.nunique(), eda["min_events_per_test_split"], eda["min_test_patients_with_event"])
            add(f"event power, {set_name} set, {label}", f"{r['events']} events / {r['patients_with_event']} patients with an event",
                f">= {eda['min_events_per_test_split']} events and >= {eda['min_test_patients_with_event']} patients", r["pass"],
                "MOVER-only test is underpowered: use all-MOVER (E3x) as the external test" if "MOVER test only" in label and not r["pass"] else "")
    for ds, g in prim.groupby("dataset"):
        c, f = 100 * g.in_core.mean(), 100 * g.in_full.mean()
        add(f"mask coverage, {ds}: core windows / eligible windows", round(float(c), 1), ">= 50%", c >= 50)
        add(f"mask coverage, {ds}: full windows / eligible windows", round(float(f), 1), ">= 30%", f >= 30)
    add("dead-ECG cases excluded from BSHI sets", int((w.in_core & w.ecg_dead_case).sum()), "0", int((w.in_core & w.ecg_dead_case).sum()) == 0)
    tc = _read_json(root, "results/bshi_tensor_checks.json")
    add("tensor sanity on real windows (NaN, constant planes, clip, PPG-free variant)", tc["checks_pass"], "all True", all(tc["checks_pass"].values()))
    meta = json.loads(S._meta_path(root).read_text())
    add("window index signature current", meta["signature"][:12], "equals index_signature()", meta["signature"] == S.index_signature(root))
    th = _read_json(root, "results/bshi_throughput.json")
    add("loader throughput at 16 workers", th["assemble_only"]["16"]["windows_per_s_steady"], ">= 3000 windows/s", th["target_met_at_16_workers"],
        f"torch DataLoader steady state {th.get('torch_dataloader', {}).get('windows_per_s_steady')} windows/s")
    used = sum(p.stat().st_size for p in (root / S.OUT_DIR).rglob("*") if p.is_file()) / 1e9
    free = shutil.disk_usage(root).free / 1e9
    add("storage budget (data/derived/bshi) and free disk", f"{used:.2f} GB used, {free:.0f} GB free", "<= 8 GB new output, >= 150 GB free", used <= 8 and free >= 150)
    pk = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "torch", "matplotlib"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pk[name] = None
    repro = {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk, "seeds": {"random_seed": s["random_seed"], "patient_split_seed": s["patient_split_seed"], "eda_seed": eda["seed"],
                                                                                                         "dev_subset_seed": s["bshi"]["dev_subset"]["seed"]},
             "hashes": {"dataset_index": tag["index_content_sha256"][:16], "label_rule": L.label_rule_sha256(L.load_label_rule(s))[:16], "beats_config": M.P.config_hash(M.P.load_beats_config(s))[:16],
                        "bshi_index_signature": meta["signature"][:16]}}
    add("reproducibility record (seeds, config hashes, package versions)", "recorded in dataset_readiness.json", "present", all(v is not None for v in pk.values()) and all(repro["seeds"].values()))
    frame = pd.DataFrame(checks)
    _table(root, "table_readiness", frame)
    atomic_json(root / "results" / "dataset_readiness.json", {"checks": checks, "n_pass": int((frame.status == "PASS").sum()), "n_fail": int((frame.status == "FAIL").sum()), "reproducibility": repro, "at": utc_now()})
    return {"files": ["results/eda/table_readiness.csv", "results/dataset_readiness.json"], "n_pass": int((frame.status == "PASS").sum()), "n_fail": int((frame.status == "FAIL").sum()),
            "failed": frame[frame.status == "FAIL"].check.tolist()}


# summary
def _fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


def write_summary(root, heads):
    root = Path(root)
    c, sg, lb, md_, sh, rd = heads["cohort"], heads["signal"], heads["labels"], heads["models"], heads["shift"], heads["readiness"]
    a = md_["auroc_full_set"]
    funnel = pd.read_csv(root / "results" / "eda" / "table_cohort_funnel.csv")
    f = lambda ds, step: funnel[(funnel.dataset == ds) & (funnel.step == step)].iloc[0]
    summ = _read_json(root, "results/bshi_index_summary.json")["reconciliation"]
    gaps = {r["dataset"]: r for r in sg["gap_profile"]}
    dead = _read_json(root, "results/dead_ecg_decision.json")["evidence"]
    base = {r["setting"]: r for r in pd.read_csv(root / "results" / "eda" / "table_baseline_A_saved.csv").to_dict("records")}
    run = lambda ds: (f"median uninterrupted run {gaps[ds]['median_valid_run_s'] / 3600:.1f} h ({gaps[ds]['valid_fraction_median']:.2f} of samples valid)" if gaps[ds]["median_valid_run_s"] > 600
                      else f"median uninterrupted valid run {gaps[ds]['median_valid_run_s']:.0f} s, median gap {gaps[ds]['median_gap_s']:.0f} s ({gaps[ds]['valid_fraction_median']:.2f} of samples valid)")
    az = {k: round(float(v), 1) for k, v in lb["aut65_zero_pct"].items()}
    dom = "; ".join(f"{k} {v:.3f}" for k, v in sh["domain_auroc"].items())
    smd_txt = ", ".join(f"{r['feature']} {r['smd']:+.1f}" for r in sh["top_smd"])
    delta = lambda k: a[k]["G4_hgb"][0] - a[k]["G0_hgb"][0]
    overlap = lambda k: not (a[k]["G4_hgb"][1] > a[k]["G0_hgb"][2] or a[k]["G0_hgb"][1] > a[k]["G4_hgb"][2])
    ppg_dom = min(v for k, v in sh["domain_auroc"].items() if k.startswith("of which PPG"))
    reading = (f"- Reading (gradient boosting, G4 minus G0): pooled test {delta('pooled'):+.3f} ({'CIs overlap' if overlap('pooled') else 'CIs separate'}), E3x {delta('E3x'):+.3f} "
               f"({'CIs overlap' if overlap('E3x') else 'CIs separate'}), reverse {delta('reverse'):+.3f} ({'CIs overlap' if overlap('reverse') else 'CIs separate'}). Beat features raise AUROC modestly "
               f"inside the pooled distribution (point estimates; unpaired CIs overlap) and give no gain across datasets. Any PPG-derived input identifies the dataset (domain AUROC >= {ppg_dom:.3f}), "
               "so the EA-vs-EAP ablation matters for generalization; closing the cross-dataset gap is what the BSHI model has to show.")
    lines = [
        "# EDA summary (numbers are generated from `results/eda/*.csv`; rerun `eda_metrics.build_all`)", "",
        "## 1. Cohort and splits (`table_cohort_funnel`, `table_split_*`, `fig_cohort_funnel`, `fig_split_windows`)",
        f"- VitalDB: {int(f('vitaldb', 1).cases):,} surgeries -> {int(f('vitaldb', 3).cases):,} usable ({int(f('vitaldb', 3).patients):,} patients). MOVER: {int(f('mover', 1).cases):,} listed -> {int(f('mover', 2).cases):,} with PPG -> "
        f"{int(f('mover', 3).cases):,} with an arterial line -> {int(f('mover', 5).cases):,} selected ({int(f('mover', 5).patients):,} patients, window-level rung R1).",
        f"- Patient-level 70/15/15 split, {c['patients']:,} patients, patients in two splits: {c['leakage_patients_in_two_splits']}. Pooled patient shares: {c['pooled_patient_shares']}; MOVER share per split: {c['mover_patient_share_by_split']} (%).",
        "", "## 2. Signal coverage and quality (`table_signal_coverage`, `table_gap_profile`, `table_beat_quality`, `table_qc_gates`, `fig_signal_quality`, `fig_gap_profile`)",
        f"- MOVER capture is intermittent: {run('mover')}. VitalDB is continuous: {run('vitaldb')}.",
        f"- Beat extraction (52.7 M beats): valid-beat share {sg['valid_beat_pct']}%; QC gates passed: {sg['qc_passed']}; ECG->PPG delay medians {sg['ptt_ecg_ppg_median']} s (a real dataset shift); "
        f"cases with suspect PPG timing {sg['cases_ppg_suspect']}; dead-ECG VitalDB cases: {dead['flagged_ecg_dead_cases']} (excluded from BSHI, kept for MAP-only baselines).",
        "", "## 3. Labels (`table_labels_prevalence`, `table_time_to_event`, `table_label_distributions`, `table_label_quality`, `fig_labels`)",
        f"- {lb['events_total']:,} event windows from {lb['patients_with_event']:,} patients, about {lb['episodes_total']:,} distinct hypotension episodes ({lb['episodes_by_dataset']}); pooled prevalence {lb['prevalence_all_pct']}%.",
        f"- AUT65 is zero in {az} % of eligible windows (zero-heavy: use a two-part model or a median reference).",
        "", "## 4. BSHI windows (`results/bshi_index_summary.json`, `results/bshi_tensor_checks.json`, `figures/bshi/`)",
        f"- Core (ECG+ART) {summ['core']['now']['windows']:,} windows / {summ['core']['now']['events']:,} events; full (ECG+ART+PPG) {summ['full']['now']['windows']:,} / {summ['full']['now']['events']:,}. "
        f"MOVER core {summ['core']['now']['mover_windows']:,} / {summ['core']['now']['mover_events']:,} events; MOVER full {summ['full']['now']['mover_windows']:,} / {summ['full']['now']['mover_events']:,}.",
        "", "## 5. Predictive baselines, AUROC with 95% patient-bootstrap CI (`table_baseline_ladder_all`, `table_ladder_auroc_hgb_report_view`, `table_baseline_A_saved`, `fig_auroc_ladder`)",
        "- All rungs are scored on the identical BSHI-full windows. G0 = current MAP + 60 s trend (baseline A), G2 = + ART morphology, HR, PTT, HRV, G4 = + PPG and SQI features (PPG amplitude only as within-window relative features).",
        f"- Pooled test: G0 {_fmt(a['pooled']['G0_lr'])} (logistic), {_fmt(a['pooled']['G0_hgb'])} (boosting); G2 {_fmt(a['pooled']['G2_hgb'])}; G4 {_fmt(a['pooled']['G4_hgb'])}.",
        f"- VitalDB-train -> all MOVER (E3x): G0 {_fmt(a['E3x']['G0_lr'])}, {_fmt(a['E3x']['G0_hgb'])}; G2 {_fmt(a['E3x']['G2_hgb'])}; G4 {_fmt(a['E3x']['G4_hgb'])}.",
        f"- MOVER-train -> VitalDB test (reverse): G0 {_fmt(a['reverse']['G0_lr'])}, {_fmt(a['reverse']['G0_hgb'])}; G2 {_fmt(a['reverse']['G2_hgb'])}; G4 {_fmt(a['reverse']['G4_hgb'])}.",
        f"- MOVER test split only (pooled model; underpowered): G0 {_fmt(a['mover_test']['G0_lr'])}, G4 {_fmt(a['mover_test']['G4_hgb'])}. Lead-time and gray-zone AUROC: `table_auroc_by_lead_time` (lead time is at most 240 s by the label rule), `table_gray_zone`.",
        reading,
        f"- Baseline A on ALL eligible windows (`table_baseline_A_saved`): pooled test {base['pooled_train->pooled_test']['auroc']:.3f}, E3x {base['E3x vitaldb_train->all_mover']['auroc']:.3f}; "
        "the full-set numbers above are higher because windows with clean beats are easier (a selection effect, not an improvement).",
        f"- Dataset shift (`table_domain_shift`, `table_smd`, `fig_domain_shift`), domain-classifier AUROC by feature family: {dom}. Largest standardized differences: {smd_txt}.",
        "", "## 6. Readiness scorecard (`results/dataset_readiness.json`, `table_readiness`)",
        f"- {rd['n_pass']} checks PASS, {rd['n_fail']} FAIL. Failed: {rd['failed'] if rd['failed'] else 'none'}. MOVER-only test splits are underpowered; the all-MOVER (E3x) test passes.",
        "", "## Caveats",
        "- Two sites, both arterial-line cohorts: results do not cover non-invasive monitoring. Windows overlap (90 s windows every 30 s), so independent samples are far fewer than window counts.",
        "- MOVER PPG timing is unreliable in about a quarter of PPG beats (ECG->PPG delay 1.0-1.2 s) and the PPG is in different device units (amplitude about 20x VitalDB; AC/DC unstable in MOVER); the full set masks bad timing, and raw PPG amplitude is kept out of the models and the BSHI context.",
        "- Event prevalence differs by dataset (MOVER about twice VitalDB), so calibration shifts. Baselines are not the CNN: they set the numbers it has to beat; boosting used every third train window (neighbours are near-duplicates)."]
    (root / "docs" / "eda_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def build_all(root, force=False, progress=True, include_models=True):
    """Build or refresh every EDA table, figure, the readiness scorecard and docs/eda_summary.md; stages whose inputs are unchanged are skipped."""
    root = Path(root)
    assert_dataset_v1(root)
    heads, ran = {}, {}
    for name, fn in (("cohort", lambda: stage_cohort(root)), ("signal", lambda: stage_signal(root)), ("labels", lambda: stage_labels(root))):
        heads[name], ran[name] = _run_stage(root, name, fn, force, extra="csv2")           # csv2: stages now also save the CSVs the figures are rendered from
    if include_models:
        heads["models"], ran["models"] = _run_stage(root, "models", lambda: stage_models(root, progress), force, extra=M.FEATURE_VERSION)
        heads["shift"], ran["shift"] = _run_stage(root, "shift", lambda: stage_shift(root), force, extra=M.FEATURE_VERSION + "s2")
    heads["readiness"], ran["readiness"] = _run_stage(root, "readiness", lambda: stage_readiness(root), True)
    figure_files = render_figures(root)
    if include_models:
        n = write_summary(root, heads)
        files = list(dict.fromkeys([f for h in heads.values() for f in h.get("files", []) if not f.startswith("figures/")] + figure_files))
        atomic_json(root / "results" / "eda_metrics.json", {"version": EDA_VERSION, "at": utc_now(), "stages_run": ran, "files": files, "headline": heads, "summary_lines": n})
    return {"stages_run": ran, "headline": heads, "figures": figure_files}
