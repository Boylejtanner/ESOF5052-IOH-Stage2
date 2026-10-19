"""Window-level beat features, the G0-G4 beat-feature baseline ladder, dataset-shift analysis (EDA metrics; these are baselines, not the CNN)."""
from __future__ import annotations
# Build the feature baselines and compare their behavior within and across hospitals.

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import beats_pipeline as P
from . import bshi as S
from . import labels as L
from .canonical import atomic_json, utc_now

FEATURE_VERSION = "wf2"
FEATURES_NAME = "window_features_v1.parquet"
ART_MORPH = ["sbp", "dbp", "pp", "map_beat", "dpdt_max", "sys_time", "art_area", "hr_bpm"]
PPG_FEATS = ["ppg_amp", "ppg_ac_dc", "ppg_upstroke", "ptt_ecg_ppg", "ptt_art_ppg", "art_ppg_corr"]
PPG_DEVICE_UNIT = ("ppg_amp", "ppg_ac_dc")        # device-native units (MOVER ~20x VitalDB): only within-window relative features are used
SQI_FEATS = ["sqi_ecg", "sqi_art", "sqi_ppg", "sqi_cross"]
BEAT_COLUMNS = ["t_r", "rr_sec", "valid_beat", "sqi_ecg", "art_ok", "ppg_ok", "ptt_ecg_ppg", "sbp", "dbp", "pp", "map_beat", "dpdt_max", "sys_time", "art_area", "hr_bpm",
                "ptt_ecg_art", "ppg_amp", "ppg_ac_dc", "ppg_upstroke", "ptt_art_ppg", "art_ppg_corr", "sqi_art", "sqi_ppg", "sqi_cross"]
MASK_OF = {**S.MASK_OF, "sbp": "art", "dbp": "art", "map_beat": "art", "hr_bpm": "ecg", "rr_sec": "ecg"}


def _tri(name):
    if name in PPG_DEVICE_UNIT:
        return [f"{name}_cv", f"{name}_rtrend"]
    return [f"{name}_mean", f"{name}_std", f"{name}_trend"]


G1_ADD = [c for n in ART_MORPH for c in _tri(n)]
G2_ADD = _tri("ptt_ecg_art") + ["rr_mean", "sdnn", "rmssd"]
G3_ADD = [c for n in PPG_FEATS for c in _tri(n)]
G4_ADD = [f"{n}_mean" for n in SQI_FEATS] + ["n_ecg_ok", "n_art_ok", "n_ppg_ok"]
RUNGS = {"G0": list(L.FEATURES)}
RUNGS["G1"] = RUNGS["G0"] + G1_ADD
RUNGS["G2"] = RUNGS["G1"] + G2_ADD
RUNGS["G3"] = RUNGS["G2"] + G3_ADD
RUNGS["G4"] = RUNGS["G3"] + G4_ADD
RUNG_TEXT = {"G0": "MAP features (current MAP, 60 s mean/min/std/slope)", "G1": "G0 + ART morphology and HR (mean/std/trend of SBP, DBP, PP, beat MAP, dP/dt, systolic time, area, HR)",
             "G2": "G1 + ECG->ART timing and HRV (PTT mean/std/trend, RR mean, SDNN, RMSSD)", "G3": "G2 + PPG features and PPG delays (full set only; PPG amplitude and AC/DC only as within-window relative features)",
             "G4": "G3 + SQI means and mask counts"}
RAW_PPG_UNITS = [f"{n}_mean" for n in PPG_DEVICE_UNIT]


# features
def masked_stats(v, valid):
    """Per-window mean, std and trend (mean of the last 10 minus the first 10 valid beats of the 60) over the unmasked beats. v, valid: (m, 60)."""
    cnt = valid.sum(axis=1)
    # Calculate the mean level for the following step.
    safe = np.maximum(cnt, 1)
    vv = np.where(valid, v, 0.0)
    mean = vv.sum(axis=1) / safe
    std = np.sqrt(np.where(valid, (v - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe)
    f_ok, l_ok = valid[:, :10], valid[:, -10:]
    first = np.where(f_ok, v[:, :10], 0.0).sum(axis=1) / np.maximum(f_ok.sum(axis=1), 1)
    last = np.where(l_ok, v[:, -10:], 0.0).sum(axis=1) / np.maximum(l_ok.sum(axis=1), 1)
    trend = np.where((f_ok.sum(axis=1) > 0) & (l_ok.sum(axis=1) > 0), last - first, np.nan)
    nan = cnt == 0
    return np.where(nan, np.nan, mean), np.where(nan, np.nan, std), np.where(nan, np.nan, trend)


def window_features(cols, masks, lo):
    """Feature frame (one row per window) from one case's beat columns. ``cols``: name -> (n,) arrays; ``masks``: (m_ecg, m_art, m_ppg); ``lo``: first beat row per window."""
    # Calculate window summaries from valid beats without changing their patient identity.
    idx = np.asarray(lo, dtype=np.int64)[:, None] + np.arange(S.N_BEATS)[None, :]
    mk = dict(zip(("ecg", "art", "ppg"), masks))
    out = {}

    def gather(name):
        v = cols[name][idx].astype(np.float64)
        return v, mk[MASK_OF[name]][idx] & np.isfinite(v)

    for name in ART_MORPH + ["ptt_ecg_art"] + PPG_FEATS:
        v, valid = gather(name)
        mean, std, trend = masked_stats(v, valid)
        if name in PPG_DEVICE_UNIT:
            scale = np.where(np.abs(mean) > 1e-12, np.abs(mean), np.nan)
            out[f"{name}_cv"], out[f"{name}_rtrend"] = std / scale, trend / scale
            out[f"{name}_mean"] = mean                                         # diagnostic only (dataset-shift analysis); in no rung and not in the BSHI context
        else:
            out[f"{name}_mean"], out[f"{name}_std"], out[f"{name}_trend"] = mean, std, trend
    rr, valid = gather("rr_sec")
    mean, std, _ = masked_stats(rr, valid)
    d = np.diff(rr, axis=1)
    pair = valid[:, 1:] & valid[:, :-1]
    out["rr_mean"], out["sdnn"] = mean, std
    out["rmssd"] = np.where(pair.sum(axis=1) > 0, np.sqrt(np.where(pair, d ** 2, 0.0).sum(axis=1) / np.maximum(pair.sum(axis=1), 1)), np.nan)
    for name in SQI_FEATS:
        v, valid = gather(name)
        out[f"{name}_mean"] = masked_stats(v, valid)[0]
    for k, name in enumerate(("n_ecg_ok", "n_art_ok", "n_ppg_ok")):
        out[name] = mk[("ecg", "art", "ppg")[k]][idx].sum(axis=1).astype(np.float64)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in out.items()})


def _feat_job(root, dataset, case_id, rows, cfg):
    # Calculate the required feature columns for the following step.
    t = P.load_beats(root, dataset, case_id, columns=BEAT_COLUMNS)
    cols = {c: t[c].to_numpy() for c in t.columns}
    masks = S.beat_masks(t.valid_beat.to_numpy(), t.sqi_ecg.to_numpy(), t.art_ok.to_numpy(), t.ppg_ok.to_numpy(), t.ptt_ecg_ppg.to_numpy(), cfg)
    f = window_features(cols, masks, rows.beat_lo.to_numpy())
    f.insert(0, "window_id", rows.window_id.to_numpy())
    return f


def features_path(root):
    return Path(root) / S.OUT_DIR / FEATURES_NAME


def build_window_features(root, workers=None, force=False, progress=False):
    """data/derived/bshi/window_features_v1.parquet for every primary-eligible core window (full windows are a subset)."""
    # Hash the content so later runs can detect changed inputs.
    root = Path(root)
    sig = hashlib.sha256((S.index_signature(root) + FEATURE_VERSION).encode()).hexdigest()
    meta = features_path(root).with_suffix(".meta.json")
    if not force and features_path(root).exists() and meta.exists() and json.loads(meta.read_text())["signature"] == sig:
        return {"status": "current", "rows": json.loads(meta.read_text())["rows"]}
    cfg = S.bshi_config(root)
    idx = S.load_window_index(root, columns=["window_id", "dataset", "case_id", "beat_lo", "primary_eligible", "in_core"])
    idx = idx[idx.primary_eligible & idx.in_core]
    n_workers = max(1, min(int(workers or cfg["workers"]), (os.cpu_count() or 8) - 2))
    started = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = [pool.submit(_feat_job, str(root), ds, cid, g.reset_index(drop=True), cfg) for (ds, cid), g in idx.groupby(["dataset", "case_id"], sort=False)]
        frames = [f.result() for f in cf.as_completed(futs)]
    # Write the verified values in the pipeline's Parquet table format.
    frame = pd.concat(frames, ignore_index=True).sort_values("window_id", kind="stable").reset_index(drop=True)
    tmp = Path(str(features_path(root)) + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, features_path(root))
    atomic_json(meta, {"signature": sig, "rows": int(len(frame)), "columns": int(frame.shape[1]) - 1, "seconds": round(time.perf_counter() - started, 1), "at": utc_now()})
    return {"status": "built", "rows": int(len(frame)), "seconds": round(time.perf_counter() - started, 1)}


def load_ladder_frame(root):
    """Index columns + features for the primary core windows (full windows flagged by ``in_full``)."""
    # Join related tables by window id to keep measurements aligned.
    idx = S.load_window_index(root, columns=["window_id", "dataset", "case_id", "patient_key", "split_pooled", "split_within_dataset", "y_event", "gray_zone", "time_to_event_sec",
                                             "primary_eligible", "in_core", "in_full", "ecg_dead_case"] + L.FEATURES)
    idx = idx[idx.primary_eligible & idx.in_core]
    f = pd.read_parquet(features_path(root))
    out = idx.merge(f, on="window_id", how="inner")
    out["y"] = out.y_event.astype(int)
    return out.dropna(subset=L.FEATURES).reset_index(drop=True)


# models
def _model(learner, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if learner == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=300))
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0, random_state=seed)


def fit_predict(train, tests, cols, learner, seed=5052, subsample=3):
    """Fit on every ``subsample``-th train window (windows overlap by 2/3, neighbours are near-duplicates) and score each frame in ``tests``."""
    # Calculate predicted risk without refitting on evaluation windows.
    tr = train.iloc[::subsample] if subsample and len(train) > 60000 else train
    m = _model(learner, seed).fit(tr[cols].to_numpy(np.float64), tr["y"].to_numpy())
    return [m.predict_proba(t[cols].to_numpy(np.float64))[:, 1] for t in tests], int(len(tr))


def evaluate(df, p, reps, seed=5052):
    # Calculate prediction metrics from these labels and model scores.
    d = df[["patient_key", "y"]].assign(p=p)
    met = L._metrics(d.y.to_numpy(), d.p.to_numpy())
    ci = L._cluster_ci(d, "y", "p", reps, seed) if reps else {}
    return {"windows": int(len(d)), "patients": int(d.patient_key.nunique()), "events": int(d.y.sum()), "prevalence": float(d.y.mean()), **met,
            "auroc_lo": ci.get("auroc", [np.nan, np.nan])[0], "auroc_hi": ci.get("auroc", [np.nan, np.nan])[1], "auprc_lo": ci.get("auprc", [np.nan, np.nan])[0],
            "auprc_hi": ci.get("auprc", [np.nan, np.nan])[1]}


def _settings(df):
    # Calculate the spread of the observations for the following step.
    ds, sd, sp = df.dataset, df.split_within_dataset, df.split_pooled
    return {"pooled_train->pooled_test": {"train": sp == "train", "tests": {"pooled_test": sp == "test", "mover_test_only": (sp == "test") & (ds == "mover"),
                                                                           "vitaldb_test_only": (sp == "test") & (ds == "vitaldb")}},
            "vitaldb_train": {"train": (ds == "vitaldb") & (sd == "train"), "tests": {"vitaldb_train->vitaldb_test": (ds == "vitaldb") & (sd == "test"),
                                                                                    "E3x vitaldb_train->all_mover": ds == "mover"}},
            "mover_train": {"train": (ds == "mover") & (sd == "train"), "tests": {"reverse mover_train->vitaldb_test": (ds == "vitaldb") & (sd == "test")}}}


def run_ladder(root, reps=200, seed=5052, progress=True, learners=("lr", "hgb"), keep_preds=("G0", "G4")):
    """Fit/score every rung x learner x model on the FULL set (identical windows for every rung) and G0-G2 on the CORE set. Returns (results frame, predictions dict)."""
    df = load_ladder_frame(root)
    full, core = df[df.in_full], df
    rows, preds = [], {}
    started = time.perf_counter()
    # Process the selected records in order and retain their individual results.
    for set_name, frame, rungs in (("full", full, list(RUNGS)), ("core", core, ["G0", "G1", "G2"])):
        settings = _settings(frame)
        for sname, spec in settings.items():
            if set_name == "core" and sname == "mover_train":
                continue
            train = frame[spec["train"].to_numpy()]
            tests = {k: frame[m.to_numpy()] for k, m in spec["tests"].items()}
            # Process the selected records in order and retain their individual results.
            for rung in rungs:
                for learner in learners:
                    scores, n_fit = fit_predict(train, list(tests.values()), RUNGS[rung], learner, seed)
                    for (tname, t), p in zip(tests.items(), scores):
                        r = evaluate(t, p, reps if len(t) < 400000 else min(reps, 100), seed)
                        rows.append({"set": set_name, "rung": rung, "learner": learner, "setting": tname, "train_windows_used": n_fit, **r})
                        if set_name == "full" and rung in keep_preds and tname in ("pooled_test", "E3x vitaldb_train->all_mover", "reverse mover_train->vitaldb_test"):
                            preds[(tname, rung, learner)] = t[["patient_key", "y", "gray_zone", "time_to_event_sec", "dataset"]].assign(p=p)
                    if progress:
                        print(f"[{time.perf_counter() - started:6.0f}s] {set_name} {sname} {rung} {learner} done", flush=True)
    return pd.DataFrame(rows), preds


# lead time, curves, shift
LEAD_BINS = [(0, 60), (60, 120), (120, 180), (180, 240)]        # an event needs >= 60 s inside the 300 s horizon, so the onset is at most 240 s ahead


def auroc_by_lead_time(preds, reps=100, seed=5052):
    rows = []
    # Apply the same operation to each named field in this group.
    for (setting, rung, learner), d in preds.items():
        if learner != "hgb" and rung != "G0":
            continue
        neg = d[d.y == 0]
        for lo, hi in LEAD_BINS:
            pos = d[(d.y == 1) & (d.time_to_event_sec > lo) & (d.time_to_event_sec <= hi)]
            sub = pd.concat([pos, neg])
            if pos.empty:
                continue
            ci = L._cluster_ci(sub, "y", "p", reps, seed).get("auroc", [np.nan, np.nan])
            rows.append({"setting": setting, "rung": rung, "learner": learner, "lead_bin_sec": f"{lo}-{hi}", "event_windows": int(len(pos)), "patients_with_event": int(pos.patient_key.nunique()),
                         "auroc": L.auroc(sub.y, sub.p), "auroc_lo": ci[0], "auroc_hi": ci[1]})
    return pd.DataFrame(rows)


def gray_zone_table(preds, reps=100, seed=5052):
    rows = []
    for (setting, rung, learner), d in preds.items():
        for name, sub in (("gray_zone 65-75 mmHg", d[d.gray_zone]), ("outside gray zone", d[~d.gray_zone])):
            if sub.y.sum() < 20:
                continue
            ci = L._cluster_ci(sub, "y", "p", reps, seed).get("auroc", [np.nan, np.nan])
            rows.append({"setting": setting, "rung": rung, "learner": learner, "subset": name, "windows": int(len(sub)), "events": int(sub.y.sum()),
                         "auroc": L.auroc(sub.y, sub.p), "auroc_lo": ci[0], "auroc_hi": ci[1]})
    return pd.DataFrame(rows)


def curve_tables(preds, points=200):
    roc, cal = [], []
    # Apply the same operation to each named field in this group.
    for (setting, rung, learner), d in preds.items():
        if learner != "hgb" and rung != "G0":
            continue
        order = np.argsort(-d.p.to_numpy(), kind="stable")
        # Summarize the records separately within each requested group.
        y = d.y.to_numpy()[order]
        tp, fp = np.cumsum(y), np.cumsum(1 - y)
        sel = np.unique(np.linspace(0, len(y) - 1, points).astype(int))
        roc.append(pd.DataFrame({"setting": setting, "rung": rung, "learner": learner, "fpr": fp[sel] / max(1, fp[-1]), "tpr": tp[sel] / max(1, tp[-1])}))
        q = pd.qcut(d.p.rank(method="first"), 10, labels=False)
        c = d.assign(bin=q).groupby("bin").agg(mean_predicted=("p", "mean"), observed_rate=("y", "mean"), windows=("y", "size")).reset_index()
        cal.append(c.assign(setting=setting, rung=rung, learner=learner))
    return pd.concat(roc, ignore_index=True), pd.concat(cal, ignore_index=True)


def dataset_shift(root, seed=5052, reps=100):
    """Domain classifier AUROC (predict dataset from window features, patient-disjoint train/test) per feature family + standardized mean differences (MOVER - VitalDB)."""
    df = load_ladder_frame(root)
    f = df[df.in_full].copy()
    f["y"] = (f.dataset == "mover").astype(int)
    ppg_delays = [c for n in ("ptt_ecg_ppg", "ptt_art_ppg") for c in _tri(n)]
    ppg_morph = [c for c in G3_ADD if c not in ppg_delays]
    families = {"MAP only (G0)": RUNGS["G0"], "ART morphology + HR (G1 additions)": G1_ADD, "ECG->ART timing + HRV (G2 additions)": G2_ADD, "PPG features + delays (G3 additions, scale-free)": G3_ADD,
                "  of which PPG delays only (ECG->PPG, ART->PPG)": ppg_delays, "  of which PPG morphology only (relative amplitude, upstroke, ART-PPG corr)": ppg_morph,
                "SQI means + counts (G4 additions)": G4_ADD, "all features (G4)": RUNGS["G4"], "RAW PPG amplitude and AC/DC (device units; excluded from every model)": RAW_PPG_UNITS}
    rows = []
    train, test = f[f.split_pooled == "train"], f[f.split_pooled == "test"]
    for name, cols in families.items():
        (p,), n_fit = fit_predict(train, [test], cols, "hgb", seed)
        ci = L._cluster_ci(test[["patient_key", "y"]].assign(p=p), "y", "p", reps, seed).get("auroc", [np.nan, np.nan])
        rows.append({"feature_family": name, "features": len(cols), "domain_auroc": L.auroc(test.y, p), "auroc_lo": ci[0], "auroc_hi": ci[1], "test_windows": int(len(test)),
                     "mover_share_test": float(test.y.mean()), "train_windows_used": n_fit})
    smd = []
    for c in RUNGS["G4"] + RAW_PPG_UNITS:
        a, b = f.loc[f.y == 1, c].dropna(), f.loc[f.y == 0, c].dropna()
        pooled = np.sqrt((a.var() + b.var()) / 2)
        smd.append({"feature": c, "mover_mean": float(a.mean()), "vitaldb_mean": float(b.mean()), "smd": float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan})
    smd = pd.DataFrame(smd).assign(abs_smd=lambda d: d.smd.abs()).sort_values("abs_smd", ascending=False).drop(columns="abs_smd")
    return pd.DataFrame(rows), smd
