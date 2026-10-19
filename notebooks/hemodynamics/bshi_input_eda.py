"""Compare candidate BSHI context features using patient-disjoint training and validation partitions.

Run as ``python -m hemodynamics.bshi_input_eda --root <workspace>``.
The completed v5 study/configuration and its test results are never modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from . import independent_features as F
from . import independence_study as I

SEED = 5052
G0 = list(I.G0)
IND = ["e_sdnn_l300", "a_map_slope_l60", "a_dpdt_min_l60"]
OUT = Path("results") / "bshi_input_eda"
REFERENCE = "ind3"


def unique(items):
    """Remove duplicate feature names while preserving their order."""
    return list(dict.fromkeys(items))


def write_json(path, obj):
    """Persist analysis metadata as UTF-8 JSON and reject non-finite values."""
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def subsample(frame):
    """Keep every third window within each case without dropping short cases."""
    return frame.loc[frame.groupby(["dataset", "case_id"], sort=False).cumcount().mod(3).eq(0)].copy()


def load_data(root):
    """Load train/validation core windows and create a patient-disjoint inner partition."""
    cols = ["window_id", "dataset", "case_id", "patient_key", "split_pooled", "t_end_sec", "primary_eligible", "in_core", "in_full",
            "y_event", "gray_zone", "time_to_event_sec"] + G0
    idx = pd.read_parquet(root / "data" / "derived" / "bshi" / "window_index_v1.parquet", columns=cols,
                          filters=[("split_pooled", "in", ["train", "validation"]), ("primary_eligible", "==", True), ("in_core", "==", True)])
    assert set(idx.split_pooled) == {"train", "validation"}
    assert idx.window_id.is_unique
    params = pd.read_parquet(root / "data" / "derived" / "independence" / F.PARAMS_NAME, columns=["window_id"] + F.column_names())
    assert params.window_id.is_unique
    df = idx.merge(params, on="window_id", how="inner", validate="one_to_one").dropna(subset=G0)
    df["y"] = df.y_event.astype(np.int8)
    df = df.sort_values(["dataset", "case_id", "t_end_sec"], kind="stable").reset_index(drop=True)
    train = df[df.split_pooled == "train"].copy()
    val = df[df.split_pooled == "validation"].copy()
    assert not set(train.patient_key) & set(val.patient_key)
    pats = train.groupby("patient_key", sort=True).agg(dataset=("dataset", "first"), event=("y", "max"))
    strata = pats.dataset + "_" + pats.event.astype(str)
    fit_pats, check_pats = train_test_split(pats.index.to_numpy(), test_size=0.2, random_state=SEED, stratify=strata)
    fit = subsample(train[train.patient_key.isin(fit_pats)])
    check = subsample(train[train.patient_key.isin(check_pats)])
    assert not set(fit.patient_key) & set(check.patient_key)
    assert set(fit.patient_key) | set(check.patient_key) == set(train.patient_key)
    return train, val, fit, check


def recipes(root):
    """Define candidate feature bundles using physiological groups and the training screen."""
    base = G0 + IND
    specs = {
        "g0": G0,
        "ind3": base,
        "ind6": base + ["a_pp_slope_l60", "e_rr_sampen_l300", "p_int_sdnn_l300"],
        "ind3_map_sd5": base + ["a_map_sd_l300"],
        "ind3_map_range5": base + ["a_map_range_l300"],
        "ind3_variability60": base + ["a_map_sd_l60", "a_pp_cv_l60", "a_spv_l60", "a_ppv_l60"],
        "ind3_multiscale": base + ["a_map_sd_l300", "a_map_sd_l60", "a_pp_cv_l300", "a_map_d60_300", "a_map_slope_l300"],
        "ind3_temporal": base + ["a_map_rsd_l300", "a_map_ac1_l300", "a_map_ktau_var_l300", "a_map_sampen_l300", "e_rr_sampen_l300"],
        "ind3_morphology": base + ["a_sbp_l60", "a_dbp_l60", "a_pp_l60", "a_dpdt_max_l60", "a_rise_l60", "a_tau_l60", "a_notch_rel_l60"],
        "ind3_hrv": base + ["e_rmssd_l300", "e_rr_sampen_l300", "e_rr_dfa_l300", "e_ectopy_l300", "e_rr_ac1_l300"],
        "ind3_quality": base + ["a_q_l60", "e_q_l60", "a_cover_l60", "e_cover_l60"],
        "ind3_self_normalized": G0 + ["e_sdnn_l300_sn", "a_map_slope_l60", "a_dpdt_min_l60_sn"],
    }
    screen = pd.read_csv(root / "results" / "independence" / "table_feasibility.csv")
    safe = screen[screen.domain_auroc.le(0.70)]
    ea = safe[safe.signal.isin(["E", "A"])].param.tolist()
    p = safe[safe.signal.eq("P") & safe.F1].param.tolist()
    specs["ind3_all_ea_low_domain"] = base + ea
    specs["ind3_optional_ppg"] = base + p
    specs["ind3_all_eap_low_domain"] = base + ea + p
    return {k: {"features": unique(v), "strategy": "standard"} for k, v in specs.items()}


def fit_predict(train, evaluation, spec):
    """Fit a missing-value-aware booster on training rows and predict evaluation risks."""
    strategy = spec["strategy"]
    options = dict(max_iter=150, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=200,
                   l2_regularization=1.0, early_stopping=False, random_state=SEED)
    if strategy == "smoother":
        options.update(max_leaf_nodes=15, l2_regularization=10.0)
    elif strategy == "longer":
        options.update(max_iter=250, learning_rate=0.06, l2_regularization=5.0)
    weights = None
    if strategy == "hospital_balanced" and train.dataset.nunique() > 1:
        weights = train.dataset.map((len(train) / (train.dataset.nunique() * train.dataset.value_counts())).to_dict()).to_numpy()
    elif strategy == "patient_balanced":
        weights = (1 / train.patient_key.map(train.patient_key.value_counts())).to_numpy()
        weights = weights / weights.mean()
    model = HistGradientBoostingClassifier(**options)
    x = train[spec["features"]].to_numpy(dtype=np.float64)
    assert np.isfinite(x).sum() > 0 and not np.isinf(x).any()
    model.fit(x, train.y.to_numpy(), sample_weight=weights)
    p = model.predict_proba(evaluation[spec["features"]].to_numpy(dtype=np.float64))[:, 1]
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return p


def scores(df, p):
    """Calculate AUROC, average precision, and Brier score when both classes are present."""
    if df.y.nunique() != 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan}
    return {"auroc": float(roc_auc_score(df.y, p)), "auprc": float(average_precision_score(df.y, p)), "brier": float(brier_score_loss(df.y, p))}


def metric_row(df, p, name, setting):
    """Attach cohort counts and model identifiers to discrimination and calibration metrics."""
    return {"recipe": name, "setting": setting, "windows": len(df), "patients": df.patient_key.nunique(), "positive_windows": int(df.y.sum()), **scores(df, p)}


def inner_compare(fit, check, specs, out, log):
    """Compare recipes on held-out training patients in pooled and source-only settings."""
    rows, predictions = [], {}
    for name, spec in specs.items():
        start = time.perf_counter()
        pooled = fit_predict(fit, check, spec)
        mover = check[check.dataset.eq("mover")]
        source = fit[fit.dataset.eq("vitaldb")]
        external = fit_predict(source, mover, spec)
        predictions[name] = (pooled, external)
        for setting, frame, pred in [("inner_pooled", check, pooled), ("inner_mover_source_only", mover, external)]:
            rows.append(metric_row(frame, pred, name, setting))
        pd.DataFrame(rows).to_csv(out / "inner_comparisons.csv", index=False)
        log(f"inner {name}: pooled {rows[-2]['auroc']:.4f}, source->MOVER {rows[-1]['auroc']:.4f}, {time.perf_counter()-start:.1f}s")
    return pd.DataFrame(rows), predictions


def select_shortlist(inner):
    """Retain references and the strongest pooled and transfer recipes from the inner check."""
    wide = inner.pivot(index="recipe", columns="setting", values="auroc")
    extra = wide.drop(index=["g0", REFERENCE])
    best_pooled = extra.inner_pooled.idxmax()
    eligible = extra[extra.inner_pooled >= wide.loc[REFERENCE, "inner_pooled"] - 0.002]
    best_external = eligible.inner_mover_source_only.idxmax()
    return unique(["g0", REFERENCE, best_pooled, best_external])


def train_eda(train, specs, root, out):
    """Summarize training-only feature coverage, correlations, and risk stratified by MAP."""
    tr = subsample(train)
    catalog = pd.read_csv(root / "results" / "independence" / "table_param_catalogue.csv")
    screen = pd.read_csv(root / "results" / "independence" / "table_feasibility.csv")
    rows = []
    all_features = unique([c for s in specs.values() for c in s["features"]])
    for ds, frame in tr.groupby("dataset"):
        for c in all_features:
            x = frame[c]
            ok = x.notna()
            rows.append({"dataset": ds, "feature": c, "coverage": float(ok.mean()), "median": float(x.median()),
                         "rho_current_map": float(x.corr(frame.map_current, method="spearman")),
                         "missing_positive_share": float(frame.loc[~ok, "y"].mean()) if (~ok).any() else np.nan})
    coverage = pd.DataFrame(rows)
    coverage.merge(screen[["param", "F1", "F3", "F6", "partial_rho_mover", "domain_auroc"]], left_on="feature", right_on="param", how="left").to_csv(out / "feature_eda.csv", index=False)
    catalog.to_csv(out / "feature_definitions.csv", index=False)
    risk = []
    for ds, frame in tr.groupby("dataset"):
        current_bin = pd.qcut(frame.map_current, 10, duplicates="drop")
        for c in ["a_map_sd_l300", "a_map_range_l300", "a_map_sd_l60", "e_sdnn_l300"]:
            feature_bin = pd.qcut(frame[c], 5, duplicates="drop")
            temp = pd.DataFrame({"map_bin": current_bin, "feature_bin": feature_bin, "y": frame.y})
            for (mb, fb), group in temp.groupby(["map_bin", "feature_bin"], observed=True):
                risk.append({"dataset": ds, "feature": c, "map_bin": str(mb), "feature_bin": str(fb), "windows": len(group), "positive_share": float(group.y.mean())})
    pd.DataFrame(risk).to_csv(out / "training_risk_by_map_and_feature.csv", index=False)


def domain_check(fit, check, specs, out):
    """Measure how well each joint feature set predicts the hospital on held-out patients."""
    a, b = fit.copy(), check.copy()
    a["y"], b["y"] = a.dataset.eq("mover").astype(int), b.dataset.eq("mover").astype(int)
    rows = []
    for name, spec in specs.items():
        ds_spec = {**spec, "strategy": "standard"}
        p = fit_predict(a, b, ds_spec)
        rows.append({"recipe": name, "inner_domain_auroc": float(roc_auc_score(b.y, p)), "features": len(spec["features"])})
    pd.DataFrame(rows).to_csv(out / "domain_identifiability.csv", index=False)


def confirmation(train, val, specs, out, reps, log):
    """Evaluate the fixed shortlist on validation patients with paired cluster intervals."""
    rows, preds, strata = [], {}, []
    fit = subsample(train)
    for name, spec in specs.items():
        p = fit_predict(fit, val, spec)
        preds[("pooled_validation", name)] = p
        rows.append(metric_row(val, p, name, "pooled_validation"))
        for ds in ["vitaldb", "mover"]:
            mask = val.dataset.eq(ds).to_numpy()
            rows.append(metric_row(val.loc[mask], p[mask], name, "pooled_fit_" + ds + "_validation"))
        mover = val[val.dataset.eq("mover")]
        ep = fit_predict(fit[fit.dataset.eq("vitaldb")], mover, spec)
        preds[("vitaldb_to_mover_validation", name)] = ep
        rows.append(metric_row(mover, ep, name, "vitaldb_to_mover_validation"))
        for setting, frame, values in [("pooled_validation", val, p), ("vitaldb_to_mover_validation", mover, ep)]:
            groups = {"gray_65_75": frame.gray_zone.to_numpy(bool), "outside_gray": ~frame.gray_zone.to_numpy(bool),
                      "PPG_available": frame.in_full.to_numpy(bool), "PPG_unavailable": ~frame.in_full.to_numpy(bool),
                      "long_ART_available": frame.a_map_sd_l300.notna().to_numpy(), "long_ART_missing": frame.a_map_sd_l300.isna().to_numpy()}
            for lo, hi in [(0, 60), (60, 120), (120, 180), (180, 240)]:
                groups[f"lead_{lo}_{hi}"] = frame.y.eq(0).to_numpy() | (frame.time_to_event_sec.gt(lo) & frame.time_to_event_sec.le(hi)).to_numpy()
            for label, mask in groups.items():
                strata.append({**metric_row(frame.loc[mask], values[mask], name, setting), "subset": label})
        log(f"validation {name}: pooled {rows[-4]['auroc']:.4f}, source->MOVER {rows[-1]['auroc']:.4f}")
        pd.DataFrame(rows).to_csv(out / "validation_comparisons.csv", index=False)
    pd.DataFrame(strata).to_csv(out / "validation_subgroups.csv", index=False)
    uncertainty = []
    for setting, frame in [("pooled_validation", val), ("vitaldb_to_mover_validation", val[val.dataset.eq("mover")])]:
        boot = I.Boot(frame.patient_key, frame.dataset, reps, SEED)
        selected = {name: preds[(setting, name)] for name in specs}
        dist = I.boot_metrics(frame.y.to_numpy(), selected, boot)
        for name, p in selected.items():
            row = {"recipe": name, "setting": setting, "bootstrap_reps": reps, **scores(frame, p)}
            for metric in ["auroc", "auprc"]:
                lo, hi, _ = I.ci(dist[name][metric])
                row[metric + "_lo"], row[metric + "_hi"] = lo, hi
                for ref in ["g0", REFERENCE]:
                    delta = dist[name][metric] - dist[ref][metric]
                    lo, hi, _ = I.ci(delta)
                    row[f"delta_{metric}_vs_{ref}"] = row[metric] - scores(frame, selected[ref])[metric]
                    row[f"delta_{metric}_vs_{ref}_lo"], row[f"delta_{metric}_vs_{ref}_hi"] = lo, hi
            uncertainty.append(row)
    pd.DataFrame(uncertainty).to_csv(out / "paired_validation_uncertainty.csv", index=False)
    pred = val[["window_id", "patient_key", "dataset", "split_pooled"]].copy()
    for name in specs:
        pred[name] = preds[("pooled_validation", name)]
    pred.to_parquet(out / "validation_predictions.parquet", index=False)
    return pd.DataFrame(rows)


def render_figures(root):
    """Plot the stored validation comparison without fitting or evaluating new models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    results = pd.read_csv(root / OUT / "paired_validation_uncertainty.csv")
    labels = {"g0": "MAP Baseline", "ind3": "Original Three", "ind3_multiscale": "Three + Multiscale ART",
              "ind3_all_eap_low_domain": "Large Feature Set", "ind3_all_eap_low_domain__smoother": "Large Set + Smoother Trees"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, setting, title in zip(axes, ["pooled_validation", "vitaldb_to_mover_validation"], ["Pooled Validation", "VitalDB To MOVER Validation"]):
        d = results[results.setting.eq(setting)].set_index("recipe").reindex(labels)
        y = np.arange(len(d))
        for pos, (name, row) in enumerate(d.iterrows()):
            color = "#147d64" if name == "ind3_multiscale" else ("#777777" if name == "g0" else "#356a9e")
            ax.errorbar(row.auroc, pos, xerr=[[max(0, row.auroc-row.auroc_lo)], [max(0, row.auroc_hi-row.auroc)]],
                        fmt="o", color=color, capsize=3)
            ax.annotate(f"{row.auroc:.3f}", (row.auroc, pos), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
        ax.set_yticks(y, list(labels.values()))
        ax.set_title(title)
        ax.set_xlabel("AUROC (Patient-Cluster 95% Interval)")
        ax.grid(axis="x", alpha=0.2)
        ax.set_ylim(len(d)-0.4, -0.7)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory Context Features: Compact Dynamics Versus More Inputs", fontsize=12)
    fig.text(0.5, 0.02, "Tabular Models, Not CNN Results. Test Split Untouched; Intervals Not Search-Adjusted.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    folder = root / "figures" / "bshi_input_eda"
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ["png", "svg"]:
        fig.savefig(folder / ("validation_comparison." + extension), dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    """Run the bounded feature-selection workflow or render figures from saved results."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args()
    root, started = args.root.resolve(), time.perf_counter()
    if args.figures_only:
        render_figures(root)
        return
    out = root / OUT
    out.mkdir(parents=True, exist_ok=False)
    def log(text):
        print(f"[{time.perf_counter()-started:7.1f}s] {text}", flush=True)
    with threadpool_limits(limits=4):
        train, val, fit, check = load_data(root)
        log(f"loaded: train {len(train)}, validation {len(val)}, inner fit {len(fit)}, inner check {len(check)}; TEST excluded")
        cohort = []
        for label, frame in [("train", train), ("validation", val), ("inner_fit_subsample", fit), ("inner_check_subsample", check)]:
            for ds, group in frame.groupby("dataset"):
                cohort.append({"partition": label, "dataset": ds, "windows": len(group), "patients": group.patient_key.nunique(), "positive_windows": int(group.y.sum())})
        pd.DataFrame(cohort).to_csv(out / "cohort.csv", index=False)
        specs = recipes(root)
        assert all(not c.startswith("x_") for spec in specs.values() for c in spec["features"])
        write_json(out / "candidate_recipes.json", specs)
        train_eda(train, specs, root, out)
        inner, _ = inner_compare(fit, check, specs, out, log)
        shortlist = select_shortlist(inner)
        best = inner[inner.setting.eq("inner_pooled") & ~inner.recipe.eq("g0")].sort_values("auroc", ascending=False).recipe.iloc[0]
        strategies = {best + "__" + strategy: {"features": specs[best]["features"], "strategy": strategy}
                      for strategy in ["smoother", "longer", "patient_balanced", "hospital_balanced"]}
        strategy_inner, _ = inner_compare(fit, check, strategies, out, log)
        pd.concat([inner, strategy_inner], ignore_index=True).to_csv(out / "inner_comparisons.csv", index=False)
        best_strategy = strategy_inner[strategy_inner.setting.eq("inner_pooled")].sort_values("auroc", ascending=False).recipe.iloc[0]
        base_score = inner.loc[inner.recipe.eq(best) & inner.setting.eq("inner_pooled"), "auroc"].iloc[0]
        frozen = {name: specs[name] for name in shortlist}
        if strategy_inner.loc[strategy_inner.recipe.eq(best_strategy) & strategy_inner.setting.eq("inner_pooled"), "auroc"].iloc[0] > base_score:
            frozen[best_strategy] = strategies[best_strategy]
        write_json(out / "shortlist_before_validation.json", {"selected_using": "patient-held-out portion of pooled TRAIN only", "recipes": frozen,
                   "selection_rule": "G0 + IND3 references; best inner pooled and best inner source->MOVER with pooled loss <= .002 vs IND3; best strategy only if inner pooled improves",
                   "test_evaluated": False})
        log("shortlist frozen before validation: " + ", ".join(frozen))
        domain_check(fit, check, frozen, out)
        result = confirmation(train, val, frozen, out, args.bootstrap, log)
        manifest = {"seed": SEED, "test_evaluated": False, "exploratory": True, "bootstrap_reps": args.bootstrap,
                    "patient_disjoint_assertions_passed": True, "inner_train_step": 3, "validation_all_windows": True,
                    "standard_learner": "HGB 150 iterations, lr .1, 31 leaves, min leaf 200, l2 1, early_stopping False",
                    "candidate_feature_sets": len(specs), "additional_strategy_checks": len(strategies), "shortlisted_models": list(frozen),
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "frozen_v5_config_sha256": hashlib.sha256((root / "configs" / "hemodynamics.json").read_bytes()).hexdigest(),
                    "runtime_seconds": round(time.perf_counter()-started, 1),
                    "caveats": ["Not a CNN experiment; tabular proxies only", "Validation previously examined during v5; no new confirmatory claim",
                                "Nominal cluster-bootstrap intervals are not corrected for feature search", "No window IDs, hospital identity, future values or test outcomes used as predictors",
                                "Positive-window counts are not unique clinical events", "Missing parameters remain NaN; HGB learns missing-value routing without dropping windows"]}
        write_json(out / "analysis_manifest.json", manifest)
        render_figures(root)
        log("complete; wrote " + str(out))
        print(result[result.setting.isin(["pooled_validation", "vitaldb_to_mover_validation"])].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
