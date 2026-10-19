"""Evaluate single-signal parameters through correlation analysis, feasibility screening, selection, and transfer comparisons.

Functions only (no global state). Tables -> ``results/independence``, figures -> ``figures/independence`` (200 dpi, Title Case, audited with ``figstyle``).
Pre-registration: ``docs/independence_study_plan.md`` (gates F1-F7, hypotheses H1-H5, success S1-S3, PPG rule). Training windows are used for the atlas,
screen and selection; validation once for confirmation; test once in ``final_comparison``.

Paired differences use ONE set of patient-cluster bootstrap weights (multinomial patient counts, stratified by dataset) shared by every model in a comparison,
so each resample contains the same patients for both models. Point estimates come from ``labels._metrics``.
"""
from __future__ import annotations
# Screen single-signal features for value beyond MAP, coverage, and hospital transfer.

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import eda_models as EM
from . import independent_features as F
from . import labels as L

SEED = 5052
G0 = list(L.FEATURES)
OUT = Path("results") / "independence"
FIG = Path("figures") / "independence"
SIG_COLOR = {"A": "#c0392b", "E": "#2471a3", "P": "#239b56", "old": "#9e9e9e"}
FAM_MARKER = {"level_shape": "o", "level_shape_sn": "D", "variability": "s", "trend": "^", "temporal": "P", "within_coupling": "X", "old": "v"}
LEAD_BINS = list(EM.LEAD_BINS)  # onset is at most 240 s ahead (>= 60 s of the event must lie inside the 300 s horizon)
OLD_ARM_CORE = EM._tri("ptt_ecg_art") + ["x_hr_map_corr_l60"]
OLD_ARM_FULL = OLD_ARM_CORE + [c for n in ("ptt_ecg_ppg", "ptt_art_ppg", "art_ppg_corr") for c in EM._tri(n)] + ["sqi_cross_mean"]
OLD_FAMILIES = {"old ART morphology (ECG-segmented)": (EM.G1_ADD, "core"), "old ECG->ART timing + HRV": (EM.G2_ADD, "core"),
                "old PPG + PPG delays": (EM.G3_ADD, "full"), "old SQI + mask counts": (EM.G4_ADD, "full"), "old cross-signal arm": (OLD_ARM_FULL, "full")}


# data
def load_frame(root, params_name=F.PARAMS_NAME):
    """Core windows (``load_ladder_frame``: index, G0, old wf2 features, ``y``) + ``delta_map``, ``t_end_sec`` + the independent window parameters."""
    # Join related tables by window id to keep measurements aligned.
    root = Path(root)
    from .bshi import load_window_index
    df = EM.load_ladder_frame(root)
    extra = load_window_index(root, "ea_core", columns=["window_id", "delta_map", "t_end_sec"])
    params = pd.read_parquet(root / F.OUT_DIR / params_name, columns=["window_id"] + F.column_names())
    df = df.merge(extra, on="window_id", how="inner").merge(params, on="window_id", how="inner")
    df["is_mover"] = (df.dataset == "mover").astype(int)
    return df.sort_values(["dataset", "case_id", "t_end_sec"], kind="stable").reset_index(drop=True)


def training_subsample(df, full=False, step=3):
    """Every ``step``-th pooled-training window (neighbouring windows overlap by 2/3); ``full`` keeps only windows with PPG (``in_full``)."""
    d = df[df.split_pooled == "train"]
    d = d.iloc[::step]
    return (d[d.in_full] if full else d).reset_index(drop=True)


def old_columns():
    return list(dict.fromkeys(EM.G1_ADD + EM.G2_ADD + EM.G3_ADD + EM.G4_ADD))


# fast paired bootstrap
class Boot:
    """Patient-cluster bootstrap weights shared across models: W[r, patient] = times the patient is drawn in resample r (stratified by dataset)."""

    def __init__(self, patient_key, strata, reps, seed=SEED):
        # Summarize the records separately within each requested group.
        codes, uniq = pd.factorize(pd.Series(patient_key).to_numpy())
        strat = pd.Series(strata).to_numpy()
        pat_stratum = pd.Series(strat).groupby(codes).first().reindex(range(len(uniq))).to_numpy()
        rng = np.random.default_rng(seed)
        W = np.zeros((reps, len(uniq)), dtype=np.float64)
        for s in pd.unique(pat_stratum):
            pats = np.flatnonzero(pat_stratum == s)
            for r in range(reps):
                W[r, pats] += np.bincount(rng.integers(0, len(pats), len(pats)), minlength=len(pats))
        self.W, self.codes, self.reps = W, codes, reps

    def window_weights(self, r):
        return self.W[r][self.codes]

    def patient_sums(self, values):
        """(patients x k) sums of the columns of ``values`` (windows x k)."""
        values = np.asarray(values, dtype=np.float64).reshape(len(self.codes), -1)
        out = np.zeros((self.W.shape[1], values.shape[1]))
        for j in range(values.shape[1]):
            out[:, j] = np.bincount(self.codes, weights=values[:, j], minlength=self.W.shape[1])
        return out


class Ranked:
    """Scores sorted once; weighted AUROC / average precision for any window weights (ties grouped)."""

    def __init__(self, y, s):
        # Order the measurements before matching intervals or comparing rows.
        s = np.asarray(s, dtype=np.float64)
        self.order = np.argsort(s, kind="mergesort")
        ss = s[self.order]
        self.starts = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1))
        self.y = np.asarray(y, dtype=np.float64)[self.order]

    def _groups(self, w):
        w = w[self.order]
        return np.add.reduceat(w * self.y, self.starts), np.add.reduceat(w * (1.0 - self.y), self.starts)

    def auroc(self, w):
        wp, wn = self._groups(w)
        P, N = wp.sum(), wn.sum()
        return float(wp @ (np.cumsum(wn) - 0.5 * wn) / (P * N)) if P > 0 and N > 0 else np.nan

    def auprc(self, w):
        wp, wn = self._groups(w)
        wp, wn = wp[::-1], wn[::-1]
        tp, fp = np.cumsum(wp), np.cumsum(wn)
        P = tp[-1]
        return float((wp * tp / np.maximum(tp + fp, 1e-12)).sum() / P) if P > 0 else np.nan


def boot_metrics(y, preds, boot, metrics=("auroc", "auprc")):
    """{model: {metric: array(reps)}} with the SAME resampled patients for every model."""
    out = {}
    for name, p in preds.items():
        rk = Ranked(y, p)
        out[name] = {m: np.array([getattr(rk, m)(boot.window_weights(r)) for r in range(boot.reps)]) for m in metrics}
    return out


def ci(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, np.nan
    lo, hi = np.percentile(a, [2.5, 97.5])
    return float(lo), float(hi), float(a.std(ddof=1)) if len(a) > 1 else np.nan


def paired_delta(y, p_new, p_ref, boot, metric="auroc"):
    """(point delta, CI lo, CI hi, two-sided normal-approximation p) of metric(p_new) - metric(p_ref) with shared patient resamples."""
    # Use the same resampled patients for both models so their difference is paired.
    f = {"auroc": L.auroc, "auprc": L.auprc}[metric]
    d0 = f(y, p_new) - f(y, p_ref)
    a, b = Ranked(y, p_new), Ranked(y, p_ref)
    ds = np.array([getattr(a, metric)(w) - getattr(b, metric)(w) for w in (boot.window_weights(r) for r in range(boot.reps))])
    lo, hi, se = ci(ds)
    from scipy.stats import norm
    pval = float(2 * norm.sf(abs(d0) / se)) if se and se > 0 else np.nan
    return float(d0), lo, hi, pval


def bh(p, q=0.05):
    """Benjamini-Hochberg: (adjusted p-values, reject mask); NaN p-values are never rejected."""
    # Exclude or flag missing values before using these measurements.
    p = np.asarray(p, dtype=np.float64)
    ok = np.isfinite(p)
    adj = np.full(len(p), np.nan)
    m = int(ok.sum())
    if m:
        # Reserve output arrays and masks with explicit initial values.
        pv = p[ok]
        order = np.argsort(pv)
        ranked = pv[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        a = np.empty(m)
        a[order] = np.minimum(ranked, 1.0)
        adj[ok] = a
    return adj, np.where(ok, adj <= q, False)


# correlations
def domain_auroc(x, is_mover):
    """Orientation-free univariate domain AUROC max(AUC, 1 - AUC) of one parameter (VitalDB vs MOVER), NaN rows dropped."""
    x = np.asarray(x, dtype=np.float64)
    ok = np.isfinite(x)
    if ok.sum() < 50 or len(np.unique(np.asarray(is_mover)[ok])) < 2:
        return np.nan
    a = L.auroc(np.asarray(is_mover)[ok], x[ok])
    return float(max(a, 1 - a))


def spearman_matrix(df, cols):
    """Pairwise-complete Spearman matrix (ranks then Pearson)."""
    return df[cols].rank().corr(min_periods=200)


def _rank_design(Z):
    R = np.column_stack([rankdata(Z[:, j]) for j in range(Z.shape[1])]) if Z.shape[1] else np.zeros((len(Z), 0))
    return np.column_stack([np.ones(len(Z)), R])


def partial_residuals(x, y, Z):
    """Rank-transform x, y and G0 (Z) and residualize the x and y ranks on the G0 ranks (linear regression)."""
    D = _rank_design(Z)
    rx, ry = rankdata(x), rankdata(y)
    bx, *_ = np.linalg.lstsq(D, rx, rcond=None)
    by, *_ = np.linalg.lstsq(D, ry, rcond=None)
    return rx - D @ bx, ry - D @ by


def corr_boot(ex, ey, boot_codes, W):
    """Point and bootstrap Pearson correlation of residuals from patient-level sufficient statistics (W: reps x patients)."""
    n_pat = W.shape[1]
    S = np.column_stack([np.bincount(boot_codes, weights=v, minlength=n_pat) for v in (np.ones_like(ex), ex, ey, ex * ex, ey * ey, ex * ey)])

    def corr(t):
        n, sx, sy, sxx, syy, sxy = t.T if t.ndim == 2 else t
        cov = sxy - sx * sy / n
        return cov / np.sqrt(np.maximum((sxx - sx ** 2 / n) * (syy - sy ** 2 / n), 1e-300))

    return float(corr(S.sum(axis=0))), corr(W @ S)


def partial_spearman(df, col, outcome="delta_map", W=None, codes=None):
    """Partial Spearman of ``col`` with ``outcome`` given G0 (rows with a finite parameter and outcome); optional bootstrap via (W, codes)."""
    # Exclude or flag missing values before using these measurements.
    x = df[col].to_numpy(np.float64)
    y = df[outcome].to_numpy(np.float64)
    Z = df[G0].to_numpy(np.float64)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(Z).all(axis=1)
    if ok.sum() < 200 or np.nanstd(x[ok]) == 0:
        return np.nan, np.nan, np.nan
    ex, ey = partial_residuals(x[ok], y[ok], Z[ok])
    if W is None:
        return float(np.corrcoef(ex, ey)[0, 1]), np.nan, np.nan
    r, rb = corr_boot(ex, ey, codes[ok], W)
    lo, hi, _ = ci(rb)
    return r, lo, hi


# figures
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_figure(fig, root, name, panel_letters=True):
    # Create the destination directory before writing any outputs.
    from . import figstyle as FS
    FS.capitalize_figure(fig, panel_letters=panel_letters)
    FS.audit_figure(fig, name)
    path = Path(root) / FIG / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    _plt().close(fig)
    return path


def save_table(df, root, name):
    # Save the comparison table so the same measurements can be inspected later.
    path = Path(root) / OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def catalogue_frame():
    return pd.DataFrame(F.catalogue()).set_index("name")


# correlation atlas
def cluster_order(rho):
    """Average-linkage order and cluster labels (cut at |rho| >= 0.8) from a Spearman matrix."""
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    # Calculate the working array for the following step.
    from scipy.spatial.distance import squareform
    d = 1.0 - np.abs(rho.to_numpy(np.float64))
    d = np.nan_to_num((d + d.T) / 2.0, nan=1.0)
    np.fill_diagonal(d, 0.0)
    Z = linkage(squareform(d, checks=False), method="average")
    return [rho.index[i] for i in leaves_list(Z)], pd.Series(fcluster(Z, t=0.2, criterion="distance"), index=rho.index)


def _upper(m):
    a = m.to_numpy(np.float64)
    iu = np.triu_indices(len(a), 1)
    return a[iu]


def within_signal_atlas(tr, root):
    """Spearman matrices per signal and dataset, redundancy clusters (pooled), cross-hospital stability; fig 01."""
    # Create the plotting panels for this comparison.
    plt = _plt()
    cat = catalogue_frame()
    sig_cols = {s: F.candidate_names(s) for s in ("A", "E", "P")}
    clusters, stab, orders = {}, [], {}
    fig, axes = plt.subplots(3, 2, figsize=(11, 15))
    # Apply the same operation to each named field in this group.
    for i, (s, cols) in enumerate(sig_cols.items()):
        base = tr[tr.in_full] if s == "P" else tr
        pooled = spearman_matrix(base, cols)
        order, lab = cluster_order(pooled)
        orders[s] = order
        clusters.update({c: f"{s}{int(k)}" for c, k in lab.items()})
        mats = {}
        for j, ds in enumerate(("vitaldb", "mover")):
            # Save the comparison table so the same measurements can be inspected later.
            m = spearman_matrix(base[base.dataset == ds], cols)
            mats[ds] = m
            m.reset_index(names="param").to_csv(Path(root) / OUT / f"table_corr_within_{ {'A': 'art', 'E': 'ecg', 'P': 'ppg'}[s] }_{ds}.csv", index=False)
            # Label the panel with the measurement names and their units.
            ax = axes[i, j]
            im = ax.imshow(m.loc[order, order].to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
            name = {"A": "Arterial Pressure", "E": "ECG", "P": "PPG"}[s]
            ax.set_xticks([]); ax.set_yticks([])
            # Label the panel with the measurement names and their units.
            ax.set_title(f"{name} Parameters, {'VitalDB' if ds == 'vitaldb' else 'MOVER'} (n = {len(cols)})")
            ax.set_xlabel("Parameter (Clustered Order, Index)"); ax.set_ylabel("Parameter (Clustered Order, Index)")
            cb = fig.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label("Spearman Rho (Unitless)")
        a, b = _upper(mats["vitaldb"]), _upper(mats["mover"])
        ok = np.isfinite(a) & np.isfinite(b)
        stab.append({"signal": s, "parameters": len(cols), "pairs": int(ok.sum()), "structure_spearman_vitaldb_vs_mover": float(pd.Series(a[ok]).corr(pd.Series(b[ok]), method="spearman")),
                     "clusters_at_0.8": int(lab.nunique()), "pairs_abs_rho_ge_0.8_vitaldb": int((np.abs(a[ok]) >= 0.8).sum()), "pairs_abs_rho_ge_0.8_mover": int((np.abs(b[ok]) >= 0.8).sum())})
        pd.concat([pooled.reset_index(names="param")]).to_csv(Path(root) / OUT / f"table_corr_within_{ {'A': 'art', 'E': 'ecg', 'P': 'ppg'}[s] }.csv", index=False)
    fig.suptitle("Within-Signal Spearman Correlation of Independent Parameters (Training Windows)")
    save_figure(fig, root, "fig_ind_01_within_signal_correlation.png")
    stab = pd.DataFrame(stab)
    save_table(stab, root, "table_corr_structure_stability.csv")
    return stab, clusters, orders


NAMED_PAIRS = [("a_hr_l60", "e_hr_l60", "ART pulse rate vs ECG HR"), ("a_hr_l60", "p_hr_l60", "ART pulse rate vs PPG rate"),
               ("e_hr_l60", "p_hr_l60", "ECG HR vs PPG rate"), ("a_ppv_l60", "p_pvi_l60", "ART PPV vs PPG PVI"),
               ("a_int_sdnn_l300", "e_sdnn_l300", "ART interval SD vs ECG SDNN"), ("a_int_rmssd_l60", "e_rmssd_l60", "ART interval RMSSD vs ECG RMSSD"),
               ("a_int_ac1_l300", "e_rr_ac1_l300", "ART interval AC1 vs RR AC1"), ("a_int_lfhf_l300", "e_rr_lfhf_l300", "ART interval LF/HF vs RR LF/HF"),
               ("a_pp_cv_l60", "p_amp_cv_l60", "ART PP CV vs PPG amplitude CV"), ("e_hr_slope_l60", "a_map_slope_l60", "ECG HR slope vs ART MAP slope")]


def between_signal_atlas(tr, root, orders):
    """A x E, A x P, E x P Spearman blocks (full windows so all three exist) per dataset and named pairs; fig 02."""
    plt = _plt()
    full = tr[tr.in_full]
    rows = []
    mats = {}
    # Process the selected records in order and retain their individual results.
    for ds in ("vitaldb", "mover"):
        sub = full[full.dataset == ds]
        cols = F.candidate_names("A") + F.candidate_names("E") + F.candidate_names("P")
        m = spearman_matrix(sub, cols)
        mats[ds] = m
        for s1, s2 in (("A", "E"), ("A", "P"), ("E", "P")):
            blk = m.loc[F.candidate_names(s1), F.candidate_names(s2)]
            st = blk.stack().rename("rho").reset_index()
            st.columns = ["param_1", "param_2", "rho"]
            st["dataset"], st["block"] = ds, f"{s1}x{s2}"
            rows.append(st)
    tab = pd.concat(rows, ignore_index=True)
    tab["abs_rho"] = tab.rho.abs()
    save_table(tab.sort_values(["block", "dataset", "abs_rho"], ascending=[True, True, False]), root, "table_corr_between.csv")
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, (s1, s2) in zip(axes.ravel()[:3], (("A", "E"), ("A", "P"), ("E", "P"))):
        # Label the panel with the measurement names and their units.
        m = mats["vitaldb"].loc[orders[s1], orders[s2]]
        im = ax.imshow(m.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto", interpolation="nearest")
        nm = {"A": "Arterial Pressure", "E": "ECG", "P": "PPG"}
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{nm[s1]} x {nm[s2]} Parameters (VitalDB)")
        ax.set_xlabel(f"{nm[s2]} Parameter (Clustered Order, Index)"); ax.set_ylabel(f"{nm[s1]} Parameter (Clustered Order, Index)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046)
        cb.set_label("Spearman Rho (Unitless)")
    ax = axes[1, 1]
    y = np.arange(len(NAMED_PAIRS))
    for k, (ds, off, col) in enumerate((("vitaldb", -0.18, "#34495e"), ("mover", 0.18, "#e67e22"))):
        vals = [mats[ds].loc[a, b] for a, b, _ in NAMED_PAIRS]
        ax.barh(y + off, vals, height=0.34, color=col, label="VitalDB" if ds == "vitaldb" else "MOVER")
    # Label the panel with the measurement names and their units.
    ax.set_yticks(y)
    ax.set_yticklabels([t for _, _, t in NAMED_PAIRS], fontsize=8)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(-1, 1)
    # Collect the per-record values into a table for comparison.
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Same Physiology Measured by Two Signals")
    ax.set_xlabel("Spearman Rho (Unitless)"); ax.set_ylabel("Parameter Pair (Signal vs Signal)")
    # Collect the per-record values into a table for comparison.
    fig.suptitle("Between-Signal Correlation of Independent Parameters (Training Windows With All Three Signals)")
    save_figure(fig, root, "fig_ind_02_between_signal_correlation.png")
    named = pd.DataFrame([{"pair": t, "param_1": a, "param_2": b, "rho_vitaldb": mats["vitaldb"].loc[a, b], "rho_mover": mats["mover"].loc[a, b]} for a, b, t in NAMED_PAIRS])
    return tab, named


TC_INDICATORS = [("a_map_ac1_l300", "MAP Lag-1 Autocorrelation", "Unitless"), ("a_sbp_ac1_l300", "SBP Lag-1 Autocorrelation", "Unitless"),
                 ("a_pp_ac1_l300", "PP Lag-1 Autocorrelation", "Unitless"), ("e_rr_ac1_l300", "RR Lag-1 Autocorrelation", "Unitless"),
                 ("a_map_rsd_l300", "MAP Residual SD", "mmHg"), ("a_sbp_rsd_l300", "SBP Residual SD", "mmHg"),
                 ("a_pp_rsd_l300", "PP Residual SD", "mmHg"), ("e_rr_rsd_l300", "RR Residual SD", "ms")]


def temporal_before_ioh(tr, root, reps=200):
    """H3: event-window value minus the mean of non-event windows of the same dataset and current-MAP decile, by time to onset; fig 03."""
    plt = _plt()
    rows = []
    for ds in ("vitaldb", "mover"):
        # Reserve output arrays and masks with explicit initial values.
        d = tr[tr.dataset == ds].reset_index(drop=True)
        dec = pd.qcut(d.map_current.rank(method="first"), 10, labels=False).to_numpy()
        ev = d.y.to_numpy() == 1
        tte = d.time_to_event_sec.to_numpy(np.float64)
        b = np.full(len(d), -1)
        for k, (lo, hi) in enumerate(LEAD_BINS):
            b[ev & (tte > lo) & (tte <= hi)] = k
        boot = Boot(d.patient_key, np.zeros(len(d)), reps, SEED)
        nb, npat = len(LEAD_BINS), boot.W.shape[1]
        for col, _, _ in TC_INDICATORS:
            # Exclude or flag missing values before using these measurements.
            v = d[col].to_numpy(np.float64)
            ok = np.isfinite(v)
            ne = ok & ~ev
            NE_s = np.zeros((npat, 10)); NE_c = np.zeros((npat, 10))
            np.add.at(NE_s, (boot.codes[ne], dec[ne]), v[ne]); np.add.at(NE_c, (boot.codes[ne], dec[ne]), 1.0)
            EV_s = np.zeros((npat, nb)); EV_c = np.zeros((npat, nb, 10))
            e_ok = ok & (b >= 0)
            np.add.at(EV_s, (boot.codes[e_ok], b[e_ok]), v[e_ok]); np.add.at(EV_c, (boot.codes[e_ok], b[e_ok], dec[e_ok]), 1.0)

            def diff(w):
                base = (w @ NE_s) / np.maximum(w @ NE_c, 1e-12)
                cnt = np.einsum("p,pbd->bd", w, EV_c)
                return ((w @ EV_s) - cnt @ base) / np.maximum(cnt.sum(axis=1), 1e-12), cnt.sum(axis=1)

            point, n_ev = diff(np.ones(npat))
            bs = np.array([diff(boot.W[r])[0] for r in range(reps)])
            for k, (lo, hi) in enumerate(LEAD_BINS):
                l, h, _ = ci(bs[:, k]) if n_ev[k] > 0 else (np.nan, np.nan, np.nan)
                rows.append({"dataset": ds, "indicator": col, "lead_bin_sec": f"{lo}-{hi}", "bin_mid_sec": (lo + hi) / 2, "event_windows": int(n_ev[k]),
                             "diff_vs_matched": float(point[k]) if n_ev[k] > 0 else np.nan, "ci_lo": l, "ci_hi": h})
    tab = pd.DataFrame(rows)
    save_table(tab, root, "table_temporal_correlation.csv")
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)
    for ax, (col, title, unit) in zip(axes.ravel(), TC_INDICATORS):
        for ds, color in (("vitaldb", "#34495e"), ("mover", "#e67e22")):
            # Plot the observed results, including uncertainty where it is available.
            t = tab[(tab.dataset == ds) & (tab.indicator == col) & (tab.event_windows > 0)]
            ax.plot(t.bin_mid_sec, t.diff_vs_matched, "o-", color=color, label="VitalDB" if ds == "vitaldb" else "MOVER")
            ax.fill_between(t.bin_mid_sec, t.ci_lo, t.ci_hi, color=color, alpha=0.2)
        ax.axhline(0, color="k", lw=0.6)
        ax.invert_xaxis()
        ax.set_title(title)
        ax.set_xlabel("Time to IOH Onset (s)"); ax.set_ylabel(f"Event - Matched Non-Event ({unit})")
        ax.legend(fontsize=7)
    fig.suptitle("Temporal Correlation Indicators Before IOH (Event Windows vs Non-Event Windows Matched on Dataset and Current-MAP Decile, 95% Patient-Bootstrap Bands)")
    save_figure(fig, root, "fig_ind_03_temporal_correlation_before_ioh.png")
    return tab


def outcome_correlations(tr, root, reps=200):
    """Spearman with delta_map, AUROC for y_event and partial Spearman given G0 (patient-bootstrap CI) per dataset, new and old parameters."""
    cat = catalogue_frame()
    new = F.candidate_names()
    old = old_columns() + ["x_hr_map_corr_l60"]
    rows = []
    # Process the selected records in order and retain their individual results.
    for ds in ("vitaldb", "mover"):
        d = tr[tr.dataset == ds].reset_index(drop=True)
        boot = Boot(d.patient_key, np.zeros(len(d)), reps, SEED)
        for col in new + old:
            # Exclude or flag missing values before using these measurements.
            base = d[d.in_full].reset_index(drop=True) if (col in new and cat.loc[col, "signal"] == "P") or col in EM.G3_ADD + EM.G4_ADD else d
            bb = boot if base is d else Boot(base.patient_key, np.zeros(len(base)), reps, SEED)
            x = base[col].to_numpy(np.float64)
            # Exclude or flag missing values before using these measurements.
            ok = np.isfinite(x)
            r, lo, hi = partial_spearman(base, col, "delta_map", bb.W, bb.codes)
            au = L.auroc(base.y.to_numpy()[ok], x[ok]) if ok.sum() > 200 else np.nan
            rows.append({"param": col, "dataset": ds, "kind": "new" if col in new else "old", "n": int(ok.sum()),
                         "spearman_delta_map": float(pd.Series(x[ok]).corr(base.delta_map[ok].reset_index(drop=True), method="spearman")) if ok.sum() > 200 else np.nan,
                         "auroc_y": au, "auroc_oriented": max(au, 1 - au) if np.isfinite(au) else np.nan,
                         "partial_rho": r, "partial_lo": lo, "partial_hi": hi})
    tab = pd.DataFrame(rows)
    save_table(tab, root, "table_partial_corr.csv")
    return tab


def domain_table(tr, root):
    """Orientation-free univariate domain AUROC (VitalDB vs MOVER) of each new and old parameter on training windows (P and full-only old features: full windows)."""
    cat = catalogue_frame()
    rows = []
    for col in F.candidate_names() + old_columns() + ["x_hr_map_corr_l60"]:
        # Calculate the working array for the following step.
        is_new = col in cat.index and cat.loc[col, "candidate"]
        full_only = (is_new and cat.loc[col, "signal"] == "P") or col in EM.G3_ADD + EM.G4_ADD
        d = tr[tr.in_full] if full_only else tr
        rows.append({"param": col, "kind": "new" if is_new else "old", "signal": cat.loc[col, "signal"] if is_new else "old",
                     "family": cat.loc[col, "family"] if is_new else "old", "normalization": cat.loc[col, "normalization"] if is_new else "raw",
                     "domain_auroc": domain_auroc(d[col], d.is_mover)})
    # Join related tables by raw, param to keep measurements aligned.
    tab = pd.DataFrame(rows)
    sn = tab[tab.param.str.endswith("_sn")].assign(raw=lambda t: t.param.str[:-3])
    pair = sn.merge(tab[["param", "domain_auroc"]].rename(columns={"param": "raw", "domain_auroc": "domain_auroc_raw"}), on="raw")
    tab = tab.merge(pair[["param", "domain_auroc_raw"]], on="param", how="left")
    save_table(tab, root, "table_domain_auroc.csv")
    return tab


def partial_corr_figure(pc, root):
    # Create the plotting panels for this comparison.
    plt = _plt()
    cat = catalogue_frame()
    w = pc.pivot_table(index=["param", "kind"], columns="dataset", values=["partial_rho", "partial_lo", "partial_hi"]).reset_index()
    w.columns = ["_".join([c for c in col if c]) for col in w.columns]
    fig, ax = plt.subplots(figsize=(9, 8))
    for _, r in w.iterrows():
        if r.kind == "old":
            color, marker = SIG_COLOR["old"], FAM_MARKER["old"]
        else:
            color, marker = SIG_COLOR[cat.loc[r.param, "signal"]], FAM_MARKER[cat.loc[r.param, "family"]]
        ax.errorbar(r.partial_rho_vitaldb, r.partial_rho_mover, xerr=[[max(r.partial_rho_vitaldb - r.partial_lo_vitaldb, 0)], [max(r.partial_hi_vitaldb - r.partial_rho_vitaldb, 0)]],
                    yerr=[[max(r.partial_rho_mover - r.partial_lo_mover, 0)], [max(r.partial_hi_mover - r.partial_rho_mover, 0)]], fmt=marker, color=color, ms=5,
                    elinewidth=0.5, alpha=0.8 if r.kind == "new" else 0.5)
    # Plot the observed results, including uncertainty where it is available.
    lim = np.nanmax(np.abs(w[["partial_rho_vitaldb", "partial_rho_mover"]].to_numpy())) * 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax.axhline(0, color="k", lw=0.4); ax.axvline(0, color="k", lw=0.4)
    from matplotlib.lines import Line2D
    # Save the labeled figure for the analysis report.
    handles = [Line2D([], [], marker="o", ls="", color=SIG_COLOR[s], label=l) for s, l in (("A", "Arterial pressure (A)"), ("E", "ECG (E)"), ("P", "PPG (P)"), ("old", "Old parameters (cross-signal era)"))]
    handles += [Line2D([], [], marker=m, ls="", color="k", label=f.replace("_", " ")) for f, m in FAM_MARKER.items() if f != "old"]
    ax.legend(handles=handles, fontsize=7, loc="upper left")
    # Save the labeled figure for the analysis report.
    ax.set_title("Partial Spearman With Future MAP Drop Given Current MAP (G0), Training Windows")
    ax.set_xlabel("VitalDB Partial Rho (Unitless)"); ax.set_ylabel("MOVER Partial Rho (Unitless)")
    save_figure(fig, root, "fig_ind_04_partial_correlation_vitaldb_vs_mover.png", panel_letters=False)
    return w


# feasibility screen
def fold_ids(groups, k):
    """Patient-grouped fold index per window (sklearn GroupKFold, deterministic)."""
    from sklearn.model_selection import GroupKFold
    out = np.full(len(groups), -1)
    for i, (_, te) in enumerate(GroupKFold(n_splits=k).split(np.zeros(len(groups)), groups=np.asarray(groups))):
        out[te] = i
    return out


def oof(d, cols, learner, folds, seed=SEED):
    """Out-of-fold probabilities of ``_model(learner)`` on the columns ``cols``."""
    X, y = d[cols].to_numpy(np.float64), d.y.to_numpy()
    p = np.zeros(len(d))
    for k in np.unique(folds):
        te = folds == k
        p[te] = EM._model(learner, seed).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    return p


def transfer_auroc(d, cols, learner="lr", seed=SEED):
    """Fit on the VitalDB rows of ``d`` (training split), AUROC on its MOVER rows (MOVER training windows)."""
    # Calculate predicted risk without refitting on evaluation windows.
    v = (d.dataset == "vitaldb").to_numpy()
    X, y = d[cols].to_numpy(np.float64), d.y.to_numpy()
    p = EM._model(learner, seed).fit(X[v], y[v]).predict_proba(X[~v])[:, 1]
    return float(L.auroc(y[~v], p))


def lead_mask(d, lo, hi):
    """All non-event windows + event windows whose onset is in (lo, hi] seconds."""
    y = d.y.to_numpy()
    t = d.time_to_event_sec.to_numpy(np.float64)
    return (y == 0) | ((y == 1) & (t > lo) & (t <= hi))


_WK = {}


def _worker_init(frame_path, reps, threads):
    # Read the stored table required for this calculation.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import sklearn.ensemble  # noqa: F401  (loads the OpenMP runtime so the limit below applies)
    from threadpoolctl import threadpool_limits
    _WK["threads"] = threadpool_limits(limits=threads)
    d = pd.read_parquet(frame_path)
    _WK["core"] = d
    _WK["full"] = d[d.in_full].reset_index(drop=True)
    _WK["reps"] = reps


def _set(s, key):
    """Lazily built per-window-set objects in a worker (folds, bootstrap, G0 baselines)."""
    k = f"{s}:{key}"
    if k not in _WK:
        d = _WK[s]
        if key == "boot":
            _WK[k] = Boot(d.patient_key, d.dataset, _WK["reps"], SEED)
        elif key.startswith("folds"):
            _WK[k] = fold_ids(d.patient_key, int(key[5:]))
        elif key == "lr0":
            _WK[k] = oof(d, G0, "lr", _set(s, "folds5"))
        elif key == "hgb0":
            _WK[k] = oof(d, G0, "hgb", _set(s, "folds3"))
        elif key == "t0":
            _WK[k] = transfer_auroc(d, G0)
    return _WK[k]


def _deltas(prefix, y, p1, p0, boot, d):
    out = {}
    for m in ("auroc", "auprc"):
        dd, lo, hi, pv = paired_delta(y, p1, p0, boot, m)
        out.update({f"{prefix}d_{m}": dd, f"{prefix}d_{m}_lo": lo, f"{prefix}d_{m}_hi": hi, f"{prefix}p_{m}": pv})
    for ds in ("vitaldb", "mover"):
        mk = (d.dataset == ds).to_numpy()
        out[f"{prefix}d_auroc_{ds}"] = float(L.auroc(y[mk], p1[mk]) - L.auroc(y[mk], p0[mk]))
    # Calculate output values for the following step.
    gz = d.gray_zone.to_numpy(bool)
    out[f"{prefix}d_auroc_gray"] = float(L.auroc(y[gz], p1[gz]) - L.auroc(y[gz], p0[gz]))
    out[f"{prefix}d_auprc_gray"] = float(L.auprc(y[gz], p1[gz]) - L.auprc(y[gz], p0[gz]))
    far = lead_mask(d, 180, 240)
    out[f"{prefix}d_auroc_180_240"] = float(L.auroc(y[far], p1[far]) - L.auroc(y[far], p0[far]))
    return out


def _lr_job(name, cols, wset):
    """F3 (5-fold OOF lr, paired bootstrap) + F4 (VitalDB -> MOVER training) for one parameter or one old family added to G0."""
    # Calculate the working array and the target or processed signal for the following step.
    d = _WK[wset]
    y = d.y.to_numpy()
    p1 = oof(d, G0 + list(cols), "lr", _set(wset, "folds5"))
    out = {"param": name, "window_set": wset, "windows": int(len(d)), "lr_auroc_g0": float(L.auroc(y, _set(wset, "lr0"))), "lr_auprc_g0": float(L.auprc(y, _set(wset, "lr0")))}
    out.update(_deltas("lr_", y, p1, _set(wset, "lr0"), _set(wset, "boot"), d))
    out["transfer_auroc_g0"] = _set(wset, "t0")
    out["transfer_auroc"] = transfer_auroc(d, G0 + list(cols))
    out["transfer_d_auroc"] = out["transfer_auroc"] - out["transfer_auroc_g0"]
    return out


def _hgb_job(name, cols, wset):
    """Boosting check: 3-fold OOF hgb G0 + parameter vs G0 (paired bootstrap)."""
    d = _WK[wset]
    y = d.y.to_numpy()
    p1 = oof(d, G0 + list(cols), "hgb", _set(wset, "folds3"))
    return {"param": name, **_deltas("hgb_", y, p1, _set(wset, "hgb0"), _set(wset, "boot"), d)}


def _oof_metric_job(tag, cols, wset):
    """Greedy step: point OOF AUROC / AUPRC of hgb on ``cols`` (3 folds)."""
    # Reserve output arrays and masks with explicit initial values.
    d = _WK[wset]
    y = d.y.to_numpy()
    p = oof(d, list(cols), "hgb", _set(wset, "folds3")) if cols else np.full(len(d), 0.5)
    return {"tag": tag, "auroc": float(L.auroc(y, p)), "auprc": float(L.auprc(y, p))}


class Pool:
    """Spawned worker pool sharing one training frame (read once per worker); ``run`` executes (fn, name, cols, wset) jobs and returns results in job order."""
    FNS = {"lr": _lr_job, "hgb": _hgb_job, "oof": _oof_metric_job}

    def __init__(self, frame_path, workers, reps=200, threads=1):
        self.args = (str(frame_path), reps, threads)
        self.workers = workers

    def __enter__(self):
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        self.ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"), initializer=_worker_init, initargs=self.args)
        return self

    def __exit__(self, *exc):
        self.ex.shutdown(wait=True, cancel_futures=True)

    def run(self, jobs, log=None):
        # Submit bounded jobs and collect completed work without losing pending tasks.
        from concurrent.futures import as_completed
        t0 = time.perf_counter()
        futs = {self.ex.submit(self.FNS[f], n, c, w): i for i, (f, n, c, w) in enumerate(jobs)}
        out = [None] * len(jobs)
        for k, fu in enumerate(as_completed(futs), 1):
            out[futs[fu]] = fu.result()
            if log and (k % 20 == 0 or k == len(jobs)) and len(jobs) >= 20:
                log(f"  pool {k}/{len(jobs)} [{time.perf_counter() - t0:.0f}s]")
        return out


def run_pool(jobs, frame_path, workers, reps=200, threads=1, log=None):
    with Pool(frame_path, workers, reps, threads) as px:
        return px.run(jobs, log)


def wset_of(col):
    cat = catalogue_frame()
    if col in cat.index:
        return "full" if cat.loc[col, "signal"] == "P" else "core"
    return "full" if col in EM.G3_ADD + EM.G4_ADD else "core"


def quality_col(signal):
    return {"A": "a_q_l60", "E": "e_q_l60", "P": "p_q_l60"}[signal]


def robustness_f7(tr, col, signal):
    """Pooled partial rho and partial rho in the bottom / top tertile (rank-based) of the parameter's own signal quality."""
    # Exclude or flag missing values before using these measurements.
    base = tr[tr.in_full] if signal == "P" else tr
    r_all, _, _ = partial_spearman(base, col)
    q = base[quality_col(signal)]
    # Exclude or flag missing values before using these measurements.
    ok = q.notna()
    terc = pd.Series(np.nan, index=base.index)
    terc[ok] = pd.qcut(q[ok].rank(method="first"), 3, labels=False)
    r_lo, _, _ = partial_spearman(base[terc == 0], col)
    # Exclude or flag missing values before using these measurements.
    r_hi, _, _ = partial_spearman(base[terc == 2], col)
    ok7 = np.isfinite([r_all, r_lo, r_hi]).all() and np.sign(r_lo) == np.sign(r_all) == np.sign(r_hi)
    return r_all, r_lo, r_hi, bool(ok7)


def feasibility_screen(tr, root, frame_path, pc, dom, workers=16, reps=200, log=print):
    """Every F1-F7 metric for each of the 147 candidates (+ old parameters and old families as reference); boosting check for cluster representatives."""
    # Measure every candidate against the same coverage, gain, transfer, and domain gates.
    cat = catalogue_frame()
    cands = F.candidate_names()
    old = old_columns() + ["x_hr_map_corr_l60"]
    # Collect the per-record values into a table for comparison.
    jobs = [("lr", c, [c], wset_of(c)) for c in cands + old]
    jobs += [("lr", f"[family] {k}", v, w) for k, (v, w) in OLD_FAMILIES.items()]
    log(f"F3/F4 lr jobs: {len(jobs)}")
    lr = pd.DataFrame(run_pool(jobs, frame_path, workers, reps, 1, log))
    rows = []
    rho_map = tr[cands + old].corrwith(tr.map_current, method="spearman")
    for c in cands:
        # Exclude or flag missing values before using these measurements.
        s = cat.loc[c, "signal"]
        base = tr[tr.in_full] if s == "P" else tr
        cov_all = float(base[c].notna().mean())
        cov_mov = float(base.loc[base.dataset == "mover", c].notna().mean())
        r_all, r_lo, r_hi, f7 = robustness_f7(tr, c, s)
        rows.append({"param": c, "signal": s, "family": cat.loc[c, "family"], "lookback": cat.loc[c, "lookback"], "normalization": cat.loc[c, "normalization"],
                     "coverage": cov_all, "coverage_mover": cov_mov, "rho_map_current": float(rho_map[c]),
                     "partial_rho_pooled": r_all, "partial_rho_q_low": r_lo, "partial_rho_q_high": r_hi, "F7": f7})
    # Join related tables by param to keep measurements aligned.
    t = pd.DataFrame(rows)
    pcw = pc[pc.kind == "new"].pivot_table(index="param", columns="dataset", values=["partial_rho", "partial_lo", "partial_hi", "auroc_oriented"])
    pcw.columns = [f"{a}_{b}" for a, b in pcw.columns]
    t = t.merge(pcw.reset_index(), on="param", how="left").merge(dom[["param", "domain_auroc", "domain_auroc_raw"]], on="param", how="left")
    t = t.merge(lr, on="param", how="left")
    # Apply the minimum overall and MOVER coverage requirements.
    t["F1"] = (t.coverage >= 0.90) & (t.coverage_mover >= 0.85)
    vit_ok = (t.partial_lo_vitaldb > 0) | (t.partial_hi_vitaldb < 0)
    t["F2"] = vit_ok & (np.sign(t.partial_rho_mover) == np.sign(t.partial_rho_vitaldb))
    # Control repeated feature comparisons before accepting an incremental discrimination gain.
    t["bh_auroc"], rej_a = bh(t.lr_p_auroc.to_numpy())
    t["bh_auprc"], rej_p = bh(t.lr_p_auprc.to_numpy())
    t["F3"] = ((t.lr_d_auroc_lo > 0) & rej_a & (t.lr_d_auroc > 0)) | ((t.lr_d_auprc_lo > 0) & rej_p & (t.lr_d_auprc > 0))
    # Summarize the records separately within each requested group.
    t["F4"] = t.transfer_d_auroc >= -0.002
    t["F5"] = t.domain_auroc <= 0.70
    rho = spearman_matrix(tr, cands)
    # Summarize the records separately within each requested group.
    _, lab = cluster_order(rho)
    t["cluster"] = t.param.map(lab).astype(int)
    t["cluster_size"] = t.groupby("cluster").param.transform("size")
    # Combine coverage, association, model gain, transfer, and domain checks before redundancy filtering.
    pre = t.F1 & t.F2 & t.F3 & t.F4 & t.F5
    rep = t[pre].sort_values("lr_d_auprc", ascending=False).groupby("cluster").param.first()
    t["cluster_representative"] = t.param.isin(rep.values)
    t["F6"] = (t.rho_map_current.abs() < 0.5) & t.cluster_representative
    t["feasible"] = pre & t.F6
    reps_f15 = t.loc[t.cluster_representative, "param"].tolist()
    log(f"pre-F6 pass: {int(pre.sum())}, cluster representatives: {len(reps_f15)}, feasible: {int(t.feasible.sum())}")
    if reps_f15:
        hg = pd.DataFrame(run_pool([("hgb", c, [c], wset_of(c)) for c in reps_f15], frame_path, max(1, workers // 2), reps, 2, log))
        t = t.merge(hg, on="param", how="left")
    ref = lr[~lr.param.isin(cands)].copy()
    save_table(t, root, "table_feasibility.csv")
    save_table(ref, root, "table_feasibility_old_reference.csv")
    return t, ref


def feasibility_figures(t, ref, dom, root):
    # Join related tables by param to keep measurements aligned.
    plt = _plt()
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.axvspan(0.5, 0.70, ymin=0, ymax=1, color="#d5f5e3", alpha=0.5, zorder=0)
    old = ref[~ref.param.str.startswith("[family]")].merge(dom[["param", "domain_auroc"]], on="param", how="left")
    ax.errorbar(old.domain_auroc, old.lr_d_auroc, yerr=[np.maximum(old.lr_d_auroc - old.lr_d_auroc_lo, 0), np.maximum(old.lr_d_auroc_hi - old.lr_d_auroc, 0)],
                fmt="v", color=SIG_COLOR["old"], ms=4, elinewidth=0.5, alpha=0.6, label="Old parameters (G1-G4, cross-signal)")
    for s, lab in (("A", "Arterial pressure (A)"), ("E", "ECG (E)"), ("P", "PPG (P)")):
        # Plot the observed results, including uncertainty where it is available.
        u = t[t.signal == s]
        ax.errorbar(u.domain_auroc, u.lr_d_auroc, yerr=[np.maximum(u.lr_d_auroc - u.lr_d_auroc_lo, 0), np.maximum(u.lr_d_auroc_hi - u.lr_d_auroc, 0)],
                    fmt="o", color=SIG_COLOR[s], ms=4, elinewidth=0.5, alpha=0.75, label=lab)
    # Plot the observed results, including uncertainty where it is available.
    f = t[t.feasible]
    ax.scatter(f.domain_auroc, f.lr_d_auroc, s=90, facecolors="none", edgecolors="k", lw=1.2, label="Feasible (F1-F6)", zorder=5)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(0.70, color="k", ls="--", lw=0.7)
    # Save the labeled figure for the analysis report.
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_title("Feasibility Quadrant: Hospital Identifiability vs Incremental Value Over Current MAP (Training, Out-of-Fold)")
    # Collect the per-record values into a table for comparison.
    ax.set_xlabel("Univariate Domain AUROC, VitalDB vs MOVER (Unitless)"); ax.set_ylabel("Delta AUROC of G0 + Parameter vs G0, Logistic (Unitless)")
    save_figure(fig, root, "fig_ind_05_feasibility_quadrant.png", panel_letters=False)

    # Collect the per-record values into a table for comparison.
    top = t.sort_values("lr_d_auprc", ascending=False).head(30)
    fam = ref[ref.param.str.startswith("[family]")].assign(signal="old")
    rows = pd.concat([fam[["param", "signal", "lr_d_auroc", "lr_d_auroc_lo", "lr_d_auroc_hi", "lr_d_auprc", "lr_d_auprc_lo", "lr_d_auprc_hi"]],
                      top[["param", "signal", "lr_d_auroc", "lr_d_auroc_lo", "lr_d_auroc_hi", "lr_d_auprc", "lr_d_auprc_lo", "lr_d_auprc_hi", "feasible"]]], ignore_index=True)
    # Create the plotting panels for this comparison.
    rows = rows.iloc[::-1].reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 12), sharey=True)
    yy = np.arange(len(rows))
    for ax, m, lab in ((axes[0], "auroc", "Delta AUROC vs G0 (Unitless)"), (axes[1], "auprc", "Delta AUPRC vs G0 (Unitless)")):
        for i, r in rows.iterrows():
            # Plot the observed results, including uncertainty where it is available.
            ax.errorbar(r[f"lr_d_{m}"], i, xerr=[[max(r[f"lr_d_{m}"] - r[f"lr_d_{m}_lo"], 0)], [max(r[f"lr_d_{m}_hi"] - r[f"lr_d_{m}"], 0)]],
                        fmt="s" if r.signal == "old" else "o", color=SIG_COLOR[r.signal], ms=4, elinewidth=0.8,
                        mfc="white" if (r.signal != "old" and not bool(r.get("feasible", False))) else SIG_COLOR[r.signal])
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab)
        ax.set_title(f"Incremental {m.upper()} Over G0, Logistic, 5-Fold Out-of-Fold")
    # Save the labeled figure for the analysis report.
    axes[0].set_yticks(yy)
    axes[0].set_yticklabels([p.replace("[family] ", "FAMILY: ") for p in rows.param], fontsize=7)
    axes[0].set_ylabel("Parameter (Top 30 by Delta AUPRC) and Old Families (Reference)")
    axes[1].set_ylabel("Parameter (Same Order)")
    fig.suptitle("Incremental Value of Single Independent Parameters vs Old Parameter Families (Filled = Feasible, Open = Not Feasible, Grey Squares = Old)")
    save_figure(fig, root, "fig_ind_06_incremental_value_forest.png")


# selection, confirmation, freeze
SETS = {"IND-A": (("A",), "core"), "IND-E": (("E",), "core"), "IND-P": (("P",), "full"), "IND-EA": (("A", "E"), "core"), "IND-EAP": (("A", "E", "P"), "full")}
SET_COLOR = {"IND-A": SIG_COLOR["A"], "IND-E": SIG_COLOR["E"], "IND-P": SIG_COLOR["P"], "IND-EA": "#8e44ad", "IND-EAP": "#d35400"}


def slug(s):
    return s.replace(" + ", "_plus_").replace(" ", "_").replace("-", "")


def selection_pools(t, cap=20):
    """Feasible parameters per set, ordered by the boosting check (hgb Delta AUPRC; lr Delta AUPRC when absent), capped at ``cap``."""
    # Order the measurements before matching intervals or comparing rows.
    f = t[t.feasible.astype(bool)].copy()
    key = "hgb_d_auprc" if "hgb_d_auprc" in f.columns else "lr_d_auprc"
    f = f.sort_values([key, "lr_d_auprc"], ascending=False)
    return {n: f[f.signal.isin(sigs)].param.head(cap).tolist() for n, (sigs, _) in SETS.items()}


def greedy_select(pools, px, log=print, max_k=12):
    """Forward selection from G0 on 3-fold out-of-fold hgb (largest Delta AUPRC); stop when the best step has Delta AUROC < 0.001 AND Delta AUPRC < 0.002."""
    refs = px.run([("oof", "G0|core", G0, "core"), ("oof", "G0|full", G0, "full"), ("oof", "G2|core", EM.RUNGS["G2"], "core"), ("oof", "G4|full", EM.RUNGS["G4"], "full")])
    ref = {r["tag"]: r for r in refs}
    st = {n: {"cols": [], "cur": ref[f"G0|{w}"], "active": bool(pools[n])} for n, (_, w) in SETS.items()}
    path = [{"set": n, "window_set": SETS[n][1], "step": 0, "added": "(G0 only)", "auroc": st[n]["cur"]["auroc"], "auprc": st[n]["cur"]["auprc"], "d_auroc": 0.0,
             "d_auprc": 0.0, "accepted": True, "candidates_tried": 0, "pool_size": len(pools[n])} for n in SETS]
    # Accumulate the calculation over the required samples or iterations.
    for step in range(1, max_k + 1):
        jobs = [("oof", f"{n}|{c}", G0 + st[n]["cols"] + [c], SETS[n][1]) for n in SETS if st[n]["active"] for c in pools[n] if c not in st[n]["cols"]]
        if not jobs:
            break
        by = {}
        for r in px.run(jobs, log):
            n, c = r["tag"].split("|", 1)
            by.setdefault(n, []).append((c, r))
        # Apply the same operation to each named field in this group.
        for n, lst in by.items():
            c, r = max(lst, key=lambda z: z[1]["auprc"])
            da, dp = r["auroc"] - st[n]["cur"]["auroc"], r["auprc"] - st[n]["cur"]["auprc"]
            ok = not (da < 0.001 and dp < 0.002)
            path.append({"set": n, "window_set": SETS[n][1], "step": step, "added": c, "auroc": r["auroc"], "auprc": r["auprc"], "d_auroc": da, "d_auprc": dp,
                         "accepted": ok, "candidates_tried": len(lst), "pool_size": len(pools[n])})
            if ok:
                st[n]["cols"].append(c)
                st[n]["cur"] = r
            if not ok or len(st[n]["cols"]) >= max_k or len(st[n]["cols"]) == len(pools[n]):
                st[n]["active"] = False
        log(f"greedy step {step}: " + ", ".join(f"{n}={len(st[n]['cols'])}" for n in SETS))
    return {n: st[n]["cols"] for n in SETS}, pd.DataFrame(path), ref


def l1_path(tr, pools, sets, Cs=None):
    """L1-logistic entry order (standardized, median-imputed G0 + pool) and overlap of its first k entrants with the greedy set of size k."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    Cs = np.logspace(-4, 0, 13) if Cs is None else Cs
    rows = []
    # Apply the same operation to each named field in this group.
    for n, (_, wset) in SETS.items():
        pool = pools[n]
        if not pool:
            continue
        d = tr[tr.in_full] if wset == "full" else tr
        X = StandardScaler().fit_transform(SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(d[G0 + pool].to_numpy(np.float64)))
        y = d.y.to_numpy()
        entry = {}
        for C in Cs:
            coef = LogisticRegression(l1_ratio=1.0, solver="liblinear", C=C, max_iter=300).fit(X, y).coef_[0][len(G0):]
            for c, v in zip(pool, coef):
                if abs(v) > 1e-8 and c not in entry:
                    entry[c] = float(C)
        order = sorted(pool, key=lambda c: (entry.get(c, np.inf), pool.index(c)))
        k = len(sets[n])
        overlap = len(set(order[:k]) & set(sets[n])) / k if k else np.nan
        rows += [{"set": n, "param": c, "l1_entry_C": entry.get(c, np.nan), "l1_rank": i + 1, "in_greedy_set": c in sets[n], "overlap_first_k": overlap}
                 for i, c in enumerate(order)]
    return pd.DataFrame(rows, columns=["set", "param", "l1_entry_C", "l1_rank", "in_greedy_set", "overlap_first_k"])


def selection_figure(path, ref, root):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # Process the selected records in order and retain their individual results.
    for ax, m in zip(axes, ("auroc", "auprc")):
        for n in SETS:
            p = path[(path.set == n) & path.accepted.astype(bool)]
            ax.plot(p.step, p[m], "o-", color=SET_COLOR[n], label=f"G0 + {n} ({SETS[n][1]} windows)")
        for tag, ls, lab in (("G0|core", "--", "G0 (core windows)"), ("G0|full", ":", "G0 (full windows)"), ("G2|core", "-.", "Old G2 (core windows)"),
                             ("G4|full", (0, (5, 1, 1, 1)), "Old G4 (full windows)")):
            ax.axhline(ref[tag][m], color="grey", ls=ls, lw=1, label=lab)
        ax.set_xlabel("Number of Added Independent Parameters (Count)")
        ax.set_ylabel(f"Out-of-Fold {m.upper()} (Unitless)")
        ax.set_title(f"Greedy Forward Selection Path, {m.upper()}")
        ax.legend(fontsize=7)
    fig.suptitle("Compact-Set Selection on Training Windows (Gradient Boosting, 3-Fold Patient-Grouped Out-of-Fold)")
    save_figure(fig, root, "fig_ind_07_selection_path.png")


def clean_settings(df, stage):
    """Pooled-split settings (patient-disjoint from the screening/selection windows): {name: (train mask, {test name: mask})}."""
    ds, sp = df.dataset, df.split_pooled
    return {f"pooled_train->pooled_{stage}": (sp == "train", {f"pooled_{stage}": sp == stage}),
            f"vitaldb_train->mover_{stage}": ((ds == "vitaldb") & (sp == "train"), {f"mover_{stage}": (ds == "mover") & (sp == stage)})}


def compare_models(y, preds, boot, refs, base_row):
    """Rows with point metrics, bootstrap CIs and paired deltas vs each reference (same resampled patients)."""
    bm = boot_metrics(y, preds, boot)
    rows = []
    # Apply the same operation to each named field in this group.
    for m, p in preds.items():
        r = {**base_row, "model": m, **L._metrics(y, p)}
        for k in ("auroc", "auprc"):
            r[f"{k}_lo"], r[f"{k}_hi"], _ = ci(bm[m][k])
        for ref in refs:
            if ref == m or ref not in preds:
                continue
            for k in ("auroc", "auprc"):
                dd, lo, hi, _ = paired_delta(y, p, preds[ref], boot, k)
                r[f"d_{k}_vs_{slug(ref)}"], r[f"d_{k}_vs_{slug(ref)}_lo"], r[f"d_{k}_vs_{slug(ref)}_hi"] = dd, lo, hi
        rows.append(r)
    return rows


def confirm_validation(df, sets, reps=1000, log=print):
    """Once on validation: hgb G0 + IND-x vs G0 and old G2 (core) / old G4 and G0 + IND-EA (full); pooled and VitalDB -> MOVER validation."""
    spec = {"core": ({"G0": G0, "old G2": EM.RUNGS["G2"], "G0 + IND-A": G0 + sets["IND-A"], "G0 + IND-E": G0 + sets["IND-E"], "G0 + IND-EA": G0 + sets["IND-EA"]},
                     ["G0", "old G2"]),
            "full": ({"G0": G0, "old G4": EM.RUNGS["G4"], "G0 + IND-EA": G0 + sets["IND-EA"], "G0 + IND-P": G0 + sets["IND-P"], "G0 + IND-EAP": G0 + sets["IND-EAP"]},
                     ["G0", "old G4", "G0 + IND-EA"])}
    rows = []
    # Apply the same operation to each named field in this group.
    for wset, (models, refs) in spec.items():
        frame = df[df.in_full] if wset == "full" else df
        for sname, (trm, tests) in clean_settings(frame, "validation").items():
            train = frame[trm.to_numpy()]
            # Apply the same operation to each named field in this group.
            for tname, tm in tests.items():
                test = frame[tm.to_numpy()]
                y = test.y.to_numpy()
                preds = {}
                for m, cols in models.items():
                    if m.startswith("G0 + ") and len(cols) == len(G0):
                        continue
                    (p,), _ = EM.fit_predict(train, [test], cols, "hgb", SEED)
                    preds[m] = p
                boot = Boot(test.patient_key, test.dataset, reps, SEED)
                base = {"setting": sname, "test": tname, "window_set": wset, "windows": int(len(test)), "events": int(y.sum()), "patients": int(test.patient_key.nunique())}
                rows += compare_models(y, preds, boot, refs, base)
                log(f"validation {wset} {sname}: {len(preds)} models")
    out = pd.DataFrame(rows)
    out.insert(3, "features", out.model.map(lambda m: len({**spec["core"][0], **spec["full"][0]}[m])))
    return out


def ppg_rule(val, sets, t):
    """Pre-registered PPG rule on validation: keep PPG only if G0 + IND-EAP beats G0 + IND-EA (paired Delta AUROC CI > 0) pooled AND VitalDB -> MOVER, P parameters pass F5."""
    # Exclude or flag missing values before using these measurements.
    p_in = [c for c in sets["IND-EAP"] if c.startswith("p_")]
    f5 = bool(t.set_index("param").loc[p_in, "F5"].astype(bool).all()) if p_in else False
    col = f"d_auroc_vs_{slug('G0 + IND-EA')}"
    # Exclude or flag missing values before using these measurements.
    r = val[(val.window_set == "full") & (val.model == "G0 + IND-EAP")]
    los = {s: float(r.loc[r.setting == s, col + "_lo"].iloc[0]) if col + "_lo" in r and (r.setting == s).any() else np.nan for s in val.setting.unique()}
    beats = bool(los) and all(np.isfinite(v) and v > 0 for v in los.values())
    keep = bool(p_in) and beats and f5
    reason = ("IND-EAP selected no PPG parameter" if not p_in else
              f"G0 + IND-EAP vs G0 + IND-EA paired Delta AUROC CI lower bounds {', '.join(f'{k}: {v:+.4f}' for k, v in los.items())}; P parameters pass F5: {f5}")
    return {"decision": "keep PPG in the main model" if keep else "remove PPG from the main model (secondary noninvasive study only)", "keep": keep,
            "ppg_params_in_IND-EAP": p_in, "reason": reason}


def noninvasive_set(sets, t):
    u = list(dict.fromkeys(sets["IND-E"] + sets["IND-P"]))
    if len(u) >= 3:
        return u, "IND-E + IND-P (selected)"
    f = t[t.signal.isin(["E", "P"]) & t.F1.astype(bool) & t.F5.astype(bool)]
    return f.param.tolist(), "all E and P candidates passing F1 and F5 (unselected; IND-E + IND-P had < 3 parameters)"


def sha256_file(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_config(root, block):
    """Additive insertion of the key ``independence`` at the end of configs/hemodynamics.json (CRLF preserved; replaces only a previous ``independence`` block)."""
    path = Path(root) / "configs" / "hemodynamics.json"
    if not path.exists():
        return None
    raw = path.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    # Store the settings and results as JSON for later inspection.
    marker = f',{nl}  "independence":'
    i = raw.find(marker)
    body = (raw[:i] + nl + "}" if i >= 0 else raw).rstrip()
    assert body.endswith("}")
    body = body[:-1].rstrip()
    # Load the recorded settings or metadata for this processing step.
    blk = json.dumps(block, indent=2).replace("\n", nl + "  ")
    new = f'{body},{nl}  "independence": {blk}{nl}}}{nl}'
    json.loads(new)
    path.write_bytes(new.encode("utf-8"))
    return path


def select_and_confirm(df, tr, t, root, frame_path, workers=16, reps=1000, log=print, config_root=None):
    """pools -> greedy (hgb OOF) -> L1 cross-check -> validation once -> PPG rule -> freeze config."""
    pools = selection_pools(t)
    log("pools: " + ", ".join(f"{n}={len(v)}" for n, v in pools.items()))
    with Pool(frame_path, workers, 200, threads=2) as px:
        sets, path, ref = greedy_select(pools, px, log)
    save_table(path, root, "table_selection.csv")
    l1 = l1_path(tr, pools, sets)
    # Calculate the PPG signal for the following step.
    save_table(l1, root, "table_selection_l1.csv")
    selection_figure(path, ref, root)
    nonin, nonin_src = noninvasive_set(sets, t)
    val = confirm_validation(df, sets, reps=reps, log=log)
    save_table(val, root, "table_validation_confirm.csv")
    # Load the recorded settings or metadata for this processing step.
    ppg = ppg_rule(val, sets, t)
    summ = json.loads((Path(config_root or root) / OUT / "summary.json").read_text(encoding="utf-8")) if (Path(config_root or root) / OUT / "summary.json").exists() else {}
    hgb = {k: v for k, v in EM._model("hgb", SEED).get_params().items() if k in ("max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")}
    block = {"version": "v5-independence", "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_sha256": summ.get("plan_sha256"),
             "window_params": F.PARAMS_NAME, "feature_code_version": F.VERSION, "seed": SEED,
             "screen": {"training_window_step": 3, "lr_folds": 5, "hgb_folds": 3, "bootstrap": 200, "bh_q": 0.05, "f1_coverage": [0.90, 0.85], "f4_min_d_auroc": -0.002,
                        "f5_max_domain_auroc": 0.70, "f6_max_abs_rho_map": 0.5, "redundancy_abs_rho": 0.8},
             "selection": {"learner": "hgb", "folds": 3, "max_params": 12, "stop_d_auroc": 0.001, "stop_d_auprc": 0.002, "pool_cap": 20},
             "sets": sets, "window_sets": {n: w for n, (_, w) in SETS.items()}, "noninvasive": {"params": nonin, "source": nonin_src},
             "learners": {"hgb": hgb, "lr": "SimpleImputer(median) + StandardScaler + LogisticRegression(max_iter=300)"}, "fit_subsample": 3,
             "ppg_rule_validation": ppg["decision"]}
    # Store the settings and results as JSON for later inspection.
    (Path(root) / OUT / "frozen_independence.json").write_text(json.dumps(block, indent=2), encoding="utf-8")
    freeze_config(config_root, block) if config_root else None
    log("frozen: " + json.dumps(sets))
    return {**sets, "ppg": ppg, "noninvasive": nonin, "validation": val, "path": path, "l1": l1}


# + final comparison (TEST used once)
CLEAN = {"pooled_test", "pooled_test:vitaldb", "pooled_test:mover", "clean vitaldb_train->mover_test", "clean mover_train->vitaldb_test"}
KEY_MODELS = {"G0", "old G2", "old G4", "G0 + IND-EA", "G0 + IND-EAP", "noninvasive IND-E + IND-P"}


def final_settings(df):
    """Pre-registered ``eda_models._settings`` (kept for comparability; transfer settings NOT clean, deviation 4) + clean pooled-split transfer settings."""
    # Calculate the spread of the observations for the following step.
    ds, sd, sp = df.dataset, df.split_within_dataset, df.split_pooled
    return {"pooled_train": (sp == "train", {"pooled_test": sp == "test", "pooled_test:vitaldb": (sp == "test") & (ds == "vitaldb"),
                                             "pooled_test:mover": (sp == "test") & (ds == "mover")}),
            "vitaldb_within_train": ((ds == "vitaldb") & (sd == "train"), {"E3x vitaldb_train->all_mover (not clean)": ds == "mover"}),
            "mover_within_train": ((ds == "mover") & (sd == "train"), {"reverse mover_train->vitaldb_test (not clean)": (ds == "vitaldb") & (sd == "test")}),
            "vitaldb_pooled_train": ((ds == "vitaldb") & (sp == "train"), {"clean vitaldb_train->mover_test": (ds == "mover") & (sp == "test")}),
            "mover_pooled_train": ((ds == "mover") & (sp == "train"), {"clean mover_train->vitaldb_test": (ds == "vitaldb") & (sp == "test")})}


def final_models(sets, nonin, t):
    p_f1 = t[(t.signal == "P") & t.F1.astype(bool)].param
    p_raw, p_sn = [c for c in p_f1 if not c.endswith("_sn")], [c for c in p_f1 if c.endswith("_sn")]
    core = {"G0": G0, "old G1": EM.RUNGS["G1"], "old G2": EM.RUNGS["G2"], "G0 + old cross arm": G0 + OLD_ARM_CORE, "G0 + IND-A": G0 + sets["IND-A"],
            "G0 + IND-E": G0 + sets["IND-E"], "G0 + IND-EA": G0 + sets["IND-EA"], "IND-EA alone": list(sets["IND-EA"])}
    full = {"G0": G0, "old G3": EM.RUNGS["G3"], "old G4": EM.RUNGS["G4"], "G0 + old cross arm": G0 + OLD_ARM_FULL, "G0 + IND-EA": G0 + sets["IND-EA"],
            "G0 + IND-P": G0 + sets["IND-P"], "G0 + IND-EAP": G0 + sets["IND-EAP"], "IND-EA alone": list(sets["IND-EA"]),
            "G0 + IND-EA + PPG raw": G0 + sets["IND-EA"] + p_raw, "G0 + IND-EA + PPG self-normalized": G0 + sets["IND-EA"] + p_sn,
            "noninvasive IND-E + IND-P": list(nonin)}
    keep = lambda d: {k: list(dict.fromkeys(v)) for k, v in d.items() if v and (k == "G0" or not (k.startswith("G0 + ") and len(set(v)) == len(G0)))}
    return {"core": keep(core), "full": keep(full)}, {"core": ["G0", "old G2"], "full": ["G0", "old G4", "G0 + IND-EA"]}


def sens_at_spec(y, p, spec=0.90):
    y, p = np.asarray(y), np.asarray(p)
    thr = np.quantile(p[y == 0], spec)
    return float((p[y == 1] > thr).mean()), float(thr)


def pick_examples(df, seed=SEED, before=600, min_windows=18):
    """One VitalDB and one MOVER training event drawn at random among events with >= ``min_windows`` windows in the ``before`` s preceding onset."""
    # Seed the generated values so this check can be repeated.
    tr = df[df.split_pooled == "train"]
    rng = np.random.default_rng(seed)
    out = {}
    for ds in ("vitaldb", "mover"):
        # Summarize the records separately within each requested group.
        d = tr[tr.dataset == ds]
        ev = d[(d.y == 1) & (d.time_to_event_sec <= 60)].assign(onset=lambda e: (e.t_end_sec + e.time_to_event_sec).round(-1))
        cands = ev[["case_id", "onset"]].drop_duplicates().sort_values(["case_id", "onset"]).to_numpy()
        t_end = {cid: g.t_end_sec.to_numpy() for cid, g in d.groupby("case_id")}
        ok = [(c, o) for c, o in cands if ((t_end[c] >= o - before) & (t_end[c] <= o)).sum() >= min_windows]
        if ok:
            c, o = ok[int(rng.integers(len(ok)))]
            w = d[(d.case_id == c) & (d.t_end_sec >= o - before) & (d.t_end_sec <= o)]
            out[ds] = {"case_id": c, "onset_sec": float(o), "window_ids": w.window_id.tolist()}
    return out


_FK = {}


def _final_init(root, params_name, threads):
    os.environ["OMP_NUM_THREADS"] = str(threads)
    import sklearn.ensemble  # noqa: F401
    from threadpoolctl import threadpool_limits
    _FK["threads"] = threadpool_limits(limits=threads)
    _FK["df"] = load_frame(root, params_name)


def _boot_subset_auroc(y, p, boot, mask):
    rk = Ranked(y[mask], p[mask])
    return np.array([rk.auroc(boot.window_weights(r)[mask]) for r in range(boot.reps)])


def _final_job(train_key, wset, learner, models, refs, reps, extras):
    df = _FK["df"]
    frame = df[df.in_full] if wset == "full" else df
    trm, tests = final_settings(frame)[train_key]
    train = frame[trm.to_numpy()]
    fit_rows = train.iloc[::3] if len(train) > 60000 else train
    rows, lead, gray = [], [], []
    if fit_rows.y.nunique() < 2:  # only possible on SMOKE subsets
        return rows, lead, gray, {}
    fitted = {n: EM._model(learner, SEED).fit(fit_rows[c].to_numpy(np.float64), fit_rows.y.to_numpy()) for n, c in models.items()}
    for tname, tm in tests.items():
        test = frame[tm.to_numpy()]
        y = test.y.to_numpy()
        if len(np.unique(y)) < 2:
            continue
        # Calculate predicted risk without refitting on evaluation windows.
        preds = {n: m.predict_proba(test[models[n]].to_numpy(np.float64))[:, 1] for n, m in fitted.items()}
        boot = Boot(test.patient_key, test.dataset, reps, SEED)
        bm = boot_metrics(y, preds, boot)
        base = {"train": train_key, "setting": tname, "clean": tname in CLEAN, "window_set": wset, "learner": learner, "windows": int(len(test)), "events": int(y.sum()),
                "patients": int(test.patient_key.nunique()), "train_windows_used": int(len(fit_rows))}
        point = {}
        for n, p in preds.items():
            # Calculate prediction metrics from these labels and model scores.
            met = L._metrics(y, p)
            point[n] = met
            r = {**base, "model": n, "features": len(models[n]), **met}
            r["sens_at_90spec"], _ = sens_at_spec(y, p)
            for k in ("auroc", "auprc"):
                r[f"{k}_lo"], r[f"{k}_hi"], _ = ci(bm[n][k])
            for ref in refs:
                if ref == n or ref not in preds:
                    continue
                for k in ("auroc", "auprc"):
                    lo, hi, _ = ci(bm[n][k] - bm[ref][k])
                    r[f"d_{k}_vs_{slug(ref)}"], r[f"d_{k}_vs_{slug(ref)}_lo"], r[f"d_{k}_vs_{slug(ref)}_hi"] = met[k] - L._metrics(y, preds[ref])[k], lo, hi
            rows.append(r)
        tte = test.time_to_event_sec.to_numpy(np.float64)
        gz = test.gray_zone.to_numpy(bool)
        # Process the selected records in order and retain their individual results.
        for n in [m for m in preds if m in KEY_MODELS]:
            p = preds[n]
            for lo_s, hi_s in LEAD_BINS:
                mk = (y == 0) | ((y == 1) & (tte > lo_s) & (tte <= hi_s))
                if (y[mk] == 1).sum() < 5:
                    continue
                b = _boot_subset_auroc(y, p, boot, mk)
                lo, hi, _ = ci(b)
                lead.append({**base, "model": n, "lead_bin_sec": f"{lo_s}-{hi_s}", "event_windows": int((y[mk] == 1).sum()), "auroc": float(L.auroc(y[mk], p[mk])), "auroc_lo": lo, "auroc_hi": hi})
            for sub, mk in (("gray zone 65-75 mmHg", gz), ("outside gray zone", ~gz)):
                if (y[mk] == 1).sum() < 5:
                    continue
                b = _boot_subset_auroc(y, p, boot, mk)
                lo, hi, _ = ci(b)
                gray.append({**base, "model": n, "subset": sub, "subset_windows": int(mk.sum()), "subset_events": int((y[mk] == 1).sum()),
                             "auroc": float(L.auroc(y[mk], p[mk])), "auroc_lo": lo, "auroc_hi": hi})
    extra_out = {}
    if extras:
        for tag, ids in extras.items():
            # Calculate predicted risk without refitting on evaluation windows.
            sub = df[df.window_id.isin(ids)]
            extra_out[tag] = {"window_id": sub.window_id.tolist(), "y": sub.y.tolist(),
                              **{n: fitted[n].predict_proba(sub[models[n]].to_numpy(np.float64))[:, 1].tolist() for n in ("G0", "G0 + IND-EA") if n in fitted}}
    return rows, lead, gray, extra_out


def _domain_job(feature_sets, reps):
    """Multivariate domain AUROC (hgb predicts MOVER vs VitalDB; pooled train -> pooled test, full windows) per feature set."""
    df = _FK["df"]
    f = df[df.in_full]
    train, test = f[f.split_pooled == "train"], f[f.split_pooled == "test"]
    fit_rows = train.iloc[::3]
    y = test.is_mover.to_numpy()
    if len(np.unique(y)) < 2 or train.is_mover.nunique() < 2:  # only possible on SMOKE subsets
        return []
    boot = Boot(test.patient_key, test.dataset, reps, SEED)
    rows = []
    for name, (cols, kind) in feature_sets.items():
        if not cols:
            continue
        # Calculate predicted risk without refitting on evaluation windows.
        m = EM._model("hgb", SEED).fit(fit_rows[cols].to_numpy(np.float64), fit_rows.is_mover.to_numpy())
        p = m.predict_proba(test[cols].to_numpy(np.float64))[:, 1]
        rk = Ranked(y, p)
        lo, hi, _ = ci([rk.auroc(boot.window_weights(r)) for r in range(reps)])
        rows.append({"feature_set": name, "kind": kind, "features": len(cols), "domain_auroc": float(L.auroc(y, p)), "domain_auroc_lo": lo, "domain_auroc_hi": hi,
                     "test_windows": int(len(test)), "mover_share": float(y.mean())})
    return rows


def ppg_cost(df):
    """What requiring PPG costs: core vs full windows, events, patients and cases per dataset (all splits)."""
    rows = []
    for ds, d in df.groupby("dataset"):
        f = d[d.in_full]
        for what, a, b in (("windows", len(d), len(f)), ("events", int(d.y.sum()), int(f.y.sum())), ("patients", d.patient_key.nunique(), f.patient_key.nunique()),
                           ("cases", d.case_id.nunique(), f.case_id.nunique())):
            rows.append({"section": "b cost of requiring PPG", "dataset": ds, "quantity": what, "core": int(a), "full": int(b), "lost": int(a - b), "lost_pct": 100.0 * (a - b) / a if a else np.nan})
    return pd.DataFrame(rows)


def _domain_sets(sets, nonin):
    return {"G0 (current MAP)": (G0, "baseline"), "old G1 additions (ART morphology, ECG-segmented)": (EM.G1_ADD, "old"),
            "old G2 additions (ECG->ART timing + HRV)": (EM.G2_ADD, "old"), "old G3 additions (PPG + PPG delays)": (EM.G3_ADD, "old"),
            "old G4 additions (SQI + counts)": (EM.G4_ADD, "old"), "old cross-signal arm": (OLD_ARM_FULL, "old"), "old G4 (all old features incl. G0)": (EM.RUNGS["G4"], "old"),
            "IND-A": (sets["IND-A"], "new"), "IND-E": (sets["IND-E"], "new"), "IND-P": (sets["IND-P"], "new"), "IND-EA": (sets["IND-EA"], "new"),
            "IND-EAP": (sets["IND-EAP"], "new"), "G0 + IND-EA": (G0 + sets["IND-EA"], "new"), "G0 + IND-EAP": (G0 + sets["IND-EAP"], "new"),
            "noninvasive IND-E + IND-P": (list(nonin), "new")}


def load_frozen(root, config_root=None):
    frozen = json.loads((Path(root) / OUT / "frozen_independence.json").read_text(encoding="utf-8"))
    cfgp = Path(config_root or root) / "configs" / "hemodynamics.json"
    if cfgp.exists():
        # Load the recorded settings or metadata for this processing step.
        cfg = json.loads(cfgp.read_text(encoding="utf-8")).get("independence")
        assert cfg and cfg["sets"] == frozen["sets"], "configs/hemodynamics.json 'independence' must match the frozen sets"
        frozen = cfg
    return frozen


def final_comparison(df, root, data_root, params_name=F.PARAMS_NAME, reps=1000, workers=8, threads=4, log=print, config_root=None):
    """+ single test pass. Every model in a comparison is scored on the same windows; paired deltas use the same patient resamples."""
    # Evaluate the frozen feature sets rather than selecting a new model on test patients.
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    root = Path(root)
    # Read the saved table and retain its named columns.
    frozen = load_frozen(root, config_root)
    sets, nonin = frozen["sets"], frozen["noninvasive"]["params"]
    t = pd.read_csv(root / OUT / "table_feasibility.csv")
    models, refs = final_models(sets, nonin, t)
    ex = pick_examples(df)
    extras = {**{ds: v["window_ids"] for ds, v in ex.items()}, "validation": df[df.split_pooled == "validation"].window_id.tolist()}
    jobs = [(k, w, l) for k in final_settings(df) for w in ("core", "full") for l in ("lr", "hgb")]
    rows, lead, gray, extra, dom = [], [], [], {}, []
    t0 = time.perf_counter()
    # Limit worker lifetime to this block and collect their processing results.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_final_init, initargs=(str(data_root), params_name, threads)) as px:
        futs = {px.submit(_domain_job, _domain_sets(sets, nonin), reps): "domain"}
        for k, w, l in jobs:
            futs[px.submit(_final_job, k, w, l, models[w], refs[w], reps, extras if (k, w, l) == ("pooled_train", "core", "hgb") else None)] = f"{k}/{w}/{l}"
        # Collect completed jobs and retain the remaining work for the next pass.
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            if futs[fu] == "domain":
                dom = r
            else:
                rows += r[0]
                lead += r[1]
                gray += r[2]
                extra.update(r[3])
            log(f"  final {i}/{len(futs)} {futs[fu]} [{time.perf_counter() - t0:.0f}s]")
    # Collect the per-record values into a table for comparison.
    res, lead, gray, dom = pd.DataFrame(rows), pd.DataFrame(lead), pd.DataFrame(gray), pd.DataFrame(dom)
    save_table(res, root, "table_old_vs_new.csv")
    save_table(lead, root, "table_lead_time.csv")
    save_table(gray, root, "table_gray_zone.csv")
    save_table(dom, root, "table_domain_old_vs_new.csv")
    thr = {}
    if "validation" in extra:
        v = extra["validation"]
        y = np.asarray(v["y"])
        thr = {n: sens_at_spec(y, np.asarray(v[n]))[1] for n in ("G0", "G0 + IND-EA") if n in v}
    cost = ppg_cost(df)
    ppg_tab = ppg_table(res, cost, root, frozen)
    save_table(ppg_tab, root, "table_ppg_decision.csv")
    figs_old_vs_new(res, lead, gray, dom, root)
    fig_examples(df, ex, extra, thr, sets, t, root)
    fig_ppg(res, cost, root)
    crit = criteria(res, dom, root, frozen, t)
    save_table(crit, root, "table_criteria.csv")
    return {"results": res, "criteria": crit, "examples": ex, "thresholds": thr}


def _pick(res, setting, wset, model, col, learner="hgb"):
    r = res[(res.setting == setting) & (res.window_set == wset) & (res.model == model) & (res.learner == learner)]
    return float(r[col].iloc[0]) if len(r) and col in r.columns and pd.notna(r[col].iloc[0]) else np.nan


def ppg_table(res, cost, root, frozen):
    rows = []
    # Process the selected records in order and retain their individual results.
    for setting in ("pooled_test", "E3x vitaldb_train->all_mover (not clean)", "clean vitaldb_train->mover_test"):
        for m in ("G0 + IND-EA", "G0 + IND-EAP", "G0 + IND-EA + PPG raw", "G0 + IND-EA + PPG self-normalized", "G0 + IND-P", "noninvasive IND-E + IND-P", "G0"):
            if np.isnan(_pick(res, setting, "full", m, "auroc")):
                continue
            rows.append({"section": "a/d models on full windows (hgb)", "dataset": setting, "quantity": m, "auroc": _pick(res, setting, "full", m, "auroc"),
                         "auroc_lo": _pick(res, setting, "full", m, "auroc_lo"), "auroc_hi": _pick(res, setting, "full", m, "auroc_hi"),
                         "auprc": _pick(res, setting, "full", m, "auprc"),
                         "d_auroc_vs_G0_plus_INDEA": _pick(res, setting, "full", m, "d_auroc_vs_G0_plus_INDEA"),
                         "d_auroc_vs_G0_plus_INDEA_lo": _pick(res, setting, "full", m, "d_auroc_vs_G0_plus_INDEA_lo"),
                         "d_auroc_vs_G0_plus_INDEA_hi": _pick(res, setting, "full", m, "d_auroc_vs_G0_plus_INDEA_hi"),
                         "d_auroc_vs_G0": _pick(res, setting, "full", m, "d_auroc_vs_G0"), "d_auroc_vs_G0_lo": _pick(res, setting, "full", m, "d_auroc_vs_G0_lo"),
                         "d_auroc_vs_G0_hi": _pick(res, setting, "full", m, "d_auroc_vs_G0_hi")})
    dom = pd.read_csv(Path(root) / OUT / "table_domain_auroc.csv")
    p = dom[(dom.kind == "new") & (dom.signal == "P")]
    for norm, sub in (("raw", p[~p.param.str.endswith("_sn")]), ("self-normalized", p[p.param.str.endswith("_sn")])):
        rows.append({"section": "c PPG domain AUROC (univariate, training)", "dataset": "VitalDB vs MOVER", "quantity": f"P candidates, {norm}",
                     "domain_auroc_median": float(sub.domain_auroc.median()), "domain_auroc_min": float(sub.domain_auroc.min()), "n_params": int(len(sub))})
    rows.append({"section": "rule (validation, pre-registered)", "dataset": "validation", "quantity": frozen.get("ppg_rule_validation", "")})
    return pd.concat([pd.DataFrame(rows), cost], ignore_index=True)


def criteria(res, dom, root, frozen, t):
    """S1-S3 and H1-H5 exactly as pre-registered (plus the clean-transfer variants of deviation 4)."""
    root = Path(root)
    sets = frozen["sets"]
    out = []

    def add(name, passed, detail):
        out.append({"criterion": name, "pass": passed if isinstance(passed, str) else bool(passed), "detail": detail})

    # Calculate the working array for the following step.
    ea_empty = not sets["IND-EA"]
    pt, e3, cl = "pooled_test", "E3x vitaldb_train->all_mover (not clean)", "clean vitaldb_train->mover_test"
    d = {k: _pick(res, pt, "core", "G0 + IND-EA", k) for k in ("d_auroc_vs_G0", "d_auroc_vs_G0_lo", "d_auroc_vs_G0_hi", "d_auprc_vs_G0")}
    e3d, cld = _pick(res, e3, "core", "G0 + IND-EA", "d_auroc_vs_G0"), _pick(res, cl, "core", "G0 + IND-EA", "d_auroc_vs_G0")
    s1 = (not ea_empty) and d["d_auroc_vs_G0_lo"] > 0 and d["d_auprc_vs_G0"] > 0 and e3d >= 0
    add("S1 beats current MAP (pre-registered)", s1, "IND-EA empty: G0 + IND-EA = G0" if ea_empty else
        f"pooled test dAUROC {d['d_auroc_vs_G0']:+.4f} [{d['d_auroc_vs_G0_lo']:+.4f}, {d['d_auroc_vs_G0_hi']:+.4f}], dAUPRC {d['d_auprc_vs_G0']:+.4f}; VitalDB->all MOVER dAUROC {e3d:+.4f}")
    add("S1 clean-transfer variant", (not ea_empty) and d["d_auroc_vs_G0_lo"] > 0 and d["d_auprc_vs_G0"] > 0 and cld >= 0, f"clean VitalDB->MOVER test dAUROC {cld:+.4f}")
    g4 = {k: _pick(res, pt, "full", "G0 + IND-EA", k) for k in ("d_auroc_vs_old_G4", "d_auroc_vs_old_G4_lo")}
    dm = dom.set_index("feature_set").domain_auroc if len(dom) else pd.Series(dtype=float)
    d_new, d_old = dm.get("G0 + IND-EA", np.nan), dm.get("old G4 (all old features incl. G0)", np.nan)
    a_new, a_old = _pick(res, e3, "full", "G0 + IND-EA", "auroc"), _pick(res, e3, "full", "old G4", "auroc")
    c_new, c_old = _pick(res, cl, "full", "G0 + IND-EA", "auroc"), _pick(res, cl, "full", "old G4", "auroc")
    s2 = (not ea_empty) and g4["d_auroc_vs_old_G4"] >= -0.003 and d_new < d_old and a_new >= a_old
    # Read the saved table and retain its named columns.
    add("S2 better than old (pre-registered, point non-inferiority)", s2, f"pooled test dAUROC vs G4 {g4['d_auroc_vs_old_G4']:+.4f} (CI lo {g4['d_auroc_vs_old_G4_lo']:+.4f}); "
        f"domain AUROC {d_new:.3f} vs G4 {d_old:.3f}; VitalDB->all MOVER AUROC {a_new:.3f} vs G4 {a_old:.3f}")
    add("S2 strict (CI lower bound >= -0.003) and clean transfer", (not ea_empty) and g4["d_auroc_vs_old_G4_lo"] >= -0.003 and d_new < d_old and c_new >= c_old,
        f"clean VitalDB->MOVER test AUROC {c_new:.3f} vs G4 {c_old:.3f}")
    # Read the saved table and retain its named columns.
    s3 = {k: _pick(res, pt, "core", "IND-EA alone", k) for k in ("auroc", "auprc", "d_auroc_vs_G0")}
    add("S3 IND-EA without G0 vs G0 (descriptive)", "descriptive", f"pooled test AUROC {s3['auroc']:.3f}, AUPRC {s3['auprc']:.3f}, dAUROC vs G0 {s3['d_auroc_vs_G0']:+.4f}")
    val = pd.read_csv(root / OUT / "table_validation_confirm.csv")
    va = val[(val.window_set == "core") & (val.model == "G0 + IND-A")]
    h1_pool = va.loc[va.setting.str.startswith("pooled"), "d_auroc_vs_G0_lo"]
    h1_mov = va.loc[va.setting.str.startswith("vitaldb"), "d_auroc_vs_G0"]
    n_a = int((t.signal.eq("A") & t.feasible.astype(bool)).sum())
    h1 = n_a >= 1 and len(h1_pool) and float(h1_pool.iloc[0]) > 0 and len(h1_mov) and float(h1_mov.iloc[0]) >= 0
    add("H1 ART single-signal parameters predict beyond G0 in both datasets", h1, f"feasible A {n_a}; validation pooled dAUROC CI lo "
        f"{float(h1_pool.iloc[0]) if len(h1_pool) else np.nan:+.4f}; VitalDB->MOVER validation dAUROC {float(h1_mov.iloc[0]) if len(h1_mov) else np.nan:+.4f}")
    # Read the saved table and retain its named columns.
    e = t[t.signal == "E"]
    e_ok = int((e.F2.astype(bool) & e.F3.astype(bool) & e.F4.astype(bool)).sum())
    esn = e[e.param.str.endswith("_sn")]
    med_sn, med_raw = float(esn.domain_auroc.median()), float(esn.domain_auroc_raw.median())
    add("H2 ECG-only HRV/temporal add beyond G0 and self-normalization lowers hospital specificity", e_ok >= 1 and med_sn <= med_raw - 0.05,
        f"E passing F2+F3+F4: {e_ok}; median domain AUROC e_*_sn {med_sn:.3f} vs raw {med_raw:.3f}")
    tc = pd.read_csv(root / OUT / "table_temporal_correlation.csv")
    res_h3 = {}
    for ind in ("a_map_ac1_l300", "a_map_rsd_l300"):
        for ds in ("vitaldb", "mover"):
            u = tc[(tc.indicator == ind) & (tc.dataset == ds)].set_index("lead_bin_sec")
            near, far = u.loc["0-60"] if "0-60" in u.index else None, u.loc["180-240"] if "180-240" in u.index else None
            res_h3[(ind, ds)] = bool(near is not None and far is not None and near.ci_lo > 0 and near.diff_vs_matched > far.diff_vs_matched)
    both = any(res_h3[(i, "vitaldb")] and res_h3[(i, "mover")] for i in ("a_map_ac1_l300", "a_map_rsd_l300"))
    vit = any(res_h3[(i, "vitaldb")] for i in ("a_map_ac1_l300", "a_map_rsd_l300"))
    add("H3 critical slowing down before IOH", "supported" if both else ("partially (VitalDB only)" if vit else "not supported"),
        "; ".join(f"{i} {ds}: {'yes' if v else 'no'}" for (i, ds), v in res_h3.items()))
    p = t[t.signal == "P"]
    med_p_raw, med_p_sn = float(p[~p.param.str.endswith("_sn")].domain_auroc.median()), float(p[p.param.str.endswith("_sn")].domain_auroc.median())
    drop = "remove" in frozen.get("ppg_rule_validation", "")
    # Collect the per-record values into a table for comparison.
    add("H4 PPG adds nothing beyond E+A and stays hospital-specific", drop and med_p_raw > 0.70 and med_p_sn > 0.70,
        f"PPG rule: {frozen.get('ppg_rule_validation')}; median domain AUROC P raw {med_p_raw:.3f}, P self-normalized {med_p_sn:.3f}")
    add("H5 G0 + IND-EA beats G0, non-inferior to G4, transfers", s1 and s2, "S1 and S2")
    return pd.DataFrame(out)


# /figures
FIG8_SETTINGS = [("pooled_test", "Pooled Train to Pooled Test"), ("E3x vitaldb_train->all_mover (not clean)", "VitalDB Train to All MOVER (Pre-Registered, Not Clean)"),
                 ("clean vitaldb_train->mover_test", "VitalDB Train to MOVER Test (Clean)"),
                 ("reverse mover_train->vitaldb_test (not clean)", "MOVER Train to VitalDB Test (Pre-Registered, Not Clean)")]
FIG8_MODELS = [("core", m) for m in ("G0", "old G1", "old G2", "G0 + old cross arm", "G0 + IND-A", "G0 + IND-E", "G0 + IND-EA", "IND-EA alone")] + \
              [("full", m) for m in ("G0", "old G4", "G0 + old cross arm", "G0 + IND-EA", "G0 + IND-EAP", "noninvasive IND-E + IND-P")]
KEY_COLOR = {"G0": "#000000", "old G4": SIG_COLOR["old"], "old G2": SIG_COLOR["old"], "G0 + IND-EA": SET_COLOR["IND-EA"], "G0 + IND-EAP": SET_COLOR["IND-EAP"],
             "noninvasive IND-E + IND-P": "#16a085"}


def _model_color(model):
    if model.startswith("old") or "cross" in model:
        return SIG_COLOR["old"]
    if model.startswith("noninvasive"):
        return "#16a085"
    for n in ("IND-EAP", "IND-EA", "IND-A", "IND-E", "IND-P"):
        if n in model:
            return SET_COLOR[n]
    return "#000000"


def figs_old_vs_new(res, lead, gray, dom, root):
    # Create the plotting panels for this comparison.
    plt = _plt()
    h = res[res.learner == "hgb"]
    fig, axes = plt.subplots(len(FIG8_SETTINGS), 2, figsize=(15, 4.6 * len(FIG8_SETTINGS)))
    # Process the selected records in order and retain their individual results.
    for i, (s, title) in enumerate(FIG8_SETTINGS):
        for j, m in enumerate(("auroc", "auprc")):
            ax, labels = axes[i, j], []
            for wset, model in FIG8_MODELS:
                r = h[(h.setting == s) & (h.window_set == wset) & (h.model == model)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(r[m], len(labels), xerr=[[max(r[m] - r[f"{m}_lo"], 0)], [max(r[f"{m}_hi"] - r[m], 0)]], fmt="s" if wset == "full" else "o",
                            color=_model_color(model), ms=5, capsize=2)
                labels.append(f"{model} ({wset})")
            g0 = h[(h.setting == s) & (h.window_set == "core") & (h.model == "G0")]
            if len(g0):
                ax.axvline(g0[m].iloc[0], color="k", ls=":", lw=0.8)
            # Label the panel with the measurement names and their units.
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{title}: {m.upper()}")
            ax.set_xlabel(f"{m.upper()} (Unitless, 95% Patient-Bootstrap CI)")
            ax.set_ylabel("Model (Window Set)")
    fig.suptitle("Independent Parameters vs Current MAP (G0) and Old Parameters, Test Split, Gradient Boosting (Circles = Core Windows, Squares = Full Windows)")
    save_figure(fig, root, "fig_ind_08_old_vs_new.png")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # Process the selected records in order and retain their individual results.
    for i, (s, title) in enumerate((("pooled_test", "Pooled Test"), ("clean vitaldb_train->mover_test", "VitalDB Train to MOVER Test (Clean)"))):
        ax = axes[i, 0]
        models = ("G0", "old G4", "G0 + IND-EA", "G0 + IND-EAP")
        for k, model in enumerate(models):
            u = lead[(lead.setting == s) & (lead.window_set == "full") & (lead.learner == "hgb") & (lead.model == model)] if len(lead) else pd.DataFrame()
            if u.empty:
                continue
            mid = np.array([np.mean([float(x) for x in b.split("-")]) for b in u.lead_bin_sec]) + (k - 1.5) * 4
            ax.errorbar(mid, u.auroc, yerr=[np.maximum(u.auroc - u.auroc_lo, 0), np.maximum(u.auroc_hi - u.auroc, 0)], fmt="o-", color=KEY_COLOR[model], capsize=2, label=model)
        # Label the panel with the measurement names and their units.
        ax.invert_xaxis()
        ax.legend(fontsize=8)
        ax.set_title(f"{title}: AUROC by Lead Time (Full Windows)")
        ax.set_xlabel("Time to IOH Onset, Bin Midpoint (s)")
        ax.set_ylabel("AUROC (Unitless)")
        ax = axes[i, 1]
        for k, model in enumerate(models):
            u = gray[(gray.setting == s) & (gray.window_set == "full") & (gray.learner == "hgb") & (gray.model == model)] if len(gray) else pd.DataFrame()
            if u.empty:
                continue
            xs = np.arange(len(u)) + (k - 1.5) * 0.18
            ax.bar(xs, u.auroc, width=0.17, color=KEY_COLOR[model], label=model, alpha=0.85)
            ax.errorbar(xs, u.auroc, yerr=[np.maximum(u.auroc - u.auroc_lo, 0), np.maximum(u.auroc_hi - u.auroc, 0)], fmt="none", ecolor="k", capsize=2)
            ax.set_xticks(np.arange(len(u)))
            ax.set_xticklabels(u.subset.tolist())
        ax.set_ylim(0.5, 1.0)
        ax.legend(fontsize=8)
        ax.set_title(f"{title}: Gray Zone (Current MAP 65-75 mmHg) vs Rest")
        ax.set_xlabel("Window Subset (Current MAP Range)")
        ax.set_ylabel("AUROC (Unitless)")
    fig.suptitle("Where the Independent Parameters Help: Lead Time and the Gray Zone (Gradient Boosting, 95% Patient-Bootstrap CI)")
    save_figure(fig, root, "fig_ind_09_lead_time_and_gray_zone.png")

    fig, ax = plt.subplots(figsize=(11, 7))
    if len(dom):
        # Plot the observed results, including uncertainty where it is available.
        d = dom.sort_values("domain_auroc").reset_index(drop=True)
        colors = [SIG_COLOR["old"] if k == "old" else ("#000000" if k == "baseline" else SET_COLOR["IND-EA"]) for k in d.kind]
        ax.barh(np.arange(len(d)), d.domain_auroc - 0.5, left=0.5, color=colors)
        ax.errorbar(d.domain_auroc, np.arange(len(d)), xerr=[np.maximum(d.domain_auroc - d.domain_auroc_lo, 0), np.maximum(d.domain_auroc_hi - d.domain_auroc, 0)],
                    fmt="none", ecolor="k", capsize=2)
        ax.set_yticks(np.arange(len(d)))
        ax.set_yticklabels([f"{n} ({c} features)" for n, c in zip(d.feature_set, d.features)], fontsize=8)
    # Save the labeled figure for the analysis report.
    ax.axvline(0.5, color="k", lw=0.6)
    ax.set_xlim(0.45, 1.01)
    ax.set_title("Multivariate Hospital Identifiability of Each Feature Set (Old = Grey, New = Purple, G0 = Black)")
    ax.set_xlabel("Domain AUROC, Classifier Predicting MOVER vs VitalDB on Pooled Test (Unitless)")
    ax.set_ylabel("Feature Set (Number of Features)")
    save_figure(fig, root, "fig_ind_10_domain_shift_old_vs_new.png", panel_letters=False)


def fig_examples(df, ex, extra, thr, sets, t, root):
    plt = _plt()
    cat = catalogue_frame()
    params = list(sets["IND-EA"])[:3]
    if len(params) < 3:
        more = [p for p in t.sort_values("lr_d_auprc", ascending=False).param if p not in params and cat.loc[p, "signal"] in ("A", "E")]
        params += more[:3 - len(params)]
    nrow = 2 + len(params)
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 2.7 * nrow))
    # Process the selected records in order and retain their individual results.
    for j, ds in enumerate(("vitaldb", "mover")):
        name = "VitalDB" if ds == "vitaldb" else "MOVER"
        col = axes[:, j]
        if ds not in ex:
            for ax in col:
                # Label the panel with the measurement names and their units.
                ax.set_title(f"{name}: No Eligible Training Event")
                ax.set_xlabel("Time Relative to IOH Onset (s)")
                ax.set_ylabel("Not Available (None)")
            continue
        # Plot the observed results, including uncertainty where it is available.
        e = ex[ds]
        w = df[df.window_id.isin(e["window_ids"])].sort_values("t_end_sec")
        x = w.t_end_sec.to_numpy() - e["onset_sec"]
        col[0].plot(x, w.map_current, "k.-")
        col[0].axhline(65, color="#c0392b", ls="--", lw=0.9, label="IOH threshold 65 mmHg")
        col[0].legend(fontsize=7)
        col[0].set_title(f"{name} Training Case {e['case_id']}: Current MAP Before Onset")
        col[0].set_ylabel("Current MAP (mmHg)")
        for i, p in enumerate(params, 1):
            # Label the panel with the measurement names and their units.
            col[i].plot(x, w[p], ".-", color=SIG_COLOR[cat.loc[p, "signal"]])
            col[i].set_title(f"{p}: {cat.loc[p, 'definition'][:70]}")
            col[i].set_ylabel(f"{p} ({cat.loc[p, 'unit']})")
        ax = col[-1]
        pr = extra.get(ds)
        if pr:
            order = pd.Series(range(len(pr["window_id"])), index=pr["window_id"]).reindex(w.window_id).to_numpy()
            for n, c in (("G0", "#000000"), ("G0 + IND-EA", SET_COLOR["IND-EA"])):
                if n in pr:
                    ax.plot(x, np.asarray(pr[n])[order], ".-", color=c, label=f"{n} risk (in-sample)")
                    if n in thr:
                        ax.axhline(thr[n], color=c, ls="--", lw=0.8, label=f"{n} alarm threshold (90% specificity, validation)")
            ax.legend(fontsize=7)
        ax.set_title("Predicted IOH Risk With Alarm Threshold (Pooled Model, Illustration Only)")
        ax.set_ylabel("Predicted Risk (Probability)")
        for ax in col:
            ax.set_xlabel("Time Relative to IOH Onset (s)")
            ax.axvline(-300, color="grey", ls=":", lw=0.7)
    fig.suptitle("Example Training Events (Random Draw, Seed 5052): MAP and the Leading Independent Parameters Over the 10 Minutes Before Onset")
    save_figure(fig, root, "fig_ind_11_example_cases.png")


def fig_ppg(res, cost, root):
    # Create the plotting panels for this comparison.
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    h = res[(res.learner == "hgb") & (res.window_set == "full")]
    ax = axes[0, 0]
    models = ["G0 + IND-EAP", "G0 + IND-EA + PPG raw", "G0 + IND-EA + PPG self-normalized"]
    # Process the selected records in order and retain their individual results.
    for k, (s, lab) in enumerate((("pooled_test", "Pooled test"), ("clean vitaldb_train->mover_test", "VitalDB to MOVER test (clean)"),
                                  ("E3x vitaldb_train->all_mover (not clean)", "VitalDB to all MOVER (not clean)"))):
        for i, m in enumerate(models):
            r = h[(h.setting == s) & (h.model == m)]
            if r.empty or "d_auroc_vs_G0_plus_INDEA" not in r.columns:
                continue
            r = r.iloc[0]
            ax.errorbar(r.d_auroc_vs_G0_plus_INDEA, i + (k - 1) * 0.22, xerr=[[max(r.d_auroc_vs_G0_plus_INDEA - r.d_auroc_vs_G0_plus_INDEA_lo, 0)],
                        [max(r.d_auroc_vs_G0_plus_INDEA_hi - r.d_auroc_vs_G0_plus_INDEA, 0)]], fmt="os^"[k], color=["#34495e", "#e67e22", "#95a5a6"][k], capsize=2,
                        label=lab if i == 0 else None)
    # Label the panel with the measurement names and their units.
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=8)
    ax.legend(fontsize=7)
    # Label the panel with the measurement names and their units.
    ax.set_title("A Adding PPG to G0 + IND-EA (Test, Full Windows)")
    ax.set_xlabel("Delta AUROC vs G0 + IND-EA (Unitless, 95% CI)")
    ax.set_ylabel("Model")
    ax = axes[0, 1]
    c = cost[cost.section.str.startswith("b")]
    qs = ["windows", "events", "patients", "cases"]
    for k, ds in enumerate(("vitaldb", "mover")):
        u = c[c.dataset == ds].set_index("quantity").reindex(qs)
        ax.bar(np.arange(len(qs)) + (k - 0.5) * 0.38, u.lost_pct, width=0.36, color=["#34495e", "#e67e22"][k], label="VitalDB" if ds == "vitaldb" else "MOVER")
    # Label the panel with the measurement names and their units.
    ax.set_xticks(np.arange(len(qs)))
    ax.set_xticklabels([q.title() for q in qs])
    ax.legend(fontsize=8)
    ax.set_title("B Cost of Requiring PPG (Core to Full Windows, All Splits)")
    # Read the saved table and retain its named columns.
    ax.set_xlabel("Quantity Lost")
    ax.set_ylabel("Lost When PPG Is Required (%)")
    ax = axes[1, 0]
    # Read the saved table and retain its named columns.
    dom = pd.read_csv(Path(root) / OUT / "table_domain_auroc.csv")
    p = dom[(dom.kind == "new") & (dom.signal == "P")]
    sn = p[p.param.str.endswith("_sn")].dropna(subset=["domain_auroc_raw"])
    raw_only = p[~p.param.str.endswith("_sn")]
    ax.scatter(np.zeros(len(raw_only)) + np.linspace(-0.08, 0.08, max(len(raw_only), 1))[:len(raw_only)], raw_only.domain_auroc, color=SIG_COLOR["P"], alpha=0.6, label="Raw PPG parameters")
    for _, r in sn.iterrows():
        ax.plot([0, 1], [r.domain_auroc_raw, r.domain_auroc], "-", color="#7f8c8d", lw=0.8)
    # Plot the observed results, including uncertainty where it is available.
    ax.scatter(np.ones(len(sn)), sn.domain_auroc, color="#1abc9c", label="Self-normalized PPG parameters")
    ax.axhline(0.70, color="k", ls="--", lw=0.8, label="F5 limit 0.70")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Self-Normalized"])
    # Label the panel with the measurement names and their units.
    ax.set_ylim(0.45, 1.02)
    ax.legend(fontsize=7)
    ax.set_title("C Hospital Identifiability of PPG Parameters (Training)")
    # Label the panel with the measurement names and their units.
    ax.set_xlabel("Normalization of the PPG Parameter")
    ax.set_ylabel("Univariate Domain AUROC (Unitless)")
    ax = axes[1, 1]
    ms = ["G0", "noninvasive IND-E + IND-P", "G0 + IND-EA"]
    for k, (s, lab) in enumerate((("pooled_test", "Pooled test"), ("clean vitaldb_train->mover_test", "VitalDB to MOVER test (clean)"),
                                  ("E3x vitaldb_train->all_mover (not clean)", "VitalDB to all MOVER (not clean)"))):
        for i, m in enumerate(ms):
            r = h[(h.setting == s) & (h.model == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.errorbar(r.auroc, i + (k - 1) * 0.22, xerr=[[max(r.auroc - r.auroc_lo, 0)], [max(r.auroc_hi - r.auroc, 0)]], fmt="os^"[k],
                        color=["#34495e", "#e67e22", "#95a5a6"][k], capsize=2, label=lab if i == 0 else None)
    # Label the panel with the measurement names and their units.
    ax.set_yticks(range(len(ms)))
    ax.set_yticklabels(ms, fontsize=8)
    ax.legend(fontsize=7)
    ax.set_title("D Noninvasive Scenario: ECG + PPG Only (No ART, No MAP) vs G0")
    ax.set_xlabel("AUROC for ART-Defined IOH (Unitless, 95% CI)")
    ax.set_ylabel("Model")
    fig.suptitle("PPG Decision Study")
    save_figure(fig, root, "fig_ind_12_ppg_decision.png", panel_letters=False)
