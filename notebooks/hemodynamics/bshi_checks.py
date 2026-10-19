"""Real-data checks of the BSHI assembler (statistics, dataset shift evidence) and a gallery of real tensors."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import bshi as S
from .canonical import atomic_json, utc_now
from .figstyle import audit_figure, capitalize_figure, finish_axes, label_colorbar

CBAR_LABEL = {"ecg": "robust z-score (ECG)", "art": "(mmHg - 80) / 40 (ART)", "ppg": "robust z-score (PPG)", "map": "z-score over the window (MAP)",
              "timing": "z-score over the window (timing bands)", "morphology": "z-score over the window (morphology bands)", "sqi": "SQI (0-1)"}

CMAPS = {"ecg": ("gray_r", -3, 3), "art": ("magma", -1.5, 2.0), "ppg": ("viridis", -3, 3), "map": ("coolwarm", -3, 3), "timing": ("coolwarm", -3, 3),
         "morphology": ("coolwarm", -3, 3), "sqi": ("viridis", 0, 1)}


def _sample(frame, cases_per_dataset, per_case, seed):
    rng = np.random.default_rng(seed)
    parts = []
    for ds, g in frame.groupby("dataset"):
        counts = g.groupby("case_id").size()
        cases = counts[counts >= per_case].index.to_numpy()
        pick = rng.choice(cases, size=min(cases_per_dataset, len(cases)), replace=False)
        for cid in pick:
            gg = g[g.case_id == cid]
            parts.append(gg.iloc[rng.choice(len(gg), per_case, replace=False)])
    return pd.concat(parts).reset_index(drop=True)


def _plane_stats(x):
    return {p: {"mean": float(x[:, i].mean()), "std": float(x[:, i].std()), "min": float(x[:, i].min()), "max": float(x[:, i].max()),
                "share_zero": float((x[:, i] == 0).mean())} for i, p in enumerate(S.PLANES)}


def run_tensor_checks(root, n_full_cases=100, per_case_full=20, n_core_cases=50, per_case_core=10, seed=5052, write=True):
    """2,000 random in_full windows (both datasets, all splits) and 500 core-only windows -> results/bshi_tensor_checks.json."""
    root = Path(root)
    started = time.perf_counter()
    idx = S.load_window_index(root)
    prim = idx[idx.primary_eligible]
    full = _sample(prim[prim.in_full], n_full_cases // 2, per_case_full, seed)
    core_only = _sample(prim[prim.in_core & ~prim.in_full], n_core_cases // 2, per_case_core, seed + 1)
    out = {"windows_full": int(len(full)), "windows_core_only": int(len(core_only)), "by_split_full": full.split_pooled.value_counts().to_dict(),
           "by_dataset_full": full.dataset.value_counts().to_dict(), "events_full": int(full.y_event.sum())}
    results = {}
    for name, rows, variant in (("full_eap", full, "eap"), ("full_ea", full, "ea"), ("core_only_ea", core_only, "ea")):
        b = S.assemble(root, rows, variant, cache=S.CaseCache(maxsize=2))
        x = b["x"]
        entry = {"variant": variant, "windows": int(len(x)), "nan_or_inf_count": int((~np.isfinite(x)).sum()) + int((~np.isfinite(b["context"])).sum()),
                 "planes": _plane_stats(x), "context_mean": dict(zip(S.CONTEXT_NAMES, np.round(b["context"].mean(axis=0), 4).tolist())),
                 "masked_beat_share": {k: float(1 - b["beat_mask"][:, i].mean()) for i, k in enumerate(("ecg", "art", "ppg"))},
                 "constant_plane": [p for i, p in enumerate(S.PLANES) if float(x[:, i].std()) < 1e-9 and not (variant == "ea" and p == "ppg")],
                 "per_dataset": {}}
        for ds, sel in rows.groupby("dataset").groups.items():
            xs = x[np.asarray(list(sel))]
            cx = b["context"][np.asarray(list(sel))]
            entry["per_dataset"][ds] = {"windows": int(len(xs)), "planes": {p: {"mean": round(v["mean"], 4), "std": round(v["std"], 4)} for p, v in _plane_stats(xs).items()},
                                        "context_mean": dict(zip(S.CONTEXT_NAMES, np.round(cx.mean(axis=0), 4).tolist())),
                                        "masked_beat_share": {k: round(float(1 - b["beat_mask"][np.asarray(list(sel))][:, i].mean()), 4) for i, k in enumerate(("ecg", "art", "ppg"))}}
        results[name] = entry
        if name == "full_eap":
            x0 = x
    out["checks"] = results
    out["checks_pass"] = {"no_nan_or_inf": all(v["nan_or_inf_count"] == 0 for v in results.values()), "no_constant_plane": all(not v["constant_plane"] for v in results.values()),
                          "ecg_ppg_within_clip": bool(max(results["full_eap"]["planes"][p]["max"] for p in ("ecg", "ppg")) <= 8.0 + 1e-6),
                          "ea_ppg_plane_is_zero": results["full_ea"]["planes"]["ppg"]["share_zero"] == 1.0,
                          "full_eap_masks_ppg_more_than_art_not_less": results["full_eap"]["masked_beat_share"]["ppg"] >= results["full_eap"]["masked_beat_share"]["art"]}
    out["seconds"] = round(time.perf_counter() - started, 1)
    out["at"] = utc_now()
    if write:
        atomic_json(root / "results" / "bshi_tensor_checks.json", out)
    return out


def _pick_gallery(root, seed=5052):
    idx = S.load_window_index(root)
    pool = idx[idx.primary_eligible & idx.in_full & idx.has60 & ~idx.gray_zone]
    rng = np.random.default_rng(seed)
    chosen = []
    for ds in ("vitaldb", "mover"):
        g = pool[pool.dataset == ds]
        ev = g[g.y_event & g.time_to_event_sec.between(90, 240) & (g.min_valid_frac >= 0.9) & (g.n_ppg_ok >= 58)]
        ctrl = g[~g.y_event & (g.min_valid_frac >= 0.9) & (g.n_ppg_ok >= 58)]
        for kind, frame in (("event", ev), ("control", ctrl)):
            cases = frame.case_id.unique()
            for cid in rng.choice(cases, size=2, replace=False):
                rows = frame[frame.case_id == cid]
                chosen.append((kind, rows.iloc[int(rng.integers(len(rows)))]))
    return chosen


def draw_gallery(root, out_dir=None, seed=5052):
    """8 real windows (2 events + 2 controls per dataset), EAP on the top row and EA below, the exact 7-plane layout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    out_dir = Path(out_dir or root / "figures" / "bshi")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for kind, row in _pick_gallery(root, seed):
        one = pd.DataFrame([row]).reset_index(drop=True)
        eap, ea = S.assemble(root, one, "eap", cache=S.CaseCache(2)), S.assemble(root, one, "ea", cache=S.CaseCache(2))
        fig, axes = plt.subplots(2, 7, figsize=(27, 8.4))
        for r, (label, b) in enumerate((("EAP (with PPG)", eap), ("EA (PPG groups zeroed)", ea))):
            for i, p in enumerate(S.PLANES):
                cmap, lo, hi = CMAPS[p]
                ax = axes[r, i]
                im = ax.imshow(b["x"][0, i], aspect="auto", cmap=cmap, vmin=lo, vmax=hi, origin="lower", interpolation="nearest")
                label_colorbar(fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02), CBAR_LABEL[p])
                bands = {"timing": (S.TIMING_BANDS, 50), "morphology": (S.MORPH_BANDS, 25), "sqi": (S.SQI_BANDS, 50)}.get(p)
                if bands:
                    for k, name in enumerate(bands[0]):
                        ax.text((k + 0.5) * bands[1], 59, name, rotation=90, ha="center", va="top", fontsize=6, color="k", bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, lw=0))
                finish_axes(ax, "phase within the R-R cycle\n(0 = R wave, 200 = next R wave, samples)", "beat in window\n(oldest to newest, 1-60)", f"{label}: {i + 1} {p}")
                ax.set_yticks([0, 19, 39, 59])
                ax.set_yticklabels([1, 20, 40, 60])
        tte = row.time_to_event_sec
        fig.suptitle(f"BSHI {row.dataset} case {str(row.case_id)[:10]}  t_end={row.t_end_sec:.0f}s | MAP {row.map_current:.0f} | "
                     + (f"EVENT in {tte:.0f}s" if row.y_event else "no event in 300 s") + f" | dMAP {row.delta_map:+.0f}, AUT65 {row.aut65:.1f} | PPG beats {int(row.n_ppg_ok)}/60",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        path = out_dir / f"bshi_{row.dataset}_{kind}_{str(row.case_id)[:8]}_{int(row.t_end_sec)}.png"
        capitalize_figure(fig)
        audit_figure(fig, path.name)
        fig.savefig(path, dpi=80)
        plt.close(fig)
        paths.append(str(path.relative_to(root)))
    return paths
