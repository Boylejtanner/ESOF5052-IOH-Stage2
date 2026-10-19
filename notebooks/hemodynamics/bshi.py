"""BSHI (beat-synchronous hemodynamic image) windows: index, on-the-fly assembler, batch iterator, compact codec.

Nothing is materialized for every window (1.4 M x 7x60x200 would be ~200 GB). ``build_window_index`` records, for every labeled window, which 60 rows of the
per-case beat table form its image and whether the window passes the beat-quality rules; ``assemble`` builds the tensors from the stored beat files.

Window (forecast origin ``t_end``): the 60 most recent beats of the case with ``t_r > t_end - 90`` whose whole cycle ``t_r + rr_sec`` is ``<= t_end`` (no beat touches
the future). Fewer than 60 -> the window has no image.

Per-beat masks (nested): ``m_ecg = valid_beat & sqi_ecg >= 0.5``; ``m_art = m_ecg & art_ok``; ``m_ppg = m_art & ppg_ok & ptt_ecg_ppg <= 0.9``.
``in_core``: >= 54 of the 60 beats have ``m_art``; ``in_full``: >= 54 have ``m_ppg``; both exclude the dead-ECG cases. Masked values are zero AFTER normalization,
their SQI entries are zero, and nothing is interpolated or filled.

Layout (7, 60, 200): beat on axis 1 (oldest -> newest), phase within the R-R cycle on axis 2.
  0 ECG | 1 ART | 2 PPG | 3 MAP z-score (map_beat broadcast over phase) | 4 timing: 4 bands x 50 columns (ptt_ecg_art, ptt_ecg_ppg, ptt_art_ppg, rr_sec)
  5 morphology: 8 bands x 25 columns (pp, dpdt_max, sys_time, art_area, ppg_amp, ppg_ac_dc, ppg_upstroke, art_ppg_corr) | 6 SQI: 4 bands x 50 (sqi_ecg, sqi_art, sqi_ppg, sqi_cross)
Normalization uses only the window itself: ECG and PPG = robust z-score (nearest-rank median / IQR of the unmasked samples) clipped to +-8; ART = fixed physical scaling
(mmHg - 80) / 40 (absolute pressure level preserved); planes 3-5 = per-window z-score over the unmasked beats; SQI = raw 0-1.
The stored ECG and PPG waveforms are already standardized per beat, so their absolute amplitude lives in the morphology bands. PPG amplitude and AC/DC are in device-native
units (about 20x apart between MOVER and VitalDB), so they appear only as per-window z-scored bands and never as raw means in the context vector.

Variants: ``eap`` = everything; ``ea`` = the PPG groups are zeroed (plane 2, timing bands ptt_ecg_ppg / ptt_art_ppg, morphology bands ppg_amp / ppg_ac_dc / ppg_upstroke /
art_ppg_corr, SQI bands sqi_ppg and sqi_cross). ``sqi_cross`` is zeroed because it contains a PPG-timing term, which would leak PPG into the PPG-free variant.
Ablation protocol (``PROTOCOLS``): EA on the core set, EAP on the full set, and EA on the full set (like-for-like, isolates the PPG effect).
``variant_mask`` (one 0/1 entry per group in ``GROUPS``) is returned with every sample. No dataset indicator is part of the inputs unless explicitly requested.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import multiprocessing
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import beats_pipeline as P
from .canonical import atomic_json, utc_now
from .snapshot import assert_dataset_v1, index_content_hash

BSHI_CODE_VERSION = "bs2"
OUT_DIR = Path("data") / "derived" / "bshi"
N_BEATS, PHASE = 60, 200
PLANES = ["ecg", "art", "ppg", "map", "timing", "morphology", "sqi"]
TIMING_BANDS = ["ptt_ecg_art", "ptt_ecg_ppg", "ptt_art_ppg", "rr_sec"]
MORPH_BANDS = ["pp", "dpdt_max", "sys_time", "art_area", "ppg_amp", "ppg_ac_dc", "ppg_upstroke", "art_ppg_corr"]
SQI_BANDS = ["sqi_ecg", "sqi_art", "sqi_ppg", "sqi_cross"]
GROUPS = ["ecg", "art", "ppg", "map"] + TIMING_BANDS + MORPH_BANDS + SQI_BANDS
PPG_GROUPS = {"ppg", "ptt_ecg_ppg", "ptt_art_ppg", "ppg_amp", "ppg_ac_dc", "ppg_upstroke", "art_ppg_corr", "sqi_ppg", "sqi_cross"}
ART_ONLY_ON = {"art", "map", "pp", "dpdt_max", "sys_time", "art_area"}
# Input recipes: the set of groups that are OFF. eap/ea are the the analysis workflow variants; the others map the proposal's matched models (A = map_only ... G = eap) onto the same layout.
VARIANTS = {"eap": frozenset(), "ea": frozenset(PPG_GROUPS),
            "wave": frozenset({"map"} | set(TIMING_BANDS) | set(MORPH_BANDS) | set(SQI_BANDS)),         # ECG + ART + PPG planes only ("BSHI waveform only")
            "wave_map": frozenset(set(TIMING_BANDS) | set(MORPH_BANDS) | set(SQI_BANDS)),               # + MAP plane
            "wave_couple": frozenset(SQI_BANDS),                                                          # + timing and morphology coupling bands, no quality plane
            "art_only": frozenset(set(GROUPS) - ART_ONLY_ON),                                             # ART plane + MAP plane + ART scalars
            "map_only": frozenset(GROUPS)}                                                                # context only (current-MAP features)
PROPOSAL_MODELS = {"A_map_only": "map_only", "B_art_only": "art_only", "E_bshi_waveform_only": "wave", "E2_bshi_wave_map": "wave_map", "F_bshi_coupling": "wave_couple",
                   "G_bshi_sqi_weighting": "eap", "G_no_ppg": "ea"}
PROTOCOLS = {"ea_core": ("ea", "core"), "eap_full": ("eap", "full"), "ea_full": ("ea", "full")}
MASK_OF = {"ptt_ecg_art": "art", "ptt_ecg_ppg": "ppg", "ptt_art_ppg": "ppg", "rr_sec": "ecg", "pp": "art", "dpdt_max": "art", "sys_time": "art", "art_area": "art",
           "ppg_amp": "ppg", "ppg_ac_dc": "ppg", "ppg_upstroke": "ppg", "art_ppg_corr": "ppg", "map_beat": "art", "sbp": "art", "dbp": "art", "hr_bpm": "ecg",
           "sqi_ecg": "ecg", "sqi_art": "art", "sqi_ppg": "ppg", "sqi_cross": "art"}
LABEL_CONTEXT = ["map_current", "map_mean_60", "map_min_60", "map_std_60", "map_slope_60"]
BEAT_CONTEXT = ["hr_mean", "sbp_mean", "dbp_mean", "n_ecg_ok", "n_art_ok", "n_ppg_ok"]
# The PPG is stored in device-native units (MOVER ppg_amp ~600 vs VitalDB ~27): a raw mean of these two features identifies the dataset perfectly. They enter the image only
# through the per-window z-scored morphology bands (scale-free) and are kept out of the context vector.
CONTEXT_EXCLUDE = {"ppg_amp", "ppg_ac_dc"}
SCALAR_MEANS = [n for n in TIMING_BANDS + MORPH_BANDS + SQI_BANDS if n not in CONTEXT_EXCLUDE]
CONTEXT_NAMES = LABEL_CONTEXT + BEAT_CONTEXT + ["mean_" + s for s in SCALAR_MEANS]
LABEL_OUT = ["y_event", "delta_map", "delta_map_robust", "aut65", "gray_zone"]
SCALAR_COLUMNS = ["t_r", "rr_sec", "hr_bpm", "sbp", "dbp", "map_beat", "valid_beat", "sqi_ecg", "art_ok", "ppg_ok", "ptt_ecg_ppg"] + \
    [c for c in dict.fromkeys(TIMING_BANDS + MORPH_BANDS + SQI_BANDS)]
INDEX_NAME = "window_index_v1.parquet"


def bshi_config(root_or_settings):
    s = root_or_settings
    if not isinstance(s, dict):
        s = json.loads((Path(s) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))
    return s["bshi"]


def variant_mask(variant):
    off = VARIANTS[variant]
    return np.array([0.0 if g in off else 1.0 for g in GROUPS], dtype=np.float32)


# index
def _cum(mask):
    return np.concatenate([[0], np.cumsum(mask, dtype=np.int64)])


def beat_masks(valid, sqi_ecg, art_ok, ppg_ok, ptt_ecg_ppg, cfg):
    m_ecg = np.asarray(valid, bool) & (np.asarray(sqi_ecg) >= cfg["sqi_ecg_min"])
    m_art = m_ecg & np.asarray(art_ok, bool)
    m_ppg = m_art & np.asarray(ppg_ok, bool) & (np.asarray(ptt_ecg_ppg, dtype=float) <= cfg["ppg_delay_max_sec"])
    return m_ecg, m_art, m_ppg


def index_windows(t_r, rr, valid, sqi_ecg, art_ok, ppg_ok, ptt_ecg_ppg, t_end, cfg):
    """Rows of the beat table that make each window's image, and the usable-beat counts (pure function of one case)."""
    t_end = np.asarray(t_end, dtype=float)
    n = len(t_r)
    out = {k: np.zeros(len(t_end), dtype=np.int32) for k in ("beat_lo", "beat_hi", "n_beats_avail", "n_ecg_ok", "n_art_ok", "n_ppg_ok")}
    out["has60"] = np.zeros(len(t_end), dtype=bool)
    if n == 0:
        out["beat_lo"][:] = -1
        out["beat_hi"][:] = -1
        return out
    t_r = np.asarray(t_r, dtype=float)
    cycle_end = np.maximum.accumulate(t_r + np.asarray(rr, dtype=np.float64))          # running max: every beat before ``hi`` ends at or before t_end
    hi = np.searchsorted(cycle_end, t_end, side="right")
    first = np.searchsorted(t_r, t_end - cfg["window_sec"], side="right")
    avail = hi - first
    has60 = avail >= cfg["n_beats"]
    lo = np.where(has60, hi - cfg["n_beats"], np.minimum(first, hi))
    m_ecg, m_art, m_ppg = beat_masks(valid, sqi_ecg, art_ok, ppg_ok, ptt_ecg_ppg, cfg)
    for name, m in (("n_ecg_ok", m_ecg), ("n_art_ok", m_art), ("n_ppg_ok", m_ppg)):
        cs = _cum(m)
        out[name] = (cs[hi] - cs[lo]).astype(np.int32)
    out.update(beat_lo=lo.astype(np.int32), beat_hi=hi.astype(np.int32), n_beats_avail=np.maximum(avail, 0).astype(np.int32), has60=has60)
    return out


def _case_sig(global_sig, root, dataset, case_id, win, flags):
    meta = P.case_dir(root, dataset, case_id) / "beats_meta.json"
    h = hashlib.sha256(global_sig.encode())
    h.update(meta.read_bytes() if meta.exists() else b"")
    h.update(pd.util.hash_pandas_object(win[["t_end_sec", "y_event", "aut65", "min_valid_frac", "eligible_onset"]], index=False).to_numpy().tobytes())
    h.update(repr(flags).encode())
    return h.hexdigest()


def _part_sig(path):
    try:
        return (pq.read_metadata(path).metadata or {}).get(b"sig", b"").decode()
    except (OSError, pa.ArrowInvalid):
        return ""


def _index_job(root, dataset, case_id, win, cfg, flags, part, global_sig):
    try:
        sig = _case_sig(global_sig, root, dataset, case_id, win, flags)
        if Path(part).exists() and _part_sig(part) == sig:
            return {"status": "reused", "case_id": case_id}
        try:
            b = P.load_beats(root, dataset, case_id, columns=["t_r", "rr_sec", "valid_beat", "sqi_ecg", "art_ok", "ppg_ok", "ptt_ecg_ppg"])
        except (OSError, FileNotFoundError):
            b = pd.DataFrame({c: [] for c in ("t_r", "rr_sec", "valid_beat", "sqi_ecg", "art_ok", "ppg_ok", "ptt_ecg_ppg")})
        mono = bool((np.diff(b.t_r.to_numpy()) > 0).all()) if len(b) > 1 else True
        r = index_windows(b.t_r.to_numpy(), b.rr_sec.to_numpy(), b.valid_beat.to_numpy(), b.sqi_ecg.to_numpy(), b.art_ok.to_numpy(), b.ppg_ok.to_numpy(),
                          b.ptt_ecg_ppg.to_numpy(), win.t_end_sec.to_numpy(), cfg)
        f = win.copy()
        for k, v in r.items():
            f[k] = v
        f["primary_eligible"] = (f.eligible_onset & ~f.label_ambiguous & (f.min_valid_frac >= 0.7 - 1e-6)).to_numpy()
        f["ecg_dead_case"], f["ppg_suspect_case"] = bool(flags[0]), bool(flags[1])
        f["beat_rule_core"] = f.has60 & (f.n_art_ok >= cfg["min_usable_beats"])
        f["beat_rule_full"] = f.has60 & (f.n_ppg_ok >= cfg["min_usable_beats"])
        f["in_core"] = f.beat_rule_core & ~f.ecg_dead_case
        f["in_full"] = f.beat_rule_full & ~f.ecg_dead_case
        table = pa.Table.from_pandas(f, preserve_index=False)
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"sig": sig.encode()})
        Path(part).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(part) + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, part)
        return {"status": "built", "case_id": case_id, "non_monotone": not mono}
    except Exception as exc:
        return {"status": "failed", "case_id": case_id, "error": f"{type(exc).__name__}: {exc}"}


def index_path(root):
    return Path(root) / OUT_DIR / INDEX_NAME


def _meta_path(root):
    return Path(root) / OUT_DIR / "window_index_v1.meta.json"


def index_signature(root):
    from . import case_quality as CQ, labels as L
    settings = json.loads((Path(root) / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))
    cfg = {k: v for k, v in settings["bshi"].items() if k not in ("dev_subset", "workers")}
    tag = json.loads((Path(root) / "data" / "manifests" / "dataset_v1.json").read_text(encoding="utf-8"))
    parts = {"code": BSHI_CODE_VERSION, "cfg": cfg, "beats": P.config_hash(P.load_beats_config(settings)), "labels": L.label_rule_sha256(L.load_label_rule(settings)),
             "labels_code": L.LABEL_CODE_VERSION, "dataset": tag["index_content_sha256"], "quality": json.loads(CQ._sig_path(root).read_text())["signature"]}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def build_window_index(root, workers=None, progress=True, force=False):
    """data/derived/bshi/window_index_v1.parquet: one row per labeled window. Resumable by case (per-case parts with their own signature)."""
    from . import case_quality as CQ, labels as L
    root = Path(root)
    assert_dataset_v1(root)
    cfg = bshi_config(root)
    sig = index_signature(root)
    if not force and index_path(root).exists() and _meta_path(root).exists() and json.loads(_meta_path(root).read_text())["signature"] == sig:
        return {"status": "current", "rows": json.loads(_meta_path(root).read_text())["rows"]}
    quality = CQ.load_case_quality(root).set_index(["dataset", "case_id"])
    labels = L.load_labels(root)
    labels["case_id"] = labels.case_id.astype(str)
    labels = labels.sort_values(["dataset", "case_id", "t_end_sec"], kind="stable")
    groups = list(labels.groupby(["dataset", "case_id"], sort=False))
    n_workers = max(1, min(int(workers or cfg["workers"]), (os.cpu_count() or 8) - 2))
    parts_dir = root / OUT_DIR / "index_parts"
    stats = {"built": 0, "reused": 0, "failed": 0, "non_monotone": 0}
    errors = []
    bar = None
    if progress:
        from tqdm.auto import tqdm
        bar = tqdm(total=len(groups), desc="window index", unit="case")
    with cf.ProcessPoolExecutor(max_workers=n_workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futs = []
        for (ds, cid), win in groups:
            q = quality.loc[(ds, cid)]
            futs.append(pool.submit(_index_job, str(root), ds, cid, win.reset_index(drop=True), cfg, (bool(q.ecg_dead), bool(q.ppg_suspect_case)),
                                    str(parts_dir / ds / f"{cid}.parquet"), sig))
        for fut in cf.as_completed(futs):
            r = fut.result()
            stats[r["status"]] += 1
            stats["non_monotone"] += int(r.get("non_monotone", False))
            if r["status"] == "failed":
                errors.append(r)
            bar and bar.update(1)
    bar and bar.close()
    if errors:
        raise RuntimeError(f"{len(errors)} case(s) failed, first: {errors[0]}")
    tables = [pq.read_table(parts_dir / ds / f"{cid}.parquet") for (ds, cid), _ in groups]
    frame = pa.concat_tables(tables).replace_schema_metadata(None).to_pandas()
    frame = frame.sort_values(["dataset", "case_id", "t_end_sec"], kind="stable").reset_index(drop=True)
    frame.insert(0, "window_id", np.arange(len(frame), dtype=np.int64))
    check_index(frame)
    index_path(root).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(index_path(root)) + ".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, index_path(root))
    atomic_json(_meta_path(root), {"signature": sig, "rows": int(len(frame)), "at": utc_now(), "stats": stats, "version": BSHI_CODE_VERSION})
    return {"status": "built", "rows": int(len(frame)), **stats}


def check_index(df):
    """Assertions every index must satisfy (raises AssertionError)."""
    assert df.window_id.is_unique, "window_id must be unique"
    spl = df.groupby("patient_key").split_pooled.nunique()
    assert (spl == 1).all(), f"{int((spl > 1).sum())} patient(s) appear in more than one pooled split"
    spl2 = df.groupby("patient_key").split_within_dataset.nunique()
    assert (spl2 == 1).all(), "a patient appears in more than one within-dataset split"
    for c in ("has60", "in_core", "in_full", "primary_eligible", "ecg_dead_case"):
        assert df[c].notna().all(), f"NaN in flag {c}"
    assert not (df.in_full & ~df.in_core).any(), "an in_full window is not in_core"
    assert not (df.in_core & df.ecg_dead_case).any(), "a dead-ECG case leaked into the core set"
    assert (df.beat_hi[df.has60] - df.beat_lo[df.has60] == N_BEATS).all(), "has60 windows must span exactly 60 beats"
    return True


def load_window_index(root, variant=None, columns=None):
    """All windows (variant=None) or the primary-eligible windows of a protocol: 'ea_core', 'eap_full', 'ea_full'."""
    df = pd.read_parquet(index_path(root), columns=columns)
    if variant is None:
        return df
    _, subset = PROTOCOLS[variant]
    flag = {"core": "in_core", "full": "in_full"}[subset]
    if columns is not None and not {"primary_eligible", flag} <= set(columns):
        df = pd.read_parquet(index_path(root), columns=list(dict.fromkeys(list(columns) + ["primary_eligible", flag])))
    return df.loc[df.primary_eligible & df[flag]].reset_index(drop=True)


ANCHORS = {"core": {"windows": 903394, "events": 28199, "mover_windows": 48737, "mover_events": 2302, "mover_patients": 391, "vitaldb_windows": 854657, "vitaldb_events": 25897},
           "full": {"windows": 794772, "events": 24540, "mover_windows": 31025, "mover_events": 1166, "mover_patients": 335, "vitaldb_windows": 763747, "vitaldb_events": 23374}}


def _cell(g):
    ev = g.loc[g.primary_eligible & g.y_event.astype(bool)]
    return {"windows": int(len(g)), "events": int(len(ev)), "patients": int(g.patient_key.nunique()), "patients_with_event": int(ev.patient_key.nunique())}


def index_summary(root, write=True):
    """Funnel (all windows -> primary-eligible -> 60 beats -> core -> full) by dataset and split, reconciled with the earlier ad-hoc counts."""
    df = load_window_index(root)
    steps = {"all_labeled": df, "primary_eligible": df[df.primary_eligible], "primary_has60": df[df.primary_eligible & df.has60],
             "primary_core": df[df.primary_eligible & df.in_core], "primary_full": df[df.primary_eligible & df.in_full]}
    out = {"version": BSHI_CODE_VERSION, "windows_total": int(len(df)), "funnel": {}, "reconciliation": {}}
    for split_col in ("split_pooled", "split_within_dataset"):
        rows = []
        for step, g in steps.items():
            for (ds, split), cell in g.groupby(["dataset", split_col]):
                rows.append({"step": step, "dataset": ds, "split": split, **_cell(cell)})
            for split, cell in g.groupby(split_col):
                rows.append({"step": step, "dataset": "all", "split": split, **_cell(cell)})
            for ds, cell in g.groupby("dataset"):
                rows.append({"step": step, "dataset": ds, "split": "all", **_cell(cell)})
            rows.append({"step": step, "dataset": "all", "split": "all", **_cell(g)})
        out["funnel"][split_col] = rows
    dead = df[df.beat_rule_core & df.ecg_dead_case & df.primary_eligible]
    out["removed_by_dead_ecg_rule"] = {"core_windows": int(len(dead)), "full_windows": int((df.beat_rule_full & df.ecg_dead_case & df.primary_eligible).sum())}
    out["windows_without_60_beats_among_primary"] = int((df.primary_eligible & ~df.has60).sum())
    for name, key in (("core", "primary_core"), ("full", "primary_full")):
        g = steps[key]
        got = {"windows": int(len(g)), "events": int(g.y_event.sum()), "mover_windows": int((g.dataset == "mover").sum()), "mover_events": int(g.y_event[g.dataset == "mover"].sum()),
               "mover_patients": int(g.loc[g.dataset == "mover", "patient_key"].nunique()), "vitaldb_windows": int((g.dataset == "vitaldb").sum()),
               "vitaldb_events": int(g.y_event[g.dataset == "vitaldb"].sum())}
        out["reconciliation"][name] = {"now": got, "earlier_adhoc_anchor": ANCHORS[name], "difference": {k: got[k] - ANCHORS[name][k] for k in got},
                                       "difference_pct_windows": round(100 * (got["windows"] - ANCHORS[name]["windows"]) / ANCHORS[name]["windows"], 3)}
    out["at"] = utc_now()
    if write:
        atomic_json(Path(root) / "results" / "bshi_index_summary.json", out)
    return out


# assembler
class CaseBeats:
    """One case's beat table (float arrays), masks and float16 waveforms, ready for window gathers."""
    __slots__ = ("dataset", "case_id", "cols", "m", "waves", "cycle_end", "n")

    def __init__(self, root, dataset, case_id, cfg):
        t = P.load_beats(root, dataset, case_id, columns=list(dict.fromkeys(SCALAR_COLUMNS)))
        waves, beat_idx = P.load_beat_waveforms(root, dataset, case_id)
        if len(beat_idx) != len(t) or not (beat_idx == np.arange(len(t))).all():
            raise ValueError(f"beat tables and waveforms are not aligned for {dataset}/{case_id}")
        self.dataset, self.case_id, self.n = dataset, str(case_id), len(t)
        self.cols = {c: t[c].to_numpy() for c in t.columns}
        self.cycle_end = self.cols["t_r"] + self.cols["rr_sec"].astype(np.float64)
        self.m = dict(zip(("ecg", "art", "ppg"), beat_masks(t.valid_beat.to_numpy(), t.sqi_ecg.to_numpy(), t.art_ok.to_numpy(), t.ppg_ok.to_numpy(), t.ptt_ecg_ppg.to_numpy(), cfg)))
        self.waves = waves


class CaseCache:
    def __init__(self, maxsize=6):
        self.maxsize, self.items = maxsize, OrderedDict()

    def get(self, root, dataset, case_id, cfg):
        key = (str(root), dataset, str(case_id))
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        item = CaseBeats(root, dataset, case_id, cfg)
        self.items[key] = item
        while len(self.items) > self.maxsize:
            self.items.popitem(last=False)
        return item


_DEFAULT_CACHE = CaseCache()


def quantile_nearest_rank(sorted_values, count, q):
    """Nearest-rank quantile used for the robust z-score (position floor((count-1)*q + 0.5) of the sorted valid samples)."""
    return sorted_values[int(np.floor((count - 1) * q + 0.5))]


def _robust_z(block, valid, clip):
    m = block.shape[0]
    flat = np.where(valid, block, np.nan).reshape(m, -1)
    srt = np.sort(flat, axis=1)                                                      # NaN (masked) samples sort last
    cnt = valid.reshape(m, -1).sum(axis=1)
    rows = np.arange(m)

    def q(p):
        return srt[rows, np.clip(np.floor((cnt - 1) * p + 0.5).astype(np.int64), 0, flat.shape[1] - 1)]

    med, scale = q(0.5), q(0.75) - q(0.25)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = np.clip((block - med[:, None, None]) / scale[:, None, None], -clip, clip)
    return np.where(valid, z, 0.0).astype(np.float32)


def _scalar_z(values, valid):
    cnt = valid.sum(axis=1)
    safe = np.maximum(cnt, 1)
    v = np.where(valid, values, 0.0)
    mean = v.sum(axis=1) / safe
    std = np.sqrt(np.where(valid, (values - mean[:, None]) ** 2, 0.0).sum(axis=1) / safe)
    den = np.where(std > 1e-9, std, 1.0)
    return np.where(valid, (values - mean[:, None]) / den[:, None], 0.0).astype(np.float32), mean.astype(np.float32), cnt


def _assemble_case(cb, lo, ctx_labels, variant, cfg, extra=None, extra_ppg=None):
    m = len(lo)
    off = VARIANTS[variant]
    idx = lo[:, None] + np.arange(N_BEATS)[None, :]
    mask = {k: cb.m[k][idx] for k in ("ecg", "art", "ppg")}
    if extra is not None:                                                    # beats treated as NOT captured (exact corruption): they leave every mask before normalization
        for k in mask:
            mask[k] = mask[k] & extra
    if extra_ppg is not None:
        mask["ppg"] = mask["ppg"] & extra_ppg
    if "ppg" in off:
        mask["ppg"] = np.zeros_like(mask["ppg"])
    x = np.zeros((m, 7, N_BEATS, PHASE), dtype=np.float32)
    clip = float(cfg["ecg_robust_clip"])
    e = cb.waves["ecg"][idx].astype(np.float32)
    x[:, 0] = _robust_z(e, mask["ecg"][:, :, None] & np.isfinite(e), clip)
    a = cb.waves["art"][idx].astype(np.float32)
    va = mask["art"][:, :, None] & np.isfinite(a)
    x[:, 1] = np.where(va, np.clip((a - cfg["art_center_mmhg"]) / cfg["art_scale_mmhg"], -clip, clip), 0.0)
    if "ppg" not in off:
        p = cb.waves["ppg"][idx].astype(np.float32)
        x[:, 2] = _robust_z(p, mask["ppg"][:, :, None] & np.isfinite(p), clip)

    def scalar(name):
        return cb.cols[name][idx].astype(np.float64)

    ctx = np.zeros((m, len(CONTEXT_NAMES)), dtype=np.float32)
    ctx[:, :len(LABEL_CONTEXT)] = ctx_labels
    pos = {n: i for i, n in enumerate(CONTEXT_NAMES)}
    z, _, _ = _scalar_z(scalar("map_beat"), mask["art"] & np.isfinite(scalar("map_beat")))
    x[:, 3] = z[:, :, None]

    def band(plane, name, k, width):
        group_off = name in off
        v = scalar(name)
        valid = mask[MASK_OF[name]] & np.isfinite(v)
        if name.startswith("sqi_"):
            vals = np.where(valid, np.clip(v, 0.0, 1.0), 0.0)
            mean = vals.sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
        else:
            vals, mean, _ = _scalar_z(v, valid)
            mean = (np.where(valid, v, 0.0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1))
        if group_off:
            vals, mean = np.zeros_like(vals), np.zeros_like(mean)
        x[:, plane, :, k * width:(k + 1) * width] = vals[:, :, None]
        if "mean_" + name in pos:
            ctx[:, pos["mean_" + name]] = mean

    for k, name in enumerate(TIMING_BANDS):
        band(4, name, k, 50)
    for k, name in enumerate(MORPH_BANDS):
        band(5, name, k, 25)
    for k, name in enumerate(SQI_BANDS):
        band(6, name, k, 50)
    for name, mk, col in (("hr_mean", "ecg", "hr_bpm"), ("sbp_mean", "art", "sbp"), ("dbp_mean", "art", "dbp")):
        v = scalar(col)
        valid = mask[mk] & np.isfinite(v)
        ctx[:, pos[name]] = np.where(valid, v, 0.0).sum(axis=1) / np.maximum(valid.sum(axis=1), 1)
    for name, k in (("n_ecg_ok", "ecg"), ("n_art_ok", "art"), ("n_ppg_ok", "ppg")):
        ctx[:, pos[name]] = mask[k].sum(axis=1)
    return x, np.nan_to_num(ctx, nan=0.0), np.stack([mask["ecg"], mask["art"], mask["ppg"]], axis=1)


def assemble(root, rows, variant="eap", cache=None, include_dataset_flag=False, return_times=False, extra_mask=None, extra_mask_ppg=None):
    """Tensors for index rows (any order). Returns a dict of numpy arrays aligned with ``rows``; see the module docstring for the layout.

    ``variant`` may be any key of ``VARIANTS``: eap/ea are assembled natively, every other recipe is eap (or ea when the recipe drops PPG) with its groups zeroed (``apply_variant``).
    ``return_times`` adds ``t_rel`` (beat start relative to t0, s) and ``rr`` (s), both (N,60), which the compact store needs for gap augmentation.
    ``extra_mask`` / ``extra_mask_ppg``: optional (N,60) bool arrays (aligned with ``rows``) of beats that count as NOT captured (all signals / PPG only): exact corruption,
    applied before normalization like a real gap."""
    cfg = bshi_config(root)
    cache = cache or _DEFAULT_CACHE
    rows = rows.reset_index(drop=True)
    n = len(rows)
    base = variant if variant in ("eap", "ea") else ("ea" if "ppg" in VARIANTS[variant] else "eap")
    x = np.zeros((n, 7, N_BEATS, PHASE), dtype=np.float32)
    ctx = np.zeros((n, len(CONTEXT_NAMES) + (1 if include_dataset_flag else 0)), dtype=np.float32)
    bmask = np.zeros((n, 3, N_BEATS), dtype=bool)
    t_rel = np.zeros((n, N_BEATS), dtype=np.float32) if return_times else None
    rr = np.zeros((n, N_BEATS), dtype=np.float32) if return_times else None
    if (rows.beat_hi - rows.beat_lo != N_BEATS).any():
        raise ValueError("every requested window must have 60 beats (has60)")
    for (ds, cid), g in rows.groupby(["dataset", "case_id"], sort=False):
        pos = g.index.to_numpy()
        cb = cache.get(root, ds, cid, cfg)
        lo = g.beat_lo.to_numpy().astype(np.int64)
        cx, cc, cm = _assemble_case(cb, lo, g[LABEL_CONTEXT].to_numpy(dtype=np.float64), base, cfg,
                                    None if extra_mask is None else np.asarray(extra_mask)[pos], None if extra_mask_ppg is None else np.asarray(extra_mask_ppg)[pos])
        x[pos], ctx[pos, :len(CONTEXT_NAMES)], bmask[pos] = cx, cc, cm
        if return_times:
            ii = lo[:, None] + np.arange(N_BEATS)[None, :]
            t_rel[pos] = (cb.cols["t_r"][ii] - g.t_end_sec.to_numpy(dtype=np.float64)[:, None]).astype(np.float32)
            rr[pos] = cb.cols["rr_sec"][ii].astype(np.float32)
    if include_dataset_flag:
        ctx[:, -1] = (rows.dataset == "vitaldb").to_numpy()
    out = {"x": x, "context": ctx, "beat_mask": bmask, "variant_mask": np.tile(variant_mask(base), (n, 1)), "window_id": rows.window_id.to_numpy(dtype=np.int64),
           "dataset_code": (rows.dataset == "vitaldb").to_numpy(dtype=np.int8)}
    for c in LABEL_OUT:
        out[c] = rows[c].to_numpy(dtype=np.float32)
    if return_times:
        out["t_rel"], out["rr"] = t_rel, rr
    return out if variant == base else apply_variant(out, variant)


PLANE_OF_GROUP = {"ecg": 0, "art": 1, "ppg": 2, "map": 3}


def context_off(off):
    """Context entries that come from the groups in ``off`` (the label-derived MAP features are always kept)."""
    names = {"mean_" + g for g in off if "mean_" + g in CONTEXT_NAMES}
    if "ecg" in off:
        names |= {"hr_mean", "n_ecg_ok"}
    if "art" in off:
        names |= {"sbp_mean", "dbp_mean", "n_art_ok"}
    if "ppg" in off:
        names |= {"n_ppg_ok"}
    return sorted(names)


def apply_variant(batch, variant):
    """Derive a variant from an assembled batch (e.g. EA from stored EAP): zero the variant's groups in the tensor and the context. Equals ``assemble(..., variant)``
    for every window whose non-PPG groups use the ART mask, i.e. all of them."""
    off = VARIANTS[variant]
    out = dict(batch)
    x = np.array(batch["x"], copy=True)
    ctx = np.array(batch["context"], copy=True)
    for g, plane in PLANE_OF_GROUP.items():
        if g in off:
            x[:, plane] = 0
    for plane, names, width in ((4, TIMING_BANDS, 50), (5, MORPH_BANDS, 25), (6, SQI_BANDS, 50)):
        for k, name in enumerate(names):
            if name in off:
                x[:, plane, :, k * width:(k + 1) * width] = 0
    for name in context_off(off):
        ctx[:, CONTEXT_NAMES.index(name)] = 0
    if "ppg" in off:
        bm = np.array(batch["beat_mask"], copy=True)
        bm[:, 2] = False
        out["beat_mask"] = bm
    out.update(x=x, context=ctx, variant_mask=np.tile(variant_mask(variant), (len(x), 1)))
    return out


def to_planes(x):
    """Named views of a batch (N,7,60,200): the three waveform planes plus the decoded per-beat scalars (N,60) for every band."""
    x = np.asarray(x)
    planes = {name: x[:, i] for i, name in enumerate(PLANES[:4])}
    planes["scalars"] = {name: x[:, 4, :, k * 50] for k, name in enumerate(TIMING_BANDS)}
    planes["scalars"].update({name: x[:, 5, :, k * 25] for k, name in enumerate(MORPH_BANDS)})
    planes["scalars"].update({name: x[:, 6, :, k * 50] for k, name in enumerate(SQI_BANDS)})
    return planes


def window_batch_iter(root, index, variant="eap", batch_size=256, shuffle=True, seed=0, cases_per_chunk=4, cache=None, include_dataset_flag=False, case_subset=None):
    """Case-grouped batches: shuffle the order of cases, gather ``cases_per_chunk`` cases, shuffle their windows, yield batches. Locality keeps the loaded-case cache hot."""
    rng = np.random.default_rng(seed)
    keys = index.groupby(["dataset", "case_id"], sort=False).indices
    names = sorted(keys)
    if case_subset is not None:
        names = names[case_subset[0]::case_subset[1]]
    if shuffle:
        names = [names[i] for i in rng.permutation(len(names))]
    cache = cache or CaseCache(maxsize=max(2, cases_per_chunk + 1))
    for i in range(0, len(names), cases_per_chunk):
        pos = np.concatenate([keys[k] for k in names[i:i + cases_per_chunk]])
        if shuffle:
            pos = pos[rng.permutation(len(pos))]
        for j in range(0, len(pos), batch_size):
            yield assemble(root, index.iloc[pos[j:j + batch_size]], variant, cache=cache, include_dataset_flag=include_dataset_flag)


TORCH_INDEX_COLUMNS = ["window_id", "dataset", "case_id", "beat_lo", "beat_hi"] + LABEL_CONTEXT + LABEL_OUT


def make_torch_dataset(root, index, variant="eap", batch_size=256, shuffle=True, seed=0, cases_per_chunk=4, include_dataset_flag=False):
    """torch IterableDataset yielding ready batches: ``DataLoader(ds, batch_size=None, num_workers=N, persistent_workers=True)`` (module-level class, so Windows spawn workers can
    unpickle it). On Windows, starting N workers takes about a minute for N=16, so use persistent workers: every new iteration inside a worker is a new epoch with a new shuffle.
    The index is slimmed to the columns the assembler needs."""
    from .bshi_torch import BshiBatches
    return BshiBatches(str(root), index[TORCH_INDEX_COLUMNS].reset_index(drop=True), variant, batch_size, shuffle, seed, cases_per_chunk, include_dataset_flag)


# compact codec
SCALAR_ORDER = ["map_beat"] + TIMING_BANDS + MORPH_BANDS + SQI_BANDS
SCALAR_LAYERS = {"map_beat": (3, None), **{n: (4, (k * 50, 50)) for k, n in enumerate(TIMING_BANDS)}, **{n: (5, (k * 25, 25)) for k, n in enumerate(MORPH_BANDS)},
                 **{n: (6, (k * 50, 50)) for k, n in enumerate(SQI_BANDS)}}


def encode_compact(batch):
    """Per window: planes 0-2 as float16 bytes, the (60 x 17) per-beat scalar matrix as float32 bytes, masks, context. Planes 3-6 are exact broadcasts of the scalars."""
    x = batch["x"]
    n = len(x)
    scal = np.stack([x[:, 3, :, 0]] + [x[:, p, :, s[0]] for name, (p, s) in SCALAR_LAYERS.items() if name != "map_beat"], axis=2).astype(np.float32)
    return {"window_id": batch["window_id"], "dataset_code": batch["dataset_code"],
            "waveforms_f16": [x[i, :3].astype(np.float16).tobytes() for i in range(n)], "scalars_f32": [scal[i].tobytes() for i in range(n)],
            "beat_mask_u8": [batch["beat_mask"][i].astype(np.uint8).tobytes() for i in range(n)], "context_f32": [batch["context"][i].astype(np.float32).tobytes() for i in range(n)],
            "variant_mask_f32": [batch["variant_mask"][i].astype(np.float32).tobytes() for i in range(n)], **{c: batch[c] for c in LABEL_OUT}}


def decode_compact(rows):
    """Inverse of ``encode_compact`` for a dict of lists/arrays (e.g. a pandas frame converted with to_dict('list')): returns the batch dict (planes 0-2 are float16-precise)."""
    n = len(rows["window_id"])
    x = np.zeros((n, 7, N_BEATS, PHASE), dtype=np.float32)
    ctx = np.stack([np.frombuffer(b, dtype=np.float32) for b in rows["context_f32"]])
    for i in range(n):
        x[i, :3] = np.frombuffer(rows["waveforms_f16"][i], dtype=np.float16).reshape(3, N_BEATS, PHASE)
        scal = np.frombuffer(rows["scalars_f32"][i], dtype=np.float32).reshape(N_BEATS, len(SCALAR_ORDER))
        x[i, 3] = scal[:, 0:1]
        for j, name in enumerate(SCALAR_ORDER[1:], start=1):
            plane, (start, width) = SCALAR_LAYERS[name]
            x[i, plane, :, start:start + width] = scal[:, j:j + 1]
    out = {"x": x, "context": ctx, "window_id": np.asarray(rows["window_id"], dtype=np.int64), "dataset_code": np.asarray(rows["dataset_code"], dtype=np.int8),
           "beat_mask": np.stack([np.frombuffer(b, dtype=np.uint8).reshape(3, N_BEATS).astype(bool) for b in rows["beat_mask_u8"]]),
           "variant_mask": np.stack([np.frombuffer(b, dtype=np.float32) for b in rows["variant_mask_f32"]])}
    for c in LABEL_OUT:
        out[c] = np.asarray(rows[c], dtype=np.float32)
    return out
