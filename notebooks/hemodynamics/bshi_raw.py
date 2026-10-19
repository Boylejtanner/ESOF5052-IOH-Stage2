"""Raw 100 Hz windows (ECG, ART, PPG) for the proposal's raw-multimodal (TCN) and spectral (STFT/CWT) baselines.

A window is the 90 s ending at ``t_end_sec`` of a window-index row: (3, 9000) float32 in the channel order ECG, ART, PPG plus a (3, 9000) validity mask (samples that are
missing or non-finite are 0 in ``x`` and False in ``mask``; nothing is interpolated). ART is in mmHg; ECG and PPG are in device-native units, so ``normalize_raw`` applies the
same normalization as BSHI (ART fixed scale, ECG/PPG per-window robust z-score over valid samples).
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .bshi import bshi_config

FS = 100
CHANNELS = ("ecg", "art", "ppg")
_COLUMNS = {"ecg": "ECG", "art": "ART", "ppg": "PPG"}


def waveform_path(root, dataset, case_id):
    return Path(root) / "data" / "parquet" / str(dataset) / f"case_id={case_id}" / "waveform.parquet"


class RawCase:
    """Dense (NaN-filled) 100 Hz arrays of one case on its own clock."""
    __slots__ = ("t0", "n", "data")

    def __init__(self, root, dataset, case_id):
        t = pq.read_table(waveform_path(root, dataset, case_id), columns=["time_sec", "ECG", "ART", "PPG"]).to_pandas()
        ts = t.time_sec.to_numpy(dtype=np.float64)
        self.t0 = float(ts[0])
        self.n = int(round((ts[-1] - self.t0) * FS)) + 1
        pos = np.rint((ts - self.t0) * FS).astype(np.int64)
        self.data = np.full((3, self.n), np.nan, dtype=np.float32)
        for i, c in enumerate(CHANNELS):
            self.data[i, pos] = t[_COLUMNS[c]].to_numpy(dtype=np.float32)


class RawCache:
    def __init__(self, maxsize=4):
        self.maxsize, self.items = maxsize, OrderedDict()

    def get(self, root, dataset, case_id):
        key = (str(root), dataset, str(case_id))
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        item = RawCase(root, dataset, case_id)
        self.items[key] = item
        while len(self.items) > self.maxsize:
            self.items.popitem(last=False)
        return item


_DEFAULT = RawCache()


def raw_windows(root, rows, window_sec=90, cache=None):
    """(x, mask) for index rows (needs dataset, case_id, t_end_sec): x (N,3,window_sec*100) float32 with 0 where invalid, mask bool."""
    cache = cache or _DEFAULT
    rows = rows.reset_index(drop=True)
    length = int(window_sec * FS)
    n = len(rows)
    x = np.zeros((n, 3, length), dtype=np.float32)
    mask = np.zeros((n, 3, length), dtype=bool)
    for (ds, cid), g in rows.groupby(["dataset", "case_id"], sort=False):
        case = cache.get(root, ds, cid)
        for i, t_end in zip(g.index.to_numpy(), g.t_end_sec.to_numpy(dtype=np.float64)):
            start = int(round((t_end - window_sec - case.t0) * FS))
            lo, hi = max(start, 0), min(start + length, case.n)
            if hi > lo:
                seg = case.data[:, lo:hi]
                ok = np.isfinite(seg)
                x[i, :, lo - start:hi - start] = np.where(ok, seg, 0.0)
                mask[i, :, lo - start:hi - start] = ok
    return x, mask


def normalize_raw(x, mask, cfg_or_root=None, clip=8.0):
    """BSHI-compatible normalization of raw windows: ART (mmHg - center) / scale, ECG and PPG robust z-score over the valid samples of the window (nearest-rank quantiles), clipped."""
    cfg = bshi_config(cfg_or_root) if cfg_or_root is not None else {"art_center_mmhg": 80, "art_scale_mmhg": 40}
    out = np.zeros_like(x)
    out[:, 1] = np.where(mask[:, 1], np.clip((x[:, 1] - cfg["art_center_mmhg"]) / cfg["art_scale_mmhg"], -clip, clip), 0.0)
    for c in (0, 2):
        for i in range(len(x)):
            v = x[i, c][mask[i, c]]
            if v.size < 10:
                continue
            s = np.sort(v)
            q = lambda p: s[int(np.floor((len(s) - 1) * p + 0.5))]
            scale = q(0.75) - q(0.25)
            scale = scale if scale > 1e-6 else 1.0
            out[i, c] = np.where(mask[i, c], np.clip((x[i, c] - q(0.5)) / scale, -clip, clip), 0.0)
    return out
