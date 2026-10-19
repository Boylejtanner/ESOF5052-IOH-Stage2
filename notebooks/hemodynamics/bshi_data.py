"""Training-side data for the BSHI network.

* ``Preprocessor``: model-side preprocessing on the GPU (clamp the z-scored scalar planes, standardize the context with the TRAIN statistics, apply the variant mask) and the training
  targets with their explicit loss masks (delta MAP, AUT65 two-part). The assembler is untouched.
* ``resample_index``: the recommended batch composition (share of MOVER, share of events inside each dataset) as a resampled index, ``prior_shift_by_dataset`` to get natural-prevalence
  probabilities back.
* ``frames_for``: the train / validation / test windows of the experiment (pooled, VitalDB-train -> all MOVER, MOVER-train -> VitalDB test). The test frames are returned but never used
  for selection by the trainer.
* loaders: the on-the-fly assembler through a torch DataLoader (persistent workers), the dev subset (smoke tests, the laptop), and the GPU decode of compact-store batches.
"""
from __future__ import annotations
# Prepare batches and normalize them using statistics from training patients.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import bshi as S
from .bshi_aug import clamp_scalar_planes
from .bshi_model import input_masks

CONTEXT_STATS = S.OUT_DIR / "context_stats_v1.json"
TARGET_STATS = S.OUT_DIR / "target_stats_v1.json"
DEV_DIR = S.OUT_DIR / "dev_subset"


def load_stats(root):
    root = Path(root)
    return {"context": json.loads((root / CONTEXT_STATS).read_text(encoding="utf-8")), "target": json.loads((root / TARGET_STATS).read_text(encoding="utf-8"))}


def ppg_free(variant):
    return "ppg" in S.VARIANTS[variant]


class Preprocessor:
    """Batch dict (cpu or gpu tensors from ``bshi.assemble`` / the compact store) -> model inputs and targets on ``device``."""

    def __init__(self, variant, stats, clamp=6.0, device="cuda", channels_last=True):
        self.variant, self.clamp, self.device, self.channels_last = variant, float(clamp), torch.device(device), channels_last
        cs = stats["context"]["ea_core" if ppg_free(variant) else "eap_full"]
        _, cmask = input_masks(variant)
        self.center = torch.tensor(cs["center"], dtype=torch.float32, device=self.device)
        self.scale = torch.tensor(cs["scale"], dtype=torch.float32, device=self.device)
        self.ctx_mask = cmask.to(self.device)
        t = stats["target"]
        self.delta_scale = float(t["delta_map"]["scale"])
        self.aut_mean, self.aut_std = float(t["aut65"]["log1p_mean_given_positive"]), float(t["aut65"]["log1p_std_given_positive"])

    def inputs(self, batch):
        """Model inputs only: (x, context, beat_mask)."""
        d = self.device
        if "x" in batch:
            x = batch["x"].to(d, non_blocking=True).float()
        else:                                                              # compact-store batch: planes 3-6 are exact broadcasts of the scalar matrix
            x = decode_compact_gpu(batch["waves"].to(d, non_blocking=True), batch["scal"].to(d, non_blocking=True))
        x = clamp_scalar_planes(x, self.clamp)
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        bm = batch["beat_mask"].to(d)
        if ppg_free(self.variant):                                        # what apply_variant does for PPG-free recipes (the PPG planes and bands are masked inside the model)
            bm = bm.clone()
            bm[:, 2] = False
        ctx = ((batch["context"].to(d, non_blocking=True).float() - self.center) / self.scale).clamp(-self.clamp, self.clamp) * self.ctx_mask
        return {"x": x, "context": ctx, "beat_mask": bm}

    def targets(self, batch):
        """Targets, loss masks and bookkeeping only (cheap: no image work); the masks give the loss normalizers of a batch."""
        # Calculate the working array and output values for the following step.
        d = self.device
        aut = batch["aut65"].to(d).float()
        delta = batch["delta_map"].to(d).float()
        reg = batch["reg_mask"].to(d).bool() if "reg_mask" in batch else torch.ones_like(aut, dtype=torch.bool)
        # Calculate output values for the following step.
        m_aut = reg & torch.isfinite(aut)
        out = {"y_event": batch["y_event"].to(d).float(),
               "delta": torch.nan_to_num(delta) / self.delta_scale, "m_delta": reg & torch.isfinite(delta),
               "aut_pos": (torch.nan_to_num(aut) > 0).float(), "m_aut": m_aut, "m_mag": m_aut & (torch.nan_to_num(aut) > 0),
               "aut_mag": torch.nan_to_num((torch.log1p(torch.nan_to_num(aut).clamp(min=0)) - self.aut_mean) / self.aut_std),
               "window_id": batch["window_id"], "dataset_code": batch["dataset_code"].to(d) if "dataset_code" in batch else None}
        out["w"] = batch["sampling_weight"].to(d).float() if "sampling_weight" in batch else None
        return out

    def __call__(self, batch):
        return {**self.inputs(batch), **self.targets(batch)}


# flags, sampling, frames
class WindowFlags:
    """Per-window loss-mask flag from the window index (``aut65_low_coverage``): attached to every batch as ``reg_mask`` (True = usable for the regression losses)."""

    def __init__(self, idx):
        # Order the measurements before matching intervals or comparing rows.
        order = np.argsort(idx.window_id.to_numpy())
        self.ids = idx.window_id.to_numpy()[order]
        self.ok = ~idx.aut65_low_coverage.to_numpy()[order]

    def attach(self, batch):
        pos = np.searchsorted(self.ids, np.asarray(batch["window_id"]))
        batch["reg_mask"] = torch.from_numpy(self.ok[pos])
        return batch


class FlaggedLoader:
    def __init__(self, loader, flags):
        self.loader, self.flags = loader, flags

    def __iter__(self):
        for b in self.loader:
            yield self.flags.attach(b)


def resample_positions(dataset, y_event, n, mover_share, event_share, seed):
    """Row positions of ``n`` windows drawn WITH replacement: ``mover_share`` of them from MOVER, ``event_share`` of each dataset's draw from event windows (uniform inside a stratum).
    A dataset that is absent gets share 0 and the other one all of it. ``dataset``: array of 'mover'/'vitaldb'; ``y_event``: bool array."""
    # Seed the generated values so this check can be repeated.
    rng = np.random.default_rng(seed)
    ds, ev = np.asarray(dataset), np.asarray(y_event).astype(bool)
    pools = {(d, e): np.flatnonzero((ds == d) & (ev == e)) for d in ("mover", "vitaldb") for e in (False, True)}
    have = {d: len(pools[(d, False)]) + len(pools[(d, True)]) > 0 for d in ("mover", "vitaldb")}
    share = {"mover": mover_share if have["mover"] and have["vitaldb"] else float(have["mover"]), "vitaldb": 0.0}
    share["vitaldb"] = 1.0 - share["mover"]
    picks = []
    for d in ("mover", "vitaldb"):
        nd = int(round(n * share[d]))
        if nd == 0:
            continue
        ne = int(round(nd * event_share)) if len(pools[(d, True)]) else 0
        for e, k in ((True, ne), (False, nd - ne)):
            if k and len(pools[(d, e)]):
                picks.append(rng.choice(pools[(d, e)], k, replace=True))
    pos = np.concatenate(picks)
    return pos[rng.permutation(len(pos))]


def resample_index(idx, n, mover_share, event_share, seed):
    """The recommended batch composition as a resampled index frame (see ``resample_positions``)."""
    return idx.iloc[resample_positions(idx.dataset.to_numpy(), idx.y_event.to_numpy(), n, mover_share, event_share, seed)].reset_index(drop=True)


def prior_shift_by_dataset(train_natural, event_share):
    """logit(natural prevalence) - logit(sampled event share) per dataset: ADD it to the event logit to get probabilities at natural prevalence."""
    out = {}
    for ds, g in train_natural.groupby("dataset"):
        p = float(g.y_event.mean())
        out[ds] = float(np.log(p / (1 - p)) - np.log(event_share / (1 - event_share)))
    return out


def frames_for(idx, train_on="pooled"):
    """train / validation / test frames of an experiment from a protocol-filtered index. Never touch ``tests`` while selecting anything."""
    ds = idx.dataset
    if train_on == "pooled":
        sp = idx.split_pooled
        return {"train": idx[sp == "train"], "val": idx[sp == "validation"],
                "tests": {"pooled_test": idx[sp == "test"], "vitaldb_test": idx[(sp == "test") & (ds == "vitaldb")], "mover_test": idx[(sp == "test") & (ds == "mover")]}}
    sp = idx.split_within_dataset
    if train_on == "vitaldb":
        return {"train": idx[(ds == "vitaldb") & (sp == "train")], "val": idx[(ds == "vitaldb") & (sp == "validation")],
                "tests": {"vitaldb_test": idx[(ds == "vitaldb") & (sp == "test")], "mover_all": idx[ds == "mover"]}}
    if train_on == "mover":
        return {"train": idx[(ds == "mover") & (sp == "train")], "val": idx[(ds == "mover") & (sp == "validation")],
                "tests": {"mover_test": idx[(ds == "mover") & (sp == "test")], "vitaldb_test": idx[(ds == "vitaldb") & (sp == "test")]}}
    raise ValueError(f"train_on must be pooled, vitaldb or mover, not {train_on!r}")


def protocol_frame(root, windows="eap_full", columns=None):
    """The primary-eligible windows of a protocol ('eap_full' = identical windows for every input recipe, 'ea_core', 'ea_full') with the columns the trainer needs."""
    cols = list(dict.fromkeys(list(S.TORCH_INDEX_COLUMNS) + ["patient_key", "split_pooled", "split_within_dataset", "aut65_low_coverage", "t_end_sec", "time_to_event_sec", "gray_zone",
                                                              "primary_eligible", "in_core", "in_full"] + list(columns or [])))
    return S.load_window_index(root, windows, columns=cols)


# loaders
def make_loader(root, index, variant="eap", batch_size=256, shuffle=True, seed=0, num_workers=16, flags=None):
    """On-the-fly assembler through a DataLoader with persistent workers (every new iteration is a new epoch with a new shuffle)."""
    ds = S.make_torch_dataset(root, index, variant, batch_size, shuffle, seed)
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=torch.cuda.is_available())
    if num_workers > 0:
        kw["prefetch_factor"] = 4
    dl = torch.utils.data.DataLoader(ds, batch_size=None, **kw)
    return FlaggedLoader(dl, flags) if flags is not None else dl


class DevSubsetBatches(torch.utils.data.IterableDataset):
    """The dev subset (parquet parts of compact windows) as ready batches, with ``sampling_weight``. Used by the smoke runs, tests and the laptop check (module level: spawn-safe)."""

    def __init__(self, root, split, variant="eap", batch_size=256, shuffle=True, seed=0, files_per_chunk=3):
        self.root, self.split, self.variant, self.batch_size, self.shuffle, self.seed, self.files_per_chunk = str(root), split, variant, batch_size, shuffle, seed, files_per_chunk
        self.files = sorted((Path(root) / DEV_DIR / f"split={split}").glob("part-*.parquet"))
        self._calls = 0

    def __iter__(self):
        # Seed the generated values so this check can be repeated.
        import pyarrow.parquet as pq
        info = torch.utils.data.get_worker_info()
        files = self.files[info.id::info.num_workers] if info else list(self.files)
        rng = np.random.default_rng(self.seed + self._calls)
        self._calls += 1
        if self.shuffle:
            files = [files[i] for i in rng.permutation(len(files))]
        cols = ["window_id", "dataset_code", "waveforms_f16", "scalars_f32", "beat_mask_u8", "context_f32", "variant_mask_f32"] + S.LABEL_OUT + ["sampling_weight"]
        # Accumulate the calculation over the required samples or iterations.
        for i in range(0, len(files), self.files_per_chunk):
            t = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files[i:i + self.files_per_chunk]], ignore_index=True)
            if self.shuffle:
                t = t.iloc[rng.permutation(len(t))].reset_index(drop=True)
            for j in range(0, len(t), self.batch_size):
                part = t.iloc[j:j + self.batch_size]
                b = S.decode_compact(part.to_dict("list"))
                if self.variant != "eap":
                    b = S.apply_variant(b, self.variant)
                b["sampling_weight"] = part.sampling_weight.to_numpy(dtype=np.float32)
                yield {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in b.items()}


def dev_loader(root, split, variant="eap", batch_size=256, shuffle=True, seed=0, num_workers=0, flags=None):
    ds = DevSubsetBatches(root, split, variant, batch_size, shuffle, seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=num_workers, persistent_workers=num_workers > 0)
    return FlaggedLoader(dl, flags) if flags is not None else dl


# compact store decode on the GPU
def store_frame(store):
    """Dataset name, event flag and patient of every window of a ``bshi_store.Store`` (for sampling and bootstrap)."""
    # Collect the per-record values into a table for comparison.
    y = np.concatenate([s["y_event"] for s in store.sc])
    code = np.concatenate([s["dataset_code"] for s in store.sc])
    return pd.DataFrame({"dataset": np.where(code == 1, "vitaldb", "mover"), "y_event": y > 0.5, "patient_key": store.meta.patient_key.to_numpy()})


def prefetch(it, depth=6):
    """Produce the batches of ``it`` in a background thread (memory-map reads release the GIL), so disk reads overlap with the GPU step. Closing the generator stops the thread."""
    # Read batches ahead on a background thread while forwarding its errors to the caller.
    import queue
    import threading
    q, stop, done = queue.Queue(depth), threading.Event(), object()

    def work():
        # Ensure the associated files, locks, or resources are released after this operation.
        try:
            for x in it:
                while not stop.is_set():
                    try:
                        q.put(x, timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    return
        except BaseException as exc:                      # re-raised in the consumer
            q.put(exc)
        finally:
            q.put(done)

    th = threading.Thread(target=work, daemon=True)
    th.start()
    # Ensure the associated files, locks, or resources are released after this operation.
    try:
        while True:
            x = q.get()
            if x is done:
                return
            if isinstance(x, BaseException):
                raise x
            yield x
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass


def store_train_loader(store, batch_size, epoch_windows=None, seed=0, sampler="natural", mover_share=0.15, event_share=0.30):
    """Callable ``epoch -> iterator`` of CPU batches. ``natural``: every epoch a new shuffle of (a random ``epoch_windows`` of) the windows; ``stratified``: ``epoch_windows`` windows drawn
    with the recommended composition (WITH replacement, a new draw per epoch)."""
    fr = store_frame(store) if sampler == "stratified" else None

    def loader(epoch):
        if sampler == "stratified":
            pos = resample_positions(fr.dataset.to_numpy(), fr.y_event.to_numpy(), epoch_windows or len(fr), mover_share, event_share, seed * 1000 + epoch)
            return prefetch(store.batches(batch_size, shuffle=False, indices=pos))
        return prefetch(store.batches(batch_size, shuffle=True, seed=seed * 1000 + epoch, limit=epoch_windows))
    return loader


def store_eval_loader(store, batch_size=512, indices=None):
    return lambda: prefetch(store.batches(batch_size, shuffle=False, indices=indices))


class StoreIterable(torch.utils.data.IterableDataset):
    """A split of a compact store as ready batches for a torch DataLoader: every worker takes a strided share of the (shuffled) window order and opens its own memory maps (module level:
    Windows spawn workers must be able to import it)."""

    def __init__(self, path, split, batch_size=256, seed=0, limit=None):
        self.path, self.split, self.batch_size, self.seed, self.limit, self._calls = str(path), split, batch_size, seed, limit, 0

    def __iter__(self):
        # Seed the generated values so this check can be repeated.
        from .bshi_store import Store
        store = Store(self.path, self.split)
        order = np.random.default_rng(self.seed + self._calls).permutation(len(store))
        self._calls += 1
        if self.limit:
            order = order[:self.limit]
        info = torch.utils.data.get_worker_info()
        if info:
            order = order[info.id::info.num_workers]
        yield from store.batches(self.batch_size, shuffle=False, indices=order)


def decode_compact_gpu(waves, scal):
    """Planes 0-2 (B,3,60,200) float16/32 + the per-beat scalar matrix (B,60,17) float32 (``bshi.SCALAR_ORDER``) -> the 7-plane tensor (B,7,60,200); planes 3-6 are exact broadcasts."""
    b = waves.shape[0]
    x = torch.empty((b, 7, S.N_BEATS, S.PHASE), dtype=torch.float32, device=waves.device)
    x[:, :3] = waves.float()
    x[:, 3] = scal[:, :, 0:1].expand(-1, -1, S.PHASE)
    for name, (plane, sl) in S.SCALAR_LAYERS.items():
        if name == "map_beat":
            continue
        j = S.SCALAR_ORDER.index(name)
        x[:, plane, :, sl[0]:sl[0] + sl[1]] = scal[:, :, j:j + 1].expand(-1, -1, sl[1])
    return x
