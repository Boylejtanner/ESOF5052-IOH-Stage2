"""Two-node data-parallel training over Gloo. NCCL does not exist on Windows; the launcher is our own (no torchrun).

    python -m hemodynamics.bshi_ddp --rank 0 --world-size 2 --master tcp://<desktop_ip>:29500 --store data/derived/bshi/sweep_subset --global-batch 256 --speeds 2500,900 --accum 4 ...

Design:
* EXACT gradient: every rank sums its per-window losses and divides by the GLOBAL normalizers (windows in the optimizer step, windows with an observed delta MAP, with an observed AUT65,
  with AUT65 > 0; exchanged once per optimizer step from the labels alone). The SUM of the ranks' gradients is then the gradient of one big batch holding all micro-batches, whatever the
  per-rank micro-batch sizes are (they are proportional to the measured speed of each GPU, ``split_micro_batches``).
* ``accum`` micro-steps per optimizer step: gradients accumulate locally and are all-reduced ONCE per optimizer step as one flat buffer (optionally compressed to float16 on the wire).
* Patient-disjoint data shards per rank in proportion to speed (``shard_patients``) or each rank's whole local data (``--shard none``).
* Rank 0 validates, decides early stopping / best weights and broadcasts the decision; checkpoints are written by rank 0 and on resume broadcast to the other rank (no shared disk).
* Every collective has a timeout; a vanished peer raises ``PeerLost`` with what to check instead of hanging.
"""
from __future__ import annotations
# Coordinate training processes and share gradients without changing patient partitions.

import argparse
import datetime
import json
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist

from . import bshi_data as D
from . import bshi_train as T
from .bshi_model import ModelConfig


class PeerLost(RuntimeError):
    pass


# network interface
def lan_interface(host):
    """Name of the network adapter that carries the traffic to ``host`` (the adapter that owns the address when ``host`` is this machine); ``None`` for loopback or when unknown.
    Gloo on Windows picks its adapter from what the computer name resolves to, which can be a Hyper-V / Docker address that the other machine cannot reach: ``GLOO_SOCKET_IFNAME`` pins it."""
    try:
        if host in ("localhost", "") or host.startswith("127."):
            return None
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect((host, 9))
            ip = s.getsockname()[0]
        if ip.startswith("127."):
            return None
        import psutil
        for name, addrs in psutil.net_if_addrs().items():
            if any(a.family == socket.AF_INET and a.address == ip for a in addrs):
                return name
    except Exception:
        return None
    return None


def pin_gloo_interface(master, environ=None):
    """Set ``GLOO_SOCKET_IFNAME`` (this process only) to the adapter that reaches the rank-0 address, unless the user already set it. Returns the name or ``None``."""
    environ = os.environ if environ is None else environ
    if environ.get("GLOO_SOCKET_IFNAME"):
        return environ["GLOO_SOCKET_IFNAME"]
    name = lan_interface(master.replace("tcp://", "").rsplit(":", 1)[0])
    if name:
        environ["GLOO_SOCKET_IFNAME"] = name
    return name


# pure helpers
def split_micro_batches(total, speeds):
    """Per-rank micro-batch sizes summing to ``total``, proportional to ``speeds`` (windows/s), each at least 1."""
    s = np.asarray(speeds, dtype=float)
    raw = total * s / s.sum()
    out = np.maximum(1, np.floor(raw).astype(int))
    while out.sum() < total:
        out[int(np.argmax(raw - out))] += 1
    while out.sum() > total:
        out[int(np.argmax(out))] -= 1
    return out.tolist()


def shard_patients(patient_counts, weights, seed=5052):
    """Assign whole patients to ranks so each rank's window count is proportional to its weight. ``patient_counts``: Series (patient -> windows). Returns {patient: rank}."""
    # Seed the generated values so this check can be repeated.
    rng = np.random.default_rng(seed)
    pats = list(patient_counts.index)
    rng.shuffle(pats)
    pats.sort(key=lambda p: -patient_counts[p])                           # big patients first, then the least-filled rank (relative to its target)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    load = np.zeros(len(w))
    out = {}
    for p in pats:
        r = int(np.argmin((load + patient_counts[p]) / w))
        out[p] = r
        load[r] += patient_counts[p]
    return out


def loss_sums(out, t, pos_weight=None):
    """Per-window loss SUMS and the counts they are normalized by (mirror of ``bshi_train.multitask_loss``, which uses means)."""
    import torch.nn.functional as F
    pw = None if pos_weight is None else torch.tensor(float(pos_weight), device=out["event"].device)
    return {"event": F.binary_cross_entropy_with_logits(out["event"], t["y_event"], pos_weight=pw, reduction="sum"),
            "delta": (F.huber_loss(out["delta_map"], t["delta"], delta=1.0, reduction="none") * t["m_delta"].float()).sum(),
            "pos": (F.binary_cross_entropy_with_logits(out["aut_logit"], t["aut_pos"], reduction="none") * t["m_aut"].float()).sum(),
            "mag": (F.huber_loss(out["aut_mag"], t["aut_mag"], delta=1.0, reduction="none") * t["m_mag"].float()).sum()}


def batch_counts(t):
    """[windows, observed delta, observed AUT65, AUT65 > 0] of a prepared target dict."""
    return torch.stack([torch.tensor(float(len(t["y_event"])), device=t["y_event"].device), t["m_delta"].sum().float(), t["m_aut"].sum().float(), t["m_mag"].sum().float()])


def local_loss(sums, counts, weights):
    n, nd, na, nm = (counts[i].clamp(min=1.0) for i in range(4))
    return weights["event"] * sums["event"] / n + weights["delta_map"] * sums["delta"] / nd + weights["aut65"] * (sums["pos"] / na + sums["mag"] / nm)


# process group
def init_group(rank, world_size, master, timeout_s=120, log=print):
    """``master``: ``tcp://host:port``. Raises ``PeerLost`` with a checklist when the other rank never shows up."""
    host_port = master.replace("tcp://", "")
    iface = pin_gloo_interface(master)
    try:
        dist.init_process_group("gloo", init_method=master, rank=rank, world_size=world_size, timeout=datetime.timedelta(seconds=timeout_s))
    except Exception as exc:
        raise PeerLost(f"could not start the group at {master} within {timeout_s} s ({type(exc).__name__}: {str(exc)[:160]}). Check: both ranks use the SAME --master address and "
                       f"--world-size {world_size}; rank 0 runs on {host_port.split(':')[0]}; port {host_port.split(':')[-1]} is open (desktop_admin_setup.ps1); both machines are on the "
                       "same network (Wi-Fi vs the Hyper-V 192.168.64.x adapter); the other rank was started.") from exc
    if rank == 0:
        log(f"group ready: {world_size} ranks via {master} (gloo, adapter {iface or 'default'}), host {socket.gethostname()}")


def guarded(fn, what):
    try:
        return fn()
    except RuntimeError as exc:
        raise PeerLost(f"lost the other rank during '{what}' ({str(exc)[:160]}). Restart both ranks with the same command: training resumes from the last epoch checkpoint on rank 0.") from exc


def allreduce_grads(params, compress="none"):
    """Sum the gradients of all ranks (in place) with ONE all-reduce of a flat buffer; ``compress='fp16'`` halves the bytes on the wire (summation error about 1e-3 relative)."""
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat = torch.cat([g.reshape(-1) for g in grads]).cpu()
    if compress == "fp16":
        flat = flat.to(torch.float16)
    guarded(lambda: dist.all_reduce(flat, op=dist.ReduceOp.SUM), "gradient all-reduce")
    flat = flat.to(torch.float32).to(grads[0].device)
    off = 0
    for p, g in zip(params, grads):
        n = g.numel()
        p.grad = flat[off:off + n].view_as(g).clone()
        off += n


# trainer
class DDPTrainer(T.Trainer):
    """``train_loader(epoch)`` yields THIS rank's micro-batches (CPU dicts); every rank must yield at least ``steps_per_epoch * accum`` of them."""

    def __init__(self, cfg, model_cfg, pre, train_loader, val_loader, out_dir, rank, world_size, accum=1, compress="none", device="cuda", steps_per_epoch=None, shift=None, log=print):
        assert cfg.ema == 0, "EMA is not supported with the DDP trainer"
        super().__init__(cfg, model_cfg, pre, train_loader, val_loader, out_dir, device=device, steps_per_epoch=steps_per_epoch, shift=shift, log=log if rank == 0 else (lambda *a: None))
        self.rank, self.world, self.accum, self.compress = rank, world_size, accum, compress
        self.sync_initial_weights()

    def sync_initial_weights(self):
        """All ranks start from rank 0's weights (the same seed already gives this; the broadcast makes it certain, also across machines with different library versions)."""
        state = [self.model.state_dict()] if self.rank == 0 else [None]
        guarded(lambda: dist.broadcast_object_list(state, src=0), "initial weight broadcast")
        if self.rank != 0:
            self.model.load_state_dict({k: v.to(self.device) for k, v in state[0].items()})

    def _train_epoch(self, epoch):
        # Track each batch update and aggregate losses over the epoch.
        cfg, total = self.cfg, self.cfg.epochs * self.steps_per_epoch
        self.model.train()
        it = iter(self.train_loader(epoch) if callable(self.train_loader) else self.train_loader)
        params = [p for p in self.model.parameters()]
        sums = {k: 0.0 for k in ("total", "event", "delta_map", "aut_pos", "aut_mag")}
        n_win, steps, lr, t0, comm = 0, 0, float("nan"), time.perf_counter(), 0.0
        prog, g_win = self.progress, 0.0
        if prog is not None:
            prog.begin_epoch(epoch)
        # Accumulate the calculation over the required samples or iterations.
        for _ in range(self.steps_per_epoch):
            micro = [next(it) for _ in range(self.accum)]
            tgts = [self.pre.targets(b) for b in micro]
            counts = sum(batch_counts(t) for t in tgts)
            c = counts.cpu()
            guarded(lambda: dist.all_reduce(c, op=dist.ReduceOp.SUM), "count all-reduce")
            counts = c.to(self.device)
            lr = T.lr_at(cfg, self.step, total, self.steps_per_epoch)
            for g in self.opt.param_groups:
                g["lr"] = lr
            local_parts = {k: 0.0 for k in ("event", "delta", "pos", "mag")}
            # Process the selected records in order and retain their individual results.
            for b, t in zip(micro, tgts):
                inp = self.pre.inputs(b)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=cfg.amp == "bf16"):
                    out = self.model(inp["x"], inp["context"], inp["beat_mask"])
                out = {k: v.float() for k, v in out.items()}
                s = loss_sums(out, t, cfg.pos_weight)
                loss = local_loss(s, counts, cfg.loss_weights)
                loss.backward()
                for k in local_parts:
                    local_parts[k] += float(s[k].detach())
                n_win += len(t["y_event"])
            # Limit the gradient norm before applying the optimizer update.
            t1 = time.perf_counter()
            allreduce_grads(params, self.compress)
            comm += time.perf_counter() - t1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip if cfg.grad_clip > 0 else float("inf"))
            # Calculate the observation count and the accumulated totals for the following step.
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            n, nd, na, nm = (float(counts[i]) for i in range(4))
            sums["event"] += local_parts["event"] / max(n, 1)
            # Calculate the accumulated totals for the following step.
            sums["delta_map"] += local_parts["delta"] / max(nd, 1)
            sums["aut_pos"] += local_parts["pos"] / max(na, 1)
            sums["aut_mag"] += local_parts["mag"] / max(nm, 1)
            steps += 1
            self.step += 1
            g_win += n
            if prog is not None and prog.due():
                w = cfg.loss_weights
                tot = (w["event"] * sums["event"] + w["delta_map"] * sums["delta_map"] + w["aut65"] * (sums["aut_pos"] + sums["aut_mag"])) / steps
                prog.update(steps, g_win, {"total": tot, "event": sums["event"] / steps}, lr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        # Calculate the output record for the following step.
        secs = time.perf_counter() - t0
        row = {k: v / max(1, steps) for k, v in sums.items() if k != "total"}
        row["total"] = float(cfg.loss_weights["event"] * row["event"] + cfg.loss_weights["delta_map"] * row["delta_map"] + cfg.loss_weights["aut65"] * (row["aut_pos"] + row["aut_mag"]))
        row.update(lr=lr, grad_norm=float("nan"), steps=steps, windows=n_win, train_seconds=round(secs, 2), train_windows_per_s=round(n_win / max(secs, 1e-9), 1), comm_seconds=round(comm, 2),
                   comm_share=round(comm / max(secs, 1e-9), 3))
        return row

    def _broadcast_state(self, state):
        box = [state]
        guarded(lambda: dist.broadcast_object_list(box, src=0), "checkpoint broadcast")
        return box[0]

    def fit(self, resume=True, until_epoch=None):
        cfg = self.cfg
        if resume:
            has = [bool(self.rank == 0 and (self.out_dir / "last.pt").exists())]
            guarded(lambda: dist.broadcast_object_list(has, src=0), "resume check")
            if has[0]:
                state = torch.load(self.out_dir / "last.pt", map_location="cpu", weights_only=False) if self.rank == 0 else None
                state = self._broadcast_state(state)
                self._load_state(state)
                self.log(f"resumed from epoch {self.epoch0}")
        stopped, t_all = False, time.perf_counter()
        prog = self.progress
        if prog is not None:
            prog.start_run(cfg.epochs, self.steps_per_epoch, getattr(self, "global_windows_per_step", cfg.batch_size), self.epoch0,
                           {"mode": f"ddp, {self.world} ranks", "optimizer": cfg.optimizer, "lr": cfg.lr, "accum": self.accum, "compress": self.compress})
        # Repeat training and validation for each epoch until the stopping rule is met.
        for epoch in range(self.epoch0, min(cfg.epochs, until_epoch or cfg.epochs)):
            row = {"epoch": epoch, **self._train_epoch(epoch)}
            decision = torch.zeros(3)
            if self.rank == 0:
                val = self.validate()
                row.update({"val_" + k: v for k, v in val.items()})
                improved, stop = self.stopper.update(val.get("auprc", float("nan")), epoch)
                decision = torch.tensor([float(improved), float(stop), float(self.stopper.best if self.stopper.best is not None else float("nan"))])
            guarded(lambda: dist.broadcast(decision, src=0), "early-stopping decision")
            improved, stop = bool(decision[0]), bool(decision[1])
            if self.rank != 0:
                self.stopper.best, self.stopper.best_epoch = (float(decision[2]) if improved else self.stopper.best), (epoch if improved else self.stopper.best_epoch)
            if improved:
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if self.rank == 0:
                    torch.save({"model": self.best_state, "epoch": epoch, "metric": self.stopper.best, "model_cfg": self.model_cfg.to_dict(), "cfg": cfg.to_dict()}, self.out_dir / "best.pt")
            row["improved"], row["best_auprc"] = improved, self.stopper.best
            self.history.append(row)
            self.epoch0 = epoch + 1
            if prog is not None:
                prog.epoch_done(row)
            if self.rank == 0:
                # Save the comparison table so the same measurements can be inspected later.
                pd.DataFrame(self.history).to_csv(self.out_dir / "epochs.csv", index=False)
                self.save_checkpoint()
                self.log(f"epoch {epoch}: loss {row['total']:.4f} | val AUPRC {row.get('val_auprc', float('nan')):.4f} | {row['train_windows_per_s']:.0f} windows/s (this rank) | "
                         f"comm {100 * row['comm_share']:.0f}% of the epoch")
            if stop:
                stopped = True
                break
        if self.best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in self.best_state.items()})
        if prog is not None:
            prog.finish({"best_epoch": self.stopper.best_epoch, "best_auprc": self.stopper.best, "stopped_early": stopped})
        return {"history": pd.DataFrame(self.history), "best_epoch": self.stopper.best_epoch, "best_auprc": self.stopper.best, "stopped_early": stopped, "seconds": round(time.perf_counter() - t_all, 1)}

    def _load_state(self, s):
        # Restore the saved model or optimizer state for this run.
        self.model.load_state_dict({k: v.to(self.device) for k, v in s["model"].items()})
        self.opt.load_state_dict(s["opt"])
        self.epoch0, self.step, self.history, self.best_state = s["epoch"], s["step"], s["history"], s["best_state"]
        self.stopper.load(s["stopper"])


# launcher
def main(argv=None):
    ap = argparse.ArgumentParser(description="Two-node data-parallel BSHI training (Gloo).")
    ap.add_argument("--rank", type=int, required=True)
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--master", required=True, help="tcp://<rank-0 host>:29500")
    ap.add_argument("--store", default="data/derived/bshi/sweep_subset")
    ap.add_argument("--root", default=os.environ.get("BSHI_ROOT", "."))
    ap.add_argument("--global-batch", type=int, default=256, help="windows per micro-step summed over all ranks")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--speeds", default="1,1", help="measured windows/s of each rank, e.g. 2500,900 (sets the micro-batch split and the data shares)")
    ap.add_argument("--accum", type=int, default=4, help="micro-steps per optimizer step (one all-reduce each)")
    ap.add_argument("--compress", choices=("none", "fp16"), default="none")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--shard", choices=("auto", "none"), default="auto")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--epoch-windows", type=int, default=0, help="windows per epoch summed over ranks (0 = every training window once)")
    ap.add_argument("--config", default="", help="JSON file with a 'train' block (defaults to configs/hemodynamics.json)")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--set", nargs="*", default=[], help="overrides like lr=0.002 optimizer=sgd")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    cfg_file = Path(a.config) if a.config else root / "configs" / "hemodynamics.json"
    over = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        over[k] = json.loads(v) if v[:1] in "[{0123456789-" or v in ("true", "false") else v
    # Load the recorded settings or metadata for this processing step.
    tcfg = T.TrainConfig.from_dict(json.loads(cfg_file.read_text(encoding="utf-8")).get("train", {}), epochs=a.epochs, num_workers=0, **over)
    out = Path(a.out or root / "runs" / f"ddp_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    init_group(a.rank, a.world_size, a.master, a.timeout)
    from . import bshi_store as ST
    speeds = [float(s) for s in a.speeds.split(",")]
    micro = split_micro_batches(a.global_batch, speeds)[a.rank]
    store = Path(a.store) if Path(a.store).is_absolute() else root / a.store
    tr, va = ST.Store(store, "train"), ST.Store(store, "validation")
    if a.shard == "auto":
        owner = shard_patients(tr.meta.groupby("patient_key").size(), speeds)
        mine = np.flatnonzero(tr.meta.patient_key.map(owner).to_numpy() == a.rank)
    else:
        mine = np.arange(len(tr))
    n_local = torch.tensor([len(mine) // (micro * a.accum)], dtype=torch.int64)
    guarded(lambda: dist.all_reduce(n_local, op=dist.ReduceOp.MIN), "step-count agreement")
    spe = int(n_local) if not a.epoch_windows else max(1, min(int(n_local), a.epoch_windows // (a.global_batch * a.accum)))
    stats = D.load_stats(root)
    pre = D.Preprocessor(tcfg.variant, stats, clamp=tcfg.clamp, device=a.device, channels_last=tcfg.channels_last)

    def loader(epoch):
        order = mine[np.random.default_rng(tcfg.seed * 1000 + epoch + 17 * a.rank).permutation(len(mine))]
        return tr.batches(micro, shuffle=False, indices=order)
    trainer = DDPTrainer(tcfg, ModelConfig(variant=tcfg.variant, gate=tcfg.gate, dropout=tcfg.dropout), pre, loader, D.store_eval_loader(va, 512), out, a.rank, a.world_size, accum=a.accum,
                         compress=a.compress, device=a.device, steps_per_epoch=spe)
    if a.rank == 0:
        from . import bshi_progress as P
        trainer.progress = P.Progress(out / "live", label=f"ddp {a.world_size} ranks")
    trainer.global_windows_per_step = a.global_batch * a.accum
    if a.rank == 0:
        print(f"rank 0: micro-batch {micro} (all ranks {split_micro_batches(a.global_batch, speeds)}), accum {a.accum}, effective batch {a.global_batch * a.accum}, {spe} optimizer steps/epoch, "
              f"{len(mine)} local training windows, compress {a.compress}", flush=True)
    res = trainer.fit(resume=not a.no_resume)
    if a.rank == 0:
        # Store the settings and results as JSON for later inspection.
        (out / "result.json").write_text(json.dumps({"best_epoch": res["best_epoch"], "best_val_auprc": res["best_auprc"], "seconds": res["seconds"], "stopped_early": res["stopped_early"],
                                                     "windows_per_s_this_rank": float(res["history"].train_windows_per_s.median()), "comm_share": float(res["history"].comm_share.median()),
                                                     "micro_batches": split_micro_batches(a.global_batch, speeds), "accum": a.accum, "compress": a.compress}, indent=1), encoding="utf-8")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
