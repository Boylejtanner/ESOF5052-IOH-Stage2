"""Machine probe: run on EACH machine, writes ``probe_<hostname>.json``; with ``--rank/--master`` it also measures the Gloo link between the two machines.

    python -m hemodynamics.cluster_probe --out results/probe                       (single machine)
    python -m hemodynamics.cluster_probe --rank 0 --world-size 2 --master tcp://<desktop_ip>:29500 --out results/probe     (desktop; start rank 1 on the laptop the same way)

It measures: CUDA / driver / free VRAM, bf16 matmul TFLOPs, training-step windows/s and peak VRAM per batch size (synthetic data), host-to-device copy speed, local read speed of the sweep
subset, the DataLoader worker count that maximizes windows/s, what this machine's hostname resolves to (Gloo needs it to be the LAN address), and (two ranks) all-reduce / broadcast
time for 1, 4, 16 and 64 MB tensors in both directions plus the round-trip latency. ``scaling_estimate`` combines the two probe files.
"""
from __future__ import annotations
# Measure a machine and its link before assigning distributed workloads.

import argparse
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

BATCH_SIZES = (16, 32, 64, 96, 128, 192, 256, 384)


def _smi(query):
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        return [v.strip() for v in out.split(",")]
    except Exception:
        return None


def machine_info():
    import platform
    import psutil
    info = {"hostname": socket.gethostname(), "platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()), "cpu_logical": os.cpu_count(), "cpu_physical": psutil.cpu_count(logical=False), "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
    try:
        info["hostname_resolves_to"] = socket.gethostbyname(socket.gethostname())
    except OSError as exc:
        info["hostname_resolves_to"] = f"unresolved: {exc}"
    try:
        from .cluster import lan_ipv4
        info["lan_ip"] = lan_ipv4()
        info["gloo_hostname_ok"] = info["hostname_resolves_to"] == info["lan_ip"]
        from .bshi_ddp import lan_interface
        info["gloo_iface"] = lan_interface(info["lan_ip"])        # the adapter bshi_ddp pins through GLOO_SOCKET_IFNAME
    except Exception as exc:
        info["lan_ip"] = f"unknown: {exc}"[:80]
    smi = _smi("name,driver_version,memory.total,memory.used,memory.free")
    if smi:
        info.update(gpu=smi[0], driver=smi[1], vram_total_gb=round(float(smi[2]) / 1024, 2), vram_used_gb=round(float(smi[3]) / 1024, 2), vram_free_gb=round(float(smi[4]) / 1024, 2))
    return info


def matmul_tflops(n=4096, iters=20):
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return round(2 * n ** 3 * iters / (time.perf_counter() - t0) / 1e12, 1)


def step_throughput(free_gb):
    """windows/s and peak VRAM per batch size (bf16, channels_last off), up to the first batch that does not fit in the free VRAM; ``max_micro_batch`` keeps 10% headroom."""
    from . import bshi_train as T
    from .bshi_model import ModelConfig
    rows, best = [], None
    for bs in BATCH_SIZES:
        r = T.benchmark_step(ModelConfig(), bs, bf16=True, channels_last=False, iters=6, warmup=3)
        rows.append(r)
        if r["oom"] or (free_gb and r["peak_vram_gb"] > 0.95 * free_gb):
            break
    ok = [r for r in rows if not r["oom"]]
    fits = [r for r in ok if not free_gb or r["peak_vram_gb"] <= 0.9 * free_gb]
    best = max(fits or ok, key=lambda r: r["windows_per_s"])
    return {"rows": rows, "best": best, "max_micro_batch": max(r["batch_size"] for r in (fits or ok))}


def h2d_gb_s(mb=256, repeats=5):
    # Wait for GPU work to finish before measuring the elapsed time.
    x = torch.empty(mb * 1024 * 256, dtype=torch.float32).pin_memory()
    y = torch.empty_like(x, device="cuda")
    y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize()
    return round(repeats * x.numel() * 4 / 1e9 / (time.perf_counter() - t0), 2)


def data_speed(store, max_workers=16):
    """Single-process read speed of the compact store and the DataLoader worker count that maximizes windows/s."""
    from . import bshi_data as D
    from . import bshi_store as ST
    st = ST.Store(store, "validation")
    t0 = time.perf_counter()
    n = 0
    for b in st.batches(256, shuffle=True, seed=1, limit=2048):
        n += len(b["window_id"])
    single = n / (time.perf_counter() - t0)
    rows = []
    cap = max(1, min(max_workers, (os.cpu_count() or 4)))
    for w in sorted({0, 2, 4, 8, cap} & set(range(0, cap + 1))):
        ds = D.StoreIterable(store, "validation", 256, seed=2, limit=4096)
        dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=w, persistent_workers=False, pin_memory=torch.cuda.is_available())
        t0, n = time.perf_counter(), 0
        for b in dl:
            n += len(b["window_id"])
        rows.append({"workers": w, "windows_per_s": round(n / (time.perf_counter() - t0), 1)})
    best = max(rows, key=lambda r: r["windows_per_s"])
    return {"single_process_windows_per_s": round(single, 1), "dataloader": rows, "best_workers": best["workers"], "best_windows_per_s": best["windows_per_s"], "store": str(store)}


def comm_probe(rank, world, master, sizes_mb=(1, 4, 16, 64), repeats=4, timeout_s=120):
    """Gloo link between the two machines: for each size the median seconds of an all-reduce (sum) and of a broadcast from each rank, and the round-trip latency of a 4-byte all-reduce."""
    # Reserve output arrays and masks with explicit initial values.
    import torch.distributed as dist
    from .bshi_ddp import guarded, init_group
    init_group(rank, world, master, timeout_s)
    out = {"world_size": world, "rank": rank, "sizes": []}
    lat = []
    tiny = torch.zeros(1)
    for _ in range(30):
        t0 = time.perf_counter()
        guarded(lambda: dist.all_reduce(tiny), "latency probe")
        lat.append(time.perf_counter() - t0)
    out["latency_ms"] = round(statistics.median(lat[5:]) * 1000, 3)
    # Process the selected records in order and retain their individual results.
    for mb in sizes_mb:
        x = torch.randn(mb * 1024 * 256)
        row = {"mb": mb}
        for name, fn in (("allreduce_s", lambda: dist.all_reduce(x)), ("broadcast_from_0_s", lambda: dist.broadcast(x, src=0)), ("broadcast_from_1_s", lambda: dist.broadcast(x, src=1))):
            times = []
            for _ in range(repeats):
                guarded(lambda: dist.barrier(), "barrier")
                t0 = time.perf_counter()
                guarded(fn, name)
                times.append(time.perf_counter() - t0)
            row[name] = round(statistics.median(times), 4)
        row["allreduce_mb_s"] = round(mb / row["allreduce_s"], 1)
        out["sizes"].append(row)
    dist.barrier()
    dist.destroy_process_group()
    return out


def fit_link(comm):
    """(bandwidth MB/s, latency s) of the link from a ``comm_probe`` result: least squares of all-reduce time = latency + MB / bandwidth."""
    mb = np.array([r["mb"] for r in comm["sizes"]], dtype=float)
    t = np.array([r["allreduce_s"] for r in comm["sizes"]], dtype=float)
    slope, icpt = np.polyfit(mb, t, 1)
    return float(1.0 / max(slope, 1e-9)), float(max(icpt, comm["latency_ms"] / 1000.0))


def run_probe(store=None, out_dir=".", comm=None, quick=False, log=print):
    res = {"info": machine_info(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    if torch.cuda.is_available():
        res["bf16_tflops"] = matmul_tflops()
        res["step"] = step_throughput(res["info"].get("vram_free_gb"))
        res["h2d_gb_s"] = h2d_gb_s()
    else:
        res["note"] = "CUDA not available: GPU measurements skipped"
    if store and (Path(store) / "validation").exists() and not quick:
        res["data"] = data_speed(store)
    if comm:
        res["comm"] = comm
        res["link"] = dict(zip(("bandwidth_mb_s", "latency_s"), fit_link(comm)))
    p = Path(out_dir) / f"probe_{res['info']['hostname']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    if log:
        i = res["info"]
        log(f"{i['hostname']}: {i.get('gpu', 'no GPU')} | free VRAM {i.get('vram_free_gb')} GB | best step {res.get('step', {}).get('best', {}).get('windows_per_s')} windows/s at batch "
            f"{res.get('step', {}).get('best', {}).get('batch_size')} | hostname resolves to {i['hostname_resolves_to']} (LAN {i.get('lan_ip')}, Gloo ok: {i.get('gloo_hostname_ok')}) -> {p}")
    return res


def main(argv=None):
    # Read command-line options instead of hard-coding this run's settings.
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="")
    ap.add_argument("--out", default="results/probe")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rank", type=int, default=-1)
    ap.add_argument("--world-size", type=int, default=2)
    ap.add_argument("--master", default="")
    # Read command-line options instead of hard-coding this run's settings.
    ap.add_argument("--timeout", type=int, default=120)
    a = ap.parse_args(argv)
    comm = comm_probe(a.rank, a.world_size, a.master, timeout_s=a.timeout) if a.rank >= 0 and a.master else None
    run_probe(a.store or None, a.out, comm, a.quick)


if __name__ == "__main__":
    main()
