"""Estimate the throughput of distributed training from measured machine and network probes.

    python -m hemodynamics.scaling_estimate probe_desktop.json probe_laptop.json [--cable] [--out results/scaling_estimate.json]

Model: per optimizer step every rank runs ``accum`` micro-steps (compute time = windows / aggregate speed, micro-batches split in proportion to the speeds and limited by each GPU's VRAM),
then ONE all-reduce of the gradient buffer: ``2 x latency + bytes / bandwidth`` plus the two GPU<->host copies of the buffer. No overlap (one flat all-reduce after accumulation).
The effective batch (global micro-batch x accum) is capped (default 2048): larger batches change the optimization, not only the speed. Speed-up = DDP windows/s over the first machine alone.
Recommendation ``ddp_2node`` when the speed-up is at least 1.25, else ``trial_parallel`` (one independent run per GPU through the Spark harness). A direct Ethernet cable is evaluated as a
second scenario (about 110 MB/s, 0.5 ms) unless the probes were already measured over it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GRAD_MB = 11.4                       # float32 gradient of the 2.85 M-parameter network
ASSUMED_WIFI = {"bandwidth_mb_s": 20.0, "latency_s": 0.005, "source": "ASSUMED (no rank probe given): Wi-Fi about 20 MB/s, 5 ms"}
ASSUMED_CABLE = {"bandwidth_mb_s": 110.0, "latency_s": 0.0005, "source": "ASSUMED gigabit Ethernet: about 110 MB/s, 0.5 ms"}
ASSUMED_INTERNET = {"bandwidth_mb_s": 2.5, "latency_s": 0.04, "source": "ASSUMED laptop elsewhere through a VPN: home upload about 20 Mbit/s = 2.5 MB/s, 40 ms"}


def speed_of(probe):
    return float(probe["step"]["best"]["windows_per_s"])


def estimate(a, b, link, grad_mb=GRAD_MB, max_effective_batch=2048, h2d_gb_s=None, threshold=1.25):
    """a = the machine that also runs rank 0 (the desktop), b = the other one. ``link``: {bandwidth_mb_s, latency_s}. Returns the best configuration and the table of candidates."""
    sa, sb = speed_of(a), speed_of(b)
    cap_a, cap_b = int(a["step"]["max_micro_batch"]), int(b["step"]["max_micro_batch"])
    copy_s = 2 * grad_mb / 1000.0 / max(0.5, min(a.get("h2d_gb_s", 10.0), b.get("h2d_gb_s", 10.0)) if h2d_gb_s is None else h2d_gb_s)
    rows = []
    for compress, factor in (("none", 1.0), ("fp16", 0.5)):
        t_comm = 2 * link["latency_s"] + grad_mb * factor / link["bandwidth_mb_s"] + copy_s
        for g in (16, 32, 48, 64, 96, 128, 192, 256, 384, 512):
            ba = round(g * sa / (sa + sb))
            bb = g - ba
            if ba > cap_a or bb > cap_b or bb < 1:
                continue
            for k in (1, 2, 4, 8, 16, 32):
                if g * k > max_effective_batch:
                    continue
                t_comp = max(ba / sa, bb / sb) * k                         # the slower of the two paces the step
                speed = g * k / (t_comp + t_comm)
                rows.append({"global_micro_batch": g, "split": [ba, bb], "accum": k, "effective_batch": g * k, "compress": compress, "step_seconds": round(t_comp + t_comm, 4),
                             "comm_seconds": round(t_comm, 4), "windows_per_s": round(speed, 1), "speedup_vs_first_machine": round(speed / sa, 3), "efficiency_vs_sum": round(speed / (sa + sb), 3)})
    best = max(rows, key=lambda r: r["windows_per_s"]) if rows else None
    rec = "ddp_2node" if best and best["speedup_vs_first_machine"] >= threshold else "trial_parallel"
    return {"link": link, "speeds_windows_per_s": [sa, sb], "max_micro_batch": [cap_a, cap_b], "best": best, "recommendation": rec, "threshold": threshold,
            "upper_bound_speedup": round((sa + sb) / sa, 3), "candidates": len(rows)}


def run(probe_a, probe_b, cable=True, **kw):
    a, b = json.loads(Path(probe_a).read_text(encoding="utf-8")), json.loads(Path(probe_b).read_text(encoding="utf-8"))
    measured = a.get("link") or b.get("link")
    link = {**measured, "source": "measured with cluster_probe ranks"} if measured else ASSUMED_WIFI
    out = {"machines": [a["info"].get("hostname"), b["info"].get("hostname")], "current_link": estimate(a, b, link, **kw)}
    if cable and link["bandwidth_mb_s"] < 80:
        out["direct_ethernet_cable"] = estimate(a, b, ASSUMED_CABLE, **kw)
    if cable and link["bandwidth_mb_s"] > ASSUMED_INTERNET["bandwidth_mb_s"]:
        out["vpn_over_internet"] = estimate(a, b, ASSUMED_INTERNET, **kw)
    gl = [p["info"].get("gloo_hostname_ok") for p in (a, b)]
    out["warnings"] = [f"{p['info'].get('hostname')}: hostname resolves to {p['info'].get('hostname_resolves_to')} but the LAN address is {p['info'].get('lan_ip')}; Gloo may bind the wrong interface"
                       for p in (a, b) if p["info"].get("gloo_hostname_ok") is False and not p["info"].get("gloo_iface")]
    out["verdict"] = out["current_link"]["recommendation"]
    return out


def format_report(res):
    lines = []
    for key in ("current_link", "direct_ethernet_cable", "vpn_over_internet"):
        if key not in res:
            continue
        r = res[key]
        b = r["best"]
        lines.append(f"[{key}] link {r['link']['bandwidth_mb_s']:.0f} MB/s, {r['link']['latency_s'] * 1000:.1f} ms ({r['link'].get('source', '')})")
        lines.append(f"  machines alone: {r['speeds_windows_per_s'][0]:.0f} + {r['speeds_windows_per_s'][1]:.0f} windows/s (upper bound {r['upper_bound_speedup']:.2f}x)")
        if b:
            lines.append(f"  best DDP: micro-batch {b['global_micro_batch']} split {b['split']}, accum {b['accum']} (effective batch {b['effective_batch']}), compress {b['compress']} -> {b['windows_per_s']:.0f} windows/s "
                         f"= {b['speedup_vs_first_machine']:.2f}x the first machine alone ({100 * b['efficiency_vs_sum']:.0f}% of the sum); all-reduce {b['comm_seconds']:.2f} s per step")
        lines.append(f"  recommendation: {r['recommendation']} (needs >= {r['threshold']}x)")
    lines += [f"WARNING: {w}" for w in res.get("warnings", [])]
    lines.append(f"VERDICT on the current link: {res['verdict']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("probe_a", help="probe of the desktop (rank 0)")
    ap.add_argument("probe_b", help="probe of the laptop")
    ap.add_argument("--out", default="results/scaling_estimate.json")
    ap.add_argument("--max-effective-batch", type=int, default=2048)
    a = ap.parse_args(argv)
    res = run(a.probe_a, a.probe_b, max_effective_batch=a.max_effective_batch)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(format_report(res))


if __name__ == "__main__":
    main()
