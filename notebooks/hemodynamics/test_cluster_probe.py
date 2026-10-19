import datetime
import json
import socket
import tempfile
import unittest
from pathlib import Path

import torch
import torch.multiprocessing as mp

from hemodynamics import cluster_probe as P
from hemodynamics import scaling_estimate as E


def probe(host, speed, max_mb, link=None, gloo_ok=True, h2d=12.0):
    p = {"info": {"hostname": host, "hostname_resolves_to": "10.0.0.5" if gloo_ok else "172.17.0.1", "lan_ip": "10.0.0.5", "gloo_hostname_ok": gloo_ok},
         "step": {"best": {"batch_size": 64, "windows_per_s": speed}, "max_micro_batch": max_mb}, "h2d_gb_s": h2d}
    if link:
        p["link"] = link
    return p


def write(tmp, name, d):
    p = Path(tmp) / name
    p.write_text(json.dumps(d), encoding="utf-8")
    return str(p)


class TestEstimator(unittest.TestCase):
    def test_slow_wifi_with_a_slower_laptop_recommends_trial_parallel_and_the_cable_helps(self):
        with tempfile.TemporaryDirectory() as t:
            res = E.run(write(t, "a.json", probe("desktop", 2500, 256)), write(t, "b.json", probe("laptop", 900, 96)))
        cur = res["current_link"]
        self.assertEqual(cur["recommendation"], "trial_parallel")
        self.assertIn("ASSUMED", cur["link"]["source"])
        self.assertIn("direct_ethernet_cable", res)
        self.assertGreater(res["direct_ethernet_cable"]["best"]["windows_per_s"], cur["best"]["windows_per_s"])
        self.assertLessEqual(cur["best"]["effective_batch"], 2048)
        self.assertEqual(res["verdict"], "trial_parallel")

    def test_two_equal_machines_on_a_fast_link_recommend_ddp(self):
        fast = {"bandwidth_mb_s": 110.0, "latency_s": 0.0005}
        with tempfile.TemporaryDirectory() as t:
            res = E.run(write(t, "a.json", probe("a", 2500, 256, fast)), write(t, "b.json", probe("b", 2500, 256, fast)))
        self.assertEqual(res["verdict"], "ddp_2node")
        self.assertGreater(res["current_link"]["best"]["speedup_vs_first_machine"], 1.25)
        self.assertNotIn("direct_ethernet_cable", res)                         # the probes were already measured on a fast link
        self.assertIn("measured", res["current_link"]["link"]["source"])

    def test_the_small_gpu_limits_the_micro_batch_split(self):
        with tempfile.TemporaryDirectory() as t:
            res = E.run(write(t, "a.json", probe("a", 2500, 384)), write(t, "b.json", probe("b", 900, 16)))
        b = res["current_link"]["best"]
        self.assertLessEqual(b["split"][1], 16)
        self.assertEqual(sum(b["split"]), b["global_micro_batch"])
        self.assertAlmostEqual(b["split"][0] / b["split"][1], 2500 / 900, delta=0.8)

    def test_effective_batch_cap_is_respected(self):
        fast = {"bandwidth_mb_s": 110.0, "latency_s": 0.0005}
        a, b = probe("a", 2500, 256, fast), probe("b", 2500, 256, fast)
        for cap in (512, 1024):
            r = E.estimate(a, b, fast, max_effective_batch=cap)
            self.assertLessEqual(r["best"]["effective_batch"], cap)

    def test_hostname_warning_and_report_text(self):
        with tempfile.TemporaryDirectory() as t:
            res = E.run(write(t, "a.json", probe("desktop", 2500, 256, gloo_ok=False)), write(t, "b.json", probe("laptop", 900, 96)))
        self.assertEqual(len(res["warnings"]), 1)
        txt = E.format_report(res)
        self.assertIn("VERDICT", txt)
        self.assertIn("WARNING", txt)

    def test_compression_lowers_the_comm_time(self):
        a = b = probe("x", 2500, 256)
        r = E.estimate(a, b, {"bandwidth_mb_s": 20.0, "latency_s": 0.005})
        self.assertEqual(r["best"]["compress"], "fp16")


class TestLinkFit(unittest.TestCase):
    def test_least_squares_recovers_bandwidth_and_latency(self):
        bw, lat = 50.0, 0.004
        comm = {"latency_ms": 4.0, "sizes": [{"mb": m, "allreduce_s": lat + m / bw} for m in (1, 4, 16, 64)]}
        got_bw, got_lat = P.fit_link(comm)
        self.assertAlmostEqual(got_bw, bw, delta=0.5)
        self.assertAlmostEqual(got_lat, lat, delta=1e-4)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _comm_worker(rank, port, out_dir):
    torch.set_num_threads(1)
    res = P.comm_probe(rank, 2, f"tcp://127.0.0.1:{port}", sizes_mb=(1, 2), repeats=2, timeout_s=60)
    if rank == 0:
        (Path(out_dir) / "comm.json").write_text(json.dumps(res), encoding="utf-8")


class TestCommProbe(unittest.TestCase):
    def test_two_local_ranks_measure_latency_and_bandwidth(self):
        with tempfile.TemporaryDirectory() as t:
            mp.spawn(_comm_worker, args=(_free_port(), t), nprocs=2, join=True)
            comm = json.loads((Path(t) / "comm.json").read_text(encoding="utf-8"))
        self.assertGreater(comm["latency_ms"], 0)
        self.assertEqual([r["mb"] for r in comm["sizes"]], [1, 2])
        for r in comm["sizes"]:
            self.assertGreater(r["allreduce_mb_s"], 1)
            self.assertGreater(r["broadcast_from_0_s"], 0)
            self.assertGreater(r["broadcast_from_1_s"], 0)
        bw, lat = P.fit_link(comm)
        self.assertGreater(bw, 1)


class TestMachineInfo(unittest.TestCase):
    def test_info_has_the_fields_the_estimator_needs(self):
        i = P.machine_info()
        for k in ("hostname", "cpu_logical", "ram_gb", "hostname_resolves_to", "torch", "cuda_available"):
            self.assertIn(k, i)


if __name__ == "__main__":
    unittest.main()
