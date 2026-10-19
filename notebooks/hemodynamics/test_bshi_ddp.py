# Check bshi ddp using controlled inputs and isolated files.
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hemodynamics import bshi as S
from hemodynamics import bshi_data as D
from hemodynamics import bshi_ddp as DD
from hemodynamics import bshi_train as T
from hemodynamics.bshi_model import BshiNet, ModelConfig
from hemodynamics.test_bshi_data import FAKE
from hemodynamics.test_bshi_train import fake_batch

SMALL = ModelConfig(widths=(8, 16, 16, 32), blocks=(1, 1, 1, 1), dropout=0.0, ctx_hidden=16, head_hidden=16)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def window_set(n=48, seed=0):
    """A fixed set of windows with some unobserved targets, to exercise the masks."""
    # Seed the generated values so this check can be repeated.
    b = fake_batch(n, seed)
    b["delta_map"] = b["delta_map"].clone()
    b["delta_map"][::7] = float("nan")
    b["reg_mask"] = torch.rand(n, generator=torch.Generator().manual_seed(seed + 1)) > 0.15
    return b


def slice_batch(b, a, c):
    return {k: v[a:c] for k, v in b.items()}


def cfg_for(epochs=2, **over):
    return T.TrainConfig.from_dict({"epochs": epochs, "patience": 50, "amp": "none", "num_workers": 0, "warmup_epochs": 0.0, "lr": 5e-3, "optimizer": "sgd", "weight_decay": 0.0, "schedule": "constant",
                                    "deterministic": True, "dropout": 0.0, "channels_last": False, "grad_clip": 0.0}, **over)


def make_trainer(out, rank, world, micro, accum, data, compress="none", epochs=2, steps=2):
    """Each rank takes its window range of ``data`` (rank r: sizes micro[r]) per micro-step; one epoch = ``steps`` optimizer steps."""
    pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
    total = sum(micro)
    offs = [sum(micro[:r]) for r in range(world)]

    def loader(epoch):
        for step in range(steps * accum):
            base = (step * total) % (len(data["y_event"]) - total + 1)
            yield slice_batch(data, base + offs[rank], base + offs[rank] + micro[rank])
    val = [slice_batch(data, 0, 16)]
    return DD.DDPTrainer(cfg_for(epochs), SMALL, pre, loader, lambda: val, out, rank, world, accum=accum, compress=compress, device="cpu", steps_per_epoch=steps, log=lambda *a: None)


def worker(rank, world, port, out, micro, accum, compress, epochs, steps, until, result_dir):
    # Fit the model or transform using the supplied fitting partition.
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world, timeout=__import__("datetime").timedelta(seconds=60))
    data = window_set(64, 0)
    tr = make_trainer(Path(out) / f"r{rank}", rank, world, micro, accum, data, compress, epochs, steps)
    tr.fit(resume=True, until_epoch=until)
    torch.save({k: v.cpu() for k, v in tr.model.state_dict().items()}, Path(result_dir) / f"rank{rank}.pt")
    dist.barrier()
    dist.destroy_process_group()


def run_ranks(tmp, micro, accum, compress="none", epochs=2, steps=2, until=None):
    # Create the destination directory before writing any outputs.
    port = free_port()
    res = Path(tmp) / "res"
    res.mkdir(exist_ok=True)
    mp.spawn(worker, args=(len(micro), port, str(tmp), micro, accum, compress, epochs, steps, until, str(res)), nprocs=len(micro), join=True)
    return [torch.load(res / f"rank{r}.pt") for r in range(len(micro))]


class TestPureHelpers(unittest.TestCase):
    def test_split_micro_batches(self):
        # Verify split micro batches.
        self.assertEqual(sum(DD.split_micro_batches(256, [2500, 900])), 256)
        a, b = DD.split_micro_batches(256, [2500, 900])
        self.assertAlmostEqual(a / b, 2500 / 900, delta=0.05)
        self.assertEqual(DD.split_micro_batches(10, [1, 1]), [5, 5])
        self.assertTrue(all(x >= 1 for x in DD.split_micro_batches(4, [100, 1, 1])))
        self.assertEqual(sum(DD.split_micro_batches(4, [100, 1, 1])), 4)

    def test_shard_patients_is_a_disjoint_proportional_partition(self):
        # Verify shard patients is a disjoint proportional partition.
        counts = pd.Series(np.random.default_rng(0).integers(1, 300, 400), index=[f"p{i}" for i in range(400)])
        own = DD.shard_patients(counts, [3.0, 1.0])
        self.assertEqual(set(own), set(counts.index))
        share0 = counts[[p for p, r in own.items() if r == 0]].sum() / counts.sum()
        self.assertAlmostEqual(float(share0), 0.75, delta=0.01)
        self.assertEqual(own, DD.shard_patients(counts, [3.0, 1.0]))

    def test_loss_sums_over_counts_equal_the_means_of_the_single_process_loss(self):
        # Verify loss sums over counts equal the means of the single process loss.
        pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
        t = pre(window_set(24, 3))
        out = {k: torch.randn(24) for k in ("event", "delta_map", "aut_logit", "aut_mag")}
        # Compare the result with the expected value or tolerance.
        w = {"event": 1.0, "delta_map": 0.3, "aut65": 0.3}
        ref, _ = T.multitask_loss(out, t, w)
        mine = DD.local_loss(DD.loss_sums(out, t), DD.batch_counts(t), w)
        self.assertAlmostEqual(float(ref), float(mine), places=6)
        # Compare the result with the expected value or tolerance.
        half = {k: v[:10] for k, v in t.items() if torch.is_tensor(v) and v.ndim and len(v) == 24}
        half2 = {k: v[10:] for k, v in t.items() if torch.is_tensor(v) and v.ndim and len(v) == 24}
        counts = DD.batch_counts({**t, **half}) + DD.batch_counts({**t, **half2})
        split = DD.local_loss(DD.loss_sums({k: v[:10] for k, v in out.items()}, {**t, **half}), counts, w) + DD.local_loss(DD.loss_sums({k: v[10:] for k, v in out.items()}, {**t, **half2}), counts, w)
        self.assertAlmostEqual(float(ref), float(split), places=5)                 # two uneven halves with GLOBAL counts add up to the full-batch loss

    def test_gloo_adapter_is_pinned_to_the_one_that_reaches_the_master(self):
        # Verify gloo adapter is pinned to the one that reaches the master.
        self.assertIsNone(DD.lan_interface("127.0.0.1"))
        self.assertIsNone(DD.lan_interface("localhost"))
        env = {}
        self.assertIsNone(DD.pin_gloo_interface("tcp://127.0.0.1:29500", env))     # loopback tests stay as they are
        self.assertNotIn("GLOO_SOCKET_IFNAME", env)
        env = {"GLOO_SOCKET_IFNAME": "Ethernet"}
        self.assertEqual(DD.pin_gloo_interface("tcp://10.9.9.9:29500", env), "Ethernet")      # a user-set value is never overridden
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect(("203.0.113.1", 9))
                own = s.getsockname()[0]
        except OSError:
            self.skipTest("no network route")
        name = DD.lan_interface(own)
        if name is None:
            self.skipTest("no adapter owns the default-route address")
        env = {}
        self.assertEqual(DD.pin_gloo_interface(f"tcp://{own}:29500", env), name)
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], name)


class TestTwoRanks(unittest.TestCase):
    def test_gradients_of_uneven_ranks_equal_one_big_batch(self):
        # Verify gradients of uneven ranks equal one big batch.
        with tempfile.TemporaryDirectory() as tmp:
            ranks = run_ranks(tmp, micro=[10, 6], accum=2, epochs=1, steps=3)
            self.assertTrue(all(torch.equal(ranks[0][k], ranks[1][k]) for k in ranks[0]), "the ranks must hold identical weights")
            # reference: ONE process; every optimizer step sees the 2 micro-steps x 16 windows as one batch of 32 (accum 1 with micro 32)
            port = free_port()
            dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=0, world_size=1)
            try:
                data = window_set(64, 0)
                pre = D.Preprocessor("eap", FAKE, device="cpu", channels_last=False)
                tr = DD.DDPTrainer(cfg_for(1), SMALL, pre, lambda e: [], lambda: [slice_batch(data, 0, 16)], Path(tmp) / "ref", 0, 1, accum=1, device="cpu", steps_per_epoch=3, log=lambda *a: None)

                def batches(epoch):                                            # the ranks' windows of optimizer step s, micro-step m: base = ((s*2+m)*16) % 49
                    for s in range(3):
                        wins = []
                        for m in range(2):
                            base = ((s * 2 + m) * 16) % 49
                            wins.append(slice_batch(data, base, base + 16))
                        yield {k: torch.cat([w[k] for w in wins]) for k in wins[0]}
                tr.train_loader = batches
                tr.sync_initial_weights = lambda: None
                tr.fit(resume=False)
                for k, v in tr.model.state_dict().items():
                    self.assertTrue(torch.allclose(v, ranks[0][k], atol=2e-5), k)
            finally:
                dist.destroy_process_group()

    def test_fp16_compression_stays_close(self):
        # Verify fp16 compression stays close.
        with tempfile.TemporaryDirectory() as tmp:
            a = run_ranks(tmp, micro=[8, 8], accum=2, epochs=1, steps=2)
        with tempfile.TemporaryDirectory() as tmp:
            b = run_ranks(tmp, micro=[8, 8], accum=2, compress="fp16", epochs=1, steps=2)
        worst = max(float((a[0][k] - b[0][k]).abs().max()) for k in a[0])
        self.assertLess(worst, 5e-3)
        self.assertTrue(all(torch.equal(b[0][k], b[1][k]) for k in b[0]))

    def test_resume_continues_exactly(self):
        # Verify resume continues exactly.
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            full = run_ranks(t1, micro=[8, 8], accum=1, epochs=2, steps=2)
            run_ranks(t2, micro=[8, 8], accum=1, epochs=2, steps=2, until=1)          # stop after one epoch
            resumed = run_ranks(t2, micro=[8, 8], accum=1, epochs=2, steps=2)           # both ranks restart; rank 1 gets the state by broadcast
            for k in full[0]:
                self.assertTrue(torch.allclose(full[0][k], resumed[0][k], atol=1e-5), k)
                self.assertTrue(torch.equal(resumed[0][k], resumed[1][k]), k)


class TestPeerLoss(unittest.TestCase):
    def test_a_missing_peer_gives_a_clear_error_instead_of_a_hang(self):
        # Verify a missing peer gives a clear error instead of a hang.
        t0 = time.time()
        with self.assertRaises(DD.PeerLost) as cm:
            DD.init_group(0, 2, f"tcp://127.0.0.1:{free_port()}", timeout_s=4)
        self.assertLess(time.time() - t0, 30)
        msg = str(cm.exception)
        self.assertIn("--master", msg)
        self.assertIn("port", msg)


if __name__ == "__main__":
    unittest.main()
