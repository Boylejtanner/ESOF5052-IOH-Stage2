import os
import tempfile
import threading
import time
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from hemodynamics import bshi_data as D
from hemodynamics import bshi_notebook as NB
from hemodynamics import bshi_progress as P
from hemodynamics import bshi_store as ST
from hemodynamics import cluster as C


class PrefetchTests(unittest.TestCase):
    def test_order_is_kept(self):
        self.assertEqual(list(D.prefetch(iter(range(50)), depth=3)), list(range(50)))

    def test_early_close_stops_the_thread(self):
        before = threading.active_count()
        g = D.prefetch(iter(range(10 ** 9)), depth=2)
        next(g)
        g.close()
        time.sleep(0.6)
        self.assertLessEqual(threading.active_count(), before)

    def test_errors_reach_the_consumer(self):
        def bad():
            yield 1
            raise ValueError("boom")
        with self.assertRaises(ValueError):
            list(D.prefetch(bad()))


def fake_index():
    rows = []
    rng = np.random.default_rng(0)
    for ds, n_pat in (("vitaldb", 24), ("mover", 12)):
        for p in range(n_pat):
            pooled = "train" if p % 4 < 2 else ("validation" if p % 4 == 2 else "test")
            within = ["train", "validation", "test"][(p * 7 + 1) % 3]        # a DIFFERENT partition than the pooled one
            for k in range(9):
                rows.append({"dataset": ds, "patient_key": f"{ds}_{p}", "case_id": f"{ds}_{p}_c", "t_end_sec": 30.0 * k, "split_pooled": pooled, "split_within_dataset": within,
                             "y_event": float(rng.random() < 0.1), "window_id": len(rows)})
    return pd.DataFrame(rows)


class ExperimentFramesTests(unittest.TestCase):
    def test_stride_weights_and_all_validation_test(self):
        idx = fake_index()
        fr = ST.experiment_frames(idx, "pooled", stride=2)
        full_train = idx[idx.split_pooled == "train"]
        self.assertEqual(len(fr["train"]), int(np.ceil(9 / 2)) * full_train.patient_key.nunique())
        self.assertTrue((fr["train"].sampling_weight == 2).all())
        self.assertEqual(len(fr["validation"]), int((idx.split_pooled == "validation").sum()))
        self.assertEqual(len(fr["test"]), int((idx.split_pooled == "test").sum()))
        self.assertTrue((fr["test"].sampling_weight == 1).all())

    def test_every_experiment_keeps_its_own_split_and_patients_stay_disjoint(self):
        idx = fake_index()
        for train_on in ("pooled", "vitaldb", "mover"):
            fr = ST.experiment_frames(idx, train_on, stride=2)
            p = {k: set(v.patient_key) for k, v in fr.items()}
            self.assertFalse(p["train"] & p["validation"], train_on)
            self.assertFalse(p["train"] & p["test"], train_on)
            self.assertFalse(p["validation"] & p["test"], train_on)
        # the pooled and the within-dataset partitions really differ in the fake data: a pooled store would leak for 'vitaldb'
        pooled_train = set(ST.experiment_frames(idx, "pooled")["train"].patient_key)
        vit_test = set(ST.experiment_frames(idx, "vitaldb")["test"].patient_key)
        self.assertTrue(pooled_train & vit_test)

    def test_store_paths_differ_per_experiment(self):
        self.assertEqual(len({ST.store_path(".", t) for t in ("pooled", "vitaldb", "mover")}), 3)


class ModeTests(unittest.TestCase):
    def test_modes_fall_back_without_a_laptop(self):
        none = {"master_running": False, "master": None, "workers": [], "remote": []}
        one = {"master_running": True, "master": "10.0.0.19", "workers": [], "remote": [{"host": "100.64.1.2"}]}
        self.assertEqual(NB.resolve_mode(".", "single_gpu", none)[0], "single_gpu")
        self.assertEqual(NB.resolve_mode(".", "ddp_2node", none)[0], "single_gpu")
        self.assertEqual(NB.resolve_mode(".", "trial_parallel", none)[0], "trial_sequential")
        self.assertEqual(NB.resolve_mode(".", "ddp_2node", one)[0], "ddp_2node")
        self.assertEqual(NB.resolve_mode(".", "trial_parallel", one)[0], "trial_parallel")
        with self.assertRaises(ValueError):
            NB.resolve_mode(".", "nonsense", none)

    def test_single_gpu_needs_no_cluster_lookup(self):
        with mock.patch.object(NB, "cluster_status", side_effect=AssertionError("must not look")):
            self.assertEqual(NB.resolve_mode(".", "single_gpu")[0], "single_gpu")

    def test_cluster_status_without_a_master_is_a_normal_answer(self):
        with mock.patch.object(C, "load_config", return_value={"master_ui": 8080}), mock.patch.object(C, "master_host", return_value="10.0.0.19"), \
                mock.patch.object(C, "master_view", return_value=None):
            st = NB.cluster_status(".")
        self.assertFalse(st["master_running"])
        self.assertEqual(st["remote"], [])

    def test_remote_worker_is_the_one_on_another_host(self):
        view = {"workers": [{"id": "w1", "host": "10.0.0.19", "cores": 8, "memory": 24000, "state": "ALIVE"}, {"id": "w2", "host": "100.64.3.4", "cores": 6, "memory": 8000, "state": "ALIVE"}]}
        with mock.patch.object(C, "load_config", return_value={"master_ui": 8080}), mock.patch.object(C, "master_host", return_value="10.0.0.19"), \
                mock.patch.object(C, "master_view", return_value=view), mock.patch.object(C, "alive_workers", return_value=view["workers"]):
            st = NB.cluster_status(".")
        self.assertEqual([w["host"] for w in st["remote"]], ["100.64.3.4"])


class VpnHostTests(unittest.TestCase):
    def test_default_is_the_lan_address(self):
        with mock.patch.dict(os.environ, {}, clear=False), mock.patch.object(C, "lan_ipv4", return_value="10.0.0.19"):
            os.environ.pop("BSHI_NETWORK", None)
            os.environ.pop("BSHI_MASTER_HOST", None)
            self.assertEqual(C.master_host({"master_host": "auto"}), "10.0.0.19")

    def test_vpn_uses_the_vpn_address_and_falls_back_to_lan(self):
        env = {k: v for k, v in os.environ.items() if k not in ("BSHI_NETWORK", "BSHI_MASTER_HOST")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(C, "lan_ipv4", return_value="10.0.0.19"):
            with mock.patch.object(C, "vpn_ipv4", return_value="100.101.102.103"):
                self.assertEqual(C.master_host({"master_host": "vpn"}), "100.101.102.103")
            with mock.patch.object(C, "vpn_ipv4", return_value=None), self.assertWarns(UserWarning):
                self.assertEqual(C.master_host({"master_host": "vpn"}), "10.0.0.19")        # desktop-only keeps working

    def test_env_network_overrides_config(self):
        with mock.patch.dict(os.environ, {"BSHI_NETWORK": "vpn"}), mock.patch.object(C, "vpn_ipv4", return_value="100.64.0.9"):
            os.environ.pop("BSHI_MASTER_HOST", None)
            self.assertEqual(C.master_host({"master_host": "auto"}), "100.64.0.9")


class BaselineColumnsTests(unittest.TestCase):
    ROOT = os.environ.get("BSHI_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def test_every_rung_column_is_in_the_index_or_the_feature_file(self):
        import pyarrow.parquet as pq
        from hemodynamics import eda_models as M
        idx_path = os.path.join(self.ROOT, "data", "derived", "bshi", "window_index_v1.parquet")
        if not (os.path.exists(idx_path) and os.path.exists(M.features_path(self.ROOT))):
            self.skipTest("real window index / features not on disk")
        have = set(pq.read_schema(idx_path).names) | set(pq.read_schema(M.features_path(self.ROOT)).names)
        for rung in ("G0", "G4"):
            self.assertEqual([c for c in M.RUNGS[rung] if c not in have], [], rung)


class ProgressStateTests(unittest.TestCase):
    def test_paused_and_stalled_runs_are_named(self):
        with tempfile.TemporaryDirectory() as td:
            p = P.Progress(td, label="unit", console=False, notebook=False, gpu=False, html_page=False)
            p.start_run(epochs=5, steps_per_epoch=10, windows_per_step=128)
            p.begin_epoch(2)
            p.pause()
            line = P.status_line(P.read_status(td))
            self.assertIn("paused", line)
            self.assertIn("epoch 3/5", line)
            st = P.read_status(td)
            st["phase"], st["updated"] = "train", "2000-01-01 00:00:00"
            self.assertIn("NO UPDATE", P.status_line(st))


if __name__ == "__main__":
    unittest.main()
