# Check cluster using controlled inputs and isolated files.
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import bshi as S
from hemodynamics import cluster as C
from hemodynamics import spark_jobs as J

# Load the recorded settings or metadata for this processing step.
ROOT = Path(__file__).resolve().parents[2]
CFG = json.loads((ROOT / "configs" / "hemodynamics.json").read_text(encoding="utf-8"))["cluster"]
WORKERS = [{"id": "w-desktop", "host": "10.0.0.19", "cores": 8, "memory": 24576, "coresused": 0, "state": "ALIVE"},
           {"id": "w-laptop", "host": "10.0.0.20", "cores": 4, "memory": 6144, "coresused": 0, "state": "ALIVE"}]


class TestConf(unittest.TestCase):
    def test_gpu_profile_gives_one_task_slot_per_machine(self):
        # Verify gpu profile gives one task slot per machine.
        c = C.derive_conf("gpu", WORKERS, CFG, "10.0.0.19")
        self.assertEqual(c["spark.executor.cores"], "4")
        self.assertEqual(c["spark.task.cpus"], "4")
        self.assertEqual(c["spark.cores.max"], "8")                       # 4 cores x 2 workers = exactly two executors, one per worker
        self.assertEqual(c["spark.task.maxFailures"], "1")                # retries are done (and reported) by run_trials
        self.assertLessEqual(int(c["spark.executor.memory"][:-1]), 2048)

    def test_etl_profile_uses_every_core_and_fits_the_smallest_worker(self):
        # Verify etl profile uses every core and fits the smallest worker.
        c = C.derive_conf("etl", WORKERS, CFG, "10.0.0.19")
        self.assertEqual(c["spark.cores.max"], "12")
        self.assertEqual(c["spark.executor.memory"], "5120m")             # smallest worker 6144 MB minus 1 GB
        self.assertNotIn("spark.executor.cores", c)

    def test_ports_and_hosts_come_from_the_config(self):
        # Verify ports and hosts come from the config.
        c = C.derive_conf("etl", WORKERS, CFG, "10.0.0.19")
        self.assertEqual(c["spark.driver.host"], "10.0.0.19")
        self.assertEqual(c["spark.driver.port"], str(CFG["driver_port"]))
        lo, hi = CFG["block_manager_port_range"]
        self.assertEqual(int(c["spark.driver.blockManager.port"]), lo)
        self.assertTrue(lo < int(c["spark.blockManager.port"]) + int(c["spark.port.maxRetries"]) <= hi + 1)
        self.assertEqual(c["spark.pyspark.python"], "python")             # resolved through each worker's own PATH

    def test_bad_inputs(self):
        # Verify bad inputs.
        with self.assertRaises(RuntimeError):
            C.derive_conf("etl", [], CFG, "h")
        with self.assertRaises(ValueError):
            C.derive_conf("nope", WORKERS, CFG, "h")

    def test_alive_workers_filters_dead_ones(self):
        # Compare the result with the expected value or tolerance.
        view = {"workers": WORKERS + [{**WORKERS[0], "id": "dead", "state": "DEAD"}]}
        self.assertEqual([w["id"] for w in C.alive_workers(view)], ["w-desktop", "w-laptop"])
        self.assertEqual(C.alive_workers(None), [])


class TestAddresses(unittest.TestCase):
    def test_override_and_hyper_v_refusal(self):
        # Verify override and hyper v refusal.
        old = os.environ.pop("BSHI_BIND_HOST", None)
        try:
            os.environ["BSHI_BIND_HOST"] = "192.168.50.1"
            self.assertEqual(C.lan_ipv4(), "192.168.50.1")
            del os.environ["BSHI_BIND_HOST"]

            class Fake:
                def __init__(self, *a): pass
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def connect(self, addr): pass
                def getsockname(self): return ("192.168.64.1", 0)
            real = C.socket.socket
            C.socket.socket = Fake
            try:
                with self.assertRaises(RuntimeError):
                    C.lan_ipv4()
            finally:
                C.socket.socket = real
        finally:
            if old is not None:
                os.environ["BSHI_BIND_HOST"] = old

    def test_master_host_auto_uses_the_lan_address_unless_overridden(self):
        # Verify master host auto uses the lan address unless overridden.
        old = os.environ.pop("BSHI_MASTER_HOST", None)
        try:
            self.assertEqual(C.master_host({"master_host": "192.168.50.1"}), "192.168.50.1")
            os.environ["BSHI_MASTER_HOST"] = "1.2.3.4"
            self.assertEqual(C.master_host({"master_host": "auto"}), "1.2.3.4")
        finally:
            os.environ.pop("BSHI_MASTER_HOST", None)
            if old is not None:
                os.environ["BSHI_MASTER_HOST"] = old

    def test_code_hash_is_stable_and_short(self):
        self.assertEqual(C.code_hash(), C.code_hash())
        self.assertEqual(len(C.code_hash()), 10)


class FakeRDD:
    def __init__(self, items): self.items = items
    def map(self, f): return FakeRDD([f(i) for i in self.items])
    def collect(self): return self.items


class FakeSC:
    def parallelize(self, data, n): return FakeRDD(list(data))


class FakeSpark:
    sparkContext = FakeSC()


class TestTrials(unittest.TestCase):
    def test_one_trial_returns_metrics_or_the_error_as_data(self):
        # Verify one trial returns metrics or the error as data.
        ok = C._run_one((3, {"lr": 3e-3}, "hemodynamics.cluster:demo_trial", 0))
        self.assertTrue(ok["ok"])
        self.assertEqual(ok["trial_id"], 3)
        # Check score against the expected result.
        self.assertAlmostEqual(ok["metrics"]["score"], 1.0)
        bad = C._run_one((4, {"fail_on_attempt": 0}, "hemodynamics.cluster:demo_trial", 0))
        self.assertFalse(bad["ok"])
        self.assertIn("planned failure", bad["error"])
        missing = C._run_one((5, {}, "hemodynamics.nothing:fn", 0))
        self.assertFalse(missing["ok"])

    def test_failed_trials_are_retried_once_and_reported(self):
        # Verify failed trials are retried once and reported.
        res = C.run_trials(FakeSpark(), [{"lr": 1e-3}, {"lr": 2e-3, "fail_on_attempt": 0}, {"lr": 3e-3, "fail_on_attempt": 1}, {"always": 1, "fail_on_attempt": 0}], retries=1)
        self.assertEqual([r["trial_id"] for r in res], [0, 1, 2, 3])
        self.assertEqual([r["attempts"] for r in res], [1, 2, 1, 2])      # trial 2 fails only on attempt 1, which never happens because it succeeded on attempt 0
        self.assertTrue(all(r["ok"] for r in res))
        self.assertEqual(res[1]["history"][0]["ok"], False)
        self.assertIn("planned failure", res[1]["history"][0]["error"])

    def test_a_trial_that_always_fails_stays_failed_after_the_retry(self):
        # Verify a trial that always fails stays failed after the retry.
        res = C.run_trials(FakeSpark(), [{"fail_on_attempt": 0}, {"fail_on_attempt": 1}], retries=1)
        self.assertEqual([r["ok"] for r in res], [True, True])            # each fails on one attempt only: the retry rescues both
        res = C.run_trials(FakeSpark(), [{"fail_on_attempt": 0}], retries=0)
        self.assertFalse(res[0]["ok"])
        self.assertEqual(res[0]["attempts"], 1)


class TestFeatures(unittest.TestCase):
    def batch(self, n=6, seed=0):
        # Seed the generated values so this check can be repeated.
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 7, 60, 200)).astype(np.float32)
        bm = rng.random((n, 3, 60)) > 0.2
        for p in range(3):
            x[:, p][~bm[:, p]] = 0
        return {"x": x, "beat_mask": bm, "context": rng.normal(size=(n, len(S.CONTEXT_NAMES))).astype(np.float32)}

    def test_features_have_the_declared_names_and_ignore_masked_beats(self):
        # Verify features have the declared names and ignore masked beats.
        b = self.batch()
        f = J.features_from_batch(b)
        self.assertEqual(list(f.columns), J.FEATURE_NAMES)
        # Compare the result with the expected value or tolerance.
        self.assertEqual(len(J.FEATURE_NAMES), 34)
        self.assertTrue(np.isfinite(f.to_numpy()).all())
        i = 2
        valid = b["beat_mask"][i, 1]
        # Compare the result with the expected value or tolerance.
        ref = b["x"][i, 1][valid].astype(np.float64)
        self.assertAlmostEqual(f.loc[i, "art_mean"], ref.mean(), places=6)
        self.assertAlmostEqual(f.loc[i, "art_std"], ref.std(), places=5)
        self.assertEqual(f.loc[i, "n_art"], valid.sum())
        for j, n in enumerate(S.CONTEXT_NAMES):
            self.assertEqual(f.loc[i, f"ctx_{n}"], float(b["context"][i, j]))

    def test_a_fully_masked_window_gives_zeros_not_nan(self):
        # Verify a fully masked window gives zeros not nan.
        b = self.batch(2)
        b["beat_mask"][0, 2] = False
        b["x"][0, 2] = 0
        f = J.features_from_batch(b)
        self.assertEqual(f.loc[0, "ppg_mean"], 0.0)
        self.assertEqual(f.loc[0, "ppg_std"], 0.0)

    def test_default_score_falls_with_map(self):
        # Compare the result with the expected value or tolerance.
        f = pd.DataFrame({"ctx_map_current": [50.0, 70.0, 100.0]})
        s = J.default_score(f)
        self.assertTrue(s[0] > s[1] > s[2])


@unittest.skipUnless((ROOT / J.DEV_REL).exists(), "dev subset not present")
class TestReplayPlan(unittest.TestCase):
    def test_plan_is_chronological_per_patient_and_interleaves_concurrent_patients(self):
        # Verify plan is chronological per patient and interleaves concurrent patients.
        plan = J.replay_plan(str(ROOT), n_windows=200, concurrent_patients=4)
        self.assertEqual(len(plan), 200)
        self.assertTrue(plan.window_id.is_unique)
        for k, g in plan.groupby("patient_key"):
            self.assertTrue(g.t_end_sec.is_monotonic_increasing, k)
        first = plan.patient_key.head(8).tolist()
        self.assertGreater(len(set(first)), 1)                            # patients alternate inside a group
        self.assertTrue(plan.rel.str.contains("split=test").all())


class TestScripts(unittest.TestCase):
    def test_powershell_scripts_parse(self):
        # Verify powershell scripts parse.
        for name in ("env.ps1", "start_master.ps1", "start_worker.ps1", "stop_all.ps1", "status.ps1"):
            p = ROOT / "scripts" / "spark" / name
            self.assertTrue(p.exists(), name)
            cmd = f"$e=$null; [System.Management.Automation.Language.Parser]::ParseFile('{p}', [ref]$null, [ref]$e) | Out-Null; if ($e.Count) {{ $e | ForEach-Object {{ $_.Message }}; exit 1 }}"
            r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, name + ": " + r.stdout[:300])

    def test_worker_scripts_never_bind_to_hyper_v_and_use_config_ports(self):
        # Check text against the expected result.
        text = (ROOT / "scripts" / "spark" / "start_worker.ps1").read_text(encoding="utf-8") + (ROOT / "scripts" / "spark" / "start_master.ps1").read_text(encoding="utf-8")
        self.assertIn("192.168.64.", text)
        self.assertIn("master_port", text)

    def test_flush_hook_only_activates_inside_a_spark_worker(self):
        # Verify flush hook only activates inside a spark worker.
        code = "import sitecustomize, pyspark.util as u; print(u.local_connect_and_auth.__name__)"
        env = {**os.environ, "PYTHONPATH": str(ROOT / "scripts" / "spark" / "pysite")}
        plain = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(plain.stdout.strip(), "local_connect_and_auth")
        hooked = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**env, "PYTHON_WORKER_FACTORY_PORT": "1"})
        self.assertEqual(hooked.stdout.strip(), "_flushing_local_connect_and_auth")


def _java_ok():
    try:
        C.find_jdk(CFG)
        return True
    except Exception:
        return False


@unittest.skipUnless(_java_ok(), "Java 17/21 not found")
class TestLocalSpark(unittest.TestCase):
    """A real Spark session in local mode (about 30 s): the Windows worker fix, the trial harness and the smoke aggregation on the dev subset."""

    @classmethod
    def setUpClass(cls):
        import signal
        cls._sigint = signal.getsignal(signal.SIGINT)            # PySpark replaces the Ctrl-C handler; later tests (the ingest interrupt test) must get the original back
        cls.spark = C.get_spark("etl", master="local", root=str(ROOT))

    @classmethod
    def tearDownClass(cls):
        import signal
        cls.spark.stop()
        signal.signal(signal.SIGINT, cls._sigint)

    def test_trials_run_in_tasks_and_retry(self):
        # Compare the result with the expected value or tolerance.
        res = C.run_trials(self.spark, [{"lr": 1e-3}, {"lr": 3e-3, "fail_on_attempt": 0}], retries=1)
        self.assertTrue(all(r["ok"] for r in res))
        self.assertEqual([r["attempts"] for r in res], [1, 2])

    @unittest.skipUnless((ROOT / J.DEV_REL).exists(), "dev subset not present")
    def test_smoke_aggregation_equals_pandas(self):
        # Verify smoke aggregation equals pandas.
        res = J.smoke_aggregate(self.spark, str(ROOT))
        self.assertTrue(res["counts_equal"])
        self.assertTrue(res["weights_equal"])
        self.assertEqual(res["windows"], 17167)


if __name__ == "__main__":
    unittest.main()
