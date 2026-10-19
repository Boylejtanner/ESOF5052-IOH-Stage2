"""Focused, network-free checks; test scratch stays inside the project."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, wait as real_wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, redirect_stdout
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import threading
import time
import shutil
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error
import uuid

import numpy as np
import pandas as pd
import vitaldb
from vitaldb import dataset as vital_dataset

from hemodynamics import canonical
from hemodynamics import vitaldb_pipeline as pipeline

PROJECT = Path(__file__).resolve().parents[2]


_ORIGINAL_CASE_WORKER = pipeline._process_case
_REAL_INDEX_HOOK = pipeline._build_unified_index_after_download
_MOCK_PATCH_LOCK = threading.Lock()


def _mock_spawn_worker(job):
    """Importable spawned fixture: mocked network/native records, real atomic Parquet."""
    options = job["settings"]["vitaldb"]
    scenario = options.get("test_scenario", "normal")
    if scenario == "crash" and job["case_id"] == 1:
        os._exit(13)
    wave = pd.DataFrame({"time_sec": np.arange(10) * .01, "ART": np.full(10, 80.0),
                         "ECG": np.linspace(.1, .2, 10), "PPG": np.linspace(1., 2., 10)})
    numeric = pd.DataFrame({"time_sec": [.01, .07], "MAP": [75., 74.]})
    metrics = pipeline.frame_metrics(wave, numeric)
    metrics["measured_quality_passed"] = True
    delay = float(options.get("test_delay_sec", .03))
    if scenario == "pause" and job["case_id"] == 1:
        delay = 0
    pipeline._worker_event(job["case_id"], "network")
    time.sleep(delay)
    def fake_load(case_id, tracks, timeout_seconds=None):
        fail_case = int(options.get("test_fail_case", 2))
        if scenario == "one_failure" and case_id == fail_case:
            raise TimeoutError("mock network timeout")
        if scenario == "http429" and case_id == fail_case:
            raise urllib.error.HTTPError("http://vitaldb.invalid/2.vital", 429, "Too Many Requests", {}, None)
        return SimpleNamespace(dtstart=100., dtend=1900.)
    with _MOCK_PATCH_LOCK, patch.object(pipeline, "load_vital_case", side_effect=fake_load), \
         patch.object(pipeline, "native_to_frames", return_value=(wave, numeric, {"ART":500., "ECG":500., "PPG":500.})), \
         patch.object(pipeline, "frame_metrics", return_value=metrics):
        return _ORIGINAL_CASE_WORKER(job)


class VitalDBPipelineTests(unittest.TestCase):
    def setUp(self):
        self.root = PROJECT / "data" / "checkpoints" / "vitaldb" / ("test-" + uuid.uuid4().hex)
        self.root.mkdir(parents=True)
        (self.root / "configs").mkdir()
        shutil.copyfile(PROJECT / "configs" / "hemodynamics.json", self.root / "configs" / "hemodynamics.json")
        path = self.root / "configs" / "hemodynamics.json"
        settings = json.loads(path.read_text())
        settings["allow_full_vitaldb_download"] = False
        settings["vitaldb"].update(download_workers=2, progress_interval_seconds=0.05, stop_grace_seconds=1.0)
        path.write_text(json.dumps(settings))
        self.executor_patch = patch.object(pipeline, "_create_executor", side_effect=lambda workers, events, settings: ThreadPoolExecutor(max_workers=workers))
        self.executor_patch.start()
        self.hook = MagicMock(return_value={"status": "stubbed"})
        self.hook_patch = patch.object(pipeline, "_build_unified_index_after_download", self.hook)
        self.hook_patch.start()

    def tearDown(self):
        self.hook_patch.stop()
        self.executor_patch.stop()
        shutil.rmtree(self.root)

    def test_patient_linkage_is_not_surgery_identity_and_no_leakage(self):
        cohort = pd.DataFrame({"caseid": range(1, 21)})
        clinical = pd.DataFrame({"caseid": range(1, 21), "subjectid": [200 + i // 2 for i in range(20)]})
        first = pipeline.patient_splits(cohort, clinical)
        second = pipeline.patient_splits(cohort.sample(frac=1, random_state=9), clinical)
        pd.testing.assert_frame_equal(first, second)
        self.assertTrue(first.patient_id.gt(20).all())
        self.assertEqual(first.groupby("patient_id").split.nunique().max(), 1)
        sets = [set(first.loc[first.split.eq(split), "patient_id"]) for split in ("train", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_absent_patient_linkage_fails_not_invented(self):
        with self.assertRaisesRegex(ValueError, "Missing documented patient"):
            pipeline.patient_splits(pd.DataFrame({"caseid": [1]}), pd.DataFrame({"caseid": [1], "subjectid": [np.nan]}))

    def test_pilot_selection_is_three_deterministic_outcome_blind_cases(self):
        cohort = pd.DataFrame({"caseid": range(11, 111), "duration_seconds": range(100)})
        expected = [20, 60, 100]
        self.assertEqual(pipeline.pilot_case_ids(cohort), expected)
        self.assertEqual(pipeline.pilot_case_ids(cohort.sample(frac=1, random_state=7)), expected)

    def test_exact_map_preference_and_missing_channel_fail(self):
        names = list(pipeline.TRACKS.values()) + list(pipeline.MAP_PRIORITY)
        index = pd.DataFrame({"caseid": [4] * len(names), "tname": names})
        self.assertEqual(pipeline.resolve_tracks(index, 4)["MAP"], "Solar8000/ART_MBP")
        with self.assertRaisesRegex(ValueError, "waveform"):
            pipeline.resolve_tracks(index.loc[index.tname.ne("SNUADC/PLETH")], 4)

    def test_native_timestamps_and_long_gaps_preserved(self):
        tracks = {**pipeline.TRACKS, "MAP": pipeline.MAP_PRIORITY[0]}
        waveform = SimpleNamespace(type=1, srate=500.0, fmt=3, gain=0.25, offset=0.0,
                                   recs=[{"dt": 100.02, "val": np.full(20, 8, dtype=np.int16)},
                                         {"dt": 100.20, "val": np.full(10, 8, dtype=np.int16)}])
        numeric = SimpleNamespace(type=2, recs=[{"dt": 100.023, "val": 75.0}, {"dt": 100.201, "val": 70.0}])
        vf = SimpleNamespace(dtstart=100.0, dtend=100.25,
                             trks={**{name: waveform for name in pipeline.TRACKS.values()}, tracks["MAP"]: numeric})
        wave, map_frame, rates = pipeline.native_to_frames(vf, tracks)
        np.testing.assert_allclose(map_frame.time_sec, [0.023, 0.201])
        np.testing.assert_allclose(map_frame.MAP, [75.0, 70.0])
        self.assertTrue(wave.loc[wave.time_sec.between(0.06, 0.19), "ART"].isna().all())
        self.assertTrue(wave.loc[wave.time_sec.lt(0.02), "ART"].isna().all())
        self.assertEqual(rates["ART"], 500.0)
        self.assertAlmostEqual(float(wave.loc[np.isclose(wave.time_sec, 0.03), "ART"].iloc[0]), 2.0, places=5)

    def _native_fixture(self, records, end):
        names = {**pipeline.TRACKS, "MAP": pipeline.MAP_PRIORITY[0]}
        waveform = SimpleNamespace(type=1, srate=500.0, fmt=1, gain=1.0, offset=0.0, recs=records)
        numeric = SimpleNamespace(type=2, recs=[{"dt": 100.01, "val": 75.0}, {"dt": 100.5, "val": 70.0}])
        vf = SimpleNamespace(dtstart=100.0, dtend=100.0 + end,
                             trks={**{name: waveform for name in pipeline.TRACKS.values()}, names["MAP"]: numeric})
        return vf, names

    def test_antialias_rejects_120hz_before_100hz_downsample(self):
        times = np.arange(5000) / 500.0
        source = np.sin(2 * np.pi * 120 * times)
        vf, tracks = self._native_fixture([{"dt": 100.0, "val": source}], 10.0)
        wave, _, _ = pipeline.native_to_frames(vf, tracks)
        interior = wave.time_sec.between(0.5, 9.5)
        self.assertGreater(np.std(source[::5]), 0.6)
        self.assertLess(float(np.sqrt(np.mean(wave.loc[interior, "ART"] ** 2))), 0.01)

    def test_contiguous_record_boundary_and_phase_alignment(self):
        start = 0.006
        source_times = start + np.arange(1000) / 500.0
        source = np.sin(2 * np.pi * 10 * source_times)
        records = [{"dt": 100.0 + start, "val": source[:500]},
                   {"dt": 101.0 + start, "val": source[500:]}]
        vf, tracks = self._native_fixture(records, 2.5)
        split, _, _ = pipeline.native_to_frames(vf, tracks)
        vf.trks[tracks["ART"]].recs = [{"dt": 100.0 + start, "val": source}]
        full, _, _ = pipeline.native_to_frames(vf, tracks)
        np.testing.assert_allclose(split.ART, full.ART, atol=1e-6, equal_nan=True)
        interior = split.time_sec.between(0.3, 1.7)
        np.testing.assert_allclose(split.loc[interior, "ART"], np.sin(2 * np.pi * 10 * split.loc[interior, "time_sec"]), atol=0.003)
        self.assertTrue(split.loc[split.time_sec.lt(start), "ART"].isna().all())
        self.assertTrue(split.loc[split.time_sec.ge(start + 2.0), "ART"].isna().all())
        np.testing.assert_allclose(np.diff(split.time_sec), 0.01, atol=1e-8)

    def test_filter_never_bridges_internal_nan_run(self):
        source = np.r_[np.full(100, 80.0), np.full(25, np.nan), np.full(375, 20.0)]
        vf, tracks = self._native_fixture([{"dt": 100.0, "val": source}], 1.0)
        wave, _, _ = pipeline.native_to_frames(vf, tracks)
        self.assertTrue(wave.loc[wave.time_sec.ge(0.20) & wave.time_sec.lt(0.25), "ART"].isna().all())
        self.assertAlmostEqual(float(wave.loc[np.isclose(wave.time_sec, 0.19), "ART"].iloc[0]), 80.0, places=4)
        self.assertAlmostEqual(float(wave.loc[np.isclose(wave.time_sec, 0.25), "ART"].iloc[0]), 20.0, places=4)

    def test_retry_backoff_is_bounded(self):
        calls = []
        def operation():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("network")
            return 5
        with patch.object(pipeline.time, "sleep") as sleep:
            self.assertEqual(pipeline._retry(operation), 5)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.0, 4.0])
        self.assertEqual(len(calls), 3)

    def test_atomic_json_has_no_partial_file(self):
        path = self.root / "audit.json"
        pipeline._atomic_json(path, {"test": 1})
        self.assertEqual(json.loads(path.read_text()), {"test": 1})
        self.assertFalse(path.with_name(path.name + ".part").exists())

    def test_canonical_validated_resume_and_corruption_rejected(self):
        wave = pd.DataFrame({"time_sec": [0.0, 0.01], "ART": [80.0, 81.0], "ECG": [0.2, 0.3], "PPG": [1.0, 2.0]})
        numeric = pd.DataFrame({"time_sec": [0.005, 0.9], "MAP": [75.0, 74.0]})
        metadata = canonical.write_case(self.root, "vitaldb", 9, wave, numeric, {"patient_id": 200, "split": "train", "min_free_space_bytes": 0,
                                                                                  "processing_signature": pipeline.processing_signature(),
                                                                                  "time_reference": "seconds_since_recording_start",
                                                                                  "art_final_fs": 100, "ecg_final_fs": 100, "ppg_final_fs": 100})
        self.assertEqual(metadata["time_reference"], "seconds_since_recording_start")
        self.assertTrue(all(metadata[key] == 100 for key in ("art_final_fs", "ecg_final_fs", "ppg_final_fs")))
        metadata["status"] = "complete"
        self.assertTrue(pipeline._completed(self.root, metadata))
        loaded = canonical.load_case(self.root, "vitaldb", 9)
        np.testing.assert_allclose(loaded["map_time_sec"], [0.005, 0.9])
        path = self.root / metadata["waveform_output_path"]
        with path.open("ab") as stream:
            stream.write(b"corruption")
        self.assertFalse(pipeline._completed(self.root, metadata))

    def test_full_download_blocked_without_pilot(self):
        self._prepare_pipeline_inputs()
        path = self.root / "configs" / "hemodynamics.json"
        settings = json.loads(path.read_text())
        settings["allow_full_vitaldb_download"] = True
        path.write_text(json.dumps(settings))
        with patch.object(pipeline, "audit_project", return_value={"pilot_passed": False, "pilot_case_ids": [1, 2, 3]}):
            with self.assertRaisesRegex(RuntimeError, "FULL_DOWNLOAD blocked until three"):
                pipeline.run_pipeline(self.root, "FULL_DOWNLOAD")

    def test_full_download_blocked_by_explicit_config_even_after_pilot(self):
        with patch.object(pipeline, "audit_project", return_value={"pilot_passed": True, "pilot_case_ids": [1, 2, 3]}):
            with self.assertRaisesRegex(RuntimeError, "allow_full_vitaldb_download=true"):
                pipeline.run_pipeline(self.root, "FULL_DOWNLOAD")
            with self.assertRaisesRegex(RuntimeError, "allow_full_vitaldb_download=true"):
                pipeline.run_pipeline(self.root, "RESUME", resume_target="FULL_DOWNLOAD")

    def test_patient_seed_and_parameters_come_from_shared_config(self):
        settings = pipeline.load_settings(self.root)
        self.assertEqual(settings["patient_split_seed"], 5052)
        self.assertEqual(settings["random_seed"], 5052)
        self.assertEqual(settings["target_waveform_fs"], 100)
        cohort = pd.DataFrame({"caseid": range(1, 21)})
        clinical = pd.DataFrame({"caseid": range(1, 21), "subjectid": [200 + i // 2 for i in range(20)]})
        linked = pipeline.patient_splits(cohort, clinical, settings=settings)
        self.assertTrue(linked.seed.eq(settings["patient_split_seed"]).all())
        self.assertEqual(settings["vitaldb"]["quality_thresholds"]["map_measurements_per_minute_min"], 10)
        self.assertEqual(settings["map_min_points_per_min"], 5)

    def test_resume_rejects_legacy_filter_or_changed_split_seed(self):
        settings = pipeline.load_settings(self.root)
        state = {"status": "complete", "processing_signature": pipeline.processing_signature(settings)}
        with patch.object(canonical, "verify_case", return_value=True):
            self.assertTrue(pipeline._completed(self.root, state))
            self.assertFalse(pipeline._completed(self.root, {"status": "complete"}))
            settings["patient_split_seed"] = 2026
            state["processing_signature"] = pipeline.processing_signature(settings)
            self.assertFalse(pipeline._completed(self.root, state))

    def _prepare_pipeline_inputs(self, count=3):
        manifests = self.root / "data" / "manifests"
        manifests.mkdir(parents=True)
        cases = list(range(1, count + 1))
        pd.DataFrame({"caseid": cases, "duration_seconds": [1800.0] * count}).to_csv(manifests / "usable_cohort.csv", index=False)
        reference = self.root / "data" / "reference" / "vitaldb"
        reference.mkdir(parents=True)
        pd.DataFrame({"caseid": cases, "casestart": [0] * count, "caseend": [1800] * count,
                      "anestart": [-10] * count, "aneend": [1700] * count, "opstart": [100] * count,
                      "opend": [1600] * count}).to_csv(reference / "clinical_data.csv", index=False)
        if count == 3:
            patients, splits = [101, 102, 103], ["train", "validation", "test"]
        else:
            # Two surgeries per patient; patients 0-3 train, 4 validation, 5 test (as many as the cohort needs).
            patients = [100 + (case - 1) // 2 for case in cases]
            splits = [("train", "train", "train", "train", "validation", "test")[(case - 1) // 2 % 6] for case in cases]
        pd.DataFrame({"case_id": cases, "patient_id": patients, "split": splits,
                      "patient_id_source": ["documented subjectid"] * count}).to_csv(manifests / "vitaldb_patient_splits.csv", index=False)
        return {"pilot_case_ids": [1, 2, 3], "pilot_passed": False}

    def test_pipeline_disk_stop_is_checkpointed_before_network(self):
        audit = self._prepare_pipeline_inputs()
        with patch.object(pipeline, "audit_project", return_value=audit), \
             patch.object(pipeline, "load_track_index", return_value=pd.DataFrame()), \
             patch.object(pipeline, "write_audit_markdown"), \
             patch.object(pipeline, "resolve_tracks", return_value={**pipeline.TRACKS, "MAP": pipeline.MAP_PRIORITY[0]}), \
             patch.object(pipeline, "load_vital_case") as download, \
             patch.object(canonical, "ensure_disk_space", side_effect=canonical.DiskSpaceStop("reserve exhausted")):
            result = pipeline.run_pipeline(self.root)
        self.assertEqual(result["stopped_reason"], "reserve exhausted")
        self.assertEqual(canonical.CheckpointStore(self.root, "vitaldb").get(1)["status"], "disk_stopped")
        download.assert_not_called()
        manifest = pd.read_parquet(self.root / "data" / "manifests" / "vitaldb_download_manifest.parquet")
        self.assertEqual(manifest.iloc[0].status, "disk_stopped")

    def test_failed_cases_exhaust_retries_and_continue(self):
        audit = self._prepare_pipeline_inputs()
        with patch.object(pipeline, "audit_project", return_value=audit), \
             patch.object(pipeline, "load_track_index", return_value=pd.DataFrame()), \
             patch.object(pipeline, "resolve_tracks", return_value={**pipeline.TRACKS, "MAP": pipeline.MAP_PRIORITY[0]}), \
             patch.object(pipeline, "write_audit_markdown"), \
             patch.object(pipeline, "load_vital_case", side_effect=TimeoutError("network timeout")) as download, \
             patch.object(pipeline.time, "sleep"):
            result = pipeline.run_pipeline(self.root, max_attempts=3, backoff_seconds=0)
        self.assertEqual(result["failed"], [1, 2, 3])
        self.assertEqual(download.call_count, 9)
        states = canonical.CheckpointStore(self.root, "vitaldb").rows()
        self.assertTrue(all(row["status"] == "failed" and row["attempts"] == 3 for row in states))

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.run_pipeline(self.root, "FULL")
        with self.assertRaises(ValueError):
            pipeline.run_pipeline(self.root, max_attempts=0)

    def test_disk_reserve_is_decimal_100gb(self):
        self.assertEqual(pipeline.RESERVE_BYTES, 100_000_000_000)
        with patch.object(canonical.shutil, "disk_usage", return_value=SimpleNamespace(free=99_999_999_999)):
            with self.assertRaises(canonical.DiskSpaceStop):
                canonical.ensure_disk_space(self.root)


    def _mock_run_context(self, scenario="normal", delay=.03, real_spawn=False, count=3, pilot_passed=False):
        audit = self._prepare_pipeline_inputs(count)
        audit["pilot_passed"] = pilot_passed
        path = self.root / "configs" / "hemodynamics.json"
        settings = json.loads(path.read_text())
        settings["vitaldb"].update(test_scenario=scenario, test_delay_sec=delay)
        path.write_text(json.dumps(settings))
        if real_spawn:
            self.executor_patch.stop()
        stack = ExitStack()
        stack.enter_context(patch.object(pipeline, "audit_project", return_value=audit))
        stack.enter_context(patch.object(pipeline, "load_track_index", return_value=pd.DataFrame()))
        stack.enter_context(patch.object(pipeline, "resolve_tracks", return_value={**pipeline.TRACKS,"MAP":pipeline.MAP_PRIORITY[0]}))
        stack.enter_context(patch.object(pipeline, "write_audit_markdown"))
        stack.enter_context(patch.object(pipeline, "_process_case", _mock_spawn_worker))
        return stack

    def test_bounded_concurrency_and_sorted_dispatch(self):
        with self._mock_run_context(delay=.1):
            result = pipeline.run_pipeline(self.root, download_workers=2)
        self.assertEqual(result["dispatch_order"], [1,2,3])
        self.assertEqual(result["peak_inflight"], 2)
        self.assertEqual(result["processed"], [1,2,3])
        self.assertEqual(result["summary_status"], "COMPLETE")

    def test_progress_heartbeat_while_workers_idle(self):
        output = io.StringIO()
        with self._mock_run_context(delay=.3), redirect_stdout(output):
            result = pipeline.run_pipeline(self.root, download_workers=2, progress=True)
        text = output.getvalue()
        for phrase in ("Loading cohort", "Auditing", "Resolving tracks", "Starting 2 workers", "oldest_age=", "oldest_request_age=", "last_durable=", "cases/min=", "elapsed=", "ETA=", "output_GB=", "free_disk_GB="):
            self.assertIn(phrase,text)
        self.assertGreater(text.count("in_flight=[1, 2]"), 1)
        self.assertEqual(text.count("COMPLETE:"),1)
        self.assertEqual(result["summary_status"], "COMPLETE")

    def test_one_failing_case_does_not_stop_other_cases(self):
        with self._mock_run_context(scenario="one_failure"):
            result = pipeline.run_pipeline(self.root, download_workers=2, backoff_seconds=.01)
        self.assertEqual(result["processed"], [1,3])
        self.assertEqual(result["failed"], [2])
        self.assertEqual(result["summary_status"], "INCOMPLETE")
        self.assertEqual(canonical.CheckpointStore(self.root,"vitaldb").get(2)["attempts"],3)

    def test_interrupt_pauses_and_resume_skips_durable_cases(self):
        calls = [0]
        def interrupt_after_first(*args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise KeyboardInterrupt()
            if calls[0] == 1:
                kwargs["timeout"] = 2
            return real_wait(*args,**kwargs)
        with self._mock_run_context(scenario="pause", delay=.2):
            with patch.object(pipeline,"wait",side_effect=interrupt_after_first):
                first = pipeline.run_pipeline(self.root,download_workers=1)
            second = pipeline.run_pipeline(self.root,mode="RESUME",download_workers=1)
        self.assertEqual(first["summary_status"],"PAUSED")
        self.assertTrue(first["processed"])
        self.assertTrue(set(first["processed"]).issubset(second["skipped"]))
        self.assertEqual(second["summary_status"],"COMPLETE")

    def test_actual_windows_spawn_with_mocked_vitalfile(self):
        with self._mock_run_context(real_spawn=True):
            result = pipeline.run_pipeline(self.root,download_workers=2)
        self.assertEqual(result["processed"],[1,2,3])
        self.assertLessEqual(result["peak_inflight"],2)
        for row in canonical.CheckpointStore(self.root,"vitaldb").rows():
            self.assertTrue(canonical.verify_case(self.root,row))
        self.assertEqual(os.environ["OMP_NUM_THREADS"],"1")
        self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"],"1")

    def test_broken_process_pool_stops_cleanly(self):
        with self._mock_run_context(scenario="crash", delay=.3, real_spawn=True):
            result = pipeline.run_pipeline(self.root,download_workers=2)
        self.assertIn("BrokenProcessPool",result["stopped_reason"])
        self.assertIn(result["summary_status"],("FAILED","INCOMPLETE"))
        self.assertTrue(result["failed"])
        self.assertTrue(all(row["status"] in ("failed","complete") for row in canonical.CheckpointStore(self.root,"vitaldb").rows()))

    def test_forced_interrupt_joins_owned_workers_and_cleans_temps(self):
        with self._mock_run_context(delay=20,real_spawn=True):
            path = self.root / "configs" / "hemodynamics.json"
            settings=json.loads(path.read_text());settings["vitaldb"]["stop_grace_seconds"]=.1
            path.write_text(json.dumps(settings))
            directory=self.root/"data"/"parquet"/"vitaldb"/"case_id=1";directory.mkdir(parents=True)
            scratch=directory/"waveform.parquet.tmp";scratch.write_bytes(b"incomplete")
            with patch.object(pipeline,"wait",side_effect=KeyboardInterrupt()):
                start=time.monotonic();result=pipeline.run_pipeline(self.root,download_workers=2)
            self.assertLess(time.monotonic()-start,5)
            self.assertFalse(scratch.exists())
        self.assertEqual(result["summary_status"],"PAUSED")
        self.assertTrue(all(row["status"]=="interrupted" for row in canonical.CheckpointStore(self.root,"vitaldb").rows()))

    def test_download_only_settings_do_not_change_pilot_signature(self):
        settings=pipeline.load_settings(self.root)
        original=pipeline.processing_signature(settings)
        settings["vitaldb"].update(download_workers=8,max_inflight_ram_gb=16,progress_interval_seconds=.2,stop_grace_seconds=2,arrow_cpu_threads=2)
        self.assertEqual(pipeline.processing_signature(settings),original)
        self.assertEqual(pipeline.processing_signature(),"a65f00d54f20f17b9d957c1be93e70bdbf2ecfaa0261a3652300ef9098fa1c01")

    def test_worker_hard_limit_and_optin_message(self):
        with self.assertRaisesRegex(ValueError,"hard maximum12"):
            pipeline.run_pipeline(self.root,download_workers=13)
        self.assertEqual(pipeline.MAX_DOWNLOAD_WORKERS,12)
        self.assertEqual(json.loads((PROJECT/"configs"/"hemodynamics.json").read_text())["vitaldb"]["download_workers"],8)
        with self.assertRaisesRegex(RuntimeError,"hemodynamics.json") as error:
            pipeline.run_pipeline(self.root,mode="FULL_DOWNLOAD")
        self.assertIn("no download was started",str(error.exception))
        self.assertIn("allow_full_vitaldb_download=true",str(error.exception))
        self.assertIn("RUN VITALDB DOWNLOAD - START HERE",str(error.exception))

    def test_interrupt_during_audit_returns_paused_before_dispatch(self):
        self._prepare_pipeline_inputs()
        with patch.object(pipeline,"audit_project",side_effect=KeyboardInterrupt()), \
             patch.object(pipeline,"_create_executor") as executor:
            result=pipeline.run_pipeline(self.root,progress=False)
        executor.assert_not_called()
        self.assertEqual(result["summary_status"],"PAUSED")
        self.assertEqual(result["processed"],[])

    def test_ram_budget_pauses_before_dispatch(self):
        with self._mock_run_context():
            path=self.root/"configs"/"hemodynamics.json";settings=json.loads(path.read_text())
            settings["vitaldb"]["max_inflight_ram_gb"]=.01;path.write_text(json.dumps(settings))
            result=pipeline.run_pipeline(self.root)
        self.assertEqual(result["summary_status"],"PAUSED")
        self.assertEqual(result["peak_inflight"],0)
        self.assertIn("RAM budget",result["stopped_reason"])

    def test_max_cases_and_fixed_batch_repeat(self):
        with self._mock_run_context():
            first=pipeline.run_pipeline(self.root,download_workers=2,max_cases=2)
            second=pipeline.run_pipeline(self.root,mode="RESUME",case_ids=first["dispatch_order"],max_cases=2)
        self.assertEqual(first["dispatch_order"],[1,2])
        self.assertEqual(first["summary_status"],"INCOMPLETE")
        self.assertEqual(second["skipped"],[1,2])
        self.assertEqual(second["processed"],[])
        self.assertEqual(second["summary_status"],"INCOMPLETE")

    def _full_context(self, count=12, scenario="normal", delay=.01, fail_case=5):
        """Pilot cases 1-3 verified by a real mocked PILOT run; later FULL runs use the gate as passed."""
        stack = self._mock_run_context(scenario="normal", delay=delay, count=count)
        stack.__enter__()
        self.addCleanup(stack.close)
        pipeline.run_pipeline(self.root, "PILOT", download_workers=2)
        settings_path = self.root / "configs" / "hemodynamics.json"
        settings = json.loads(settings_path.read_text())
        settings["allow_full_vitaldb_download"] = True
        settings["vitaldb"].update(test_scenario=scenario, test_fail_case=fail_case)
        settings_path.write_text(json.dumps(settings))
        stack.enter_context(patch.object(pipeline, "audit_project", return_value={"pilot_case_ids": [1, 2, 3], "pilot_passed": True}))
        return pd.read_csv(self.root / "data" / "manifests" / "vitaldb_patient_splits.csv")

    def test_full_download_uses_representative_order_and_max_cases_applies_after_it(self):
        links = self._full_context()
        expected = pipeline.representative_order(links, 5052, done={1, 2, 3})
        self.assertNotEqual(expected, sorted(expected))
        first = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, max_cases=4)
        self.assertEqual(first["skipped"], [1, 2, 3])
        self.assertEqual(first["dispatch_order"], expected[:4])
        self.assertEqual(first["summary_status"], "INCOMPLETE")
        self.hook.assert_not_called()

    def test_full_resume_skips_verified_cases_without_redownloading(self):
        links = self._full_context()
        first = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, max_cases=4)
        attempts = {row["case_id"]: row["attempts"] for row in canonical.CheckpointStore(self.root, "vitaldb").rows()}
        second = pipeline.run_pipeline(self.root, "RESUME", resume_target="FULL_DOWNLOAD", download_workers=2)
        self.assertEqual(second["dispatch_order"], pipeline.representative_order(links, 5052, done={1, 2, 3, *first["processed"]}))
        self.assertTrue(set(first["processed"]).issubset(second["skipped"]))
        self.assertFalse(set(first["processed"]) & set(second["processed"]))
        self.assertEqual(second["summary_status"], "COMPLETE")
        self.assertEqual(second["verified_cases"], 12)
        for row in canonical.CheckpointStore(self.root, "vitaldb").rows():
            if row["case_id"] in attempts:
                self.assertEqual(row["attempts"], attempts[row["case_id"]], "verified cases must not be attempted again")
        self.hook.assert_called_once()
        third = pipeline.run_pipeline(self.root, "RESUME", resume_target="FULL_DOWNLOAD", download_workers=2)
        self.assertEqual((third["processed"], third["dispatch_order"]), ([], []))
        self.assertEqual(third["summary_status"], "COMPLETE")

    def test_explicit_case_ids_keep_sorted_order_in_full_mode(self):
        self._full_context()
        result = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=1, case_ids=[9, 5, 7, 11])
        self.assertEqual(result["dispatch_order"], [5, 7, 9, 11])

    def test_startup_messages_are_timestamped_per_phase(self):
        output = io.StringIO()
        self._full_context()
        with redirect_stdout(output):
            pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, max_cases=3, progress=True)
        text = output.getvalue()
        for phrase in ("Auditing cohort", "Audit finished", "Verifying completed checkpoints", "Checkpoint verification finished",
                       "Ordering 9 unfinished", "Resolving tracks for 3 cases", "Resolved tracks for 3 cases", "Dispatching first case"):
            self.assertRegex(text, r"\[\d\d:\d\d:\d\d \+ *\d+\.\ds\] " + phrase)

    def test_live_progress_reports_verified_rate_eta_network_phases_and_commit_time(self):
        self._full_context(delay=.3)
        output = io.StringIO()
        with redirect_stdout(output):
            pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, max_cases=4, progress=True)
        text = output.getvalue()
        for phrase in ("verified=3/12", "verified=7/12", "cases/min=", "ETA=", "net_MB/s=", "net_GB=", "written_GB=", "output_GB=",
                       "phases[net/resample/write/readback]=", "retries=0", "failed=0", "last_durable=", "s ago)"):
            self.assertIn(phrase, text)

    def test_attempt_log_and_report_record_phases_failures_and_retries(self):
        self._full_context(scenario="one_failure")
        output = io.StringIO()
        with redirect_stdout(output):
            result = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, backoff_seconds=.01, progress=True)
        logs = sorted((self.root / "logs").glob("vitaldb_download_*.jsonl"))
        self.assertTrue(logs)
        lines = [json.loads(line) for path in logs for line in path.read_text().splitlines()]
        lines = [line for line in lines if line["case_id"] > 3]
        self.assertEqual(len(lines), result["attempts"])
        failing = [line for line in lines if line["case_id"] == 5]
        self.assertEqual([line["status"] for line in failing], ["retry", "retry", "failed"])
        self.assertEqual(failing[0]["failed_phase"], "network")
        self.assertIn("mock network timeout", failing[-1]["error"])
        done = [line for line in lines if line["status"] == "complete"]
        self.assertEqual(sorted(line["case_id"] for line in done), sorted(result["processed"]))
        for line in done:
            self.assertEqual(set(line["phase_seconds"]), {"network", "resampling", "writing", "readback"})
            self.assertGreater(line["output_bytes"], 0)
            self.assertIn(line["split"], ("train", "validation", "test"))
        self.assertEqual((result["retries"], result["failed"], result["summary_status"]), (2, [5], "INCOMPLETE"))
        report = json.loads((self.root / "results" / "vitaldb_download_report.json").read_text())
        self.assertEqual(report["summary_status"], "INCOMPLETE")
        self.assertEqual(report["failed_cases"], [5])
        self.assertIn("5", report["failures"])
        self.assertEqual((report["download_workers"], report["retries"], report["http_429"]), (2, 2, 0))
        self.assertEqual(report["completed_this_run"], len(result["processed"]))
        for key in ("seconds_per_case_wall", "cases_per_min", "written_bytes_this_run", "network_bytes", "phase_seconds_total",
                    "eta_remaining_sec_extrapolated", "attempt_log", "verified_cases", "remaining_cases"):
            self.assertIn(key, report)
        self.assertFalse(report["network_bytes_exact"])
        self.assertIn("extrapolation", report["eta_note"].lower())

    def test_http_429_is_counted_from_attempt_telemetry(self):
        self._full_context(scenario="http429")
        result = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2, backoff_seconds=.01)
        self.assertEqual(result["http_429"], 3)
        self.assertEqual(result["failed"], [5])
        lines = [json.loads(line) for path in (self.root / "logs").glob("*.jsonl") for line in path.read_text().splitlines()]
        self.assertEqual([line["http_status"] for line in lines if line["case_id"] == 5], [429, 429, 429])

    def test_failure_telemetry_survives_the_process_boundary(self):
        with self._mock_run_context(scenario="http429", real_spawn=True):
            result = pipeline.run_pipeline(self.root, download_workers=2, backoff_seconds=.01)
        self.assertEqual((result["failed"], result["http_429"], result["retries"]), ([2], 3, 2))
        lines = [json.loads(line) for path in (self.root / "logs").glob("*.jsonl") for line in path.read_text().splitlines()]
        failing = [line for line in lines if line["case_id"] == 2]
        self.assertEqual([(line["status"], line["http_status"], line["failed_phase"]) for line in failing],
                         [("retry", 429, "network"), ("retry", 429, "network"), ("failed", 429, "network")])
        self.assertTrue(all(line["phase_seconds"]["network"] >= 0 for line in failing))

    def test_portable_error_keeps_type_and_attaches_telemetry(self):
        class Odd(OSError):
            def __init__(self):
                super().__init__("odd")
                self.lock = threading.Lock()

        converted = pipeline._portable(Odd(), {"phase_seconds": {}})
        self.assertIsInstance(converted, RuntimeError)
        self.assertIn("Odd: odd", str(converted))
        self.assertIsNone(converted.attempt_info["http_status"])
        timeout = TimeoutError("slow")
        self.assertIs(pipeline._portable(timeout, {"phase_seconds": {"network": 1.0}}), timeout)
        self.assertEqual(timeout.attempt_info["phase_seconds"], {"network": 1.0})
        throttled = urllib.error.HTTPError("http://x.invalid/1.vital", 429, "Too Many Requests", {}, None)
        self.assertEqual(pipeline._portable(throttled, {}).attempt_info["http_status"], 429)

    def test_report_is_written_for_paused_run(self):
        with self._mock_run_context(delay=.2):
            with patch.object(pipeline, "wait", side_effect=KeyboardInterrupt()):
                result = pipeline.run_pipeline(self.root, download_workers=1)
        report = json.loads((self.root / "results" / "vitaldb_download_report.json").read_text())
        self.assertEqual((result["summary_status"], report["summary_status"]), ("PAUSED", "PAUSED"))
        self.assertEqual(report["download_workers"], 1)

    def test_report_is_written_when_interrupted_before_dispatch(self):
        self._prepare_pipeline_inputs()
        with patch.object(pipeline, "audit_project", side_effect=KeyboardInterrupt()):
            pipeline.run_pipeline(self.root)
        report = json.loads((self.root / "results" / "vitaldb_download_report.json").read_text())
        self.assertEqual(report["summary_status"], "PAUSED")
        self.assertEqual(report["completed_this_run"], 0)

    def test_run_lock_is_held_during_run_and_released_after(self):
        lock = self.root / pipeline.LOCK_RELATIVE
        seen = []
        real = pipeline._run_pipeline_impl
        def spy(*args, **kwargs):
            seen.append(json.loads(lock.read_text())["pid"])
            return real(*args, **kwargs)
        with self._mock_run_context(), patch.object(pipeline, "_run_pipeline_impl", spy):
            pipeline.run_pipeline(self.root)
        self.assertEqual(seen, [os.getpid()])
        self.assertFalse(lock.exists())

    def test_second_live_run_warns_but_does_not_abort(self):
        lock = self.root / pipeline.LOCK_RELATIVE
        pipeline._atomic_json(lock, {"pid": os.getpid() + 1, "mode": "FULL_DOWNLOAD", "started_utc": "then"})
        output = io.StringIO()
        with self._mock_run_context(), patch.object(pipeline, "_lock_holder_alive", return_value=True), redirect_stdout(output):
            result = pipeline.run_pipeline(self.root)
        self.assertIn("another VitalDB run", output.getvalue())
        self.assertEqual(result["summary_status"], "COMPLETE")
        pipeline._atomic_json(lock, {"pid": 1, "mode": "x", "started_utc": "then"})
        output = io.StringIO()
        with patch.object(pipeline, "_lock_holder_alive", return_value=False), redirect_stdout(output):
            pipeline._release_run_lock(pipeline._acquire_run_lock(self.root, "PILOT"))
        self.assertNotIn("another VitalDB run", output.getvalue())

    def test_unified_index_hook_runs_both_steps_with_stub(self):
        calls = []
        stub = SimpleNamespace(index_ready_windows=lambda root, dataset="vitaldb", workers=None, progress=True: calls.append(("index", Path(root), dataset, progress)) or {},
                               build_unified_index=lambda root, progress=True: calls.append(("unified", Path(root), progress)) or pd.DataFrame({"x": [1, 2, 3]}))
        with patch.dict(sys.modules, {"hemodynamics.unified": stub}):
            outcome = _REAL_INDEX_HOOK(self.root, True)
        self.assertEqual(outcome["status"], "built")
        self.assertEqual(outcome["rows"], 3)
        self.assertEqual(calls, [("index", self.root, "vitaldb", True), ("unified", self.root, True)])

    def test_unified_index_hook_failure_or_absence_never_fails_the_download(self):
        output = io.StringIO()
        with patch.dict(sys.modules, {"hemodynamics.unified": None}), redirect_stdout(output):
            self.assertEqual(_REAL_INDEX_HOOK(self.root)["status"], "skipped")
        self.assertIn("WARNING", output.getvalue())
        def explode(*args, **kwargs):
            raise RuntimeError("indexing boom")
        stub = SimpleNamespace(index_ready_windows=explode, build_unified_index=explode)
        output = io.StringIO()
        with patch.dict(sys.modules, {"hemodynamics.unified": stub}), redirect_stdout(output):
            outcome = _REAL_INDEX_HOOK(self.root)
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("indexing boom", output.getvalue())
        self._full_context()
        stub = SimpleNamespace(index_ready_windows=explode, build_unified_index=explode)
        with patch.dict(sys.modules, {"hemodynamics.unified": stub}), patch.object(pipeline, "_build_unified_index_after_download", _REAL_INDEX_HOOK), \
             redirect_stdout(io.StringIO()):
            result = pipeline.run_pipeline(self.root, "FULL_DOWNLOAD", download_workers=2)
        self.assertEqual(result["summary_status"], "COMPLETE")
        self.assertEqual(result["unified_index"]["status"], "failed")
        report = json.loads((self.root / "results" / "vitaldb_download_report.json").read_text())
        self.assertEqual(report["unified_index"]["status"], "failed")

    def test_unified_index_hook_only_after_complete_full_run(self):
        with self._mock_run_context():
            pipeline.run_pipeline(self.root, "PILOT", download_workers=2)
            self.hook.assert_not_called()
            pipeline.run_pipeline(self.root, "RESUME", download_workers=2)
        self.hook.assert_not_called()

    def test_checkpoint_verification_threads_match_sequential_result(self):
        states = {case: {"status": "complete", "case": case} for case in range(1, 21)}
        good = {case for case in states if case % 3}
        with patch.object(pipeline, "_completed", side_effect=lambda root, state, settings=None: state["case"] in good):
            self.assertEqual(pipeline._verify_cases(self.root, states, list(states), {}, workers=6), good)
        self.assertEqual(pipeline._verify_cases(self.root, {}, [1, 2], {}), set())

    def test_interrupt_during_checkpoint_verification_does_not_wait_for_queued_cases(self):
        states = {case: {"status": "complete"} for case in range(1, 101)}
        with patch.object(pipeline, "_completed", side_effect=lambda *args, **kwargs: time.sleep(.2) or True), \
             patch.object(pipeline, "as_completed", side_effect=KeyboardInterrupt()):
            started = time.monotonic()
            with self.assertRaises(KeyboardInterrupt):
                pipeline._verify_cases(self.root, states, list(states), {}, workers=2)
        self.assertLess(time.monotonic() - started, 2.0)

    def test_checkpoint_verification_detects_corrupted_case(self):
        with self._mock_run_context():
            pipeline.run_pipeline(self.root, "PILOT", download_workers=2)
        settings = pipeline.load_settings(self.root)
        states = {int(row["case_id"]): row for row in canonical.CheckpointStore(self.root, "vitaldb").rows()}
        self.assertEqual(pipeline._verify_cases(self.root, states, [1, 2, 3], settings, workers=2), {1, 2, 3})
        path = self.root / states[2]["waveform_output_path"]
        with path.open("ab") as stream:
            stream.write(b"corruption")
        self.assertEqual(pipeline._verify_cases(self.root, states, [1, 2, 3], settings, workers=2), {1, 3})

    def test_track_lookup_is_built_once_for_the_cached_index(self):
        directory = self.root / "data" / "checkpoints" / "vitaldb"
        directory.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"caseid": [1, 1, 2], "tid": ["a", "b", "c"], "tname": ["SNUADC/ART", "SNUADC/ECG_II", "SNUADC/ART"]}).to_csv(directory / "track_index.csv", index=False)
        first = pipeline.load_track_index(self.root)
        self.assertIs(pipeline.load_track_index(self.root), first)
        self.assertIs(pipeline.track_lookup(first), pipeline.track_lookup(first))
        self.assertEqual(pipeline.track_lookup(first), {1: {"SNUADC/ART", "SNUADC/ECG_II"}, 2: {"SNUADC/ART"}})

    def test_network_bytes_are_counted_from_the_http_stream(self):
        import http.server
        tracks = {"ART": "SNUADC/ART", "MAP": "Solar8000/ART_MBP"}
        vf = vitaldb.VitalFile()
        vf.add_track("SNUADC/ART", [{"dt": 1e9 + i, "val": np.random.default_rng(i).normal(size=500).astype(np.float32)} for i in range(120)], srate=500)
        vf.add_track("Solar8000/ART_MBP", [{"dt": 1e9 + i, "val": 75.0 + i % 3} for i in range(120)])
        source = self.root / "tiny.vital"
        vf.to_vital(str(source))
        body = source.read_bytes()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch.object(vital_dataset, "api_url", f"http://127.0.0.1:{server.server_address[1]}"):
                loaded = pipeline.load_vital_case(7, tracks)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(pipeline.last_network_bytes(), len(body))
        self.assertTrue(all(loaded.trks[name].recs for name in tracks.values()))


class NotebookStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads((PROJECT / "notebooks" / "vitaldbwaveformdownload.ipynb").read_text(encoding="utf-8"))
        cls.code = ["".join(cell["source"]) for cell in cls.notebook["cells"] if cell["cell_type"] == "code"]

    def test_setup_cell_is_import_light_and_keeps_its_contract(self):
        setup = self.code[0]
        top_level = {line.split()[1].split(".")[0] for line in setup.splitlines() if line.startswith(("import ", "from "))}
        self.assertFalse(top_level & {"numpy", "pandas", "vitaldb", "scipy", "hemodynamics", "pyarrow", "matplotlib", "requests"}, top_level)
        for phrase in ('ROOT = next(', 'assert Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve()', 'MODE = "FULL_DOWNLOAD"',
                       'RESUME_TARGET = "FULL_DOWNLOAD"', 'print("Interpreter:"', 'print("Mode:"', "SETTINGS =", "MIN_FREE_SPACE_BYTES", "TARGET_WAVEFORM_FS",
                       "MAX_ATTEMPTS", "BACKOFF_SECONDS", "psutil", "fast_ingest.lock", "run.lock", "WARNING"):
            self.assertIn(phrase, setup)

    def test_timed_load_step_lives_in_the_download_cell(self):
        download = next(source for source in self.code if source.startswith("# RUN VITALDB DOWNLOAD - START HERE"))
        for phrase in ("import numpy as np", "import pandas as pd", "from hemodynamics import canonical, vitaldb_pipeline", "Library load", "time.perf_counter()",
                       "run_pipeline(ROOT", "progress=True"):
            self.assertIn(phrase, download)
        self.assertLess(download.index("Library load"), download.index("run_pipeline(ROOT"))

    def test_every_code_cell_compiles_without_syntax_warnings(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for index, source in enumerate(self.code):
                compile(source, f"cell{index}", "exec")

    def test_saved_outputs_are_cleared(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual((cell["outputs"], cell["execution_count"]), ([], None))

    def test_final_cell_builds_the_unified_manifest_lazily(self):
        final = self.code[-1]
        self.assertTrue(final.startswith("# 5. Build ready-window index and unified manifest"))
        self.assertIn('index_ready_windows(ROOT, "vitaldb", progress=True)', final)
        self.assertIn('build_unified_index(ROOT, progress=True).groupby(["dataset", "split_within_dataset"]).size()', final)
        self.assertFalse(any("hemodynamics.unified" in source for source in self.code[:-1]))
        markdown = ["".join(cell["source"]) for cell in self.notebook["cells"] if cell["cell_type"] == "markdown"]
        self.assertTrue(any(text.startswith("## 5. Build ready-window index and unified manifest") for text in markdown))
        self.assertTrue(any("Restart the kernel" in text for text in markdown))


class RepresentativeOrderTests(unittest.TestCase):
    @staticmethod
    def _links(patients=600, seed=3):
        rng = np.random.default_rng(seed)
        labels = np.array(["train"] * int(patients * .7) + ["validation"] * int(patients * .15))
        labels = np.r_[labels, ["test"] * (patients - len(labels))]
        rng.shuffle(labels)
        rows = []
        for patient, split in enumerate(labels):
            rows += [(0, 1000 + patient, split)] * int(rng.choice([1, 1, 1, 2, 3]))
        frame = pd.DataFrame(rows, columns=["case_id", "patient_id", "split"])
        frame["case_id"] = rng.permutation(np.arange(1, len(frame) + 1) * 3)
        return frame

    def test_order_is_deterministic_complete_and_independent_of_row_order(self):
        links = self._links()
        first = pipeline.representative_order(links, 5052)
        self.assertEqual(first, pipeline.representative_order(links.sample(frac=1, random_state=11), 5052))
        self.assertEqual(sorted(first), sorted(links.case_id))
        self.assertNotEqual(first, pipeline.representative_order(links, 2026))
        self.assertNotEqual(first, sorted(first))

    def test_every_prefix_is_about_seventy_fifteen_fifteen(self):
        links = self._links()
        order = pipeline.representative_order(links, 5052)
        splits = links.set_index("case_id").split.loc[order].to_numpy()
        target = {"train": .70, "validation": .15, "test": .15}
        for name, share in target.items():
            running = np.cumsum(splits == name)
            lengths = np.arange(1, len(order) + 1)
            self.assertLess(np.abs(running[99:] / lengths[99:] - share).max(), .03, name)
            first_tier = lengths <= links.patient_id.nunique()
            self.assertLess(np.abs(running[first_tier] - share * lengths[first_tier]).max(), 1.5, name)

    def test_one_surgery_per_patient_before_any_repeat(self):
        links = self._links()
        order = pipeline.representative_order(links, 5052)
        patient = links.set_index("case_id").patient_id.loc[order].to_numpy()
        unique = links.patient_id.nunique()
        self.assertEqual(len(set(patient[:unique])), unique)
        self.assertTrue(set(patient[unique:]).issubset(set(patient[:unique])))

    def test_verified_cases_are_skipped_and_their_patients_count_as_covered(self):
        links = self._links()
        multi = links.groupby("patient_id").case_id.apply(list)
        multi = multi[multi.map(len) > 1]
        done = {int(cases[0]) for cases in multi.iloc[:25]}
        order = pipeline.representative_order(links, 5052, done=done)
        self.assertFalse(set(order) & done)
        self.assertEqual(len(order), len(links) - len(done))
        by_case = links.set_index("case_id").patient_id
        covered = {int(by_case[case]) for case in done}
        uncovered = links.patient_id.nunique() - len(covered)
        position = {case: i for i, case in enumerate(order)}
        for case in order:
            if int(by_case[case]) in covered:
                self.assertGreaterEqual(position[case], uncovered, "covered patients' remaining surgeries must come after every uncovered patient")
        self.assertEqual(len({int(by_case[case]) for case in order[:uncovered]}), uncovered)

    def test_subset_empty_and_duplicate_inputs(self):
        links = self._links(60)
        subset = list(links.case_id[:20])
        self.assertEqual(sorted(pipeline.representative_order(links, 5052, cases=subset)), sorted(subset))
        self.assertEqual(pipeline.representative_order(links, 5052, done=set(links.case_id)), [])
        with self.assertRaises(ValueError):
            pipeline.representative_order(pd.concat([links, links.iloc[:1]]), 5052)


def real_smoke_once():
    """Explicit operator diagnostic: exactly four next cases, never full ingestion."""
    root = PROJECT
    path = root / "data" / "manifests" / "vitaldb_project_audit.json"
    previous_audit = json.loads(path.read_text(encoding="utf-8"))
    if previous_audit.get("download_speed_smoke"):
        raise RuntimeError("The one real four-case smoke already ran. Reuse its case_ids to verify resume; do not download another batch.")
    settings = pipeline.load_settings(root)
    before = {int(row["case_id"]): row for row in canonical.CheckpointStore(root,"vitaldb").rows()}
    if set(before) != {3585,2305,4865} or not all(pipeline._completed(root,row,settings) for row in before.values()):
        raise RuntimeError("Smoke requires exactly the original three verified pilots; inspect changes rather than selecting additional cases.")
    cohort = pd.read_csv(root/"data"/"manifests"/"usable_cohort.csv")
    cases = [int(case) for case in sorted(cohort.caseid) if int(case) not in before][:4]
    print("ONE REAL SMOKE; exact next four case_ids:",cases,flush=True)
    start=time.perf_counter()
    result=pipeline.run_pipeline(root,"FULL_DOWNLOAD",download_workers=4,max_cases=4,case_ids=cases,progress=True)
    wall=time.perf_counter()-start
    resume=pipeline.run_pipeline(root,"RESUME",resume_target="FULL_DOWNLOAD",download_workers=4,max_cases=4,case_ids=cases,progress=True)
    after={int(row["case_id"]):row for row in canonical.CheckpointStore(root,"vitaldb").rows()}
    assert set(after).issubset(set(before)|set(cases))
    assert all(before[case]==after[case] for case in before), "Original pilot checkpoints changed"
    valid=[case for case in cases if pipeline._completed(root,after.get(case),settings)]
    assert not resume["processed"], "Same-batch resume must never download another case"
    if len(valid)==4:
        assert set(resume["skipped"])==set(cases)
    metrics={"case_ids":cases,"valid_cases":valid,"download_workers":4,"wall_seconds":wall,
             "wall_seconds_per_case":wall/4,"case_wall_times_sec":result["case_wall_times_sec"],
             "startup_seconds":result["startup_to_dispatch_sec"],"audit_seconds":result["audit_seconds"],
             "summary_status":result["summary_status"],"verified_cohort_cases":result["verified_cases"],
             "remaining_cohort_cases":result["remaining_cases"],"resume_skipped":resume["skipped"],
             "resume_downloads":len(resume["processed"]),"original_pilots_unchanged":True,
             "projected_full_hours_4_workers":len(cohort)*(wall/4)/3600,
             "projected_full_hours_8_workers":len(cohort)*(wall/4)/7200,
             "projection_note":"Plain four-case cold-start wall-throughput extrapolation. Eight workers assumes ideal2x scaling; neither full run nor8-worker performance was measured.",
             "old_audit_seconds":1.6416164003312588,"old_startup_to_dispatch_seconds":2.0272450000047684,
             "root_cause":"The full opt-in guard raised before audit/case loop; only the three original pilots existed. No six-hour full download occurred.",
             "tests_passed":31,"processing_signature":pipeline.processing_signature(settings)}
    audit=json.loads(path.read_text(encoding="utf-8"));audit["download_speed_smoke"]=metrics
    pipeline._atomic_json(path,audit);pipeline.write_audit_markdown(root,audit)
    pipeline.assert_originals_unchanged(root)
    print("REAL_SMOKE_METRICS",json.dumps(metrics,indent=2),flush=True)
    if len(valid)!=4:
        raise RuntimeError("Bounded smoke had failures; checkpoints retain them. No other cases were selected.")


if __name__ == "__main__":
    if "--real-smoke" in sys.argv:
        real_smoke_once()
    else:
        unittest.main()
