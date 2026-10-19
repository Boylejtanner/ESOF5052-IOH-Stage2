import _thread
import base64
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hemodynamics import mover_fast
from hemodynamics.canonical import DiskSpaceStop, load_case
from hemodynamics.mover_pipeline import audit_linkage, discover, pilot, process_selected, select_cohort

SURGERY_START = datetime(2021, 1, 1, 17, tzinfo=timezone.utc)


def make_xml(start=SURGERY_START, kind="s5", seconds=60, pulsatile=False):
    packets = []
    specs = ((("INVP1", 100, .01), ("ECG1", 300, 1.), ("PLETH", 100, 1.)) if kind == "s5"
             else (("GE_ART", 180, .25), ("GE_ECG", 180, 1.)))
    device = ('<device make="GE_S5" model="GE_S5">' if kind == "s5"
              else '<device make="AnalogWaves" model="GE_Solar_8000i">')
    for second in range(seconds):
        groups = []
        for name, fs, gain in specs:
            if name in ("INVP1", "GE_ART"):
                if pulsatile and name == "INVP1":
                    # 80 mmHg mean with a 1.2 Hz, +/-20 mmHg pulse (raw units are 0.01 mmHg).
                    phase = (second * fs + np.arange(fs)) / fs
                    raw = np.rint((80 + 20 * np.sin(2 * np.pi * 1.2 * phase)) / .01).astype("<i2")
                else:
                    raw = np.full(fs, 8000 if name == "INVP1" else 320, dtype="<i2")
            else:
                raw = np.rint(np.sin(np.arange(fs) / fs * 2 * np.pi * (1 + second % 3)) * 100).astype("<i2")
            fields = {"Wave": base64.b64encode(raw.tobytes()).decode(), "Points": fs, "PointsBytes": 2,
                      "Min": -500, "Max": 1500, "Offset": 0, "Gain": gain, "Hz": fs}
            groups.append(f'<mg name="{name}">' + "".join(f'<m name="{k}">{v}</m>' for k, v in fields.items()) + "</mg>")
        stamp = (start + timedelta(seconds=second)).isoformat().replace("+00:00", "Z")
        numeric = '<m name="IBP_MEAN">80</m>' if kind == "s5" else ""
        packets.append(f'<cpc datetime="{stamp}">{device}<measurements>' + "".join(groups) + numeric + "</measurements></device></cpc>")
    return ('<cpcArchive duration="30">' + "".join(packets) + "</cpcArchive>").encode()


def make_tar(path, items):
    with tarfile.open(path, "w:gz") as archive:
        for name, data in items:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))


def pid(index):
    return f"{index:016x}"


def build_fixture(root, patients=4, extras=True, fast_overrides=None, corrupt_patient=None, single_archive=False, corrupt_all=False,
                  config_overrides=None, seconds=60, surgery_minutes=1, pulsatile=False):
    """patients are spread over the 3 archives; every patient has one surgery of ``seconds`` seconds."""
    root = Path(root)
    (root / "configs").mkdir(parents=True)
    source = root / "data" / "reference" / "mover"
    source.mkdir(parents=True)
    for folder in ("data/manifests", "results", "figures"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    config = json.loads((Path(__file__).resolve().parents[2] / "configs" / "hemodynamics.json").read_text())
    config.update(min_free_space_gb=0, minimum_duration_sec=1, map_min_valid_points=10, discovery_members_per_archive=2,
                  pilot_cases=3, allow_full_archive_scan=True, allow_full_process=True, eligibility_mode="legacy_case")
    config.update(config_overrides or {})
    config["fast_ingest"].update(decode_workers=2, finalize_workers=2, progress_interval_sec=.2, commit_interval_sec=.2,
                                 canary_min_fragments=2, **(fast_overrides or {}))
    (root / "configs" / "hemodynamics.json").write_text(json.dumps(config))
    members = {name: [] for name in config["mover_archives"]}
    crosswalk, clinical = [], []
    for i in range(1, patients + 1):
        archive = config["mover_archives"][0 if single_archive else (i - 1) % 3]
        crosswalk.append({"LOG_ID": f"{i + 100:016x}", "PAT_ID": pid(i), "MRN": f"{i + 500:016x}"})
        clinical.append({"LOG_ID": f"{i + 100:016x}", "MRN": f"{i + 500:016x}", "IN_OR_DTTM": "01/01/21 09:00",
                         "OUT_OR_DTTM": f"01/01/21 {9 + surgery_minutes // 60:02d}:{surgery_minutes % 60:02d}"})
        payload = make_xml(seconds=seconds, pulsatile=pulsatile)
        if corrupt_all or corrupt_patient == i:
            payload = payload[:len(payload) // 2]
        members[archive].append((f"Waveforms/{pid(i)}CB-2021-01-01-17-00-00-000Z.xml", payload))
        if extras:
            members[archive].append((f"Waveforms/{pid(i)}IP-2021-01-01-17-00-00-000Z.xml", make_xml(kind="ge")))
            members[archive].append((f"Waveforms/{pid(i)}CB-2021-03-01-10-00-00-000Z.xml", make_xml(datetime(2021, 3, 1, 10, tzinfo=timezone.utc))))
    if extras:
        members[config["mover_archives"][0]].append((f"Waveforms/{'f' * 16}CB-2021-01-01-17-00-00-000Z.xml", make_xml()))
    for archive, items in members.items():
        make_tar(source / archive, items or [("Waveforms/empty.txt", b"")])
    pd.DataFrame(crosswalk).to_csv(source / "EPIC_MRN_PAT_ID.csv", index=False)
    make_tar(source / "EPIC_EMR.tar.gz", [("EPIC_EMR/EMR/patient_information.csv", pd.DataFrame(clinical).to_csv(index=False).encode())])
    return config


class FastIngestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def second_root(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        return Path(other.name)

    def test_full_run_is_complete_verified_and_skips_irrelevant_members(self):
        build_fixture(self.root)
        result = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "COMPLETE", result.get("reason"))
        counters = result["ingest"]["counters"]
        self.assertEqual(sum(c.get("skipped_patient", 0) for c in counters.values()), 1)
        self.assertEqual(sum(c.get("skipped_no_pleth", 0) for c in counters.values()), 4)
        self.assertEqual(sum(c.get("skipped_no_overlap", 0) for c in counters.values()), 4)
        self.assertEqual(sum(c.get("decoded", 0) for c in counters.values()), 4)
        selected = pd.read_parquet(self.root / "data/manifests/mover_selected_cohort.parquet")
        self.assertEqual(len(selected), 4)
        splits = pd.read_parquet(self.root / "data/manifests/mover_patient_splits.parquet")
        self.assertEqual(sorted(splits.split.unique()), ["test", "train", "validation"])
        for case_id in selected.case_id:
            case = load_case(self.root, "mover", case_id)
            for stream in ("ART", "ECG", "PPG", "MAP"):
                self.assertTrue(np.isfinite(case[stream]).any())
            self.assertEqual(case["split"], selected.loc[selected.case_id == case_id, "split"].iloc[0])
        self.assertFalse((self.root / "data/parquet/mover/staging").exists())
        readiness = pd.read_csv(self.root / "data/manifests/mover_raw_archive_deletion_readiness.csv")
        self.assertTrue(readiness.safe_to_delete.all(), readiness.reason.tolist())
        self.assertTrue(all((self.root / "data/reference/mover" / a).is_file() for a in ("epic_wave_1_v2.tar.gz", "epic_wave_2_v2.tar.gz", "epic_wave_3_v2.tar.gz")))
        eda = json.loads((self.root / "results/mover_fast_eda.json").read_text())
        self.assertEqual(eda["completed_cases"], 4)
        from hemodynamics.validate_data import validate_outputs
        validation = validate_outputs(self.root)["datasets"]["mover"]
        self.assertTrue(validation["ready_for_ml_bshi"], validation)
        self.assertEqual(validation["valid_completed_cases"], 4)
        self.assertEqual(eda["patient_intersections"], {"train_validation": 0, "train_test": 0, "validation_test": 0})
        again = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(again["status"], "COMPLETE")
        self.assertEqual(again.get("reason"), "already finalized")

    def test_outputs_equal_the_trusted_legacy_pipeline(self):
        build_fixture(self.root, patients=3, extras=False)
        legacy = self.second_root()
        shutil.copytree(self.root, legacy, dirs_exist_ok=True)
        fast = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(fast["status"], "COMPLETE", fast.get("reason"))
        audit_linkage(legacy)
        discover(legacy, complete_scan=True)
        pilot(legacy, limit=3)
        select_cohort(legacy)
        process_selected(legacy)
        selected = pd.read_parquet(self.root / "data/manifests/mover_selected_cohort.parquet")
        self.assertEqual(len(selected), 3)
        for case_id in selected.case_id:
            a, b = load_case(self.root, "mover", case_id), load_case(legacy, "mover", case_id)
            for key in ("waveform_time_sec", "map_time_sec", "ART", "ECG", "PPG", "MAP"):
                np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    def test_results_do_not_depend_on_worker_count(self):
        build_fixture(self.root)
        other = self.second_root()
        shutil.copytree(self.root, other, dirs_exist_ok=True)
        self.assertEqual(mover_fast.run(self.root, workers=1, progress=False)["status"], "COMPLETE")
        self.assertEqual(mover_fast.run(other, workers=3, progress=False)["status"], "COMPLETE")
        a = pd.read_parquet(self.root / "data/manifests/mover_selected_cohort.parquet").sort_values("case_id")
        b = pd.read_parquet(other / "data/manifests/mover_selected_cohort.parquet").sort_values("case_id")
        self.assertEqual(a[["case_id", "patient_id", "split"]].values.tolist(), b[["case_id", "patient_id", "split"]].values.tolist())
        for case_id in a.case_id:
            np.testing.assert_array_equal(load_case(self.root, "mover", case_id)["ART"], load_case(other, "mover", case_id)["ART"])

    def test_stop_hook_pauses_then_resume_matches_uninterrupted(self):
        build_fixture(self.root, single_archive=True)
        control = self.second_root()
        shutil.copytree(self.root, control, dirs_exist_ok=True)
        paused = mover_fast.run(self.root, workers=2, progress=False, _stop_after=2)
        self.assertEqual(paused["status"], "PAUSED", paused.get("reason"))
        self.assertFalse((self.root / "data/manifests/mover_selected_cohort.parquet").exists())
        refused = mover_fast.finalize(self.root, workers=2, progress=False)
        self.assertEqual(refused["status"], "INCOMPLETE")
        self.assertFalse((self.root / "data/manifests/mover_selected_cohort.parquet").exists())
        resumed = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(resumed["status"], "COMPLETE", resumed.get("reason"))
        self.assertGreater(sum(c.get("resume_skipped", 0) for c in resumed["ingest"]["counters"].values()), 0)
        self.assertEqual(mover_fast.run(control, workers=2, progress=False)["status"], "COMPLETE")
        a = pd.read_parquet(self.root / "data/manifests/mover_selected_cohort.parquet").sort_values("case_id")
        b = pd.read_parquet(control / "data/manifests/mover_selected_cohort.parquet").sort_values("case_id")
        self.assertEqual(a.case_id.tolist(), b.case_id.tolist())
        for case_id in a.case_id:
            np.testing.assert_array_equal(load_case(self.root, "mover", case_id)["PPG"], load_case(control, "mover", case_id)["PPG"])

    def test_keyboard_interrupt_is_a_clean_pause_and_releases_the_lock(self):
        build_fixture(self.root, fast_overrides={"debug_task_delay_sec": 0.6})
        timer = threading.Timer(2.0, _thread.interrupt_main)
        timer.start()
        try:
            paused = mover_fast.ingest(self.root, workers=2, progress=False)
        finally:
            timer.cancel()
        self.assertEqual(paused["status"], "PAUSED", paused.get("reason"))
        self.assertFalse((self.root / "data/checkpoints/mover/fast_ingest.lock").exists())
        config = json.loads((self.root / "configs/hemodynamics.json").read_text())
        config["fast_ingest"]["debug_task_delay_sec"] = 0.0
        (self.root / "configs/hemodynamics.json").write_text(json.dumps(config))
        self.assertEqual(mover_fast.run(self.root, workers=2, progress=False)["status"], "COMPLETE")

    def test_unreadable_pleth_member_quarantines_only_its_surgery(self):
        build_fixture(self.root, patients=4, corrupt_patient=1)
        result = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "INCOMPLETE", result.get("reason"))
        cases = pd.read_parquet(self.root / "data/manifests/mover_all_cases.parquet")
        bad = cases.loc[cases.patient_id == pid(1)].iloc[0]
        self.assertFalse(bad.final_usable)
        self.assertIn("quarantined_unreadable_pleth_member", bad.exclusion_reason)
        selected = pd.read_parquet(self.root / "data/manifests/mover_selected_cohort.parquet")
        self.assertEqual(len(selected), 3)
        self.assertTrue((self.root / "data/manifests/mover_processing_errors.csv").read_text().strip())
        readiness = pd.read_csv(self.root / "data/manifests/mover_raw_archive_deletion_readiness.csv").set_index("archive")
        self.assertFalse(readiness.loc["epic_wave_1_v2.tar.gz", "safe_to_delete"])
        self.assertIn("unreadable PLETH", readiness.loc["epic_wave_1_v2.tar.gz", "reason"])

    def test_gzip_checksum_failure_never_certifies_an_archive(self):
        build_fixture(self.root)
        path = self.root / "data/reference/mover/epic_wave_1_v2.tar.gz"
        data = bytearray(path.read_bytes())
        data[-8] ^= 1
        path.write_bytes(data)
        result = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "INCOMPLETE", result.get("reason"))
        self.assertFalse((self.root / "data/manifests/mover_selected_cohort.parquet").exists())

    def test_systematic_decode_failures_stop_the_run_early(self):
        build_fixture(self.root, patients=30, extras=False, single_archive=True, corrupt_all=True)
        result = mover_fast.ingest(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "FAILED", result.get("reason"))
        self.assertIn("Systematic decode failures", result["reason"])
        self.assertFalse((self.root / "data/manifests/mover_selected_cohort.parquet").exists())

    def test_tiny_payload_queue_backpressure_does_not_deadlock(self):
        build_fixture(self.root, fast_overrides={"max_inflight_payload_mb": 0, "max_inflight_tasks": 1})
        self.assertEqual(mover_fast.run(self.root, workers=2, progress=False)["status"], "COMPLETE")

    def test_disk_guard_pauses_without_losing_state(self):
        build_fixture(self.root)
        with patch("hemodynamics.mover_fast.ensure_disk_space", side_effect=DiskSpaceStop("test disk guard")):
            result = mover_fast.ingest(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "PAUSED")
        self.assertIn("test disk guard", result["reason"])
        self.assertEqual(mover_fast.run(self.root, workers=2, progress=False)["status"], "COMPLETE")

    def test_lock_refuses_a_live_second_run_but_takes_over_a_dead_one(self):
        import psutil
        lock = self.root / "data/checkpoints/mover/fast_ingest.lock"
        lock.parent.mkdir(parents=True)
        lock.write_text(json.dumps({"pid": 2_000_000_000, "create_time": 1.0}))
        with mover_fast._RunLock(self.root):
            self.assertEqual(json.loads(lock.read_text())["pid"], psutil.Process().pid)
        self.assertFalse(lock.exists())
        parent = psutil.Process().parent()
        lock.write_text(json.dumps({"pid": parent.pid, "create_time": parent.create_time()}))
        with self.assertRaises(RuntimeError):
            with mover_fast._RunLock(self.root):
                pass
        self.assertTrue(lock.exists())

    def test_canary_stops_a_systematically_wrong_scale(self):
        config = build_fixture(self.root)
        fast = mover_fast._fast(config)
        context = mover_fast._Context(self.root, config, fast, {}, {}, "sig", None)
        good = {"art_n": 5, "art_median": 90., "ecg_n": 5, "ppg_n": 5, "ppg_std": 3., "map_n": 5, "map_median": 85., "all_four": True}
        mover_fast._canary(context, good, False)
        mover_fast._canary(context, good, False)
        self.assertEqual(context.canary["status"], "passed")
        context = mover_fast._Context(self.root, config, fast, {}, {}, "sig", None)
        bad = dict(good, art_median=9000.)
        for _ in range(3):
            mover_fast._canary(context, bad, False)
        self.assertEqual(context.canary["status"], "FAILED")
        self.assertTrue(context.fatal and context.stop.is_set())

    def test_canary_tolerates_real_non_arterial_channels_but_catches_scale_faults(self):
        config = build_fixture(self.root)
        fast = dict(mover_fast._fast(config), canary_min_fragments=20)   # production decision window, not the fixture's 2
        live = {"art_n": 5, "art_median": 75., "ecg_n": 5, "ppg_n": 5, "ppg_std": 3., "map_n": 5, "map_median": 80., "all_four": True}
        # Values measured on real MOVER fragments in the failed run: the waveform and the numeric MAP agree, the line is just not arterial.
        real_non_arterial = [
            dict(live, art_median=2.8, map_median=3.0),                    # flat line near 3 mmHg
            dict(live, art_median=6.7, map_median=7.0),                    # CVP-like trace
            dict(live, art_median=-2.9, map_n=0, map_median=None),         # zeroed/disconnected line, no MAP numerics
            dict(live, art_median=245.2, map_n=0, map_median=None),        # saturated flat line, no MAP numerics
        ]
        for sanity in real_non_arterial:
            self.assertFalse(mover_fast._scale_violation(sanity, fast), sanity)
        # 3 of the first 20 fragments being non-arterial must no longer abort the pass.
        context = mover_fast._Context(self.root, config, fast, {}, {}, "sig", None)
        for sanity in [live] * 17 + real_non_arterial[:3]:
            mover_fast._canary(context, sanity, False)
        self.assertEqual(context.canary["status"], "passed")
        self.assertEqual(context.canary["violations"], 0)
        # Genuine decode faults are still violations: impossible magnitude, gain 100x too small/large, offset, broken PPG.
        faults = [dict(live, art_median=9000., map_median=9000.),          # impossible magnitude even if both agree
                  dict(live, art_median=0.8, map_median=80.),              # gain 100x too small: waveform vs numeric MAP
                  dict(live, art_median=8000., map_median=80.),            # gain 100x too large
                  dict(live, art_median=130., map_median=80.),             # 50 mmHg offset
                  dict(live, map_median=-400.),
                  dict(live, ppg_std=float("nan")), dict(live, ppg_std=None)]
        for sanity in faults:
            self.assertTrue(mover_fast._scale_violation(sanity, fast), sanity)
        context = mover_fast._Context(self.root, config, fast, {}, {}, "sig", None)
        for sanity in [live] * 17 + faults[1:4]:
            mover_fast._canary(context, sanity, False)
        self.assertEqual(context.canary["status"], "FAILED")

    def test_member_span_and_window_matching(self):
        payload = make_xml(seconds=30)
        start, end = mover_fast.member_span(payload)
        self.assertAlmostEqual(start, SURGERY_START.timestamp())
        self.assertAlmostEqual(end, SURGERY_START.timestamp() + 30)
        self.assertIsNone(mover_fast.member_span(b"<cpcArchive></cpcArchive>"))
        entries = [("a", start - 10, start + 5), ("b", end + 500, end + 900), ("c", end - 1, end + 100)]
        self.assertEqual([e[0] for e in mover_fast.match_cases(entries, (start, end), 120)], ["a", "c"])
        self.assertEqual(len(mover_fast.match_cases(entries, None, 120)), 3)
        self.assertEqual([e[0] for e in mover_fast.match_cases(entries, (start, end), 1000)], ["a", "b", "c"])

    def test_patient_level_selection_is_deterministic_and_size_bounded(self):
        config = {"max_mover_size_gb": 1.0, "target_mover_size_gb": 0.5, "random_seed": 5052}
        rows = [{"patient_id": f"p{i:03d}", "case_id": f"c{i}", "estimated_bytes": 40_000_000} for i in range(60)]
        frame = pd.DataFrame(rows)
        chosen, mode = mover_fast.select_patients(frame, config)
        self.assertEqual(mode, "seeded_patient_subset_to_target_size")
        self.assertLessEqual(len(chosen) * 40_000_000, 500_000_000)
        shuffled, _ = mover_fast.select_patients(frame.sample(frac=1, random_state=1), config)
        self.assertEqual(chosen, shuffled)
        small, mode = mover_fast.select_patients(frame.head(10), config)
        self.assertEqual((len(small), mode), (10, "all_eligible_patients"))


WINDOW_LADDER = [{"name": "R1", "theta": 0.75, "min_ready_span_min": 3}, {"name": "R2", "theta": 0.70, "min_ready_span_min": 3},
                 {"name": "R3", "theta": 0.60, "min_ready_span_min": 3}, {"name": "R4", "theta": 0.50, "min_ready_span_min": 3}]


def window_config(**targets):
    return {"eligibility_mode": "window_v1", "eligibility_ladder": WINDOW_LADDER,
            "eligibility_targets": {"min_cases": 2, "min_patients": 2, "floor_cases": 1, **targets}}


class WindowModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        return build_fixture(self.root, seconds=900, surgery_minutes=15, pulsatile=True, **kwargs)

    def test_window_mode_run_writes_cases_stats_windows_and_frozen_cohort(self):
        self.build(config_overrides=window_config())
        result = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "COMPLETE", result.get("reason"))
        info = result["window_eligibility"]
        self.assertEqual(info["rung"], "R1")
        manifests = self.root / "data" / "manifests"
        stats = pd.read_parquet(manifests / "mover_case_window_stats.parquet")
        self.assertEqual(len(stats), 4)
        self.assertTrue((stats.ready_span_min_theta_75 >= 3).all())
        windows = pd.read_parquet(manifests / "ready_windows_mover.parquet")
        self.assertEqual(sorted(windows.case_id.unique()), sorted(stats.case_id))
        self.assertTrue(windows.is_ready.any())
        frozen = json.loads((manifests / "mover_cohort_frozen.json").read_text())
        self.assertEqual((frozen["selected_cases"], frozen["rung"]), (4, "R1"))
        self.assertEqual(len(frozen["patient_list_sha256"]), 64)
        selected = pd.read_parquet(manifests / "mover_selected_cohort.parquet")
        self.assertEqual(len(selected), 4)
        for case_id, split in zip(selected.case_id, selected.split):
            self.assertEqual(load_case(self.root, "mover", case_id)["split"], split)
        archives = mover_fast._State(self.root)
        try:
            self.assertTrue(all(a.get("ingest_profile") == mover_fast.INGEST_PROFILE for a in archives.archives().values()))
        finally:
            archives.close()
        again = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(again.get("reason"), "already finalized")

    def test_legacy_completed_marker_does_not_short_circuit_and_valid_cases_are_reused(self):
        self.build()
        first = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(first["status"], "COMPLETE", first.get("reason"))
        case_dirs = sorted((self.root / "data" / "parquet" / "mover").glob("case_id=*"))
        before = {d.name: [mover_fast.sha256_file(d / "waveform.parquet"), mover_fast.sha256_file(d / "map.parquet")] for d in case_dirs}
        config_path = self.root / "configs" / "hemodynamics.json"
        config = json.loads(config_path.read_text())
        config.update(window_config())
        config_path.write_text(json.dumps(config))
        second = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(second["status"], "COMPLETE", second.get("reason"))
        self.assertGreater(sum(c.get("decoded", 0) for c in second["ingest"]["counters"].values()), 0, "the legacy marker must not skip the new pass")
        self.assertEqual(second["window_eligibility"]["existing_cases_reused"], 4)
        after = {d.name: [mover_fast.sha256_file(d / "waveform.parquet"), mover_fast.sha256_file(d / "map.parquet")] for d in case_dirs}
        self.assertEqual(before, after, "verified waveform/map files must not be rewritten")

    def test_resplit_is_metadata_only(self):
        self.build()
        self.assertEqual(mover_fast.run(self.root, workers=2, progress=False)["status"], "COMPLETE")
        store = mover_fast.CheckpointStore(self.root, "mover")
        prior = store.rows()[0]
        directory = self.root / "data" / "parquet" / "mover" / f"case_id={prior['case_id']}"
        hashes = [mover_fast.sha256_file(directory / "waveform.parquet"), mover_fast.sha256_file(directory / "map.parquet")]
        new_split = "test" if prior["split"] != "test" else "train"
        updated = mover_fast._resplit_case(self.root, store, prior["case_id"], prior, new_split)
        self.assertEqual(updated["split"], new_split)
        self.assertEqual(load_case(self.root, "mover", prior["case_id"])["split"], new_split)
        self.assertEqual(hashes, [mover_fast.sha256_file(directory / "waveform.parquet"), mover_fast.sha256_file(directory / "map.parquet")])
        self.assertTrue(mover_fast.verify_case(self.root, store.get(prior["case_id"])))

    def test_below_floor_creates_no_cohort_but_keeps_stats_for_every_case(self):
        self.build(config_overrides=window_config(min_cases=100, min_patients=100, floor_cases=10))
        result = mover_fast.run(self.root, workers=2, progress=False)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("not creating a cohort", result["reason"])
        manifests = self.root / "data" / "manifests"
        self.assertEqual(len(pd.read_parquet(manifests / "mover_case_window_stats.parquet")), 4)
        self.assertFalse((manifests / "mover_selected_cohort.parquet").exists())
        self.assertFalse((manifests / "mover_cohort_frozen.json").exists())
        self.assertTrue((self.root / "data" / "parquet" / "mover" / "staging").exists(), "fragments are kept for inspection")

    def test_stage_filter_requires_all_three_waveforms_only_in_window_mode(self):
        sanity = {"art_n": 0, "ecg_n": 10, "ppg_n": 10, "map_n": 5}
        fast = mover_fast._fast({})
        self.assertFalse(mover_fast._stage_worthy(sanity, {"eligibility_mode": "window_v1"}, fast))
        self.assertTrue(mover_fast._stage_worthy(sanity, {"eligibility_mode": "legacy_case"}, fast))
        self.assertTrue(mover_fast._stage_worthy({**sanity, "art_n": 3}, {"eligibility_mode": "window_v1"}, fast))

    def test_speed_settings_are_unchanged(self):
        fast = mover_fast._fast(json.loads((Path(__file__).resolve().parents[2] / "configs" / "hemodynamics.json").read_text()))
        for key, value in {"decode_workers": 16, "finalize_workers": 20, "archive_readers": 3, "max_inflight_payload_mb": 6144,
                           "min_available_ram_gb": 16, "tar_buffer_kb": 256}.items():
            self.assertEqual(fast[key], value)


if __name__ == "__main__":
    unittest.main()
