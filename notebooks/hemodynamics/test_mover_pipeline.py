import base64
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from hemodynamics.canonical import DiskSpaceStop
from hemodynamics.mover_pipeline import (DiscoveryIncomplete, _quality_from_missing, _union_count,
                                        audit_linkage, discover, load_config, patient_splits,
                                        pilot, process_selected, select_cohort)


class MoverPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        self.source = self.root / "data/reference/mover"
        self.source.mkdir(parents=True)
        for folder in ("data/manifests", "results", "figures"):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        config_path = Path(__file__).resolve().parents[2] / "configs/hemodynamics.json"
        config = json.loads(config_path.read_text())
        config.update(min_free_space_gb=0, minimum_duration_sec=1, map_min_valid_points=10,
                      discovery_members_per_archive=2, pilot_cases=3, allow_full_archive_scan=True,
                      allow_full_process=True, maximum_waveform_missingness=.2)
        self.config = config
        (self.root / "configs/hemodynamics.json").write_text(json.dumps(config))
        crosswalk, clinical = [], []
        for index, archive in enumerate(config["mover_archives"]):
            patient, case = f"{index + 1:016x}", f"{index + 11:016x}"
            crosswalk.append({"LOG_ID": case, "PAT_ID": patient, "MRN": f"{index + 101:016x}"})
            clinical.append({"LOG_ID": case, "MRN": crosswalk[-1]["MRN"],
                             "IN_OR_DTTM": "01/01/21 09:00", "OUT_OR_DTTM": "01/01/21 09:01"})
            payload = self.xml()
            self.make_tar(self.source / archive, [(f"Waveforms/{patient}CB-2021-01-01-17-00-00-000Z.xml", payload)])
        pd.DataFrame(crosswalk).to_csv(self.source / "EPIC_MRN_PAT_ID.csv", index=False)
        self.make_tar(self.source / "EPIC_EMR.tar.gz", [("EPIC_EMR/EMR/patient_information.csv", pd.DataFrame(clinical).to_csv(index=False).encode())])

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def make_tar(path, items):
        with tarfile.open(path, "w:gz") as archive:
            for name, data in items:
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    @staticmethod
    def xml():
        packets = []
        start = datetime(2021, 1, 1, 17, tzinfo=timezone.utc)
        for second in range(60):
            groups = []
            for name, fs, gain in (("INVP1", 100, .01), ("ECG1", 300, 1.), ("PLETH", 100, 1.)):
                raw = np.full(fs, 8000, dtype="<i2") if name == "INVP1" else np.rint(np.sin(np.arange(fs) / fs * 2 * np.pi) * 100).astype("<i2")
                fields = {"Wave": base64.b64encode(raw.tobytes()).decode(), "Points": fs, "PointsBytes": 2,
                          "Min": -500, "Max": 1500, "Offset": 0, "Gain": gain, "Hz": fs}
                groups.append(f'<mg name="{name}">' + "".join(f'<m name="{k}">{v}</m>' for k, v in fields.items()) + "</mg>")
            stamp = (start + timedelta(seconds=second)).isoformat().replace("+00:00", "Z")
            packets.append(f'<cpc datetime="{stamp}"><device make="GE_S5" model="GE_S5"><measurements>' + "".join(groups) + '<m name="IBP_MEAN">80</m></measurements></device></cpc>')
        return ('<cpcArchive duration="30">' + "".join(packets) + "</cpcArchive>").encode()

    def test_live_progress_and_completion_on_fixture_archives(self):
        audit_linkage(self.root)
        output, bars = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(bars):
            inventory, metrics = discover(self.root, complete_scan=True, progress=True)
        self.assertTrue(inventory.scan_completed.all())
        self.assertTrue(metrics["full_scan_complete"])
        self.assertIn("STARTING COMPLETE MOVER DISCOVERY", output.getvalue())
        self.assertIn("[1/3] SCANNING", output.getvalue())
        self.assertIn("DISCOVERY COMPLETE: 3/3", output.getvalue())
        self.assertIn("Archive 3/3", bars.getvalue())
        self.assertIn("Checking surgery eligibility", bars.getvalue())
        with redirect_stdout(output), redirect_stderr(bars):
            discover(self.root, complete_scan=True, progress=True)
        self.assertIn("SKIPPED — verified complete scan already cached", output.getvalue())

    def test_progress_closes_when_disk_guard_pauses(self):
        audit_linkage(self.root)
        output, bars = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(bars), patch("hemodynamics.mover_pipeline.ensure_disk_space", side_effect=DiskSpaceStop("test pause")):
            with self.assertRaises(DiskSpaceStop):
                discover(self.root, progress=True)
        self.assertIn("PAUSED", output.getvalue())
        self.assertIn("[PAUSED]", bars.getvalue())
        self.assertNotIn("DISCOVERY COMPLETE:", output.getvalue())

    def test_byte_bar_uses_compressed_file_position(self):
        from hemodynamics.mover_pipeline import _ArchiveProgress
        with patch("hemodynamics.mover_pipeline.tqdm") as factory:
            bar = factory.return_value
            bar.total, bar.n = 1000, 0
            progress = _ArchiveProgress("archive", 1000, True)
            progress.update(250, seen=2, skipped=1)
            bar.update.assert_called_with(250)
            bar.n = 250
            progress.update(1500, seen=3)
            bar.update.assert_called_with(750)
            progress.close("COMPLETED")
            bar.close.assert_called_once()
            self.assertIn("COMPLETED", bar.set_description.call_args.args[0])

    def test_sample_does_not_certify_full_archive_or_cohort(self):
        cases, report = audit_linkage(self.root)
        self.assertEqual(len(cases), 3)
        self.assertTrue(cases.linkage_usable.all())
        inventory, metrics = discover(self.root, members_per_archive=1)
        self.assertFalse(inventory.scan_completed.any())
        eligible = pd.read_parquet(self.root / "data/manifests/mover_eligible_cohort.parquet")
        self.assertEqual(len(eligible), 0)
        readiness = pd.read_csv(self.root / "data/manifests/mover_raw_archive_deletion_readiness.csv")
        self.assertFalse(readiness.safe_to_delete.any())
        pilot(self.root, limit=3)
        with self.assertRaises(DiscoveryIncomplete):
            select_cohort(self.root)

    def test_complete_synthetic_archive_pipeline_and_patient_splits(self):
        audit_linkage(self.root)
        inventory, _ = discover(self.root, complete_scan=True)
        self.assertTrue(inventory.scan_completed.all())
        metrics = pilot(self.root, limit=3)
        self.assertEqual(metrics["eligible_cases_processed"], 3)
        selected = select_cohort(self.root)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected.split.nunique(), 3)
        result = process_selected(self.root)
        self.assertEqual(result["selected_surgeries_processed"], 3)
        readiness = pd.read_csv(self.root / "data/manifests/mover_raw_archive_deletion_readiness.csv")
        self.assertTrue(readiness.safe_to_delete.all())
        self.assertFalse(any((self.root / "data/parquet/mover/pilot_fragments").rglob("*.parquet")))
        self.assertTrue(all((self.source / archive).is_file() for archive in self.config["mover_archives"]))

    def test_bad_member_is_logged_and_other_archives_continue(self):
        first = self.source / self.config["mover_archives"][0]
        self.make_tar(first, [("Waveforms/0000000000000001CB-bad.xml", b"not XML")])
        audit_linkage(self.root)
        inventory, _ = discover(self.root, members_per_archive=2)
        self.assertEqual(int(inventory.sampled_errors.sum()), 1)
        self.assertEqual(len(inventory), 3)
        self.assertTrue((self.root / "data/manifests/mover_processing_errors.csv").exists())

    def test_one_bad_case_does_not_block_other_cases_eligibility(self):
        self.make_tar(self.source / self.config["mover_archives"][0], [("Waveforms/0000000000000001CB-bad.xml", b"bad XML")])
        audit_linkage(self.root)
        inventory, _ = discover(self.root, complete_scan=True)
        self.assertTrue(inventory.scan_completed.all())
        eligible = pd.read_parquet(self.root / "data/manifests/mover_eligible_cohort.parquet")
        self.assertEqual(len(eligible), 2)
        all_cases = pd.read_parquet(self.root / "data/manifests/mover_all_cases.parquet")
        bad = all_cases.loc[all_cases.case_id.eq("000000000000000b")].iloc[0]
        self.assertFalse(bad.final_usable)
        self.assertIn("corrupt_or_unknown_member", bad.exclusion_reason)

    def test_gzip_crc_failure_never_certifies_archive_scan(self):
        path = self.source / self.config["mover_archives"][0]
        data = bytearray(path.read_bytes())
        data[-8] ^= 1
        path.write_bytes(data)
        audit_linkage(self.root)
        inventory, _ = discover(self.root, complete_scan=True)
        self.assertFalse(bool(inventory.iloc[0].scan_completed))
        self.assertEqual(inventory.iloc[0].status, "failed")
        readiness = pd.read_csv(self.root / "data/manifests/mover_raw_archive_deletion_readiness.csv")
        self.assertFalse(readiness.safe_to_delete.any())

    def test_resume_preserves_fragment_hash_and_size(self):
        audit_linkage(self.root)
        discover(self.root, members_per_archive=1)
        paths = list((self.root / "data/parquet/mover/pilot_fragments").rglob("*.parquet"))
        self.assertGreater(len(paths), 0)
        stamps = {path: path.stat().st_mtime_ns for path in paths}
        discover(self.root, members_per_archive=1)
        self.assertEqual(stamps, {path: path.stat().st_mtime_ns for path in paths})

    def test_full_modes_remain_explicitly_guarded(self):
        self.config.update(allow_full_archive_scan=False, allow_full_process=False)
        (self.root / "configs/hemodynamics.json").write_text(json.dumps(self.config))
        with self.assertRaises(DiscoveryIncomplete):
            discover(self.root, complete_scan=True)
        with self.assertRaises(DiscoveryIncomplete):
            process_selected(self.root)

    def test_disk_guard_checkpoints_paused_archive(self):
        audit_linkage(self.root)
        with patch("hemodynamics.mover_pipeline.ensure_disk_space", side_effect=DiskSpaceStop("test pause")):
            with self.assertRaises(DiskSpaceStop):
                discover(self.root)
        import sqlite3
        connection = sqlite3.connect(self.root / "data/checkpoints/mover/inventory.sqlite")
        try:
            state = json.loads(connection.execute("SELECT payload FROM archives LIMIT 1").fetchone()[0])
        finally:
            connection.close()
        self.assertEqual(state["status"], "paused")
        self.assertFalse(state["scan_completed"])

    def test_unobserved_pressure_scaling_and_device_are_quarantined(self):
        from hemodynamics.mover_pipeline import _assert_observed_pressure_schema
        metadata = {"groups": {"INVP1": {"sample_rates": [100.], "variants": [{"Gain": ".01", "Offset": "0", "device": {"make": "GE_S5", "model": "GE_S5"}}]}}, "group_names": ["INVP1"]}
        _assert_observed_pressure_schema(metadata)
        metadata["groups"]["INVP1"]["variants"][0]["Gain"] = "1"
        with self.assertRaises(ValueError):
            _assert_observed_pressure_schema(metadata)
        metadata["groups"]["INVP1"]["variants"][0]["Gain"] = ".01"
        metadata["groups"]["INVP1"]["variants"][0]["device"]["model"] = "unknown"
        with self.assertRaises(ValueError):
            _assert_observed_pressure_schema(metadata)

    def test_union_and_long_map_gap_quality(self):
        self.assertEqual(_union_count([[0, 5], [3, 8], [10, 12]]), 10)
        case = pd.Series({"duration_sec": 3600})
        numeric = pd.DataFrame({"time_sec": [0., 3500.], "MAP": [80., 80.]})
        assessment = _quality_from_missing(case, {f"{c}_missingness": 0. for c in ("art", "ecg", "ppg")}, numeric, self.config)
        self.assertLess(assessment["map_coverage"], .01)
        self.assertFalse(assessment["final_usable"])


if __name__ == "__main__":
    unittest.main()
