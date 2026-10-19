"""Focused synthetic and bounded, in-memory real-member MOVER decoder tests.

Run: .venv\\Scripts\\python.exe -m unittest discover -s notebooks\\hemodynamics
Real archive tests stream only a bounded prefix; no member is extracted to disk.
"""

# Check mover signals using controlled inputs and isolated files.
import base64
from datetime import datetime, timezone
from itertools import islice
from io import BytesIO
import json
from pathlib import Path
import tarfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from notebooks.hemodynamics.mover_signals import decode_xml, inspect_xml


_ORIGIN = datetime(2021, 1, 1, tzinfo=timezone.utc).timestamp()
_ROOT = Path(__file__).resolve().parents[2]


def _fixture(packets):
    root = ET.Element("cpcArchive", duration="30")
    # Process the selected records in order and retain their individual results.
    for item in packets:
        time = item.get("time", 0)
        cpc = ET.SubElement(root, "cpc", datetime=datetime.fromtimestamp(
            _ORIGIN + time, timezone.utc).isoformat())
        device = ET.SubElement(cpc, "device", make=item.get("make", "AnalogWaves"),
                               model="test", bid="test")
        ms = ET.SubElement(device, "measurements")
        for name, value in item.get("numeric", []):
            ET.SubElement(ms, "m", name=name).text = str(value)
        if "gap" in item:
            ET.SubElement(ms, "m", name="WaveDataGap").text = str(item["gap"])
        # Process the selected records in order and retain their individual results.
        for spec in item.get("waves", []):
            name, values, fs, gain = spec[:4]
            options = spec[4] if len(spec) > 4 else {}
            group = ET.SubElement(ms, "mg", name=name)
            raw = np.asarray(values, dtype="<i2")
            fields = {"Wave": base64.b64encode(raw.tobytes()).decode(),
                      "Points": str(len(raw)), "PointsBytes": "2", "Hz": str(fs),
                      "Gain": str(gain), "Offset": "0", "Min": "1500", "Max": "3000"}
            fields.update(options)
            for field, value in fields.items():
                ET.SubElement(group, "m", name=field).text = value
    return ET.tostring(root)


class SyntheticMOVERTests(unittest.TestCase):
    def decode(self, fixture, start=0, end=20, fs=100, track_map=None):
        return decode_xml(fixture, _ORIGIN + start, _ORIGIN + end, fs, track_map)

    def test_schema_gain_and_absolute_numeric_no_clinical_min_max_filter(self):
        # Verify schema gain and absolute numeric no clinical min max filter.
        payload = _fixture([{"waves": [("INVP1", [9000] * 100, 100, .01)],
                             "numeric": [("IBP_MEAN", 87), ("NIBP_MEAN", 65)]}])
        inspected = inspect_xml(payload)
        # Check group names against the expected result.
        self.assertEqual(inspected["group_names"], ["INVP1"])
        self.assertEqual(inspected["original_fs"], {"INVP1": 100.0})
        self.assertEqual(inspected["member_start_utc"], _ORIGIN)
        self.assertEqual(inspected["member_end_utc"], _ORIGIN + 1)
        # Check valid samples against the expected result.
        self.assertEqual(inspected["waveform_counts"]["ART"]["valid_samples"], 100)
        wf, maps, meta = self.decode(payload)
        np.testing.assert_allclose(wf.ART, 90)
        self.assertTrue(wf.ECG.isna().all())
        self.assertTrue(wf.PPG.isna().all())
        self.assertEqual(maps.MAP.tolist(), [87])
        self.assertEqual(meta["groups"]["INVP1"]["variants"][0]["Min"], "1500")

    def test_absent_tracks_are_empty_not_cuff_or_derived_map(self):
        # Verify absent tracks are empty not cuff or derived MAP.
        wf, maps, _ = self.decode(_fixture([{"numeric": [("NIBP_MEAN", 75)]}]))
        self.assertTrue(wf.empty)
        self.assertTrue(maps.empty)
        self.assertEqual(wf.columns.tolist(), ["time_sec", "ART", "ECG", "PPG"])
        with self.assertRaisesRegex(ValueError, "never NIBP"):
            self.decode(_fixture([]), track_map={"MAP": "NIBP_MEAN"})

    def test_numeric_only_and_explicit_ambiguous_channel_selection(self):
        # Verify numeric only and explicit ambiguous channel selection.
        payload = _fixture([{"numeric": [("IBP_MEAN", 80), ("IBP_MEAN", 60)]},
                            {"time": 1, "numeric": [("IBP_MEAN", "--"), ("IBP_MEAN", 62)]}])
        wf, maps, meta = self.decode(payload)
        # Check empty against the expected result.
        self.assertTrue(wf.empty)
        self.assertTrue(maps.empty)
        self.assertIn("Ambiguous", meta["map_unavailable_reason"])
        _, maps, meta = self.decode(payload, track_map={"MAP": "IBP_MEAN:2"})
        # Compare the result with the expected value or tolerance.
        self.assertEqual(maps.MAP.tolist(), [60, 62])
        self.assertEqual(meta["numeric_channel_counts"]["IBP_MEAN"], 2)
        self.assertIn("site is not established", meta["map_channel_identity"])

    def test_observed_s5_mapping_evidence_does_not_automatically_select_map(self):
        # Verify observed s5 mapping evidence does not automatically select MAP.
        numeric = [(name, value) for _ in range(6)
                   for name, value in (("IBP_SYS", 110), ("IBP_MEAN", 85), ("IBP_DIA", 65))]
        payload = _fixture([{"make": "GE_S5", "waves": [("INVP1", [8500] * 100, 100, .01)],
                             "numeric": numeric}])
        # Check supported explicit selector against the expected result.
        meta = inspect_xml(payload)
        evidence = meta["map_mapping_evidence"]
        self.assertEqual(evidence["supported_explicit_selector"], "IBP_MEAN:1")
        # Check different patients against the expected result.
        self.assertEqual(evidence["different_patients"], 3)
        json.dumps(meta, allow_nan=False)
        _, maps, _ = self.decode(payload)
        # Check empty against the expected result.
        self.assertTrue(maps.empty)
        _, maps, _ = self.decode(payload, track_map={"MAP": "IBP_MEAN:1"})
        self.assertEqual(maps.MAP.tolist(), [85])
        mismatched = ET.fromstring(payload)
        original = mismatched.find("cpc/device/measurements")
        other = ET.SubElement(mismatched.find("cpc"), "device", make="Other", model="test", bid="other")
        second = ET.SubElement(other, "measurements")
        for m in list(original.findall("m")):
            original.remove(m)
            second.append(m)
        self.assertIsNone(inspect_xml(ET.tostring(mismatched))["map_mapping_evidence"])

    def test_sentinel_and_timestamp_gap_remain_nan(self):
        # Verify sentinel and timestamp gap remain nan.
        values = [400] * 180
        values[70:110] = [-32767] * 40
        payload = _fixture([{"waves": [("GE_ART", values, 180, .25)]},
                            {"time": 2, "waves": [("GE_ART", [400] * 180, 180, .25)]}])
        # Compare the result with the expected value or tolerance.
        wf, _, meta = self.decode(payload, end=3)
        self.assertTrue(wf.loc[(wf.time_sec >= .4) & (wf.time_sec < .6), "ART"].isna().all())
        self.assertTrue(wf.loc[(wf.time_sec >= 1) & (wf.time_sec < 2), "ART"].isna().all())
        # Compare the result with the expected value or tolerance.
        np.testing.assert_allclose(wf.loc[(wf.time_sec >= 2.1) & (wf.time_sec < 2.8), "ART"],
                                   100, rtol=.002)
        self.assertEqual(meta["groups"]["GE_ART"]["missing_samples"], 40)

    def test_wave_data_gap_masks_packet_and_does_not_filter_across(self):
        # Verify wave data gap masks packet and does not filter across.
        payload = _fixture([{"waves": [("GE_ART", [400] * 180, 180, .25)]},
                            {"time": 1, "gap": "TRUE", "waves": [("GE_ART", [900] * 180, 180, .25)]},
                            {"time": 2, "waves": [("GE_ART", [800] * 180, 180, .25)]}])
        wf, _, meta = self.decode(payload, end=3)
        self.assertTrue(wf.loc[(wf.time_sec >= 1) & (wf.time_sec < 2), "ART"].isna().all())
        self.assertLess(abs(wf.ART.iloc[90] - 100), .2)
        self.assertLess(abs(wf.ART.iloc[210] - 200), .3)
        self.assertEqual(meta["packet_summaries"]["wave_data_gap_packets"], 1)
        with self.assertRaisesRegex(ValueError, "WaveDataGap"):
            self.decode(_fixture([{"gap": "MAYBE"}]))

    def test_start_aligned_grid_and_half_open_bounds_bounded_to_member(self):
        # Verify start aligned grid and half open bounds bounded to member.
        payload = _fixture([{"time": 7200.003, "waves": [("INVP1", [8500] * 100, 100, .01)]}])
        wf, _, _ = self.decode(payload, end=86400)
        self.assertLessEqual(len(wf), 100)
        # Compare the result with the expected value or tolerance.
        self.assertGreaterEqual(wf.time_sec.min(), 7200.003)
        np.testing.assert_allclose(wf.time_sec * 100, np.round(wf.time_sec * 100), atol=1e-8)
        wf, _, _ = self.decode(payload, start=7200.003, end=7200.503)
        self.assertEqual(len(wf), 50)
        # Compare the result with the expected value or tolerance.
        self.assertEqual(wf.time_sec.iloc[0], 0)
        self.assertLess(wf.time_sec.max(), .5)
        outside, _, _ = self.decode(payload, end=3600)
        self.assertTrue(outside.empty)

    def test_anti_alias_polyphase_across_contiguous_packets(self):
        # Verify anti alias polyphase across contiguous packets.
        fs = 300
        t = np.arange(fs * 6) / fs
        signal = np.round(1000 * (np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 80 * t))).astype(int)
        # Calculate the target or processed signal for the following step.
        payload = _fixture([{"time": second, "waves": [("ECG1", signal[second * fs:(second + 1) * fs], fs, 1)]}
                            for second in range(6)])
        wf, _, _ = self.decode(payload, end=6)
        interior = wf[(wf.time_sec > .5) & (wf.time_sec < 5.5)]
        # Check amplitude 10 against the expected result.
        t = interior.time_sec.to_numpy()
        y = interior.ECG.to_numpy() / 1000
        amplitude_10 = abs(2 * np.mean(y * np.exp(-2j * np.pi * 10 * t)))
        alias_20 = abs(2 * np.mean(y * np.exp(-2j * np.pi * 20 * t)))
        self.assertGreater(amplitude_10, .95)
        self.assertLess(alias_20, .02)

    def test_equal_duplicates_collapse_and_conflicts_mask_deterministically(self):
        # Verify equal duplicates collapse and conflicts mask deterministically.
        first = {"waves": [("GE_ART", [400] * 180, 180, .25)], "numeric": [("IBP_MEAN", 90)]}
        wf, maps, meta = self.decode(_fixture([first, first]))
        self.assertEqual(meta["duplicate_wave_packets"], 1)
        # Check conflicting wave timestamps against the expected result.
        self.assertEqual(meta["conflicting_wave_timestamps"], 0)
        self.assertEqual(len(wf), 100)
        self.assertEqual(maps.MAP.tolist(), [90])
        second = {"waves": [("GE_ART", [600] * 180, 180, .25)], "numeric": [("IBP_MEAN", 100)]}
        # Compare the result with the expected value or tolerance.
        wf, maps, meta = self.decode(_fixture([first, second]))
        reversed_wf, reversed_maps, _ = self.decode(_fixture([second, first]))
        self.assertTrue(wf.ART.isna().all())
        self.assertTrue(maps.MAP.isna().all())
        self.assertEqual(meta["conflicting_wave_timestamps"], 1)
        self.assertEqual(meta["conflicting_map_timestamps"], 1)
        self.assertTrue(wf.equals(reversed_wf))
        self.assertTrue(maps.equals(reversed_maps))

    def test_overlap_conflicts_are_masked(self):
        # Verify overlap conflicts are masked.
        payload = _fixture([{"waves": [("INVP1", [8000] * 100, 100, .01)]},
                            {"time": .5, "waves": [("INVP1", [10000] * 100, 100, .01)]}])
        wf, _, meta = self.decode(payload, end=2)
        self.assertTrue(wf.loc[(wf.time_sec >= .5) & (wf.time_sec < 1), "ART"].isna().all())
        self.assertEqual(meta["overlap_conflicting_samples"], 50)

    def test_overlapping_missing_support_is_applied_before_filtering(self):
        # Verify overlapping missing support is applied before filtering.
        values = np.r_[np.full(90, 400), np.full(90, 900)]
        payload = _fixture([{"waves": [("GE_ART", values, 180, .25)]},
                            {"time": .5, "gap": "TRUE", "waves": [("GE_ART", [800] * 180, 180, .25)]},
                            {"time": 1.5, "waves": [("GE_ART", [1200] * 180, 180, .25)]}])
        wf, _, _ = self.decode(payload, end=2.5)
        self.assertTrue(wf.loc[(wf.time_sec >= .5) & (wf.time_sec < 1.5), "ART"].isna().all())
        self.assertLess(abs(wf.ART.iloc[48] - 100), .2)
        self.assertLess(abs(wf.ART.iloc[151] - 300), .5)

    def test_metadata_is_strict_json_and_large_case_does_not_expand_member(self):
        # Verify metadata is strict json and large case does not expand member.
        payload = _fixture([{"waves": [("INVP1", [-32767] * 100, 100, .01)],
                             "numeric": [("IBP_MEAN", "--")]}])
        json.dumps(inspect_xml(payload), allow_nan=False)
        wf, _, meta = self.decode(payload, end=1_000_000)
        self.assertEqual(len(wf), 100)
        json.dumps(meta, allow_nan=False)
        distant = _fixture([{"waves": [("INVP1", [8000] * 100, 100, .01)]},
                            {"time": 100_000, "waves": [("INVP1", [8000] * 100, 100, .01)]}])
        with self.assertRaisesRegex(ValueError, "bounded grid"):
            self.decode(distant, end=200_000)

    def test_original_clocks_not_used_even_when_stale(self):
        # Verify original clocks not used even when stale.
        payload = _fixture([{"waves": [("ECG1", [10] * 300, 300, 1)],
                             "numeric": [("PacketTime", "2000-01-01T00:00:00"),
                                         ("POLLTIME", "2000-01-01T08:00:00Z")]}])
        wf, _, meta = self.decode(payload, end=1)
        self.assertEqual(len(wf), 100)
        self.assertEqual(meta["packet_summaries"]["clock_field_counts"]["POLLTIME"], 1)
        self.assertEqual(meta["member_start_utc"], _ORIGIN)

    def test_reject_malformed_encoding_metadata_and_naive_clock(self):
        # Verify reject malformed encoding metadata and naive clock.
        cases = [{"Wave": "!!!"}, {"Wave": "AAAA"}, {"PointsBytes": "4"},
                 {"Hz": "nan"}, {"Offset": "1"}, {"Gain": "nan"}]
        for override in cases:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.decode(_fixture([{"waves": [("GE_ART", [400] * 180, 180, .25, override)]}]))
        payload = _fixture([{"waves": [("GE_ART", [400] * 180, 180, .25)]}])
        tree = ET.fromstring(payload)
        tree.find("cpc").set("datetime", "2021-01-01T00:00:00")
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            self.decode(ET.tostring(tree))
        with self.assertRaisesRegex(ValueError, "DTD"):
            inspect_xml(b'<!DOCTYPE cpcArchive [<!ENTITY x "test">]><cpcArchive/>')

    def test_track_selection_rate_changes_and_invalid_bounds(self):
        # Verify track selection rate changes and invalid bounds.
        payload = _fixture([{"waves": [("GE_ART", [400] * 180, 180, .25),
                                        ("INVP1", [8500] * 100, 100, .01)]},
                            {"time": 1, "waves": [("GE_ART", [400] * 300, 300, .25)]}])
        self.assertEqual(inspect_xml(payload)["original_fs"]["GE_ART"], [180.0, 300.0])
        with self.assertRaisesRegex(ValueError, "Ambiguous ART"):
            self.decode(payload)
        wf, _, _ = self.decode(payload, track_map={"ART": "INVP1"})
        np.testing.assert_allclose(wf.ART, 85)
        for start, end, fs in [(0, 0, 100), (0, 1, 0), (0, 1, float("nan"))]:
            with self.assertRaises(ValueError):
                self.decode(payload, start, end, fs)


class RealMOVERMemberTests(unittest.TestCase):
    def test_one_representative_member_per_shard_without_extraction(self):
        # These bounded indices were inspected directly: physiological GE pressure
        # in shards 1/2 and the real all-missing INVP1/PLETH/ECG1 S5 member in shard3.
        for shard, index in ((1, 94), (2, 52), (3, 49)):
            archive = _ROOT / "data" / "reference" / "mover" / f"epic_wave_{shard}_v2.tar.gz"
            with self.subTest(shard=shard):
                if not archive.exists():
                    self.skipTest(f"Local reference archive unavailable: shard {shard}")
                # Read or write through a managed handle so the file is closed on exit.
                with tarfile.open(archive, "r|gz") as tar:
                    members = (m for m in tar if m.isfile() and m.name.endswith(".xml"))
                    for position, member in enumerate(islice(members, index + 1)):
                        if position == index:
                            stream = tar.extractfile(member)
                            self.assertIsNotNone(stream)
                            with stream:
                                payload = stream.read()
                            break
                    else:
                        self.fail("Reference archive has fewer than the bounded expected members")
                # Check duration against the expected result.
                meta = inspect_xml(payload)
                self.assertEqual(meta["archive_attributes"]["duration"], "30")
                self.assertLessEqual(meta["member_end_utc"] - meta["member_start_utc"], 1802)
                start = meta["member_start_utc"]
                wf, maps, decoded = decode_xml(payload, start, start + 15)
                self.assertLessEqual(len(wf), 1500)
                self.assertTrue(wf.time_sec.ge(0).all())
                self.assertTrue(wf.time_sec.lt(15).all())
                if shard < 3:
                    # Check GE ART against the expected result.
                    self.assertEqual(meta["original_fs"]["GE_ART"], 180.0)
                    self.assertTrue(wf.ART.notna().any())
                    self.assertTrue(wf.PPG.isna().all())
                    self.assertTrue(maps.empty)
                    self.assertNotIn("IBP_MEAN", meta["numeric_names"])
                else:
                    # Check original fs against the expected result.
                    self.assertEqual(meta["original_fs"], {"ECG1": 300.0, "INVP1": 100.0, "PLETH": 100.0})
                    self.assertGreater(meta["groups"]["INVP1"]["missing_samples"], 0)
                    self.assertTrue(wf.ART.isna().all())
                    self.assertEqual(meta["numeric_channel_counts"]["IBP_MEAN"], 6)
                    self.assertIn("map_unavailable_reason", decoded)
                    self.assertTrue(maps.empty)
                json.dumps(meta, allow_nan=False)
                json.dumps(decoded, allow_nan=False)
                if shard < 3:
                    # Compare real native samples exactly, without filtering or
                    # centering, to independently decoded little-endian bytes.
                    for event, element in ET.iterparse(BytesIO(payload), events=("end",)):
                        if element.tag != "cpc":
                            continue
                        group = element.find(".//mg[@name='GE_ART']")
                        if group is None:
                            element.clear()
                            continue
                        # Check expected against the expected result.
                        fields = {m.get("name"): m.text for m in group.findall("m")}
                        expected = np.frombuffer(base64.b64decode(fields["Wave"]), dtype="<i2").astype(float)
                        expected[expected == -32767] = np.nan
                        expected *= float(fields["Gain"])
                        packet_start = datetime.fromisoformat(element.get("datetime")).timestamp()
                        native, _, _ = decode_xml(payload, packet_start, packet_start + 1, 180)
                        np.testing.assert_allclose(native.ART.to_numpy(), expected, equal_nan=True)
                        break
                del payload


if __name__ == "__main__":
    unittest.main()
