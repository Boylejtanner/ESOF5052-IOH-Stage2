# Check bshi raw using controlled inputs and isolated files.
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hemodynamics import bshi_raw as R

FS = 100


def make_waveform(root, dataset, case_id, seconds=300.0, seed=0, gaps=True):
    # Collect the per-record values into a table for comparison.
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = 12.5 + np.arange(n) / FS                                   # the case clock does not start at 0
    df = pd.DataFrame({"time_sec": t, "ART": rng.normal(90, 10, n).astype("float32"), "ECG": rng.normal(0, 1, n).astype("float32"), "PPG": rng.normal(500, 30, n).astype("float32")})
    if gaps:
        df.loc[5000:5600, "PPG"] = np.nan
        df = df.drop(index=range(9000, 9300)).reset_index(drop=True)      # missing rows (a capture gap)
    d = R.waveform_path(root, dataset, case_id)
    d.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d, index=False)
    return df


class TestRawWindows(unittest.TestCase):
    def test_windows_equal_a_direct_slice_of_the_waveform_file(self):
        # Verify windows equal a direct slice of the waveform file.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            df = make_waveform(root, "vitaldb", "7")
            rows = pd.DataFrame({"dataset": "vitaldb", "case_id": "7", "t_end_sec": [120.0, 160.5, 12.5 + 250.0, 60.0]})
            x, mask = R.raw_windows(root, rows)
            self.assertEqual(x.shape, (4, 3, 9000))
            # Process the selected records in order and retain their individual results.
            for i, t_end in enumerate(rows.t_end_sec):
                sel = df[(df.time_sec >= t_end - 90 - 1e-6) & (df.time_sec < t_end - 1e-6)]
                pos = np.rint((sel.time_sec.to_numpy() - (t_end - 90)) * FS).astype(int)
                for c, name in enumerate(("ECG", "ART", "PPG")):
                    # Compare the result with the expected value or tolerance.
                    v = sel[name].to_numpy()
                    ok = np.isfinite(v) & (pos >= 0) & (pos < 9000)
                    self.assertTrue(np.array_equal(x[i, c, pos[ok]], v[ok]), (i, name))
                    self.assertTrue(mask[i, c, pos[ok]].all())
                self.assertEqual(int(mask[i, 1].sum()), int(np.isfinite(sel.ART.to_numpy()).sum()) - int(((pos < 0) | (pos >= 9000)).sum()))

    def test_missing_and_nan_samples_are_zero_and_masked_never_filled(self):
        # Verify missing and nan samples are zero and masked never filled.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_waveform(root, "mover", "abc")
            rows = pd.DataFrame({"dataset": "mover", "case_id": "abc", "t_end_sec": [12.5 + 110.0]})   # window [32.5, 122.5) holds the PPG NaN block and 3 s of missing rows
            x, mask = R.raw_windows(root, rows)
            self.assertTrue((x[~mask] == 0).all())
            self.assertEqual(int((~mask[0, 0]).sum()), 300)                    # missing rows are masked for every channel
            self.assertEqual(int((~mask[0, 2]).sum()), 300 + 601)              # plus the PPG-only NaN samples

    def test_window_before_the_recording_is_padded_with_masked_zeros(self):
        # Verify window before the recording is padded with masked zeros.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_waveform(root, "vitaldb", "1", gaps=False)
            x, mask = R.raw_windows(root, pd.DataFrame({"dataset": "vitaldb", "case_id": "1", "t_end_sec": [12.5 + 30.0]}))
            self.assertEqual(int(mask[0, 1].sum()), 3000)
            self.assertTrue((x[0, :, :6000] == 0).all() and not mask[0, :, :6000].any())

    def test_normalization_is_scale_free_for_ecg_ppg_and_physical_for_art(self):
        # Verify normalization is scale free for ECG PPG and physical for ART.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_waveform(root, "vitaldb", "2", gaps=False)
            # Collect the per-record values into a table for comparison.
            rows = pd.DataFrame({"dataset": "vitaldb", "case_id": "2", "t_end_sec": [150.0, 200.0]})
            x, mask = R.raw_windows(root, rows)
            a = R.normalize_raw(x, mask)
            scaled = x.copy()
            scaled[:, 2] *= 16
            # Compare the result with the expected value or tolerance.
            scaled[:, 0] *= 16
            b = R.normalize_raw(scaled, mask)
            self.assertLess(float(np.abs(a - b).max()), 1e-5)                              # a different device unit changes nothing
            self.assertLessEqual(float(np.abs(a).max()), 8.0 + 1e-6)
            self.assertAlmostEqual(float(a[0, 1].mean()), (x[0, 1].mean() - 80) / 40, delta=0.05)


if __name__ == "__main__":
    unittest.main()
