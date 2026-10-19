import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from hemodynamics import bshi_dryrun as DR

ROOT = Path(os.environ.get("BSHI_ROOT", Path(__file__).resolve().parents[2]))


class MonitorTests(unittest.TestCase):
    def test_summary_uses_only_training_intervals(self):
        s = [{"t": t, "gpu_util": u, "vram_mb": 4096 + t, "cpu_util": 50.0} for t, u in [(0, 0), (1, 90), (2, 95), (3, 5)]]
        m = DR.summarize_monitor(s, [(1, 2)])
        self.assertEqual(m["samples"], 2)
        self.assertEqual(m["gpu_util_mean_%"], 92.5)
        self.assertEqual(m["share_samples_gpu_util_ge_80_%"], 100.0)

    def test_bottleneck_wording(self):
        self.assertTrue(DR.bottleneck({"share_samples_gpu_util_ge_80_%": 90, "cpu_util_mean_%": 10}).startswith("none"))
        self.assertIn("loader", DR.bottleneck({"share_samples_gpu_util_ge_80_%": 20, "cpu_util_mean_%": 80}))
        self.assertTrue(DR.bottleneck({}).startswith("unknown"))


@unittest.skipUnless(torch.cuda.is_available() and (ROOT / "data" / "derived" / "bshi" / "sweep_subset" / "MANIFEST.sha256").exists(), "needs CUDA and the sweep subset")
class SmokeRunTests(unittest.TestCase):
    def test_smoke_run_writes_checkpoint_and_reloads(self):
        with tempfile.TemporaryDirectory() as td:
            out = DR.run_dry(ROOT, td, epochs=2, smoke=True, epoch_windows=512, log=lambda *a: None)
            self.assertTrue((Path(td) / "best.pt").exists() and (Path(td) / "last.pt").exists())
            self.assertTrue(out["checkpoint_reload"]["ok"], out["checkpoint_reload"])
            self.assertEqual(len(out["epoch_wall_seconds"]), 2)
            self.assertTrue(np.isfinite(out["val_auroc"]).all())


if __name__ == "__main__":
    unittest.main()
