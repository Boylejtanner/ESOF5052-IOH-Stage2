# Check bshi progress using controlled inputs and isolated files.
import json
import tempfile
import time
import unittest
from pathlib import Path

from hemodynamics import bshi_progress as P


class ProgressTests(unittest.TestCase):
    def run_fake(self, d, notebook=False):
        p = P.Progress(d, label="unit", every_s=0.0, log_every_s=0.0, plot_every_s=0.0, console=False, notebook=notebook, gpu=False)
        p.start_run(epochs=2, steps_per_epoch=10, windows_per_step=128)
        # Repeat training and validation for each epoch until the stopping rule is met.
        for epoch in range(2):
            p.begin_epoch(epoch)
            for step in range(1, 11):
                self.assertTrue(p.due())
                p.update(step, step * 128, {"total": 1.0 / (epoch * 10 + step), "event": 0.5}, 1e-3)
            p.begin_validation(5)
            for i in range(5):
                p.validation_batch(i)
            p.epoch_done({"epoch": epoch, "total": 0.2, "val_auprc": 0.2 + 0.01 * epoch, "val_auroc": 0.85, "val_brier": 0.1, "improved": True, "best_auprc": 0.2 + 0.01 * epoch,
                          "train_windows_per_s": 900.0, "train_seconds": 12.0})
        p.finish({"best_epoch": 1})
        return p

    def test_files_fields_and_final_state(self):
        # Verify files fields and final state.
        with tempfile.TemporaryDirectory() as td:
            self.run_fake(td)
            st = json.loads((Path(td) / "progress.json").read_text(encoding="utf-8"))
            self.assertEqual(st["phase"], "done")
            self.assertEqual(st["epochs"], 2)
            self.assertEqual(len(st["history"]), 2)
            self.assertAlmostEqual(st["val"]["auprc"], 0.21)
            self.assertEqual(st["eta_run_s"], 0.0)
            for name in ("progress.jsonl", "progress.png", "progress.html"):
                self.assertTrue((Path(td) / name).stat().st_size > 0, name)
            lines = [json.loads(x) for x in (Path(td) / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertTrue(any(r.get("event") == "epoch_done" for r in lines))
            self.assertIn("http-equiv='refresh'", (Path(td) / "progress.html").read_text(encoding="utf-8"))

    def test_status_line_and_eta(self):
        # Verify status line and eta.
        with tempfile.TemporaryDirectory() as td:
            p = P.Progress(td, label="unit", every_s=0.0, console=False, notebook=False, gpu=False, html_page=False)
            p.start_run(epochs=3, steps_per_epoch=100, windows_per_step=256)
            p.begin_epoch(0)
            # Check line against the expected result.
            time.sleep(0.2)
            p.update(10, 2560, {"total": 0.7}, 1e-3)
            time.sleep(0.2)
            p.update(40, 10240, {"total": 0.6}, 1e-3)
            # Check line against the expected result.
            line = P.status_line(P.read_status(td))
            self.assertIn("epoch 1/3", line)
            self.assertIn("40% of the epoch", line)
            self.assertIn("loss 0.6000", line)
            self.assertGreater(P.read_status(td)["eta_run_s"], 0)

    def test_format_seconds(self):
        # Verify format seconds.
        self.assertEqual(P.fmt_seconds(59), "59s")
        self.assertEqual(P.fmt_seconds(125), "2m05s")
        self.assertEqual(P.fmt_seconds(7260), "2h01m")
        self.assertEqual(P.fmt_seconds(None), "?")

    def test_cli_status(self):
        # Verify cli status.
        with tempfile.TemporaryDirectory() as td:
            self.run_fake(td)
            self.assertEqual(P.main(["status", td]), 0)
            self.assertEqual(P.status_line(None), "no progress file yet")


if __name__ == "__main__":
    unittest.main()
