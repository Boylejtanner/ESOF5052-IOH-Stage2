# Check bshi aug using controlled inputs and isolated files.
import tempfile
import unittest

import numpy as np
import torch

from hemodynamics import bshi as S
from hemodynamics import bshi_aug as A
from hemodynamics.test_bshi import World

STATS = {"valid_run_quantiles_s": np.linspace(4.0, 8.0, 101).tolist(), "gap_quantiles_s": np.linspace(2.0, 4.0, 101).tolist()}


def synth_batch(b=32, seed=0):
    # Seed the generated values so this check can be repeated.
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((b, 7, 60, 200), generator=g)
    rr = torch.full((b, 60), 0.8)
    t_rel = -(torch.arange(60, 0, -1, dtype=torch.float32) * 0.8).unsqueeze(0).repeat(b, 1)
    return {"x": x, "beat_mask": torch.ones((b, 3, 60), dtype=torch.bool), "context": torch.randn((b, len(S.CONTEXT_NAMES))), "t_rel": t_rel, "rr": rr}


class TestGaps(unittest.TestCase):
    def test_masked_beat_share_matches_mover_one_boundary_beat_per_run(self):
        # Verify masked beat share matches MOVER one boundary beat per run.
        b = synth_batch(256)
        keep = A.gap_keep_mask(b["rr"], STATS, torch.Generator().manual_seed(1))
        lost = 1.0 - float(keep.float().mean())
        # about one lost beat per valid run of ~6 s (7.5 beats): about 0.8 / (6 + 0.8) = 12%, like MOVER's measured 11% invalid beats
        self.assertTrue(0.07 < lost < 0.17, lost)
        self.assertEqual(keep.shape, (256, 60))
        self.assertGreater(float(keep.float().mean(dim=1).min()), 0.6)

    def test_deterministic_given_the_generator_and_different_across_seeds(self):
        # Verify deterministic given the generator and different across seeds.
        b = synth_batch(16)
        k1 = A.gap_keep_mask(b["rr"], STATS, torch.Generator().manual_seed(3))
        k2 = A.gap_keep_mask(b["rr"], STATS, torch.Generator().manual_seed(3))
        k3 = A.gap_keep_mask(b["rr"], STATS, torch.Generator().manual_seed(4))
        self.assertTrue(torch.equal(k1, k2))
        self.assertFalse(torch.equal(k1, k3))

    def test_drop_beats_zeroes_planes_masks_and_counts(self):
        # Verify drop beats zeroes planes masks and counts.
        b = synth_batch(8)
        keep = torch.rand((8, 60), generator=torch.Generator().manual_seed(0)) > 0.4
        out = A.drop_beats(b, keep)
        self.assertTrue((out["x"].permute(0, 2, 1, 3)[~keep] == 0).all())
        self.assertTrue(torch.equal(out["x"].permute(0, 2, 1, 3)[keep], b["x"].permute(0, 2, 1, 3)[keep]))
        self.assertTrue((out["beat_mask"].permute(0, 2, 1)[~keep] == 0).all())
        self.assertTrue(torch.equal(out["context"][:, S.CONTEXT_NAMES.index("n_art_ok")], out["beat_mask"][:, 1].sum(dim=1).float()))
        self.assertTrue(torch.equal(b["x"], synth_batch(8)["x"]))                      # the input batch is not modified

    def test_mover_gaps_wrapper_returns_the_keep_mask(self):
        out = A.mover_gaps(synth_batch(8), STATS, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(out["beat_mask"][:, 0], out["keep"]))


class TestCorruptions(unittest.TestCase):
    def test_ppg_dropout_only_touches_ppg_groups(self):
        # Verify PPG dropout only touches PPG groups.
        b = synth_batch(32)
        out = A.ppg_dropout(b, 0.3, 0.2, torch.Generator().manual_seed(5))
        drop = out["dropped"]
        # Compare the result with the expected value or tolerance.
        self.assertGreater(int(drop.sum()), 0)
        self.assertTrue(torch.equal(out["x"][:, [0, 1, 3]], b["x"][:, [0, 1, 3]]))
        self.assertTrue((out["x"][:, 2].permute(0, 1, 2)[drop] == 0).all())
        plane, sl = A.band_slice("sqi", "sqi_ppg")
        # Compare the result with the expected value or tolerance.
        self.assertTrue((out["x"][:, plane, :, sl][drop] == 0).all())
        plane, sl = A.band_slice("sqi", "sqi_ecg")
        self.assertTrue(torch.equal(out["x"][:, plane, :, sl], b["x"][:, plane, :, sl]))
        self.assertTrue(torch.equal(out["beat_mask"][:, :2], b["beat_mask"][:, :2]))
        self.assertTrue((out["beat_mask"][:, 2][drop] == 0).all())

    def test_ecg_noise_has_the_requested_snr(self):
        # Verify ECG noise has the requested snr.
        b = synth_batch(64)
        out = A.ecg_noise(b, 10.0, torch.Generator().manual_seed(6))
        noise = out["x"][:, 0] - b["x"][:, 0]
        snr = 10 * torch.log10((b["x"][:, 0] ** 2).mean() / (noise ** 2).mean())
        self.assertAlmostEqual(float(snr), 10.0, delta=0.5)
        self.assertTrue(torch.equal(out["x"][:, 1:], b["x"][:, 1:]))

    def test_art_damping_reduces_waveform_variation_and_keeps_other_planes(self):
        # Verify ART damping reduces waveform variation and keeps other planes.
        b = synth_batch(8)
        out = A.art_damping(b, 9)
        self.assertLess(float(out["x"][:, 1].std()), 0.6 * float(b["x"][:, 1].std()))
        self.assertTrue(torch.equal(out["x"][:, [0, 2, 3, 4, 5, 6]], b["x"][:, [0, 2, 3, 4, 5, 6]]))

    def test_flatline_makes_selected_beats_constant(self):
        # Verify flatline makes selected beats constant.
        b = synth_batch(16)
        out = A.art_flatline(b, 0.3, torch.Generator().manual_seed(7))
        flat = out["flat"]
        self.assertGreater(int(flat.sum()), 0)
        self.assertLess(float(out["x"][:, 1][flat].std(dim=1).max()), 1e-6)
        self.assertTrue(torch.equal(out["x"][:, 1][~flat], b["x"][:, 1][~flat]))

    def test_timing_jitter_zero_is_identity_and_only_waveform_planes_move(self):
        # Verify timing jitter zero is identity and only waveform planes move.
        b = synth_batch(8)
        same = A.timing_jitter(b, 0.0, torch.Generator().manual_seed(8))
        self.assertTrue(torch.equal(same["x"], b["x"]))
        out = A.timing_jitter(b, 0.05, torch.Generator().manual_seed(8))
        self.assertFalse(torch.equal(out["x"][:, :3], b["x"][:, :3]))
        self.assertTrue(torch.equal(out["x"][:, 3:], b["x"][:, 3:]))
        self.assertAlmostEqual(float(out["x"][:, :3].mean()), float(b["x"][:, :3].mean()), places=5)    # a circular shift conserves the content

    def test_registry_declares_exactness_for_every_function(self):
        for name, spec in A.AUGMENTATIONS.items():
            self.assertTrue(spec["exactness"].startswith(("EXACT", "APPROXIMATION")), name)
            self.assertTrue(callable(spec["fn"]))


class TestAgainstTheAssembler(unittest.TestCase):
    def test_exact_gap_path_and_batch_times_work_together(self):
        # Verify exact gap path and batch times work together.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(24, seed=31).reset_index(drop=True)
            # Compare the result with the expected value or tolerance.
            base = S.assemble(w.root, rows, "eap", cache=S.CaseCache(), return_times=True)
            batch = A.to_torch(base)
            keep = A.gap_keep_mask(batch["rr"], STATS, torch.Generator().manual_seed(9))
            exact = S.assemble(w.root, rows, "eap", cache=S.CaseCache(), extra_mask=keep.numpy())
            approx = A.drop_beats(batch, keep)
            # Compare the result with the expected value or tolerance.
            self.assertTrue(np.array_equal(exact["beat_mask"], approx["beat_mask"].numpy()))
            gone = ~keep.numpy()
            self.assertTrue((exact["x"].transpose(0, 2, 1, 3)[gone] == 0).all())


if __name__ == "__main__":
    unittest.main()
