# Check bshi variants using controlled inputs and isolated files.
import tempfile
import unittest

import numpy as np

from hemodynamics import beats_pipeline as P
from hemodynamics import bshi as S
from hemodynamics.test_bshi import World


class TestRecipes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Build the requested beat images, numerical context, and beat masks.
        cls.tmp = tempfile.TemporaryDirectory()
        cls.w = World(cls.tmp.name)
        cls.rows = cls.w.sample(60, seed=21)
        cls.eap = S.assemble(cls.w.root, cls.rows, "eap", cache=S.CaseCache())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_every_recipe_equals_eap_with_its_groups_zeroed(self):
        # Verify every recipe equals eap with its groups zeroed.
        for v in S.VARIANTS:
            direct = S.assemble(self.w.root, self.rows, v, cache=S.CaseCache())
            derived = S.apply_variant(self.eap, v)
            for k in ("x", "context", "beat_mask", "variant_mask"):
                self.assertTrue(np.array_equal(direct[k], derived[k]), f"{v}:{k}")

    def test_recipes_are_nested_as_the_proposal_ladder(self):
        # Verify recipes are nested as the proposal ladder.
        on = lambda v: set(S.GROUPS) - S.VARIANTS[v]
        self.assertTrue(on("wave") < on("wave_map") < on("wave_couple") < on("eap"))
        self.assertTrue(on("map_only") == set())
        self.assertTrue(on("art_only") == S.ART_ONLY_ON)
        for model, v in S.PROPOSAL_MODELS.items():
            self.assertIn(v, S.VARIANTS, model)

    def test_zeroed_planes_and_bands(self):
        # Verify zeroed planes and bands.
        x = {v: S.apply_variant(self.eap, v)["x"] for v in S.VARIANTS}
        self.assertTrue((x["wave"][:, 3:] == 0).all() and np.abs(x["wave"][:, :3]).max() > 0)
        self.assertTrue((x["wave_map"][:, 4:] == 0).all() and np.abs(x["wave_map"][:, 3]).max() > 0)
        self.assertTrue((x["wave_couple"][:, 6] == 0).all() and np.abs(x["wave_couple"][:, 4]).max() > 0 and np.abs(x["wave_couple"][:, 5]).max() > 0)
        self.assertTrue((x["map_only"] == 0).all())
        art = x["art_only"]
        self.assertTrue((art[:, 0] == 0).all() and (art[:, 2] == 0).all() and (art[:, 4] == 0).all() and (art[:, 6] == 0).all())
        self.assertGreater(np.abs(art[:, 1]).max(), 0)
        for k, name in enumerate(S.MORPH_BANDS):
            band = art[:, 5, :, k * 25:(k + 1) * 25]
            self.assertEqual(bool(np.abs(band).max() > 0), name in S.ART_ONLY_ON, name)

    def test_context_keeps_only_the_label_features_for_map_only(self):
        # Verify context keeps only the label features for MAP only.
        ctx = S.apply_variant(self.eap, "map_only")["context"]
        keep = [S.CONTEXT_NAMES.index(c) for c in S.LABEL_CONTEXT]
        rest = np.setdiff1d(np.arange(ctx.shape[1]), keep)
        self.assertTrue((ctx[:, rest] == 0).all())
        self.assertTrue(np.array_equal(ctx[:, keep], self.eap["context"][:, keep]))
        self.assertTrue(set(S.context_off(S.VARIANTS["eap"])) == set())

    def test_return_times_gives_beat_starts_before_t0(self):
        # Verify return times gives beat starts before t0.
        out = S.assemble(self.w.root, self.rows, "eap", cache=S.CaseCache(), return_times=True)
        self.assertEqual(out["t_rel"].shape, (len(self.rows), 60))
        self.assertTrue((out["t_rel"] + out["rr"] <= 1e-3).all())                      # whole cycles end by t0
        self.assertTrue((np.diff(out["t_rel"], axis=1) > 0).all())
        self.assertTrue((out["t_rel"][:, 0] > -90.0).all())
        self.assertTrue(np.array_equal(out["x"], self.eap["x"]))                         # extra output does not change the tensor


class TestExactCorruption(unittest.TestCase):
    def test_extra_mask_equals_beats_missing_in_the_files(self):
        # Verify extra mask equals beats missing in the files.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(40, seed=22).reset_index(drop=True)
            drop = {(ds, cid): set(np.random.default_rng(5).choice(len(df), len(df) // 12, replace=False).tolist()) for (ds, cid), (df, _) in w.cases.items()}
            extra = np.ones((len(rows), 60), dtype=bool)
            for i, r in enumerate(rows.itertuples()):
                extra[i] = [(r.beat_lo + k) not in drop[(r.dataset, r.case_id)] for k in range(60)]
            masked = S.assemble(w.root, rows, "eap", cache=S.CaseCache(), extra_mask=extra)
            self.assertFalse(extra.all())
            for (ds, cid), (df, waves) in w.cases.items():
                df2 = df.copy()
                df2.loc[sorted(drop[(ds, cid)]), "valid_beat"] = False
                d = P.case_dir(w.root, ds, cid)
                P._atomic_parquet(df2, d / "beats.parquet")
                P._atomic_parquet(P.waveform_table(df2, waves, 200), d / "beat_waveforms.parquet")
            files_dropped = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            for k in ("x", "context", "beat_mask"):
                self.assertTrue(np.array_equal(masked[k], files_dropped[k]), k)

    def test_extra_mask_ppg_only_touches_ppg(self):
        # Verify extra mask PPG only touches PPG.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(40, seed=23).reset_index(drop=True)
            # Compare the result with the expected value or tolerance.
            base = S.assemble(w.root, rows, "eap", cache=S.CaseCache())
            extra = np.ones((len(rows), 60), dtype=bool)
            extra[:, ::5] = False
            out = S.assemble(w.root, rows, "eap", cache=S.CaseCache(), extra_mask_ppg=extra)
            self.assertTrue(np.array_equal(out["x"][:, [0, 1, 3]], base["x"][:, [0, 1, 3]]))        # ECG, ART and MAP planes unchanged
            self.assertFalse(np.array_equal(out["x"][:, 2], base["x"][:, 2]))
            self.assertTrue((out["x"][:, 2][:, ::5] == 0).all())
            self.assertTrue(np.array_equal(out["beat_mask"][:, :2], base["beat_mask"][:, :2]))
            self.assertTrue((out["beat_mask"][:, 2][:, ::5] == 0).all())

    def test_all_true_extra_mask_changes_nothing(self):
        # Verify all true extra mask changes nothing.
        with tempfile.TemporaryDirectory() as tmp:
            w = World(tmp)
            rows = w.sample(30, seed=24).reset_index(drop=True)
            a = S.assemble(w.root, rows, "ea", cache=S.CaseCache())
            b = S.assemble(w.root, rows, "ea", cache=S.CaseCache(), extra_mask=np.ones((len(rows), 60), dtype=bool))
            self.assertTrue(np.array_equal(a["x"], b["x"]) and np.array_equal(a["context"], b["context"]))


if __name__ == "__main__":
    unittest.main()
