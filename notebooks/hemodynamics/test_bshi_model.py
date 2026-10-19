import unittest

import torch

from hemodynamics import bshi as S
from hemodynamics import bshi_aug as A
from hemodynamics.bshi_model import BshiNet, ModelConfig, input_masks

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def batch(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn((n, 7, 60, 200), generator=g)
    return x.to(DEV), torch.randn((n, len(S.CONTEXT_NAMES)), generator=g).to(DEV), (torch.rand((n, 3, 60), generator=g) > 0.1).to(DEV)


class TestShapes(unittest.TestCase):
    def test_every_variant_gives_finite_heads_of_the_right_shape(self):
        x, ctx, bm = batch()
        for variant in S.VARIANTS:
            net = BshiNet(ModelConfig(variant=variant)).to(DEV).eval()
            out = net(x, ctx, bm)
            self.assertEqual(set(out), {"event", "delta_map", "aut_logit", "aut_mag"}, variant)
            for k, v in out.items():
                self.assertEqual(tuple(v.shape), (6,), (variant, k))
                self.assertTrue(torch.isfinite(v).all(), (variant, k))

    def test_parameter_budget(self):
        n = BshiNet(ModelConfig()).n_params()
        self.assertLess(n, 3_100_000)
        self.assertGreater(n, 2_000_000)
        self.assertLess(BshiNet(ModelConfig(variant="map_only")).n_params(), 100_000)   # context only: no image trunk

    def test_masks_match_the_recipes(self):
        m, cm = input_masks("eap")
        self.assertEqual(float(m.sum()), 7 * 200)
        self.assertEqual(float(cm.sum()), len(S.CONTEXT_NAMES))
        m, cm = input_masks("ea")
        self.assertEqual(float(m[:, 2].sum()), 0.0)                       # PPG plane off
        self.assertEqual(float(m[:, 0].sum()), 200.0)
        self.assertEqual(float(m[0, 4, 0, 50:100].sum()), 0.0)             # ptt_ecg_ppg band (second timing band) off
        self.assertEqual(float(m[0, 4, 0, 0:50].sum()), 50.0)              # ptt_ecg_art band on
        self.assertEqual(float(cm[S.CONTEXT_NAMES.index("n_ppg_ok")]), 0.0)
        m, _ = input_masks("map_only")
        self.assertEqual(float(m.sum()), 0.0)


class TestInputIsolation(unittest.TestCase):
    def test_zeroed_groups_cannot_change_the_output(self):
        x, ctx, bm = batch()
        for variant in ("ea", "art_only", "wave", "map_only"):
            net = BshiNet(ModelConfig(variant=variant)).to(DEV).eval()
            ref = net(x, ctx, bm)
            mask, cmask = input_masks(variant)
            noise = torch.randn_like(x) * 5
            x2 = x + noise * (1 - mask.to(DEV))                           # change ONLY the groups the variant turns off
            ctx2 = ctx + torch.randn_like(ctx) * 5 * (1 - cmask.to(DEV))
            out = net(x2, ctx2, bm)
            for k in ref:
                self.assertTrue(torch.allclose(ref[k], out[k], atol=1e-5), (variant, k))

    def test_a_window_does_not_depend_on_the_other_windows_in_its_batch(self):
        x, ctx, bm = batch(8)
        net = BshiNet(ModelConfig()).to(DEV).eval()
        full = net(x, ctx, bm)
        for i in (0, 5):
            single = net(x[i:i + 1], ctx[i:i + 1], bm[i:i + 1])
            for k in full:
                self.assertTrue(torch.allclose(full[k][i], single[k][0], atol=2e-4), (i, k))        # GroupNorm: no batch statistics

    def test_no_batchnorm_anywhere(self):
        self.assertFalse(any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in BshiNet(ModelConfig()).modules()))


class TestGate(unittest.TestCase):
    def test_gate_starts_as_identity(self):
        x, ctx, bm = batch(4)
        net = BshiNet(ModelConfig(gate=True)).to(DEV).eval()
        w = net.gate(x, bm)
        self.assertTrue(torch.allclose(w, torch.ones_like(w)))
        a = net(x, ctx, bm)
        b = BshiNet(ModelConfig(gate=False)).to(DEV).eval()
        b.load_state_dict({k: v for k, v in net.state_dict().items() if not k.startswith("gate")})
        c = b(x, ctx, bm)
        for k in a:
            self.assertTrue(torch.allclose(a[k], c[k], atol=1e-5))

    def test_gate_scales_only_the_waveform_planes_by_the_learned_weights(self):
        x, ctx, bm = batch(3)
        net = BshiNet(ModelConfig(gate=True)).to(DEV).eval()
        with torch.no_grad():
            net.gate.net[-1].bias.copy_(torch.tensor([-100.0, 0.0, 100.0]))      # weights ~0 (ECG), 1 (ART), 2 (PPG)
        w = net.gate(x, bm)
        self.assertTrue(torch.allclose(w[:, 0], torch.zeros_like(w[:, 0]), atol=1e-4))
        self.assertTrue(torch.allclose(w[:, 1], torch.ones_like(w[:, 1]), atol=1e-4))
        self.assertTrue(torch.allclose(w[:, 2], 2 * torch.ones_like(w[:, 2]), atol=1e-4))
        seen = {}

        def grab(module, inputs, output):
            seen.setdefault("x", inputs[0].detach())

        net.stem.register_forward_hook(grab)
        net(x, ctx, bm)
        s = seen["x"]
        self.assertTrue(torch.allclose(s[:, 0], torch.zeros_like(s[:, 0]), atol=1e-3))
        self.assertTrue(torch.allclose(s[:, 1], x[:, 1], atol=1e-3))
        self.assertTrue(torch.allclose(s[:, 2], 2 * x[:, 2], atol=1e-3))
        self.assertTrue(torch.equal(s[:, 3:], x[:, 3:]))                    # scalar planes untouched

    def test_gate_is_absent_without_waveform_planes_or_when_switched_off(self):
        self.assertIsNone(BshiNet(ModelConfig(gate=False)).gate)
        self.assertIsNone(BshiNet(ModelConfig(variant="map_only")).gate)
        self.assertIsNotNone(BshiNet(ModelConfig(variant="ea")).gate)


class TestClamp(unittest.TestCase):
    def test_clamp_touches_only_scalar_planes_3_to_5(self):
        x = torch.full((2, 7, 60, 200), 9.0)
        y = A.clamp_scalar_planes(x, 6.0)
        self.assertTrue((y[:, 3:6] == 6.0).all())
        self.assertTrue((y[:, :3] == 9.0).all() and (y[:, 6] == 9.0).all())
        self.assertTrue((x == 9.0).all())                                   # input not modified


if __name__ == "__main__":
    unittest.main()
