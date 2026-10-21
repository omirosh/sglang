"""ROCm coverage for the online MXFP4 Qwen3.5/3.8 GDN projections."""

import itertools
import unittest

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, suite="stage-b-test-1-gpu-small-amd")

# Qwen3.8-2.4T TP8: in_proj_qkvz and out_proj.
SHAPES = ((4608, 8192), (8192, 2048))
# Both sides of the preshuffled-scale split at 32 rows and of the 64-row cut.
ROWS = (1, 4, 16, 31, 32, 64, 65, 256)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.version.hip and is_gfx95_supported(),
    "requires AMD gfx95",
)
class TestQwen35GdnOnlineMxfp4(CustomTestCase):
    def test_matches_dequantized_reference(self):
        from aiter import per_1x32_f4_quant_hip

        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            AiterMxfp4OnlineLinearMethod,
            _dequant_mxfp4_to_bf16,
            _dynamic_mxfp4_quant_orig,
        )

        for (n, k), m in itertools.product(SHAPES, ROWS):
            with self.subTest(n=n, k=k, m=m):
                torch.manual_seed(m)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) * 0.02
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                w_q, w_s = _dynamic_mxfp4_quant_orig(w)
                x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
                ref = torch.nn.functional.linear(
                    _dequant_mxfp4_to_bf16(
                        x_q.view(torch.uint8), x_s.view(torch.uint8)[:m]
                    ).float(),
                    _dequant_mxfp4_to_bf16(w_q, w_s.contiguous()).float(),
                )

                layer = torch.nn.Module()
                layer.weight = torch.nn.Parameter(w, requires_grad=False)
                AiterMxfp4OnlineLinearMethod.quantize_(layer)
                y = layer.quant_method.apply(layer, x)

                self.assertEqual(y.shape, (m, n))
                rel = ((y.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 1e-2)

    def test_packed_qkvz_ba(self):
        from types import SimpleNamespace

        from aiter import per_1x32_f4_quant_hip

        from sglang.srt.layers.quantization.quark.schemes.quark_w4a4_mxfp4 import (
            AiterMxfp4OnlineLinearMethod,
            _dequant_mxfp4_to_bf16,
            _dynamic_mxfp4_quant_orig,
        )

        (n, k), ba = SHAPES[0], 32
        torch.manual_seed(0)
        w = torch.randn(n + ba, k, device="cuda", dtype=torch.bfloat16) * 0.02
        packed = SimpleNamespace(weight=w)
        AiterMxfp4OnlineLinearMethod.quantize_(packed, config_n=n)
        self.assertTrue(packed.mxfp4_preshuffle_tuned)
        # The model reads in_proj_qkvz from the head of the packed weight.
        qkvz = torch.nn.Module()
        qkvz.weight = torch.nn.Parameter(w[:n].clone(), requires_grad=False)
        AiterMxfp4OnlineLinearMethod.quantize_(qkvz)
        self.assertTrue(torch.equal(packed.weight[:n], qkvz.weight))
        self.assertTrue(torch.equal(packed.weight_scale[:n], qkvz.weight_scale))

        w_q, w_s = _dynamic_mxfp4_quant_orig(w)
        w_ref = _dequant_mxfp4_to_bf16(w_q, w_s.contiguous()).float()
        for m in (1, 4, 16, 31, 32, 64):
            with self.subTest(m=m):
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
                x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=False)
                ref = torch.nn.functional.linear(
                    _dequant_mxfp4_to_bf16(
                        x_q.view(torch.uint8), x_s.view(torch.uint8)[:m]
                    ).float(),
                    w_ref,
                )
                y = packed.quant_method.apply(packed, x)
                self.assertEqual(y.shape, (m, n + ba))
                rel = ((y.float() - ref).norm() / ref.norm()).item()
                self.assertLess(rel, 1e-2)


if __name__ == "__main__":
    unittest.main()
