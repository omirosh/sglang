"""MRotaryEmbedding on 1-D (text) positions, as an MTP draft passes them: the fused
Triton kernel on the same position on each axis gives the native rotation."""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.mrope import MRotaryEmbedding
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")
register_amd_ci(est_time=3, suite="base-a-test-1-gpu-small-amd")


@unittest.skipUnless(torch.cuda.is_available(), "the Triton kernel needs a GPU")
class TestMRopeTextPositions(CustomTestCase):
    def setUp(self):
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    def test_text_positions_match_the_native_rotation(self):
        # Qwen3.5's attention head: 256, a quarter rotary; one TP rank's heads
        for interleaved in (False, True):
            with self.subTest(interleaved=interleaved):
                rope = MRotaryEmbedding(
                    head_size=256,
                    rotary_dim=64,
                    max_position_embeddings=1 << 18,
                    base=10_000_000,
                    is_neox_style=True,
                    dtype=torch.bfloat16,
                    mrope_section=[11, 11, 10],
                    mrope_interleaved=interleaved,
                ).cuda()
                g = torch.Generator(device="cuda").manual_seed(0)
                pos = torch.randint(0, 1 << 18, (8,), device="cuda", generator=g)
                shape = (8, 8 * 256), (8, 256)
                q, k = (
                    torch.randn(s, device="cuda", dtype=torch.bfloat16, generator=g)
                    for s in shape
                )
                rope._match_cos_sin_cache_dtype(q)
                want = rope.forward_native(pos, q.clone(), k.clone())
                got = rope.forward_cuda(pos, q.clone(), k.clone())
                for w, x in zip(want, got):
                    w, x = w.to(x.dtype).view(8, -1, 256), x.view(8, -1, 256)
                    torch.testing.assert_close(x[..., 64:], w[..., 64:], atol=0, rtol=0)
                    torch.testing.assert_close(
                        x[..., :64], w[..., :64], atol=2e-2, rtol=1e-2
                    )


if __name__ == "__main__":
    unittest.main()
