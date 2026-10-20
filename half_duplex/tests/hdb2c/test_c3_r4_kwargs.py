"""Regression for NeMo keyword-only block invocation in R4 tracing.

Uses real torch.Module hook dispatch, not a hand-called callback. It never
downloads model weights or runs synthesized ASR.
"""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from hdb2c.c3_r4 import attach_nemo_r4_hooks, verify_r4_hooks, STAGES
from hdb2c.c3_r3 import read_tensor


class ToyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        for attr in (
            "norm_feed_forward1", "feed_forward1", "norm_self_att",
            "self_attn", "norm_conv", "conv", "norm_feed_forward2",
            "feed_forward2", "norm_out"
        ):
            setattr(self, attr, nn.Identity())

    def forward(self, x):
        residual = x
        y = self.norm_feed_forward1(residual)
        residual = residual + self.feed_forward1(y) * 0.5
        y = self.norm_self_att(residual)
        residual = residual + self.self_attn(y)
        y = self.norm_conv(residual)
        residual = residual + self.conv(y)
        y = self.norm_feed_forward2(residual)
        residual = residual + self.feed_forward2(y) * 0.5
        return self.norm_out(residual)


class ToyPosition(nn.Module):
    def forward(self, x):
        return x, x.clone()


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([ToyBlock()] + [nn.Identity() for _ in range(23)])
        self.pos_enc = ToyPosition()


class R4KwargsTest(unittest.TestCase):
    def test_keyword_and_positional_neural_hook_execution(self):
        for as_keyword in (True, False):
            with self.subTest(keyword=as_keyword), tempfile.TemporaryDirectory() as temp:
                encoder = ToyEncoder()
                folder = Path(temp)
                handles, seen = attach_nemo_r4_hooks(encoder, folder)
                x = torch.ones((1, 5, 4), dtype=torch.float32)
                try:
                    encoder.pos_enc(x)
                    if as_keyword:
                        encoder.layers[0](x=x)
                    else:
                        encoder.layers[0](x)
                    verify_r4_hooks(seen)
                    self.assertEqual(set(STAGES) | {"block0_pos_emb"}, seen)
                    self.assertEqual(read_tensor(folder / "block0_input.f32le").shape, (1, 5, 4))
                    self.assertTrue((read_tensor(folder / "block0_input.f32le") == 1.).all())
                finally:
                    for handle in handles:
                        handle.remove()

    def test_missing_keyword_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            encoder = ToyEncoder()
            handles, _ = attach_nemo_r4_hooks(encoder, Path(temp))
            try:
                with self.assertRaisesRegex(RuntimeError, "R4 hook without input: block0_input"):
                    encoder.layers[0](unexpected=torch.ones((1, 2, 3)))
            finally:
                for handle in handles:
                    handle.remove()

    def test_ambiguous_input_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            encoder = ToyEncoder()
            handles, _ = attach_nemo_r4_hooks(encoder, Path(temp))
            try:
                with self.assertRaisesRegex(RuntimeError, "R4 ambiguous positional/keyword input"):
                    encoder.layers[0](torch.ones((1, 2, 3)), x=torch.ones((1, 2, 3)))
            finally:
                for handle in handles:
                    handle.remove()


if __name__ == "__main__":
    unittest.main()
