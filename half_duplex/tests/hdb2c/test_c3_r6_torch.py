"""Actual Torch hook dispatch for masks supplied by keyword arguments."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from hdb2c.c3_r3 import read_tensor
from hdb2c.c3_r6 import attach_nemo_r6_hooks


class MaskBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn=nn.Module(); self.self_attn.use_pytorch_sdpa=False
        self.conv=nn.Module(); self.conv.pointwise_activation='glu_'
        self.conv.depthwise_conv=nn.Identity()

    def forward(self,x,att_mask=None,pad_mask=None):
        return self.conv.depthwise_conv(x.transpose(1,2)).transpose(1,2)


class MaskEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention_model='rel_pos'; self.att_context_style='regular'
        self.att_context_size=[-1,-1]; self.use_pad_mask=True
        self.layers=nn.ModuleList([MaskBlock()])


class R6HookTests(unittest.TestCase):
    def test_masks_and_actual_depthwise_input_are_captured(self):
        with tempfile.TemporaryDirectory() as temp:
            enc=MaskEncoder().eval(); folder=Path(temp)
            hooks,seen=attach_nemo_r6_hooks(enc,folder)
            x=torch.zeros((1,4,2)); x[:,:3]=1.
            pad=torch.tensor([[False,False,False,True]])
            att=pad.unsqueeze(2)|pad.unsqueeze(1)
            try:
                enc.layers[0](x=x,att_mask=att,pad_mask=pad)
                self.assertEqual(seen,{'encoder_pad_mask','encoder_att_mask','block0_conv_after_mask'})
                self.assertEqual(read_tensor(folder/'encoder_att_mask.f32le').sum(),7)
                self.assertTrue((read_tensor(folder/'block0_conv_after_mask.f32le')[:,3:]==0).all())
            finally:
                for hook in hooks:hook.remove()

    def test_missing_real_masks_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            enc=MaskEncoder().eval()
            hooks,_=attach_nemo_r6_hooks(enc,Path(temp))
            try:
                with self.assertRaisesRegex(RuntimeError,'masks missing'):
                    enc.layers[0](x=torch.ones((1,4,2)))
            finally:
                for hook in hooks:hook.remove()


if __name__=='__main__':
    unittest.main()
