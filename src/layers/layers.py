# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from .cuda_inference import CUSTOMIZED_CUDA_INFERENCE
if CUSTOMIZED_CUDA_INFERENCE:
    from .cuda_inference import DepthConvProxy, SubpelConv2xProxy


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, to_cat, cat_at_front)

        return self.forward_cuda(x, to_cat, cat_at_front)

    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)

    def forward_cuda(self, x, to_cat=None, cat_at_front=True):
        if self.proxy is None:
            self.proxy = SubpelConv2xProxy()
            self.proxy.set_param(self.conv[0].weight, self.conv[0].bias, self.padding)

        if to_cat is None:
            return self.proxy.forward(x)

        return self.proxy.forward_with_cat(x, to_cat, cat_at_front)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )

        self.proxy = None

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step, to_cat, cat_at_front)

        return self.forward_cuda(x, quant_step, to_cat, cat_at_front)

    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out

    def forward_cuda(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        if self.proxy is None:
            self.proxy = DepthConvProxy()
            if self.adaptor is not None:
                self.proxy.set_param_with_adaptor(self.dc[0].weight, self.dc[0].bias,
                                                  self.dc[2].weight, self.dc[2].bias,
                                                  self.dc[3].weight, self.dc[3].bias,
                                                  self.ffn[0].weight, self.ffn[0].bias,
                                                  self.ffn[2].weight, self.ffn[2].bias,
                                                  self.adaptor.weight, self.adaptor.bias,
                                                  self.shortcut)
            else:
                self.proxy.set_param(self.dc[0].weight, self.dc[0].bias,
                                     self.dc[2].weight, self.dc[2].bias,
                                     self.dc[3].weight, self.dc[3].bias,
                                     self.ffn[0].weight, self.ffn[0].bias,
                                     self.ffn[2].weight, self.ffn[2].bias,
                                     self.shortcut)

        if quant_step is not None:
            return self.proxy.forward_with_quant_step(x, quant_step)
        if to_cat is not None:
            return self.proxy.forward_with_cat(x, to_cat, cat_at_front)

        return self.proxy.forward(x)


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out
