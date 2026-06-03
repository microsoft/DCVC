# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F

from torch import nn
from torch.autograd import Function

from ..utils.transforms import ycbcr2rgb


def bit_estimator_z_prob(x, h, b, a):
    # this is accumulated prob
    for i in range(4):
        x = x * F.softplus(h[:, :, i:i+1, None]) + b[:, :, i:i+1, None]
        if i != 3:
            x = x + torch.tanh(x) * torch.tanh(a[:, :, i:i+1, None])
    return torch.sigmoid(x)


def bit_estimator_z_fwd(x, h, b, a):
    dtype = x.dtype
    x = x.float()
    h = h.float()
    b = b.float()
    a = a.float()
    lower = bit_estimator_z_prob(x - 0.5, h, b, a)
    upper = bit_estimator_z_prob(x + 0.5, h, b, a)
    prob = upper - lower
    return prob.to(dtype)


def get_mse_yuv_rgb(x, x_hat):
    mse_yuv = torch.sum(F.mse_loss(x, x_hat, reduction='none'), dim=(2, 3))
    org_rgb = ycbcr2rgb(x, clamp=False)
    rec_rgb = ycbcr2rgb(x_hat, clamp=False)
    mse_rgb = torch.sum(F.mse_loss(org_rgb, rec_rgb, reduction='none'), dim=(1, 2, 3))
    return mse_yuv, mse_rgb


def mse_8frames_sum(mse, dist_weights):
    result = (mse[0] + mse[2] + mse[4] + mse[6]) * dist_weights[1] + \
        (mse[1] + mse[3] + mse[5]) * dist_weights[2] + mse[7] * dist_weights[0]
    return result


def mse_weighted_average(mse_yuv, mse_rgb, pixel_num):
    dtype = mse_yuv.dtype
    mse_yuv = mse_yuv.float()
    mse_rgb = mse_rgb.float()
    mse_yuv = mse_yuv / pixel_num
    mse_y, mse_u, mse_v = mse_yuv[:, 0], mse_yuv[:, 1], mse_yuv[:, 2]
    mse_yuv = torch.exp(
        0.0833 * (10 * torch.log(torch.clamp_min(mse_y, 1e-6))
                  + torch.log(torch.clamp_min(mse_u, 1e-6))
                  + torch.log(torch.clamp_min(mse_v, 1e-6)))) * 3
    mse_rgb = mse_rgb / pixel_num
    mse = mse_yuv * 0.8 + mse_rgb * 0.2
    return mse.to(dtype)


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        ctx.save_for_backward(inputs)
        ctx.bound = bound
        return torch.clamp_min(inputs, bound)

    @staticmethod
    def backward(ctx, grad):
        inputs = ctx.saved_tensors[0]
        bound = ctx.bound

        pass_through_1 = inputs >= bound
        pass_through_2 = grad < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through * grad, None


class QuantFunc(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad):
        return grad


class SubpelConv2x(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding=0, force_bias=False):
        super().__init__()
        has_bias = (kernel_size > 1) or force_bias
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding, bias=has_bias),
            nn.PixelShuffle(2),
        )
        self.padding = padding

    def forward(self, x):
        return self.conv(x)


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.wsilu = WSiLU()

    def forward(self, x):
        x = self.wsilu(x)
        x1 = x[:, 0::4, :, :]
        x2 = x[:, 1::4, :, :]
        x3 = x[:, 2::4, :, :]
        x4 = x[:, 3::4, :, :]
        return x1 + x2 + x3 + x4


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        ch_ratio = 1
        if dcb2:
            assert not shortcut
            ch_ratio = 2
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch // ch_ratio, 1),
            WSiLU(),
            nn.Conv2d(out_ch // ch_ratio, out_ch // ch_ratio, 3, padding=1,
                      groups=out_ch // ch_ratio),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4 // ch_ratio, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch // ch_ratio, out_ch, 1),
        )

    def forward(self, x):
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out


class ResidualBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True, force_bias=False):
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1, force_bias=force_bias)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)
        return out


class ResidualBlockWithStride2(nn.Module):
    def __init__(self, in_ch, out_ch, dcb2=False, shortcut=True):
        super().__init__()
        self.down = nn.Conv2d(in_ch * 4, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, dcb2=dcb2, shortcut=shortcut)

        self.shortcut = shortcut

    def forward(self, x):
        out = F.pixel_unshuffle(x, 2)
        out = self.down(out)
        out = self.conv(out)
        return out
