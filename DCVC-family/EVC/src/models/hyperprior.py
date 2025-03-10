# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from torch import nn


class DepthConv(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1, slope=0.01):
        super().__init__()
        dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, dw_ch, 1, stride=stride),
            nn.LeakyReLU(negative_slope=slope),
        )

        self.depth_conv = nn.Conv2d(dw_ch, dw_ch, depth_kernel, padding=depth_kernel // 2,
                                    groups=dw_ch)
        self.conv2 = nn.Conv2d(dw_ch, out_ch, 1)

        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        elif in_ch != out_ch:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)
        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)
        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch, slope=0.1):
        super().__init__()
        internal_ch = max(min(in_ch * 4, 1024), in_ch * 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, internal_ch, 1),
            nn.LeakyReLU(negative_slope=slope),
            nn.Conv2d(internal_ch, in_ch, 1),
            nn.LeakyReLU(negative_slope=slope),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1,
                 slope_depth_conv=0.01, slope_ffn=0.1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, stride, slope=slope_depth_conv),
            ConvFFN(out_ch, slope=slope_ffn),
        )

    def forward(self, x):
        return self.block(x)


class DepthConvBlockUpsample(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, slope_depth_conv=0.01, slope_ffn=0.1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel, slope=slope_depth_conv),
            ConvFFN(out_ch, slope=slope_ffn),
            nn.Conv2d(out_ch, out_ch * 4, 1),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.block(x)


def get_hyperprior(channel=192):
    N = channel
    hyper_enc = nn.Sequential(
        DepthConvBlock(N, N, stride=1),
        nn.Conv2d(N, N, 3, stride=2, padding=1),
        nn.LeakyReLU(),
        nn.Conv2d(N, N, 3, stride=2, padding=1),
    )
    hyper_dec = nn.Sequential(
        DepthConvBlockUpsample(N, N),
        DepthConvBlockUpsample(N, N),
        DepthConvBlock(N, N),
    )
    y_prior_fusion = nn.Sequential(
        DepthConvBlock(N, N * 2),
        DepthConvBlock(N * 2, N * 3),
    )
    return hyper_enc, hyper_dec, y_prior_fusion


def get_dualprior(channel=192):
    N = channel
    y_spatial_prior = nn.Sequential(
        DepthConvBlock(N * 4, N * 3),
        DepthConvBlock(N * 3, N * 2),
        DepthConvBlock(N * 2, N * 2),
    )
    return y_spatial_prior
