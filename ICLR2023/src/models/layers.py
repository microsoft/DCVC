# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from torch.autograd import Function


# pylint: disable=W0221
class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None
# pylint: enable=W0221


def conv3x3(in_ch, out_ch, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def subpel_conv3x3(in_ch, out_ch, r=1):
    """3x3 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=3, padding=1), nn.PixelShuffle(r)
    )


def subpel_conv1x1(in_ch, out_ch, r=1):
    """1x1 sub-pixel convolution for up-sampling."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch * r ** 2, kernel_size=1, padding=0), nn.PixelShuffle(r)
    )


def conv1x1(in_ch, out_ch, stride=1, bias=True):
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, bias=bias, kernel_size=1, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.
    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """
    def __init__(self, in_ch1, in_ch2, in_ch3=None, stride=2):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.conv1 = conv3x3(in_ch1, in_ch2, stride=stride)
        self.leaky_relu = nn.LeakyReLU()
        self.conv2 = conv3x3(in_ch2, in_ch3)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        if stride != 1:
            self.downsample = conv1x1(in_ch1, in_ch3, stride=stride)
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch1, in_ch2, in_ch3=None, upsample=2):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.subpel_conv = subpel_conv1x1(in_ch1, in_ch2, upsample)
        self.leaky_relu = nn.LeakyReLU()
        self.conv = conv3x3(in_ch2, in_ch3)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.upsample = subpel_conv1x1(in_ch1, in_ch3, upsample)

    def forward(self, x):
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.leaky_relu2(out)
        identity = self.upsample(x)
        out = out + identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch1, in_ch2, in_ch3=None, leaky_relu_slope=0.01):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
        self.conv1 = conv3x3(in_ch1, in_ch2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.conv2 = conv3x3(in_ch2, in_ch3)
        self.adaptor = None
        if in_ch1 != in_ch3:
            self.adaptor = conv1x1(in_ch1, in_ch3)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        out = out + identity
        return out


class DepthConv(nn.Module):
    def __init__(self, in_ch1, in_ch2, in_ch3=None, depth_kernel=3, stride=1):
        super().__init__()
        if in_ch3 is None:
            in_ch3 = in_ch2
            in_ch2 = in_ch1
        # dw_ch = in_ch * 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1, stride=stride),
            nn.LeakyReLU(),
        )
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch2, depth_kernel, padding=depth_kernel // 2, groups=in_ch2),
            nn.LeakyReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch2, in_ch3, 1),
            nn.LeakyReLU(),
        )
        self.adaptor = None
        if stride != 1:
            assert stride == 2
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 2, stride=2)
        elif in_ch1 != in_ch3:
            self.adaptor = nn.Conv2d(in_ch1, in_ch3, 1)

    def forward(self, x):
        identity = x
        if self.adaptor is not None:
            identity = self.adaptor(identity)

        out = self.conv1(x)
        out = self.depth_conv(out)
        out = self.conv2(out)

        return out + identity


class ConvFFN(nn.Module):
    def __init__(self, in_ch1, in_ch2=None):
        super().__init__()
        if in_ch2 is None:
            in_ch2 = in_ch1 * 4
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch1, in_ch2, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch2, in_ch1, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        return identity + self.conv(x)


class DepthConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, depth_kernel=3, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            DepthConv(in_ch, out_ch, depth_kernel=depth_kernel, stride=stride),
            ConvFFN(out_ch),
        )

    def forward(self, x):
        return self.block(x)


def get_enc_dec_models(input_channel, output_channel, channels=[64, 64, 128, 192]):
    channel_2x = channels[0]
    channel_4x = channels[1]
    channel_8x = channels[2]
    channel_16x = channels[3]

    enc = nn.Sequential(
        ResidualBlockWithStride(input_channel, channel_2x, stride=2),
        DepthConvBlock(channel_2x, channel_2x),
        ResidualBlockWithStride(channel_2x, channel_4x, stride=2),
        DepthConvBlock(channel_4x, channel_4x),
        ResidualBlockWithStride(channel_4x, channel_8x, stride=2),
        DepthConvBlock(channel_8x, channel_8x),
        conv3x3(channel_8x, channel_16x, stride=2),
    )

    dec = nn.Sequential(
        DepthConvBlock(channel_16x, channel_16x),
        ResidualBlockUpsample(channel_16x, channel_8x, upsample=2),
        DepthConvBlock(channel_8x, channel_8x),
        ResidualBlockUpsample(channel_8x, channel_4x, upsample=2),
        DepthConvBlock(channel_4x, channel_4x),
        ResidualBlockUpsample(channel_4x, channel_2x, upsample=2),
        DepthConvBlock(channel_2x, channel_2x),
        subpel_conv1x1(channel_2x, output_channel, 2),
    )
    return enc, dec
