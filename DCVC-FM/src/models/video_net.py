import torch
from torch import nn
import torch.nn.functional as F

from .layers import subpel_conv1x1, DepthConvBlock2, DepthConvBlock4
from .block_mc import block_mc_func


def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature


def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, channel, slope=0.01, end_with_relu=False,
                 bottleneck=False, inplace=False):
        super().__init__()
        in_channel = channel // 2 if bottleneck else channel
        self.first_layer = nn.LeakyReLU(negative_slope=slope, inplace=False)
        self.conv1 = nn.Conv2d(channel, in_channel, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=slope, inplace=inplace)
        self.conv2 = nn.Conv2d(in_channel, channel, 3, padding=1)
        self.last_layer = self.relu if end_with_relu else nn.Identity()

    def forward(self, x):
        identity = x
        out = self.first_layer(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.last_layer(out)
        return identity + out


class MEBasic(nn.Module):
    def __init__(self, complexity_level=0):
        super().__init__()
        self.relu = nn.ReLU()
        self.by_pass = False
        if complexity_level < 0:
            self.by_pass = True
        elif complexity_level == 0:
            self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 3:
            self.conv1 = nn.Conv2d(8, 32, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
            self.conv3 = nn.Conv2d(64, 32, 5, 1, padding=2)
            self.conv4 = nn.Conv2d(32, 16, 5, 1, padding=2)
            self.conv5 = nn.Conv2d(16, 2, 5, 1, padding=2)

    def forward(self, x):
        if self.by_pass:
            return x[:, -2:, :, :]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.me_8x = MEBasic(0)
        self.me_4x = MEBasic(0)
        self.me_2x = MEBasic(3)
        self.me_1x = MEBasic(3)

    def forward(self, im1, im2):
        batchsize = im1.size()[0]

        im1_1x = im1
        im1_2x = F.avg_pool2d(im1_1x, kernel_size=2, stride=2)
        im1_4x = F.avg_pool2d(im1_2x, kernel_size=2, stride=2)
        im1_8x = F.avg_pool2d(im1_4x, kernel_size=2, stride=2)
        im2_1x = im2
        im2_2x = F.avg_pool2d(im2_1x, kernel_size=2, stride=2)
        im2_4x = F.avg_pool2d(im2_2x, kernel_size=2, stride=2)
        im2_8x = F.avg_pool2d(im2_4x, kernel_size=2, stride=2)

        shape_fine = im1_8x.size()
        zero_shape = [batchsize, 2, shape_fine[2], shape_fine[3]]
        flow_8x = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        flow_8x = self.me_8x(torch.cat((im1_8x, im2_8x, flow_8x), dim=1))

        flow_4x = bilinearupsacling(flow_8x) * 2.0
        flow_4x = flow_4x + self.me_4x(torch.cat((im1_4x,
                                                  block_mc_func(im2_4x, flow_4x),
                                                  flow_4x),
                                                 dim=1))

        flow_2x = bilinearupsacling(flow_4x) * 2.0
        flow_2x = flow_2x + self.me_2x(torch.cat((im1_2x,
                                                  block_mc_func(im2_2x, flow_2x),
                                                  flow_2x),
                                                 dim=1))

        flow_1x = bilinearupsacling(flow_2x) * 2.0
        flow_1x = flow_1x + self.me_1x(torch.cat((im1_1x,
                                                  block_mc_func(im2_1x, flow_1x),
                                                  flow_1x),
                                                 dim=1))
        return flow_1x


class UNet(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.conv1 = DepthConvBlock2(in_ch, 32, inplace=inplace)
        self.down1 = nn.Conv2d(32, 32, 2, stride=2)
        self.conv2 = DepthConvBlock2(32, 64, inplace=inplace)
        self.down2 = nn.Conv2d(64, 64, 2, stride=2)
        self.conv3 = DepthConvBlock2(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
            DepthConvBlock2(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock2(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock2(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.down1(x1)

        x2 = self.conv2(x2)
        x3 = self.down2(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2


class UNet2(nn.Module):
    def __init__(self, in_ch=64, out_ch=64, inplace=False):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = DepthConvBlock4(in_ch, 32, inplace=inplace)
        self.conv2 = DepthConvBlock4(32, 64, inplace=inplace)
        self.conv3 = DepthConvBlock4(64, 128, inplace=inplace)

        self.context_refine = nn.Sequential(
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
            DepthConvBlock4(128, 128, inplace=inplace),
        )

        self.up3 = subpel_conv1x1(128, 64, 2)
        self.up_conv3 = DepthConvBlock4(128, 64, inplace=inplace)

        self.up2 = subpel_conv1x1(64, 32, 2)
        self.up_conv2 = DepthConvBlock4(64, out_ch, inplace=inplace)

    def forward(self, x):
        # encoding path
        x1 = self.conv1(x)
        x2 = self.max_pool(x1)

        x2 = self.conv2(x2)
        x3 = self.max_pool(x2)

        x3 = self.conv3(x3)
        x3 = self.context_refine(x3)

        # decoding + concat path
        d3 = self.up3(x3)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.up_conv2(d2)
        return d2
