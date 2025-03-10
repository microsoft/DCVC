# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn


from .common_model import CompressionModel
from .layers import conv3x3, DepthConvBlock2, DepthConvBlock3, DepthConvBlock4, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from .video_net import UNet
from ..utils.stream_helper import write_ip, get_downsampled_shape


class IntraEncoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.enc_1 = nn.Sequential(
            ResidualBlockWithStride2(3, 128, inplace=inplace),
            DepthConvBlock3(128, 128, inplace=inplace),
        )
        self.enc_2 = nn.Sequential(
            ResidualBlockWithStride2(128, 192, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockWithStride2(192, N, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            nn.Conv2d(N, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = self.enc_1(x)
        out = out * quant_step
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N, inplace=False):
        super().__init__()

        self.dec_1 = nn.Sequential(
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample(N, N, 2, inplace=inplace),
            DepthConvBlock3(N, N, inplace=inplace),
            ResidualBlockUpsample(N, 192, 2, inplace=inplace),
            DepthConvBlock3(192, 192, inplace=inplace),
            ResidualBlockUpsample(192, 128, 2, inplace=inplace),
        )
        self.dec_2 = nn.Sequential(
            DepthConvBlock3(128, 128, inplace=inplace),
            ResidualBlockUpsample(128, 16, 2, inplace=inplace),
        )

    def forward(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        return self.dec_2(out)


class DMCI(CompressionModel):
    def __init__(self, N=256, z_channel=128, ec_thread=False, stream_part=1, inplace=False):
        super().__init__(y_distribution='gaussian', z_channel=z_channel,
                         ec_thread=ec_thread, stream_part=stream_part)

        self.enc = IntraEncoder(N, inplace)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock4(N, z_channel, inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
            nn.LeakyReLU(inplace=inplace),
            nn.Conv2d(z_channel, z_channel, 3, stride=2, padding=1),
        )
        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            ResidualBlockUpsample(z_channel, z_channel, 2, inplace=inplace),
            DepthConvBlock4(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock4(N, N * 2, inplace=inplace),
            DepthConvBlock4(N * 2, N * 2 + 2, inplace=inplace),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock2(N * 2, N * 2, inplace=inplace)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
            DepthConvBlock2(N * 2, N * 2, inplace=inplace),
        )

        self.dec = IntraDecoder(N, inplace)
        self.refine = nn.Sequential(
            UNet(16, 16, inplace=inplace),
            conv3x3(16, 3),
        )

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), 128, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), 128, 1, 1)))

    def forward_one_frame(self, x, q_index=None):
        _, _, H, W = x.size()
        device = x.device
        index = self.get_index_tensor(q_index, device)
        curr_q_enc = torch.index_select(self.q_scale_enc, 0, index)
        curr_q_dec = torch.index_select(self.q_scale_dec, 0, index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_q = self.quant(z)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_res, y_q, y_hat, scales_hat = self.forward_four_part_prior(
            y, params,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        x_hat = self.dec(y_hat, curr_q_dec)
        x_hat = self.refine(x_hat)

        y_for_bit = y_q
        z_for_bit = z_q
        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z, index)
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        bits = torch.sum(bpp_y + bpp_z) * pixel_num

        return {
            "x_hat": x_hat,
            "bit": bits,
        }

    def encode(self, x, q_index, sps_id=0, output_file=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_file is None:
            encoded = self.forward_one_frame(x, q_index)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
            }
            return result

        compressed = self.compress(x, q_index)
        bit_stream = compressed['bit_stream']
        written = write_ip(output_file, True, sps_id, bit_stream)
        result = {
            'bit': written * 8,
            'x_hat': compressed['x_hat'],
        }
        return result

    def compress(self, x, q_index):
        device = x.device
        index = self.get_index_tensor(q_index, device)
        curr_q_enc = torch.index_select(self.q_scale_enc, 0, index)
        curr_q_dec = torch.index_select(self.q_scale_dec, 0, index)

        y = self.enc(x, curr_q_enc)
        y_pad, slice_shape = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_q = torch.round(z)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, \
            scales_w_0, scales_w_1, scales_w_2, scales_w_3, y_hat = self.compress_four_part_prior(
                y, params, self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
                y_spatial_prior_reduction=self.y_spatial_prior_reduction)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_q, q_index)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.gaussian_encoder.encode(y_q_w_2, scales_w_2)
        self.gaussian_encoder.encode(y_q_w_3, scales_w_3)
        self.entropy_coder.flush()

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, sps):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        index = self.get_index_tensor(sps['qp'], device)
        curr_q_dec = torch.index_select(self.q_scale_dec, 0, index)

        self.entropy_coder.set_stream(bit_stream)
        z_size = get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = get_downsampled_shape(sps['height'], sps['width'], 16)
        slice_shape = self.get_to_y_slice_shape(y_height, y_width)
        z_q = self.bit_estimator_z.decode_stream(z_size, dtype, device, sps['qp'])
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = self.slice_to_y(params, slice_shape)
        y_hat = self.decompress_four_part_prior(params,
                                                self.y_spatial_prior_adaptor_1,
                                                self.y_spatial_prior_adaptor_2,
                                                self.y_spatial_prior_adaptor_3,
                                                self.y_spatial_prior,
                                                self.y_spatial_prior_reduction)

        x_hat = self.refine(self.dec(y_hat, curr_q_dec)).clamp_(0, 1)
        return {"x_hat": x_hat}
