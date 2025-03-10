# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
import torch.nn.functional as F


from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8

g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 368


class IntraEncoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, N, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(out, quant_step)

        return self.forward_cuda(out, quant_step)

    def forward_torch(self, out, quant_step):
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)

    def forward_cuda(self, out, quant_step):
        out = self.enc_1(out, quant_step=quant_step)
        return self.enc_2(out)


class IntraDecoder(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(N, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
        )
        self.dec_2 = DepthConvBlock(g_ch_enc_dec, g_ch_src)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out

    def forward_cuda(self, x, quant_step):
        out = self.dec_1[0](x)
        out = self.dec_1[1](out)
        out = self.dec_1[2](out)
        out = self.dec_1[3](out)
        out = self.dec_1[4](out)
        out = self.dec_1[5](out)
        out = self.dec_1[6](out)
        out = self.dec_1[7](out)
        out = self.dec_1[8](out)
        out = self.dec_1[9](out)
        out = self.dec_1[10](out)
        out = self.dec_1[11](out)
        out = self.dec_1[12](out, quant_step=quant_step)
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class DMCI(CompressionModel):
    def __init__(self, N=256, z_channel=128):
        super().__init__(z_channel=z_channel)

        self.enc = IntraEncoder(N)

        self.hyper_enc = nn.Sequential(
            DepthConvBlock(N, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
            ResidualBlockWithStride2(z_channel, z_channel),
        )

        self.hyper_dec = nn.Sequential(
            ResidualBlockUpsample(z_channel, z_channel),
            ResidualBlockUpsample(z_channel, z_channel),
            DepthConvBlock(z_channel, N),
        )

        self.y_prior_fusion = nn.Sequential(
            DepthConvBlock(N, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2 + 2, 1),
        )

        self.y_spatial_prior_reduction = nn.Conv2d(N * 2 + 2, N * 1, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(N * 2, N * 2, force_adaptor=True)
        self.y_spatial_prior = nn.Sequential(
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            DepthConvBlock(N * 2, N * 2),
            nn.Conv2d(N * 2, N * 2, 1),
        )

        self.dec = IntraDecoder(N)

        self.q_scale_enc = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.get_qp_num(), g_ch_enc_dec, 1, 1)))

    def compress(self, x, qp):
        device = x.device
        curr_q_enc = self.q_scale_enc[qp:qp+1, :, :, :]
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        y = self.enc(x, curr_q_enc)
        y_pad = self.pad_for_y(y)
        z = self.hyper_enc(y_pad)
        z_hat, z_hat_write = round_and_to_int8(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW].contiguous()
        y_q_w_0, y_q_w_1, y_q_w_2, y_q_w_3, s_w_0, s_w_1, s_w_2, s_w_3, y_hat = \
            self.compress_prior_4x(
                y, params, self.y_spatial_prior_reduction,
                self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
                self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        cuda_event = torch.cuda.Event()
        cuda_event.record()
        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            cuda_event.wait()
            self.entropy_coder.reset()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.gaussian_encoder.encode_y(y_q_w_2, s_w_2)
            self.gaussian_encoder.encode_y(y_q_w_3, s_w_3)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        curr_q_dec = self.q_scale_dec[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        y_height, y_width = self.get_downsampled_shape(sps['height'], sps['width'], 16)
        self.bit_estimator_z.decode_z(z_size, qp)
        z_q = self.bit_estimator_z.get_z(z_size, device, dtype)
        z_hat = z_q

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        params = params[:, :, :y_height, :y_width].contiguous()
        y_hat = self.decompress_prior_4x(params, self.y_spatial_prior_reduction,
                                         self.y_spatial_prior_adaptor_1,
                                         self.y_spatial_prior_adaptor_2,
                                         self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec).clamp_(0, 1)
        return {"x_hat": x_hat}
