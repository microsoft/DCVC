# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .common_model import CompressionModel
from ..layers.layers import DepthConvBlock, QuantFunc, ResidualBlockUpsample, ResidualBlockWithStride2
from ..utils.common import loss_func


g_ch_src = 3 * 8 * 8
g_ch_enc_dec = 384
g_ch_y = 256
g_ch_z = 128


class IntraDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.dec_1 = nn.Sequential(
            ResidualBlockUpsample(g_ch_y, g_ch_enc_dec),
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
        out = self.dec_1(x)
        out = out * quant_step
        out = self.dec_2(out)
        out = F.pixel_shuffle(out, 8)
        return out


class IntraEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_1 = DepthConvBlock(g_ch_src, g_ch_enc_dec)
        self.enc_2 = nn.Sequential(
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            DepthConvBlock(g_ch_enc_dec, g_ch_enc_dec),
            nn.Conv2d(g_ch_enc_dec, g_ch_y, 3, stride=2, padding=1),
        )

    def forward(self, x, quant_step):
        out = F.pixel_unshuffle(x, 8)
        out = self.enc_1(out)
        out = out * quant_step
        return self.enc_2(out)


class IntraHyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class IntraHyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class IntraSpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class IntraYPriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class DMCI(CompressionModel):
    def __init__(self):
        super().__init__(z_channel=g_ch_z)

        self.enc = IntraEncoder()
        self.hyper_enc = IntraHyperEncoder()

        self.hyper_dec = IntraHyperDecoder()
        self.y_prior_fusion = IntraYPriorFusion()

        self.y_spatial_prior_reduction = nn.Conv2d(g_ch_y * 2, g_ch_y, 1)
        self.y_spatial_prior_adaptor_1 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_2 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior_adaptor_3 = DepthConvBlock(g_ch_y * 2, g_ch_y * 2, force_adaptor=True)
        self.y_spatial_prior = IntraSpatialPrior()

        self.dec = IntraDecoder()

        self.q_scale_enc = nn.Parameter(torch.ones((self.qp_num(), g_ch_enc_dec)))
        self.q_scale_dec = nn.Parameter(torch.ones((self.qp_num(), g_ch_enc_dec)))
        self.q_scale_y_enc = nn.Parameter(torch.ones((self.qp_num(), g_ch_y)))
        self.q_scale_y_dec = nn.Parameter(torch.ones((self.qp_num(), g_ch_y)))
        self._initialize_weights()

    def forward_one_frame(self, x, qp, recon_only=False):
        curr_q_enc = self.index_select_dim0(self.q_scale_enc, qp)
        curr_q_dec = self.index_select_dim0(self.q_scale_dec, qp)
        curr_y_q_enc = self.index_select_dim0(self.q_scale_y_enc, qp)
        curr_y_q_dec = self.index_select_dim0(self.q_scale_y_dec, qp)

        y = self.enc(x, curr_q_enc)
        z = self.hyper_enc(y)
        z_hat = QuantFunc.apply(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        _, _, yH, yW = y.shape
        params = params[:, :, :yH, :yW]
        y_res, y_q, y_hat, scales_hat = self.forward_prior_4x(
            y, curr_y_q_enc, curr_y_q_dec,
            params, self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior)

        x_hat = self.dec(y_hat, curr_q_dec)
        if recon_only:
            return x_hat

        y_for_bit = self.add_noise(y_res)
        z_for_bit = self.add_noise(z)
        bits_y = self.get_y_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, qp)

        mse = self.get_mse(x, x_hat)
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp = (bits_y + bits_z) / pixel_num

        return {
            'x_hat': x_hat,
            'mse': mse,
            'bpp': bpp,
            'bits_y': bits_y,
            'bits_z': bits_z,
        }

    def compress(self, x, qp, padding_b, padding_r):
        if self.proxy is None:
            try:
                from inference_extensions_cuda import DMCIProxy
            except Exception:
                raise NotImplementedError(
                    'cannot import cuda implementation for inference. '
                    'Please build the inference extensions first.'
                )
            state_dict = self.state_dict()
            state_dict = self.add_cdf_to_state_dict(state_dict)
            self.proxy = DMCIProxy()
            self.proxy.set_param(state_dict, self.gaussian_encoder.skip_thres)
        bit_stream, x_hat, ec_parallel = self.proxy.compress(x, qp, padding_b, padding_r)
        return {
            'bit_stream': bit_stream.tobytes(),
            'x_hat': x_hat,
            'ec_parallel': ec_parallel,
        }

    def decompress(self, bit_stream, sps, qp, ec_part):
        x_hat = self.proxy.decompress(
            np.frombuffer(bit_stream, dtype=np.uint8), qp, sps['height'], sps['width'], ec_part)
        return {'x_hat': x_hat}

    def forward(self, x, qp, lambdas=None, get_loss_info=False, recon_only=False):
        result = self.forward_one_frame(x, qp, recon_only)
        loss = loss_func(result, lambdas)
        info = None
        if get_loss_info:
            _, _, H, W = x.size()
            pixel_num = H * W
            info = self.get_loss_info(result, loss, pixel_num)
        return loss['loss'], info
