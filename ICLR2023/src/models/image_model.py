# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import torch
from torch import nn

from .common_model import CompressionModel
from .layers import get_enc_dec_models
from .hyperprior import get_hyperprior, get_dualprior
from ..utils.stream_helper import encode_i, decode_i, get_downsampled_shape, filesize, \
    get_rounded_q, get_state_dict


class EVC(CompressionModel):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(y_distribution='gaussian', z_channel=N, ec_thread=ec_thread)
        channels = [192, 192, 192, 192]
        self.enc, self.dec = get_enc_dec_models(3, 3, channels)
        self.hyper_enc, self.hyper_dec, self.y_prior_fusion = get_hyperprior(N)
        self.y_spatial_prior = get_dualprior(N)

        self.q_basic = nn.Parameter(torch.ones((1, N, 1, 1)))
        self.q_scale = nn.Parameter(torch.ones((anchor_num, 1, 1, 1)))
        # the exact q_step is q_basic * q_scale
        self.N = int(N)
        self.anchor_num = int(anchor_num)

    def single_encode(self, x, q_scale=None):
        curr_q = self.get_curr_q(q_scale, self.q_basic)
        y = self.enc(x)
        y = y / curr_q
        return x, y, curr_q

    def hyperprior(self, y):
        z = self.hyper_enc(y)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        q_step, scales, means = self.separate_prior(params)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        y_for_bit = y_q
        z_for_bit = z_hat

        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        return y_hat, bits_y, bits_z

    def forward(self, x, q_scale=None):
        x, y, curr_q = self.single_encode(x, q_scale)
        y_hat, bits_y, bits_z = self.hyperprior(y)
        y_hat = y_hat * curr_q
        x_hat = self.dec(y_hat)
        return self.compute_loss(x, x_hat, bits_y, bits_z)

    @staticmethod
    def get_q_scales_from_ckpt(ckpt_path):
        ckpt = get_state_dict(ckpt_path)
        q_scales = ckpt["q_scale"]
        return q_scales.reshape(-1)

    def compute_loss(self, x, x_hat, bits_y, bits_z):
        B, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num

        bits = torch.sum(bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    def encode_decode(self, x, q_scale, output_path=None, pic_width=None, pic_height=None):
        # pic_width and pic_height may be different from x's size. X here is after padding
        # x_hat has the same size with x
        if output_path is None:
            torch.cuda.synchronize()
            start_time = time.time()
            x, y, curr_q = self.single_encode(x, q_scale)
            y_hat, bits_y, bits_z = self.hyperprior(y)
            y_hat = y_hat * curr_q
            x_hat = self.dec(y_hat)
            torch.cuda.synchronize()
            latency = time.time() - start_time
            encoded = self.compute_loss(x, x_hat, bits_y, bits_z)
            result = {
                'bit': encoded['bit'].item(),
                'x_hat': encoded['x_hat'],
                'latency': latency,
            }
            return result

        assert pic_height is not None
        assert pic_width is not None
        q_scale, q_index = get_rounded_q(q_scale)
        torch.cuda.synchronize()
        start_time = time.time()
        compressed = self.compress(x, q_scale)
        torch.cuda.synchronize()
        enc_time = time.time() - start_time

        bit_stream = compressed['bit_stream']
        encode_i(pic_height, pic_width, q_index, bit_stream, output_path)
        bit = filesize(output_path) * 8

        height, width, q_index, bit_stream = decode_i(output_path)
        decompressed = self.decompress(bit_stream, height, width, q_index / 100)

        x_hat = decompressed['x_hat']
        dec_time = decompressed['dec_time']

        result = {
            'bit': bit,
            'x_hat': x_hat,
            'enc_time': enc_time,
            'dec_time': dec_time,
            'latency': enc_time + dec_time,
        }
        return result

    def compress(self, x, q_scale):
        curr_q = self.get_curr_q(q_scale, self.q_basic)
        y = self.enc(x)
        y = y / curr_q
        z = self.hyper_enc(y)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        q_step, scales, means = self.separate_prior(params)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)
        y_hat = y_hat * curr_q

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.entropy_coder.flush()

        x_hat = self.dec(y_hat).clamp_(0, 1)

        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat,
        }
        return result

    def decompress(self, bit_stream, height, width, q_scale):
        torch.cuda.synchronize()
        start_time = time.time()
        curr_q = self.get_curr_q(q_scale, self.q_basic)

        self.entropy_coder.set_stream(bit_stream)
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bit_estimator_z.decode_stream(z_size, dtype, device)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        q_step, scales, means = self.separate_prior(params)
        y_hat = self.decompress_dual_prior(means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.dec(y_hat).clamp_(0, 1)
        torch.cuda.synchronize()
        dec_time = time.time() - start_time
        return {"x_hat": x_hat, 'dec_time': dec_time}

    def load_state_dict(self, state_dict, verbose=True, **kwargs):
        sd = self.state_dict()
        for skey in sd:
            if skey in state_dict and state_dict[skey].shape == sd[skey].shape:
                sd[skey] = state_dict[skey]
                # print(f"load {skey}")
            elif 'q_scale' in skey and skey in state_dict:
                # TODO: load q_scale according to cuda_id
                print(f"q_scale: this {sd[skey].shape}, load {state_dict[skey].shape}")
                cuda_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))
                sd[skey][0] = state_dict[skey][cuda_id % 4]
                if verbose:
                    print(f"cuda {cuda_id} load q_scale: {sd[skey]}")
            elif verbose and skey not in state_dict:
                print(f"NOT load {skey}, not find it in state_dict")
            elif verbose:
                print(f"NOT load {skey}, this {sd[skey].shape}, load {state_dict[skey].shape}")
        super().load_state_dict(sd, **kwargs)


class EVC_LL(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [192, 192, 192, 192]
        self.enc, self.dec = get_enc_dec_models(3, 3, channels)


class EVC_LM(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [128, 128, 192, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)


class EVC_LS(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [64, 64, 128, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)


class EVC_SL(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [64, 64, 128, 192]
        self.enc, _ = get_enc_dec_models(3, 3, channels)


class EVC_ML(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [128, 128, 192, 192]
        self.enc, _ = get_enc_dec_models(3, 3, channels)


class EVC_SS(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [64, 64, 128, 192]
        self.enc, self.dec = get_enc_dec_models(3, 3, channels)


class EVC_MM(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [128, 128, 192, 192]
        self.enc, self.dec = get_enc_dec_models(3, 3, channels)


class EVC_MS(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False):
        super().__init__(N, anchor_num, ec_thread)
        channels = [128, 128, 192, 192]
        self.enc, _ = get_enc_dec_models(3, 3, channels)
        channels = [64, 64, 128, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)
