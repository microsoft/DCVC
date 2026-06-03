# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
import torch.utils.checkpoint

from torch import nn

from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder
from ..layers.layers import LowerBound, QuantFunc, mse_weighted_average, get_mse_yuv_rgb
from ..utils.common import generate_str


class CkptModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.use_ckpt = False

    def internal_forward(self, *args, **kwargs):
        raise NotImplementedError

    def set_use_ckpt(self, use_ckpt=True):
        self.use_ckpt = use_ckpt

    def forward(self, *args, **kwargs):
        if self.use_ckpt:
            return torch.utils.checkpoint.checkpoint(self.internal_forward, *args, **kwargs,
                                                     preserve_rng_state=False, use_reentrant=False)
        return self.internal_forward(*args, **kwargs)


class CompressionModel(nn.Module):
    def __init__(self, z_channel):
        super().__init__()

        self.z_channel = z_channel
        self.entropy_coder = None
        self.proxy = None
        self.bit_estimator_z = BitEstimator(self.qp_num(), z_channel)
        self.gaussian_encoder = GaussianEncoder()

        self.masks = {}

        self.prob_to_bits_factor = -1.0 / math.log(2.0)

        self.recon_range = 0.5

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, 1.)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)

    def _reset_cuda_proxies(self):
        for module in self.modules():
            if hasattr(module, 'proxy'):
                try:
                    module.proxy = None
                except Exception:
                    pass

    def add_cdf_to_state_dict(self, state_dict):
        keys = ['quantized_cdf', 'cdf_length']
        state_dict.update({'gaussian_encoder.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.gaussian_encoder.get_cdf_info())})
        state_dict.update({'bit_estimator_z.'+key: torch.from_numpy(val)
                           for key, val in zip(keys, self.bit_estimator_z.get_cdf_info())})
        return state_dict

    @staticmethod
    def add_noise(x):
        noise = torch.nn.init.uniform_(torch.empty_like(x), -0.5, 0.5)
        return x + noise.detach()

    @staticmethod
    def get_loss_info(rd, loss, pixel_num):
        info = {
            'bpp_y': generate_str(rd['bits_y'] / pixel_num),
            'bpp_z': generate_str(rd['bits_z'] / pixel_num),
            'mse': generate_str(rd['mse']),
            'losses': generate_str(loss['losses']),
        }
        return info

    def get_mse(self, x, x_hat):
        mse_yuv, mse_rgb = get_mse_yuv_rgb(x, x_hat)

        _, _, H, W = x.size()
        pixel_num = H * W
        mse = mse_weighted_average(mse_yuv, mse_rgb, pixel_num)
        return mse

    @staticmethod
    def get_one_mask(micro_mask, H, W):
        mask = torch.tensor(micro_mask, dtype=torch.bool)
        mask = mask.repeat((H + 1) // 2, (W + 1) // 2)
        mask = mask[None, None, :H, :W]
        return mask

    @staticmethod
    def get_padding_size(height, width, p=64):
        new_h = (height + p - 1) // p * p
        new_w = (width + p - 1) // p * p
        padding_right = new_w - width
        padding_bottom = new_h - height
        return padding_right, padding_bottom

    def index_select_dim0(self, x, index):
        if x is None:
            return x
        out = torch.index_select(x, 0, index)[:, :, None, None]
        return out

    def probs_to_bits(self, probs):
        dtype = probs.dtype
        probs = probs.float()
        bits = torch.log(LowerBound.apply(probs, 1e-6)) * self.prob_to_bits_factor
        bits = LowerBound.apply(bits, 0)
        return bits.to(dtype)

    @staticmethod
    def process_with_mask(y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = QuantFunc.apply(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    @staticmethod
    def qp_num():
        return 64

    def separate_prior_image(self, params):
        scales, means = params.chunk(2, 1)
        return scales, means

    def separate_prior_video(self, params):
        quant_step, scales, means = params.chunk(3, 1)
        quant_step = LowerBound.apply(quant_step, 0.5)
        q_enc = 1. / quant_step
        q_dec = quant_step
        return q_enc, q_dec, scales, means

    def set_entropy_coder_parallel(self, entropy_coder_parallel):
        self.entropy_coder.set_entropy_coder_parallel(entropy_coder_parallel)

    def update(self, skip_thres):
        self.entropy_coder = EntropyCoder()
        self.gaussian_encoder.update(self.entropy_coder, skip_thres=skip_thres)
        self.bit_estimator_z.update(self.entropy_coder)

    @torch.no_grad()
    def get_mask_2x(self, B, C, H, W, device):
        curr_mask_str = f'{B}_{C}_{H}_{W}_2x'
        if curr_mask_str not in self.masks:
            assert C % 2 == 0
            m = torch.ones((B, C // 2, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 1)), H, W)
            m1 = self.get_one_mask(((0, 1), (1, 0)), H, W)

            mask_0 = torch.cat((m * m0, m * m1), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1]
        return self.masks[curr_mask_str]

    @torch.no_grad()
    def get_mask_4x(self, B, C, H, W, device):
        curr_mask_str = f'{B}_{C}_{H}_{W}_4x'
        if curr_mask_str not in self.masks:
            assert C % 4 == 0
            m = torch.ones((B, C // 4, H, W), dtype=torch.bool)
            m0 = self.get_one_mask(((1, 0), (0, 0)), H, W)
            m1 = self.get_one_mask(((0, 1), (0, 0)), H, W)
            m2 = self.get_one_mask(((0, 0), (1, 0)), H, W)
            m3 = self.get_one_mask(((0, 0), (0, 1)), H, W)

            mask_0 = torch.cat((m * m0, m * m1, m * m2, m * m3), dim=1)
            mask_0 = mask_0.to(device=device)
            mask_1 = torch.cat((m * m3, m * m2, m * m1, m * m0), dim=1)
            mask_1 = mask_1.to(device=device)
            mask_2 = torch.cat((m * m2, m * m3, m * m0, m * m1), dim=1)
            mask_2 = mask_2.to(device=device)
            mask_3 = torch.cat((m * m1, m * m0, m * m3, m * m2), dim=1)
            mask_3 = mask_3.to(device=device)

            self.masks[curr_mask_str] = [mask_0, mask_1, mask_2, mask_3]
        return self.masks[curr_mask_str]

    def get_y_bits(self, y, sigma):
        probs = self.gaussian_encoder.get_prob_train(y, sigma)
        return self.probs_to_bits(probs)

    def get_z_bits(self, z, index):
        probs = self.bit_estimator_z.get_prob(z, index)
        return self.probs_to_bits(probs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        res = super().load_state_dict(state_dict, *args, **kwargs)
        # CUDA proxy params are cached via .set_param(self.state_dict()) on first use.
        # When weights are reloaded, proxies must be reset to avoid using stale params.
        self._reset_cuda_proxies()
        return res

    def forward_prior_2x(self, y, common_params, y_spatial_prior):
        q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
        y = y * q_enc
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1 = self.get_mask_2x(B, C, H, W, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)
        means = y_spatial_prior(y_hat_0, common_params)
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat = y_hat_0 + y_hat_1
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else y_res_0 + y_res_1
        y_q = None if y_q_0 is None else y_q_0 + y_q_1
        scales_hat = s_hat_0 + s_hat_1
        return y_res, y_q, y_hat, scales_hat

    def forward_prior_4x(self, y, q_enc, q_dec,
                         common_params, y_spatial_prior_reduction,
                         y_spatial_prior_adaptor_1, y_spatial_prior_adaptor_2,
                         y_spatial_prior_adaptor_3, y_spatial_prior,
                         spatial_prior_has_scales=False):
        if q_enc is None:
            q_enc, q_dec, scales, means = self.separate_prior_video(common_params)
            y = y * q_enc
        else:
            spatial_prior_has_scales = True
            scales, means = self.separate_prior_image(common_params)
            y = y * q_enc

        common_params = y_spatial_prior_reduction(common_params)
        device = common_params.device
        B, C, H, W = y.size()
        mask_0, mask_1, mask_2, mask_3 = self.get_mask_4x(B, C, H, W, device)

        y_res_0, y_q_0, y_hat_0, s_hat_0 = self.process_with_mask(y, scales, means, mask_0)

        y_hat_so_far = y_hat_0
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_1(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_1(y_hat_so_far, common_params))
        y_res_1, y_q_1, y_hat_1, s_hat_1 = self.process_with_mask(y, scales, means, mask_1)

        y_hat_so_far = y_hat_so_far + y_hat_1
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_2(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_2(y_hat_so_far, common_params))
        y_res_2, y_q_2, y_hat_2, s_hat_2 = self.process_with_mask(y, scales, means, mask_2)

        y_hat_so_far = y_hat_so_far + y_hat_2
        if spatial_prior_has_scales:
            params = torch.cat((y_hat_so_far, common_params), dim=1)
            scales, means = y_spatial_prior(y_spatial_prior_adaptor_3(params)).chunk(2, 1)
        else:
            means = y_spatial_prior(y_spatial_prior_adaptor_3(y_hat_so_far, common_params))
        y_res_3, y_q_3, y_hat_3, s_hat_3 = self.process_with_mask(y, scales, means, mask_3)

        y_hat = y_hat_so_far + y_hat_3
        y_hat = y_hat * q_dec

        y_res = None if y_res_0 is None else (y_res_0 + y_res_1) + (y_res_2 + y_res_3)
        y_q = None if y_q_0 is None else (y_q_0 + y_q_1) + (y_q_2 + y_q_3)
        scales_hat = (s_hat_0 + s_hat_1) + (s_hat_2 + s_hat_3)

        return y_res, y_q, y_hat, scales_hat
