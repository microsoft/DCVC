# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch
from torch import nn

from .layers import LowerBound
from .entropy_models import BitEstimator, GaussianEncoder, EntropyCoder


class CompressionModel(nn.Module):
    def __init__(self, y_distribution, z_channel, ec_thread=False):
        super().__init__()
        self.y_distribution = y_distribution
        self.z_channel = z_channel
        self.entropy_coder = None
        self.bit_estimator_z = BitEstimator(z_channel)
        self.gaussian_encoder = GaussianEncoder(distribution=y_distribution)
        self.ec_thread = ec_thread
        self.mse = nn.MSELoss(reduction='none')

    def quant(self, x):
        return torch.round(x)

    @staticmethod
    def get_curr_q(q_scale, q_basic):
        q_basic = LowerBound.apply(q_basic, 0.5)
        return q_basic * q_scale

    @staticmethod
    def probs_to_bits(probs):
        bits = -1.0 * torch.log(probs + 1e-5) / math.log(2.0)
        bits = LowerBound.apply(bits, 0)
        return bits

    def get_y_gaussian_bits(self, y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(0.11, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def get_z_bits(self, z, bit_estimator):
        probs = bit_estimator.get_cdf(z + 0.5) - bit_estimator.get_cdf(z - 0.5)
        return CompressionModel.probs_to_bits(probs)

    def update(self, force=False):
        self.entropy_coder = EntropyCoder(self.ec_thread)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)

    @staticmethod
    def separate_prior(params):
        return params.chunk(3, 1)

    @staticmethod
    def get_mask(height, width, dtype, device):
        micro_mask = torch.tensor(((1, 0), (0, 1)), dtype=dtype, device=device)
        mask_0 = micro_mask.repeat(height // 2, width // 2)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_0 = torch.unsqueeze(mask_0, 0)
        mask_1 = torch.ones_like(mask_0) - mask_0
        return mask_0, mask_1

    def process_with_mask(self, y, scales, means, mask):
        scales_hat = scales * mask
        means_hat = means * mask

        y_res = (y - means_hat) * mask
        y_q = self.quant(y_res)
        y_hat = y_q + means_hat

        return y_res, y_q, y_hat, scales_hat

    def forward_dual_prior(self, y, means, scales, quant_step, y_spatial_prior, write=False):
        '''
        y_0 means split in channel, the first half
        y_1 means split in channel, the second half
        y_?_0, means multiply with mask_0
        y_?_1, means multiply with mask_1
        '''
        dtype = y.dtype
        device = y.device
        _, _, H, W = y.size()
        mask_0, mask_1 = self.get_mask(H, W, dtype, device)

        quant_step = torch.clamp_min(quant_step, 0.5)
        y = y / quant_step
        y_0, y_1 = y.chunk(2, 1)

        scales_0, scales_1 = scales.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)

        y_res_0_0, y_q_0_0, y_hat_0_0, scales_hat_0_0 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_0)
        y_res_1_1, y_q_1_1, y_hat_1_1, scales_hat_1_1 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_1)

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        scales_0, means_0, scales_1, means_1 = y_spatial_prior(params).chunk(4, 1)

        y_res_0_1, y_q_0_1, y_hat_0_1, scales_hat_0_1 = \
            self.process_with_mask(y_0, scales_0, means_0, mask_1)
        y_res_1_0, y_q_1_0, y_hat_1_0, scales_hat_1_0 = \
            self.process_with_mask(y_1, scales_1, means_1, mask_0)

        y_res_0 = y_res_0_0 + y_res_0_1
        y_q_0 = y_q_0_0 + y_q_0_1
        y_hat_0 = y_hat_0_0 + y_hat_0_1
        scales_hat_0 = scales_hat_0_0 + scales_hat_0_1

        y_res_1 = y_res_1_1 + y_res_1_0
        y_q_1 = y_q_1_1 + y_q_1_0
        y_hat_1 = y_hat_1_1 + y_hat_1_0
        scales_hat_1 = scales_hat_1_1 + scales_hat_1_0

        y_res = torch.cat((y_res_0, y_res_1), dim=1)
        y_q = torch.cat((y_q_0, y_q_1), dim=1)
        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        scales_hat = torch.cat((scales_hat_0, scales_hat_1), dim=1)

        y_hat = y_hat * quant_step

        if write:
            y_q_w_0 = y_q_0_0 + y_q_1_1
            y_q_w_1 = y_q_0_1 + y_q_1_0
            scales_w_0 = scales_hat_0_0 + scales_hat_1_1
            scales_w_1 = scales_hat_0_1 + scales_hat_1_0
            return y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat
        return y_res, y_q, y_hat, scales_hat

    def compress_dual_prior(self, y, means, scales, quant_step, y_spatial_prior):
        return self.forward_dual_prior(y, means, scales, quant_step, y_spatial_prior, write=True)

    def decompress_dual_prior(self, means, scales, quant_step, y_spatial_prior):
        dtype = means.dtype
        device = means.device
        _, _, H, W = means.size()
        mask_0, mask_1 = self.get_mask(H, W, dtype, device)
        quant_step = torch.clamp_min(quant_step, 0.5)

        scales_0, scales_1 = scales.chunk(2, 1)
        means_0, means_1 = means.chunk(2, 1)

        scales_r_0 = scales_0 * mask_0 + scales_1 * mask_1
        y_q_r_0 = self.gaussian_encoder.decode_stream(scales_r_0, dtype, device)
        y_hat_0_0 = (y_q_r_0 + means_0) * mask_0
        y_hat_1_1 = (y_q_r_0 + means_1) * mask_1

        params = torch.cat((y_hat_0_0, y_hat_1_1, means, scales, quant_step), dim=1)
        scales_0, means_0, scales_1, means_1 = y_spatial_prior(params).chunk(4, 1)

        scales_r_1 = scales_0 * mask_1 + scales_1 * mask_0
        y_q_r_1 = self.gaussian_encoder.decode_stream(scales_r_1, dtype, device)
        y_hat_0_1 = (y_q_r_1 + means_0) * mask_1
        y_hat_1_0 = (y_q_r_1 + means_1) * mask_0

        y_hat_0 = y_hat_0_0 + y_hat_0_1
        y_hat_1 = y_hat_1_1 + y_hat_1_0
        y_hat = torch.cat((y_hat_0, y_hat_1), dim=1)
        y_hat = y_hat * quant_step

        return y_hat
