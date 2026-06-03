# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
import torch

from torch import nn

from ..layers.layers import bit_estimator_z_fwd, bit_estimator_z_prob, LowerBound


MAX_ENTROPY_CODING_VALUE = 8


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self._quantized_cdf = None
        self._cdf_length = None

    def get_cdf_info(self):
        return self._quantized_cdf, self._cdf_length

    def set_cdf_info(self, quantized_cdf, cdf_length):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()


class EntropyCoder():
    def __init__(self):
        super().__init__()

        from MLCodec_extensions_cpp import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf):
        from MLCodec_extensions_cpp import pmf_to_quantized_cdf as _pmf_to_cdf
        cdf = _pmf_to_cdf(pmf.tolist())
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def reorder_prob(prob):
        # prob is with length l, with the last one tail_prob
        # the first (l-1) probs are sysmmetric about 0, in increasing order
        # we want to make the first a few probs in 0, 1, -1, 2, -2, ... order
        length = prob.size(0)
        prob1 = prob.clone()
        center = (length - 1) // 2
        prob1[0] = prob[center]
        for i in range(1, center + 1):
            prob1[2 * i - 1] = prob[center + i]
            prob1[2 * i - 0] = prob[center - i]
        return prob1

    def set_cdf(self, cdf, cdf_length, index):
        self.encoder.set_cdf(cdf, cdf_length, index)
        self.decoder.set_cdf(cdf, cdf_length, index)

    def set_entropy_coder_parallel(self, entropy_coder_parallel):
        self.encoder.set_entropy_coder_parallel(entropy_coder_parallel)
        self.decoder.set_entropy_coder_parallel(entropy_coder_parallel)

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            prob1 = EntropyCoder.reorder_prob(prob)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob1)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, qp_num, channel):
        super().__init__()
        self.layer_num = 4
        self.h = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num]), 0, 0.01))
        self.b = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num]), 0, 0.01))
        self.a = nn.Parameter(
            torch.nn.init.normal_(torch.empty([qp_num, channel, self.layer_num - 1]), 0, 0.01))
        self.qp_num = qp_num
        self.channel = channel

    def get_hba(self, index, dtype=None):
        if isinstance(index, int):
            h = self.h[index:index+1, :, :]
            b = self.b[index:index+1, :, :]
            a = self.a[index:index+1, :, :]
        else:
            h = torch.index_select(self.h, 0, index)
            b = torch.index_select(self.b, 0, index)
            a = torch.index_select(self.a, 0, index)
        if dtype is not None:
            return h.to(dtype), b.to(dtype), a.to(dtype)
        return h, b, a

    def forward(self, x, index):
        h, b, a = self.get_hba(index)
        return bit_estimator_z_prob(x, h, b, a)

    def get_prob(self, x, index):
        h, b, a = self.get_hba(index, dtype=x.dtype)
        prob = bit_estimator_z_fwd(x, h, b, a)
        return prob

    @torch.inference_mode()
    def update(self, entropy_coder):
        self.entropy_coder = entropy_coder

        device = next(self.parameters()).device
        zeros = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
        index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

        sym_range = zeros + MAX_ENTROPY_CODING_VALUE
        for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
            neg_probs = self.forward(zeros - i, index)
            pos_probs = self.forward(zeros + i, index)
            sym_range = torch.where(torch.logical_and(neg_probs < 0.001, pos_probs > 0.999),
                                    i, sym_range)

        sym_range = sym_range.int()
        pmf_length = sym_range * 2 + 1

        max_length = MAX_ENTROPY_CODING_VALUE * 2 + 1
        samples = torch.arange(max_length, device=device)
        samples = samples[None, None, None, :] - sym_range

        lower = self.forward(samples - 0.5, index)
        upper = self.forward(samples + 0.5, index)
        pmf = upper - lower

        pmf = pmf[:, :, 0, :]
        upper = self.forward(sym_range.float(), index)
        tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

        pmf = pmf.reshape([-1, max_length])
        tail_mass = tail_mass.reshape([-1, 1])
        pmf_length = pmf_length.reshape([-1])
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        cdf_length = pmf_length + 2
        self.set_cdf_info(quantized_cdf, cdf_length)
        self.entropy_coder.set_cdf(*self.get_cdf_info(), 0)


class GaussianEncoder(AEHelper, nn.Module):
    def __init__(self):
        super().__init__()
        self.scale_min = 0.11
        self.scale_max = 16.0
        self.scale_level = 128
        self.scale_table = self.get_scale_table()

        self.skip_thres = 0

    @staticmethod
    def get_prob_train(values, scales):
        def _cdf2(inputs):
            const = float(-(2 ** -0.5))
            return torch.erfc(const * inputs)

        dtype = values.dtype
        values = values.float()
        scales = scales.float()
        scales = LowerBound.apply(scales, 0.11)
        values = torch.abs(values)
        upper = _cdf2((0.5 - values) / scales)
        lower = _cdf2((-0.5 - values) / scales)
        prob = upper - lower
        prob = torch.clamp_min(0.5 * prob, 1e-9)
        return prob.to(dtype)

    def get_scale_table(self):
        return torch.exp(torch.linspace(math.log(self.scale_min),
                                        math.log(self.scale_max),
                                        self.scale_level))

    def update(self, entropy_coder, skip_thres):
        self.entropy_coder = entropy_coder
        self.skip_thres = skip_thres

        zeros = torch.zeros_like(self.scale_table)
        sym_range = zeros + MAX_ENTROPY_CODING_VALUE
        scales = self.scale_table
        cdf_distribution = torch.distributions.normal.Normal(0., scales)
        for i in range(MAX_ENTROPY_CODING_VALUE, 1, -1):
            samples = zeros + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            sym_range = torch.where(probs > 0.999, i, sym_range)

        sym_range = sym_range.int()
        pmf_length = 2 * sym_range + 1
        max_length = 2 * MAX_ENTROPY_CODING_VALUE + 1

        samples = torch.arange(max_length, device=sym_range.device) - sym_range[:, None]
        samples = samples.float()

        scales = self.scale_table[:, None]
        cdf_distribution = torch.distributions.normal.Normal(0., scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.set_cdf_info(quantized_cdf, pmf_length+2)
        self.entropy_coder.set_cdf(*self.get_cdf_info(), 1)
