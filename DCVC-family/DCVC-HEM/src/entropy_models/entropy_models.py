import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class EntropyCoder():
    def __init__(self):
        super().__init__()

        from .MLCodec_rans import BufferedRansEncoder, RansDecoder
        self.encoder = BufferedRansEncoder()
        self.decoder = RansDecoder()

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    @staticmethod
    def pmf_to_cdf(pmf, tail_mass, pmf_length, max_length):
        entropy_coder_precision = 16
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = EntropyCoder.pmf_to_quantized_cdf(prob, entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def set_stream(self, stream):
        self.decoder.set_stream(stream)

    def encode_with_indexes(self, symbols_list, indexes_list, cdf, cdf_length, offset):
        self.encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_length, offset)
        return None

    def flush_encoder(self):
        return self.encoder.flush()

    def reset_encoder(self):
        self.encoder.reset()

    def decode_stream(self, indexes, cdf, cdf_length, offset):
        rv = self.decoder.decode_stream(indexes, cdf, cdf_length, offset)
        rv = np.array(rv)
        rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
        return rv


class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        x = x * F.softplus(self.h) + self.b
        if self.final:
            return x

        return x + torch.tanh(x) * torch.tanh(self.a)


class CdfHelper():
    def __init__(self):
        super().__init__()
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf(self, offset, quantized_cdf, cdf_length):
        self._offset = offset.reshape(-1).int().cpu().numpy()
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel

        self.entropy_coder = None
        self.cdf_helper = None

    def forward(self, x):
        return self.get_cdf(x)

    def get_logits_cdf(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def get_cdf(self, x):
        return torch.sigmoid(self.get_logits_cdf(x))

    def update(self, force=False, entropy_coder=None):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self.entropy_coder is not None and not force:  # pylint: disable=E0203
            return

        self.entropy_coder = entropy_coder
        self.cdf_helper = CdfHelper()
        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.channel), device=device)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                samples = samples[None, :, None, None]
                probs = self.forward(samples)
                probs = torch.squeeze(probs)
                maxima = torch.where(probs > torch.zeros_like(medians) + 0.9999,
                                     torch.zeros_like(medians) + i, maxima)

            minima = minima.int()
            maxima = maxima.int()

            offset = -minima

            pmf_start = medians - minima
            pmf_length = maxima + minima + 1

            max_length = pmf_length.max()
            device = pmf_start.device
            samples = torch.arange(max_length, device=device)

            samples = samples[None, :] + pmf_start[:, None, None]

            half = float(0.5)

            lower = self.forward(samples - half).squeeze(0)
            upper = self.forward(samples + half).squeeze(0)
            pmf = upper - lower

            pmf = pmf[:, 0, :]
            tail_mass = lower[:, 0, :1] + (1.0 - upper[:, 0, -1:])

            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.cdf_helper.set_cdf(offset, quantized_cdf, cdf_length)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def encode(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.encode_with_indexes(x.reshape(-1).int().cpu().numpy(),
                                                      indexes[0].reshape(-1).int().cpu().numpy(),
                                                      *self.cdf_helper.get_cdf_info())

    def decode_stream(self, size):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1).int().cpu().numpy(),
                                               *self.cdf_helper.get_cdf_info())
        val = val.reshape(indexes.shape)
        return val.float()


class GaussianEncoder():
    def __init__(self, distribution='laplace'):
        assert distribution in ['laplace', 'gaussian']
        self.distribution = distribution
        if distribution == 'laplace':
            self.cdf_distribution = torch.distributions.laplace.Laplace
            self.scale_min = 0.01
            self.scale_max = 64.0
            self.scale_level = 256
        elif distribution == 'gaussian':
            self.cdf_distribution = torch.distributions.normal.Normal
            self.scale_min = 0.11
            self.scale_max = 64.0
            self.scale_level = 256
        self.scale_table = self.get_scale_table(self.scale_min, self.scale_max, self.scale_level)

        self.log_scale_min = math.log(self.scale_min)
        self.log_scale_max = math.log(self.scale_max)
        self.log_scale_step = (self.log_scale_max - self.log_scale_min) / (self.scale_level - 1)
        self.entropy_coder = None
        self.cdf_helper = None

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def update(self, force=False, entropy_coder=None):
        if self.entropy_coder is not None and not force:
            return
        self.entropy_coder = entropy_coder
        self.cdf_helper = CdfHelper()

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = cdf_distribution.cdf(samples)
            probs = torch.squeeze(probs)
            pmf_center = torch.where(probs > torch.zeros_like(pmf_center) + 0.9999,
                                     torch.zeros_like(pmf_center) + i, pmf_center)

        pmf_center = pmf_center.int()
        pmf_length = 2 * pmf_center + 1
        max_length = torch.max(pmf_length).item()

        device = pmf_center.device
        samples = torch.arange(max_length, device=device) - pmf_center[:, None]
        samples = samples.float()

        scales = torch.zeros_like(samples) + self.scale_table[:, None]
        mu = torch.zeros_like(scales)
        cdf_distribution = self.cdf_distribution(mu, scales)

        upper = cdf_distribution.cdf(samples + 0.5)
        lower = cdf_distribution.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)

        self.cdf_helper.set_cdf(-pmf_center, quantized_cdf, pmf_length+2)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1).int().cpu().numpy(),
                                                      indexes.reshape(-1).int().cpu().numpy(),
                                                      *self.cdf_helper.get_cdf_info())

    def decode_stream(self, scales):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1).int().cpu().numpy(),
                                               *self.cdf_helper.get_cdf_info())
        val = val.reshape(scales.shape)
        return val.float()

    def set_decoder_cdf(self):
        self.entropy_coder.set_decoder_cdf(*self.cdf_helper.get_cdf_info())

    def encode_with_indexes(self, symbols_list, indexes_list):
        return self.entropy_coder.encode_with_indexes(symbols_list, indexes_list,
                                                      *self.cdf_helper.get_cdf_info())
