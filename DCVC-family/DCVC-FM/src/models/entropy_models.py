import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class EntropyCoder():
    def __init__(self, ec_thread=False, stream_part=1):
        super().__init__()

        from .MLCodec_rans import RansEncoder, RansDecoder  # pylint: disable=E0401
        self.encoder = RansEncoder(ec_thread, stream_part)
        self.decoder = RansDecoder(stream_part)

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_cdf  # pylint: disable=E0401
        cdf = _pmf_to_cdf(pmf.tolist(), precision)
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

    def reset(self):
        self.encoder.reset()

    def add_cdf(self, cdf, cdf_length, offset):
        enc_cdf_idx = self.encoder.add_cdf(cdf, cdf_length, offset)
        dec_cdf_idx = self.decoder.add_cdf(cdf, cdf_length, offset)
        assert enc_cdf_idx == dec_cdf_idx
        return enc_cdf_idx

    def encode_with_indexes(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clamp(-30000, 30000).to(torch.int16).cpu().numpy(),
                                         indexes.to(torch.int16).cpu().numpy(),
                                         cdf_group_index)

    def encode_with_indexes_np(self, symbols, indexes, cdf_group_index):
        self.encoder.encode_with_indexes(symbols.clip(-30000, 30000).astype(np.int16).reshape(-1),
                                         indexes.astype(np.int16).reshape(-1),
                                         cdf_group_index)

    def flush(self):
        self.encoder.flush()

    def get_encoded_stream(self):
        return self.encoder.get_encoded_stream().tobytes()

    def set_stream(self, stream):
        self.decoder.set_stream((np.frombuffer(stream, dtype=np.uint8)))

    def decode_stream(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.to(torch.int16).cpu().numpy(),
                                        cdf_group_index)
        rv = torch.Tensor(rv)
        return rv

    def decode_stream_np(self, indexes, cdf_group_index):
        rv = self.decoder.decode_stream(indexes.astype(np.int16).reshape(-1),
                                        cdf_group_index)
        return rv


class Bitparm(nn.Module):
    def __init__(self, qp_num, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(
            torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(
                torch.empty([qp_num, channel, 1, 1]), 0, 0.01))
        else:
            self.a = None

    def forward(self, x, index):
        h = torch.index_select(self.h, 0, index)
        b = torch.index_select(self.b, 0, index)
        x = x * F.softplus(h) + b
        if self.final:
            return x

        a = torch.index_select(self.a, 0, index)
        return x + torch.tanh(x) * torch.tanh(a)


class AEHelper():
    def __init__(self):
        super().__init__()
        self.entropy_coder = None
        self.cdf_group_index = None
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def set_cdf_info(self, quantized_cdf, cdf_length, offset):
        self._quantized_cdf = quantized_cdf.cpu().numpy()
        self._cdf_length = cdf_length.reshape(-1).int().cpu().numpy()
        self._offset = offset.reshape(-1).int().cpu().numpy()

    def get_cdf_info(self):
        return self._quantized_cdf, \
            self._cdf_length, \
            self._offset


class BitEstimator(AEHelper, nn.Module):
    def __init__(self, qp_num, channel):
        super().__init__()
        self.f1 = Bitparm(qp_num, channel)
        self.f2 = Bitparm(qp_num, channel)
        self.f3 = Bitparm(qp_num, channel)
        self.f4 = Bitparm(qp_num, channel, True)
        self.qp_num = qp_num
        self.channel = channel

    def forward(self, x, index):
        return self.get_cdf(x, index)

    def get_logits_cdf(self, x, index):
        x = self.f1(x, index)
        x = self.f2(x, index)
        x = self.f3(x, index)
        x = self.f4(x, index)
        return x

    def get_cdf(self, x, index):
        return torch.sigmoid(self.get_logits_cdf(x, index))

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

        with torch.no_grad():
            device = next(self.parameters()).device
            medians = torch.zeros((self.qp_num, self.channel, 1, 1), device=device)
            index = torch.arange(self.qp_num, device=device, dtype=torch.int32)

            minima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) - i
                probs = self.forward(samples, index)
                minima = torch.where(probs < torch.zeros_like(medians) + 0.0001,
                                     torch.zeros_like(medians) + i, minima)

            maxima = medians + 50
            for i in range(50, 1, -1):
                samples = torch.zeros_like(medians) + i
                probs = self.forward(samples, index)
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

            samples = samples[None, None, None, :] + pmf_start

            half = float(0.5)

            lower = self.forward(samples - half, index)
            upper = self.forward(samples + half, index)
            pmf = upper - lower

            pmf = pmf[:, :, 0, :]
            tail_mass = lower[:, :, 0, :1] + (1.0 - upper[:, :, 0, -1:])

            pmf = pmf.reshape([-1, max_length])
            tail_mass = tail_mass.reshape([-1, 1])
            pmf_length = pmf_length.reshape([-1])
            offset = offset.reshape([-1])
            quantized_cdf = EntropyCoder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.set_cdf_info(quantized_cdf, cdf_length, offset)
            self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, size, qp):
        B, C, H, W = size
        indexes = torch.arange(C, dtype=torch.int).view(1, -1, 1, 1) + qp * self.channel
        return indexes.repeat(B, 1, H, W)

    def build_indexes_np(self, size, qp):
        return self.build_indexes(size, qp).cpu().numpy()

    def encode(self, x, qp):
        indexes = self.build_indexes(x.size(), qp)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, size, dtype, device, qp):
        output_size = (1, self.channel, size[0], size[1])
        indexes = self.build_indexes(output_size, qp)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1), self.cdf_group_index)
        val = val.reshape(indexes.shape)
        return val.to(dtype).to(device)


class GaussianEncoder(AEHelper):
    def __init__(self, distribution='laplace'):
        super().__init__()
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

    @staticmethod
    def get_scale_table(min_val, max_val, levels):
        return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))

    def update(self, force=False, entropy_coder=None):
        assert entropy_coder is not None
        self.entropy_coder = entropy_coder

        if not force and self._offset is not None:
            return

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

        self.set_cdf_info(quantized_cdf, pmf_length+2, -pmf_center)
        self.cdf_group_index = self.entropy_coder.add_cdf(*self.get_cdf_info())

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = (torch.log(scales) - self.log_scale_min) / self.log_scale_step
        indexes = indexes.clamp_(0, self.scale_level - 1)
        return indexes.int()

    def encode(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.encode_with_indexes(x.reshape(-1), indexes.reshape(-1),
                                                      self.cdf_group_index)

    def decode_stream(self, scales, dtype, device):
        indexes = self.build_indexes(scales)
        val = self.entropy_coder.decode_stream(indexes.reshape(-1),
                                               self.cdf_group_index)
        val = val.reshape(scales.shape)
        return val.to(device).to(dtype)
