# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class EntropyCoder(object):
    def __init__(self, entropy_coder_precision=16):
        super().__init__()

        from .MLCodec_rans import RansEncoder, RansDecoder
        self.encoder = RansEncoder()
        self.decoder = RansDecoder()
        self.entropy_coder_precision = int(entropy_coder_precision)
        self._offset = None
        self._quantized_cdf = None
        self._cdf_length = None

    def encode_with_indexes(self, *args, **kwargs):
        return self.encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self.decoder.decode_with_indexes(*args, **kwargs)

    def set_cdf_states(self, offset, quantized_cdf, cdf_length):
        self._offset = offset
        self._quantized_cdf = quantized_cdf
        self._cdf_length = cdf_length

    @staticmethod
    def pmf_to_quantized_cdf(pmf, precision=16):
        from .MLCodec_CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
        cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
        cdf = torch.IntTensor(cdf)
        return cdf

    def pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros((len(pmf_length), max_length + 2), dtype=torch.int32)
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = self.pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes):
        """
        """
        if len(inputs.size()) != 4:
            raise ValueError("Invalid `inputs` size. Expected a 4-D tensor.")

        if inputs.size() != indexes.size():
            raise ValueError("`inputs` and `indexes` should have the same size.")
        symbols = inputs.int()

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        assert symbols.size(0) == 1
        rv = self.encode_with_indexes(
            symbols[0].reshape(-1).int().tolist(),
            indexes[0].reshape(-1).int().tolist(),
            self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(),
            self._offset.reshape(-1).int().tolist(),
        )
        return rv

    def decompress(self, strings, indexes):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
        """

        assert indexes.size(0) == 1

        if len(indexes.size()) != 4:
            raise ValueError("Invalid `indexes` size. Expected a 4-D tensor.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        cdf = self._quantized_cdf
        outputs = cdf.new(indexes.size())

        values = self.decode_with_indexes(
            strings,
            indexes[0].reshape(-1).int().tolist(),
            self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(),
            self._offset.reshape(-1).int().tolist(),
        )
        outputs[0] = torch.Tensor(values).reshape(outputs[0].size())
        return outputs.float()

    def set_stream(self, stream):
        self.decoder.set_stream(stream)

    def decode_stream(self, indexes):
        rv = self.decoder.decode_stream(
            indexes.squeeze().int().tolist(),
            self._quantized_cdf.tolist(),
            self._cdf_length.reshape(-1).int().tolist(),
            self._offset.reshape(-1).int().tolist(),
        )
        rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
        return rv


class Bitparm(nn.Module):
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
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
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    def __init__(self, channel):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        self.channel = channel
        self.entropy_coder = None

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)

    def update(self, force=False):
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self.entropy_coder is not None and not force:  # pylint: disable=E0203
            return

        self.entropy_coder = EntropyCoder()
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

            quantized_cdf = self.entropy_coder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
            cdf_length = pmf_length + 2
            self.entropy_coder.set_cdf_states(offset, quantized_cdf, cdf_length)

    @staticmethod
    def build_indexes(size):
        N, C, H, W = size
        indexes = torch.arange(C).view(1, -1, 1, 1)
        indexes = indexes.int()
        return indexes.repeat(N, 1, H, W)

    def compress(self, x):
        indexes = self.build_indexes(x.size())
        return self.entropy_coder.compress(x, indexes)

    def decompress(self, strings, size):
        output_size = (1, self.entropy_coder._quantized_cdf.size(0), size[0], size[1])
        indexes = self.build_indexes(output_size)
        return self.entropy_coder.decompress(strings, indexes)


class GaussianEncoder(object):
    def __init__(self):
        self.scale_table = self.get_scale_table()
        self.entropy_coder = None

    @staticmethod
    def get_scale_table(min=0.01, max=16, levels=64):  # pylint: disable=W0622
        return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

    def update(self, force=False):
        if self.entropy_coder is not None and not force:
            return
        self.entropy_coder = EntropyCoder()

        pmf_center = torch.zeros_like(self.scale_table) + 50
        scales = torch.zeros_like(pmf_center) + self.scale_table
        mu = torch.zeros_like(scales)
        gaussian = torch.distributions.laplace.Laplace(mu, scales)
        for i in range(50, 1, -1):
            samples = torch.zeros_like(pmf_center) + i
            probs = gaussian.cdf(samples)
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
        gaussian = torch.distributions.laplace.Laplace(mu, scales)

        upper = gaussian.cdf(samples + 0.5)
        lower = gaussian.cdf(samples - 0.5)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self.entropy_coder.pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self.entropy_coder.set_cdf_states(-pmf_center, quantized_cdf, pmf_length+2)

    def build_indexes(self, scales):
        scales = torch.maximum(scales, torch.zeros_like(scales) + 1e-5)
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
        for s in self.scale_table[:-1]:
            indexes -= (scales <= s).int()
        return indexes

    def compress(self, x, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.compress(x, indexes)

    def decompress(self, strings, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.decompress(strings, indexes)

    def set_stream(self, stream):
        self.entropy_coder.set_stream(stream)

    def decode_stream(self, scales):
        indexes = self.build_indexes(scales)
        return self.entropy_coder.decode_stream(indexes)
