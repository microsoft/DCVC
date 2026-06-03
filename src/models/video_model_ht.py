# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn

from .common_model import CkptModule, CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, QuantFunc, ResidualBlockUpsample, \
    ResidualBlockWithStride2, mse_8frames_sum
from ..utils.common import ModelStructure, loss_func


g_frame_delay = 8
g_ch_src_d_intra = 3 * 8 * 8
g_ch_src_d = g_ch_src_d_intra * g_frame_delay
g_ch_y = 256
g_ch_z = 128
g_ch_d = 512
g_ch_m = 512
g_ch_recon = 256


class Decoder(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.up = SubpelConv2x(g_ch_y, g_ch_d, 1)
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_d * 2, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1, force_bias=True)
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_d * 2, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )

    def internal_forward(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = feature * quant_step
        return feature


class Encoder(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_src_d + g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.conv1 = nn.Sequential(
                DepthConvBlock(g_ch_src_d + g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

    def internal_forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = feature * quant_step
        feature = self.down(feature)
        return feature


class FeatureAdaptorI(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_src_d_intra, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_src_d_intra, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
            )

    def internal_forward(self, x):
        return self.conv(x)


class FeatureAdaptorM(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m + g_ch_d, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
                DepthConvBlock(g_ch_m, g_ch_m, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m + g_ch_d, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
                DepthConvBlock(g_ch_m, g_ch_m),
            )

    def internal_forward(self, memory, feature):
        return self.conv(torch.cat((memory, feature), dim=1))


class FeatureExtractor(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
                DepthConvBlock(g_ch_d, g_ch_d, dcb2=True),
            )
        else:
            self.conv = nn.Sequential(
                DepthConvBlock(g_ch_m, g_ch_d),
                DepthConvBlock(g_ch_d, g_ch_d),
            )

    def internal_forward(self, x):
        return self.conv(x)


class HyperDecoder(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {"force_bias": True}
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_y, **kwargs),
            ResidualBlockUpsample(g_ch_y, g_ch_y, **kwargs),
            DepthConvBlock(g_ch_y, g_ch_y),
        )

    def internal_forward(self, x):
        return self.conv(x)


class HyperEncoder(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {}
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_y),
            ResidualBlockWithStride2(g_ch_y, g_ch_y, **kwargs),
            ResidualBlockWithStride2(g_ch_y, g_ch_z, **kwargs),
        )

    def internal_forward(self, x):
        return self.conv(x)


class PriorFusion(CkptModule):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def internal_forward(self, x1, x2):
        return self.conv(torch.cat((x1, x2), dim=1))


class ReconHead(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        self.is_hts = is_hts
        if is_hts:
            self.conv1 = nn.ModuleList()
            for _ in range(g_frame_delay // 2):
                self.conv1.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_d),
                    )
                )

            self.conv2 = nn.ModuleList()
            for _ in range(g_frame_delay):
                self.conv2.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        nn.Conv2d(g_ch_recon, g_ch_src_d_intra, 1),
                    )
                )
        else:
            self.conv = nn.ModuleList()
            for _ in range(g_frame_delay):
                self.conv.append(
                    nn.Sequential(
                        DepthConvBlock(g_ch_d, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        DepthConvBlock(g_ch_recon, g_ch_recon),
                        nn.Conv2d(g_ch_recon, g_ch_src_d_intra, 1),
                    )
                )

    def internal_forward(self, x, for_reset=False):
        if self.is_hts:
            if for_reset:
                out = self.conv1[-1](x)
                return self.conv2[-1](out)

            all_out = []
            for i in range(g_frame_delay):
                if i % 2 == 0:
                    common = self.conv1[i // 2](x)
                out = self.conv2[i](common)
                out = F.pixel_shuffle(out, 8)
                all_out.append(out)
            return all_out
        else:
            if for_reset:
                return self.conv[-1](x)

            all_out = []
            for i in range(g_frame_delay):
                out = self.conv[i](x)
                out = F.pixel_shuffle(out, 8)
                all_out.append(out)
            return all_out


class SpatialPrior(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        out_ch = g_ch_y * 1 if is_hts else g_ch_y * 2
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            DepthConvBlock(g_ch_y * 2, g_ch_y * 2),
            nn.Conv2d(g_ch_y * 2, out_ch, 1),
        )

    def internal_forward(self, x):
        return self.conv(x)


class SpatialPriorAdaptor(DepthConvBlock):
    def __init__(self, in_ch, out_ch, *, dcb2=False, shortcut=False, force_adaptor=True,
                 is_hts=True):
        assert force_adaptor
        super().__init__(in_ch, out_ch, dcb2=dcb2, shortcut=shortcut, force_adaptor=force_adaptor)
        self.is_hts = is_hts

    def forward(self, x1, x2=None):
        if self.is_hts:
            return super().forward(torch.cat((x1, x2), dim=1))
        else:
            return super().forward(x1)


class TemporalPriorEncoder(CkptModule):
    def __init__(self, is_hts):
        super().__init__()
        if is_hts:
            kwargs = {"shortcut": False}
        else:
            kwargs = {}
        self.conv = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2, **kwargs)

    def internal_forward(self, x, quant):
        return self.conv(x * quant)


class DMC(CompressionModel):
    def __init__(self, model_structure=ModelStructure.HTS):
        super().__init__(z_channel=g_ch_z)

        self.is_hts = model_structure == ModelStructure.HTS

        self.feature_adaptor_i = FeatureAdaptorI(self.is_hts)
        self.feature_adaptor_m = FeatureAdaptorM(self.is_hts)
        self.feature_extractor = FeatureExtractor(self.is_hts)

        self.encoder = Encoder(self.is_hts)
        self.hyper_encoder = HyperEncoder(self.is_hts)

        self.hyper_decoder = HyperDecoder(self.is_hts)
        self.temporal_prior_encoder = TemporalPriorEncoder(self.is_hts)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior_reduction = nn.Conv2d(g_ch_y * 3, g_ch_y, 1)
        self.y_spatial_prior_adaptor_1 = SpatialPriorAdaptor(
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior_adaptor_2 = SpatialPriorAdaptor(
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior_adaptor_3 = SpatialPriorAdaptor(
            g_ch_y * 2, g_ch_y * 2, is_hts=self.is_hts)
        self.y_spatial_prior = SpatialPrior(self.is_hts)
        self.decoder = Decoder(self.is_hts)

        self.recon_head = ReconHead(self.is_hts)

        self.q_encoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_decoder = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self.q_feature = nn.Parameter(torch.ones((self.qp_num(), g_ch_d)))
        self._initialize_weights()

        self.ref_feature = None
        self.memory = None
        self.ctx = None

    def apply_feature_adaptor(self):
        if self.memory is None:
            self.memory = self.feature_adaptor_i(self.ref_feature)
        else:
            self.memory = self.feature_adaptor_m(self.memory, self.ref_feature)
        self.ctx = self.feature_extractor(self.memory)

    def clear_dpb(self):
        self.ref_feature = None
        self.memory = None
        self.ctx = None

    def get_rd_info(self, res):
        dist_weights = [1.5, 0.16, 0.4]
        assert g_frame_delay == 8
        mse = mse_8frames_sum(res['mse'], dist_weights)

        rd = {
            'bits_y': res['bits_y'],
            'bits_z': res['bits_z'],
            'bpp': res['bpp'],
            'mse': mse,
        }
        return rd

    def get_recon_and_feature(self, y_hat, ctx, q_decoder):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_head(feature)
        return x_hat, feature

    def res_prior_param_decoder(self, z_hat, memory, q_feature):
        temporal_params = self.temporal_prior_encoder(memory, q_feature)
        hyper_params = self.hyper_decoder(z_hat)
        params = self.y_prior_fusion(hyper_params, temporal_params)
        return params

    def set_use_ckpt(self, use_ckpt=True):
        self.feature_adaptor_i.set_use_ckpt(use_ckpt)
        self.feature_adaptor_m.set_use_ckpt(use_ckpt)
        self.feature_extractor.set_use_ckpt(use_ckpt)
        self.temporal_prior_encoder.set_use_ckpt(use_ckpt)
        self.encoder.set_use_ckpt(use_ckpt)
        self.decoder.set_use_ckpt(use_ckpt)
        self.recon_head.set_use_ckpt(use_ckpt)
        self.hyper_encoder.set_use_ckpt(use_ckpt)
        self.hyper_decoder.set_use_ckpt(use_ckpt)
        self.y_prior_fusion.set_use_ckpt(use_ckpt)
        self.y_spatial_prior.set_use_ckpt(use_ckpt)

    def set_ref_feature(self, feature, reset_feature_memory):
        self.ref_feature = feature
        if reset_feature_memory:
            feature = self.recon_head(feature, for_reset=True)
            self.clear_dpb()
            self.ref_feature = feature

    def add_ref_feature_from_frame(self, frame, apply_feature_adaptor=True):
        if self.training:
            self.ref_feature = F.pixel_unshuffle(frame, 8)
            return
        if self.proxy is None:
            try:
                if self.is_hts:
                    from inference_extensions_cuda import DMCHTSProxy
                    proxy_cls = DMCHTSProxy
                else:
                    from inference_extensions_cuda import DMCHTLProxy
                    proxy_cls = DMCHTLProxy
            except Exception:
                raise NotImplementedError(
                    'cannot import cuda implementation for inference. '
                    'Please build the inference extensions first.'
                )
            state_dict = self.state_dict()
            state_dict = self.add_cdf_to_state_dict(state_dict)
            self.proxy = proxy_cls()
            self.proxy.set_param(state_dict, self.gaussian_encoder.skip_thres)
        return self.proxy.add_ref_feature_from_frame(frame, apply_feature_adaptor)

    def compress(self, x, qp, reset_feature_memory, padding_b, padding_r):
        bit_stream, ec_parallel = self.proxy.compress(
            x, qp, reset_feature_memory, padding_b, padding_r)
        return {
            'bit_stream': bit_stream.tobytes(),
            'ec_parallel': ec_parallel,
        }

    def decompress(self, bit_stream, sps, qp, ec_part, reset_feature_memory):
        x_hat = self.proxy.decompress((np.frombuffer(bit_stream, dtype=np.uint8)),
                                      qp, sps['height'], sps['width'], ec_part,
                                      reset_feature_memory)
        return {
            'x_hat': x_hat,
        }

    def forward_one_frame(self, x, qp, reset_feature_memory=False):
        q_encoder = self.index_select_dim0(self.q_encoder, qp)
        q_decoder = self.index_select_dim0(self.q_decoder, qp)
        q_feature = self.index_select_dim0(self.q_feature, qp)
        self.apply_feature_adaptor()

        y = self.encoder(x, self.ctx, q_encoder)

        z = self.hyper_encoder(y)
        z_hat = QuantFunc.apply(z)

        params = self.res_prior_param_decoder(z_hat, self.memory, q_feature)
        y_res, y_q, y_hat, scales_hat = self.forward_prior_4x(
            y, None, None, params, self.y_spatial_prior_reduction,
            self.y_spatial_prior_adaptor_1, self.y_spatial_prior_adaptor_2,
            self.y_spatial_prior_adaptor_3, self.y_spatial_prior,
            spatial_prior_has_scales=not self.is_hts)

        x_hat, feature = self.get_recon_and_feature(y_hat, self.ctx, q_decoder)

        self.set_ref_feature(feature, reset_feature_memory)

        y_for_bit = self.add_noise(y_res)
        z_for_bit = self.add_noise(z)
        bits_y = self.get_y_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, qp)

        x_split = x.chunk(g_frame_delay, 1)
        mse_list = []
        for x_one, x_hat_one in zip(x_split, x_hat):
            mse = self.get_mse(x_one, x_hat_one)
            mse_list.append(mse)
        bits_y = torch.sum(bits_y, dim=(1, 2, 3))
        bits_z = torch.sum(bits_z, dim=(1, 2, 3))
        _, _, H, W = x.size()
        pixel_num = H * W
        bpp = (bits_y + bits_z) / pixel_num

        return {
            'bits_y': bits_y,
            'bits_z': bits_z,
            'bpp': bpp,
            'mse': mse_list,
            'x_hat': x_hat,
        }

    def forward(self, x, qp, lambdas=None, get_loss_info=False, curr_poc=0):
        if not isinstance(x, list):
            result = self.forward_one_frame(x, qp)
            rd = self.get_rd_info(result)
            loss = loss_func(rd, lambdas)
            self.ref_feature = self.ref_feature.detach()
            self.memory = self.memory.detach()
            info = None
            if get_loss_info:
                _, _, H, W = x.size()
                pixel_num = H * W
                info = self.get_loss_info(rd, loss, pixel_num)
            return loss['loss'], info

        frame_nums = len(x)
        losses = []
        for frame_index in range(frame_nums):
            cur_frame = x[frame_index]
            result = self.forward_one_frame(cur_frame, qp)
            rd = self.get_rd_info(result)
            loss = loss_func(rd, lambdas)
            losses.append(loss['loss'])
        info = None
        if get_loss_info:
            _, _, H, W = x[0].size()
            pixel_num = H * W
            info = self.get_loss_info(rd, loss, pixel_num)
        loss = torch.mean(torch.stack(losses))
        return loss, info
