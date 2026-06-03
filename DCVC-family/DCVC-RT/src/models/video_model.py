# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import torch.nn.functional as F
from torch import nn

from .common_model import CompressionModel
from ..layers.layers import SubpelConv2x, DepthConvBlock, \
    ResidualBlockUpsample, ResidualBlockWithStride2
from ..layers.cuda_inference import CUSTOMIZED_CUDA_INFERENCE, round_and_to_int8, \
    bias_pixel_shuffle_8, bias_quant


qp_shift = [0, 8, 4]
extra_qp = max(qp_shift)

g_ch_src_d = 3 * 8 * 8
g_ch_recon = 320
g_ch_y = 128
g_ch_z = 128
g_ch_d = 256


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )

    def forward(self, x, quant):
        x1, ctx_t = self.forward_part1(x, quant)
        ctx = self.forward_part2(x1)
        return ctx, ctx_t

    def forward_part1(self, x, quant):
        x1 = self.conv1(x)
        ctx_t = x1 * quant
        return x1, ctx_t

    def forward_part2(self, x1):
        ctx = self.conv2(x1)
        return ctx


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(g_ch_src_d, g_ch_d, 1)
        self.conv2 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv3 = DepthConvBlock(g_ch_d, g_ch_d)
        self.down = nn.Conv2d(g_ch_d, g_ch_y, 3, stride=2, padding=1)

        self.fuse_conv1_flag = False

    def forward(self, x, ctx, quant_step):
        feature = F.pixel_unshuffle(x, 8)
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(feature, ctx, quant_step)
        return self.forward_cuda(feature, ctx, quant_step)

    def forward_torch(self, feature, ctx, quant_step):
        feature = self.conv1(feature)
        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature)
        feature = feature * quant_step
        feature = self.down(feature)
        return feature

    def forward_cuda(self, feature, ctx, quant_step):
        if not self.fuse_conv1_flag:
            fuse_weight1 = torch.matmul(
                self.conv2[0].adaptor.weight[:, :g_ch_d, 0, 0],
                self.conv1.weight[:, :, 0, 0]
            )[:, :, None, None]
            fuse_weight2 = self.conv2[0].adaptor.weight[:, g_ch_d:]
            self.conv2[0].adaptor.bias.data = self.conv2[0].adaptor.bias + \
                torch.matmul(self.conv2[0].adaptor.weight[:, :g_ch_d, 0, 0],
                             self.conv1.bias[:, None])[:, 0]
            self.conv2[0].adaptor.weight.data = torch.cat([fuse_weight1, fuse_weight2], dim=1)
            self.fuse_conv1_flag = True

        feature = self.conv2(torch.cat((feature, ctx), dim=1))
        feature = self.conv3(feature, quant_step=quant_step)
        feature = self.down(feature)
        return feature


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = SubpelConv2x(g_ch_y, g_ch_d, 3, padding=1)
        self.conv1 = nn.Sequential(
            DepthConvBlock(g_ch_d * 2, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
            DepthConvBlock(g_ch_d, g_ch_d),
        )
        self.conv2 = nn.Conv2d(g_ch_d, g_ch_d, 1)

    def forward(self, x, ctx, quant_step,):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, ctx, quant_step)

        return self.forward_cuda(x, ctx, quant_step)

    def forward_torch(self, x, ctx, quant_step):
        feature = self.up(x)
        feature = self.conv1(torch.cat((feature, ctx), dim=1))
        feature = self.conv2(feature)
        feature = feature * quant_step
        return feature

    def forward_cuda(self, x, ctx, quant_step):
        feature = self.up(x, to_cat=ctx, cat_at_front=False)
        feature = self.conv1(feature)
        feature = F.conv2d(feature, self.conv2.weight)
        feature = bias_quant(feature, self.conv2.bias, quant_step)
        return feature


class ReconGeneration(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_d,     g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
            DepthConvBlock(g_ch_recon, g_ch_recon),
        )
        self.head = nn.Conv2d(g_ch_recon, g_ch_src_d, 1)

    def forward(self, x, quant_step):
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step)

        return self.forward_cuda(x, quant_step)

    def forward_torch(self, x, quant_step):
        out = self.conv(x)
        out = out * quant_step
        out = self.head(out)
        out = F.pixel_shuffle(out, 8)
        out = torch.clamp(out, 0., 1.)
        return out

    def forward_cuda(self, x, quant_step):
        out = self.conv[0](x)
        out = self.conv[1](out)
        out = self.conv[2](out)
        out = self.conv[3](out, quant_step=quant_step)
        out = F.conv2d(out, self.head.weight)
        return bias_pixel_shuffle_8(out, self.head.bias)


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
            ResidualBlockWithStride2(g_ch_z, g_ch_z),
        )

    def forward(self, x):
        return self.conv(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            ResidualBlockUpsample(g_ch_z, g_ch_z),
            DepthConvBlock(g_ch_z, g_ch_y),
        )

    def forward(self, x):
        return self.conv(x)


class PriorFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 3, 1),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPrior(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            DepthConvBlock(g_ch_y * 4, g_ch_y * 3),
            DepthConvBlock(g_ch_y * 3, g_ch_y * 3),
            nn.Conv2d(g_ch_y * 3, g_ch_y * 2, 1),
        )

    def forward(self, x):
        return self.conv(x)


class RefFrame():
    def __init__(self):
        self.frame = None
        self.feature = None
        self.poc = None


class DMC(CompressionModel):
    def __init__(self):
        super().__init__(z_channel=g_ch_z, extra_qp=extra_qp)
        self.qp_shift = qp_shift

        self.feature_adaptor_i = DepthConvBlock(g_ch_src_d, g_ch_d)
        self.feature_adaptor_p = nn.Conv2d(g_ch_d, g_ch_d, 1)
        self.feature_extractor = FeatureExtractor()

        self.encoder = Encoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.temporal_prior_encoder = ResidualBlockWithStride2(g_ch_d, g_ch_y * 2)
        self.y_prior_fusion = PriorFusion()
        self.y_spatial_prior = SpatialPrior()
        self.decoder = Decoder()
        self.recon_generation_net = ReconGeneration()

        self.q_encoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_decoder = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_feature = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_d, 1, 1)))
        self.q_recon = nn.Parameter(torch.ones((self.get_qp_num() + extra_qp, g_ch_recon, 1, 1)))

        self.dpb = []
        self.max_dpb_size = 1
        self.curr_poc = 0

    def reset_ref_feature(self):
        if len(self.dpb) > 0:
            self.dpb[0].feature = None

    def add_ref_frame(self, feature=None, frame=None, increase_poc=True):
        ref_frame = RefFrame()
        ref_frame.poc = self.curr_poc
        ref_frame.frame = frame
        ref_frame.feature = feature
        if len(self.dpb) >= self.max_dpb_size:
            self.dpb.pop(-1)
        self.dpb.insert(0, ref_frame)
        if increase_poc:
            self.curr_poc += 1

    def clear_dpb(self):
        self.dpb.clear()

    def set_curr_poc(self, poc):
        self.curr_poc = poc

    def apply_feature_adaptor(self):
        if self.dpb[0].feature is None:
            return self.feature_adaptor_i(F.pixel_unshuffle(self.dpb[0].frame, 8))
        return self.feature_adaptor_p(self.dpb[0].feature)

    def res_prior_param_decoder(self, z_hat, ctx_t):
        hierarchical_params = self.hyper_decoder(z_hat)
        temporal_params = self.temporal_prior_encoder(ctx_t)
        _, _, H, W = temporal_params.shape
        hierarchical_params = hierarchical_params[:, :, :H, :W].contiguous()
        params = self.y_prior_fusion(
            torch.cat((hierarchical_params, temporal_params), dim=1))
        return params

    def get_recon_and_feature(self, y_hat, ctx, q_decoder, q_recon):
        feature = self.decoder(y_hat, ctx, q_decoder)
        x_hat = self.recon_generation_net(feature, q_recon)
        return x_hat, feature

    def prepare_feature_adaptor_i(self, last_qp):
        if self.dpb[0].frame is None:
            q_recon = self.q_recon[last_qp:last_qp+1, :, :, :]
            self.dpb[0].frame = self.recon_generation_net(self.dpb[0].feature, q_recon).clamp_(0, 1)
            self.reset_ref_feature()

    def compress(self, x, qp):
        # pic_width and pic_height may be different from x's size. x here is after padding
        # x_hat has the same size with x
        device = x.device
        q_encoder = self.q_encoder[qp:qp+1, :, :, :]
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]

        feature = self.apply_feature_adaptor()
        ctx, ctx_t = self.feature_extractor(feature, q_feature)
        y = self.encoder(x, ctx, q_encoder)

        hyper_inp = self.pad_for_y(y)

        z = self.hyper_encoder(hyper_inp)
        z_hat, z_hat_write = round_and_to_int8(z)
        cuda_event_z_ready = torch.cuda.Event()
        cuda_event_z_ready.record()
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        y_q_w_0, y_q_w_1, s_w_0, s_w_1, y_hat = \
            self.compress_prior_2x(y, params, self.y_spatial_prior)

        cuda_event_y_ready = torch.cuda.Event()
        cuda_event_y_ready.record()
        feature = self.decoder(y_hat, ctx, q_decoder)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            self.entropy_coder.reset()
            cuda_event_z_ready.wait()
            self.bit_estimator_z.encode_z(z_hat_write, qp)
            cuda_event_y_ready.wait()
            self.gaussian_encoder.encode_y(y_q_w_0, s_w_0)
            self.gaussian_encoder.encode_y(y_q_w_1, s_w_1)
            self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        torch.cuda.synchronize(device=device)
        self.add_ref_frame(feature, None)
        return {
            'bit_stream': bit_stream,
        }

    def decompress(self, bit_stream, sps, qp):
        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        q_decoder = self.q_decoder[qp:qp+1, :, :, :]
        q_feature = self.q_feature[qp:qp+1, :, :, :]
        q_recon = self.q_recon[qp:qp+1, :, :, :]

        self.entropy_coder.set_use_two_entropy_coders(sps['ec_part'] == 1)
        self.entropy_coder.set_stream(bit_stream)
        z_size = self.get_downsampled_shape(sps['height'], sps['width'], 64)
        self.bit_estimator_z.decode_z(z_size, qp)

        feature = self.apply_feature_adaptor()
        c1, ctx_t = self.feature_extractor.forward_part1(feature, q_feature)

        z_hat = self.bit_estimator_z.get_z(z_size, device, dtype)
        params = self.res_prior_param_decoder(z_hat, ctx_t)
        infos = self.decompress_prior_2x_part1(params)

        ctx = self.feature_extractor.forward_part2(c1)

        cuda_stream = self.get_cuda_stream(device=device, priority=-1)
        with torch.cuda.stream(cuda_stream):
            y_hat = self.decompress_prior_2x_part2(params, self.y_spatial_prior, infos)
            cuda_event = torch.cuda.Event()
            cuda_event.record()

        cuda_event.wait()
        x_hat, feature = self.get_recon_and_feature(y_hat, ctx, q_decoder, q_recon)

        self.add_ref_frame(feature, x_hat)
        return {
            'x_hat': x_hat,
        }

    def shift_qp(self, qp, fa_idx):
        return qp + self.qp_shift[fa_idx]
