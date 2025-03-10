# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import time
import torch
from torch import nn

from .image_model import EVC
from .layers import get_enc_dec_models
from ..utils.stream_helper import encode_i, decode_i, filesize, \
    get_rounded_q


def scalable_add(inputs):
    # inputs: S x B x C x H x W
    scalable_num = inputs.size(0)
    inputs_detach = inputs.detach()
    out = []
    for i in range(scalable_num):
        o = (inputs_detach[:i].sum(0) + inputs[i]) / (i + 1)
        out.append(o)
    out = torch.stack(out)
    return out


class ScalableEnc(EVC):
    def __init__(self, N=192, anchor_num=4, ec_thread=False, enc_num=4, forward_enc_id=None):
        super().__init__(N, anchor_num, ec_thread)
        self.enc = None
        channels = [64, 64, 128, 192]
        encs = []
        self.enc_num = enc_num
        for i in range(enc_num):
            encs.append(get_enc_dec_models(3, 3, channels)[0])
        self.encs = nn.ModuleList(encs)
        self.scalable_add = scalable_add
        channels = [192, 192, 192, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)
        self.rate = None
        self.lmbdas = [0.0022, 0.0050, 0.012, 0.027]
        self.forward_enc_id = forward_enc_id
        # print(f"multi enc: forward_enc_id={self.forward_enc_id}")

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
            elif 'enc' in skey:
                tmp = ['enc'] + skey.split('.')[2:]
                tkey = '.'.join(tmp)
                if tkey in state_dict and state_dict[tkey].shape == sd[skey].shape:
                    sd[skey] = state_dict[tkey]
                elif verbose:
                    print(f"NOT load {skey}")
            elif verbose and skey not in state_dict:
                print(f"NOT load {skey}, not find it in state_dict")
            elif verbose:
                print(f"NOT load {skey}, this {sd[skey].shape}, load {state_dict[skey].shape}")
        super().load_state_dict(sd, **kwargs)

    def multi_encode(self, x, q_scale=None):
        curr_q = self.get_curr_q(q_scale, self.q_basic)

        x_list = []
        y_list = []
        for enc_id in range(self.enc_num):
            y = self.encs[enc_id](x)
            x_list.append(x)
            y_list.append(y)
            if self.forward_enc_id is not None and self.forward_enc_id == enc_id:
                break

        ys = torch.stack(y_list)
        y_out = self.scalable_add(ys)

        if self.forward_enc_id is not None:
            y = y_out[self.forward_enc_id]
            x = x_list[self.forward_enc_id]
        else:
            S, B, C, H, W = y_out.shape
            y = y_out.reshape(S * B, C, H, W)
            curr_q = curr_q.repeat(len(y_list), 1, 1, 1)
            x = torch.cat(x_list, dim=0)

        y = y / curr_q
        return x, y, curr_q

    def hyperprior_decode(self, x, y, curr_q):
        z = self.hyper_enc(y)
        z_hat = self.quant(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        q_step, scales, means = self.separate_prior(params)
        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.dec(y_hat)

        y_for_bit = y_q
        z_for_bit = z_hat

        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        mse = self.mse(x, x_hat)

        B, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num

        bits = (bpp_y + bpp_z) * pixel_num
        bpp = bpp_y + bpp_z

        return {
            "x_hat": x_hat,
            "mse": mse,
            "bit": bits,
            "bpp": bpp,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
        }

    def forward(self, x, q_scale=None):
        x, y, curr_q = self.multi_encode(x, q_scale)
        return self.hyperprior_decode(x, y, curr_q)

    def set_rate(self, rate):
        self.rate = rate

    def encode_decode(self, x, q_scale, output_path=None, pic_width=None, pic_height=None):
        if output_path is None:
            result = {
                'bit': None,
                'x_hat': None,
            }
            chose_id = 0
            lmbda = self.lmbdas[self.rate]
            encoded = self.forward(x, q_scale)
            mse, bpp = encoded['mse'], encoded['bpp']
            cost = (lmbda * 255 * 255 * mse + bpp).flatten()
            if len(cost) == 1:
                chose_id = 0
            else:
                chose_id = cost.argmin()
            result['bit'] = encoded['bit'][chose_id].item()
            result['x_hat'] = encoded['x_hat'][chose_id].unsqueeze(0)
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

        x_list = []
        y_list = []
        for enc_id in range(self.enc_num):
            y = self.encs[enc_id](x)
            x_list.append(x)
            y_list.append(y)
            if self.forward_enc_id is not None and self.forward_enc_id == enc_id:
                break

        ys = torch.stack(y_list)
        y_out = self.scalable_add(ys)

        if self.forward_enc_id is not None:
            y = y_out[self.forward_enc_id]
            x = x_list[self.forward_enc_id]
        else:
            S, B, C, H, W = y_out.shape
            y = y_out.reshape(S * B, C, H, W)
            curr_q = curr_q.repeat(len(y_list), 1, 1, 1)
            x = torch.cat(x_list, dim=0)

        y = y / curr_q
        z = self.hyper_enc(y)
        z_hat = torch.round(z)

        params = self.hyper_dec(z_hat)
        params = self.y_prior_fusion(params)
        q_step, scales, means = self.separate_prior(params)

        y_res, y_q, y_hat, scales_hat = self.forward_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        y_hat = y_hat * curr_q
        x_hat = self.dec(y_hat)

        y_for_bit = y_q
        z_for_bit = z_hat

        bits_y = self.get_y_gaussian_bits(y_for_bit, scales_hat)
        bits_z = self.get_z_bits(z_for_bit, self.bit_estimator_z)
        mse = self.mse(x, x_hat)

        B, _, H, W = x.size()
        pixel_num = H * W
        bpp_y = torch.sum(bits_y, dim=(1, 2, 3)) / pixel_num
        bpp_z = torch.sum(bits_z, dim=(1, 2, 3)) / pixel_num
        mse = torch.sum(mse, dim=(1, 2, 3)) / pixel_num

        bpp = bpp_y + bpp_z

        chose_id = 0
        lmbda = self.lmbdas[self.rate]
        cost = (lmbda * 255 * 255 * mse + bpp).flatten()
        if len(cost) == 1:
            chose_id = 0
        else:
            chose_id = cost.argmin()

        y = y[chose_id].unsqueeze(0)
        means = means[chose_id].unsqueeze(0)
        scales = scales[chose_id].unsqueeze(0)
        q_step = q_step[chose_id].unsqueeze(0)
        z_hat = z_hat[chose_id].unsqueeze(0)
        y_q_w_0, y_q_w_1, scales_w_0, scales_w_1, y_hat = self.compress_dual_prior(
            y, means, scales, q_step, self.y_spatial_prior)

        self.entropy_coder.reset()
        self.bit_estimator_z.encode(z_hat)
        self.gaussian_encoder.encode(y_q_w_0, scales_w_0)
        self.gaussian_encoder.encode(y_q_w_1, scales_w_1)
        self.entropy_coder.flush()

        bit_stream = self.entropy_coder.get_encoded_stream()

        result = {
            "bit_stream": bit_stream,
            "x_hat": x_hat[chose_id].unsqueeze(0),
        }
        return result


class Scale_EVC_SS(ScalableEnc):
    def __init__(self, N=192, anchor_num=4, ec_thread=False, enc_num=4, forward_enc_id=None):
        super().__init__(N, anchor_num, ec_thread, enc_num, forward_enc_id)
        channels = [64, 64, 128, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)


class Scale_EVC_SL(ScalableEnc):
    def __init__(self, N=192, anchor_num=4, ec_thread=False, enc_num=4, forward_enc_id=None):
        super().__init__(N, anchor_num, ec_thread, enc_num, forward_enc_id)
        channels = [192, 192, 192, 192]
        _, self.dec = get_enc_dec_models(3, 3, channels)
