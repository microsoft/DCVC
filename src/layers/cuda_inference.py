# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch
import torch.nn.functional as F


CUSTOMIZED_CUDA_INFERENCE = False
try:
    from inference_extensions_cuda import process_with_mask_cuda, combine_for_reading_2x_cuda, \
        restore_y_2x_cuda, restore_y_4x_cuda, build_index_dec_cuda, \
        round_and_to_int8_cuda, clamp_reciprocal_with_quant_cuda, bias_quant_cuda, \
        add_and_multiply_cuda, bias_pixel_shuffle_8_cuda, replicate_pad_cuda, \
        build_index_enc_cuda, DepthConvProxy, SubpelConv2xProxy  # noqa: F401
    CUSTOMIZED_CUDA_INFERENCE = True
except Exception:  # pylint: disable=W0718
    pass


if not CUSTOMIZED_CUDA_INFERENCE and 'SUPPRESS_CUSTOM_KERNEL_WARNING' not in os.environ:
    print("cannot import cuda implementation for inference, fallback to pytorch.")


def round_and_to_int8(z):
    if CUSTOMIZED_CUDA_INFERENCE and z.is_cuda:
        z_int8 = round_and_to_int8_cuda(z)
        return z, z_int8

    z_hat = torch.clamp(torch.round(z), -128., 127.)
    z_hat_write = z_hat.to(dtype=torch.int8)
    return z_hat, z_hat_write


def clamp_reciprocal_with_quant(q_dec, y, min_val):
    if CUSTOMIZED_CUDA_INFERENCE and q_dec.is_cuda:
        # q_dec is not inplace modified at decoder side
        q_dec = clamp_reciprocal_with_quant_cuda(q_dec, y, min_val)
        return q_dec, y

    q_dec = torch.clamp_min(q_dec, min_val)
    q_enc = torch.reciprocal(q_dec)
    y = y * q_enc
    return q_dec, y


def add_and_multiply(y_hat_0, y_hat_1, q_dec):
    if CUSTOMIZED_CUDA_INFERENCE and y_hat_0.is_cuda:
        add_and_multiply_cuda(y_hat_0, y_hat_1, q_dec)
        return y_hat_0

    y_hat = y_hat_0 + y_hat_1
    y_hat = y_hat * q_dec
    return y_hat


def process_with_mask(y, scales, means, mask, force_zero_thres):
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda:
        thres = force_zero_thres if force_zero_thres is not None else -1.
        return process_with_mask_cuda(y, scales, means, mask, thres)

    scales_hat = scales * mask
    means_hat = means * mask

    y_res = (y - means_hat) * mask
    y_q = torch.round(y_res)
    if force_zero_thres is not None:
        cond = scales_hat > force_zero_thres
        y_q = y_q * cond
    y_q = torch.clamp(y_q, -128., 127.)
    y_hat = y_q + means_hat

    return y_res, y_q, y_hat, scales_hat


def combine_for_reading_2x(x, mask, inplace=False):
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda and x.is_contiguous():
        B, C, H, W = x.shape
        if inplace:
            out = x[:, :C // 2, :, :]
        else:
            out = torch.empty((B, C // 2, H, W), dtype=x.dtype, layout=x.layout, device=x.device)
        combine_for_reading_2x_cuda(out, x, mask)
        return out

    x = x * mask
    x0, x1 = x.chunk(2, 1)
    return x0 + x1


def restore_y_2x(y, means, mask):
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_2x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y), dim=1) + means) * mask


def restore_y_2x_with_cat_after(y, means, mask, to_cat):
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        B, C1, H, W = means.shape
        C2 = to_cat.shape[1]
        out = torch.empty((B, C1 + C2, H, W), dtype=means.dtype, layout=means.layout,
                          device=means.device)
        restore_y_2x_cuda(out[:, :C1, :, :], y, means, mask)
        out[:, C1:, :, :] = to_cat
        return out[:, :C1, :, :], out

    out = (torch.cat((y, y), dim=1) + means) * mask
    return out, torch.cat((out, to_cat), dim=1)


def restore_y_4x(y, means, mask):
    if CUSTOMIZED_CUDA_INFERENCE and y.is_cuda and y.is_contiguous():
        out = torch.empty_like(means)
        restore_y_4x_cuda(out, y, means, mask)
        return out

    return (torch.cat((y, y, y, y), dim=1) + means) * mask


def build_index_dec(scales, scale_min, scale_max, log_scale_min, log_step_recip, skip_thres=None):
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.uint8)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_dec_cuda(out, skip_cond, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)
        return out, skip_cond

    skip_cond = None
    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
    return indexes, skip_cond


def build_index_enc(symbols, scales, scale_min, scale_max, log_scale_min,
                    log_step_recip, skip_thres=None):
    if CUSTOMIZED_CUDA_INFERENCE and scales.is_cuda:
        out = torch.empty_like(scales, dtype=torch.int16)
        skip_cond = None
        if skip_thres is not None:
            skip_cond = torch.empty_like(scales, dtype=torch.bool)
        else:
            skip_thres = -1.

        build_index_enc_cuda(out, skip_cond, symbols, scales, scale_min, scale_max, log_scale_min,
                             log_step_recip, skip_thres)

        out = out[skip_cond]
        return out

    scales = scales.clamp_(scale_min, scale_max)
    indexes = (torch.log(scales) - log_scale_min) * log_step_recip
    indexes = indexes.to(dtype=torch.uint8)
    symbols = symbols.to(dtype=torch.int16)
    out = (symbols << 8) + indexes
    out = out.to(dtype=torch.int16)
    if skip_thres is not None:
        skip_cond = scales > skip_thres
        out = out[skip_cond]
    return out


def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        return replicate_pad_cuda(x, pad_b, pad_r)
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def bias_pixel_shuffle_8(x, bias):
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        B, C, H, W = x.shape
        assert B == 1
        out = torch.empty((B, 3, H * 8, W * 8), dtype=x.dtype, device=x.device, layout=x.layout)
        bias_pixel_shuffle_8_cuda(out, x, bias, C, H * W, W, True)
        return out

    out = x + bias[None, :, None, None]
    out = F.pixel_shuffle(out, 8)
    out = torch.clamp(out, 0., 1.)
    return out


def bias_quant(x, bias, quant_step):
    if CUSTOMIZED_CUDA_INFERENCE and x.is_cuda:
        bias_quant_cuda(x, bias, quant_step)
        return x

    out = x + bias[None, :, None, None]
    out = out * quant_step
    return out
