# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import torch


CUSTOMIZED_CUDA = False
try:
    from .extensions.block_mc_cpp_cuda import block_mc_forward  # pylint: disable=E0401, E0611
    CUSTOMIZED_CUDA = True
except Exception:  # pylint: disable=W0718
    pass

if not CUSTOMIZED_CUDA:
    try:
        from block_mc_cpp_cuda import block_mc_forward  # pylint: disable=E0401  # noqa: F811
        CUSTOMIZED_CUDA = True
    except Exception:  # pylint: disable=W0718
        pass

if not CUSTOMIZED_CUDA and 'SUPPRESS_CUSTOM_KERNEL_WARNING' not in os.environ:
    print("cannot import motion compensation in cuda, fallback to pytorch grid_sample.")


backward_grid = [{} for _ in range(9)]    # 0~7 for GPU, -1 for CPU
FORCE_RECALCULATE_GRID = False


def set_force_recalculate_grid(force):
    global FORCE_RECALCULATE_GRID
    FORCE_RECALCULATE_GRID = force


def add_grid_cache(flow):
    device_id = -1 if flow.device == torch.device('cpu') else flow.device.index
    if str(flow.size()) not in backward_grid[device_id] or FORCE_RECALCULATE_GRID:
        B, _, H, W = flow.size()
        tensor_hor = torch.linspace(-1.0, 1.0, W, device=flow.device, dtype=torch.float32).view(
            1, 1, 1, W).expand(B, -1, H, -1)
        tensor_ver = torch.linspace(-1.0, 1.0, H, device=flow.device, dtype=torch.float32).view(
            1, 1, H, 1).expand(B, -1, -1, W)
        backward_grid[device_id][str(flow.size())] = torch.cat([tensor_hor, tensor_ver], 1)


def torch_warp(feature, flow):
    device_id = -1 if feature.device == torch.device('cpu') else feature.device.index
    add_grid_cache(flow)
    flow = torch.cat([flow[:, 0:1, :, :] / ((feature.size(3) - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((feature.size(2) - 1.0) / 2.0)], 1)

    grid = backward_grid[device_id][str(flow.size())] + flow
    return torch.nn.functional.grid_sample(input=feature,
                                           grid=grid.permute(0, 2, 3, 1),
                                           mode='bilinear',
                                           padding_mode='border',
                                           align_corners=True)


def flow_warp(im, flow):
    is_float16 = False
    if im.dtype == torch.float16:
        is_float16 = True
        im = im.to(torch.float32)
        flow = flow.to(torch.float32)
    warp = torch_warp(im, flow)
    if is_float16:
        warp = warp.to(torch.float16)
    return warp


def block_mc_func(im, flow):
    if not CUSTOMIZED_CUDA:
        return flow_warp(im, flow)
    with torch.no_grad():
        B, C, H, W = im.size()
        out = torch.empty_like(im)
        block_mc_forward(out, im, flow, B, C, H, W)
        return out
