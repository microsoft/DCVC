// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../common_cu.h"
#include "../def_elementwise.h"

__global__ void pad_and_unshuffle_8_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x,
                                           const int padB, const int padR, const CudaCHW chw)
{
    // C, H, and W are the shape of output tensor
    const auto [c, h, w] = global_idx_chw();
    const int originalH = chw.H * 8 - padB;
    const int originalW = chw.W * 8 - padR;
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    const int src_c0 = c0 / (8 * 8);
    const int src_c1 = c1 / (8 * 8);
    int src_h0 = h * 8 + (c0 % (8 * 8)) / 8;
    int src_h1 = h * 8 + (c1 % (8 * 8)) / 8;
    int src_w0 = w * 8 + (c0 % (8 * 8)) % 8;
    int src_w1 = w * 8 + (c1 % (8 * 8)) % 8;
    src_h0 = min(src_h0, originalH - 1);
    src_h1 = min(src_h1, originalH - 1);
    src_w0 = min(src_w0, originalW - 1);
    src_w1 = min(src_w1, originalW - 1);
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        store2(&out[0][c0][h][w], x[0][src_c0][src_h0][src_w0], x[0][src_c1][src_h1][src_w1]);
    }
}

at::Tensor pad_and_unshuffle_8_cuda(const at::Tensor& x, const int padB, const int padR,
                                    at::Tensor& out_buf)
{
    const at::IntArrayRef x_shape = x.sizes();
    const int C = x_shape[1] * 8 * 8;
    const int H = (x_shape[2] + padB) / 8;
    const int W = (x_shape[3] + padR) / 8;
    cudaLaunchConfig_t config;
    CudaCHW chw;
    if (C == 3 * 8 * 8) {
        std::tie(config, chw) = get_kernel_config_4D<2, 16, 4, 4>(C, H, W);
    } else {
        std::tie(config, chw) = get_kernel_config_4D<2, 256, 1, 1>(C, H, W);
    }

    cudaLaunchKernelEx(&config, &pad_and_unshuffle_8_kernel, out_buf, x, padB, padR, chw);

    return out_buf;
}

__global__ void replicate_pad_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x,
                                     const int padB, const int padR, const CudaCHW chw)
{
    // C, H, and W are the shape of output tensor
    const auto [c, h, w] = global_idx_chw();
    const int originalH = chw.H - padB;
    const int originalW = chw.W - padR;
    const int src_h = min(h, originalH - 1);
    const int src_w = min(w, originalW - 1);
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        out[0][c][h][w] = x[0][c][src_h][src_w];
    }
}

at::Tensor replicate_pad_cuda(const at::Tensor& x, const int padB, const int padR, at::Tensor& out_buf)
{
    const at::IntArrayRef x_shape = x.sizes();
    const int C = x_shape[1];
    const int H = x_shape[2] + padB;
    const int W = x_shape[3] + padR;
    if (C <= 4) {
        auto [config, chw] = get_kernel_config_4D<1, 4, 16, 16>(C, H, W);

        cudaLaunchKernelEx(&config, &replicate_pad_kernel, out_buf, x, padB, padR, chw);
    } else {
        auto [config, chw] = get_kernel_config_4D<1>(C, H, W);

        cudaLaunchKernelEx(&config, &replicate_pad_kernel, out_buf, x, padB, padR, chw);
    }
    return out_buf;
}

__global__ void slice_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x, const CudaCHW chw)
{
    // C, H, and W are the shape of output tensor
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _x0;
        at::Half _x1;
        load2(&x[0][c0][h][w], _x0, _x1);
        store2(&out[0][c0][h][w], _x0, _x1);
    }
}

at::Tensor slice_cuda(const at::Tensor& x, const int H, const int W, at::Tensor& out_buf)
{
    const int C = x.size(1);
    auto [config, chw] = get_kernel_config_4D<2>(C, H, W);

    cudaLaunchKernelEx(&config, &slice_kernel, out_buf, x, chw);
    return out_buf;
}
