// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../common_cu.h"
#include "../def_elementwise.h"

__global__ void pixel_shuffle_2_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x,
                                       const CudaCHW chw)
{
    // C, H, and W are the shape of out tensor
    const auto [c, h, w] = global_idx_chw();
    const int src_c = c * 2 * 2 + (h % 2) * 2 + (w % 2);
    const int src_h = h / 2;
    const int src_w = w / 2;
    cudaGDC();

    if (c < chw.C && w < chw.W && h < chw.H) {
        out[0][c][h][w] = x[0][src_c][src_h][src_w];
    }
}

__forceinline__ void pixel_shuffle_2_dispatcher(at::Tensor& out, const at::Tensor& x, const int C,
                                                const int H, const int W)
{
    auto [config, chw] = get_kernel_config_4D<1, 4, 8, 8>(C, H, W);
    cudaLaunchKernelEx(&config, &pixel_shuffle_2_kernel, out, x, chw);
}

at::Tensor pixel_shuffle_2_cuda(const at::Tensor& x, at::Tensor& out_buf)
{
    const at::IntArrayRef x_shape = x.sizes();
    const int B = x_shape[0];
    const int C = x_shape[1] / 4;
    const int H = x_shape[2] * 2;
    const int W = x_shape[3] * 2;
    pixel_shuffle_2_dispatcher(out_buf, x, C, H, W);
    return out_buf;
}

template <bool clamp = false>
__global__ void pixel_shuffle_8_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x,
                                       const CudaCHW chw)
{
    // C, H, and W are the shape of out tensor
    const auto [c, h, w] = global_idx_chw();
    const int src_c = c * 8 * 8 + (h % 8) * 8 + (w % 8);
    const int src_h = h / 8;
    const int src_w = w / 8;
    cudaGDC();

    if (c < chw.C && w < chw.W && h < chw.H) {
        at::Half _out = x[0][src_c][src_h][src_w];
        if constexpr (clamp) {
            _out = max(_out, static_cast<at::Half>(-0.5f));
            _out = min(_out, static_cast<at::Half>(0.5f));
        }
        out[0][c][h][w] = _out;
    }
}

__forceinline__ void pixel_shuffle_8_dispatcher(at::Tensor& out, const at::Tensor& x, const int C,
                                                const int H, const int W, bool clamp)
{
    auto [config, chw] = get_kernel_config_4D<1, 4, 8, 8>(C, H, W);

    if (clamp) {
        cudaLaunchKernelEx(&config, &pixel_shuffle_8_kernel<true>, out, x, chw);
    } else {
        cudaLaunchKernelEx(&config, &pixel_shuffle_8_kernel<false>, out, x, chw);
    }
}

template <bool clamp = false>
__global__ void pixel_shuffle_8_out_3_kernel(GPUTensor4D<at::Half> out,
                                             const GPUTensor4D<at::Half> x, const CudaCHW chw)
{
    // C, H, and W are the shape of out tensor
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c + 0;
    const int c1 = c + 1;
    const int c2 = c + 2;
    const int src_c0 = c0 * 8 * 8 + (h % 8) * 8 + (w % 8);
    const int src_c1 = c1 * 8 * 8 + (h % 8) * 8 + (w % 8);
    const int src_c2 = c2 * 8 * 8 + (h % 8) * 8 + (w % 8);
    const int src_h = h / 8;
    const int src_w = w / 8;
    cudaGDC();

    if (c < chw.C && w < chw.W && h < chw.H) {
        at::Half _out0 = x[0][src_c0][src_h][src_w];
        at::Half _out1 = x[0][src_c1][src_h][src_w];
        at::Half _out2 = x[0][src_c2][src_h][src_w];
        if constexpr (clamp) {
            _out0 = max(_out0, static_cast<at::Half>(-0.5f));
            _out0 = min(_out0, static_cast<at::Half>(0.5f));
            _out1 = max(_out1, static_cast<at::Half>(-0.5f));
            _out1 = min(_out1, static_cast<at::Half>(0.5f));
            _out2 = max(_out2, static_cast<at::Half>(-0.5f));
            _out2 = min(_out2, static_cast<at::Half>(0.5f));
        }
        out[0][c0][h][w] = _out0;
        out[0][c1][h][w] = _out1;
        out[0][c2][h][w] = _out2;
    }
}

__forceinline__ void pixel_shuffle_8_out_3_dispatcher(at::Tensor& out, const at::Tensor& x,
                                                      const int C, const int H, const int W, bool clamp)
{
    auto [config, chw] = get_kernel_config_4D<1, 1, 16, 16>(C, H, W);

    if (clamp) {
        cudaLaunchKernelEx(&config, &pixel_shuffle_8_out_3_kernel<true>, out, x, chw);
    } else {
        cudaLaunchKernelEx(&config, &pixel_shuffle_8_out_3_kernel<false>, out, x, chw);
    }
}

at::Tensor pixel_shuffle_8_cuda(const at::Tensor& x, bool clamp, at::Tensor& out_buf)
{
    const at::IntArrayRef x_shape = x.sizes();
    const int B = x_shape[0];
    const int C = x_shape[1] / 64;
    const int H = x_shape[2] * 8;
    const int W = x_shape[3] * 8;
    if (C == 3) {
        pixel_shuffle_8_out_3_dispatcher(out_buf, x, C / 3, H, W, clamp);
    } else {
        pixel_shuffle_8_dispatcher(out_buf, x, C, H, W, clamp);
    }
    return out_buf;
}

template <int downscale_factor>
__global__ void pixel_unshuffle_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> x,
                                       const CudaCHW chw)
{
    // C, H, and W are the shape of output tensor
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    const int src_c0 = c0 / (downscale_factor * downscale_factor);
    const int src_c1 = c1 / (downscale_factor * downscale_factor);
    const int src_h0 =
        h * downscale_factor + (c0 % (downscale_factor * downscale_factor)) / downscale_factor;
    const int src_h1 =
        h * downscale_factor + (c1 % (downscale_factor * downscale_factor)) / downscale_factor;
    const int src_w0 =
        w * downscale_factor + (c0 % (downscale_factor * downscale_factor)) % downscale_factor;
    const int src_w1 =
        w * downscale_factor + (c1 % (downscale_factor * downscale_factor)) % downscale_factor;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        store2(&out[0][c0][h][w], x[0][src_c0][src_h0][src_w0], x[0][src_c1][src_h1][src_w1]);
    }
}

__forceinline__ void pixel_unshuffle_dispatcher(at::Tensor& out, const at::Tensor& x,
                                                const int downscale_factor, const int C,
                                                const int H, const int W)
{
    // good shape for 8x unshuffle, but may not be optimal for others
    auto [config, chw] = get_kernel_config_4D<1, 16, 4, 4>(C, H, W);

    if (downscale_factor == 2) {
        cudaLaunchKernelEx(&config, &pixel_unshuffle_kernel<2>, out, x, chw);
    } else if (downscale_factor == 4) {
        cudaLaunchKernelEx(&config, &pixel_unshuffle_kernel<4>, out, x, chw);
    } else if (downscale_factor == 8) {
        cudaLaunchKernelEx(&config, &pixel_unshuffle_kernel<8>, out, x, chw);
    } else {
        assert(false);
    }
}

at::Tensor pixel_unshuffle_cuda(const at::Tensor& x, const int downscale_factor, at::Tensor& out_buf)
{
    const at::IntArrayRef x_shape = x.sizes();
    const int C = x_shape[1] * downscale_factor * downscale_factor;
    const int H = x_shape[2] / downscale_factor;
    const int W = x_shape[3] / downscale_factor;
    pixel_unshuffle_dispatcher(out_buf, x, downscale_factor, C / 2, H, W);
    return out_buf;
}
