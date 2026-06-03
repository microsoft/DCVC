// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <ATen/cuda/CUDAContext.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "common.h"
#include "def.h"
#include <vector>

template <typename vec_t>
__forceinline__ __host__ bool can_vectorize(void* pointer)
{
    uint64_t address = reinterpret_cast<uint64_t>(pointer);
    constexpr int vec4_alignment = std::alignment_of<vec_t>::value;
    return address % vec4_alignment == 0;
}

template <typename vec_t>
__forceinline__ std::tuple<dim3, dim3, at::cuda::CUDAStream, bool, bool, int, int>
get_kernel_launch_info(const torch::Tensor& x, const int cDiv = 1, const bool allow_useVec = true)
{
    const torch::IntArrayRef x_shape = x.sizes();
    const int B = x_shape[0];
    assert(B == 1);
    const int C = x_shape[1];
    const int HW = x_shape[2] * x_shape[3];
    const int N = C * HW / cDiv;
    const int BLOCK_SIZE = 128;
    const dim3 blockDim(BLOCK_SIZE);
    const bool useVec = allow_useVec && N % 4 == 0 && can_vectorize<vec_t>(x.data_ptr());
    const bool biasSafe = HW % 4 == 0;
    const int factor = useVec ? 4 : 1;
    const dim3 gridDim((N / factor + BLOCK_SIZE - 1) / BLOCK_SIZE);
    return { blockDim, gridDim, at::cuda::getCurrentCUDAStream(), useVec, biasSafe, N / factor, HW };
}

template <typename vec_t>
__forceinline__ std::tuple<dim3, dim3, at::cuda::CUDAStream, bool, int>
get_kernel_launch_info_flatten(const torch::Tensor& x)
{
    const int N = x.numel();
    const int BLOCK_SIZE = 128;
    const dim3 blockDim(BLOCK_SIZE);
    const bool useVec = N % 4 == 0 && can_vectorize<vec_t>(x.data_ptr());
    const int factor = useVec ? 4 : 1;
    const dim3 gridDim((N / factor + BLOCK_SIZE - 1) / BLOCK_SIZE);
    return { blockDim, gridDim, at::cuda::getCurrentCUDAStream(), useVec, N / factor };
}

template <typename scalar_t, typename T, bool forceZero = false>
__global__ void process_with_mask_kernel(GPUTensor1D<T> y_res, GPUTensor1D<T> y_q,
                                         GPUTensor1D<T> y_hat, GPUTensor1D<T> s_hat,
                                         const GPUTensor1D<T> y, const GPUTensor1D<T> scales,
                                         const GPUTensor1D<T> means, const GPUTensor1D<T> mask,
                                         const scalar_t force_zero_thres, const int N)
{
    const scalar_t __min_val = static_cast<scalar_t>(-128.f);
    const scalar_t __max_val = static_cast<scalar_t>(127.f);
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        T _y = y[chw];
        T _scale = scales[chw];
        T _means = means[chw];
        T _mask = mask[chw];

        T _s_hat = _scale * _mask;
        T _means_hat = _means * _mask;
        T _y_res = (_y - _means_hat) * _mask;
        T _y_q = round(_y_res);

        if constexpr (forceZero) {
            _y_q = _y_q * (_s_hat > force_zero_thres);
        }
        _y_q = max(min(_y_q, __max_val), __min_val);
        T _y_hat = _y_q + _means_hat;

        y_res[chw] = _y_res;
        y_q[chw] = _y_q;
        y_hat[chw] = _y_hat;
        s_hat[chw] = _s_hat;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void
process_with_mask_dispatcher(torch::Tensor& y_res, torch::Tensor& y_q, torch::Tensor& y_hat,
                             torch::Tensor& s_hat, const torch::Tensor& y,
                             const torch::Tensor& scales, const torch::Tensor& means,
                             const torch::Tensor& mask, const float force_zero_thres)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(y);
    const bool force_zero = force_zero_thres > 0.f;

    auto launch_kernel = [&](auto in_v) {
        using in_t = decltype(in_v);
        if (force_zero) {
            process_with_mask_kernel<scalar_t, in_t, true>
                <<<gridDim, blockDim, 0, stream>>>(y_res, y_q, y_hat, s_hat, y, scales, means, mask,
                                                   static_cast<scalar_t>(force_zero_thres), N);
        } else {
            process_with_mask_kernel<scalar_t, in_t, false>
                <<<gridDim, blockDim, 0, stream>>>(y_res, y_q, y_hat, s_hat, y, scales, means, mask,
                                                   static_cast<scalar_t>(force_zero_thres), N);
        }
    };

    if (useVec) {
        launch_kernel(vec_t{});
    } else {
        launch_kernel(scalar_t{});
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
process_with_mask_cuda(const torch::Tensor& y, const torch::Tensor& scales, const torch::Tensor& means,
                       const torch::Tensor& mask, const float force_zero_thres)
{
    auto y_res = torch::empty_like(y);
    auto y_q = torch::empty_like(y);
    auto y_hat = torch::empty_like(y);
    auto s_hat = torch::empty_like(y);

    if (y.dtype() == torch::kFloat32) {
        process_with_mask_dispatcher<float, float4>(y_res, y_q, y_hat, s_hat, y, scales, means,
                                                    mask, force_zero_thres);
    } else if (y.dtype() == torch::kFloat16) {
        process_with_mask_dispatcher<c10::Half, Half4>(y_res, y_q, y_hat, s_hat, y, scales, means,
                                                       mask, force_zero_thres);
    }

    return { y_res, y_q, y_hat, s_hat };
}

template <typename T>
__global__ void combine_for_reading_2x_kernel(GPUTensor1D<T> out, const GPUTensor1D<T> x,
                                              const GPUTensor1D<T> mask, const int N)
{
    const int chw1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int chw2 = chw1 + N;

    if (chw1 < N) {
        T _s1 = x[chw1];
        T _s2 = x[chw2];
        T _m1 = mask[chw1];
        T _m2 = mask[chw2];

        _s1 = _s1 * _m1;
        _s2 = _s2 * _m2;
        out[chw1] = _s1 + _s2;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void combine_for_reading_2x_dispatcher(torch::Tensor& out, const torch::Tensor& x,
                                                       const torch::Tensor& mask)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x, 2);
    if (useVec) {
        combine_for_reading_2x_kernel<vec_t><<<gridDim, blockDim, 0, stream>>>(out, x, mask, N);
    } else {
        combine_for_reading_2x_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(out, x, mask, N);
    }
}

void combine_for_reading_2x_cuda(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& mask)
{
    if (x.dtype() == torch::kFloat32) {
        combine_for_reading_2x_dispatcher<float, float4>(out, x, mask);
    } else if (x.dtype() == torch::kFloat16) {
        combine_for_reading_2x_dispatcher<c10::Half, Half4>(out, x, mask);
    }
}

template <typename T>
__global__ void restore_y_2x_kernel(GPUTensor1D<T> out, const GPUTensor1D<T> y,
                                    const GPUTensor1D<T> means, const GPUTensor1D<T> mask, const int N)
{
    const int chw1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int chw2 = chw1 + N;

    if (chw1 < N) {
        T _y = y[chw1];
        T _means1 = means[chw1];
        T _means2 = means[chw2];
        T _mask1 = mask[chw1];
        T _mask2 = mask[chw2];

        _means1 = (_y + _means1) * _mask1;
        _means2 = (_y + _means2) * _mask2;
        out[chw1] = _means1;
        out[chw2] = _means2;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void restore_y_2x_dispatcher(torch::Tensor& out, const torch::Tensor& y,
                                             const torch::Tensor& means, const torch::Tensor& mask)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(y);
    if (useVec) {
        restore_y_2x_kernel<vec_t><<<gridDim, blockDim, 0, stream>>>(out, y, means, mask, N);
    } else {
        restore_y_2x_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(out, y, means, mask, N);
    }
}

void restore_y_2x_cuda(torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& means,
                       const torch::Tensor& mask)
{
    if (y.dtype() == torch::kFloat32) {
        restore_y_2x_dispatcher<float, float4>(out, y, means, mask);
    } else if (y.dtype() == torch::kFloat16) {
        restore_y_2x_dispatcher<c10::Half, Half4>(out, y, means, mask);
    }
}

template <typename T>
__global__ void restore_y_4x_kernel(GPUTensor1D<T> out, const GPUTensor1D<T> y,
                                    const GPUTensor1D<T> means, const GPUTensor1D<T> mask, const int N)
{
    const int chw1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int chw2 = chw1 + N;
    const int chw3 = chw2 + N;
    const int chw4 = chw3 + N;

    if (chw1 < N) {
        T _y = y[chw1];
        T _means1 = means[chw1];
        T _means2 = means[chw2];
        T _means3 = means[chw3];
        T _means4 = means[chw4];
        T _mask1 = mask[chw1];
        T _mask2 = mask[chw2];
        T _mask3 = mask[chw3];
        T _mask4 = mask[chw4];

        _means1 = (_y + _means1) * _mask1;
        _means2 = (_y + _means2) * _mask2;
        _means3 = (_y + _means3) * _mask3;
        _means4 = (_y + _means4) * _mask4;
        out[chw1] = _means1;
        out[chw2] = _means2;
        out[chw3] = _means3;
        out[chw4] = _means4;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void restore_y_4x_dispatcher(torch::Tensor& out, const torch::Tensor& y,
                                             const torch::Tensor& means, const torch::Tensor& mask)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(y);
    if (useVec) {
        restore_y_4x_kernel<vec_t><<<gridDim, blockDim, 0, stream>>>(out, y, means, mask, N);
    } else {
        restore_y_4x_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(out, y, means, mask, N);
    }
}

void restore_y_4x_cuda(torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& means,
                       const torch::Tensor& mask)
{
    if (y.dtype() == torch::kFloat32) {
        restore_y_4x_dispatcher<float, float4>(out, y, means, mask);
    } else if (y.dtype() == torch::kFloat16) {
        restore_y_4x_dispatcher<c10::Half, Half4>(out, y, means, mask);
    }
}

template <typename T, typename scalar_t>
__forceinline__ __device__ T scale_to_index(T scale, const scalar_t scale_min,
                                            const scalar_t scale_max, const scalar_t log_scale_min,
                                            const scalar_t log_step_recip)
{
    scale = max(scale, scale_min);
    scale = min(scale, scale_max);
    scale = log(scale) - log_scale_min;
    scale = scale * log_step_recip;
    return scale;
}

template <typename scalar_t, typename in_t, typename out_t, typename cond_out_t, bool with_cond = false>
__global__ void build_index_dec_kernel(GPUTensor1D<out_t> out, GPUTensor1D<cond_out_t> cond_out,
                                       const GPUTensor1D<in_t> scales, const scalar_t scale_min,
                                       const scalar_t scale_max, const scalar_t log_scale_min,
                                       const scalar_t log_step_recip, const scalar_t skip_thres,
                                       const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        in_t _scale = scales[n];
        in_t _index = scale_to_index(_scale, scale_min, scale_max, log_scale_min, log_step_recip);
        out[n] = to_uint8(_index);
        if constexpr (with_cond) {
            cond_out_t _cond = _scale > skip_thres;
            cond_out[n] = _cond;
        }
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void
build_index_dec_dispatcher(torch::Tensor& out, torch::optional<torch::Tensor>& cond_out,
                           const torch::Tensor& scales, const scalar_t scale_min,
                           const scalar_t scale_max, const scalar_t log_scale_min,
                           const scalar_t log_step_recip, const scalar_t skip_thres)
{
    auto [blockDim, gridDim, stream, useVec, N] = get_kernel_launch_info_flatten<vec_t>(scales);
    const bool with_cond = static_cast<float>(skip_thres) > 0.f;

    auto launch_kernel = [&](auto in_v, auto out_v, auto cond_out_v) {
        using in_t = decltype(in_v);
        using out_t = decltype(out_v);
        using cond_out_t = decltype(cond_out_v);
        if (with_cond) {
            build_index_dec_kernel<scalar_t, in_t, out_t, cond_out_t, true>
                <<<gridDim, blockDim, 0, stream>>>(out, cond_out.value(), scales, scale_min, scale_max,
                                                   log_scale_min, log_step_recip, skip_thres, N);
        } else {
            build_index_dec_kernel<scalar_t, in_t, out_t, cond_out_t, false>
                <<<gridDim, blockDim, 0, stream>>>(out, nullptr, scales, scale_min, scale_max,
                                                   log_scale_min, log_step_recip, skip_thres, N);
        }
    };

    if (useVec) {
        launch_kernel(vec_t{}, uchar4{}, bool4{});
    } else {
        launch_kernel(scalar_t{}, uint8_t{}, bool{});
    }
}

void build_index_dec_cuda(torch::Tensor& out, torch::optional<torch::Tensor>& cond_out,
                          const torch::Tensor& scales, const float scale_min, const float scale_max,
                          const float log_scale_min, const float log_step_recip, const float skip_thres)
{
    if (scales.dtype() == torch::kFloat32) {
        build_index_dec_dispatcher<float, float4>(out, cond_out, scales, scale_min, scale_max,
                                                  log_scale_min, log_step_recip, skip_thres);
    } else if (scales.dtype() == torch::kFloat16) {
        build_index_dec_dispatcher<c10::Half, Half4>(
            out, cond_out, scales, static_cast<c10::Half>(scale_min),
            static_cast<c10::Half>(scale_max), static_cast<c10::Half>(log_scale_min),
            static_cast<c10::Half>(log_step_recip), static_cast<c10::Half>(skip_thres));
    }
}

template <typename scalar_t, typename in_t, typename out_t, typename cond_out_t, bool with_cond = false>
__global__ void build_index_enc_kernel(GPUTensor1D<out_t> out, GPUTensor1D<cond_out_t> cond_out,
                                       const GPUTensor1D<in_t> symbols, const GPUTensor1D<in_t> scales,
                                       const scalar_t scale_min, const scalar_t scale_max,
                                       const scalar_t log_scale_min, const scalar_t log_step_recip,
                                       const scalar_t skip_thres, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < N) {
        in_t _scale = scales[n];
        in_t _symbol = symbols[n];
        in_t _index = scale_to_index(_scale, scale_min, scale_max, log_scale_min, log_step_recip);

        out[n] = (to_int16(_symbol) << 8) + to_int16(_index);
        if constexpr (with_cond) {
            cond_out_t _cond = _scale > skip_thres;
            cond_out[n] = _cond;
        }
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void build_index_enc_dispatcher(
    torch::Tensor& out, torch::optional<torch::Tensor>& cond_out, const torch::Tensor& symbols,
    const torch::Tensor& scales, const scalar_t scale_min, const scalar_t scale_max,
    const scalar_t log_scale_min, const scalar_t log_step_recip, const scalar_t skip_thres)
{
    auto [blockDim, gridDim, stream, useVec, N] = get_kernel_launch_info_flatten<vec_t>(scales);
    const bool with_cond = static_cast<float>(skip_thres) > 0.f;

    auto launch_kernel = [&](auto in_v, auto out_v, auto cond_out_v) {
        using in_t = decltype(in_v);
        using out_t = decltype(out_v);
        using cond_out_t = decltype(cond_out_v);
        if (with_cond) {
            build_index_enc_kernel<scalar_t, in_t, out_t, cond_out_t, true>
                <<<gridDim, blockDim, 0, stream>>>(out, cond_out.value(), symbols, scales,
                                                   scale_min, scale_max, log_scale_min,
                                                   log_step_recip, skip_thres, N);
        } else {
            build_index_enc_kernel<scalar_t, in_t, out_t, cond_out_t, false>
                <<<gridDim, blockDim, 0, stream>>>(out, nullptr, symbols, scales, scale_min, scale_max,
                                                   log_scale_min, log_step_recip, skip_thres, N);
        }
    };

    if (useVec) {
        launch_kernel(vec_t{}, short4{}, bool4{});
    } else {
        launch_kernel(scalar_t{}, int16_t{}, bool{});
    }
}

void build_index_enc_cuda(torch::Tensor& out, torch::optional<torch::Tensor>& cond_out,
                          const torch::Tensor& symbols, const torch::Tensor& scales,
                          const float scale_min, const float scale_max, const float log_scale_min,
                          const float log_step_recip, const float skip_thres)
{
    if (scales.dtype() == torch::kFloat32) {
        build_index_enc_dispatcher<float, float4>(out, cond_out, symbols, scales, scale_min, scale_max,
                                                  log_scale_min, log_step_recip, skip_thres);
    } else if (scales.dtype() == torch::kFloat16) {
        build_index_enc_dispatcher<c10::Half, Half4>(
            out, cond_out, symbols, scales, static_cast<c10::Half>(scale_min),
            static_cast<c10::Half>(scale_max), static_cast<c10::Half>(log_scale_min),
            static_cast<c10::Half>(log_step_recip), static_cast<c10::Half>(skip_thres));
    }
}

template <typename vec_t, typename scalar_t, bool biasSafe = false>
__forceinline__ __device__ vec_t get_bias(const GPUTensor1D<scalar_t> bias, const int HW, const int chw)
{
    vec_t _bias;
    if constexpr (sizeof(vec_t) / sizeof(scalar_t) == 4) {
        if constexpr (biasSafe) {
            scalar_t b = bias[(chw * 4 + 0) / HW];
            _bias = make_vec4(b, b, b, b);
        } else {
            _bias = make_vec4(bias[(chw * 4 + 0) / HW], bias[(chw * 4 + 1) / HW],
                              bias[(chw * 4 + 2) / HW], bias[(chw * 4 + 3) / HW]);
        }
    } else {
        _bias = bias[(chw) / HW];
    }
    return _bias;
}

template <typename scalar_t, typename vec_t, bool biasSafe>
__global__ void bias_wsilu_kernel(GPUTensor1D<vec_t> x, const GPUTensor1D<scalar_t> bias,
                                  const int N, const int HW)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t _bias = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw);
        vec_t _x = x[chw];
        _x = _x + _bias;
        _x = wsilu(_x);
        x[chw] = _x;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void bias_wsilu_dispatcher(torch::Tensor& x, const torch::Tensor& bias)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x);
    if (useVec) {
        if (biasSafe) {
            bias_wsilu_kernel<scalar_t, vec_t, true><<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
        } else {
            bias_wsilu_kernel<scalar_t, vec_t, false><<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
        }
    } else {
        bias_wsilu_kernel<scalar_t, scalar_t, true><<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
    }
}

void bias_wsilu_cuda(torch::Tensor& x, const torch::Tensor& bias)
{
    if (x.dtype() == torch::kFloat32) {
        bias_wsilu_dispatcher<float, float4>(x, bias);
    } else if (x.dtype() == torch::kFloat16) {
        bias_wsilu_dispatcher<c10::Half, Half4>(x, bias);
    }
}

template <typename scalar_t, typename vec_t, bool biasSafe, bool with_shortcut = true, bool with_quant = false>
__global__ void bias_shortcut_kernel(GPUTensor1D<vec_t> x, const GPUTensor1D<scalar_t> bias,
                                     const GPUTensor1D<scalar_t> quant_step,
                                     const GPUTensor1D<vec_t> shortcut, const int N, const int HW)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t _x = x[chw];
        vec_t _bias = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw);
        _x = _x + _bias;
        if constexpr (with_shortcut) {
            vec_t _s = shortcut[chw];
            _x = _x + _s;
        }
        if constexpr (with_quant) {
            vec_t _q = get_bias<vec_t, scalar_t, biasSafe>(quant_step, HW, chw);
            _x = _x * _q;
        }
        x[chw] = _x;
    }
}

template <typename scalar_t, typename vec_t, bool with_shortcut = true, bool with_quant = false>
__forceinline__ void bias_shortcut_dispatcher(torch::Tensor& x, const torch::Tensor& bias,
                                              const torch::Tensor& quant_step,
                                              const torch::Tensor& shortcut)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x);
    if (useVec) {
        if (biasSafe) {
            bias_shortcut_kernel<scalar_t, vec_t, true, with_shortcut, with_quant>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, quant_step, shortcut, N, HW);
        } else {
            bias_shortcut_kernel<scalar_t, vec_t, false, with_shortcut, with_quant>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, quant_step, shortcut, N, HW);
        }
    } else {
        bias_shortcut_kernel<scalar_t, scalar_t, true, with_shortcut, with_quant>
            <<<gridDim, blockDim, 0, stream>>>(x, bias, quant_step, shortcut, N, HW);
    }
}

void bias_shortcut_cuda(torch::Tensor& x, const torch::Tensor& bias, const torch::Tensor& shortcut)
{
    if (x.dtype() == torch::kFloat32) {
        bias_shortcut_dispatcher<float, float4, true, false>(x, bias, bias, shortcut);
    } else if (x.dtype() == torch::kFloat16) {
        bias_shortcut_dispatcher<c10::Half, Half4, true, false>(x, bias, bias, shortcut);
    }
}

void bias_quant_cuda(torch::Tensor& x, const torch::Tensor& bias, const torch::Tensor& quant_step)
{
    if (x.dtype() == torch::kFloat32) {
        bias_shortcut_dispatcher<float, float4, false, true>(x, bias, quant_step, bias);
    } else if (x.dtype() == torch::kFloat16) {
        bias_shortcut_dispatcher<c10::Half, Half4, false, true>(x, bias, quant_step, bias);
    }
}

template <typename scalar_t, typename vec_t, bool biasSafe>
__global__ void bias_shortcut_no_inplace_kernel(GPUTensor1D<vec_t> out, const GPUTensor1D<vec_t> x,
                                                const GPUTensor1D<scalar_t> bias,
                                                const GPUTensor1D<vec_t> shortcut, const int N,
                                                const int HW)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t _x = x[chw];
        vec_t _bias = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw);
        _x = _x + _bias;
        vec_t _s = shortcut[chw];
        _x = _x + _s;
        out[chw] = _x;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void bias_shortcut_no_inplace_dispatcher(torch::Tensor& out, const torch::Tensor& x,
                                                         const torch::Tensor& bias,
                                                         const torch::Tensor& shortcut)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x);
    if (useVec) {
        if (biasSafe) {
            bias_shortcut_no_inplace_kernel<scalar_t, vec_t, true>
                <<<gridDim, blockDim, 0, stream>>>(out, x, bias, shortcut, N, HW);
        } else {
            bias_shortcut_no_inplace_kernel<scalar_t, vec_t, false>
                <<<gridDim, blockDim, 0, stream>>>(out, x, bias, shortcut, N, HW);
        }
    } else {
        bias_shortcut_no_inplace_kernel<scalar_t, scalar_t, true>
            <<<gridDim, blockDim, 0, stream>>>(out, x, bias, shortcut, N, HW);
    }
}
void bias_shortcut_no_inplace_cuda(torch::Tensor& out, const torch::Tensor& x,
                                   const torch::Tensor& bias, const torch::Tensor& shortcut)
{
    if (x.dtype() == torch::kFloat32) {
        bias_shortcut_no_inplace_dispatcher<float, float4>(out, x, bias, shortcut);
    } else if (x.dtype() == torch::kFloat16) {
        bias_shortcut_no_inplace_dispatcher<c10::Half, Half4>(out, x, bias, shortcut);
    }
}

template <typename scalar_t, typename vec_t, bool biasSafe>
__global__ void bias_shortcut_2_kernel(GPUTensor1D<vec_t> x, const GPUTensor1D<scalar_t> bias,
                                       GPUTensor1D<vec_t> shortcut, const int N, const int HW)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t _x = x[chw];
        vec_t _bias = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw);
        _x = _x + _bias;
        vec_t _s = shortcut[chw];
        _x = _x + _s;
        x[chw] = _x;
        shortcut[chw] = _x + _s;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void bias_shortcut_2_dispatcher(torch::Tensor& x, const torch::Tensor& bias,
                                                torch::Tensor& shortcut)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x);
    if (useVec) {
        if (biasSafe) {
            bias_shortcut_2_kernel<scalar_t, vec_t, true>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, shortcut, N, HW);
        } else {
            bias_shortcut_2_kernel<scalar_t, vec_t, false>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, shortcut, N, HW);
        }
    } else {
        bias_shortcut_2_kernel<scalar_t, scalar_t, true>
            <<<gridDim, blockDim, 0, stream>>>(x, bias, shortcut, N, HW);
    }
}

void bias_shortcut_2_cuda(torch::Tensor& x, const torch::Tensor& bias, torch::Tensor& shortcut)
{
    if (x.dtype() == torch::kFloat32) {
        bias_shortcut_2_dispatcher<float, float4>(x, bias, shortcut);
    } else if (x.dtype() == torch::kFloat16) {
        bias_shortcut_2_dispatcher<c10::Half, Half4>(x, bias, shortcut);
    }
}

void bias_shortcut_with_quant_step_cuda(torch::Tensor& x, const torch::Tensor& bias,
                                        const torch::Tensor& quant_step, const torch::Tensor& shortcut)
{
    if (x.dtype() == torch::kFloat32) {
        bias_shortcut_dispatcher<float, float4, true, true>(x, bias, quant_step, shortcut);
    } else if (x.dtype() == torch::kFloat16) {
        bias_shortcut_dispatcher<c10::Half, Half4, true, true>(x, bias, quant_step, shortcut);
    }
}

template <typename scalar_t, typename vec_t, bool biasSafe>
__global__ void bias_wsilu_chunk_add_kernel(GPUTensor1D<vec_t> x, const GPUTensor1D<scalar_t> bias,
                                            const int N, const int HW)
{
    const int chw1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int chw2 = chw1 + N;

    if (chw1 < N) {
        vec_t _x1 = x[chw1];
        vec_t _bias1 = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw1);
        _x1 = _x1 + _bias1;
        _x1 = wsilu(_x1);

        vec_t _x2 = x[chw2];
        vec_t _bias2 = get_bias<vec_t, scalar_t, biasSafe>(bias, HW, chw2);
        _x2 = _x2 + _bias2;
        _x2 = wsilu(_x2);

        x[chw1] = _x1 + _x2;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void bias_wsilu_chunk_add_dispatcher(torch::Tensor& x, const torch::Tensor& bias)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x, 2);
    if (useVec) {
        if (biasSafe) {
            bias_wsilu_chunk_add_kernel<scalar_t, vec_t, true>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
        } else {
            bias_wsilu_chunk_add_kernel<scalar_t, vec_t, false>
                <<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
        }
    } else {
        bias_wsilu_chunk_add_kernel<scalar_t, scalar_t, true>
            <<<gridDim, blockDim, 0, stream>>>(x, bias, N, HW);
    }
}

void bias_wsilu_chunk_add_cuda(torch::Tensor& x, const torch::Tensor& bias)
{
    if (x.dtype() == torch::kFloat32) {
        bias_wsilu_chunk_add_dispatcher<float, float4>(x, bias);
    } else if (x.dtype() == torch::kFloat16) {
        bias_wsilu_chunk_add_dispatcher<c10::Half, Half4>(x, bias);
    }
    const torch::IntArrayRef x_shape = x.sizes();
    x = x.narrow(1, 0, x_shape[1] / 2);
}

template <typename scalar_t>
__global__ void bias_pixel_shuffle_2_kernel(Packed4DTensorAccessor32<scalar_t> out,
                                            const Packed4DTensorAccessor32<scalar_t> x,
                                            const Packed1DTensorAccessor32<scalar_t> bias,
                                            const int N, const int W)
{
    const int c = blockIdx.y * 4;
    const int c1 = c / 4;
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = (hw / W) * 2;
    const int w = (hw % W) * 2;

    __shared__ scalar_t _bias[4];
    if (threadIdx.x < 4) {
        _bias[threadIdx.x] = bias[c + threadIdx.x];
    }
    __syncthreads();

    if (hw < N) {
        scalar_t _bias_0 = _bias[0];
        scalar_t _bias_1 = _bias[1];
        scalar_t _bias_2 = _bias[2];
        scalar_t _bias_3 = _bias[3];

        scalar_t _x0 = x[0][c + 0][0][hw];
        scalar_t _x1 = x[0][c + 1][0][hw];
        scalar_t _x2 = x[0][c + 2][0][hw];
        scalar_t _x3 = x[0][c + 3][0][hw];

        _x0 = _x0 + _bias_0;
        _x1 = _x1 + _bias_1;
        _x2 = _x2 + _bias_2;
        _x3 = _x3 + _bias_3;

        out[0][c1][h + 0][w + 0] = _x0;
        out[0][c1][h + 0][w + 1] = _x1;
        out[0][c1][h + 1][w + 0] = _x2;
        out[0][c1][h + 1][w + 1] = _x3;
    }
}

template <typename scalar_t>
__forceinline__ void bias_pixel_shuffle_2_dispatcher(torch::Tensor& out, const torch::Tensor& x,
                                                     const torch::Tensor& bias, const int C,
                                                     const int N, const int W)
{
    const int BLOCK_SIZE = 128;
    const dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, C / 4);
    const dim3 blockDim(BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();
    bias_pixel_shuffle_2_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(
        out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(), N, W);
}

void bias_pixel_shuffle_2_cuda(torch::Tensor& out, const torch::Tensor& x,
                               const torch::Tensor& bias, const int C, const int N, const int W)
{
    if (x.dtype() == torch::kFloat32) {
        bias_pixel_shuffle_2_dispatcher<float>(out, x, bias, C, N, W);
    } else if (x.dtype() == torch::kFloat16) {
        bias_pixel_shuffle_2_dispatcher<at::Half>(out, x, bias, C, N, W);
    }
}

template <typename scalar_t, bool clamp = false>
__global__ void bias_pixel_shuffle_8_kernel(Packed4DTensorAccessor32<scalar_t> out,
                                            const Packed4DTensorAccessor32<scalar_t> x,
                                            const Packed1DTensorAccessor32<scalar_t> bias,
                                            const int N, const int W)
{
    const int c = blockIdx.y * 64;
    const int c1 = c / 64;
    const int hw = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = (hw / W) * 8;
    const int w = (hw % W) * 8;

    __shared__ scalar_t _bias[64];
    if (threadIdx.x < 64) {
        _bias[threadIdx.x] = bias[c + threadIdx.x];
    }
    __syncthreads();

    if (hw < N) {
        for (int i = 0; i < 64; i++) {
            scalar_t _x = x[0][c + i][0][hw];
            _x = _x + _bias[i];
            const int out_y_offset = i >> 3;
            const int out_x_offset = i & 7;
            if constexpr (clamp) {
                _x = max(_x, static_cast<scalar_t>(0.f));
                _x = min(_x, static_cast<scalar_t>(1.f));
            }
            out[0][c1][h + out_y_offset][w + out_x_offset] = _x;
        }
    }
}

template <typename scalar_t>
__forceinline__ void bias_pixel_shuffle_8_dispatcher(torch::Tensor& out, const torch::Tensor& x,
                                                     const torch::Tensor& bias, const int C,
                                                     const int N, const int W, bool clamp)
{
    const int BLOCK_SIZE = 128;
    const dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, C / 64);
    const dim3 blockDim(BLOCK_SIZE);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (clamp) {
        bias_pixel_shuffle_8_kernel<scalar_t, true><<<gridDim, blockDim, 0, stream>>>(
            out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(), N, W);
    } else {
        bias_pixel_shuffle_8_kernel<scalar_t, false><<<gridDim, blockDim, 0, stream>>>(
            out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            bias.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(), N, W);
    }
}

void bias_pixel_shuffle_8_cuda(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& bias,
                               const int C, const int N, const int W, bool clamp)
{
    if (x.dtype() == torch::kFloat32) {
        bias_pixel_shuffle_8_dispatcher<float>(out, x, bias, C, N, W, clamp);
    } else if (x.dtype() == torch::kFloat16) {
        bias_pixel_shuffle_8_dispatcher<at::Half>(out, x, bias, C, N, W, clamp);
    }
}

template <typename scalar_t, typename vec_t1, typename vec_t2>
__global__ void round_and_to_int8_kernel(GPUTensor1D<vec_t1> z, GPUTensor1D<vec_t2> z_int8, const int N)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t1 _z = z[chw];
        _z = round(_z);
        _z = max(_z, static_cast<scalar_t>(-128.f));
        _z = min(_z, static_cast<scalar_t>(127.f));
        z[chw] = _z;
        vec_t2 _z_int8 = to_int8(_z);
        z_int8[chw] = _z_int8;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void round_and_to_int8_dispatcher(torch::Tensor& z, torch::Tensor& z_int8)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(z);
    if (useVec) {
        round_and_to_int8_kernel<scalar_t, vec_t, char4>
            <<<gridDim, blockDim, 0, stream>>>(z, z_int8, N);
    } else {
        round_and_to_int8_kernel<scalar_t, scalar_t, int8_t>
            <<<gridDim, blockDim, 0, stream>>>(z, z_int8, N);
    }
}

torch::Tensor round_and_to_int8_cuda(torch::Tensor& z)
{
    auto z_int8 = torch::empty_like(z, at::TensorOptions().dtype(torch::kInt8));
    if (z.dtype() == torch::kFloat32) {
        round_and_to_int8_dispatcher<float, float4>(z, z_int8);
    } else if (z.dtype() == torch::kFloat16) {
        round_and_to_int8_dispatcher<c10::Half, Half4>(z, z_int8);
    }
    return z_int8;
}

template <typename scalar_t, typename vec_t>
__global__ void clamp_reciprocal_with_quant_kernel(GPUTensor1D<vec_t> q_dec_clamp,
                                                   const GPUTensor1D<vec_t> q_dec, GPUTensor1D<vec_t> y,
                                                   const scalar_t min_val, const int N)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        vec_t _q_dec = q_dec[chw];
        vec_t _y = y[chw];
        _q_dec = max(_q_dec, min_val);
        q_dec_clamp[chw] = _q_dec;
        vec_t _q_enc = reciprocal(_q_dec);
        _y = _y * _q_enc;
        y[chw] = _y;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void clamp_reciprocal_with_quant_dispatcher(torch::Tensor& q_dec_clamp,
                                                            const torch::Tensor& q_dec,
                                                            torch::Tensor& y, const float min_val)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(q_dec);
    if (useVec) {
        clamp_reciprocal_with_quant_kernel<scalar_t, vec_t><<<gridDim, blockDim, 0, stream>>>(
            q_dec_clamp, q_dec, y, static_cast<scalar_t>(min_val), N);
    } else {
        clamp_reciprocal_with_quant_kernel<scalar_t, scalar_t><<<gridDim, blockDim, 0, stream>>>(
            q_dec_clamp, q_dec, y, static_cast<scalar_t>(min_val), N);
    }
}

torch::Tensor clamp_reciprocal_with_quant_cuda(const torch::Tensor& q_dec, torch::Tensor& y,
                                               const float min_val)
{
    auto q_dec_clamp = torch::empty_like(q_dec);
    if (q_dec.dtype() == torch::kFloat32) {
        clamp_reciprocal_with_quant_dispatcher<float, float4>(q_dec_clamp, q_dec, y, min_val);
    } else if (q_dec.dtype() == torch::kFloat16) {
        clamp_reciprocal_with_quant_dispatcher<c10::Half, Half4>(q_dec_clamp, q_dec, y, min_val);
    }
    return q_dec_clamp;
}

template <typename T>
__global__ void add_and_multiply_kernel(GPUTensor1D<T> x0, const GPUTensor1D<T> x1,
                                        const GPUTensor1D<T> q, const int N)
{
    const int chw = blockIdx.x * blockDim.x + threadIdx.x;

    if (chw < N) {
        T _x0 = x0[chw];
        T _x1 = x1[chw];
        T _q = q[chw];
        _x0 = _x0 + _x1;
        _x0 = _x0 * _q;
        x0[chw] = _x0;
    }
}

template <typename scalar_t, typename vec_t>
__forceinline__ void add_and_multiply_dispatcher(torch::Tensor& x0, const torch::Tensor& x1,
                                                 const torch::Tensor& q)
{
    auto [blockDim, gridDim, stream, useVec, biasSafe, N, HW] = get_kernel_launch_info<vec_t>(x0);
    if (useVec) {
        add_and_multiply_kernel<vec_t><<<gridDim, blockDim, 0, stream>>>(x0, x1, q, N);
    } else {
        add_and_multiply_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(x0, x1, q, N);
    }
}

void add_and_multiply_cuda(torch::Tensor& x0, const torch::Tensor& x1, const torch::Tensor q)
{
    if (x0.dtype() == torch::kFloat32) {
        add_and_multiply_dispatcher<float, float4>(x0, x1, q);
    } else if (x0.dtype() == torch::kFloat16) {
        add_and_multiply_dispatcher<c10::Half, Half4>(x0, x1, q);
    }
}

template <typename scalar_t>
__global__ void replicate_pad_kernel(Packed4DTensorAccessor32<scalar_t> out,
                                     const Packed4DTensorAccessor32<scalar_t> x, const int C,
                                     const int H, const int W, const int H_padded, const int W_padded)
{
    const int b = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n < H_padded * W_padded) {
        const int dst_y = n / W_padded;
        const int dst_x = n % W_padded;
        const int src_y = min(dst_y, H - 1);
        const int src_x = min(dst_x, W - 1);
        for (int i = 0; i < C; i++) {
            scalar_t _x = x[b][i][src_y][src_x];
            out[b][i][dst_y][dst_x] = _x;
        }
    }
}

template <typename scalar_t>
__forceinline__ void replicate_pad_dispatcher(torch::Tensor& out, const torch::Tensor& x,
                                              const int B, const int C, const int H, const int W,
                                              const int padB, const int padR)
{
    const int totalOutPixel = (H + padB) * (W + padR);
    const int BLOCK_SIZE = 128;
    const dim3 blockDim(BLOCK_SIZE);
    const dim3 gridDim((totalOutPixel + BLOCK_SIZE - 1) / BLOCK_SIZE, B);
    auto stream = at::cuda::getCurrentCUDAStream();

    replicate_pad_kernel<scalar_t><<<gridDim, blockDim, 0, stream>>>(
        out.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
        x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(), C, H, W, H + padB, W + padR);
}

torch::Tensor replicate_pad_cuda(const torch::Tensor& x, const int padB, const int padR)
{
    const torch::IntArrayRef x_shape = x.sizes();
    const int B = x_shape[0];
    const int C = x_shape[1];
    const int H = x_shape[2];
    const int W = x_shape[3];
    auto out = torch::empty({ B, C, H + padB, W + padR }, x.options());
    if (x.dtype() == torch::kFloat32) {
        replicate_pad_dispatcher<float>(out, x, B, C, H, W, padB, padR);
    } else if (x.dtype() == torch::kFloat16) {
        replicate_pad_dispatcher<c10::Half>(out, x, B, C, H, W, padB, padR);
    } else if (x.dtype() == torch::kInt8) {
        replicate_pad_dispatcher<int8_t>(out, x, B, C, H, W, padB, padR);
    } else if (x.dtype() == torch::kInt16) {
        replicate_pad_dispatcher<int16_t>(out, x, B, C, H, W, padB, padR);
    }
    return out;
}

template <typename T, typename T1, int BLOCK_SIZE, int THREAD_NUM_X, int THREAD_NUM_Y>
__global__ void bias_wsilu_depthwise_conv2d_kernel(Packed4DTensorAccessor32<T> out,
                                                   const Packed4DTensorAccessor32<T> x,
                                                   const Packed4DTensorAccessor32<T> weight,
                                                   const Packed1DTensorAccessor32<T> bias,
                                                   const int B, const int C, const int H, const int W)
{
    const int b = blockIdx.z / C;
    const int c = blockIdx.z % C;
    const int h = blockIdx.y * BLOCK_SIZE;  // start of the block
    const int w = blockIdx.x * BLOCK_SIZE;
    const int THREAD_NUM = THREAD_NUM_Y * THREAD_NUM_X;
    const int t_idx = threadIdx.y * THREAD_NUM_X + threadIdx.x;

    __shared__ T1 x_shared[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    const T1 __bias = static_cast<T1>(bias[c]);
    T1 __weight[3][3];
#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            __weight[i][j] = static_cast<T1>(weight[c][0][i][j]);
        }
    }

    // load boundary padded pixels
    const int read_times = (BLOCK_SIZE * 4 + THREAD_NUM - 1) / THREAD_NUM;
    const int boundary_pos = BLOCK_SIZE + 1;

    for (int i = 0; i < read_times; i++) {
        int pixel_idx = i * THREAD_NUM + t_idx;
        if (pixel_idx < BLOCK_SIZE * 2) {
            const int y_offset = pixel_idx / 2 + 1;
            const int x_offset = (pixel_idx & 1) * boundary_pos;
            const int curr_y = h + y_offset - 1;
            const int curr_x = w + x_offset - 1;
            if (curr_y < 0 || curr_x < 0 || curr_y >= H || curr_x >= W) {
                x_shared[y_offset][x_offset] = static_cast<T1>(0.f);
            } else {
                T1 x_tmp = static_cast<T1>(x[b][c][curr_y][curr_x]);
                x_shared[y_offset][x_offset] = wsilu(x_tmp + __bias);
            }
        } else if (pixel_idx < BLOCK_SIZE * 4) {
            pixel_idx -= BLOCK_SIZE * 2;
            const int y_offset = (pixel_idx & 1) * boundary_pos;
            const int x_offset = pixel_idx / 2 + 1;
            const int curr_y = h + y_offset - 1;
            const int curr_x = w + x_offset - 1;
            if (curr_y < 0 || curr_x < 0 || curr_y >= H || curr_x >= W) {
                x_shared[y_offset][x_offset] = static_cast<T1>(0.f);
            } else {
                T1 x_tmp = static_cast<T1>(x[b][c][curr_y][curr_x]);
                x_shared[y_offset][x_offset] = wsilu(x_tmp + __bias);
            }
        }
    }

    // load corner 4 pixels
    if (t_idx < 4) {
        const int y_offset = (t_idx / 2) * boundary_pos;
        const int x_offset = (t_idx & 1) * boundary_pos;
        const int curr_y = h + y_offset - 1;
        const int curr_x = w + x_offset - 1;
        if (curr_y < 0 || curr_x < 0 || curr_y >= H || curr_x >= W) {
            x_shared[y_offset][x_offset] = static_cast<T1>(0.f);
        } else {
            T1 x_tmp = static_cast<T1>(x[b][c][curr_y][curr_x]);
            x_shared[y_offset][x_offset] = wsilu(x_tmp + __bias);
        }
    }

    const int per_y_thread_pix_num = BLOCK_SIZE / THREAD_NUM_Y;
    const int per_x_thread_pix_num = BLOCK_SIZE / THREAD_NUM_X;

    for (int t_y = 0; t_y < per_y_thread_pix_num; t_y++) {
        for (int t_x = 0; t_x < per_x_thread_pix_num; t_x++) {
            const int h_offset = threadIdx.y * per_y_thread_pix_num + t_y + 1;
            const int w_offset = threadIdx.x * per_x_thread_pix_num + t_x + 1;
            const int curr_y = h + h_offset - 1;
            const int curr_x = w + w_offset - 1;
            // curr_x and curr_y cannot < 0
            if (curr_y >= H || curr_x >= W) {
                x_shared[h_offset][w_offset] = static_cast<T1>(0.f);
            } else {
                T1 x_tmp = static_cast<T1>(x[b][c][curr_y][curr_x]);
                x_shared[h_offset][w_offset] = wsilu(x_tmp + __bias);
            }
        }
    }
    __syncthreads();

    // calculation
    for (int t_y = 0; t_y < per_y_thread_pix_num; t_y++) {
        for (int t_x = 0; t_x < per_x_thread_pix_num; t_x++) {
            const int h_offset = threadIdx.y * per_y_thread_pix_num + t_y;
            const int w_offset = threadIdx.x * per_x_thread_pix_num + t_x;
            if (h + h_offset < H && w + w_offset < W) {
                T1 r = static_cast<T1>(0.f);
#pragma unroll
                for (int i = 0; i < 3; i++) {
#pragma unroll
                    for (int j = 0; j < 3; j++) {
                        r = multiply_add(__weight[i][j], x_shared[h_offset + i][w_offset + j], r);
                    }
                }

                out[b][c][h + h_offset][w + w_offset] = static_cast<T>(r);
            }
        }
    }
}

torch::Tensor bias_wsilu_depthwise_conv2d_cuda(const torch::Tensor& x, const torch::Tensor& weight,
                                               const torch::Tensor& bias)
{
    const torch::IntArrayRef x_shape = x.sizes();
    const int B = x_shape[0];
    const int C = x_shape[1];
    const int H = x_shape[2];
    const int W = x_shape[3];

    auto out = torch::empty_like(x);

    const int BLOCK_SIZE = 32;
    const int THREAD_NUM_X = 16;
    const int THREAD_NUM_Y = 8;
    const dim3 gridDim((W + BLOCK_SIZE - 1) / BLOCK_SIZE, (H + BLOCK_SIZE - 1) / BLOCK_SIZE, B * C);
    const dim3 blockDim(THREAD_NUM_X, THREAD_NUM_Y);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (x.dtype() == torch::kFloat32) {
        bias_wsilu_depthwise_conv2d_kernel<float, float, BLOCK_SIZE, THREAD_NUM_X, THREAD_NUM_Y>
            <<<gridDim, blockDim, 0, stream>>>(
                out.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                x.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), B, C, H, W);
    } else if (x.dtype() == torch::kFloat16) {
        bias_wsilu_depthwise_conv2d_kernel<c10::Half, float, BLOCK_SIZE, THREAD_NUM_X, THREAD_NUM_Y>
            <<<gridDim, blockDim, 0, stream>>>(
                out.packed_accessor32<c10::Half, 4, torch::RestrictPtrTraits>(),
                x.packed_accessor32<c10::Half, 4, torch::RestrictPtrTraits>(),
                weight.packed_accessor32<c10::Half, 4, torch::RestrictPtrTraits>(),
                bias.packed_accessor32<c10::Half, 1, torch::RestrictPtrTraits>(), B, C, H, W);
    }
    return out;
}
