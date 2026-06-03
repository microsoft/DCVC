// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "../common_cu.h"
#include "../cuda_check.h"
#include "../def_elementwise.h"

__global__ void add_and_multiply_broadcast_kernel(GPUTensor4D<at::Half> out,
                                                  const GPUTensor4D<at::Half> x0,
                                                  const GPUTensor4D<at::Half> x1,
                                                  const GPUTensor1D<at::Half> q, const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _x0_0;
        at::Half _x0_1;
        load2(&x0[0][c0][h][w], _x0_0, _x0_1);
        at::Half _x1_0;
        at::Half _x1_1;
        load2(&x1[0][c0][h][w], _x1_0, _x1_1);
        at::Half _q_0 = q[c0];
        at::Half _q_1 = q[c1];
        store2(&out[0][c0][h][w], (_x0_0 + _x1_0) * _q_0, (_x0_1 + _x1_1) * _q_1);
    }
}

at::Tensor add_and_multiply_broadcast_cuda(const at::Tensor& x0, const at::Tensor& x1,
                                           const at::Tensor& q, at::Tensor& out_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(x0);

    cudaLaunchKernelEx(&config, &add_and_multiply_broadcast_kernel, out_buf, x0, x1, q, chw);
    return out_buf;
}

__global__ void add_and_multiply_with_clamp_min_inplace_kernel(GPUTensor4D<at::Half> x0,
                                                               const GPUTensor4D<at::Half> x1,
                                                               const GPUTensor4D<at::Half> q,
                                                               const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _x0_0;
        at::Half _x0_1;
        load2(&x0[0][c0][h][w], _x0_0, _x0_1);
        at::Half _x1_0;
        at::Half _x1_1;
        load2(&x1[0][c0][h][w], _x1_0, _x1_1);
        at::Half _q_0;
        at::Half _q_1;
        load2(&q[0][c0][h][w], _q_0, _q_1);
        _q_0 = max(_q_0, static_cast<at::Half>(0.5f));
        _q_1 = max(_q_1, static_cast<at::Half>(0.5f));
        _x0_0 = (_x0_0 + _x1_0) * _q_0;
        _x0_1 = (_x0_1 + _x1_1) * _q_1;
        store2(&x0[0][c0][h][w], _x0_0, _x0_1);
    }
}

at::Tensor add_and_multiply_with_clamp_min_inplace_cuda(at::Tensor& x0, const at::Tensor& x1,
                                                        const at::Tensor& q)
{
    auto [config, chw] = get_kernel_config_4D<2>(x0);

    cudaLaunchKernelEx(&config, &add_and_multiply_with_clamp_min_inplace_kernel, x0, x1, q, chw);
    return x0;
}

__forceinline__ __device__ at::Half scale_to_index(at::Half scale, const at::Half scale_min,
                                                   const at::Half scale_max,
                                                   const at::Half log_scale_min,
                                                   const at::Half log_step_recip)
{
    scale = max(scale, scale_min);
    scale = min(scale, scale_max);
    scale = log(scale) - log_scale_min;
    scale = scale * log_step_recip;
    return scale;
}

__global__ void build_index_dec_kernel(GPUTensor1D<uint8_t> out, GPUTensor1D<bool> cond_out,
                                       const GPUTensor4D<at::Half> scales,
                                       const at::Half skip_thres_half, const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    const int n0 = (h * chw.W * chw.C + w * chw.C + c) * 2 + 0;
    const int n1 = (h * chw.W * chw.C + w * chw.C + c) * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _scale0;
        at::Half _scale1;
        load2(&scales[0][c0][h][w], _scale0, _scale1);
        at::Half _index0 =
            scale_to_index(_scale0, SCALE_MIN, SCALE_MAX, LOG_SCALE_MIN, LOG_SCALE_STEP_RECIP);
        at::Half _index1 =
            scale_to_index(_scale1, SCALE_MIN, SCALE_MAX, LOG_SCALE_MIN, LOG_SCALE_STEP_RECIP);
        uint8_t _out0 = to_uint8(_index0);
        uint8_t _out1 = to_uint8(_index1);
        store2(&out[n0], _out0, _out1);
        bool _cond0 = _scale0 > skip_thres_half;
        bool _cond1 = _scale1 > skip_thres_half;
        store2(&cond_out[n0], _cond0, _cond1);
    }
}

std::tuple<at::Tensor, at::Tensor> build_index_dec_cuda(const at::Tensor& scales,
                                                        const float skip_thres_float,
                                                        const at::optional<at::Tensor>& fx,
                                                        const int32_t skip_thres_int,
                                                        at::Tensor& out_buf, at::Tensor& cond_out_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(scales);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &build_index_dec_kernel, out_buf, cond_out_buf, scales,
                       skip_thres_float, chw);
    return { out_buf, cond_out_buf };
}

__global__ void build_index_enc_kernel(GPUTensor1D<int16_t> out, GPUTensor1D<bool> cond_out,
                                       const GPUTensor4D<at::Half> symbols,
                                       const GPUTensor4D<at::Half> scales,
                                       const at::Half skip_thres_half, const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    const int n0 = (h * chw.W * chw.C + w * chw.C + c) * 2 + 0;
    const int n1 = (h * chw.W * chw.C + w * chw.C + c) * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _scale0;
        at::Half _scale1;
        load2(&scales[0][c0][h][w], _scale0, _scale1);
        at::Half _symbol0;
        at::Half _symbol1;
        load2(&symbols[0][c0][h][w], _symbol0, _symbol1);
        at::Half _index0 =
            scale_to_index(_scale0, SCALE_MIN, SCALE_MAX, LOG_SCALE_MIN, LOG_SCALE_STEP_RECIP);
        at::Half _index1 =
            scale_to_index(_scale1, SCALE_MIN, SCALE_MAX, LOG_SCALE_MIN, LOG_SCALE_STEP_RECIP);

        int16_t _out0 = (to_int16(_symbol0) << 8) + to_int16(_index0);
        int16_t _out1 = (to_int16(_symbol1) << 8) + to_int16(_index1);
        store2(&out[n0], _out0, _out1);
        bool _cond0 = _scale0 > skip_thres_half;
        bool _cond1 = _scale1 > skip_thres_half;
        store2(&cond_out[n0], _cond0, _cond1);
    }
}

std::tuple<at::Tensor, at::Tensor>
build_index_enc_cuda(const at::Tensor& symbols, const at::Tensor& scales,
                     const float skip_thres_float, const at::optional<at::Tensor>& fx,
                     const int32_t skip_thres_int, at::Tensor& out_buf, at::Tensor& cond_out_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(scales);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &build_index_enc_kernel, out_buf, cond_out_buf, symbols, scales,
                       skip_thres_float, chw);
    return { out_buf, cond_out_buf };
}

template <int THREAD_NUM1, int PER_THREAD_NUM>
__global__ void conditional_index_step1_kernel(GPUTensor2D<int> num, const GPUTensor1D<bool> m,
                                               const int N)
{
    __shared__ int s_sum[THREAD_NUM1];
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const int start_idx = PER_THREAD_NUM * (bidx * THREAD_NUM1 + tidx);
    const int end_idx = min(start_idx + PER_THREAD_NUM, N);
    cudaGDC();

    int s = 0;
    for (int i = start_idx; i < end_idx; i++) {
        if (m[i]) {
            s++;
        }
    }
    s_sum[tidx] = s;
    __syncthreads();

    // calculate prefix sum
    int curr_group_size_log2 = 1;
    int curr_group_size = 1 << curr_group_size_log2;
    while (curr_group_size <= THREAD_NUM1) {
        if (tidx < THREAD_NUM1 / 2) {
            const int half_group_size_log2 = curr_group_size_log2 - 1;
            const int half_group_size = 1 << half_group_size_log2;
            const int start_idx = tidx >> half_group_size_log2 << curr_group_size_log2;
            const int src_idx = start_idx + (half_group_size - 1);
            const int dst_idx =
                start_idx + half_group_size + (tidx & ((1 << half_group_size_log2) - 1));
            s_sum[dst_idx] += s_sum[src_idx];
        }

        curr_group_size_log2++;
        curr_group_size <<= 1;
        __syncthreads();
    }

    num[bidx][tidx] = s_sum[tidx];
}

template <int THREAD_NUM1, int THREAD_NUM2>
__global__ void conditional_index_step2_kernel(GPUTensor2D<int> num, GPUTensor1D<int> total_num,
                                               const int N1)
{
    __shared__ int s_sum[THREAD_NUM2];
    const int tidx = threadIdx.x;
    cudaGDC();

    if (tidx < N1) {
        s_sum[tidx] = num[tidx][THREAD_NUM1 - 1];
    } else {
        s_sum[tidx] = 0;
    }
    __syncthreads();

    // calculate prefix sum
    int curr_group_size_log2 = 1;
    int curr_group_size = 1 << curr_group_size_log2;
    while (curr_group_size <= THREAD_NUM2) {
        if (tidx < THREAD_NUM2 / 2) {
            const int half_group_size_log2 = curr_group_size_log2 - 1;
            const int half_group_size = 1 << half_group_size_log2;
            const int start_idx = tidx >> half_group_size_log2 << curr_group_size_log2;
            const int src_idx = start_idx + (half_group_size - 1);
            const int dst_idx =
                start_idx + half_group_size + (tidx & ((1 << half_group_size_log2) - 1));
            s_sum[dst_idx] += s_sum[src_idx];
        }

        curr_group_size_log2++;
        curr_group_size <<= 1;
        __syncthreads();
    }

    if (tidx < N1) {
        num[tidx][THREAD_NUM1 - 1] = s_sum[tidx];
    }

    if (tidx == 0) {
        total_num[0] = s_sum[THREAD_NUM2 - 1];
    }
}

template <typename T, int THREAD_NUM1, int PER_THREAD_NUM>
__global__ void conditional_index_step3_kernel(GPUTensor1D<T> out, const GPUTensor2D<int> num,
                                               const GPUTensor1D<T> x, const GPUTensor1D<bool> m,
                                               const int N)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const int start_idx = PER_THREAD_NUM * (bidx * THREAD_NUM1 + tidx);
    const int end_idx = min(start_idx + PER_THREAD_NUM, N);
    cudaGDC();

    // set according to prefix sum
    int curr_dst_idx = tidx > 0 ? num[bidx][tidx - 1] : 0;
    curr_dst_idx += (bidx > 0 ? num[bidx - 1][THREAD_NUM1 - 1] : 0);
    for (int i = start_idx; i < end_idx; i++) {
        if (m[i]) {
            out[curr_dst_idx] = x[i];
            curr_dst_idx++;
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
conditional_index_part1_cuda(const at::Tensor& x, const at::Tensor& m, at::Tensor& out_buf,
                             at::Tensor& total_num_buf, at::Tensor& num_buf)
{
    const int N = x.numel();
    const int block_num = (N + COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM - 1)
                          / (COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM);
    assert(block_num <= COND_KERNEL_THREAD_NUM2);

    static_assert(COND_KERNEL_THREAD_NUM1 == 1024 || COND_KERNEL_THREAD_NUM1 == 512
                  || COND_KERNEL_THREAD_NUM1 == 256 || COND_KERNEL_THREAD_NUM1 == 128);

    cudaLaunchConfig_t config1;
    config1.gridDim = block_num;
    config1.blockDim = COND_KERNEL_THREAD_NUM1;
    config1.dynamicSmemBytes = 0;
    config1.stream = at::cuda::getCurrentCUDAStream();
    auto attr = get_cuda_launch_attribute();
    config1.attrs = &attr;
    config1.numAttrs = 1;

    cudaLaunchConfig_t config2;
    config2.gridDim = 1;
    config2.blockDim = COND_KERNEL_THREAD_NUM2;
    config2.dynamicSmemBytes = 0;
    config2.stream = at::cuda::getCurrentCUDAStream();
    config2.attrs = &attr;
    config2.numAttrs = 1;

    cudaLaunchKernelEx(
        &config1, &conditional_index_step1_kernel<COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
        num_buf, m, N);
    cudaLaunchKernelEx(
        &config2, &conditional_index_step2_kernel<COND_KERNEL_THREAD_NUM1, COND_KERNEL_THREAD_NUM2>,
        num_buf, total_num_buf, block_num);

    if (x.dtype() == at::kShort) {
        cudaLaunchKernelEx(
            &config1,
            &conditional_index_step3_kernel<int16_t, COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
            out_buf, num_buf, x, m, N);
    } else if (x.dtype() == at::kByte) {
        cudaLaunchKernelEx(
            &config1,
            &conditional_index_step3_kernel<uint8_t, COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
            out_buf, num_buf, x, m, N);
    } else if (x.dtype() == at::kChar) {
        cudaLaunchKernelEx(
            &config1,
            &conditional_index_step3_kernel<int8_t, COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
            out_buf, num_buf, x, m, N);
    } else {
        assert(false);
    }

    return { out_buf, total_num_buf, num_buf };
}

at::Tensor conditional_index_part2_cuda(const at::Tensor& x, const at::Tensor& s, int* size_ptr)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (size_ptr == nullptr) {
        int s_cpu;
        CUDA_CHECK(cudaMemcpyAsync(&s_cpu, s.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost,
                                   stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto out = x.slice(0, 0, s_cpu);
        return out;
    }
    CUDA_CHECK(cudaMemcpyAsync(size_ptr, s.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    auto out = x.slice(0, 0, *size_ptr);
    return out;
}

template <typename T, int THREAD_NUM1, int PER_THREAD_NUM>
__global__ void conditional_recover_with_type_conversion_kernel(GPUTensor1D<T> out,
                                                                const GPUTensor2D<int> num,
                                                                const GPUTensor1D<int8_t> x,
                                                                const GPUTensor1D<bool> m, const int N)
{
    const int tidx = threadIdx.x;
    const int bidx = blockIdx.x;

    const int start_idx = PER_THREAD_NUM * (bidx * THREAD_NUM1 + tidx);
    const int end_idx = min(start_idx + PER_THREAD_NUM, N);
    cudaGDC();

    // set according to prefix sum
    int curr_src_idx = tidx > 0 ? num[bidx][tidx - 1] : 0;
    curr_src_idx += (bidx > 0 ? num[bidx - 1][THREAD_NUM1 - 1] : 0);
    for (int i = start_idx; i < end_idx; i++) {
        if (m[i]) {
            out[i] = static_cast<T>(x[curr_src_idx]);
            curr_src_idx++;
        } else {
            out[i] = static_cast<T>(0);
        }
    }
}

at::Tensor conditional_recover_with_type_conversion_cuda(const at::Tensor& x, const at::Tensor& m,
                                                         const at::Tensor& num,
                                                         const at::ScalarType dtype, at::Tensor& out_buf)
{
    assert(x.dtype() == at::kChar);
    const int N = m.numel();
    const int block_num = (N + COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM - 1)
                          / (COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM);
    static_assert(COND_KERNEL_THREAD_NUM1 == 1024 || COND_KERNEL_THREAD_NUM1 == 512
                  || COND_KERNEL_THREAD_NUM1 == 256 || COND_KERNEL_THREAD_NUM1 == 128);

    cudaLaunchConfig_t config1;
    config1.gridDim = block_num;
    config1.blockDim = COND_KERNEL_THREAD_NUM1;
    config1.dynamicSmemBytes = 0;
    config1.stream = at::cuda::getCurrentCUDAStream();
    auto attr = get_cuda_launch_attribute();
    config1.attrs = &attr;
    config1.numAttrs = 1;

    if (dtype == at::kHalf) {
        cudaLaunchKernelEx(
            &config1,
            &conditional_recover_with_type_conversion_kernel<at::Half, COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
            out_buf, num, x, m, N);
    } else if (dtype == at::kChar) {
        cudaLaunchKernelEx(
            &config1,
            &conditional_recover_with_type_conversion_kernel<int8_t, COND_KERNEL_THREAD_NUM1, COND_KERNEL_PER_THREAD_NUM>,
            out_buf, num, x, m, N);
    } else {
        assert(false);
    }

    return out_buf;
}

__global__ void divide_with_clamp_inplace_kernel(GPUTensor4D<at::Half> y,
                                                 const GPUTensor4D<at::Half> q, const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _y0;
        at::Half _y1;
        load2(&y[0][c0][h][w], _y0, _y1);
        at::Half _q0;
        at::Half _q1;
        load2(&q[0][c0][h][w], _q0, _q1);
        _q0 = max(_q0, static_cast<at::Half>(0.5f));
        _q1 = max(_q1, static_cast<at::Half>(0.5f));
        _y0 = _y0 * reciprocal(_q0);
        _y1 = _y1 * reciprocal(_q1);
        store2(&y[0][c0][h][w], _y0, _y1);
    }
}

at::Tensor divide_with_clamp_min_inplace_cuda(at::Tensor& y, const at::Tensor& q)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);

    cudaLaunchKernelEx(&config, &divide_with_clamp_inplace_kernel, y, q, chw);

    return y;
}

template <typename T>
__global__ void int8_to_dtype_kernel(GPUTensor1D<T> out, const GPUTensor1D<int8_t> x, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int n0 = n * 2 + 0;
    const int n1 = n * 2 + 1;
    cudaGDC();

    if (n < N) {
        int8_t _x0 = x[n0];
        int8_t _x1 = x[n1];
        store2(&out[n0], static_cast<T>(_x0), static_cast<T>(_x1));
    }
}

at::Tensor int8_to_dtype_cuda(const at::Tensor& x, const at::ScalarType dtype, const int C,
                              const int H, const int W, at::Tensor& out_buf)
{
    auto [config, N] = get_kernel_config_1D<2>(x);

    if (dtype == at::kHalf) {
        cudaLaunchKernelEx(&config, &int8_to_dtype_kernel<at::Half>, out_buf, x, N);
    } else if (dtype == at::kShort) {
        cudaLaunchKernelEx(&config, &int8_to_dtype_kernel<int16_t>, out_buf, x, N);
    } else {
        assert(false);
    }
    return out_buf;
}

template <int unrolling_factor>
__global__ void multiply_with_broadcast_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> a,
                                               const GPUTensor1D<at::Half> b, const int C, const int N)
{
    static_assert(unrolling_factor == 1 || unrolling_factor == 2 || unrolling_factor == 4);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    const int n0 = n * unrolling_factor;
    cudaGDC();

    if (c < C && n < N) {
        const at::Half _b = b[c];
        if (unrolling_factor >= 1) {
            out[0][c][0][n0 + 0] = a[0][c][0][n0 + 0] * _b;
        }
        if (unrolling_factor >= 2) {
            out[0][c][0][n0 + 1] = a[0][c][0][n0 + 1] * _b;
        }
        if (unrolling_factor >= 4) {
            out[0][c][0][n0 + 2] = a[0][c][0][n0 + 2] * _b;
            out[0][c][0][n0 + 3] = a[0][c][0][n0 + 3] * _b;
        }
    }
}

at::Tensor multiply_with_broadcast_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out_buf)
{
    auto attr = get_cuda_launch_attribute();

    const int C = a.size(1);
    int N = a.size(2) * a.size(3);
    int factor;
    if (N % 4 == 0) {
        factor = 4;
        N /= 4;
    } else if (N % 2 == 0) {
        factor = 2;
        N /= 2;
    } else {
        factor = 1;
    }
    const int BLOCK_SIZE_C = 256;
    const int BLOCK_SIZE_N = 1;
    const dim3 blockDim(BLOCK_SIZE_C, BLOCK_SIZE_N);
    const dim3 gridDim((C + BLOCK_SIZE_C - 1) / BLOCK_SIZE_C, (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N);

    cudaLaunchConfig_t config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    config.attrs = &attr;
    config.numAttrs = 1;

    if (factor == 4) {
        cudaLaunchKernelEx(&config, &multiply_with_broadcast_kernel<4>, out_buf, a, b, C, N);
    } else if (factor == 2) {
        cudaLaunchKernelEx(&config, &multiply_with_broadcast_kernel<2>, out_buf, a, b, C, N);
    } else {
        cudaLaunchKernelEx(&config, &multiply_with_broadcast_kernel<1>, out_buf, a, b, C, N);
    }
    return out_buf;
}

template <bool scale_out = true, bool add_inplace = false, bool multiply = false>
__global__ void process_with_mask_kernel(GPUTensor4D<at::Half> y_q, GPUTensor4D<at::Half> y_hat,
                                         GPUTensor4D<at::Half> s_hat, const GPUTensor4D<at::Half> y,
                                         const GPUTensor4D<at::Half> scales,
                                         const GPUTensor4D<at::Half> means,
                                         const GPUTensor4D<bool> mask, const GPUTensor4D<at::Half> quant,
                                         const at::Half force_zero_thres, const CudaCHW chw)
{
    static_assert(!scale_out || !multiply);
    static_assert(!multiply || add_inplace);
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        bool _mask0;
        bool _mask1;
        load2(&mask[0][c0][h][w], _mask0, _mask1);
        at::Half _y0;
        at::Half _y1;
        load2(&y[0][c0][h][w], _y0, _y1);

        const at::Half _min_val = static_cast<at::Half>(-128.f);
        const at::Half _max_val = static_cast<at::Half>(127.f);

        at::Half _scale0;
        at::Half _scale1;
        load2(&scales[0][c0][h][w], _scale0, _scale1);
        at::Half _means0;
        at::Half _means1;
        load2(&means[0][c0][h][w], _means0, _means1);

        at::Half _s_hat0 = _scale0 * _mask0;
        at::Half _s_hat1 = _scale1 * _mask1;
        at::Half _means_hat0 = _means0 * _mask0;
        at::Half _means_hat1 = _means1 * _mask1;
        at::Half _y_res0 = (_y0 - _means_hat0) * _mask0;
        at::Half _y_res1 = (_y1 - _means_hat1) * _mask1;
        at::Half _y_q0 = round(_y_res0);
        at::Half _y_q1 = round(_y_res1);
        _y_q0 = _y_q0 * (_s_hat0 > force_zero_thres);
        _y_q1 = _y_q1 * (_s_hat1 > force_zero_thres);
        _y_q0 = max(min(_y_q0, _max_val), _min_val);
        _y_q1 = max(min(_y_q1, _max_val), _min_val);
        at::Half _y_hat0 = _y_q0 + _means_hat0;
        at::Half _y_hat1 = _y_q1 + _means_hat1;

        if constexpr (add_inplace) {
            at::Half _t0;
            at::Half _t1;
            load2(&y_q[0][c0][h][w], _t0, _t1);
            store2(&y_q[0][c0][h][w], _t0 + _y_q0, _t1 + _y_q1);
        } else {
            store2(&y_q[0][c0][h][w], _y_q0, _y_q1);
        }

        if constexpr (multiply) {
            at::Half _q0;
            at::Half _q1;
            load2(&quant[0][c0][h][w], _q0, _q1);
            _q0 = max(_q0, static_cast<at::Half>(0.5f));
            _q1 = max(_q1, static_cast<at::Half>(0.5f));
            at::Half _t0;
            at::Half _t1;
            load2(&y_hat[0][c0][h][w], _t0, _t1);
            _y_hat0 += _t0;
            _y_hat1 += _t1;
            store2(&y_hat[0][c0][h][w], _y_hat0 * _q0, _y_hat1 * _q1);
        } else if constexpr (add_inplace) {
            at::Half _t0;
            at::Half _t1;
            load2(&y_hat[0][c0][h][w], _t0, _t1);
            store2(&y_hat[0][c0][h][w], _y_hat0 + _t0, _y_hat1 + _t1);
        } else {
            store2(&y_hat[0][c0][h][w], _y_hat0, _y_hat1);
        }

        if constexpr (scale_out) {
            store2(&s_hat[0][c0][h][w], _s_hat0, _s_hat1);
        }
    }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor>
process_with_mask_cuda(const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means,
                       const at::Tensor& mask, const float force_zero_thres_float,
                       const int32_t force_zero_thres_int, at::Tensor& y_q_buf,
                       at::Tensor& y_hat_buf, at::Tensor& s_hat_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &process_with_mask_kernel<>, y_q_buf, y_hat_buf, s_hat_buf, y,
                       scales, means, mask, nullptr, force_zero_thres_float, chw);

    return { y_q_buf, y_hat_buf, s_hat_buf };
}

std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_add_and_multiply_inplace_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    at::Tensor& y_q, at::Tensor& y_hat, const at::Tensor& quant, const float force_zero_thres_float,
    const int32_t force_zero_thres_int)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &process_with_mask_kernel<false, true, true>, y_q, y_hat, nullptr,
                       y, scales, means, mask, quant, force_zero_thres_float, chw);

    return { y_q, y_hat };
}

std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_add_inplace_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    at::Tensor& y_q, at::Tensor& y_hat, const float force_zero_thres_float,
    const int32_t force_zero_thres_int)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &process_with_mask_kernel<false, true>, y_q, y_hat, nullptr, y,
                       scales, means, mask, nullptr, force_zero_thres_float, chw);

    return { y_q, y_hat };
}

std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    const float force_zero_thres_float, const int32_t force_zero_thres_int, at::Tensor& y_q_buf,
    at::Tensor& y_hat_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    assert(scales.scalar_type() == at::kHalf);
    cudaLaunchKernelEx(&config, &process_with_mask_kernel<false>, y_q_buf, y_hat_buf, nullptr, y,
                       scales, means, mask, nullptr, force_zero_thres_float, chw);

    return { y_q_buf, y_hat_buf };
}

template <bool add_inplace = false, bool multiply = false>
__global__ void restore_y_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> y,
                                 const GPUTensor4D<at::Half> means, const GPUTensor4D<bool> mask,
                                 const GPUTensor4D<at::Half> q, const CudaCHW chw)
{
    static_assert(!multiply || add_inplace);
    // C, H, and W are the shape of y tensor
    const auto [c, h, w] = global_idx_chw();
    const int c0 = c * 2 + 0;
    const int c1 = c * 2 + 1;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        bool _mask0 = mask[0][c0][h][w];
        bool _mask1 = mask[0][c1][h][w];

        at::Half _y0;
        at::Half _y1;
        load2(&y[0][c0][h][w], _y0, _y1);
        at::Half _means0;
        at::Half _means1;
        load2(&means[0][c0][h][w], _means0, _means1);

        _y0 = (_y0 + _means0) * _mask0;
        _y1 = (_y1 + _means1) * _mask1;
        if constexpr (add_inplace) {
            at::Half _out_0;
            at::Half _out_1;
            load2(&out[0][c0][h][w], _out_0, _out_1);

            _y0 = _y0 + _out_0;
            _y1 = _y1 + _out_1;
            if constexpr (multiply) {
                at::Half _q0;
                at::Half _q1;
                load2(&q[0][c0][h][w], _q0, _q1);
                _q0 = max(_q0, static_cast<at::Half>(0.5f));
                _q1 = max(_q1, static_cast<at::Half>(0.5f));
                _y0 = _y0 * _q0;
                _y1 = _y1 * _q1;
            }
        }
        store2(&out[0][c0][h][w], _y0, _y1);
    }
}

at::Tensor restore_y_and_add_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                          const at::Tensor& mask, at::Tensor& y_hat)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    cudaLaunchKernelEx(&config, &restore_y_kernel<true>, y_hat, y, means, mask, nullptr, chw);
    return y_hat;
}

at::Tensor restore_y_and_add_multiply_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                                   const at::Tensor& mask, at::Tensor& y_hat,
                                                   const at::Tensor& q)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    cudaLaunchKernelEx(&config, &restore_y_kernel<true, true>, y_hat, y, means, mask, q, chw);
    return y_hat;
}

at::Tensor restore_y_cuda(const at::Tensor& y, const at::Tensor& means, const at::Tensor& mask,
                          at::Tensor& out_buf)
{
    auto [config, chw] = get_kernel_config_4D<2>(y);
    cudaLaunchKernelEx(&config, &restore_y_kernel<>, out_buf, y, means, mask, nullptr, chw);
    return out_buf;
}

template <bool add_inplace = false>
__global__ void restore_y_4x_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> y,
                                    const GPUTensor4D<at::Half> means, const GPUTensor4D<bool> mask,
                                    const CudaCHW chw)
{
    // C, H, and W are the shape of y tensor
    const auto [c, h, w] = global_idx_chw();
    const int c1 = c;
    const int c2 = c1 + chw.C;
    const int c3 = c2 + chw.C;
    const int c4 = c3 + chw.C;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        bool _mask1 = mask[0][c1][h][w];
        bool _mask2 = mask[0][c2][h][w];
        bool _mask3 = mask[0][c3][h][w];
        bool _mask4 = mask[0][c4][h][w];

        at::Half _y = y[0][c][h][w];
        at::Half _t1 = (_y + means[0][c1][h][w]) * _mask1;
        at::Half _t2 = (_y + means[0][c2][h][w]) * _mask2;
        at::Half _t3 = (_y + means[0][c3][h][w]) * _mask3;
        at::Half _t4 = (_y + means[0][c4][h][w]) * _mask4;
        if constexpr (add_inplace) {
            out[0][c1][h][w] = out[0][c1][h][w] + _t1;
            out[0][c2][h][w] = out[0][c2][h][w] + _t2;
            out[0][c3][h][w] = out[0][c3][h][w] + _t3;
            out[0][c4][h][w] = out[0][c4][h][w] + _t4;
        } else {
            out[0][c1][h][w] = _t1;
            out[0][c2][h][w] = _t2;
            out[0][c3][h][w] = _t3;
            out[0][c4][h][w] = _t4;
        }
    }
}

at::Tensor restore_y_4x_and_add_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                             const at::Tensor& mask, at::Tensor& y_hat)
{
    const int C = y.size(1);
    const int H = y.size(2);
    const int W = y.size(3);
    auto [config, chw] = get_kernel_config_4D(C, H, W);

    cudaLaunchKernelEx(&config, &restore_y_4x_kernel<true>, y_hat, y, means, mask, chw);
    return y_hat;
}

at::Tensor restore_y_4x_cuda(const at::Tensor& y, const at::Tensor& means, const at::Tensor& mask,
                             at::Tensor& out_buf)
{
    const int B = y.size(0);
    const int C = y.size(1);
    const int H = y.size(2);
    const int W = y.size(3);

    auto [config, chw] = get_kernel_config_4D(C, H, W);

    cudaLaunchKernelEx(&config, &restore_y_4x_kernel<>, out_buf, y, means, mask, chw);
    return out_buf;
}

__global__ void restore_y_4x_add_and_multiply_broadcast_kernel(
    GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> y_hat_so_far,
    const GPUTensor4D<at::Half> y, const GPUTensor4D<at::Half> means, const GPUTensor4D<bool> mask,
    const GPUTensor1D<at::Half> q, const CudaCHW chw)
{
    const auto [c, h, w] = global_idx_chw();
    const int c1 = c;
    const int c2 = c1 + chw.C;
    const int c3 = c2 + chw.C;
    const int c4 = c3 + chw.C;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        bool _mask1 = mask[0][c1][h][w];
        bool _mask2 = mask[0][c2][h][w];
        bool _mask3 = mask[0][c3][h][w];
        bool _mask4 = mask[0][c4][h][w];

        at::Half _y = y[0][c][h][w];
        out[0][c1][h][w] = (y_hat_so_far[0][c1][h][w] + (_y + means[0][c1][h][w]) * _mask1) * q[c1];
        out[0][c2][h][w] = (y_hat_so_far[0][c2][h][w] + (_y + means[0][c2][h][w]) * _mask2) * q[c2];
        out[0][c3][h][w] = (y_hat_so_far[0][c3][h][w] + (_y + means[0][c3][h][w]) * _mask3) * q[c3];
        out[0][c4][h][w] = (y_hat_so_far[0][c4][h][w] + (_y + means[0][c4][h][w]) * _mask4) * q[c4];
    }
}

at::Tensor restore_y_4x_add_and_multiply_broadcast_cuda(const at::Tensor& y, const at::Tensor& means,
                                                        const at::Tensor& mask,
                                                        const at::Tensor& y_hat_so_far,
                                                        const at::Tensor& q, at::Tensor& out_buf)
{
    const int C = y.size(1);
    const int H = y.size(2);
    const int W = y.size(3);

    auto [config, chw] = get_kernel_config_4D(C, H, W);

    cudaLaunchKernelEx(&config, &restore_y_4x_add_and_multiply_broadcast_kernel, out_buf,
                       y_hat_so_far, y, means, mask, q, chw);
    return out_buf;
}

__global__ void round_z_kernel(GPUTensor1D<at::Half> z_hat, GPUTensor1D<int8_t> z_int8,
                               const GPUTensor1D<at::Half> z, const int N)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int n0 = n * 2 + 0;
    cudaGDC();

    if (n < N) {
        at::Half _z0;
        at::Half _z1;
        load2(&z[n0], _z0, _z1);
        _z0 = round(_z0);
        _z1 = round(_z1);
        _z0 = max(_z0, static_cast<at::Half>(-64.f));
        _z1 = max(_z1, static_cast<at::Half>(-64.f));
        _z0 = min(_z0, static_cast<at::Half>(63.f));
        _z1 = min(_z1, static_cast<at::Half>(63.f));
        int8_t _z0_int8 = static_cast<int8_t>(_z0);
        int8_t _z1_int8 = static_cast<int8_t>(_z1);
        store2(&z_int8[n0], _z0_int8, _z1_int8);
        store2(&z_hat[n0], _z0, _z1);
    }
}

std::tuple<at::Tensor, at::Tensor> round_z_cuda(const at::Tensor& z, at::Tensor& z_hat_buf,
                                                at::Tensor& z_int8_buf)
{
    auto [config, N] = get_kernel_config_1D<2>(z);

    cudaLaunchKernelEx(&config, &round_z_kernel, z_hat_buf, z_int8_buf, z, N);

    return { z_hat_buf, z_int8_buf };
}

__global__ void single_part_for_reading_4x_kernel(GPUTensor4D<at::Half> out,
                                                  const GPUTensor4D<at::Half> x,
                                                  const GPUTensor4D<bool> mask, const CudaCHW chw)
{
    // C, H, and W are the shape of out tensor
    const auto [c, h, w] = global_idx_chw();
    const int c1 = c;
    const int c2 = c1 + chw.C;
    const int c3 = c2 + chw.C;
    const int c4 = c3 + chw.C;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _x1 = x[0][c1][h][w] * mask[0][c1][h][w];
        at::Half _x2 = x[0][c2][h][w] * mask[0][c2][h][w];
        at::Half _x3 = x[0][c3][h][w] * mask[0][c3][h][w];
        at::Half _x4 = x[0][c4][h][w] * mask[0][c4][h][w];
        out[0][c][h][w] = _x1 + _x2 + _x3 + _x4;
    }
}

at::Tensor single_part_for_reading_4x_cuda(const at::Tensor& x, const at::Tensor& mask,
                                           at::Tensor& out_buf)
{
    const int B = x.size(0);
    const int C = x.size(1) / 4;
    const int H = x.size(2);
    const int W = x.size(3);

    auto [config, chw] = get_kernel_config_4D(C, H, W);

    cudaLaunchKernelEx(&config, &single_part_for_reading_4x_kernel, out_buf, x, mask, chw);
    return out_buf;
}

__global__ void single_part_for_writing_4x_kernel(GPUTensor4D<at::Half> out,
                                                  const GPUTensor4D<at::Half> x, const CudaCHW chw)
{
    // C, H, and W are the shape of out tensor
    const auto [c, h, w] = global_idx_chw();
    const int c1 = c;
    const int c2 = c1 + chw.C;
    const int c3 = c2 + chw.C;
    const int c4 = c3 + chw.C;
    cudaGDC();

    if (c < chw.C && h < chw.H && w < chw.W) {
        at::Half _x1 = x[0][c1][h][w];
        at::Half _x2 = x[0][c2][h][w];
        at::Half _x3 = x[0][c3][h][w];
        at::Half _x4 = x[0][c4][h][w];
        out[0][c][h][w] = _x1 + _x2 + _x3 + _x4;
    }
}

at::Tensor single_part_for_writing_4x_cuda(const at::Tensor& x, at::Tensor& out_buf)
{
    const int B = x.size(0);
    const int C = x.size(1) / 4;
    const int H = x.size(2);
    const int W = x.size(3);

    auto [config, chw] = get_kernel_config_4D(C, H, W);

    cudaLaunchKernelEx(&config, &single_part_for_writing_4x_kernel, out_buf, x, chw);
    return out_buf;
}
