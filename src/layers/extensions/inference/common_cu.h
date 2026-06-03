// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "common_cutlass.h"
#include "def_const.h"

constexpr uint64_t conv_key(uint64_t sm, uint64_t W, uint64_t H, uint64_t C2, uint64_t C1,
                            uint64_t stride, uint64_t kernel)
{
    /**
     * Valid range:
     *   kernel:  0 - 1 ()     2^0  - 2^0
     *   stride:  0 - 1 (<<1)  2^1  - 2^1
     *   C1: 2^3 - 2^13 (>>1)  2^2  - 2^12
     *   C2: 2^3 - 2^13 (<<10)  2^13 - 2^23
     *   H:  2^0 - 2^15 (<<24) 2^24 - 2^39
     *   W:  2^0 - 2^15 (<<40) 2^40 - 2^55
     *   SM: 1 - 0xFF (8bits) (<<56)
     */
    assert(C2 <= (1 << 13) && (C2 & 7) == 0 && C1 <= (1 << 13) && (C1 & 7) == 0 && H <= (1 << 15)
           && W <= (1 << 15) && sm <= 0xFF);
    return kernel | (stride << 1) | (C1 >> 1) | (C2 << 10) | (H << 24) | (W << 40) | (sm << 56);
}

struct CudaCHW {
    int C;
    int H;
    int W;
};

template <int factor = 1>
__forceinline__ std::tuple<cudaLaunchConfig_t, int> get_kernel_config_1D(const at::Tensor& x)
{
    int N = x.numel();
    assert(N % factor == 0);
    N /= factor;
    const int BLOCK_SIZE = 256;
    const dim3 blockDim(BLOCK_SIZE);
    const dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    static cudaLaunchAttribute attrs = get_cuda_launch_attribute();
    cudaLaunchConfig_t config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    config.attrs = &attrs;
    config.numAttrs = 1;

    return { config, N };
}

template <int factor = 1, int BLOCK_SIZE_X = 64, int BLOCK_SIZE_Y = 2, int BLOCK_SIZE_Z = 2>
__forceinline__ std::tuple<cudaLaunchConfig_t, CudaCHW> get_kernel_config_4D(const int C,
                                                                             const int H, const int W)
{
    assert(C % factor == 0);
    const dim3 gridDim((C / factor + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                       (W + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y, (H + BLOCK_SIZE_Z - 1) / BLOCK_SIZE_Z);
    const dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

    static cudaLaunchAttribute attrs = get_cuda_launch_attribute();
    cudaLaunchConfig_t config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    config.attrs = &attrs;
    config.numAttrs = 1;
    return { config, { C / factor, H, W } };
}

template <int factor = 1, int BLOCK_SIZE_X = 64, int BLOCK_SIZE_Y = 2, int BLOCK_SIZE_Z = 2>
__forceinline__ std::tuple<cudaLaunchConfig_t, CudaCHW> get_kernel_config_4D(const at::Tensor& x)
{
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    return get_kernel_config_4D<factor, BLOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z>(C, H, W);
}

struct __align__(2) bool2
{
    bool x;
    bool y;
};

template <typename T>
__forceinline__ __device__ void load2(const T* src, T& dst0, T& dst1)
{
    if constexpr (std::is_same_v<T, at::Half>) {
        half2 v = *reinterpret_cast<const half2*>(src);
        dst0 = __low2half(v);
        dst1 = __high2half(v);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        short2 v = *reinterpret_cast<const short2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, int8_t>) {
        char2 v = *reinterpret_cast<const char2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        uchar2 v = *reinterpret_cast<const uchar2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else if constexpr (std::is_same_v<T, bool>) {
        bool2 v = *reinterpret_cast<const bool2*>(src);
        dst0 = v.x;
        dst1 = v.y;
    } else {
        dst0 = src[0];
        dst1 = src[1];
    }
}

__forceinline__ __device__ at::Half log(const at::Half& a)
{
    return hlog(a);
}

__forceinline__ __device__ bool2 make_bool2(bool x, bool y)
{
    bool2 t;
    t.x = x;
    t.y = y;
    return t;
}

__forceinline__ __device__ at::Half max(const at::Half& a, const at::Half& b)
{
    return __hmax(a, b);
}

__forceinline__ __device__ at::Half min(const at::Half& a, const at::Half& b)
{
    return __hmin(a, b);
}

__forceinline__ __device__ at::Half operator*(const at::Half& a, const bool b)
{
    return b ? a : static_cast<at::Half>(0.f);
}

__forceinline__ __device__ bool operator>(const at::Half& a, const at::Half& b)
{
    return __hgt(a, b);
}

__forceinline__ __device__ at::Half reciprocal(const at::Half& a)
{
    return hrcp(a);
}

template <typename T>
__forceinline__ __device__ void store2(T* dst, T src0, T src1)
{
    if constexpr (std::is_same_v<T, at::Half>) {
        *reinterpret_cast<half2*>(dst) = __halves2half2(src0, src1);
    } else if constexpr (std::is_same_v<T, int16_t>) {
        *reinterpret_cast<short2*>(dst) = make_short2(src0, src1);
    } else if constexpr (std::is_same_v<T, int8_t>) {
        *reinterpret_cast<char2*>(dst) = make_char2(src0, src1);
    } else if constexpr (std::is_same_v<T, uint8_t>) {
        *reinterpret_cast<uchar2*>(dst) = make_uchar2(src0, src1);
    } else if constexpr (std::is_same_v<T, bool>) {
        *reinterpret_cast<bool2*>(dst) = make_bool2(src0, src1);
    } else {
        dst[0] = src0;
        dst[1] = src1;
    }
}

__forceinline__ __device__ int16_t to_int16(const at::Half& a)
{
    return static_cast<int16_t>(__half2int_rd(a));
}

__forceinline__ __device__ uint8_t to_uint8(const at::Half& a)
{
    return static_cast<uint8_t>(__half2uint_rd(a));
}
