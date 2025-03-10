// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

// T maybe vector type, and may be different from t.dtype
template <typename T>
struct GPUTensor1D {
    GPUTensor1D(torch::Tensor& t) : ptr(static_cast<T*>(t.data_ptr())) {}
    GPUTensor1D(const torch::Tensor& t) : ptr(static_cast<T*>(t.data_ptr())) {}
    GPUTensor1D(T* t) : ptr(static_cast<T*>(t)) { assert(t == nullptr); }

    __device__ T& operator[](int idx) { return ptr[idx]; }
    __device__ T& operator[](int idx) const { return ptr[idx]; }

    T* __restrict__ const ptr;
};

template <typename scalar_t>
using Packed4DTensorAccessor32 = torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits>;
template <typename scalar_t>
using Packed1DTensorAccessor32 = torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits>;

struct __align__(8) Half4
{
    c10::Half x;
    c10::Half y;
    c10::Half z;
    c10::Half w;
};

struct __align__(4) bool4
{
    bool x;
    bool y;
    bool z;
    bool w;
};

__forceinline__ __device__ float4 make_vec4(const float& x, const float& y, const float& z,
                                            const float& w)
{
    return make_float4(x, y, z, w);
}

__forceinline__ __device__ Half4 make_vec4(const c10::Half& x, const c10::Half& y,
                                           const c10::Half& z, const c10::Half& w)
{
    Half4 t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}

__forceinline__ __device__ Half4 make_Half4(const c10::Half& x, const c10::Half& y,
                                            const c10::Half& z, const c10::Half& w)
{
    Half4 t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}

__forceinline__ __device__ bool4 make_vec4(const bool& x, const bool& y, const bool& z, const bool& w)
{
    bool4 t;
    t.x = x;
    t.y = y;
    t.z = z;
    t.w = w;
    return t;
}

__forceinline__ __device__ c10::Half round(const c10::Half& a)
{
    return static_cast<c10::Half>(__half2int_rn(a));
}

template <typename T>
__forceinline__ __device__ T round(const T& a)
{
    return make_vec4(round(a.x), round(a.y), round(a.z), round(a.w));
}

__forceinline__ __device__ int8_t to_int8(const float& a)
{
    return static_cast<int8_t>(a);
}

__forceinline__ __device__ int8_t to_int8(const c10::Half& a)
{
    return static_cast<int8_t>(a);
}

template <typename T>
__forceinline__ __device__ char4 to_int8(const T& a)
{
    return make_char4(to_int8(a.x), to_int8(a.y), to_int8(a.z), to_int8(a.w));
}

__forceinline__ __device__ uint8_t to_uint8(const float& a)
{
    return static_cast<uint8_t>(a);
}

__forceinline__ __device__ uint8_t to_uint8(const c10::Half& a)
{
    return static_cast<uint8_t>(__half2uint_rd(a));
}

template <typename T>
__forceinline__ __device__ uchar4 to_uint8(const T& a)
{
    return make_uchar4(to_uint8(a.x), to_uint8(a.y), to_uint8(a.z), to_uint8(a.w));
}

__forceinline__ __device__ int16_t to_int16(const float& a)
{
    return static_cast<int16_t>(a);
}

__forceinline__ __device__ int16_t to_int16(const c10::Half& a)
{
    return static_cast<int16_t>(__half2int_rd(a));
}

template <typename T>
__forceinline__ __device__ short4 to_int16(const T& a)
{
    return make_short4(to_int16(a.x), to_int16(a.y), to_int16(a.z), to_int16(a.w));
}

__forceinline__ __device__ short4 operator<<(const short4& a, const int b)
{
    return make_short4(a.x << b, a.y << b, a.z << b, a.w << b);
}

__forceinline__ __device__ c10::Half min(const c10::Half& a, const c10::Half& b)
{
    return __hmin(a, b);
}
__forceinline__ __device__ c10::Half max(const c10::Half& a, const c10::Half& b)
{
    return __hmax(a, b);
}

__forceinline__ __device__ bool operator>(const c10::Half& a, const c10::Half& b)
{
    return __hgt(a, b);
}

__forceinline__ __device__ bool operator<(const c10::Half& a, const c10::Half& b)
{
    return __hlt(a, b);
}

__forceinline__ __device__ c10::Half log(const c10::Half& a)
{
    return hlog(a);
}

__forceinline__ __device__ short4 operator+(const short4& a, const short4& b)
{
    return make_short4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ float4 operator+(const float4& a, const float4& b)
{
    return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__forceinline__ __device__ Half4 operator+(const Half4& a, const Half4& b)
{
    return make_Half4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

template <typename T>
__forceinline__ __device__ T operator-(const T& a, const T& b)
{
    return make_vec4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__forceinline__ __device__ float4 operator-(const float4& a, const float& b)
{
    return make_vec4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__forceinline__ __device__ Half4 operator-(const Half4& a, const c10::Half& b)
{
    return make_vec4(a.x - b, a.y - b, a.z - b, a.w - b);
}

__forceinline__ __device__ c10::Half operator*(const c10::Half& a, const bool b)
{
    return b ? a : static_cast<c10::Half>(0.f);
}

template <typename T>
__forceinline__ __device__ T operator*(const T& a, const T& b)
{
    return make_vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template <typename T>
__forceinline__ __device__ T operator*(const T& a, const bool4& b)
{
    return make_vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

template <typename T1, typename T2>
__forceinline__ __device__ T1 operator*(const T1& a, const T2& b)
{
    return make_vec4(a.x * b, a.y * b, a.z * b, a.w * b);
}

template <typename T>
__forceinline__ __device__ T max(const T& a, const T& b)
{
    return make_vec4(max(a.x, b.x), max(a.y, b.y), max(a.z, b.z), max(a.w, b.w));
}

template <typename T1, typename T2>
__forceinline__ __device__ T1 max(const T1& a, const T2& b)
{
    return make_vec4(max(a.x, b), max(a.y, b), max(a.z, b), max(a.w, b));
}

template <typename T>
__forceinline__ __device__ T min(const T& a, const T& b)
{
    return make_vec4(min(a.x, b.x), min(a.y, b.y), min(a.z, b.z), min(a.w, b.w));
}

template <typename T1, typename T2>
__forceinline__ __device__ T1 min(const T1& a, const T2& b)
{
    return make_vec4(min(a.x, b), min(a.y, b), min(a.z, b), min(a.w, b));
}

template <typename T>
__forceinline__ __device__ T log(const T& a)
{
    return make_vec4(log(a.x), log(a.y), log(a.z), log(a.w));
}

__forceinline__ __device__ float reciprocal(const float& a)
{
    return __frcp_rd(a);
}

__forceinline__ __device__ c10::Half reciprocal(const c10::Half& a)
{
    return hrcp(a);
}

template <typename T>
__forceinline__ __device__ T reciprocal(const T& a)
{
    return make_vec4(reciprocal(a.x), reciprocal(a.y), reciprocal(a.z), reciprocal(a.w));
}

template <typename T1, typename T2>
__forceinline__ __device__ bool4 operator>(const T1& a, const T2& b)
{
    return make_vec4(a.x > b, a.y > b, a.z > b, a.w > b);
}

__forceinline__ __device__ float sigmoid(const float x)
{
    return 1.0f / (1.0f + expf(-x));
}

__forceinline__ __device__ float wsilu(const float x)
{
    return x * sigmoid(4.0f * x);
}

__forceinline__ __device__ c10::Half wsilu(const c10::Half x)
{
    return __float2half_rn(wsilu(__half2float(x)));
}

__forceinline__ __device__ float4 wsilu(float4 data)
{
    data.x = wsilu(data.x);
    data.y = wsilu(data.y);
    data.z = wsilu(data.z);
    data.w = wsilu(data.w);
    return data;
}

__forceinline__ __device__ Half4 wsilu(Half4 data)
{
    data.x = wsilu(data.x);
    data.y = wsilu(data.y);
    data.z = wsilu(data.z);
    data.w = wsilu(data.w);
    return data;
}

__forceinline__ __device__ float multiply_add(const float a, const float b, const float c)
{
    return __fmaf_rn(a, b, c);
}

__forceinline__ __device__ c10::Half multiply_add(const c10::Half a, const c10::Half b, const c10::Half c)
{

    return __hfma(a, b, c);
}