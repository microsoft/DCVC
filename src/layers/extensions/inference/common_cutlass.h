// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/numeric_conversion.h>

#if CURRENT_DEVICE_SM == 90 || CURRENT_DEVICE_SM == 100
    #define cudaGDC() cudaGridDependencySynchronize()
#else
    #define cudaGDC() ((void)0)
#endif

template <typename T>
CUTLASS_HOST_DEVICE T four();

template <>
CUTLASS_HOST_DEVICE float four<float>()
{
    return 4.f;
}

template <>
CUTLASS_HOST_DEVICE cutlass::half_t four<cutlass::half_t>()
{
    uint16_t bits = 0x4400u;
    return reinterpret_cast<cutlass::half_t const&>(bits);
}

__forceinline__ cudaLaunchAttribute get_cuda_launch_attribute()
{
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
#if CURRENT_DEVICE_SM == 90 || CURRENT_DEVICE_SM == 100
    attr.val.programmaticStreamSerializationAllowed = 1;
#else
    attr.val.programmaticStreamSerializationAllowed = 0;
#endif
    return attr;
}

__device__ __forceinline__ std::tuple<int, int, int> global_idx_chw()
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int w = blockIdx.y * blockDim.y + threadIdx.y;
    const int h = blockIdx.z * blockDim.z + threadIdx.z;
    return { c, h, w };
}

// T maybe vector type, and may be different from t.dtype
template <typename T>
struct GPUTensor1D {
    GPUTensor1D(at::Tensor& t) : ptr(static_cast<T*>(t.data_ptr())) {}
    GPUTensor1D(const at::Tensor& t) : ptr(static_cast<T*>(t.data_ptr())) {}
    GPUTensor1D(std::nullptr_t) : ptr(nullptr) {}

    __device__ T& operator[](int idx) { return ptr[idx]; }
    __device__ T& operator[](int idx) const { return ptr[idx]; }

    T* __restrict__ const ptr;
};

template <typename T, size_t N>
class GPUTensorND : public at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits> {
    using Base = at::PackedTensorAccessor32<T, N, at::RestrictPtrTraits>;

public:
    GPUTensorND(at::Tensor& t) : Base(t.packed_accessor32<T, N, at::RestrictPtrTraits>()) {}
    GPUTensorND(const at::Tensor& t) : Base(t.packed_accessor32<T, N, at::RestrictPtrTraits>()) {}
    GPUTensorND(std::nullptr_t) : Base(make_null_base()) {}

    static inline Base make_null_base()
    {
        int32_t sizes[N] = { 0 };
        int32_t strides[N] = { 0 };
        return Base(nullptr, sizes, strides);
    }
};

template <typename T>
using GPUTensor2D = GPUTensorND<T, 2>;

template <typename T>
using GPUTensor4D = GPUTensorND<T, 4>;

constexpr int HINT_NUM = 30;
constexpr int HINT_NUM_EX = 96;

template <typename T>
struct WSiLUOp;

template <typename T, int N>
struct WSiLUOp<cutlass::Array<T, N>> {
    static const bool kIsHeavy = true;

    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& value) const
    {
        cutlass::epilogue::thread::Sigmoid<cutlass::Array<T, N>> sigmoid_op;
        cutlass::multiplies<cutlass::Array<T, N>> mul;
        return mul(value, sigmoid_op(mul(four<T>(), value)));
    }
};

// clang-format off
// Macro for Sm80 Hint definitions
#define DEFINE_HINT_SM80(N, TB_M, TB_N, TB_K, W_M, W_N, W_K, OP_M, OP_N, OP_K, STAGES) \
struct Hint##N {                                                                        \
    using SmArch = cutlass::arch::Sm80;                                                     \
    using ShapeThreadBlock = cutlass::gemm::GemmShape<TB_M, TB_N, TB_K>;                    \
    using ShapeWarp = cutlass::gemm::GemmShape<W_M, W_N, W_K>;                              \
    using ShapeOp = cutlass::gemm::GemmShape<OP_M, OP_N, OP_K>;                             \
    static constexpr int Stages = STAGES;                                                   \
}

//                        N    TB_M TB_N TB_K  W_M W_N W_K  OP_M OP_N OP_K  STAGES
DEFINE_HINT_SM80(    0,   256, 128,  32,  64, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    1,   128, 256,  32,  64, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    2,   128, 128,  32,  64, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    3,    64, 256,  32,  64, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    4,   256,  64,  32,  64, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    5,    64, 128,  32,  32, 64, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    6,   128,  64,  32,  64, 32, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    7,    64,  64,  32,  32, 32, 32,   16,   8,   8,      2);
DEFINE_HINT_SM80(    8,    64,  64,  64,  32, 32, 64,   16,   8,  16,      5);
DEFINE_HINT_SM80(    9,   256, 128,  32,  64, 64, 32,   16,   8,  16,      3);
DEFINE_HINT_SM80(   10,   128, 256,  32,  64, 64, 32,   16,   8,  16,      3);
DEFINE_HINT_SM80(   11,   256,  64,  32,  64, 64, 32,   16,   8,  16,      3);
DEFINE_HINT_SM80(   12,   256,  64,  32,  64, 64, 32,   16,   8,  16,      4);
DEFINE_HINT_SM80(   13,    64, 256,  32,  64, 64, 32,   16,   8,  16,      4);
DEFINE_HINT_SM80(   14,   128, 128,  32,  64, 64, 32,   16,   8,  16,      3);  // default Sm80, Sm86, Sm90
DEFINE_HINT_SM80(   15,   128, 128,  32,  64, 64, 32,   16,   8,  16,      4);
DEFINE_HINT_SM80(   16,   128, 128,  32,  64, 64, 32,   16,   8,  16,      5);
DEFINE_HINT_SM80(   17,   128,  64,  32,  64, 32, 32,   16,   8,  16,      6);
DEFINE_HINT_SM80(   18,    64, 128,  32,  32, 64, 32,   16,   8,  16,      6);
DEFINE_HINT_SM80(   19,    64,  64,  32,  32, 32, 32,   16,   8,  16,     10);
DEFINE_HINT_SM80(   20,   256, 128,  64,  64, 64, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   21,   128, 256,  64,  64, 64, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   22,   256,  64,  64,  64, 64, 64,   16,   8,  16,      4);
DEFINE_HINT_SM80(   23,    64, 256,  64,  64, 64, 64,   16,   8,  16,      4);
DEFINE_HINT_SM80(   24,   128, 128,  64,  64, 64, 64,   16,   8,  16,      4);
DEFINE_HINT_SM80(   25,   256,  64,  64,  64, 64, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   26,    64, 256,  64,  64, 64, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   27,   128, 128,  64,  64, 64, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   28,   128,  64,  64,  64, 32, 64,   16,   8,  16,      3);
DEFINE_HINT_SM80(   29,    64, 128,  64,  32, 64, 64,   16,   8,  16,      3);

// Macro for Sm90 Hint definitions
#define DEFINE_HINT_SM90(N, T_M, T_N, T_K, C_X, C_Y, C_Z)                             \
struct Hint##N {                                                                       \
    using SmArch = cutlass::arch::Sm90;                                                    \
    using TileShape = cute::Shape<cute::T_M, cute::T_N, cute::T_K>;                        \
    using ClusterShape = cute::Shape<cute::C_X, cute::C_Y, cute::C_Z>;                     \
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;             \
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperative;             \
}

//                        N   T_M   T_N   T_K  C_X  C_Y  C_Z
DEFINE_HINT_SM90(   30, _128, _128,  _64,  _1,  _2,  _1);
DEFINE_HINT_SM90(   31, _128, _128,  _64,  _2,  _1,  _1);
DEFINE_HINT_SM90(   32, _128, _256,  _64,  _1,  _2,  _1);
DEFINE_HINT_SM90(   33, _128, _256,  _64,  _2,  _1,  _1);
DEFINE_HINT_SM90(   34, _256, _128,  _64,  _1,  _2,  _1);
DEFINE_HINT_SM90(   35, _256, _128,  _64,  _2,  _1,  _1);
DEFINE_HINT_SM90(   36, _256, _256,  _64,  _1,  _2,  _1);
DEFINE_HINT_SM90(   37, _256, _256,  _64,  _2,  _1,  _1);

// Macros for Sm100 Hint definitions with different schedule combinations
// Type A: TmaWarpSpecialized2Sm + KernelTmaWarpSpecialized2SmSm100
#define DEFINE_HINT_SM100_2SM(N, T_M, T_N, T_K, C_X, C_Y, C_Z)                        \
struct Hint##N {                                                                       \
    using SmArch = cutlass::arch::Sm100;                                                   \
    using TileShape = cute::Shape<cute::T_M, cute::T_N, cute::T_K>;                        \
    using ClusterShape = cute::Shape<C_X, C_Y, cute::C_Z>;                                 \
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized2Sm;                     \
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;                \
}

// Type B: TmaWarpSpecialized1Sm + KernelTmaWarpSpecialized1SmSm100
#define DEFINE_HINT_SM100_1SM(N, T_M, T_N, T_K, C_X, C_Y, C_Z)                        \
struct Hint##N {                                                                       \
    using SmArch = cutlass::arch::Sm100;                                                   \
    using TileShape = cute::Shape<cute::T_M, cute::T_N, cute::T_K>;                        \
    using ClusterShape = cute::Shape<C_X, C_Y, cute::C_Z>;                                 \
    using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecialized1Sm;                     \
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;                \
}

// Type C: EpilogueScheduleAuto + KernelTmaWarpSpecialized2SmSm100
#define DEFINE_HINT_SM100_AUTO_2SM(N, T_M, T_N, T_K, C_X, C_Y, C_Z)                   \
struct Hint##N {                                                                       \
    using SmArch = cutlass::arch::Sm100;                                                   \
    using TileShape = cute::Shape<cute::T_M, cute::T_N, cute::T_K>;                        \
    using ClusterShape = cute::Shape<C_X, C_Y, cute::C_Z>;                                 \
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;          \
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized2SmSm100;                \
}

// Type D: EpilogueScheduleAuto + KernelTmaWarpSpecialized1SmSm100
#define DEFINE_HINT_SM100_AUTO_1SM(N, T_M, T_N, T_K, C_X, C_Y, C_Z)                   \
struct Hint##N {                                                                       \
    using SmArch = cutlass::arch::Sm100;                                                   \
    using TileShape = cute::Shape<cute::T_M, cute::T_N, cute::T_K>;                        \
    using ClusterShape = cute::Shape<C_X, C_Y, cute::C_Z>;                                 \
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;          \
    using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecialized1SmSm100;                \
}

// Hints 38-51: TmaWarpSpecialized2Sm + KernelTmaWarpSpecialized2SmSm100
//                            N   T_M   T_N   T_K      C_X      C_Y  C_Z
DEFINE_HINT_SM100_2SM(  38, _128, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_2SM(  39, _128, _128,  _64, cute::_2, cute::_1,  _1);
DEFINE_HINT_SM100_2SM(  40, _128, _128,  _64, cute::_2, cute::_2,  _1);
DEFINE_HINT_SM100_2SM(  41, _128, _128,  _64, cute::_4, cute::_1,  _1);
DEFINE_HINT_SM100_2SM(  42, _128, _128,  _64, cute::_2, cute::_4,  _1);
DEFINE_HINT_SM100_2SM(  43, _128, _128,  _64, cute::_4, cute::_2,  _1);
DEFINE_HINT_SM100_2SM(  44, _128, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_2SM(  45, _128, _256,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_2SM(  46, _128, _256,  _64, cute::_2, cute::_1,  _1);
DEFINE_HINT_SM100_2SM(  47, _128, _256,  _64, cute::_2, cute::_2,  _1);
DEFINE_HINT_SM100_2SM(  48, _128, _256,  _64, cute::_4, cute::_1,  _1);
DEFINE_HINT_SM100_2SM(  49, _128, _256,  _64, cute::_2, cute::_4,  _1);
DEFINE_HINT_SM100_2SM(  50, _128, _256,  _64, cute::_4, cute::_2,  _1);
DEFINE_HINT_SM100_2SM(  51, _128, _256,  _64, cute::_4, cute::_4,  _1);

// Hints 52-66: TmaWarpSpecialized1Sm + KernelTmaWarpSpecialized1SmSm100
//                            N   T_M   T_N   T_K      C_X      C_Y  C_Z
DEFINE_HINT_SM100_1SM(  52,  _64, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_1SM(  53,  _64, _128,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_1SM(  54,  _64, _128,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_1SM(  55,  _64, _128,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_1SM(  56,  _64, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_1SM(  57, _128, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_1SM(  58, _128, _128,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_1SM(  59, _128, _128,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_1SM(  60, _128, _128,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_1SM(  61, _128, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_1SM(  62, _128, _256,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_1SM(  63, _128, _256,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_1SM(  64, _128, _256,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_1SM(  65, _128, _256,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_1SM(  66, _128, _256,  _64, cute::_4, cute::_4,  _1);

// Hints 67-80: EpilogueScheduleAuto + KernelTmaWarpSpecialized2SmSm100
//                                N   T_M   T_N   T_K      C_X      C_Y  C_Z
DEFINE_HINT_SM100_AUTO_2SM( 67, _256, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 68, _256, _128,  _64, cute::_2, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 69, _256, _128,  _64, cute::_2, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 70, _256, _128,  _64, cute::_4, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 71, _256, _128,  _64, cute::_2, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 72, _256, _128,  _64, cute::_4, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 73, _256, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 74, _256, _256,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 75, _256, _256,  _64, cute::_2, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 76, _256, _256,  _64, cute::_2, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 77, _256, _256,  _64, cute::_4, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 78, _256, _256,  _64, cute::_2, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 79, _256, _256,  _64, cute::_4, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_2SM( 80, _256, _256,  _64, cute::_4, cute::_4,  _1);

// Hints 81-95: EpilogueScheduleAuto + KernelTmaWarpSpecialized1SmSm100
//                                N   T_M   T_N   T_K      C_X      C_Y  C_Z
DEFINE_HINT_SM100_AUTO_1SM( 81,  _64, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 82,  _64, _128,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 83,  _64, _128,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 84,  _64, _128,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 85,  _64, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 86, _128, _128,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 87, _128, _128,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 88, _128, _128,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 89, _128, _128,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 90, _128, _128,  _64, cute::_4, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 91, _128, _256,  _64,     int,     int,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 92, _128, _256,  _64, cute::_1, cute::_1,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 93, _128, _256,  _64, cute::_1, cute::_2,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 94, _128, _256,  _64, cute::_1, cute::_4,  _1);
DEFINE_HINT_SM100_AUTO_1SM( 95, _128, _256,  _64, cute::_4, cute::_4,  _1);
// clang-format on
