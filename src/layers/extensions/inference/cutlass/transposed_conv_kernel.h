// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/tile_scheduler_params.h>
#include <cutlass/util/packed_stride.hpp>
// clang-format on

#include "cutlass_epilogue.h"
#include "cutlass_kernel.h"
#include "../common_cu.h"
#include "../cutlass_helper_implicit_gemm.h"
#include "../def_cutlass.h"

template <typename Sm>
at::Tensor transposed_conv_generic_cutlass(at::Tensor& out_buf, const at::Tensor& feature,
                                           const at::Tensor& weight, const int stride)
{
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t, 128 / cutlass::sizeof_bits_v<cutlass::half_t>, cutlass::half_t,
        cutlass::half_t, cutlass::epilogue::thread::ScaleType::Nothing>;

    using Conv2dDgradKernel = typename CustomDefaultConv2dDgrad<
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::arch::OpClassTensorOp,
        typename Sm::SmArch, typename Sm::ShapeThreadBlock, typename Sm::ShapeWarp, typename Sm::ShapeOp,
        EpilogueOp, cutlass::conv::threadblock::StridedDgradIdentityThreadblockSwizzle<8>, Sm::Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

    using ImplicitGemm = CustomImplicitGemmConvolution<Conv2dDgradKernel>;

    const int B = feature.size(0);
    const int Cin = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int Cin_ = weight.size(0);
    const int Cout = weight.size(1);
    const int kernel = weight.size(2);
    const int kernel_ = weight.size(3);
    assert(Cin == Cin_ && kernel == kernel_ && kernel == stride);

    cutlass::conv::Conv2dProblemSize problem_size({ B, H * stride, W * stride, Cout },  // out
                                                  { Cin, kernel, kernel, Cout },        // kernel
                                                  { 0, 0, 0, 0 },                       // pad
                                                  { stride, stride },                   // stride
                                                  { 1, 1 },                             // dilation
                                                  { B, H, W, Cin },                     // input
                                                  cutlass::conv::Mode::kCrossCorrelation,
                                                  1  // split_k_slices
    );

    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> d_feature(
        static_cast<cutlass::half_t*>(feature.data_ptr()),
        cutlass::make_Coord(feature.stride(3), feature.stride(2), feature.stride(0)));
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> d_weight(
        static_cast<cutlass::half_t*>(weight.data_ptr()),
        cutlass::make_Coord(weight.stride(3), weight.stride(2), weight.stride(0)));
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::TensorNHWC> d_out(
        static_cast<cutlass::half_t*>(out_buf.data_ptr()),
        cutlass::make_Coord(out_buf.stride(3), out_buf.stride(2), out_buf.stride(0)));

    typename ImplicitGemm::Arguments args(problem_size, d_feature, d_weight, d_out, d_out, {});

    ImplicitGemm implicit_gemm_op;
    cutlass::Status status = implicit_gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to configure convolution operation." << std::endl;
        return at::Tensor();
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    status = implicit_gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run convolution operation." << std::endl;
        return at::Tensor();
    }

    return out_buf;
}
