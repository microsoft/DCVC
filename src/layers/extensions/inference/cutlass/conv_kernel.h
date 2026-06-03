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
at::Tensor conv_bias_generic_cutlass(at::Tensor& out_buf, const at::Tensor& feature,
                                     const at::Tensor& weight, const at::Tensor& bias, const int stride)
{
    using EpilogueOp =
        LinearCombination<0, false, false, false, cutlass::half_t, cutlass::half_t, cutlass::half_t,
                          128 / cutlass::sizeof_bits_v<cutlass::half_t>>;

    using Conv2dFpropKernel = typename CustomDefaultConv2dFpropWithBroadcast<
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::arch::OpClassTensorOp,
        typename Sm::SmArch, typename Sm::ShapeThreadBlock, typename Sm::ShapeWarp, typename Sm::ShapeOp,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Sm::Stages,
        cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kOptimized>::Kernel;

    using ImplicitGemm = CustomImplicitGemmConvolution<Conv2dFpropKernel>;

    const int B = feature.size(0);
    const int Cin = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int Cout = weight.size(0);
    const int Cin_ = weight.size(1);
    const int kernel = weight.size(2);
    const int kernel_ = weight.size(3);
    assert(Cin == Cin_ && kernel == kernel_);
    const int pad = (kernel - 1) / 2;

    cutlass::conv::Conv2dProblemSize problem_size({ B, H, W, Cin },                     // input
                                                  { Cout, kernel, kernel, Cin },        // kernel
                                                  { pad, pad, pad, pad },               // pad
                                                  { stride, stride },                   // stride
                                                  { 1, 1 },                             // dilation
                                                  { B, H / stride, W / stride, Cout },  // out
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

    typename ImplicitGemm::Arguments args(problem_size, d_feature, d_weight, d_out, d_out, {},
                                          cutlass::conv::SplitKMode::kSerial, bias.data_ptr(),
                                          nullptr, 0, 0);

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
