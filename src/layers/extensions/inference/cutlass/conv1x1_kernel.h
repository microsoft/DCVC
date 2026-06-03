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

#include "conv1x1_bias_wsilu_chunk_add_kernel.h"
#include "cutlass_epilogue.h"
#include "cutlass_kernel.h"
#include "../common_cu.h"
#include "../cutlass_helper_gemm.h"
#include "../def_cutlass.h"

template <typename Sm, bool WithBias, int NumShortcuts, bool WithQuant, bool WSiLU, bool ChunkAdd>
at::Tensor conv1x1_bias_generic_cutlass2(at::Tensor& out_buf, const at::Tensor& feature,
                                         const at::Tensor& weight, const at::optional<at::Tensor>& bias,
                                         const at::optional<at::Tensor>& shortcut1,
                                         const at::optional<at::Tensor>& shortcut2,
                                         const at::optional<at::Tensor>& quant)
{
    constexpr int NumAdditionalInput = NumShortcuts + (WithQuant ? 1 : 0);
    static_assert(NumAdditionalInput <= 2);
    using EpilogueOp =
        LinearCombination<NumAdditionalInput, WSiLU, WithQuant, ChunkAdd, cutlass::half_t,
                          cutlass::half_t, std::conditional_t<WSiLU, float, cutlass::half_t>,
                          128 / cutlass::sizeof_bits_v<cutlass::half_t>>;

    using Gemm = CustomGemmUniversalWithBroadcast<
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::layout::ColumnMajor,
        cutlass::half_t, cutlass::layout::RowMajor, cutlass::half_t, cutlass::arch::OpClassTensorOp,
        typename Sm::SmArch, typename Sm::ShapeThreadBlock, typename Sm::ShapeWarp, typename Sm::ShapeOp,
        EpilogueOp, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>, Sm::Stages>;

    const int B = feature.size(0);
    const int C1 = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C2 = weight.size(0);
    const int C1_ = weight.size(1);
    const int KH_ = weight.size(2);
    const int KW_ = weight.size(3);
    assert(C1 == C1_ && KH_ == 1 && KW_ == 1);
    const int M = H * W;
    const int N = C2;
    const int K = C1;
    const cutlass::gemm::GemmCoord problem_size(M, N, K);

    void* additional_input1_ptr = nullptr;
    void* additional_input2_ptr = nullptr;
    int additional_input1_stride = 0;
    int additional_input2_stride = 0;
    if constexpr (NumAdditionalInput == 1) {
        if constexpr (WithQuant) {
            additional_input1_ptr = quant.value().data_ptr();
            additional_input1_stride = 0;
        } else {
            additional_input1_ptr = shortcut1.value().data_ptr();
            additional_input1_stride = shortcut1.value().stride(3);
        }
    }
    if constexpr (NumAdditionalInput == 2) {
        additional_input1_ptr = shortcut1.value().data_ptr();
        additional_input1_stride = shortcut1.value().stride(3);
        if constexpr (WithQuant) {
            additional_input2_ptr = quant.value().data_ptr();
            additional_input2_stride = 0;
        } else {
            additional_input2_ptr = shortcut2.value().data_ptr();
            additional_input2_stride = shortcut2.value().stride(3);
        }
    }

    // Note: the following tensors may be part of a pre-allocated tensor of shape [B, C1 + C2, H, W], channels_last.
    typename Gemm::Arguments args;
    if constexpr (NumAdditionalInput < 2) {
        args = {
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,   // int batch_count
            {},  // args for epilogue
            feature.data_ptr(),
            weight.data_ptr(),
            additional_input1_ptr,
            ChunkAdd ? nullptr : out_buf.data_ptr(),
            WithBias ? bias.value().data_ptr() : nullptr,
            ChunkAdd ? out_buf.data_ptr() : nullptr,  // pointer of 2nd output
            0,                                        // int64_t batch_stride_A
            0,                                        // int64_t batch_stride_B
            0,                                        // int64_t batch_stride_C (shortcut)
            0,                                        // int64_t batch_stride_D
            0,                                        // int64_t batch_stride_Vector (bias)
            0,                                        // int64_t batch_stride_Tensor (2nd output)
            feature.stride(3),                        // typename LayoutA::Stride::Index lda
            weight.stride(3),                         // typename LayoutB::Stride::Index ldb
            additional_input1_stride,
            ChunkAdd ? 0 : out_buf.stride(3),     // typename LayoutC::Stride::Index ldd
            0,                                    // typename LayoutC::Stride::Index ldr (bias)
            ChunkAdd ? out_buf.stride(3) * 4 : 0  // typename LayoutC::Stride::Index ldt (2nd output)
        };
    } else if constexpr (NumAdditionalInput == 2) {
        args = {
            cutlass::gemm::GemmUniversalMode::kGemm,
            problem_size,
            1,   // int batch_count
            {},  // args for epilogue
            feature.data_ptr(),
            weight.data_ptr(),
            additional_input1_ptr,
            additional_input2_ptr,
            ChunkAdd ? nullptr : out_buf.data_ptr(),
            WithBias ? bias.value().data_ptr() : nullptr,
            ChunkAdd ? out_buf.data_ptr() : nullptr,  // pointer of 2nd output
            0,                                        // int64_t batch_stride_A
            0,                                        // int64_t batch_stride_B
            0,                                        // int64_t batch_stride_C1 (shortcut1)
            0,                                        // int64_t batch_stride_C2 (shortcut2)
            0,                                        // int64_t batch_stride_D
            0,                                        // int64_t batch_stride_Vector (bias)
            0,                                        // int64_t batch_stride_Tensor (2nd output)
            feature.stride(3),                        // typename LayoutA::Stride::Index lda
            weight.stride(3),                         // typename LayoutB::Stride::Index ldb
            additional_input1_stride,
            additional_input2_stride,
            ChunkAdd ? 0 : out_buf.stride(3),  // typename LayoutC::Stride::Index ldd
            0,                                 // typename LayoutC::Stride::Index ldr (bias)
            ChunkAdd ? out_buf.stride(3) : 0   // typename LayoutC::Stride::Index ldt (2nd output)
        };
    } else {
        assert(false);
    }

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM cannot implement: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return at::Tensor();
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    status = gemm_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return at::Tensor();
    }

    return out_buf;
}

#if CURRENT_DEVICE_SM == 90 || CURRENT_DEVICE_SM == 100
template <typename Sm, bool WithBias, int NumShortcuts, bool WithQuant, bool WSiLU, bool ChunkAdd>
at::Tensor conv1x1_bias_generic_cutlass3(at::Tensor& out_buf, const at::Tensor& feature,
                                         const at::Tensor& weight, const at::optional<at::Tensor>& bias,
                                         const at::optional<at::Tensor>& shortcut1,
                                         const at::optional<at::Tensor>& shortcut2,
                                         const at::optional<at::Tensor>& quant)
{
    static_assert(!ChunkAdd);
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementCompute = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    static constexpr int AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
    static constexpr int AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
    static constexpr int AlignmentC = 128 / cutlass::sizeof_bits_v<ElementC>;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;
    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

    using SmArch = typename Sm::SmArch;
    using TileShape = typename Sm::TileShape;
    using ClusterShape = typename Sm::ClusterShape;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueSchedule = typename Sm::EpilogueSchedule;
    using KernelSchedule = typename Sm::KernelSchedule;

    using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<LayoutC>;
    using EpilogueDescriptor = std::conditional_t<
        std::is_same_v<typename Sm::SmArch, cutlass::arch::Sm90>,
        cutlass::epilogue::collective::detail::EpilogueDescriptor<TileShape, EpilogueTileType, ElementC, ElementD, EpilogueSchedule>,
        cutlass::epilogue::collective::detail::Sm100EpilogueDescriptor<
            cutlass::arch::OpClassTensorOp, TileShape, EpilogueTileType, ElementCompute, ElementC,
            ElementD, EpilogueSchedule, GmemStrideTypeC, GmemStrideTypeC, false, false>>;
    using AuxLoadDescriptor = std::conditional_t<
        std::is_same_v<typename Sm::SmArch, cutlass::arch::Sm90>,
        cutlass::epilogue::collective::detail::AuxLoadDescriptor<EpilogueDescriptor, LayoutC, ElementC>,
        cutlass::epilogue::collective::detail::Sm100AuxLoadDescriptor<EpilogueDescriptor, LayoutC, ElementC>>;
    using AuxLoad = cutlass::epilogue::fusion::Sm90AuxLoad<
        AuxLoadDescriptor::Stages, typename AuxLoadDescriptor::EpilogueTile,
        typename AuxLoadDescriptor::Element, typename AuxLoadDescriptor::Stride,
        typename AuxLoadDescriptor::SmemLayoutAtom, typename AuxLoadDescriptor::CopyOpS2R>;

    using AccEVT = cutlass::epilogue::fusion::Sm90AccFetch;

    using BiasEVT = std::conditional_t<
        WithBias,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementC, ElementCompute, RoundStyle>,  // plus
            cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementC, ElementCompute>,  // bias
            AccEVT  // acc
            >,
        AccEVT>;

    using Shortcut1EVT = std::conditional_t<
        NumShortcuts >= 1,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementC, ElementCompute, RoundStyle>,  // plus
            cutlass::epilogue::fusion::Sm90SrcFetch<ElementC>,  // shortcut1
            BiasEVT                                             // acc + bias
            >,
        BiasEVT>;

    using Shortcut2EVT = std::conditional_t<
        NumShortcuts == 2,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementC, ElementCompute, RoundStyle>,  // plus
            AuxLoad,      // shortcut2
            Shortcut1EVT  // acc + bias + shortcut1
            >,
        Shortcut1EVT>;

    using QuantEVT = std::conditional_t<
        WithQuant,
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, ElementC, ElementCompute, RoundStyle>,  // multiply
            cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementC, ElementCompute>,  // quant
            Shortcut2EVT  // acc + bias + shortcut1 + shortcut2
            >,
        Shortcut2EVT>;

    using FinalEVT = std::conditional_t<
        WSiLU,
        cutlass::epilogue::fusion::Sm90EVT<cutlass::epilogue::fusion::Sm90Compute<WSiLUOp, ElementC, float, RoundStyle>,  // wsilu
                                           QuantEVT  // previous
                                           >,
        QuantEVT>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        SmArch, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, EpilogueTileType,
        ElementCompute, ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutC,
        AlignmentD, EpilogueSchedule, FinalEVT>::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        SmArch, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB,
        AlignmentB, ElementCompute, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
            sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;

    using GemmKernel =
        cutlass::gemm::kernel::GemmUniversal<cute::Shape<int, int, int, int>,  // Indicates ProblemShape
                                             CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    const int B = feature.size(0);
    const int C1 = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C2 = weight.size(0);
    const int C1_ = weight.size(1);
    const int KH_ = weight.size(2);
    const int KW_ = weight.size(3);
    assert(C1 == C1_ && KH_ == 1 && KW_ == 1);
    const int M = H * W;
    const int N = C2;
    const int K = C1;
    const typename GemmKernel::ProblemShape problem_size{ M, N, K, 1 };

    // Note: the following tensors may be part of a pre-allocated tensor of shape [B, C1 + C2, H, W], channels_last.
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        { static_cast<ElementA*>(feature.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ feature.stride(3), cute::_1{}, 0 },
          static_cast<ElementB*>(weight.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ weight.stride(3), cute::_1{}, 0 } },
        { {},  // epilogue arguments, updated later
          shortcut1.has_value() ? static_cast<ElementC*>(shortcut1.value().data_ptr()) : nullptr,
          cute::Stride<int64_t, cute::_1, int64_t>{
              shortcut1.has_value() ? shortcut1.value().stride(3) : 0, cute::_1{}, 0 },
          static_cast<ElementD*>(out_buf.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ out_buf.stride(3), cute::_1{}, 0 } },
    };

    typename AccEVT::Arguments acc_args = {};
    typename BiasEVT::Arguments bias_args;

    if constexpr (WithBias) {
        bias_args = {
            { static_cast<ElementC*>(bias.value().data_ptr()) },  // bias
            acc_args,                                             // acc
            {}                                                    // plus
        };
    } else {
        bias_args = acc_args;
    }

    typename Shortcut1EVT::Arguments shortcut1_args;

    if constexpr (NumShortcuts >= 1) {
        shortcut1_args = {
            {},         // shortcut1
            bias_args,  // acc + bias
            {}          // plus
        };
    } else {
        shortcut1_args = bias_args;
    }

    typename Shortcut2EVT::Arguments shortcut2_args;

    if constexpr (NumShortcuts == 2) {
        shortcut2_args = {
            {
                static_cast<ElementC*>(shortcut2.value().data_ptr()),  // ptr
                static_cast<ElementC>(0),                              // null_default
                cute::Stride<int64_t, cute::_1, int64_t>{ shortcut2.value().stride(3), cute::_1{}, 0 }  // stride
            },               // shortcut2
            shortcut1_args,  // acc + bias + shortcut1
            {}               // plus
        };
    } else {
        shortcut2_args = shortcut1_args;
    }

    typename QuantEVT::Arguments quant_args;

    if constexpr (WithQuant) {
        quant_args = {
            {
                static_cast<ElementC*>(quant.value().data_ptr()),  // ptr
                static_cast<ElementC>(1)                           // null_default
            },                                                     // quant
            shortcut2_args,  // acc + bias + shortcut1 + shortcut2
            {}               // multiply
        };
    } else {
        quant_args = shortcut2_args;
    }

    typename FinalEVT::Arguments final_args;

    if constexpr (WSiLU) {
        final_args = {
            quant_args,  // previous
            {}           // wsilu
        };
    } else {
        final_args = quant_args;
    }

    args.epilogue.thread = final_args;

    args.scheduler.max_swizzle_size = 1;
    args.scheduler.raster_order =
        cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions::Heuristic;

    Gemm gemm_op;
    cutlass::Status status = gemm_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM cannot implement: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return at::Tensor();
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    status = gemm_op(args, nullptr, stream, /* cuda_adapter */ nullptr, /* launch_with_pdl */ true);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return at::Tensor();
    }

    return out_buf;
}
#endif

template <typename Sm, bool WithBias = true, int NumShortcuts = 0, bool WithQuant = false,
          bool WSiLU = false, bool ChunkAdd = false>
at::Tensor conv1x1_bias_generic_cutlass(at::Tensor& out_buf, const at::Tensor& feature,
                                        const at::Tensor& weight,
                                        const at::optional<at::Tensor>& bias = at::nullopt,
                                        const at::optional<at::Tensor>& shortcut1 = at::nullopt,
                                        const at::optional<at::Tensor>& shortcut2 = at::nullopt,
                                        const at::optional<at::Tensor>& quant = at::nullopt,
                                        const at::optional<at::Tensor>& weight0 = at::nullopt,
                                        const at::optional<at::Tensor>& weight1 = at::nullopt,
                                        const at::optional<at::Tensor>& weight2 = at::nullopt,
                                        const at::optional<at::Tensor>& weight3 = at::nullopt,
                                        const at::optional<at::Tensor>& bias0 = at::nullopt,
                                        const at::optional<at::Tensor>& bias1 = at::nullopt,
                                        const at::optional<at::Tensor>& bias2 = at::nullopt,
                                        const at::optional<at::Tensor>& bias3 = at::nullopt)
{
    if constexpr (std::is_same_v<typename Sm::SmArch, cutlass::arch::Sm90>) {
#if CURRENT_DEVICE_SM == 90
        if constexpr (ChunkAdd) {
            return conv1x1_bias_wsilu_chunk_add_generic_cutlass3<Sm>(
                out_buf, feature, weight0.value(), weight1.value(), weight2.value(),
                weight3.value(), bias0.value(), bias1.value(), bias2.value(), bias3.value());
        } else {
            return conv1x1_bias_generic_cutlass3<Sm, WithBias, NumShortcuts, WithQuant, WSiLU, ChunkAdd>(
                out_buf, feature, weight, bias, shortcut1, shortcut2, quant);
        }
#else
        return at::Tensor();
#endif
    } else if constexpr (std::is_same_v<typename Sm::SmArch, cutlass::arch::Sm100>) {
#if CURRENT_DEVICE_SM == 100
        if constexpr (ChunkAdd) {
            return at::Tensor();
        } else {
            return conv1x1_bias_generic_cutlass3<Sm, WithBias, NumShortcuts, WithQuant, WSiLU, ChunkAdd>(
                out_buf, feature, weight, bias, shortcut1, shortcut2, quant);
        }
#else
        return at::Tensor();
#endif
    } else {
        return conv1x1_bias_generic_cutlass2<Sm, WithBias, NumShortcuts, WithQuant, WSiLU, ChunkAdd>(
            out_buf, feature, weight, bias, shortcut1, shortcut2, quant);
    }
}
