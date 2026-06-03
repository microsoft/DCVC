// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/util/packed_stride.hpp>
// clang-format on

#include "cutlass_kernel.h"
#include "../common_cu.h"
#include "sm90_quad_gemm/sm90_builder_epilogue.h"
#include "sm90_quad_gemm/sm90_builder_mainloop.h"
#include "sm90_quad_gemm/sm90_gemm_tma_warpspecialized_cooperative_quad.h"

#if CURRENT_DEVICE_SM == 90
template <typename Sm>
at::Tensor
conv1x1_bias_wsilu_chunk_add_generic_cutlass3(at::Tensor& out_buf, const at::Tensor& feature,
                                              const at::Tensor& weight0, const at::Tensor& weight1,
                                              const at::Tensor& weight2, const at::Tensor& weight3,
                                              const at::Tensor& bias0, const at::Tensor& bias1,
                                              const at::Tensor& bias2, const at::Tensor& bias3)
{
    if constexpr (std::is_same_v<std::tuple_element_t<1, typename Sm::TileShape>, cute::_256>) {
        std::cerr << "conv1x1_bias_wsilu_chunk_add_generic_cutlass3: The second value of "
                     "Sm::TileShape cannot be cute::_256!"
                  << std::endl;
        return at::Tensor();
    }

    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementD = cutlass::half_t;
    using ElementCompute = cutlass::half_t;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::RowMajor;

    constexpr int AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>;
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>;
    constexpr int AlignmentC = 128 / cutlass::sizeof_bits_v<ElementC>;
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits_v<ElementD>;
    static constexpr auto RoundStyle = cutlass::FloatRoundStyle::round_to_nearest;

    using SmArch = typename Sm::SmArch;
    using TileShape = typename Sm::TileShape;
    using ClusterShape = typename Sm::ClusterShape;
    using EpilogueTileType = cutlass::epilogue::collective::EpilogueTileAuto;
    using EpilogueSchedule = typename Sm::EpilogueSchedule;
    using KernelSchedule = typename Sm::KernelSchedule;

    using FinalEVT = cutlass::epilogue::fusion::Sm90EVT<
        cutlass::epilogue::fusion::Sm90Compute<WSiLUOp, ElementD, float, RoundStyle>,  // wsilu
        cutlass::epilogue::fusion::Sm90EVT<
            cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, ElementC, ElementCompute, RoundStyle>,  // plus
            cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, ElementC, ElementCompute>,  // bias
            cutlass::epilogue::fusion::Sm90AccFetch  // acc
            >>;

    using CollectiveEpilogue = typename QuadCollectiveBuilderEpilogue<
        SmArch, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, EpilogueTileType,
        ElementCompute, ElementCompute, ElementC, LayoutC, AlignmentC, ElementD, LayoutC,
        AlignmentD, EpilogueSchedule, FinalEVT>::CollectiveOp;

    using CollectiveMainloop = typename QuadCollectiveBuilderMainloop<
        SmArch, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB,
        AlignmentB, ElementCompute, TileShape, ClusterShape,
        cutlass::gemm::collective::StageCount<2>, KernelSchedule>::CollectiveOp;

    using GemmKernel = QuadGemmUniversal<cute::Shape<int, int, int, int>,  // Indicates ProblemShape
                                         CollectiveMainloop, CollectiveEpilogue>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

    const int B = feature.size(0);
    const int C1 = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C2 = weight0.size(0);
    const int C1_ = weight0.size(1);
    const int KH_ = weight0.size(2);
    const int KW_ = weight0.size(3);
    assert(C1 == C1_ && KH_ == 1 && KW_ == 1);
    assert(weight0.sizes() == weight1.sizes());
    assert(weight0.sizes() == weight2.sizes());
    assert(weight0.sizes() == weight3.sizes());
    assert(bias0.sizes() == bias1.sizes());
    assert(bias0.sizes() == bias2.sizes());
    assert(bias0.sizes() == bias3.sizes());
    const int M = H * W;
    const int N = C2;
    const int K = C1;
    const typename GemmKernel::ProblemShape problem_size{ M, N, K, 1 };

    // Note: It is assumed that the 4 weights are interleaved splits of a contiguous weight.
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem_size,
        { static_cast<ElementA*>(feature.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ feature.stride(3), cute::_1{}, 0 },
          static_cast<ElementB*>(weight0.data_ptr()), static_cast<ElementB*>(weight1.data_ptr()),
          static_cast<ElementB*>(weight2.data_ptr()), static_cast<ElementB*>(weight3.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ weight0.stride(3) * 4, cute::_1{}, 0 } },
        { {},
          {},
          {},
          {},  // epilogue arguments, updated later
          nullptr,
          cute::Stride<int64_t, cute::_1, int64_t>{ 0, cute::_1{}, 0 },
          static_cast<ElementD*>(out_buf.data_ptr()),
          cute::Stride<int64_t, cute::_1, int64_t>{ out_buf.stride(3), cute::_1{}, 0 } },
    };

    args.epilogue.thread0 = {
        {
            { static_cast<ElementC*>(bias0.data_ptr()) },  // bias
            {},                                            // acc
            {}                                             // plus
        },
        {}  // WSiLU
    };
    args.epilogue.thread1 = {
        {
            { static_cast<ElementC*>(bias1.data_ptr()) },  // bias
            {},                                            // acc
            {}                                             // plus
        },
        {}  // WSiLU
    };
    args.epilogue.thread2 = {
        {
            { static_cast<ElementC*>(bias2.data_ptr()) },  // bias
            {},                                            // acc
            {}                                             // plus
        },
        {}  // WSiLU
    };
    args.epilogue.thread3 = {
        {
            { static_cast<ElementC*>(bias3.data_ptr()) },  // bias
            {},                                            // acc
            {}                                             // plus
        },
        {}  // WSiLU
    };

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
