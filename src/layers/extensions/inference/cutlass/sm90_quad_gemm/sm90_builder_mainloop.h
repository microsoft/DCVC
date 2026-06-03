// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>
// clang-format on

#include "sm90_mma_tma_gmma_ss_warpspecialized_quad.h"

/**
 * These 1 classes/structs are rewritten:
 *   1. cutlass::gemm::collective::CollectiveBuilder -> QuadCollectiveBuilderMainloop
 */

// SM90 Collective Builders should be used only starting CUDA 12.0
#if (__CUDACC_VER_MAJOR__ >= 12)
    #define CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
#endif

// Primary template declaration to enable partial/specialized definitions below.
template <class ArchTag, class OpClass, class ElementA, class GmemLayoutA, int AlignmentA, class ElementB,
          class GmemLayoutB, int AlignmentB, class ElementAccumulator, class TileShape_MNK,
          class ClusterShape_MNK, class StageCountType, class KernelScheduleType, class Enable = void>
struct QuadCollectiveBuilderMainloop {
    static_assert(sizeof(ElementA) == 0,
                  "QuadCollectiveBuilderMainloop: unsupported configuration.");
};

// Original file: cutlass/gemm/collective/builders/sm90_gmma_builder.inl
// to define QuadCollectiveMma as CollectiveOp
// GMMA_TMA_WS_SS
template <class ElementA, class GmemLayoutATag, int AlignmentA, class ElementB,
          class GmemLayoutBTag, int AlignmentB, class ElementAccumulator, class TileShape_MNK,
          class ClusterShape_MNK, class StageCountType, class KernelScheduleType>
struct QuadCollectiveBuilderMainloop<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, GmemLayoutATag, AlignmentA, ElementB,
    GmemLayoutBTag, AlignmentB, ElementAccumulator, TileShape_MNK, ClusterShape_MNK, StageCountType, KernelScheduleType,
    cute::enable_if_t<(cute::is_same_v<KernelScheduleType, cutlass::gemm::KernelTmaWarpSpecializedCooperative>)
                      && not cutlass::gemm::collective::detail::is_use_rmem_A<
                          ElementA, GmemLayoutATag, ElementB, GmemLayoutBTag>()>> {
    static_assert(cute::is_static<TileShape_MNK>::value);
    static_assert(cute::is_static<ClusterShape_MNK>::value);
#ifndef CUTLASS_SM90_COLLECTIVE_BUILDER_SUPPORTED
    static_assert(cutlass::detail::dependent_false<ElementA>,
                  "Unsupported Toolkit for SM90 Collective Builder\n");
#endif
    static_assert(
        cutlass::gemm::collective::detail::is_aligned<ElementA, AlignmentA, ElementB, AlignmentB,
                                                      cutlass::gemm::collective::detail::tma_alignment_bytes>(),
        "Should meet TMA alignment requirement\n");

    using ElementAMma =
        cute::conditional_t<cute::is_same_v<ElementA, float>, cutlass::tfloat32_t, ElementA>;
    using ElementBMma =
        cute::conditional_t<cute::is_same_v<ElementB, float>, cutlass::tfloat32_t, ElementB>;

    static constexpr cute::GMMA::Major GmmaMajorA =
        cutlass::gemm::collective::detail::gmma_ss_tag_to_major_A<ElementAMma, GmemLayoutATag>();
    static constexpr cute::GMMA::Major GmmaMajorB =
        cutlass::gemm::collective::detail::gmma_ss_tag_to_major_B<ElementBMma, GmemLayoutBTag>();

    static constexpr bool IsCooperative =
        cute::is_same_v<KernelScheduleType, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using AtomLayoutMNK =
        cute::conditional_t<IsCooperative, cute::Layout<cute::Shape<cute::_2, cute::_1, cute::_1>>,
                            cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>>;

    using TiledMma = decltype(cute::make_tiled_mma(
        cute::GMMA::ss_op_selector<ElementAMma, ElementBMma, ElementAccumulator, TileShape_MNK,
                                   GmmaMajorA, GmmaMajorB>(),
        AtomLayoutMNK{}));

    using GmemTiledCopyA = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(
        cute::shape<1>(ClusterShape_MNK{})));
    using GmemTiledCopyB = decltype(cutlass::gemm::collective::detail::sm90_cluster_shape_to_tma_atom(
        cute::shape<0>(ClusterShape_MNK{})));

    using SmemLayoutAtomA = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                     GmmaMajorA, ElementAMma, decltype(cute::get<0>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{}))>());
    using SmemLayoutAtomB = decltype(cutlass::gemm::collective::detail::ss_smem_selector<
                                     GmmaMajorB, ElementBMma, decltype(cute::get<1>(TileShape_MNK{})),
                                     decltype(cute::get<2>(TileShape_MNK{}))>());

    static constexpr size_t TensorMapStorage = 0;
    static constexpr size_t SchedulerPipelineStorage =
        cute::is_pointer_v<cutlass::detail::TagToStrideA_t<GmemLayoutATag>>
            ? sizeof(cutlass::PipelineDetail::PipelineAsyncSharedStorage<8>)
            : 0;
    static constexpr int KernelSmemCarveout =
        static_cast<int>(TensorMapStorage + SchedulerPipelineStorage);
    static constexpr int Sm90ReducedSmemCapacityBytes =
        cutlass::gemm::collective::detail::sm90_smem_capacity_bytes - KernelSmemCarveout;

    static constexpr int PipelineStages =
        cutlass::gemm::collective::detail::compute_stage_count_or_override<
            Sm90ReducedSmemCapacityBytes, ElementAMma, ElementBMma, TileShape_MNK>(StageCountType{});

    using DispatchPolicy =
        cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<PipelineStages, ClusterShape_MNK, KernelScheduleType>;

    using SmemCopyAtomA = void;
    using SmemCopyAtomB = void;

    using CollectiveOp = QuadCollectiveMma<
        DispatchPolicy, TileShape_MNK, ElementA, cutlass::detail::TagToStrideA_t<GmemLayoutATag>, ElementB,
        cutlass::detail::TagToStrideB_t<GmemLayoutBTag>, TiledMma, GmemTiledCopyA, SmemLayoutAtomA,
        SmemCopyAtomA, cute::identity, GmemTiledCopyB, SmemLayoutAtomB, SmemCopyAtomB, cute::identity>;
};
