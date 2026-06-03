// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
// clang-format on

#include "sm90_epilogue_tma_warpspecialized_quad.h"

/**
 * These 2 classes/structs are rewritten:
 *   1. cutlass::epilogue::collective::detail::Sm90TmaBuilderImpl -> QuadSm90TmaBuilderImpl
 *   2. cutlass::epilogue::collective::CollectiveBuilder -> QuadCollectiveBuilderEpilogue
 */

// Original file: cutlass/epilogue/collective/builders/sm90_builder.inl
// to define QuadCollectiveEpilogue as CollectiveOp
// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <class TileShape_MNK, class EpilogueTile_MN, class ElementAccumulator, class ElementCompute,
          class ElementC_, class GmemLayoutTagC_, int AlignmentC, class ElementD_,
          class GmemLayoutTagD, int AlignmentD, class FusionOpOrCallbacks, class DispatchPolicy>
struct QuadSm90TmaBuilderImpl {
    // C/D should meet TMA alignment requirement if not void
    static_assert(
        cutlass::epilogue::collective::detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
        "C/D Should meet TMA alignment requirement\n");
    // Passing void D disables destination store + smem allocation
    using ElementD =
        cute::conditional_t<cute::is_void_v<ElementD_>,
                            cutlass::epilogue::fusion::get_element_aux_t<FusionOpOrCallbacks>, ElementD_>;

    // Passing void C disables source load + smem allocation
    using ElementC =
        cute::conditional_t<cute::is_void_v<ElementC_>, ElementD, ElementC_>;  // prevents void ref breakages
    using GmemLayoutTagC =
        cute::conditional_t<cute::is_void_v<ElementC_>, GmemLayoutTagD, GmemLayoutTagC_>;

    using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
    using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;

    using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
    using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

    using CopyOpS2G =
        cute::conditional_t<cutlass::epilogue::collective::detail::is_im2col_mode<GmemLayoutTagD>,
                            cute::SM90_TMA_STORE_IM2COL, cute::SM90_TMA_STORE>;
    using CopyOpG2S =
        cute::conditional_t<cutlass::epilogue::collective::detail::is_im2col_mode<GmemLayoutTagC>,
                            cute::SM90_TMA_LOAD_IM2COL, cute::SM90_TMA_LOAD>;

    // Get the smallest tiled copy we can use to retile the accumulators
    // using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
    using CopyAtomC = cute::conditional_t<
        cute::size<1>(EpilogueTile_MN{}) % 16 == 0, cute::Copy_Atom<cute::SM90_U32x4_STSM_N, cutlass::half_t>,
        cute::conditional_t<cute::size<1>(EpilogueTile_MN{}) % 8 == 0,
                            cute::Copy_Atom<cute::SM90_U32x2_STSM_N, cutlass::half_t>, void>>;
    static_assert(!cute::is_same_v<CopyAtomC, void>,
                  "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");
    // Get register to register tiled copy that happen before shared memory store.
    // Apply void as no register transform op needed currently.
    using CopyOpR2R = void;

    // TMA builder allows for passing callbacks directly, which is either a cutlass::epilogue::fusion::FusionCallbacks
    // instance or a direct visitor implementation, e.g. cutlass::epilogue::fusion::Sm90LinearCombination
    using FusionCallbacks = typename cutlass::epilogue::collective::detail::CallbacksBuilder<
        DispatchPolicy, FusionOpOrCallbacks, TileShape_MNK, EpilogueTile_MN, ElementAccumulator>::Callbacks;

    using CollectiveOp = QuadCollectiveEpilogue<
        DispatchPolicy, TileShape_MNK, EpilogueTile_MN,
        ElementC_,  // Need to pass void through to expose via QuadGemmUniversal
        GmemStrideTypeC, ElementD_, GmemStrideTypeD, FusionCallbacks, CopyOpG2S,
        decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
                 UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
        decltype(cutlass::epilogue::collective::detail::sm90_get_smem_load_op_for_source<
                 UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
        CopyOpS2G,
        decltype(cutlass::epilogue::collective::detail::sm90_get_epilogue_smem_swizzle_layout_atom<
                 UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
        decltype(cutlass::epilogue::collective::detail::sm90_get_smem_store_op_for_accumulator<
                 UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
        CopyAtomC, CopyOpR2R>;
};

// Primary template declaration to enable partial/specialized definitions below.
template <class ArchTag, class OpClass, class TileShape_MNK, class ClusterShape_MNK, class EpilogueTileType,
          class ElementAccumulator, class ElementCompute, class ElementC, class GmemLayoutTagC,
          int AlignmentC, class ElementD, class GmemLayoutTagD, int AlignmentD, class EpilogueScheduleType,
          class FusionOpOrCallbacks =
              cutlass::epilogue::fusion::LinearCombination<ElementD, ElementCompute, ElementC, ElementCompute>,
          class Enable = void>
struct QuadCollectiveBuilderEpilogue {
    static_assert(cutlass::detail::dependent_false<ArchTag>,
                  "Could not build a collective epilogue for given parameters.");
};

// Original file: cutlass/epilogue/collective/builders/sm90_builder.inl
// to define QuadSm90TmaBuilderImpl as CollectiveOp
// Tma warp-specialized builder
template <class OpClass, class TileShape_MNK, class ClusterShape_MNK, class EpilogueTileType,
          class ElementAccumulator, class ElementCompute, class ElementC, class GmemLayoutTagC, int AlignmentC,
          class ElementD_, class GmemLayoutTagD, int AlignmentD, class Schedule, class FusionOperation>
struct QuadCollectiveBuilderEpilogue<
    cutlass::arch::Sm90, OpClass, TileShape_MNK, ClusterShape_MNK, EpilogueTileType, ElementAccumulator, ElementCompute,
    ElementC, GmemLayoutTagC, AlignmentC, ElementD_, GmemLayoutTagD, AlignmentD, Schedule, FusionOperation,
    cute::enable_if_t<cute::is_any_of_v<Schedule, cutlass::epilogue::TmaWarpSpecialized,
                                        cutlass::epilogue::TmaWarpSpecializedCooperative>>> {
private:
    using ElementD =
        cute::conditional_t<cute::is_void_v<ElementD_>,
                            cutlass::epilogue::fusion::get_element_aux_t<FusionOperation>, ElementD_>;
    using EpilogueTile_MN =
        decltype(cutlass::epilogue::collective::detail::sm90_compute_tile_shape_or_override<
                 ElementD, EpilogueTileType, Schedule, TileShape_MNK>());
    using DispatchPolicy = decltype(cutlass::epilogue::collective::detail::sm90_get_tma_dispatch_policy<
                                    TileShape_MNK, EpilogueTile_MN, ElementC, ElementD, Schedule>());

public:
    using CollectiveOp =
        typename QuadSm90TmaBuilderImpl<TileShape_MNK, EpilogueTile_MN, ElementAccumulator, ElementCompute,
                                        ElementC, GmemLayoutTagC, AlignmentC, ElementD_, GmemLayoutTagD,
                                        AlignmentD, FusionOperation, DispatchPolicy>::CollectiveOp;
};
