// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/epilogue/dispatch_policy.hpp>
// clang-format on

/**
 * These 1 classes/structs are rewritten:
 *   1. cutlass::epilogue::collective::CollectiveEpilogue -> QuadCollectiveEpilogue
 */

template <typename T>
struct QuadAddOp;

template <typename T, int N>
struct QuadAddOp<cutlass::Array<T, N>> {
    static const bool kIsHeavy = true;

    CUTLASS_HOST_DEVICE
    cutlass::Array<T, N> operator()(cutlass::Array<T, N> const& accum0,
                                    cutlass::Array<T, N> const& accum1,
                                    cutlass::Array<T, N> const& accum2,
                                    cutlass::Array<T, N> const& accum3) const
    {
        cutlass::plus<cutlass::Array<T, N>> plus_op;
        return plus_op(plus_op(plus_op(accum0, accum1), accum2), accum3);
    }
};

// Primary template declaration to enable partial/specialized definitions below.
template <class DispatchPolicy, class... Args>
class QuadCollectiveEpilogue {
    static_assert(cutlass::detail::dependent_false<DispatchPolicy>,
                  "Could not find an epilogue specialization.");
};

// Original file: cutlass/epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp
// to define QuadCollectiveEpilogue as CollectiveOp
template <int StagesC_, int StagesD_, int FragmentSize_, bool ReuseSmemC_, bool DelayTmaStore_,
          class CtaTileMNK_,    // (CTA_M,CTA_N,CTA_K)
          class EpilogueTile_,  // (EPI_TILE_M,EPI_TILE_N)
          class ElementC_, class StrideC_, class ElementD_, class StrideD_, class FusionCallbacks_,
          class CopyOpG2S_, class SmemLayoutAtomC_, class CopyOpS2R_, class CopyOpS2G_,
          class SmemLayoutAtomD_, class CopyOpR2S_, class CopyAtomC_, class CopyOpR2R_>
class QuadCollectiveEpilogue<
    cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>,
    CtaTileMNK_, EpilogueTile_, ElementC_, StrideC_, ElementD_, StrideD_, FusionCallbacks_, CopyOpG2S_,
    SmemLayoutAtomC_, CopyOpS2R_, CopyOpS2G_, SmemLayoutAtomD_, CopyOpR2S_, CopyAtomC_, CopyOpR2R_> {
public:
    //
    // Type Aliases
    //
    using DispatchPolicy =
        cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC_, StagesD_, FragmentSize_, ReuseSmemC_, DelayTmaStore_>;
    using CtaTileMNK = CtaTileMNK_;
    using EpilogueTile = EpilogueTile_;
    using FusionCallbacks = FusionCallbacks_;
    using ElementC = ElementC_;
    using StrideC = StrideC_;
    using ElementD = ElementD_;
    using StrideD = StrideD_;
    using CopyOpG2S = CopyOpG2S_;
    using SmemLayoutAtomC = SmemLayoutAtomC_;
    using CopyOpS2R = CopyOpS2R_;
    using CopyOpS2G = CopyOpS2G_;
    using SmemLayoutAtomD = SmemLayoutAtomD_;
    using CopyOpR2S = CopyOpR2S_;
    using CopyAtomC = CopyAtomC_;
    using CopyOpR2R = CopyOpR2R_;

    using ThreadEpilogueOp =
        typename cutlass::epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::Operation;
    using GmemTiledCopyC = CopyOpG2S;
    using GmemTiledCopyD = CopyOpS2G;

    static_assert(!cute::is_layout<EpilogueTile>::value && cutlass::is_tuple<EpilogueTile>::value,
                  "EpilogueTile must be a cute::Tile or cute::Shape");
    static_assert(cute::rank(CtaTileMNK{}) == 3,
                  "CtaTileMNK must be rank-3: [CTA_M, CTA_N, CTA_K]");
    static_assert(cute::rank(EpilogueTile{}) == 2,
                  "EpilogueTile must be rank-2: [EPI_TILE_M, EPI_TILE_N]");
    static_assert(cute::size<0>(CtaTileMNK{}) % cute::size<0>(cute::shape(EpilogueTile{})) == 0,
                  "EPI_TILE_M must divide CTA_M");
    static_assert(cute::size<1>(CtaTileMNK{}) % cute::size<1>(cute::shape(EpilogueTile{})) == 0,
                  "EPI_TILE_N must divide CTA_N");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]");

private:
    constexpr static bool is_source_supported = not cute::is_void_v<ElementC>;
    constexpr static bool is_destination_supported = not cute::is_void_v<ElementD>;
    using NonVoidElementD =
        cute::conditional_t<not is_destination_supported,
                            cutlass::epilogue::fusion::get_element_aux_t<FusionCallbacks>, ElementD>;
    static_assert(not cute::is_void_v<NonVoidElementD>, "SmemElementD is void");
    using NonVoidElementC = cute::conditional_t<not is_source_supported, NonVoidElementD, ElementC>;  // prevents void ref breakages

    using TmaElementD =
        cute::conditional_t<cute::is_same_v<NonVoidElementD, cutlass::complex<float>>, uint64_t, NonVoidElementD>;
    using TmaElementC =
        cute::conditional_t<cute::is_same_v<NonVoidElementC, cutlass::complex<float>>, uint64_t, NonVoidElementC>;

    using SmemElementC = typename cutlass::detail::get_unpacked_element_type<NonVoidElementC>::type;
    using SmemElementD = typename cutlass::detail::get_unpacked_element_type<NonVoidElementD>::type;

    constexpr static int StagesC = StagesC_;
    constexpr static int StagesD = StagesD_;
    constexpr static bool ReuseSmemC = ReuseSmemC_ and is_destination_supported;
    constexpr static bool DelayTmaStore = DelayTmaStore_;

    constexpr static bool is_m_major_C = cutlass::epilogue::collective::detail::is_m_major<StrideC>();
    constexpr static bool is_m_major_D = cutlass::epilogue::collective::detail::is_m_major<StrideD>();

    constexpr static bool is_im2col_C = cute::is_same_v<CopyOpG2S, cute::SM90_TMA_LOAD_IM2COL>;
    constexpr static bool is_im2col_D = cute::is_same_v<CopyOpS2G, cute::SM90_TMA_STORE_IM2COL>;

    // Check if register transformation is needed before copying register to shared memory.
    constexpr static bool IsUseR2R = !cute::is_void_v<CopyOpR2R>;

    using SmemLayoutC = decltype(cute::tile_to_shape(
        SmemLayoutAtomC{},
        cute::make_shape(cute::size<0>(EpilogueTile{}), cute::size<1>(EpilogueTile{}),
                         cute::Int<StagesC>{}),
        cute::conditional_t<is_m_major_C, cute::Step<cute::_2, cute::_1, cute::_3>,
                            cute::Step<cute::_1, cute::_2, cute::_3>>{}));
    using SmemLayoutD = decltype(cute::tile_to_shape(
        SmemLayoutAtomD{},
        cute::make_shape(cute::size<0>(EpilogueTile{}), cute::size<1>(EpilogueTile{}),
                         cute::Int<ReuseSmemC ? StagesC : StagesD>{}),
        cute::conditional_t<is_m_major_D, cute::Step<cute::_2, cute::_1, cute::_3>,
                            cute::Step<cute::_1, cute::_2, cute::_3>>{}));

    constexpr static bool support_smem_reuse =
        is_source_supported && is_destination_supported && StagesD <= StagesC
        && cute::cosize(cute::take<0, 2>(SmemLayoutC{}))
               == cute::cosize(cute::take<0, 2>(SmemLayoutD{}));
    static_assert(not(ReuseSmemC && not support_smem_reuse), "Smem reuse requirements not met");

    constexpr static size_t SmemAlignmentD = cutlass::detail::alignment_for_swizzle(SmemLayoutD{});
    constexpr static size_t SmemAlignmentC = cutlass::detail::alignment_for_swizzle(SmemLayoutC{});
    constexpr static size_t MaxSmemAlignment = cute::max(SmemAlignmentC, SmemAlignmentD);

    using SmemArrayTypeC = cute::ArrayEngine<SmemElementC, cute::cosize_v<SmemLayoutC>>;
    using SmemArrayTypeD = cute::ArrayEngine<SmemElementD, cute::cosize_v<SmemLayoutD>>;

    using EmptyType = cute::tuple<>;
    using SmemCStorage =
        cute::conditional_t<is_source_supported and (not ReuseSmemC), SmemArrayTypeC, EmptyType>;
    using SmemDStorage = cute::conditional_t<is_destination_supported, SmemArrayTypeD, EmptyType>;

    struct CollectiveStorageWithC {
        alignas(SmemAlignmentC) cute::ArrayEngine<SmemElementC, cute::cosize_v<SmemLayoutC>> smem_C;
        alignas(SmemAlignmentD) cute::ArrayEngine<SmemElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };

    union CollectiveStorageWithoutC {
        cute::array<SmemElementC, 0> smem_C;
        alignas(SmemAlignmentD) cute::ArrayEngine<SmemElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };

    union CollectiveStorageReuseC {
        alignas(MaxSmemAlignment) cute::ArrayEngine<SmemElementC, cute::cosize_v<SmemLayoutC>> smem_C;
        alignas(MaxSmemAlignment) cute::ArrayEngine<SmemElementD, cute::cosize_v<SmemLayoutD>> smem_D;
    };

public:
    // TMA pipeline for loading C
    using LoadPipeline = cutlass::PipelineTransactionAsync<StagesC>;
    using LoadPipelineState = cutlass::PipelineState<StagesC>;
    constexpr static uint32_t TmaTransactionBytes =
        (cute::size(cute::take<0, 2>(SmemLayoutC{}))
         * static_cast<uint32_t>(cutlass::sizeof_bits<SmemElementC>::value))
        / 8;
    constexpr static bool RequiresTransactionBytes = true;

    // TMA pipeline for storing D
    using StorePipeline =
        cute::conditional_t<ReuseSmemC, cutlass::PipelineTmaStore<StagesC, StagesD - 1>,
                            cutlass::PipelineTmaStore<StagesD>>;
    using StorePipelineState = cutlass::PipelineState<ReuseSmemC ? StagesC : StagesD>;

    struct SharedStorage {
        struct TensorStorage {
            using CollectiveStorage =
                cute::conditional_t<not is_source_supported, CollectiveStorageWithoutC,
                                    cute::conditional_t<ReuseSmemC, CollectiveStorageReuseC, CollectiveStorageWithC>>;
            CollectiveStorage collective;

            using FusionStorage = typename FusionCallbacks::SharedStorage;
            FusionStorage thread0;
            FusionStorage thread1;
            FusionStorage thread2;
            FusionStorage thread3;
        } tensors;

        using PipelineStorage = typename LoadPipeline::SharedStorage;
        PipelineStorage pipeline;
    };
    using TensorStorage = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Host side epilogue arguments
    struct Arguments {
        typename FusionCallbacks::Arguments thread0{};
        typename FusionCallbacks::Arguments thread1{};
        typename FusionCallbacks::Arguments thread2{};
        typename FusionCallbacks::Arguments thread3{};
        ElementC const* ptr_C;
        StrideC dC;
        ElementD const* ptr_D;
        StrideD dD;
    };

    // Device side epilogue params
    struct Params {
        using TMA_C = decltype(cute::make_tma_copy(
            CopyOpG2S{},
            cute::make_tensor(cute::make_gmem_ptr<TmaElementC const>(nullptr),
                              cute::repeat_like(StrideC{}, int32_t(0)), StrideC{}),
            cute::take<0, 2>(SmemLayoutC{}), EpilogueTile{}, cute::_1{}));
        using TMA_D = decltype(cute::make_tma_copy(
            CopyOpS2G{},
            cute::make_tensor(cute::make_gmem_ptr<TmaElementD>(nullptr),
                              cute::repeat_like(StrideD{}, int32_t(0)), StrideD{}),
            cute::take<0, 2>(SmemLayoutD{}), EpilogueTile{}, cute::_1{}));

        typename FusionCallbacks::Params thread0{};
        typename FusionCallbacks::Params thread1{};
        typename FusionCallbacks::Params thread2{};
        typename FusionCallbacks::Params thread3{};
        TMA_C tma_load_c;
        TMA_D tma_store_d;
        uint32_t tma_transaction_bytes = TmaTransactionBytes;
    };

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    QuadCollectiveEpilogue(Params const& params_, TensorStorage& shared_tensors)
        : params(params_)
        , fusion_callbacks0(params_.thread0, shared_tensors.thread0)
        , fusion_callbacks1(params_.thread1, shared_tensors.thread1)
        , fusion_callbacks2(params_.thread2, shared_tensors.thread2)
        , fusion_callbacks3(params_.thread3, shared_tensors.thread3)
    {
    }

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args)
    {
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        auto shape = cute::make_shape(M, N, L);

        bool implementable = true;
        if constexpr (is_destination_supported) {
            constexpr int tma_alignment_bits_D =
                cutlass::detail::get_output_alignment_bits<ElementD>();
            constexpr int min_tma_aligned_elements_D =
                tma_alignment_bits_D / cutlass::sizeof_bits<ElementD>::value;
            if constexpr (cute::is_same_v<CopyOpS2G, cute::SM90_TMA_STORE_IM2COL>) {  // ignore L stride for implicit gemm
                implementable = cutlass::detail::check_alignment<min_tma_aligned_elements_D>(
                    cute::take<0, 2>(shape), cute::take<0, 2>(StrideD{}));
            } else {
                implementable =
                    cutlass::detail::check_alignment<min_tma_aligned_elements_D>(shape, StrideD{});
            }
        }

        if constexpr (not cute::is_void_v<ElementC>) {
            constexpr int tma_alignment_bits_C = cutlass::detail::get_input_alignment_bits<ElementC>();
            constexpr int min_tma_aligned_elements_C =
                tma_alignment_bits_C / cutlass::sizeof_bits<ElementC>::value;
            if constexpr (cute::is_same_v<CopyOpG2S, cute::SM90_TMA_LOAD_IM2COL>) {  // ignore L stride for implicit gemm
                implementable = implementable
                                && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(
                                    cute::take<0, 2>(shape), cute::take<0, 2>(StrideC{}));
            } else {
                implementable =
                    implementable
                    && cutlass::detail::check_alignment<min_tma_aligned_elements_C>(shape, StrideC{});
            }
        }

        if (!implementable) {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
                "TMA.\n");
        }

        bool fusion_implementable = true;
        ;
        fusion_implementable =
            fusion_implementable && FusionCallbacks::can_implement(problem_shape, args.thread0);
        fusion_implementable =
            fusion_implementable && FusionCallbacks::can_implement(problem_shape, args.thread1);
        fusion_implementable =
            fusion_implementable && FusionCallbacks::can_implement(problem_shape, args.thread2);
        fusion_implementable =
            fusion_implementable && FusionCallbacks::can_implement(problem_shape, args.thread3);

        if (!fusion_implementable) {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum requirements for "
                "FusionCallbacks.\n");
        }

        return implementable && fusion_implementable;
    }

    template <class TileShapeMNK>
    CUTLASS_HOST_DEVICE static constexpr int get_load_pipe_increment(TileShapeMNK tile_shape_MNK)
    {
        // Compute number of epilogue subtiles
        return cute::size<1>(
            zipped_divide(cute::make_layout(cute::take<0, 2>(tile_shape_MNK)), EpilogueTile{}));
    }

    template <class TileShapeMNK>
    CUTLASS_HOST_DEVICE static constexpr int get_store_pipe_increment(TileShapeMNK tile_shape_MNK)
    {
        return get_load_pipe_increment(tile_shape_MNK);
    }

    template <class ProblemShape>
    static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args)
    {
        return 0;
    }

    template <class ProblemShape>
    static cutlass::Status initialize_workspace(ProblemShape const& problem_shape, Arguments const& args,
                                                void* workspace, cudaStream_t stream,
                                                cutlass::CudaHostAdapter* cuda_adapter = nullptr)
    {
        return cutlass::Status::kSuccess;
    }

    CUTLASS_DEVICE
    bool is_producer_load_needed() const { return fusion_callbacks0.is_producer_load_needed(); }

    template <class ProblemShapeMNKL, class TileShapeMNK, class TileCoordMNKL, class TiledMma>
    CUTLASS_DEVICE auto load(LoadPipeline load_pipeline, LoadPipelineState load_pipe_producer_state,
                             ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape_MNK,
                             TileCoordMNKL tile_coord_mnkl, TiledMma tiled_mma, int thread_idx,
                             TensorStorage& shared_tensors, int subtile_idx = -1)
    {
        // Indexing variables
        auto [M, N, K, L] = problem_shape_mnkl;
        auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

        // The tma tensor C under im2col mode only has two modes (M, N) which
        // should be local tiled with only (m_coord, n_coord).
        auto coord_shape = cute::conditional_return<is_im2col_C>(
            cute::make_coord(m_coord, n_coord), cute::make_coord(m_coord, n_coord, l_coord));

        // Represent the full source tensor, slice to get the tile this CTA is currently responsible for
        cute::Tensor mC_mn = params.tma_load_c.get_tma_tensor(cute::make_shape(M, N, L));  // (M,N,L)
        cute::Tensor mC = cute::coalesce(mC_mn, cute::take<0, 2>(CtaTileMNK{}));
        cute::Tensor gC =
            cute::local_tile(mC, cute::take<0, 2>(CtaTileMNK{}), coord_shape);  // (CTA_M,CTA_N)

        // Apply epilogue subtile, get matching smem tensor
        auto ptr_sC = shared_tensors.collective.smem_C.begin();
        cute::Tensor gC_epi = cute::flat_divide(gC, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)
        cute::Tensor sC_epi = cute::make_tensor(cute::make_smem_ptr(ptr_sC),
                                                SmemLayoutC{});  // (EPI_TILE_M,EPI_TILE_N,PIPE_C)

        // Prepare the thread(b)lock's (G)mem to (S)mem TMA tiled copy (bGS_)
        cute::ThrCopy thrblk_g2s = params.tma_load_c.get_slice(cute::_0{});
        cute::Tensor bGS_gC = thrblk_g2s.partition_S(gC_epi);  // (G2S,G2S_M,G2S_N,EPI_M,EPI_N)
        cute::Tensor bGS_sC = thrblk_g2s.partition_D(sC_epi);  // (G2S,G2S_M,G2S_N,PIPE_C)

        // Get the fusion callbacks for the producer load warp
        auto pld_args = cutlass::epilogue::fusion::detail::ProducerLoadArgs(
            problem_shape_mnkl, CtaTileMNK{}, tile_coord_mnkl, tiled_mma, EpilogueTile{}, thread_idx);
        auto pld_callbacks = fusion_callbacks0.get_producer_load_callbacks(pld_args);
        bool is_C_load_needed = is_source_supported && fusion_callbacks0.is_C_load_needed();

        // Predication for TMA load (one thread issues TMA load)
        bool issue_tma_load = cute::elect_one_sync();

        // Pre-loop fusion callback entry point
        pld_callbacks.begin();

        CUTLASS_PRAGMA_UNROLL
        for (int epi_n = 0; epi_n < cute::size<3>(gC_epi); ++epi_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_m = 0; epi_m < cute::size<2>(gC_epi); ++epi_m) {
                if (subtile_idx != -1
                    && (epi_n * static_cast<int>(cute::size<2>(gC_epi)) + epi_m) != subtile_idx) {
                    continue;
                }
                // Acquire the lock for this stage
                constexpr uint16_t mcast_mask = 0;
                uint64_t* tma_barrier = load_pipeline.producer_get_barrier(load_pipe_producer_state);
                load_pipeline.producer_acquire(load_pipe_producer_state);

                // Loop fusion callback entry point
                pld_callbacks.step(tma_barrier, epi_m, epi_n, load_pipe_producer_state.count(),
                                   issue_tma_load);

                // Execute the TMA load for C if needed
                if (issue_tma_load && is_C_load_needed) {
                    cute::copy(params.tma_load_c.with(*tma_barrier, mcast_mask),
                               bGS_gC(cute::_, cute::_, cute::_, epi_m, epi_n),
                               bGS_sC(cute::_, cute::_, cute::_, load_pipe_producer_state.index()));
                    load_pipeline.producer_expect_transaction(load_pipe_producer_state);
                }

                // Commit TMA loads for this stage and release the lock
                load_pipeline.producer_commit(load_pipe_producer_state);
                ++load_pipe_producer_state;
            }
        }

        // Post-loop fusion callback entry point
        pld_callbacks.end();

        return load_pipe_producer_state;
    }

    CUTLASS_DEVICE auto load_tail(LoadPipeline load_pipeline, LoadPipelineState load_pipe_producer_state)
    {
        bool issue_tma_load = cute::elect_one_sync();
        if (issue_tma_load) {
            load_pipeline.producer_tail(load_pipe_producer_state);
        }

        return load_pipe_producer_state;
    }

    template <class ProblemShapeMNKL, class TileShapeMNK, class TileCoordMNKL, class AccEngine,
              class AccLayout, class TiledMma>
    CUTLASS_DEVICE auto
    store(LoadPipeline load_pipeline, LoadPipelineState load_pipe_consumer_state,
          StorePipeline store_pipeline, StorePipelineState store_pipe_producer_state,
          ProblemShapeMNKL problem_shape_mnkl, TileShapeMNK tile_shape_MNK,
          TileCoordMNKL tile_coord_mnkl, cute::Tensor<AccEngine, AccLayout> accumulators0,
          cute::Tensor<AccEngine, AccLayout> accumulators1,
          cute::Tensor<AccEngine, AccLayout> accumulators2,
          cute::Tensor<AccEngine, AccLayout> accumulators3, TiledMma tiled_mma, int thread_idx,
          TensorStorage& shared_tensors, int subtile_idx = -1)
    {
        using ElementAccumulator = typename AccEngine::value_type;
        using ElementCompute_ =
            typename cutlass::epilogue::fusion::FusionCallbacksTraits<FusionCallbacks>::ElementCompute;
        using ElementCompute =
            cute::conditional_t<cute::is_void_v<ElementCompute_>, ElementAccumulator, ElementCompute_>;

        static_assert(cute::is_rmem<AccEngine>::value, "Accumulator must be RF resident.");
        static_assert(cute::rank(AccLayout{}) == 3,
                      "Accumulator must be MMA-partitioned: (MMA,MMA_M,MMA_N)");
        static_assert(cute::rank(ProblemShapeMNKL{}) == 4, "ProblemShapeMNKL must be rank 4");
        static_assert(cute::is_static<TileShapeMNK>::value, "TileShapeMNK must be static");
        static_assert(cute::rank(TileShapeMNK{}) == 3, "TileShapeMNK must be rank 3");
        static_assert(cute::rank(TileCoordMNKL{}) == 4, "TileCoordMNKL must be rank 4");

        // Indexing variables
        auto [M, N, K, L] = problem_shape_mnkl;
        auto [m_coord, n_coord, k_coord, l_coord] = tile_coord_mnkl;

        // The tma tensor D under im2col mode only has two modes (M, N) which
        // should be local tiled with only (m_coord, n_coord).
        auto coord_shape = cute::conditional_return<is_im2col_D>(
            cute::make_coord(m_coord, n_coord), cute::make_coord(m_coord, n_coord, l_coord));

        // Represent the full output tensor, slice to get the tile this CTA is responsible for
        cute::Tensor mD_mn = params.tma_store_d.get_tma_tensor(cute::make_shape(M, N, L));  // (M,N,L)
        cute::Tensor mD = cute::coalesce(mD_mn, cute::take<0, 2>(CtaTileMNK{}));
        cute::Tensor gD =
            cute::local_tile(mD, cute::take<0, 2>(CtaTileMNK{}), coord_shape);  // (CTA_M,CTA_N)

        // Apply epilogue subtiling
        cute::Tensor gD_epi = cute::flat_divide(gD, EpilogueTile{});  // (EPI_TILE_M,EPI_TILE_N,EPI_M,EPI_N)

        // Construct the corresponding pipelined smem tensors
        auto ptr_sC = shared_tensors.collective.smem_C.begin();
        auto ptr_sD = shared_tensors.collective.smem_D.begin();
        cute::Tensor sC_epi = cute::as_position_independent_swizzle_tensor(cute::make_tensor(
            cute::make_smem_ptr(ptr_sC), SmemLayoutC{}));  // (EPI_TILE_M,EPI_TILE_N,PIPE_C)
        cute::Tensor sD_epi = cute::as_position_independent_swizzle_tensor(cute::make_tensor(
            cute::make_smem_ptr(ptr_sD), SmemLayoutD{}));  // (EPI_TILE_M,EPI_TILE_N,PIPE_D)

        cute::TiledCopy tiled_copy_C_atom = cute::make_tiled_copy_C_atom(CopyAtomC{}, tiled_mma);

        // (t)hread-partition for (r)egister to (r)egister copy (tRR_)
        cute::TiledCopy tiled_r2r = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            if constexpr (IsUseR2R) {
                return cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2R, ElementCompute>{},
                                               tiled_copy_C_atom);
            } else {
                return cute::make_tiled_copy_S(
                    cute::Copy_Atom<cute::AutoVectorizingCopyWithAssumedAlignment<128>, ElementCompute>{},
                    tiled_copy_C_atom);
            }
        }();
        cute::ThrCopy thread_r2r = tiled_r2r.get_slice(thread_idx);

        // (t)hread-partition for (r)egister to (s)mem copy (tRS_)
        cute::TiledCopy tiled_r2s = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            if constexpr (IsUseR2R) {
                return cute::make_tiled_copy_D(cute::Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_r2r);
            } else {
                return cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, SmemElementD>{},
                                               tiled_copy_C_atom);
            }
        }();
        cute::ThrCopy thread_r2s = tiled_r2s.get_slice(thread_idx);
        cute::Tensor tRS_rAcc0 = thread_r2s.retile_S(accumulators0);  // ((R2S,R2S_V),MMA_M,MMA_N)
        cute::Tensor tRS_rAcc1 = thread_r2s.retile_S(accumulators1);  // ((R2S,R2S_V),MMA_M,MMA_N)
        cute::Tensor tRS_rAcc2 = thread_r2s.retile_S(accumulators2);  // ((R2S,R2S_V),MMA_M,MMA_N)
        cute::Tensor tRS_rAcc3 = thread_r2s.retile_S(accumulators3);  // ((R2S,R2S_V),MMA_M,MMA_N)

        cute::Tensor tRS_sD = thread_r2s.partition_D(sD_epi);  // (R2S,R2S_M,R2S_N,PIPE_D)

        auto mma_tile_m = cute::size<0>(TileShapeMNK{}) / cute::size<1>(tRS_rAcc0);
        auto mma_tile_n = cute::size<1>(TileShapeMNK{}) / cute::size<2>(tRS_rAcc0);
        auto epi_tile_m = cute::size<0>(EpilogueTile{});
        auto epi_tile_n = cute::size<1>(EpilogueTile{});

        // Allocate D registers
        cute::Layout tRS_rD_layout =
            cute::make_layout(cute::take<0, 3>(cute::shape(thread_r2s.partition_S(sD_epi))));
        cute::Tensor tRS_rD = cute::make_tensor<SmemElementD>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)

        // Vectorized fragment view
        constexpr int FragmentSize = DispatchPolicy::FragmentSize;
        cute::Tensor tRS_rAcc_frg0 =
            cute::recast<cutlass::Array<ElementAccumulator, FragmentSize>>(tRS_rAcc0);
        cute::Tensor tRS_rAcc_frg1 =
            cute::recast<cutlass::Array<ElementAccumulator, FragmentSize>>(tRS_rAcc1);
        cute::Tensor tRS_rAcc_frg2 =
            cute::recast<cutlass::Array<ElementAccumulator, FragmentSize>>(tRS_rAcc2);
        cute::Tensor tRS_rAcc_frg3 =
            cute::recast<cutlass::Array<ElementAccumulator, FragmentSize>>(tRS_rAcc3);

        cute::Tensor tRS_rD_frg = cute::recast<cutlass::Array<SmemElementD, FragmentSize>>(tRS_rD);
        CUTE_STATIC_ASSERT(cute::size<0>(tRS_rAcc0) % FragmentSize == 0,
                           "Fragment size does not vectorize properly");

        // (t)hread-partition for (s)mem to (r)egister copy (tSR_)
        cute::TiledCopy tiled_s2r =
            cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpS2R, SmemElementC>{}, tiled_copy_C_atom);
        cute::ThrCopy thread_s2r = tiled_s2r.get_slice(thread_idx);
        cute::Tensor tSR_sC = thread_s2r.partition_S(sC_epi);  // (S2R,S2R_M,S2R_N,PIPE_C)
        cute::Layout tSR_rC_layout = thread_s2r.retile_D(tRS_rD).layout();  // (S2R,S2R_M,S2R_N)

        // Allocate C registers
        // If C smem load is a non-vectorized dst(i) = src(i) then we can allocate C registers directly in the compute type
        // to eliminate some redundant pack+unpack instruction sequences for sub-word types
        constexpr bool IsDirectS2R =
            cute::is_same_v<CopyOpS2R, cute::AutoVectorizingCopyWithAssumedAlignment<128>>
            && decltype(cute::max_common_vector(tSR_rC_layout, tSR_sC.layout()))::value <= 1;
        using RegisterElementC = cute::conditional_t<IsDirectS2R, ElementCompute, SmemElementC>;
        cute::Tensor tRS_rC = cute::make_tensor<RegisterElementC>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)
        cute::Tensor tSR_rC = thread_s2r.retile_D(tRS_rC);  // (S2R,S2R_M,S2R_N)

        // thread(b)lock-partition for (s)mem to (g)mem copy (bSG_)
        cute::ThrCopy thrblk_s2g = params.tma_store_d.get_slice(cute::_0{});
        cute::Tensor bSG_sD = thrblk_s2g.partition_S(sD_epi);  // (S2G,S2G_M,S2G_N,PIPE_D)
        cute::Tensor bSG_gD = thrblk_s2g.partition_D(gD_epi);  // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

        // OOB predication for tile quantization "residue"
        // Absolute coordinate tensors (dynamic)
        cute::Tensor mD_crd = cute::make_identity_tensor(cute::make_shape(M, N));  // (M,N)
        cute::Tensor cD_mn = cute::local_tile(mD_crd, cute::take<0, 2>(CtaTileMNK{}),
                                              cute::make_coord(m_coord, n_coord));  // (CTA_M,CTA_N)
        cute::Tensor tRS_cD_mn = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            if constexpr (IsUseR2R) {
                // (t)hread-partition for ConsumerStoreCallbacks.
                cute::TiledCopy tiled_cst = cute::make_tiled_copy_S(
                    cute::Copy_Atom<CopyOpR2S, SmemElementC>{}, tiled_copy_C_atom);
                cute::ThrCopy thread_cst = tiled_cst.get_slice(thread_idx);
                return thread_cst.partition_S(
                    cute::flat_divide(cD_mn, EpilogueTile{}));  // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
            } else {
                return thread_r2s.partition_S(
                    cute::flat_divide(cD_mn, EpilogueTile{}));  // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
            }
        }();
        // Relative coordinate tensors (static)
        cute::Tensor cD = cute::make_coord_tensor(cD_mn.layout());          // (CTA_M,CTA_N)
        cute::Tensor tRS_cD = cute::make_coord_tensor(tRS_cD_mn.layout());  // (R2S,R2S_M,R2S_N,EPI_M,EPI_N)
        // Subtract the global "bottom right" corner from the local "top left" corner to get the max relative coordinate
        auto residue_cD = cute::make_coord(M, N) - cD_mn(cute::_0{});          // (m,n)
        auto residue_tRS_cD = cute::make_coord(M, N) - tRS_cD_mn(cute::_0{});  // (m,n)

        CUTE_STATIC_ASSERT(epi_tile_m % mma_tile_m == 0, "MMA_TILE_M must divide EPI_TILE_M");

        if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
            // When the epilogue subtile is larger than the MMA tiles, loop over multiple MMA tiles
            CUTE_STATIC_ASSERT(epi_tile_n % mma_tile_n == 0, "MMA_TILE_N must divide EPI_TILE_N");
        } else {
            CUTE_STATIC_ASSERT(mma_tile_n % epi_tile_n == 0, "EPI_TILE_N must divide MMA_TILE_N");
        }

        // Get cute::TiledCopy for partition reference when consumer store.
        cute::TiledCopy tiled_copy_partition_ref =
            cute::make_tiled_copy_S(cute::Copy_Atom<CopyOpR2S, SmemElementD>{}, tiled_copy_C_atom);
        // Get the fusion callbacks for the consumer store warps
        constexpr bool RefSrc = true;  // Register tensors reference tiled copy src layout
        auto cst_args = cutlass::epilogue::fusion::detail::ConsumerStoreArgs(
            problem_shape_mnkl, CtaTileMNK{}, tile_coord_mnkl, tiled_mma, EpilogueTile{},
            tiled_copy_partition_ref, cD, residue_cD, tRS_cD, residue_tRS_cD, tRS_rC, thread_idx);
        auto cst_callbacks0 = fusion_callbacks0.template get_consumer_store_callbacks<RefSrc>(cst_args);
        auto cst_callbacks1 = fusion_callbacks1.template get_consumer_store_callbacks<RefSrc>(cst_args);
        auto cst_callbacks2 = fusion_callbacks2.template get_consumer_store_callbacks<RefSrc>(cst_args);
        auto cst_callbacks3 = fusion_callbacks3.template get_consumer_store_callbacks<RefSrc>(cst_args);
        bool is_producer_load_needed = fusion_callbacks0.is_producer_load_needed();
        bool is_C_load_needed = is_source_supported && fusion_callbacks0.is_C_load_needed();

        using FragmentVisit = decltype(cst_callbacks0.visit(tRS_rAcc_frg0(0), 0, 0, 0));
        constexpr bool IsDirectR2S =
            cute::is_same_v<FragmentVisit, cutlass::Array<SmemElementD, FragmentSize>>;
        using RegisterElementD = cute::conditional_t<!IsDirectR2S, ElementCompute, SmemElementD>;
        cute::Tensor tRS_rCompute = cute::make_tensor<RegisterElementD>(tRS_rD_layout);  // (R2S,R2S_M,R2S_N)
        cute::Tensor tRS_rCompute_frg =
            cute::recast<cutlass::Array<RegisterElementD, FragmentSize>>(tRS_rCompute);

        // Thread synchronizer for previously issued waits or fences
        // to ensure visibility of smem reads/writes to threads or TMA unit
        auto synchronize = [&]() CUTLASS_LAMBDA_FUNC_INLINE {
            cutlass::arch::NamedBarrier::sync(size(TiledMma{}),
                                              cutlass::arch::ReservedNamedBarriers::EpilogueBarrier);
        };

        // Predication for TMA store (one warp issues TMA store)
        bool issue_tma_store = (thread_idx / cutlass::NumThreadsPerWarp) == 0;

        // In the reuse smem configuration we have StagesC smem buffers and at most StagesD
        // committed TMA stores in flight. The TMA store pipeline producer acquire returns when at
        // most StagesD-1 committed stores are in-flight, so we can only guarantee store completion
        // after StagesD iterations, then we can begin issuing releases on the smem buffer locks.
        // store_pipe_producer_state tracks the acquire and load_pipe_consumer_state tracks the
        // release, in circular buffer fashion.
        LoadPipelineState load_wait_state = load_pipe_consumer_state;
        if constexpr (ReuseSmemC) {
            load_wait_state = store_pipe_producer_state;
            load_wait_state.phase_ ^= 1;
        }

        // We can delay issue of TMA store by one iteration to achieve better interleaving of non-TMA instructions
        // Sync requirements of smem reuse may preclude this optimization
        // Delayed stores cause delayed stage releases which causes deadlock when StagesC == StagesD
        [[maybe_unused]] int epi_m_prev = 0;
        [[maybe_unused]] int epi_n_prev = 0;
        static_assert(not(DelayTmaStore and ReuseSmemC and StagesC <= StagesD),
                      "This TMA epilogue configuration will deadlock");

        // The TMA store sequence for one subtile iteration
        auto tma_store_fn = [&](int epi_m, int epi_n) CUTLASS_LAMBDA_FUNC_INLINE {
            // Write the tile from smem to gmem with TMA
            cutlass::arch::fence_view_async_shared();  // ensure smem writes are visible to TMA
            synchronize();  // ensure all threads have issued their async fence
            if constexpr (is_destination_supported) {
                if (issue_tma_store) {
                    cute::copy(params.tma_store_d,
                               bSG_sD(cute::_, cute::_, cute::_, store_pipe_producer_state.index()),
                               bSG_gD(cute::_, cute::_, cute::_, epi_m, epi_n));
                }
            }

            // Post async fence, pre TMA commit callback entry point
            cst_callbacks0.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);
            cst_callbacks1.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);
            cst_callbacks2.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);
            cst_callbacks3.tma_store(epi_m, epi_n, store_pipe_producer_state.count(), issue_tma_store);

            // Commit the TMA stores for this stage
            if (issue_tma_store) {
                store_pipeline.producer_commit(store_pipe_producer_state);
            }
            ++store_pipe_producer_state;
            ++issued_stores;

            // Wait for the next smem buffer to be available
            if (issue_tma_store) {
                store_pipeline.producer_acquire(store_pipe_producer_state);
            }
            synchronize();

            if constexpr (ReuseSmemC) {
                // producer_acquire returns when at most StagesD-1 committed stores are pending
                bool store_finished = issued_stores > StorePipeline::UnacquiredStages;
                // Let dma warp know earliest smem buffer is consumed and empty after StagesD producer commits
                if (store_finished) {
                    if (is_producer_load_needed) {
                        load_pipeline.consumer_release(load_pipe_consumer_state);
                    }
                    ++load_pipe_consumer_state;
                }
            }
        };

        //
        // BEGIN EPILOGUE
        //

        // Pre-loop fusion callback entry point
        cst_callbacks0.begin();
        cst_callbacks1.begin();
        cst_callbacks2.begin();
        cst_callbacks3.begin();

        if (cst_callbacks0.begin_sync_needed()) {
            synchronize();
        }

        // For each output tile
        CUTLASS_PRAGMA_UNROLL
        for (int epi_n = 0; epi_n < cute::size<3>(gD_epi); ++epi_n) {
            CUTLASS_PRAGMA_UNROLL
            for (int epi_m = 0; epi_m < cute::size<2>(gD_epi); ++epi_m) {
                [[maybe_unused]] bool is_first_iteration = epi_m == 0 && epi_n == 0;
                bool is_last_iteration =
                    epi_m == cute::size<2>(gD_epi) - 1 && epi_n == cute::size<3>(gD_epi) - 1;

                if (subtile_idx != -1
                    && (epi_n * static_cast<int>(cute::size<2>(gD_epi)) + epi_m) != subtile_idx) {
                    continue;
                }

                cst_callbacks0.begin_loop(epi_m, epi_n);
                cst_callbacks1.begin_loop(epi_m, epi_n);
                cst_callbacks2.begin_loop(epi_m, epi_n);
                cst_callbacks3.begin_loop(epi_m, epi_n);

                if (is_producer_load_needed) {
                    // Wait for the producer load to fill smem
                    load_pipeline.consumer_wait(load_wait_state);

                    if (is_C_load_needed) {
                        // Copy source tile from smem to register
                        cute::copy(tiled_s2r,
                                   tSR_sC(cute::_, cute::_, cute::_, load_wait_state.index()), tSR_rC);
                        // Ensure smem loads are complete before reusing smem for mixed types/layouts
                        if constexpr (ReuseSmemC && not(SmemLayoutC{} == SmemLayoutD{})) {
                            synchronize();
                        }
                    }
                }

                // First loop fusion callback entry point
                cst_callbacks0.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);
                cst_callbacks1.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);
                cst_callbacks2.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);
                cst_callbacks3.previsit(epi_m, epi_n, load_wait_state.count(), is_producer_load_needed);

                if (is_producer_load_needed) {
                    if constexpr (not ReuseSmemC) {
                        // Let producer load warp know smem buffers are consumed and empty
                        cutlass::arch::fence_view_async_shared();
                        load_pipeline.consumer_release(load_pipe_consumer_state);
                        ++load_pipe_consumer_state;
                    }
                    ++load_wait_state;
                }

                if constexpr (epi_tile_m * epi_tile_n > mma_tile_m * mma_tile_n) {
                    // When the epilogue subtile is larger than the MMA tiles, loop over multiple
                    // MMA tiles
                    static constexpr int MmaMPerEpiM = epi_tile_m / mma_tile_m;
                    static constexpr int MmaNPerEpiN = epi_tile_n / mma_tile_n;

                    CUTLASS_PRAGMA_UNROLL
                    for (int mma_n_in_epi = 0; mma_n_in_epi < MmaNPerEpiN; ++mma_n_in_epi) {
                        int mma_n = (epi_n * MmaNPerEpiN) + mma_n_in_epi;

                        CUTLASS_PRAGMA_UNROLL
                        for (int mma_m_in_epi = 0; mma_m_in_epi < MmaMPerEpiM; ++mma_m_in_epi) {
                            int mma_m = (epi_m * MmaMPerEpiM) + mma_m_in_epi;
                            cute::Tensor tRS_rAcc_frg_mn0 = tRS_rAcc_frg0(cute::_, mma_m, mma_n);
                            cute::Tensor tRS_rAcc_frg_mn1 = tRS_rAcc_frg1(cute::_, mma_m, mma_n);
                            cute::Tensor tRS_rAcc_frg_mn2 = tRS_rAcc_frg2(cute::_, mma_m, mma_n);
                            cute::Tensor tRS_rAcc_frg_mn3 = tRS_rAcc_frg3(cute::_, mma_m, mma_n);

                            int idx_in_epi_subtile = (mma_n_in_epi * MmaMPerEpiM + mma_m_in_epi);

                            auto comp0 = cst_callbacks0.visit(tRS_rAcc_frg_mn0(0),
                                                              idx_in_epi_subtile, epi_m, epi_n);
                            auto comp1 = cst_callbacks1.visit(tRS_rAcc_frg_mn1(0),
                                                              idx_in_epi_subtile, epi_m, epi_n);
                            auto comp2 = cst_callbacks2.visit(tRS_rAcc_frg_mn2(0),
                                                              idx_in_epi_subtile, epi_m, epi_n);
                            auto comp3 = cst_callbacks3.visit(tRS_rAcc_frg_mn3(0),
                                                              idx_in_epi_subtile, epi_m, epi_n);
                            QuadAddOp<cutlass::Array<RegisterElementD, FragmentSize>> quad_add_op;
                            auto combined_comp = quad_add_op(comp0, comp1, comp2, comp3);
                            tRS_rCompute_frg(idx_in_epi_subtile) = combined_comp;
                        }
                    }
                } else {
                    int mma_m = epi_m;
                    int mma_n = (epi_n * cute::size<1>(EpilogueTile{})) / mma_tile_n;
                    cute::Tensor tRS_rAcc_frg_mn0 = tRS_rAcc_frg0(cute::_, mma_m, mma_n);
                    cute::Tensor tRS_rAcc_frg_mn1 = tRS_rAcc_frg1(cute::_, mma_m, mma_n);
                    cute::Tensor tRS_rAcc_frg_mn2 = tRS_rAcc_frg2(cute::_, mma_m, mma_n);
                    cute::Tensor tRS_rAcc_frg_mn3 = tRS_rAcc_frg3(cute::_, mma_m, mma_n);

                    // Vectorized fragment loop with visitor callback entry point
                    int epi_n_in_mma = epi_n % (mma_tile_n / epi_tile_n);
                    int r2s_v = epi_n_in_mma * size(tRS_rCompute_frg);
                    CUTLASS_PRAGMA_UNROLL
                    for (int epi_v = 0; epi_v < size(tRS_rCompute_frg); ++epi_v) {
                        auto comp0 = cst_callbacks0.visit(tRS_rAcc_frg_mn0(r2s_v + epi_v), epi_v,
                                                          epi_m, epi_n);
                        auto comp1 = cst_callbacks1.visit(tRS_rAcc_frg_mn1(r2s_v + epi_v), epi_v,
                                                          epi_m, epi_n);
                        auto comp2 = cst_callbacks2.visit(tRS_rAcc_frg_mn2(r2s_v + epi_v), epi_v,
                                                          epi_m, epi_n);
                        auto comp3 = cst_callbacks3.visit(tRS_rAcc_frg_mn3(r2s_v + epi_v), epi_v,
                                                          epi_m, epi_n);
                        QuadAddOp<cutlass::Array<RegisterElementD, FragmentSize>> quad_add_op;
                        auto combined_comp = quad_add_op(comp0, comp1, comp2, comp3);
                        tRS_rCompute_frg(epi_v) = combined_comp;
                    }
                }

                // The latest we can delay the TMA store is right before the smem store of the next iteration
                // since the current TMA store needs to be committed before we can acquire the next smem buffer
                if constexpr (DelayTmaStore) {
                    // Issue TMA stores for the previous subtile
                    if (not is_first_iteration and subtile_idx == -1) {
                        tma_store_fn(epi_m_prev, epi_n_prev);
                    }
                    epi_m_prev = epi_m;
                    epi_n_prev = epi_n;
                }

                // Smem reduction callback entry point using current store buffer for workspace
                cst_callbacks0.reduce(sD_epi(cute::_, cute::_, store_pipe_producer_state.index()),
                                      synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);
                cst_callbacks1.reduce(sD_epi(cute::_, cute::_, store_pipe_producer_state.index()),
                                      synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);
                cst_callbacks2.reduce(sD_epi(cute::_, cute::_, store_pipe_producer_state.index()),
                                      synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);
                cst_callbacks3.reduce(sD_epi(cute::_, cute::_, store_pipe_producer_state.index()),
                                      synchronize, epi_m, epi_n, is_last_iteration, tRS_rCompute_frg);

                // Copy tile from register to regiser if needed
                if constexpr (IsUseR2R) {
                    // retile source and destination for tiled_r2r
                    cute::Tensor tRR_rD_src =
                        thread_r2r.retile_S(tRS_rCompute);  // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)
                    cute::Tensor tRR_rD_dst =
                        thread_r2r.retile_D(tRS_rCompute);  // (R2R,R2R_M,R2R_N,EPI_M,EPI_N)

                    // Output register transformation before copying to shared memory.
                    cute::copy(tiled_r2r, tRR_rD_src, tRR_rD_dst);
                }

                CUTLASS_PRAGMA_UNROLL
                for (int i = 0; i < size(tRS_rD_frg); ++i) {
                    tRS_rD_frg(i) =
                        cutlass::NumericArrayConverter<SmemElementD, RegisterElementD, FragmentSize>{}(
                            tRS_rCompute_frg(i));
                }

                // Copy tile from register to smem
                if constexpr (is_destination_supported) {
                    cute::copy(tiled_r2s, tRS_rD,
                               tRS_sD(cute::_, cute::_, cute::_, store_pipe_producer_state.index()));
                }

                // Post reduction, pre TMA store callback entry point
                constexpr bool issue_smem_store = true;  // No smem store predication
                cst_callbacks0.postreduce(epi_m, epi_n, store_pipe_producer_state.count(),
                                          issue_smem_store);
                cst_callbacks1.postreduce(epi_m, epi_n, store_pipe_producer_state.count(),
                                          issue_smem_store);
                cst_callbacks2.postreduce(epi_m, epi_n, store_pipe_producer_state.count(),
                                          issue_smem_store);
                cst_callbacks3.postreduce(epi_m, epi_n, store_pipe_producer_state.count(),
                                          issue_smem_store);

                if constexpr (not DelayTmaStore) {
                    // Issue TMA stores for this subtile
                    tma_store_fn(epi_m, epi_n);
                }

                cst_callbacks0.end_loop(epi_m, epi_n);
                cst_callbacks1.end_loop(epi_m, epi_n);
                cst_callbacks2.end_loop(epi_m, epi_n);
                cst_callbacks3.end_loop(epi_m, epi_n);
            }  // for epi_m
        }  // for epi_n

        if constexpr (DelayTmaStore) {
            // Issue TMA stores for the last subtile
            tma_store_fn(epi_m_prev, epi_n_prev);
        }

        // Post-loop fusion callback entry point
        cst_callbacks0.end();
        cst_callbacks1.end();
        cst_callbacks2.end();
        cst_callbacks3.end();

        return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
    }

    CUTLASS_DEVICE auto store_tail(LoadPipeline load_pipeline,
                                   LoadPipelineState load_pipe_consumer_state,
                                   StorePipeline store_pipeline,
                                   StorePipelineState store_pipe_producer_state)
    {
        // wait for all TMA stores to complete
        store_pipeline.producer_tail(store_pipe_producer_state);
        // reset store counter
        issued_stores = 0;

        if constexpr (ReuseSmemC) {
            if (fusion_callbacks0.is_producer_load_needed()) {
                // Issue releases on up to StagesD-1 previously issued TMA stores
                constexpr int release_stages =
                    cute::min(StorePipeline::UnacquiredStages, get_load_pipe_increment(CtaTileMNK{}));
                CUTLASS_PRAGMA_UNROLL
                for (int stage = 0; stage < release_stages; ++stage) {
                    load_pipeline.consumer_release(load_pipe_consumer_state);
                    ++load_pipe_consumer_state;
                }
            }
        }

        return cute::make_tuple(load_pipe_consumer_state, store_pipe_producer_state);
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& epilogue_params)
    {
        if constexpr (is_source_supported) {
            cute::prefetch_tma_descriptor(epilogue_params.tma_load_c.get_tma_descriptor());
        }
        if constexpr (is_destination_supported) {
            cute::prefetch_tma_descriptor(epilogue_params.tma_store_d.get_tma_descriptor());
        }
    }

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                    Arguments const& args,
                                                    [[maybe_unused]] void* workspace)
    {
        // Optionally append 1s until problem shape is rank-4 in case its is only rank-3 (MNK)
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        uint32_t transaction_bytes = TmaTransactionBytes;
        typename Params::TMA_C tma_load_c{};
        if constexpr (is_source_supported) {
            cute::Tensor tensor_c =
                cute::make_tensor(cute::make_gmem_ptr<TmaElementC const>(args.ptr_C),
                                  cute::make_layout(cute::make_shape(M, N, L), args.dC));
            tma_load_c = cute::make_tma_copy_C_sm90(CopyOpG2S{}, tensor_c,
                                                    cute::take<0, 2>(SmemLayoutC{}), EpilogueTile{});
        }

        typename Params::TMA_D tma_store_d{};
        if constexpr (is_destination_supported) {
            cute::Tensor tensor_d =
                cute::make_tensor(cute::make_gmem_ptr<TmaElementD>(args.ptr_D),
                                  cute::make_layout(cute::make_shape(M, N, L), args.dD));
            tma_store_d = cute::make_tma_copy_C_sm90(CopyOpS2G{}, tensor_d,
                                                     cute::take<0, 2>(SmemLayoutD{}), EpilogueTile{});
        }

        return { FusionCallbacks::to_underlying_arguments(problem_shape, args.thread0, workspace),
                 FusionCallbacks::to_underlying_arguments(problem_shape, args.thread1, workspace),
                 FusionCallbacks::to_underlying_arguments(problem_shape, args.thread2, workspace),
                 FusionCallbacks::to_underlying_arguments(problem_shape, args.thread3, workspace),
                 tma_load_c,
                 tma_store_d,
                 transaction_bytes };
    }

private:
    Params const& params;
    FusionCallbacks fusion_callbacks0;
    FusionCallbacks fusion_callbacks1;
    FusionCallbacks fusion_callbacks2;
    FusionCallbacks fusion_callbacks3;
    int issued_stores = 0;
};
