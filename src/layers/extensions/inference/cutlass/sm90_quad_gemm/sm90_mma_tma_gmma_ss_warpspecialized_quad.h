// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/dispatch_policy.hpp>
// clang-format on

/**
 * These 1 classes/structs are rewritten:
 *   1. cutlass::gemm::collective::CollectiveMma -> QuadCollectiveMma
 */

// Primary template declaration to enable partial/specialized definitions below.
template <class DispatchPolicy, class TileShape, class ElementA, class StrideA, class ElementB,
          class StrideB, class TiledMma, class GmemTiledCopyA, class SmemLayoutAtomsA, class SmemCopyAtomsA,
          class TransformA, class GmemTiledCopyB, class SmemLayoutAtomsB, class SmemCopyAtomsB, class TransformB>
struct QuadCollectiveMma {
    static_assert(cutlass::detail::dependent_false<ElementA>,
                  "Could not find a mainloop specialization.");
};

// Original file: cutlass/gemm/collective/sm90_mma_tma_gmma_ss_warpspecialized.hpp
// to add multiple accumulators to mma
// WarpSpecialized Mainloop
template <int Stages, class ClusterShape, class KernelSchedule, class TileShape_, class ElementA_,
          class StrideA_, class ElementB_, class StrideB_, class TiledMma_, class GmemTiledCopyA_,
          class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_, class GmemTiledCopyB_,
          class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct QuadCollectiveMma<
    cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>,
    TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_, GmemTiledCopyA_, SmemLayoutAtomA_,
    SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_, TransformB_> {
    //
    // Type Aliases
    //
    using DispatchPolicy =
        cutlass::gemm::MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>;
    using TileShape = TileShape_;
    using ElementA = ElementA_;
    using StrideA = StrideA_;
    using ElementB = ElementB_;
    using StrideB = StrideB_;
    using TiledMma = TiledMma_;
    using ElementAccumulator = typename TiledMma::ValTypeC;
    using GmemTiledCopyA = GmemTiledCopyA_;
    using GmemTiledCopyB = GmemTiledCopyB_;
    using SmemLayoutAtomA = SmemLayoutAtomA_;
    using SmemLayoutAtomB = SmemLayoutAtomB_;
    using SmemCopyAtomA = SmemCopyAtomA_;
    using SmemCopyAtomB = SmemCopyAtomB_;
    using TransformA = TransformA_;
    using TransformB = TransformB_;
    using ArchTag = typename DispatchPolicy::ArchTag;

    using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
    using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
    using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

    using PipelineParams = typename MainloopPipeline::Params;

    // One threads per CTA are producers (1 for operand tile)
    static constexpr int NumProducerThreadEvents = 1;

    static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((cute::size<0>(TileShape{}) % cute::size<0>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomA{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
    static_assert((cute::size<1>(TileShape{}) % cute::size<0>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");
    static_assert((cute::size<2>(TileShape{}) % cute::size<1>(SmemLayoutAtomB{})) == 0,
                  "SmemLayoutAtom must evenly divide tile shape.");

    // Tile along modes in a way that maximizes the TMA box size.
    using SmemLayoutA = decltype(cute::tile_to_shape(
        SmemLayoutAtomA{},
        cute::make_shape(cute::shape<0>(TileShape{}), cute::shape<2>(TileShape{}),
                         cute::Int<DispatchPolicy::Stages>{}),
        cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideA>(),
                            cute::Step<cute::_2, cute::_1, cute::_3>,
                            cute::Step<cute::_1, cute::_2, cute::_3>>{}));
    using SmemLayoutB = decltype(cute::tile_to_shape(
        SmemLayoutAtomB{},
        cute::make_shape(cute::shape<1>(TileShape{}), cute::shape<2>(TileShape{}),
                         cute::Int<DispatchPolicy::Stages>{}),
        cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideB>(),
                            cute::Step<cute::_2, cute::_1, cute::_3>,
                            cute::Step<cute::_1, cute::_2, cute::_3>>{}));

    static_assert(DispatchPolicy::Stages >= 2,
                  "Specialization requires Stages set to value 2 or more.");
    static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value
                      && cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                  "MMA atom must source both A and B operand from smem_desc for this mainloop.");
    static_assert(cute::is_same_v<GmemTiledCopyA, cute::SM90_TMA_LOAD>
                      || cute::is_same_v<GmemTiledCopyA, cute::SM90_TMA_LOAD_MULTICAST>,
                  "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
    static_assert(cute::is_same_v<GmemTiledCopyB, cute::SM90_TMA_LOAD>
                      || cute::is_same_v<GmemTiledCopyB, cute::SM90_TMA_LOAD_MULTICAST>,
                  "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

    // TMA converts f32 input to tf32 when copying from GMEM to SMEM
    // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
    static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
    static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
    using InternalElementA = cute::conditional_t<ConvertF32toTF32A, cutlass::tfloat32_t,
                                                 cute::uint_bit_t<cutlass::sizeof_bits_v<ElementA>>>;
    using InternalElementB = cute::conditional_t<ConvertF32toTF32B, cutlass::tfloat32_t,
                                                 cute::uint_bit_t<cutlass::sizeof_bits_v<ElementB>>>;

    struct SharedStorage {
        struct TensorStorage : cute::aligned_struct<128, cute::_0> {
            cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B0;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B1;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B2;
            cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B3;
        } tensors;

        using PipelineStorage = typename MainloopPipeline::SharedStorage;
        PipelineStorage pipeline;
    };
    using TensorStorage = typename SharedStorage::TensorStorage;
    using PipelineStorage = typename SharedStorage::PipelineStorage;

    // Host side kernel arguments
    struct Arguments {
        ElementA const* ptr_A;
        StrideA dA;
        ElementB const* ptr_B0;
        ElementB const* ptr_B1;
        ElementB const* ptr_B2;
        ElementB const* ptr_B3;
        StrideB dB;
        uint32_t mma_promotion_interval = 4;
    };

    // Device side kernel params
    struct Params {
        // Assumption: StrideA is congruent with Problem_MK
        using TMA_A = decltype(cute::make_tma_copy_A_sm90(
            GmemTiledCopyA{},
            cute::make_tensor(static_cast<InternalElementA const*>(nullptr),
                              repeat_like(StrideA{}, int32_t(0)), StrideA{}),
            SmemLayoutA{}(cute::_, cute::_, cute::_0{}), TileShape{}, ClusterShape{}));
        // Assumption: StrideB is congruent with Problem_NK
        using TMA_B = decltype(cute::make_tma_copy_B_sm90(
            GmemTiledCopyB{},
            cute::make_tensor(static_cast<InternalElementB const*>(nullptr),
                              repeat_like(StrideB{}, int32_t(0)), StrideB{}),
            SmemLayoutB{}(cute::_, cute::_, cute::_0{}), TileShape{}, ClusterShape{}));
        TMA_A tma_load_a;
        TMA_B tma_load_b0;
        TMA_B tma_load_b1;
        TMA_B tma_load_b2;
        TMA_B tma_load_b3;

        uint32_t tma_transaction_bytes = TmaTransactionBytes;
        uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
        uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    };

    //
    // Methods
    //

    template <class ProblemShape>
    static bool can_implement(ProblemShape const& problem_shape, [[maybe_unused]] Arguments const& args)
    {
        constexpr int tma_alignment_bits = 128;
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        bool implementable = true;
        constexpr int min_tma_aligned_elements_A =
            tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
        implementable = implementable
                        && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                            cute::make_shape(M, K, L), StrideA{});
        constexpr int min_tma_aligned_elements_B =
            tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
        implementable = implementable
                        && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                            cute::make_shape(N, K, L), StrideB{});

        if (!implementable) {
            CUTLASS_TRACE_HOST(
                "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
                "TMA.\n");
        }
        return implementable;
    }

    static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
    static constexpr int K_PIPE_MMAS = 1;
    static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(cute::size<0>(SmemLayoutA{}) * cute::size<1>(SmemLayoutA{})
                               * static_cast<uint32_t>(cutlass::sizeof_bits<ElementA>::value));
    static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(cute::size<0>(SmemLayoutB{}) * cute::size<1>(SmemLayoutB{})
                               * static_cast<uint32_t>(cutlass::sizeof_bits<ElementB>::value));
    static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK * 4;

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Producer Perspective
    template <class TensorA, class TensorB, class KTileIterator, class BlockCoord>
    CUTLASS_DEVICE void
    load(Params const& mainloop_params, MainloopPipeline pipeline, PipelineState smem_pipe_write,
         cute::tuple<TensorA, TensorB, TensorB, TensorB, TensorB> const& load_inputs,
         BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
         uint32_t block_rank_in_cluster, TensorStorage& shared_tensors)
    {
        int lane_predicate = cute::elect_one_sync();

        if (lane_predicate) {
            cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_A.data()),
                                                SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
            cute::Tensor sB0 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B0.data()),
                                                 SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
            cute::Tensor sB1 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B1.data()),
                                                 SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
            cute::Tensor sB2 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B2.data()),
                                                 SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
            cute::Tensor sB3 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B3.data()),
                                                 SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

            //
            // Prepare the TMA loads for A and B
            //

            constexpr uint32_t cluster_shape_x = cute::get<0>(typename DispatchPolicy::ClusterShape());
            uint2 cluster_local_block_id = { block_rank_in_cluster % cluster_shape_x,
                                             block_rank_in_cluster / cluster_shape_x };

            cute::Tensor gA_mkl = cute::get<0>(load_inputs);
            cute::Tensor gB0_nkl = cute::get<1>(load_inputs);
            cute::Tensor gB1_nkl = cute::get<2>(load_inputs);
            cute::Tensor gB2_nkl = cute::get<3>(load_inputs);
            cute::Tensor gB3_nkl = cute::get<4>(load_inputs);

            auto block_tma_a = mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
            auto block_tma_b0 = mainloop_params.tma_load_b0.get_slice(cluster_local_block_id.x);
            auto block_tma_b1 = mainloop_params.tma_load_b1.get_slice(cluster_local_block_id.x);
            auto block_tma_b2 = mainloop_params.tma_load_b2.get_slice(cluster_local_block_id.x);
            auto block_tma_b3 = mainloop_params.tma_load_b3.get_slice(cluster_local_block_id.x);

            // Partition the inputs based on the current block coordinates.
            auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
            cute::Tensor gA = gA_mkl(cute::_, cute::_, m_coord, cute::_, l_coord);  // (BLK_M,BLK_K,k)
            cute::Tensor gB0 = gB0_nkl(cute::_, cute::_, n_coord, cute::_, l_coord);  // (BLK_N,BLK_K,k)
            cute::Tensor gB1 = gB1_nkl(cute::_, cute::_, n_coord, cute::_, l_coord);  // (BLK_N,BLK_K,k)
            cute::Tensor gB2 = gB2_nkl(cute::_, cute::_, n_coord, cute::_, l_coord);  // (BLK_N,BLK_K,k)
            cute::Tensor gB3 = gB3_nkl(cute::_, cute::_, n_coord, cute::_, l_coord);  // (BLK_N,BLK_K,k)

            // Applies the mapping from block_tma_a
            cute::Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA,TMA_M,TMA_K,k)
            cute::Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA,TMA_M,TMA_K,PIPE)

            cute::Tensor tBgB0 = block_tma_b0.partition_S(gB0);  // (TMA,TMA_N,TMA_K,k)
            cute::Tensor tBsB0 = block_tma_b0.partition_D(sB0);  // (TMA,TMA_N,TMA_K,PIPE)
            cute::Tensor tBgB1 = block_tma_b1.partition_S(gB1);  // (TMA,TMA_N,TMA_K,k)
            cute::Tensor tBsB1 = block_tma_b1.partition_D(sB1);  // (TMA,TMA_N,TMA_K,PIPE)
            cute::Tensor tBgB2 = block_tma_b2.partition_S(gB2);  // (TMA,TMA_N,TMA_K,k)
            cute::Tensor tBsB2 = block_tma_b2.partition_D(sB2);  // (TMA,TMA_N,TMA_K,PIPE)
            cute::Tensor tBgB3 = block_tma_b3.partition_S(gB3);  // (TMA,TMA_N,TMA_K,k)
            cute::Tensor tBsB3 = block_tma_b3.partition_D(sB3);  // (TMA,TMA_N,TMA_K,PIPE)

            uint16_t mcast_mask_a = 0;
            uint16_t mcast_mask_b = 0;

            // Issue TmaLoads
            // Maps the tile -> block, value
            if constexpr (cute::is_same_v<GmemTiledCopyA, cute::SM90_TMA_LOAD_MULTICAST>) {
                auto block_layout =
                    cute::Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
                for (int n = 0; n < cute::size<1>(block_layout); ++n) {
                    mcast_mask_a |=
                        (uint16_t(1) << block_layout(cluster_local_block_id.x, n, cute::_0{}));
                }
            }

            if constexpr (cute::is_same_v<GmemTiledCopyB, cute::SM90_TMA_LOAD_MULTICAST>) {
                auto block_layout =
                    cute::Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
                for (int m = 0; m < cute::size<0>(block_layout); ++m) {
                    mcast_mask_b |=
                        (uint16_t(1) << block_layout(m, cluster_local_block_id.y, cute::_0{}));
                }
            }

            // Mainloop
            CUTLASS_PRAGMA_NO_UNROLL
            for (; k_tile_count > 0; --k_tile_count) {
                // LOCK smem_pipe_write for _writing_
                pipeline.producer_acquire(smem_pipe_write);

                //
                // Copy gmem to smem for *k_tile_iter
                //

                using BarrierType = typename MainloopPipeline::ProducerBarrierType;
                BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

                int write_stage = smem_pipe_write.index();
                copy(mainloop_params.tma_load_a.with(*tma_barrier, mcast_mask_a),
                     tAgA(cute::_, cute::_, cute::_, *k_tile_iter),
                     tAsA(cute::_, cute::_, cute::_, write_stage));
                copy(mainloop_params.tma_load_b0.with(*tma_barrier, mcast_mask_b),
                     tBgB0(cute::_, cute::_, cute::_, *k_tile_iter),
                     tBsB0(cute::_, cute::_, cute::_, write_stage));
                copy(mainloop_params.tma_load_b1.with(*tma_barrier, mcast_mask_b),
                     tBgB1(cute::_, cute::_, cute::_, *k_tile_iter),
                     tBsB1(cute::_, cute::_, cute::_, write_stage));
                copy(mainloop_params.tma_load_b2.with(*tma_barrier, mcast_mask_b),
                     tBgB2(cute::_, cute::_, cute::_, *k_tile_iter),
                     tBsB2(cute::_, cute::_, cute::_, write_stage));
                copy(mainloop_params.tma_load_b3.with(*tma_barrier, mcast_mask_b),
                     tBgB3(cute::_, cute::_, cute::_, *k_tile_iter),
                     tBsB3(cute::_, cute::_, cute::_, write_stage));

                ++k_tile_iter;

                // Advance smem_pipe_write
                ++smem_pipe_write;
            }
        }
    }

    /// Set up the data needed by this collective for load and mma.
    /// Returns a tuple of tensors. The collective and the kernel layer have the contract
    /// Returned tuple must contain at least two elements, with the first two elements being:
    /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
    /// gB0_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
    /// gB1_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
    /// gB2_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
    /// gB3_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)

    /// The rest of the tensors can be specified as needed by this collective.
    template <class ProblemShape_MNKL>
    CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                  Params const& mainloop_params) const
    {
        // Separate out problem shape for convenience
        auto [M, N, K, L] = problem_shape_MNKL;

        // TMA requires special handling of strides to deal with coord codomain mapping
        // Represent the full tensors -- get these from TMA
        cute::Tensor mA_mkl =
            mainloop_params.tma_load_a.get_tma_tensor(cute::make_shape(M, K, L));  // (m,k,l)
        cute::Tensor mB0_nkl =
            mainloop_params.tma_load_b0.get_tma_tensor(cute::make_shape(N, K, L));  // (n,k,l)
        cute::Tensor mB1_nkl =
            mainloop_params.tma_load_b1.get_tma_tensor(cute::make_shape(N, K, L));  // (n,k,l)
        cute::Tensor mB2_nkl =
            mainloop_params.tma_load_b2.get_tma_tensor(cute::make_shape(N, K, L));  // (n,k,l)
        cute::Tensor mB3_nkl =
            mainloop_params.tma_load_b3.get_tma_tensor(cute::make_shape(N, K, L));  // (n,k,l)

        // Make tiled views, defer the slice
        cute::Tensor gA_mkl =
            cute::local_tile(mA_mkl, TileShape{}, make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::_1, cute::Underscore, cute::_1>{});  // (BLK_M,BLK_K,m,k,l)
        cute::Tensor gB0_nkl =
            cute::local_tile(mB0_nkl, TileShape{}, make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::Underscore, cute::_1, cute::_1>{});  // (BLK_N,BLK_K,n,k,l)
        cute::Tensor gB1_nkl =
            cute::local_tile(mB1_nkl, TileShape{}, make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::Underscore, cute::_1, cute::_1>{});  // (BLK_N,BLK_K,n,k,l)
        cute::Tensor gB2_nkl =
            cute::local_tile(mB2_nkl, TileShape{}, make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::Underscore, cute::_1, cute::_1>{});  // (BLK_N,BLK_K,n,k,l)
        cute::Tensor gB3_nkl =
            cute::local_tile(mB3_nkl, TileShape{}, make_coord(cute::_, cute::_, cute::_),
                             cute::Step<cute::Underscore, cute::_1, cute::_1>{});  // (BLK_N,BLK_K,n,k,l)

        return cute::make_tuple(gA_mkl, gB0_nkl, gB1_nkl, gB2_nkl, gB3_nkl);
    }

    /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
    CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write)
    {
        int lane_predicate = cute::elect_one_sync();

        // Issue the epilogue waits
        if (lane_predicate) {
            /* This helps avoid early exit of blocks in Cluster
             * Waits for all stages to either be released (all
             * Consumer UNLOCKs), or if the stage was never used
             * then would just be acquired since the phase was
             * still inverted from make_producer_start_state
             */
            pipeline.producer_tail(smem_pipe_write);
        }
    }

    /// Perform a collective-scoped matrix multiply-accumulate
    /// Consumer Perspective
    template <class FrgTensorC>
    CUTLASS_DEVICE void mma(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                            FrgTensorC& accum0, FrgTensorC& accum1, FrgTensorC& accum2,
                            FrgTensorC& accum3, int k_tile_count, int thread_idx,
                            TensorStorage& shared_tensors, Params const& mainloop_params)
    {
        static_assert(cute::is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
        static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
        static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
        static_assert(
            cute::is_void_v<SmemCopyAtomA>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
        static_assert(
            cute::is_void_v<SmemCopyAtomB>,
            "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

        cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_A.data()),
                                            SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
        cute::Tensor sB0 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B0.data()),
                                             SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
        cute::Tensor sB1 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B1.data()),
                                             SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
        cute::Tensor sB2 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B2.data()),
                                             SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
        cute::Tensor sB3 = cute::make_tensor(cute::make_smem_ptr(shared_tensors.smem_B3.data()),
                                             SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

        //
        // Define C accumulators and A/B partitioning
        //

        // Layout of warp group to thread mapping

        static_assert(
            cute::stride<0>(typename TiledMma::ALayout{}) == 0
                and cute::stride<0>(typename TiledMma::BLayout{}) == 0
                and cute::size<0>(typename TiledMma::ALayout{}) == cutlass::NumThreadsPerWarpGroup
                and cute::size<0>(typename TiledMma::BLayout{}) == cutlass::NumThreadsPerWarpGroup,
            "Stride of the first mode must be 0 and the size of the mode must be "
            "cutlass::NumThreadsPerWarpGroup");

        constexpr int MmaWarpGroups = cute::size(TiledMma{}) / cutlass::NumThreadsPerWarpGroup;
        cute::Layout warp_group_thread_layout = cute::make_layout(
            cute::Int<MmaWarpGroups>{}, cute::Int<cutlass::NumThreadsPerWarpGroup>{});

        int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / cutlass::NumThreadsPerWarpGroup, 0);

        TiledMma tiled_mma;
        auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

        cute::Tensor tCsA = thread_mma.partition_A(sA);    // (MMA,MMA_M,MMA_K,PIPE)
        cute::Tensor tCsB0 = thread_mma.partition_B(sB0);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCsB1 = thread_mma.partition_B(sB1);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCsB2 = thread_mma.partition_B(sB2);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCsB3 = thread_mma.partition_B(sB3);  // (MMA,MMA_N,MMA_K,PIPE)

        // Allocate "fragments/descriptors"
        cute::Tensor tCrA = thread_mma.make_fragment_A(tCsA);    // (MMA,MMA_M,MMA_K,PIPE)
        cute::Tensor tCrB0 = thread_mma.make_fragment_B(tCsB0);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCrB1 = thread_mma.make_fragment_B(tCsB1);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCrB2 = thread_mma.make_fragment_B(tCsB2);  // (MMA,MMA_N,MMA_K,PIPE)
        cute::Tensor tCrB3 = thread_mma.make_fragment_B(tCsB3);  // (MMA,MMA_N,MMA_K,PIPE)

        CUTE_STATIC_ASSERT_V(cute::size<1>(tCsA) == cute::size<1>(accum0));   // M
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB0) == cute::size<2>(accum0));  // N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB1) == cute::size<2>(accum1));  // N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB2) == cute::size<2>(accum2));  // N
        CUTE_STATIC_ASSERT_V(cute::size<1>(tCsB3) == cute::size<2>(accum3));  // N

        CUTE_STATIC_ASSERT_V(cute::size<2>(tCsA) == cute::size<2>(tCsB0));  // K
        CUTE_STATIC_ASSERT_V(cute::size<2>(tCsA) == cute::size<2>(tCsB1));  // K
        CUTE_STATIC_ASSERT_V(cute::size<2>(tCsA) == cute::size<2>(tCsB2));  // K
        CUTE_STATIC_ASSERT_V(cute::size<2>(tCsA) == cute::size<2>(tCsB3));  // K

        CUTE_STATIC_ASSERT_V(cute::size<3>(tCsA) == cute::size<3>(tCsB0));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::size<3>(tCsA) == cute::size<3>(tCsB1));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::size<3>(tCsA) == cute::size<3>(tCsB2));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::size<3>(tCsA) == cute::size<3>(tCsB3));  // PIPE

        CUTE_STATIC_ASSERT_V(cute::Int<DispatchPolicy::Stages>{} == cute::size<2>(sA));   // PIPE
        CUTE_STATIC_ASSERT_V(cute::Int<DispatchPolicy::Stages>{} == cute::size<2>(sB0));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::Int<DispatchPolicy::Stages>{} == cute::size<2>(sB1));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::Int<DispatchPolicy::Stages>{} == cute::size<2>(sB2));  // PIPE
        CUTE_STATIC_ASSERT_V(cute::Int<DispatchPolicy::Stages>{} == cute::size<2>(sB3));  // PIPE

        //
        // PIPELINED MAIN LOOP
        //
        static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS < K_PIPE_MAX),
                      "ERROR : Incorrect number of MMAs in flight");

        // We release buffers to producer warps(dma load) with some mmas in flight
        cutlass::PipelineState smem_pipe_release = smem_pipe_read;

        // Prologue GMMAs
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
        assert(k_tile_count >= 1);
        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
        cute::warpgroup_fence_operand(accum0);
        cute::warpgroup_fence_operand(accum1);
        cute::warpgroup_fence_operand(accum2);
        cute::warpgroup_fence_operand(accum3);
        {
            // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            int read_stage = smem_pipe_read.index();
            cute::warpgroup_arrive();

            // Unroll the K mode manually to set scale D to 1 (B0)
            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB0(cute::_, cute::_, k_block, read_stage), accum0);
                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
            }

            // Unroll the K mode manually to set scale D to 1 (B1)
            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB1(cute::_, cute::_, k_block, read_stage), accum1);
                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
            }

            // Unroll the K mode manually to set scale D to 1 (B2)
            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB2(cute::_, cute::_, k_block, read_stage), accum2);
                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
            }

            // Unroll the K mode manually to set scale D to 1 (B3)
            tiled_mma.accumulate_ = cute::GMMA::ScaleOut::Zero;
            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                // (V,M,K) x (V,N,K) => (V,M,N)
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB3(cute::_, cute::_, k_block, read_stage), accum3);
                tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;
            }

            cute::warpgroup_commit_batch();

            ++smem_pipe_read;
        }

        tiled_mma.accumulate_ = cute::GMMA::ScaleOut::One;

        cute::warpgroup_fence_operand(accum0);
        cute::warpgroup_fence_operand(accum1);
        cute::warpgroup_fence_operand(accum2);
        cute::warpgroup_fence_operand(accum3);

        CUTLASS_PRAGMA_UNROLL
        for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue) {
            // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            int read_stage = smem_pipe_read.index();
            cute::warpgroup_arrive();
            // (V,M,K) x (V,N,K) => (V,M,N)
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, cute::_, read_stage),
                       tCrB0(cute::_, cute::_, cute::_, read_stage), accum0);
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, cute::_, read_stage),
                       tCrB1(cute::_, cute::_, cute::_, read_stage), accum1);
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, cute::_, read_stage),
                       tCrB2(cute::_, cute::_, cute::_, read_stage), accum2);
            cute::gemm(tiled_mma, tCrA(cute::_, cute::_, cute::_, read_stage),
                       tCrB3(cute::_, cute::_, cute::_, read_stage), accum3);

            cute::warpgroup_commit_batch();

            ++smem_pipe_read;
        }

        cute::warpgroup_fence_operand(accum0);
        cute::warpgroup_fence_operand(accum1);
        cute::warpgroup_fence_operand(accum2);
        cute::warpgroup_fence_operand(accum3);

        // Mainloop GMMAs
        k_tile_count -= prologue_mma_count;

        CUTLASS_PRAGMA_NO_UNROLL
        for (; k_tile_count > 0; --k_tile_count) {
            // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
            auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            //
            // Compute on k_tile
            //

            int read_stage = smem_pipe_read.index();
            cute::warpgroup_fence_operand(accum0);
            cute::warpgroup_fence_operand(accum1);
            cute::warpgroup_fence_operand(accum2);
            cute::warpgroup_fence_operand(accum3);

            cute::warpgroup_arrive();
            // (V,M,K) x (V,N,K) => (V,M,N)

            CUTLASS_PRAGMA_UNROLL
            for (int k_block = 0; k_block < cute::size<2>(tCrA); ++k_block) {
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB0(cute::_, cute::_, k_block, read_stage), accum0);
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB1(cute::_, cute::_, k_block, read_stage), accum1);
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB2(cute::_, cute::_, k_block, read_stage), accum2);
                cute::gemm(tiled_mma, tCrA(cute::_, cute::_, k_block, read_stage),
                           tCrB3(cute::_, cute::_, k_block, read_stage), accum3);
            }
            cute::warpgroup_commit_batch();

            /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
            cute::warpgroup_wait<K_PIPE_MMAS>();
            cute::warpgroup_fence_operand(accum0);
            cute::warpgroup_fence_operand(accum1);
            cute::warpgroup_fence_operand(accum2);
            cute::warpgroup_fence_operand(accum3);

            // UNLOCK smem_pipe_release, done _computing_ on it
            pipeline.consumer_release(smem_pipe_release);

            // Advance smem_pipe_read and smem_pipe_release
            ++smem_pipe_read;
            ++smem_pipe_release;
        }

        cute::warpgroup_fence_operand(accum0);
        cute::warpgroup_fence_operand(accum1);
        cute::warpgroup_fence_operand(accum2);
        cute::warpgroup_fence_operand(accum3);
    }

    /// Perform a Consumer Epilogue to release all buffers
    CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release,
                                 int k_tile_count)
    {
        // Prologue GMMAs
        int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
        k_tile_count -= prologue_mma_count;

        smem_pipe_release.advance(k_tile_count);

        // Wait on all GMMAs to complete
        cute::warpgroup_wait<0>();

        for (int count = 0; count < prologue_mma_count; ++count) {
            pipeline.consumer_release(
                smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
            ++smem_pipe_release;
        }
    }

    /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
    CUTLASS_DEVICE
    static void prefetch_tma_descriptors(Params const& mainloop_params)
    {
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b0.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b1.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b2.get_tma_descriptor());
        cute::prefetch_tma_descriptor(mainloop_params.tma_load_b3.get_tma_descriptor());
    }

    template <class ProblemShape>
    static constexpr Params to_underlying_arguments(ProblemShape const& problem_shape,
                                                    Arguments const& args, void* workspace)
    {
        (void)workspace;

        // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
        auto problem_shape_MNKL = cute::append<4>(problem_shape, 1);
        auto [M, N, K, L] = problem_shape_MNKL;

        auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
        auto ptr_B0 = reinterpret_cast<InternalElementB const*>(args.ptr_B0);
        auto ptr_B1 = reinterpret_cast<InternalElementB const*>(args.ptr_B1);
        auto ptr_B2 = reinterpret_cast<InternalElementB const*>(args.ptr_B2);
        auto ptr_B3 = reinterpret_cast<InternalElementB const*>(args.ptr_B3);

        cute::Tensor tensor_a =
            cute::make_tensor(ptr_A, cute::make_layout(cute::make_shape(M, K, L), args.dA));
        cute::Tensor tensor_b0 =
            cute::make_tensor(ptr_B0, cute::make_layout(cute::make_shape(N, K, L), args.dB));
        cute::Tensor tensor_b1 =
            cute::make_tensor(ptr_B1, cute::make_layout(cute::make_shape(N, K, L), args.dB));
        cute::Tensor tensor_b2 =
            cute::make_tensor(ptr_B2, cute::make_layout(cute::make_shape(N, K, L), args.dB));
        cute::Tensor tensor_b3 =
            cute::make_tensor(ptr_B3, cute::make_layout(cute::make_shape(N, K, L), args.dB));

        typename Params::TMA_A tma_load_a = cute::make_tma_copy_A_sm90(
            GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(cute::_, cute::_, cute::_0{}), TileShape{},
            ClusterShape{});
        typename Params::TMA_B tma_load_b0 = cute::make_tma_copy_B_sm90(
            GmemTiledCopyB{}, tensor_b0, SmemLayoutB{}(cute::_, cute::_, cute::_0{}), TileShape{},
            ClusterShape{});
        typename Params::TMA_B tma_load_b1 = cute::make_tma_copy_B_sm90(
            GmemTiledCopyB{}, tensor_b1, SmemLayoutB{}(cute::_, cute::_, cute::_0{}), TileShape{},
            ClusterShape{});
        typename Params::TMA_B tma_load_b2 = cute::make_tma_copy_B_sm90(
            GmemTiledCopyB{}, tensor_b2, SmemLayoutB{}(cute::_, cute::_, cute::_0{}), TileShape{},
            ClusterShape{});
        typename Params::TMA_B tma_load_b3 = cute::make_tma_copy_B_sm90(
            GmemTiledCopyB{}, tensor_b3, SmemLayoutB{}(cute::_, cute::_, cute::_0{}), TileShape{},
            ClusterShape{});
        uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
        uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
        uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk * 4;

        return { tma_load_a,  tma_load_b0,       tma_load_b1,          tma_load_b2,
                 tma_load_b3, transaction_bytes, transaction_bytes_mk, transaction_bytes_nk };
    }
};
