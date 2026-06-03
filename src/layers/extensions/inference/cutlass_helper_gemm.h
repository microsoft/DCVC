// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/trace.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
// clang-format on

/**
 * These 3 classes/structs are rewritten:
 *   1. cutlass::gemm::kernel::GemmWithFusedEpilogue -> CustomGemmWithFusedEpilogue
 *   2. cutlass::gemm::kernel::DefaultGemmWithBroadcast -> CustomDefaultGemmWithBroadcast
 *   3. cutlass::gemm::device::GemmUniversalWithBroadcast -> CustomGemmUniversalWithBroadcast
 */

/// Original file: cutlass/gemm/kernel/gemm_with_fused_epilogue.h
/// to add PDL sync points
/// depending on IsSingleSource, there are 2 specializations
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, bool IsSingleSource = Epilogue_::kIsSingleSource>
struct CustomGemmWithFusedEpilogue;

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct CustomGemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, false> {
    using Base =
        cutlass::gemm::kernel::GemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, false>;
    using Mma = typename Base::Mma;
    using Epilogue = typename Base::Epilogue;
    using EpilogueOutputOp = typename Base::EpilogueOutputOp;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;
    using Arguments = typename Base::Arguments;

    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;

    static cutlass::ComplexTransform const kTransformA = Base::kTransformA;
    static cutlass::ComplexTransform const kTransformB = Base::kTransformB;
    static int const kThreadCount = Base::kThreadCount;
    using Operator = typename Base::Operator;

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
            cudaGDC();
            return;
        }

        int offset_k = 0;
        int problem_size_k = params.problem_size.k();

        ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
        ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * Mma::Shape::kM,
            offset_k,
        };

        cutlass::MatrixCoord tb_offset_B{ offset_k, threadblock_tile_offset.n() * Mma::Shape::kN };

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params.params_A, ptr_A,
                                           { params.problem_size.m(), problem_size_k }, thread_idx,
                                           tb_offset_A);

        typename Mma::IteratorB iterator_B(params.params_B, ptr_B,
                                           { problem_size_k, params.problem_size.n() }, thread_idx,
                                           tb_offset_B);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

        int lane_idx = threadIdx.x % 32;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        cudaGDC();
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        //
        // Masked tile iterators constructed from members
        //

        threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // assume identity swizzle
        cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                                threadblock_tile_offset.n() * Mma::Shape::kN);

        int block_idx =
            threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

        ElementC* ptr_C1 = static_cast<ElementC*>(params.ptr_C1);
        ElementC* ptr_C2 = static_cast<ElementC*>(params.ptr_C2);
        ElementC* ptr_D = static_cast<ElementC*>(params.ptr_D);
        typename Epilogue::ElementTensor* ptr_Tensor =
            static_cast<typename Epilogue::ElementTensor*>(params.ptr_Tensor);

        // Define the reduction output pointer and move to the appropriate place
        typename Epilogue::ElementVector* ptr_Vector =
            static_cast<typename Epilogue::ElementVector*>(params.ptr_Vector);

        // Tile iterators loading from source tensors.
        typename Epilogue::OutputTileIterator iterator_C1(
            params.params_C1, ptr_C1, params.problem_size.mn(), thread_idx, threadblock_offset);

        typename Epilogue::OutputTileIterator iterator_C2(
            params.params_C2, ptr_C2, params.problem_size.mn(), thread_idx, threadblock_offset);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params.params_D, ptr_D, params.problem_size.mn(), thread_idx, threadblock_offset);

        // Additional tensor to load from
        typename Epilogue::TensorTileIterator tensor_iterator(params.params_Tensor,
                                                              // Only the final block outputs Tensor
                                                              ptr_Tensor, params.problem_size.mn(),
                                                              thread_idx, threadblock_offset);

        // Construct the epilogue
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Move to appropriate location for this output tile
        if (ptr_Vector) {
            ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
        }

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, ptr_Vector, iterator_D, accumulators, iterator_C1, iterator_C2,
                 tensor_iterator, params.problem_size.mn(), threadblock_offset);
    }

    // Factory invocation
    CUTLASS_DEVICE
    static void invoke(Params const& params, SharedStorage& shared_storage)
    {
        CustomGemmWithFusedEpilogue op;
        op(params, shared_storage);
    }

    static cutlass::Status can_implement(Arguments const& args)
    {
        return Base::can_implement(args);
    }
};

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct CustomGemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, true> {
    using Base =
        cutlass::gemm::kernel::GemmWithFusedEpilogue<Mma_, Epilogue_, ThreadblockSwizzle_, true>;
    using Mma = typename Base::Mma;
    using Epilogue = typename Base::Epilogue;
    using EpilogueOutputOp = typename Base::EpilogueOutputOp;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;
    using Arguments = typename Base::Arguments;

    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;

    static cutlass::ComplexTransform const kTransformA = Base::kTransformA;
    static cutlass::ComplexTransform const kTransformB = Base::kTransformB;
    static int const kThreadCount = Base::kThreadCount;
    using Operator = typename Base::Operator;

    /// Executes one GEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_offset =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
            cudaGDC();
            return;
        }

        int offset_k = 0;
        int problem_size_k = params.problem_size.k();

        ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
        ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

        // Compute initial location in logical coordinates
        cutlass::MatrixCoord tb_offset_A{
            threadblock_tile_offset.m() * Mma::Shape::kM,
            offset_k,
        };

        cutlass::MatrixCoord tb_offset_B{ offset_k, threadblock_tile_offset.n() * Mma::Shape::kN };

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(params.params_A, ptr_A,
                                           { params.problem_size.m(), problem_size_k }, thread_idx,
                                           tb_offset_A);

        typename Mma::IteratorB iterator_B(params.params_B, ptr_B,
                                           { problem_size_k, params.problem_size.n() }, thread_idx,
                                           tb_offset_B);

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = cutlass::canonical_warp_idx_sync();

        int lane_idx = threadIdx.x % 32;

        //
        // Main loop
        //

        // Construct thread-scoped matrix multiply
        Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Compute threadblock-scoped matrix multiply-add
        int gemm_k_iterations = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

        // Compute threadblock-scoped matrix multiply-add
        cudaGDC();
        mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        //
        // Masked tile iterators constructed from members
        //

        threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // assume identity swizzle
        cutlass::MatrixCoord threadblock_offset(threadblock_tile_offset.m() * Mma::Shape::kM,
                                                threadblock_tile_offset.n() * Mma::Shape::kN);

        int block_idx =
            threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

        ElementC* ptr_C = static_cast<ElementC*>(params.ptr_C);
        ElementC* ptr_D = static_cast<ElementC*>(params.ptr_D);
        typename Epilogue::ElementTensor* ptr_Tensor =
            static_cast<typename Epilogue::ElementTensor*>(params.ptr_Tensor);

        // Define the reduction output pointer and move to the appropriate place
        typename Epilogue::ElementVector* ptr_Vector =
            static_cast<typename Epilogue::ElementVector*>(params.ptr_Vector);

        // Tile iterators loading from source tensors.
        typename Epilogue::OutputTileIterator iterator_C(
            params.params_C, ptr_C, params.problem_size.mn(), thread_idx, threadblock_offset);

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D(
            params.params_D, ptr_D, params.problem_size.mn(), thread_idx, threadblock_offset);

        // Additional tensor to load from
        typename Epilogue::TensorTileIterator tensor_iterator(params.params_Tensor,
                                                              // Only the final block outputs Tensor
                                                              ptr_Tensor, params.problem_size.mn(),
                                                              thread_idx, threadblock_offset);

        // Construct the epilogue
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Move to appropriate location for this output tile
        if (ptr_Vector) {
            ptr_Vector += threadblock_offset.column() + threadblock_tile_offset.m() * params.ldr;
        }

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, ptr_Vector, iterator_D, accumulators, iterator_C, tensor_iterator,
                 params.problem_size.mn(), threadblock_offset);
    }

    // Factory invocation
    CUTLASS_DEVICE
    static void invoke(Params const& params, SharedStorage& shared_storage)
    {
        CustomGemmWithFusedEpilogue op;
        op(params, shared_storage);
    }

    static cutlass::Status can_implement(Arguments const& args)
    {
        return Base::can_implement(args);
    }
};

/// Original file: cutlass/gemm/kernel/default_gemm_with_broadcast.h
/// to invoke CustomGemmWithFusedEpilogue
template <typename ElementA_, typename LayoutA_, cutlass::ComplexTransform TransformA,
          int AlignmentA, typename ElementB_, typename LayoutB_, cutlass::ComplexTransform TransformB,
          int AlignmentB, typename ElementC_, typename LayoutC_, typename ElementAccumulator_,
          typename OperatorClass_, typename ArchTag_, typename ThreadblockShape_,
          typename WarpShape_, typename InstructionShape_, typename EpilogueOutputOp_,
          typename ThreadblockSwizzle_, int Stages, typename Operator_, typename Enable = void>
struct CustomDefaultGemmWithBroadcast {
    using Base = typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
        ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_, LayoutB_, TransformB, AlignmentB,
        ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
        WarpShape_, InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, Operator_>;
    using GemmKernel =
        CustomGemmWithFusedEpilogue<typename Base::GemmBase::Mma, typename Base::Epilogue, ThreadblockSwizzle_>;
};

/// Original file: cutlass/gemm/device/gemm_universal_with_broadcast.h
/// to add "cudaLaunchKernelEx" to run()
template <typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_,
          typename ElementC_, typename LayoutC_, typename ElementAccumulator_, typename OperatorClass_,
          typename ArchTag_, typename ThreadblockShape_, typename WarpShape_, typename InstructionShape_,
          typename EpilogueOutputOp_, typename ThreadblockSwizzle_, int Stages,
          int AlignmentA = cutlass::gemm::device::DefaultGemmConfiguration<
              OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentA,
          int AlignmentB = cutlass::gemm::device::DefaultGemmConfiguration<
              OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::kAlignmentB,
          typename Operator_ = typename cutlass::gemm::device::DefaultGemmConfiguration<
              OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_, ElementAccumulator_>::Operator,
          cutlass::ComplexTransform TransformA = cutlass::ComplexTransform::kNone,
          cutlass::ComplexTransform TransformB = cutlass::ComplexTransform::kNone>
class CustomGemmUniversalWithBroadcast
    : public cutlass::gemm::device::GemmUniversalBase<typename CustomDefaultGemmWithBroadcast<
          ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_, LayoutB_, TransformB, AlignmentB,
          ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_,
          InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, Operator_>::GemmKernel> {
public:
    using Base = cutlass::gemm::device::GemmUniversalBase<typename CustomDefaultGemmWithBroadcast<
        ElementA_, LayoutA_, TransformA, AlignmentA, ElementB_, LayoutB_, TransformB, AlignmentB,
        ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_,
        InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, Operator_>::GemmKernel>;

    using Arguments = typename Base::Arguments;
    using GemmKernel = typename Base::GemmKernel;

    static constexpr size_t kSharedStorageSize = Base::kSharedStorageSize;

    cutlass::Status run(cudaStream_t stream = nullptr)
    {
        CUTLASS_TRACE_HOST("GemmUniversalBase::run()");

        // Configure grid and block dimensions
        dim3 block(GemmKernel::kThreadCount, 1, 1);
        dim3 grid = this->params_.get_grid_dims();

        // Launch kernel
        CUTLASS_TRACE_HOST(
            "  "
            "grid: ("
            << grid
            << "), "
               "block: ("
            << block
            << "), "
               "SMEM: ("
            << Base::kSharedStorageSize << ")");

        cutlass::arch::synclog_setup();

        cudaLaunchConfig_t config;
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = kSharedStorageSize;
        config.stream = stream;
        auto attr = get_cuda_launch_attribute();
        config.attrs = &attr;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, &cutlass::Kernel2<GemmKernel>, this->params_);

        // Query for errors
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) {
            CUTLASS_TRACE_HOST("  grid launch failed with error " << cudaGetErrorString(result));
            return cutlass::Status::kErrorInternal;
        }

        return cutlass::Status::kSuccess;
    }

    cutlass::Status operator()(Arguments const& args, void* workspace = nullptr,
                               cudaStream_t stream = nullptr)
    {
        cutlass::Status status = this->initialize(args, workspace, stream);

        if (status == cutlass::Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }
};
