// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h>
#include <cutlass/conv/kernel/implicit_gemm_convolution_strided_dgrad.h>
#include <cutlass/conv/kernel/default_conv2d_dgrad.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/numeric_conversion.h>
// clang-format on

/**
 * These 5 classes/structs are rewritten:
 *   1. cutlass::conv::kernel::ImplicitGemmConvolutionWithFusedEpilogue -> CustomImplicitGemmConvolutionWithFusedEpilogue
 *   2. cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast -> CustomDefaultConv2dFpropWithBroadcast
 *   3. cutlass::conv::kernel::ImplicitGemmConvolutionStridedDgrad -> CustomImplicitGemmConvolutionStridedDgrad
 *   4. cutlass::conv::kernel::DefaultConv2dDgrad -> CustomDefaultConv2dDgrad
 *   5. cutlass::conv::device::ImplicitGemmConvolution -> CustomImplicitGemmConvolution
 */

/// Original file: cutlass/conv/kernel/implicit_gemm_convolution_with_fused_epilogue.h
/// to add PDL sync points
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          cutlass::conv::Operator ConvOperator, typename ConvProblemSize_ = cutlass::conv::Conv2dProblemSize>
struct CustomImplicitGemmConvolutionWithFusedEpilogue {
    using Base = cutlass::conv::kernel::ImplicitGemmConvolutionWithFusedEpilogue<
        Mma_, Epilogue_, ThreadblockSwizzle_, ConvOperator, ConvProblemSize_>;
    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using ElementCompute = typename Base::ElementCompute;
    using OperatorClass = typename Base::OperatorClass;
    using ArchTag = typename Base::ArchTag;
    using ThreadblockShape = typename Base::ThreadblockShape;
    using WarpShape = typename Base::WarpShape;
    using InstructionShape = typename Base::InstructionShape;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
    using EpilogueOutputOp = typename Base::EpilogueOutputOp;
    static int const kStages = Base::kStages;
    static int const kConvDim = Base::kConvDim;
    using WarpMmaOperator = typename Base::WarpMmaOperator;
    using ArchMmaOperator = typename Base::ArchMmaOperator;
    using MathOperator = typename Base::MathOperator;

    static cutlass::conv::Operator const kConvolutionalOperator = Base::kConvolutionalOperator;
    static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = Base::kIteratorAlgorithm;
    static cutlass::conv::StrideSupport const kStrideSupport = Base::kStrideSupport;
    static cutlass::conv::GroupMode const kGroupMode = Base::kGroupMode;

    using Arguments = typename Base::Arguments;
    using Mma = typename Base::Mma;
    using Epilogue = typename Base::Epilogue;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    using ConvOutputIteratorParameter = typename Base::ConvOutputIteratorParameter;

    /// Executes one ImplicitGEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_idx =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
            cudaGDC();
            return;
        }

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
            params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
            cutlass::MatrixCoord(threadblock_tile_idx.m() * Mma::Shape::kM,
                                 threadblock_tile_idx.k() * Mma::Shape::kK));

        typename Mma::IteratorB iterator_B(
            params.iterator_B, params.problem_size, params.ptr_B, thread_idx,
            cutlass::MatrixCoord(threadblock_tile_idx.k() * Mma::Shape::kK,
                                 threadblock_tile_idx.n() * Mma::Shape::kN));

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
        cudaGDC();
        mma(params.gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        // Compute logical position within grid
        threadblock_tile_idx = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        cutlass::MatrixCoord threadblock_offset(threadblock_tile_idx.m() * Mma::Shape::kM,
                                                threadblock_tile_idx.n() * Mma::Shape::kN);

        // Tile iterator writing to destination tensor
        typename Epilogue::OutputTileIterator iterator_D(
            params.iterator_D, params.ptr_D,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx, threadblock_offset);

        // Tile iterator reading from source accumulator tensor
        typename Epilogue::OutputTileIterator iterator_C(
            params.iterator_C, params.ptr_C,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx, threadblock_offset);

        typename Epilogue::ElementTensor* ptr_Tensor =
            static_cast<typename Epilogue::ElementTensor*>(params.ptr_Tensor);

        // Define the reduction output pointer and move to the appropriate place
        typename Epilogue::ElementVector* ptr_Vector =
            static_cast<typename Epilogue::ElementVector*>(params.ptr_Vector);

        // Additional tensor to load from
        typename Epilogue::TensorTileIterator tensor_iterator(
            params.params_Tensor, ptr_Tensor,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx, threadblock_offset);

        // Construct the epilogue
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Move to appropriate location for this output tile
        if (ptr_Vector) {
            ptr_Vector += threadblock_offset.column() + threadblock_tile_idx.m() * params.ldr;
        }

        // Execute the epilogue operator to update the destination tensor.
        epilogue(output_op, ptr_Vector, iterator_D, accumulators, iterator_C, tensor_iterator,
                 ConvOutputIteratorParameter::extent(params.problem_size), threadblock_offset);
    }
};

/// Original file: cutlass/conv/kernel/default_conv2d_fprop_with_broadcast.h
/// to invoke CustomImplicitGemmConvolutionWithFusedEpilogue
template <typename ElementA, typename LayoutA, typename ElementB, typename LayoutB,
          typename ElementC, typename LayoutC, typename ElementAccumulator, typename OperatorClass,
          typename ArchTag, typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp, typename ThreadblockSwizzle,
          int Stages, typename MathOperatorTag, cutlass::conv::IteratorAlgorithm IteratorAlgorithm,
          cutlass::conv::StrideSupport StrideSupport = cutlass::conv::StrideSupport::kUnity,
          int AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>,
          int AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>>
struct CustomDefaultConv2dFpropWithBroadcast {
    using Base = cutlass::conv::kernel::DefaultConv2dFpropWithBroadcast<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
        ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, IteratorAlgorithm, StrideSupport, AlignmentA, AlignmentB>;
    using Kernel =
        CustomImplicitGemmConvolutionWithFusedEpilogue<typename Base::ImplicitGemmBase::Mma,
                                                       typename Base::Epilogue, ThreadblockSwizzle,
                                                       cutlass::conv::Operator::kFprop>;
};

/// Original file: cutlass/conv/kernel/implicit_gemm_convolution_strided_dgrad.h
/// to add PDL sync points
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_,
          cutlass::conv::Operator ConvOperator, typename ConvProblemSize_ = cutlass::conv::Conv2dProblemSize>
struct CustomImplicitGemmConvolutionStridedDgrad {
    using Base =
        cutlass::conv::kernel::ImplicitGemmConvolutionStridedDgrad<Mma_, Epilogue_, ThreadblockSwizzle_,
                                                                   ConvOperator, ConvProblemSize_>;
    using ElementA = typename Base::ElementA;
    using LayoutA = typename Base::LayoutA;
    using ElementB = typename Base::ElementB;
    using LayoutB = typename Base::LayoutB;
    using ElementC = typename Base::ElementC;
    using LayoutC = typename Base::LayoutC;
    using ElementAccumulator = typename Base::ElementAccumulator;
    using ElementCompute = typename Base::ElementCompute;
    using OperatorClass = typename Base::OperatorClass;
    using ArchTag = typename Base::ArchTag;
    using ThreadblockShape = typename Base::ThreadblockShape;
    using WarpShape = typename Base::WarpShape;
    using InstructionShape = typename Base::InstructionShape;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
    using EpilogueOutputOp = typename Base::EpilogueOutputOp;
    static int const kStages = Base::kStages;
    static int const kConvDim = Base::kConvDim;
    using WarpMmaOperator = typename Base::WarpMmaOperator;
    using ArchMmaOperator = typename Base::ArchMmaOperator;
    using MathOperator = typename Base::MathOperator;

    static cutlass::conv::Operator const kConvolutionalOperator = Base::kConvolutionalOperator;
    static cutlass::conv::IteratorAlgorithm const kIteratorAlgorithm = Base::kIteratorAlgorithm;
    static cutlass::conv::StrideSupport const kStrideSupport = Base::kStrideSupport;
    static cutlass::conv::GroupMode const kGroupMode = Base::kGroupMode;

    using Arguments = typename Base::Arguments;
    using Mma = typename Base::Mma;
    using Epilogue = typename Base::Epilogue;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    using ConvOutputIteratorParameter = typename Base::ConvOutputIteratorParameter;

    /// Executes one ImplicitGEMM
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_idx =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if CTA is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
            cudaGDC();
            return;
        }

        // Compute position within threadblock
        int thread_idx = threadIdx.x;

        // Compute starting filter position for strided dgrad
        int tile_m_per_filter =
            strided_dgrad_tile_m_per_filter(params.problem_size, ThreadblockShape::kM);
        int filter_tile_m = (threadblock_tile_idx.m() / tile_m_per_filter);

        int start_r;
        int start_s;
        params.stride_w_divmod(start_r, start_s, filter_tile_m);

        int filter_r = start_r;
        int filter_s = start_s;

        // Starting h, w positions for filter position in gemm_k=0
        int start_h;
        int start_w;
        strided_dgrad_starting_coords(params.problem_size, params.stride_h_divmod,
                                      params.stride_w_divmod, filter_r, filter_s, start_h, start_w);

        if (start_h >= params.problem_size.H || start_w >= params.problem_size.W) {
            cudaGDC();
            return;
        }

        typename Mma::FragmentC accumulators;

        accumulators.clear();

        // Broadcast the warp_id computed by lane 0 to ensure dependent code
        // is compiled as warp-uniform.
        int warp_idx = cutlass::canonical_warp_idx_sync();
        int lane_idx = threadIdx.x % 32;

        // Check if CTA contributes valid MMA (Dy * w) and accumulator will be non-zero after MMA
        if (start_r < params.problem_size.R && start_s < params.problem_size.S) {
            // Scale gemm_k_iterations for strided dgrad
            int gemm_k_iterations =
                (params.gemm_k_iterations / (params.problem_size.R * params.problem_size.S))
                * params.problem_size.num_gemm_k_filter_positions(start_r, start_s);

            // Construct iterators to A and B operands
            typename Mma::IteratorA iterator_A(
                params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
                params.stride_h_divmod, params.stride_w_divmod, start_r, start_s,
                cutlass::MatrixCoord(threadblock_tile_idx.m() * Mma::Shape::kM,
                                     threadblock_tile_idx.k() * Mma::Shape::kK));

            typename Mma::IteratorB iterator_B(
                params.iterator_B, params.problem_size, params.ptr_B, thread_idx, start_r, start_s,
                cutlass::MatrixCoord(threadblock_tile_idx.k() * Mma::Shape::kK,
                                     threadblock_tile_idx.n() * Mma::Shape::kN));

            //
            // Main loop
            //

            // Construct thread-scoped matrix multiply
            Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

            // Compute threadblock-scoped matrix multiply-add
            cudaGDC();
            mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);
        } else {
            cudaGDC();
        }

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        // Construct the semaphore.
        int block_idx =
            threadblock_tile_idx.m() + threadblock_tile_idx.n() * params.grid_tiled_shape.m();
        cutlass::Semaphore semaphore(params.semaphore + block_idx, thread_idx);

        // Compute logical position within grid
        threadblock_tile_idx = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // If performing a reduction via split-K, fetch the initial synchronization
        if (params.split_k_mode == cutlass::conv::SplitKMode::kSerial
            && params.grid_tiled_shape.k() > 1) {

            // Fetch the synchronization lock initially but do not block.
            semaphore.fetch();

            // Indicate which position in a serial reduction the output operator is currently updating
            output_op.set_k_partition(threadblock_tile_idx.k(), params.grid_tiled_shape.k());
        }

        cutlass::MatrixCoord threadblock_offset(threadblock_tile_idx.m() * Mma::Shape::kM,
                                                threadblock_tile_idx.n() * Mma::Shape::kN);

        // Tile iterator writing to destination tensor
        typename Epilogue::OutputTileIterator iterator_D(
            params.iterator_D, params.ptr_D,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx,
            params.stride_h_divmod, params.stride_w_divmod, start_r, start_s, threadblock_offset);

        // Construct the epilogue
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        if (output_op.is_source_needed()) {
            // Tile iterator reading from source accumulator tensor
            typename Epilogue::OutputTileIterator iterator_C(
                params.iterator_C, params.ptr_C,
                ConvOutputIteratorParameter::extent(params.problem_size), thread_idx,
                params.stride_h_divmod, params.stride_w_divmod, start_r, start_s, threadblock_offset);

            // Wait on the semaphore - this latency may have been covered by iterator construction
            if (params.split_k_mode == cutlass::conv::SplitKMode::kSerial
                && params.grid_tiled_shape.k() > 1) {

                // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
                if (threadblock_tile_idx.k()) {
                    iterator_C = iterator_D;
                }

                semaphore.wait(threadblock_tile_idx.k());
            }

            // Run epilogue with addend source iterator
            epilogue(output_op, iterator_D, accumulators, iterator_C);
        } else {
            // Run epilogue without addend source iterator
            epilogue(output_op, iterator_D, accumulators);
        }

        //
        // Release the semaphore
        //

        if (params.split_k_mode == cutlass::conv::SplitKMode::kSerial
            && params.grid_tiled_shape.k() > 1) {

            int lock = 0;
            if (params.grid_tiled_shape.k() == threadblock_tile_idx.k() + 1) {
                // The final threadblock resets the semaphore for subsequent grids.
                lock = 0;
            } else {
                // Otherwise, the semaphore is incremented
                lock = threadblock_tile_idx.k() + 1;
            }

            semaphore.release(lock);
        }
    }
};

/// Original file: cutlass/conv/kernel/default_conv2d_dgrad.h
/// to invoke CustomImplicitGemmConvolutionStridedDgrad
template <typename ElementA, typename LayoutA, typename ElementB, typename LayoutB,
          typename ElementC, typename LayoutC, typename ElementAccumulator, typename OperatorClass,
          typename ArchTag, typename ThreadblockShape, typename WarpShape, typename InstructionShape,
          typename EpilogueOutputOp, typename ThreadblockSwizzle, int Stages, typename MathOperatorTag,
          cutlass::conv::IteratorAlgorithm IteratorAlgorithm = cutlass::conv::IteratorAlgorithm::kOptimized,
          cutlass::conv::StrideSupport StrideSupport = cutlass::conv::StrideSupport::kStrided,
          int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value,
          int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value>
struct CustomDefaultConv2dDgrad {
    using Base = cutlass::conv::kernel::DefaultConv2dDgrad<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
        ArchTag, ThreadblockShape, WarpShape, InstructionShape, EpilogueOutputOp, ThreadblockSwizzle,
        Stages, MathOperatorTag, IteratorAlgorithm, StrideSupport, AlignmentA, AlignmentB>;
    using Kernel =
        CustomImplicitGemmConvolutionStridedDgrad<typename Base::Mma, typename Base::Epilogue,
                                                  ThreadblockSwizzle, cutlass::conv::Operator::kDgrad>;
};

/// Original file: cutlass/conv/device/implicit_gemm_convolution.h
/// to add "cudaLaunchKernelEx" to run()
template <typename ImplicitGemmKernel_>
class CustomImplicitGemmConvolution {
public:
    using Base = cutlass::conv::device::ImplicitGemmConvolution<ImplicitGemmKernel_>;
    using UnderlyingKernel = typename Base::UnderlyingKernel;
    using Arguments = typename Base::Arguments;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;

    static int const kWarpCount = Base::kWarpCount;

private:
    typename UnderlyingKernel::Params params_;

public:
    cutlass::Status run(cudaStream_t stream = nullptr)
    {
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(32 * kWarpCount, 1, 1);

        int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));
        cutlass::Status launch_result = cutlass::Status::kSuccess;

        cutlass::arch::synclog_setup();

        cudaLaunchConfig_t config;
        config.gridDim = grid;
        config.blockDim = block;
        config.dynamicSmemBytes = smem_size;
        config.stream = stream;
        auto attr = get_cuda_launch_attribute();
        config.attrs = &attr;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, &cutlass::Kernel<UnderlyingKernel>, params_);

        cudaError_t result = cudaGetLastError();
        if (cudaSuccess == result && cutlass::Status::kSuccess == launch_result) {
            return cutlass::Status::kSuccess;
        } else {
            CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
            return cutlass::Status::kErrorInternal;
        }
    }

    cutlass::Status initialize(Arguments const& args, void* workspace = nullptr,
                               cudaStream_t stream = nullptr)
    {
        // initialize the params structure from the arguments
        params_ = typename UnderlyingKernel::Params(args, static_cast<int*>(workspace));

        int smem_size = int(sizeof(typename UnderlyingKernel::SharedStorage));

        if (smem_size >= (48 << 10)) {
            cudaError_t result =
                cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess) {
                return cutlass::Status::kErrorInternal;
            }
        }

        return cutlass::Status::kSuccess;
    }

    cutlass::Status operator()(Arguments const& args, void* workspace = nullptr,
                               cudaStream_t stream = nullptr)
    {

        cutlass::Status status = initialize(args, workspace, stream);

        if (status == cutlass::Status::kSuccess) {
            status = run(stream);
        }

        return status;
    }

    static cutlass::Status can_implement(Arguments const& args)
    {
        return Base::can_implement(args);
    }
};
