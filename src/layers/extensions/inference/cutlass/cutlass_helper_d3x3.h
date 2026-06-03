// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/conv/kernel/default_depthwise_fprop.h>
#include <cutlass/conv/device/direct_convolution.h>
// clang-format on

/**
 * These 3 classes/structs are rewritten:
 *   1. cutlass::conv::kernel::DirectConvolution -> CustomDirectConvolutionKernel
 *   2. cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop -> CustomDefaultDepthwiseDirect2dConvFprop
 *   3. cutlass::conv::device::DirectConvolution -> CustomDirectConvolutionDevice
 */

/// Original file: cutlass/conv/kernel/direct_convolution.h
/// to add PDL sync points
template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_, cutlass::conv::Operator ConvOperator,
          typename ConvProblemSize_, cutlass::conv::GroupMode GroupMode_, typename ThreadBlockOutputShape_>
struct CustomDirectConvolutionKernel {
    using Base =
        cutlass::conv::kernel::DirectConvolution<Mma_, Epilogue_, ThreadblockSwizzle_, ConvOperator,
                                                 ConvProblemSize_, GroupMode_, ThreadBlockOutputShape_>;
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
    using ReorderKernel = typename Base::ReorderKernel;
    using Mma = typename Base::Mma;
    using Epilogue = typename Base::Epilogue;
    using Params = typename Base::Params;
    using SharedStorage = typename Base::SharedStorage;

    using TensorRefC = cutlass::TensorRef<ElementC, LayoutC>;
    using ConvProblemSize = ConvProblemSize_;
    using ConvOutputIteratorParameter = cutlass::epilogue::threadblock::ConvOutputIteratorParameter<
        LayoutC, typename Epilogue::OutputTileIterator::Layout, TensorRefC, ConvOperator, ConvProblemSize>;

    /// Executes one ImplicitGEMM
    /// The aim is to add PDL sync point to this function.
    CUTLASS_DEVICE
    void operator()(Params const& params, SharedStorage& shared_storage)
    {
        // Compute threadblock location
        ThreadblockSwizzle threadblock_swizzle;

        cutlass::gemm::GemmCoord threadblock_tile_idx =
            threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        // Early exit if threadblock is out of range
        if (params.grid_tiled_shape.m() <= threadblock_tile_idx.m()
            || params.grid_tiled_shape.n() <= threadblock_tile_idx.n()) {
            cudaGDC();
            return;
        }

        // Compute position within threadblock
        int thread_idx = threadIdx.x;
        int iterator_column_offset = 0;
        int filter_row_offset = 0;
        if (kGroupMode != cutlass::conv::GroupMode::kNone) {
            if (kGroupMode == cutlass::conv::GroupMode::kDepthwise) {
                iterator_column_offset += threadblock_tile_idx.n() * Mma::Shape::kN;
            }
        }

        // Construct iterators to A and B operands
        typename Mma::IteratorA iterator_A(
            params.iterator_A, params.problem_size, params.ptr_A, thread_idx,
            cutlass::MatrixCoord(threadblock_tile_idx.m() + threadblock_tile_idx.k(),
                                 iterator_column_offset));

        typename Mma::IteratorB iterator_B(
            params.iterator_B, params.problem_size, params.ptr_reordered_B, thread_idx,
            cutlass::MatrixCoord(filter_row_offset, iterator_column_offset));

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

        //
        // Epilogue
        //

        EpilogueOutputOp output_op(params.output_op);

        // Compute logical position within grid
        threadblock_tile_idx = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

        cutlass::MatrixCoord threadblock_offset(threadblock_tile_idx.m() + threadblock_tile_idx.k(),
                                                threadblock_tile_idx.n() * Mma::Shape::kN);

        // Tile iterator writing to destination tensor
        typename Epilogue::OutputTileIterator iterator_D(
            params.iterator_D, params.ptr_D,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx, threadblock_offset);

        // Tile iterator reading from source accumulator tensor
        typename Epilogue::OutputTileIterator iterator_C(
            params.iterator_C, params.ptr_C,
            ConvOutputIteratorParameter::extent(params.problem_size), thread_idx, threadblock_offset);

        // Construct the epilogue
        Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

        // Compute threadblock-scoped matrix multiply-add
        // Epilogue is fused in the mainloop
        cudaGDC();
        mma(params.gemm_k_iterations, accumulators, iterator_A, params.iterator_A, iterator_B,
            params.iterator_B, accumulators, epilogue, output_op, iterator_D, iterator_C,
            params.split_k_slices);
    }
};

/// Original file: cutlass/conv/kernel/default_depthwise_fprop.h
/// to invoke CustomDirectConvolutionKernel
template <typename ElementA, typename LayoutA, typename ElementB, typename LayoutB, typename ElementC,
          typename LayoutC, typename ElementAccumulator, typename OperatorClass, typename ArchTag,
          typename ThreadblockShape, typename ThreadBlockOutputShape, typename FilterShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp, typename ThreadblockSwizzle,
          int Stages, typename MathOperatorTag, cutlass::conv::IteratorAlgorithm IteratorAlgorithm,
          cutlass::conv::StrideSupport StrideSupport, typename StrideShape, typename DilationShape,
          int AlignmentA = 128 / cutlass::sizeof_bits_v<ElementA>,
          int AlignmentB = 128 / cutlass::sizeof_bits_v<ElementB>>
struct CustomDefaultDepthwiseDirect2dConvFprop {
    using Base = cutlass::conv::kernel::DefaultDepthwiseDirect2dConvFprop<
        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator, OperatorClass,
        ArchTag, ThreadblockShape, ThreadBlockOutputShape, FilterShape, WarpShape, InstructionShape,
        EpilogueOutputOp, ThreadblockSwizzle, Stages, MathOperatorTag, IteratorAlgorithm,
        StrideSupport, StrideShape, DilationShape, AlignmentA, AlignmentB>;
    using Kernel =
        CustomDirectConvolutionKernel<typename Base::Mma, typename Base::Epilogue, ThreadblockSwizzle,
                                      cutlass::conv::Operator::kFprop, cutlass::conv::Conv2dProblemSize,
                                      cutlass::conv::GroupMode::kDepthwise, ThreadBlockOutputShape>;
};

/// Original file: cutlass/conv/device/direct_convolution.h
/// to remove the launch of the ReorderKernel
/// to add "cudaLaunchKernelEx" to run()
template <typename DirectConvolutionKernel_>
class CustomDirectConvolutionDevice {
public:
    using Base = cutlass::conv::device::DirectConvolution<DirectConvolutionKernel_>;
    using UnderlyingKernel = typename Base::UnderlyingKernel;
    using ThreadblockShape = typename Base::ThreadblockShape;
    using ThreadblockSwizzle = typename Base::ThreadblockSwizzle;
    using Arguments = typename Base::Arguments;

    static cutlass::conv::Operator const kConvolutionalOperator = Base::kConvolutionalOperator;
    static cutlass::conv::GroupMode const kGroupMode = Base::kGroupMode;
    static int const kWarpCount = Base::kWarpCount;

private:
    typename UnderlyingKernel::Params params_;

public:
    static cutlass::Status can_implement(Arguments const& args)
    {
        return Base::can_implement(args);
    }

    cutlass::Status initialize(Arguments const& args, void* workspace = nullptr,
                               cudaStream_t stream = nullptr)
    {
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

    // Only this function is modified. To remove the launch of ReorderKernel.
    cutlass::Status run(cudaStream_t stream = nullptr)
    {
        // Launch main kernel
        ThreadblockSwizzle threadblock_swizzle;

        dim3 grid = threadblock_swizzle.get_grid_shape(params_.grid_tiled_shape);
        dim3 block(32 * kWarpCount, 1, 1);

        // Dynamic SMEM size based on input params.
        int smem_size = int(params_.get_smem_size());

        // Make sure we can use that much shared memory.
        cudaError_t status =
            cudaFuncSetAttribute(cutlass::Kernel<UnderlyingKernel>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
        if (status != cudaSuccess) {
            return cutlass::Status::kErrorInternal;
        }

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

        return result == cudaSuccess ? cutlass::Status::kSuccess : cutlass::Status::kErrorInternal;
    }
};
