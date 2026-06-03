// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

// clang-format off
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
// clang-format on

#include "cutlass_epilogue.h"
#include "cutlass_helper_d3x3.h"
#include "../common_cu.h"
#include "../def_cutlass.h"

static constexpr int CUTLASS_START_INDEX = 65536;
static constexpr int HINT_NUM_CUTLASS = 151;
static constexpr int HINT_NUM_CUDA = 25;

template <typename SmArch, int stages, int split_k_slices>
at::Tensor d3x3_cutlass(const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& out_buf)
{
    static constexpr int groups_per_cta = 64;
    using ThreadBlockOutputShape = cutlass::conv::TensorNHWCShape<1, 8, 8, groups_per_cta>;
    using FilterShape = cutlass::MatrixShape<3, 3>;
    using ThreadblockShape =
        cutlass::gemm::GemmShape<ThreadBlockOutputShape::kNHW, groups_per_cta, FilterShape::kCount>;
    using WarpShape = cutlass::gemm::GemmShape<16, groups_per_cta, FilterShape::kCount>;
    using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;
    using SwizzleThreadBlock =
        cutlass::conv::threadblock::DepthwiseDirect2dConvIdentityThreadblockSwizzle<
            1, ThreadBlockOutputShape::kN, ThreadBlockOutputShape::kH, ThreadBlockOutputShape::kW>;
    using StrideShape = cutlass::MatrixShape<1, 1>;
    using DilationShape = cutlass::MatrixShape<1, 1>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        cutlass::half_t,                                // Data type of output matrix.
        128 / cutlass::sizeof_bits_v<cutlass::half_t>,  // The number of elements per vectorized.
        cutlass::half_t,                                // Data type of accumulator
        cutlass::half_t,  // Data type for alpha/beta in linear combination
        cutlass::epilogue::thread::ScaleType::Nothing>;  // Epilogue scaling operation.

    using DepthwiseDirect2dConv = typename CustomDefaultDepthwiseDirect2dConvFprop<
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::layout::TensorNHWC,
        cutlass::half_t, cutlass::layout::TensorNHWC, cutlass::half_t, cutlass::arch::OpClassSimt, SmArch,
        ThreadblockShape, ThreadBlockOutputShape, FilterShape, WarpShape, InstructionShape, EpilogueOp,
        SwizzleThreadBlock, stages, cutlass::arch::OpMultiplyAdd, cutlass::conv::IteratorAlgorithm::kFixedStrideDilation,
        cutlass::conv::StrideSupport::kFixed, StrideShape, DilationShape>::Kernel;

    using Direct2dConv = CustomDirectConvolutionDevice<DepthwiseDirect2dConv>;

    // Note: d3x3 weight is reordered from [C, 1, 3, 3] to [1, C, 3, 3].
    const int B = feature.size(0);
    const int C = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C1_ = weight.size(0);
    const int C2_ = weight.size(1);
    const int KH_ = weight.size(2);
    const int KW_ = weight.size(3);
    assert(C == C2_ && C1_ == 1 && KH_ == 3 && KW_ == 3);

    cutlass::conv::Conv2dProblemSize problem_size({ B, H, W, C },  // input
                                                  { C, 3, 3, 1 },  // kernel
                                                  { 1, 1, 1, 1 },  // pad
                                                  { 1, 1 },        // stride
                                                  { 1, 1 },        // dilation
                                                  { B, H, W, C },  // out
                                                  cutlass::conv::Mode::kCrossCorrelation,
                                                  split_k_slices,  // split_k_slices
                                                  C                // groups
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

    typename Direct2dConv::Arguments args(problem_size, d_feature, d_weight, d_out, d_out, {}, d_weight);

    Direct2dConv direct2dconv_op;
    cutlass::Status status = direct2dconv_op.can_implement(args);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to configure convolution operation." << std::endl;
        return at::Tensor();
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    status = direct2dconv_op(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "Failed to run convolution operation." << std::endl;
        return at::Tensor();
    }

    return out_buf;
}

template <int H_BLOCK_SIZE, int W_BLOCK_SIZE>
__global__ void d3x3_kernel(GPUTensor4D<at::Half> out, const GPUTensor4D<at::Half> feature,
                            const GPUTensor4D<at::Half> weight, const int C, const int H, const int W)
{
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = n % C;
    const int hw = n / C;
    const int h = (hw % ((H + H_BLOCK_SIZE - 1) / H_BLOCK_SIZE)) * H_BLOCK_SIZE;
    const int w = (hw / ((H + H_BLOCK_SIZE - 1) / H_BLOCK_SIZE)) * W_BLOCK_SIZE;
    if (c >= C || h >= H || w >= W) {
        cudaGDC();
        return;
    }

    float _weight[3][3];
    float _feature[H_BLOCK_SIZE + 2][W_BLOCK_SIZE + 2];

#pragma unroll
    for (int i = 0; i < 3; i++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
            _weight[i][j] = static_cast<float>(weight[0][c][i][j]);
        }
    }
    cudaGDC();

#pragma unroll
    for (int i = 0; i < H_BLOCK_SIZE + 2; i++) {
#pragma unroll
        for (int j = 0; j < W_BLOCK_SIZE + 2; j++) {
            const int y = h + i - 1;
            const int x = w + j - 1;
            float curr_feature = 0.f;
            if (y >= 0 && y < H && x >= 0 && x < W) {
                curr_feature = static_cast<float>(feature[0][c][y][x]);
            }
            _feature[i][j] = curr_feature;
        }
    }

    for (int h_offset = 0; h_offset < H_BLOCK_SIZE; h_offset++) {
        if (h + h_offset >= H) {
            break;
        }
        for (int w_offset = 0; w_offset < W_BLOCK_SIZE; w_offset++) {
            if (w + w_offset >= W) {
                break;
            }

            float r = 0.f;
#pragma unroll
            for (int i = 0; i < 3; i++) {
#pragma unroll
                for (int j = 0; j < 3; j++) {
                    float curr_weight = _weight[i][j];
                    float curr_feature = _feature[i + h_offset][j + w_offset];
                    r += curr_feature * curr_weight;
                }
            }
            out[0][c][h + h_offset][w + w_offset] = r;
        }
    }
}

template <int H_BLOCK_SIZE, int W_BLOCK_SIZE>
__forceinline__ auto d3x3_launch(const at::Tensor& feature, const at::Tensor& weight,
                                 const at::Tensor& out_buf)
{
    // Note: d3x3 weight is reordered from [C, 1, 3, 3] to [1, C, 3, 3].
    const int C = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C1_ = weight.size(0);
    const int C2_ = weight.size(1);
    const int KH_ = weight.size(2);
    const int KW_ = weight.size(3);
    assert(C == C2_ && C1_ == 1 && KH_ == 3 && KW_ == 3);
    const int total_threads =
        C * ((H + H_BLOCK_SIZE - 1) / H_BLOCK_SIZE) * ((W + W_BLOCK_SIZE - 1) / W_BLOCK_SIZE);

    const int BLOCK_SIZE = 256;
    const dim3 gridDim((total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE);
    const dim3 blockDim(BLOCK_SIZE);

    cudaLaunchConfig_t config;
    config.gridDim = gridDim;
    config.blockDim = blockDim;
    config.dynamicSmemBytes = 0;
    config.stream = at::cuda::getCurrentCUDAStream();
    auto attr = get_cuda_launch_attribute();
    config.attrs = &attr;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, &d3x3_kernel<H_BLOCK_SIZE, W_BLOCK_SIZE>, out_buf, feature, weight,
                       C, H, W);
    return out_buf;
}

__forceinline__ auto d3x3_launch_switcher(const int sm, const at::Tensor& feature,
                                          const at::Tensor& weight, const at::Tensor& out_buf,
                                          const int hint)
{
    if (sm >= 80) {
        /**
         * 0: pytorch
         * 1 - 25: cuda
         * 65536 - 65686: cutlass
         */
        switch (hint) {
        // remove the original case 0 because "at::conv2d" does not support out_buf
        case 0:
            return d3x3_launch<2, 2>(feature, weight, out_buf);
        case 1:
            return d3x3_launch<2, 3>(feature, weight, out_buf);
        case 2:
            return d3x3_launch<2, 4>(feature, weight, out_buf);
        case 3:
            return d3x3_launch<2, 5>(feature, weight, out_buf);
        case 4:
            return d3x3_launch<2, 6>(feature, weight, out_buf);
        case 5:
            return d3x3_launch<3, 2>(feature, weight, out_buf);
        case 6:
            return d3x3_launch<3, 3>(feature, weight, out_buf);
        case 7:
            return d3x3_launch<3, 4>(feature, weight, out_buf);
        case 8:
            return d3x3_launch<3, 5>(feature, weight, out_buf);
        case 9:
            return d3x3_launch<3, 6>(feature, weight, out_buf);
        case 10:
            return d3x3_launch<4, 2>(feature, weight, out_buf);
        case 11:
            return d3x3_launch<4, 3>(feature, weight, out_buf);
        case 12:
            return d3x3_launch<4, 4>(feature, weight, out_buf);
        case 13:
            return d3x3_launch<4, 5>(feature, weight, out_buf);
        case 14:
            return d3x3_launch<4, 6>(feature, weight, out_buf);
        case 15:
            return d3x3_launch<5, 2>(feature, weight, out_buf);
        case 16:
            return d3x3_launch<5, 3>(feature, weight, out_buf);
        case 17:
            return d3x3_launch<5, 4>(feature, weight, out_buf);
        case 18:
            return d3x3_launch<5, 5>(feature, weight, out_buf);
        case 19:
            return d3x3_launch<5, 6>(feature, weight, out_buf);
        case 20:
            return d3x3_launch<6, 2>(feature, weight, out_buf);
        case 21:
            return d3x3_launch<6, 3>(feature, weight, out_buf);
        case 22:
            return d3x3_launch<6, 4>(feature, weight, out_buf);
        case 23:
            return d3x3_launch<6, 5>(feature, weight, out_buf);
        case 24:
            return d3x3_launch<6, 6>(feature, weight, out_buf);
        case 65536:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 1>(feature, weight, out_buf);
        case 65537:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 3>(feature, weight, out_buf);
        case 65538:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 5>(feature, weight, out_buf);
        case 65539:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 7>(feature, weight, out_buf);
        case 65540:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 9>(feature, weight, out_buf);
        case 65541:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 11>(feature, weight, out_buf);
        case 65542:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 13>(feature, weight, out_buf);
        case 65543:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 15>(feature, weight, out_buf);
        case 65544:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 17>(feature, weight, out_buf);
        case 65545:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 19>(feature, weight, out_buf);
        case 65546:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 21>(feature, weight, out_buf);
        case 65547:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 23>(feature, weight, out_buf);
        case 65548:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 25>(feature, weight, out_buf);
        case 65549:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 27>(feature, weight, out_buf);
        case 65550:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 29>(feature, weight, out_buf);
        case 65551:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 31>(feature, weight, out_buf);
        case 65552:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 33>(feature, weight, out_buf);
        case 65553:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 35>(feature, weight, out_buf);
        case 65554:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 37>(feature, weight, out_buf);
        case 65555:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 39>(feature, weight, out_buf);
        case 65556:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 41>(feature, weight, out_buf);
        case 65557:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 43>(feature, weight, out_buf);
        case 65558:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 45>(feature, weight, out_buf);
        case 65559:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 47>(feature, weight, out_buf);
        case 65560:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 49>(feature, weight, out_buf);
        case 65561:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 51>(feature, weight, out_buf);
        case 65562:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 53>(feature, weight, out_buf);
        case 65563:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 55>(feature, weight, out_buf);
        case 65564:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 57>(feature, weight, out_buf);
        case 65565:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 59>(feature, weight, out_buf);
        case 65566:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 61>(feature, weight, out_buf);
        case 65567:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 63>(feature, weight, out_buf);
        case 65568:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 65>(feature, weight, out_buf);
        case 65569:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 67>(feature, weight, out_buf);
        case 65570:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 69>(feature, weight, out_buf);
        case 65571:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 71>(feature, weight, out_buf);
        case 65572:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 73>(feature, weight, out_buf);
        case 65573:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 75>(feature, weight, out_buf);
        case 65574:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 77>(feature, weight, out_buf);
        case 65575:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 79>(feature, weight, out_buf);
        case 65576:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 81>(feature, weight, out_buf);
        case 65577:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 83>(feature, weight, out_buf);
        case 65578:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 85>(feature, weight, out_buf);
        case 65579:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 87>(feature, weight, out_buf);
        case 65580:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 89>(feature, weight, out_buf);
        case 65581:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 91>(feature, weight, out_buf);
        case 65582:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 93>(feature, weight, out_buf);
        case 65583:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 95>(feature, weight, out_buf);
        case 65584:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 97>(feature, weight, out_buf);
        case 65585:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 99>(feature, weight, out_buf);
        case 65586:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 101>(feature, weight, out_buf);
        case 65587:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 103>(feature, weight, out_buf);
        case 65588:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 105>(feature, weight, out_buf);
        case 65589:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 107>(feature, weight, out_buf);
        case 65590:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 109>(feature, weight, out_buf);
        case 65591:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 111>(feature, weight, out_buf);
        case 65592:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 113>(feature, weight, out_buf);
        case 65593:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 115>(feature, weight, out_buf);
        case 65594:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 117>(feature, weight, out_buf);
        case 65595:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 119>(feature, weight, out_buf);
        case 65596:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 121>(feature, weight, out_buf);
        case 65597:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 123>(feature, weight, out_buf);
        case 65598:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 125>(feature, weight, out_buf);
        case 65599:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 127>(feature, weight, out_buf);
        case 65600:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 129>(feature, weight, out_buf);
        case 65601:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 131>(feature, weight, out_buf);
        case 65602:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 133>(feature, weight, out_buf);
        case 65603:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 135>(feature, weight, out_buf);
        case 65604:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 137>(feature, weight, out_buf);
        case 65605:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 139>(feature, weight, out_buf);
        case 65606:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 141>(feature, weight, out_buf);
        case 65607:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 143>(feature, weight, out_buf);
        case 65608:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 145>(feature, weight, out_buf);
        case 65609:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 147>(feature, weight, out_buf);
        case 65610:
            return d3x3_cutlass<cutlass::arch::Sm80, 2, 149>(feature, weight, out_buf);
        case 65611:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 1>(feature, weight, out_buf);
        case 65612:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 3>(feature, weight, out_buf);
        case 65613:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 5>(feature, weight, out_buf);
        case 65614:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 7>(feature, weight, out_buf);
        case 65615:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 9>(feature, weight, out_buf);
        case 65616:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 11>(feature, weight, out_buf);
        case 65617:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 13>(feature, weight, out_buf);
        case 65618:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 15>(feature, weight, out_buf);
        case 65619:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 17>(feature, weight, out_buf);
        case 65620:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 19>(feature, weight, out_buf);
        case 65621:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 21>(feature, weight, out_buf);
        case 65622:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 23>(feature, weight, out_buf);
        case 65623:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 25>(feature, weight, out_buf);
        case 65624:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 27>(feature, weight, out_buf);
        case 65625:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 29>(feature, weight, out_buf);
        case 65626:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 31>(feature, weight, out_buf);
        case 65627:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 33>(feature, weight, out_buf);
        case 65628:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 35>(feature, weight, out_buf);
        case 65629:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 37>(feature, weight, out_buf);
        case 65630:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 39>(feature, weight, out_buf);
        case 65631:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 41>(feature, weight, out_buf);
        case 65632:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 43>(feature, weight, out_buf);
        case 65633:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 45>(feature, weight, out_buf);
        case 65634:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 47>(feature, weight, out_buf);
        case 65635:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 49>(feature, weight, out_buf);
        case 65636:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 51>(feature, weight, out_buf);
        case 65637:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 53>(feature, weight, out_buf);
        case 65638:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 55>(feature, weight, out_buf);
        case 65639:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 57>(feature, weight, out_buf);
        case 65640:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 59>(feature, weight, out_buf);
        case 65641:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 61>(feature, weight, out_buf);
        case 65642:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 63>(feature, weight, out_buf);
        case 65643:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 65>(feature, weight, out_buf);
        case 65644:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 67>(feature, weight, out_buf);
        case 65645:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 69>(feature, weight, out_buf);
        case 65646:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 71>(feature, weight, out_buf);
        case 65647:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 73>(feature, weight, out_buf);
        case 65648:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 75>(feature, weight, out_buf);
        case 65649:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 77>(feature, weight, out_buf);
        case 65650:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 79>(feature, weight, out_buf);
        case 65651:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 81>(feature, weight, out_buf);
        case 65652:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 83>(feature, weight, out_buf);
        case 65653:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 85>(feature, weight, out_buf);
        case 65654:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 87>(feature, weight, out_buf);
        case 65655:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 89>(feature, weight, out_buf);
        case 65656:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 91>(feature, weight, out_buf);
        case 65657:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 93>(feature, weight, out_buf);
        case 65658:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 95>(feature, weight, out_buf);
        case 65659:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 97>(feature, weight, out_buf);
        case 65660:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 99>(feature, weight, out_buf);
        case 65661:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 101>(feature, weight, out_buf);
        case 65662:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 103>(feature, weight, out_buf);
        case 65663:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 105>(feature, weight, out_buf);
        case 65664:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 107>(feature, weight, out_buf);
        case 65665:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 109>(feature, weight, out_buf);
        case 65666:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 111>(feature, weight, out_buf);
        case 65667:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 113>(feature, weight, out_buf);
        case 65668:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 115>(feature, weight, out_buf);
        case 65669:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 117>(feature, weight, out_buf);
        case 65670:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 119>(feature, weight, out_buf);
        case 65671:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 121>(feature, weight, out_buf);
        case 65672:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 123>(feature, weight, out_buf);
        case 65673:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 125>(feature, weight, out_buf);
        case 65674:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 127>(feature, weight, out_buf);
        case 65675:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 129>(feature, weight, out_buf);
        case 65676:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 131>(feature, weight, out_buf);
        case 65677:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 133>(feature, weight, out_buf);
        case 65678:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 135>(feature, weight, out_buf);
        case 65679:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 137>(feature, weight, out_buf);
        case 65680:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 139>(feature, weight, out_buf);
        case 65681:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 141>(feature, weight, out_buf);
        case 65682:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 143>(feature, weight, out_buf);
        case 65683:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 145>(feature, weight, out_buf);
        case 65684:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 147>(feature, weight, out_buf);
        case 65685:
            return d3x3_cutlass<cutlass::arch::Sm80, 3, 149>(feature, weight, out_buf);
        case 65686:  // default cutlass hint for sm80
            return d3x3_cutlass<cutlass::arch::Sm80, 4, 54>(feature, weight, out_buf);
        default:
            assert(false);
        }
    }

    assert(sm == 75);
    return d3x3_cutlass<cutlass::arch::Sm75, 2, 32>(feature, weight, out_buf);
}

__forceinline__ auto d3x3_launch_helper(const int sm, const at::Tensor& feature,
                                        const at::Tensor& weight, const at::Tensor& out_buf,
                                        std::unordered_map<uint64_t, int>& map,
                                        const std::string& map_name)
{
    if (sm == 75) {
        return d3x3_launch_switcher(sm, feature, weight, out_buf, -1);
    }
    const int C = feature.size(1);
    const int H = feature.size(2);
    const int W = feature.size(3);
    const uint64_t key = conv_key(sm, W, H, C, C, 0, 0);
    int best_hint = -1;
    auto it = map.find(key);
    if (it == map.end()) {
        return d3x3_launch_switcher(sm, feature, weight, out_buf,
                                    CUTLASS_START_INDEX + HINT_NUM_CUTLASS - 1);
    } else {
        best_hint = it->second;
    }
    return d3x3_launch_switcher(sm, feature, weight, out_buf, best_hint);
}
