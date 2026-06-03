// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "dmc_common.h"

#include <algorithm>

#include "def_elementwise.h"

void DMCCommon::capture_cuda_graph(cudaGraphExec_t& gexec, const cudaStream_t& stream,
                                   const std::function<void()>& lambda)
{
    cudaGraph_t graph;
    destroy_cuda_graph_exec(gexec);
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    try {
        lambda();
    } catch (...) {
        // End capture to restore the stream from capture mode, then discard the graph.
        cudaStreamEndCapture(stream, &graph);
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        throw;
    }
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    CUDA_CHECK(cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0));
    cudaGraphDestroy(graph);
}

int DMCCommon::compute_ec_parallel(int symbol_count)
{
    int n = symbol_count / MIN_SYMBOLS_PER_STREAM;
    return std::max(1, std::min(MAX_EC_PARALLEL, n));
}

at::Tensor DMCCommon::crop_hyper_params(const at::Tensor& hyper_params, const int H, const int W,
                                        at::Tensor& hyper_params_out)
{
    const int H0 = hyper_params.size(2);
    const int W0 = hyper_params.size(3);
    if (H0 == H && W0 == W) {
        return hyper_params;
    }
    hyper_params_out = slice_cuda(hyper_params, H, W, hyper_params_out);
    return hyper_params_out;
}

void DMCCommon::destroy_cuda_graph_exec(cudaGraphExec_t& gexec)
{
    if (gexec != nullptr) {
        cudaGraphExecDestroy(gexec);
        gexec = nullptr;
    }
}

at::Tensor DMCCommon::get_one_mask(const at::Tensor& micro_mask, int H, int W)
{
    auto mask = micro_mask.repeat({ (H + 1) / 2, (W + 1) / 2 });
    mask = mask.slice(0, 0, H).slice(1, 0, W);
    return mask.unsqueeze(0).unsqueeze(0);
}

std::tuple<int, int> DMCCommon::get_padding_size(const int height, const int width, const int p)
{
    const int new_h = (height + p - 1) / p * p;
    const int new_w = (width + p - 1) / p * p;
    const int padding_right = new_w - width;
    const int padding_bottom = new_h - height;
    return { padding_right, padding_bottom };
}

at::Tensor DMCCommon::pad_for_y(const at::Tensor& y, at::Tensor& y_pad)
{
    const int H = y.size(2);
    const int W = y.size(3);
    auto [padding_right, padding_bottom] = get_padding_size(H, W, 4);
    if (padding_right == 0 && padding_bottom == 0) {
        return y;
    }
    y_pad = replicate_pad_cuda(y, padding_bottom, padding_right, y_pad);
    return y_pad;
}

void DMCCommon::run(const std::function<void()>& lambda, cudaGraphExec_t& gexec)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gexec == nullptr) {
        capture_cuda_graph(gexec, stream, lambda);
    }

    CUDA_CHECK(cudaGraphLaunch(gexec, stream));
}

void DMCCommon::run(const std::function<void(int)>& lambda, cudaGraphExec_t (&gexec)[g_qp_num],
                    const int qp)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gexec[0] == nullptr) {
        for (int i = 0; i < g_qp_num; ++i) {
            capture_cuda_graph(gexec[i], stream, [&]() { lambda(i); });
        }
    }

    CUDA_CHECK(cudaGraphLaunch(gexec[qp], stream));
}

void DMCCommon::run(const std::function<void(bool)>& lambda, cudaGraphExec_t (&gexec)[2],
                    const bool bvalue)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gexec[0] == nullptr) {
        for (int i = 0; i < 2; ++i) {
            capture_cuda_graph(gexec[i], stream, [&]() { lambda(static_cast<bool>(i)); });
        }
    }

    CUDA_CHECK(cudaGraphLaunch(gexec[bvalue ? 1 : 0], stream));
}

void DMCCommon::run(const std::function<void(int, bool)>& lambda,
                    cudaGraphExec_t (&gexec)[g_qp_num][2], const int qp, const bool bvalue)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    if (gexec[0][0] == nullptr) {
        for (int i = 0; i < g_qp_num; ++i) {
            for (int j = 0; j < 2; ++j) {
                capture_cuda_graph(gexec[i][j], stream, [&]() { lambda(i, static_cast<bool>(j)); });
            }
        }
    }

    CUDA_CHECK(cudaGraphLaunch(gexec[qp][bvalue ? 1 : 0], stream));
}
