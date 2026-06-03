// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "cuda_check.h"
#include "def_const.h"
#include "layers_proxy.h"
#include "py_rans.h"

enum class DMCWorkType {
    Encode,
    DecodeZ,
    DecodeY,
};

class DMCCommon {
public:
    static constexpr int g_qp_num = 64;

public:
    DMCCommon() = default;
    ~DMCCommon() = default;

public:
    static void capture_cuda_graph(cudaGraphExec_t& gexec, const cudaStream_t& stream,
                                   const std::function<void()>& lambda);
    static int compute_ec_parallel(int symbol_count);
    static void destroy_cuda_graph_exec(cudaGraphExec_t& gexec);
    static at::Tensor get_one_mask(const at::Tensor& micro_mask, int H, int W);
    static std::tuple<int, int> get_padding_size(const int height, const int width, const int p);
    static void run(const std::function<void()>& lambda, cudaGraphExec_t& gexec);
    static void run(const std::function<void(int)>& lambda, cudaGraphExec_t (&gexec)[g_qp_num],
                    const int qp);
    static void run(const std::function<void(bool)>& lambda, cudaGraphExec_t (&gexec)[2],
                    const bool bvalue);
    static void run(const std::function<void(int, bool)>& lambda,
                    cudaGraphExec_t (&gexec)[g_qp_num][2], const int qp, const bool bvalue);

    template <typename T, bool is_gpu = false>
    static std::shared_ptr<std::vector<T>> tensor_to_vector_1d(const at::Tensor& x)
    {
        auto vec = std::make_shared<std::vector<T>>(x.numel());
        if constexpr (is_gpu) {
            CUDA_CHECK(cudaMemcpyAsync(vec->data(), x.data_ptr<T>(), x.numel() * sizeof(T),
                                       cudaMemcpyDeviceToHost, at::cuda::getCurrentCUDAStream()));
            CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
        } else {
            memcpy(vec->data(), x.data_ptr<T>(), x.numel() * sizeof(T));
        }
        return vec;
    }

protected:
    at::Tensor crop_hyper_params(const at::Tensor& hyper_params, const int H, const int W,
                                 at::Tensor& hyper_params_out);
    at::Tensor pad_for_y(const at::Tensor& y, at::Tensor& y_pad);

protected:
    int m_entropy_coder_parallel{ 0 };
    RansEncoder m_entropy_encoder;
    RansDecoder m_entropy_decoder;
};
