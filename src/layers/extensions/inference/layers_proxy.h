// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "def_const.h"
#include "py_rans.h"

std::tuple<at::Tensor, at::Tensor> chunk_tensors_2(const at::Tensor& x);
std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_tensors_3(const at::Tensor& x);
std::map<std::string, at::Tensor> get_submodule_state_dict(const std::map<std::string, at::Tensor>& state_dict,
                                                           const std::string& prefix);
std::tuple<at::Tensor, at::Tensor> split_tensors_2(const at::Tensor& x, const int C1, const int C2);
std::tuple<at::Tensor, at::Tensor, at::Tensor> split_tensors_3(const at::Tensor& x, const int C1,
                                                               const int C2, const int C3);

class ConvParam {
public:
    at::Tensor weight;
    at::Tensor bias;

    ConvParam() = default;
    ~ConvParam() = default;

    void set_param(const std::map<std::string, at::Tensor>& state_dict, const std::string& prefix);
};

class DepthConvBlockProxy {
public:
    DepthConvBlockProxy() = default;
    ~DepthConvBlockProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::optional<at::Tensor>& quant = at::nullopt);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const bool input_further_use = false,
                                    const at::optional<at::Tensor>& out_buf = at::nullopt);
    void release_tensors();
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const bool shortcut = false);

private:
    at::Tensor m_adaptor_weight;
    at::Tensor m_adaptor_bias;
    at::Tensor m_dc_conv1_weight;
    at::Tensor m_dc_conv1_bias;
    at::Tensor m_dc_depth_conv_weight;
    at::Tensor m_dc_conv2_weight;
    at::Tensor m_dc_conv2_bias;
    at::Tensor m_ffn_conv1_weight;
    at::Tensor m_ffn_conv1_bias;
    at::Tensor m_ffn_conv1_weight0;
    at::Tensor m_ffn_conv1_weight1;
    at::Tensor m_ffn_conv1_weight2;
    at::Tensor m_ffn_conv1_weight3;
    at::Tensor m_ffn_conv1_bias0;
    at::Tensor m_ffn_conv1_bias1;
    at::Tensor m_ffn_conv1_bias2;
    at::Tensor m_ffn_conv1_bias3;
    at::Tensor m_ffn_conv2_weight;
    at::Tensor m_ffn_conv2_bias;
    bool m_adaptor{ false };
    bool m_shortcut{ false };

    // for cuda graphs
    at::Tensor m_adaptor_out;
    at::Tensor m_dc_conv1_out;
    at::Tensor m_dc_depth_conv_out;
    at::Tensor m_dc_conv2_out;
    at::Tensor m_ffn_conv1_out;
    at::Tensor m_ffn_conv2_out;
};

class SubpelConv2xProxy {
public:
    SubpelConv2xProxy() = default;
    ~SubpelConv2xProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, at::Tensor& out_buf);
    void release_tensors();
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    at::Tensor m_weight;
    at::Tensor m_weight_transposed;
    at::Tensor m_bias;

    // for cuda graphs
    at::Tensor m_conv_out;
    at::Tensor m_shuffle_out;
};

class ResidualBlockWithStride2Proxy {
public:
    ResidualBlockWithStride2Proxy() = default;
    ~ResidualBlockWithStride2Proxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x,
                                    const at::optional<at::Tensor>& out_buf = at::nullopt);
    void release_tensors();
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const bool shortcut);

private:
    at::Tensor m_weight;
    at::Tensor m_bias;
    DepthConvBlockProxy m_conv;

    // for cuda graphs
    at::Tensor m_conv_out;
};

class ResidualBlockUpsampleProxy {
public:
    ResidualBlockUpsampleProxy() = default;
    ~ResidualBlockUpsampleProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void release_tensors();
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const bool shortcut);

private:
    SubpelConv2xProxy m_up;
    DepthConvBlockProxy m_conv;
};
