// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "def.h"
namespace F = torch::nn::functional;

void DepthConvProxy::set_param(
    const torch::Tensor& dc_conv1_weight, const torch::Tensor& dc_conv1_bias,
    const torch::Tensor& dc_depth_conv_weight, const torch::Tensor& dc_depth_conv_bias,
    const torch::Tensor& dc_conv2_weight, const torch::Tensor& dc_conv2_bias,
    const torch::Tensor& ffn_conv1_weight, const torch::Tensor& ffn_conv1_bias,
    const torch::Tensor& ffn_conv2_weight, const torch::Tensor& ffn_conv2_bias, const bool shortcut)
{
    _dc_conv1_weight = dc_conv1_weight;
    _dc_conv1_bias = dc_conv1_bias;
    _dc_depth_conv_weight = dc_depth_conv_weight;
    _dc_conv2_weight = dc_conv2_weight;
    _dc_conv2_bias = F::conv2d(dc_depth_conv_bias.reshape({ 1, -1, 1, 1 }), dc_conv2_weight);
    _dc_conv2_bias = _dc_conv2_bias.index({ 0, torch::indexing::Slice(), 0, 0 }) + dc_conv2_bias;
    _ffn_conv1_weight = ffn_conv1_weight;
    _ffn_conv1_bias = ffn_conv1_bias;
    _ffn_conv2_weight = ffn_conv2_weight;
    _ffn_conv2_bias = ffn_conv2_bias;
    _shortcut = shortcut;
}

void DepthConvProxy::set_param_with_adaptor(
    const torch::Tensor& dc_conv1_weight, const torch::Tensor& dc_conv1_bias,
    const torch::Tensor& dc_depth_conv_weight, const torch::Tensor& dc_depth_conv_bias,
    const torch::Tensor& dc_conv2_weight, const torch::Tensor& dc_conv2_bias,
    const torch::Tensor& ffn_conv1_weight, const torch::Tensor& ffn_conv1_bias,
    const torch::Tensor& ffn_conv2_weight, const torch::Tensor& ffn_conv2_bias,
    const torch::Tensor& adaptor_weight, const torch::Tensor& adaptor_bias, const bool shortcut)
{
    _dc_conv1_weight = F::conv2d(torch::transpose(adaptor_weight, 0, 1), dc_conv1_weight);
    _dc_conv1_weight = torch::transpose(_dc_conv1_weight, 0, 1);
    _dc_conv1_weight = torch::cat({ _dc_conv1_weight, adaptor_weight }, 0);
    _dc_conv1_bias = F::conv2d(adaptor_bias.reshape({ 1, -1, 1, 1 }), dc_conv1_weight);
    _dc_conv1_bias = _dc_conv1_bias.index({ 0, torch::indexing::Slice(), 0, 0 }) + dc_conv1_bias;
    _dc_depth_conv_weight = dc_depth_conv_weight;
    _dc_conv2_weight = dc_conv2_weight;
    _dc_conv2_bias = F::conv2d(dc_depth_conv_bias.reshape({ 1, -1, 1, 1 }), dc_conv2_weight);
    _dc_conv2_bias = _dc_conv2_bias.index({ 0, torch::indexing::Slice(), 0, 0 }) + dc_conv2_bias;
    _dc_conv2_bias = _dc_conv2_bias + adaptor_bias;
    _ffn_conv1_weight = ffn_conv1_weight;
    _ffn_conv1_bias = ffn_conv1_bias;
    _ffn_conv2_weight = ffn_conv2_weight;
    _ffn_conv2_bias = ffn_conv2_bias;
    _shortcut = shortcut;
    _adaptor = true;
}

std::tuple<torch::Tensor, torch::Tensor> DepthConvProxy::forward_common(const torch::Tensor& x)
{
    auto identity = x;
    // depthconv
    torch::Tensor out;
    if (_adaptor) {
        // NOTE: Here we always fuse adaptor with the first conv1x1 (when even in_ch > out_ch).
        //       It brings larger MACs, but it faster on A100 due to lower memory cost.
        auto out_identity = F::conv2d(identity, _dc_conv1_weight);
        auto chunks = torch::chunk(out_identity, 2, 1);
        out = chunks[0];
        identity = chunks[1];
    } else {
        out = F::conv2d(identity, _dc_conv1_weight);
    }
    out = bias_wsilu_depthwise_conv2d_cuda(out, _dc_depth_conv_weight, _dc_conv1_bias);
    out = F::conv2d(out, _dc_conv2_weight);

    if (_shortcut) {
        bias_shortcut_2_cuda(out, _dc_conv2_bias, identity);
    } else {
        bias_shortcut_cuda(out, _dc_conv2_bias, identity);
        identity = out;
    }
    // ffn
    out = F::conv2d(out, _ffn_conv1_weight);
    bias_wsilu_chunk_add_cuda(out, _ffn_conv1_bias);
    out = F::conv2d(out, _ffn_conv2_weight);
    return { out, identity };
}

torch::Tensor DepthConvProxy::forward(const torch::Tensor& x)
{
    auto [out, identity] = forward_common(x);
    bias_shortcut_cuda(out, _ffn_conv2_bias, identity);
    return out;
}

torch::Tensor DepthConvProxy::forward_with_quant_step(const torch::Tensor& x,
                                                      const torch::Tensor& quant_step)
{
    auto [out, identity] = forward_common(x);
    bias_shortcut_with_quant_step_cuda(out, _ffn_conv2_bias, quant_step, identity);
    return out;
}

torch::Tensor DepthConvProxy::forward_with_cat(const torch::Tensor& x, const torch::Tensor& to_cat,
                                               const bool cat_at_front)
{
    auto [t, identity] = forward_common(x);

    auto t_shape = t.sizes();
    auto B = t_shape[0];
    auto C = t_shape[1];
    auto H = t_shape[2];
    auto W = t_shape[3];
    auto add_ch = to_cat.sizes()[1];
    auto out = torch::empty({ B, C + add_ch, H, W }, t.options());
    if (cat_at_front) {
        auto t_out = out.narrow(1, add_ch, C);
        bias_shortcut_no_inplace_cuda(t_out, t, _ffn_conv2_bias, identity);
        out.narrow(1, 0, add_ch) = to_cat;
    } else {
        auto t_out = out.narrow(1, 0, C);
        bias_shortcut_no_inplace_cuda(t_out, t, _ffn_conv2_bias, identity);
        out.narrow(1, C, add_ch) = to_cat;
    }
    return out;
}

void SubpelConv2xProxy::set_param(const torch::Tensor& weight, const torch::Tensor& bias,
                                  const int padding)
{
    _weight = weight;
    _bias = bias;
    _padding = padding;
}

torch::Tensor SubpelConv2xProxy::forward(const torch::Tensor& x)
{
    auto t = F::conv2d(x, _weight, F::Conv2dFuncOptions().padding(_padding));
    auto t_shape = t.sizes();
    auto B = t_shape[0];
    auto C = t_shape[1];
    auto H = t_shape[2];
    auto W = t_shape[3];
    auto out = torch::empty({ B, C / 4, H * 2, W * 2 }, t.options());
    assert(B == 1);
    bias_pixel_shuffle_2_cuda(out, t, _bias, C, H * W, W);
    return out;
}

torch::Tensor SubpelConv2xProxy::forward_with_cat(const torch::Tensor& x, const torch::Tensor& to_cat,
                                                  const bool cat_at_front)
{
    auto t = F::conv2d(x, _weight, F::Conv2dFuncOptions().padding(_padding));
    auto t_shape = t.sizes();
    auto B = t_shape[0];
    auto C = t_shape[1];
    auto H = t_shape[2];
    auto W = t_shape[3];
    auto add_ch = to_cat.sizes()[1];
    auto out = torch::empty({ B, add_ch + C / 4, H * 2, W * 2 }, t.options());
    assert(B == 1);
    if (cat_at_front) {
        auto t_out = out.narrow(1, add_ch, C / 4);
        bias_pixel_shuffle_2_cuda(t_out, t, _bias, C, H * W, W);
        out.narrow(1, 0, add_ch) = to_cat;
    } else {
        auto t_out = out.narrow(1, 0, C / 4);
        bias_pixel_shuffle_2_cuda(t_out, t, _bias, C, H * W, W);
        out.narrow(1, C / 4, add_ch) = to_cat;
    }
    return out;
}
