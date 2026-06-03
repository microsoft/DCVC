// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/torch.h>

#include "common_cpp.h"
#include "def_cutlass.h"
#include "def_elementwise.h"
#include "layers_proxy.h"
#include "memory_pool.h"

std::tuple<at::Tensor, at::Tensor> chunk_tensors_2(const at::Tensor& x)
{
    auto out = x.chunk(2, 1);
    return { out[0], out[1] };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> chunk_tensors_3(const at::Tensor& x)
{
    auto out = x.chunk(3, 1);
    return { out[0], out[1], out[2] };
}

at::Tensor conv2d(const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& bias,
                  const int stride, at::Tensor& out_buf)
{
    const int sm = get_gpu_sm();

    const int kernel = weight.size(2);
    if (kernel == 1 && stride == 1) {
        return conv1x1_bias(sm, feature, weight, bias, out_buf);
    }
    return conv_bias(sm, feature, weight, bias, stride, out_buf);
}

std::map<std::string, at::Tensor>
get_submodule_state_dict(const std::map<std::string, at::Tensor>& state_dict, const std::string& prefix)
{

    std::map<std::string, at::Tensor> submodule_state_dict;

    for (const auto& param : state_dict) {
        if (param.first.rfind(prefix, 0) == 0) {
            std::string new_key = param.first.substr(prefix.size());
            submodule_state_dict[new_key] = param.second;
        }
    }

    return submodule_state_dict;
}

std::tuple<at::Tensor, at::Tensor> split_tensors_2(const at::Tensor& x, const int C1, const int C2)
{
    auto out = x.split_with_sizes({ C1, C2 }, 1);
    return { out[0], out[1] };
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> split_tensors_3(const at::Tensor& x, const int C1,
                                                               const int C2, const int C3)
{
    auto out = x.split_with_sizes({ C1, C2, C3 }, 1);
    return { out[0], out[1], out[2] };
}

void ConvParam::set_param(const std::map<std::string, at::Tensor>& state_dict, const std::string& prefix)
{
    weight = state_dict.at(prefix + "weight");
    bias = state_dict.at(prefix + "bias");
}

at::Tensor DepthConvBlockProxy::forward(const at::Tensor& x, const at::optional<at::Tensor>& quant)
{
    const int sm = get_gpu_sm();

    auto out = x;
    if (m_adaptor) {
        out = conv1x1_bias(sm, x, m_adaptor_weight, m_adaptor_bias, m_adaptor_out);
    }
    auto shortcut = out;
    out = conv1x1_bias_wsilu(sm, out, m_dc_conv1_weight, m_dc_conv1_bias, m_dc_conv1_out);
    out = d3x3(sm, out, m_dc_depth_conv_weight, m_dc_depth_conv_out);
    out = conv1x1_bias_shortcut(sm, out, m_dc_conv2_weight, m_dc_conv2_bias, shortcut, m_dc_conv2_out);

    auto shortcut_ffn = out;
    out = conv1x1_bias_wsilu_chunk_add(sm, out, m_ffn_conv1_weight, m_ffn_conv1_bias,
                                       m_ffn_conv1_weight0, m_ffn_conv1_weight1, m_ffn_conv1_weight2,
                                       m_ffn_conv1_weight3, m_ffn_conv1_bias0, m_ffn_conv1_bias1,
                                       m_ffn_conv1_bias2, m_ffn_conv1_bias3, m_ffn_conv1_out);
    if (m_shortcut) {
        out = conv1x1_bias_shortcut2(sm, out, m_ffn_conv2_weight, m_ffn_conv2_bias, shortcut_ffn,
                                     shortcut, m_ffn_conv2_out);
    } else if (quant.has_value()) {
        out = conv1x1_bias_shortcut_with_quant(sm, out, m_ffn_conv2_weight, m_ffn_conv2_bias,
                                               shortcut_ffn, quant.value(), m_ffn_conv2_out);
    } else {
        out = conv1x1_bias_shortcut(sm, out, m_ffn_conv2_weight, m_ffn_conv2_bias, shortcut_ffn,
                                    m_ffn_conv2_out);
    }

    return out;
}

at::Tensor DepthConvBlockProxy::pre_allocate_tensors(at::Tensor& x, const bool input_further_use,
                                                     const at::optional<at::Tensor>& out_buf)
{
    const int H = x.size(2);
    const int W = x.size(3);
    bool dc_input_modifiable = !m_shortcut && (!input_further_use || m_adaptor);

    auto options = m_dc_conv1_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int ch_dc = m_dc_conv1_weight.size(0);
    const int C = m_dc_conv1_weight.size(1);
    const int ch_ffn = m_ffn_conv2_weight.size(1);

    auto dc_input = x;
    if (m_adaptor) {
        const int C1 = m_adaptor_weight.size(0);
        m_adaptor_out = g_tensor_pool.empty({ 1, C1, H, W }, options);
        dc_input = m_adaptor_out;
    }

    m_dc_conv1_out = g_tensor_pool.empty({ 1, ch_dc, H, W }, options);
    m_dc_depth_conv_out = g_tensor_pool.empty_like(m_dc_conv1_out);
    if (out_buf.has_value()) {
        m_dc_conv2_out = out_buf.value();
    } else if (dc_input_modifiable) {
        m_dc_conv2_out = dc_input;
    } else {
        m_dc_conv2_out = g_tensor_pool.empty({ 1, C, H, W }, options);
    }
    m_ffn_conv1_out = g_tensor_pool.empty({ 1, ch_ffn, H, W }, options);
    m_ffn_conv2_out = m_dc_conv2_out;

    if (!input_further_use) {
        if (m_adaptor || out_buf.has_value()) {
            g_tensor_pool.release(x);
        }
    }
    if (out_buf.has_value()) {
        if (m_adaptor) {
            g_tensor_pool.release(m_adaptor_out);
        }
    }

    g_tensor_pool.release(m_dc_conv1_out);
    g_tensor_pool.release(m_dc_depth_conv_out);
    g_tensor_pool.release(m_ffn_conv1_out);

    return m_ffn_conv2_out;
}

void DepthConvBlockProxy::release_tensors()
{
    if (m_adaptor_out.defined()) {
        g_tensor_pool.release(m_adaptor_out);
    }
    g_tensor_pool.release(m_ffn_conv2_out);
}

void DepthConvBlockProxy::set_param(const std::map<std::string, at::Tensor>& state_dict,
                                    const bool shortcut)
{
    auto it = state_dict.find("adaptor.weight");
    if (it != state_dict.end()) {
        m_adaptor_weight = state_dict.at("adaptor.weight");
        m_adaptor_bias = state_dict.at("adaptor.bias");
        m_adaptor = true;
    }
    m_dc_conv1_weight = state_dict.at("dc.0.weight");
    m_dc_conv1_bias = state_dict.at("dc.0.bias");
    // Note: d3x3 weight is reordered from [C, 1, 3, 3] to [1, C, 3, 3].
    m_dc_depth_conv_weight =
        state_dict.at("dc.2.weight").permute({ 1, 0, 2, 3 }).contiguous(at::MemoryFormat::ChannelsLast);
    m_dc_conv2_weight = state_dict.at("dc.3.weight");
    m_dc_conv2_bias = at::conv2d(state_dict.at("dc.2.bias").reshape({ 1, -1, 1, 1 }),
                                 state_dict.at("dc.3.weight"));
    m_dc_conv2_bias =
        m_dc_conv2_bias.index({ 0, at::indexing::Slice(), 0, 0 }) + state_dict.at("dc.3.bias");
    m_ffn_conv1_weight = state_dict.at("ffn.0.weight");
    m_ffn_conv1_bias = state_dict.at("ffn.0.bias");

    m_ffn_conv1_weight0 =
        m_ffn_conv1_weight.index({ at::indexing::Slice(0, at::nullopt, 4), at::indexing::Slice(),
                                   at::indexing::Slice(), at::indexing::Slice() });
    m_ffn_conv1_weight1 =
        m_ffn_conv1_weight.index({ at::indexing::Slice(1, at::nullopt, 4), at::indexing::Slice(),
                                   at::indexing::Slice(), at::indexing::Slice() });
    m_ffn_conv1_weight2 =
        m_ffn_conv1_weight.index({ at::indexing::Slice(2, at::nullopt, 4), at::indexing::Slice(),
                                   at::indexing::Slice(), at::indexing::Slice() });
    m_ffn_conv1_weight3 =
        m_ffn_conv1_weight.index({ at::indexing::Slice(3, at::nullopt, 4), at::indexing::Slice(),
                                   at::indexing::Slice(), at::indexing::Slice() });
    m_ffn_conv1_bias0 =
        m_ffn_conv1_bias.index({ at::indexing::Slice(0, at::nullopt, 4) }).contiguous();
    m_ffn_conv1_bias1 =
        m_ffn_conv1_bias.index({ at::indexing::Slice(1, at::nullopt, 4) }).contiguous();
    m_ffn_conv1_bias2 =
        m_ffn_conv1_bias.index({ at::indexing::Slice(2, at::nullopt, 4) }).contiguous();
    m_ffn_conv1_bias3 =
        m_ffn_conv1_bias.index({ at::indexing::Slice(3, at::nullopt, 4) }).contiguous();

    m_ffn_conv2_weight = state_dict.at("ffn.2.weight");
    m_ffn_conv2_bias = state_dict.at("ffn.2.bias");
    m_shortcut = shortcut;
}

at::Tensor ResidualBlockUpsampleProxy::forward(const at::Tensor& x)
{
    auto out = m_up.forward(x);
    return m_conv.forward(out);
}

at::Tensor ResidualBlockUpsampleProxy::pre_allocate_tensors(at::Tensor& x)
{
    auto out = m_up.pre_allocate_tensors(x);
    out = m_conv.pre_allocate_tensors(out);
    return out;
}

void ResidualBlockUpsampleProxy::release_tensors()
{
    m_up.release_tensors();
    m_conv.release_tensors();
}

void ResidualBlockUpsampleProxy::set_param(const std::map<std::string, at::Tensor>& state_dict,
                                           const bool shortcut)
{
    m_up.set_param(get_submodule_state_dict(state_dict, "up."));
    m_conv.set_param(get_submodule_state_dict(state_dict, "conv."), shortcut);
}

at::Tensor ResidualBlockWithStride2Proxy::forward(const at::Tensor& x)
{
    const int sm = get_gpu_sm();
    auto out = conv_bias(sm, x, m_weight, m_bias, 2, m_conv_out);
    out = m_conv.forward(out);
    return out;
}

at::Tensor ResidualBlockWithStride2Proxy::pre_allocate_tensors(at::Tensor& x,
                                                               const at::optional<at::Tensor>& out_buf)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int C = m_weight.size(0);
    m_conv_out = g_tensor_pool.empty({ 1, C, H / 2, W / 2 }, options);
    // because we will downsample first, for the DCB, input will not be further used
    return m_conv.pre_allocate_tensors(m_conv_out, false, out_buf);
}

void ResidualBlockWithStride2Proxy::release_tensors()
{
    g_tensor_pool.release(m_conv_out);
    m_conv.release_tensors();
}

void ResidualBlockWithStride2Proxy::set_param(const std::map<std::string, at::Tensor>& state_dict,
                                              const bool shortcut)
{
    m_weight = state_dict.at("down.weight");
    m_weight = at::pixel_shuffle(m_weight, 2).contiguous(at::MemoryFormat::ChannelsLast);
    m_bias = state_dict.at("down.bias");
    m_conv.set_param(get_submodule_state_dict(state_dict, "conv."), shortcut);
}

at::Tensor SubpelConv2xProxy::forward(const at::Tensor& x)
{
    if (m_bias.defined()) {
        auto out = conv2d(x, m_weight, m_bias, 1, m_conv_out);
        return pixel_shuffle_2_cuda(out, m_shuffle_out);
    }
    return transposed_conv(get_gpu_sm(), x, m_weight_transposed, 2, m_conv_out);
}

at::Tensor SubpelConv2xProxy::pre_allocate_tensors(at::Tensor& x)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int C = m_weight.size(0);
    if (m_bias.defined()) {
        m_conv_out = g_tensor_pool.empty({ 1, C, H, W }, options);
        m_shuffle_out = g_tensor_pool.empty({ 1, C / 4, H * 2, W * 2 }, options);
        return m_shuffle_out;
    }
    m_conv_out = g_tensor_pool.empty({ 1, C / 4, H * 2, W * 2 }, options);
    return m_conv_out;
}

at::Tensor SubpelConv2xProxy::pre_allocate_tensors(at::Tensor& x, at::Tensor& out_buf)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int C = m_weight.size(0);
    if (m_bias.defined()) {
        m_conv_out = g_tensor_pool.empty({ 1, C, H, W }, options);
        m_shuffle_out = out_buf;
        return m_shuffle_out;
    }
    m_conv_out = out_buf;
    return m_conv_out;
}

void SubpelConv2xProxy::release_tensors()
{
    g_tensor_pool.release(m_conv_out);
    g_tensor_pool.release(m_shuffle_out);
}

void SubpelConv2xProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_weight = state_dict.at("conv.0.weight");
    auto it = state_dict.find("conv.0.bias");
    if (it != state_dict.end()) {
        m_bias = state_dict.at("conv.0.bias");
    } else {
        m_weight_transposed =
            at::pixel_shuffle(m_weight.permute({ 1, 0, 2, 3 }), 2).contiguous(at::MemoryFormat::ChannelsLast);
    }
}
