// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
process_with_mask_cuda(const torch::Tensor& y, const torch::Tensor& scales, const torch::Tensor& means,
                       const torch::Tensor& mask, const float force_zero_thres);

void combine_for_reading_2x_cuda(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& mask);
void restore_y_2x_cuda(torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& means,
                       const torch::Tensor& mask);
void restore_y_4x_cuda(torch::Tensor& out, const torch::Tensor& y, const torch::Tensor& means,
                       const torch::Tensor& mask);

void build_index_dec_cuda(torch::Tensor& out, torch::optional<torch::Tensor>& cond_out,
                          const torch::Tensor& scales, const float scale_min, const float scale_max,
                          const float log_scale_min, const float log_step_recip,
                          const float skip_thres);

void build_index_enc_cuda(torch::Tensor& out, torch::optional<torch::Tensor>& cond_out,
                          const torch::Tensor& symbols, const torch::Tensor& scales,
                          const float scale_min, const float scale_max, const float log_scale_min,
                          const float log_step_recip, const float skip_thres);

void bias_wsilu_cuda(torch::Tensor& x, const torch::Tensor& bias);

void bias_shortcut_cuda(torch::Tensor& x, const torch::Tensor& bias, const torch::Tensor& shortcut);
void bias_shortcut_no_inplace_cuda(torch::Tensor& out, const torch::Tensor& x,
                                   const torch::Tensor& bias, const torch::Tensor& shortcut);
void bias_shortcut_2_cuda(torch::Tensor& x, const torch::Tensor& bias, torch::Tensor& shortcut);
void bias_shortcut_with_quant_step_cuda(torch::Tensor& x, const torch::Tensor& bias,
                                        const torch::Tensor& quant_step, const torch::Tensor& shortcut);

void bias_quant_cuda(torch::Tensor& x, const torch::Tensor& bias, const torch::Tensor& quant_step);

void bias_wsilu_chunk_add_cuda(torch::Tensor& x, const torch::Tensor& bias);

void bias_pixel_shuffle_2_cuda(torch::Tensor& out, const torch::Tensor& x,
                               const torch::Tensor& bias, const int C, const int N, const int W);
void bias_pixel_shuffle_8_cuda(torch::Tensor& out, const torch::Tensor& x, const torch::Tensor& bias,
                               const int C, const int N, const int W, bool clamp);
torch::Tensor replicate_pad_cuda(const torch::Tensor& x, const int padB, const int padR);

torch::Tensor round_and_to_int8_cuda(torch::Tensor& z);
torch::Tensor clamp_reciprocal_with_quant_cuda(const torch::Tensor& q_dec, torch::Tensor& y,
                                               const float min_val);
void add_and_multiply_cuda(torch::Tensor& x0, const torch::Tensor& x1, const torch::Tensor q);

torch::Tensor bias_wsilu_depthwise_conv2d_cuda(const torch::Tensor& x, const torch::Tensor& weight,
                                               const torch::Tensor& bias);

class DepthConvProxy {
public:
    DepthConvProxy() = default;
    ~DepthConvProxy() = default;

    void set_param(const torch::Tensor& dc_conv1_weight, const torch::Tensor& dc_conv1_bias,
                   const torch::Tensor& dc_depth_conv_weight,
                   const torch::Tensor& dc_depth_conv_bias, const torch::Tensor& dc_conv2_weight,
                   const torch::Tensor& dc_conv2_bias, const torch::Tensor& ffn_conv1_weight,
                   const torch::Tensor& ffn_conv1_bias, const torch::Tensor& ffn_conv2_weight,
                   const torch::Tensor& ffn_conv2_bias, const bool shortcut);
    void set_param_with_adaptor(
        const torch::Tensor& dc_conv1_weight, const torch::Tensor& dc_conv1_bias,
        const torch::Tensor& dc_depth_conv_weight, const torch::Tensor& dc_depth_conv_bias,
        const torch::Tensor& dc_conv2_weight, const torch::Tensor& dc_conv2_bias,
        const torch::Tensor& ffn_conv1_weight, const torch::Tensor& ffn_conv1_bias,
        const torch::Tensor& ffn_conv2_weight, const torch::Tensor& ffn_conv2_bias,
        const torch::Tensor& adaptor_weight, const torch::Tensor& adaptor_bias, const bool shortcut);
    torch::Tensor forward(const torch::Tensor& x);
    torch::Tensor forward_with_quant_step(const torch::Tensor& x, const torch::Tensor& quant_step);
    torch::Tensor forward_with_cat(const torch::Tensor& x, const torch::Tensor& to_cat,
                                   const bool cat_at_front);

private:
    std::tuple<torch::Tensor, torch::Tensor> forward_common(const torch::Tensor& x);

private:
    torch::Tensor _dc_conv1_weight;
    torch::Tensor _dc_conv1_bias;
    torch::Tensor _dc_depth_conv_weight;
    torch::Tensor _dc_conv2_weight;
    torch::Tensor _dc_conv2_bias;
    torch::Tensor _ffn_conv1_weight;
    torch::Tensor _ffn_conv1_bias;
    torch::Tensor _ffn_conv2_weight;
    torch::Tensor _ffn_conv2_bias;
    bool _adaptor{ false };
    bool _shortcut{ false };
};

class SubpelConv2xProxy {
public:
    SubpelConv2xProxy() = default;
    ~SubpelConv2xProxy() = default;

    void set_param(const torch::Tensor& weight, const torch::Tensor& bias, const int padding);
    torch::Tensor forward(const torch::Tensor& x);
    torch::Tensor forward_with_cat(const torch::Tensor& x, const torch::Tensor& to_cat,
                                   const bool cat_at_front);

private:
    torch::Tensor _weight;
    torch::Tensor _bias;
    int _padding{ 0 };
};
