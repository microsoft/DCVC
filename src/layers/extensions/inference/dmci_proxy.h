// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "dmc_common.h"
#include "layers_proxy.h"

class DMCIDecoderProxy {
public:
    DMCIDecoderProxy() = default;
    ~DMCIDecoderProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::Tensor& quant);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockUpsampleProxy m_up;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
    DepthConvBlockProxy m_conv4;
    DepthConvBlockProxy m_conv5;
    DepthConvBlockProxy m_conv6;
    DepthConvBlockProxy m_conv7;
    DepthConvBlockProxy m_conv8;
    DepthConvBlockProxy m_conv9;
    DepthConvBlockProxy m_conv10;
    DepthConvBlockProxy m_conv11;
    DepthConvBlockProxy m_conv12;
    DepthConvBlockProxy m_dec_2;

    // for cuda graph
    at::Tensor m_shuffle_out;
};

class DMCIEncoderProxy {
public:
    DMCIEncoderProxy() = default;
    ~DMCIEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::Tensor& quant);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_enc_1;
    DepthConvBlockProxy m_enc_2_0;
    DepthConvBlockProxy m_enc_2_1;
    DepthConvBlockProxy m_enc_2_2;
    DepthConvBlockProxy m_enc_2_3;
    DepthConvBlockProxy m_enc_2_4;
    DepthConvBlockProxy m_enc_2_5;
    at::Tensor m_down_weight;
    at::Tensor m_down_bias;

    // for cuda graph
    at::Tensor m_down_out;
};

class DMCIHyperDecoderProxy {
public:
    DMCIHyperDecoderProxy() = default;
    ~DMCIHyperDecoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockUpsampleProxy m_conv0;
    ResidualBlockUpsampleProxy m_conv1;
    DepthConvBlockProxy m_conv2;
};

class DMCIHyperEncoderProxy {
public:
    DMCIHyperEncoderProxy() = default;
    ~DMCIHyperEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    ResidualBlockWithStride2Proxy m_conv1;
    ResidualBlockWithStride2Proxy m_conv2;
};

class DMCISpatialPriorProxy {
public:
    DMCISpatialPriorProxy() = default;
    ~DMCISpatialPriorProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_conv3_weight;
    at::Tensor m_conv3_bias;

    // for cuda graph
    at::Tensor m_conv3_out;
};

class DMCIYPriorFusionProxy {
public:
    DMCIYPriorFusionProxy() = default;
    ~DMCIYPriorFusionProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_conv3_weight;
    at::Tensor m_conv3_bias;

    // for cuda graph
    at::Tensor m_conv3_out;
};

class DMCIProxy : public DMCCommon {
public:
    static constexpr int g_ch_src = 3 * 8 * 8;
    static constexpr int g_ch_enc_dec = 384;
    static constexpr int g_ch_y = 256;
    static constexpr int g_ch_z = 128;

    DMCIProxy();
    ~DMCIProxy();

    py::tuple compress(const at::Tensor& x, const int qp, const int padding_b, const int padding_r);
    at::Tensor decompress(const py::array_t<uint8_t>& bit_stream, const int qp, const int height,
                          const int width, const int entropy_coder_parallel);
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const float skip_threshold);

private:
    // cuda graph related
    void clear_cuda_graph();

    at::Tensor decode_y_step_build_index(const at::Tensor& scales, const at::Tensor& mask,
                                         at::Tensor& skip_cond_out, at::Tensor& indexes_num_out);

    // helper functions
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> get_mask_4x(const at::Tensor& y);

    void pre_allocate_tensors(const int height, const int width);

    void worker();

private:
    at::Tensor m_q_scale_enc[g_qp_num];
    at::Tensor m_q_scale_dec[g_qp_num];
    at::Tensor m_q_scale_y_enc[g_qp_num];
    at::Tensor m_q_scale_y_dec[g_qp_num];

    DMCIEncoderProxy m_encoder;
    DMCIHyperEncoderProxy m_hyper_encoder;
    DMCIHyperDecoderProxy m_hyper_decoder;
    DMCIYPriorFusionProxy m_y_prior_fusion;
    ConvParam m_y_spatial_prior_reduction;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_1;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_2;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_3;
    DMCISpatialPriorProxy m_y_spatial_prior;
    DMCIDecoderProxy m_decoder;

    float m_skip_threshold{ 0.f };

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    bool m_pending{ false };
    DMCWorkType m_pending_work;

    uint8_t* m_bit_stream{ nullptr };
    int m_bit_stream_size{ 0 };
    int m_z_width{ 0 };
    int m_z_height{ 0 };

    cudaEvent_t m_event_y;
    cudaEvent_t m_event_z_ready;
    cudaEvent_t m_event_y_ready;
    int* m_size_ptr{ nullptr };

    int m_qp{ 0 };
    std::shared_ptr<std::vector<int16_t>> m_y_to_encode[4]{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> m_y_to_decode{ nullptr };
    int m_y_to_encode_size[4]{ 0 };

    at::ScalarType m_dtype;
    at::Device m_device{ at::kCUDA };

    // for cuda graph
    int m_H8{ 0 };
    int m_W8{ 0 };
    cudaGraphExec_t m_gexec_enc_0[g_qp_num]{ nullptr };
    cudaGraphExec_t m_gexec_enc_1[g_qp_num]{ nullptr };
    cudaGraphExec_t m_gexec_dec_0{ nullptr };
    cudaGraphExec_t m_gexec_dec_1{ nullptr };
    cudaGraphExec_t m_gexec_dec_2{ nullptr };
    cudaGraphExec_t m_gexec_dec_3{ nullptr };
    cudaGraphExec_t m_gexec_dec_4[g_qp_num]{ nullptr };
    at::Tensor m_mask_0;
    at::Tensor m_mask_1;
    at::Tensor m_mask_2;
    at::Tensor m_mask_3;
    at::Tensor m_enc_unshuffle_out;
    at::Tensor m_y;
    at::Tensor m_y_pad;
    at::Tensor m_z_hat;
    at::Tensor m_z_hat_io;
    at::Tensor m_hyper_params;
    at::Tensor m_cropped_params;
    at::Tensor m_y_q;
    at::Tensor m_y_hat;
    at::Tensor m_s_hat;
    at::Tensor m_reduced_params;
    at::Tensor m_y_q_w;
    at::Tensor m_s_w;
    at::Tensor m_y_symbol[4];
    at::Tensor m_y_symbol_prev;
    at::Tensor m_skip_cond;
    at::Tensor m_y_size[4];
    at::Tensor m_y_num;
    at::Tensor m_x_hat;
    at::Tensor m_y_q_r;
    at::Tensor m_scales_r;
    at::Tensor m_y_decoded;
    at::Tensor m_y_indexes;
    at::Tensor m_y_indexes_prev;
    at::Tensor m_y_indexes_skip_cond;
    at::Tensor m_y_indexes_size;
    at::Tensor m_y_indexes_num;
    at::Tensor m_spatial_prior_out;

    // concatenated tensor for spatial prior adaptor input
    at::Tensor m_cat_spatial_prior_adaptor;
    at::Tensor m_y_hat_so_far;  // view into m_cat_spatial_prior_adaptor
};
