// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "dmc_common.h"

class DMCHTSDecoderProxy {
public:
    DMCHTSDecoderProxy() = default;
    ~DMCHTSDecoderProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::Tensor& conv1_input, const at::Tensor& quant);
    at::Tensor forward_part1(const at::Tensor& x);
    at::Tensor forward_part2(const at::Tensor& x, const at::Tensor& quant);
    at::Tensor pre_allocate_tensors(at::Tensor& x, at::Tensor& up_out_buf,
                                    at::Tensor& conv1_input_buf, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    SubpelConv2xProxy m_up;
    DepthConvBlockProxy m_conv10;
    DepthConvBlockProxy m_conv11;
    DepthConvBlockProxy m_conv12;
    DepthConvBlockProxy m_conv13;
    DepthConvBlockProxy m_conv14;
    DepthConvBlockProxy m_conv15;
    DepthConvBlockProxy m_conv16;

    // for cuda graphs
    at::Tensor m_conv1_out;
};

class DMCHTSEncoderProxy {
public:
    DMCHTSEncoderProxy() = default;
    ~DMCHTSEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::Tensor& quant);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv10;
    DepthConvBlockProxy m_conv11;
    DepthConvBlockProxy m_conv12;
    DepthConvBlockProxy m_conv13;
    DepthConvBlockProxy m_conv14;
    DepthConvBlockProxy m_conv15;
    at::Tensor m_down_weight;
    at::Tensor m_down_bias;

    // for cuda graph
    at::Tensor m_down_out;
};

class DMCHTSFeatureAdaptorIProxy {
public:
    DMCHTSFeatureAdaptorIProxy() = default;
    ~DMCHTSFeatureAdaptorIProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
};

class DMCHTSFeatureAdaptorMProxy {
public:
    DMCHTSFeatureAdaptorMProxy() = default;
    ~DMCHTSFeatureAdaptorMProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
    DepthConvBlockProxy m_conv4;
    DepthConvBlockProxy m_conv5;
};

class DMCHTSFeatureExtractorProxy {
public:
    DMCHTSFeatureExtractorProxy() = default;
    ~DMCHTSFeatureExtractorProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
    DepthConvBlockProxy m_conv4;
};

class DMCHTSHyperDecoderProxy {
public:
    DMCHTSHyperDecoderProxy() = default;
    ~DMCHTSHyperDecoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockUpsampleProxy m_conv0;
    ResidualBlockUpsampleProxy m_conv1;
    DepthConvBlockProxy m_conv2;
};

class DMCHTSHyperEncoderProxy {
public:
    DMCHTSHyperEncoderProxy() = default;
    ~DMCHTSHyperEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    ResidualBlockWithStride2Proxy m_conv1;
    ResidualBlockWithStride2Proxy m_conv2;
};

class DMCHTSPriorFusionProxy {
public:
    DMCHTSPriorFusionProxy() = default;
    ~DMCHTSPriorFusionProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_conv3_weight;
    at::Tensor m_conv3_bias;

    // for cuda graphs
    at::Tensor m_conv3_out;
};

class DMCHTSReconHeadProxy {
public:
    static constexpr int g_frame_delay = 8;
    DMCHTSReconHeadProxy() = default;
    ~DMCHTSReconHeadProxy() = default;

    std::tuple<std::vector<at::Tensor>, at::Tensor> forward(const at::Tensor& x);
    at::Tensor forward_reset(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv1_0[g_frame_delay / 2];
    DepthConvBlockProxy m_conv2_0[g_frame_delay];
    DepthConvBlockProxy m_conv2_1[g_frame_delay];
    DepthConvBlockProxy m_conv2_2[g_frame_delay];
    at::Tensor m_head_weight[g_frame_delay];
    at::Tensor m_head_bias[g_frame_delay];

    // for cuda graphs
    at::Tensor m_head_out[g_frame_delay];
    at::Tensor m_shuffle_out[g_frame_delay];
};

class DMCHTSSpatialPriorProxy {
public:
    DMCHTSSpatialPriorProxy() = default;
    ~DMCHTSSpatialPriorProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_conv3_weight;
    at::Tensor m_conv3_bias;

    // for cuda graphs
    at::Tensor m_conv3_out;
};

class DMCHTSTemporalPriorEncoderProxy {
public:
    DMCHTSTemporalPriorEncoderProxy() = default;
    ~DMCHTSTemporalPriorEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockWithStride2Proxy m_conv;
};

class DMCHTSProxy : public DMCCommon {
public:
    static constexpr int g_frame_delay = 8;
    static constexpr int g_ch_src_d_intra = 3 * 8 * 8;
    static constexpr int g_ch_src_d = g_ch_src_d_intra * g_frame_delay;
    static constexpr int g_ch_y = 256;
    static constexpr int g_ch_z = 128;
    static constexpr int g_ch_d = 512;
    static constexpr int g_ch_m = 512;
    static constexpr int g_ch_recon = 256;

    DMCHTSProxy();
    ~DMCHTSProxy();

    void add_ref_feature_from_frame(const at::Tensor& frame, const bool apply_adaptor);
    py::tuple compress(const at::Tensor& x, const int qp, const bool reset_feature_memory,
                       const int padding_b, const int padding_r);
    std::vector<at::Tensor> decompress(const py::array_t<uint8_t>& bit_stream, const int qp,
                                       const int height, const int width,
                                       const int entropy_coder_parallel,
                                       const bool reset_feature_memory);
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const float skip_threshold);

private:
    void worker();

public:
private:
    // compress and decompress components
    void add_ref_feature_and_apply_adaptor(const at::Tensor& feature, const bool reset_feature_memory);
    void apply_feature_adaptor(const bool memory_none);

    // cuda graph related
    void clear_cuda_graph();

    // helper functions
    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> get_mask_4x(const at::Tensor& y);

    void pre_allocate_tensors(const at::Tensor& frame);

private:
    at::Tensor m_q_encoder[g_qp_num];
    at::Tensor m_q_decoder[g_qp_num];
    at::Tensor m_q_feature[g_qp_num];

    DMCHTSEncoderProxy m_encoder;
    DMCHTSHyperEncoderProxy m_hyper_encoder;
    DMCHTSTemporalPriorEncoderProxy m_temporal_prior_encoder;
    DMCHTSHyperDecoderProxy m_hyper_decoder;
    DMCHTSPriorFusionProxy m_y_prior_fusion;
    ConvParam m_y_spatial_prior_reduction;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_1;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_2;
    DepthConvBlockProxy m_y_spatial_prior_adaptor_3;
    DMCHTSSpatialPriorProxy m_y_spatial_prior;
    DMCHTSDecoderProxy m_decoder;
    DMCHTSFeatureAdaptorIProxy m_feature_adaptor_i;
    DMCHTSFeatureAdaptorMProxy m_feature_adaptor_m;
    DMCHTSFeatureExtractorProxy m_feature_extractor;
    DMCHTSReconHeadProxy m_recon_head;

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
    std::shared_ptr<std::vector<int16_t>> m_y_to_encode{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> m_y_to_decode{ nullptr };

    at::ScalarType m_dtype;
    at::Device m_device{ at::kCUDA };

    // for cuda graph
    int m_H8{ 0 };
    int m_W8{ 0 };
    cudaGraphExec_t m_gexec_enc_0[g_qp_num]{ nullptr };
    cudaGraphExec_t m_gexec_enc_1[g_qp_num][2]{ nullptr };
    cudaGraphExec_t m_gexec_dec_0[g_qp_num][2]{ nullptr };
    cudaGraphExec_t m_gexec_dec_1{ nullptr };
    cudaGraphExec_t m_gexec_dec_2{ nullptr };
    cudaGraphExec_t m_gexec_dec_3[g_qp_num]{ nullptr };
    bool m_memory_has_value{ true };
    at::Tensor m_mask_0;
    at::Tensor m_mask_1;
    at::Tensor m_mask_2;
    at::Tensor m_mask_3;
    at::Tensor m_enc_unshuffle_out;
    at::Tensor m_feature_i;  // output of image frame unshuffle or recon_head reset
    at::Tensor m_feature_p;  // output of decoder
    at::Tensor m_memory;     // output of both feature_adaptor_i and feature_adaptor_m
    at::Tensor m_ctx;
    at::Tensor m_temporal_input;
    at::Tensor m_temporal_params;
    at::Tensor m_y;
    at::Tensor m_y_pad;
    at::Tensor m_hyper_params;
    at::Tensor m_hyper_params_pad4;
    at::Tensor m_common_params;  // no allocation
    at::Tensor m_z_hat;
    at::Tensor m_z_hat_io;
    at::Tensor m_means;
    at::Tensor m_reduced_params;
    at::Tensor m_y_q;
    at::Tensor m_y_q_r;  // only for decompress
    at::Tensor m_y_hat;
    at::Tensor m_y_symbol;
    at::Tensor m_y_symbol_prev;
    at::Tensor m_y_indexes;
    at::Tensor m_y_indexes_prev;
    at::Tensor m_skip_cond;
    at::Tensor m_y_decoded;
    at::Tensor m_y_size;
    at::Tensor m_y_num;
    at::Tensor m_decoder_up_out;
    std::vector<at::Tensor> m_x_hat;  // no allocation. length: g_frame_delay

    // 5 concatenated tensors
    at::Tensor m_cat_encoder;
    at::Tensor m_cat_decoder;
    at::Tensor m_cat_feature_adaptor_m;
    at::Tensor m_cat_prior_fusion;
    at::Tensor m_cat_spatial_prior_adaptor;
};
