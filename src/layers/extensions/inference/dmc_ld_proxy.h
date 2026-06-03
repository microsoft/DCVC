// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "dmc_common.h"

class DMCLDDecoderProxy {
public:
    DMCLDDecoderProxy() = default;
    ~DMCLDDecoderProxy() = default;

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
    at::Tensor m_conv2_weight;
    at::Tensor m_conv2_bias;

    // for cuda graphs
    at::Tensor m_conv2_out;
};

class DMCLDEncoderProxy {
public:
    DMCLDEncoderProxy() = default;
    ~DMCLDEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x, const at::Tensor& quant);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv10;
    DepthConvBlockProxy m_conv11;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_down_weight;
    at::Tensor m_down_bias;

    // for cuda graph
    at::Tensor m_down_out;
};

class DMCLDFeatureAdaptorIProxy {
public:
    DMCLDFeatureAdaptorIProxy() = default;
    ~DMCLDFeatureAdaptorIProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
};

class DMCLDFeatureAdaptorMProxy {
public:
    DMCLDFeatureAdaptorMProxy() = default;
    ~DMCLDFeatureAdaptorMProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    DepthConvBlockProxy m_conv3;
};

class DMCLDFeatureExtractorProxy {
public:
    DMCLDFeatureExtractorProxy() = default;
    ~DMCLDFeatureExtractorProxy() = default;

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

class DMCLDHyperDecoderProxy {
public:
    DMCLDHyperDecoderProxy() = default;
    ~DMCLDHyperDecoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockUpsampleProxy m_conv0;
    ResidualBlockUpsampleProxy m_conv1;
    DepthConvBlockProxy m_conv2;
};

class DMCLDHyperEncoderProxy {
public:
    DMCLDHyperEncoderProxy() = default;
    ~DMCLDHyperEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    ResidualBlockWithStride2Proxy m_conv1;
    ResidualBlockWithStride2Proxy m_conv2;
};

class DMCLDPriorFusionProxy {
public:
    DMCLDPriorFusionProxy() = default;
    ~DMCLDPriorFusionProxy() = default;

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

class DMCLDReconHeadProxy {
public:
    DMCLDReconHeadProxy() = default;
    ~DMCLDReconHeadProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor forward_reset(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    DepthConvBlockProxy m_conv2;
    at::Tensor m_head_weight;
    at::Tensor m_head_bias;

    // for cuda graphs
    at::Tensor m_head_out;
    at::Tensor m_shuffle_out;
};

class DMCLDSpatialPriorProxy {
public:
    DMCLDSpatialPriorProxy() = default;
    ~DMCLDSpatialPriorProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    DepthConvBlockProxy m_conv0;
    DepthConvBlockProxy m_conv1;
    at::Tensor m_conv2_weight;
    at::Tensor m_conv2_bias;

    // for cuda graphs
    at::Tensor m_conv2_out;
};

class DMCLDTemporalPriorEncoderProxy {
public:
    DMCLDTemporalPriorEncoderProxy() = default;
    ~DMCLDTemporalPriorEncoderProxy() = default;

    at::Tensor forward(const at::Tensor& x);
    at::Tensor pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf);
    void set_param(const std::map<std::string, at::Tensor>& state_dict);

private:
    ResidualBlockWithStride2Proxy m_conv;
};

class DMCLDProxy : public DMCCommon {
public:
    static constexpr int g_ch_src_d = 3 * 8 * 8;
    static constexpr int g_ch_y = 128;
    static constexpr int g_ch_z = 128;
    static constexpr int g_ch_d = 256;
    static constexpr int g_ch_m = 256;

    DMCLDProxy();
    ~DMCLDProxy();

    void add_ref_feature_from_frame(const at::Tensor& frame, const bool apply_adaptor);
    py::tuple compress(const at::Tensor& x, const int qp, const bool reset_feature_memory,
                       const int padding_b, const int padding_r);
    at::Tensor decompress(const py::array_t<uint8_t>& bit_stream, const int qp, const int height,
                          const int width, const int entropy_coder_parallel,
                          const bool reset_feature_memory);
    void set_param(const std::map<std::string, at::Tensor>& state_dict, const float skip_threshold);

private:
    // compress and decompress components
    void add_ref_feature_and_apply_adaptor(const at::Tensor& feature, const bool reset_feature_memory);
    void apply_feature_adaptor(const bool memory_none);

    // cuda graph related
    void clear_cuda_graph();

    // helper functions
    std::tuple<at::Tensor, at::Tensor> get_mask_2x(const at::Tensor& y);

    void pre_allocate_tensors(const at::Tensor& frame);
    void worker();

private:
    at::Tensor m_q_encoder[g_qp_num];
    at::Tensor m_q_decoder[g_qp_num];
    at::Tensor m_q_feature[g_qp_num];

    DMCLDEncoderProxy m_encoder;
    DMCLDHyperEncoderProxy m_hyper_encoder;
    DMCLDTemporalPriorEncoderProxy m_temporal_prior_encoder;
    DMCLDHyperDecoderProxy m_hyper_decoder;
    DMCLDPriorFusionProxy m_y_prior_fusion;
    DMCLDSpatialPriorProxy m_y_spatial_prior;
    DMCLDDecoderProxy m_decoder;
    DMCLDFeatureAdaptorIProxy m_feature_adaptor_i;
    DMCLDFeatureAdaptorMProxy m_feature_adaptor_m;
    DMCLDFeatureExtractorProxy m_feature_extractor;
    DMCLDReconHeadProxy m_recon_head;

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
    cudaGraphExec_t m_gexec_dec_0[2]{ nullptr };
    cudaGraphExec_t m_gexec_dec_1[g_qp_num]{ nullptr };
    cudaGraphExec_t m_gexec_dec_2{ nullptr };
    cudaGraphExec_t m_gexec_dec_3[g_qp_num]{ nullptr };
    bool m_memory_has_value{ true };
    at::Tensor m_mask_0;
    at::Tensor m_mask_1;
    at::Tensor m_enc_unshuffle_out;
    at::Tensor m_feature_i;  // output of both image frame unshuffle and recon_head reset
    at::Tensor m_feature_p;  // output of decoder
    at::Tensor m_memory;     // output of both feature_adaptor_i and feature_adaptor_m
    at::Tensor m_ctx;
    at::Tensor m_temporal_params;
    at::Tensor m_y;
    at::Tensor m_y_pad;
    at::Tensor m_hyper_params;
    at::Tensor m_hyper_params_pad4;
    at::Tensor m_z_hat;
    at::Tensor m_z_hat_io;
    at::Tensor m_common_params;
    at::Tensor m_means;
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
    at::Tensor m_x_hat;  // no allocation

    // 5 concatenated tensors
    at::Tensor m_cat_encoder;
    at::Tensor m_cat_decoder;
    at::Tensor m_cat_feature_adaptor_m;
    at::Tensor m_cat_prior_fusion;
    at::Tensor m_cat_spatial_prior;
};
