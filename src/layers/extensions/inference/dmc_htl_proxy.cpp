// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <c10/cuda/CUDAGuard.h>
#include <torch/torch.h>

#include "common_cpp.h"
#include "cuda_check.h"
#include "def_cutlass.h"
#include "def_elementwise.h"
#include "dmc_htl_proxy.h"
#include "memory_pool.h"

at::Tensor DMCHTLDecoderProxy::forward(const at::Tensor& x, const at::Tensor& conv1_input,
                                       const at::Tensor& quant)
{
    auto out = forward_part1(x);
    out = forward_part2(conv1_input, quant);
    return out;
}

at::Tensor DMCHTLDecoderProxy::forward_part1(const at::Tensor& x)
{
    return m_up.forward(x);
}

at::Tensor DMCHTLDecoderProxy::forward_part2(const at::Tensor& x, const at::Tensor& quant)
{
    auto out = m_conv10.forward(x);
    out = m_conv11.forward(out);
    out = m_conv12.forward(out);
    out = m_conv13.forward(out);
    out = m_conv14.forward(out);
    out = m_conv15.forward(out);
    out = m_conv16.forward(out);
    out = m_conv17.forward(out);
    out = m_conv18.forward(out);
    out = m_conv19.forward(out);
    out = m_conv1a.forward(out);
    out = multiply_with_broadcast_cuda(out, quant, out);
    return out;
}

at::Tensor DMCHTLDecoderProxy::pre_allocate_tensors(at::Tensor& x, at::Tensor& up_out_buf,
                                                    at::Tensor& conv1_input_buf,
                                                    const at::Tensor& out_buf)
{
    m_up.pre_allocate_tensors(x, up_out_buf);
    m_conv1_out = out_buf;
    auto out = m_conv10.pre_allocate_tensors(conv1_input_buf, true);
    out = m_conv11.pre_allocate_tensors(out);
    out = m_conv12.pre_allocate_tensors(out);
    out = m_conv13.pre_allocate_tensors(out);
    out = m_conv14.pre_allocate_tensors(out);
    out = m_conv15.pre_allocate_tensors(out);
    out = m_conv16.pre_allocate_tensors(out);
    out = m_conv17.pre_allocate_tensors(out);
    out = m_conv18.pre_allocate_tensors(out);
    out = m_conv19.pre_allocate_tensors(out);
    out = m_conv1a.pre_allocate_tensors(out, false, m_conv1_out);

    m_conv10.release_tensors();
    m_conv11.release_tensors();
    m_conv12.release_tensors();
    m_conv13.release_tensors();
    m_conv14.release_tensors();
    m_conv15.release_tensors();
    m_conv16.release_tensors();
    m_conv17.release_tensors();
    m_conv18.release_tensors();
    m_conv19.release_tensors();
    m_conv1a.release_tensors();
    g_tensor_pool.set_reusable(m_conv1_out, false);
    return m_conv1_out;
}

void DMCHTLDecoderProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_up.set_param(get_submodule_state_dict(state_dict, "up."));
    m_conv10.set_param(get_submodule_state_dict(state_dict, "conv1.0."));
    m_conv11.set_param(get_submodule_state_dict(state_dict, "conv1.1."));
    m_conv12.set_param(get_submodule_state_dict(state_dict, "conv1.2."));
    m_conv13.set_param(get_submodule_state_dict(state_dict, "conv1.3."));
    m_conv14.set_param(get_submodule_state_dict(state_dict, "conv1.4."));
    m_conv15.set_param(get_submodule_state_dict(state_dict, "conv1.5."));
    m_conv16.set_param(get_submodule_state_dict(state_dict, "conv1.6."));
    m_conv17.set_param(get_submodule_state_dict(state_dict, "conv1.7."));
    m_conv18.set_param(get_submodule_state_dict(state_dict, "conv1.8."));
    m_conv19.set_param(get_submodule_state_dict(state_dict, "conv1.9."));
    m_conv1a.set_param(get_submodule_state_dict(state_dict, "conv1.10."));
}

at::Tensor DMCHTLEncoderProxy::forward(const at::Tensor& x, const at::Tensor& quant)
{
    const int sm = get_gpu_sm();
    auto out = m_conv10.forward(x);
    out = m_conv11.forward(out);
    out = m_conv12.forward(out);
    out = m_conv13.forward(out);
    out = m_conv14.forward(out);
    out = m_conv15.forward(out);
    out = m_conv16.forward(out);
    out = multiply_with_broadcast_cuda(out, quant, out);
    out = conv_bias(sm, out, m_down_weight, m_down_bias, 2, m_down_out);
    return out;
}

at::Tensor DMCHTLEncoderProxy::pre_allocate_tensors(at::Tensor& x)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_down_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int ch_down = m_down_weight.size(0);
    m_down_out = g_tensor_pool.empty({ 1, ch_down, H / 2, W / 2 }, options);
    auto out = m_conv10.pre_allocate_tensors(x, true);
    out = m_conv11.pre_allocate_tensors(out);
    out = m_conv12.pre_allocate_tensors(out);
    out = m_conv13.pre_allocate_tensors(out);
    out = m_conv14.pre_allocate_tensors(out);
    out = m_conv15.pre_allocate_tensors(out);
    out = m_conv16.pre_allocate_tensors(out);

    m_conv10.release_tensors();
    m_conv11.release_tensors();
    m_conv12.release_tensors();
    m_conv13.release_tensors();
    m_conv14.release_tensors();
    m_conv15.release_tensors();
    m_conv16.release_tensors();
    return m_down_out;
}

void DMCHTLEncoderProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv10.set_param(get_submodule_state_dict(state_dict, "conv1.0."));
    m_conv11.set_param(get_submodule_state_dict(state_dict, "conv1.1."));
    m_conv12.set_param(get_submodule_state_dict(state_dict, "conv1.2."));
    m_conv13.set_param(get_submodule_state_dict(state_dict, "conv1.3."));
    m_conv14.set_param(get_submodule_state_dict(state_dict, "conv1.4."));
    m_conv15.set_param(get_submodule_state_dict(state_dict, "conv1.5."));
    m_conv16.set_param(get_submodule_state_dict(state_dict, "conv1.6."));
    m_down_weight = state_dict.at("down.weight");
    m_down_bias = state_dict.at("down.bias");
}

at::Tensor DMCHTLFeatureAdaptorIProxy::forward(const at::Tensor& x)
{
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    return out;
}

at::Tensor DMCHTLFeatureAdaptorIProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    auto out = m_conv0.pre_allocate_tensors(x, true);
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out, false, out_buf);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLFeatureAdaptorIProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."));
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."));
}

at::Tensor DMCHTLFeatureAdaptorMProxy::forward(const at::Tensor& x)
{
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    out = m_conv3.forward(out);
    out = m_conv4.forward(out);
    out = m_conv5.forward(out);
    out = m_conv6.forward(out);
    out = m_conv7.forward(out);
    out = m_conv8.forward(out);
    out = m_conv9.forward(out);
    return out;
}

at::Tensor DMCHTLFeatureAdaptorMProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    auto out = m_conv0.pre_allocate_tensors(x, true);
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out);
    out = m_conv3.pre_allocate_tensors(out);
    out = m_conv4.pre_allocate_tensors(out);
    out = m_conv5.pre_allocate_tensors(out);
    out = m_conv6.pre_allocate_tensors(out);
    out = m_conv7.pre_allocate_tensors(out);
    out = m_conv8.pre_allocate_tensors(out);
    out = m_conv9.pre_allocate_tensors(out, false, out_buf);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    m_conv3.release_tensors();
    m_conv4.release_tensors();
    m_conv5.release_tensors();
    m_conv6.release_tensors();
    m_conv7.release_tensors();
    m_conv8.release_tensors();
    m_conv9.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLFeatureAdaptorMProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."));
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."));
    m_conv3.set_param(get_submodule_state_dict(state_dict, "conv.3."));
    m_conv4.set_param(get_submodule_state_dict(state_dict, "conv.4."));
    m_conv5.set_param(get_submodule_state_dict(state_dict, "conv.5."));
    m_conv6.set_param(get_submodule_state_dict(state_dict, "conv.6."));
    m_conv7.set_param(get_submodule_state_dict(state_dict, "conv.7."));
    m_conv8.set_param(get_submodule_state_dict(state_dict, "conv.8."));
    m_conv9.set_param(get_submodule_state_dict(state_dict, "conv.9."));
}

at::Tensor DMCHTLFeatureExtractorProxy::forward(const at::Tensor& x)
{
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    return out;
}

at::Tensor DMCHTLFeatureExtractorProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    auto out = m_conv0.pre_allocate_tensors(x, true);  // must not modify!
    out = m_conv1.pre_allocate_tensors(out, false, out_buf);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLFeatureExtractorProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."));
}

at::Tensor DMCHTLHyperDecoderProxy::forward(const at::Tensor& x)
{
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    return out;
}

at::Tensor DMCHTLHyperDecoderProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    auto out = m_conv0.pre_allocate_tensors(x);
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out, false, out_buf);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLHyperDecoderProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."), true);
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."), true);
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."));
}

at::Tensor DMCHTLHyperEncoderProxy::forward(const at::Tensor& x)
{
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    return out;
}

at::Tensor DMCHTLHyperEncoderProxy::pre_allocate_tensors(at::Tensor& x)
{
    auto out = m_conv0.pre_allocate_tensors(x, true);  // must not modify!
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLHyperEncoderProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."), true);
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."), true);
}

at::Tensor DMCHTLPriorFusionProxy::forward(const at::Tensor& x)
{
    const int sm = get_gpu_sm();
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    out = conv1x1_bias(sm, out, m_conv3_weight, m_conv3_bias, m_conv3_out);
    return out;
}

at::Tensor DMCHTLPriorFusionProxy::pre_allocate_tensors(at::Tensor& x)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_conv3_weight.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int ch_conv3 = m_conv3_weight.size(0);
    m_conv3_out = g_tensor_pool.empty({ 1, ch_conv3, H, W }, options);
    // x is a cat parameter, will be further used
    auto out = m_conv0.pre_allocate_tensors(x, true);
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    return m_conv3_out;
}

void DMCHTLPriorFusionProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."));
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."));
    m_conv3_weight = state_dict.at("conv.3.weight");
    m_conv3_bias = state_dict.at("conv.3.bias");
}

at::Tensor DMCHTLReconHeadProxy::forward_reset(const at::Tensor& x)
{
    constexpr int conv_index = g_frame_delay - 1;
    const int sm = get_gpu_sm();
    auto out = m_conv0[conv_index].forward(x);
    out = m_conv1[conv_index].forward(out);
    out = m_conv2[conv_index].forward(out);
    out = m_conv3[conv_index].forward(out);
    out = m_conv4[conv_index].forward(out);
    out = conv1x1_bias(sm, out, m_head_weight[conv_index], m_head_bias[conv_index],
                       m_head_out[conv_index]);
    return out;
}

std::tuple<std::vector<at::Tensor>, at::Tensor> DMCHTLReconHeadProxy::forward(const at::Tensor& x)
{
    at::Tensor for_reset;
    const int sm = get_gpu_sm();
    auto all_out = std::vector<at::Tensor>(g_frame_delay);
    for (int i = 0; i < g_frame_delay; i++) {
        auto out = m_conv0[i].forward(x);
        out = m_conv1[i].forward(out);
        out = m_conv2[i].forward(out);
        out = m_conv3[i].forward(out);
        out = m_conv4[i].forward(out);
        out = conv1x1_bias(sm, out, m_head_weight[i], m_head_bias[i], m_head_out[i]);
        if (i == g_frame_delay - 1) {
            for_reset = out;
        }
        out = pixel_shuffle_8_cuda(out, true, m_shuffle_out[i]);
        all_out[i] = out;
    }
    return { all_out, for_reset };
}

at::Tensor DMCHTLReconHeadProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    const int H = x.size(2);
    const int W = x.size(3);
    auto options = m_head_weight[0].options().memory_format(at::MemoryFormat::ChannelsLast);
    const int C = m_head_weight[0].size(0);
    for (int i = 0; i < g_frame_delay; i++) {
        if (i != g_frame_delay - 1) {
            m_head_out[i] = g_tensor_pool.empty({ 1, C, H, W }, options);
        } else {
            m_head_out[i] = out_buf;
        }
        m_shuffle_out[i] = at::empty({ 1, C / 64, H * 8, W * 8 }, options);

        auto out = m_conv0[i].pre_allocate_tensors(x, true);  // must not modify!
        out = m_conv1[i].pre_allocate_tensors(out);
        out = m_conv2[i].pre_allocate_tensors(out);
        out = m_conv3[i].pre_allocate_tensors(out);
        out = m_conv4[i].pre_allocate_tensors(out);
        if (i != g_frame_delay - 1) {
            g_tensor_pool.release(m_head_out[i]);
        }
        m_conv0[i].release_tensors();
        m_conv1[i].release_tensors();
        m_conv2[i].release_tensors();
        m_conv3[i].release_tensors();
        m_conv4[i].release_tensors();
    }
    return m_head_out[g_frame_delay - 1];
}

void DMCHTLReconHeadProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    for (int i = 0; i < g_frame_delay; i++) {
        m_conv0[i].set_param(get_submodule_state_dict(state_dict, "conv." + std::to_string(i) + ".0."));
        m_conv1[i].set_param(get_submodule_state_dict(state_dict, "conv." + std::to_string(i) + ".1."));
        m_conv2[i].set_param(get_submodule_state_dict(state_dict, "conv." + std::to_string(i) + ".2."));
        m_conv3[i].set_param(get_submodule_state_dict(state_dict, "conv." + std::to_string(i) + ".3."));
        m_conv4[i].set_param(get_submodule_state_dict(state_dict, "conv." + std::to_string(i) + ".4."));
        m_head_weight[i] = state_dict.at("conv." + std::to_string(i) + ".5.weight");
        m_head_bias[i] = state_dict.at("conv." + std::to_string(i) + ".5.bias");
    }
}

at::Tensor DMCHTLSpatialPriorProxy::forward(const at::Tensor& x)
{
    const int sm = get_gpu_sm();
    auto out = m_conv0.forward(x);
    out = m_conv1.forward(out);
    out = m_conv2.forward(out);
    out = conv1x1_bias(sm, out, m_conv3_weight, m_conv3_bias, m_conv3_out);
    return out;
}

at::Tensor DMCHTLSpatialPriorProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    m_conv3_out = out_buf;
    auto out = m_conv0.pre_allocate_tensors(x, true);
    out = m_conv1.pre_allocate_tensors(out);
    out = m_conv2.pre_allocate_tensors(out);

    m_conv0.release_tensors();
    m_conv1.release_tensors();
    m_conv2.release_tensors();
    return m_conv3_out;
}

void DMCHTLSpatialPriorProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv0.set_param(get_submodule_state_dict(state_dict, "conv.0."));
    m_conv1.set_param(get_submodule_state_dict(state_dict, "conv.1."));
    m_conv2.set_param(get_submodule_state_dict(state_dict, "conv.2."));
    m_conv3_weight = state_dict.at("conv.3.weight");
    m_conv3_bias = state_dict.at("conv.3.bias");
}

at::Tensor DMCHTLTemporalPriorEncoderProxy::forward(const at::Tensor& x)
{
    return m_conv.forward(x);
}

at::Tensor DMCHTLTemporalPriorEncoderProxy::pre_allocate_tensors(at::Tensor& x, const at::Tensor& out_buf)
{
    auto out = m_conv.pre_allocate_tensors(x, out_buf);
    m_conv.release_tensors();
    g_tensor_pool.set_reusable(out, false);
    return out;
}

void DMCHTLTemporalPriorEncoderProxy::set_param(const std::map<std::string, at::Tensor>& state_dict)
{
    m_conv.set_param(get_submodule_state_dict(state_dict, "conv."), true);
}

DMCHTLProxy::DMCHTLProxy()
{
    CUDA_CHECK(cudaMallocHost((void**)&m_size_ptr, sizeof(int)));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_event_y, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_event_y_ready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&m_event_z_ready, cudaEventDisableTiming));
    m_thread = std::thread(&DMCHTLProxy::worker, this);
}

DMCHTLProxy::~DMCHTLProxy()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        m_finish = true;
    }
    m_cv_pending.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
    cudaEventDestroy(m_event_y);
    cudaEventDestroy(m_event_y_ready);
    cudaEventDestroy(m_event_z_ready);
    cudaFreeHost(m_size_ptr);

    clear_cuda_graph();
}

void DMCHTLProxy::worker()
{
    at::cuda::CUDAStream high_priority_stream = at::cuda::getStreamFromPool(true);
    at::cuda::CUDAStreamGuard guard(high_priority_stream);

    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return m_pending || m_finish; });
        if (m_finish) {
            break;
        }
        m_pending = false;
        lk.unlock();

        if (m_pending_work == DMCWorkType::Encode) {
            m_entropy_encoder.reset();
            CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), m_event_y));

            // build_index_enc and conditional_index for each of the 4 mask steps
            // Process in reverse order (3,2,1,0) to match Python encode order
            int total_symbol_count = 0;
            for (int i = 3; i >= 0; i--) {
                auto out = conditional_index_part2_cuda(m_y_symbol[i], m_y_size[i], m_size_ptr);
                m_y_to_encode_size[i] = *m_size_ptr;
                total_symbol_count += *m_size_ptr;
                CUDA_CHECK(cudaMemcpyAsync(m_y_to_encode[i]->data(), out.data_ptr<int16_t>(),
                                           (*m_size_ptr) * sizeof(int16_t), cudaMemcpyDeviceToHost,
                                           at::cuda::getCurrentCUDAStream()));
            }
            CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));

            m_entropy_coder_parallel = compute_ec_parallel(total_symbol_count);
            m_entropy_encoder.set_entropy_coder_parallel(m_entropy_coder_parallel);

            // Encode in reverse order (3,2,1,0) to match Python
            for (int i = 3; i >= 0; i--) {
                m_entropy_encoder.encode_y(m_y_to_encode[i], m_y_to_encode_size[i]);
            }

            auto t = tensor_to_vector_1d<int8_t, true>(m_z_hat_io);
            m_entropy_encoder.encode_z(t, m_qp * g_ch_z, g_ch_z);
            m_entropy_encoder.flush();
        } else if (m_pending_work == DMCWorkType::DecodeZ) {
            m_entropy_decoder.set_entropy_coder_parallel(m_entropy_coder_parallel);
            m_entropy_decoder.set_stream(m_bit_stream, m_bit_stream_size);
            const int z_num = g_ch_z * m_z_height * m_z_width;
            m_entropy_decoder.decode_z(z_num, m_qp * g_ch_z, g_ch_z);

            auto z_cpu = m_entropy_decoder.get_decoded_tensor_cpp();
            CUDA_CHECK(cudaMemcpyAsync(m_z_hat_io.data_ptr<int8_t>(), z_cpu->data(),
                                       z_num * sizeof(int8_t), cudaMemcpyHostToDevice,
                                       at::cuda::getCurrentCUDAStream()));
            CUDA_CHECK(cudaEventRecord(m_event_z_ready, at::cuda::getCurrentCUDAStream()));
        } else if (m_pending_work == DMCWorkType::DecodeY) {
            CUDA_CHECK(cudaStreamWaitEvent(at::cuda::getCurrentCUDAStream(), m_event_y));
            auto out = conditional_index_part2_cuda(m_y_indexes, m_y_indexes_size, m_size_ptr);
            const int y_num = *m_size_ptr;
            CUDA_CHECK(cudaMemcpyAsync(m_y_to_decode->data(), out.data_ptr<uint8_t>(),
                                       y_num * sizeof(uint8_t), cudaMemcpyDeviceToHost,
                                       at::cuda::getCurrentCUDAStream()));

            m_entropy_decoder.decode_y(m_y_to_decode, y_num);

            auto y_cpu = m_entropy_decoder.get_decoded_tensor_cpp();
            CUDA_CHECK(cudaMemcpyAsync(m_y_decoded.data_ptr<int8_t>(), y_cpu->data(),
                                       y_num * sizeof(int8_t), cudaMemcpyHostToDevice,
                                       at::cuda::getCurrentCUDAStream()));
            CUDA_CHECK(cudaEventRecord(m_event_y_ready, at::cuda::getCurrentCUDAStream()));
        } else {
            assert(false);
        }

        {
            std::lock_guard<std::mutex> lk_result(m_mutex_result);
            m_result_ready = true;
        }
        m_cv_result.notify_one();
    }
}

void DMCHTLProxy::add_ref_feature_from_frame(const at::Tensor& frame, const bool apply_adaptor)
{
    pre_allocate_tensors(frame);
    // must NOT use cuda graph becase frame.data_ptr() is non-const
    m_feature_i = pixel_unshuffle_cuda(frame, 8, m_feature_i);
    if (apply_adaptor) {
        m_memory = m_feature_adaptor_i.forward(m_feature_i);
        m_ctx = m_feature_extractor.forward(m_memory);
    }
    m_memory_has_value = apply_adaptor;
}

py::tuple DMCHTLProxy::compress(const at::Tensor& x, const int qp, const bool reset_feature_memory,
                                const int padding_b, const int padding_r)
{
    m_result_ready = false;
    m_qp = qp;

    auto stream = at::cuda::getCurrentCUDAStream();

    // must NOT use cuda graph becase x.data_ptr() is non-const
    m_enc_unshuffle_out = pad_and_unshuffle_8_cuda(x, padding_b, padding_r, m_enc_unshuffle_out);

    // ----------- start of lambda_enc_0 -----------
    auto lambda_enc_0 = [&](const int qp) {
        // need to output [m_z_hat_io, m_y_symbol[0..3], m_y_size[0..3]] for cpu thread
        auto y = m_encoder.forward(m_cat_encoder, m_q_encoder[qp]);
        auto y_pad = pad_for_y(y, m_y_pad);
        auto z = m_hyper_encoder.forward(y_pad);
        std::tie(m_z_hat, m_z_hat_io) = round_z_cuda(z, m_z_hat, m_z_hat_io);
        m_temporal_input = multiply_with_broadcast_cuda(m_memory, m_q_feature[qp], m_temporal_input);
        m_temporal_params = m_temporal_prior_encoder.forward(m_temporal_input);
        m_hyper_params_pad4 = m_hyper_decoder.forward(m_z_hat);
        m_hyper_params = crop_hyper_params(m_hyper_params_pad4, m_temporal_params.size(2),
                                           m_temporal_params.size(3), m_hyper_params);
        m_common_params = m_y_prior_fusion.forward(m_cat_prior_fusion);

        auto [q_dec, scales, means] = chunk_tensors_3(m_common_params);
        y = divide_with_clamp_min_inplace_cuda(y, q_dec);

        const int sm = get_gpu_sm();
        m_reduced_params = conv1x1_bias(sm, m_common_params, m_y_spatial_prior_reduction.weight,
                                        m_y_spatial_prior_reduction.bias, m_reduced_params);

        // Step 0: process with mask_0
        std::tie(m_y_q, m_y_hat, m_s_hat) = process_with_mask_cuda(
            y, scales, means, m_mask_0, m_skip_threshold, 0, m_y_q, m_y_hat, m_s_hat);
        m_y_q_w = single_part_for_writing_4x_cuda(m_y_q, m_y_q_w);
        m_s_w = single_part_for_writing_4x_cuda(m_s_hat, m_s_w);
        std::tie(m_y_symbol_prev, m_skip_cond) = build_index_enc_cuda(
            m_y_q_w, m_s_w, m_skip_threshold, at::nullopt, 0, m_y_symbol_prev, m_skip_cond);
        std::tie(m_y_symbol[0], m_y_size[0], m_y_num) = conditional_index_part1_cuda(
            m_y_symbol_prev, m_skip_cond, m_y_symbol[0], m_y_size[0], m_y_num);

        // copy y_hat to y_hat_so_far (first 256 channels of m_cat_spatial_prior_adaptor)
        m_y_hat_so_far.copy_(m_y_hat);

        // Step 1: adaptor_1 → spatial_prior → chunk2 → new scales, means
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_1.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        std::tie(m_y_q, m_y_hat, m_s_hat) = process_with_mask_cuda(
            y, scales, means, m_mask_1, m_skip_threshold, 0, m_y_q, m_y_hat, m_s_hat);
        m_y_q_w = single_part_for_writing_4x_cuda(m_y_q, m_y_q_w);
        m_s_w = single_part_for_writing_4x_cuda(m_s_hat, m_s_w);
        std::tie(m_y_symbol_prev, m_skip_cond) = build_index_enc_cuda(
            m_y_q_w, m_s_w, m_skip_threshold, at::nullopt, 0, m_y_symbol_prev, m_skip_cond);
        std::tie(m_y_symbol[1], m_y_size[1], m_y_num) = conditional_index_part1_cuda(
            m_y_symbol_prev, m_skip_cond, m_y_symbol[1], m_y_size[1], m_y_num);

        // y_hat_so_far += y_hat_1
        m_y_hat_so_far.add_(m_y_hat);

        // Step 2: adaptor_2 → spatial_prior → chunk2 → new scales, means
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_2.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        std::tie(m_y_q, m_y_hat, m_s_hat) = process_with_mask_cuda(
            y, scales, means, m_mask_2, m_skip_threshold, 0, m_y_q, m_y_hat, m_s_hat);
        m_y_q_w = single_part_for_writing_4x_cuda(m_y_q, m_y_q_w);
        m_s_w = single_part_for_writing_4x_cuda(m_s_hat, m_s_w);
        std::tie(m_y_symbol_prev, m_skip_cond) = build_index_enc_cuda(
            m_y_q_w, m_s_w, m_skip_threshold, at::nullopt, 0, m_y_symbol_prev, m_skip_cond);
        std::tie(m_y_symbol[2], m_y_size[2], m_y_num) = conditional_index_part1_cuda(
            m_y_symbol_prev, m_skip_cond, m_y_symbol[2], m_y_size[2], m_y_num);

        // y_hat_so_far += y_hat_2
        m_y_hat_so_far.add_(m_y_hat);

        // Step 3: adaptor_3 → spatial_prior → chunk2 → final add_and_multiply
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_3.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        std::tie(m_y_q, m_y_hat, m_s_hat) = process_with_mask_cuda(
            y, scales, means, m_mask_3, m_skip_threshold, 0, m_y_q, m_y_hat, m_s_hat);
        m_y_q_w = single_part_for_writing_4x_cuda(m_y_q, m_y_q_w);
        m_s_w = single_part_for_writing_4x_cuda(m_s_hat, m_s_w);
        std::tie(m_y_symbol_prev, m_skip_cond) = build_index_enc_cuda(
            m_y_q_w, m_s_w, m_skip_threshold, at::nullopt, 0, m_y_symbol_prev, m_skip_cond);
        std::tie(m_y_symbol[3], m_y_size[3], m_y_num) = conditional_index_part1_cuda(
            m_y_symbol_prev, m_skip_cond, m_y_symbol[3], m_y_size[3], m_y_num);

        // y_hat_final = add_and_multiply_with_clamp_min_inplace(y_hat_3, y_hat_so_far, q_dec)
        // swap args so result is in m_y_hat (decoder's pre-allocated input)
        m_y_hat = add_and_multiply_with_clamp_min_inplace_cuda(m_y_hat, m_y_hat_so_far, q_dec);
    };

    run(lambda_enc_0, m_gexec_enc_0, m_qp);
    // ------------ end of lambda_enc_0 ------------

    CUDA_CHECK(cudaEventRecord(m_event_y, stream));
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::Encode;
    }
    m_cv_pending.notify_one();

    // ----------- start of lambda_enc_1 -----------
    auto lambda_enc_1 = [&](const int qp, const bool reset_feature_memory) {
        m_feature_p = m_decoder.forward(m_y_hat, m_cat_decoder, m_q_decoder[qp]);
        add_ref_feature_and_apply_adaptor(m_feature_p, reset_feature_memory);
    };

    run(lambda_enc_1, m_gexec_enc_1, m_qp, reset_feature_memory);
    // ------------ end of lambda_enc_1 ------------

    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    if (m_result_ready) {
        return py::make_tuple(m_entropy_encoder.get_encoded_stream(), m_entropy_coder_parallel);
    }
    return py::make_tuple(py::array_t<uint8_t>(0), 1);
}

std::vector<at::Tensor> DMCHTLProxy::decompress(const py::array_t<uint8_t>& bit_stream,
                                                const int qp, const int height, const int width,
                                                const int entropy_coder_parallel,
                                                const bool reset_feature_memory)
{
    m_result_ready = false;
    m_entropy_coder_parallel = entropy_coder_parallel;
    py::buffer_info bit_stream_buf = bit_stream.request();
    m_bit_stream = static_cast<uint8_t*>(bit_stream_buf.ptr);
    m_bit_stream_size = static_cast<int>(bit_stream.size());
    m_z_height = (height + 63) / 64;
    m_z_width = (width + 63) / 64;
    m_qp = qp;

    auto stream = at::cuda::getCurrentCUDAStream();

    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::DecodeZ;
    }
    m_cv_pending.notify_one();

    // ----------- start of lambda_dec_0 -----------
    auto lambda_dec_0 = [&](const int qp, const bool memory_has_value) {
        apply_feature_adaptor(memory_has_value);
        m_temporal_input = multiply_with_broadcast_cuda(m_memory, m_q_feature[qp], m_temporal_input);
        m_temporal_params = m_temporal_prior_encoder.forward(m_temporal_input);
    };

    run(lambda_dec_0, m_gexec_dec_0, m_qp, m_memory_has_value);
    // ------------ end of lambda_dec_0 ------------

    {
        std::unique_lock<std::mutex> lk(m_mutex_result);
        m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
        if (!m_result_ready) {
            return std::vector<at::Tensor>();
        }
        m_result_ready = false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_event_z_ready));

    at::Tensor q_dec;
    at::Tensor scales;
    at::Tensor means;
    // ----------- start of lambda_dec_1 -----------
    auto lambda_dec_1 = [&]() {
        m_z_hat = int8_to_dtype_cuda(m_z_hat_io, m_dtype, g_ch_z, m_z_height, m_z_width, m_z_hat);
        m_hyper_params_pad4 = m_hyper_decoder.forward(m_z_hat);
        m_hyper_params = crop_hyper_params(m_hyper_params_pad4, m_temporal_params.size(2),
                                           m_temporal_params.size(3), m_hyper_params);
        m_common_params = m_y_prior_fusion.forward(m_cat_prior_fusion);

        const int sm = get_gpu_sm();
        m_reduced_params = conv1x1_bias(sm, m_common_params, m_y_spatial_prior_reduction.weight,
                                        m_y_spatial_prior_reduction.bias, m_reduced_params);

        m_ctx = m_feature_extractor.forward(m_memory);

        std::tie(q_dec, scales, means) = chunk_tensors_3(m_common_params);
        // Step 0: build index
        m_y_indexes =
            decode_y_step_build_index(scales, m_mask_0, m_y_indexes_skip_cond, m_y_indexes_num);
    };

    run(lambda_dec_1, m_gexec_dec_1);
    // ------------ end of lambda_dec_1 ------------

    // Step 0: entropy (worker thread)
    CUDA_CHECK(cudaEventRecord(m_event_y, stream));
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::DecodeY;
    }
    m_cv_pending.notify_one();
    {
        std::unique_lock<std::mutex> lk(m_mutex_result);
        m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
        m_result_ready = false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_event_y_ready));

    // ----------- start of lambda_dec_2 -----------
    auto lambda_dec_2 = [&]() {
        // Step 0: recover + restore
        m_y_q_r = conditional_recover_with_type_conversion_cuda(m_y_decoded, m_y_indexes_skip_cond,
                                                                m_y_indexes_num, m_dtype, m_y_q_r);
        m_y_hat_so_far = restore_y_4x_cuda(m_y_q_r, means, m_mask_0, m_y_hat_so_far);
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_1.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        // Step 1: build index
        m_y_indexes =
            decode_y_step_build_index(scales, m_mask_1, m_y_indexes_skip_cond, m_y_indexes_num);
    };
    run(lambda_dec_2, m_gexec_dec_2);
    // ------------ end of lambda_dec_2 ------------

    // Step 1: entropy (worker thread)
    CUDA_CHECK(cudaEventRecord(m_event_y, stream));
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::DecodeY;
    }
    m_cv_pending.notify_one();
    {
        std::unique_lock<std::mutex> lk(m_mutex_result);
        m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
        m_result_ready = false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_event_y_ready));

    // ----------- start of lambda_dec_3 -----------
    auto lambda_dec_3 = [&]() {
        // Step 1: recover + restore
        m_y_q_r = conditional_recover_with_type_conversion_cuda(m_y_decoded, m_y_indexes_skip_cond,
                                                                m_y_indexes_num, m_dtype, m_y_q_r);
        restore_y_4x_and_add_inplace_cuda(m_y_q_r, means, m_mask_1, m_y_hat_so_far);
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_2.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        // Step 2: build index
        m_y_indexes =
            decode_y_step_build_index(scales, m_mask_2, m_y_indexes_skip_cond, m_y_indexes_num);
    };
    run(lambda_dec_3, m_gexec_dec_3);
    // ------------ end of lambda_dec_3 ------------

    // Step 2: entropy (worker thread)
    CUDA_CHECK(cudaEventRecord(m_event_y, stream));
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::DecodeY;
    }
    m_cv_pending.notify_one();
    {
        std::unique_lock<std::mutex> lk(m_mutex_result);
        m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
        m_result_ready = false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_event_y_ready));

    // ----------- start of lambda_dec_4 -----------
    auto lambda_dec_4 = [&]() {
        // Step 2: recover + restore
        m_y_q_r = conditional_recover_with_type_conversion_cuda(m_y_decoded, m_y_indexes_skip_cond,
                                                                m_y_indexes_num, m_dtype, m_y_q_r);
        restore_y_4x_and_add_inplace_cuda(m_y_q_r, means, m_mask_2, m_y_hat_so_far);
        m_spatial_prior_out = m_y_spatial_prior.forward(
            m_y_spatial_prior_adaptor_3.forward(m_cat_spatial_prior_adaptor));
        std::tie(scales, means) = chunk_tensors_2(m_spatial_prior_out);
        // Step 3: build index
        m_y_indexes =
            decode_y_step_build_index(scales, m_mask_3, m_y_indexes_skip_cond, m_y_indexes_num);
    };
    run(lambda_dec_4, m_gexec_dec_4);
    // ------------ end of lambda_dec_4 ------------

    // Step 3: entropy (worker thread)
    CUDA_CHECK(cudaEventRecord(m_event_y, stream));
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending = true;
        m_pending_work = DMCWorkType::DecodeY;
    }
    m_cv_pending.notify_one();
    {
        std::unique_lock<std::mutex> lk(m_mutex_result);
        m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
        m_result_ready = false;
    }
    CUDA_CHECK(cudaStreamWaitEvent(stream, m_event_y_ready));

    // ----------- start of lambda_dec_5 -----------
    auto lambda_dec_5 = [&](const int qp) {
        // Step 3: recover + restore (QP-dependent due to q_dec)
        m_y_q_r = conditional_recover_with_type_conversion_cuda(m_y_decoded, m_y_indexes_skip_cond,
                                                                m_y_indexes_num, m_dtype, m_y_q_r);
        m_y_hat = restore_y_4x_cuda(m_y_q_r, means, m_mask_3, m_y_hat);
        // swap args so result is in m_y_hat (decoder's pre-allocated input)
        m_y_hat = add_and_multiply_with_clamp_min_inplace_cuda(m_y_hat, m_y_hat_so_far, q_dec);
        // decoder + recon_head
        m_feature_p = m_decoder.forward(m_y_hat, m_cat_decoder, m_q_decoder[qp]);
        std::tie(m_x_hat, m_feature_i) =
            m_recon_head.forward(m_feature_p);  // m_feature_i is also updated for reset purpose;
    };

    run(lambda_dec_5, m_gexec_dec_5, m_qp);
    // ------------ end of lambda_dec_5 ------------

    m_memory_has_value = !reset_feature_memory;  // host operation. must be outside of cuda graph

    return m_x_hat;
}

void DMCHTLProxy::set_param(const std::map<std::string, at::Tensor>& state_dict,
                            const float skip_threshold)
{
    m_device = state_dict.at("q_decoder").device();
    m_dtype = state_dict.at("q_decoder").scalar_type();
    auto q_encoder = state_dict.at("q_encoder");
    auto q_decoder = state_dict.at("q_decoder");
    auto q_feature = state_dict.at("q_feature");
    for (int i = 0; i < g_qp_num; i++) {
        m_q_encoder[i] = q_encoder.index({ at::indexing::Slice(i, i + 1), at::indexing::Slice(),
                                           at::indexing::None, at::indexing::None });
        m_q_decoder[i] = q_decoder.index({ at::indexing::Slice(i, i + 1), at::indexing::Slice(),
                                           at::indexing::None, at::indexing::None });
        m_q_feature[i] = q_feature.index({ at::indexing::Slice(i, i + 1), at::indexing::Slice(),
                                           at::indexing::None, at::indexing::None });
    }
    m_encoder.set_param(get_submodule_state_dict(state_dict, "encoder."));
    m_hyper_encoder.set_param(get_submodule_state_dict(state_dict, "hyper_encoder."));
    m_temporal_prior_encoder.set_param(
        get_submodule_state_dict(state_dict, "temporal_prior_encoder."));
    m_hyper_decoder.set_param(get_submodule_state_dict(state_dict, "hyper_decoder."));
    m_y_prior_fusion.set_param(get_submodule_state_dict(state_dict, "y_prior_fusion."));
    m_y_spatial_prior_reduction.set_param(state_dict, "y_spatial_prior_reduction.");
    m_y_spatial_prior_adaptor_1.set_param(
        get_submodule_state_dict(state_dict, "y_spatial_prior_adaptor_1."));
    m_y_spatial_prior_adaptor_2.set_param(
        get_submodule_state_dict(state_dict, "y_spatial_prior_adaptor_2."));
    m_y_spatial_prior_adaptor_3.set_param(
        get_submodule_state_dict(state_dict, "y_spatial_prior_adaptor_3."));
    m_y_spatial_prior.set_param(get_submodule_state_dict(state_dict, "y_spatial_prior."));
    m_decoder.set_param(get_submodule_state_dict(state_dict, "decoder."));
    m_feature_adaptor_i.set_param(get_submodule_state_dict(state_dict, "feature_adaptor_i."));
    m_feature_adaptor_m.set_param(get_submodule_state_dict(state_dict, "feature_adaptor_m."));
    m_feature_extractor.set_param(get_submodule_state_dict(state_dict, "feature_extractor."));
    m_recon_head.set_param(get_submodule_state_dict(state_dict, "recon_head."));

    m_skip_threshold = skip_threshold;
    m_entropy_encoder.set_cdf(
        tensor_to_vector_1d<int>(state_dict.at("bit_estimator_z.quantized_cdf")),
        tensor_to_vector_1d<int>(state_dict.at("bit_estimator_z.cdf_length")), 0);
    m_entropy_encoder.set_cdf(
        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.quantized_cdf")),
        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.cdf_length")), 1);

    m_entropy_decoder.set_cdf(
        tensor_to_vector_1d<int>(state_dict.at("bit_estimator_z.quantized_cdf")),
        tensor_to_vector_1d<int>(state_dict.at("bit_estimator_z.cdf_length")), 0);
    m_entropy_decoder.set_cdf(
        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.quantized_cdf")),
        tensor_to_vector_1d<int>(state_dict.at("gaussian_encoder.cdf_length")), 1);
}

void DMCHTLProxy::add_ref_feature_and_apply_adaptor(const at::Tensor& feature,
                                                    const bool reset_feature_memory)
{
    if (reset_feature_memory) {
        auto out = m_recon_head.forward_reset(feature);
        m_memory = m_feature_adaptor_i.forward(out);
    } else {
        m_memory = m_feature_adaptor_m.forward(m_cat_feature_adaptor_m);
    }
    m_ctx = m_feature_extractor.forward(m_memory);
}

void DMCHTLProxy::apply_feature_adaptor(const bool memory_has_value)
{
    if (memory_has_value) {
        m_memory = m_feature_adaptor_m.forward(m_cat_feature_adaptor_m);
    } else {
        m_memory = m_feature_adaptor_i.forward(m_feature_i);
    }
}

void DMCHTLProxy::clear_cuda_graph()
{
    for (int i = 0; i < g_qp_num; ++i) {
        destroy_cuda_graph_exec(m_gexec_enc_0[i]);
        for (int j = 0; j < 2; ++j) {
            destroy_cuda_graph_exec(m_gexec_enc_1[i][j]);
        }
        for (int j = 0; j < 2; ++j) {
            destroy_cuda_graph_exec(m_gexec_dec_0[i][j]);
        }
        destroy_cuda_graph_exec(m_gexec_dec_5[i]);
    }
    destroy_cuda_graph_exec(m_gexec_dec_1);
    destroy_cuda_graph_exec(m_gexec_dec_2);
    destroy_cuda_graph_exec(m_gexec_dec_3);
    destroy_cuda_graph_exec(m_gexec_dec_4);
}

at::Tensor DMCHTLProxy::decode_y_step_build_index(const at::Tensor& scales, const at::Tensor& mask,
                                                  at::Tensor& skip_cond_out, at::Tensor& indexes_num_out)
{
    m_scales_r = single_part_for_reading_4x_cuda(scales, mask, m_scales_r);
    std::tie(m_y_indexes_prev, skip_cond_out) = build_index_dec_cuda(
        m_scales_r, m_skip_threshold, at::nullopt, 0, m_y_indexes_prev, skip_cond_out);
    std::tie(m_y_indexes, m_y_indexes_size, indexes_num_out) = conditional_index_part1_cuda(
        m_y_indexes_prev, skip_cond_out, m_y_indexes, m_y_indexes_size, indexes_num_out);
    return m_y_indexes;
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> DMCHTLProxy::get_mask_4x(const at::Tensor& y)
{
    const int C = y.size(1);
    const int H = y.size(2);
    const int W = y.size(3);
    auto m = at::ones({ 1, C / 4, H, W }, y.options().dtype(at::kBool));
    auto m0 = get_one_mask(torch::tensor({ { 1, 0 }, { 0, 0 } }, y.options().dtype(at::kBool)), H, W);
    auto m1 = get_one_mask(torch::tensor({ { 0, 1 }, { 0, 0 } }, y.options().dtype(at::kBool)), H, W);
    auto m2 = get_one_mask(torch::tensor({ { 0, 0 }, { 1, 0 } }, y.options().dtype(at::kBool)), H, W);
    auto m3 = get_one_mask(torch::tensor({ { 0, 0 }, { 0, 1 } }, y.options().dtype(at::kBool)), H, W);

    auto mask_0 =
        at::cat({ m * m0, m * m1, m * m2, m * m3 }, 1).contiguous(at::MemoryFormat::ChannelsLast);
    auto mask_1 =
        at::cat({ m * m3, m * m2, m * m1, m * m0 }, 1).contiguous(at::MemoryFormat::ChannelsLast);
    auto mask_2 =
        at::cat({ m * m2, m * m3, m * m0, m * m1 }, 1).contiguous(at::MemoryFormat::ChannelsLast);
    auto mask_3 =
        at::cat({ m * m1, m * m0, m * m3, m * m2 }, 1).contiguous(at::MemoryFormat::ChannelsLast);

    return { mask_0, mask_1, mask_2, mask_3 };
}

void DMCHTLProxy::pre_allocate_tensors(const at::Tensor& frame)
{
    int H = frame.size(2);
    int W = frame.size(3);
    H = (H + 16 - 1) / 16 * 16;
    W = (W + 16 - 1) / 16 * 16;
    const int H8 = H / 8;
    const int W8 = W / 8;
    if (m_H8 == H8 && m_W8 == W8) {
        return;
    }

    m_H8 = H8;
    m_W8 = W8;
    clear_cuda_graph();
    g_tensor_pool.clear(H8, W8);

    auto options = frame.options().memory_format(at::MemoryFormat::ChannelsLast);
    const int H16 = H8 / 2;
    const int W16 = W8 / 2;
    auto [padR, padB] = get_padding_size(H16, W16, 4);
    const int H16_pad4 = H16 + padB;
    const int W16_pad4 = W16 + padR;
    const int H64 = H16_pad4 / 4;
    const int W64 = W16_pad4 / 4;

    const std::vector<int64_t> feature_i_shape = { 1, g_ch_src_d_intra, H8, W8 };
    const std::vector<int64_t> memory_shape = { 1, g_ch_m, H8, W8 };
    const std::vector<int64_t> y_shape = { 1, g_ch_y, H16, W16 };
    const std::vector<int64_t> y_pad_shape = { 1, g_ch_y, H16_pad4, W16_pad4 };
    const std::vector<int64_t> z_shape = { 1, g_ch_z, H64, W64 };

    auto y = at::empty(y_shape, options);

    // pre allocate masks
    std::tie(m_mask_0, m_mask_1, m_mask_2, m_mask_3) = get_mask_4x(y);

    // pre allocate entropy coder buffers
    const int y_quarter_numel = g_ch_y / 4 * H16 * W16;
    for (int i = 0; i < 4; i++) {
        m_y_to_encode[i] = std::make_shared<std::vector<int16_t>>(y_quarter_numel);
    }

    //
    // pre allocate concatenated tensors
    //

    // 1. deal with m_cat_encoder, m_cat_decoder, m_enc_unshuffle_out, m_decoder_up_out, and m_ctx
    assert(g_ch_src_d > g_ch_d);
    at::Tensor tensor_placeholder;
    m_cat_encoder = at::empty({ 1, g_ch_src_d + g_ch_d, H8, W8 }, options);
    std::tie(tensor_placeholder, m_cat_decoder) =
        split_tensors_2(m_cat_encoder, g_ch_src_d - g_ch_d, g_ch_d * 2);
    std::tie(tensor_placeholder, m_decoder_up_out, m_ctx) =
        split_tensors_3(m_cat_encoder, g_ch_src_d - g_ch_d, g_ch_d, g_ch_d);
    std::tie(m_enc_unshuffle_out, m_ctx) = split_tensors_2(m_cat_encoder, g_ch_src_d, g_ch_d);

    // 2. deal with m_cat_feature_adaptor_m, m_memory, and m_feature_p
    m_cat_feature_adaptor_m = at::empty({ 1, g_ch_m + g_ch_d, H8, W8 }, options);
    std::tie(m_memory, m_feature_p) = split_tensors_2(m_cat_feature_adaptor_m, g_ch_m, g_ch_d);

    // 3. deal with m_cat_prior_fusion, m_hyper_params, and m_temporal_params
    m_cat_prior_fusion = at::empty({ 1, g_ch_y * 3, H16, W16 }, options);
    std::tie(m_hyper_params, m_temporal_params) =
        split_tensors_2(m_cat_prior_fusion, g_ch_y, g_ch_y * 2);
    m_hyper_params_pad4 = (padR == 0 && padB == 0) ? m_hyper_params : at::empty(y_pad_shape, options);

    // 4. deal with m_cat_spatial_prior_adaptor, m_y_hat_so_far, and m_reduced_params
    m_cat_spatial_prior_adaptor = at::empty({ 1, g_ch_y * 2, H16, W16 }, options);
    std::tie(m_y_hat_so_far, m_reduced_params) =
        split_tensors_2(m_cat_spatial_prior_adaptor, g_ch_y, g_ch_y);

    // pre allocate other tensors
    m_feature_i = at::empty(feature_i_shape, options);
    m_temporal_input = at::empty(memory_shape, options);

    m_z_hat = at::empty(z_shape, options.dtype(at::kHalf));
    m_z_hat_io = at::empty(z_shape, options.dtype(at::kChar));

    m_spatial_prior_out = at::empty({ 1, g_ch_y * 2, H16, W16 }, options);
    m_y_q = at::empty_like(y);
    m_y_hat = at::empty_like(y);
    m_s_hat = at::empty_like(y);

    m_y_q_w = at::empty({ 1, g_ch_y / 4, H16, W16 }, options);
    m_s_w = at::empty({ 1, g_ch_y / 4, H16, W16 }, options);
    m_y_symbol_prev = at::empty({ y_quarter_numel }, y.options().dtype(at::kShort));
    m_skip_cond = at::empty({ y_quarter_numel }, y.options().dtype(at::kBool));
    {
        const int block_num_quarter =
            (y_quarter_numel + COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM - 1)
            / (COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM);
        m_y_num =
            at::empty({ block_num_quarter, COND_KERNEL_THREAD_NUM1 }, y.options().dtype(at::kInt));
    }
    for (int i = 0; i < 4; i++) {
        m_y_symbol[i] = at::empty({ y_quarter_numel }, y.options().dtype(at::kShort));
        m_y_size[i] = at::empty({ 1 }, y.options().dtype(at::kInt));
    }

    // decompress-only buffers
    m_y_q_r = at::empty({ 1, g_ch_y / 4, H16, W16 }, options);
    m_scales_r = at::empty({ 1, g_ch_y / 4, H16, W16 }, options);
    m_y_decoded = at::empty({ y_quarter_numel }, y.options().dtype(at::kChar));
    m_y_to_decode = std::make_shared<std::vector<uint8_t>>(y_quarter_numel);
    m_y_indexes = at::empty({ y_quarter_numel }, y.options().dtype(at::kByte));
    m_y_indexes_prev = at::empty_like(m_y_indexes);
    m_y_indexes_skip_cond = at::empty({ y_quarter_numel }, y.options().dtype(at::kBool));
    {
        const int block_num_quarter =
            (y_quarter_numel + COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM - 1)
            / (COND_KERNEL_THREAD_NUM1 * COND_KERNEL_PER_THREAD_NUM);
        m_y_indexes_size = at::empty({ 1 }, y.options().dtype(at::kInt));
        m_y_indexes_num =
            at::empty({ block_num_quarter, COND_KERNEL_THREAD_NUM1 }, y.options().dtype(at::kInt));
    }

    // pre allocate modules
    m_y = m_encoder.pre_allocate_tensors(m_cat_encoder);
    m_y_pad = (padR == 0 && padB == 0) ? m_y : at::empty(y_pad_shape, options);
    auto enc_z = m_hyper_encoder.pre_allocate_tensors(m_y_pad);
    m_temporal_params =
        m_temporal_prior_encoder.pre_allocate_tensors(m_temporal_input, m_temporal_params);
    m_hyper_params_pad4 = m_hyper_decoder.pre_allocate_tensors(m_z_hat, m_hyper_params_pad4);
    m_common_params = m_y_prior_fusion.pre_allocate_tensors(m_cat_prior_fusion);
    auto adaptor_out =
        m_y_spatial_prior_adaptor_1.pre_allocate_tensors(m_cat_spatial_prior_adaptor, true);
    m_y_spatial_prior_adaptor_2.pre_allocate_tensors(m_cat_spatial_prior_adaptor, true, adaptor_out);
    m_y_spatial_prior_adaptor_3.pre_allocate_tensors(m_cat_spatial_prior_adaptor, true, adaptor_out);
    m_spatial_prior_out = m_y_spatial_prior.pre_allocate_tensors(adaptor_out, m_spatial_prior_out);
    m_feature_p = m_decoder.pre_allocate_tensors(m_y_hat, m_decoder_up_out, m_cat_decoder, m_feature_p);
    m_memory = m_feature_adaptor_i.pre_allocate_tensors(m_feature_i, m_memory);
    m_memory = m_feature_adaptor_m.pre_allocate_tensors(m_cat_feature_adaptor_m, m_memory);
    m_ctx = m_feature_extractor.pre_allocate_tensors(m_memory, m_ctx);
    m_feature_i = m_recon_head.pre_allocate_tensors(
        m_feature_p, m_feature_i);  // the recon is not used in cuda graph, only use feature for reset
}
