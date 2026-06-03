// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

#include "def_const.h"

at::Tensor add_and_multiply_broadcast_cuda(const at::Tensor& x0, const at::Tensor& x1,
                                           const at::Tensor& q, at::Tensor& out_buf);
at::Tensor add_and_multiply_with_clamp_min_inplace_cuda(at::Tensor& x0, const at::Tensor& x1,
                                                        const at::Tensor& q);
std::tuple<at::Tensor, at::Tensor>
build_index_dec_cuda(const at::Tensor& scales, const float skip_thres_float,
                     const at::optional<at::Tensor>& fx, const int32_t skip_thres_int,
                     at::Tensor& out_buf, at::Tensor& cond_out_buf);
std::tuple<at::Tensor, at::Tensor>
build_index_enc_cuda(const at::Tensor& symbols, const at::Tensor& scales,
                     const float skip_thres_float, const at::optional<at::Tensor>& fx,
                     const int32_t skip_thres_int, at::Tensor& out_buf, at::Tensor& cond_out_buf);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
conditional_index_part1_cuda(const at::Tensor& x, const at::Tensor& m, at::Tensor& out_buf,
                             at::Tensor& total_num_buf, at::Tensor& num_buf);
at::Tensor conditional_index_part2_cuda(const at::Tensor& x, const at::Tensor& s,
                                        int* size_ptr = nullptr);
at::Tensor conditional_recover_with_type_conversion_cuda(const at::Tensor& x, const at::Tensor& m,
                                                         const at::Tensor& num,
                                                         const at::ScalarType dtype,
                                                         at::Tensor& out_buf);
at::Tensor divide_with_clamp_min_inplace_cuda(at::Tensor& y, const at::Tensor& q);
at::Tensor int8_to_dtype_cuda(const at::Tensor& x, const at::ScalarType dtype, const int C,
                              const int H, const int W, at::Tensor& out_buf);
at::Tensor multiply_with_broadcast_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out_buf);
at::Tensor pad_and_unshuffle_8_cuda(const at::Tensor& x, const int padB, const int padR,
                                    at::Tensor& out_buf);
at::Tensor pixel_shuffle_2_cuda(const at::Tensor& x, at::Tensor& out_buf);
at::Tensor pixel_shuffle_8_cuda(const at::Tensor& x, bool clamp, at::Tensor& out_buf);
at::Tensor pixel_unshuffle_cuda(const at::Tensor& x, const int downscale_factor, at::Tensor& out_buf);
std::tuple<at::Tensor, at::Tensor, at::Tensor>
process_with_mask_cuda(const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means,
                       const at::Tensor& mask, const float force_zero_thres_float,
                       const int32_t force_zero_thres_int, at::Tensor& y_q_buf,
                       at::Tensor& y_hat_buf, at::Tensor& s_hat_buf);
std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_add_and_multiply_inplace_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    at::Tensor& y_q, at::Tensor& y_hat, const at::Tensor& quant, const float force_zero_thres_float,
    const int32_t force_zero_thres_int);
std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_add_inplace_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    at::Tensor& y_q, at::Tensor& y_hat, const float force_zero_thres_float,
    const int32_t force_zero_thres_int);
std::tuple<at::Tensor, at::Tensor> process_with_mask_no_scale_cuda(
    const at::Tensor& y, const at::Tensor& scales, const at::Tensor& means, const at::Tensor& mask,
    const float force_zero_thres_float, const int32_t force_zero_thres_int, at::Tensor& y_q_buf,
    at::Tensor& y_hat_buf);
at::Tensor replicate_pad_cuda(const at::Tensor& x, const int padB, const int padR, at::Tensor& out_buf);
at::Tensor restore_y_4x_add_and_multiply_broadcast_cuda(const at::Tensor& y, const at::Tensor& means,
                                                        const at::Tensor& mask,
                                                        const at::Tensor& y_hat_so_far,
                                                        const at::Tensor& q, at::Tensor& out_buf);
at::Tensor restore_y_4x_and_add_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                             const at::Tensor& mask, at::Tensor& y_hat);
at::Tensor restore_y_4x_cuda(const at::Tensor& y, const at::Tensor& means, const at::Tensor& mask,
                             at::Tensor& out_buf);
at::Tensor restore_y_and_add_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                          const at::Tensor& mask, at::Tensor& y_hat);
at::Tensor restore_y_and_add_multiply_inplace_cuda(const at::Tensor& y, const at::Tensor& means,
                                                   const at::Tensor& mask, at::Tensor& y_hat,
                                                   const at::Tensor& q);
at::Tensor restore_y_cuda(const at::Tensor& y, const at::Tensor& means, const at::Tensor& mask,
                          at::Tensor& out_buf);
std::tuple<at::Tensor, at::Tensor> round_z_cuda(const at::Tensor& z, at::Tensor& z_hat_buf,
                                                at::Tensor& z_int8_buf);
at::Tensor single_part_for_reading_4x_cuda(const at::Tensor& x, const at::Tensor& mask,
                                           at::Tensor& out_buf);
at::Tensor single_part_for_writing_4x_cuda(const at::Tensor& x, at::Tensor& out_buf);
at::Tensor slice_cuda(const at::Tensor& x, const int H, const int W, at::Tensor& out_buf);
