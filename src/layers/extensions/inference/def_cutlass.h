// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <torch/extension.h>

#include "def_const.h"

at::Tensor conv1x1_bias(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                        const at::Tensor& bias, at::Tensor& out_buf);
at::Tensor conv1x1_bias_shortcut(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                                 const at::Tensor& bias, const at::Tensor& shortcut,
                                 at::Tensor& out_buf);
at::Tensor conv1x1_bias_shortcut2(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                                  const at::Tensor& bias, const at::Tensor& shortcut1,
                                  const at::Tensor& shortcut2, at::Tensor& out_buf);
at::Tensor conv1x1_bias_shortcut_with_quant(const int sm, const at::Tensor& feature,
                                            const at::Tensor& weight, const at::Tensor& bias,
                                            const at::Tensor& shortcut, const at::Tensor& quant,
                                            at::Tensor& out_buf);
at::Tensor conv1x1_bias_with_quant(const int sm, const at::Tensor& feature,
                                   const at::Tensor& weight, const at::Tensor& bias,
                                   const at::Tensor& quant, at::Tensor& out_buf);
at::Tensor conv1x1_bias_wsilu(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                              const at::Tensor& bias, at::Tensor& out_buf);
at::Tensor conv1x1_bias_wsilu_chunk_add(const int sm, const at::Tensor& feature,
                                        const at::Tensor& weight, const at::Tensor& bias,
                                        const at::Tensor& weight0, const at::Tensor& weight1,
                                        const at::Tensor& weight2, const at::Tensor& weight3,
                                        const at::Tensor& bias0, const at::Tensor& bias1,
                                        const at::Tensor& bias2, const at::Tensor& bias3,
                                        at::Tensor& out_buf);
at::Tensor conv2d(const at::Tensor& feature, const at::Tensor& weight, const at::Tensor& bias,
                  const int stride, at::Tensor& out_buf);
at::Tensor conv_bias(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                     const at::Tensor& bias, const int stride, at::Tensor& out_buf);
at::Tensor d3x3(const int sm, const at::Tensor& feature, const at::Tensor& weight, at::Tensor& out_buf);
at::Tensor transposed_conv(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                           const int stride, at::Tensor& out_buf);
