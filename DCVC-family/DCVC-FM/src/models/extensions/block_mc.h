// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <torch/extension.h>

void block_mc_forward_cuda(torch::Tensor &out, const torch::Tensor &im,
                           const torch::Tensor &flow, const int B, const int C,
                           const int H, const int W);
