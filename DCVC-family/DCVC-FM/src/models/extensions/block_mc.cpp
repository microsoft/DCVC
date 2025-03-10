// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "block_mc.h"
#include <torch/extension.h>

void block_mc_forward(torch::Tensor &out, const torch::Tensor &im,
                      const torch::Tensor &flow, const int B, const int C,
                      const int H, const int W) {
  block_mc_forward_cuda(out, im, flow, B, C, H, W);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("block_mc_forward", &block_mc_forward, "Motion Compensation forward");
}
