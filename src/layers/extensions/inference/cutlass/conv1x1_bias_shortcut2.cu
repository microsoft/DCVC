// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "conv1x1_kernel.h"
#include "../def_cutlass.h"

const std::string HINT_MAP_BIAS_SHORTCUT2_NAME = "HINT_MAP_BIAS_SHORTCUT2";
std::unordered_map<uint64_t, int> HINT_MAP_BIAS_SHORTCUT2 = {
#if CURRENT_DEVICE_SM == 80
    { conv_key(80, 7, 4, 128, 128, 0, 0), 19 },     { conv_key(80, 13, 8, 128, 128, 0, 0), 8 },
    { conv_key(80, 14, 8, 128, 128, 0, 0), 8 },     { conv_key(80, 20, 12, 128, 128, 0, 0), 7 },
    { conv_key(80, 26, 16, 128, 128, 0, 0), 7 },    { conv_key(80, 28, 16, 128, 128, 0, 0), 29 },
    { conv_key(80, 30, 17, 128, 128, 0, 0), 8 },    { conv_key(80, 40, 24, 128, 128, 0, 0), 8 },
    { conv_key(80, 40, 24, 256, 256, 0, 0), 8 },    { conv_key(80, 52, 30, 384, 384, 0, 0), 29 },
    { conv_key(80, 52, 32, 128, 128, 0, 0), 19 },   { conv_key(80, 60, 34, 128, 128, 0, 0), 8 },
    { conv_key(80, 60, 34, 256, 256, 0, 0), 29 },   { conv_key(80, 80, 45, 512, 512, 0, 0), 29 },
    { conv_key(80, 80, 48, 128, 128, 0, 0), 29 },   { conv_key(80, 80, 48, 256, 256, 0, 0), 14 },
    { conv_key(80, 104, 60, 384, 384, 0, 0), 14 },  { conv_key(80, 120, 68, 128, 128, 0, 0), 7 },
    { conv_key(80, 120, 68, 256, 256, 0, 0), 14 },  { conv_key(80, 120, 68, 512, 512, 0, 0), 14 },
    { conv_key(80, 160, 90, 384, 384, 0, 0), 14 },  { conv_key(80, 240, 135, 512, 512, 0, 0), 14 },
    { conv_key(80, 240, 136, 128, 128, 0, 0), 2 },  { conv_key(80, 240, 136, 256, 256, 0, 0), 14 },
    { conv_key(80, 240, 136, 384, 384, 0, 0), 14 }, { conv_key(80, 480, 270, 384, 384, 0, 0), 14 },
#elif CURRENT_DEVICE_SM == 89
    { conv_key(89, 7, 4, 128, 128, 0, 0), 19 },     { conv_key(89, 13, 8, 128, 128, 0, 0), 8 },
    { conv_key(89, 14, 8, 128, 128, 0, 0), 7 },     { conv_key(89, 20, 12, 128, 128, 0, 0), 7 },
    { conv_key(89, 26, 16, 128, 128, 0, 0), 19 },   { conv_key(89, 28, 16, 128, 128, 0, 0), 19 },
    { conv_key(89, 30, 17, 128, 128, 0, 0), 7 },    { conv_key(89, 40, 24, 128, 128, 0, 0), 19 },
    { conv_key(89, 40, 24, 256, 256, 0, 0), 19 },   { conv_key(89, 52, 30, 384, 384, 0, 0), 29 },
    { conv_key(89, 52, 32, 128, 128, 0, 0), 19 },   { conv_key(89, 60, 34, 128, 128, 0, 0), 8 },
    { conv_key(89, 60, 34, 256, 256, 0, 0), 8 },    { conv_key(89, 80, 45, 512, 512, 0, 0), 16 },
    { conv_key(89, 80, 48, 128, 128, 0, 0), 19 },   { conv_key(89, 80, 48, 256, 256, 0, 0), 7 },
    { conv_key(89, 104, 60, 384, 384, 0, 0), 14 },  { conv_key(89, 120, 68, 128, 128, 0, 0), 7 },
    { conv_key(89, 120, 68, 256, 256, 0, 0), 14 },  { conv_key(89, 120, 68, 512, 512, 0, 0), 14 },
    { conv_key(89, 160, 90, 384, 384, 0, 0), 14 },  { conv_key(89, 240, 135, 512, 512, 0, 0), 14 },
    { conv_key(89, 240, 136, 128, 128, 0, 0), 2 },  { conv_key(89, 240, 136, 256, 256, 0, 0), 14 },
    { conv_key(89, 240, 136, 384, 384, 0, 0), 14 }, { conv_key(89, 480, 270, 384, 384, 0, 0), 14 },
#elif CURRENT_DEVICE_SM == 90
    { conv_key(90, 7, 4, 128, 128, 0, 0), 8 },      { conv_key(90, 13, 8, 128, 128, 0, 0), 8 },
    { conv_key(90, 14, 8, 128, 128, 0, 0), 8 },     { conv_key(90, 20, 12, 128, 128, 0, 0), 8 },
    { conv_key(90, 26, 16, 128, 128, 0, 0), 8 },    { conv_key(90, 28, 16, 128, 128, 0, 0), 8 },
    { conv_key(90, 30, 17, 128, 128, 0, 0), 8 },    { conv_key(90, 40, 24, 128, 128, 0, 0), 8 },
    { conv_key(90, 40, 24, 256, 256, 0, 0), 8 },    { conv_key(90, 52, 30, 384, 384, 0, 0), 29 },
    { conv_key(90, 52, 32, 128, 128, 0, 0), 8 },    { conv_key(90, 60, 34, 128, 128, 0, 0), 8 },
    { conv_key(90, 60, 34, 256, 256, 0, 0), 8 },    { conv_key(90, 80, 45, 512, 512, 0, 0), 30 },
    { conv_key(90, 80, 48, 128, 128, 0, 0), 8 },    { conv_key(90, 80, 48, 256, 256, 0, 0), 29 },
    { conv_key(90, 104, 60, 384, 384, 0, 0), 34 },  { conv_key(90, 120, 68, 128, 128, 0, 0), 29 },
    { conv_key(90, 120, 68, 256, 256, 0, 0), 30 },  { conv_key(90, 120, 68, 512, 512, 0, 0), 32 },
    { conv_key(90, 160, 90, 384, 384, 0, 0), 31 },  { conv_key(90, 240, 135, 512, 512, 0, 0), 40 },
    { conv_key(90, 240, 136, 128, 128, 0, 0), 11 }, { conv_key(90, 240, 136, 256, 256, 0, 0), 30 },
    { conv_key(90, 240, 136, 384, 384, 0, 0), 31 }, { conv_key(90, 480, 270, 384, 384, 0, 0), 35 },
#elif CURRENT_DEVICE_SM == 100
    { conv_key(100, 7, 4, 128, 128, 0, 0), 25 },
    { conv_key(100, 13, 8, 128, 128, 0, 0), 25 },
    { conv_key(100, 14, 8, 128, 128, 0, 0), 6 },
    { conv_key(100, 20, 12, 128, 128, 0, 0), 7 },
    { conv_key(100, 26, 16, 128, 128, 0, 0), 17 },
    { conv_key(100, 28, 16, 128, 128, 0, 0), 17 },
    { conv_key(100, 30, 17, 128, 128, 0, 0), 6 },
    { conv_key(100, 40, 24, 128, 128, 0, 0), 17 },
    { conv_key(100, 40, 24, 256, 256, 0, 0), 58 },
    { conv_key(100, 52, 30, 384, 384, 0, 0), 8 },
    { conv_key(100, 52, 32, 128, 128, 0, 0), 7 },
    { conv_key(100, 60, 34, 128, 128, 0, 0), 7 },
    { conv_key(100, 60, 34, 256, 256, 0, 0), 53 },
    { conv_key(100, 80, 45, 512, 512, 0, 0), 88 },
    { conv_key(100, 80, 48, 128, 128, 0, 0), 19 },
    { conv_key(100, 80, 48, 256, 256, 0, 0), 63 },
    { conv_key(100, 104, 60, 384, 384, 0, 0), 87 },
    { conv_key(100, 120, 68, 128, 128, 0, 0), 8 },
    { conv_key(100, 120, 68, 256, 256, 0, 0), 83 },
    { conv_key(100, 120, 68, 512, 512, 0, 0), 47 },
    { conv_key(100, 160, 90, 384, 384, 0, 0), 68 },
    { conv_key(100, 240, 135, 512, 512, 0, 0), 46 },
    { conv_key(100, 240, 136, 128, 128, 0, 0), 87 },
    { conv_key(100, 240, 136, 256, 256, 0, 0), 77 },
    { conv_key(100, 240, 136, 384, 384, 0, 0), 68 },
    { conv_key(100, 480, 270, 384, 384, 0, 0), 68 },
#endif
};

at::Tensor conv1x1_bias_shortcut2(const int sm, const at::Tensor& feature, const at::Tensor& weight,
                                  const at::Tensor& bias, const at::Tensor& shortcut1,
                                  const at::Tensor& shortcut2, at::Tensor& out_buf)
{
    if (sm < 75) {
        auto out = at::conv2d(feature, weight);
        return out + bias.reshape({ 1, -1, 1, 1 }) + shortcut1 + shortcut2;
    }
    auto launch_cutlass = [&](auto sm_v) {
        using sm_t = decltype(sm_v);
        return conv1x1_bias_generic_cutlass<sm_t, true, 2>(out_buf, feature, weight, bias,
                                                           shortcut1, shortcut2);
    };
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C2 = weight.size(0);
    const int C1 = weight.size(1);
    return launch_cutlass_helper(sm, launch_cutlass, HINT_MAP_BIAS_SHORTCUT2,
                                 HINT_MAP_BIAS_SHORTCUT2_NAME, C2, C1, H, W);
}
