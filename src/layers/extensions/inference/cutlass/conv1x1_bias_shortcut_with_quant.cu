// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "conv1x1_kernel.h"
#include "../def_cutlass.h"

const std::string HINT_MAP_BIAS_SHORTCUT_WITH_QUANT_NAME = "HINT_MAP_BIAS_SHORTCUT_WITH_QUANT";
std::unordered_map<uint64_t, int> HINT_MAP_BIAS_SHORTCUT_WITH_QUANT = {
#if CURRENT_DEVICE_SM == 80
    { conv_key(80, 52, 30, 256, 128, 0, 0), 8 },    { conv_key(80, 52, 30, 512, 256, 0, 0), 18 },
    { conv_key(80, 104, 60, 256, 128, 0, 0), 14 },  { conv_key(80, 104, 60, 512, 256, 0, 0), 14 },
    { conv_key(80, 160, 90, 256, 128, 0, 0), 14 },  { conv_key(80, 160, 90, 512, 256, 0, 0), 14 },
    { conv_key(80, 160, 90, 512, 512, 0, 0), 14 },  { conv_key(80, 240, 136, 256, 128, 0, 0), 14 },
    { conv_key(80, 240, 136, 512, 256, 0, 0), 14 }, { conv_key(80, 240, 136, 512, 512, 0, 0), 14 },
    { conv_key(80, 480, 270, 256, 128, 0, 0), 14 }, { conv_key(80, 480, 270, 512, 256, 0, 0), 14 },
    { conv_key(80, 480, 270, 512, 512, 0, 0), 14 },
#elif CURRENT_DEVICE_SM == 89
    { conv_key(89, 52, 30, 256, 128, 0, 0), 19 },  { conv_key(89, 52, 30, 512, 256, 0, 0), 18 },
    { conv_key(89, 104, 60, 256, 128, 0, 0), 5 },  { conv_key(89, 104, 60, 512, 256, 0, 0), 14 },
    { conv_key(89, 160, 90, 256, 128, 0, 0), 14 }, { conv_key(89, 160, 90, 512, 256, 0, 0), 14 },
    { conv_key(89, 160, 90, 512, 512, 0, 0), 14 }, { conv_key(89, 240, 136, 256, 128, 0, 0), 14 },
    { conv_key(89, 240, 136, 512, 256, 0, 0), 2 }, { conv_key(89, 240, 136, 512, 512, 0, 0), 2 },
    { conv_key(89, 480, 270, 256, 128, 0, 0), 7 }, { conv_key(89, 480, 270, 512, 256, 0, 0), 10 },
    { conv_key(89, 480, 270, 512, 512, 0, 0), 9 },
#elif CURRENT_DEVICE_SM == 90
    { conv_key(90, 52, 30, 256, 128, 0, 0), 8 },    { conv_key(90, 52, 30, 512, 256, 0, 0), 29 },
    { conv_key(90, 104, 60, 256, 128, 0, 0), 30 },  { conv_key(90, 104, 60, 512, 256, 0, 0), 32 },
    { conv_key(90, 160, 90, 256, 128, 0, 0), 34 },  { conv_key(90, 160, 90, 512, 256, 0, 0), 36 },
    { conv_key(90, 160, 90, 512, 512, 0, 0), 36 },  { conv_key(90, 240, 136, 256, 128, 0, 0), 33 },
    { conv_key(90, 240, 136, 512, 256, 0, 0), 30 }, { conv_key(90, 240, 136, 512, 512, 0, 0), 46 },
    { conv_key(90, 480, 270, 256, 128, 0, 0), 30 }, { conv_key(90, 480, 270, 512, 256, 0, 0), 37 },
    { conv_key(90, 480, 270, 512, 512, 0, 0), 46 },
#elif CURRENT_DEVICE_SM == 100
    { conv_key(100, 52, 30, 256, 128, 0, 0), 8 },
    { conv_key(100, 52, 30, 512, 256, 0, 0), 18 },
    { conv_key(100, 104, 60, 256, 128, 0, 0), 29 },
    { conv_key(100, 104, 60, 512, 256, 0, 0), 82 },
    { conv_key(100, 160, 90, 256, 128, 0, 0), 63 },
    { conv_key(100, 160, 90, 512, 256, 0, 0), 75 },
    { conv_key(100, 160, 90, 512, 512, 0, 0), 75 },
    { conv_key(100, 240, 136, 256, 128, 0, 0), 63 },
    { conv_key(100, 240, 136, 512, 256, 0, 0), 75 },
    { conv_key(100, 240, 136, 512, 512, 0, 0), 76 },
    { conv_key(100, 480, 270, 256, 128, 0, 0), 92 },
    { conv_key(100, 480, 270, 512, 256, 0, 0), 92 },
    { conv_key(100, 480, 270, 512, 512, 0, 0), 75 },
#endif
};

at::Tensor conv1x1_bias_shortcut_with_quant(const int sm, const at::Tensor& feature,
                                            const at::Tensor& weight, const at::Tensor& bias,
                                            const at::Tensor& shortcut, const at::Tensor& quant,
                                            at::Tensor& out_buf)
{
    if (sm < 75) {
        auto out = at::conv2d(feature, weight);
        return (out + bias.reshape({ 1, -1, 1, 1 }) + shortcut) * quant;
    }
    auto launch_cutlass = [&](auto sm_v) {
        using sm_t = decltype(sm_v);
        return conv1x1_bias_generic_cutlass<sm_t, true, 1, true>(out_buf, feature, weight, bias,
                                                                 shortcut, at::nullopt, quant);
    };
    const int H = feature.size(2);
    const int W = feature.size(3);
    const int C2 = weight.size(0);
    const int C1 = weight.size(1);
    return launch_cutlass_helper(sm, launch_cutlass, HINT_MAP_BIAS_SHORTCUT_WITH_QUANT,
                                 HINT_MAP_BIAS_SHORTCUT_WITH_QUANT_NAME, C2, C1, H, W);
}
