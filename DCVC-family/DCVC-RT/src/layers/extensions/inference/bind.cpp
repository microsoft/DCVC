// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "def.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("process_with_mask_cuda", &process_with_mask_cuda);
    m.def("combine_for_reading_2x_cuda", &combine_for_reading_2x_cuda);
    m.def("restore_y_2x_cuda", &restore_y_2x_cuda);
    m.def("restore_y_4x_cuda", &restore_y_4x_cuda);
    m.def("build_index_dec_cuda", &build_index_dec_cuda);
    m.def("build_index_enc_cuda", &build_index_enc_cuda);
    m.def("bias_quant_cuda", &bias_quant_cuda);
    m.def("round_and_to_int8_cuda", &round_and_to_int8_cuda);
    m.def("clamp_reciprocal_with_quant_cuda", &clamp_reciprocal_with_quant_cuda);
    m.def("add_and_multiply_cuda", &add_and_multiply_cuda);
    m.def("bias_pixel_shuffle_8_cuda", &bias_pixel_shuffle_8_cuda);
    m.def("replicate_pad_cuda", &replicate_pad_cuda);
    m.def("bias_wsilu_depthwise_conv2d_cuda", &bias_wsilu_depthwise_conv2d_cuda);

    py::class_<DepthConvProxy>(m, "DepthConvProxy")
        .def(py::init<>())
        .def("set_param", &DepthConvProxy::set_param)
        .def("set_param_with_adaptor", &DepthConvProxy::set_param_with_adaptor)
        .def("forward", &DepthConvProxy::forward)
        .def("forward_with_quant_step", &DepthConvProxy::forward_with_quant_step)
        .def("forward_with_cat", &DepthConvProxy::forward_with_cat);

    py::class_<SubpelConv2xProxy>(m, "SubpelConv2xProxy")
        .def(py::init<>())
        .def("set_param", &SubpelConv2xProxy::set_param)
        .def("forward", &SubpelConv2xProxy::forward)
        .def("forward_with_cat", &SubpelConv2xProxy::forward_with_cat);
}
