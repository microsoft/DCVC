// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(MLCodec_extensions_cpp, m)
{
    py::class_<RansEncoder>(m, "RansEncoder")
        .def(py::init<>())
        .def("encode_y", py::overload_cast<const py::array_t<int16_t>&>(&RansEncoder::encode_y))
        .def("encode_z", py::overload_cast<const py::array_t<int8_t>&, const int, const int>(
                             &RansEncoder::encode_z))
        .def("flush", &RansEncoder::flush)
        .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
        .def("reset", &RansEncoder::reset)
        .def("set_cdf",
             py::overload_cast<const py::array_t<int32_t>&, const py::array_t<int32_t>&, const int>(
                 &RansEncoder::set_cdf))
        .def("set_entropy_coder_parallel", &RansEncoder::set_entropy_coder_parallel);

    py::class_<RansDecoder>(m, "RansDecoder")
        .def(py::init<>())
        .def("set_stream", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::set_stream))
        .def("decode_y", py::overload_cast<const py::array_t<uint8_t>&>(&RansDecoder::decode_y))
        .def("decode_z", &RansDecoder::decode_z)
        .def("set_cdf",
             py::overload_cast<const py::array_t<int32_t>&, const py::array_t<int32_t>&, const int>(
                 &RansDecoder::set_cdf))
        .def("set_entropy_coder_parallel", &RansDecoder::set_entropy_coder_parallel);

    m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "Return quantized CDF for a given PMF");
}
