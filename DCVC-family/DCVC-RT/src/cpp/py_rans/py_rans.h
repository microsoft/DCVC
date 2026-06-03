// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "rans.h"
#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class RansEncoder {
public:
    RansEncoder();

    RansEncoder(const RansEncoder&) = delete;
    RansEncoder(RansEncoder&&) = delete;
    RansEncoder& operator=(const RansEncoder&) = delete;
    RansEncoder& operator=(RansEncoder&&) = delete;

    void encode_y(const py::array_t<int16_t>& symbols, const int cdf_group_index);
    void encode_z(const py::array_t<int8_t>& symbols, const int cdf_group_index,
                  const int start_offset, const int per_channel_size);
    void flush();
    py::array_t<uint8_t> get_encoded_stream();
    void reset();
    int add_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                const py::array_t<int32_t>& offsets);
    void empty_cdf_buffer();
    void set_use_two_encoders(bool b);
    bool get_use_two_encoders();

private:
    std::shared_ptr<RansEncoderLib> m_encoder0;
    std::shared_ptr<RansEncoderLib> m_encoder1;
    bool m_use_two_encoders{ false };
};

class RansDecoder {
public:
    RansDecoder();

    RansDecoder(const RansDecoder&) = delete;
    RansDecoder(RansDecoder&&) = delete;
    RansDecoder& operator=(const RansDecoder&) = delete;
    RansDecoder& operator=(RansDecoder&&) = delete;

    void set_stream(const py::array_t<uint8_t>&);

    void decode_y(const py::array_t<uint8_t>& indexes, const int cdf_group_index);
    py::array_t<int8_t> decode_and_get_y(const py::array_t<uint8_t>& indexes, const int cdf_group_index);
    void decode_z(const int total_size, const int cdf_group_index, const int start_offset,
                  const int per_channel_size);
    py::array_t<int8_t> get_decoded_tensor();
    int add_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                const py::array_t<int32_t>& offsets);
    void empty_cdf_buffer();
    void set_use_two_decoders(bool b);
    bool get_use_two_decoders();

private:
    std::shared_ptr<RansDecoderLib> m_decoder0;
    std::shared_ptr<RansDecoderLib> m_decoder1;
    bool m_use_two_decoders{ false };
};

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf, int precision);
