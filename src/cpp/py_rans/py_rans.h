// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "rans.h"
#include <memory>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr int MAX_EC_PARALLEL = 8;

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf);

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class RansEncoder {
public:
    RansEncoder();

    RansEncoder(const RansEncoder&) = delete;
    RansEncoder(RansEncoder&&) = delete;
    RansEncoder& operator=(const RansEncoder&) = delete;
    RansEncoder& operator=(RansEncoder&&) = delete;

    // symbols may contain more elements than symbolSize
    void encode_y(const std::shared_ptr<std::vector<int16_t>>& symbols, const int symbolSize);
    void encode_y(const py::array_t<int16_t>& symbols);
    void encode_z(const std::shared_ptr<std::vector<int8_t>>& symbols, const int cdf_offset,
                  const int ch);
    void encode_z(const py::array_t<int8_t>& symbols, const int cdf_offset, const int ch);
    void flush();
    py::array_t<uint8_t> get_encoded_stream();
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                 const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index);
    void set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                 const int index);
    void set_entropy_coder_parallel(int n);

private:
    std::vector<std::shared_ptr<RansEncoderLib>> m_encoders;
    int m_entropy_coder_parallel{ 1 };
};

class RansDecoder {
public:
    RansDecoder();

    RansDecoder(const RansDecoder&) = delete;
    RansDecoder(RansDecoder&&) = delete;
    RansDecoder& operator=(const RansDecoder&) = delete;
    RansDecoder& operator=(RansDecoder&&) = delete;

    void decode_y(const std::shared_ptr<std::vector<uint8_t>>& indexes, const int indexSize);
    void decode_y(const py::array_t<uint8_t>& indexes);
    // if is_stream_nhmw, ch is channel number
    // otherwise, ch is per channel element number
    void decode_z(const int total_size, const int cdf_offset, const int ch);
    std::shared_ptr<std::vector<int8_t>> get_decoded_tensor_cpp();
    void set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                 const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index);
    void set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                 const int index);
    void set_entropy_coder_parallel(int n);
    void set_stream(const py::array_t<uint8_t>&);
    void set_stream(const uint8_t* ptr, const int size);

private:
    std::shared_ptr<std::vector<int8_t>> m_decoded_tensor;
    int m_current_decoded_tensor_size{ 0 };
    std::vector<std::shared_ptr<RansDecoderLib>> m_decoders;
    int m_entropy_coder_parallel{ 1 };
};
