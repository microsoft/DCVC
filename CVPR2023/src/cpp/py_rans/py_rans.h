// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once
#include "rans.h"
#include <memory>
#include <pybind11/numpy.h>

namespace py = pybind11;

// the classes in this file only perform the type conversion
// from python type (numpy) to C++ type (vector)
class RansEncoder {
public:
  RansEncoder(bool multiThread, int streamPart);

  RansEncoder(const RansEncoder &) = delete;
  RansEncoder(RansEncoder &&) = delete;
  RansEncoder &operator=(const RansEncoder &) = delete;
  RansEncoder &operator=(RansEncoder &&) = delete;

  void encode_with_indexes(const py::array_t<int16_t> &symbols,
                           const py::array_t<int16_t> &indexes,
                           const py::array_t<int32_t> &cdfs,
                           const py::array_t<int32_t> &cdfs_sizes,
                           const py::array_t<int32_t> &offsets);
  void flush();
  py::array_t<uint8_t> get_encoded_stream();
  void reset();

private:
  std::vector<std::shared_ptr<RansEncoderLib>> m_encoders;
};

class RansDecoder {
public:
  RansDecoder(int streamPart);

  RansDecoder(const RansDecoder &) = delete;
  RansDecoder(RansDecoder &&) = delete;
  RansDecoder &operator=(const RansDecoder &) = delete;
  RansDecoder &operator=(RansDecoder &&) = delete;

  void set_stream(const py::array_t<uint8_t> &);

  py::array_t<int16_t> decode_stream(const py::array_t<int16_t> &indexes,
                                     const py::array_t<int32_t> &cdfs,
                                     const py::array_t<int32_t> &cdfs_sizes,
                                     const py::array_t<int32_t> &offsets);

private:
  std::vector<std::shared_ptr<RansDecoderLib>> m_decoders;
};
