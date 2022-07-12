/* Copyright 2020 InterDigital Communications, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wsign-compare"
#elif _MSC_VER
#pragma warning(push, 0)
#endif

#include <rans64.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#elif _MSC_VER
#pragma warning(pop)
#endif

namespace py = pybind11;

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class BufferedRansEncoder {
public:
  BufferedRansEncoder() = default;

  BufferedRansEncoder(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder(BufferedRansEncoder &&) = delete;
  BufferedRansEncoder &operator=(const BufferedRansEncoder &) = delete;
  BufferedRansEncoder &operator=(BufferedRansEncoder &&) = delete;

  void encode_with_indexes(const py::array_t<int32_t> &symbols,
                           const py::array_t<int32_t> &indexes,
                           const py::array_t<int32_t> &cdfs,
                           const py::array_t<int32_t> &cdfs_sizes,
                           const py::array_t<int32_t> &offsets);
  py::bytes flush();
  void reset();

private:
  std::vector<RansSymbol> _syms;
};

class RansDecoder {
public:
  RansDecoder() = default;

  RansDecoder(const RansDecoder &) = delete;
  RansDecoder(RansDecoder &&) = delete;
  RansDecoder &operator=(const RansDecoder &) = delete;
  RansDecoder &operator=(RansDecoder &&) = delete;

  void set_stream(const std::string &stream);

  py::array_t<int32_t>
  decode_stream(const py::array_t<int32_t> &indexes,
                const py::array_t<int32_t> &cdfs,
                const py::array_t<int32_t> &cdfs_sizes,
                const py::array_t<int32_t> &offsets);

private:
  Rans64State _rans;
  std::string _stream;
  uint32_t *_ptr;
};
