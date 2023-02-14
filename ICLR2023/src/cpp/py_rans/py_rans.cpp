// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <vector>
#include <future>

namespace py = pybind11;

RansEncoder::RansEncoder(bool multiThread, int streamPart=1) {
  bool useMultiThread = multiThread || streamPart > 1;
  if (useMultiThread) {
    for (int i=0; i<streamPart; i++) {
      m_encoders.push_back(std::make_shared<RansEncoderLibMultiThread>());
    }
  } else {
    for (int i=0; i<streamPart; i++) {
      m_encoders.push_back(std::make_shared<RansEncoderLib>());
    }
  }
}

void RansEncoder::encode_with_indexes(const py::array_t<int16_t> &symbols,
                                      const py::array_t<int16_t> &indexes,
                                      const py::array_t<int32_t> &cdfs,
                                      const py::array_t<int32_t> &cdfs_sizes,
                                      const py::array_t<int32_t> &offsets) {
  py::buffer_info symbols_buf = symbols.request();
  py::buffer_info indexes_buf = indexes.request();
  py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
  py::buffer_info offsets_buf = offsets.request();
  int16_t *symbols_ptr = static_cast<int16_t *>(symbols_buf.ptr);
  int16_t *indexes_ptr = static_cast<int16_t *>(indexes_buf.ptr);
  int32_t *cdfs_sizes_ptr = static_cast<int32_t *>(cdfs_sizes_buf.ptr);
  int32_t *offsets_ptr = static_cast<int32_t *>(offsets_buf.ptr);

  std::vector<int32_t> vec_cdfs_sizes(cdfs_sizes.size());
  memcpy(vec_cdfs_sizes.data(), cdfs_sizes_ptr,
         sizeof(int32_t) * cdfs_sizes.size());
  std::vector<int32_t> vec_offsets(offsets.size());
  memcpy(vec_offsets.data(), offsets_ptr, sizeof(int32_t) * offsets.size());

  std::vector<std::vector<int32_t>> vec_cdfs;
  int cdf_num = static_cast<int>(cdfs_sizes.size());
  int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
  auto cdfs_raw = cdfs.unchecked<2>();
  for (int i = 0; i < cdf_num; i++) {
    std::vector<int32_t> t(per_vector_size);
    memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
    vec_cdfs.push_back(t);
  }

  int encoderNum = m_encoders.size();
  int perEncoderSymbolSize = symbols.size() / encoderNum;
  int lastEncoderSymbolSize = symbols.size() - perEncoderSymbolSize * (encoderNum - 1);
  for (int i=0; i < encoderNum - 1; i++) {
    std::vector<int16_t> vec_symbols(perEncoderSymbolSize);
    memcpy(vec_symbols.data(), symbols_ptr + i*perEncoderSymbolSize, sizeof(int16_t) * perEncoderSymbolSize);
    std::vector<int16_t> vec_indexes(perEncoderSymbolSize);
    memcpy(vec_indexes.data(), indexes_ptr + i*perEncoderSymbolSize, sizeof(int16_t) * perEncoderSymbolSize);
    m_encoders[i]->encode_with_indexes(vec_symbols, vec_indexes, vec_cdfs,
                                vec_cdfs_sizes, vec_offsets);
  }

  std::vector<int16_t> vec_symbols(lastEncoderSymbolSize);
  memcpy(vec_symbols.data(), symbols_ptr + (encoderNum - 1)*perEncoderSymbolSize, sizeof(int16_t) * lastEncoderSymbolSize);
  std::vector<int16_t> vec_indexes(perEncoderSymbolSize);
  memcpy(vec_indexes.data(), indexes_ptr + (encoderNum - 1)*perEncoderSymbolSize, sizeof(int16_t) * lastEncoderSymbolSize);
  m_encoders[encoderNum - 1]->encode_with_indexes(vec_symbols, vec_indexes, vec_cdfs,
                              vec_cdfs_sizes, vec_offsets);
}

void RansEncoder::flush() {
  for (int i=0; i<static_cast<int>(m_encoders.size()); i++) {
    m_encoders[i]->flush();
 }
}

py::array_t<uint8_t> RansEncoder::get_encoded_stream() {
  std::vector<std::vector<uint8_t>> results;
  int maximumSize = 0;
  int totalSize = 0;
  int encoderNumber = static_cast<int>(m_encoders.size());
  for (int i=0; i<encoderNumber; i++) {
    std::vector<uint8_t> result = m_encoders[i]->get_encoded_stream();
    results.push_back(result);
    int nbytes = static_cast<int>(result.size());
    if (i < encoderNumber - 1 && nbytes > maximumSize) {
      maximumSize = nbytes;
    }
    totalSize += nbytes;
  }

  int overhead = 1;
  int perStreamHeader = maximumSize > 65535 ? 4 : 2;
  if (encoderNumber > 1) {
    overhead += ((encoderNumber - 1) * perStreamHeader);
  }

  py::array_t<uint8_t> stream(totalSize + overhead);
  py::buffer_info stream_buf = stream.request();
  uint8_t *stream_ptr = static_cast<uint8_t *>(stream_buf.ptr);

  uint8_t flag = ((encoderNumber - 1) << 4) + (perStreamHeader == 2 ? 1 : 0);
  memcpy(stream_ptr, &flag, 1);
  for (int i=0; i<encoderNumber - 1; i++) {
    if (perStreamHeader == 2) {
      uint16_t perStreamSize = static_cast<uint16_t>(results[i].size());
      memcpy(stream_ptr + 1 + 2 * i, &perStreamSize, 2);
    }
    else {
      uint32_t perStreamSize = static_cast<uint32_t>(results[i].size());
      memcpy(stream_ptr + 1 + 4 * i, &perStreamSize, 4);
    }
  }

  int offset = overhead;
  for (int i=0; i<encoderNumber; i++) {
    int nbytes = static_cast<int>(results[i].size());
    memcpy(stream_ptr + offset, results[i].data(), nbytes);
    offset += nbytes;
  }
  return stream;
}

void RansEncoder::reset() {
  for (int i=0; i<static_cast<int>(m_encoders.size()); i++) {
    m_encoders[i]->reset();
  }
}

RansDecoder::RansDecoder(int streamPart) {
  for (int i=0; i<streamPart; i++) {
    m_decoders.push_back(std::make_shared<RansDecoderLib>());
  }
}

void RansDecoder::set_stream(const py::array_t<uint8_t> &encoded) {
  py::buffer_info encoded_buf = encoded.request();
  uint8_t flag = *(static_cast<uint8_t*>(encoded_buf.ptr));
  int numberOfStreams = (flag >> 4) + 1;
  assert(numberOfStreams == m_decoders.size());

  uint8_t perStreamSizeLength = (flag & 0x0f) == 1 ? 2 : 4;
  if (numberOfStreams == 1) {
    int nbytes = static_cast<int>(encoded.size()) - 1;
    std::vector<uint8_t> stream(nbytes);
    memcpy(stream.data(), static_cast<void *>(static_cast<uint8_t*>(encoded_buf.ptr) + 1), nbytes);
    m_decoders[0]->set_stream(stream);
  }
  else {
    std::vector<uint32_t> perStreamSize;
    int offset = 1;
    int totalSize = 0;
    for (int i=0; i<numberOfStreams - 1; i++) {
      if (perStreamSizeLength == 2) {
        uint16_t streamSize = *(reinterpret_cast<uint16_t *>(static_cast<uint8_t *>(encoded_buf.ptr) + offset));
        offset += 2;
        perStreamSize.push_back(streamSize);
        totalSize += streamSize;
      }
      else {
        uint32_t streamSize = *(reinterpret_cast<uint32_t *>(static_cast<uint8_t *>(encoded_buf.ptr) + offset));
        offset += 4;
        perStreamSize.push_back(streamSize);
        totalSize += streamSize;
      }
    }
    perStreamSize.push_back(static_cast<int>(encoded.size()) - offset - totalSize);
    for (int i=0; i<numberOfStreams; i++) {
      std::vector<uint8_t> stream(perStreamSize[i]);
      memcpy(stream.data(), static_cast<void *>(static_cast<uint8_t*>(encoded_buf.ptr) + offset), perStreamSize[i]);
      m_decoders[i]->set_stream(stream);
      offset += perStreamSize[i];
    }
  }
}

std::vector<int16_t> decode_future(RansDecoderLib* pDecoder, PendingDecoding* pendingDecoding) {
  return pDecoder->decode_stream(*(pendingDecoding->indexes), *(pendingDecoding->cdfs), *(pendingDecoding->cdfs_sizes), *(pendingDecoding->offsets));
}

py::array_t<int16_t>
RansDecoder::decode_stream(const py::array_t<int16_t> &indexes,
                           const py::array_t<int32_t> &cdfs,
                           const py::array_t<int32_t> &cdfs_sizes,
                           const py::array_t<int32_t> &offsets) {
  py::buffer_info indexes_buf = indexes.request();
  py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
  py::buffer_info offsets_buf = offsets.request();
  int16_t *indexes_ptr = static_cast<int16_t *>(indexes_buf.ptr);
  int32_t *cdfs_sizes_ptr = static_cast<int32_t *>(cdfs_sizes_buf.ptr);
  int32_t *offsets_ptr = static_cast<int32_t *>(offsets_buf.ptr);

  std::vector<int32_t> vec_cdfs_sizes(cdfs_sizes.size());
  memcpy(vec_cdfs_sizes.data(), cdfs_sizes_ptr,
         sizeof(int32_t) * cdfs_sizes.size());
  std::vector<int32_t> vec_offsets(offsets.size());
  memcpy(vec_offsets.data(), offsets_ptr, sizeof(int32_t) * offsets.size());

  std::vector<std::vector<int32_t>> vec_cdfs;
  int cdf_num = static_cast<int>(cdfs_sizes.size());
  int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
  auto cdfs_raw = cdfs.unchecked<2>();
  for (int i = 0; i < cdf_num; i++) {
    std::vector<int32_t> t(per_vector_size);
    memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
    vec_cdfs.push_back(t);
  }
  int decoderNum = m_decoders.size();
  int perDecoderSymbolSize = indexes.size() / decoderNum;
  int lastDecoderSymbolSize = indexes.size() - perDecoderSymbolSize * (decoderNum - 1);

  std::vector<std::shared_future<std::vector<int16_t>>> results;
  std::vector<std::vector<int16_t>> vec_indexes(decoderNum);
  std::vector<PendingDecoding> pendingDecoding(decoderNum);

  for (int i=0; i<decoderNum - 1; i++) {
    vec_indexes[i].resize(perDecoderSymbolSize);
    memcpy(vec_indexes[i].data(), indexes_ptr + i * perDecoderSymbolSize, sizeof(int16_t) * perDecoderSymbolSize);

    pendingDecoding[i].indexes = &(vec_indexes[i]);
    pendingDecoding[i].cdfs = &(vec_cdfs);
    pendingDecoding[i].cdfs_sizes = &(vec_cdfs_sizes);
    pendingDecoding[i].offsets = &(vec_offsets);
    std::shared_future<std::vector<int16_t>> result = std::async(std::launch::async, decode_future, m_decoders[i].get(), &(pendingDecoding[i]));
    results.push_back(result);
  }
  vec_indexes[decoderNum - 1].resize(lastDecoderSymbolSize);
  memcpy(vec_indexes[decoderNum - 1].data(), indexes_ptr + (decoderNum - 1) * perDecoderSymbolSize, sizeof(int16_t) * lastDecoderSymbolSize);

  pendingDecoding[decoderNum - 1].indexes = &(vec_indexes[decoderNum - 1]);
  pendingDecoding[decoderNum - 1].cdfs = &(vec_cdfs);
  pendingDecoding[decoderNum - 1].cdfs_sizes = &(vec_cdfs_sizes);
  pendingDecoding[decoderNum - 1].offsets = &(vec_offsets);
  std::shared_future<std::vector<int16_t>> result = std::async(std::launch::async, decode_future, m_decoders[decoderNum - 1].get(), &(pendingDecoding[decoderNum - 1]));
  results.push_back(result);

  py::array_t<int16_t> output(indexes.size());
  py::buffer_info buf = output.request();
  int offset = 0;
  for (int i=0; i<decoderNum; i++) {
    std::vector<int16_t> result = results[i].get();
    int resultSize = static_cast<int>(result.size());
    memcpy(static_cast<void *>(static_cast<int16_t *>(buf.ptr) + offset), result.data(), sizeof(int16_t) * resultSize);
    offset += resultSize;
  }

  return output;
}

PYBIND11_MODULE(MLCodec_rans, m) {
  m.attr("__name__") = "MLCodec_rans";

  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<bool, int>())
      .def("encode_with_indexes", &RansEncoder::encode_with_indexes)
      .def("flush", &RansEncoder::flush)
      .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
      .def("reset", &RansEncoder::reset);

  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<int>())
      .def("set_stream", &RansDecoder::set_stream)
      .def("decode_stream", &RansDecoder::decode_stream);
}
