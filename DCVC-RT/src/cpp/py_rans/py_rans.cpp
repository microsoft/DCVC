// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <numeric>
#include <vector>

namespace py = pybind11;

RansEncoder::RansEncoder()
{
    m_encoder0 = std::make_shared<RansEncoderLibMultiThread>();
    m_encoder1 = std::make_shared<RansEncoderLibMultiThread>();
}

void RansEncoder::encode_y(const py::array_t<int16_t>& symbols, const int cdf_group_index)
{
    py::buffer_info symbols_buf = symbols.request();
    int16_t* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    if (m_use_two_encoders) {
        int symbolSize0 = symbolSize / 2;
        int symbolSize1 = symbolSize - symbolSize0;

        auto vec_symbols0 = std::make_shared<std::vector<int16_t>>(symbolSize0);
        memcpy(vec_symbols0->data(), symbols_ptr, symbolSize0 * sizeof(int16_t));
        m_encoder0->encode_y(vec_symbols0, cdf_group_index);
        auto vec_symbols1 = std::make_shared<std::vector<int16_t>>(symbolSize1);
        memcpy(vec_symbols1->data(), symbols_ptr + symbolSize0, symbolSize1 * sizeof(int16_t));
        m_encoder1->encode_y(vec_symbols1, cdf_group_index);
    } else {
        auto vec_symbols0 = std::make_shared<std::vector<int16_t>>(symbolSize);
        memcpy(vec_symbols0->data(), symbols_ptr, symbolSize * sizeof(int16_t));
        m_encoder0->encode_y(vec_symbols0, cdf_group_index);
    }
}

void RansEncoder::encode_z(const py::array_t<int8_t>& symbols, const int cdf_group_index,
                           const int start_offset, const int per_channel_size)
{
    py::buffer_info symbols_buf = symbols.request();
    int8_t* symbols_ptr = static_cast<int8_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    if (m_use_two_encoders) {
        int symbolSize0 = symbolSize / 2;
        int symbolSize1 = symbolSize - symbolSize0;
        int channel_half = symbolSize0 / per_channel_size;

        auto vec_symbols0 = std::make_shared<std::vector<int8_t>>(symbolSize0);
        memcpy(vec_symbols0->data(), symbols_ptr, symbolSize0 * sizeof(int8_t));
        m_encoder0->encode_z(vec_symbols0, cdf_group_index, start_offset, per_channel_size);
        auto vec_symbols1 = std::make_shared<std::vector<int8_t>>(symbolSize1);
        memcpy(vec_symbols1->data(), symbols_ptr + symbolSize0, symbolSize1 * sizeof(int8_t));
        m_encoder1->encode_z(vec_symbols1, cdf_group_index, start_offset + channel_half,
                             per_channel_size);
    } else {
        auto vec_symbols0 = std::make_shared<std::vector<int8_t>>(symbolSize);
        memcpy(vec_symbols0->data(), symbols_ptr, symbolSize * sizeof(int8_t));
        m_encoder0->encode_z(vec_symbols0, cdf_group_index, start_offset, per_channel_size);
    }
}

int RansEncoder::add_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                         const py::array_t<int32_t>& offsets)
{
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    py::buffer_info offsets_buf = offsets.request();
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);
    int32_t* offsets_ptr = static_cast<int32_t*>(offsets_buf.ptr);

    int cdf_num = static_cast<int>(cdfs_sizes.size());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdf_num);
    memcpy(vec_cdfs_sizes->data(), cdfs_sizes_ptr, sizeof(int32_t) * cdf_num);
    auto vec_offsets = std::make_shared<std::vector<int32_t>>(offsets.size());
    memcpy(vec_offsets->data(), offsets_ptr, sizeof(int32_t) * offsets.size());

    int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
    auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
    auto cdfs_raw = cdfs.unchecked<2>();
    for (int i = 0; i < cdf_num; i++) {
        std::vector<int32_t> t(per_vector_size);
        memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
        vec_cdfs->at(i) = t;
    }

    int cdf_idx = m_encoder0->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
    m_encoder1->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
    return cdf_idx;
}

void RansEncoder::empty_cdf_buffer()
{
    m_encoder0->empty_cdf_buffer();
    m_encoder1->empty_cdf_buffer();
}

void RansEncoder::flush()
{
    m_encoder0->flush();
    m_encoder1->flush();
}

py::array_t<uint8_t> RansEncoder::get_encoded_stream()
{
    if (m_use_two_encoders) {
        auto result0 = m_encoder0->get_encoded_stream();
        int nbytes0 = static_cast<int>(result0->size());
        auto result1 = m_encoder1->get_encoded_stream();
        int nbytes1 = static_cast<int>(result1->size());

        int identical_bytes = 0;
        int check_bytes = std::min(nbytes0, nbytes1);
        check_bytes = std::min(check_bytes, 8);
        for (int i = 0; i < check_bytes; i++) {
            if (result0->at(nbytes0 - 1 - i) != 0) {
                break;
            }
            if (result1->at(nbytes1 - 1 - i) != 0) {
                break;
            }
            identical_bytes++;
        }
        if (identical_bytes == 0 && result0->at(nbytes0 - 1) == result1->at(nbytes1 - 1)) {
            identical_bytes = 1;
        }

        py::array_t<uint8_t> stream(nbytes0 + nbytes1 - identical_bytes);
        py::buffer_info stream_buf = stream.request();
        uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);

        std::copy(result0->begin(), result0->end(), stream_ptr);
        std::reverse_copy(result1->begin(), result1->end() - identical_bytes, stream_ptr + nbytes0);
        return stream;
    }

    auto result0 = m_encoder0->get_encoded_stream();
    int nbytes0 = static_cast<int>(result0->size());

    py::array_t<uint8_t> stream(nbytes0);
    py::buffer_info stream_buf = stream.request();
    uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);

    std::copy(result0->begin(), result0->end(), stream_ptr);
    return stream;
}

void RansEncoder::reset()
{
    m_encoder0->reset();
    m_encoder1->reset();
}

void RansEncoder::set_use_two_encoders(bool b)
{
    m_use_two_encoders = b;
}

bool RansEncoder::get_use_two_encoders()
{
    return m_use_two_encoders;
}

RansDecoder::RansDecoder()
{
    m_decoder0 = std::make_shared<RansDecoderLibMultiThread>();
    m_decoder1 = std::make_shared<RansDecoderLibMultiThread>();
}

void RansDecoder::set_stream(const py::array_t<uint8_t>& encoded)
{
    py::buffer_info encoded_buf = encoded.request();
    const uint8_t* encoded_ptr = static_cast<uint8_t*>(encoded_buf.ptr);
    const int encoded_size = static_cast<int>(encoded.size());
    auto stream0 = std::make_shared<std::vector<uint8_t>>(encoded.size());
    std::copy(encoded_ptr, encoded_ptr + encoded_size, stream0->data());
    m_decoder0->set_stream(stream0);
    if (m_use_two_decoders) {
        auto stream1 = std::make_shared<std::vector<uint8_t>>(encoded.size());
        std::reverse_copy(encoded_ptr, encoded_ptr + encoded_size, stream1->data());
        m_decoder1->set_stream(stream1);
    }
}

void RansDecoder::decode_y(const py::array_t<uint8_t>& indexes, const int cdf_group_index)
{
    py::buffer_info indexes_buf = indexes.request();
    uint8_t* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);

    int indexSize = static_cast<int>(indexes.size());
    if (m_use_two_decoders) {
        int indexSize0 = indexSize / 2;
        int indexSize1 = indexSize - indexSize0;

        auto vec_indexes0 = std::make_shared<std::vector<uint8_t>>(indexSize0);
        std::copy(indexes_ptr, indexes_ptr + indexSize0, vec_indexes0->data());
        m_decoder0->decode_y(vec_indexes0, cdf_group_index);

        auto vec_indexes1 = std::make_shared<std::vector<uint8_t>>(indexSize1);
        std::copy(indexes_ptr + indexSize0, indexes_ptr + indexSize, vec_indexes1->data());
        m_decoder1->decode_y(vec_indexes1, cdf_group_index);
    } else {
        auto vec_indexes0 = std::make_shared<std::vector<uint8_t>>(indexSize);
        std::copy(indexes_ptr, indexes_ptr + indexSize, vec_indexes0->data());
        m_decoder0->decode_y(vec_indexes0, cdf_group_index);
    }
}

py::array_t<int8_t> RansDecoder::decode_and_get_y(const py::array_t<uint8_t>& indexes,
                                                  const int cdf_group_index)
{
    decode_y(indexes, cdf_group_index);
    return get_decoded_tensor();
}

void RansDecoder::decode_z(const int total_size, const int cdf_group_index, const int start_offset,
                           const int per_channel_size)
{
    if (m_use_two_decoders) {
        int symbolSize0 = total_size / 2;
        int symbolSize1 = total_size - symbolSize0;
        int channel_half = symbolSize0 / per_channel_size;
        m_decoder0->decode_z(symbolSize0, cdf_group_index, start_offset, per_channel_size);
        m_decoder1->decode_z(symbolSize1, cdf_group_index, start_offset + channel_half,
                             per_channel_size);
    } else {
        m_decoder0->decode_z(total_size, cdf_group_index, start_offset, per_channel_size);
    }
}

py::array_t<int8_t> RansDecoder::get_decoded_tensor()
{
    if (m_use_two_decoders) {
        auto result0 = m_decoder0->get_decoded_tensor();
        const int total_size0 = static_cast<int>(result0->size());

        auto result1 = m_decoder1->get_decoded_tensor();
        const int total_size1 = static_cast<int>(result1->size());
        py::array_t<int8_t> output(total_size0 + total_size1);
        py::buffer_info buf = output.request();
        int8_t* buf_ptr = static_cast<int8_t*>(buf.ptr);
        std::copy(result0->begin(), result0->end(), buf_ptr);
        std::copy(result1->begin(), result1->end(), buf_ptr + total_size0);

        return output;
    }

    auto result0 = m_decoder0->get_decoded_tensor();
    const int total_size0 = static_cast<int>(result0->size());

    py::array_t<int8_t> output(total_size0);
    py::buffer_info buf = output.request();
    int8_t* buf_ptr = static_cast<int8_t*>(buf.ptr);
    std::copy(result0->begin(), result0->end(), buf_ptr);

    return output;
}

int RansDecoder::add_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                         const py::array_t<int32_t>& offsets)
{
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    py::buffer_info offsets_buf = offsets.request();
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);
    int32_t* offsets_ptr = static_cast<int32_t*>(offsets_buf.ptr);

    int cdf_num = static_cast<int>(cdfs_sizes.size());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdf_num);
    memcpy(vec_cdfs_sizes->data(), cdfs_sizes_ptr, sizeof(int32_t) * cdf_num);
    auto vec_offsets = std::make_shared<std::vector<int32_t>>(offsets.size());
    memcpy(vec_offsets->data(), offsets_ptr, sizeof(int32_t) * offsets.size());

    int per_vector_size = static_cast<int>(cdfs.size() / cdf_num);
    auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
    auto cdfs_raw = cdfs.unchecked<2>();
    for (int i = 0; i < cdf_num; i++) {
        std::vector<int32_t> t(per_vector_size);
        memcpy(t.data(), cdfs_raw.data(i, 0), sizeof(int32_t) * per_vector_size);
        vec_cdfs->at(i) = t;
    }
    int cdf_idx = m_decoder0->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
    m_decoder1->add_cdf(vec_cdfs, vec_cdfs_sizes, vec_offsets);
    return cdf_idx;
}

void RansDecoder::empty_cdf_buffer()
{
    m_decoder0->empty_cdf_buffer();
    m_decoder1->empty_cdf_buffer();
}

void RansDecoder::set_use_two_decoders(bool b)
{
    m_use_two_decoders = b;
}

bool RansDecoder::get_use_two_decoders()
{
    return m_use_two_decoders;
}

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf, int precision)
{
    /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
     * although it's only run once per model after training. See TF/compression
     * implementation for an optimized version. */

    std::vector<uint32_t> cdf(pmf.size() + 1);
    cdf[0] = 0; /* freq 0 */

    std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1, [=](float p) {
        return static_cast<uint32_t>(std::round(p * (1 << precision)) + 0.5);
    });

    const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);

    std::transform(cdf.begin(), cdf.end(), cdf.begin(), [precision, total](uint32_t p) {
        return static_cast<uint32_t>((((1ull << precision) * p) / total));
    });

    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = 1 << precision;

    for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
        if (cdf[i] == cdf[i + 1]) {
            /* Try to steal frequency from low-frequency symbols */
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
                uint32_t freq = cdf[j + 1] - cdf[j];
                if (freq > 1 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }

            assert(best_steal != -1);

            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; ++j) {
                    cdf[j]--;
                }
            } else {
                assert(best_steal > i);
                for (int j = i + 1; j <= best_steal; ++j) {
                    cdf[j]++;
                }
            }
        }
    }

    assert(cdf[0] == 0);
    assert(cdf.back() == (1u << precision));
    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        assert(cdf[i + 1] > cdf[i]);
    }

    return cdf;
}

PYBIND11_MODULE(MLCodec_extensions_cpp, m)
{
    py::class_<RansEncoder>(m, "RansEncoder")
        .def(py::init<>())
        .def("encode_y", &RansEncoder::encode_y)
        .def("encode_z", &RansEncoder::encode_z)
        .def("flush", &RansEncoder::flush)
        .def("get_encoded_stream", &RansEncoder::get_encoded_stream)
        .def("reset", &RansEncoder::reset)
        .def("add_cdf", &RansEncoder::add_cdf)
        .def("empty_cdf_buffer", &RansEncoder::empty_cdf_buffer)
        .def("set_use_two_encoders", &RansEncoder::set_use_two_encoders)
        .def("get_use_two_encoders", &RansEncoder::get_use_two_encoders);

    py::class_<RansDecoder>(m, "RansDecoder")
        .def(py::init<>())
        .def("set_stream", &RansDecoder::set_stream)
        .def("decode_y", &RansDecoder::decode_y)
        .def("decode_and_get_y", &RansDecoder::decode_and_get_y)
        .def("decode_z", &RansDecoder::decode_z)
        .def("get_decoded_tensor", &RansDecoder::get_decoded_tensor)
        .def("add_cdf", &RansDecoder::add_cdf)
        .def("empty_cdf_buffer", &RansDecoder::empty_cdf_buffer)
        .def("set_use_two_decoders", &RansDecoder::set_use_two_decoders)
        .def("get_use_two_decoders", &RansDecoder::get_use_two_decoders);

    m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf, "Return quantized CDF for a given PMF");
}
