// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "py_rans.h"

#include <algorithm>
#include <future>
#include <numeric>
#include <vector>

namespace py = pybind11;

// Count trailing identical zero bytes shared between two encoded streams.
// This allows overlapping the reversed stream to save space.
static int compute_identical_bytes(const std::vector<uint8_t>& a, int na,
                                   const std::vector<uint8_t>& b, int nb)
{
    int identical_bytes = 0;
    int check_bytes = std::min({ na, nb, 8 });
    for (int i = 0; i < check_bytes; i++) {
        if (a[na - 1 - i] != 0) {
            break;
        }
        if (b[nb - 1 - i] != 0) {
            break;
        }
        identical_bytes++;
    }
    if (identical_bytes == 0 && a[na - 1] == b[nb - 1]) {
        identical_bytes = 1;
    }
    return identical_bytes;
}

std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float>& pmf)
{
    /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
     * although it's only run once per model after training. See TF/compression
     * implementation for an optimized version. */
    constexpr int precision = 16;
    constexpr uint32_t prob_max = (1u << precision);
    constexpr int min_freq = 1;

    std::vector<uint32_t> cdf(pmf.size() + 1);
    cdf[0] = 0; /* freq 0 */

    std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                   [=](float p) { return static_cast<uint32_t>(p * prob_max + 0.5); });

    const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);

    std::transform(cdf.begin(), cdf.end(), cdf.begin(), [=](uint32_t p) {
        return static_cast<uint32_t>(((static_cast<uint64_t>(prob_max) * p) / total));
    });

    std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());
    cdf.back() = prob_max;

    for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
        if (cdf[i] + min_freq > cdf[i + 1]) {
            /* Try to steal frequency from low-frequency symbols */
            uint32_t best_freq = ~0u;
            int best_steal = -1;
            for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
                uint32_t freq = cdf[j + 1] - cdf[j];
                if (freq >= min_freq * 2 && freq < best_freq) {
                    best_freq = freq;
                    best_steal = j;
                }
            }

            assert(best_steal != -1);

            if (best_steal < i) {
                for (int j = best_steal + 1; j <= i; ++j) {
                    cdf[j] -= min_freq;
                }
            } else {
                assert(best_steal > i);
                for (int j = i + 1; j <= best_steal; ++j) {
                    cdf[j] += min_freq;
                }
            }
        }
    }

    assert(cdf[0] == 0);
    assert(cdf.back() == prob_max);
    for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
        assert(cdf[i + 1] > cdf[i]);
    }

    return cdf;
}

RansEncoder::RansEncoder()
{
    m_encoders.resize(MAX_EC_PARALLEL);
    for (int i = 0; i < MAX_EC_PARALLEL; i++) {
        m_encoders[i] = std::make_shared<RansEncoderLib>();
    }
}

void RansEncoder::encode_y(const std::shared_ptr<std::vector<int16_t>>& symbols, const int symbolSize)
{
    int n = m_entropy_coder_parallel;
    int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_encoders[i]->encode_y(symbols, size0, size0 * i);
    }
    m_encoders[n - 1]->encode_y(symbols, symbolSize - size0 * (n - 1), size0 * (n - 1));
}

void RansEncoder::encode_y(const py::array_t<int16_t>& symbols)
{
    py::buffer_info symbols_buf = symbols.request();
    int16_t* symbols_ptr = static_cast<int16_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    auto vec_symbols = std::make_shared<std::vector<int16_t>>(symbolSize);
    std::copy(symbols_ptr, symbols_ptr + symbolSize, vec_symbols->data());
    encode_y(vec_symbols, symbolSize);
}

void RansEncoder::encode_z(const std::shared_ptr<std::vector<int8_t>>& symbols,
                           const int cdf_offset, const int ch)
{
    int symbolSize = static_cast<int>(symbols->size());
    int n = m_entropy_coder_parallel;
    int size0 = symbolSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_encoders[i]->encode_z(symbols, size0, size0 * i, cdf_offset, ch);
    }
    m_encoders[n - 1]->encode_z(symbols, symbolSize - size0 * (n - 1), size0 * (n - 1), cdf_offset, ch);
}

void RansEncoder::encode_z(const py::array_t<int8_t>& symbols, const int cdf_offset, const int ch)
{
    py::buffer_info symbols_buf = symbols.request();
    int8_t* symbols_ptr = static_cast<int8_t*>(symbols_buf.ptr);

    int symbolSize = static_cast<int>(symbols.size());
    auto vec_symbols = std::make_shared<std::vector<int8_t>>(symbolSize);
    std::copy(symbols_ptr, symbols_ptr + symbolSize, vec_symbols->data());
    encode_z(vec_symbols, cdf_offset, ch);
}

void RansEncoder::flush()
{
    int n = m_entropy_coder_parallel;
    for (int i = 0; i < n; i++) {
        m_encoders[i]->flush();
    }
}

py::array_t<uint8_t> RansEncoder::get_encoded_stream()
{
    int n = m_entropy_coder_parallel;

    // Collect all encoded streams
    std::vector<std::shared_ptr<std::vector<uint8_t>>> results(n);
    std::vector<int> nbytes(n);
    for (int i = 0; i < n; i++) {
        results[i] = m_encoders[i]->get_encoded_stream();
        nbytes[i] = static_cast<int>(results[i]->size());
    }

    if (n == 1) {
        /**
         * The input arguments to py::array_t<uint8_t> constructor are array size and stride.
         * The code may fail to perform correct encoding-decoding by omitting array stride.
         * It would be more robust by explicitly specifying the stride.
         *
         * By only passing the array size, i.e., "py::array_t<uint8_t> stream(nbytes0);",
         * our empirical results are as follows:
         *   pybind11==3.0.1,  numpy==2.3.3:  succeeded
         *   pybind11==2.10.4, numpy==1.26.0: succeeded
         *   pybind11==2.10.4, numpy==2.3.3:  failed. the array stride defaults to 0
         */
        py::array_t<uint8_t> stream({ nbytes[0] }, { sizeof(uint8_t) });
        py::buffer_info stream_buf = stream.request();
        uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);
        std::copy(results[0]->begin(), results[0]->end(), stream_ptr);
        return stream;
    }

    // For n >= 2, streams are paired: (0,1), (2,3), ...
    // Each pair is merged: stream[2k] forward + stream[2k+1] reversed with zero-byte overlap.
    // If n is odd, the last stream stands alone (forward only).
    int num_pairs = n / 2;
    bool has_tail = (n % 2 != 0);

    // Compute group sizes (each pair merged) and the tail
    std::vector<int> group_sizes(num_pairs);
    std::vector<int> identical(num_pairs);
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        identical[p] = compute_identical_bytes(*results[i0], nbytes[i0], *results[i1], nbytes[i1]);
        group_sizes[p] = nbytes[i0] + nbytes[i1] - identical[p];
    }
    int tail_size = has_tail ? nbytes[n - 1] : 0;

    // Header: (num_pairs - 1 + has_tail) int32_t offsets
    // For n==2 (1 pair, no tail): 0 offsets, no header
    int num_offsets = num_pairs - 1 + (has_tail ? 1 : 0);
    int header_size = num_offsets * 4;

    int total_size = header_size;
    for (int p = 0; p < num_pairs; p++) {
        total_size += group_sizes[p];
    }
    total_size += tail_size;

    py::array_t<uint8_t> stream({ total_size }, { sizeof(uint8_t) });
    py::buffer_info stream_buf = stream.request();
    uint8_t* stream_ptr = static_cast<uint8_t*>(stream_buf.ptr);

    // Write header: cumulative offsets for groups after the first
    // offset[k] = cumulative size of groups 0..k (so decoder knows where group k+1 starts)
    int cumulative = group_sizes[0];
    for (int k = 0; k < num_offsets; k++) {
        *reinterpret_cast<int32_t*>(stream_ptr + k * 4) = cumulative;
        if (k + 1 < num_pairs) {
            cumulative += group_sizes[k + 1];
        } else {
            // This offset points to the tail start (== total payload without header)
            // Already correct: cumulative == sum of all group_sizes
        }
    }

    // Write groups
    int pos = header_size;
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        std::copy(results[i0]->begin(), results[i0]->end(), stream_ptr + pos);
        std::reverse_copy(results[i1]->begin(), results[i1]->end() - identical[p],
                          stream_ptr + pos + nbytes[i0]);
        pos += group_sizes[p];
    }

    // Write tail
    if (has_tail) {
        std::copy(results[n - 1]->begin(), results[n - 1]->end(), stream_ptr + pos);
    }

    return stream;
}

void RansEncoder::reset()
{
    for (int i = 0; i < MAX_EC_PARALLEL; i++) {
        m_encoders[i]->reset();
    }
}

void RansEncoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                          const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index)
{
    int cdf_num = static_cast<int>(cdfs_sizes->size());
    int per_vector_size = static_cast<int>(cdfs->size() / cdf_num);
    auto max_value = std::make_shared<std::vector<int8_t>>(cdf_num);
    auto ransSymbols = std::make_shared<std::vector<std::vector<RansSymbol>>>(cdf_num);
    for (int i = 0; i < cdf_num; i++) {
        max_value->at(i) = static_cast<int8_t>(cdfs_sizes->at(i) - 2);

        const int32_t* cdf = cdfs->data() + i * per_vector_size;
        std::vector<RansSymbol> ransSym(per_vector_size);
        const int ransSize = per_vector_size - 1;
        for (int j = 0; j < ransSize; j++) {
            ransSym[j] = RansSymbol(
                { static_cast<uint16_t>(cdf[j]), static_cast<uint16_t>(cdf[j + 1] - cdf[j]) });
        }
        ransSymbols->at(i) = std::move(ransSym);
    }

    for (int i = 0; i < MAX_EC_PARALLEL; i++) {
        m_encoders[i]->set_cdf(ransSymbols, max_value, index);
    }
}

void RansEncoder::set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                          const int index)
{
    py::buffer_info cdfs_buf = cdfs.request();
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    int32_t* cdfs_ptr = static_cast<int32_t*>(cdfs_buf.ptr);
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);

    auto vec_cdfs = std::make_shared<std::vector<int32_t>>(cdfs.size());
    std::copy(cdfs_ptr, cdfs_ptr + cdfs.size(), vec_cdfs->data());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdfs_sizes.size());
    std::copy(cdfs_sizes_ptr, cdfs_sizes_ptr + cdfs_sizes.size(), vec_cdfs_sizes->data());

    set_cdf(vec_cdfs, vec_cdfs_sizes, index);
}

void RansEncoder::set_entropy_coder_parallel(int n)
{
    assert(n >= 1 && n <= MAX_EC_PARALLEL);
    m_entropy_coder_parallel = n;
}

RansDecoder::RansDecoder()
{
    m_decoded_tensor = std::make_shared<std::vector<int8_t>>(3840 * 2160 / 16 / 16 * 128 * 2);
    m_decoders.resize(MAX_EC_PARALLEL);
    for (int i = 0; i < MAX_EC_PARALLEL; i++) {
        m_decoders[i] = std::make_shared<RansDecoderLib>();
    }
}

void RansDecoder::decode_y(const std::shared_ptr<std::vector<uint8_t>>& indexes, const int indexSize)
{
    m_current_decoded_tensor_size = indexSize;
    if (m_decoded_tensor == nullptr || static_cast<int>(m_decoded_tensor->size()) < indexSize) {
        m_decoded_tensor = std::make_shared<std::vector<int8_t>>(indexSize * 2);
    }
    int8_t* decoded_ptr = m_decoded_tensor->data();

    int n = m_entropy_coder_parallel;
    int size0 = indexSize / n;
    for (int i = 0; i < n - 1; i++) {
        m_decoders[i]->decode_y(decoded_ptr, indexes, size0, size0 * i);
    }
    m_decoders[n - 1]->decode_y(decoded_ptr, indexes, indexSize - size0 * (n - 1), size0 * (n - 1));
}

void RansDecoder::decode_y(const py::array_t<uint8_t>& indexes)
{
    py::buffer_info indexes_buf = indexes.request();
    uint8_t* indexes_ptr = static_cast<uint8_t*>(indexes_buf.ptr);

    int indexSize = static_cast<int>(indexes.size());
    auto vec_indexes = std::make_shared<std::vector<uint8_t>>(indexSize);
    std::copy(indexes_ptr, indexes_ptr + indexSize, vec_indexes->data());

    decode_y(vec_indexes, indexSize);
}

void RansDecoder::decode_z(const int total_size, const int cdf_offset, const int ch)
{
    m_current_decoded_tensor_size = total_size;
    if (m_decoded_tensor == nullptr || static_cast<int>(m_decoded_tensor->size()) < total_size) {
        m_decoded_tensor = std::make_shared<std::vector<int8_t>>(total_size * 2);
    }
    int8_t* decoded_ptr = m_decoded_tensor->data();

    int n = m_entropy_coder_parallel;
    int size0 = total_size / n;
    for (int i = 0; i < n - 1; i++) {
        m_decoders[i]->decode_z(decoded_ptr, size0, size0 * i, cdf_offset, ch);
    }
    m_decoders[n - 1]->decode_z(decoded_ptr, total_size - size0 * (n - 1), size0 * (n - 1),
                                cdf_offset, ch);
}

std::shared_ptr<std::vector<int8_t>> RansDecoder::get_decoded_tensor_cpp()
{
    int n = m_entropy_coder_parallel;
    for (int i = 0; i < n; i++) {
        m_decoders[i]->wait_for_decoding_finish();
    }
    return m_decoded_tensor;
}

void RansDecoder::set_cdf(const std::shared_ptr<std::vector<int32_t>>& cdfs,
                          const std::shared_ptr<std::vector<int32_t>>& cdfs_sizes, const int index)
{
    int cdf_num = static_cast<int>(cdfs_sizes->size());

    int per_vector_size = static_cast<int>(cdfs->size() / cdf_num);
    auto vec_cdfs = std::make_shared<std::vector<std::vector<int32_t>>>(cdf_num);
    auto max_value = std::make_shared<std::vector<int8_t>>(cdf_num);
    for (int i = 0; i < cdf_num; i++) {
        max_value->at(i) = static_cast<int8_t>(cdfs_sizes->at(i) - 2);

        std::vector<int32_t> t(per_vector_size);
        std::copy(cdfs->data() + i * per_vector_size,
                  cdfs->data() + i * per_vector_size + per_vector_size, t.data());
        vec_cdfs->at(i) = std::move(t);
    }

    for (int i = 0; i < MAX_EC_PARALLEL; i++) {
        m_decoders[i]->set_cdf(vec_cdfs, max_value, index);
    }
}

void RansDecoder::set_cdf(const py::array_t<int32_t>& cdfs, const py::array_t<int32_t>& cdfs_sizes,
                          const int index)
{
    py::buffer_info cdfs_buf = cdfs.request();
    py::buffer_info cdfs_sizes_buf = cdfs_sizes.request();
    int32_t* cdfs_ptr = static_cast<int32_t*>(cdfs_buf.ptr);
    int32_t* cdfs_sizes_ptr = static_cast<int32_t*>(cdfs_sizes_buf.ptr);

    auto vec_cdfs = std::make_shared<std::vector<int32_t>>(cdfs.size());
    std::copy(cdfs_ptr, cdfs_ptr + cdfs.size(), vec_cdfs->data());
    auto vec_cdfs_sizes = std::make_shared<std::vector<int32_t>>(cdfs_sizes.size());
    std::copy(cdfs_sizes_ptr, cdfs_sizes_ptr + cdfs_sizes.size(), vec_cdfs_sizes->data());

    set_cdf(vec_cdfs, vec_cdfs_sizes, index);
}

void RansDecoder::set_entropy_coder_parallel(int n)
{
    assert(n >= 1 && n <= MAX_EC_PARALLEL);
    m_entropy_coder_parallel = n;
}

void RansDecoder::set_stream(const uint8_t* ptr, const int size)
{
    int n = m_entropy_coder_parallel;

    if (n == 1) {
        auto stream0 = std::make_shared<std::vector<uint8_t>>(size);
        std::copy(ptr, ptr + size, stream0->data());
        m_decoders[0]->set_stream(stream0);
        return;
    }

    if (n == 2) {
        // No header, entire buffer is one pair
        auto stream0 = std::make_shared<std::vector<uint8_t>>(size);
        std::copy(ptr, ptr + size, stream0->data());
        m_decoders[0]->set_stream(stream0);

        auto stream1 = std::make_shared<std::vector<uint8_t>>(size);
        std::reverse_copy(ptr, ptr + size, stream1->data());
        m_decoders[1]->set_stream(stream1);
        return;
    }

    // n >= 3: header + groups + optional tail
    int num_pairs = n / 2;
    bool has_tail = (n % 2 != 0);
    int num_offsets = num_pairs - 1 + (has_tail ? 1 : 0);
    int header_size = num_offsets * 4;

    // Read cumulative offsets
    std::vector<int> offsets(num_offsets);
    for (int k = 0; k < num_offsets; k++) {
        offsets[k] = *reinterpret_cast<const int32_t*>(ptr + k * 4);
    }

    // Compute group start/end positions (relative to after header)
    const uint8_t* payload = ptr + header_size;
    int payload_size = size - header_size;

    std::vector<int> group_start(num_pairs);
    std::vector<int> group_size(num_pairs);
    group_start[0] = 0;
    group_size[0] = offsets[0];  // first offset == size of first group
    for (int p = 1; p < num_pairs; p++) {
        group_start[p] = offsets[p - 1];
        if (p < num_offsets) {
            group_size[p] = offsets[p] - offsets[p - 1];
        } else {
            // Last group: extends to end of payload (or to tail start)
            int groups_end = has_tail ? offsets[num_offsets - 1] : payload_size;
            group_size[p] = groups_end - offsets[p - 1];
        }
    }

    // Set streams for each pair
    for (int p = 0; p < num_pairs; p++) {
        int i0 = p * 2;
        int i1 = p * 2 + 1;
        const uint8_t* group_ptr = payload + group_start[p];
        int gs = group_size[p];

        auto s0 = std::make_shared<std::vector<uint8_t>>(gs);
        std::copy(group_ptr, group_ptr + gs, s0->data());
        m_decoders[i0]->set_stream(s0);

        auto s1 = std::make_shared<std::vector<uint8_t>>(gs);
        std::reverse_copy(group_ptr, group_ptr + gs, s1->data());
        m_decoders[i1]->set_stream(s1);
    }

    // Tail
    if (has_tail) {
        int tail_start = offsets[num_offsets - 1];
        int tail_size = payload_size - tail_start;
        const uint8_t* tail_ptr = payload + tail_start;

        auto s = std::make_shared<std::vector<uint8_t>>(tail_size);
        std::copy(tail_ptr, tail_ptr + tail_size, s->data());
        m_decoders[n - 1]->set_stream(s);
    }
}

void RansDecoder::set_stream(const py::array_t<uint8_t>& encoded)
{
    py::buffer_info encoded_buf = encoded.request();
    const uint8_t* encoded_ptr = static_cast<uint8_t*>(encoded_buf.ptr);
    const int encoded_size = static_cast<int>(encoded.size());
    set_stream(encoded_ptr, encoded_size);
}
