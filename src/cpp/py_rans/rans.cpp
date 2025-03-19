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

/* Rans64 extensions from:
 * https://fgiesen.wordpress.com/2015/12/21/rans-in-practice/
 * Unbounded range coding from:
 * https://github.com/tensorflow/compression/blob/master/tensorflow_compression/cc/kernels/unbounded_index_range_coding_kernels.cc
 **/

#include "rans.h"

#include <algorithm>
#include <cassert>
#include <cstring>

constexpr uint16_t bypass_precision = 2; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

inline void RansEncPutBits(RansState& r, uint8_t*& ptr, uint32_t val)
{
    RansAssert(bypass_precision <= 8);
    RansAssert(val < (1u << bypass_precision));

    constexpr uint32_t freq = 1 << (SCALE_BITS - bypass_precision);
    constexpr uint32_t x_max = freq << ENC_RENORM_SHIFT_BITS;
    while (r >= x_max) {
        *(--ptr) = static_cast<uint8_t>(r & 0xff);
        r >>= 8;
    }

    r = (r << bypass_precision) | val;
}

inline uint32_t RansDecGetBits(RansState& r, uint8_t*& ptr)
{
    uint32_t val = r & ((1u << bypass_precision) - 1);

    /* Re-normalize */
    r = r >> bypass_precision;
    if (r < RANS_BYTE_L) {
        r = (r << 8) | *ptr++;
        RansAssert(r >= RANS_BYTE_L);
    }

    return val;
}

RansEncoderLib::RansEncoderLib()
{
    _stream = std::make_shared<std::vector<uint8_t>>();
}

int RansEncoderLib::add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                            const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                            const std::shared_ptr<std::vector<int32_t>> offsets)
{

    auto ransSymbols = std::make_shared<std::vector<std::vector<RansSymbol>>>(cdfs->size());
    for (int i = 0; i < static_cast<int>(cdfs->size()); i++) {
        const int32_t* cdf = cdfs->at(i).data();
        std::vector<RansSymbol> ransSym(cdfs->at(i).size());
        const int ransSize = static_cast<int>(ransSym.size() - 1);
        for (int j = 0; j < ransSize; j++) {
            ransSym[j] = RansSymbol(
                { static_cast<uint16_t>(cdf[j]), static_cast<uint16_t>(cdf[j + 1] - cdf[j]) });
        }
        ransSymbols->at(i) = ransSym;
    }

    _ransSymbols.push_back(ransSymbols);
    _cdfs_sizes.push_back(cdfs_sizes);
    _offsets.push_back(offsets);
    return static_cast<int>(_ransSymbols.size()) - 1;
}

void RansEncoderLib::empty_cdf_buffer()
{
    _ransSymbols.clear();
    _cdfs_sizes.clear();
    _offsets.clear();
}

FORCE_INLINE void RansEncoderLib::encode_one_symbol(uint8_t*& ptr, RansState& rans, const int32_t symbol,
                                                    const int32_t cdf_size, const int32_t offset,
                                                    const std::vector<RansSymbol>& ransSymbols)
{
    const int32_t max_value = cdf_size - 2;
    int32_t value = symbol - offset;

    uint32_t raw_val = 0;
    if (value < 0) {
        raw_val = -2 * value - 1;
        value = max_value;
    } else if (value >= max_value) {
        raw_val = 2 * (value - max_value);
        value = max_value;
    }

    if (value == max_value) {
        std::vector<uint16_t> bypassBins;
        bypassBins.reserve(20);
        /* Determine the number of bypasses (in bypass_precision size) needed to
         * encode the raw value. */
        int32_t n_bypass = 0;
        while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
            ++n_bypass;
        }

        /* Encode number of bypasses */
        int32_t val = n_bypass;
        while (val >= max_bypass_val) {
            bypassBins.push_back(max_bypass_val);
            val -= max_bypass_val;
        }
        bypassBins.push_back(static_cast<uint16_t>(val));

        /* Encode raw value */
        for (int32_t j = 0; j < n_bypass; ++j) {
            const int32_t val1 = (raw_val >> (j * bypass_precision)) & max_bypass_val;
            bypassBins.push_back(static_cast<uint16_t>(val1));
        }

        for (auto it = bypassBins.rbegin(); it < bypassBins.rend(); it++) {
            RansEncPutBits(rans, ptr, *it);
        }
    }
    RansEncPut(rans, ptr, ransSymbols[value].start, ransSymbols[value].range);
}

void RansEncoderLib::encode_y(const std::shared_ptr<std::vector<int16_t>> symbols,
                              const int cdf_group_index)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.symbols_y = symbols;
    p.cdf_group_index = cdf_group_index;
    m_pendingEncodingList.push_back(p);
}

void RansEncoderLib::encode_z(const std::shared_ptr<std::vector<int8_t>> symbols,
                              const int cdf_group_index, const int start_offset,
                              const int per_channel_size)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.symbols_z = symbols;
    p.cdf_group_index = cdf_group_index;
    p.start_offset = start_offset;
    p.per_channel_size = per_channel_size;
    m_pendingEncodingList.push_back(p);
}
#include <iostream>
FORCE_INLINE void RansEncoderLib::encode_y_internal(uint8_t*& ptr, RansState& rans,
                                                    const std::shared_ptr<std::vector<int16_t>> symbols,
                                                    const int cdf_group_index)
{
    // backward loop on symbols from the end;
    const int16_t* symbols_ptr = symbols->data();
    const int32_t* cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
    const int32_t* offsets_ptr = _offsets[cdf_group_index]->data();
    const int symbol_size = static_cast<int>(symbols->size());

    for (int i = symbol_size - 1; i >= 0; i--) {
        const int32_t combined_symbol = symbols_ptr[i];
        const int32_t cdf_idx = combined_symbol & 0xff;
        const int32_t s = combined_symbol >> 8;
        encode_one_symbol(ptr, rans, s, cdfs_sizes_ptr[cdf_idx], offsets_ptr[cdf_idx],
                          _ransSymbols[cdf_group_index]->at(cdf_idx));
    }
}

FORCE_INLINE void RansEncoderLib::encode_z_internal(uint8_t*& ptr, RansState& rans,
                                                    const std::shared_ptr<std::vector<int8_t>> symbols,
                                                    const int cdf_group_index, const int start_offset,
                                                    const int per_channel_size)
{
    // backward loop on symbols from the end;
    const int8_t* symbols_ptr = symbols->data();
    const int32_t* cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
    const int32_t* offsets_ptr = _offsets[cdf_group_index]->data();
    const int symbol_size = static_cast<int>(symbols->size());

    for (int i = symbol_size - 1; i >= 0; i--) {
        const int32_t cdf_idx = i / per_channel_size + start_offset;
        encode_one_symbol(ptr, rans, symbols_ptr[i], cdfs_sizes_ptr[cdf_idx], offsets_ptr[cdf_idx],
                          _ransSymbols[cdf_group_index]->at(cdf_idx));
    }
}

void RansEncoderLib::flush()
{
    RansState rans;
    RansEncInit(rans);

    int32_t total_symbol_size = 0;
    for (auto it = m_pendingEncodingList.begin(); it != m_pendingEncodingList.end(); it++) {
        if (it->workType == WorkType::EncodeDecodeY) {
            total_symbol_size += static_cast<int32_t>(it->symbols_y->size());
        } else if (it->workType == WorkType::EncodeDecodeZ) {
            total_symbol_size += static_cast<int32_t>(it->symbols_z->size());
        }
    }

    if (total_symbol_size == 0) {
        _stream->resize(0);
        return;
    }

    uint8_t* output = new uint8_t[total_symbol_size];  // too much space ?
    uint8_t* ptrEnd = output + total_symbol_size;
    uint8_t* ptr = ptrEnd;
    assert(ptr != nullptr);

    for (auto it = m_pendingEncodingList.rbegin(); it != m_pendingEncodingList.rend(); it++) {
        PendingTask p = *it;
        if (p.workType == WorkType::EncodeDecodeY) {
            encode_y_internal(ptr, rans, p.symbols_y, p.cdf_group_index);
        } else if (p.workType == WorkType::EncodeDecodeZ) {
            encode_z_internal(ptr, rans, p.symbols_z, p.cdf_group_index, p.start_offset,
                              p.per_channel_size);
        }
    }

    RansEncFlush(rans, ptr);

    const int nbytes = static_cast<int>(std::distance(ptr, ptrEnd));

    _stream->resize(nbytes);
    memcpy(_stream->data(), ptr, nbytes);
    delete[] output;
}

std::shared_ptr<std::vector<uint8_t>> RansEncoderLib::get_encoded_stream()
{
    return _stream;
}

void RansEncoderLib::reset()
{
    m_pendingEncodingList.clear();
    _stream->clear();
}

RansEncoderLibMultiThread::RansEncoderLibMultiThread()
    : RansEncoderLib()
    , m_finish(false)
    , m_result_ready(false)
{
    m_thread = std::thread(&RansEncoderLibMultiThread::worker, this);
}
RansEncoderLibMultiThread::~RansEncoderLibMultiThread()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        std::lock_guard<std::mutex> lk1(m_mutex_result);
        m_finish = true;
    }
    m_cv_pending.notify_one();
    m_cv_result.notify_one();
    m_thread.join();
}

void RansEncoderLibMultiThread::flush()
{
    PendingTask p;
    p.workType = WorkType::Flush;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push_back(p);
    }
    m_cv_pending.notify_one();
}

std::shared_ptr<std::vector<uint8_t>> RansEncoderLibMultiThread::get_encoded_stream()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    return RansEncoderLib::get_encoded_stream();
}

void RansEncoderLibMultiThread::reset()
{
    RansEncoderLib::reset();
    std::lock_guard<std::mutex> lk(m_mutex_result);
    m_result_ready = false;
}

void RansEncoderLibMultiThread::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return m_pending.size() > 0 || m_finish; });
        if (m_finish) {
            lk.unlock();
            break;
        }
        if (m_pending.size() == 0) {
            lk.unlock();
            // std::cout << "contine in worker" << std::endl;
            continue;
        }
        while (m_pending.size() > 0) {
            auto p = m_pending.front();
            m_pending.pop_front();
            lk.unlock();
            if (p.workType == WorkType::Flush) {
                RansEncoderLib::flush();
                {
                    std::lock_guard<std::mutex> lk_result(m_mutex_result);
                    m_result_ready = true;
                }
                m_cv_result.notify_one();
            }
            lk.lock();
        }
        lk.unlock();
    }
}

void RansDecoderLib::set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded)
{
    _stream = encoded;
    _ptr8 = (uint8_t*)(_stream->data());
    RansDecInit(_rans, _ptr8);
}

int RansDecoderLib::add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                            const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                            const std::shared_ptr<std::vector<int32_t>> offsets)
{
    _cdfs.push_back(cdfs);
    _cdfs_sizes.push_back(cdfs_sizes);
    _offsets.push_back(offsets);
    return static_cast<int>(_cdfs.size()) - 1;
}

void RansDecoderLib::empty_cdf_buffer()
{
    _cdfs.clear();
    _cdfs_sizes.clear();
    _offsets.clear();
}

FORCE_INLINE int8_t RansDecoderLib::decode_one_symbol(const int32_t* cdf, const int32_t cdf_size,
                                                      const int32_t offset)
{
    const int32_t max_value = cdf_size - 2;
    const int32_t cum_freq = static_cast<int32_t>(RansDecGet(_rans));

    int s = 1;
    while (cdf[s++] <= cum_freq) {
    }
    s -= 2;

    RansDecAdvance(_rans, _ptr8, cdf[s], cdf[s + 1] - cdf[s]);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
        /* Bypass decoding mode */
        int32_t val = RansDecGetBits(_rans, _ptr8);
        int32_t n_bypass = val;

        while (val == max_bypass_val) {
            val = RansDecGetBits(_rans, _ptr8);
            n_bypass += val;
        }

        int32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
            val = RansDecGetBits(_rans, _ptr8);
            raw_val |= val << (j * bypass_precision);
        }
        value = raw_val >> 1;
        if (raw_val & 1) {
            value = -value - 1;
        } else {
            value += max_value;
        }
    }

    return static_cast<int8_t>(value + offset);
}

void RansDecoderLib::decode_y(const std::shared_ptr<std::vector<uint8_t>> indexes,
                              const int cdf_group_index)
{
    int index_size = static_cast<int>(indexes->size());
    m_decoded = std::make_shared<std::vector<int8_t>>(index_size);

    int8_t* outout_ptr = m_decoded->data();
    const uint8_t* indexes_ptr = indexes->data();
    const int32_t* cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
    const int32_t* offsets_ptr = _offsets[cdf_group_index]->data();
    const auto& cdfs = _cdfs[cdf_group_index];
    for (int i = 0; i < index_size; ++i) {
        const int32_t cdf_idx = indexes_ptr[i];
        outout_ptr[i] = decode_one_symbol(cdfs->at(cdf_idx).data(), cdfs_sizes_ptr[cdf_idx],
                                          offsets_ptr[cdf_idx]);
    }
}

void RansDecoderLib::decode_z(const int total_size, const int cdf_group_index,
                              const int start_offset, const int per_channel_size)
{
    m_decoded = std::make_shared<std::vector<int8_t>>(total_size);

    int8_t* outout_ptr = m_decoded->data();
    const int32_t* cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
    const int32_t* offsets_ptr = _offsets[cdf_group_index]->data();
    const auto& cdfs = _cdfs[cdf_group_index];
    for (int i = 0; i < total_size; ++i) {
        const int32_t cdf_idx = i / per_channel_size + start_offset;
        outout_ptr[i] = decode_one_symbol(cdfs->at(cdf_idx).data(), cdfs_sizes_ptr[cdf_idx],
                                          offsets_ptr[cdf_idx]);
    }
}

std::shared_ptr<std::vector<int8_t>> RansDecoderLib::get_decoded_tensor()
{
    return m_decoded;
}

RansDecoderLibMultiThread::RansDecoderLibMultiThread()
    : RansDecoderLib()
    , m_finish(false)
    , m_result_ready(false)
{
    m_thread = std::thread(&RansDecoderLibMultiThread::worker, this);
}

RansDecoderLibMultiThread::~RansDecoderLibMultiThread()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        std::lock_guard<std::mutex> lk1(m_mutex_result);
        m_finish = true;
    }
    m_cv_pending.notify_one();
    m_cv_result.notify_one();
    m_thread.join();
}

void RansDecoderLibMultiThread::decode_y(const std::shared_ptr<std::vector<uint8_t>> indexes,
                                         const int cdf_group_index)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.indexes = indexes;
    p.cdf_group_index = cdf_group_index;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push_back(p);
    }
    m_cv_pending.notify_one();
}

void RansDecoderLibMultiThread::decode_z(const int total_size, const int cdf_group_index,
                                         const int start_offset, const int per_channel_size)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.total_size = total_size;
    p.cdf_group_index = cdf_group_index;
    p.start_offset = start_offset;
    p.per_channel_size = per_channel_size;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push_back(p);
    }
    m_cv_pending.notify_one();
}

std::shared_ptr<std::vector<int8_t>> RansDecoderLibMultiThread::get_decoded_tensor()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    return RansDecoderLib::get_decoded_tensor();
}

void RansDecoderLibMultiThread::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return m_pending.size() > 0 || m_finish; });
        if (m_finish) {
            lk.unlock();
            break;
        }
        if (m_pending.size() == 0) {
            lk.unlock();
            // std::cout << "contine in worker" << std::endl;
            continue;
        }
        while (m_pending.size() > 0) {
            auto p = m_pending.front();
            m_pending.pop_front();
            lk.unlock();
            if (p.workType == WorkType::EncodeDecodeY) {
                RansDecoderLib::decode_y(p.indexes, p.cdf_group_index);
            } else if (p.workType == WorkType::EncodeDecodeZ) {
                RansDecoderLib::decode_z(p.total_size, p.cdf_group_index, p.start_offset,
                                         p.per_channel_size);
            }
            {
                std::lock_guard<std::mutex> lk_result(m_mutex_result);
                m_result_ready = true;
            }
            m_cv_result.notify_one();
            lk.lock();
        }
        lk.unlock();
    }
}
