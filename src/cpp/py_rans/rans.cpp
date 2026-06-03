// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "rans.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

constexpr int SCALE_BITS = 16;
constexpr int RANS_SHIFT_BITS = 23;
constexpr uint32_t RANS_BYTE_L = 1u << RANS_SHIFT_BITS;
constexpr int ENC_RENORM_SHIFT_BITS = RANS_SHIFT_BITS - SCALE_BITS + 8;
constexpr uint32_t DEC_MASK = (1u << SCALE_BITS) - 1;
constexpr uint16_t BYPASS_PRECISION = 2;  // number of bits in bypass mode
constexpr uint16_t MAX_BYPASS_VAL = (1 << BYPASS_PRECISION) - 1;

#if defined(__GNUC__) || defined(__clang__)
    #define FORCE_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
    #define FORCE_INLINE __forceinline
#else
    #define FORCE_INLINE inline
#endif

FORCE_INLINE void RansDecAdvance(RansState& r, const uint8_t*& ptr, uint32_t start, uint32_t freq)
{
    r = freq * (r >> SCALE_BITS) + (r & DEC_MASK) - start;

    // renormalize
    while (r < RANS_BYTE_L) {
        r = (r << 8) | *ptr++;
    }
}

FORCE_INLINE int32_t RansDecGet(RansState& r)
{
    return r & DEC_MASK;
}

FORCE_INLINE uint32_t RansDecGetBits(RansState& r, const uint8_t*& ptr)
{
    uint32_t val = r & ((1u << BYPASS_PRECISION) - 1);

    // renormalize
    r = r >> BYPASS_PRECISION;
    if (r < RANS_BYTE_L) {
        r = (r << 8) | *ptr++;
        assert(r >= RANS_BYTE_L);
    }

    return val;
}

FORCE_INLINE void RansDecInit(RansState& r, const uint8_t*& ptr)
{
    r = (*ptr++) << 0;
    r |= (*ptr++) << 8;
    r |= (*ptr++) << 16;
    r |= (*ptr++) << 24;
}

FORCE_INLINE void RansEncFlush(const RansState& r, uint8_t*& ptr)
{
    ptr -= 4;
    ptr[0] = static_cast<uint8_t>(r >> 0);
    ptr[1] = static_cast<uint8_t>(r >> 8);
    ptr[2] = static_cast<uint8_t>(r >> 16);
    ptr[3] = static_cast<uint8_t>(r >> 24);
}

FORCE_INLINE void RansEncInit(RansState& r)
{
    r = RANS_BYTE_L;
}

FORCE_INLINE void RansEncPut(RansState& r, uint8_t*& ptr, uint32_t start, uint32_t freq)
{
    // renormalize
    const uint32_t r_max = freq << ENC_RENORM_SHIFT_BITS;
    while (r >= r_max) {
        // converting to uint8_t will only keep the lowest 8 bits, equal to r & 0xff
        *(--ptr) = static_cast<uint8_t>(r);
        r >>= 8;
    }

    r = ((r / freq) << SCALE_BITS) + (r % freq) + start;
}

FORCE_INLINE void RansEncPutBits(RansState& r, uint8_t*& ptr, uint32_t val)
{
    static_assert(BYPASS_PRECISION <= 8);
    assert(val < (1u << BYPASS_PRECISION));

    constexpr uint32_t freq = 1 << (SCALE_BITS - BYPASS_PRECISION);
    constexpr uint32_t x_max = freq << ENC_RENORM_SHIFT_BITS;
    while (r >= x_max) {
        *(--ptr) = static_cast<uint8_t>(r);
        r >>= 8;
    }

    r = (r << BYPASS_PRECISION) | val;
}

FORCE_INLINE int8_t decode_one_symbol(const uint8_t*& ptr8, RansState& rans, const int32_t* cdf,
                                      const int8_t max_value)
{
    const int32_t cum_freq = RansDecGet(rans);

    int s = 1;
    while (cdf[s] <= cum_freq) {
        s++;
    }
    s--;

    RansDecAdvance(rans, ptr8, cdf[s], cdf[s + 1] - cdf[s]);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
        // Bypass decoding mode
        int32_t val = RansDecGetBits(rans, ptr8);
        int32_t n_bypass = val;

        while (val == MAX_BYPASS_VAL) {
            val = RansDecGetBits(rans, ptr8);
            n_bypass += val;
        }

        int32_t raw_val = 0;
        for (int j = 0; j < n_bypass; ++j) {
            val = RansDecGetBits(rans, ptr8);
            raw_val |= val << (j * BYPASS_PRECISION);
        }
        value = raw_val + max_value;
    }

    return static_cast<int8_t>((value % 2 == 1) ? (value + 1) / 2 : -(value + 1) / 2);
}

FORCE_INLINE void encode_one_symbol(uint8_t*& ptr, RansState& rans, const int32_t symbol,
                                    const int8_t max_value, const std::vector<RansSymbol>& ransSymbols)
{
    int32_t value = abs(symbol) * 2 - (symbol > 0);

    if (value >= max_value) {
        const uint32_t raw_val = value - max_value;
        value = max_value;

        std::vector<uint16_t> bypassBins;
        bypassBins.reserve(20);
        // Determine the number of bypasses (in BYPASS_PRECISION size)
        // needed to encode the raw value.
        int32_t n_bypass = 0;
        while ((raw_val >> (n_bypass * BYPASS_PRECISION)) != 0) {
            ++n_bypass;
        }

        // Encode number of bypasses
        int32_t val = n_bypass;
        while (val >= MAX_BYPASS_VAL) {
            bypassBins.push_back(MAX_BYPASS_VAL);
            val -= MAX_BYPASS_VAL;
        }
        bypassBins.push_back(static_cast<uint16_t>(val));

        // Encode raw value
        for (int32_t j = 0; j < n_bypass; ++j) {
            const int32_t val1 = (raw_val >> (j * BYPASS_PRECISION)) & MAX_BYPASS_VAL;
            bypassBins.push_back(static_cast<uint16_t>(val1));
        }

        for (auto it = bypassBins.rbegin(); it < bypassBins.rend(); it++) {
            RansEncPutBits(rans, ptr, *it);
        }
    }
    RansEncPut(rans, ptr, ransSymbols[value].start, ransSymbols[value].range);
}

RansEncoderLib::RansEncoderLib()
{
    m_ransSymbols.resize(2);
    m_max_value.resize(2);
    m_stream_buffer = new uint8_t[max_stream_buffer_size];
    m_stream = std::make_shared<std::vector<uint8_t>>();
    m_thread = std::thread(&RansEncoderLib::worker, this);
}

RansEncoderLib::~RansEncoderLib()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        m_finish = true;
    }
    m_cv_pending.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }

    if (m_stream_buffer != nullptr) {
        delete[] m_stream_buffer;
        m_stream_buffer = nullptr;
    }
}

void RansEncoderLib::check_buffer_capacity(int symbol_count)
{
    // Conservative upper bound: each symbol can emit up to 4 bytes (3 bytes renorm + overhead).
    // The flush at the end writes 4 bytes for the final state.
    const int required_bytes = symbol_count * 4 + 4;
    const int remaining_bytes = static_cast<int>(m_ptr - m_stream_buffer);
    if (remaining_bytes < required_bytes) {
        throw std::runtime_error("rANS buffer overflow: encoding " + std::to_string(symbol_count)
                                 + " symbols requires up to " + std::to_string(required_bytes)
                                 + " bytes, but only " + std::to_string(remaining_bytes)
                                 + " bytes remain in the buffer (capacity: "
                                 + std::to_string(max_stream_buffer_size) + ")");
    }
}

void RansEncoderLib::encode_y(const std::shared_ptr<std::vector<int16_t>>& symbols,
                              const int symbol_size, const int symbol_offset)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.symbols_y = symbols;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_y_internal(const std::shared_ptr<std::vector<int16_t>>& symbols,
                                       const int symbol_size, const int symbol_offset)
{
    check_buffer_capacity(symbol_size);

    // backward loop on symbols from the end;
    const int16_t* symbols_ptr = symbols->data();
    const int8_t* max_value_ptr = m_max_value[1]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = m_ransSymbols[1]->data();
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; i--) {
        const int16_t combined_symbol = symbols_ptr[i];
        const int32_t cdf_idx = combined_symbol & 0xff;
        const int32_t s = static_cast<int8_t>(combined_symbol >> 8);
        encode_one_symbol(m_ptr, m_rans, s, max_value_ptr[cdf_idx], ransSymbols_ptr[cdf_idx]);
    }
}

void RansEncoderLib::encode_z(const std::shared_ptr<std::vector<int8_t>>& symbols, const int symbol_size,
                              const int symbol_offset, const int cdf_offset, const int ch)
{
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.symbols_z = symbols;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.cdf_offset = cdf_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansEncoderLib::encode_z_internal(const std::shared_ptr<std::vector<int8_t>>& symbols,
                                       const int symbol_size, const int symbol_offset,
                                       const int cdf_offset, const int ch)
{
    check_buffer_capacity(symbol_size);

    // backward loop on symbols from the end;
    const int8_t* symbols_ptr = symbols->data();
    const int8_t* max_value_ptr = m_max_value[0]->data();
    const std::vector<RansSymbol>* ransSymbols_ptr = m_ransSymbols[0]->data();
    const int symbol_start = symbol_offset;
    const int symbol_end = symbol_offset + symbol_size - 1;

    for (int i = symbol_end; i >= symbol_start; i--) {
        const int32_t cdf_idx = (i % ch) + cdf_offset;
        encode_one_symbol(m_ptr, m_rans, symbols_ptr[i], max_value_ptr[cdf_idx],
                          ransSymbols_ptr[cdf_idx]);
    }
}

void RansEncoderLib::flush()
{
    PendingTask p;
    p.workType = WorkType::Flush;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

std::shared_ptr<std::vector<uint8_t>> RansEncoderLib::get_encoded_stream()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    return m_stream;
}

void RansEncoderLib::reset()
{
    m_stream->clear();

    RansEncInit(m_rans);

    uint8_t* ptrEnd = m_stream_buffer + max_stream_buffer_size;
    m_ptr = ptrEnd;

    std::lock_guard<std::mutex> lk(m_mutex_result);
    m_result_ready = false;
}

void RansEncoderLib::set_cdf(const std::shared_ptr<std::vector<std::vector<RansSymbol>>>& ransSymbols,
                             const std::shared_ptr<std::vector<int8_t>>& max_value, const int index)
{
    assert(index < 2);
    m_ransSymbols[index] = ransSymbols;
    m_max_value[index] = max_value;
}

void RansEncoderLib::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return !m_pending.empty() || m_finish; });
        if (m_finish) {
            break;
        }
        while (!m_pending.empty()) {
            auto p = std::move(m_pending.front());
            m_pending.pop();
            lk.unlock();
            if (p.workType == WorkType::EncodeDecodeY) {
                RansEncoderLib::encode_y_internal(p.symbols_y, p.symbol_size, p.symbol_offset);
            } else if (p.workType == WorkType::EncodeDecodeZ) {
                RansEncoderLib::encode_z_internal(p.symbols_z, p.symbol_size, p.symbol_offset,
                                                  p.cdf_offset, p.ch);
            } else if (p.workType == WorkType::Flush) {
                RansEncFlush(m_rans, m_ptr);

                uint8_t* ptrEnd = m_stream_buffer + max_stream_buffer_size;
                const int nbytes = static_cast<int>(std::distance(m_ptr, ptrEnd));

                if (m_ptr < m_stream_buffer || nbytes > max_stream_buffer_size) {
                    throw std::runtime_error("rANS stream buffer overflow: encoded size ("
                                             + std::to_string(nbytes) + ") exceeds buffer capacity ("
                                             + std::to_string(max_stream_buffer_size) + ")");
                }

                m_stream->resize(nbytes);
                std::copy(m_ptr, m_ptr + nbytes, m_stream->data());
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

RansDecoderLib::RansDecoderLib()
{
    m_cdfs.resize(2);
    m_max_value.resize(2);
    m_thread = std::thread(&RansDecoderLib::worker, this);
}

RansDecoderLib::~RansDecoderLib()
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_pending);
        m_finish = true;
    }
    m_cv_pending.notify_all();
    if (m_thread.joinable()) {
        m_thread.join();
    }
}

void RansDecoderLib::decode_y(int8_t* decoded_ptr, const std::shared_ptr<std::vector<uint8_t>>& indexes,
                              const int symbol_size, const int symbol_offset)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeY;
    p.indexes = indexes;
    p.decoded_ptr = decoded_ptr;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_y_internal(int8_t* decoded_ptr,
                                       const std::shared_ptr<std::vector<uint8_t>>& indexes,
                                       const int symbol_size, const int symbol_offset)
{
    const uint8_t* indexes_ptr = indexes->data();
    const int8_t* max_value_ptr = m_max_value[1]->data();
    const std::vector<int32_t>* cdfs = m_cdfs[1]->data();
    for (int i = 0; i < symbol_size; ++i) {
        const int32_t cdf_idx = indexes_ptr[i + symbol_offset];
        decoded_ptr[i + symbol_offset] =
            decode_one_symbol(m_ptr8, m_rans, cdfs[cdf_idx].data(), max_value_ptr[cdf_idx]);
    }
}

void RansDecoderLib::decode_z(int8_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                              const int cdf_offset, const int ch)
{
    {
        std::lock_guard<std::mutex> lk(m_mutex_result);
        m_result_ready = false;
    }
    PendingTask p;
    p.workType = WorkType::EncodeDecodeZ;
    p.decoded_ptr = decoded_ptr;
    p.symbol_size = symbol_size;
    p.symbol_offset = symbol_offset;
    p.cdf_offset = cdf_offset;
    p.ch = ch;
    {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_pending.push(std::move(p));
    }
    m_cv_pending.notify_one();
}

void RansDecoderLib::decode_z_internal(int8_t* decoded_ptr, const int symbol_size,
                                       const int symbol_offset, const int cdf_offset, const int ch)
{
    const int8_t* max_value_ptr = m_max_value[0]->data();
    const std::vector<int32_t>* cdfs = m_cdfs[0]->data();

    for (int i = 0; i < symbol_size; ++i) {
        const int32_t cdf_idx = ((i + symbol_offset) % ch) + cdf_offset;
        decoded_ptr[i + symbol_offset] =
            decode_one_symbol(m_ptr8, m_rans, cdfs[cdf_idx].data(), max_value_ptr[cdf_idx]);
    }
}

void RansDecoderLib::set_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
                             const std::shared_ptr<std::vector<int8_t>>& max_value, const int index)
{
    assert(index < 2);
    m_cdfs[index] = cdfs;
    m_max_value[index] = max_value;
}

void RansDecoderLib::set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded)
{
    m_stream = encoded;
    m_ptr8 = m_stream->data();
    RansDecInit(m_rans, m_ptr8);
}

bool RansDecoderLib::wait_for_decoding_finish()
{
    std::unique_lock<std::mutex> lk(m_mutex_result);
    m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
    return true;
}

void RansDecoderLib::worker()
{
    while (!m_finish) {
        std::unique_lock<std::mutex> lk(m_mutex_pending);
        m_cv_pending.wait(lk, [this] { return !m_pending.empty() || m_finish; });
        if (m_finish) {
            break;
        }
        while (!m_pending.empty()) {
            auto p = std::move(m_pending.front());
            m_pending.pop();
            lk.unlock();
            if (p.workType == WorkType::EncodeDecodeY) {
                decode_y_internal(p.decoded_ptr, p.indexes, p.symbol_size, p.symbol_offset);
            } else if (p.workType == WorkType::EncodeDecodeZ) {
                decode_z_internal(p.decoded_ptr, p.symbol_size, p.symbol_offset, p.cdf_offset, p.ch);
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
