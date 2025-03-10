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

#include <condition_variable>
#include <list>
#include <thread>
#include <vector>

#ifdef __GNUC__
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wpedantic"
    #pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#ifdef _MSC_VER
    #pragma warning(disable : 4244)
#endif

#include "rans_byte.h"

#ifdef _MSC_VER
    #pragma warning(default : 4244)
#endif

#ifdef __GNUC__
    #pragma GCC diagnostic pop
#endif

#ifdef _MSC_VER
    #define FORCE_INLINE __forceinline
#endif

#ifdef __GNUC__
    #define FORCE_INLINE __attribute__((always_inline)) inline
#endif

struct RansSymbol {
    uint16_t start;
    uint16_t range;  // range for normal coding and 0 for bypass coding
};

enum class WorkType {
    EncodeDecodeY,
    EncodeDecodeZ,
    Flush,
};

struct PendingTask {
    WorkType workType;
    std::shared_ptr<std::vector<int16_t>> symbols_y;
    std::shared_ptr<std::vector<int8_t>> symbols_z;
    std::shared_ptr<std::vector<uint8_t>> indexes;
    int total_size{ 0 };
    int cdf_group_index{ 0 };
    int start_offset{ 0 };
    int per_channel_size{ 0 };
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class RansEncoderLib {
public:
    RansEncoderLib();
    virtual ~RansEncoderLib() = default;

    RansEncoderLib(const RansEncoderLib&) = delete;
    RansEncoderLib(RansEncoderLib&&) = delete;
    RansEncoderLib& operator=(const RansEncoderLib&) = delete;
    RansEncoderLib& operator=(RansEncoderLib&&) = delete;

    void encode_y(const std::shared_ptr<std::vector<int16_t>> symbols, const int cdf_group_index);
    void encode_z(const std::shared_ptr<std::vector<int8_t>> symbols, const int cdf_group_index,
                  const int start_offset, const int per_channel_size);

    FORCE_INLINE void encode_y_internal(uint8_t*& ptr, RansState& rans,
                                        const std::shared_ptr<std::vector<int16_t>> symbols,
                                        const int cdf_group_index);
    FORCE_INLINE void encode_z_internal(uint8_t*& ptr, RansState& rans,
                                        const std::shared_ptr<std::vector<int8_t>> symbols,
                                        const int cdf_group_index, const int start_offset,
                                        const int per_channel_size);
    FORCE_INLINE void encode_one_symbol(uint8_t*& ptr, RansState& rans, const int32_t symbol,
                                        const int32_t cdf_size, const int32_t offset,
                                        const std::vector<RansSymbol>& ransSymbols);
    virtual void flush();
    virtual std::shared_ptr<std::vector<uint8_t>> get_encoded_stream();
    virtual void reset();
    virtual int add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                        const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                        const std::shared_ptr<std::vector<int32_t>> offsets);
    virtual void empty_cdf_buffer();

private:
    std::shared_ptr<std::vector<uint8_t>> _stream;

    std::vector<std::shared_ptr<std::vector<std::vector<RansSymbol>>>> _ransSymbols;
    std::vector<std::shared_ptr<std::vector<int32_t>>> _cdfs_sizes;
    std::vector<std::shared_ptr<std::vector<int32_t>>> _offsets;

    std::list<PendingTask> m_pendingEncodingList;
};

class RansEncoderLibMultiThread : public RansEncoderLib {
public:
    RansEncoderLibMultiThread();
    virtual ~RansEncoderLibMultiThread();
    virtual void flush() override;
    virtual std::shared_ptr<std::vector<uint8_t>> get_encoded_stream() override;
    virtual void reset() override;

    void worker();

private:
    bool m_finish;
    bool m_result_ready;
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::list<PendingTask> m_pending;
};

class RansDecoderLib {
public:
    RansDecoderLib() {}
    virtual ~RansDecoderLib() = default;

    RansDecoderLib(const RansDecoderLib&) = delete;
    RansDecoderLib(RansDecoderLib&&) = delete;
    RansDecoderLib& operator=(const RansDecoderLib&) = delete;
    RansDecoderLib& operator=(RansDecoderLib&&) = delete;

    virtual void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);

    FORCE_INLINE int8_t decode_one_symbol(const int32_t* cdf, const int32_t cdf_size,
                                          const int32_t offset);

    virtual void decode_y(const std::shared_ptr<std::vector<uint8_t>> indexes,
                          const int cdf_group_index);
    virtual void decode_z(const int total_size, const int cdf_group_index, const int start_offset,
                          const int per_channel_size);

    virtual std::shared_ptr<std::vector<int8_t>> get_decoded_tensor();

    virtual int add_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                        const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                        const std::shared_ptr<std::vector<int32_t>> offsets);
    virtual void empty_cdf_buffer();

private:
    RansState _rans;
    uint8_t* _ptr8;
    std::shared_ptr<std::vector<uint8_t>> _stream;
    std::shared_ptr<std::vector<int8_t>> m_decoded;

    std::vector<std::shared_ptr<std::vector<std::vector<int32_t>>>> _cdfs;
    std::vector<std::shared_ptr<std::vector<int32_t>>> _cdfs_sizes;
    std::vector<std::shared_ptr<std::vector<int32_t>>> _offsets;
};

class RansDecoderLibMultiThread : public RansDecoderLib {
public:
    RansDecoderLibMultiThread();
    virtual ~RansDecoderLibMultiThread();

    virtual void decode_y(const std::shared_ptr<std::vector<uint8_t>> indexes,
                          const int cdf_group_index) override;

    virtual void decode_z(const int total_size, const int cdf_group_index, const int start_offset,
                          const int per_channel_size) override;

    virtual std::shared_ptr<std::vector<int8_t>> get_decoded_tensor() override;

    void worker();

private:
    bool m_finish;
    bool m_result_ready;
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::list<PendingTask> m_pending;
};