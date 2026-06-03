// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <condition_variable>
#include <queue>
#include <thread>
#include <vector>

struct RansSymbol {
    uint16_t start;
    uint16_t range;
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
    int8_t* decoded_ptr{ nullptr };
    int total_size{ 0 };
    int cdf_offset{ 0 };
    int symbol_size{ 0 };
    int symbol_offset{ 0 };
    int ch{ 0 };

    PendingTask() = default;
    PendingTask(PendingTask&&) noexcept = default;
    PendingTask& operator=(PendingTask&&) noexcept = default;

    PendingTask(const PendingTask&) = delete;
    PendingTask& operator=(const PendingTask&) = delete;
};

using RansState = uint32_t;

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class RansEncoderLib {
public:
    RansEncoderLib();
    virtual ~RansEncoderLib();

    RansEncoderLib(const RansEncoderLib&) = delete;
    RansEncoderLib(RansEncoderLib&&) = delete;
    RansEncoderLib& operator=(const RansEncoderLib&) = delete;
    RansEncoderLib& operator=(RansEncoderLib&&) = delete;

    void check_buffer_capacity(int symbol_count);
    void encode_y(const std::shared_ptr<std::vector<int16_t>>& symbols, const int symbol_size,
                  const int symbol_offset);
    void encode_y_internal(const std::shared_ptr<std::vector<int16_t>>& symbols,
                           const int symbol_size, const int symbol_offset);
    void encode_z(const std::shared_ptr<std::vector<int8_t>>& symbols, const int symbol_size,
                  const int symbol_offset, const int cdf_offset, const int ch);
    void encode_z_internal(const std::shared_ptr<std::vector<int8_t>>& symbols, const int symbol_size,
                           const int symbol_offset, const int cdf_offset, const int ch);
    void flush();
    std::shared_ptr<std::vector<uint8_t>> get_encoded_stream();
    void reset();
    void set_cdf(const std::shared_ptr<std::vector<std::vector<RansSymbol>>>& ransSymbols,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);
    void worker();

private:
    static const int max_stream_buffer_size = 10 * 1000 * 1000;
    uint8_t* m_stream_buffer{ nullptr };
    RansState m_rans;
    uint8_t* m_ptr{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> m_stream;

    std::vector<std::shared_ptr<std::vector<std::vector<RansSymbol>>>> m_ransSymbols;
    std::vector<std::shared_ptr<std::vector<int8_t>>> m_max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};

class RansDecoderLib {
public:
    RansDecoderLib();
    virtual ~RansDecoderLib();

    RansDecoderLib(const RansDecoderLib&) = delete;
    RansDecoderLib(RansDecoderLib&&) = delete;
    RansDecoderLib& operator=(const RansDecoderLib&) = delete;
    RansDecoderLib& operator=(RansDecoderLib&&) = delete;

    void decode_y(int8_t* decoded_ptr, const std::shared_ptr<std::vector<uint8_t>>& indexes,
                  const int symbol_size, const int symbol_offset);
    void decode_y_internal(int8_t* decoded_ptr, const std::shared_ptr<std::vector<uint8_t>>& indexes,
                           const int symbol_size, const int symbol_offset);
    void decode_z(int8_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                  const int cdf_offset, const int ch);
    void decode_z_internal(int8_t* decoded_ptr, const int symbol_size, const int symbol_offset,
                           const int cdf_offset, const int ch);
    void set_cdf(const std::shared_ptr<std::vector<std::vector<int32_t>>>& cdfs,
                 const std::shared_ptr<std::vector<int8_t>>& max_value, const int index);
    void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);
    bool wait_for_decoding_finish();
    void worker();

private:
    RansState m_rans;
    const uint8_t* m_ptr8{ nullptr };
    std::shared_ptr<std::vector<uint8_t>> m_stream;

    std::vector<std::shared_ptr<std::vector<std::vector<int32_t>>>> m_cdfs;
    std::vector<std::shared_ptr<std::vector<int8_t>>> m_max_value;

    bool m_finish{ false };
    bool m_result_ready{ false };
    std::thread m_thread;
    std::mutex m_mutex_result;
    std::mutex m_mutex_pending;
    std::condition_variable m_cv_pending;
    std::condition_variable m_cv_result;
    std::queue<PendingTask> m_pending;
};
