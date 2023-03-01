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

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} // namespace

void RansEncoderLib::encode_with_indexes(
    const std::shared_ptr<std::vector<int16_t>> symbols,
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {

  // backward loop on symbols from the end;
  const int16_t *symbols_ptr = symbols->data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = cdfs_sizes->data();
  const int32_t *offsets_ptr = offsets->data();
  const int symbol_size = static_cast<int>(symbols->size());
  _syms.reserve(symbol_size * 3 / 2);
  for (int i = 0; i < symbol_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    if (cdf_idx < 0) {
      continue;
    }
    const int32_t *cdf = cdfs->at(cdf_idx).data();
    const int32_t max_value = cdfs_sizes_ptr[cdf_idx] - 2;
    int32_t value = symbols_ptr[i] - offsets_ptr[cdf_idx];

    uint32_t raw_val = 0;
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});

    /* Bypass coding mode (value == max_value -> sentinel flag) */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val1 =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back({static_cast<uint16_t>(val1),
                         static_cast<uint16_t>(val1 + 1), true});
      }
    }
  }
}

void RansEncoderLib::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size()); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes = static_cast<int>(
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t));

  _stream.resize(nbytes);
  memcpy(_stream.data(), ptr, nbytes);
}

std::vector<uint8_t> RansEncoderLib::get_encoded_stream() { return _stream; }

void RansEncoderLib::reset() { _syms.clear(); }

RansEncoderLibMultiThread::RansEncoderLibMultiThread()
    : RansEncoderLib(), m_finish(false), m_result_ready(false),
      m_thread(std::thread(&RansEncoderLibMultiThread::worker, this)) {}

RansEncoderLibMultiThread::~RansEncoderLibMultiThread() {
  {
    std::lock_guard<std::mutex> lk(m_mutex_pending);
    std::lock_guard<std::mutex> lk1(m_mutex_result);
    m_finish = true;
  }
  m_cv_pending.notify_one();
  m_cv_result.notify_one();
  m_thread.join();
}

void RansEncoderLibMultiThread::encode_with_indexes(
    const std::shared_ptr<std::vector<int16_t>> symbols,
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {
  PendingTask p;
  p.workType = WorkType::Encode;
  p.symbols = symbols;
  p.indexes = indexes;
  p.cdfs = cdfs;
  p.cdfs_sizes = cdfs_sizes;
  p.offsets = offsets;
  {
    std::unique_lock<std::mutex> lk(m_mutex_pending);
    m_pending.push_back(p);
  }
  m_cv_pending.notify_one();
}

void RansEncoderLibMultiThread::flush() {
  PendingTask p;
  p.workType = WorkType::Flush;
  {
    std::unique_lock<std::mutex> lk(m_mutex_pending);
    m_pending.push_back(p);
  }
  m_cv_pending.notify_one();
}

std::vector<uint8_t> RansEncoderLibMultiThread::get_encoded_stream() {
  std::unique_lock<std::mutex> lk(m_mutex_result);
  m_cv_result.wait(lk, [this] { return m_result_ready || m_finish; });
  return RansEncoderLib::get_encoded_stream();
}

void RansEncoderLibMultiThread::reset() {
  RansEncoderLib::reset();
  std::lock_guard<std::mutex> lk(m_mutex_result);
  m_result_ready = false;
}

void RansEncoderLibMultiThread::worker() {
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
      if (p.workType == WorkType::Encode) {
        RansEncoderLib::encode_with_indexes(p.symbols, p.indexes, p.cdfs,
                                            p.cdfs_sizes, p.offsets);
      } else if (p.workType == WorkType::Flush) {
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

void RansDecoderLib::set_stream(
    const std::shared_ptr<std::vector<uint8_t>> encoded) {
  _stream = encoded;
  _ptr = (uint32_t *)(_stream->data());
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int16_t> RansDecoderLib::decode_stream(
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {
  int index_size = static_cast<int>(indexes->size());
  std::vector<int16_t> output(index_size);

  int16_t *outout_ptr = output.data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = cdfs_sizes->data();
  const int32_t *offsets_ptr = offsets->data();
  for (int i = 0; i < index_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    const int32_t offset = offsets_ptr[cdf_idx];
    if (cdf_idx < 0) {
      outout_ptr[i] = static_cast<int16_t>(offset);
      continue;
    }
    const int32_t *cdf = cdfs->at(cdf_idx).data();
    const int32_t max_value = cdfs_sizes_ptr[cdf_idx] - 2;
    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf + cdfs_sizes_ptr[cdf_idx];
    const auto it = std::find_if(cdf, cdf_end, [cum_freq](int v) {
      return static_cast<uint32_t>(v) > cum_freq;
    });
    const uint32_t s = static_cast<uint32_t>(std::distance(cdf, it) - 1);

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    outout_ptr[i] = static_cast<int16_t>(value + offset);
  }
  return output;
}
