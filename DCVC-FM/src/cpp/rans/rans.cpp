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

constexpr uint16_t bypass_precision = 2; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;
namespace {

inline void RansEncPutBits(RansState *r, uint8_t **pptr, uint32_t val,
                           uint32_t nbits) {
  assert(nbits <= 8);
  assert(val < (1u << nbits));

  uint32_t x = *r;
  uint32_t freq = 1 << (precision - nbits);
  // uint32_t x_max =
  //((RANS_BYTE_L >> precision) << 8) * freq; // this turns into a shift.
  uint32_t x_max = freq << 15;
  while (x >= x_max) {
    *(--(*pptr)) = static_cast<uint8_t>(x & 0xff);
    x >>= 8;
  }

  *r = (x << nbits) | val;
}

inline uint32_t RansDecGetBits(RansState *r, uint8_t **pptr, uint32_t n_bits) {
  uint32_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS_BYTE_L) {
    x = (x << 8) | **pptr;
    *pptr += 1;
    RansAssert(x >= RANS_BYTE_L);
  }

  *r = x;

  return val;
}
} // namespace

int RansEncoderLib::add_cdf(
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {

  auto ransSymbols =
      std::make_shared<std::vector<std::vector<RansSymbol>>>(cdfs->size());
  for (int i = 0; i < static_cast<int>(cdfs->size()); i++) {
    const int32_t *cdf = cdfs->at(i).data();
    std::vector<RansSymbol> ransSym(cdfs->at(i).size());
    const int ransSize = static_cast<int>(ransSym.size() - 1);
    for (int j = 0; j < ransSize; j++) {
      ransSym[j] = RansSymbol({static_cast<uint16_t>(cdf[j]),
                               static_cast<uint16_t>(cdf[j + 1] - cdf[j])});
    }
    ransSymbols->at(i) = ransSym;
  }

  _ransSymbols.push_back(ransSymbols);
  _cdfs_sizes.push_back(cdfs_sizes);
  _offsets.push_back(offsets);
  return static_cast<int>(_ransSymbols.size()) - 1;
}

void RansEncoderLib::empty_cdf_buffer() {
  _ransSymbols.clear();
  _cdfs_sizes.clear();
  _offsets.clear();
}

void RansEncoderLib::encode_with_indexes(
    const std::shared_ptr<std::vector<int16_t>> symbols,
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const int cdf_group_index) {

  // backward loop on symbols from the end;
  const int16_t *symbols_ptr = symbols->data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
  const int32_t *offsets_ptr = _offsets[cdf_group_index]->data();
  const int symbol_size = static_cast<int>(symbols->size());
  _syms.reserve(symbol_size * 3 / 2);
  auto ransSymbols = _ransSymbols[cdf_group_index];

  for (int i = 0; i < symbol_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    if (cdf_idx < 0) {
      continue;
    }
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

    _syms.push_back(ransSymbols->at(cdf_idx)[value]);

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
        _syms.push_back({max_bypass_val, 0});
        val -= max_bypass_val;
      }
      _syms.push_back({static_cast<uint16_t>(val), 0});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val1 =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;
        _syms.push_back({static_cast<uint16_t>(val1), 0});
      }
    }
  }
}

void RansEncoderLib::flush() {
  RansState rans;
  RansEncInit(&rans);

  std::vector<uint8_t> output(_syms.size()); // too much space ?
  uint8_t *ptrEnd = output.data() + output.size();
  uint8_t *ptr = ptrEnd;
  assert(ptr != nullptr);

  for (auto it = _syms.rbegin(); it < _syms.rend(); it++) {
    const RansSymbol sym = *it;

    if (sym.range != 0) {
      RansEncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      RansEncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
  }

  RansEncFlush(&rans, &ptr);

  const int nbytes = static_cast<int>(std::distance(ptr, ptrEnd));

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
    const int cdf_group_index) {
  PendingTask p;
  p.workType = WorkType::Encode;
  p.symbols = symbols;
  p.indexes = indexes;
  p.cdf_group_index = cdf_group_index;
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
        RansEncoderLib::encode_with_indexes(p.symbols, p.indexes,
                                            p.cdf_group_index);
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
  _ptr8 = (uint8_t *)(_stream->data());
  RansDecInit(&_rans, &_ptr8);
}

int RansDecoderLib::add_cdf(
    const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
    const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
    const std::shared_ptr<std::vector<int32_t>> offsets) {
  _cdfs.push_back(cdfs);
  _cdfs_sizes.push_back(cdfs_sizes);
  _offsets.push_back(offsets);
  return static_cast<int>(_cdfs.size()) - 1;
}

void RansDecoderLib::empty_cdf_buffer() {
  _cdfs.clear();
  _cdfs_sizes.clear();
  _offsets.clear();
}

std::vector<int16_t> RansDecoderLib::decode_stream(
    const std::shared_ptr<std::vector<int16_t>> indexes,
    const int cdf_group_index) {

  int index_size = static_cast<int>(indexes->size());
  std::vector<int16_t> output(index_size);

  int16_t *outout_ptr = output.data();
  const int16_t *indexes_ptr = indexes->data();
  const int32_t *cdfs_sizes_ptr = _cdfs_sizes[cdf_group_index]->data();
  const int32_t *offsets_ptr = _offsets[cdf_group_index]->data();
  const auto &cdfs = _cdfs[cdf_group_index];
  for (int i = 0; i < index_size; ++i) {
    const int32_t cdf_idx = indexes_ptr[i];
    if (cdf_idx < 0) {
      outout_ptr[i] = 0;
      continue;
    }
    const int32_t *cdf = cdfs->at(cdf_idx).data();
    const int32_t max_value = cdfs_sizes_ptr[cdf_idx] - 2;
    const uint32_t cum_freq = RansDecGet(&_rans, precision);

    const auto cdf_end = cdf + cdfs_sizes_ptr[cdf_idx];
    const auto it = std::find_if(cdf, cdf_end, [cum_freq](int v) {
      return static_cast<uint32_t>(v) > cum_freq;
    });
    const uint32_t s = static_cast<uint32_t>(std::distance(cdf, it) - 1);

    RansDecAdvance(&_rans, &_ptr8, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = RansDecGetBits(&_rans, &_ptr8, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = RansDecGetBits(&_rans, &_ptr8, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = RansDecGetBits(&_rans, &_ptr8, bypass_precision);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    const int32_t offset = offsets_ptr[cdf_idx];
    outout_ptr[i] = static_cast<int16_t>(value + offset);
  }
  return output;
}
