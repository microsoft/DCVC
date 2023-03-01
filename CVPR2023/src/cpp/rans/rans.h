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

#include <rans64.h>

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

struct RansSymbol {
  uint16_t start;
  uint16_t range;
  bool bypass; // bypass flag to write raw bits to the stream
};

enum class WorkType {
  Encode,
  Flush,
};

struct PendingTask {
  WorkType workType;
  std::shared_ptr<std::vector<int16_t>> symbols;
  std::shared_ptr<std::vector<int16_t>> indexes;
  std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs;
  std::shared_ptr<std::vector<int32_t>> cdfs_sizes;
  std::shared_ptr<std::vector<int32_t>> offsets;
};

/* NOTE: Warning, we buffer everything for now... In case of large files we
 * should split the bitstream into chunks... Or for a memory-bounded encoder
 **/
class RansEncoderLib {
public:
  RansEncoderLib() = default;
  virtual ~RansEncoderLib() = default;

  RansEncoderLib(const RansEncoderLib &) = delete;
  RansEncoderLib(RansEncoderLib &&) = delete;
  RansEncoderLib &operator=(const RansEncoderLib &) = delete;
  RansEncoderLib &operator=(RansEncoderLib &&) = delete;

  virtual void encode_with_indexes(
      const std::shared_ptr<std::vector<int16_t>> symbols,
      const std::shared_ptr<std::vector<int16_t>> indexes,
      const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
      const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
      const std::shared_ptr<std::vector<int32_t>> offsets);
  virtual void flush();
  virtual std::vector<uint8_t> get_encoded_stream();
  virtual void reset();

private:
  std::vector<RansSymbol> _syms;
  std::vector<uint8_t> _stream;
};

class RansEncoderLibMultiThread : public RansEncoderLib {
public:
  RansEncoderLibMultiThread();
  virtual ~RansEncoderLibMultiThread();

  virtual void encode_with_indexes(
      const std::shared_ptr<std::vector<int16_t>> symbols,
      const std::shared_ptr<std::vector<int16_t>> indexes,
      const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
      const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
      const std::shared_ptr<std::vector<int32_t>> offsets) override;
  virtual void flush() override;
  virtual std::vector<uint8_t> get_encoded_stream() override;
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
  RansDecoderLib() = default;
  virtual ~RansDecoderLib() = default;

  RansDecoderLib(const RansDecoderLib &) = delete;
  RansDecoderLib(RansDecoderLib &&) = delete;
  RansDecoderLib &operator=(const RansDecoderLib &) = delete;
  RansDecoderLib &operator=(RansDecoderLib &&) = delete;

  void set_stream(const std::shared_ptr<std::vector<uint8_t>> encoded);

  virtual std::vector<int16_t>
  decode_stream(const std::shared_ptr<std::vector<int16_t>> indexes,
                const std::shared_ptr<std::vector<std::vector<int32_t>>> cdfs,
                const std::shared_ptr<std::vector<int32_t>> cdfs_sizes,
                const std::shared_ptr<std::vector<int32_t>> offsets);

private:
  Rans64State _rans;
  uint32_t *_ptr;
  std::shared_ptr<std::vector<uint8_t>> _stream;
};
