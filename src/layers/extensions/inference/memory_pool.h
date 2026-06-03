// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/core/Device.h>
#include <c10/util/SmallVector.h>
#include <cuda_runtime.h>
#include <mutex>
#include <torch/extension.h>
#include <vector>

struct TensorEntry {
    at::Tensor tensor;
    c10::SmallVector<int64_t, 8> sizes;
    c10::Device device{ at::kCUDA, 0 };
    at::ScalarType dtype{ at::kFloat };
    c10::Layout layout{ c10::kStrided };
    bool is_channel_last{ true };
    bool reusable{ false };
};

inline bool sizes_equal(c10::IntArrayRef a, c10::IntArrayRef b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return false;
    return true;
}

class TensorPool {
public:
    TensorPool() = default;

    at::Tensor acquire(c10::IntArrayRef sizes, const at::TensorOptions& options)
    {
        std::lock_guard<std::mutex> g(mu_);
        const c10::Device device = options.device();
        const at::ScalarType dtype = c10::typeMetaToScalarType(options.dtype());
        const c10::Layout layout = options.layout();
        const std::optional<at::MemoryFormat> mem_format = options.memory_format_opt();
        bool is_channel_last =
            mem_format.has_value() && mem_format.value() == at::MemoryFormat::ChannelsLast;

        for (auto& e : entries_) {
            if (!e.reusable) {
                continue;
            }
            if (e.device != device || e.dtype != dtype || e.layout != layout
                || e.is_channel_last != is_channel_last) {
                continue;
            }
            if (!sizes_equal(e.sizes, sizes)) {
                continue;
            }

            e.reusable = false;
            return e.tensor;
        }

        auto t = at::empty(sizes, options);

        TensorEntry e;
        e.tensor = t;
        e.sizes.assign(sizes.begin(), sizes.end());
        e.device = device;
        e.dtype = dtype;
        e.layout = layout;
        e.is_channel_last = is_channel_last;
        e.reusable = false;
        entries_.emplace_back(std::move(e));
        return entries_.back().tensor;
    }

    void clear(int H, int W)
    {
        std::lock_guard<std::mutex> g(mu_);
        if (H == m_H && W == m_W) {
            return;
        }
        entries_.clear();
        m_H = H;
        m_W = W;
    }

    at::Tensor empty(c10::IntArrayRef sizes, const at::TensorOptions& options)
    {
        return acquire(sizes, options);
    }

    at::Tensor empty_like(const at::Tensor& a)
    {
        return acquire(a.sizes(), a.options().memory_format(a.is_contiguous(at::MemoryFormat::ChannelsLast)
                                                                ? at::MemoryFormat::ChannelsLast
                                                                : at::MemoryFormat::Contiguous));
    }

    bool release(const at::Tensor& t)
    {
        if (!t.defined()) {
            return false;
        }
        return set_reusable(t, true);
    }

    bool set_reusable(const at::Tensor& t, bool flag = true)
    {
        std::lock_guard<std::mutex> g(mu_);
        auto* impl = t.unsafeGetTensorImpl();
        for (auto& e : entries_) {
            if (e.tensor.unsafeGetTensorImpl() == impl) {
                e.reusable = flag;
                return true;
            }
        }
        return false;
    }

private:
    mutable std::mutex mu_;
    std::vector<TensorEntry> entries_;
    int m_H{ 0 };
    int m_W{ 0 };
};

extern TensorPool g_tensor_pool;
