// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

template <typename T> struct GPUTensor {
  GPUTensor(torch::Tensor &t) : ptr(static_cast<T *>(t.data_ptr())) {
    assert(sizeof(T) == t.element_size());
    assert(t.dim() <= 8);
    for (int i = 0; i < t.dim(); i++) {
      stride[i] = static_cast<int>(t.stride(i));
    }
  }
  GPUTensor(const torch::Tensor &t) : ptr(static_cast<T *>(t.data_ptr())) {
    assert(sizeof(T) == t.element_size());
    assert(t.dim() <= 8);
    for (int i = 0; i < t.dim(); i++) {
      stride[i] = static_cast<int>(t.stride(i));
    }
  }

  T *__restrict__ const ptr;
  int stride[8] = {0};
};

template <typename T> inline __device__ T __totype(float a) {
  return static_cast<T>(a);
}

template <> inline __device__ __half __totype(float a) {
  return __float2half(a);
}
