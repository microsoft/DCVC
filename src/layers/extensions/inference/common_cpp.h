// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

class GPUInfo {
public:
    GPUInfo()
    {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
            return;
        }

        // only query the first GPU
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        computeCapability = deviceProp.major * 10 + deviceProp.minor;
    }
    ~GPUInfo() {}

public:
    int computeCapability{ 0 };
};

inline GPUInfo& get_gpu_info()
{
    static GPUInfo info;
    return info;
}

inline int get_gpu_sm()
{
    return get_gpu_info().computeCapability;
}
