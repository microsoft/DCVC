// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

constexpr float SCALE_MIN = 0.11f;
constexpr float SCALE_MAX = 16.f;
constexpr float LOG_SCALE_MIN = -2.2073f;  // std::log(SCALE_MIN);
constexpr float LOG_SCALE_MAX = 2.7726f;   // std::log(SCALE_MAX)
constexpr int SCALE_LEVEL = 128;
constexpr float LOG_SCALE_STEP = (LOG_SCALE_MAX - LOG_SCALE_MIN) / (SCALE_LEVEL - 1);
constexpr float LOG_SCALE_STEP_RECIP = 1.f / LOG_SCALE_STEP;

constexpr int COND_KERNEL_THREAD_NUM1 = 1024;
constexpr int COND_KERNEL_THREAD_NUM2 = 1024;
constexpr int COND_KERNEL_PER_THREAD_NUM = 8;

constexpr int MIN_SYMBOLS_PER_STREAM = 32768;
