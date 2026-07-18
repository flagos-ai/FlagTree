// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_MACA_COMMON_H
#define TRITON_MACA_COMMON_H

#include "Utility.h"

using ValueTable = std::map<std::pair<unsigned, unsigned>, Value>;

enum class TensorCoreType : uint8_t {
  FP32_FP16_FP16_FP32 = 0,
  FP32_BF16_BF16_FP32,
  FP32_FP32_FP32_FP32,
  FP32_TF32_TF32_FP32,
  FP64_FP64_FP64_FP64,
  INT32_INT8_INT8_INT32,
  INT32_INT8_INT8_INT32_1,
  FP32_FP8_FP8_FP32,
  FP32_BF8_BF8_FP32,
  NOT_APPLICABLE,
};

#endif
