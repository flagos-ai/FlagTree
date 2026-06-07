/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_FLAG_ID_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_FLAG_ID_MANAGER_H

#include <optional>
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace triton {

// Flag ID Manager for synchronization between cores
// Purpose: Globally scan and allocate unique flag_id for synchronization
class FlagIdManager {
public:
  // Maximum flag count (hardware limitation)
  static constexpr int MAX_FLAG_ID = 14;
  static constexpr int INVALID_FLAG_ID = -1;

  // Constructor: initialize with Module for scanning
  FlagIdManager(ModuleOp module);

  // Acquire an available ID: try to reuse existing flags first (conservative analysis),
  // if cannot reuse, then increment and allocate.
  // insertionPoint: the position where sync operation will be inserted, used for
  // linear comparison in reuse analysis. Can be nullptr.
  int acquireId(Operation* insertionPoint);

private:
  // Scan existing Flag IDs
  // Traverse Module to find all existing flag_id to prevent duplicate allocation
  void scanExistingFlags(ModuleOp module);

  // Currently allocated maximum ID
  int64_t currentMaxId = 0;

  // Save module for reuse analysis
  ModuleOp module;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_FLAG_ID_MANAGER_H
