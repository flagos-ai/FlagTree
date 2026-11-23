/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_
#define TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir::triton::tle {

enum class PointerAddressClass {
  Global,
  Shared,
  Unknown,
};

PointerAddressClass classifyPointerAddress(Value value);
std::optional<Value> getSharedPointerMemDescRoot(Value ptr);

bool mayReadSharedMemory(Operation *op);
bool mayWriteSharedMemory(Operation *op);
bool mayWriteSharedMemoryAlias(Operation *op, Value memdesc);
bool hasInterveningSharedMemoryWrite(Operation *from, Operation *to);
bool hasInterveningSharedMemoryWriteAlias(Operation *from, Operation *to,
                                          Value memdesc);

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_
