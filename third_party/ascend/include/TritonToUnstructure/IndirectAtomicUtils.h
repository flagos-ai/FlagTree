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

#ifndef TRITON_ADAPTER_INDIRECTATOMICUTILS_H
#define TRITON_ADAPTER_INDIRECTATOMICUTILS_H

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace IndirectAtomicUtils {
using namespace mlir;
using namespace triton;

// The fast path requires a statically shaped result tensor whose element type
// is not int8. When offsetValue is provided it must be a statically shaped
// tensor or a scalar int/index. Additionally, CAS and XCHG do not use the
// fast path for f16/bf16 results.
bool canUseIndirectAtomicFastPath(triton::AtomicRMWOp op,
                                  Value offsetValue = Value());

bool canUseIndirectAtomicFastPath(triton::AtomicCASOp op,
                                  Value offsetValue = Value());

// Lowers tt.atomic_rmw to the indirect atomic custom op and restores the
// flattened custom result back to the original result tensor type.
FailureOr<Value> tryConvertAtomicRmwToIndirectCustom(triton::AtomicRMWOp op,
                                                     Value srcPtr,
                                                     Value offsetValue,
                                                     PatternRewriter &rewriter);

// Lowers tt.atomic_cas to the indirect atomic custom op and restores the
// flattened custom result back to the original result tensor type.
FailureOr<Value> tryConvertAtomicCasToIndirectCustom(triton::AtomicCASOp op,
                                                     Value srcPtr,
                                                     Value offsetValue,
                                                     PatternRewriter &rewriter);

} // namespace IndirectAtomicUtils

#endif // TRITON_ADAPTER_INDIRECTATOMICUTILS_H
