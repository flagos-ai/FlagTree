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

#ifndef TRITON_TO_GRAPH_ENTRY_ARG_POINTER_ALIAS_ANALYSIS_H
#define TRITON_TO_GRAPH_ENTRY_ARG_POINTER_ALIAS_ANALYSIS_H

#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace triton {
namespace cfg {

class AliasAnalysis;

// Relationship between pointer values that can be traced to global pointer
// arguments of the current tt.func. Distinct roots use the graph-optimization
// ABI assumption that distinct entry pointers do not share storage.
enum class EntryArgPointerRelation : uint8_t {
  SameEntryRoot,
  DistinctEntryRoots,
  Unknown,
};

// Classifies pointer values by their entry-function global pointer root. This
// intentionally builds on AliasAnalysis provenance without changing the
// language-level semantics of AliasAnalysis::mayAlias().
class EntryArgPointerAliasAnalysis {
public:
  EntryArgPointerAliasAnalysis(triton::FuncOp function,
                               const AliasAnalysis &aliasAnalysis);

  EntryArgPointerRelation classify(Value lhs, Value rhs) const;

private:
  // Returns the current function's global pointer block argument from which
  // pointer is derived, or a null Value when provenance is unsupported.
  Value getEntryPointerRoot(Value pointer) const;

  const AliasAnalysis &aliasAnalysis;
  SmallVector<Value> entryPointerRoots;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_ENTRY_ARG_POINTER_ALIAS_ANALYSIS_H
