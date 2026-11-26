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

#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"

#include "TritonToGraph/AliasAnalysis.h"

using namespace mlir;
using namespace triton;
using namespace cfg;

EntryArgPointerAliasAnalysis::EntryArgPointerAliasAnalysis(
    triton::FuncOp function, const AliasAnalysis &aliasAnalysis)
    : aliasAnalysis(aliasAnalysis) {
  for (BlockArgument argument : function.getArguments()) {
    if (!AliasAnalysis::isPointerType(argument.getType()) ||
        !AliasAnalysis::isGlobalMemoryType(argument.getType()) ||
        !aliasAnalysis.getTensorObject(argument))
      continue;
    entryPointerRoots.push_back(argument);
  }
}

EntryArgPointerRelation
EntryArgPointerAliasAnalysis::classify(Value lhs, Value rhs) const {
  Value lhsRoot = getEntryPointerRoot(lhs);
  Value rhsRoot = getEntryPointerRoot(rhs);
  if (!lhsRoot || !rhsRoot)
    return EntryArgPointerRelation::Unknown;

  return lhsRoot == rhsRoot ? EntryArgPointerRelation::SameEntryRoot
                            : EntryArgPointerRelation::DistinctEntryRoots;
}

Value EntryArgPointerAliasAnalysis::getEntryPointerRoot(Value pointer) const {
  if (!pointer || !AliasAnalysis::isPointerType(pointer.getType()))
    return {};

  Value base = aliasAnalysis.getBasePointer(pointer);
  if (!base || !AliasAnalysis::isPointerType(base.getType()))
    return {};

  // AliasAnalysis only creates TensorObjects for global-memory function
  // arguments. Require that tracked provenance before treating a root as an
  // entry ABI pointer.
  if (!aliasAnalysis.getTensorObject(base))
    return {};

  for (Value entryPointerRoot : entryPointerRoots) {
    if (entryPointerRoot == base)
      return entryPointerRoot;
  }

  return {};
}
