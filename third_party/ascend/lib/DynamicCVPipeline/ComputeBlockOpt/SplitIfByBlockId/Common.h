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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_COMPUTE_OPT_SPLIT_IF_BY_BLOCK_ID_COMMON_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMPUTE_BLOCK_COMPUTE_OPT_SPLIT_IF_BY_BLOCK_ID_COMMON_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"

#include "DynamicCVPipeline/Common/Utils.h"

namespace mlir::CVPipeline::SplitIf {

/// One group of ops sharing the same block_id inside an scf.if.
/// N=0: all storage is heap-allocated, keeping sizeof small.
struct BlockGroup {
  int blockId;
  SmallVector<Operation *, 0> ops;     // ops in this group (original order)
  SmallVector<scf::IfOp, 0> nestedIfs; // nested ifs that belong to this group
};

// Logic: find until the first tensor dependency
class ScalarClosure {
private:
  mlir::Block *block = nullptr;
  mlir::Block *parentBlock = nullptr;
  bool includeParent = true;
  llvm::ArrayRef<Operation *> ops;

  void collectScalarClosure(Value val);
  void collectOuterDependency(Operation *op);

public:
  static constexpr size_t kExpectedMaxScalarOps = 4;
  llvm::SmallPtrSet<Operation *, kExpectedMaxScalarOps> scalarOps;
  ScalarClosure(BlockGroup &group, ArrayRef<Operation *> ops);
  ScalarClosure(Block *block, ArrayRef<Operation *> ops, bool includeParent);
  void collect();

  // Safety: both operations must be in scalarOps
  bool isBefore(Operation *a, Operation *b);

  std::pair<SmallVector<Operation *>, IRMapping> capture(int blockId);
};

struct WalkMainLoopResult {
  CoreType coreType = UNDETERMINED;
  bool containsMainLoop = false;
};

llvm::FailureOr<WalkMainLoopResult>
walkMainLoop(Operation *op,
             llvm::function_ref<llvm::LogicalResult(Operation *)> pred);

} // namespace mlir::CVPipeline::SplitIf

#endif
