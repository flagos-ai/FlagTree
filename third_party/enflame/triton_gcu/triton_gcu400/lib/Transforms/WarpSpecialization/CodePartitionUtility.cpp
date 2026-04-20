/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <list>
#include <unordered_set>
#include <utility>

#include "CodePartitionUtility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-utility"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gcu {

// Get the tensor type from the load type.
RankedTensorType getTensorType(Type loadType) {
  if (auto ptrType = dyn_cast<tt::PointerType>(loadType)) {
    return dyn_cast<RankedTensorType>(ptrType.getPointeeType());
  }
  return dyn_cast<RankedTensorType>(loadType);
}

// Check to see if op is enclosed under ifOp.
bool enclosing(scf::IfOp ifOp, Operation *op) {
  return ifOp->isProperAncestor(op);
}

bool enclosing(scf::ForOp forOp, Operation *op) {
  return forOp->isProperAncestor(op);
}

// Check to see if there is no outer loop that is enclosed under ifOp.
bool immediateEnclosing(scf::IfOp ifOp, Operation *subOp) {
  auto pOp = subOp->getParentOfType<scf::ForOp>();
  if (!pOp)
    return true;
  return !enclosing(ifOp, pOp.getOperation());
}

// Return total number of AccumCnts for the given ctrlOp.
// We need one for each nested region that contains a channel.
unsigned getAccumCnts(Operation *ctrlOp,
                      const DenseSet<Operation *> &regionsWithChannels) {
  unsigned cnt = 0;
  LDBG() << "getAccumCnts: " << ctrlOp;
  for (auto *op : regionsWithChannels) {
    LDBG() << "-- getAccumCnts: " << ctrlOp << " regionsWithChannels " << op;
    if (ctrlOp == op) {
      ++cnt;
    } else if (auto forOp = dyn_cast<scf::ForOp>(ctrlOp)) {
      if (enclosing(forOp, op))
        ++cnt;
    } else if (auto ifOp = dyn_cast<scf::IfOp>(ctrlOp)) {
      if (enclosing(ifOp, op))
        ++cnt;
    } else {
      llvm_unreachable("region op other than If/For is not supported");
    }
  }
  return cnt;
}

// Assume parentForOp has accumCnt for the specified ctrlOp.
unsigned getAccumArgIdx(scf::ForOp parentForOp, Operation *ctrlOp,
                        const DenseSet<Operation *> &regionsWithChannels) {
  // Walk parentForOp in preorder.
  unsigned preOrderId = 0;
  unsigned ctrlId = 0;
  bool found = false;
  parentForOp->walk<WalkOrder::PreOrder>([&](Operation *subOp) {
    if (subOp == ctrlOp) {
      ctrlId = preOrderId;
      found = true;
    }
    for (auto *op : regionsWithChannels) {
      if (op == subOp) {
        LDBG() << "getAccumArgIdx: saw ctrlOp enclosing channel " << subOp;
        ++preOrderId;
      }
    }
  });
  assert(found && "error in getAccumArgIdx");
  LDBG() << "getAccumArgIdx: " << parentForOp.getOperation() << " " << ctrlOp
         << " ctrlId: " << ctrlId;
  return ctrlId;
}

// Compute and return the buffer index and phase for a given accumulate count.
std::pair<Value, Value> getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder,
                                             Location loc, Value accumCnt,
                                             unsigned numBuffers) {
  Value numBuffersVal =
      builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, numBuffers, 32);
  numBuffersVal = builder.createWithAsyncTaskIds<arith::ExtSIOp>(
      loc, builder.getI64Type(), numBuffersVal);
  // Calculate accumCnt / numBuffers
  // initBufferIdx = accumCnt - accumCnt / numBuffers * numBuffers
  //    -> accumCnt % numBuffers
  // initPhase = (accumCnt / numBuffers) & 1
  Value bufferIdx = builder.createWithAsyncTaskIds<arith::DivUIOp>(
      loc, accumCnt, numBuffersVal);
  Value initBufferIdx = builder.createWithAsyncTaskIds<arith::SubIOp>(
      loc, accumCnt,
      builder.createWithAsyncTaskIds<arith::MulIOp>(loc, bufferIdx,
                                                    numBuffersVal));
  initBufferIdx = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI32Type(), initBufferIdx);

  Value one = builder.createWithAsyncTaskIds<arith::ConstantIntOp>(loc, 1, 64);
  bufferIdx =
      builder.createWithAsyncTaskIds<arith::AndIOp>(loc, bufferIdx, one);
  Value initPhase = builder.createWithAsyncTaskIds<arith::TruncIOp>(
      loc, builder.getI1Type(), bufferIdx);
  return {initBufferIdx, initPhase};
}

// Get the current accumulation count for the given op within its immediate
// scope.
// ForA (accumForA, accumIfA, accumForB, accumIfB)
//   IfA (accumIfA, accumForB)
//     Channel A --> uses ForA.arg[accumIfA]
//     ForB (accumForB)
//       Channel B --> uses ForB.arg[accumForB]
//   ThenYield ForA.arg[accumIfA] + 1, ForB.res[accumForB]
//   ElseYield ForA.arg[accumIfA], ForA.arg[accumForB]
//   ForC (accumForC, accumIfB)
//     IfB
//       Channel C --> uses ForC.arg[accumIfB]
//     ThenYield ForC.arg[accumIfB] + 1
//     ElseYield ForC.arg[accumIfB]
//   Channel D --> uses ForA.arg[accumForA]
Value getAccumCount(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                    const DenseSet<Operation *> &regionsWithChannels) {
  auto parentForOp = op->getParentOfType<scf::ForOp>();
  auto *pOp = op->getParentOp();
  // Get parentForOp.arg[pOp]
  unsigned tSize = parentForOp.getBody()->getArguments().size();
  unsigned parentTCnts = getAccumCnts(parentForOp, regionsWithChannels);
  unsigned accumArgId = getAccumArgIdx(parentForOp, pOp, regionsWithChannels);
  Value accumCnt =
      parentForOp.getBody()->getArgument(tSize - parentTCnts + accumArgId);

  LDBG() << "getAccumCount: parentForOp " << parentForOp.getOperation()
         << " pOp " << pOp << " " << tSize << " " << parentTCnts << " "
         << accumArgId;
  return accumCnt;
}

void getBufferIdxAndPhase(OpBuilderWithAsyncTaskIds &builder, Operation *op,
                          unsigned numBuffers,
                          const DenseSet<Operation *> &regionsWithChannels,
                          Value &bufferIdx, Value &phase) {
  Value accumCnt = getAccumCount(builder, op, regionsWithChannels);
  std::tie(bufferIdx, phase) =
      getBufferIdxAndPhase(builder, op->getLoc(), accumCnt, numBuffers);
}

Value getBarrierForPipelineStage(OpBuilderWithAsyncTaskIds &builder,
                                 Value barrierAlloc, Value bufferIdx) {
  auto context = barrierAlloc.getContext();
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(context);
  ttg::MemDescType barrierTy = ttg::MemDescType::get(
      {1}, builder.getI64Type(),
      cast<ttg::MemDescType>(barrierAlloc.getType()).getEncoding(),
      sharedMemorySpace,
      /*mutableMemory=*/true);

  // Create barrierForTMA from barrierAlloc.
  return builder.createWithAsyncTaskIds<ttg::MemDescIndexOp>(
      barrierAlloc.getLoc(), barrierTy, barrierAlloc, bufferIdx);
}

} // namespace mlir::triton::gcu
