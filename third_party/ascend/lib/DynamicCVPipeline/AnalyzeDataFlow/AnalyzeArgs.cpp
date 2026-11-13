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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-args-in-forOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;

namespace {

// Data collected for each tensor iter_arg: first block_id that uses it, and set
// of all block_ids
struct TensorArgBlockInfo {
  int firstBlockId = -1;
  llvm::DenseSet<int> blockIds;
};

// Collect block info for all tensor-type iter_args in the given body.
static llvm::DenseMap<unsigned, TensorArgBlockInfo>
collectTensorArgBlockInfo(ArrayRef<Value> iterArgs, Block *body) {
  llvm::DenseMap<unsigned, TensorArgBlockInfo> result;

  if (!body) {
    return result;
  }

  for (Operation &op : body->without_terminator()) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt)
      continue;
    int blockId = *blockIdOpt;

    for (OpOperand &operand : op.getOpOperands()) {
      int argIdx = CVPipeline::getTensorIterArgIndex(operand.get(), iterArgs);
      if (argIdx >= 0) {
        auto &info = result[argIdx];
        info.blockIds.insert(blockId);
        if (info.firstBlockId < 0)
          info.firstBlockId = blockId;
      }
    }
  }

  return result;
}

// Check if any tensor-type iter_arg appears in different block_ids.
static bool checkMultiBlockUse(
    const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo) {
  for (auto &p : argBlockInfo) {
    if (p.second.blockIds.size() > 1)
      return true;
  }
  return false;
}

// Check if the first block_id differs from the block_id of yield's defining op.
static bool checkUseUpdateMismatch(
    ArrayRef<Value> iterArgs, Block *body,
    const llvm::DenseMap<unsigned, TensorArgBlockInfo> &argBlockInfo) {
  if (!body)
    return false;

  for (unsigned i = 0; i < iterArgs.size(); ++i) {
    if (!isa<RankedTensorType>(iterArgs[i].getType()))
      continue;

    auto it = argBlockInfo.find(i);
    if (it == argBlockInfo.end())
      continue;

    Operation *defOp = CVPipeline::getLoopCarriedDefOp(iterArgs[i], body);
    if (!defOp)
      continue;

    if (CVPipeline::getOpBlockId(defOp) != it->second.firstBlockId)
      return true;
  }
  return false;
}

// Main check function that combines both conditions
static bool hasTensorArgInDifferentBlockIds(ArrayRef<Value> iterArgs,
                                            Block *body) {
  auto argBlockInfo = collectTensorArgBlockInfo(iterArgs, body);
  return checkMultiBlockUse(argBlockInfo) ||
         checkUseUpdateMismatch(iterArgs, body, argBlockInfo);
}

// Detect "update-before-use": a tensor iter_arg's update-defining op
// appears in code order BEFORE another direct-child use of that iter_arg
// whose ssbuffer.block_id differs from the update's block_id.
static bool checkUpdateBeforeUse(ArrayRef<Value> iterArgs, Block *body) {
  if (!body)
    return false;
  auto yieldOp = dyn_cast<scf::YieldOp>(body->getTerminator());
  if (!yieldOp)
    return false;

  for (Value iterArg : iterArgs) {
    if (!isa<RankedTensorType>(iterArg.getType()))
      continue;

    Operation *defOp = CVPipeline::getLoopCarriedDefOp(iterArg, body);
    if (!defOp)
      continue;

    auto defBlockId = CVPipeline::getOpBlockId(defOp);
    if (!defBlockId)
      continue;

    for (Operation *user : iterArg.getUsers()) {
      Operation *directChild = body->findAncestorOpInBlock(*user);
      if (!directChild)
        continue;
      // must come AFTER the update-defining op in code order
      if (!defOp->isBeforeInBlock(directChild))
        continue;

      if (CVPipeline::getOpBlockId(directChild) != defBlockId)
        return true;
    }
  }
  return false;
}

} // namespace

bool checkTensorArgsInMainLoop(ModuleOp module) {
  return module
      .walk([&](Operation *op) -> WalkResult {
        if (!CVPipeline::isMainLoopOp(op))
          return WalkResult::advance();

        CVPipeline::MainLoop mainLoop(op);
        return hasTensorArgInDifferentBlockIds(mainLoop.getIterArgs(),
                                               mainLoop.getBody())
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

bool checkUpdateBeforeUseInMainLoop(ModuleOp module) {
  return module
      .walk([&](Operation *op) -> WalkResult {
        if (!CVPipeline::isMainLoopOp(op))
          return WalkResult::advance();

        CVPipeline::MainLoop mainLoop(op);
        return checkUpdateBeforeUse(mainLoop.getIterArgs(), mainLoop.getBody())
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

void AnalyzeArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Before AnalyzeArgs:\n" << module << "\n");

  BufferCountManager bufferCountMgr(module);
  int intraBufNum = bufferCountMgr.getBufferCountByType(
      BufferCountManager::DepType::IntraCore);
  int interBufNum = bufferCountMgr.getBufferCountByType(
      BufferCountManager::DepType::InterCore);

  if (checkUpdateBeforeUseInMainLoop(module)) {
    LDBG("Found tensor iter_args update-before-use in mainloop!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }

  if (intraBufNum == 3 && interBufNum == 2 &&
      checkTensorArgsInMainLoop(module)) {
    LDBG("Found tensor iter_args dependency in mainloop!");
    CVPipeline::setFallbackAttr(module,
                                CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED);
    return;
  }

  LDBG("After AnalyzeArgs:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeArgsPass() {
  return std::make_unique<AnalyzeArgsPass>();
}

} // namespace triton
} // namespace mlir
