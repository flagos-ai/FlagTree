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

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

static Value peelSqmmaIssueOperand(Value value) {
  llvm::SmallPtrSet<void *, 8> visited;
  while (value) {
    if (!visited.insert(value.getAsOpaquePointer()).second)
      break;

    if (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = cvt.getSrc();
      continue;
    }
    if (auto load = value.getDefiningOp<ttg::LocalLoadOp>()) {
      value = load.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescTransOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescReshapeOp>()) {
      value = view.getSrc();
      continue;
    }
    break;
  }
  return value;
}

static bool isIssueBarrier(ttg::BarrierOp barrier) {
  return barrier && barrier.hasLocal() &&
         barrier.getAddrSpace() != ttg::AddrSpace::Local;
}

static bool
shouldInsertIssueBarrierBefore(triton::musa::AsyncTMECopyLocalToGlobalOp op) {
  Operation *prev = op->getPrevNode();
  auto prevBarrier = dyn_cast_or_null<ttg::BarrierOp>(prev);
  return !isIssueBarrier(prevBarrier);
}

static bool shouldInsertIssueBarrierBefore(triton::musa::SquadDotOp op) {
  Operation *prev = op->getPrevNode();
  auto prevBarrier = dyn_cast_or_null<ttg::BarrierOp>(prev);
  if (isIssueBarrier(prevBarrier))
    return false;

  Value aMemDesc = peelSqmmaIssueOperand(op.getA());
  Value bMemDesc = peelSqmmaIssueOperand(op.getB());
  if (!aMemDesc || !bMemDesc)
    return true;
  if (!isa<ttg::MemDescType>(aMemDesc.getType()) ||
      !isa<ttg::MemDescType>(bMemDesc.getType()))
    return true;

  return triton::musa::needsSqmmaIssueBarrier(aMemDesc, bMemDesc);
}

static void insertIssueBarrierBefore(Operation *op, RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  // MUSA lowers non-local TTG barriers to llvm.musa.barrier0.
  ttg::BarrierOp::create(rewriter, op->getLoc(), ttg::AddrSpace::All);
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUISSUEBARRIERINSERTION
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUIssueBarrierInsertionPass
    : impl::TritonMUSAGPUIssueBarrierInsertionBase<
          TritonMUSAGPUIssueBarrierInsertionPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> candidates;

    mod.walk([&](Operation *op) {
      if (isa<triton::musa::AsyncTMECopyLocalToGlobalOp,
              triton::musa::SquadDotOp>(op))
        candidates.push_back(op);
    });

    for (Operation *op : candidates) {
      if (auto store =
              dyn_cast<triton::musa::AsyncTMECopyLocalToGlobalOp>(op)) {
        if (shouldInsertIssueBarrierBefore(store))
          insertIssueBarrierBefore(op, rewriter);
        continue;
      }

      if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
        if (shouldInsertIssueBarrierBefore(sqmma))
          insertIssueBarrierBefore(op, rewriter);
        continue;
      }
    }
  }
};

} // namespace mlir
