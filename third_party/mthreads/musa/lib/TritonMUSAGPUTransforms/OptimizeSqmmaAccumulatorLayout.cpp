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
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/BitVector.h"

#include <optional>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace {

static Value unwrapSqmmaWaitResult(Value value) {
  while (auto result = dyn_cast<OpResult>(value)) {
    auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(result.getOwner());
    if (!wait)
      break;
    unsigned idx = result.getResultNumber();
    if (idx >= wait.getInputs().size())
      break;
    value = wait.getInputs()[idx];
  }
  return value;
}

static bool isSqmmaAccumulatorLoopArg(Value loopArg) {
  for (Operation *user : loopArg.getUsers()) {
    auto sqmma = dyn_cast<triton::musa::SquadDotOp>(user);
    if (!sqmma)
      continue;
    if (sqmma.getC() == loopArg)
      return true;
  }
  return false;
}

static bool sinkLoopCarriedSqmmaAccumulatorConvert(scf::ForOp &forOp,
                                                   RewriterBase &rewriter) {
  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp)
    return false;

  unsigned numIterArgs = forOp.getNumRegionIterArgs();
  if (yieldOp.getNumOperands() != numIterArgs)
    return false;

  struct Candidate {
    unsigned blockedIdx;
    unsigned mmaIdx;
    Operation *waitToCleanup = nullptr;
  };

  SmallVector<Candidate> candidates;
  llvm::SmallBitVector blockedIdxSeen(numIterArgs, false);
  for (unsigned blockedIdx = 0; blockedIdx < numIterArgs; ++blockedIdx) {
    auto cvt =
        yieldOp.getOperand(blockedIdx).getDefiningOp<ttg::ConvertLayoutOp>();
    if (!cvt || cvt->getBlock() != forOp.getBody())
      continue;

    Value mmaValue = unwrapSqmmaWaitResult(cvt.getSrc());
    std::optional<unsigned> mmaIdx;
    for (unsigned idx = 0; idx < numIterArgs; ++idx) {
      if (unwrapSqmmaWaitResult(yieldOp.getOperand(idx)) == mmaValue) {
        mmaIdx = idx;
        break;
      }
    }
    if (!mmaIdx || *mmaIdx == blockedIdx)
      continue;
    if (blockedIdxSeen.test(blockedIdx))
      continue;
    if (forOp.getResult(blockedIdx).getType() != cvt.getType())
      continue;
    if (forOp.getResult(*mmaIdx).getType() != mmaValue.getType())
      continue;
    if (!isSqmmaAccumulatorLoopArg(forOp.getRegionIterArg(*mmaIdx)))
      continue;

    blockedIdxSeen.set(blockedIdx);
    Operation *waitToCleanup = nullptr;
    if (auto waitResult = dyn_cast<OpResult>(cvt.getSrc()))
      waitToCleanup =
          dyn_cast<triton::musa::SquadDotWaitOp>(waitResult.getOwner())
              .getOperation();

    candidates.push_back({blockedIdx, *mmaIdx, waitToCleanup});
  }

  if (candidates.empty())
    return false;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(forOp);
  for (const Candidate &candidate : candidates) {
    Value converted = ttg::ConvertLayoutOp::create(
        rewriter, forOp.getLoc(),
        forOp.getResult(candidate.blockedIdx).getType(),
        forOp.getResult(candidate.mmaIdx));
    forOp.getResult(candidate.blockedIdx).replaceAllUsesWith(converted);
  }

  llvm::BitVector eraseBits(numIterArgs);
  for (const Candidate &candidate : candidates)
    eraseBits.set(candidate.blockedIdx);
  eraseLoopCarriedValues(forOp, eraseBits);

  llvm::SmallDenseSet<Operation *> waitsToCleanup;
  for (const Candidate &candidate : candidates) {
    if (candidate.waitToCleanup)
      waitsToCleanup.insert(candidate.waitToCleanup);
  }
  for (Operation *waitOp : waitsToCleanup) {
    if (waitOp->use_empty())
      rewriter.eraseOp(waitOp);
  }
  return true;
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUOPTIMIZESQMMAACCUMULATORLAYOUT
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUOptimizeSqmmaAccumulatorLayoutPass
    : impl::TritonMUSAGPUOptimizeSqmmaAccumulatorLayoutBase<
          TritonMUSAGPUOptimizeSqmmaAccumulatorLayoutPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    for (tt::FuncOp func : mod.getOps<tt::FuncOp>()) {
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<scf::ForOp> forOps;
        func.walk([&](scf::ForOp loop) { forOps.push_back(loop); });
        for (scf::ForOp loop : forOps) {
          if (!loop->getBlock())
            continue;
          if (sinkLoopCarriedSqmmaAccumulatorConvert(loop, rewriter))
            changed = true;
        }
      }
    }
  }
};

} // namespace mlir
