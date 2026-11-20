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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/MarkGMLoadPass.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "MarkGMLoad";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace {

static constexpr int kDefaultVBufferCount = 2;
static constexpr int kDefaultCBufferCount = 1;

struct MarkCandidate {
  memref::CopyOp copyOp;
  memref::AllocOp destAlloc; // dest backing alloc after view-like piercing
  scope::ScopeOp scopeOp;    // nearest enclosing scope, should not be null
  int bufferCount;           // filled in Phase 2
  bool fromHint = false;     // true if bufferCount came from gm_load hint
};

// Resolve whether a BlockArgument of func::FuncOp traces to a GM pointer.
// Returns:
//  - true if the arg belongs to the entry function (=> GM load source)
//  - false + non-null nextValue: trace through the caller's corresponding
//  operand, continue tracing
//  - false + null nextValue: tracing failed (no module / no caller)
static bool resolveFuncBlockArg(BlockArgument blockArg, func::FuncOp funcOp,
                                Value &nextValue) {
  auto moduleOp = funcOp->getParentOfType<ModuleOp>();
  if (!moduleOp) {
    LOG_DEBUG("[warning] moduleOp is null");
    return false;
  }
  if (SymbolTable::symbolKnownUseEmpty(funcOp.getNameAttr(), moduleOp)) {
    return true; // entry func argument => GM load
  }
  // Non-entry func: trace through callers.
  auto symbolUses = SymbolTable::getSymbolUses(funcOp.getNameAttr(), moduleOp);
  if (symbolUses && !symbolUses->empty()) {
    // get funcOp's caller, continue tracing
    auto callOp = cast<func::CallOp>((*symbolUses->begin()).getUser());
    nextValue = callOp.getOperands()[blockArg.getArgNumber()];
    return false;
  }
  LOG_DEBUG("[warning] Non-entry func: no caller uses "
            << funcOp.getNameAttr());
  return false;
}

// Rule 1: Pierce view-like ops and trace scf.for / scf.while iter_args back to
// their init values. Returns true only when the terminal is a BlockArgument
// owned by the entry func::FuncOp (i.e. a GM pointer function argument).
static bool traceSourceToFuncArg(Value v) {
  while (true) {
    // 1. Pierce view-like ops and tensor.extract_slice.
    while (auto *defOp = v.getDefiningOp()) {
      if (auto viewLike = dyn_cast<ViewLikeOpInterface>(defOp)) {
        v = viewLike.getViewSource();
        continue;
      }
      if (auto extractSlice = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
        v = extractSlice.getSource();
        continue;
      }
      break;
    }
    // 2. Check terminal: must be a BlockArgument.
    auto blockArg = dyn_cast<BlockArgument>(v);
    if (!blockArg) {
      return false; // ends at a defining op, not a GM arg
    }
    Operation *parentOp = blockArg.getOwner()->getParentOp();

    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      Value nextV;
      if (resolveFuncBlockArg(blockArg, funcOp, nextV)) {
        return true; // find GM load source in funcOp arg
      }
      if (nextV) {
        v = nextV;
        continue;
      }
      return false;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      if (blockArg.getArgNumber() == 0)
        return false; // induction variable, cannot be a GM load source
      // iter_arg: trace its init value (skip induction var at index 0).
      v = forOp.getInitArgs()[blockArg.getArgNumber() - 1];
      continue;
    }
    if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      // before-block arg: trace to the corresponding init operand.
      if (blockArg.getOwner() == whileOp.getBeforeBody()) {
        v = whileOp.getInits()[blockArg.getArgNumber()];
        continue;
      }
      // after-block arg: trace to the corresponding condition operand.
      if (blockArg.getOwner() == whileOp.getAfterBody()) {
        auto condOp =
            cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
        v = condOp.getArgs()[blockArg.getArgNumber()];
        continue;
      }
      LOG_DEBUG("[warning] unsupport while op ");
      return false;
    }
    LOG_DEBUG("[warning] unsupport iterarg or control flow");
    return false; // other BlockArgument kinds
  }
}

// Rule 2: Pierce view-like ops and tensor.extract_slice on the dest chain
// and return the backing memref::AllocOp, or null if the chain does not
// terminate at one.
static memref::AllocOp traceDestToAlloc(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (auto viewLikeOp = dyn_cast<ViewLikeOpInterface>(defOp)) {
      v = viewLikeOp.getViewSource();
      continue;
    }
    if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
      v = extractSliceOp.getSource();
      continue;
    }
    break;
  }
  return dyn_cast_or_null<memref::AllocOp>(v.getDefiningOp());
}

// Rule 3: Resolve multi-buffer count N from the enclosing scope.
// Returns -1 when scopeOp is null or has an unexpected tcore_type,
// which should normally never happen.
static int resolveBufferCount(scope::ScopeOp scopeOp) {
  int buffer_num = -1;
  if (!scopeOp) {
    return buffer_num;
  }
  bool isCube = false;
  bool isVector = false;
  if (failed(getScopeType(scopeOp, isCube, isVector))) {
    return buffer_num;
  }
  if (isVector) {
    buffer_num = kDefaultVBufferCount;
  } else if (isCube) {
    buffer_num = kDefaultCBufferCount;
  }
  LOG_DEBUG("return buffer num = " << buffer_num);
  return buffer_num;
}

// Check whether the dest alloc has a user-specified gm_load compile
// hint (annotation.mark with gm_load attr).  Returns the hint value
// (0/1 = force-off, >=2 = force-on with specified depth), or -1 if no hint.
//
// The hint may be attached either directly to the alloc result
// (post-bufferization) or to a tensor produced by bufferization.to_tensor
// (pre-bufferization).  In the latter case we pierce through the to_tensor op
// to find the annotation.
static int resolveHintBufferCount(memref::AllocOp destAlloc) {
  Value allocResult = destAlloc.getResult();
  std::optional<int> foundHint;
  auto hasConflictingHint = [&](annotation::MarkOp markOp) -> bool {
    auto attr = markOp->getAttrOfType<IntegerAttr>(
        CVPipeline::kGMLoadMultiBufferHintAttr);
    if (!attr)
      return false;
    int val = static_cast<int>(attr.getInt());
    markOp->removeAttr(CVPipeline::kGMLoadMultiBufferHintAttr);
    if (foundHint && *foundHint != val) {
      LOG_DEBUG("conflicting gm_load hints: " << *foundHint << " vs " << val);
      return true; // signal conflict
    }
    foundHint = val;
    return false;
  };
  for (auto *user : allocResult.getUsers()) {
    // Pre-bufferization: hint is on the tensor produced by to_tensor.
    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
      for (auto *tensorUser : toTensor.getResult().getUsers()) {
        if (auto markOp = dyn_cast<annotation::MarkOp>(tensorUser)) {
          if (hasConflictingHint(markOp))
            return -1;
        }
      }
    }
  }
  return foundHint.value_or(-1);
}

// Apply Rules 1 & 2 to a memref::CopyOp and build a MarkCandidate if eligible.
// Returns std::nullopt when any rule fails.
static std::optional<MarkCandidate> collectCandidate(memref::CopyOp copyOp) {
  // Rule 1: source must trace back to a func argument (GM load).
  if (!traceSourceToFuncArg(copyOp.getSource())) {
    return std::nullopt;
  }
  // Rule 2: dest must be created by a memref::AllocOp.
  auto allocOp = traceDestToAlloc(copyOp.getTarget());
  if (!allocOp) {
    LOG_DEBUG("dest not created by memref::AllocOp, skip");
    return std::nullopt;
  }
  auto scopeOp = copyOp->getParentOfType<scope::ScopeOp>();
  return MarkCandidate{copyOp, allocOp, scopeOp};
}

// Phase 3: insert or update annotation::MarkOp with multi_buffer attr on the
// dest alloc. Returns false when bufferCount <= 1 (skip marking), true
// otherwise.
static bool markGMLoadCandidate(MarkCandidate &c) {
  if (c.bufferCount <= 1) {
    LOG_DEBUG("bufferCount <= 1, skip marking");
    return false;
  }
  // Check if an annotation::MarkOp already exists for this alloc.
  annotation::MarkOp existingMarkOp = nullptr;
  for (auto *user : c.destAlloc->getUsers()) {
    if (auto markOp = dyn_cast<annotation::MarkOp>(user)) {
      existingMarkOp = markOp;
      break;
    }
  }
  OpBuilder builder(c.destAlloc);
  if (existingMarkOp) {
    existingMarkOp->setAttr(hivm::MultiBufferAttr::name,
                            builder.getI32IntegerAttr(c.bufferCount));
    if (c.fromHint)
      existingMarkOp->setAttr(CVPipeline::kGMLoadHintAttr,
                              builder.getUnitAttr());
  } else {
    builder.setInsertionPointAfter(c.destAlloc);
    auto markOp = builder.create<annotation::MarkOp>(c.destAlloc->getLoc(),
                                                     c.destAlloc.getResult());
    markOp->setAttr(hivm::MultiBufferAttr::name,
                    builder.getI32IntegerAttr(c.bufferCount));
    if (c.fromHint)
      markOp->setAttr(CVPipeline::kGMLoadHintAttr, builder.getUnitAttr());
  }
  LOG_DEBUG("marked multi_buffer = " << c.bufferCount << " on " << c.destAlloc);
  return true;
}

} // namespace

void MarkGMLoadPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<annotation::AnnotationDialect,
                  bufferization::BufferizationDialect, hivm::HIVMDialect,
                  memref::MemRefDialect, scope::ScopeDialect, scf::SCFDialect,
                  func::FuncDialect>();
}

void MarkGMLoadPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LOG_DEBUG("Enter MarkGMLoad pass");

  // Phase 1: collect candidates (read-only).
  SmallVector<MarkCandidate, 8> candidates;
  module.walk([&](memref::CopyOp copyOp) {
    if (auto candidate = collectCandidate(copyOp)) {
      candidates.push_back(*candidate);
    }
  });

  if (candidates.empty()) {
    LOG_DEBUG("no GM load candidate found");
    return;
  }

  // Phase 2 & 3: resolve buffer count N per candidate and insert
  // annotation::MarkOp (mutation).
  for (auto &c : candidates) {
    int hintVal = resolveHintBufferCount(c.destAlloc);
    if (hintVal >= 0) {
      // User specified a compile hint.
      if (hintVal <= 1) {
        // Force-off: skip marking entirely.
        LOG_DEBUG("hint force-off (val=" << hintVal << "), skip marking");
        continue;
      }
      c.bufferCount = hintVal;
      c.fromHint = true;
      LOG_DEBUG("hint force-on, bufferCount = " << hintVal);
    } else {
      // No hint: automatic resolution.
      c.bufferCount = resolveBufferCount(c.scopeOp);
      if (c.bufferCount < 0) {
        LOG_DEBUG("resolveBufferCount failed for " << c.copyOp << ", fallback");
        CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
        return;
      }
    }
    markGMLoadCandidate(c);
  }

  LOG_DEBUG("after MarkGMLoad: " << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMarkGMLoadPass() {
  return std::make_unique<MarkGMLoadPass>();
}

} // namespace triton
} // namespace mlir
