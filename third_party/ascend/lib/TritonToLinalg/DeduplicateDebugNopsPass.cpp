/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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
//===----------------------------------------------------------------------===//
// DeduplicateDebugNopsPass
//
// Walks each function and removes redundant `llvm.inline_asm "nop"` ops that
// were inserted as DWARF anchors by the triton-to-linalg converters. Keeps
// the first NOP per unique source line; erases the rest.
//
// Source lines are resolved with the shared user-frame resolver
// (debug::unwrapToUserFileLineCol), so a NOP carrying an inlined call-site loc
// keys on the user's line rather than the inlined callee. Run this AFTER
// CanonicalizeDebugLocationsPass.
//
// Runtime-gated via the LLVM_EXTRACT_DI_LOCAL_VARIABLES environment variable.
//===----------------------------------------------------------------------===//

#include "TritonToLinalg/DeduplicateDebugNopsPass.h"
#include "Utils/DebugUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dedup-debug-nops"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct DeduplicateDebugNopsPass
    : public PassWrapper<DeduplicateDebugNopsPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DeduplicateDebugNopsPass)

  StringRef getArgument() const final { return "deduplicate-debug-nops"; }

  StringRef getDescription() const final {
    return "Deduplicate llvm.inline_asm 'nop' debug-anchor ops by source line.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    // Opt-in via env var; do nothing in production builds.
    if (!mlir::triton::debug::isDebugNopEnabled())
      return;

    ModuleOp moduleOp = getOperation();
    // Only read inside LLVM_DEBUG, which is a no-op under NDEBUG.
    [[maybe_unused]] unsigned totalDropped = 0;
    [[maybe_unused]] unsigned totalKept = 0;

    moduleOp.walk([&](func::FuncOp func) {
      // Per-function dedup: key = (filename, line). Column intentionally
      // ignored -- for debugger stepping behaviour, two NOPs on the same line
      // at different columns are equivalent. The StringRef points into a
      // StringAttr owned by the MLIRContext, which outlives this pass.
      llvm::DenseSet<std::pair<StringRef, unsigned>> seen;
      llvm::SmallVector<Operation *, 16> toErase;

      func.walk([&](Operation *op) {
        if (!mlir::triton::debug::isDebugNop(op))
          return;

        // Resolve to the user's source line (caller frame for call sites).
        FileLineColLoc flc =
            mlir::triton::debug::unwrapToUserFileLineCol(op->getLoc());
        if (!flc) {
          // No resolvable source location; leave the NOP alone.
          LLVM_DEBUG(llvm::dbgs()
                     << "[dedup-nops] NOP with no FileLineColLoc, keeping: "
                     << *op << "\n");
          ++totalKept;
          return;
        }

        auto key = std::make_pair(flc.getFilename().getValue(), flc.getLine());
        if (!seen.insert(key).second) {
          toErase.push_back(op);
        }
      });

      totalKept += seen.size();
      totalDropped += toErase.size();

      for (Operation *op : toErase)
        op->erase();
    });

    LLVM_DEBUG(llvm::dbgs()
               << "[dedup-nops] dropped " << totalDropped
               << " duplicate NOPs, kept " << totalKept << " unique anchors\n");
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createDeduplicateDebugNopsPass() {
  return std::make_unique<DeduplicateDebugNopsPass>();
}

} // namespace triton
} // namespace mlir
