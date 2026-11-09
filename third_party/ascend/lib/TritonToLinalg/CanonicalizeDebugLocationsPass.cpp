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
// CanonicalizeDebugLocationsPass
//
// Collapses call-site locations whose callee is an inlined Triton stdlib
// helper (under site-packages) down to their caller (user-file) frame, so the
// emitted DWARF references the user's source line instead of the library.
// Fixes phantom line-table rows such as standard.py:306 produced by an inlined
// tl.sum reduction.
//
// Run BEFORE DeduplicateDebugNopsPass so NOP anchors are deduped against
// already-canonicalized (user-frame) locations.
//
// Gated by env var LLVM_EXTRACT_DI_LOCAL_VARIABLES=1.
//===----------------------------------------------------------------------===//

#include "TritonToLinalg/CanonicalizeDebugLocationsPass.h"
#include "Utils/DebugUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "canonicalize-debug-locations"

using namespace mlir;
using namespace mlir::triton;

namespace {

struct CanonicalizeDebugLocationsPass
    : public PassWrapper<CanonicalizeDebugLocationsPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CanonicalizeDebugLocationsPass)

  StringRef getArgument() const final { return "canonicalize-debug-locations"; }

  StringRef getDescription() const final {
    return "Collapse inlined stdlib call-site locations to the user frame.";
  }

  void runOnOperation() override {
    if (!mlir::triton::debug::isDebugNopEnabled())
      return;

    [[maybe_unused]] unsigned rewritten = 0;
    getOperation().walk([&](Operation *op) {
      Location oldLoc = op->getLoc();
      Location newLoc = mlir::triton::debug::collapseForeignCallsites(oldLoc);
      if (newLoc != oldLoc) {
        op->setLoc(newLoc);
        ++rewritten;
      }
    });

    LLVM_DEBUG(llvm::dbgs() << "[canon-debug-locs] rewrote " << rewritten
                            << " foreign call-site locations\n");
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<Pass> createCanonicalizeDebugLocationsPass() {
  return std::make_unique<CanonicalizeDebugLocationsPass>();
}

} // namespace triton
} // namespace mlir
