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
// DebugUtils.cpp
//
// Implementations for DebugUtils.h:
//   * NOP insertion helpers, runtime-gated via the
//     LLVM_EXTRACT_DI_LOCAL_VARIABLES environment variable, and
//   * the shared location analysis / rewrite helpers used by
//     CanonicalizeDebugLocationsPass and DeduplicateDebugNopsPass.
//
// All definitions live in mlir::triton::debug; DebugUtils.h re-exports the NOP
// helpers at global scope via using declarations.
//===----------------------------------------------------------------------===//

#include "Utils/DebugUtils.h"

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>

#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

using namespace mlir;

namespace mlir {
namespace triton {
namespace debug {

// Guard against pathologically deep / cyclic location trees.
static constexpr unsigned kMaxLocDepth = 16;

//===----------------------------------------------------------------------===//
// NOP insertion helpers.
//===----------------------------------------------------------------------===//

static bool moduleContainsScope(PatternRewriter &rewriter) {
  Block *insertionBlock = rewriter.getInsertionBlock();
  if (!insertionBlock)
    return false;

  Operation *parentOp = insertionBlock->getParentOp();
  if (!parentOp)
    return false;

  ModuleOp module = dyn_cast<ModuleOp>(parentOp);
  if (!module)
    module = parentOp->getParentOfType<ModuleOp>();
  if (!module)
    return false;

  bool found = false;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "scope.scope")
      return WalkResult::advance();
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

Location unwrapFusedLocForDebug(Location loc, unsigned depth) {
  if (depth > kMaxLocDepth)
    return loc;
  if (auto cs = dyn_cast<CallSiteLoc>(loc))
    return unwrapFusedLocForDebug(cs.getCaller(), depth + 1);
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (auto inner : llvm::reverse(fused.getLocations())) {
      if (!isa<UnknownLoc>(inner))
        return unwrapFusedLocForDebug(inner, depth + 1);
    }
  }
  return loc;
}

void insertDebugNop(Location loc, PatternRewriter &rewriter) {
  if (!isDebugNopEnabled() || moduleContainsScope(rewriter))
    return;
  auto unwrapped = unwrapFusedLocForDebug(loc);

  auto ctx = rewriter.getContext();
  rewriter.create<LLVM::InlineAsmOp>(
      unwrapped,
      /*resultTypes=*/TypeRange(),
      /*operands=*/ValueRange(),
      /*asm_string=*/"nop",
      /*constraints=*/"",
      /*has_side_effects=*/true,
      /*is_align_stack=*/false, LLVM::tailcallkind::TailCallKind::None,
      LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT), ArrayAttr());
}

void insertDebugNopForMask(Value mask, PatternRewriter &rewriter) {
  if (!mask)
    return;
  if (Operation *def = mask.getDefiningOp())
    insertDebugNop(def->getLoc(), rewriter);
}

// Collect every distinct user FileLineColLoc reachable through NameLoc /
// CallSiteLoc / (nested) FusedLoc wrappers. Locations are context-uniqued, so
// the Location handle itself is an exact dedup key (file, line, column).
static void collectUserLineLocs(Location loc,
                                llvm::SmallDenseSet<Location> &seen,
                                llvm::SmallVectorImpl<Location> &out,
                                unsigned depth = 0) {
  if (depth > kMaxLocDepth)
    return;
  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    collectUserLineLocs(cs.getCaller(), seen, out, depth + 1);
    return;
  }
  if (auto named = dyn_cast<NameLoc>(loc)) {
    collectUserLineLocs(named.getChildLoc(), seen, out, depth + 1);
    return;
  }
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (Location inner : fused.getLocations())
      collectUserLineLocs(inner, seen, out, depth + 1);
    return;
  }
  if (auto flc = dyn_cast<FileLineColLoc>(loc))
    if (seen.insert(flc).second)
      out.push_back(loc);
}

void insertDebugNopForAllLines(Location loc, PatternRewriter &rewriter) {
  if (!isDebugNopEnabled() || moduleContainsScope(rewriter))
    return;

  llvm::SmallDenseSet<Location> seen;
  llvm::SmallVector<Location, 4> lineLocs;
  collectUserLineLocs(loc, seen, lineLocs);
  if (lineLocs.empty()) {
    insertDebugNop(loc, rewriter);
    return;
  }
  for (Location l : lineLocs)
    insertDebugNop(l, rewriter);
}

//===----------------------------------------------------------------------===//
// Shared location analysis / rewrite helpers (no self-gating; passes gate).
//===----------------------------------------------------------------------===//

bool isForeignFile(llvm::StringRef filename) {
  // A library/stdlib file inlined into the kernel (e.g. triton/language/
  // standard.py). Heuristic: lives under a site-packages or dist-packages tree
  return filename.contains("/site-packages/") ||
         filename.contains("/dist-packages/");
}

bool isDebugNop(Operation *op) {
  auto asmOp = dyn_cast<LLVM::InlineAsmOp>(op);
  if (!asmOp)
    return false;
  if (!asmOp.getHasSideEffects())
    return false;
  if (asmOp.getAsmString() != "nop")
    return false;
  // NOPs have no results and no operands.
  if (asmOp->getNumResults() != 0 || asmOp->getNumOperands() != 0)
    return false;
  return true;
}

FileLineColLoc unwrapToUserFileLineCol(Location loc) {
  // Reuse the existing caller-preferring unwrapper, then descend NameLoc to the
  // underlying FileLineColLoc. callsite(stdlib at user) -> the user's line.
  Location l = unwrapFusedLocForDebug(loc);
  while (auto named = dyn_cast<NameLoc>(l))
    l = named.getChildLoc();
  return dyn_cast<FileLineColLoc>(l);
}

Location collapseForeignCallsites(Location loc, unsigned depth) {
  if (depth > kMaxLocDepth)
    return loc;

  if (auto named = dyn_cast<NameLoc>(loc)) {
    Location child = collapseForeignCallsites(named.getChildLoc(), depth + 1);
    return child == named.getChildLoc()
               ? loc
               : Location(NameLoc::get(named.getName(), child));
  }

  if (auto cs = dyn_cast<CallSiteLoc>(loc)) {
    Location caller = collapseForeignCallsites(cs.getCaller(), depth + 1);
    // The callee is the inlined helper frame; resolve its own file. (The
    // callee is not itself a call site, so caller-preference is moot here.)
    if (FileLineColLoc calleeFlc = unwrapToUserFileLineCol(cs.getCallee()))
      if (isForeignFile(calleeFlc.getFilename().getValue()))
        return caller;
    Location callee = collapseForeignCallsites(cs.getCallee(), depth + 1);
    if (callee == cs.getCallee() && caller == cs.getCaller())
      return loc;
    return Location(CallSiteLoc::get(callee, caller));
  }

  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    llvm::SmallVector<Location> newLocs;
    bool changed = false;
    for (Location sub : fused.getLocations()) {
      Location c = collapseForeignCallsites(sub, depth + 1);
      changed |= (c != sub);
      newLocs.push_back(c);
    }
    return changed ? Location(FusedLoc::get(loc.getContext(), newLocs,
                                            fused.getMetadata()))
                   : loc;
  }

  return loc;
}

bool isDebugNopEnabled() {
  // Environment variables do not change mid-process; read once. Function-local
  // static initialisation is thread-safe since C++11.
  static const bool enabled = [] {
    const char *s = std::getenv("LLVM_EXTRACT_DI_LOCAL_VARIABLES");
    if (!s)
      return false;
    std::string v(s);
    std::transform(v.begin(), v.end(), v.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return v == "1" || v == "true" || v == "on";
  }();
  return enabled;
}

} // namespace debug
} // namespace triton
} // namespace mlir
