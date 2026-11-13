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

#ifndef TRITON_ASCEND_UTILS_DEBUGUTILS_H
#define TRITON_ASCEND_UTILS_DEBUGUTILS_H

#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Value.h>

namespace mlir {
class Operation;
class PatternRewriter;
} // namespace mlir

//===----------------------------------------------------------------------===//
// msdebug helpers. All declarations live in mlir::triton::debug; the NOP
// insertion helpers are additionally exposed at global scope via using
// declarations (see the bottom of this header) so existing converter call
// sites remain unqualified.
//
// The NOP insertion helpers are runtime-gated via the
// LLVM_EXTRACT_DI_LOCAL_VARIABLES environment variable (see isDebugNopEnabled);
// they are not compiled out. Definitions in DebugUtils.cpp.
//===----------------------------------------------------------------------===//

namespace mlir {
namespace triton {
namespace debug {

//===----------------------------------------------------------------------===//
// NOP insertion helpers.
//===----------------------------------------------------------------------===//

/// Unwrap CallSiteLoc (caller-preferring) and FusedLoc (last non-unknown) down
/// to a single representative location for tagging a debug NOP.
/// `depth` tracks recursion internally; callers should use the default (0).
/// Recursion is bounded by an implementation-defined limit (currently 16) to
/// guard against pathologically deep or cyclic location trees; on hitting the
/// limit the location is returned as-is.
Location unwrapFusedLocForDebug(Location loc, unsigned depth = 0);

/// Insert a side-effecting `llvm.inline_asm "nop"` carrying `loc`, so the
/// source line stays anchored in the DWARF line table. No-op unless
/// LLVM_EXTRACT_DI_LOCAL_VARIABLES is enabled. Must be called before the op
/// carrying the location is erased.
void insertDebugNop(Location loc, PatternRewriter &rewriter);

/// Anchor the source line of a mask operand: inserts a debug NOP at the
/// location of the op defining `mask`, so a masked load/store/atomic keeps a
/// breakable PC for the `mask = ...` line. No-op when `mask` is null or has no
/// defining op.
void insertDebugNopForMask(Value mask, PatternRewriter &rewriter);

/// Insert one debug NOP per unique FileLineColLoc reachable from `loc`,
/// descending through CallSiteLoc (caller frame), NameLoc, and nested FusedLoc
/// wrappers. Keeps every distinct user source line fused into a single op
/// breakable. Falls back to a single anchor on `loc` when no concrete
/// FileLineColLoc is reachable.
void insertDebugNopForAllLines(Location loc, PatternRewriter &rewriter);

//===----------------------------------------------------------------------===//
// Shared location analysis / rewrite helpers used by the debug passes
// (CanonicalizeDebugLocationsPass, DeduplicateDebugNopsPass).
// These do NOT self-gate; the passes gate.
//===----------------------------------------------------------------------===//

/// True if `filename` is an inlined library/stdlib file (lives under a
/// site-packages or dist-packages tree).
bool isForeignFile(llvm::StringRef filename);

/// True if `op` is one of our debug NOPs: llvm.inline_asm "nop", side
/// effecting, with no results and no operands.
bool isDebugNop(Operation *op);

/// Resolve `loc` to the FileLineColLoc the user should see -- caller frame for
/// call sites (via unwrapFusedLocForDebug), descending NameLoc.
/// Returns a null FileLineColLoc when no user-facing file location is
/// reachable; callers MUST check `if (!result)` before use.
FileLineColLoc unwrapToUserFileLineCol(Location loc);

/// Rewrite call-site locations whose callee resolves to a foreign (stdlib)
/// file so they collapse to their caller (user) frame. Recurses through
/// NameLoc / FusedLoc / nested call sites. `depth` tracks recursion
/// internally; callers should use the default (0). Recursion is bounded by an
/// implementation-defined limit (currently 16).
Location collapseForeignCallsites(Location loc, unsigned depth = 0);

/// Whitelist-free replacement for tools::getBoolEnv -- reads the
/// LLVM_EXTRACT_DI_LOCAL_VARIABLES gate directly so no core-Triton
/// (GetEnv.hpp) change is required. The environment is read once and cached.
bool isDebugNopEnabled();

} // namespace debug
} // namespace triton
} // namespace mlir

//===----------------------------------------------------------------------===//
// Backward-compatible global-scope aliases for the NOP insertion helpers, so
// the TritonToLinalg converters can keep calling them unqualified.
//===----------------------------------------------------------------------===//

using mlir::triton::debug::insertDebugNop;
using mlir::triton::debug::insertDebugNopForAllLines;
using mlir::triton::debug::insertDebugNopForMask;

#endif // TRITON_ASCEND_UTILS_DEBUGUTILS_H
