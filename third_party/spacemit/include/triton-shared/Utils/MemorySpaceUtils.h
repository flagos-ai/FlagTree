//===- MemorySpaceUtils.h - Unified memory space helpers --------*- C++ -*-===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Provides a single source of truth for memory space representation across
// all Triton-to-memref conversion passes.
//
// Design: ALL memory spaces use xsmt::MemorySpaceAttr — a custom attribute
// that implements ptr::MemorySpaceAttrInterface.  Unlike IntegerAttr(0),
// MLIR does NOT normalize custom attrs to null, so the memory space is
// preserved on memref types and matches ptr::PtrType's memory space.
//
// The four canonical memory space names match spine-mlir's XSMT dialect
// (see spine-mlir/lib/Common/Dialect/XSMT/IR/XSMT.cc) so that IR emitted
// by spine-triton is accepted by spine-mlir's `memory-space-inference`
// and backend lowering passes:
//
//   global              — module-level / externally visible buffer
//   thread_local        — compiler-managed private scratch/stack space
//   cluster_shared_l2   — cluster-shared L2 scratchpad
//   fragment            — register file / fragment storage
//
// Triton's user-facing `scope` strings (from smt.alloc / smt.alloc_copies)
// are mapped to the canonical names by `scopeToMemorySpace()`:
//
//   "" / "global"  -> global
//   "tcm"          -> thread_local      (TCM is the thread-local scratch)
//   "l2"           -> cluster_shared_l2
//   "fragment"     -> fragment
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H
#define TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTAttrs.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir::triton {

//===----------------------------------------------------------------------===//
// Canonical name helpers (mirror spine-mlir's API)
//===----------------------------------------------------------------------===//

inline llvm::StringRef getGlobalMemorySpaceName() { return "global"; }
inline llvm::StringRef getThreadLocalMemorySpaceName() {
  return "thread_local";
}
inline llvm::StringRef getClusterSharedL2MemorySpaceName() {
  return "cluster_shared_l2";
}
inline llvm::StringRef getFragmentMemorySpaceName() { return "fragment"; }

/// Return true iff `memorySpace` is one of the four canonical names.
inline bool isSupportedMemorySpace(llvm::StringRef memorySpace) {
  return memorySpace == getGlobalMemorySpaceName() ||
         memorySpace == getThreadLocalMemorySpaceName() ||
         memorySpace == getClusterSharedL2MemorySpaceName() ||
         memorySpace == getFragmentMemorySpaceName();
}

/// If `memorySpace` is an xsmt::MemorySpaceAttr, return its value;
/// otherwise return std::nullopt. Used to distinguish "no memory space",
/// "XSMT memory space", and "non-XSMT memory space (e.g. #ptr.generic_space)".
inline std::optional<llvm::StringRef>
getMemorySpaceName(Attribute memorySpace) {
  if (auto xsmtMemorySpace =
          dyn_cast_or_null<xsmt::MemorySpaceAttr>(memorySpace))
    return xsmtMemorySpace.getValue();
  return std::nullopt;
}

inline xsmt::MemorySpaceAttr getGlobalMemorySpaceAttr(MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  return xsmt::MemorySpaceAttr::get(ctx, getGlobalMemorySpaceName());
}

inline xsmt::MemorySpaceAttr getThreadLocalMemorySpaceAttr(MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  return xsmt::MemorySpaceAttr::get(ctx, getThreadLocalMemorySpaceName());
}

inline xsmt::MemorySpaceAttr
getClusterSharedL2MemorySpaceAttr(MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  return xsmt::MemorySpaceAttr::get(ctx, getClusterSharedL2MemorySpaceName());
}

inline xsmt::MemorySpaceAttr getFragmentMemorySpaceAttr(MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  return xsmt::MemorySpaceAttr::get(ctx, getFragmentMemorySpaceName());
}

/// Return a copy of `type` with its memory space replaced by `memorySpace`.
inline MemRefType withMemorySpace(MemRefType type, Attribute memorySpace) {
  return MemRefType::get(type.getShape(), type.getElementType(),
                         type.getLayout(), memorySpace);
}

//===----------------------------------------------------------------------===//
// Bridge / scope-mapping helpers
//===----------------------------------------------------------------------===//

/// Return the canonical "bridge" memory space used when converting Triton
/// pointer types to memref types.  Returns #xsmt.memory_space<global>.
///
/// Every conversion pass that deals with tt.ptr ↔ memref should call this.
/// The returned xsmt::MemorySpaceAttr implements ptr::MemorySpaceAttrInterface,
/// so it is compatible with ptr::PtrType and all ptr dialect operations.
///
/// Note: Ensures the XSMT dialect is loaded in the context.
inline Attribute getDefaultBridgeMemorySpace(MLIRContext *ctx) {
  return getGlobalMemorySpaceAttr(ctx);
}

/// Map a user-facing scope name to a memref memory-space attribute.
///
/// Frontend scope names (from smt.alloc / smt.alloc_copies / BufferType)
/// are translated to the canonical spine-mlir names:
///
///   "" / "global"  -> #xsmt.memory_space<global>
///   "tcm"          -> #xsmt.memory_space<thread_local>
///   "l2"           -> #xsmt.memory_space<cluster_shared_l2>
///   "fragment"     -> #xsmt.memory_space<fragment>
///
/// Unknown scopes fall back to #xsmt.memory_space<global>.
inline Attribute scopeToMemorySpace(llvm::StringRef scope, MLIRContext *ctx) {
  ctx->getOrLoadDialect<xsmt::XSMTDialect>();
  if (scope.empty() || scope == getGlobalMemorySpaceName())
    return getGlobalMemorySpaceAttr(ctx);
  if (scope == "tcm")
    return getThreadLocalMemorySpaceAttr(ctx);
  if (scope == "l2")
    return getClusterSharedL2MemorySpaceAttr(ctx);
  if (scope == getFragmentMemorySpaceName())
    return getFragmentMemorySpaceAttr(ctx);
  // Already-canonical name passed through verbatim.
  if (isSupportedMemorySpace(scope))
    return xsmt::MemorySpaceAttr::get(ctx, scope);
  // Unknown scope — default to global.
  return getGlobalMemorySpaceAttr(ctx);
}

} // namespace mlir::triton

#endif // TRITON_SHARED_UTILS_MEMORYSPACEUTILS_H
