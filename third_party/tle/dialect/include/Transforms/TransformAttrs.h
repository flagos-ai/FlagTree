// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#ifndef TRITON_TLE_TRANSFORM_ATTRS_H
#define TRITON_TLE_TRANSFORM_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::tle {

// Marks direct async-copy producer ops that originate from TLE local-pointer
// staging canonicalization. Downstream TLE pipelining passes use this
// provenance to distinguish TLE-owned direct-async families from generic
// Triton async-copy loops.
inline constexpr llvm::StringLiteral
    kTleLocalPointerAsyncStoreAttr("tle.local_ptr_async_store");

// Marks a TLE pipe commit whose payload readiness is produced by prior
// cp.async copies. NVWS token lowering uses this to attach copy completion to
// the pipe full barrier instead of forcing a producer-side cp.async wait.
inline constexpr llvm::StringLiteral
    kTlePipeCommitCpAsyncAttr("tle.pipe_commit_cp_async");

// Marks TMA store ops whose commit-group boundary is represented explicitly by
// a following tle.tma_store.commit_group op.
inline constexpr llvm::StringLiteral
    kTleTMAStoreExplicitCommitAttr("tle.tma_store_explicit_commit");

// Marks modules that may use TLE-specific encoding rematerialization hooks in
// native TritonGPU passes.
inline constexpr llvm::StringLiteral kTleEnableEncodingRematerializationAttr(
    "tle.enable_encoding_rematerialization");

// Marks scf.for loops (produced from tle.range(..., reorder=True)) whose loads
// should be clustered ahead of other ops after the LoopUnroll pass unrolls the
// body. Consumed by the TLE reorder-loop-loads hook in the upstream LoopUnroll
// pass. The name matches the frontend attribute set in code_generator.py.
inline constexpr llvm::StringLiteral kTleReorderLoopLoadsAttr("tt.reorder");

} // namespace mlir::triton::tle

#endif // TRITON_TLE_TRANSFORM_ATTRS_H
