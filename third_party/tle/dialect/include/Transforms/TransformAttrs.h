/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

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

} // namespace mlir::triton::tle

#endif // TRITON_TLE_TRANSFORM_ATTRS_H
