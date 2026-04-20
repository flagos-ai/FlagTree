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

} // namespace mlir::triton::tle

#endif // TRITON_TLE_TRANSFORM_ATTRS_H
