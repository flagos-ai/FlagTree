// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#pragma once

#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LinearLayout.h"

namespace mlir::triton::tle {

// Choose a canonical thread owner; register deduplication and ordering remain
// the caller's responsibility, including any same-thread repeated stores.
// Replicated reduction results need not be bit-identical after FP contraction.
inline Value sharedStoreOwnerPredicate(Location loc, RewriterBase &rewriter,
                                       const LinearLayout &storeLayout) {
  TritonLLVMOpBuilder b(loc, rewriter);
  auto masks = storeLayout.getFreeVariableMasks();
  auto *ctx = rewriter.getContext();
  int32_t laneMask = masks.lookup(StringAttr::get(ctx, "lane"));
  int32_t warpMask = masks.lookup(StringAttr::get(ctx, "warp"));
  if (laneMask == 0 && warpMask == 0)
    return b.true_val();

  auto [laneId, warpId] = getLaneAndWarpId(rewriter, loc);
  Value predicate = b.true_val();
  if (laneMask != 0)
    predicate = b.icmp_eq(b.and_(laneId, b.i32_val(laneMask)), b.i32_val(0));
  if (warpMask != 0) {
    Value warpOwner =
        b.icmp_eq(b.and_(warpId, b.i32_val(warpMask)), b.i32_val(0));
    predicate = b.and_(predicate, warpOwner);
  }
  return predicate;
}

} // namespace mlir::triton::tle
