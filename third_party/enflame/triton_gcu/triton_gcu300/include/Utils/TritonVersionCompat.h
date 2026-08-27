/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #ifndef KURAMA_TRITON_VERSION_COMPAT_H
 #define KURAMA_TRITON_VERSION_COMPAT_H

#include <utility>
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "llvm/Support/ErrorHandling.h"

namespace triton_gcu {
namespace compat {

// Wrapper for the free function getCTALayout(attr) / getCGALayout(attr).
// Also usable as replacement for member function encoding.getCTALayout() /
// encoding.getCGALayout() since typed attrs implicitly convert to Attribute.
inline auto getCGALayout(mlir::Attribute encoding) {
#if TRITON_VERSION >= 37
  return mlir::triton::gpu::getCGALayout(encoding);
#else
  return mlir::triton::gpu::getCTALayout(encoding);
#endif
}

// Wrapper for CTAEncodingAttr::getDefault / CGAEncodingAttr::get1CTALayout.
// Creates the default single-CTA layout for a given rank.
inline auto getDefaultCGALayout(mlir::MLIRContext *ctx, unsigned rank) {
#if TRITON_VERSION == 35
  return mlir::triton::gpu::CTALayoutAttr::getDefault(ctx, rank);
#elif TRITON_VERSION >= 37
  return mlir::triton::gpu::CGAEncodingAttr::get1CTALayout(ctx, rank);
#else
  return mlir::triton::gpu::CTAEncodingAttr::getDefault(ctx, rank);
#endif
}

// Wrapper for CTAEncodingAttr::get(ctx, ll) / CGAEncodingAttr::get(ctx, ll).
// Constructs CGA layout from a LinearLayout.
template <typename LinearLayoutT>
inline auto getCGALayoutFromLL(mlir::MLIRContext *ctx, LinearLayoutT &&ll) {
#if TRITON_VERSION >= 37
  return mlir::triton::gpu::CGAEncodingAttr::get(ctx,
                                           std::forward<LinearLayoutT>(ll));
#elif TRITON_VERSION == 35
  // Triton 3.5 has no LinearLayout-based CTA layout constructor; callers select
  // getCGALayoutFromSplitParams on 3.5, so this overload is never used. Guard it
  // so the (non-dependent) CTAEncodingAttr name below is not parsed on 3.5.
  (void)ctx;
  (void)ll;
  llvm_unreachable("getCGALayoutFromLL is not supported on Triton 3.5");
#else
  return mlir::triton::gpu::CTAEncodingAttr::get(ctx,
                                           std::forward<LinearLayoutT>(ll));
#endif
}

// Wrapper for CTALayoutAttr::get / CTAEncodingAttr::fromSplitParams /
// CGAEncodingAttr::fromSplitParams.
inline auto getCGALayoutFromSplitParams(mlir::MLIRContext *ctx,
                                        llvm::ArrayRef<unsigned> ctasPerCGA,
                                        llvm::ArrayRef<unsigned> ctaSplitNum,
                                        llvm::ArrayRef<unsigned> ctaOrder) {
#if TRITON_VERSION == 35
  return mlir::triton::gpu::CTALayoutAttr::get(ctx, ctasPerCGA, ctaSplitNum,
                                               ctaOrder);
#elif TRITON_VERSION >= 37
  return mlir::triton::gpu::CGAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                             ctaSplitNum, ctaOrder);
#else
  return mlir::triton::gpu::CTAEncodingAttr::fromSplitParams(ctx, ctasPerCGA,
                                                             ctaSplitNum, ctaOrder);
#endif
}

// Wrapper for vector::ScatterOp creation across Triton/LLVM versions.
// Triton 3.7 LLVM adds TypeRange resultTypes (first) and IntegerAttr alignment
// (last) parameters to the ScatterOp builder.
inline void createVectorScatterOp(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::Value base, mlir::ValueRange indices,
                                  mlir::Value indexVec, mlir::Value mask,
                                  mlir::Value valueToStore) {
#if TRITON_VERSION >= 37
  builder.create<mlir::vector::ScatterOp>(loc, mlir::TypeRange{}, base, indices,
                                          indexVec, mask, valueToStore,
                                          /*alignment=*/mlir::IntegerAttr());
#else
  builder.create<mlir::vector::ScatterOp>(loc, base, indices, indexVec, mask,
                                          valueToStore);
#endif
}

// --- WarpSpecialize captures compatibility wrappers ---
// Triton 3.7 moved explicitCaptures from WarpSpecializeOp to
// WarpSpecializePartitionsOp, accessed via wsOp.getPartitionOp().

// Get explicit captures from a triton::gpu::WarpSpecializeOp.
inline mlir::OperandRange getWsExplicitCaptures(
    mlir::triton::gpu::WarpSpecializeOp wsOp) {
#if TRITON_VERSION >= 37
  return wsOp.getPartitionOp().getExplicitCaptures();
#else
  return wsOp.getExplicitCaptures();
#endif
}

// Insert a capture operand into a triton::gpu::WarpSpecializeOp.
inline void insertWsCapture(mlir::triton::gpu::WarpSpecializeOp wsOp,
                            mlir::Value capture) {
#if TRITON_VERSION >= 37
  auto partOp = wsOp.getPartitionOp();
  partOp->insertOperands(partOp.getNumOperands(), capture);
#else
  wsOp->insertOperands(wsOp.getNumOperands(), capture);
#endif
}

// Get explicit captures from a WarpSpecializePartitionsOp (used in analysis).
// In 3.7 the PartitionsOp holds captures directly; in <3.7 they are on the
// parent WarpSpecializeOp.
inline mlir::OperandRange getPartitionsExplicitCaptures(
    mlir::triton::gpu::WarpSpecializePartitionsOp partOp) {
#if TRITON_VERSION >= 37
  return partOp.getExplicitCaptures();
#else
  return partOp.getParentOp().getExplicitCaptures();
#endif
}

} // namespace compat
} // namespace triton_gcu
#endif
