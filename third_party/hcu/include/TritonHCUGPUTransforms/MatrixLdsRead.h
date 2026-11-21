// The N32 panel layout follows Hygon triton-das DsReadMLayout.h (MIT).
// Copyright (c) 2026 Hygon Information Technology Co., Ltd.
// SPDX-License-Identifier: MIT
#ifndef TRITON_HCU_MATRIX_LDS_READ_H
#define TRITON_HCU_MATRIX_LDS_READ_H

#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::HCU {

inline unsigned getMatrixLdsTileK(Type type) {
  if (type.isF16() || type.isBF16())
    return 16;
  if (isa<Float8E4M3FNType>(type))
    return 32;
  return 0;
}

inline bool isMatrixLdsPanelShape(ArrayRef<int64_t> shape, unsigned tileK) {
  return tileK && shape.size() == 2 && shape[0] >= tileK && shape[1] >= 32 &&
         llvm::isPowerOf2_64(shape[0]) && llvm::isPowerOf2_64(shape[1]);
}

// gfx938 LDS panel policy. Physical storage is [N/32, K, 32].
// K-dependent XOR keeps each N vector aligned. In b8, low panel bits also
// skew physical K to distribute stores; the reader must undo this skew.
inline gpu::SharedLinearEncodingAttr
getMatrixLdsPanelEncoding(MLIRContext *ctx, ArrayRef<int64_t> shape,
                          unsigned tileK) {
  bool isB8 = tileK == 32;
  int32_t vec = isB8 ? 16 : 8;
  int32_t perPhase = isB8 ? 1 : 2;
  std::vector<std::vector<int32_t>> bases;
  for (int32_t n = 1; n < 32; n *= 2)
    bases.push_back({0, n});
  for (int32_t k = 1; k < shape[0]; k *= 2)
    bases.push_back({k, vec * ((k / perPhase) & 1)});
  for (int32_t n = 32; n < shape[1]; n *= 2) {
    int32_t k = isB8 && n < 128 ? n / 32 : 0;
    bases.push_back({k, n + vec * ((k / perPhase) & 1)});
  }
  LinearLayout layout({{StringAttr::get(ctx, "offset"), std::move(bases)},
                       {StringAttr::get(ctx, "block"), {}}},
                      standardOutDimNames(ctx, 2));
  return gpu::SharedLinearEncodingAttr::get(ctx, layout, 16);
}

} // namespace mlir::triton::HCU
#endif
