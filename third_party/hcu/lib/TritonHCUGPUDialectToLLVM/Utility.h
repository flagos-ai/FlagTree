#ifndef TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUDIALECTTOLLVM_UTILITY_H_
#define TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUDIALECTTOLLVM_UTILITY_H_

#include "triton/Tools/LinearLayout.h"

namespace tt = mlir::triton;

namespace mlir::LLVM::HCU {
using ElemLocationKey = SmallVector<std::pair<StringAttr, int32_t>>;

ElemLocationKey getElemCoordinatesFromRegisters(tt::LinearLayout ll,
                                                unsigned regId,
                                                MLIRContext *ctx);

std::optional<int> getRegFromCoordinates(tt::LinearLayout ll,
                                         ElemLocationKey coordinates,
                                         MLIRContext *ctx);

} // namespace mlir::LLVM::HCU
#endif // TRITON_THIRD_PARTY_HCU_LIB_TRITONHCUGPUDIALECTTOLLVM_UTILITY_H_
