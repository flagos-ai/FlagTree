#ifndef TRITONMUSA_COMMON_RLC_ATTRS_H
#define TRITONMUSA_COMMON_RLC_ATTRS_H

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::musa {

// Per-thread vector width to preserve when lowering an i32-to-f32 cast.
inline constexpr llvm::StringLiteral kRlcPreserveIntToFpVectorWidthAttr =
    "musa.rlc_preserve_int_to_fp_vector_width";

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_RLC_ATTRS_H
