//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MK_TO_TX81_CONVERSION_PASSES_H
#define MK_TO_TX81_CONVERSION_PASSES_H

#include "tsingmicro-tx81/Conversion/MKToTx81/MKToTx81.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  MK_TO_TX81_CONVERSION_PASSES_H
