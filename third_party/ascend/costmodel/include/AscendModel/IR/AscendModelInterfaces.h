//===- AscendModelInterfaces.h - AscendModel Op Interfaces -------*- C++
//-*-===//
//
// Defines operation interfaces for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_IR_INTERFACES_H
#define ASCEND_MODEL_IR_INTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace ascend {

// Forward declarations
class HardwareConfig;
enum class HWUnit : uint32_t;

} // namespace ascend
} // namespace mlir

/// Include the generated interface declarations.
/// Use include guard to prevent multiple inclusion.
#ifndef ASCEND_MODEL_INTERFACES_INC_
#define ASCEND_MODEL_INTERFACES_INC_
#include "AscendModel/IR/AscendModelInterfaces.h.inc"
#endif // ASCEND_MODEL_INTERFACES_INC_

#endif // ASCEND_MODEL_IR_INTERFACES_H
