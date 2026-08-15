#ifndef XSMT_TYPES_H
#define XSMT_TYPES_H

#include <optional>

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Types.h"

namespace mlir {
namespace xsmt {} // namespace xsmt
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.h.inc"

#endif // XSMT_TYPES_H
