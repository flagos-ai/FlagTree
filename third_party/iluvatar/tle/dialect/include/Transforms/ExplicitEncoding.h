// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#ifndef ILUVATAR_TLE_EXPLICIT_ENCODING_H
#define ILUVATAR_TLE_EXPLICIT_ENCODING_H

#include <string>
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::iluvatar_tle {

inline Attribute explicitEncoding(Value value) {
  if (auto result = dyn_cast<OpResult>(value))
    return result.getOwner()->getAttr(
        "tle.explicit_encoding." + std::to_string(result.getResultNumber()));
  return {};
}

inline void preserveExplicitEncoding(Value source, Value destination) {
  if (Attribute encoding = explicitEncoding(source))
    if (auto result = dyn_cast<OpResult>(destination))
      result.getOwner()->setAttr(
          "tle.explicit_encoding." + std::to_string(result.getResultNumber()),
          encoding);
}

inline bool isExplicitLayoutConversion(Operation *op) {
  return isa<gpu::ConvertLayoutOp>(op) &&
         op->hasAttr("tle.explicit_encoding.0");
}

inline void preserveExplicitEncoding(Operation *source, Operation *destination) {
  for (NamedAttribute attr : source->getAttrs())
    if (attr.getName().strref().starts_with("tle.explicit_encoding."))
      destination->setAttr(attr.getName(), attr.getValue());
}

inline Attribute explicitMemoryEncoding(Operation *op) {
  if (Attribute encoding = op->getAttr("tle.explicit_memory_encoding"))
    return encoding;
  for (Value value : op->getResults())
    if (Attribute encoding = explicitEncoding(value))
      return encoding;
  for (Value value : op->getOperands())
    if (Attribute encoding = explicitEncoding(value))
      return encoding;
  return {};
}

} // namespace mlir::triton::iluvatar_tle
#endif
