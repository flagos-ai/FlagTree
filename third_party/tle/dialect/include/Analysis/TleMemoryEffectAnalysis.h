#ifndef TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_
#define TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include <optional>

namespace mlir::triton::tle {

enum class PointerAddressClass {
  Global,
  Shared,
  Unknown,
};

PointerAddressClass classifyPointerAddress(Value value);
std::optional<Value> getSharedPointerMemDescRoot(Value ptr);

bool mayReadSharedMemory(Operation *op);
bool mayWriteSharedMemory(Operation *op);
bool mayWriteSharedMemoryAlias(Operation *op, Value memdesc);
bool hasInterveningSharedMemoryWrite(Operation *from, Operation *to);
bool hasInterveningSharedMemoryWriteAlias(Operation *from, Operation *to,
                                          Value memdesc);

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_MEMORY_EFFECT_ANALYSIS_H_
