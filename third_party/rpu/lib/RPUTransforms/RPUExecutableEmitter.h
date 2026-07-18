#ifndef RPU_EXECUTABLE_EMITTER_H
#define RPU_EXECUTABLE_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace mlir {
namespace rpu {

struct RPUExecutableEmissionResult {
  std::string kernelName;
  std::string sourceKind;
  std::string source;
};

FailureOr<RPUExecutableEmissionResult>
emitRPUDSLFromExecutableModule(ModuleOp module);

} // namespace rpu
} // namespace mlir

#endif // RPU_EXECUTABLE_EMITTER_H
