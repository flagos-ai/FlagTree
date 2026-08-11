#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "IR/Dialect.cpp.inc"

using namespace mlir;

namespace mlir::triton::iluvatar_tle {

void IluvatarTleDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "IR/Ops.cpp.inc"
      >();
}

} // namespace mlir::triton::iluvatar_tle

#define GET_OP_CLASSES
#include "IR/Ops.cpp.inc"

#endif // __ILUVATAR_TLE__
