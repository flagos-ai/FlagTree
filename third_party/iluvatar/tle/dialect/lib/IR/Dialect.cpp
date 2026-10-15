#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

#include "IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "IR/IluvatarTleAttrDefs.cpp.inc"
#include "IR/OpsEnums.cpp.inc"

using namespace mlir;

namespace mlir::triton::iluvatar_tle {

void IluvatarTleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "IR/IluvatarTleAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "IR/Ops.cpp.inc"
      >();

#ifdef FLAGCX_ENABLED
  addOperations<
#define GET_OP_LIST
#include "IR/FlagCxOps.cpp.inc"
      >();
#endif
}

std::optional<MemoryOrder> parseMemoryOrder(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<MemoryOrder>>(str)
      .Case("relaxed", MemoryOrder::RELAXED)
      .Case("acquire", MemoryOrder::ACQUIRE)
      .Case("release", MemoryOrder::RELEASE)
      .Case("acq_rel", MemoryOrder::ACQ_REL)
      .Case("acqrel", MemoryOrder::ACQ_REL)
      .Default(std::nullopt);
}

} // namespace mlir::triton::iluvatar_tle

#define GET_OP_CLASSES
#include "IR/Ops.cpp.inc"

#ifdef FLAGCX_ENABLED
#define GET_OP_CLASSES
#include "IR/FlagCxOps.cpp.inc"
#endif

#endif // __ILUVATAR_TLE__
