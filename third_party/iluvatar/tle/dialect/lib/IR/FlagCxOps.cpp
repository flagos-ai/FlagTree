#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "IR/VerfiyUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

namespace mlir::triton::iluvatar_tle {

LogicalResult GetLocalRankOp::verify() {
  // The FlagCX device ABI takes the communicator handle as a 64-bit integer.
  // Tensor and pointer sources come from the distributed pointer ops and are
  // unpacked during lowering, so only the integer form is constrained here.
  auto srcTy = getSrc().getType();
  if (isa<IntegerType>(srcTy) && !srcTy.isSignlessInteger(64))
    return emitOpError("comm pointer must be represented as i64");

  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult DeviceIntraBarrierOp::verify() {
  auto *op = getOperation();

  auto barrierTypeAttr = getBarrierTypeAttr();

  auto emitInvalidStrAttr = [&](StringRef attrName, StringRef value,
                                StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " '" << value
                             << "', expected one of: " << expected;
  };

  // barrier_type
  if (barrierTypeAttr) {
    StringRef barrierType = barrierTypeAttr.getValue();

    bool valid = llvm::StringSwitch<bool>(barrierType)
                     .Case("arrive", true)
                     .Case("wait", true)
                     .Case("sync", true)
                     .Default(false);

    if (!valid)
      return emitInvalidStrAttr("barrier_type", barrierType,
                                "arrive, wait, sync");
  }

  return success();
}

} // namespace mlir::triton::iluvatar_tle

#endif // __ILUVATAR_TLE__
