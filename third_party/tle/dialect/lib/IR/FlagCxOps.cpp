#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>

#include "tle/dialect/include/IR/VerfiyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::tle {

LogicalResult GetLocalRankOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult DeviceIntraBarrierOp::verify() {
  auto *op = getOperation();
  auto spaceAttr = op->getAttrOfType<StringAttr>("space");
  if (spaceAttr && spaceAttr.getValue() != "device")
    return success();
  auto emitInvalidAttr = [&](StringRef attrName, StringRef value,
                             StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " '" << value
                             << "', expected one of: " << expected;
  };
  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto barrierTypeAttr = op->getAttrOfType<StringAttr>("barrier_type");
  auto orderAttr = op->getAttrOfType<StringAttr>("order");

  bool kindValid = llvm::StringSwitch<bool>(kindAttr.getValue())
                       .Case("thread", true)
                       .Case("warp", true)
                       .Case("block", true)
                       .Case("tile_span", true)
                       .Case("lanes", true)
                       .Default(false);

  if (!kindValid) {
    return emitInvalidAttr("group_kind", kindAttr.getValue(),
                           "thread, warp, block, tile_span, lanes");
  }
  bool barrierTypeValid = llvm::StringSwitch<bool>(barrierTypeAttr.getValue())
                              .Case("arrive", true)
                              .Case("wait", true)
                              .Case("sync", true)
                              .Default(false);
  if (!barrierTypeValid) {
    return emitInvalidAttr("barrier_type", barrierTypeAttr.getValue(),
                           "arrive, wait, sync");
  }
  bool orderValid = llvm::StringSwitch<bool>(orderAttr.getValue())
                        .Case("acqrel", true)
                        .Case("acquire", true)
                        .Case("release", true)
                        .Case("relaxed", true)
                        .Default(false);
  if (!orderValid) {
    return emitInvalidAttr("order", orderAttr.getValue(),
                           "relaxed, acqrel, acquire, release");
  }
  return success();
}

} // namespace mlir::triton::tle
