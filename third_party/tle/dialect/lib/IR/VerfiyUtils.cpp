#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
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
#include <iostream>

namespace mlir::triton::tle {
namespace RemotePointers {
llvm::LogicalResult verifyDeviceSpace(mlir::Value src, mlir::Value result) {
  // flagcxGetIntraPointerC accept raw device pointers represented as signless
  // i64 values.
  if (!src.getType().isSignlessInteger(64))
    return failure();

  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    auto ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
    if (!ptr)
      return failure();
    return success();
  }
  return success();
}
} // namespace RemotePointers

namespace DistributedBarrier {
llvm::LogicalResult verifyDeviceSpace(mlir::Operation *op) {
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

} // namespace DistributedBarrier

} // namespace mlir::triton::tle
