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

enum class CoopKind : int32_t {
  Thread = 0,
  Warp = 1,
  Block = 2,
  TileSpan = 3,
  Lanes = 4,
};

enum class MemoryOrder : int32_t {
  Relaxed = 0,
  Acquire = 1,
  Release = 2,
  AcqRel = 3,
};

namespace mlir::triton::tle {

LogicalResult GetLocalRankOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult DeviceIntraBarrierOp::verify() {
  auto *op = getOperation();

  auto barrierTypeAttr = getBarrierTypeAttr();
  auto coopKindAttr = getCoopKindAttr();
  auto orderAttr = getOrderAttr();

  auto emitInvalidIntAttr = [&](StringRef attrName, int64_t value,
                                StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " (" << value
                             << "), expected one of: " << expected;
  };

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

  // coop_kind
  if (coopKindAttr) {
    auto coopKind = static_cast<CoopKind>(coopKindAttr.getInt());

    switch (coopKind) {
    case CoopKind::Thread:
    case CoopKind::Warp:
    case CoopKind::Block:
    case CoopKind::TileSpan:
    case CoopKind::Lanes:
      break;
    default:
      return emitInvalidIntAttr(
          "coop_kind", coopKindAttr.getInt(),
          "Thread(0), Warp(1), Block(2), TileSpan(3), Lanes(4)");
    }
  }

  // order
  if (orderAttr) {
    auto order = static_cast<MemoryOrder>(orderAttr.getInt());

    switch (order) {
    case MemoryOrder::Relaxed:
    case MemoryOrder::Acquire:
    case MemoryOrder::Release:
    case MemoryOrder::AcqRel:
      break;
    default:
      return emitInvalidIntAttr(
          "order", orderAttr.getInt(),
          "Relaxed(0), Acquire(1), Release(2), AcqRel(3)");
    }
  }

  return success();
}
} // namespace mlir::triton::tle
