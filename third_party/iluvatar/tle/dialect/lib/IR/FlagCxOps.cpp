#ifdef __ILUVATAR_TLE__

#include "IR/Dialect.h"
#include "IR/VerfiyUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/StringSwitch.h"

using namespace mlir;

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

} // namespace mlir::triton::iluvatar_tle

#endif // __ILUVATAR_TLE__
