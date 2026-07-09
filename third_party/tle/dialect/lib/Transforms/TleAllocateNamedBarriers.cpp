#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include <algorithm>
#include <optional>

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLEALLOCATENAMEDBARRIERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

#if defined(__HCU__)
namespace {

struct TritonTleAllocateNamedBarriers
    : public impl::TritonTleAllocateNamedBarriersBase<
          TritonTleAllocateNamedBarriers> {
  void runOnOperation() override {}
};

} // namespace
#else
namespace {

constexpr int64_t kFirstVirtualNamedBarrierId = 16;
constexpr int64_t kNumPhysicalNamedBarriers = 16;
// Keep virtual TLE allocations in the historical user-visible range. Existing
// physical id 0 ops are preserved and still block future virtual allocations.
constexpr int64_t kFirstTleAllocatedPhysicalNamedBarrierId = 1;
constexpr int64_t kDefaultWarpGroupBarrierIdx = 0;
constexpr int64_t kSwitchLoopBarrierIdx = 1;
constexpr int64_t kNumWarpSpecializeReservedBarriers = 2;

static std::optional<int64_t> getConstantI32(Value value) {
  if (auto op = value.getDefiningOp<arith::ConstantIntOp>())
    return op.value();
  if (auto op = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto attr = dyn_cast<IntegerAttr>(op.getValue()))
      return attr.getInt();
  }
  return std::nullopt;
}

static void reserveWarpSpecializeBarrierIds(triton::FuncOp func,
                                            llvm::SmallBitVector &reserved) {
  int64_t maxPartitions = 0;
  func.walk([&](ttg::WarpSpecializeOp ws) {
    maxPartitions =
        std::max<int64_t>(maxPartitions, ws.getPartitionRegions().size());
  });

  if (maxPartitions == 0)
    return;

  reserved.set(kDefaultWarpGroupBarrierIdx);
  reserved.set(kSwitchLoopBarrierIdx);
  for (int64_t i = 0; i < maxPartitions; ++i) {
    int64_t id = kNumWarpSpecializeReservedBarriers + i;
    if (id < kNumPhysicalNamedBarriers)
      reserved.set(id);
  }
}

static LogicalResult
collectNamedBarrierId(Operation *op, Value idValue,
                      llvm::SmallBitVector &reserved,
                      llvm::SmallBitVector &used,
                      llvm::MapVector<int64_t, int64_t> &virtualToPhysical) {
  std::optional<int64_t> id = getConstantI32(idValue);
  if (!id)
    return op->emitOpError("requires a constant named barrier id before "
                           "physical allocation");

  if (*id >= kFirstVirtualNamedBarrierId) {
    virtualToPhysical.insert({*id, -1});
    return success();
  }

  if (*id < 0 || *id >= kNumPhysicalNamedBarriers)
    return op->emitOpError("has invalid physical named barrier id ")
           << *id << "; expected 0..15 or a TLE virtual id >= "
           << kFirstVirtualNamedBarrierId;

  // Physical ids are treated as fixed allocations. They may have been created
  // by lower-level NVIDIA passes, while TLE virtual ids are remapped below.
  used.set(*id);
  return success();
}

static LogicalResult
allocateVirtualIds(triton::FuncOp func, llvm::SmallBitVector &reserved,
                   llvm::SmallBitVector &used,
                   llvm::MapVector<int64_t, int64_t> &virtualToPhysical) {
  for (int64_t i = 0; i < kNumPhysicalNamedBarriers; ++i)
    if (reserved.test(i))
      used.set(i);
  for (auto &entry : virtualToPhysical) {
    int64_t physicalId = -1;
    for (int64_t candidate = kFirstTleAllocatedPhysicalNamedBarrierId;
         candidate < kNumPhysicalNamedBarriers; ++candidate) {
      if (!used.test(candidate)) {
        physicalId = candidate;
        break;
      }
    }

    if (physicalId < 0)
      return func.emitError("cannot allocate physical NVIDIA named barrier "
                            "id for virtual TLE named barrier ")
             << entry.first << "; all ids in allocatable range ["
             << kFirstTleAllocatedPhysicalNamedBarrierId << ", "
             << (kNumPhysicalNamedBarriers - 1)
             << "] are already reserved or used";

    entry.second = physicalId;
    used.set(physicalId);
  }
  return success();
}

static void rewriteNamedBarrierId(
    Operation *op, Value idValue,
    const llvm::MapVector<int64_t, int64_t> &virtualToPhysical) {
  std::optional<int64_t> id = getConstantI32(idValue);
  if (!id || *id < kFirstVirtualNamedBarrierId)
    return;

  auto it = virtualToPhysical.find(*id);
  if (it == virtualToPhysical.end())
    return;

  OpBuilder builder(op);
  Value physicalId =
      builder.create<arith::ConstantIntOp>(op->getLoc(), it->second, 32);
  op->setOperand(0, physicalId);
}

static LogicalResult allocateForFunction(triton::FuncOp func) {
  llvm::SmallBitVector reserved(kNumPhysicalNamedBarriers, false);
  llvm::SmallBitVector used(kNumPhysicalNamedBarriers, false);
  llvm::MapVector<int64_t, int64_t> virtualToPhysical;

  reserveWarpSpecializeBarrierIds(func, reserved);

  WalkResult collectResult = func.walk([&](Operation *op) -> WalkResult {
    if (auto wait = dyn_cast<ttng::NamedBarrierWaitOp>(op)) {
      if (failed(collectNamedBarrierId(op, wait.getBar(), reserved, used,
                                       virtualToPhysical)))
        return WalkResult::interrupt();
    } else if (auto arrive = dyn_cast<ttng::NamedBarrierArriveOp>(op)) {
      if (failed(collectNamedBarrierId(op, arrive.getBar(), reserved, used,
                                       virtualToPhysical)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (collectResult.wasInterrupted())
    return failure();

  if (failed(allocateVirtualIds(func, reserved, used, virtualToPhysical)))
    return failure();

  func.walk([&](Operation *op) {
    if (auto wait = dyn_cast<ttng::NamedBarrierWaitOp>(op))
      rewriteNamedBarrierId(op, wait.getBar(), virtualToPhysical);
    else if (auto arrive = dyn_cast<ttng::NamedBarrierArriveOp>(op))
      rewriteNamedBarrierId(op, arrive.getBar(), virtualToPhysical);
  });

  return success();
}

struct TritonTleAllocateNamedBarriers
    : public impl::TritonTleAllocateNamedBarriersBase<
          TritonTleAllocateNamedBarriers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](triton::FuncOp func) -> WalkResult {
      if (failed(allocateForFunction(func)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
#endif

} // namespace mlir::triton::tle
