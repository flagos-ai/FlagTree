#ifndef TRITONMUSA_COMMON_BARRIER_UTILS_H
#define TRITONMUSA_COMMON_BARRIER_UTILS_H

#include "Dialect/MUSA/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::musa {

inline constexpr llvm::StringLiteral kNextBarrierIdAttr = "musa.next_bar_id";
inline constexpr llvm::StringLiteral kMaxBarrierIdAttr = "musa.max_bar_id";
inline constexpr int32_t kMaxBarrierId = 63;

inline ModuleOp getEnclosingModule(Operation *op) {
  if (auto mod = dyn_cast<ModuleOp>(op))
    return mod;
  return op->getParentOfType<ModuleOp>();
}

inline FunctionOpInterface getEnclosingFunction(Operation *op) {
  while (op) {
    if (auto func = dyn_cast<FunctionOpInterface>(op))
      return func;
    op = op->getParentOp();
  }
  return nullptr;
}

inline int32_t getImplicitAsyncBarrierFloor(FunctionOpInterface func) {
  if (!func)
    return 0;

  bool hasAsyncCommitGroup = false;
  func.walk(
      [&](triton::gpu::AsyncCommitGroupOp) { hasAsyncCommitGroup = true; });
  return hasAsyncCommitGroup ? 1 : 0;
}

inline FailureOr<int32_t> reserveBarrierIdRange(Operation *anchorOp,
                                                int32_t numSlots) {
  if (numSlots <= 0 || numSlots > kMaxBarrierId)
    return failure();

  auto func = getEnclosingFunction(anchorOp);
  if (!func)
    return failure();

  auto *ctx = func->getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto nextAttr = func->getAttrOfType<IntegerAttr>(kNextBarrierIdAttr);
  auto maxAttr = func->getAttrOfType<IntegerAttr>(kMaxBarrierIdAttr);
  int32_t current = nextAttr ? static_cast<int32_t>(nextAttr.getInt()) : 0;
  if (!nextAttr && maxAttr)
    current = static_cast<int32_t>(maxAttr.getInt());
  current = std::max(current, getImplicitAsyncBarrierFloor(func));
  int32_t base = current + 1;
  int32_t next = current + numSlots;
  if (next > kMaxBarrierId)
    return failure();

  func->setAttr(kNextBarrierIdAttr, IntegerAttr::get(i32Ty, next));
  int32_t funcMax = maxAttr ? static_cast<int32_t>(maxAttr.getInt()) : 0;
  if (next > funcMax)
    func->setAttr(kMaxBarrierIdAttr, IntegerAttr::get(i32Ty, next));
  return base;
}

inline FailureOr<int32_t> reserveFreshBarrierId(Operation *anchorOp) {
  return reserveBarrierIdRange(anchorOp, /*numSlots=*/1);
}

inline int32_t getReservedBarrierCount(FunctionOpInterface func) {
  auto attr = func->getAttrOfType<IntegerAttr>(kMaxBarrierIdAttr);
  return attr ? static_cast<int32_t>(attr.getInt()) : 0;
}

inline void finalizeBarRecord(FunctionOpInterface func,
                              RewriterBase &rewriter) {
  SmallVector<BarRecordOp> records;
  func->walk([&](BarRecordOp op) { records.push_back(op); });

  int32_t barCount = std::max(getReservedBarrierCount(func),
                              getImplicitAsyncBarrierFloor(func));
  if (barCount <= 0) {
    for (BarRecordOp record : records)
      record.erase();
    func->removeAttr(kNextBarrierIdAttr);
    func->removeAttr(kMaxBarrierIdAttr);
    return;
  }

  auto loc = func.getLoc();
  auto i32Ty = IntegerType::get(func->getContext(), 32);
  func->setAttr(kMaxBarrierIdAttr, IntegerAttr::get(i32Ty, barCount));
  func->removeAttr(kNextBarrierIdAttr);

  for (BarRecordOp record : records)
    record.erase();

  OpBuilder::InsertionGuard guard(rewriter);
  Block &entry = func.getFunctionBody().front();
  rewriter.setInsertionPointToStart(&entry);
  auto count = arith::ConstantIntOp::create(rewriter, loc, barCount, 32);
  BarRecordOp::create(rewriter, loc, count);
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_BARRIER_UTILS_H
