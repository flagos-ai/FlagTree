#include "Analysis/TkoOpSemantics.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::tileir;

namespace mlir::triton::tileir::tko_op_semantics {

EffectKind getEffectKind(Operation *op) {
  return llvm::TypeSwitch<Operation *, EffectKind>(op)
      .Case<cuda_tile::LoadPtrTkoOp, cuda_tile::LoadViewTkoOp>(
          [](auto) { return EffectKind::Read; })
      .Case<cuda_tile::StorePtrTkoOp, cuda_tile::StoreViewTkoOp>(
          [](auto) { return EffectKind::Write; })
      .Case<cuda_tile::AtomicRMWTkoOp, cuda_tile::AtomicCASTkoOp,
            cuda_tile::AtomicRedViewTkoOp>(
          [](auto) { return EffectKind::Atomic; })
      .Default(EffectKind::None);
}

bool isMemOp(Operation *op) { return getEffectKind(op) != EffectKind::None; }

bool isWriteLike(Operation *op) {
  EffectKind kind = getEffectKind(op);
  return kind == EffectKind::Write || kind == EffectKind::Atomic;
}

bool isReadOnly(Operation *op) { return getEffectKind(op) == EffectKind::Read; }

bool isAcquireFence(Operation *op) {
  // GDC wait is modeled as acquire: its output orders later local memory ops
  // after predecessor completion. gpu.barrier is a total fence, so it has both
  // acquire and release sides.
  return isa<cuda_tile::GdcWaitTkoOp, mlir::gpu::BarrierOp>(op);
}

bool isReleaseFence(Operation *op) {
  // gpu.barrier is a total fence, so it has both acquire and release sides.
  return isa<mlir::gpu::BarrierOp>(op);
}

bool isDependentLaunchSignal(Operation *op) {
  return isa<cuda_tile::GdcLaunchDependentsTkoOp>(op);
}

bool isTotalFence(Operation *op) { return isa<mlir::gpu::BarrierOp>(op); }

bool hasUserToken(Operation *op) {
  return llvm::TypeSwitch<Operation *, bool>(op)
      .Case<cuda_tile::LoadPtrTkoOp>([](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::StorePtrTkoOp>([](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::AtomicRMWTkoOp>(
          [](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::AtomicCASTkoOp>(
          [](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::AtomicRedViewTkoOp>(
          [](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::LoadViewTkoOp>([](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::StoreViewTkoOp>(
          [](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::GdcWaitTkoOp>([](auto o) { return bool(o.getToken()); })
      .Case<cuda_tile::GdcLaunchDependentsTkoOp>(
          [](auto o) { return bool(o.getToken()); })
      .Default(false);
}

Value getAccessValue(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<cuda_tile::LoadPtrTkoOp>([](auto o) { return o.getSource(); })
      .Case<cuda_tile::StorePtrTkoOp>([](auto o) { return o.getDestination(); })
      .Case<cuda_tile::AtomicRMWTkoOp>([](auto o) { return o.getPointers(); })
      .Case<cuda_tile::AtomicCASTkoOp>([](auto o) { return o.getPointers(); })
      .Case<cuda_tile::AtomicRedViewTkoOp>([](auto o) { return o.getView(); })
      .Case<cuda_tile::LoadViewTkoOp>([](auto o) { return o.getView(); })
      .Case<cuda_tile::StoreViewTkoOp>([](auto o) { return o.getView(); })
      .Default(Value());
}

Value getPassthroughAccessValue(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<cuda_tile::MakeTensorViewOp>([](auto o) { return o.getBase(); })
      .Case<cuda_tile::MakePartitionViewOp>(
          [](auto o) { return o.getTensorView(); })
      .Case<cuda_tile::MakeStridedViewOp>(
          [](auto o) { return o.getTensorView(); })
      .Case<cuda_tile::MakeGatherScatterViewOp>(
          [](auto o) { return o.getTensorView(); })
      .Case<cuda_tile::PtrToPtrOp>([](auto o) { return o.getSource(); })
      .Case<cuda_tile::OffsetOp>([](auto o) { return o.getPtr(); })
      .Case<cuda_tile::AssumeOp>([](auto o) { return o.getValue(); })
      .Case<cuda_tile::BroadcastOp>(
          [](auto o) -> Value { return o->getOperand(0); })
      .Case<cuda_tile::ReshapeOp>(
          [](auto o) -> Value { return o->getOperand(0); })
      .Default(Value());
}

} // namespace mlir::triton::tileir::tko_op_semantics
