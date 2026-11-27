#include "TritonAMDGPUToLLVM/MembarUtility.h"
#include "AsyncUtility.h"
#include "Dialect/TritonAMDGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef __TLE__
#include "mlir/IR/SymbolTable.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/TypeSwitch.h"
#endif

namespace mlir::triton::AMD {
namespace {
// Returns true if one of the operands is a LocalLoad synced via AsyncWait.
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2) {
  auto isAsyncLoad = [](Operation *op) {
    return llvm::isa<triton::gpu::AsyncCopyGlobalToLocalOp,
                     triton::amdgpu::BufferLoadToLocalOp>(op);
  };
  auto isLocalLoadWithAsyncWaitToken = [](Operation *op) {
    auto localLoad = llvm::dyn_cast<triton::gpu::LocalLoadOp>(op);
    return localLoad && isSyncedViaAsyncWait(localLoad);
  };

  // Early return if neither or both operands are an AsyncLoad
  if (isAsyncLoad(op1) == isAsyncLoad(op2)) {
    return false;
  }

  return isLocalLoadWithAsyncWaitToken(op1) ||
         isLocalLoadWithAsyncWaitToken(op2);
};

bool filterLDSMemoryBarriersDependencies(Operation *op1, Operation *op2) {
  auto isLDSMemoryBarrierOp = [](Operation *op) {
    return llvm::isa<triton::amdgpu::InitBarrierOp,
                     triton::amdgpu::ArriveBarrierOp,
                     triton::amdgpu::AsyncCopyMbarrierArriveOp,
                     triton::amdgpu::WaitBarrierOp>(op);
  };

  return (isLDSMemoryBarrierOp(op1) && isLDSMemoryBarrierOp(op2));
}

#ifdef __TLE__
// Only accesses with a single, explicit LDS descriptor are considered. Keep
// scratch and unknown pointer effects in the conservative analysis.
Value getAccessDescriptor(Operation *op) {
  return llvm::TypeSwitch<Operation *, Value>(op)
      .Case<gpu::LocalLoadOp>([](auto load) { return load.getSrc(); })
      .Case<gpu::LocalStoreOp>([](auto store) { return store.getDst(); })
      .Case<amdgpu::WaitBarrierOp, amdgpu::ArriveBarrierOp>(
          [](auto barrier) { return barrier.getAlloc(); })
      .Default([](Operation *) { return Value(); });
}

Value stripSlotIndex(Value descriptor) {
  while (auto index = descriptor.getDefiningOp<gpu::MemDescIndexOp>())
    descriptor = index.getSrc();
  return descriptor;
}

bool isEntryArgument(Value value, triton::FuncOp function) {
  auto argument = dyn_cast<BlockArgument>(value);
  return argument && argument.getOwner() == &function.getBody().front();
}

triton::CallOp getOnlyCallSite(triton::FuncOp function,
                               const SymbolUserMap &users) {
  ArrayRef<Operation *> callSites = users.getUsers(function);
  if (!function.isPrivate() || callSites.size() != 1)
    return {};
  return dyn_cast<triton::CallOp>(callSites.front());
}

// Outlined pipe helpers borrow the payload and barrier arrays from the same
// caller. Compare their allocations in that caller's LDS frame. A dynamic
// slot cannot escape its allocation, so the whole allocation suffices.
bool haveDisjointBorrowedAllocations(Value lhs, Value rhs,
                                     triton::FuncOp function,
                                     ModuleAllocation &allocation,
                                     const SymbolUserMap &users) {
  while (true) {
    lhs = stripSlotIndex(lhs);
    rhs = stripSlotIndex(rhs);
    if (lhs == rhs)
      return false;

    auto lhsAlloc = lhs.getDefiningOp<gpu::LocalAllocOp>();
    auto rhsAlloc = rhs.getDefiningOp<gpu::LocalAllocOp>();
    if (lhsAlloc && rhsAlloc) {
      auto *frame = allocation.getFuncData(function);
      auto lhsId = frame->getBufferId(lhs);
      auto rhsId = frame->getBufferId(rhs);
      auto lhsOffset =
          lhsAlloc->getAttrOfType<IntegerAttr>("allocation.offset");
      auto rhsOffset =
          rhsAlloc->getAttrOfType<IntegerAttr>("allocation.offset");
      if (lhsId == Allocation::InvalidBufferId ||
          rhsId == Allocation::InvalidBufferId || !lhsOffset || !rhsOffset)
        return false;
      size_t lhsStart = lhsOffset.getInt();
      size_t rhsStart = rhsOffset.getInt();
      Interval lhsRange(lhsStart, lhsStart + frame->getAllocatedSize(lhsId));
      Interval rhsRange(rhsStart, rhsStart + frame->getAllocatedSize(rhsId));
      return !lhsRange.intersects(rhsRange);
    }

    auto call = getOnlyCallSite(function, users);
    if (!call || !isEntryArgument(lhs, function) ||
        !isEntryArgument(rhs, function))
      return false;
    lhs = call.getOperand(cast<BlockArgument>(lhs).getArgNumber());
    rhs = call.getOperand(cast<BlockArgument>(rhs).getArgNumber());
    function = call->getParentOfType<triton::FuncOp>();
  }
}

// A single wave revisiting a fixed ring slot executes the same store with the
// same lane mapping. Distinct slots cannot overlap. Leave potentially
// replicated stores across different waves to ordinary dependency analysis.
bool isSameSlotStore(Operation *lhs, Operation *rhs, triton::FuncOp function,
                     const SymbolUserMap &users) {
  auto store = dyn_cast<gpu::LocalStoreOp>(lhs);
  if (lhs != rhs || !store || gpu::lookupNumWarps(store) != 1)
    return false;
  auto index = store.getDst().getDefiningOp<gpu::MemDescIndexOp>();
  if (!index)
    return false;
  if (!isa<gpu::SwizzledSharedEncodingAttr, gpu::NVMMASharedEncodingAttr>(
          index.getType().getEncoding()))
    return false;
  // The call must use a fixed backing allocation. An incoming subview could
  // move the ring between calls and cause partially overlapping slots.
  Value ring = index.getSrc();
  auto call = getOnlyCallSite(function, users);
  if (!call || !isEntryArgument(ring, function))
    return false;
  Value actualRing = call.getOperand(cast<BlockArgument>(ring).getArgNumber());
  auto alloc = actualRing.getDefiningOp<gpu::LocalAllocOp>();
  auto caller = call->getParentOfType<triton::FuncOp>();
  return alloc && alloc->getBlock() == &caller.getBody().front();
}
#endif
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return (filterAsyncLocalLoadsDependencies(op1, op2) ||
          filterLDSMemoryBarriersDependencies(op1, op2));
}

#ifdef __TLE__
MembarFilterFn makeMembarFilter(const SymbolUserMap &users,
                                ModuleAllocation &allocation) {
  return [&users, &allocation](Operation *lhs, Operation *rhs) {
    if (membarFilter(lhs, rhs))
      return true;
    auto function = lhs->getParentOfType<triton::FuncOp>();
    if (!function || function != rhs->getParentOfType<triton::FuncOp>())
      return false;
    Value lhsDescriptor = getAccessDescriptor(lhs);
    Value rhsDescriptor = getAccessDescriptor(rhs);
    return lhsDescriptor && rhsDescriptor &&
           (isSameSlotStore(lhs, rhs, function, users) ||
            haveDisjointBorrowedAllocations(lhsDescriptor, rhsDescriptor,
                                            function, allocation, users));
  };
}
#endif
} // namespace mlir::triton::AMD
