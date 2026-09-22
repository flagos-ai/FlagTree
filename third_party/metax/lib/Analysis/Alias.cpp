#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/LLVM.h"
#ifdef __MCTLE__
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {

#ifdef __MCTLE__
// mctle.local_pointers turns a shared buffer into plain !tt.ptr values, and
// every later tt.load/tt.store/tt.atomic_rmw reaches the buffer through
// those pointers. Without the cases below the pointers carry no alias, so
// Allocation sees the buffer's last use at local_pointers itself and hands
// its bytes to reduction/scan scratch and to other local_alloc buffers.
// Ported from lib/Analysis/Alias.cpp (#ifdef __TLE__, tle.local_pointers).
static bool isTritonPtrLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType>(tensorTy.getElementType());
  return false;
}
#endif

AliasInfo AliasInfo::join(const AliasInfo &lhs, const AliasInfo &rhs) {
  if (lhs == rhs)
    return lhs;
  AliasInfo ret;
  for (auto value : lhs.allocs) {
    ret.insert(value);
  }
  for (auto value : rhs.allocs) {
    ret.insert(value);
  }
  return ret;
}

LogicalResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
  AliasInfo aliasInfo;
  bool pessimistic = true;
  auto result = op->getResult(0);
  // skip ops that return memdesc in a different memory space.
  if (auto memdescTy = dyn_cast<triton::gpu::MemDescType>(result.getType())) {
    if (!isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
            memdescTy.getMemorySpace()))
      return success();
  }

  // Only LocalAllocOp creates a new buffer.
  if (isa<triton::gpu::LocalAllocOp>(op)) {
    aliasInfo.insert(result);
    pessimistic = false;
  } else if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else if (isa<ub::PoisonOp>(op)) {
    aliasInfo = AliasInfo();
    pessimistic = false;
#ifdef __MCTLE__
  } else if (op->getName().getStringRef() == "mctle.local_pointers" &&
             !operands.empty()) {
    // Local pointer views alias their source memdesc (operand 0).
    aliasInfo = operands[0]->getValue();
    pessimistic = false;
  } else {
    // Pointer-producing ops (tt.splat / tt.broadcast / tt.addptr chains, and
    // any op with a pointer among several results) inherit the aliases of all
    // operands, result by result: an op whose FIRST result is not a pointer
    // may still return one later. Other results keep the entry state.
    for (auto *operand : operands)
      aliasInfo = AliasInfo::join(aliasInfo, operand->getValue());
    for (auto [idx, res] : llvm::enumerate(results)) {
      Value value = op->getResult(idx);
      if (isTritonPtrLikeType(value.getType())) {
        propagateIfChanged(res, res->join(aliasInfo));
      } else {
        assert(!isa<triton::gpu::MemDescType>(value.getType()) &&
               "unknown operation creating memory descriptor");
        setToEntryState(res);
      }
    }
    return success();
#else
  } else {
    assert(!isa<triton::gpu::MemDescType>(result.getType()) &&
           "unknown operation creating memory descriptor");
#endif
  }

  if (pessimistic) {
    setAllToEntryStates(results);
    return success();
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));

  return success();
}

void SharedMemoryAliasAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<dataflow::Lattice<AliasInfo> *> argLattices, unsigned firstIndex) {
  auto wsOp = dyn_cast<triton::gpu::WarpSpecializePartitionsOp>(op);
  if (!wsOp) {
    setAllToEntryStates(argLattices.take_front(firstIndex));
    setAllToEntryStates(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
    return;
  }

  // Propagate aliases from the parent operation's operands to the block
  // arguments.
  assert(!successor.isParent());
  ProgramPoint *point = getProgramPointAfter(wsOp);

  for (auto [capture, argLattice] :
       llvm::zip(wsOp.getParentOp().getExplicitCaptures(), argLattices)) {
    propagateIfChanged(
        argLattice,
        argLattice->join(getLatticeElementFor(point, capture)->getValue()));
  }
}

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}

} // namespace mlir
