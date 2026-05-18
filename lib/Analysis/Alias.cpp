#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/LLVM.h"
#ifdef __TLE__
#include "triton/Dialect/Triton/IR/Types.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir {

static bool isSharedMemDescType(Type type) {
  auto memdescTy = dyn_cast<triton::gpu::MemDescType>(type);
  return memdescTy && isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                          memdescTy.getMemorySpace());
}

static bool isMemDescType(Type type) {
  return isa<triton::gpu::MemDescType>(type);
}

#ifdef __TLE__
static bool isTritonPtrLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType>(tensorTy.getElementType());
  return false;
}
#endif

static AliasInfo
joinOperandAliases(ArrayRef<const dataflow::Lattice<AliasInfo> *> operands) {
  AliasInfo aliasInfo;
  for (auto *operand : operands)
    aliasInfo = AliasInfo::join(aliasInfo, operand->getValue());
  return aliasInfo;
}

static bool getSameTypedOperandAlias(
    Operation *op, unsigned resultIdx,
    ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    AliasInfo &aliasInfo) {
  Type resultTy = op->getResult(resultIdx).getType();
  if (resultIdx < op->getNumOperands() &&
      op->getOperand(resultIdx).getType() == resultTy) {
    aliasInfo = operands[resultIdx]->getValue();
    return true;
  }

  std::optional<unsigned> matchingOperand;
  for (unsigned i = 0, e = op->getNumOperands(); i < e; ++i) {
    if (op->getOperand(i).getType() != resultTy)
      continue;
    if (!matchingOperand) {
      matchingOperand = i;
      aliasInfo = operands[i]->getValue();
      continue;
    }
    aliasInfo = AliasInfo::join(aliasInfo, operands[i]->getValue());
  }
  return matchingOperand.has_value();
}

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
  if (results.empty())
    return success();

  // Only LocalAllocOp creates a new buffer.
  if (isa<triton::gpu::LocalAllocOp>(op)) {
    for (auto [value, result] : llvm::zip(op->getResults(), results)) {
      AliasInfo aliasInfo;
      if (isSharedMemDescType(value.getType()))
        aliasInfo.insert(value);
      propagateIfChanged(result, result->join(aliasInfo));
    }
    return success();
  }

  if (op->hasTrait<OpTrait::MemDescViewTrait>()) {
    assert(!operands.empty() && "memdesc view op must have a source operand");
    AliasInfo aliasInfo = operands[0]->getValue();
    for (auto *result : results)
      propagateIfChanged(result, result->join(aliasInfo));
    return success();
  }

#ifdef __TLE__
  if (op->getName().getStringRef() == "tle.local_pointers" &&
      !operands.empty()) {
    // Treat local pointer views as aliases of their source memdesc.
    AliasInfo aliasInfo = operands[0]->getValue();
    for (auto *result : results)
      propagateIfChanged(result, result->join(aliasInfo));
    return success();
  }
#endif

  if (isa<ub::PoisonOp>(op)) {
    for (auto *result : results)
      propagateIfChanged(result, result->join(AliasInfo()));
    return success();
  }

  for (auto [idx, result] : llvm::enumerate(results)) {
    Value value = op->getResult(idx);
    AliasInfo aliasInfo;
    bool propagateAlias = false;

    if (isSharedMemDescType(value.getType())) {
      // Some synchronization ops, e.g. ttng.warp_group_dot_wait, forward
      // memdesc operands as results so later users keep the same buffer live.
      propagateAlias = getSameTypedOperandAlias(op, idx, operands, aliasInfo);
      if (!propagateAlias)
        return op->emitOpError("creates a shared memory descriptor result that "
                               "alias analysis cannot map to an operand");
    } else if (isMemDescType(value.getType())) {
      propagateAlias = true;
#ifdef __TLE__
    } else if (isTritonPtrLikeType(value.getType())) {
      // Propagate aliases through pointer-producing/view-like ops such as
      // tt.splat/tt.broadcast/tt.addptr chains so shared buffers stay live
      // across pointer arithmetic users.
      aliasInfo = joinOperandAliases(operands);
      propagateAlias = true;
#endif
    }

    if (propagateAlias)
      propagateIfChanged(result, result->join(aliasInfo));
    else
      setToEntryState(result);
  }

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
