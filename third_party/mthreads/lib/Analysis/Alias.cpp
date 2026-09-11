#include "triton/Analysis/Alias.h"

#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#ifdef __TLE__
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/STLExtras.h"
#endif

namespace mlir {

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

#ifdef __TLE__
static bool isTritonPtrLikeType(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return isa<triton::PointerType>(tensorTy.getElementType());
  return false;
}

static AliasInfo
joinOperandAliases(ArrayRef<const dataflow::Lattice<AliasInfo> *> operands) {
  AliasInfo aliasInfo;
  for (auto *operand : operands)
    aliasInfo = AliasInfo::join(aliasInfo, operand->getValue());
  return aliasInfo;
}
#endif // __TLE__

LogicalResult SharedMemoryAliasAnalysis::visitOperation(
    Operation *op, ArrayRef<const dataflow::Lattice<AliasInfo> *> operands,
    ArrayRef<dataflow::Lattice<AliasInfo> *> results) {
#ifdef __TLE__
  if (results.empty())
    return success();

  StringRef opName = op->getName().getStringRef();
  if (opName == "ttmg.squad_dot_wait" || opName == "mtgpu.sqmma_wait") {
    // These waits forward each dependency unchanged; they do not allocate
    // storage. Handle every result, including memdescs after an accumulator.
    if (operands.size() != results.size())
      return op->emitOpError("SQMMA wait alias propagation requires matching "
                             "operand and result counts");
    for (unsigned idx = 0; idx < results.size(); ++idx) {
      if (op->getOperand(idx).getType() != op->getResult(idx).getType())
        return op->emitOpError("SQMMA wait alias propagation requires matching "
                               "operand and result types");
    }
    for (auto [idx, result] : llvm::enumerate(results)) {
      auto memdescTy =
          dyn_cast<triton::gpu::MemDescType>(op->getResult(idx).getType());
      if (memdescTy && isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                           memdescTy.getMemorySpace()))
        propagateIfChanged(result, result->join(operands[idx]->getValue()));
      else
        setToEntryState(result);
    }
    return success();
  }
#endif // __TLE__

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
#ifdef __TLE__
  } else if (op->getName().getStringRef() == "musa_tle.local_pointers" &&
             !operands.empty()) {
    // Treat local pointer views as aliases of their source memdesc.
    aliasInfo = operands[0]->getValue();
    pessimistic = false;
#endif // __TLE__
  } else if (isa<ub::PoisonOp>(op)) {
    aliasInfo = AliasInfo();
    pessimistic = false;
  } else {
    assert(!isa<triton::gpu::MemDescType>(result.getType()) &&
           "unknown operation creating memory descriptor");
  }

  if (pessimistic) {
#ifdef __TLE__
    // Propagate aliases through pointer-producing ops such as
    // tt.splat / tt.broadcast / tt.addptr so that shared buffers
    // allocated via tle.gpu.alloc stay live across pointer arithmetic
    // users (tl.load / tl.store through derived pointers).
    bool propagated = false;
    for (auto [idx, result] : llvm::enumerate(results)) {
      Value value = op->getResult(idx);
      if (isTritonPtrLikeType(value.getType())) {
        AliasInfo ptrAlias = joinOperandAliases(operands);
        propagateIfChanged(result, result->join(ptrAlias));
        propagated = true;
      }
    }
    if (!propagated)
      setAllToEntryStates(results);
#else
    setAllToEntryStates(results);
#endif
    return success();
  }
  // Join all lattice elements
  for (auto *result : results)
    propagateIfChanged(result, result->join(aliasInfo));

  return success();
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
