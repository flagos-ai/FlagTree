#include "Analysis/Alias.h"
#include "Address/Dialect/IR/AddressDialect.h"
#include "Analysis/Utility.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"

namespace mlir::triton::alias {

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
  // TODO: Check if the memory space is shared memory
  if (isa<addr::ToMemRefOp, memref::TransposeOp>(op))
    return success();

  // Only LocalAllocOp creates a new buffer.
  if (isa<memref::AllocOp>(op)) {
    aliasInfo.insert(result);
    pessimistic = false;
  } else if (isa<memref::ReinterpretCastOp, memref::SubViewOp,
                 memref::ExtractStridedMetadataOp,
                 memref::ExtractAlignedPointerAsIndexOp, memref::CastOp,
                 memref::ReshapeOp, memref::CollapseShapeOp,
                 memref::ExpandShapeOp, mk::BitcastOp>(op)) {
    // FIXME: memref::viewOp and memref::transposeOp
    // FIXME: A common trait or interface to handle all memref op
    // FIXME: memref::SubViewOp may need handled before this analysis.
    aliasInfo = AliasInfo(operands[0]->getValue());
    pessimistic = false;
  } else {
    if (isa<MemRefType>(result.getType())) {
      op->dump();
      fflush(stdout);
    }
    assert(!isa<MemRefType>(result.getType()) &&
           "unknown operation creating memory descriptor");
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

AliasResult SharedMemoryAliasAnalysis::alias(Value lhs, Value rhs) {
  // TODO: implement
  return AliasResult::MayAlias;
}

ModRefResult SharedMemoryAliasAnalysis::getModRef(Operation *op,
                                                  Value location) {
  // TODO: implement
  return ModRefResult::getModAndRef();
}

} // namespace mlir::triton::alias
