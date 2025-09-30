#include "triton/Analysis/Utility.h"

namespace mlir {

bool isMmaToDotSlowShortcut(RankedTensorType &srcTy, RankedTensorType &dstTy) {

  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  if (!srcLayout.isa<triton::gpu::IluvatarMmaEncodingAttr>())
    return false;
  auto mmaLayout = srcLayout.cast<triton::gpu::IluvatarMmaEncodingAttr>();
  if (!dstLayout.isa<triton::gpu::DotOperandEncodingAttr>())
    return false;
  auto dotOperandLayout = dstLayout.cast<triton::gpu::DotOperandEncodingAttr>();
  auto dstParLayout = dotOperandLayout.getParent();
  if (!dstParLayout.isa<triton::gpu::IluvatarMmaEncodingAttr>())
    return false;
  auto dstMmaLayout =
      dstParLayout.dyn_cast<triton::gpu::IluvatarMmaEncodingAttr>();
  return !isMmaToDotShortcut(srcTy, dstTy) &&
         mmaLayout.getVersionMajor() == 1 &&
         dstMmaLayout.getVersionMajor() == 1 &&
         mmaLayout.getWarpsPerCTA()[0] == dstMmaLayout.getWarpsPerCTA()[0] &&
         dotOperandLayout.getOpIdx() == 0 && !srcTy.getElementType().isF32();
}

void getBackwardSliceImplCorex(Operation *op,
                               SetVector<Operation *> *backwardSlice,
                               TransitiveFilter filter,
                               bool omitBlockArguments) {
  if (!op || op->hasTrait<OpTrait::IsIsolatedFromAbove>())
    return;

  // Evaluate whether we should keep this def.
  // This is useful in particular to implement scoping; i.e. return the
  // transitive backwardSlice in the current scope.
  if (filter && !filter(op))
    return;

  for (const auto &en : llvm::enumerate(op->getOperands())) {
    auto operand = en.value();
    if (auto *definingOp = operand.getDefiningOp()) {
      if (backwardSlice->count(definingOp) == 0)
        getBackwardSliceImplCorex(definingOp, backwardSlice, filter,
                                  omitBlockArguments);
    } else if (auto blockArg = operand.dyn_cast<BlockArgument>()) {
      if (omitBlockArguments)
        continue;

      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      // TODO: determine whether we want to recurse backward into the other
      // blocks of parentOp, which are not technically backward unless they flow
      // into us. For now, just bail.
      if (parentOp && backwardSlice->count(parentOp) == 0) {
        // assert(parentOp->getNumRegions() == 1 &&
        //        parentOp->getRegion(0).getBlocks().size() == 1);
        getBackwardSliceImplCorex(parentOp, backwardSlice, filter,
                                  omitBlockArguments);
      }
    } else {
      llvm_unreachable("No definingOp and not a block argument.");
    }
  }

  backwardSlice->insert(op);
}

void getBackwardSliceCorex(Operation *op, SetVector<Operation *> *backwardSlice,
                           TransitiveFilter filter, bool omitBlockArguments) {
  getBackwardSliceImplCorex(op, backwardSlice, filter, omitBlockArguments);

  // Don't insert the top level operation, we just queried on it and don't
  // want it in the results.
  backwardSlice->remove(op);
}

SetVector<Operation *> multiRootGetSlice(Operation *op,
                                         TransitiveFilter backwardFilter,
                                         TransitiveFilter forwardFilter,
                                         bool omitBlockArguments) {
  SetVector<Operation *> slice;
  slice.insert(op);

  unsigned currentIndex = 0;
  SetVector<Operation *> backwardSlice;
  SetVector<Operation *> forwardSlice;
  while (currentIndex != slice.size()) {
    auto *currentOp = (slice)[currentIndex];
    // Compute and insert the backwardSlice starting from currentOp.
    backwardSlice.clear();
    BackwardSliceOptions opt;
    opt.omitBlockArguments = true;
    opt.filter = backwardFilter;
    getBackwardSliceCorex(currentOp, &backwardSlice, opt.filter,
                          opt.omitBlockArguments);
    slice.insert(backwardSlice.begin(), backwardSlice.end());

    // Compute and insert the forwardSlice starting from currentOp.
    forwardSlice.clear();
    getForwardSlice(currentOp, &forwardSlice, forwardFilter);
    slice.insert(forwardSlice.begin(), forwardSlice.end());
    ++currentIndex;
  }
  return multiRootTopologicalSort(slice);
}

}