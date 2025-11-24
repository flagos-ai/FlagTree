#include "triton/Analysis/Utility.h"
#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {

using namespace triton::gpu;

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

bool maybeSharedAllocationOp(Operation *op) {
  // TODO(Keren): This function can be replaced by adding
  // MemoryEffectOpInterface. We can then use the MemoryEffectOpInterface to
  // query the memory effects of the op.
  auto *dialect = op->getDialect();
  return dialect &&
         (dialect->getTypeID() == TypeID::get<TritonGPUDialect>() ||
          dialect->getTypeID() == TypeID::get<triton::TritonDialect>() ||
          dialect->getTypeID() == TypeID::get<arith::ArithDialect>() ||
          dialect->getTypeID() == TypeID::get<tensor::TensorDialect>());
}

bool supportMMA(triton::DotOp op, int version) {
  // Refer to mma section for the data type supported by Volta and Hopper
  // Tensor Core in
  // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16
  auto aElemTy = op.getA().getType().getElementType();
  auto bElemTy = op.getB().getType().getElementType();
  if (version == 3) {
    if (triton::tools::getBoolEnv("DISABLE_MMA_V3"))
      return false;
    auto retType = op.getType();
    auto retShapePerCTA = getShapePerCTA(retType);
    auto rank = retShapePerCTA.size();
    auto mod = op->getParentOfType<ModuleOp>();
    int numWarps = TritonGPUDialect::getNumWarps(mod);
    if (!(numWarps % 4 == 0 && retShapePerCTA[rank - 2] % 64 == 0 &&
          retShapePerCTA[rank - 1] % 8 == 0 &&
          (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ() ||
           aElemTy.isInteger(8) || aElemTy.isF16() || aElemTy.isBF16() ||
           aElemTy.isF32()))) {
      return false;
    }
    // We cannot use MMA_V3 if we need to accumulate in F32 within the MMA op.
    if (op.getMaxNumImpreciseAcc() < 32 &&
        (aElemTy.isFloat8E5M2() || aElemTy.isFloat8E4M3FNUZ()) &&
        cast<RankedTensorType>(op.getType()).getElementType().isF32()) {
      return false;
    }
  }
  auto retElemTy =
      op.getResult().getType().cast<RankedTensorType>().getElementType();
  if (retElemTy.isF16()) {
    return false;
  }
  return supportMMA(op.getA(), version) && supportMMA(op.getB(), version);
}

bool supportMMA(Value value, int version) {
  // Tell whether a DotOp support MMA by the operand type(either $a or $b).
  // We cannot get both the operand types(in TypeConverter), here we assume the
  // types of both the operands are identical here.
  assert((version == 1 || version == 2) &&
         "Unexpected MMA layout version found");
  auto elemTy = cast<TensorOrMemDesc>(value.getType()).getElementType();
  return elemTy.isF16() || elemTy.isBF16() || elemTy.isF32() ||
         elemTy.isInteger(8);
}

bool isMmaToMmaShortcut(Attribute srcEncoding, Attribute dstEncoding) {
  auto src = dyn_cast<IluvatarMmaEncodingAttr>(srcEncoding);
  auto dst = dyn_cast<IluvatarMmaEncodingAttr>(dstEncoding);
  if (!src || !dst)
    return false;
  return src.getVersionMinor() == 0 && dst.getVersionMinor() > 0;
}

bool isMmaToDotShortcut(RankedTensorType srcTy, RankedTensorType dstTy) {
  // dot_op<opIdx=0, parent=#mma> = #mma
  // when #mma = MmaEncoding<version=2, warpsPerCTA=[..., 1]>
  auto srcLayout = srcTy.getEncoding();
  auto dstLayout = dstTy.getEncoding();
  auto mmaLayout = mlir::cast<IluvatarMmaEncodingAttr>(srcLayout);
  auto dotOperandLayout = mlir::cast<DotOperandEncodingAttr>(dstLayout);
  return mmaLayout.getWarpsPerCTA()[1] == 1 &&
         dotOperandLayout.getOpIdx() == 0 &&
         dotOperandLayout.getParent() == mmaLayout &&
         !srcTy.getElementType().isF32();
}

} // namespace mlir
