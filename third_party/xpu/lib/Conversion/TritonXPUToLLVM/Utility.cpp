#include "Utility.h"
// LLVM22 compatibility: re-introduce dragon-style free macros (i32_val, etc.).
// Must be included AFTER the upstream Utility.h pulled in above.
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h"

namespace mlir::LLVM::XPU {

Value llGetPid(Location loc, RewriterBase &rewriter, ModuleOp moduleOp,
               int axis) {
  assert(axis >= 0);
  assert(axis < 3);
  assert(moduleOp);
  static constexpr mlir::gpu::Dimension dims[] = {mlir::gpu::Dimension::x,
                                                  mlir::gpu::Dimension::y,
                                                  mlir::gpu::Dimension::z};

  // TODO[dyq]: add Dimension:y & Dimension:z mapping
  Value blockId;
  switch (axis) {
  case 0: {
    blockId = rewriter.create<::mlir::gpu::BlockIdOp>(loc, dims[axis]);
    break;
  }
  case 1:
  case 2: {
    blockId = i32_val(0);
    break;
  }
  default: {
    llvm_unreachable("ProgramIdOp Get Invalid Axis");
  }
  }

  return rewriter.create<arith::IndexCastOp>(loc, i32_ty, blockId);
}

Type getFunctionType(mlir::OpBuilder &builder, ValueRange operands) {
  SmallVector<Type> operandTypes(operands.getTypes());
  mlir::MLIRContext *ctx = builder.getContext();
  auto voidTy = mlir::LLVM::LLVMVoidType::get(ctx);
  return LLVM::LLVMFunctionType::get(voidTy, operandTypes);
}

Value createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                       Operation *op, Type &elemTy, ValueRange &operands,
                       Location &loc) {
  Type funcType = mlir::triton::gpu::getFunctionType(elemTy, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  return rewriter.create<LLVM::CallOp>(loc, funcOp, operands).getResult();
}

void createDeviceCall(StringRef funcName, ConversionPatternRewriter &rewriter,
                      Operation *op, ValueRange &operands, Location &loc) {
  OpBuilder builder(op);
  Type funcType = getFunctionType(builder, operands);
  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, op, funcName, funcType, "", "");
  rewriter.create<LLVM::CallOp>(loc, funcOp, operands);
  return;
}

} // namespace mlir::LLVM::XPU

namespace mlir::LLVM::XPU {

// Ceil-based cluster-layout offset emission, ported from Triton 3.0's
// `triton::xpu::emitOffsetForClusterLayout`. 3.0 used `getShapePerCTATile`
// (= sizePerCore*coresPerGroup*groupsPerCluster per dim) and `getShapePerCTA`
// (= raw shape, since XPU CTASplitNum is all 1). 3.6 dropped the free
// `getShapePerCTATile(Attribute,...)`, so we recompute the per-dim products
// directly here.
SmallVector<SmallVector<unsigned>>
emitOffsetForClusterLayout(const triton::xpu::ClusterLayoutAttr &clusterLayout,
                           RankedTensorType type) {
  auto shape = type.getShape();
  auto sizePerThread = clusterLayout.getSizePerCore();
  auto threadsPerWarp = clusterLayout.getCoresPerGroup();
  auto warpsPerCTA = clusterLayout.getGroupsPerCluster();
  auto order = clusterLayout.getOrder();

  unsigned rank = shape.size();
  // XPU has no inter-CTA concept (CTASplitNum == 1), so shapePerCTA == shape.
  SmallVector<int64_t> shapePerCTA(shape.begin(), shape.end());
  SmallVector<unsigned> shapePerCTATile(rank);
  for (unsigned k = 0; k < rank; ++k)
    shapePerCTATile[k] = sizePerThread[k] * threadsPerWarp[k] * warpsPerCTA[k];

  SmallVector<unsigned> tilesPerDim(rank);
  for (unsigned k = 0; k < rank; ++k)
    tilesPerDim[k] = ceil<unsigned>(shapePerCTA[k], shapePerCTATile[k]);

  unsigned elemsPerThread =
      triton::gpu::getTotalElemsPerThread(type); // ceil-based (XPU dispatch)
  unsigned totalSizePerThread = product<unsigned>(sizePerThread);
  SmallVector<SmallVector<unsigned>> reorderedOffset(elemsPerThread);
  for (unsigned n = 0; n < elemsPerThread; ++n) {
    unsigned linearNanoTileId = n / totalSizePerThread;
    unsigned linearNanoTileElemId = n % totalSizePerThread;
    // 3.6 delinearize(linear, shape, order) is numerically identical to 3.0's
    // getMultiDimIndex(linear, shape, order).
    SmallVector<unsigned> multiDimNanoTileId =
        delinearize(linearNanoTileId, tilesPerDim, order);
    SmallVector<unsigned> multiDimNanoTileElemId =
        delinearize(linearNanoTileElemId, sizePerThread, order);
    for (unsigned k = 0; k < rank; ++k) {
      unsigned reorderedMultiDimId =
          (multiDimNanoTileId[k] * shapePerCTATile[k] +
           multiDimNanoTileElemId[k]) %
          shapePerCTA[k];
      reorderedOffset[n].push_back(reorderedMultiDimId);
    }
  }
  return reorderedOffset;
}

// Slice-layout offset emission for XPU-backed slices, ported from 3.0's
// `emitOffsetForSliceLayout`.
static SmallVector<SmallVector<unsigned>>
emitOffsetForSliceLayoutXPU(const triton::gpu::SliceEncodingAttr &sliceLayout,
                            RankedTensorType type) {
  auto parentEncoding = sliceLayout.getParent();
  unsigned dim = sliceLayout.getDim();
  auto parentShape = sliceLayout.paddedShape(type.getShape());
  RankedTensorType parentTy =
      RankedTensorType::get(parentShape, type.getElementType(), parentEncoding);
  auto parentOffsets = emitOffsetForLayoutXPU(parentEncoding, parentTy);
  if (parentOffsets.empty())
    return {};

  SmallVector<SmallVector<unsigned>> resultOffsets;
  std::set<SmallVector<unsigned>> uniqueOffsets;
  for (unsigned i = 0; i < parentOffsets.size(); ++i) {
    SmallVector<unsigned> offsets(parentOffsets[i].begin(),
                                  parentOffsets[i].end());
    offsets.erase(offsets.begin() + dim);
    if (auto [it, inserted] = uniqueOffsets.insert(offsets); inserted)
      resultOffsets.push_back(offsets);
  }

  // After deduplicating, resultOffsets may have fewer than
  // getTotalElemsPerThread() elements; repeat the sequence to fill.
  int elemsPerThread = triton::gpu::getTotalElemsPerThread(type);
  assert(resultOffsets.size() > 0);
  assert(elemsPerThread % resultOffsets.size() == 0);
  int numRepeats = elemsPerThread / resultOffsets.size();
  SmallVector<SmallVector<unsigned>> ret;
  for (int i = 0; i < numRepeats; ++i)
    for (unsigned j = 0; j < resultOffsets.size(); ++j)
      ret.push_back(SmallVector<unsigned>(resultOffsets[j]));
  return ret;
}

SmallVector<SmallVector<unsigned>>
emitOffsetForLayoutXPU(Attribute layout, RankedTensorType type) {
  if (auto clusterLayout =
          mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(layout))
    return emitOffsetForClusterLayout(clusterLayout, type);
  if (auto sliceLayout = mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(layout))
    return emitOffsetForSliceLayoutXPU(sliceLayout, type);
  // Fall back to the shared LinearLayout-based emitter for any non-XPU layout.
  return ::mlir::emitOffsetForLayout(layout, type);
}

} // namespace mlir::LLVM::XPU
