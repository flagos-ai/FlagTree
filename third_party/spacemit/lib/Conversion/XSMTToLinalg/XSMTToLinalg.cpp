//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "xsmt-to-linalg"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::memref;
using namespace mlir::tensor;
using namespace mlir::bufferization;
using namespace mlir::xsmt;

#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

// ===----------------------------------------------------------------------===//
// Common helpers (used by multiple patterns)
// ===----------------------------------------------------------------------===//

Value createZeroConstant(PatternRewriter &rewriter, Location loc,
                         Type elementType) {
  auto zeroAttr = rewriter.getZeroAttr(elementType);
  if (!zeroAttr)
    return nullptr;
  return arith::ConstantOp::create(rewriter, loc, elementType, zeroAttr);
}

// ===----------------------------------------------------------------------===//
// Subview elision helpers (XSMT-specific; generic OFR helpers live in
// OpFoldResultUtils.h)
// ===----------------------------------------------------------------------===//

using mlir::isConstOneOFR;
using mlir::isConstZeroOFR;
using mlir::sameOFR;

static bool canElideDescriptorSubview(Value rankedMemRef, Value off0,
                                      Value off1, Value size0, Value size1) {
  (void)size0;
  (void)size1;
  if (!isConstZeroOFR(off0) || !isConstZeroOFR(off1))
    return false;

  auto memrefTy = llvm::dyn_cast<MemRefType>(rankedMemRef.getType());
  return memrefTy && memrefTy.getRank() == 2;
}

static bool canElideIdentitySubview(Value memrefBaseValue,
                                    ArrayRef<OpFoldResult> sizes,
                                    ArrayRef<OpFoldResult> offsets,
                                    ArrayRef<OpFoldResult> strides) {
  auto memrefTy = llvm::dyn_cast<MemRefType>(memrefBaseValue.getType());
  if (!memrefTy)
    return false;
  if (memrefTy.getRank() != static_cast<int64_t>(offsets.size()) ||
      memrefTy.getRank() != static_cast<int64_t>(strides.size()) ||
      memrefTy.getRank() != static_cast<int64_t>(sizes.size()))
    return false;

  if (!llvm::all_of(offsets,
                    [](OpFoldResult ofr) { return isConstZeroOFR(ofr); }) ||
      !llvm::all_of(strides,
                    [](OpFoldResult ofr) { return isConstOneOFR(ofr); }))
    return false;

  if (auto reinterpretCast =
          memrefBaseValue.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto baseSizes = reinterpretCast.getMixedSizes();
    if (baseSizes.size() != sizes.size())
      return false;
    return llvm::all_of(llvm::zip(sizes, baseSizes), [](auto it) {
      return sameOFR(std::get<0>(it), std::get<1>(it));
    });
  }

  return true;
}

// ===----------------------------------------------------------------------===//
// Loop detection helper
// ===----------------------------------------------------------------------===//

inline bool isInsideLoop(mlir::Operation *op) {
  mlir::Operation *parent = op->getParentOp();
  while (parent) {
    if (llvm::isa<mlir::scf::ForOp, mlir::scf::WhileOp, mlir::scf::ParallelOp>(
            parent)) {
      return true;
    }
    parent = parent->getParentOp();
  }
  return false;
}

// ===----------------------------------------------------------------------===//
// Phase 1: Validation patterns
// ===----------------------------------------------------------------------===//

template <typename OpTy>
struct MBarrierLoopCheckPattern : public OpRewritePattern<OpTy> {
  using mlir::OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Operation *currentParent = op->getParentOp();

    while (currentParent) {
      if (isInsideLoop(op.getOperation())) {
        return op.emitError() << "'" << op->getName()
                              << "' operation cannot be used inside a loop. ";
      }
      if (isa<FunctionOpInterface>(currentParent)) {
        break;
      }
      currentParent = currentParent->getParentOp();
    }
    return failure();
  }
};

template <typename OpTy>
struct CheckGlobalMBarrierInLoopPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Value barrier = op.getMbarrier();

    Operation *definingOp = barrier.getDefiningOp();
    if (!definingOp) {
      return failure();
    }

    if (!llvm::isa<xsmt::GlobalMBarrierInitOp>(definingOp)) {
      return failure();
    }

    if (!isInsideLoop(op.getOperation())) {
      return failure();
    }

    op.emitError()
        << "'" << op->getName() << "' operation with barrier defined by "
        << "'xsmt.global_mbarrier_init' cannot be used inside a loop. "
        << "Global barriers are not designed for loop-level synchronization.";

    return failure();
  }
};

// ===----------------------------------------------------------------------===//
// Phase 2: XSMT → Linalg conversion patterns
// (DescriptorLoadViewOp, MMT4D, GetThread, Proton, MBarrier release)
// ===----------------------------------------------------------------------===//

// ===----------------------------------------------------------------------===//
// PackOpPattern: xsmt.pack (2D → 4D)
// Extracted from ViewOpPattern::convertFillDirect + convertNormal
// ===----------------------------------------------------------------------===//
struct PackOpPattern : public OpRewritePattern<xsmt::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::PackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value base = op.getBase();
    ValueRange offsets = op.getOffsets();
    ArrayRef<int32_t> shape = op.getShape();
    ArrayRef<int32_t> packedSize = op.getPackedSize();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultType.getElementType();
    auto baseType = cast<RankedTensorType>(base.getType());

    if (shape.size() != 2 || packedSize.size() != 2 || offsets.size() != 2)
      return rewriter.notifyMatchFailure(
          op, "Expected 2D shape/packed_size/offsets");

    SmallVector<OpFoldResult> offsetValues;
    for (Value off : offsets)
      offsetValues.push_back(ensureIndexType(loc, off, rewriter));

    // FillOp fast path: skip alloc+materialize roundtrip
    if (auto fillOp = base.getDefiningOp<linalg::FillOp>()) {
      auto tensorType = cast<RankedTensorType>(fillOp.getResult(0).getType());
      SmallVector<OpFoldResult> sizes;
      for (int64_t s : tensorType.getShape())
        sizes.push_back(rewriter.getIndexAttr(s));

      auto sliceInfo =
          computeClampedSlice2D(loc, sizes, offsetValues, shape, rewriter);

      SmallVector<int64_t, 2> dynShape = {ShapedType::kDynamic,
                                          ShapedType::kDynamic};
      auto sliceType = RankedTensorType::get(dynShape, elementType);
      Value slice = tensor::ExtractSliceOp::create(
          rewriter, loc, sliceType, base,
          ArrayRef<OpFoldResult>{sliceInfo.offsetValues[0],
                                 sliceInfo.offsetValues[1]},
          ArrayRef<OpFoldResult>{sliceInfo.sizeRows, sliceInfo.sizeCols},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                 rewriter.getIndexAttr(1)});

      FailureOr<Value> packed = pack2DToPackedTiles(
          loc, slice, elementType, packedSize[0], packedSize[1],
          sliceInfo.sizeRows, sliceInfo.sizeCols, rewriter, true);
      if (failed(packed))
        return rewriter.notifyMatchFailure(op, "Unsupported element type");

      rewriter.replaceOp(
          op, tensor::CastOp::create(rewriter, loc, resultType, *packed));
      return success();
    }

    // Standard path: extract_slice from base tensor + linalg.pack
    // Use destination if provided (DPS), otherwise use base directly.
    Value sourceTensor = op.getDestination() ? op.getDestination() : base;
    auto sourceType = cast<RankedTensorType>(sourceTensor.getType());
    SmallVector<OpFoldResult> sizes;
    for (int64_t s : sourceType.getShape())
      sizes.push_back(rewriter.getIndexAttr(s));

    auto sliceInfo =
        computeClampedSlice2D(loc, sizes, offsetValues, shape, rewriter);

    SmallVector<int64_t, 2> dynShape = {ShapedType::kDynamic,
                                        ShapedType::kDynamic};
    auto sliceType = RankedTensorType::get(dynShape, elementType);
    Value tensor = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, sourceTensor,
        ArrayRef<OpFoldResult>{sliceInfo.offsetValues[0],
                               sliceInfo.offsetValues[1]},
        ArrayRef<OpFoldResult>{sliceInfo.sizeRows, sliceInfo.sizeCols},
        ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                               rewriter.getIndexAttr(1)});

    FailureOr<Value> packed = pack2DToPackedTiles(
        loc, tensor, elementType, packedSize[0], packedSize[1],
        sliceInfo.sizeRows, sliceInfo.sizeCols, rewriter, true);
    if (failed(packed))
      return rewriter.notifyMatchFailure(op, "Unsupported element type");

    rewriter.replaceOp(
        op, tensor::CastOp::create(rewriter, loc, resultType, *packed));
    return success();
  }

private:
  struct ClampedViewSlice2D {
    SmallVector<OpFoldResult> offsetValues;
    Value sizeRows;
    Value sizeCols;
  };

  ClampedViewSlice2D computeClampedSlice2D(Location loc,
                                           ArrayRef<OpFoldResult> sizes,
                                           ArrayRef<OpFoldResult> offsetValues,
                                           ArrayRef<int32_t> shape,
                                           PatternRewriter &rewriter) const {
    auto rowDelta = subOFRs(sizes[0], offsetValues[0], loc, rewriter);
    Value rowBase = ofrToIndexValue(rowDelta, loc, rewriter);
    Value shapeRows = ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value sizeRows = MinSIOp::create(rewriter, loc, rowBase, shapeRows);

    auto colDelta = subOFRs(sizes[1], offsetValues[1], loc, rewriter);
    Value colBase = ofrToIndexValue(colDelta, loc, rewriter);
    Value shapeCols = ConstantIndexOp::create(rewriter, loc, shape[1]);
    Value sizeCols = MinSIOp::create(rewriter, loc, colBase, shapeCols);

    return {SmallVector<OpFoldResult>(offsetValues.begin(), offsetValues.end()),
            sizeRows, sizeCols};
  }

  FailureOr<Value> pack2DToPackedTiles(Location loc, Value source2D,
                                       Type elementType, int32_t tileRows,
                                       int32_t tileCols, Value rows, Value cols,
                                       PatternRewriter &rewriter,
                                       bool withOuterDimsPerm) const {
    Value tileRowsValue =
        arith::ConstantIndexOp::create(rewriter, loc, tileRows);
    Value tileColsValue =
        arith::ConstantIndexOp::create(rewriter, loc, tileCols);
    Value outerRows = createCeilDivUI(rewriter, loc, rows, tileRowsValue);
    Value outerCols = createCeilDivUI(rewriter, loc, cols, tileColsValue);

    SmallVector<int64_t> packedShape = {
        ShapedType::kDynamic, ShapedType::kDynamic, tileRows, tileCols};
    auto packedTy = RankedTensorType::get(packedShape, elementType);
    Value packedEmpty = tensor::EmptyOp::create(rewriter, loc, packedTy,
                                                {outerRows, outerCols});

    Value padding = createZeroConstant(rewriter, loc, elementType);
    if (!padding)
      return failure();

    if (withOuterDimsPerm) {
      return linalg::PackOp::create(
                 rewriter, loc, source2D, packedEmpty, ArrayRef<int64_t>{0, 1},
                 ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileRows),
                                        rewriter.getIndexAttr(tileCols)},
                 padding, ArrayRef<int64_t>{0, 1})
          .getResult();
    }
    return linalg::PackOp::create(
               rewriter, loc, source2D, packedEmpty, ArrayRef<int64_t>{0, 1},
               ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileRows),
                                      rewriter.getIndexAttr(tileCols)},
               padding)
        .getResult();
  }
};

// ===----------------------------------------------------------------------===//
// UnpackOpPattern: xsmt.unpack (4D → 2D)
// Extracted from ViewOpPattern::convertWithDifferentPackedSize
// (packedSize=={1,1})
// ===----------------------------------------------------------------------===//
struct UnpackOpPattern : public OpRewritePattern<xsmt::UnpackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::UnpackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value base = op.getBase();
    ValueRange offsets = op.getOffsets();
    ArrayRef<int32_t> shape = op.getShape();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultType.getElementType();
    auto baseType = cast<RankedTensorType>(base.getType());

    if (baseType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "Expected 4D input for unpack");

    int64_t tileRows = baseType.getDimSize(2);
    int64_t tileCols = baseType.getDimSize(3);

    // Use base directly as the 4D tensor source (no alloc roundtrip needed).
    Value packedTensor = op.getDestination() ? op.getDestination() : base;

    // Unpack 4D → 2D
    Value outerRows = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
    Value outerCols = tensor::DimOp::create(rewriter, loc, packedTensor, 1);
    Value flatRows = arith::MulIOp::create(
        rewriter, loc, outerRows,
        arith::ConstantIndexOp::create(rewriter, loc, tileRows));
    Value flatCols = arith::MulIOp::create(
        rewriter, loc, outerCols,
        arith::ConstantIndexOp::create(rewriter, loc, tileCols));

    auto flatTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value flatEmpty =
        tensor::EmptyOp::create(rewriter, loc, flatTy, {flatRows, flatCols});

    Value flatTensor =
        linalg::UnPackOp::create(
            rewriter, loc, packedTensor, flatEmpty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileRows),
                                   rewriter.getIndexAttr(tileCols)})
            .getResult();

    // Extract slice to target shape
    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);
    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

    Value srcRows = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
    Value srcCols = tensor::DimOp::create(rewriter, loc, flatTensor, 1);
    Value availRows = arith::SubIOp::create(rewriter, loc, srcRows, off0);
    Value availCols = arith::SubIOp::create(rewriter, loc, srcCols, off1);
    Value actualRows =
        arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols =
        arith::MinUIOp::create(rewriter, loc, reqCols, availCols);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols}, ArrayRef<Value>{c1, c1});

    rewriter.replaceOp(
        op, tensor::CastOp::create(rewriter, loc, resultType, slice));
    return success();
  }
};

// ===----------------------------------------------------------------------===//
// RepackOpPattern: xsmt.repack (4D → 4D with different packed)
// Extracted from ViewOpPattern::convertWithDifferentPackedSize
// (packedSize!={1,1})
// ===----------------------------------------------------------------------===//
struct RepackOpPattern : public OpRewritePattern<xsmt::RepackOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::RepackOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value base = op.getBase();
    ValueRange offsets = op.getOffsets();
    ArrayRef<int32_t> shape = op.getShape();
    ArrayRef<int32_t> packedSize = op.getPackedSize();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Type elementType = resultType.getElementType();
    auto baseType = cast<RankedTensorType>(base.getType());

    if (baseType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "Expected 4D input for repack");

    int64_t oldTileRows = baseType.getDimSize(2);
    int64_t oldTileCols = baseType.getDimSize(3);

    // Use base directly as the 4D tensor source (no alloc roundtrip needed).
    Value packedTensor = op.getDestination() ? op.getDestination() : base;

    // Step 1: Unpack old 4D → 2D
    Value outerRows = tensor::DimOp::create(rewriter, loc, packedTensor, 0);
    Value outerCols = tensor::DimOp::create(rewriter, loc, packedTensor, 1);
    Value flatRows = arith::MulIOp::create(
        rewriter, loc, outerRows,
        arith::ConstantIndexOp::create(rewriter, loc, oldTileRows));
    Value flatCols = arith::MulIOp::create(
        rewriter, loc, outerCols,
        arith::ConstantIndexOp::create(rewriter, loc, oldTileCols));

    auto flatTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value flatEmpty =
        tensor::EmptyOp::create(rewriter, loc, flatTy, {flatRows, flatCols});

    Value flatTensor =
        linalg::UnPackOp::create(
            rewriter, loc, packedTensor, flatEmpty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(oldTileRows),
                                   rewriter.getIndexAttr(oldTileCols)})
            .getResult();

    // Step 2: Extract slice
    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);
    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shape[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shape[1]);

    Value srcRows = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
    Value srcCols = tensor::DimOp::create(rewriter, loc, flatTensor, 1);
    Value availRows = arith::SubIOp::create(rewriter, loc, srcRows, off0);
    Value availCols = arith::SubIOp::create(rewriter, loc, srcCols, off1);
    Value actualRows =
        arith::MinUIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols =
        arith::MinUIOp::create(rewriter, loc, reqCols, availCols);
    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols}, ArrayRef<Value>{c1, c1});

    // Step 3: Repack with new packed tiles
    Value newTileRowsVal =
        arith::ConstantIndexOp::create(rewriter, loc, packedSize[0]);
    Value newTileColsVal =
        arith::ConstantIndexOp::create(rewriter, loc, packedSize[1]);
    Value newOuterRows =
        createCeilDivUI(rewriter, loc, actualRows, newTileRowsVal);
    Value newOuterCols =
        createCeilDivUI(rewriter, loc, actualCols, newTileColsVal);

    SmallVector<int64_t> packedShape = {ShapedType::kDynamic,
                                        ShapedType::kDynamic, packedSize[0],
                                        packedSize[1]};
    auto packedTy = RankedTensorType::get(packedShape, elementType);
    Value packedEmpty = tensor::EmptyOp::create(rewriter, loc, packedTy,
                                                {newOuterRows, newOuterCols});

    Value padding = createZeroConstant(rewriter, loc, elementType);
    if (!padding)
      return rewriter.notifyMatchFailure(op, "Cannot create padding value");

    Value repacked =
        linalg::PackOp::create(
            rewriter, loc, slice, packedEmpty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(packedSize[0]),
                                   rewriter.getIndexAttr(packedSize[1])},
            padding)
            .getResult();

    rewriter.replaceOp(
        op, tensor::CastOp::create(rewriter, loc, resultType, repacked));
    return success();
  }
};

struct LowerXSMTMMT4D : public OpRewritePattern<xsmt::MMT4DOp> {
  using OpRewritePattern::OpRewritePattern;

  /// Returns true if the memref type is known to be contiguous in the last
  /// dimension (i.e. last stride == 1). We use this as a conservative guard for
  /// the "unpack directly into output buffer" optimization because downstream
  /// lowering (e.g. spestruct.unpack) may require the destination/result memref
  /// types to match exactly.
  static bool hasUnitStrideInLastDim(MemRefType ty) {
    if (!ty.hasStaticShape()) {
      // Shape dynamic is fine; we only care about layout.
    }
    if (!ty.getLayout().isIdentity()) {
      int64_t offset = 0;
      SmallVector<int64_t, 4> strides;
      if (failed(ty.getStridesAndOffset(strides, offset)))
        return false;
      if (strides.empty())
        return false;
      return strides.back() == 1;
    }
    // Identity layout implies last-dim stride is 1.
    return true;
  }

  /// Try to find a linalg.generic(addf) user that wraps this mmt4d result
  /// as an accumulation pattern: generic(addf(mmt4d_result, acc)).
  /// Returns the genericOp if found, nullptr otherwise.
  static linalg::GenericOp findAccumulationGeneric(xsmt::MMT4DOp op) {
    Value result = op.getResult();
    // The mmt4d must have no accumulator (2 operands: a, b)
    if (op.getNumOperands() != 2)
      return nullptr;

    for (OpOperand &use : result.getUses()) {
      auto genericOp = dyn_cast<linalg::GenericOp>(use.getOwner());
      if (!genericOp)
        continue;

      // Check: all parallel iterators
      bool allParallel = true;
      for (auto it : genericOp.getIteratorTypesArray()) {
        if (it != utils::IteratorType::parallel) {
          allParallel = false;
          break;
        }
      }
      if (!allParallel)
        continue;

      // Check: 3 indexing maps (2 inputs + 1 output)
      if (genericOp.getIndexingMaps().size() != 3)
        continue;

      // Check: single addf in body
      auto &block = genericOp.getRegion().front();
      if (!llvm::hasSingleElement(block.without_terminator()))
        continue;
      auto addOp = dyn_cast<arith::AddFOp>(*block.begin());
      if (!addOp)
        continue;
      if (addOp.getLhs() != block.getArgument(0) ||
          addOp.getRhs() != block.getArgument(1))
        continue;

      // Check: one input is the mmt4d result, the other matches the output
      Value otherInput;
      if (genericOp.getInputs()[0] == result)
        otherInput = genericOp.getInputs()[1];
      else if (genericOp.getInputs()[1] == result)
        otherInput = genericOp.getInputs()[0];
      else
        continue;

      Value accTensor = genericOp.getOutputs()[0];
      if (accTensor != otherInput)
        continue;

      return genericOp;
    }
    return nullptr;
  }

  /// Handle the case where xsmt.mmt4d is wrapped in linalg.generic(addf)
  /// for accumulation. This inlines the logic from ConvertMMT4DAddPattern.
  LogicalResult convertWithAccumulation(xsmt::MMT4DOp op,
                                        linalg::GenericOp genericOp,
                                        PatternRewriter &rewriter) const {
    Value mmt4dResult = op.getResult();
    auto resType = dyn_cast<RankedTensorType>(mmt4dResult.getType());
    if (!resType || resType.getRank() != 4 || !resType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "non-static 4D result type");

    auto shape = resType.getShape();
    Location loc = genericOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(genericOp);

    Value a = op.getA();
    Value b = op.getB();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();

    // Determine which input is the accumulator
    Value accTensor = genericOp.getOutputs()[0];

    SmallVector<OpFoldResult> offsets(4, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(4, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes;

    Value dim0 = tensor::DimOp::create(rewriter, loc, a, 0);
    Value dim1 = tensor::DimOp::create(rewriter, loc, b, 0);
    sizes.push_back(dim0);
    sizes.push_back(dim1);
    sizes.push_back(rewriter.getIndexAttr(shape[2]));
    sizes.push_back(rewriter.getIndexAttr(shape[3]));

    SmallVector<int64_t> sliceShape = {
        ShapedType::kDynamic, ShapedType::kDynamic, shape[2], shape[3]};
    auto sliceType =
        RankedTensorType::get(sliceShape, resType.getElementType());

    auto extractOp = tensor::ExtractSliceOp::create(
        rewriter, loc, sliceType, accTensor, offsets, sizes, strides);

    auto mmt4dResType =
        RankedTensorType::get(sliceShape, resType.getElementType());
    auto newMmt4dOp =
        linalg::Mmt4DOp::create(rewriter, loc, mmt4dResType, ValueRange{a, b},
                                ValueRange{extractOp.getResult()});

    auto insertOp =
        tensor::InsertSliceOp::create(rewriter, loc, newMmt4dOp.getResult(0),
                                      accTensor, offsets, sizes, strides);

    rewriter.replaceOp(genericOp, insertOp.getResult());
    rewriter.eraseOp(op);

    return success();
  }

  LogicalResult matchAndRewrite(xsmt::MMT4DOp op,
                                PatternRewriter &rewriter) const override {
    // First, check if this mmt4d is wrapped in a generic(addf) accumulation
    // pattern. If so, handle it inline (replaces the old ConvertMMT4DAddPattern
    // stage).
    if (auto genericUser = findAccumulationGeneric(op))
      return convertWithAccumulation(op, genericUser, rewriter);

    Location loc = op.getLoc();
    Value a = op.getA();
    Value b = op.getB();
    Value c = op.getC();

    if (auto cast = a.getDefiningOp<tensor::CastOp>())
      a = cast.getSource();
    if (auto cast = b.getDefiningOp<tensor::CastOp>())
      b = cast.getSource();

    bool unpack = false;

    if (c) {
      if (c.hasOneUse() && op.getResult().use_empty())
        unpack = true;
      if (unpack) {
        if (auto cast = c.getDefiningOp<tensor::CastOp>())
          c = cast.getSource();
        auto cPackOp = c.getDefiningOp<linalg::PackOp>();
        if (!cPackOp)
          return rewriter.notifyMatchFailure(op,
                                             "C must be result of linalg.pack");

        Value packInput = cPackOp.getSource();
        auto toTensorOp = packInput.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp)
          return rewriter.notifyMatchFailure(
              op, "Packed input must come from to_tensor");

        Value subview = toTensorOp->getOperand(0);
        Value mmt4dResult = createZeroFilledMmt4d(rewriter, loc, a, b, c);

        // Unpack directly into the output buffer (subview).
        // Use the *same* memref subview that backs the pack input to keep
        // dest/result types stable for downstream lowerings.
        auto destMemrefTy = cast<MemRefType>(subview.getType());
        auto dstTensorTy =
            RankedTensorType::get({ShapedType::kDynamic, ShapedType::kDynamic},
                                  destMemrefTy.getElementType());
        auto dstTensor = bufferization::ToTensorOp::create(
            rewriter, loc, dstTensorTy, subview,
            /*restrict=*/rewriter.getUnitAttr(),
            /*writable=*/rewriter.getUnitAttr());

        ArrayRef<int64_t> staticInnerTiles = cPackOp.getStaticInnerTiles();
        if (staticInnerTiles.empty())
          return rewriter.notifyMatchFailure(op, "static_inner_tiles is empty");
        SmallVector<OpFoldResult> innerTiles;
        for (int64_t tile : staticInnerTiles)
          innerTiles.push_back(rewriter.getIndexAttr(tile));

        auto unpackOp = linalg::UnPackOp::create(
            rewriter, loc, mmt4dResult, dstTensor.getResult(),
            cPackOp.getInnerDimsPos(), innerTiles, cPackOp.getOuterDimsPerm());

        auto materializeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, unpackOp.getLoc(), unpackOp.getResult(), subview);
        materializeOp.setWritable(true);

        rewriter.replaceOp(op, unpackOp.getResult());
        return success();
      } else {
        if (auto cast = c.getDefiningOp<tensor::CastOp>())
          c = cast.getSource();
        auto toTensorOp = c.getDefiningOp<bufferization::ToTensorOp>();
        if (!toTensorOp)
          return rewriter.notifyMatchFailure(op, "c must come from to_tensor");

        Value subview = toTensorOp->getOperand(0);
        Value mmt4dResult = createZeroFilledMmt4d(rewriter, loc, a, b, c);

        auto materializeOp = bufferization::MaterializeInDestinationOp::create(
            rewriter, loc, mmt4dResult, subview);
        materializeOp.setWritable(true);
        rewriter.replaceOp(op, mmt4dResult);
        return success();
      }
    } else {
      auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
      Value initTensor = tensor::EmptyOp::create(
          rewriter, loc, resultType.getShape(), resultType.getElementType());
      Value mmt4dResult =
          createZeroFilledMmt4d(rewriter, loc, a, b, initTensor);
      rewriter.replaceOp(op, mmt4dResult);
      return success();
    }
  }

private:
  /// Create: zero → fill(initTensor) → mmt4d(a, b, filled). Returns mmt4d
  /// result.
  static Value createZeroFilledMmt4d(PatternRewriter &rewriter, Location loc,
                                     Value a, Value b, Value initTensor) {
    auto tensorType = cast<RankedTensorType>(initTensor.getType());
    auto zeroAttr = FloatAttr::get(tensorType.getElementType(), 0.0);
    auto zero = arith::ConstantOp::create(rewriter, loc, zeroAttr);
    auto fillOp = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                         ValueRange{initTensor});
    auto mmt4dOp = linalg::Mmt4DOp::create(rewriter, loc, ValueRange{a, b},
                                           ValueRange{fillOp.getResult(0)});
    return mmt4dOp.getResult(0);
  }
};

// ===----------------------------------------------------------------------===//
// Phase 3: Loop parallelization and remaining cleanup
// ===----------------------------------------------------------------------===//

class ForToForallPattern : public OpRewritePattern<scf::ForOp> {
public:
  ForToForallPattern(MLIRContext *context)
      : OpRewritePattern<scf::ForOp>(context) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto bindSubBlockAttr = forOp->getAttrOfType<BoolAttr>("bind_sub_block");
    if (!bindSubBlockAttr || !bindSubBlockAttr.getValue()) {
      return failure();
    }

    Value lb = forOp.getLowerBound();
    Value ub = forOp.getUpperBound();
    Value step = forOp.getStep();
    Value indVar = forOp.getInductionVar();
    auto convertToIndex = [&](Value val) -> Value {
      if (val.getType().isIndex())
        return val;
      return arith::IndexCastOp::create(rewriter, forOp.getLoc(),
                                        rewriter.getIndexType(), val);
    };

    SmallVector<OpFoldResult> lbs = {convertToIndex(lb)};
    SmallVector<OpFoldResult> ubs = {convertToIndex(ub)};
    SmallVector<OpFoldResult> steps = {convertToIndex(step)};
    SmallVector<Value> outputs;
    auto initArgs = forOp.getInitArgs();
    for (auto initArg : initArgs) {
      outputs.push_back(initArg);
    }

    auto forallOp = scf::ForallOp::create(rewriter, forOp.getLoc(), lbs, ubs,
                                          steps, outputs, std::nullopt);

    Block &forallBlock = forallOp.getRegion().front();
    Block &forBody = *forOp.getBody();
    auto inParallelOp = cast<scf::InParallelOp>(forallBlock.getTerminator());
    rewriter.setInsertionPoint(inParallelOp);
    IRMapping mapping;

    Value forallIndVar = forallBlock.getArgument(0);
    auto castIndVar = arith::IndexCastOp::create(
        rewriter, forOp.getLoc(), indVar.getType(), forallIndVar);
    mapping.map(indVar, castIndVar);

    unsigned numInitArgs = initArgs.size();
    for (unsigned i = 0; i < numInitArgs; ++i) {
      mapping.map(forBody.getArgument(i + 1), forallBlock.getArgument(i + 1));
    }

    for (auto &op : forBody.without_terminator()) {
      rewriter.clone(op, mapping);
    }
    rewriter.replaceOp(forOp, forallOp.getResults());
    return success();
  }
};

struct GetThreadOpToLLVMCallPattern
    : public OpRewritePattern<xsmt::GetThreadOp> {
  using OpRewritePattern<xsmt::GetThreadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(xsmt::GetThreadOp getThreadOp,
                                PatternRewriter &rewriter) const override {
    Location loc = getThreadOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = getThreadOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    StringRef funcName = "spine_get_stream_threads";
    Type resultType = rewriter.getI32Type();

    auto llvmFuncOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFuncOp) {
      auto funcType = LLVM::LLVMFunctionType::get(resultType, /*params=*/{},
                                                  /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      llvmFuncOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
      llvmFuncOp.setLinkage(LLVM::Linkage::External);
    }

    auto callOp =
        LLVM::CallOp::create(rewriter, loc, resultType, funcName, ValueRange{});

    rewriter.replaceOp(getThreadOp, callOp.getResults());

    return success();
  }
};

// Pattern to convert proton::RecordOp to RISC-V rdtime inline assembly +
// runtime call Note: rdcycle is disabled in user mode on many RISC-V systems,
// so we use rdtime instead
struct ProtonRecordOpPattern : public OpRewritePattern<proton::RecordOp> {
  using OpRewritePattern<proton::RecordOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(proton::RecordOp recordOp,
                                PatternRewriter &rewriter) const override {
    Location loc = recordOp.getLoc();
    MLIRContext *ctx = rewriter.getContext();

    auto moduleOp = recordOp->getParentOfType<ModuleOp>();
    if (!moduleOp) {
      return failure();
    }

    // 1. Create rdtime inline assembly to read time counter
    // Note: rdcycle is often disabled in user mode, rdtime is more portable
    Type i64Type = rewriter.getI64Type();

    // RISC-V rdtime instruction: reads the time CSR (available in user mode)
    auto asmDialectAttr =
        LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT);

    auto inlineAsmOp = LLVM::InlineAsmOp::create(
        rewriter, loc,
        /*resultTypes=*/i64Type,
        /*operands=*/ValueRange{},
        /*asm_string=*/"rdtime $0",
        /*constraints=*/"=r",
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*tail_call_kind=*/LLVM::tailcallkind::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr{});

    Value cycleValue = inlineAsmOp.getResult(0);

    // 2. Declare and call runtime function: proton_record(const char* name,
    // int64_t cycle, int is_start)
    StringRef funcName = "proton_record";

    // Get or create the runtime function declaration
    auto llvmFuncOp = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName);
    if (!llvmFuncOp) {
      // Function signature: void proton_record(const char* name, int64_t cycle,
      // int32_t is_start)
      Type voidType = LLVM::LLVMVoidType::get(ctx);
      Type ptrType = LLVM::LLVMPointerType::get(ctx);
      Type i32Type = rewriter.getI32Type();

      auto funcType = LLVM::LLVMFunctionType::get(
          voidType, {ptrType, i64Type, i32Type}, /*isVarArg=*/false);

      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());
      llvmFuncOp = LLVM::LLVMFuncOp::create(rewriter, loc, funcName, funcType);
      llvmFuncOp.setLinkage(LLVM::Linkage::External);
    }

    // 3. Create global string constant for the scope name
    std::string scopeName = recordOp.getName().str();
    std::string globalName = "proton_scope_" + scopeName;

    // Check if global already exists
    auto globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      // Create null-terminated string
      std::string strWithNull = scopeName + '\0';
      auto strAttr = rewriter.getStringAttr(strWithNull);
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                                strWithNull.size());

      globalOp = LLVM::GlobalOp::create(
          rewriter, loc, arrayType,
          /*isConstant=*/true, LLVM::Linkage::Internal, globalName, strAttr);
    }

    // 4. Get pointer to the global string
    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    auto addrOp = LLVM::AddressOfOp::create(rewriter, loc, ptrType, globalName);

    // 5. Create is_start constant (1 for start, 0 for end)
    int32_t isStartVal = recordOp.getIsStart() ? 1 : 0;
    auto isStartConst =
        LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                 rewriter.getI32IntegerAttr(isStartVal));

    // 6. Call the runtime function
    LLVM::CallOp::create(
        rewriter, loc, TypeRange{}, funcName,
        ValueRange{addrOp.getResult(), cycleValue, isStartConst.getResult()});

    // 7. Erase the original op
    rewriter.eraseOp(recordOp);

    return success();
  }
};

struct InsertMBarrierReleasePattern
    : public mlir::OpRewritePattern<xsmt_async::MBarrierAllocOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(xsmt_async::MBarrierAllocOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value root = op.getResult();

    if (hasExistingRelease(root))
      return mlir::failure();

    mlir::Block *allocBlock = op->getBlock();

    llvm::SmallPtrSet<mlir::Value, 16> aliasSet;
    collectAliasClosure(root, aliasSet);

    mlir::Operation *lastUser = nullptr;
    bool escaped = false;
    findLastUser(aliasSet, allocBlock, op.getOperation(), lastUser, escaped);

    if (escaped || !lastUser)
      return mlir::failure();

    insertRelease(rewriter, op.getLoc(), root, lastUser);

    return mlir::success();
  }

private:
  bool hasExistingRelease(mlir::Value barrier) const {
    for (mlir::Operation *user : barrier.getUsers()) {
      if (mlir::isa<xsmt_async::MBarrierReleaseOp>(user))
        return true;
    }
    return false;
  }

  bool
  isBarrierForwardingOp(mlir::Operation *user, mlir::Value v,
                        llvm::SmallVectorImpl<mlir::Value> &newAliases) const {
    if (auto fromElements = dyn_cast<tensor::FromElementsOp>(user)) {
      for (Value operand : fromElements.getOperands()) {
        if (operand == v) {
          newAliases.push_back(fromElements.getResult());
          return true;
        }
      }
    }

    if (auto extract = dyn_cast<tensor::ExtractOp>(user)) {
      if (extract.getTensor() == v) {
        newAliases.push_back(extract.getResult());
        return true;
      }
    }

    if (auto cast = dyn_cast<tensor::CastOp>(user)) {
      if (cast.getSource() == v) {
        newAliases.push_back(cast.getResult());
        return true;
      }
    }

    if (auto ucc = dyn_cast<UnrealizedConversionCastOp>(user)) {
      for (Value input : ucc.getInputs()) {
        if (input == v) {
          for (Value output : ucc.getOutputs())
            newAliases.push_back(output);
          return true;
        }
      }
    }

    if (auto select = dyn_cast<arith::SelectOp>(user)) {
      if (select.getTrueValue() == v || select.getFalseValue() == v) {
        newAliases.push_back(select.getResult());
        return true;
      }
    }

    if (auto insert = dyn_cast<tensor::InsertOp>(user)) {
      if (insert.getScalar() == v || insert.getDest() == v) {
        newAliases.push_back(insert.getResult());
        return true;
      }
    }

    if (auto ifOp = dyn_cast<scf::IfOp>(user)) {
      for (auto &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
        if (!yield)
          continue;
        for (auto [idx, operand] : llvm::enumerate(yield.getOperands())) {
          if (operand == v && idx < ifOp.getNumResults()) {
            newAliases.push_back(ifOp.getResult(idx));
          }
        }
      }
      if (!newAliases.empty())
        return true;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg == v) {
          newAliases.push_back(forOp.getRegionIterArg(idx));
          newAliases.push_back(forOp.getResult(idx));
        }
      }
      if (!newAliases.empty())
        return true;
    }

    return false;
  }

  void collectAliasClosure(mlir::Value root,
                           llvm::SmallPtrSetImpl<mlir::Value> &aliasSet) const {
    llvm::SmallVector<mlir::Value, 16> worklist;

    aliasSet.insert(root);
    worklist.push_back(root);

    while (!worklist.empty()) {
      mlir::Value current = worklist.pop_back_val();

      for (mlir::Operation *user : current.getUsers()) {
        llvm::SmallVector<mlir::Value, 4> newAliases;
        isBarrierForwardingOp(user, current, newAliases);

        for (mlir::Value alias : newAliases) {
          if (aliasSet.insert(alias).second) {
            worklist.push_back(alias);
          }
        }
      }
    }
  }

  void findLastUser(const llvm::SmallPtrSetImpl<mlir::Value> &aliasSet,
                    mlir::Block *allocBlock, mlir::Operation *allocOp,
                    mlir::Operation *&lastUser, bool &escaped) const {
    lastUser = allocOp;
    escaped = false;

    for (mlir::Value v : aliasSet) {
      for (mlir::Operation *user : v.getUsers()) {
        if (mlir::isa<xsmt_async::MBarrierReleaseOp>(user))
          continue;
        mlir::Operation *ancestor = allocBlock->findAncestorOpInBlock(*user);

        if (!ancestor) {
          escaped = true;
          return;
        }
        if (lastUser->isBeforeInBlock(ancestor)) {
          lastUser = ancestor;
        }
      }
    }
  }

  void insertRelease(mlir::PatternRewriter &rewriter, mlir::Location loc,
                     mlir::Value barrier, mlir::Operation *lastUser) const {
    if (lastUser->hasTrait<mlir::OpTrait::IsTerminator>()) {
      rewriter.setInsertionPoint(lastUser);
    } else {
      rewriter.setInsertionPointAfter(lastUser);
    }

    xsmt_async::MBarrierReleaseOp::create(rewriter, loc, barrier);
  }
};

// ===----------------------------------------------------------------------===//
// Bufferization cleanup: rewrite unpack+materialize into direct subview write.
// Runs AFTER the main conversion stage where linalg::UnPackOp and
// MaterializeInDestinationOp have already been generated.
// ===----------------------------------------------------------------------===//

class MaterializeUnpackIntoSubviewPattern
    : public OpRewritePattern<bufferization::MaterializeInDestinationOp> {
public:
  MaterializeUnpackIntoSubviewPattern(MLIRContext *context)
      : OpRewritePattern<bufferization::MaterializeInDestinationOp>(
            context,
            /*benefit=*/2) {}

private:
  static bool isZeroIndex(Value value) {
    APInt intValue;
    return matchPattern(value, m_ConstantInt(&intValue)) && intValue.isZero();
  }

  static bool isIdentitySubview(memref::SubViewOp subview) {
    if (!hasZeroOffsets(subview.getOffsets()))
      return false;
    if (!hasUnitStrides(subview.getStrides()))
      return false;

    auto srcTy = dyn_cast<MemRefType>(subview.getSource().getType());
    auto dstTy = dyn_cast<MemRefType>(subview.getType());
    if (!srcTy || !dstTy || srcTy.getRank() != dstTy.getRank())
      return false;

    for (auto [srcDim, dstDim] :
         llvm::zip(srcTy.getShape(), dstTy.getShape())) {
      if (srcDim != ShapedType::kDynamic && dstDim != ShapedType::kDynamic &&
          srcDim != dstDim)
        return false;
    }
    return true;
  }

  static bool comesFromDimOf(Value value, Value source) {
    auto dimOp = value.getDefiningOp<tensor::DimOp>();
    return dimOp && dimOp.getSource() == source;
  }

  static bool isBoundedByUnpackDim(Value size, Value unpackResult) {
    if (comesFromDimOf(size, unpackResult))
      return true;

    auto minOp = size.getDefiningOp<arith::MinUIOp>();
    if (!minOp)
      return false;

    return comesFromDimOf(minOp.getLhs(), unpackResult) ||
           comesFromDimOf(minOp.getRhs(), unpackResult);
  }

  static bool isStaticSliceWithinUnpackDim(OpFoldResult size,
                                           int64_t unpackDim) {
    auto attr =
        llvm::dyn_cast_if_present<IntegerAttr>(size.dyn_cast<Attribute>());
    return attr && unpackDim != ShapedType::kDynamic &&
           attr.getInt() <= unpackDim;
  }

  static bool isSliceSizeCompatible(OpFoldResult size, Value unpackResult,
                                    int64_t unpackDim) {
    if (isStaticSliceWithinUnpackDim(size, unpackDim))
      return true;

    Value sizeVal = size.dyn_cast<Value>();
    return sizeVal && isBoundedByUnpackDim(sizeVal, unpackResult);
  }

  static bool hasUnitStrides(ValueRange strides) {
    return llvm::all_of(
        strides, [](Value stride) { return isConstantIntValue(stride, 1); });
  }

  static bool hasZeroOffsets(ValueRange offsets) {
    return llvm::all_of(offsets, isZeroIndex);
  }

  LogicalResult matchAndRewrite(bufferization::MaterializeInDestinationOp mat,
                                PatternRewriter &rewriter) const override {
    auto fail = [&](StringRef reason) -> LogicalResult {
      (void)reason;
      return failure();
    };

    if (!mat.getWritable())
      return failure();

    // Match:
    // materialize_in_destination(extract_slice(cast(extract_slice(unpack(...)))))
    auto outerSlice = mat.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!outerSlice)
      return fail("source is not outer extract_slice");

    if (outerSlice.getMixedOffsets().size() != 2 ||
        outerSlice.getMixedSizes().size() != 2 ||
        outerSlice.getMixedStrides().size() != 2)
      return fail("expected rank-2 outer slice");

    if (!hasZeroOffsets(outerSlice.getOffsets()))
      return fail("expected zero outer slice offsets");

    if (!hasUnitStrides(outerSlice.getStrides()))
      return fail("expected unit outer slice strides");

    auto cast = outerSlice.getSource().getDefiningOp<tensor::CastOp>();
    if (!cast)
      return fail("outer slice source is not tensor.cast");

    auto innerSlice = cast.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!innerSlice)
      return fail("cast source is not inner extract_slice");

    auto unpack = innerSlice.getSource().getDefiningOp<linalg::UnPackOp>();
    if (!unpack)
      return fail("inner slice source is not linalg.unpack");

    if (innerSlice.getMixedOffsets().size() != 2 ||
        innerSlice.getMixedSizes().size() != 2 ||
        innerSlice.getMixedStrides().size() != 2)
      return fail("expected rank-2 inner slice");

    if (!hasZeroOffsets(innerSlice.getOffsets()))
      return fail("expected zero inner slice offsets");

    if (!hasUnitStrides(innerSlice.getStrides()))
      return fail("expected unit inner slice strides");

    auto unpackTy = dyn_cast<RankedTensorType>(unpack.getResult().getType());
    if (!unpackTy || unpackTy.getRank() != 2)
      return fail("expected rank-2 unpack result");

    Value unpackResult = unpack.getResult();
    ArrayRef<int64_t> unpackShape = unpackTy.getShape();
    if (!llvm::all_of(llvm::zip(innerSlice.getMixedSizes(), unpackShape),
                      [&](auto it) {
                        return isSliceSizeCompatible(
                            std::get<0>(it), unpackResult, std::get<1>(it));
                      }))
      return fail("inner slice sizes must be derived from unpack dimensions");

    // Destination subview memref.
    Value dstMemref = mat.getDest();
    if (auto subview = dstMemref.getDefiningOp<memref::SubViewOp>()) {
      if (isIdentitySubview(subview)) {
        dstMemref = subview.getSource();
      }
    }
    auto dstTy = dyn_cast<MemRefType>(dstMemref.getType());
    if (!dstTy || dstTy.getRank() != 2)
      return fail("destination is not rank-2 memref");

    if (outerSlice.getMixedSizes().size() !=
        static_cast<size_t>(dstTy.getRank()))
      return fail("outer slice rank does not match destination rank");

    Location loc = mat.getLoc();
    rewriter.setInsertionPoint(mat);

    Value castedDst = dstMemref;

    // Create writable tensor view of the destination.
    auto dstTensorTy =
        RankedTensorType::get(dstTy.getShape(), dstTy.getElementType());
    Value dstTensor =
        bufferization::ToTensorOp::create(rewriter, loc, dstTensorTy, castedDst,
                                          /*restrict=*/rewriter.getUnitAttr(),
                                          /*writable=*/rewriter.getUnitAttr());

    // Recreate unpack with destination init = dstTensor.
    auto newUnpack = linalg::UnPackOp::create(
        rewriter, loc, unpack.getSource(), dstTensor, unpack.getInnerDimsPos(),
        unpack.getMixedTiles(), unpack.getOuterDimsPerm());

    // Replace materialize to use the unpack result directly.
    auto newMat = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, newUnpack.getResult(), castedDst);
    newMat.setWritable(true);

    rewriter.eraseOp(mat);
    return success();
  }
};

struct DescriptorLoadViewOpPattern
    : public OpRewritePattern<DescriptorLoadViewOp> {
  using OpRewritePattern<DescriptorLoadViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DescriptorLoadViewOp op,
                                PatternRewriter &rewriter) const override {
    Value result = op.getResult();
    bool hasTranspose = false;
    linalg::TransposeOp transposeOp;

    for (OpOperand &use : result.getUses()) {
      if (auto t = dyn_cast<linalg::TransposeOp>(use.getOwner())) {
        if (hasTranspose)
          return failure();
        hasTranspose = true;
        transposeOp = t;
      }
    }

    if (hasTranspose) {
      if (!result.hasOneUse())
        return failure();
      if (transposeOp.getInput() != result)
        return failure();
    }
    Value sourcePtr = op.getBase();
    auto unrealizedCast = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealizedCast)
      return failure();
    Value rankedMemRef = unrealizedCast.getOperand(0);

    // Trace through memref.cast to get the original reinterpret_cast if present
    // This allows us to work with the actual dynamic sizes from
    // reinterpret_cast instead of the static sizes from memref.cast
    if (auto castOp = rankedMemRef.getDefiningOp<memref::CastOp>()) {
      rankedMemRef = castOp.getSource();
    }

    auto memrefTy = dyn_cast<MemRefType>(rankedMemRef.getType());
    if (!memrefTy)
      return failure();

    if (memrefTy.getRank() == 4) {
      if (hasTranspose)
        return failure();
      return convertRank4ToUnpack(op, rankedMemRef, rewriter);
    }

    if (memrefTy.getRank() != 2)
      return failure();

    Location loc = op.getLoc();

    Value dim0Value, dim1Value;
    if (auto rc = rankedMemRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      // Get actual sizes from reinterpret_cast
      auto sizes = rc.getMixedSizes();
      if (sizes.size() != 2)
        return failure();
      dim0Value = ofrToIndexValue(loc, sizes[0], rewriter);
      dim1Value = ofrToIndexValue(loc, sizes[1], rewriter);
    } else {
      dim0Value =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/0, rewriter);
      dim1Value =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/1, rewriter);
    }

    auto packedSizeAttr = op.getPackedSize();
    if (packedSizeAttr.size() != 2)
      return failure();
    int32_t tile0 = packedSizeAttr[0];
    int32_t tile1 = packedSizeAttr[1];

    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto offsets = op.getOffsets();
    Value offset0 = offsets[0];
    Value offset1 = offsets[1];

    Value indexOffset0 = ensureIndexType(loc, offset0, rewriter);
    Value indexOffset1 = ensureIndexType(loc, offset1, rewriter);
    Value indexDim0Value = ensureIndexType(loc, dim0Value, rewriter);
    Value indexDim1Value = ensureIndexType(loc, dim1Value, rewriter);

    auto shapes = op.getShape();
    Value shape0 = arith::ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value shape1 = arith::ConstantIndexOp::create(rewriter, loc, shapes[1]);

    Value size_m1 =
        arith::SubIOp::create(rewriter, loc, indexDim0Value, indexOffset0);
    Value size_m2 = arith::MinSIOp::create(rewriter, loc, size_m1, shape0);

    Value size_k1 =
        arith::SubIOp::create(rewriter, loc, indexDim1Value, indexOffset1);
    Value size_k2 = arith::MinSIOp::create(rewriter, loc, size_k1, shape1);

    SmallVector<OpFoldResult> subviewOffsets{indexOffset0, indexOffset1};
    SmallVector<OpFoldResult> subviewSizes{size_m2, size_k2};
    SmallVector<OpFoldResult> subviewStrides{c1, c1};

    Value toTensorSource = rankedMemRef;
    auto rankedMemRefType = cast<MemRefType>(rankedMemRef.getType());
    auto tensorType = RankedTensorType::get(rankedMemRefType.getShape(),
                                            rankedMemRefType.getElementType());

    bool elideSubview =
        canElideDescriptorSubview(rankedMemRef, indexOffset0, indexOffset1,
                                  size_m2, size_k2) ||
        canElideIdentitySubview(rankedMemRef, subviewSizes, subviewOffsets,
                                subviewStrides);

    if (!elideSubview) {
      Value subview = memref::SubViewOp::create(rewriter, loc, rankedMemRef,
                                                /*offsets=*/subviewOffsets,
                                                /*sizes=*/subviewSizes,
                                                /*strides=*/subviewStrides);

      auto subviewType = cast<MemRefType>(subview.getType());
      tensorType = RankedTensorType::get(subviewType.getShape(),
                                         subviewType.getElementType());
      toTensorSource = subview;
    }

    Value tensor = bufferization::ToTensorOp::create(
        rewriter, loc, tensorType, toTensorSource,
        /*restrict=*/rewriter.getUnitAttr());

    if (!hasTranspose) {
      auto shapes = op.getShape();
      return convertWithoutTranspose(op, tensor, size_m2, size_k2, shapes[0],
                                     shapes[1], tile0, tile1, rewriter);
    }

    auto permutation = transposeOp.getPermutation();
    if (permutation.size() != 4 || permutation[0] != 1 || permutation[1] != 0 ||
        permutation[2] != 3 || permutation[3] != 2) {
      return failure();
    }

    return convertWithTranspose(op, transposeOp, tensor, size_m2, size_k2,
                                shapes[0], shapes[1], tile0, tile1, rewriter);
  }

private:
  LogicalResult convertRank4ToUnpack(DescriptorLoadViewOp op,
                                     Value rankedMemRef,
                                     PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    auto memrefTy = dyn_cast<MemRefType>(rankedMemRef.getType());
    if (!memrefTy || memrefTy.getRank() != 4)
      return failure();

    Value outer0, outer1, tile0Val, tile1Val;
    if (auto rc = rankedMemRef.getDefiningOp<memref::ReinterpretCastOp>()) {
      auto sizes = rc.getMixedSizes();
      if (sizes.size() != 4)
        return failure();
      outer0 = ofrToIndexValue(loc, sizes[0], rewriter);
      outer1 = ofrToIndexValue(loc, sizes[1], rewriter);
      tile0Val = ofrToIndexValue(loc, sizes[2], rewriter);
      tile1Val = ofrToIndexValue(loc, sizes[3], rewriter);
    } else {
      outer0 =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/0, rewriter);
      outer1 =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/1, rewriter);
      tile0Val =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/2, rewriter);
      tile1Val =
          getDimValueFromTypeOrDimOp(loc, rankedMemRef, /*dim=*/3, rewriter);
    }

    int64_t tile0Static = memrefTy.getDimSize(2);
    int64_t tile1Static = memrefTy.getDimSize(3);
    if (ShapedType::isDynamic(tile0Static) ||
        ShapedType::isDynamic(tile1Static))
      return failure();

    auto tensor4DType =
        RankedTensorType::get(memrefTy.getShape(), memrefTy.getElementType());
    Value packedTensor = bufferization::ToTensorOp::create(
        rewriter, loc, tensor4DType, rankedMemRef, rewriter.getUnitAttr());

    Value flatRows = arith::MulIOp::create(rewriter, loc, outer0, tile0Val);
    Value flatCols = arith::MulIOp::create(rewriter, loc, outer1, tile1Val);

    auto elementType = memrefTy.getElementType();
    auto flatTy = RankedTensorType::get(
        {ShapedType::kDynamic, ShapedType::kDynamic}, elementType);
    Value empty = tensor::EmptyOp::create(rewriter, loc, flatTy,
                                          ValueRange{flatRows, flatCols});

    Value flatTensor =
        linalg::UnPackOp::create(
            rewriter, loc, packedTensor, empty, ArrayRef<int64_t>{0, 1},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0Static),
                                   rewriter.getIndexAttr(tile1Static)})
            .getResult();

    auto offsets = op.getOffsets();
    if (offsets.size() != 2)
      return failure();

    Value off0 = ensureIndexType(loc, offsets[0], rewriter);
    Value off1 = ensureIndexType(loc, offsets[1], rewriter);

    auto shapes = op.getShape();
    if (shapes.size() != 2)
      return failure();

    Value reqRows = arith::ConstantIndexOp::create(rewriter, loc, shapes[0]);
    Value reqCols = arith::ConstantIndexOp::create(rewriter, loc, shapes[1]);

    Value flatDim0 = tensor::DimOp::create(rewriter, loc, flatTensor, 0);
    Value flatDim1 = tensor::DimOp::create(rewriter, loc, flatTensor, 1);

    Value availRows = arith::SubIOp::create(rewriter, loc, flatDim0, off0);
    Value availCols = arith::SubIOp::create(rewriter, loc, flatDim1, off1);
    Value actualRows =
        arith::MinSIOp::create(rewriter, loc, reqRows, availRows);
    Value actualCols =
        arith::MinSIOp::create(rewriter, loc, reqCols, availCols);

    Value c1 = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value slice = tensor::ExtractSliceOp::create(
        rewriter, loc, flatTensor, ArrayRef<Value>{off0, off1},
        ArrayRef<Value>{actualRows, actualCols}, ArrayRef<Value>{c1, c1});

    auto originalResultType = cast<RankedTensorType>(op.getResult().getType());
    Value castResult =
        tensor::CastOp::create(rewriter, loc, originalResultType, slice);

    rewriter.replaceOp(op, castResult);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  Value getDimValueFromTypeOrDimOp(Location loc, Value memref, int64_t dim,
                                   PatternRewriter &rewriter) const {
    auto ty = cast<MemRefType>(memref.getType());
    if (!ty.isDynamicDim(dim)) {
      return arith::ConstantIndexOp::create(rewriter, loc, ty.getDimSize(dim));
    }
    return memref::DimOp::create(rewriter, loc, memref, dim);
  }

  LogicalResult convertWithoutTranspose(DescriptorLoadViewOp op, Value tensor,
                                        Value dim0, Value dim1, int32_t shape0,
                                        int32_t shape1, int32_t tile0,
                                        int32_t tile1,
                                        PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    // Use expected shape (not actual data size) to calculate packed dimensions
    // This ensures the output has the expected static shape, with padding for
    // boundary cases
    int64_t packedRowsStatic = (shape0 + tile0 - 1) / tile0;
    int64_t packedColsStatic = (shape1 + tile1 - 1) / tile1;

    SmallVector<int64_t, 4> shapeDims = {packedRowsStatic, packedColsStatic,
                                         tile0, tile1};

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();
    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    // Static shape, no dynamic sizes needed
    Value emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, emptyType, ValueRange{});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, tensor, emptyTensor,
            /*innerDimsPos=*/ArrayRef<int64_t>{0, 1},
            /*innerTiles=*/
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile0),
                                   rewriter.getIndexAttr(tile1)},
            /*paddingValue=*/paddingValue,
            /*outerDimsPerm=*/ArrayRef<int64_t>{0, 1})
            .getResult();

    auto originalResultType = cast<RankedTensorType>(op.getResult().getType());
    Value castResult =
        tensor::CastOp::create(rewriter, loc, originalResultType, packedResult);

    rewriter.replaceOp(op, castResult);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  LogicalResult convertWithTranspose(DescriptorLoadViewOp op,
                                     linalg::TransposeOp transposeOp,
                                     Value tensor, Value dim0, Value dim1,
                                     int32_t shape0, int32_t shape1,
                                     int32_t tile0, int32_t tile1,
                                     PatternRewriter &rewriter) const {
    Location loc = op.getLoc();

    // Use expected shape (not actual data size) to calculate packed dimensions
    // This ensures static output shapes for downstream operations
    // For transpose case: output shape is [ceil(shape1/tile1),
    // ceil(shape0/tile0), tile1, tile0]
    int64_t packedRowsStatic = (shape1 + tile1 - 1) / tile1;
    int64_t packedColsStatic = (shape0 + tile0 - 1) / tile0;

    auto tensorType = cast<RankedTensorType>(tensor.getType());
    Type elementType = tensorType.getElementType();

    SmallVector<int64_t, 4> shapeDims = {packedRowsStatic, packedColsStatic,
                                         tile1, tile0};

    auto emptyType = RankedTensorType::get(shapeDims, elementType);

    // Static shape, no dynamic sizes needed
    Value emptyTensor =
        tensor::EmptyOp::create(rewriter, loc, emptyType, ValueRange{});
    Value paddingValue = createZeroConstant(rewriter, loc, elementType);

    Value packedResult =
        linalg::PackOp::create(
            rewriter, loc, tensor, emptyTensor,
            /*innerDimsPos=*/ArrayRef<int64_t>{1, 0},
            /*innerTiles=*/
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tile1),
                                   rewriter.getIndexAttr(tile0)},
            /*paddingValue=*/paddingValue,
            /*outerDimsPerm=*/ArrayRef<int64_t>{1, 0})
            .getResult();

    auto transposeResultType =
        cast<RankedTensorType>(transposeOp->getResult(0).getType());

    // Use the transpose result type directly, as packedResult should match it
    Value castResult = tensor::CastOp::create(
        rewriter, loc, transposeResultType, packedResult);

    rewriter.replaceOp(transposeOp, castResult);
    rewriter.eraseOp(op);
    cleanupUnusedOps(op, rewriter);
    return success();
  }

  void cleanupUnusedOps(DescriptorLoadViewOp op,
                        PatternRewriter &rewriter) const {
    Value sourcePtr = op.getBase();
    auto unrealized = sourcePtr.getDefiningOp<UnrealizedConversionCastOp>();
    if (!unrealized)
      return;

    Value v = unrealized.getOperand(0);
    memref::CastOp castOp = v.getDefiningOp<memref::CastOp>();
    if (castOp)
      v = castOp.getSource();
    memref::ReinterpretCastOp rcOp =
        v.getDefiningOp<memref::ReinterpretCastOp>();

    if (unrealized->use_empty())
      rewriter.eraseOp(unrealized);

    if (castOp && castOp->use_empty())
      rewriter.eraseOp(castOp);

    if (rcOp && rcOp->use_empty())
      rewriter.eraseOp(rcOp);
  }
};

// ===----------------------------------------------------------------------===//
// Stage registration
// ===----------------------------------------------------------------------===//

void mlir::triton::populateXSMTValidationPatterns(RewritePatternSet &patterns) {
  patterns.add<MBarrierLoopCheckPattern<xsmt_async::MBarrierAllocOp>>(
      patterns.getContext());
  patterns.add<MBarrierLoopCheckPattern<xsmt::GlobalMBarrierInitOp>>(
      patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierArriveOp>>(
      patterns.getContext());
  patterns.add<CheckGlobalMBarrierInLoopPattern<xsmt_async::MBarrierWaitOp>>(
      patterns.getContext());
  patterns.add<InsertMBarrierReleasePattern>(patterns.getContext());
}

void mlir::triton::populateXSMTConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<DescriptorLoadViewOpPattern>(patterns.getContext());
  patterns.add<PackOpPattern>(patterns.getContext());
  patterns.add<UnpackOpPattern>(patterns.getContext());
  patterns.add<RepackOpPattern>(patterns.getContext());
  patterns.add<LowerXSMTMMT4D>(patterns.getContext());
  patterns.add<GetThreadOpToLLVMCallPattern>(patterns.getContext());
  patterns.add<ForToForallPattern>(patterns.getContext());
  patterns.add<ProtonRecordOpPattern>(patterns.getContext());
}

void mlir::triton::populateXSMTBufferizationCleanupPatterns(
    RewritePatternSet &patterns) {
  patterns.add<MaterializeUnpackIntoSubviewPattern>(patterns.getContext());
}
