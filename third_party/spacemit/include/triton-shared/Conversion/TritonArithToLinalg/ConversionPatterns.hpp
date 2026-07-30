
#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. ALL rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights
// reserved. SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Conversion/TritonArithToLinalg/ConversionTools.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Utils/Utils.h"

#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "TypeConverter.hpp"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include <numeric>
#include <optional>
#include <type_traits>

namespace mlir {
namespace triton {

static bool keepFp16ReduceAccInLinalg() { return true; }

//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//

// Extract a scalar value from v.
// If v is a scalar, return that directly. Otherwise, parse through operations
// (currently only support splat, sitofp, and truncf) that produce it to
// extract the underlying scalar value. We then reconstruct the chain of
// operations that can produce this constant with the original type. If no
// scalar value can be extracted, a nullptr is returned.
static Value getScalarValue(Value operand, Location loc,
                            ConversionPatternRewriter &rewriter) {
  SmallVector<Operation *> ops;

  auto reconstructScalarValue = [&](Value src) {
    for (auto op = ops.rbegin(); op != ops.rend(); ++op) {
      src = TypeSwitch<Operation *, Value>(*op)
                .Case<arith::SIToFPOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::SIToFPOp::create(rewriter, loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return arith::TruncFOp::create(rewriter, loc, resType, src);
                })
                .Default([](Operation *op) {
                  llvm_unreachable("unsupported op in generating ");
                  return nullptr;
                });
    }
    return src;
  };

  while (true) {
    if (!dyn_cast<ShapedType>(operand.getType())) {
      return reconstructScalarValue(operand);
    } else if (auto op = operand.getDefiningOp<arith::ConstantOp>()) {
      if (auto attr = dyn_cast<DenseElementsAttr>(op.getValue())) {
        if (!attr.isSplat()) {
          InFlightDiagnostic diag = emitRemark(loc)
                                    << "other value used in masked load "
                                       "produced by unsupported instruction";
          return nullptr;
        }
        auto elemValue = attr.getSplatValue<Attribute>();
        auto constOp = arith::ConstantOp::materialize(
            rewriter, elemValue, attr.getElementType(), op.getLoc());
        return reconstructScalarValue(constOp.getResult());
      }
    } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
      operand = op.getSrc();
    } else if (auto op = operand.getDefiningOp<arith::SIToFPOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else if (auto op = operand.getDefiningOp<arith::TruncFOp>()) {
      ops.push_back(op.getOperation());
      operand = op.getIn();
    } else {
      InFlightDiagnostic diag = emitRemark(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

static Value castScalarToElementType(Value scalar, Type elementType,
                                     Location loc,
                                     ConversionPatternRewriter &rewriter) {
  auto srcType = scalar.getType();
  if (srcType == elementType)
    return scalar;

  if (isa<IntegerType>(srcType) && isa<IntegerType>(elementType)) {
    auto srcInt = cast<IntegerType>(srcType);
    auto dstInt = cast<IntegerType>(elementType);
    unsigned srcWidth = srcInt.getWidth();
    unsigned dstWidth = dstInt.getWidth();
    if (srcWidth == dstWidth)
      return scalar;
    if (srcWidth < dstWidth)
      return arith::ExtUIOp::create(rewriter, loc, elementType, scalar);
    return arith::TruncIOp::create(rewriter, loc, elementType, scalar);
  }

  if (isa<FloatType>(srcType) && isa<FloatType>(elementType)) {
    auto srcFloat = cast<FloatType>(srcType);
    auto dstFloat = cast<FloatType>(elementType);
    unsigned srcWidth = srcFloat.getWidth();
    unsigned dstWidth = dstFloat.getWidth();
    if (srcWidth == dstWidth)
      return scalar;
    if (srcWidth < dstWidth)
      return arith::ExtFOp::create(rewriter, loc, elementType, scalar);
    return arith::TruncFOp::create(rewriter, loc, elementType, scalar);
  }

  if (isa<IntegerType>(srcType) && isa<FloatType>(elementType))
    return arith::SIToFPOp::create(rewriter, loc, elementType, scalar);

  if (isa<FloatType>(srcType) && isa<IntegerType>(elementType))
    return arith::FPToSIOp::create(rewriter, loc, elementType, scalar);

  return scalar;
}

// if order is empty, transpose the last two dimensions
// otherwise, use the provided order.
// The order must be a permutation of the source rank.
static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter,
                                llvm::ArrayRef<int32_t> order = {}) {
  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  SmallVector<int64_t> transposedShape(sourceType.getShape());
  if (order.empty()) {
    std::iota(std::begin(perm), std::end(perm), 0);
    std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);
    std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);
  } else {
    // Use the provided order
    assert(order.size() == sourceRank && "Order size must match source rank");
    for (unsigned i = 0; i < sourceRank; ++i) {
      perm[i] = order[i];
      transposedShape[i] = sourceType.getShape()[order[i]];
    }
  }

  Value transposeInit = tensor::EmptyOp::create(rewriter, loc, transposedShape,
                                                sourceType.getElementType());

  Value transpose =
      linalg::TransposeOp::create(rewriter, loc, source, transposeInit, perm)
          .getResults()[0];

  return transpose;
}

// for IntLike and FloatLike types
static std::optional<unsigned> getBitWidth(Type a) {
  if (auto type = dyn_cast<TensorType>(a)) {
    auto elementType = type.getElementType();
    if (elementType.isIntOrFloat()) {
      return type.getElementType().getIntOrFloatBitWidth();
    }
    return std::nullopt;
  }

  if (a.isIntOrFloat())
    return a.getIntOrFloatBitWidth();

  return std::nullopt;
}

static bool createReassociationMaps(
    OpBuilder &builder, llvm::ArrayRef<int64_t> expandedShape,
    llvm::ArrayRef<int64_t> collapsedShape,
    llvm::SmallVector<ReassociationExprs, 4> &reassociationMap) {
  if (collapsedShape.empty()) {
    reassociationMap = {};
    return true;
  }

  // As tensor.expand_shape/tensor.collapse_shape expected rank
  // expansion/reduction.
  if (expandedShape.size() == collapsedShape.size())
    return false;
  if (ShapedType::isDynamicShape(expandedShape) ||
      ShapedType::isDynamicShape(collapsedShape))
    return false;

  reassociationMap.resize(collapsedShape.size());
  unsigned currExpandDim = 0, currCollapseDim = 0;
  while (currExpandDim < expandedShape.size() &&
         currCollapseDim < collapsedShape.size()) {
    int64_t dstSize = collapsedShape[currCollapseDim];
    int64_t srcSize = expandedShape[currExpandDim];
    while (srcSize < dstSize && currExpandDim < expandedShape.size()) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      srcSize *= expandedShape[currExpandDim];
    }
    if (srcSize == dstSize) {
      reassociationMap[currCollapseDim].push_back(
          builder.getAffineDimExpr(currExpandDim++));
      // If the next dim in collapsedShape is not 1, treat subsequent dims in
      // expandedShape which are 1 to be collapsed.
      if (currCollapseDim == collapsedShape.size() - 1 ||
          collapsedShape[currCollapseDim + 1] != 1) {
        while (currExpandDim < expandedShape.size() &&
               expandedShape[currExpandDim] == 1) {
          reassociationMap[currCollapseDim].push_back(
              builder.getAffineDimExpr(currExpandDim++));
        }
      }
    }
    // If the reassociationMap for the currCollapseDim is empty, clear all
    // mappings and return false.
    if (reassociationMap[currCollapseDim].empty()) {
      reassociationMap.clear();
      return false;
    }
    currCollapseDim++;
  }
  // If both iterators didn't reach the end, we have leftover dimentions which
  // implies that we have a mismatch in shape.
  return currExpandDim == expandedShape.size() &&
         currCollapseDim == collapsedShape.size();
}

static Value sliceFirst(ConversionPatternRewriter &rewriter, Location loc,
                        Value input, int64_t dim, bool reverse = false) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  if (reverse)
    offsets[dim] = rewriter.getIndexAttr(inputType.getDimSize(dim) - 1);
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return tensor::ExtractSliceOp::create(rewriter, loc, input, offsets, sizes,
                                        strides);
}

static Value sliceRemaining(ConversionPatternRewriter &rewriter, Location loc,
                            Value input, int64_t dim, bool reverse = false) {
  ShapedType inputType = cast<ShapedType>(input.getType());
  auto sizes =
      llvm::to_vector(llvm::map_range(inputType.getShape(), [&](int64_t t) {
        return OpFoldResult(rewriter.getI64IntegerAttr(t));
      }));
  int64_t rank = inputType.getRank();
  // Retrieve slice sizes of input.
  sizes[dim] = rewriter.getIndexAttr(inputType.getDimSize(dim) - 1);
  // Retrieve slice offsets of input.
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  if (!reverse)
    offsets[dim] = rewriter.getIndexAttr(1);
  // Retrieve slice strides of input.
  SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));
  // Create the slice of input.
  return tensor::ExtractSliceOp::create(rewriter, loc, input, offsets, sizes,
                                        strides);
}

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

//-----------------------------
// Begin of monolithic only
//-----------------------------
struct AdvanceConverter : public OpConversionPattern<triton::AdvanceOp> {
  using OpConversionPattern<triton::AdvanceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::AdvanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrState pointerState;
    PtrAnalysis::rewriteAdvanceOp(op, rewriter, knownPtrs);
    return success();
  }
};

struct MakeTensorPtrConverter
    : public OpConversionPattern<triton::MakeTensorPtrOp> {
  using OpConversionPattern<triton::MakeTensorPtrOp>::OpConversionPattern;

  void populateVectorAsIndex(SmallVector<OpFoldResult> &vec,
                             Operation::operand_range ops,
                             ConversionPatternRewriter &rewriter,
                             Location loc) const {
    for (auto opnd : ops) {
      if (isa<IntegerType>(opnd.getType())) {
        auto castOp = arith::IndexCastOp::create(rewriter, loc,
                                                 rewriter.getIndexType(), opnd);
        vec.push_back(castOp.getResult());
      } else {
        assert(isa<IndexType>(opnd.getType()));
        vec.push_back(opnd);
      }
    }
  }

  LogicalResult
  matchAndRewrite(triton::MakeTensorPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    PtrState pointerState;

    auto orderSize = op.getOrder().size();
    if (orderSize > 1) {
      for (auto [first, second] :
           llvm::zip(op.getOrder().slice(0, orderSize - 2),
                     op.getOrder().slice(1, orderSize - 1))) {
        assert(first == second + 1 &&
               "Currently only support default order on block pointers");
      }
    }

    pointerState.source = rewriter.getRemappedValue(op.getBase());
    populateVectorAsIndex(pointerState.offsets, op.getOffsets(), rewriter, loc);
    populateVectorAsIndex(pointerState.strides, op.getStrides(), rewriter, loc);

    SmallVector<Value> newOffsets;
    for (auto [offset, stride] :
         llvm::zip(pointerState.offsets, pointerState.strides)) {
      auto mulOp = arith::MulIOp::create(rewriter, loc, cast<Value>(offset),
                                         cast<Value>(stride));
      newOffsets.push_back(mulOp.getResult());
    }

    pointerState.offsets.clear();

    for (auto offset : newOffsets) {
      pointerState.offsets.push_back(offset);
    }

    ArrayRef<int64_t> resultShape;
    auto pointerType =
        cast<mlir::triton::PointerType>(op.getResult().getType());
    if (auto shapedType = dyn_cast<ShapedType>(pointerType.getPointeeType())) {
      resultShape = shapedType.getShape();
      for (auto dim_size : resultShape) {
        pointerState.sizes.push_back(
            IntegerAttr::get(IntegerType::get(op.getContext(), 64), dim_size));
      }
    } else {
      // scalar pointer, should produce a one dimensional memref
      SmallVector<int64_t> scalarShape(1, 1);
      resultShape = scalarShape;
      assert(pointerState.getRank() == 1);
    }

    auto castOp = pointerState.createCastOp(resultShape, loc, rewriter);
    rewriter.replaceOp(op, castOp.getResult());
    return success();
  }
};

struct LegacyAddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::rewriteAddptrOp(op, rewriter, knownPtrs);
    return success();
  }
};

struct LoadConverter : public OpConversionPattern<triton::LoadOp> {
private:
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  void createSideBySideCopies(Value block1, Value block2, Value dst,
                              Location loc,
                              ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));

    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{zero, block1Col},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(0));
    auto one =
        arith::ConstantOp::create(rewriter, loc, rewriter.getIndexAttr(1));

    Value block1Row = memref::DimOp::create(rewriter, loc, block1, 0);
    Value block1Col = memref::DimOp::create(rewriter, loc, block1, 1);

    Value block2Row = memref::DimOp::create(rewriter, loc, block2, 0);
    Value block2Col = memref::DimOp::create(rewriter, loc, block2, 1);

    auto block1Dst = memref::SubViewOp::create(rewriter, loc, dst, /* offsets */
                                               ValueRange{zero, zero},
                                               /* sizes */
                                               ValueRange{block1Row, block1Col},
                                               /* strides */
                                               ValueRange{one, one});

    auto block2Dst = memref::SubViewOp::create(rewriter, loc, dst,
                                               /* offsets */
                                               ValueRange{block1Row, zero},
                                               /* sizes */
                                               ValueRange{block2Row, block2Col},
                                               /* strides */
                                               ValueRange{one, one});

    memref::CopyOp::create(rewriter, loc, block1, block1Dst);
    memref::CopyOp::create(rewriter, loc, block2, block2Dst);
  }

public:
  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto mask = op.getMask();
    auto other = op.getOther();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar loads
    if (!isa<ShapedType>(op.getResult().getType())) {
      auto sMemRef = PtrAnalysis::getScalarMemRef(op.getPtr(), adaptor.getPtr(),
                                                  loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      auto loadOp = affine::AffineLoadOp::create(rewriter, op.getLoc(), sMemRef,
                                                 zeroMap, ValueRange{});
      rewriter.replaceOp(op, loadOp.getResult());
      return success();
    }

    // 1. Simple case where no mask is used.
    auto type = dyn_cast<MemRefType>(ptr.getType());
    if (!type) {
      // Seen when implicit broadcasting is done late in a chain of operations.
      // The workaround is to broadcast the pointers early in the address
      // calculation. A proper fix is complicated, but at least we can provide a
      // better error message.
      return rewriter.notifyMatchFailure(
          op, "LoadOp expects a memref, not a memref of pointers");
    }

    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = memref::AllocOp::create(
        rewriter, loc, MemRefType::get(type.getShape(), type.getElementType()));

    // Initialize masked-load temporary buffer eagerly when `other` is absent.
    // This preserves Triton semantics where masked-out lanes read as zero.
    if (mask && !other) {
      auto zero =
          arith::ConstantOp::create(rewriter, loc,
                                    rewriter.getZeroAttr(type.getElementType()))
              .getResult();
      linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                             ValueRange{alloc});
    }

    if (!mask) {
      assert(!other && "other value used in non-masked load");
      if (auto unrealizedCast =
              ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
                ModuloState::WraparoundAttr)) {

          auto memrefs = unrealizedCast.getOperands();
          auto block1 = memrefs[0];
          auto block2 = memrefs[1];

          if (wrapType.getValue() == ModuloState::WraparoundSideBySide) {
            createSideBySideCopies(block1, block2, alloc, loc, rewriter);
          } else if (wrapType.getValue() == ModuloState::WraparoundStacked) {
            createStackedCopies(block1, block2, alloc, loc, rewriter);
          } else {
            llvm_unreachable("unexpected wraparound type");
          }
        } else {
          llvm_unreachable("unexpected unrealized cast op");
        }

      } else {
        memref::CopyOp::create(rewriter, loc, ptr, alloc);
      }

      Value tensor = bufferization::ToTensorOp::create(
          rewriter, loc, tensorType, alloc, true /* restrict */,
          true /* writable */);
      rewriter.replaceOp(op, tensor);

      return success();
    }

    // 2. Continuous masked loads.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed()) {
      return rewriter.notifyMatchFailure(
          op, "Cannot lower continuous masked loads");
    }

    // fill load destination with other value
    if (other) {
      auto scalarOther = getScalarValue(other, loc, rewriter);
      assert(scalarOther && "other value used in masked load produced by "
                            "unsupported instruction");
      scalarOther = castScalarToElementType(scalarOther, type.getElementType(),
                                            loc, rewriter);

      // For each dimension check if mstate.dims[i] < shape[i], or-accumulate
      // the result
      auto shape = type.getShape();
      auto accBase =
          arith::ConstantOp::create(rewriter, loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < type.getShape().size(); i++) {
        auto shapei = arith::ConstantOp::create(
            rewriter, loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = dyn_cast<Value>(mstate.dims[i]);
        if (!dimi) {
          dimi = arith::ConstantOp::create(
              rewriter, loc,
              cast<IntegerAttr>(cast<Attribute>(mstate.dims[i])));
        }

        auto cmpOp = arith::CmpIOp::create(
            rewriter, loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase =
            arith::OrIOp::create(rewriter, loc, accBase, cmpOp.getResult())
                .getResult();
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      scf::IfOp::create(
          rewriter, loc, accBase, [&](OpBuilder &builder, Location loc) {
            linalg::FillOp::create(builder, loc, ValueRange{scalarOther},
                                   ValueRange{alloc});
            scf::YieldOp::create(builder, loc);
          });
    }

    if (auto unrealizedCast = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (auto wrapType = unrealizedCast->getAttrOfType<StringAttr>(
              ModuloState::WraparoundAttr)) {

        auto memrefs = unrealizedCast.getOperands();
        auto block1 = memrefs[0];
        auto block2 = memrefs[1];

        if (wrapType.getValue() == ModuloState::WraparoundSideBySide) {
          auto [subview1, subview2] =
              mstate.getSideBySideSubviews(block1, block2, loc, rewriter);

          createSideBySideCopies(subview1, subview2, alloc, loc, rewriter);
        } else if (wrapType.getValue() == ModuloState::WraparoundStacked) {
          auto [subview1, subview2] =
              mstate.getStackedSubviews(block1, block2, loc, rewriter);

          createStackedCopies(subview1, subview2, alloc, loc, rewriter);
        } else {
          llvm_unreachable("unexpected wraparound type");
        }

      } else {
        llvm_unreachable("unexpected unrealized cast op");
      }

    } else {
      memref::SubViewOp srcSubview = mstate.getSubview(ptr, loc, rewriter);
      memref::SubViewOp dstSubview = mstate.getSubview(alloc, loc, rewriter);
      memref::CopyOp::create(rewriter, loc, srcSubview, dstSubview);
    }

    Value tensor = bufferization::ToTensorOp::create(rewriter, loc, tensorType,
                                                     alloc, true /* restrict */,
                                                     true /* writable */);
    rewriter.replaceOp(op, tensor);

    return success();
  }
};

struct StoreConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = adaptor.getPtr();
    auto val = adaptor.getValue();
    auto mask = op.getMask();
    auto loc = op.getLoc();

    // 0. Shortcut for scalar stores
    if (!isa<ShapedType>(val.getType())) {
      auto sMemRef =
          PtrAnalysis::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);
      auto zeroMap = AffineMap::getConstantMap(0, rewriter.getContext());
      affine::AffineStoreOp::create(rewriter, loc, val, sMemRef, zeroMap,
                                    ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    // 1. Simple case where no mask is used.
    if (!mask) {
      auto storeOp = bufferization::MaterializeInDestinationOp::create(
          rewriter, loc, val, ptr);
      storeOp.setWritable(true);
      rewriter.eraseOp(op);
      return success();
    }

    // 2. Continuous masked stores.
    // Analyze the mask operand to determine at runtime the size of the data we
    // are moving.
    MaskState mstate;
    auto isContMask = mstate.parse(mask, loc, rewriter);

    if (isContMask.failed())
      return failure();

    auto srcSlice = mstate.getExtractSlice(val, loc, rewriter);
    auto dstSubview = mstate.getSubview(ptr, loc, rewriter);

    auto storeOp = bufferization::MaterializeInDestinationOp::create(
        rewriter, loc, srcSlice, dstSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);

    return success();
  }
};

struct LoopConverter : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    llvm::SmallDenseMap<Value, PtrState> knownPtrs;
    PtrAnalysis::IndexMapSet
        levelToBlockArgIndex; // level -> set of block arg index to be replaced

    PtrAnalysis::rewriteForOp(op, rewriter, levelToBlockArgIndex, 0, knownPtrs);
    return success();
  }
};

struct YieldConverter : public OpConversionPattern<scf::YieldOp> {
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

// Remove all Meta ops except for AddPtr which is handled by AddPtrConverter.
// Use benefit == 10 to ensure that this pattern always takes precedence over
// other patterns.
struct MetaOpConverter : public RewritePattern {
private:
  // UseAnalysis will tag operations whose results are used only as meta-data
  // with "MetaUse" tag.
  bool isMetaUse(Operation *op) const { return op->hasAttr("MetaUse"); }

public:
  MetaOpConverter(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/10, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final {

    if (isa<triton::AddPtrOp>(op)) {
      return rewriter.notifyMatchFailure(op,
                                         "AddPtrOp will be handled separately");
    }

    if (isMetaUse(op)) {
      rewriter.eraseOp(op);
      return success();
    }

    return rewriter.notifyMatchFailure(op, "requires meta ops");
  }
};

struct UnrealizedCastConverter
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern<UnrealizedConversionCastOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

//-----------------------------
// End of monolithic only
//-----------------------------

struct SplatConverter : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto opType = cast<RankedTensorType>(op.getType());
    auto loc = op.getLoc();

    auto init = tensor::EmptyOp::create(rewriter, loc, opType.getShape(),
                                        opType.getElementType());

    Value filledTensor;
    if (isa<triton::PointerType>(adaptor.getSrc().getType())) {
      SmallVector<AffineMap> indexingMaps = {
          rewriter.getMultiDimIdentityMap(opType.getRank())};
      auto linalgOp = linalg::GenericOp::create(
          rewriter, loc, op->getResultTypes(), ValueRange{}, ValueRange{init},
          indexingMaps, getNParallelLoopsAttrs(opType.getRank()),
          [&](OpBuilder &nestedBuilder, Location nestedLoc,
              ValueRange blockArgs) {
            linalg::YieldOp::create(nestedBuilder, nestedLoc, adaptor.getSrc());
          });
      filledTensor = linalgOp.getResult(0);
    } else {
      filledTensor =
          linalg::FillOp::create(rewriter, loc, ValueRange{adaptor.getSrc()},
                                 ValueRange{init})
              .result();
    }

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    assert(op->getNumResults() == 1 && "code assumes single result!");
    RankedTensorType sourceType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    RankedTensorType resultType = cast<RankedTensorType>(op.getType());
    auto elementType = resultType.getElementType();
    size_t resultRank = resultType.getRank();

    SmallVector<AffineMap> indexingMaps;
    indexingMaps.reserve(op->getNumOperands() + op->getNumResults());

    indexingMaps.push_back(getBroadcastAffineMap(
        op->getContext(), sourceType.getShape(), resultType.getShape()));
    indexingMaps.append(op->getNumResults(),
                        rewriter.getMultiDimIdentityMap(resultRank));

    assert(op->getNumResults() == 1 && "code assumes single result!");
    auto init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                        elementType);

    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          linalg::YieldOp::create(nestedBuilder, loc, opResult);
        });

    linalgOp->setAttr("broadcastDims",
                      rewriter.getDenseI64ArrayAttr(
                          getBroadcastDims(sourceType, resultType)));

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct ExpandDimsConverter : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern<triton::ExpandDimsOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto src = adaptor.getSrc();
    auto srcRank = cast<RankedTensorType>(src.getType()).getRank();
    auto resType = cast<RankedTensorType>(op->getResultTypes()[0]);
    SmallVector<ReassociationIndices> reassoc;
    int64_t c = 0;
    for (int64_t i = 0; i < srcRank; i++) {
      ReassociationIndices g;
      g.push_back(c++);
      if (op.getAxis() == i) {
        g.push_back(c++);
      } else if (op.getAxis() == i + 1 && i == srcRank - 1) {
        g.push_back(c++);
      }
      reassoc.push_back(g);
    }

    auto expandShapeOp = tensor::ExpandShapeOp::create(rewriter, op.getLoc(),
                                                       resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res = getTransposedValue(adaptor.getSrc(), op.getLoc(), rewriter,
                                  op.getOrder());
    rewriter.replaceOp(op, res);
    return success();
  }
};

struct MakeRangeConverter : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto type = cast<TensorType>(op.getResult().getType());
    auto shape = type.getShape();
    auto elementType = type.getElementType();
    auto context = rewriter.getContext();

    assert(type.getShape().size() == 1 &&
           type.getElementType().getIntOrFloatBitWidth() == 32 &&
           "make range can only return 1D int32 tensor");

    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{mlir::getAffineDimExpr(0, context)}, context)};

    auto init = tensor::EmptyOp::create(rewriter, loc, shape, elementType);
    auto linalgOp = linalg::GenericOp::create(
        rewriter, loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = linalg::IndexOp::create(nestedBuilder, loc, 0);
          Value res = arith::IndexCastOp::create(nestedBuilder, loc,
                                                 type.getElementType(), index);
          if (op.getStart()) {
            auto start = mlir::arith::ConstantIntOp::create(
                rewriter, op.getLoc(), op.getStart(),
                type.getElementType().getIntOrFloatBitWidth());
            res = arith::AddIOp::create(nestedBuilder, loc, res, start);
          }
          linalg::YieldOp::create(nestedBuilder, loc, res);
        });

    rewriter.replaceOp(op, linalgOp->getResults());
    return success();
  }
};

struct AssertConverter : public OpConversionPattern<triton::AssertOp> {
  using OpConversionPattern<triton::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = rewriter.getContext();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto func = op->getParentOfType<FunctionOpInterface>();

    // 1. Extract program IDs from function arguments (same as PrintOpConverter)
    auto numArgs = func.getNumArguments();
    Value pid0 = func.getArgument(numArgs - 3);
    Value pid1 = func.getArgument(numArgs - 2);
    Value pid2 = func.getArgument(numArgs - 1);

    // 2. Reduce tensor condition to scalar i1 via AND reduction
    Value condVal = op.getCondition();
    Value scalarCond;
    if (isa<mlir::IntegerType>(condVal.getType())) {
      scalarCond = condVal;
    } else if (auto tensorType =
                   dyn_cast<RankedTensorType>(condVal.getType())) {
      scalarCond = reduceCondTensor(rewriter, loc, condVal, tensorType);
    } else {
      op.emitError("Unexpected type in triton::AssertOp");
      return failure();
    }

    // 3. Extract location info (file, line, func) from MLIR Location
    StringRef file = "unknown";
    StringRef funcName = "unknown";
    int line = 0;
    extractLocationInfo(loc, file, funcName, line);

    // 4. Create string constants for message, file, func
    Value msgStr = createAssertStringConstant(rewriter, loc, moduleOp,
                                              op.getMessage(), "assert_msg_");
    Value fileStr = createAssertStringConstant(rewriter, loc, moduleOp, file,
                                               "assert_file_");
    Value funcStr = createAssertStringConstant(rewriter, loc, moduleOp,
                                               funcName, "assert_func_");

    // 5. Create line number constant
    auto i32Type = IntegerType::get(ctx, 32);
    Value lineVal = arith::ConstantOp::create(rewriter, loc, i32Type,
                                              rewriter.getI32IntegerAttr(line));

    // 6. Call spine_assert(pid0, pid1, pid2, cond, msg, file, line, func)
    auto assertFn = getOrAddSpineAssertDecl(rewriter, moduleOp);
    SmallVector<Value> callArgs = {pid0,   pid1,    pid2,    scalarCond,
                                   msgStr, fileStr, lineVal, funcStr};
    LLVM::CallOp::create(rewriter, loc, assertFn, callArgs);

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Reduce a tensor<...xi1> to a scalar i1 via AND reduction.
  Value reduceCondTensor(ConversionPatternRewriter &rewriter, Location loc,
                         Value condTensor, RankedTensorType tensorType) const {
    auto ctx = rewriter.getContext();
    auto i1Type = IntegerType::get(ctx, 1);
    int64_t rank = tensorType.getRank();

    // Create rank-0 init tensor with value `true`
    auto initTensorType = RankedTensorType::get({}, i1Type);
    Value trueVal = arith::ConstantOp::create(
        rewriter, loc, i1Type, rewriter.getIntegerAttr(i1Type, 1));
    Value initTensor = bufferization::AllocTensorOp::create(
        rewriter, loc, initTensorType, ValueRange{});
    initTensor = tensor::InsertOp::create(rewriter, loc, trueVal, initTensor,
                                          ValueRange{});

    // linalg.reduce with AND across all dimensions
    SmallVector<int64_t> reductionDims;
    for (int64_t i = 0; i < rank; ++i)
      reductionDims.push_back(i);

    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc, condTensor, initTensor, reductionDims,
        [&](OpBuilder &b, Location innerLoc, ValueRange args) {
          Value andVal = arith::AndIOp::create(b, innerLoc, args[0], args[1]);
          linalg::YieldOp::create(b, innerLoc, andVal);
        });

    // Extract scalar from rank-0 tensor result
    Value result = reduceOp.getResults()[0];
    return tensor::ExtractOp::create(rewriter, loc, result, ValueRange{});
  }

  /// Extract file, func, line from MLIR Location (following triton-cpu).
  static void extractLocationInfo(Location loc, StringRef &file,
                                  StringRef &funcName, int &line) {
    while (auto callLoc = dyn_cast<CallSiteLoc>(loc))
      loc = callLoc.getCallee();

    if (auto fileLineColLoc = dyn_cast<FileLineColLoc>(loc)) {
      file = fileLineColLoc.getFilename();
      line = fileLineColLoc.getLine();
    }
  }

  /// Create a global string constant and return a pointer to it.
  Value createAssertStringConstant(ConversionPatternRewriter &rewriter,
                                   Location loc, ModuleOp moduleOp,
                                   StringRef str,
                                   StringRef globalNamePrefix) const {
    auto ctx = rewriter.getContext();

    std::string globalName = (globalNamePrefix + str).str();
    for (auto &c : globalName) {
      if (!llvm::isAlnum(c) && c != '_')
        c = '_';
    }
    if (globalName.size() > 64)
      globalName.resize(64);

    auto globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      std::string strWithNull = str.str() + '\0';
      auto strAttr = rewriter.getStringAttr(strWithNull);
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                                strWithNull.size());

      globalOp = LLVM::GlobalOp::create(
          rewriter, loc, arrayType,
          /*isConstant=*/true, LLVM::Linkage::Internal, globalName, strAttr);
    }

    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    return LLVM::AddressOfOp::create(rewriter, loc, ptrType, globalName)
        .getResult();
  }

  /// Get or create LLVM declaration for spine_assert.
  /// Signature: void spine_assert(i32, i32, i32, i1, ptr, ptr, i32, ptr)
  LLVM::LLVMFuncOp getOrAddSpineAssertDecl(ConversionPatternRewriter &rewriter,
                                           ModuleOp moduleOp) const {
    StringRef funcName = "spine_assert";
    if (auto existing = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return existing;

    auto ctx = rewriter.getContext();
    auto i32Type = IntegerType::get(ctx, 32);
    auto i1Type = IntegerType::get(ctx, 1);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto voidType = LLVM::LLVMVoidType::get(ctx);

    SmallVector<Type> argsType = {
        i32Type, i32Type, i32Type, // pid_x, pid_y, pid_z
        i1Type,                    // condition
        ptrType,                   // message
        ptrType,                   // file
        i32Type,                   // line
        ptrType                    // func
    };
    auto funcType =
        LLVM::LLVMFunctionType::get(voidType, argsType, /*isVarArg=*/false);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fn = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                       funcType);
    fn.setLinkage(LLVM::Linkage::External);
    return fn;
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Type inputType = adaptor.getSrc().getType();
    Type outputType = op.getResult().getType();
    if (inputType == outputType) {
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }

    // arith::bitcast does not support casting pointers
    if (triton::isPtrTypeLike(op.getType())) {
      return failure();
    }

    auto arithBitcast = arith::BitcastOp::create(rewriter, op.getLoc(),
                                                 op.getType(), op.getOperand());

    rewriter.replaceOp(op, arithBitcast.getResult());
    return success();
  }
};

struct CallConverter : public OpConversionPattern<triton::CallOp> {
  using OpConversionPattern<triton::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> args = adaptor.getOperands();

    // We need to pass extra arguments added by addProgramInfo which are
    // num_programs and program_ids
    if (FuncOp parentFunc = op->getParentOfType<triton::FuncOp>()) {
      SymbolRefAttr calleeAttr = op.getCalleeAttr();
      StringRef calleeName = calleeAttr.getRootReference();

      if (ModuleOp module = op->getParentOfType<ModuleOp>()) {
        if (FuncOp calleeFunc = module.lookupSymbol<FuncOp>(calleeName)) {
          size_t argsNeed = calleeFunc.getFunctionType().getInputs().size();
          Block &entryBlock = parentFunc.front();
          auto parentInputs = entryBlock.getArguments();
          size_t argsParent = parentInputs.size();

          if (argsNeed > args.size()) {
            int missing = argsNeed - args.size();
            int missingArgsStart = argsParent - missing;
            for (int i = 0; i < missing; i++) {
              args.push_back(parentInputs[missingArgsStart + i]);
            }
          }
        }
      }
    }

    auto call = func::CallOp::create(rewriter, op.getLoc(), op.getCallee(),
                                     op.getResultTypes(), args);

    if (!call) {
      op.emitError("Failed to create func::CallOp");
      return failure();
    }

    rewriter.replaceOp(op, call);
    return success();
  }
};

struct FpToFpConverter : public OpConversionPattern<triton::FpToFpOp> {
  using OpConversionPattern<triton::FpToFpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FpToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto roundingMode = triton::RoundingMode::RTNE; // default

    auto roundingModeAttr = op.getRounding();
    if (roundingModeAttr.has_value()) {
      roundingMode = roundingModeAttr.value();
    }

    assert(roundingMode != triton::RoundingMode::RTZ &&
           "Rounding Towards Zero is not supported");

    Type resultType = op.getResult().getType();

    auto operandWidth = getBitWidth(op.getOperand().getType());
    auto resultWidth = getBitWidth(resultType);

    assert(operandWidth.has_value() && resultWidth.has_value() &&
           "Not a float-like operand or result");

    if (operandWidth.value() > resultWidth.value()) {
      Value truncatedValue = arith::TruncFOp::create(
          rewriter, op.getLoc(), resultType, op.getOperand());
      rewriter.replaceOp(op, truncatedValue);
      return success();
    }

    Value extendedValue = arith::ExtFOp::create(rewriter, op.getLoc(),
                                                resultType, op.getOperand());
    rewriter.replaceOp(op, extendedValue);

    return success();
  }
};

struct ClampConverter : public OpConversionPattern<triton::ClampFOp> {
  using OpConversionPattern<triton::ClampFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ClampFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bool propagateNan = op.getPropagateNan() == triton::PropagateNan::ALL;

    Location loc = op.getLoc();
    Value x = adaptor.getOperands()[0];
    Value min = adaptor.getOperands()[1];
    Value max = adaptor.getOperands()[2];

    Value clamp;
    if (propagateNan) {
      Value maxMin = arith::MaximumFOp::create(rewriter, loc, x, min);
      clamp = arith::MinimumFOp::create(rewriter, loc, maxMin, max);
    } else {
      Value maxMin = arith::MaxNumFOp::create(rewriter, loc, x, min);
      clamp = arith::MinNumFOp::create(rewriter, loc, maxMin, max);
    }
    rewriter.replaceOp(op, clamp);

    return success();
  }
};

struct PreciseSqrtConverter
    : public OpConversionPattern<triton::PreciseSqrtOp> {
  using OpConversionPattern<triton::PreciseSqrtOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseSqrtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        math::SqrtOp::create(rewriter, op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct PreciseDivConverter : public OpConversionPattern<triton::PreciseDivFOp> {
  using OpConversionPattern<triton::PreciseDivFOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PreciseDivFOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement =
        arith::DivFOp::create(rewriter, op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct CatConverter : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement = tensor::ConcatOp::create(
        rewriter, op.getLoc(), 0 /* concat dimension */, adaptor.getOperands());

    rewriter.replaceOp(op, replacement);

    return success();
  }
};

struct SplitConverter : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern<triton::SplitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value input = op.getOperand();
    auto inputType = cast<RankedTensorType>(input.getType());

    Type resultType = op.getResults().front().getType();
    auto resultTensor = cast<RankedTensorType>(resultType);
    auto shape = inputType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    SmallVector<Value> results;

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      Value slice = tensor::ExtractSliceOp::create(
          rewriter, loc, resultTensor, input, offsets, sizes, strides);
      results.push_back(slice);
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct JoinConverter : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern<triton::JoinOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::JoinOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ValueRange inputs = op.getOperands();

    auto resultType = cast<RankedTensorType>(op.getResult().getType());

    auto loc = op.getLoc();
    Value result = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                           resultType.getElementType());

    auto shape = resultType.getShape();

    SmallVector<OpFoldResult> offsets(shape.size(), rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> strides(shape.size(), rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> sizes = llvm::to_vector(
        llvm::map_range(shape, [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    for (int i = 0; i < 2; ++i) {
      offsets.pop_back();
      sizes.pop_back();

      offsets.push_back(rewriter.getIndexAttr(i));
      sizes.push_back(rewriter.getIndexAttr(1));
      result = tensor::InsertSliceOp::create(rewriter, loc, inputs[i], result,
                                             offsets, sizes, strides);
    }

    rewriter.replaceOp(op, result);

    return success();
  }
};

struct MulHiUIOpConverter : public OpConversionPattern<triton::MulhiUIOp> {
  using OpConversionPattern<triton::MulhiUIOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MulhiUIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto mulResult =
        arith::MulUIExtendedOp::create(rewriter, loc, adaptor.getOperands());
    rewriter.replaceOp(op, mulResult.getHigh());

    return success();
  }
};

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  // true means tensor elements are zeros
  // false means not zero or it cannot be determined
  bool isZeroTensor(Value &v, bool integers) const {
    if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
      if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
        if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
          return val.getValueAsDouble() == 0.;
        }
        if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
          return val.getValue() == 0;
        }
      }
      return false;
    }

    if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          if (integers)
            return denseAttr.getSplatValue<APInt>().isZero();
          return denseAttr.getSplatValue<APFloat>().isZero();
        }
      }
    }

    return false;
  }

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // use adaptor values to respect earlier type conversions
    auto opa = adaptor.getA();
    auto opb = adaptor.getB();
    auto opc = adaptor.getC();

    llvm::SmallVector<xsmt::AnnotationOp> annotationOps;
    llvm::SmallDenseMap<StringAttr, Attribute> annotations;
    for (Operation *user : op->getUsers()) {
      if (auto annotateOp = dyn_cast<xsmt::AnnotationOp>(user)) {
        annotationOps.push_back(annotateOp);
        for (const NamedAttribute &attr : annotateOp->getAttrs()) {
          if (attr.getName() == "name" || attr.getName() == "loc")
            continue;
          annotations[attr.getName()] = attr.getValue();
        }
      }
    }

    auto dstType = cast<RankedTensorType>(op.getType());
    auto dstElemType = dstType.getElementType();
    bool integers = dstElemType.isInteger();
    bool skipC = isZeroTensor(opc, integers);

    // When the dot op lives inside an scf.for loop with f16 inputs but an f32
    // accumulator, perform the matmul in f16 and extend the result back to f32.
    auto opaType = dyn_cast<RankedTensorType>(opa.getType());
    auto opbType = dyn_cast<RankedTensorType>(opb.getType());
    bool inputsAreF16 = opaType && opbType &&
                        opaType.getElementType().isF16() &&
                        opbType.getElementType().isF16();
    bool useF16Matmul = (op->getParentOfType<scf::ForOp>() != nullptr) &&
                        inputsAreF16 && dstElemType.isF32();
    Type matmulElemType = useF16Matmul ? opaType.getElementType() : dstElemType;

    Value res;

    {
      auto init = tensor::EmptyOp::create(rewriter, loc, dstType.getShape(),
                                          matmulElemType);

      TypedAttr constantAttr =
          integers ? static_cast<TypedAttr>(
                         rewriter.getIntegerAttr(matmulElemType, 0))
                   : static_cast<TypedAttr>(
                         rewriter.getFloatAttr(matmulElemType, 0));

      auto zero = arith::ConstantOp::create(rewriter, loc, matmulElemType,
                                            constantAttr);

      auto zeroes = linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                                           ValueRange{init})
                        .result();

      auto matmulOp = linalg::MatmulOp::create(
          rewriter, loc, ValueRange{opa, opb}, ValueRange{zeroes});

      for (auto &kv : annotations) {
        matmulOp->setAttr(kv.getFirst(), kv.getSecond());
      }

      res = matmulOp.getResult(0);

      // Extend the f16 accumulator result back to the f32 destination type.
      if (useF16Matmul) {
        res = arith::ExtFOp::create(rewriter, loc, dstType, res);
      }

      if (!skipC) {
        if (integers) {
          res = arith::AddIOp::create(rewriter, loc, opc, res);
        } else {
          res = arith::AddFOp::create(rewriter, loc, opc, res);
        }
      }
    }

    rewriter.replaceOp(op, res);
    for (auto annotateOp : annotationOps) {
      rewriter.eraseOp(annotateOp);
    }

    return success();
  }
};

struct ReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

private:
  llvm::SmallVector<Operation *> getRedOps(triton::ReduceOp redOp) const {
    auto reduceBlock = redOp.getBody();
    return llvm::map_to_vector(reduceBlock->without_terminator(),
                               [](Operation &op) { return &op; });
  }

  bool isReductionOpSupported(Operation *redOp) const {
    return isa<arith::AddFOp, arith::AddIOp, arith::AndIOp, arith::MaximumFOp,
               arith::MulFOp, arith::MulIOp, arith::MaxNumFOp,
               arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
               arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::OrIOp,
               arith::XOrIOp>(redOp);
  }

  arith::ConstantOp getRedBaseConstOp(ConversionPatternRewriter &rewriter,
                                      Operation *redOp,
                                      Type constantType) const {
    const int64_t bitWidth = constantType.getIntOrFloatBitWidth();

    auto attr =
        llvm::TypeSwitch<Operation *, TypedAttr>(redOp)
            .Case([&](arith::AddFOp) {
              return rewriter.getFloatAttr(constantType, 0.f);
            })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case<arith::MaximumFOp, arith::MaxNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, -std::numeric_limits<float>::infinity());
            })
            .Case<arith::MinimumFOp, arith::MinNumFOp>([&](auto) {
              return rewriter.getFloatAttr(
                  constantType, std::numeric_limits<float>::infinity());
            })
            .Case([&](arith::MinSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxIntN(bitWidth));
            })
            .Case([&](arith::MinUIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Case([&](arith::MaxSIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::minIntN(bitWidth));
            })
            .Case<arith::MaxUIOp, arith::XOrIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 0); })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case<arith::MulIOp, arith::AndIOp>(
                [&](auto) { return rewriter.getIntegerAttr(constantType, 1); })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::AndIOp) {
              return rewriter.getIntegerAttr(constantType, 1);
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return arith::ConstantOp::create(rewriter, redOp->getLoc(), constantType,
                                     attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    if (keepFp16ReduceAccInLinalg() && elemType.isF16()) {
      return false;
    }
    unsigned width =
        cast<FloatType>(Float32Type::get(elemType.getContext())).getWidth();
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() < width &&
           isa<arith::AddFOp>(redOp);
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
          if (convertLhsToF32Precision) {
            lhs = arith::ExtFOp::create(b, loc,
                                        Float32Type::get(b.getContext()), lhs);
          }
          return decltype(redOp)::create(b, loc, lhs, rhs);
        })
        .Case([&](arith::MulFOp) {
          if (convertLhsToF32Precision) {
            lhs = arith::ExtFOp::create(b, loc,
                                        Float32Type::get(b.getContext()), lhs);
          }
          return arith::MulFOp::create(b, loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::AndIOp, arith::XOrIOp, arith::MaximumFOp,
              arith::MaxNumFOp, arith::MulIOp, arith::MinimumFOp,
              arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
              arith::MaxUIOp, arith::OrIOp, arith::AndIOp, arith::OrIOp>(
            [&](auto redOp) {
              return decltype(redOp)::create(b, loc, lhs, rhs);
            })
        .Default([](Operation *op) {
          op->dump();
          llvm_unreachable("Reduction op not yet supported");
          return nullptr;
        });
  }

  LogicalResult
  convertToLinalgReduce(triton::ReduceOp op,
                        typename triton::ReduceOp::Adaptor adaptor,
                        ConversionPatternRewriter &rewriter) const {
    auto source = adaptor.getOperands().front();
    auto sourceType = cast<RankedTensorType>(source.getType());
    auto elemType = sourceType.getElementType();
    auto resType = op.getResult().front().getType();
    auto loc = op.getLoc();
    auto reductionOps = getRedOps(op);

    // Reduction of arbitrary operations isn't supported because using the first
    // element across the reduction dimension requires us to iterate over a
    // subview that skips over each first element.
    if (reductionOps.size() != 1 ||
        !isReductionOpSupported(reductionOps.front())) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering reduction with body "
              "containing 1 max(i/f), addf, ori, or mulf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto rank = sourceType.getRank();
    auto isVectorReduce = (rank == 1);

    // if it is not a vector reduce, we can transpose the source
    // so that the reduction axis is the first dimension.
    if (!isVectorReduce && axis != 0) {
      SmallVector<int32_t> order;
      order.reserve(rank);
      order.push_back(axis);
      for (int i = 0; i < rank; ++i) {
        if (i != axis) {
          order.push_back(i);
        }
      }
      source = getTransposedValue(source, op.getLoc(), rewriter, order);
      axis = 0;
    }

    bool convertToF32Precision = requiresF32Conversion(resType, rop);

    auto constantType = convertToF32Precision
                            ? Float32Type::get(rewriter.getContext())
                            : elemType;

    auto accBaseConstOp = getRedBaseConstOp(rewriter, rop, constantType);
    Value initTensor;

    if (isVectorReduce) {
      // The affine vectorizer cannot vectorize affine loops generated from
      // linalg.reduce for the vector reduce case, so we must rewrite the
      // linalg.reduce to affine loops manually. Here we lower to AllocTensor
      // directly instead of EmptyOp so that the subsequent pass can recognize
      // the patterns (EmptyOp is susceptible to being CSE'd away, making it
      // harder to match the patterns correctly).
      initTensor = bufferization::AllocTensorOp::create(
          rewriter, loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = tensor::InsertOp::create(rewriter, loc, accBaseConstOp,
                                            initTensor, ValueRange{});
    } else {
      Value init = tensor::EmptyOp::create(
          rewriter, loc, cast<RankedTensorType>(resType).getShape(),
          constantType);
      initTensor =
          linalg::FillOp::create(rewriter, loc, ValueRange{accBaseConstOp},
                                 ValueRange{init})
              .result();
    }

    Value finalResult =
        linalg::ReduceOp::create(
            rewriter, loc, ValueRange{source}, ValueRange{initTensor},
            SmallVector<int64_t>{axis},
            [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
              assert(inputs.size() == 2);
              Value result = getRedElement(inputs[0], inputs[1], loc, rop,
                                           opBuilder, convertToF32Precision);
              linalg::YieldOp::create(opBuilder, loc, result);
            })
            .getResult(0);

    if (isVectorReduce) {
      finalResult =
          tensor::ExtractOp::create(rewriter, loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult =
          arith::TruncFOp::create(rewriter, loc, resType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
    return success();
  }

  LogicalResult
  convertToTensorExtract(triton::ReduceOp op,
                         typename triton::ReduceOp::Adaptor adaptor,
                         ConversionPatternRewriter &rewriter) const {
    assert(llvm::hasSingleElement(op.getSrcs()));

    auto returnOp = cast<triton::ReduceReturnOp>(*op.getOps().begin());
    assert(llvm::hasSingleElement(returnOp.getResult()));
    assert(cast<BlockArgument>(returnOp.getResult().front()).getArgNumber() ==
           0);

    auto source = op.getSrcs().front();
    auto zeroIdx =
        rewriter.createOrFold<arith::ConstantIndexOp>(op.getLoc(), 0);
    rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, source, zeroIdx);
    return success();
  }

public:
  LogicalResult
  matchAndRewrite(triton::ReduceOp op,
                  typename triton::ReduceOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto sourceType =
        cast<RankedTensorType>(adaptor.getOperands().front().getType());
    assert(sourceType.hasRank() && "Expected input is "
                                   "ranked");

    int64_t axis = op.getAxis();
    assert(axis >= 0 && axis < sourceType.getRank() &&
           "Expected reduction "
           "axis is within "
           "operand's rank");

    // Unsplat is implemented as a single element, rank 1 reduction where
    // single element is yielded immediately. This can be simplified into
    // a single element extract.
    if (llvm::hasSingleElement(op.getOps()) && sourceType.getRank() == 1 &&
        sourceType.getShape()[0] == 1) {
      return convertToTensorExtract(op, adaptor, rewriter);
    }

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  // We're looking for an op that looks like this:
  //
  // %9:2 = "tt.reduce"(%8, %3) <{axis = 0 : i32}> ({
  // ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
  // -------------------------------------------------
  // `matchTieBreakValue`                                |
  //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32         |
  //   %12 = arith.cmpi slt, %arg10, %arg12 : i32        |   1.
  //   %13 = arith.andi %11, %12 : i1                    |
  // -------------------------------------------------   |-> `matchShouldUpdate`
  // `matchUpdateCondition`                              |
  //   %14 = arith.cmpf ogt, %arg9, %arg11 : f32         |   2.
  // -------------------------------------------------   |
  //   %15 = arith.ori %14, %13 : i1                     |
  // -------------------------------------------------
  //   %16 = arith.select %15, %arg9, %arg11 : f32
  //   %17 = arith.select %15, %arg10, %arg12 : i32
  //   tt.reduce.return %16, %17 : f32, i32
  // }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
  //
  // The above mlir code is lowered from this combinator in triton's
  // standard.py:
  //
  //  def _argmax_combine(value1, index1, value2, index2, tie_break_left):
  //    if tie_break_left:
  //        tie = value1 == value2 and index1 < index2
  //    else:
  //        tie = False
  //    gt = value1 > value2 or tie
  //    v_ret = core.where(gt, value1, value2)
  //    i_ret = core.where(gt, index1, index2)
  //    return v_ret, i_ret

  LogicalResult matchTieBreakResult(Value currValue, Value currIndex,
                                    Value reduceValue, Value reduceIndex,
                                    mlir::Block::iterator &it,
                                    Value &tileBreakValue) const {
    // Match the following (section 1. of the above)
    //
    //   %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    //   %12 = arith.cmpi slt, %arg10, %arg12 : i32
    //   %13 = arith.andi %11, %12 : i1
    //
    // which is equivalent to the following python code
    //
    //   tie = value1 == value2 and index1 < index2

    // matching: %11 = arith.cmpf oeq, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto eqCmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (eqCmpOp) {
      if (eqCmpOp.getPredicate() != arith::CmpFPredicate::OEQ) {
        return failure();
      }
      if (currValue != eqCmpOp.getLhs() || reduceValue != eqCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %12 = arith.cmpi slt, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto sltCmpOp = dyn_cast<arith::CmpIOp>(*it++);
    if (sltCmpOp) {
      if (sltCmpOp.getPredicate() != arith::CmpIPredicate::slt) {
        return failure();
      }
      if (currIndex != sltCmpOp.getLhs() || reduceIndex != sltCmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: %13 = arith.andi %11, %12 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto andOp = dyn_cast<arith::AndIOp>(*it++);
    if (andOp) {
      if (andOp.getLhs() != eqCmpOp || andOp.getRhs() != sltCmpOp) {
        return failure();
      }
    } else {
      return failure();
    }

    tileBreakValue = andOp;
    return success();
  }

  LogicalResult matchShouldUpdateValue(Value currValue, Value currIndex,
                                       Value reduceValue, Value reduceIndex,
                                       mlir::Block::iterator &it,
                                       Value &shouldUpdate) const {
    Value tieResult;
    if (failed(matchTieBreakResult(currValue, currIndex, reduceValue,
                                   reduceIndex, it, tieResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Tie break result match failed\n");
      return failure();
    }

    Value comparisonResult;
    if (failed(T::matchComparisonResult(currValue, currIndex, reduceValue,
                                        reduceIndex, it, comparisonResult))) {
      LLVM_DEBUG(llvm::dbgs() << "Comparison result match failed\n");
      return failure();
    }

    // matching: %15 = arith.ori %14, %13 : i1
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto orOp = dyn_cast<arith::OrIOp>(*it++);
    if (orOp) {
      if (orOp.getLhs() != comparisonResult || orOp.getRhs() != tieResult) {
        return failure();
      }
    } else {
      return failure();
    }

    shouldUpdate = orOp;
    return success();
  }

  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue,
                      Location loc) const {
    if (shape.empty()) {
      auto initTensorType = RankedTensorType::get({}, fillValue.getType());
      Value initTensor = bufferization::AllocTensorOp::create(
          rewriter, loc, initTensorType, ValueRange{});
      return tensor::InsertOp::create(rewriter, loc, fillValue, initTensor,
                                      ValueRange{});
    }

    Value initTensor =
        tensor::EmptyOp::create(rewriter, loc, shape, fillValue.getType());
    return linalg::FillOp::create(rewriter, loc, ValueRange{fillValue},
                                  ValueRange{initTensor})
        .result();
  }

  LogicalResult rewriteScalarReduceToScfFor(ReduceOp op, OpAdaptor adaptor,
                                            ConversionPatternRewriter &rewriter,
                                            Value valuesAccBaseVal,
                                            Value indicesAccBaseVal) const {
    auto loc = op.getLoc();
    auto valueInputType =
        dyn_cast<RankedTensorType>(adaptor.getOperands()[0].getType());
    auto indexInputType =
        dyn_cast<RankedTensorType>(adaptor.getOperands()[1].getType());
    auto indexType = dyn_cast<IntegerType>(indicesAccBaseVal.getType());
    if (!valueInputType || !indexInputType || valueInputType.getRank() != 1 ||
        indexInputType.getRank() != 1 || !indexType ||
        valueInputType.isDynamicDim(0) || indexInputType.isDynamicDim(0) ||
        adaptor.getAxis() != 0)
      return failure();

    int64_t tripCount = valueInputType.getDimSize(0);
    if (tripCount != indexInputType.getDimSize(0))
      return failure();

    auto makeRangeOp = op.getOperands()[1].getDefiningOp<triton::MakeRangeOp>();
    if (!makeRangeOp || makeRangeOp.getStart() != 0 ||
        makeRangeOp.getEnd() != tripCount)
      return failure();

    Value lowerBound = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value upperBound = arith::ConstantIndexOp::create(rewriter, loc, tripCount);
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto forOp = scf::ForOp::create(
        rewriter, loc, lowerBound, upperBound, step,
        ValueRange{valuesAccBaseVal, indicesAccBaseVal},
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          Value currValue = tensor::ExtractOp::create(
              b, loc, adaptor.getOperands()[0], ValueRange{iv});
          Value currIndex = arith::IndexCastOp::create(b, loc, indexType, iv);
          Value reduceValue = iterArgs[0];
          Value reduceIndex = iterArgs[1];

          Value eq = arith::CmpFOp::create(b, loc, arith::CmpFPredicate::OEQ,
                                           currValue, reduceValue);
          Value slt = arith::CmpIOp::create(b, loc, arith::CmpIPredicate::slt,
                                            currIndex, reduceIndex);
          Value tie = arith::AndIOp::create(b, loc, eq, slt);
          Value cmp = T::createComparisonResult(b, loc, currValue, reduceValue);
          Value shouldUpdate = arith::OrIOp::create(b, loc, cmp, tie);
          Value nextValue = arith::SelectOp::create(b, loc, shouldUpdate,
                                                    currValue, reduceValue);
          Value nextIndex = arith::SelectOp::create(b, loc, shouldUpdate,
                                                    currIndex, reduceIndex);
          scf::YieldOp::create(b, loc, ValueRange{nextValue, nextIndex});
        });

    rewriter.replaceOp(op, forOp.getResults());
    return success();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override final {
    if (op.getBody()->getNumArguments() != 4) {
      return failure();
    }

    auto block = op.getBody();
    auto ops = block->without_terminator();

    Value currValue = block->getArgument(0);
    Value currIndex = block->getArgument(1);
    Value reduceValue = block->getArgument(2);
    Value reduceIndex = block->getArgument(3);

    auto opsIt = ops.begin();
    Value shouldUpdate;
    if (failed(matchShouldUpdateValue(currValue, currIndex, reduceValue,
                                      reduceIndex, opsIt, shouldUpdate))) {
      return failure();
    }

    // matching: %16 = arith.select %15, %arg9, %arg11 : f32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto valueSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (valueSelectOp) {
      if (valueSelectOp.getCondition() != shouldUpdate ||
          currValue != valueSelectOp.getTrueValue() ||
          reduceValue != valueSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching:%17 = arith.select %15, %arg10, %arg12 : i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto indexSelectOp = dyn_cast<arith::SelectOp>(*opsIt++);
    if (indexSelectOp) {
      if (indexSelectOp.getCondition() != shouldUpdate ||
          currIndex != indexSelectOp.getTrueValue() ||
          reduceIndex != indexSelectOp.getFalseValue()) {
        return failure();
      }
    } else {
      return failure();
    }

    // matching: tt.reduce.return %16, %17 : f32, i32
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *opsIt << "\n");
    auto termOp = dyn_cast<triton::ReduceReturnOp>(*opsIt++);
    if (termOp && termOp == block->getTerminator()) {
      auto opnds = termOp.getOperands();
      if (opnds != ArrayRef<Value>{valueSelectOp, indexSelectOp}) {
        return failure();
      }
    } else {
      return failure();
    }

    auto loc = op.getLoc();

    auto elemTypes = op.getElementTypes();

    // Set the initial value of the rank-0 tensor containing
    // the result value to either -inf or +inf depending on
    // whether we're dealing with argmax or argmin
    auto valueType = elemTypes[0];
    auto valuesAccBaseVal = arith::ConstantOp::create(
        rewriter, loc, valueType,
        rewriter.getFloatAttr(valueType, T::getBaseReductionValue()));

    // Set the initial value of the rank-0 tensor containing the index of the
    // min or max value to -1
    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = arith::ConstantOp::create(
        rewriter, loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    // Get the shape of the resulting tensors (both for values and indices). If
    // we are reducing to a single scalar, then the result's type is a tensor of
    // rank-0, otherwise we can reuse the original result shape
    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    if (isScalarReduce &&
        succeeded(rewriteScalarReduceToScfFor(
            op, adaptor, rewriter, valuesAccBaseVal, indicesAccBaseVal))) {
      return success();
    }

    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc)};

    auto linalgOp = linalg::ReduceOp::create(
        rewriter, loc, adaptor.getOperands(), outputs,
        SmallVector<int64_t>{adaptor.getAxis()},
        [&](OpBuilder &b, Location loc, ValueRange inputs) {
          assert(inputs.size() == 4);

          auto tritonReduceBlock = op.getBody();
          IRMapping mapping;
          mapping.map(tritonReduceBlock->getArguments(), inputs);

          for (auto &op : tritonReduceBlock->without_terminator()) {
            b.clone(op, mapping);
          }

          auto tritonYield = tritonReduceBlock->getTerminator();
          auto results =
              llvm::map_to_vector(tritonYield->getOperands(), [&](Value val) {
                return mapping.lookup(val);
              });
          linalg::YieldOp::create(b, loc, results);
        });

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          tensor::ExtractOp::create(rewriter, loc, valueType,
                                    linalgOp.getResults()[0], ValueRange{}),
          tensor::ExtractOp::create(rewriter, loc, indexType,
                                    linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
    return success();
  }
};

struct ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf ogt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OGT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return -std::numeric_limits<float>::infinity();
  }

  static Value createComparisonResult(OpBuilder &builder, Location loc,
                                      Value currValue, Value reduceValue) {
    return arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OGT,
                                 currValue, reduceValue);
  }

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

struct ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
  static LogicalResult matchComparisonResult(Value currValue, Value currIndex,
                                             Value reduceValue,
                                             Value reduceIndex,
                                             mlir::Block::iterator &it,
                                             Value &comparisonResult) {
    // %14 = arith.cmpf olt, %arg9, %arg11 : f32
    // This corresponds to section 2. of the sample snippet in
    // ArgMinMaxBaseConverter
    LLVM_DEBUG(llvm::dbgs() << "Matching: " << *it << "\n");
    auto cmpOp = dyn_cast<arith::CmpFOp>(*it++);
    if (cmpOp) {
      if (cmpOp.getPredicate() != arith::CmpFPredicate::OLT ||
          currValue != cmpOp.getLhs() || reduceValue != cmpOp.getRhs()) {
        return failure();
      }
    } else {
      return failure();
    }

    comparisonResult = cmpOp;
    return success();
  }

  static float getBaseReductionValue() {
    return std::numeric_limits<float>::infinity();
  }

  static Value createComparisonResult(OpBuilder &builder, Location loc,
                                      Value currValue, Value reduceValue) {
    return arith::CmpFOp::create(builder, loc, arith::CmpFPredicate::OLT,
                                 currValue, reduceValue);
  }

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {}
};

struct normalReduceConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Derive types. `tt.reduce` treats reducing a 1-D tensor with a special
    // case that returns a scalar, but we treat it as a 0-D tensor in these
    // types.
    auto convertedInputTensorTypes =
        llvm::map_range(adaptor.getOperands().getTypes(),
                        [](Type t) { return cast<TensorType>(t); });
    assert(llvm::all_equal(llvm::map_range(
        convertedInputTensorTypes, [](TensorType t) { return t.getShape(); })));
    static_cast<void>(convertedInputTensorTypes);

    auto originalResultTensorTypes =
        llvm::map_range(op.getResultTypes(), [](Type t) -> TensorType {
          if (auto tensorType = dyn_cast<TensorType>(t))
            return tensorType;
          return RankedTensorType::get({}, t);
        });
    assert(llvm::all_equal(llvm::map_range(
        originalResultTensorTypes, [](TensorType t) { return t.getShape(); })));
    ArrayRef<int64_t> resultShape =
        (*originalResultTensorTypes.begin()).getShape();
    auto convertedResultTensorTypes =
        llvm::map_range(originalResultTensorTypes, [&](TensorType t) {
          return RankedTensorType::get(resultShape, t.getElementType());
        });

    llvm::SmallVector<Value> initVals;
    llvm::SmallVector<Value> inputVals;
    // To lowering to linalg.reduce, we use the first slice of the reduction
    // axis of input operands as the init value of init operands. And then,
    // reduce the remaining elements of input operands.
    // We assume that the number of input operands is same as init operands and
    // corresponds one to one.
    // TODO: This restriction will need to be relaxed in the future.
    assert(adaptor.getOperands().size() == op.getNumResults() &&
           "tt.reduce requires the same input number and init number");
    for (auto [inputVal, initTy] :
         llvm::zip(adaptor.getOperands(), convertedResultTensorTypes)) {
      ShapedType inputTy = cast<ShapedType>(inputVal.getType());
      ArrayRef<int64_t> inputShape = inputTy.getShape();

      // If the size of reduce axis is 1, we will replace init operands by input
      // operands, so we should resize the input operands' shape by init
      // operands.
      if (inputShape[op.getAxis()] <= 1) {
        assert(inputVals.empty() &&
               "tt.reduce requires the same shape of all input operands");
        SmallVector<ReassociationExprs, 4> reassociationMap;
        [[maybe_unused]] bool res = createReassociationMaps(
            rewriter, inputShape, initTy.getShape(), reassociationMap);
        assert(res && "attempting to collapse into an incompatible shape");
        auto collapse = tensor::CollapseShapeOp::create(rewriter, loc, inputVal,
                                                        reassociationMap);
        initVals.push_back(collapse);
        continue;
      }

      // 1. Slice the first elements of input operands, and use them as init
      //    operands' init value.
      {
        Value slice = sliceFirst(rewriter, loc, inputVal, op.getAxis());
        auto sliceShape = cast<ShapedType>(slice.getType()).getShape();

        // Resize slice value's shape by init operand.
        SmallVector<ReassociationExprs, 4> reassociationMap;
        [[maybe_unused]] bool res = createReassociationMaps(
            rewriter, sliceShape, initTy.getShape(), reassociationMap);
        assert(res && "attempting to collapse into an incompatible shape");
        auto collapse = tensor::CollapseShapeOp::create(rewriter, loc, slice,
                                                        reassociationMap);
        initVals.push_back(collapse);
      }

      // 2. Slice the remaining elements of input operands, reduce them and
      //    init value.
      {
        Value slice = sliceRemaining(rewriter, loc, inputVal, op.getAxis());
        inputVals.push_back(slice);
      }
    }

    // If the results are scalar, we need to extract the scalar from the
    // 0-ranked result tensor.
    auto getFinalResults = [&](ValueRange results) -> SmallVector<Value> {
      if (!resultShape.empty())
        return results;
      SmallVector<Value> extractResults;
      for (auto [tensor, type] :
           llvm::zip(results, convertedResultTensorTypes)) {
        Value scalar =
            tensor::ExtractOp::create(rewriter, loc, type.getElementType(),
                                      tensor, /*indices=*/ValueRange{});
        extractResults.push_back(scalar);
      }
      return extractResults;
    };

    // If the the size of reduce axis is 1, we just replace the init operands by
    // input operands.
    if (inputVals.empty()) {
      rewriter.replaceOp(op, getFinalResults(initVals));
      return success();
    }

    // Create a linalg.reduce on the same input and move the combine region
    // there. (ReduceReturnOpConversion will take care of the terminator.)
    auto reduceOp = linalg::ReduceOp::create(
        rewriter, loc,
        /*resultTypes=*/SmallVector<Type>(convertedResultTensorTypes),
        /*inputs=*/inputVals, /*inits=*/initVals,
        /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
    rewriter.inlineRegionBefore(op.getCombineOp(), reduceOp.getCombiner(),
                                reduceOp.getCombiner().end());

    rewriter.replaceOp(op, getFinalResults(reduceOp.getResults()));
    return success();
  }
};

struct TritonReduceReturnPattern
    : public OpConversionPattern<triton::ReduceReturnOp> {
  using OpConversionPattern<triton::ReduceReturnOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

// get_program_id and get_num_programs:
// When launching triton kernels, we pass 6 additional arguments to indicate
// num_programs and program_id. Amongst those six, we have 3 arguments
// correspond to each axis for num_programs followed by 3 additional arguments
// for program_id.
//
// For instance, with triton kernel example_kernel(a, b, c), we have:
//  example_kernel(
//    a, b, c,
//    num_programs_axis_0,
//    num_programs_axis_1,
//    num_programs_axis_2,
//    program_id_axis_0,
//    program_id_axis_1,
//    program_id_axis_2,
//   )
//
struct GetProgramIDConverter
    : public OpConversionPattern<triton::GetProgramIdOp> {
  using OpConversionPattern<triton::GetProgramIdOp>::OpConversionPattern;
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

struct GetNumProgramsConverter
    : public OpConversionPattern<triton::GetNumProgramsOp> {
  using OpConversionPattern<triton::GetNumProgramsOp>::OpConversionPattern;

private:
  static uint32_t constexpr LAUNCH_GRID_RANK =
      getMaxEnumValForProgramIDDim() + 1;

public:
  GetNumProgramsConverter(MLIRContext *context)
      : OpConversionPattern(context) {}

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto axis = (uint32_t)op.getAxis();
    assert(axis < LAUNCH_GRID_RANK && "program_id expects "
                                      "axis to be either 0, "
                                      "1, or 2");

    auto func = op->getParentOfType<FunctionOpInterface>();
    auto numArgs = func.getNumArguments();
    auto id = func.getArgument(numArgs - LAUNCH_GRID_RANK * 2 + axis);

    rewriter.replaceOp(op, id);
    return success();
  }
};

// Convert a pair of cmpf and select to either min or max.
// Leave the pattern as simple as possible because triton has plans to emit
// min and max directly.
template <typename CmpOp>
struct MinMaxConverter : public OpRewritePattern<CmpOp> {
  using OpRewritePattern<CmpOp>::OpRewritePattern;

  MinMaxConverter(MLIRContext *context)
      : OpRewritePattern<CmpOp>(context, /*benefit=*/10) {}

  LogicalResult matchAndRewrite(CmpOp cmpOp,
                                PatternRewriter &rewriter) const final {
    if (!cmpOp.getResult().hasOneUse()) {
      return failure();
    }
    auto selectOp =
        dyn_cast<arith::SelectOp>(*cmpOp.getResult().getUsers().begin());
    if (!selectOp) {
      return failure();
    }

    if (!(cmpOp.getResult() == selectOp.getCondition() &&
          cmpOp.getLhs() == selectOp.getTrueValue() &&
          cmpOp.getRhs() == selectOp.getFalseValue())) {
      return failure();
    }

    rewriteOpWithMinMax(rewriter, cmpOp, selectOp, cmpOp.getPredicate());
    rewriter.eraseOp(cmpOp);

    return success();
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpFOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpFPredicate pred) const {
    switch (pred) {
    case arith::CmpFPredicate::OGT:
    case arith::CmpFPredicate::OGE:
      rewriter.replaceOpWithNewOp<arith::MaximumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    case arith::CmpFPredicate::OLT:
    case arith::CmpFPredicate::OLE:
      rewriter.replaceOpWithNewOp<arith::MinimumFOp>(selectOp, cmpOp.getLhs(),
                                                     cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }

  void rewriteOpWithMinMax(PatternRewriter &rewriter, arith::CmpIOp cmpOp,
                           arith::SelectOp selectOp,
                           arith::CmpIPredicate pred) const {
    switch (pred) {
    case arith::CmpIPredicate::sgt:
      rewriter.replaceOpWithNewOp<arith::MaxSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ugt:
      rewriter.replaceOpWithNewOp<arith::MaxUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::slt:
      rewriter.replaceOpWithNewOp<arith::MinSIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    case arith::CmpIPredicate::ult:
      rewriter.replaceOpWithNewOp<arith::MinUIOp>(selectOp, cmpOp.getLhs(),
                                                  cmpOp.getRhs());
      break;
    default:
      llvm_unreachable("Unhandled predicate");
    }
  }
};

struct DenseConstantConverter : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto attr = cast<DenseElementsAttr>(op.getValue());
    auto loc = op.getLoc();

    auto splatConst = arith::ConstantOp::materialize(
        rewriter, attr.getSplatValue<Attribute>(), attr.getElementType(), loc);

    auto init = tensor::EmptyOp::create(
        rewriter, loc,
        cast<RankedTensorType>(op.getResult().getType()).getShape(),
        attr.getElementType());

    rewriter.replaceOpWithNewOp<linalg::FillOp>(op, ValueRange{splatConst},
                                                ValueRange{init});

    return success();
  }
};

class CumSumConverter : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;

  // CumSum is a specific instance of Scan that looks like the following:
  //       %1 = "tt.scan"(%0) <{axis = 1 : i32}> ({
  //       ^bb0(%arg0: f32, %arg1: f32):
  //         %2 = arith.addf %arg0, %arg1 : f32
  //         tt.scan.return %2 : f32
  //       }) : (tensor<4x4xf32>) -> tensor<4x4xf32>
  bool isCumSum(triton::ScanOp op) const {
    auto scanBlock = op.getBody();
    auto ops = llvm::map_to_vector(scanBlock->without_terminator(),
                                   [](Operation &op) { return &op; });

    if (ops.size() != 1) {
      return false;
    }

    auto addOp = ops.front();
    if (isa<arith::AddFOp, arith::AddIOp>(addOp)) {
      if (addOp->getResult(0) != scanBlock->getTerminator()->getOperand(0)) {
        return false;
      }

      auto blockArgs =
          llvm::map_range(scanBlock->getArguments(), [](BlockArgument arg) {
            return dyn_cast<Value>(arg);
          });

      auto addArgs = addOp->getOperands();

      return DenseSet<Value>(blockArgs.begin(), blockArgs.end()) ==
             DenseSet<Value>(addArgs.begin(), addArgs.end());
    }

    return false;
  }

public:
  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isCumSum(op)) {
      return rewriter.notifyMatchFailure(
          op, "Only support cumsum variant of scan op");
    }

    auto input = op.getOperand(0);
    auto axis = op.getAxis();
    auto type = dyn_cast<RankedTensorType>(input.getType());

    if ((type.getRank() != 1 && type.getRank() != 2) ||
        axis != type.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering scan op to cumsum with rank "
              "= {1, 2} and axis = rank - 1");
    }

    Value init = tensor::EmptyOp::create(rewriter, op.getLoc(), type.getShape(),
                                         type.getElementType());

    rewriter.replaceOpWithNewOp<ttx::CumSumOp>(
        op, input, rewriter.getUI32IntegerAttr(axis), init);

    return success();
  }
};

class AddPtrConverter : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resType = op.getResult().getType();
    assert(isa<ShapedType>(resType));
    auto rank = cast<RankedTensorType>(resType).getRank();
    SmallVector<AffineMap, 3> indexingMaps(
        /*numResult + numOperands*/ 3, rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType, 6> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {op.getPtr()};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton op X operating on tensors of pointers to a linalg.generic
// wrapping op X to operate on single pointer.
// This pattern rewriter is almost identical to AddPtrConverter above, except
// that the out param for the linalg op is an empty op instead of reusing one
// of the existing operands. This is because depending on the templatized op,
// the type of the operands might be different, so we cannot pick a default
// operand to reuse for all cases.
template <typename OpType>
class TensorOpConverter : public OpConversionPattern<OpType> {
  using OpConversionPattern<OpType>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpType op, typename OpType::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTensorType =
        dyn_cast<RankedTensorType>(op.getResult().getType());
    if (!resultTensorType) {
      return failure();
    }
    auto rank = resultTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op->getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs = {tensor::EmptyOp::create(
        rewriter, op->getLoc(), resultTensorType.getShape(),
        resultTensorType.getElementType())};
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

// Convert triton store op operating on tensors of pointers to a linalg.generic
// wrapping op a triton store op on single pointer.
// Note that this linalg.generic op has an empty `out` param.
class StorePtrToLinalgConverter : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto storeTensorType = dyn_cast<RankedTensorType>(op.getValue().getType());
    if (!storeTensorType) {
      return failure();
    }
    auto rank = storeTensorType.getRank();
    SmallVector<AffineMap> indexingMaps(
        /*numResult + numOperands*/ op->getNumResults() + op.getNumOperands(),
        rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Value> outputs;
    rewriter.replaceOpWithNewOp<linalg::GenericOp>(
        op, op->getResultTypes(), op->getOperands(), outputs, indexingMaps,
        iteratorTypes,
        [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
          auto resultTypes =
              llvm::map_to_vector(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              });
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          linalg::YieldOp::create(builder, loc, scalarOp->getResults());
        });
    return success();
  }
};

class ReshapeConverter : public OpConversionPattern<triton::ReshapeOp> {
  using OpConversionPattern<triton::ReshapeOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getSrc();
    auto output = op.getResult();

    auto inputType = input.getType();
    auto outputType = output.getType();
    if (!outputType.hasStaticShape()) {
      return failure();
    }

    if (auto maybeReassociationMap =
            getReassociationIndicesForReshape(inputType, outputType)) {
      auto reassociationMap = *maybeReassociationMap;
      if (outputType.getRank() < inputType.getRank()) {
        rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
            op, outputType, input, reassociationMap);
      } else {
        rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
            op, outputType, input, reassociationMap);
      }
      return success();
    }

    ArrayRef<int64_t> outputShape = outputType.getShape();

    auto shape = arith::ConstantOp::create(
        rewriter, loc, rewriter.getI64TensorAttr(outputShape));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input,
                                                   shape);

    return success();
  }
};

class ConvertExternElementwise
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

  ConvertExternElementwise(MLIRContext *context)
      : OpConversionPattern(context) {
    opMap.insert({"linalg.exp",
                  {1, createLinalgOpFunc<linalg::ExpOp>(),
                   createMathUnaryOpFunc<math::ExpOp>()}});
    opMap.insert({"linalg.log",
                  {1, createLinalgOpFunc<linalg::LogOp>(),
                   createMathUnaryOpFunc<math::LogOp>()}});
    opMap.insert({"linalg.abs",
                  {1, createLinalgOpFunc<linalg::AbsOp>(),
                   createMathUnaryOpFunc<math::AbsFOp>()}});
    opMap.insert({"linalg.ceil",
                  {1, createLinalgOpFunc<linalg::CeilOp>(),
                   createMathUnaryOpFunc<math::CeilOp>()}});
    opMap.insert({"linalg.floor",
                  {1, createLinalgOpFunc<linalg::FloorOp>(),
                   createMathUnaryOpFunc<math::FloorOp>()}});
    opMap.insert({"linalg.round",
                  {1, createLinalgOpFunc<linalg::RoundOp>(),
                   createMathUnaryOpFunc<math::RoundOp>()}});
    opMap.insert({"linalg.rsqrt",
                  {1, createLinalgOpFunc<linalg::RsqrtOp>(),
                   createMathUnaryOpFunc<math::RsqrtOp>()}});
    opMap.insert({"linalg.sqrt",
                  {1, createLinalgOpFunc<linalg::SqrtOp>(),
                   createMathUnaryOpFunc<math::SqrtOp>()}});
    opMap.insert({"linalg.tanh",
                  {1, createLinalgOpFunc<linalg::TanhOp>(),
                   createMathUnaryOpFunc<math::TanhOp>()}});
    opMap.insert({"linalg.erf",
                  {1, createLinalgOpFunc<linalg::ErfOp>(),
                   createMathUnaryOpFunc<math::ErfOp>()}});
    opMap.insert({"linalg.powf",
                  {2, createLinalgOpFunc<linalg::PowFOp>(),
                   createMathBinaryOpFunc<math::PowFOp>()}});
  }

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto inputs = adaptor.getSrcs();

    auto symbol = op.getSymbol().str();
    auto it = opMap.find(symbol);
    if (it == opMap.end()) {
      return failure();
    }

    int requiredOperands = it->second.numOperands;
    auto createLinalgFunc = it->second.createLinalgFunc;
    auto createMathFunc = it->second.createMathFunc;

    if (inputs.size() < requiredOperands) {
      return failure();
    }

    SmallVector<Value> normalizedInputs(inputs.take_front(requiredOperands));
    auto resultType = op.getResult().getType();

    // For powf, allow int32 exponent input and normalize it to float before
    // running the existing type checks and lowering path.
    if (symbol == "linalg.powf" && normalizedInputs.size() >= 2) {
      Value exponent = normalizedInputs[1];
      if (auto exponentTensorType =
              dyn_cast<RankedTensorType>(exponent.getType())) {
        if (auto resultTensorType = dyn_cast<RankedTensorType>(resultType)) {
          Type resultElementType = resultTensorType.getElementType();
          Type exponentElementType = exponentTensorType.getElementType();
          if (exponentElementType.isInteger(32) &&
              isa<FloatType>(resultElementType)) {
            auto castType = RankedTensorType::get(exponentTensorType.getShape(),
                                                  resultElementType);
            normalizedInputs[1] =
                arith::SIToFPOp::create(rewriter, loc, castType, exponent);
          }
        }
      } else if (exponent.getType().isInteger(32) &&
                 isa<FloatType>(resultType)) {
        normalizedInputs[1] =
            arith::SIToFPOp::create(rewriter, loc, resultType, exponent);
      }
    }

    ValueRange opInputs(normalizedInputs);

    if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
      auto elementType = tensorType.getElementType();
      if (!isa<FloatType>(elementType)) {
        return failure();
      }

      for (Value input : opInputs) {
        auto inputTensorType = dyn_cast<RankedTensorType>(input.getType());
        if (!inputTensorType ||
            inputTensorType.getElementType() != elementType) {
          return rewriter.notifyMatchFailure(
              op, "extern elementwise tensor lowering requires input/output "
                  "element types to match");
        }
      }

      auto init = tensor::EmptyOp::create(rewriter, loc, tensorType.getShape(),
                                          elementType);

      Value output = init.getResult();
      SmallVector<Value, 1> outputs{output};

      Operation *resultOp = createLinalgFunc(rewriter, loc, opInputs, outputs);
      if (!resultOp || resultOp->getNumResults() == 0) {
        return failure();
      }

      rewriter.replaceOp(op, resultOp->getResult(0));
      return success();
    } else if (isa<FloatType>(resultType)) {
      for (Value input : opInputs) {
        if (!isa<FloatType>(input.getType()) || input.getType() != resultType) {
          return rewriter.notifyMatchFailure(
              op, "extern elementwise scalar lowering requires input/output "
                  "types to match");
        }
      }

      Operation *resultOp = createMathFunc(rewriter, loc, opInputs);
      if (!resultOp || resultOp->getNumResults() == 0) {
        return failure();
      }

      rewriter.replaceOp(op, resultOp->getResult(0));
      return success();
    }

    return failure();
  }

private:
  using CreateLinalgOpFunc = std::function<Operation *(
      ConversionPatternRewriter &, Location, ValueRange, ValueRange)>;
  using CreateMathOpFunc = std::function<Operation *(
      ConversionPatternRewriter &, Location, ValueRange)>;

  struct OpInfo {
    int numOperands;
    CreateLinalgOpFunc createLinalgFunc;
    CreateMathOpFunc createMathFunc;
  };

  template <typename OpType> static CreateLinalgOpFunc createLinalgOpFunc() {
    return [](ConversionPatternRewriter &rewriter, Location loc,
              ValueRange inputs, ValueRange outputs) -> Operation * {
      return OpType::create(rewriter, loc, inputs, outputs);
    };
  }

  template <typename OpType> static CreateMathOpFunc createMathUnaryOpFunc() {
    return [](ConversionPatternRewriter &rewriter, Location loc,
              ValueRange inputs) -> Operation * {
      if (inputs.size() != 1) {
        return nullptr;
      }

      Type resultType = inputs[0].getType();
      auto fastmath = arith::FastMathFlags::none;

      OperationState state(loc, OpType::getOperationName());
      OpType::build(rewriter, state, resultType, inputs[0], fastmath);
      return rewriter.create(state);
    };
  }

  template <typename OpType> static CreateMathOpFunc createMathBinaryOpFunc() {
    return [](ConversionPatternRewriter &rewriter, Location loc,
              ValueRange inputs) -> Operation * {
      if (inputs.size() != 2) {
        return nullptr;
      }

      Type resultType = inputs[0].getType();
      auto fastmath = arith::FastMathFlags::none;

      OperationState state(loc, OpType::getOperationName());
      OpType::build(rewriter, state, resultType, inputs[0], inputs[1],
                    fastmath);
      return rewriter.create(state);
    };
  }

  std::unordered_map<std::string, OpInfo> opMap;
};

class ConvertExternSpecialMath
    : public OpConversionPattern<triton::ExternElementwiseOp> {
public:
  using OpConversionPattern::OpConversionPattern;

private:
  enum class DivRoundingMode { RN, RZ, RD, RU };

  static std::optional<DivRoundingMode> getDivRoundingMode(StringRef symbol) {
    return llvm::StringSwitch<std::optional<DivRoundingMode>>(symbol)
        .Case("linalg.div_rn", DivRoundingMode::RN)
        .Case("linalg.div_rz", DivRoundingMode::RZ)
        .Case("linalg.div_rd", DivRoundingMode::RD)
        .Case("linalg.div_ru", DivRoundingMode::RU)
        .Default(std::nullopt);
  }

  static LogicalResult
  validateBinaryFloatOp(triton::ExternElementwiseOp op, ValueRange operands,
                        RankedTensorType outputType,
                        ConversionPatternRewriter &rewriter, StringRef opName) {
    if (operands.size() != 2) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << opName << " expects exactly two inputs";
      });
    }

    auto lhsType = dyn_cast<RankedTensorType>(operands[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(operands[1].getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << opName << " lowering requires ranked tensor operands";
      });
    }

    if (lhsType.getShape() != rhsType.getShape() ||
        lhsType.getShape() != outputType.getShape()) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << opName << " lowering requires matching operand/result shapes";
      });
    }

    auto lhsElemType = lhsType.getElementType();
    auto rhsElemType = rhsType.getElementType();
    auto outputElemType = outputType.getElementType();
    if (lhsElemType != rhsElemType || lhsElemType != outputElemType) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << opName
             << " lowering requires matching operand/result element types";
      });
    }

    if (!isa<FloatType>(lhsElemType)) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << opName << " lowering only supports float element types";
      });
    }

    return success();
  }

  static LogicalResult validateDivLikeOp(triton::ExternElementwiseOp op,
                                         ValueRange operands,
                                         RankedTensorType outputType,
                                         ConversionPatternRewriter &rewriter) {
    if (operands.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "linalg.div_r* expects exactly two inputs");
    }

    auto lhsType = dyn_cast<RankedTensorType>(operands[0].getType());
    auto rhsType = dyn_cast<RankedTensorType>(operands[1].getType());
    if (!lhsType || !rhsType) {
      return rewriter.notifyMatchFailure(
          op, "linalg.div_r* lowering requires ranked tensor operands");
    }

    if (lhsType.getShape() != rhsType.getShape() ||
        lhsType.getShape() != outputType.getShape()) {
      return rewriter.notifyMatchFailure(
          op, "linalg.div_r* lowering requires matching operand/result "
              "shapes");
    }

    auto lhsElemType = lhsType.getElementType();
    auto rhsElemType = rhsType.getElementType();
    auto outputElemType = outputType.getElementType();
    if (lhsElemType != rhsElemType || lhsElemType != outputElemType) {
      return rewriter.notifyMatchFailure(
          op, "linalg.div_r* lowering requires matching operand/result "
              "element types");
    }

    if (!isa<FloatType, IntegerType>(lhsElemType)) {
      return rewriter.notifyMatchFailure(
          op, "linalg.div_r* lowering only supports integer or float "
              "element types");
    }

    return success();
  }

  static Value buildIntegerDivOp(OpBuilder &b, Location loc, Value lhs,
                                 Value rhs, IntegerType intType) {
    if (intType.isUnsigned()) {
      return arith::DivUIOp::create(b, loc, lhs, rhs);
    }
    return arith::DivSIOp::create(b, loc, lhs, rhs);
  }

  static Value buildFloatDivOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                               DivRoundingMode mode) {
    Value quotient = arith::DivFOp::create(b, loc, lhs, rhs);
    switch (mode) {
    case DivRoundingMode::RN:
      return quotient;
    case DivRoundingMode::RZ:
      return math::TruncOp::create(b, loc, quotient);
    case DivRoundingMode::RD:
      return math::FloorOp::create(b, loc, quotient);
    case DivRoundingMode::RU:
      return math::CeilOp::create(b, loc, quotient);
    }
    llvm_unreachable("unsupported div rounding mode");
  }

  static Value buildDivLikeOp(OpBuilder &b, Location loc, Value lhs, Value rhs,
                              DivRoundingMode mode) {
    if (auto intType = dyn_cast<IntegerType>(lhs.getType())) {
      return buildIntegerDivOp(b, loc, lhs, rhs, intType);
    }
    return buildFloatDivOp(b, loc, lhs, rhs, mode);
  }

  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    StringRef symbol = op.getSymbol();
    bool isIsNaN = (symbol == "math.isnan");
    bool isIsInf = (symbol == "math.isinf");
    bool isFinite = (symbol == "math.isfinite" || symbol == "linalg.isfinited");
    bool isCos = (symbol == "math.cos");
    bool isSin = (symbol == "math.sin");
    bool isTrunc = (symbol == "math.trunc");
    bool isAtan2 = (symbol == "math.atan2");
    bool isFmod = (symbol == "linalg.fmod");
    auto divRoundingMode = getDivRoundingMode(symbol);
    bool isDivLike = divRoundingMode.has_value();

    if (!isIsNaN && !isIsInf && !isFinite && !isCos && !isSin && !isTrunc &&
        !isAtan2 && !isFmod && !isDivLike) {
      return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
        diag << "unsupported extern operation: " << symbol;
      });
    }

    Location loc = op.getLoc();
    auto operands = adaptor.getOperands();
    Value input = operands.front();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    if (!inputType) {
      return rewriter.notifyMatchFailure(op,
                                         "input is not a ranked tensor type");
    }

    auto outputType = cast<RankedTensorType>(op.getType());
    auto outputElemType = outputType.getElementType();
    auto inputElemType = inputType.getElementType();
    auto floatType = dyn_cast<FloatType>(inputElemType);

    if (!isDivLike && !floatType) {
      return rewriter.notifyMatchFailure(op, "element type is not float");
    }

    if ((isCos || isSin || isTrunc) &&
        (!isa<FloatType>(outputElemType) ||
         outputElemType != inputType.getElementType())) {
      return rewriter.notifyMatchFailure(
          op, "math.sin/math.cos/math.trunc lowering requires float output "
              "element type "
              "matching input element type");
    }

    if (isDivLike) {
      if (failed(validateDivLikeOp(op, operands, outputType, rewriter))) {
        return failure();
      }
    } else if (isAtan2) {
      if (failed(validateBinaryFloatOp(op, operands, outputType, rewriter,
                                       "math.atan2"))) {
        return failure();
      }
    } else if (isFmod) {
      if (failed(validateBinaryFloatOp(op, operands, outputType, rewriter,
                                       "linalg.fmod"))) {
        return failure();
      }
    }

    Value outputTensor = tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), outputElemType);

    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(
        inputType.getRank(), rewriter.getContext());
    SmallVector<AffineMap> indexingMaps;
    indexingMaps.append(operands.size(), identityMap);
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);

    OperationState state(loc, linalg::GenericOp::getOperationName());

    linalg::GenericOp::build(
        rewriter, state,
        /*resultTensorTypes=*/TypeRange{outputType},
        /*inputs=*/operands,
        /*outputs=*/ValueRange{outputTensor},
        /*indexingMaps=*/indexingMaps,
        /*iteratorTypes=*/iteratorTypes,
        /*doc=*/StringRef(),
        /*libraryCall=*/StringRef(),
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value inputVal = args[0];
          Value outputVal;

          if (isIsNaN) {
            Value specialOp = math::IsNaNOp::create(b, loc, inputVal);
            if (outputElemType.isInteger(1)) {
              outputVal = specialOp;
            } else if (auto intTy = dyn_cast<IntegerType>(outputElemType)) {
              Value one = arith::ConstantOp::create(b, loc, intTy,
                                                    b.getIntegerAttr(intTy, 1));
              Value zero = arith::ConstantOp::create(
                  b, loc, intTy, b.getIntegerAttr(intTy, 0));
              outputVal = arith::SelectOp::create(b, loc, specialOp, one, zero);
            } else {
              outputVal =
                  arith::ExtUIOp::create(b, loc, outputElemType, specialOp);
            }
          } else if (isIsInf) { // isIsInf
            // Avoid math.isinf canonicalization paths that can introduce
            // problematic i1 scalable-vector conversions in some backends.
            auto elemTy = cast<FloatType>(inputVal.getType());
            Value absVal = math::AbsFOp::create(b, loc, inputVal);
            APFloat inf = APFloat::getInf(elemTy.getFloatSemantics());
            Value infCst = arith::ConstantOp::create(
                b, loc, elemTy, FloatAttr::get(elemTy, inf));
            Value specialOp = arith::CmpFOp::create(
                b, loc, arith::CmpFPredicate::OEQ, absVal, infCst);

            if (outputElemType.isInteger(1)) {
              outputVal = specialOp;
            } else if (auto intTy = dyn_cast<IntegerType>(outputElemType)) {
              Value one = arith::ConstantOp::create(b, loc, intTy,
                                                    b.getIntegerAttr(intTy, 1));
              Value zero = arith::ConstantOp::create(
                  b, loc, intTy, b.getIntegerAttr(intTy, 0));
              outputVal = arith::SelectOp::create(b, loc, specialOp, one, zero);
            } else {
              outputVal =
                  arith::ExtUIOp::create(b, loc, outputElemType, specialOp);
            }
          } else if (isFinite) {
            Value specialOp = math::IsFiniteOp::create(b, loc, inputVal);
            if (outputElemType.isInteger(1)) {
              outputVal = specialOp;
            } else if (auto intTy = dyn_cast<IntegerType>(outputElemType)) {
              Value one = arith::ConstantOp::create(b, loc, intTy,
                                                    b.getIntegerAttr(intTy, 1));
              Value zero = arith::ConstantOp::create(
                  b, loc, intTy, b.getIntegerAttr(intTy, 0));
              outputVal = arith::SelectOp::create(b, loc, specialOp, one, zero);
            } else {
              outputVal =
                  arith::ExtUIOp::create(b, loc, outputElemType, specialOp);
            }
          } else if (isCos) {
            outputVal = math::CosOp::create(b, loc, inputVal);
          } else if (isTrunc) {
            outputVal = math::TruncOp::create(b, loc, inputVal);
          } else if (isAtan2) {
            outputVal = math::Atan2Op::create(b, loc, args[0], args[1]);
          } else if (isFmod) {
            outputVal = arith::RemFOp::create(b, loc, args[0], args[1]);
          } else if (isDivLike) {
            outputVal =
                buildDivLikeOp(b, loc, args[0], args[1], *divRoundingMode);
          } else {
            assert(isSin && "expected math.sin path");
            outputVal = math::SinOp::create(b, loc, inputVal);
          }

          linalg::YieldOp::create(b, loc, outputVal);
        });

    auto genericOp = cast<linalg::GenericOp>(rewriter.create(state));

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct ConvertExternGeluNone
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern<triton::ExternElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSymbol().str() != "linalg.gelu_none") {
      return failure();
    }

    Value input = op.getOperand(0);
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    auto inputElemTy = inputType.getElementType();
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto outputElemTy = outputType.getElementType();

    auto floatTy = dyn_cast<FloatType>(inputElemTy);
    if (!floatTy || inputElemTy != outputElemTy) {
      return rewriter.notifyMatchFailure(
          op,
          "linalg.gelu_none lowering requires float input and matching output "
          "element type");
    }

    Location loc = op.getLoc();

    Value initTensor =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), ValueRange{});

    SmallVector<AffineMap> indexingMaps;
    auto ctx = rewriter.getContext();
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Attribute> iteratorAttrs;
    for (auto type : iteratorTypes) {
      iteratorAttrs.push_back(
          linalg::IteratorTypeAttr::get(rewriter.getContext(), type));
    }

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, initTensor.getType(), input, initTensor,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorAttrs), nullptr, nullptr);

    Block *body = new Block();
    genericOp.getRegion().push_back(body);
    body->addArguments({inputElemTy, outputElemTy}, {loc, loc});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    // output = 0.5 * x * (1 + erf(x * 0.7071067811))
    Value c0_5 = arith::ConstantOp::create(rewriter, loc, inputElemTy,
                                           rewriter.getFloatAttr(floatTy, 0.5));
    Value c0_07071067811 =
        arith::ConstantOp::create(rewriter, loc, inputElemTy,
                                  rewriter.getFloatAttr(floatTy, 0.7071067811));
    Value c1 = arith::ConstantOp::create(rewriter, loc, inputElemTy,
                                         rewriter.getFloatAttr(floatTy, 1.0));

    Value x = body->getArgument(0);

    // x * 0.7071067811
    Value xSq = arith::MulFOp::create(rewriter, loc, x, c0_07071067811);

    // erf(x * 0.7071067811)
    Value xerf = math::ErfOp::create(rewriter, loc, xSq);

    // 1 + erf(x * 0.7071067811)
    Value poly = arith::AddFOp::create(rewriter, loc, xerf, c1);

    // 0.5 * x
    Value halfX = arith::MulFOp::create(rewriter, loc, x, c0_5);

    // 0.5 * x * (1 + erf(x * 0.7071067811))
    Value result = arith::MulFOp::create(rewriter, loc, halfX, poly);

    linalg::YieldOp::create(rewriter, loc, result);

    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

struct ConvertExternGeluTanh
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern<triton::ExternElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSymbol().str() != "linalg.gelu_tanh") {
      return failure();
    }

    Value input = op.getOperand(0);
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    auto inputElemTy = inputType.getElementType();
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto outputElemTy = outputType.getElementType();

    auto floatTy = dyn_cast<FloatType>(inputElemTy);
    if (!floatTy || inputElemTy != outputElemTy) {
      return rewriter.notifyMatchFailure(
          op,
          "linalg.gelu_tanh lowering requires float input and matching output "
          "element type");
    }

    Location loc = op.getLoc();

    Value initTensor =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), ValueRange{});

    SmallVector<AffineMap> indexingMaps;
    auto ctx = rewriter.getContext();
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Attribute> iteratorAttrs;
    for (auto type : iteratorTypes) {
      iteratorAttrs.push_back(
          linalg::IteratorTypeAttr::get(rewriter.getContext(), type));
    }

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, initTensor.getType(), input, initTensor,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorAttrs), nullptr, nullptr);

    Block *body = new Block();
    genericOp.getRegion().push_back(body);
    body->addArguments({inputElemTy, outputElemTy}, {loc, loc});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    // 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 *
    // pow(x.to(tl.float32), 2))))
    Value c0_5 = arith::ConstantOp::create(rewriter, loc, inputElemTy,
                                           rewriter.getFloatAttr(floatTy, 0.5));
    Value c0_044715 = arith::ConstantOp::create(
        rewriter, loc, inputElemTy, rewriter.getFloatAttr(floatTy, 0.044715));
    Value c0_797885 = arith::ConstantOp::create(
        rewriter, loc, inputElemTy, rewriter.getFloatAttr(floatTy, 0.79788456));
    Value c1 = arith::ConstantOp::create(rewriter, loc, inputElemTy,
                                         rewriter.getFloatAttr(floatTy, 1.0));

    Value x = body->getArgument(0);

    // x²
    Value xSq = arith::MulFOp::create(rewriter, loc, x, x);

    // 0.044715 * x²
    Value term = arith::MulFOp::create(rewriter, loc, c0_044715, xSq);

    // 1 + 0.044715x²
    Value poly = arith::AddFOp::create(rewriter, loc, term, c1);

    // x * poly
    Value xPoly = arith::MulFOp::create(rewriter, loc, x, poly);

    // apply 0.79788456
    Value scaled = arith::MulFOp::create(rewriter, loc, xPoly, c0_797885);

    // tanh
    Value tanhVal = math::TanhOp::create(rewriter, loc, scaled);

    // 1 + tanh
    Value sum = arith::AddFOp::create(rewriter, loc, tanhVal, c1);

    // 0.5 * x
    Value halfX = arith::MulFOp::create(rewriter, loc, x, c0_5);

    Value result = arith::MulFOp::create(rewriter, loc, halfX, sum);

    linalg::YieldOp::create(rewriter, loc, result);

    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

struct ConvertExternSilu
    : public OpRewritePattern<triton::ExternElementwiseOp> {
  using OpRewritePattern<triton::ExternElementwiseOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ExternElementwiseOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getSymbol().str() != "linalg.silu") {
      return failure();
    }

    Value input = op.getOperand(0);
    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    int64_t rank = inputType.getRank();
    auto inputElemTy = inputType.getElementType();
    auto outputType = mlir::cast<RankedTensorType>(op.getResult().getType());
    auto outputElemTy = outputType.getElementType();

    if (inputElemTy != outputElemTy) {
      return rewriter.notifyMatchFailure(
          op, "linalg.silu lowering requires input and output element types to "
              "match");
    }

    Location loc = op.getLoc();

    Value initTensor =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), ValueRange{});

    SmallVector<AffineMap> indexingMaps;
    auto ctx = rewriter.getContext();
    AffineMap identityMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
    indexingMaps.push_back(identityMap);
    indexingMaps.push_back(identityMap);

    SmallVector<utils::IteratorType> iteratorTypes(
        rank, utils::IteratorType::parallel);
    SmallVector<Attribute> iteratorAttrs;
    for (auto type : iteratorTypes) {
      iteratorAttrs.push_back(
          linalg::IteratorTypeAttr::get(rewriter.getContext(), type));
    }

    auto genericOp = linalg::GenericOp::create(
        rewriter, loc, initTensor.getType(), input, initTensor,
        rewriter.getAffineMapArrayAttr(indexingMaps),
        rewriter.getArrayAttr(iteratorAttrs), nullptr, nullptr);

    Block *body = new Block();
    genericOp.getRegion().push_back(body);
    body->addArguments({inputElemTy, outputElemTy}, {loc, loc});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(body);

    // Compute SiLU in input element type (e.g. f16 path).
    Value c0 = arith::ConstantOp::create(
        rewriter, loc, inputElemTy, rewriter.getFloatAttr(inputElemTy, 0.0));
    Value c1 = arith::ConstantOp::create(
        rewriter, loc, inputElemTy, rewriter.getFloatAttr(inputElemTy, 1.0));

    Value x = body->getArgument(0);
    Value xneg = arith::SubFOp::create(rewriter, loc, c0, x);
    Value xexp = math::ExpOp::create(rewriter, loc, xneg);
    Value xaddf = arith::AddFOp::create(rewriter, loc, xexp, c1);
    Value result = arith::DivFOp::create(rewriter, loc, x, xaddf);

    if (result.getType() != outputElemTy) {
      auto srcWidth = cast<FloatType>(result.getType()).getWidth();
      auto dstWidth = cast<FloatType>(outputElemTy).getWidth();
      if (srcWidth > dstWidth)
        result = arith::TruncFOp::create(rewriter, loc, outputElemTy, result);
      else
        result = arith::ExtFOp::create(rewriter, loc, outputElemTy, result);
    }

    linalg::YieldOp::create(rewriter, loc, result);

    rewriter.replaceOp(op, genericOp.getResult(0));

    return success();
  }
};

struct TritonPtrToIntPattern : public OpConversionPattern<triton::PtrToIntOp> {
private:
  using OpConversionPattern::OpConversionPattern;

public:
  TritonPtrToIntPattern(const TypeConverter &typeConverter,
                        MLIRContext *context)
      : OpConversionPattern<triton::PtrToIntOp>(typeConverter, context) {}

  LogicalResult
  matchAndRewrite(triton::PtrToIntOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    Value src = adaptor.getSrc();
    Type targetType = rewriter.getIntegerType(64);

    rewriter.replaceOpWithNewOp<LLVM::PtrToIntOp>(op, targetType, src);
    return success();
  }
};

class ExternElementwiseBinaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!op.getPure() || op.getSrcs().size() != 2)
      return failure();
#define POPULATE_BINARY_OP(FUNC_NAME, DST_OP)                                  \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0], op.getSrcs()[1]); \
    return success();                                                          \
  }

    POPULATE_BINARY_OP("__nv_atan2f", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_atan2", math::Atan2Op);
    POPULATE_BINARY_OP("__nv_powf", math::PowFOp);
    POPULATE_BINARY_OP("__nv_pow", math::PowFOp);

#undef POPULATE_BINARY_OP
    return failure();
  }
};

class ExternElementwiseUnaryOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!op.getPure() || op.getSrcs().size() != 1)
      return failure();
#define POPULATE_UNARY_OP(FUNC_NAME, DST_OP)                                   \
  if (!op.getSymbol().compare(FUNC_NAME)) {                                    \
    rewriter.replaceOpWithNewOp<DST_OP>(op, op.getSrcs()[0]);                  \
    return success();                                                          \
  }

    POPULATE_UNARY_OP("__nv_fabsf", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_fabs", math::AbsFOp);
    POPULATE_UNARY_OP("__nv_sinf", math::SinOp);
    POPULATE_UNARY_OP("__nv_sin", math::SinOp);
    POPULATE_UNARY_OP("__nv_cosf", math::CosOp);
    POPULATE_UNARY_OP("__nv_cos", math::CosOp);
    POPULATE_UNARY_OP("__nv_tanf", math::TanOp);
    POPULATE_UNARY_OP("__nv_tan", math::TanOp);
    POPULATE_UNARY_OP("__nv_asinf", math::AsinOp);
    POPULATE_UNARY_OP("__nv_asin", math::AsinOp);
    POPULATE_UNARY_OP("__nv_acosf", math::AcosOp);
    POPULATE_UNARY_OP("__nv_acos", math::AcosOp);
    POPULATE_UNARY_OP("__nv_atanf", math::AtanOp);
    POPULATE_UNARY_OP("__nv_atan", math::AtanOp);
    POPULATE_UNARY_OP("__nv_sinhf", math::SinhOp);
    POPULATE_UNARY_OP("__nv_sinh", math::SinhOp);
    POPULATE_UNARY_OP("__nv_coshf", math::CoshOp);
    POPULATE_UNARY_OP("__nv_cosh", math::CoshOp);
    POPULATE_UNARY_OP("__nv_tanhf", math::TanhOp);
    POPULATE_UNARY_OP("__nv_tanhf", math::TanhOp);
    POPULATE_UNARY_OP("__nv_acoshf", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_acosh", math::AcoshOp);
    POPULATE_UNARY_OP("__nv_asinhf", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_asinh", math::AsinhOp);
    POPULATE_UNARY_OP("__nv_atanhf", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_atanhf", math::AtanhOp);
    POPULATE_UNARY_OP("__nv_logf", math::LogOp);
    POPULATE_UNARY_OP("__nv_log", math::LogOp);
    POPULATE_UNARY_OP("__nv_log10f", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log10", math::Log10Op);
    POPULATE_UNARY_OP("__nv_log1pf", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_log1p", math::Log1pOp);
    POPULATE_UNARY_OP("__nv_expf", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp", math::ExpOp);
    POPULATE_UNARY_OP("__nv_exp2f", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_exp2", math::Exp2Op);
    POPULATE_UNARY_OP("__nv_erff", math::ErfOp);
    POPULATE_UNARY_OP("__nv_erf", math::ErfOp);
    POPULATE_UNARY_OP("__nv_sqrtf", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_sqrt", math::SqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrtf", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_rsqrt", math::RsqrtOp);
    POPULATE_UNARY_OP("__nv_ceilf", math::CeilOp);
    POPULATE_UNARY_OP("__nv_ceil", math::CeilOp);
    POPULATE_UNARY_OP("__nv_floorf", math::FloorOp);
    POPULATE_UNARY_OP("__nv_floor", math::FloorOp);
    POPULATE_UNARY_OP("__nv_truncf", math::TruncOp);
    POPULATE_UNARY_OP("__nv_trunc", math::TruncOp);

#undef POPULATE_UNARY_OP
    return failure();
  }
};

static void populateExternElementwiseOpToMLIROps(RewritePatternSet &patterns) {
  patterns.add<ExternElementwiseBinaryOpConverter,
               ExternElementwiseUnaryOpConverter>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// PrintOp Conversion
//===----------------------------------------------------------------------===//

struct PrintOpConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern<triton::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto func = op->getParentOfType<FunctionOpInterface>();

    // Extract program IDs from function arguments
    // spine-triton convention: last 6 args = [num_progs_x/y/z, pid_x/y/z]
    auto numArgs = func.getNumArguments();
    Value pid0 = func.getArgument(numArgs - 3); // pid_x
    Value pid1 = func.getArgument(numArgs - 2); // pid_y
    Value pid2 = func.getArgument(numArgs - 1); // pid_z

    StringRef prefix = op.getPrefix();
    bool hex = op.getHex();
    auto isSigned = op.getIsSigned();

    // Case 1: No operands (string-only print)
    if (op.getNumOperands() == 0) {
      createPrintfCall(rewriter, loc, moduleOp, prefix, pid0, pid1, pid2,
                       std::nullopt, false, false);
      rewriter.eraseOp(op);
      return success();
    }

    // Case 2: Process each operand
    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = adaptor.getOperands()[i];
      bool isSignedVal = isSigned[i];

      if (auto tensorType = dyn_cast<RankedTensorType>(operand.getType())) {
        // Tensor operand: alloc memref + call spine_print_unranked_memref
        createTensorPrintCall(rewriter, loc, moduleOp, prefix, operand,
                              tensorType, pid0, pid1, pid2, isSignedVal, hex);
      } else {
        // Scalar operand: call printf
        createPrintfCall(rewriter, loc, moduleOp, prefix, pid0, pid1, pid2,
                         operand, isSignedVal, hex);
      }
    }

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// Create a global string constant and return a pointer to it.
  /// Uses LLVM::GlobalOp pattern (same as XSMTToLinalg proton support).
  Value createStringConstant(ConversionPatternRewriter &rewriter, Location loc,
                             ModuleOp moduleOp, StringRef str,
                             StringRef globalNamePrefix) const {
    auto ctx = rewriter.getContext();

    // Build unique global name
    std::string globalName = (globalNamePrefix + str).str();
    for (auto &c : globalName) {
      if (!llvm::isAlnum(c) && c != '_')
        c = '_';
    }
    if (globalName.size() > 64)
      globalName.resize(64);

    auto globalOp = moduleOp.lookupSymbol<LLVM::GlobalOp>(globalName);
    if (!globalOp) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      std::string strWithNull = str.str() + '\0';
      auto strAttr = rewriter.getStringAttr(strWithNull);
      auto arrayType = LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                                strWithNull.size());

      globalOp = LLVM::GlobalOp::create(
          rewriter, loc, arrayType,
          /*isConstant=*/true, LLVM::Linkage::Internal, globalName, strAttr);
    }

    Type ptrType = LLVM::LLVMPointerType::get(ctx);
    return LLVM::AddressOfOp::create(rewriter, loc, ptrType, globalName)
        .getResult();
  }

  /// Get or create LLVM declaration for printf (variadic).
  LLVM::LLVMFuncOp getOrAddPrintfDecl(ConversionPatternRewriter &rewriter,
                                      ModuleOp moduleOp) const {
    StringRef funcName = "printf";
    if (auto existing = moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(funcName))
      return existing;

    auto ctx = rewriter.getContext();
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto i32Type = IntegerType::get(ctx, 32);
    auto funcType =
        LLVM::LLVMFunctionType::get(i32Type, {ptrType}, /*isVarArg=*/true);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fn = LLVM::LLVMFuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                       funcType);
    fn.setLinkage(LLVM::Linkage::External);
    return fn;
  }

  /// Get or create func declaration for spine_print_unranked_memref.
  /// Uses func::FuncOp so that unranked memref types are automatically
  /// lowered to LLVM struct {i64, ptr} by the memref-to-llvm pass.
  func::FuncOp getOrAddPrintMemrefDecl(ConversionPatternRewriter &rewriter,
                                       ModuleOp moduleOp, Type elemType) const {
    StringRef funcName = "spine_print_unranked_memref";
    if (auto existing = moduleOp.lookupSymbol<func::FuncOp>(funcName))
      return existing;

    auto ctx = rewriter.getContext();
    auto i32Type = IntegerType::get(ctx, 32);
    auto ptrType = LLVM::LLVMPointerType::get(ctx);
    auto unrankedType = UnrankedMemRefType::get(elemType, /*memorySpace=*/0);

    SmallVector<Type> argsType = {
        i32Type,      i32Type, i32Type, // pid_x, pid_y, pid_z
        ptrType,                        // prefix string
        unrankedType,                   // unranked memref
        i32Type,                        // bitWidth
        i32Type,                        // isInteger
        i32Type,                        // isSigned
        i32Type                         // asHex
    };
    auto funcType = FunctionType::get(ctx, argsType, {i32Type});

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(moduleOp.getBody());
    auto fn = func::FuncOp::create(rewriter, UnknownLoc::get(ctx), funcName,
                                   funcType);
    fn.setPrivate();
    return fn;
  }

  /// Get printf format specifier for a scalar type.
  std::string getFormatSubstr(Type type, bool hex, bool isSigned) const {
    if (hex) {
      unsigned bw = type.getIntOrFloatBitWidth();
      std::string ret = "0x%0" + std::to_string(bw / 4);
      if (bw > 32)
        ret += "ll";
      ret += "x";
      return ret;
    }
    if (type.isBF16() || type.isF16() || type.isF32() || type.isF64())
      return "%f";
    if (type.isInteger()) {
      if (type.getIntOrFloatBitWidth() == 64)
        return isSigned ? "%lli" : "%llu";
      return isSigned ? "%i" : "%u";
    }
    return "%d";
  }

  /// Promote value for printf ABI: int<32 → i32, float<64 → f64.
  Value printfPromoteValue(ConversionPatternRewriter &rewriter, Location loc,
                           Value value, bool isSigned) const {
    auto type = value.getType();
    auto ctx = rewriter.getContext();
    if (type.isIntOrIndex() && type.getIntOrFloatBitWidth() < 32) {
      auto i32Type = IntegerType::get(ctx, 32);
      return isSigned ? arith::ExtSIOp::create(rewriter, loc, i32Type, value)
                            .getResult()
                      : arith::ExtUIOp::create(rewriter, loc, i32Type, value)
                            .getResult();
    }
    if (type.isBF16() || type.isF16() || type.isF32()) {
      return arith::ExtFOp::create(rewriter, loc, Float64Type::get(ctx), value)
          .getResult();
    }
    return value;
  }

  /// Emit printf call for scalar or string-only print.
  void createPrintfCall(ConversionPatternRewriter &rewriter, Location loc,
                        ModuleOp moduleOp, StringRef prefix, Value pid0,
                        Value pid1, Value pid2, std::optional<Value> arg,
                        bool isSigned, bool hex) const {
    // Format: "(%i, %i, %i) prefix: <value>\n"
    std::string fmt = "(%i, %i, %i) " + prefix.str();
    if (arg.has_value()) {
      fmt += ": ";
      fmt += getFormatSubstr(arg->getType(), hex, isSigned);
    }
    fmt += "\n";

    Value fmtStr =
        createStringConstant(rewriter, loc, moduleOp, fmt, "printf_fmt_");

    SmallVector<Value> args = {fmtStr, pid0, pid1, pid2};
    if (arg.has_value())
      args.push_back(printfPromoteValue(rewriter, loc, *arg, isSigned));

    LLVM::CallOp::create(rewriter, loc, getOrAddPrintfDecl(rewriter, moduleOp),
                         args);
  }

  /// Emit tensor print: alloc memref → copy → cast unranked → call runtime →
  /// dealloc.
  void createTensorPrintCall(ConversionPatternRewriter &rewriter, Location loc,
                             ModuleOp moduleOp, StringRef prefix,
                             Value tensorOperand, RankedTensorType tensorType,
                             Value pid0, Value pid1, Value pid2, bool isSigned,
                             bool hex) const {
    auto ctx = rewriter.getContext();
    auto elemType = tensorType.getElementType();

    // Pointer element → i64
    if (isa<triton::PointerType>(elemType))
      elemType = IntegerType::get(ctx, 64);

    // 1. Allocate memref
    auto memrefType = MemRefType::get(tensorType.getShape(), elemType);
    Value alloc = memref::AllocOp::create(rewriter, loc, memrefType);

    // 2. Copy tensor → memref: materialize tensor to buffer first, then copy
    Value srcMemref = bufferization::ToBufferOp::create(
        rewriter, loc, memrefType, tensorOperand);
    memref::CopyOp::create(rewriter, loc, srcMemref, alloc);

    // 3. Cast to UnrankedMemRef
    auto unrankedType =
        UnrankedMemRefType::get(elemType, memrefType.getMemorySpace());
    Value unranked = memref::CastOp::create(rewriter, loc, unrankedType, alloc);

    // 4. Prepare call arguments
    Value prefixVal =
        createStringConstant(rewriter, loc, moduleOp, prefix, "print_pfx_");
    auto i32Type = IntegerType::get(ctx, 32);
    Value bw = arith::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(elemType.getIntOrFloatBitWidth()));
    Value isInt = arith::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(elemType.isInteger() ? 1 : 0));
    Value isSig = arith::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(isSigned ? 1 : 0));
    Value asHex = arith::ConstantOp::create(
        rewriter, loc, i32Type, rewriter.getI32IntegerAttr(hex ? 1 : 0));

    // 5. Call spine_print_unranked_memref via func::CallOp
    // The unranked memref will be automatically lowered to {i64, ptr} struct
    // by the memref-to-llvm pass
    SmallVector<Value> callArgs = {pid0, pid1,  pid2,  prefixVal, unranked,
                                   bw,   isInt, isSig, asHex};
    auto funcDecl = getOrAddPrintMemrefDecl(rewriter, moduleOp, elemType);
    func::CallOp::create(rewriter, loc, funcDecl, callArgs);

    // Note: intentionally skip memref.dealloc — spine-mlir's e2e pipeline
    // does not support free side-effects. The small leak is acceptable for
    // debug printing.
  }
};

} // namespace triton
} // namespace mlir

#endif
