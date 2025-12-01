#ifndef TRITON_CONVERSION_PATTERNS
#define TRITON_CONVERSION_PATTERNS

//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation, Meta Platforms.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "triton-shared/Analysis/MaskAnalysis.h"
#include "triton-shared/Analysis/OpFoldResultUtils.h"
#include "triton-shared/Analysis/PtrAnalysis.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "utils/FusionHelper.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>
#include <numeric>
#include <optional>
#include <type_traits>

using namespace mlir;
using namespace triton;

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
                  return rewriter.create<arith::SIToFPOp>(loc, resType, src);
                })
                .Case<arith::TruncFOp>([&](Operation *op) {
                  auto resType = op->getResults()[0].getType();
                  if (auto shapedType = dyn_cast<ShapedType>(resType)) {
                    resType = shapedType.getElementType();
                  }
                  return rewriter.create<arith::TruncFOp>(loc, resType, src);
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
          InFlightDiagnostic diag = emitError(loc)
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
      InFlightDiagnostic diag = emitError(loc)
                                << "other value used in masked load produced "
                                   "by unsupported instruction";
      return nullptr;
    }
  }
  return nullptr;
}

static SmallVector<utils::IteratorType> getNParallelLoopsAttrs(unsigned n) {
  return SmallVector<utils::IteratorType>(n, utils::IteratorType::parallel);
}

static Value getTransposedValue(Value source, const Location loc,
                                ConversionPatternRewriter &rewriter) {

  auto sourceType = cast<RankedTensorType>(source.getType());
  auto sourceRank = sourceType.getRank();

  SmallVector<int64_t> perm(sourceRank);
  std::iota(std::begin(perm), std::end(perm), 0);
  std::swap(perm[sourceRank - 1], perm[sourceRank - 2]);

  SmallVector<int64_t> transposedShape(sourceType.getShape());
  std::swap(transposedShape[sourceRank - 1], transposedShape[sourceRank - 2]);

  Value transposeInit = rewriter.create<tensor::EmptyOp>(
      loc, transposedShape, sourceType.getElementType());

  Value transpose =
      rewriter.create<linalg::TransposeOp>(loc, source, transposeInit, perm)
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

//===----------------------------------------------------------------------===//
// Op Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {

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
        auto castOp = rewriter.create<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), opnd);
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
      auto mulOp = rewriter.create<arith::MulIOp>(loc, cast<Value>(offset),
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
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{zero, block1Col},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
  }

  void createStackedCopies(Value block1, Value block2, Value dst, Location loc,
                           ConversionPatternRewriter &rewriter) const {

    auto zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto one =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));

    Value block1Row = rewriter.create<memref::DimOp>(loc, block1, 0);
    Value block1Col = rewriter.create<memref::DimOp>(loc, block1, 1);

    Value block2Row = rewriter.create<memref::DimOp>(loc, block2, 0);
    Value block2Col = rewriter.create<memref::DimOp>(loc, block2, 1);

    auto block1Dst =
        rewriter.create<memref::SubViewOp>(loc, dst, /* offsets */
                                           ValueRange{zero, zero},
                                           /* sizes */
                                           ValueRange{block1Row, block1Col},
                                           /* strides */
                                           ValueRange{one, one});

    auto block2Dst =
        rewriter.create<memref::SubViewOp>(loc, dst,
                                           /* offsets */
                                           ValueRange{block1Row, zero},
                                           /* sizes */
                                           ValueRange{block2Row, block2Col},
                                           /* strides */
                                           ValueRange{one, one});

    rewriter.create<memref::CopyOp>(loc, block1, block1Dst);
    rewriter.create<memref::CopyOp>(loc, block2, block2Dst);
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
      auto loadOp = rewriter.create<affine::AffineLoadOp>(
          op.getLoc(), sMemRef, zeroMap, std::nullopt);
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
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

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
        rewriter.create<memref::CopyOp>(loc, ptr, alloc);
      }

      Value tensor = rewriter.create<bufferization::ToTensorOp>(
          loc, tensorType, alloc, true /* restrict */, true /* writable */);
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

      // For each dimension check if mstate.dims[i] < shape[i], or-accumulate
      // the result
      auto shape = type.getShape();
      auto accBase =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
              .getResult();
      for (size_t i = 0; i < type.getShape().size(); i++) {
        auto shapei = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIndexAttr(shape[i]));

        Value dimi = dyn_cast<Value>(mstate.dims[i]);
        if (!dimi) {
          dimi = rewriter.create<arith::ConstantOp>(
              loc, cast<IntegerAttr>(cast<Attribute>(mstate.dims[i])));
        }

        auto cmpOp = rewriter.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt, dimi, shapei);
        accBase = rewriter.create<arith::OrIOp>(loc, accBase, cmpOp.getResult())
                      .getResult();
      }

      // condition the memset on the or-accumulation
      // initialize with padding prior to CopyOp
      rewriter.create<scf::IfOp>(
          loc, accBase, [&](OpBuilder &builder, Location loc) {
            builder.create<linalg::FillOp>(loc, ValueRange{scalarOther},
                                           ValueRange{alloc});
            builder.create<scf::YieldOp>(loc);
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
      rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    }

    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
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
      rewriter.create<affine::AffineStoreOp>(loc, val, sMemRef, zeroMap,
                                             std::nullopt);
      rewriter.eraseOp(op);
      return success();
    }

    // 1. Simple case where no mask is used.
    if (!mask) {
      auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
          loc, val, ptr);
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

    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, srcSlice, dstSubview);
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
    auto opType = cast<TensorType>(op.getType());
    auto loc = op.getLoc();

    auto init = rewriter.create<tensor::EmptyOp>(loc, opType.getShape(),
                                                 opType.getElementType());

    auto filledTensor =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{adaptor.getSrc()},
                                    ValueRange{init})
            .result();

    rewriter.replaceOp(op, filledTensor);
    return success();
  }
};

struct BroadcastConverter : public OpConversionPattern<triton::BroadcastOp> {
private:
  using OpConversionPattern<triton::BroadcastOp>::OpConversionPattern;

  SmallVector<int64_t> getBroadcastDims(RankedTensorType src,
                                        RankedTensorType dst) const {
    SmallVector<int64_t> broadcastDims;
    auto srcShape = src.getShape();
    auto dstShape = dst.getShape();

    for (size_t i = 0; i < srcShape.size(); i++) {
      if (dstShape[i] != srcShape[i]) {
        assert(srcShape[i] == 1);
        broadcastDims.push_back(i);
      }
    }
    assert(!broadcastDims.empty() && "cannot identify broadcast dimension");
    return broadcastDims;
  }

  // Broadcasts input tensor based on TosaToLinalg's broadcastToShape
  AffineMap getBroadcastAffineMap(MLIRContext *context,
                                  ArrayRef<int64_t> inputShape,
                                  ArrayRef<int64_t> broadcastToShape) const {

    assert(broadcastToShape.size() >= inputShape.size());

    // Create affine map and shapes for tensor initialization.
    SmallVector<AffineExpr> outExpr;

    size_t diff = broadcastToShape.size() - inputShape.size();
    for (size_t i = 0; i < broadcastToShape.size(); i++) {
      if (i < diff) {
        continue;
      }
      size_t j = i - diff;
      if (inputShape[j] == 1) {
        // Broadcast singleton dimension
        outExpr.push_back(mlir::getAffineConstantExpr(0, context));
        continue;
      }
      // Non-broadcast case
      outExpr.push_back(mlir::getAffineDimExpr(i, context));
    }
    return AffineMap::get(broadcastToShape.size(), 0, outExpr, context);
  }

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
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                 elementType);

    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), ValueRange{adaptor.getSrc()},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(resultRank),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value opResult = blockArgs[0];
          nestedBuilder.create<linalg::YieldOp>(loc, opResult);
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

    auto expandShapeOp = rewriter.create<tensor::ExpandShapeOp>(
        op.getLoc(), resType, src, reassoc);

    rewriter.replaceOp(op, expandShapeOp.getResult());
    return success();
  }
};

struct TransposeConverter : public OpConversionPattern<triton::TransOp> {
  using OpConversionPattern<triton::TransOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto source = adaptor.getSrc();

    auto sourceType = cast<RankedTensorType>(source.getType());
    auto sourceShape = sourceType.getShape();
    auto sourceRank = sourceType.getRank();

    auto order = op.getOrder();
    SmallVector<int64_t> perm(order.begin(), order.end());

    SmallVector<int64_t> transposedShape(sourceType.getShape());
    for (int i = 0; i < sourceRank; i++)
      transposedShape[i] = sourceShape[perm[i]];

    Value transposeInit = rewriter.create<tensor::EmptyOp>(
        op->getLoc(), transposedShape, sourceType.getElementType());

    Value transpose = rewriter
                          .create<linalg::TransposeOp>(op->getLoc(), source,
                                                       transposeInit, perm)
                          .getResults()[0];

    rewriter.replaceOp(op, transpose);
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

    auto init = rewriter.create<tensor::EmptyOp>(loc, shape, elementType);
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, op->getResultTypes(), /* operands */ ValueRange{},
        ValueRange{init}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
          Value res = nestedBuilder.create<arith::IndexCastOp>(
              loc, type.getElementType(), index);
          nestedBuilder.create<linalg::YieldOp>(loc, res);
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
    Value condVal = op.getCondition();

    if (isa<mlir::TensorType>(condVal.getType())) {
      auto scalarVal = getScalarValue(op.getCondition(), op.getLoc(), rewriter);
      condVal = scalarVal ? scalarVal : condVal;
    }
    assert(condVal && isa<mlir::IntegerType>(condVal.getType()) &&
           "Only asserts on scalars are currently supported");

    if (!condVal.getType().isInteger(1)) {
      auto zero =
          rewriter.create<mlir::arith::ConstantIntOp>(op.getLoc(), 0, 32);
      auto newCond = rewriter.create<mlir::arith::CmpIOp>(
          op.getLoc(), arith::CmpIPredicate::ne, condVal, zero);
      condVal = newCond.getResult();
    }

    auto assertMessage =
        llvm::formatv("Assertion `{0}` failed", op.getMessage());
    rewriter.create<mlir::cf::AssertOp>(op.getLoc(), condVal,
                                        assertMessage.str());

    rewriter.eraseOp(op);
    return success();
  }
};

// FIXME: There is no triton::BarrierOp currently.
struct BarrierConverter : public OpConversionPattern<mlir::gpu::BarrierOp> {
  using OpConversionPattern<mlir::gpu::BarrierOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    rewriter.create<mk::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

// Similar with triton-cpu.
struct PrintOpConverter : public OpConversionPattern<triton::PrintOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // If the op has no operands, we can just print the prefix.
    if (op.getNumOperands() == 0) {
      rewriter.create<mk::PrintOp>(loc, TypeRange{}, op.getPrefix(),
                                   op.getHex(), ValueRange{},
                                   llvm::SmallVector<int, 0>{});
      rewriter.eraseOp(op);
      return success();
    }

    for (size_t i = 0; i < op.getNumOperands(); i++) {
      Value operand = op.getOperands()[i];
      auto isSigned = {op.getIsSigned()[i]};
      // If the operand is not a ranked tensor, we should create a new tensor.
      // See mlir/lib/Interfaces/DestinationStyleOpInterface.cpp#L39
      if (!isa<RankedTensorType>(operand.getType())) {
        auto operandTensor = rewriter.create<tensor::FromElementsOp>(
            loc, RankedTensorType::get({}, operand.getType()), operand);
        rewriter.create<mk::PrintOp>(loc, operandTensor.getType(),
                                     op.getPrefix(), op.getHex(),
                                     operandTensor.getResult(), isSigned);
        continue;
      }

      auto operandType = cast<RankedTensorType>(operand.getType());
      auto flattenTensor = operand;
      if (operandType.getRank() != 1) {
        SmallVector<int64_t> flatten_shape = {operandType.getNumElements()};
        auto targetType =
            RankedTensorType::get(flatten_shape, operandType.getElementType());
        auto shapeAttr = rewriter.getI64TensorAttr(flatten_shape);
        auto shapeConst = rewriter.create<arith::ConstantOp>(loc, shapeAttr);
        flattenTensor = rewriter.create<tensor::ReshapeOp>(
            loc, targetType, flattenTensor, shapeConst);
      }

      rewriter.create<mk::PrintOp>(loc, operandType, op.getPrefix(),
                                   op.getHex(), flattenTensor, isSigned);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

struct BitcastConverter : public OpConversionPattern<triton::BitcastOp> {
  using OpConversionPattern<triton::BitcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto inputType = op.getOperand().getType();
    auto resultType = op.getType();
    if (isa<PointerType>(inputType) || isa<PointerType>(resultType)) {
      return success();
    }
    auto arithBitcast = rewriter.create<arith::BitcastOp>(
        op.getLoc(), op.getType(), op.getOperand());

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
            for (int i = 0; i < missing; i++) {
              args.push_back(parentInputs[args.size()]);
            }
          }
        }
      }
    }

    auto call = rewriter.create<func::CallOp>(op.getLoc(), op.getCallee(),
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
      Value truncatedValue = rewriter.create<arith::TruncFOp>(
          op.getLoc(), resultType, op.getOperand());
      rewriter.replaceOp(op, truncatedValue);
      return success();
    }

    Value extendedValue = rewriter.create<arith::ExtFOp>(
        op.getLoc(), resultType, op.getOperand());
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

    Value clamp = x;
    auto maxMin = min;

    if (propagateNan) {
      // Handle NaN propagation
      maxMin = rewriter.create<arith::MaximumFOp>(loc, x, min);
      clamp = rewriter.create<arith::MinimumFOp>(loc, maxMin, max);
    } else {
      // No NaN propagation.
      maxMin = rewriter.create<arith::MaxNumFOp>(loc, x, min);
      clamp = rewriter.create<arith::MinNumFOp>(loc, maxMin, max);
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
        rewriter.create<math::SqrtOp>(op.getLoc(), adaptor.getOperands());

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
        rewriter.create<arith::DivFOp>(op.getLoc(), adaptor.getOperands());

    rewriter.replaceOp(op, replacement);
    return success();
  }
};

struct CatConverter : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern<triton::CatOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto replacement = rewriter.create<tensor::ConcatOp>(
        op.getLoc(), 0 /* concat dimension */, adaptor.getOperands());

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
      Value slice = rewriter.create<tensor::ExtractSliceOp>(
          loc, resultTensor, input, offsets, sizes, strides);
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
    Value result = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

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
      result = rewriter.create<tensor::InsertSliceOp>(loc, inputs[i], result,
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
        rewriter.create<arith::MulUIExtendedOp>(loc, adaptor.getOperands());
    rewriter.replaceOp(op, mulResult.getHigh());

    return success();
  }
};

// Check if tensor is all zeros
// Returns true if all tensor elements are zero
// Returns false if tensor is not all zeros or cannot be determined
bool isZeroTensor(Value &v, bool integers) {
  // Check SplatOp case
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>()) {
    if (auto constOp = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
      // Check floating point constant
      if (auto val = dyn_cast<FloatAttr>(constOp.getValue())) {
        return val.getValueAsDouble() == 0.;
      }
      // Check integer constant
      if (auto val = dyn_cast<IntegerAttr>(constOp.getValue())) {
        return val.getValue() == 0;
      }
    }
    return false;
  }

  // Check ConstantOp case
  if (auto constOp = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat()) {
        // Check zero value based on type
        if (integers)
          return denseAttr.getSplatValue<APInt>().isZero();
        return denseAttr.getSplatValue<APFloat>().isZero();
      }
    }
  }

  return false;
}

struct MatmulConverter : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern<triton::DotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto opa = op.getA();
    auto opb = op.getB();
    auto opc = op.getC();

    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();
    bool integers = elementType.isInteger();
    bool skipC = isZeroTensor(opc, integers);
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);
    TypedAttr constantAttr =
        integers
            ? static_cast<TypedAttr>(rewriter.getIntegerAttr(elementType, 0))
            : static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));

    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), elementType, constantAttr);

    auto zeroes =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    auto res = rewriter
                   .create<linalg::MatmulOp>(loc, ValueRange{opa, opb},
                                             ValueRange{zeroes})
                   .getResult(0);

    if (!skipC) {
      if (integers) {
        res = rewriter.create<arith::AddIOp>(loc, opc, res);
      } else {
        res = rewriter.create<arith::AddFOp>(loc, opc, res);
      }
    }

    rewriter.replaceOp(op, res);
    return success();
  }
};

struct DotScaledConverter : public OpConversionPattern<triton::DotScaledOp> {
  using OpConversionPattern<triton::DotScaledOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get operands
    auto a = op.getA();
    auto b = op.getB();
    auto c = op.getC();
    Value aScale = op.getAScale();
    Value bScale = op.getBScale();
    auto aElemType = op.getAElemTypeAttr();
    auto bElemType = op.getBElemTypeAttr();
    auto fastMath = op.getFastMathAttr();

    // Get type information
    auto aType = a.getType();
    auto bType = b.getType();
    auto dstType = cast<RankedTensorType>(op.getType());
    auto elementType = dstType.getElementType();

    // Create initial zero tensor
    auto init =
        rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), elementType);
    TypedAttr constantAttr =
        static_cast<TypedAttr>(rewriter.getFloatAttr(elementType, 0));
    auto zero = rewriter.create<mlir::arith::ConstantOp>(
        op.getLoc(), elementType, constantAttr);
    auto zeroes =
        rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{init})
            .result();

    // Perform scaled dot product operation
    Value res = rewriter
                    .create<mk::DotScaledOp>(loc, TypeRange{op.getType()}, a,
                                             aScale, b, bScale, zeroes,
                                             aElemType, bElemType, fastMath)
                    .getResult(0);

    // Check if C needs to be added
    bool skipC = isZeroTensor(c, false);
    if (!skipC) {
      res = rewriter.create<arith::AddFOp>(loc, c, res);
    }

    rewriter.replaceOp(op, res);
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
    return isa<arith::AddFOp, arith::AddIOp, arith::MulIOp, arith::MulFOp,
               arith::MaximumFOp, arith::MaxNumFOp, arith::MinimumFOp,
               arith::MinNumFOp, arith::MinSIOp, arith::MinUIOp, arith::MaxSIOp,
               arith::MaxUIOp, arith::XOrIOp, arith::AndIOp, arith::OrIOp>(
        redOp);
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
            .Case([&](arith::AddIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::MulFOp) {
              return rewriter.getFloatAttr(constantType, 1.f);
            })
            .Case([&](arith::MulIOp) {
              return rewriter.getIntegerAttr(constantType, 1);
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
            .Case([&](arith::MaxUIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::OrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::XOrIOp) {
              return rewriter.getIntegerAttr(constantType, 0);
            })
            .Case([&](arith::AndIOp) {
              return rewriter.getIntegerAttr(constantType,
                                             llvm::maxUIntN(bitWidth));
            })
            .Default([](Operation *op) {
              op->dump();
              llvm_unreachable("Reduction op not yet supported");
              return nullptr;
            });

    return rewriter.create<arith::ConstantOp>(redOp->getLoc(), constantType,
                                              attr);
  }

  bool requiresF32Conversion(const Type elemType, Operation *redOp) const {
    return isa<FloatType>(elemType) &&
           elemType.getIntOrFloatBitWidth() <
               cast<FloatType>(Float32Type::get(elemType.getContext()))
                   .getWidth() &&
           (isa<arith::AddFOp>(redOp) || isa<arith::MulFOp>(redOp));
  }

  Value getRedElement(Value lhs, Value rhs, const Location loc,
                      Operation *redOp, OpBuilder &b,
                      const bool convertLhsToF32Precision) const {
    return llvm::TypeSwitch<Operation *, Value>(redOp)
        .Case<arith::AddFOp, arith::MulFOp>([&](auto redOp) {
          if (convertLhsToF32Precision) {
            lhs = b.create<arith::ExtFOp>(loc, Float32Type::get(b.getContext()),
                                          lhs);
          }
          return b.create<decltype(redOp)>(loc, lhs, rhs);
        })
        .Case<arith::AddIOp, arith::MulIOp, arith::MaximumFOp, arith::MaxNumFOp,
              arith::MinimumFOp, arith::MinNumFOp, arith::MinSIOp,
              arith::MinUIOp, arith::MaxSIOp, arith::MaxUIOp, arith::XOrIOp,
              arith::AndIOp, arith::OrIOp>([&](auto redOp) {
          return b.create<decltype(redOp)>(loc, lhs, rhs);
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
              "containing 1 max(i/f) or addf.");
    }

    auto rop = reductionOps.front();
    auto axis = op.getAxis();
    auto isVectorReduce = sourceType.getRank() == 1;

    if (axis == sourceType.getRank() - 1 && !isVectorReduce) {
      source = getTransposedValue(source, op.getLoc(), rewriter);
      axis = sourceType.getRank() - 2;
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
      initTensor = rewriter.create<bufferization::AllocTensorOp>(
          loc, RankedTensorType::get({}, constantType), ValueRange{});
      initTensor = rewriter.create<tensor::InsertOp>(loc, accBaseConstOp,
                                                     initTensor, ValueRange{});
    } else {
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, cast<RankedTensorType>(resType).getShape(), constantType);
      initTensor = rewriter
                       .create<linalg::FillOp>(loc, ValueRange{accBaseConstOp},
                                               ValueRange{init})
                       .result();
    }

    Value finalResult =
        rewriter
            .create<linalg::ReduceOp>(
                loc, ValueRange{source}, ValueRange{initTensor},
                SmallVector<int64_t>{axis},
                [&](OpBuilder &opBuilder, Location loc, ValueRange inputs) {
                  assert(inputs.size() == 2);
                  Value result =
                      getRedElement(inputs[0], inputs[1], loc, rop, opBuilder,
                                    convertToF32Precision);
                  opBuilder.create<linalg::YieldOp>(loc, result);
                })
            .getResult(0);

    if (sourceType.getRank() == 1) {
      finalResult =
          rewriter.create<tensor::ExtractOp>(loc, constantType, finalResult);
    }

    if (convertToF32Precision) {
      finalResult = rewriter.create<arith::TruncFOp>(loc, resType, finalResult);
    }

    rewriter.replaceOp(op, finalResult);
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

    return convertToLinalgReduce(op, adaptor, rewriter);
  }
};

template <typename T>
class ArgMinMaxBaseConverter : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern<triton::ReduceOp>::OpConversionPattern;

  Value getInitTensor(ConversionPatternRewriter &rewriter,
                      ArrayRef<int64_t> shape, Value fillValue,
                      Location loc) const {
    Value initTensor =
        rewriter.create<tensor::EmptyOp>(loc, shape, fillValue.getType());
    return rewriter
        .create<linalg::FillOp>(loc, ValueRange{fillValue},
                                ValueRange{initTensor})
        .result();
  }

public:
  ArgMinMaxBaseConverter(MLIRContext *context) : OpConversionPattern(context) {}
  bool isArgMin;

  LogicalResult match(ReduceOp op) const override final {
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
    Value indexSelectOp, valueSelectOp;
    if (failed(matchArgMinMax(currValue, currIndex, reduceValue, reduceIndex,
                              opsIt, indexSelectOp, valueSelectOp, isArgMin))) {
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

    return success();
  }

  void rewrite(ReduceOp op, OpAdaptor adaptor,
               ConversionPatternRewriter &rewriter) const override final {
    auto loc = op.getLoc();

    auto elemTypes = op.getElementTypes();

    // Set the initial value of the rank-0 tensor containing
    // the result value to either -inf or +inf depending on
    // whether we're dealing with argmax or argmin
    auto valueType = elemTypes[0];

    auto valuesAccBaseVal =
        valueType.isInteger()
            ? rewriter.create<arith::ConstantOp>(
                  loc, valueType,
                  rewriter.getIntegerAttr(
                      valueType, T::getBaseReductionIntValue(
                                     elemTypes[0].getIntOrFloatBitWidth())))
            : rewriter.create<arith::ConstantOp>(
                  loc, valueType,
                  rewriter.getFloatAttr(valueType,
                                        T::getBaseReductionFloatValue()));

    // Set the initial value of the rank-0 tensor containing the index of the
    // min or max value to -1
    auto indexType = elemTypes[1];
    auto indicesAccBaseVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIntegerAttr(indexType, -1));

    // Get the shape of the resulting tensors (both for values and indices). If
    // we are reducing to a single scalar, then the result's type is a tensor of
    // rank-0, otherwise we can reuse the original result shape
    auto valueResultType = dyn_cast<RankedTensorType>(op.getType(0));
    const auto isScalarReduce = valueResultType == nullptr;
    SmallVector<int64_t> reductionResultShape{
        isScalarReduce ? SmallVector<int64_t>{}
                       : SmallVector<int64_t>(valueResultType.getShape())};

    SmallVector<Value> outputs{
        getInitTensor(rewriter, reductionResultShape, valuesAccBaseVal, loc),
        getInitTensor(rewriter, reductionResultShape, indicesAccBaseVal, loc)};

    auto linalgOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getOperands(), outputs,
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
          b.create<linalg::YieldOp>(loc, results);
        });

    if (isScalarReduce) {
      SmallVector<Value> reduceResults{
          rewriter.create<tensor::ExtractOp>(
              loc, valueType, linalgOp.getResults()[0], ValueRange{}),
          rewriter.create<tensor::ExtractOp>(
              loc, indexType, linalgOp.getResults()[1], ValueRange{})};
      rewriter.replaceOp(op, reduceResults);
    } else {
      rewriter.replaceOp(op, linalgOp);
    }
  }
};

struct ArgMaxConverter : public ArgMinMaxBaseConverter<ArgMaxConverter> {
  // TODO: min value according the bitwidth? Now this is not used in tx8
  static int64_t getBaseReductionIntValue(int bitwidth) {
    return llvm::minIntN(bitwidth);
  }
  static float getBaseReductionFloatValue() {
    return -std::numeric_limits<float>::infinity();
  }

  ArgMaxConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {
    isArgMin = false;
  }
};

struct ArgMinConverter : public ArgMinMaxBaseConverter<ArgMinConverter> {
  static int64_t getBaseReductionIntValue(int bitwidth) {
    return llvm::maxIntN(bitwidth);
  }
  static float getBaseReductionFloatValue() {
    return std::numeric_limits<float>::infinity();
  }

  ArgMinConverter(MLIRContext *context) : ArgMinMaxBaseConverter(context) {
    isArgMin = true;
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

    auto init = rewriter.create<tensor::EmptyOp>(
        loc, cast<RankedTensorType>(op.getResult().getType()).getShape(),
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

    if (type.getRank() != 1 && type.getRank() != 2 &&
        axis != type.getRank() - 1) {
      return rewriter.notifyMatchFailure(
          op, "Only support lowering scan op to cumsum with rank "
              "= {1, 2} and axis = rank - 1");
    }

    Value init = rewriter.create<tensor::EmptyOp>(op.getLoc(), type.getShape(),
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
          auto resultTypes = llvm::to_vector<6>(
              llvm::map_range(op->getResultTypes(), [](Type type) {
                return cast<TensorType>(type).getElementType();
              }));
          auto *scalarOp =
              builder.create(loc, op->getName().getIdentifier(),
                             regionArgs.take_front(op->getNumOperands()),
                             resultTypes, op->getAttrs());
          builder.create<linalg::YieldOp>(loc, scalarOp->getResults());
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

    auto shape = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64TensorAttr(outputShape));
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, outputType, input,
                                                   shape);

    return success();
  }
};

struct GatherConverter : public OpConversionPattern<triton::GatherOp> {
  using OpConversionPattern<triton::GatherOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    Value dstInit = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), resultType.getShape(), resultType.getElementType());

    auto gatherOp =
        rewriter.create<mk::GatherOp>(op.getLoc(), op.getType(), op.getSrc(),
                                      op.getIndices(), dstInit, op.getAxis());

    rewriter.replaceOp(op, gatherOp.getResult());
    return success();
  }
};

class ExternElementwiseFiniteOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (!op.getPure() || op.getSrcs().size() != 1 ||
        op.getSymbol() != "__nv_finitef")
      return failure();

    // get input value
    Value x = op.getSrcs()[0];
    auto shapedType = cast<ShapedType>(x.getType());
    auto elemType = shapedType.getElementType();

    if (!elemType.isF32()) {
      return failure();
    }

    // step 1: check x == x (not NaN)
    Value notNaN =
        rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, x, x);

    // step 2: check x + 1 != x (not Infinity)
    Value one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elemType, 1.0f));

    // create constant tensor
    auto constTensor =
        rewriter.create<tensor::EmptyOp>(loc, shapedType.getShape(), elemType);
    auto filledTensor = rewriter
                            .create<linalg::FillOp>(loc, ValueRange{one},
                                                    ValueRange{constTensor})
                            .getResult(0);

    // x + 1
    Value xPlusOne = rewriter.create<arith::AddFOp>(loc, x, filledTensor);
    // x + 1 != x
    Value notInf = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ONE, xPlusOne, x);

    // step 3: combine two conditions (notNaN && notInf)
    Value isFinite = rewriter.create<arith::AndIOp>(loc, notNaN, notInf);

    rewriter.replaceOp(op, isFinite);
    return success();
  }
};

class ExternElementwiseFmodOpConverter
    : public OpConversionPattern<triton::ExternElementwiseOp> {
  using OpConversionPattern<triton::ExternElementwiseOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(triton::ExternElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (op.getSrcs().size() != 2 ||
        (op.getSymbol() != "__nv_fmodf" && op.getSymbol() != "__nv_fmod")) {
      return failure();
    }

    // get input values
    Value input1 = op.getSrcs()[0];
    Value input2 = op.getSrcs()[1];
    auto elemType = cast<ShapedType>(input1.getType()).getElementType();

    if (!elemType.isF32()) {
      return failure();
    }
    auto divResult = rewriter.create<arith::DivFOp>(loc, input1, input2);
    auto truncResult = rewriter.create<math::TruncOp>(loc, divResult);
    auto mulResult = rewriter.create<arith::MulFOp>(loc, truncResult, input2);
    auto subResult = rewriter.create<arith::SubFOp>(loc, input1, mulResult);

    rewriter.replaceOp(op, subResult);
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
    POPULATE_BINARY_OP("__nv_powif", math::FPowIOp);
    POPULATE_BINARY_OP("__nv_powi", math::FPowIOp);

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
    POPULATE_UNARY_OP("__nv_tanh", math::TanhOp);
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
    POPULATE_UNARY_OP("__nv_isnanf", math::IsNaNOp);
    POPULATE_UNARY_OP("__nv_isnand", math::IsNaNOp);
    POPULATE_UNARY_OP("__nv_isinff", math::IsInfOp);
    POPULATE_UNARY_OP("__nv_isinfd", math::IsInfOp);
    POPULATE_UNARY_OP("__nv_rintf", math::RoundOp);
    POPULATE_UNARY_OP("__nv_rint", math::RoundOp);

#undef POPULATE_UNARY_OP
    return failure();
  }
};

static void populateExternElementwiseOpToMLIROps(RewritePatternSet &patterns) {
  patterns
      .add<ExternElementwiseFiniteOpConverter, ExternElementwiseFmodOpConverter,
           ExternElementwiseBinaryOpConverter,
           ExternElementwiseUnaryOpConverter>(patterns.getContext());
}

struct HistogramOpConversion : public OpConversionPattern<triton::HistogramOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::HistogramOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = rewriter.getRemappedValue(op.getSrc());

    auto srcTy = dyn_cast<RankedTensorType>(src.getType());
    auto resTy = dyn_cast<RankedTensorType>(op.getType());

    auto flattenTensor = src;
    // NOTE: Triton only support rank1. But flatten the tensor to 1D can always
    // implement the histogram operation.
    if (srcTy.getRank() != 1) {
      SmallVector<int64_t> flatten_shape = {srcTy.getNumElements()};
      auto targetType =
          RankedTensorType::get(flatten_shape, srcTy.getElementType());

      auto shapeAttr = rewriter.getI64TensorAttr(flatten_shape);
      auto shapeConst = rewriter.create<arith::ConstantOp>(loc, shapeAttr);
      flattenTensor = rewriter.create<tensor::ReshapeOp>(
          loc, targetType, flattenTensor, shapeConst);
    }

    Value zero = rewriter.create<arith::ConstantOp>(
        loc, resTy, rewriter.getZeroAttr(resTy));
    Value one = rewriter.create<arith::ConstantOp>(loc, resTy,
                                                   rewriter.getOneAttr(resTy));
    RankedTensorType cmpVecTy =
        RankedTensorType::get(resTy.getShape(), srcTy.getElementType());

    Value allocatedRangeVec = rewriter.create<tensor::EmptyOp>(
        loc, resTy.getShape(), cmpVecTy.getElementType());
    SmallVector<AffineMap> indexingMaps{AffineMap::get(
        /* dimCount */ 1, /* symbolCount */ 0,
        SmallVector<AffineExpr>{
            mlir::getAffineDimExpr(0, rewriter.getContext())},
        rewriter.getContext())};
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, cmpVecTy, /* operands */ ValueRange{},
        ValueRange{allocatedRangeVec}, indexingMaps, getNParallelLoopsAttrs(1),
        [&](OpBuilder &nestedBuilder, Location nestedLoc,
            ValueRange blockArgs) {
          Value index = nestedBuilder.create<linalg::IndexOp>(loc, 0);
          Value res = nestedBuilder.create<arith::IndexCastOp>(
              loc, resTy.getElementType(), index);
          nestedBuilder.create<linalg::YieldOp>(loc, res);
        });

    Value res = zero;

    // Create loop bounds
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound =
        rewriter.create<arith::ConstantIndexOp>(loc, srcTy.getNumElements());
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, ValueRange{res},
        [&](OpBuilder &builder, Location loc, Value iv, ValueRange iterArgs) {
          Value currentRes = iterArgs[0];
          // Extract element at current index
          Value elem =
              builder.create<tensor::ExtractOp>(loc, flattenTensor, iv);
          SmallVector<Value> elems(resTy.getNumElements(), elem);
          // Create a splat of the element
          Value elemVec = builder.create<tensor::FromElementsOp>(loc, elems);

          // Compare with range vector
          Value mask = builder.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, elemVec, linalgOp.getResult(0));
          // Select based on mask
          Value delta =
              builder.create<arith::SelectOp>(loc, resTy, mask, one, zero);
          // Add to running result
          Value newRes = builder.create<arith::AddIOp>(loc, currentRes, delta);
          // Yield the updated result
          builder.create<scf::YieldOp>(loc, newRes);
        });

    // Replace the original op with the final histogram result
    rewriter.replaceOp(op, forOp.getResults()[0]);
    return success();
  }

  TypedAttr makeRangeAttr(RankedTensorType resTy,
                          ConversionPatternRewriter &rewriter) const {
    Type elemTy = resTy.getElementType();
    if (elemTy.isInteger(32)) {
      SmallVector<int32_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI32TensorAttr(range);
    } else if (elemTy.isInteger(64)) {
      SmallVector<int64_t> range(resTy.getShape()[0]);
      std::iota(range.begin(), range.end(), 0);
      return rewriter.getI64TensorAttr(range);
    } else {
      llvm_unreachable(
          "unsupported src elem type for histogram (expected i32 or i64)");
    }
  }
};

// It provides accumulation function that clones operations from the
// original combine region and applies them on provided tensor.
// Also, it handles multi-dimensional cases reducing them to two
// possible options: lowering for a 1-D tensor inputs and lowering
// the operation over the leading dimension.
//
// Specialized pattern should implement lower1DInput to handle
// trailing dimension case and lowerLeadingDimension to handle the leading
// dimension case through accumulation of sub-tensors.
struct ScanOpConverter : public OpConversionPattern<triton::ScanOp> {
private:
  mutable IRMapping invariantsMap;
  using OpConversionPattern<triton::ScanOp>::OpConversionPattern;
  using OpConversionPattern<triton::ScanOp>::getTypeConverter;
  using OpAdaptor = typename triton::ScanOp::Adaptor;

  using LoweringFuncType = SmallVector<Value> (ScanOpConverter::*)(
      ValueRange inputs, triton::ScanOp op,
      ConversionPatternRewriter &rewriter) const;

  // Though function ptr to call lower1DInput/lowerLeadingDimension to handle
  // the tensor.
  SmallVector<Value> lowering(ValueRange inputs, triton::ScanOp op,
                              ConversionPatternRewriter &rewriter,
                              uint32_t axis, LoweringFuncType handle) const {
    auto loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto shape = inputType.getShape();

    SmallVector<Value> res(inputs.size());
    std::transform(inputs.begin(), inputs.end(), res.begin(), [&](auto val) {
      return rewriter.create<tensor::EmptyOp>(loc, shape,
                                              inputType.getElementType());
    });

    SmallVector<scf::ForOp> loops;
    SmallVector<Value, 8> loopIndices;

    std::function<void(unsigned)> buildLoops = [&](unsigned d) {
      // Setup loop bounds and step.
      Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, shape[d]);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      SmallVector<Value> curRes =
          d == 0 ? res : loops[d - 1].getInitArgs().take_front(res.size());
      auto loop = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound, step,
                                              ValueRange{curRes});
      auto loopIdx = loop.getInductionVar();
      loopIndices.push_back(loopIdx);
      loops.push_back(loop);

      rewriter.setInsertionPointToStart(loop.getBody());
      if (d == axis - 1) {
        SmallVector<Value, 8> dynamicIndices = loopIndices;
        dynamicIndices.insert(dynamicIndices.end(), shape.size() - axis,
                              rewriter.create<arith::ConstantIndexOp>(loc, 0));

        SmallVector<int64_t> static_size(axis, 1);
        static_size.insert(static_size.end(), shape.begin() + axis,
                           shape.end());
        SmallVector<int64_t> static_stride(shape.size(), 1);

        // {1,1,..shape[axis],shape[axis+1],..shape[rank]}
        SmallVector<int64_t> extract_shape = static_size;

        // {shape[axis],shape[axis+1],..shape[rank]}
        SmallVector<int64_t> reshape_shape(shape.begin() + axis, shape.end());

        SmallVector<Value> subInputs(inputs.size());
        std::transform(
            inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
              auto extract_tensor = rewriter.create<tensor::ExtractSliceOp>(
                  loc,
                  RankedTensorType::get(extract_shape,
                                        inputType.getElementType()),
                  val, dynamicIndices, /*sizes*/ ValueRange(),
                  /*strides*/ ValueRange(),
                  SmallVector<int64_t>(shape.size(), ShapedType::kDynamic),
                  static_size, static_stride);

              auto shapeAttr = rewriter.getI64TensorAttr(reshape_shape);
              auto shapeConst =
                  rewriter.create<arith::ConstantOp>(loc, shapeAttr);

              // {1,1,..shape[axis],shape[axis+1],..shape[rank]} ->
              // {shape[axis],shape[axis+1],..shape[rank]}
              return rewriter.create<tensor::ReshapeOp>(
                  loc,
                  RankedTensorType::get(reshape_shape,
                                        inputType.getElementType()),
                  extract_tensor, shapeConst);
            });

        auto resElems = (this->*handle)(subInputs, op, rewriter);
        for (size_t i = 0; i < res.size(); ++i) {

          auto targetType =
              RankedTensorType::get(extract_shape, inputType.getElementType());

          auto shapeAttr = rewriter.getI64TensorAttr(extract_shape);
          auto shapeConst = rewriter.create<arith::ConstantOp>(loc, shapeAttr);

          // {shape[axis],shape[axis+1],..shape[rank]} ->
          // {1,1,..shape[axis],shape[axis+1],..shape[rank]}
          auto reshaped = rewriter.create<tensor::ReshapeOp>(
              loc, targetType, resElems[i], shapeConst);

          curRes[i] = rewriter.create<tensor::InsertSliceOp>(
              loc, reshaped, curRes[i], dynamicIndices,
              /*sizes*/ ValueRange(),
              /*strides*/ ValueRange(),
              SmallVector<int64_t>(shape.size(), ShapedType::kDynamic),
              static_size, static_stride);

          rewriter.create<scf::YieldOp>(loc, curRes);
        }
        return;
      }
      buildLoops(d + 1);
      // Terminate the loop body.
      rewriter.setInsertionPointToEnd(loop.getBody());
      SmallVector<Value> yieldValues =
          loops[d + 1].getResults().take_front(res.size());
      rewriter.create<scf::YieldOp>(loc, yieldValues);
      rewriter.setInsertionPointAfter(loop);
    };

    buildLoops(0);

    // Extract result tensors from forOp;
    SmallVector<Value> results;
    for (size_t i = 0; i < inputs.size(); ++i) {
      results.push_back(loops.front().getResult(i));
    }
    return results;
  }

  // To handle the trailing dimension case, we extract all input vectors
  // and process them through lower1DInput, then build the resulting
  // vector using inserts.
  LogicalResult
  lowerTrailingDimension(triton::ScanOp op,
                         ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());

    // 1-D input case.
    if (inputType.getRank() == 1) {
      auto res = lower1DInput(inputs, op, rewriter);
      rewriter.replaceOp(op, res);
      return success();
    }

    // TODO: Optimization: The last two dimensions' data can be read with column
    // stride and computed in parallel.
    uint32_t axis = op.getAxis();
    assert(axis == (inputType.getRank() - 1) &&
           "Expected reduction axis is the last one");
    SmallVector<Value> res =
        lowering(inputs, op, rewriter, axis, &ScanOpConverter::lower1DInput);

    rewriter.replaceOp(op, res);
    return success();
  }

  // In this case we either call lowerLeadingDimension to process the input
  // or extract sub-vectors, call lowerLeadingDimension, and then reconstruct
  // the result.
  LogicalResult
  lowerNonTrailingDimension(triton::ScanOp op,
                            ConversionPatternRewriter &rewriter) const {

    SmallVector<Value> inputs;
    if (failed(rewriter.getRemappedValues(op.getOperands(), inputs)))
      return failure();

    uint32_t axis = op.getAxis();
    if (axis == 0) {
      rewriter.replaceOp(op, lowerLeadingDimension(inputs, op, rewriter));
      return success();
    }

    SmallVector<Value> res = lowering(inputs, op, rewriter, axis,
                                      &ScanOpConverter::lowerLeadingDimension);

    rewriter.replaceOp(op, res);
    return success();
  }

  // Accumulate inputs and existing accumulators into a new accumulators
  // applying operations from the combine region.
  SmallVector<Value> accumulate(ValueRange inputs, ValueRange acc,
                                Region &combineOp, OpBuilder &rewriter) const {
    if (acc.empty())
      return inputs;

    auto type = inputs[0].getType();
    SmallVector<int64_t> shape;
    if (isa<RankedTensorType>(type)) {
      auto temp = cast<RankedTensorType>(type).getShape();
      shape.insert(shape.end(), temp.begin(), temp.end());
    } else {
      shape.push_back(1);
    }
    auto &block = combineOp.getBlocks().front();
    IRMapping map;
    // Map block arguments to the current inputs and accumulators.
    for (unsigned i = 0; i < acc.size(); ++i) {
      map.map(block.getArgument(i), acc[i]);
      map.map(block.getArgument(acc.size() + i), inputs[i]);
    }
    for (auto &op : block.getOperations()) {
      // Returned values are a new accumulator.
      if (isa<triton::ScanReturnOp>(op)) {
        SmallVector<Value> res;
        for (auto operand : op.getOperands()) {
          res.push_back(map.lookup(operand));
        }
        return res;
      }

      // Clone operation mapping its inputs and building vector
      // result types using the input shape.
      OperationState newState(op.getLoc(), op.getName());
      for (auto operand : op.getOperands()) {
        newState.operands.push_back(
            lookupMappedValue(map, operand, shape, rewriter));
      }
      for (auto ty : op.getResultTypes()) {
        isa<RankedTensorType>(type)
            ? newState.types.push_back(RankedTensorType::get(shape, ty))
            : newState.types.push_back(ty);
      }
      newState.attributes = op.getAttrs();
      auto newOp = rewriter.create(newState);

      // Add new values to the map.
      for (auto [oldVal, newVal] :
           llvm::zip(op.getResults(), newOp->getResults())) {
        map.map(oldVal, newVal);
      }
    }
    llvm_unreachable("No return op found in scan/reduce region");
  }

  Value lookupMappedValue(IRMapping &localMap, Value val,
                          ArrayRef<int64_t> shape, OpBuilder &rewriter) const {

    Value res = localMap.lookupOrNull(val);
    if (!res) {
      // If value is not found then it's an invariant defined in the outer
      // region. We check if it has been already translated and add a splat
      // operation if it hasn't.
      res = invariantsMap.lookupOrNull(val);
      if (!res) {
        auto ip = rewriter.saveInsertionPoint();
        rewriter.setInsertionPointAfterValue(val);
        res = rewriter.create<tensor::SplatOp>(
            val.getLoc(), RankedTensorType::get(shape, val.getType()), val);
        invariantsMap.map(val, res);
        rewriter.restoreInsertionPoint(ip);
      }
    }
    return res;
  }

  SmallVector<Value> lower1DInput(ValueRange inputs, ScanOp op,
                                  ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto shape = inputType.getShape();

    SmallVector<Value> res(inputs.size());
    std::transform(inputs.begin(), inputs.end(), res.begin(), [&](auto val) {
      return rewriter.create<tensor::EmptyOp>(loc, shape,
                                              inputType.getElementType());
    });

    SmallVector<Value> acc(inputs.size());
    std::transform(inputs.begin(), inputs.end(), acc.begin(), [&](auto val) {
      return rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(inputType.getElementType()));
    });

    // scf.for loop bounds and step
    // NOTE: Scf only support positive step, so we need to handle reverse
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    // Use tuple to pass acc and res as loop-carried variables
    SmallVector<Value> initVals;
    for (auto v : res)
      initVals.push_back(v);
    for (auto v : acc)
      initVals.push_back(v);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, initVals,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          // iterArgs: [res..., acc...]
          SmallVector<Value> curRes(iterArgs.begin(),
                                    iterArgs.begin() + inputs.size());
          SmallVector<Value> currAcc(iterArgs.begin() + inputs.size(),
                                     iterArgs.end());

          SmallVector<Value> idxIndex;
          if (reverse) {
            auto actualIdx = b.create<arith::SubIOp>(loc, upperBound, iv);
            actualIdx = b.create<arith::SubIOp>(loc, actualIdx, step);
            idxIndex.push_back(actualIdx);
          } else {
            idxIndex.push_back(iv);
          }
          SmallVector<Value> inputsElem(inputs.size());
          std::transform(
              inputs.begin(), inputs.end(), inputsElem.begin(), [&](auto val) {
                return rewriter.create<tensor::ExtractOp>(loc, val, idxIndex);
              });

          // Check if this is the first iteration
          Value isFirstValue = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, iv, lowerBound);
          scf::IfOp ifOp = b.create<scf::IfOp>(
              loc, isFirstValue,
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(loc, inputsElem);
              },
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(
                    loc, accumulate(inputsElem, currAcc, combineOp, b));
              });
          currAcc = ifOp.getResults();

          assert(acc.size() == inputs.size() &&
                 "accumulate should return the same number of results as "
                 "inputs");
          for (int i = 0; i < res.size(); ++i) {
            curRes[i] = rewriter.create<tensor::InsertOp>(loc, currAcc[i],
                                                          curRes[i], idxIndex);
          }

          SmallVector<Value> yieldVals;
          for (auto v : curRes)
            yieldVals.push_back(v);
          for (auto v : currAcc)
            yieldVals.push_back(v);

          b.create<scf::YieldOp>(loc, yieldVals);
        });
    // Extract result tensors from forOp
    SmallVector<Value> results;
    for (size_t i = 0; i < inputs.size(); ++i) {
      results.push_back(forOp.getResult(i));
    }
    return results;
  }

  SmallVector<Value>
  lowerLeadingDimension(ValueRange inputs, ScanOp op,
                        ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    Region &combineOp = op.getRegion();
    bool reverse = op.getReverse();

    auto inputType = cast<RankedTensorType>(inputs[0].getType());
    auto shape = inputType.getShape();

    SmallVector<Type> resTypes;
    for (const auto &resTy : op.getResultTypes()) {
      resTypes.push_back(RankedTensorType::get(
          shape, cast<RankedTensorType>(resTy).getElementType()));
    }

    // Initialize result tensors
    SmallVector<Value> res(inputs.size());
    std::transform(inputs.begin(), inputs.end(), res.begin(), [&](auto val) {
      return rewriter.create<tensor::EmptyOp>(loc, shape,
                                              inputType.getElementType());
    });

    // Initialize accumulators as empty tensors of shape [1, ...]
    SmallVector<int64_t> accShape({1});
    accShape.insert(accShape.end(), shape.begin() + 1, shape.end());

    SmallVector<Value> acc(inputs.size());
    std::transform(inputs.begin(), inputs.end(), acc.begin(), [&](auto val) {
      return rewriter.create<tensor::EmptyOp>(loc, accShape,
                                              inputType.getElementType());
    });

    // scf.for loop bounds and step
    // NOTE: Scf only support positive step, so we need to handle reverse
    Value lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value upperBound = rewriter.create<arith::ConstantIndexOp>(loc, shape[0]);
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Use tuple to pass acc and res as loop-carried variables
    SmallVector<Value> initVals;
    for (auto v : res)
      initVals.push_back(v);
    for (auto v : acc)
      initVals.push_back(v);

    auto forOp = rewriter.create<scf::ForOp>(
        loc, lowerBound, upperBound, step, initVals,
        [&](OpBuilder &b, Location loc, Value iv, ValueRange iterArgs) {
          // iterArgs: [res..., acc...]
          SmallVector<Value> curRes(iterArgs.begin(),
                                    iterArgs.begin() + inputs.size());
          SmallVector<Value> currAcc(iterArgs.begin() + inputs.size(),
                                     iterArgs.end());

          // Build offsets, sizes, and strides for ExtractSlice
          SmallVector<Value> dynOffsets;
          if (reverse) {
            auto actualIdx = b.create<arith::SubIOp>(loc, upperBound, iv);
            actualIdx = b.create<arith::SubIOp>(loc, actualIdx, step);
            dynOffsets.push_back(actualIdx);
          } else {
            dynOffsets.push_back(iv);
          }

          for (size_t j = 1; j < shape.size(); ++j) {
            dynOffsets.push_back(b.create<arith::ConstantIndexOp>(loc, 0));
          }
          SmallVector<int64_t> sizeVal({1});
          sizeVal.insert(sizeVal.end(), shape.begin() + 1, shape.end());
          SmallVector<int64_t> strides(shape.size(), 1);

          // iv is a Value, so build dynamic offsets for ExtractSlice
          SmallVector<Value> subInputs(inputs.size());
          std::transform(
              inputs.begin(), inputs.end(), subInputs.begin(), [&](auto val) {
                return b.create<tensor::ExtractSliceOp>(
                    loc,
                    RankedTensorType::get(
                        sizeVal,
                        cast<RankedTensorType>(resTypes[0]).getElementType()),
                    val, dynOffsets, /*sizes*/ ValueRange(),
                    /*strides*/ ValueRange(),
                    /*static_offsets*/
                    SmallVector<int64_t>(shape.size(), ShapedType::kDynamic),
                    sizeVal, strides);
              });

          // Check if this is the first iteration
          Value isFirstValue = b.create<arith::CmpIOp>(
              loc, arith::CmpIPredicate::eq, iv, lowerBound);
          scf::IfOp ifOp = b.create<scf::IfOp>(
              loc, isFirstValue,
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(loc, subInputs);
              },
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(
                    loc, accumulate(subInputs, currAcc, combineOp, b));
              });
          currAcc = ifOp.getResults();

          // Insert current accumulator into result tensor
          for (size_t i = 0; i < res.size(); ++i) {
            curRes[i] = b.create<tensor::InsertSliceOp>(
                loc, resTypes[i], currAcc[i], curRes[i], dynOffsets,
                /*sizes*/ ValueRange(), /*strides*/ ValueRange(),
                /*static_offsets*/
                SmallVector<int64_t>(shape.size(), ShapedType::kDynamic),
                sizeVal, strides);
          }

          SmallVector<Value> yieldVals;
          for (auto v : curRes)
            yieldVals.push_back(v);
          for (auto v : currAcc)
            yieldVals.push_back(v);

          b.create<scf::YieldOp>(loc, yieldVals);
        });

    // Extract result tensors from forOp
    SmallVector<Value> results;
    for (size_t i = 0; i < inputs.size(); ++i) {
      results.push_back(forOp.getResult(i));
    }
    return results;
  }

public:
  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto rank = cast<RankedTensorType>(op.getOperand(0).getType()).getRank();
    auto axis = op.getAxis();
    assert(axis < rank && "Expected axis is within the input rank");
    if (axis == (rank - 1))
      return lowerTrailingDimension(op, rewriter);

    return lowerNonTrailingDimension(op, rewriter);
  }
};

} // namespace

#endif
