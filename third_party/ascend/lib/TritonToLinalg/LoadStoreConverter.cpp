/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/TritonToLinalg/LoadStoreConverter.h"
#include "ascend/include/TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/TritonToLinalg/MaskAnalysis.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Utils/DebugUtils.h"
#include "ascend/include/Utils/InterleaveOptimization.h"
#include "ascend/include/Utils/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "Utils/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "llvm/Support/Debug.h"

#include <cassert>
#include <numeric>
#include <type_traits>

#define DEBUG_TYPE "triton-load-store-converter"

namespace LoadStoreConverter {
using namespace mlir;
using namespace triton;

const std::string MayImplicitTransposeWithLastAxisTAG =
    "MayImplicitTransposeWithLastAxis";
static constexpr llvm::StringLiteral kAlreadySyncAttr = "already_sync";

// During dialect conversion the adaptor may expose the converted memref
// directly, while a failed rewrite later leaves a one-input, one-output UCC in
// the diagnostic IR. Accept both representations, but only for the canonical
// i1-to-i8 scalar pointer normalization.
static Value
unwrapCanonicalI8ScalarPointerMaterialization(Value originalPointer,
                                              Value convertedPointer) {
  auto bitcastOp = originalPointer.getDefiningOp<triton::BitcastOp>();
  if (!bitcastOp)
    return nullptr;

  auto sourceType = dyn_cast<triton::PointerType>(bitcastOp.getSrc().getType());
  auto resultType = dyn_cast<triton::PointerType>(originalPointer.getType());
  if (!sourceType || !resultType || !sourceType.getPointeeType().isInteger(1) ||
      !resultType.getPointeeType().isInteger(8))
    return nullptr;

  Value memref = convertedPointer;
  if (auto castOp = memref.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (castOp.getInputs().size() != 1 || castOp.getOutputs().size() != 1 ||
        castOp.getOutputs().front() != memref)
      return nullptr;
    memref = castOp.getInputs().front();
  }

  auto memrefType = dyn_cast<MemRefType>(memref.getType());
  if (!memrefType || memrefType.getRank() != 1 ||
      !memrefType.getElementType().isInteger(8) ||
      !memrefType.getLayout().isIdentity() || memrefType.getMemorySpace() ||
      memrefType.getDimSize(0) != ShapedType::kDynamic)
    return nullptr;

  return memref;
}

static bool shouldMaterializeCustomPointer(Value ptr) {
  auto result = dyn_cast<OpResult>(ptr);
  if (!result)
    return false;

  Operation *defOp = result.getOwner();
  if (!defOp || !isDistributedTypeCustomOp(defOp))
    return false;

  auto srcIndices = defOp->getAttrOfType<DenseI32ArrayAttr>(
      ConverterUtils::customSrcPtrIndexAttrName);
  if (!srcIndices)
    return false;

  auto values = srcIndices.asArrayRef();
  unsigned resultIdx = result.getResultNumber();
  return resultIdx < values.size() && values[resultIdx] >= 0;
}

static FailureOr<Value>
resolveMemoryPointer(Value originalPtr, Value convertedPtr,
                     ConversionPatternRewriter &rewriter) {
  llvm::SmallDenseMap<Value, BlockData> known;

  if (shouldMaterializeCustomPointer(originalPtr))
    return BlockDataParser::materializePointer(originalPtr, rewriter, known);

  if (isa<MemRefType>(convertedPtr.getType()))
    return convertedPtr;

  return BlockDataParser::materializePointer(originalPtr, rewriter, known);
}

LogicalResult
AddPtrConverter::matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  Location loc = op.getLoc();
  insertDebugNop(loc, rewriter);
  llvm::SmallDenseMap<Value, BlockData> known;
  return BlockDataParser::rewriteAddPtr(op, adaptor, rewriter, known);
}

LogicalResult LoadConverter::toTensorAndReplace(
    triton::LoadOp &op, RankedTensorType &tensorType, Value localMem,
    bool mayImplicitTransposeWithLastAxis, const Location &loc,
    ConversionPatternRewriter &rewriter) const {
  Value loadedTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorType, localMem, true, true);
  propagateWasBoolToInt8Attr(op.getOperation(), loadedTensor.getDefiningOp(),
                             rewriter);

  if (mayImplicitTransposeWithLastAxis) {
    auto markOp = rewriter.create<annotation::MarkOp>(loc, loadedTensor);
    markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                    UnitAttr::get(rewriter.getContext()));
  }
  rewriter.replaceOp(op, loadedTensor);
  return success();
}

/// @brief Check whether the triton::LoadOp has been modified to the specified
/// state by the AddPtrConverter.
/// @param op The triton::LoadOp operation to be checked.
/// @return Return success if the operation conforms to the specified state;
/// otherwise, return failure.
LogicalResult
LoadConverter::checkModifiedByAddPtrConverter(triton::LoadOp &op) const {
  if (!isa<scf::ForOp>(op->getParentOp())) {
    return failure();
  }
  if (!op->hasAttr("IndirectLoad")) {
    return failure();
  }
  auto ptrOp = op.getPtr().getDefiningOp();
  auto ptrBlock = ptrOp->getBlock();
  auto opBlock = op->getBlock();
  if (ptrBlock == opBlock) {
    return failure();
  }

  return success();
}

void LoadConverter::propagateWasBoolToInt8Attr(
    Operation *srcLoadOp, Operation *dstOp, PatternRewriter &rewriter) const {
  const std::string WasBoolToInt8TAG = "was_bool_to_int8";
  if (!srcLoadOp || !dstOp)
    return;
  if (srcLoadOp->hasAttr(WasBoolToInt8TAG)) {
    dstOp->setAttr(WasBoolToInt8TAG, rewriter.getBoolAttr(true));
  }
}

/// @brief Continue to modify the triton::LoadOp from the state modified by the
/// AddPtrConverter.
/// @param op The triton::LoadOp operation to be processed.
/// @param adaptor The adaptor for the operation, used to obtain operands.
/// @param rewriter The pattern rewriter used to rewrite the operation.
/// @return Return success if the operation is successful; otherwise, return
/// failure.
LogicalResult LoadConverter::continueModifyFromAddPtrConverter(
    triton::LoadOp &op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto forOp = op->getParentOfType<scf::ForOp>();
  Operation *firstOp = &forOp.getBody()->front();
  auto extractOp = cast<tensor::ExtractOp>(firstOp);
  auto ivs = extractOp.getIndices();
  // Single iterArg which is inserted by AddPtrConverter.
  auto iterArg = forOp.getRegionIterArg(0);
  auto ptr = adaptor.getPtr();

  rewriter.setInsertionPointAfter(op);
  Value castVal = ptr.getDefiningOp<memref::ReinterpretCastOp>();
  Value idxZero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value loadVal =
      rewriter.create<memref::LoadOp>(loc, castVal, ValueRange{idxZero});
  propagateWasBoolToInt8Attr(op.getOperation(), loadVal.getDefiningOp(),
                             rewriter);
  Value insertedVal =
      rewriter.create<tensor::InsertOp>(loc, loadVal, iterArg, ValueRange{ivs});
  // a yield op is already created by AddPtrConverter.
  // so we need to replace it with a new yield op.
  Operation *terminator = forOp.getBody()->getTerminator();
  scf::YieldOp oldYieldOp = cast<scf::YieldOp>(terminator);
  auto yieldOp = rewriter.create<scf::YieldOp>(loc, ValueRange{insertedVal});
  rewriter.replaceOp(oldYieldOp, yieldOp);
  // Now the scf.for is complete, we can replace tt.load with it.
  auto rank = cast<ShapedType>(op.getResult().getType()).getShape().size();
  Operation *rootForOp = op;
  while (rank != 0) {
    rank--;
    rootForOp = rootForOp->getParentOfType<scf::ForOp>();
  }
  rewriter.replaceOp(op, rootForOp);
  LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(rootForOp) << "\n"; });
  return success();
}

void LoadConverter::fillTensorWithOtherForMaskScenario(
    Value other, Value localMem, ArrayRef<OpFoldResult> maskDim,
    ConversionPatternRewriter &rewriter) const {
  auto loc = localMem.getLoc();
  MemRefType originalType = cast<MemRefType>(localMem.getType());
  assert(originalType.hasStaticShape() && "only support static shape");
  assert(originalType.getRank() == maskDim.size() &&
         "shape and mask must have same rank");

  auto fillFlag =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false))
          .getResult();

  for (size_t i = 0; i < originalType.getShape().size(); ++i) {
    // Use dynamic value to judge whether overstep boundary
    auto shapeVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexAttr(originalType.getDimSize(i)));

    Value maskDimVal;
    if (isa<Attribute>(maskDim[i]))
      maskDimVal = rewriter.create<arith::ConstantOp>(
          loc, cast<IntegerAttr>(cast<Attribute>(maskDim[i])));
    else
      maskDimVal = cast<Value>(maskDim[i]);

    auto curCmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 maskDimVal, shapeVal);

    fillFlag = rewriter.create<arith::OrIOp>(loc, fillFlag, curCmp.getResult())
                   .getResult();
  }
  auto ifOp = rewriter.create<scf::IfOp>(loc, fillFlag);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
    rewriter.create<linalg::FillOp>(loc, ValueRange{other},
                                    ValueRange{localMem});
  }
  ifOp->setAttr(rewriter.getStringAttr("hivm.unlikely_condition"),
                UnitAttr::get(rewriter.getContext()));
}

LoadConverter::LoadConverter(MLIRContext *context)
    : OpConversionPattern<triton::LoadOp>(context) {}

// Masked load whose `other` is a non-scalar tensor.
// ---------------------------------------------------------------------------
// Case 1: other = mask   (Python: tl.load(ptr, mask=mask, other=mask))
//
// DSL:
//   mask = offs < N
//   x = tl.load(in_ptr + offs, mask=mask, other=mask)
//   tl.store(out_ptr + offs, x, mask=mask)
//
// TTIR (fp — uitofp; int — extui/extsi):
//   %mask  = arith.cmpi slt, %offs, %N : tensor<Nx i1>
//   %other = arith.uitofp %mask …            // or arith.extui for i8
//   %x     = tt.load %ptr, %mask, %other
//
// Lowering (linalg):
//   %alloc = memref.alloc()
//   memref.copy %src_view, %dst_view         // active SRC only
//   %loaded = bufferization.to_tensor %alloc
//   %zeros  = linalg.fill 0                  // [OTHER] cast(mask)→0 on pad
//   %x = arith.select %mask, %loaded, %zeros
// ---------------------------------------------------------------------------
// Case 2: other = prior tensor
//   (Python: other = tl.load(fill_ptr+offs); x = tl.load(ptr, mask=mask,
//   other=other); tl.store(out, x)  # full store — pad must stay OTHER)
//
// TTIR:
//   %other = tt.load %fill_ptr
//   %x     = tt.load %ptr, %mask, %other
//
// Lowering:
//   %other_t = bufferization.to_tensor …    // prior load
//   %alloc = memref.alloc()
//   memref.copy …                           // active SRC
//   %loaded = bufferization.to_tensor %alloc
//   %x = arith.select %mask, %loaded, %other_t
// ---------------------------------------------------------------------------
LogicalResult LoadConverter::replaceMaskedLoadWithTensorOther(
    triton::LoadOp op, Value alloc, bool mayImplicitTransposeWithLastAxis,
    ConversionPatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto tensorType = cast<RankedTensorType>(op.getResult().getType());
  Value mask = op.getMask();
  Value tensorOther = op.getOther();
  if (!mask || !tensorOther) {
    return rewriter.notifyMatchFailure(
        op,
        "replaceMaskedLoadWithTensorOther requires non-null mask and other");
  }

  Value loadedTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorType, alloc, true, true);
  propagateWasBoolToInt8Attr(op.getOperation(), loadedTensor.getDefiningOp(),
                             rewriter);
  if (mayImplicitTransposeWithLastAxis) {
    auto markOp = rewriter.create<annotation::MarkOp>(loc, loadedTensor);
    markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                    UnitAttr::get(rewriter.getContext()));
  }

  // Case 1: cast(mask) → inactive is 0. Cover fp (uitofp/sitofp) and int
  // (extui/extsi). Case 2: keep prior tensor as-is.
  Value otherVal = tensorOther;
  if (Value remappedOther = rewriter.getRemappedValue(tensorOther))
    otherVal = remappedOther;

  bool otherIsZeroOnInactive = false;
  if (auto uitofp = tensorOther.getDefiningOp<arith::UIToFPOp>())
    otherIsZeroOnInactive = (uitofp.getIn() == mask);
  else if (auto sitofp = tensorOther.getDefiningOp<arith::SIToFPOp>())
    otherIsZeroOnInactive = (sitofp.getIn() == mask);
  else if (auto extui = tensorOther.getDefiningOp<arith::ExtUIOp>())
    otherIsZeroOnInactive = (extui.getIn() == mask);
  else if (auto extsi = tensorOther.getDefiningOp<arith::ExtSIOp>())
    otherIsZeroOnInactive = (extsi.getIn() == mask);

  if (otherIsZeroOnInactive) {
    auto empty = rewriter.create<tensor::EmptyOp>(loc, tensorType.getShape(),
                                                  tensorType.getElementType());
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(tensorType.getElementType()));
    otherVal =
        rewriter
            .create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{empty})
            .getResult(0);
  }

  if (Value remappedMask = rewriter.getRemappedValue(mask))
    mask = remappedMask;

  Value result =
      rewriter.create<arith::SelectOp>(loc, mask, loadedTensor, otherVal);
  rewriter.replaceOp(op, result);
  return success();
}

static bool skipNonComplementaryStaticDeinterleave(triton::LoadOp lhs) {
  if (lhs->hasAttr("skip_deinterleave"))
    return true;
  auto lhsAddPtr = lhs.getPtr().getDefiningOp<triton::AddPtrOp>();
  if (!lhsAddPtr)
    return false;
  auto lhsOffset = getConstantIntValue(lhsAddPtr.getOffset());
  if (!lhsOffset)
    return false;
  auto pairBase = [](int64_t offset) {
    int64_t lane = offset % 2;
    return offset - (lane < 0 ? lane + 2 : lane);
  };

  for (triton::LoadOp rhs : lhs->getBlock()->getOps<triton::LoadOp>()) {
    if (lhs == rhs || rhs.getMask() || rhs.getOther() ||
        lhs.getResult().getType() != rhs.getResult().getType())
      continue;
    auto rhsAddPtr = rhs.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!rhsAddPtr || lhsAddPtr.getPtr() != rhsAddPtr.getPtr())
      continue;
    auto rhsOffset = getConstantIntValue(rhsAddPtr.getOffset());
    if (!rhsOffset)
      continue;
    if (lhsOffset.value() != rhsOffset.value() &&
        pairBase(lhsOffset.value()) == pairBase(rhsOffset.value()))
      continue;
    // The first load can be rewritten before its sibling; tag both now.
    auto skipAttr = UnitAttr::get(lhs->getContext());
    lhs->setAttr("skip_deinterleave", skipAttr);
    rhs->setAttr("skip_deinterleave", skipAttr);
    return true;
  }
  return false;
}

LogicalResult
LoadConverter::matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {

  // Check if tt.load is modified by AddPtrConverter to a specified state.
  if (checkModifiedByAddPtrConverter(op).succeeded()) {
    return continueModifyFromAddPtrConverter(op, adaptor, rewriter);
  }

  auto ptr = adaptor.getPtr();
  auto mask = op.getMask();
  auto other = op.getOther();
  auto loc = op.getLoc();
  insertDebugNopForMask(mask, rewriter);

  bool isScalarLoad = !isa<ShapedType>(op.getResult().getType());
  Value normalizedScalarMemref;
  if (isScalarLoad && op.getResult().getType().isInteger(8))
    normalizedScalarMemref =
        unwrapCanonicalI8ScalarPointerMaterialization(op.getPtr(), ptr);

  if (normalizedScalarMemref) {
    ptr = normalizedScalarMemref;
  } else {
    FailureOr<Value> resolvedPtr =
        resolveMemoryPointer(op.getPtr(), ptr, rewriter);
    if (failed(resolvedPtr))
      return rewriter.notifyMatchFailure(
          op, "unable to materialize the load pointer as a memref");
    ptr = *resolvedPtr;
  }
  // handling scalar
  if (isScalarLoad) {
    Value scalarMemref = normalizedScalarMemref;
    if (!scalarMemref) {
      FailureOr<Value> resolvedScalarMemref =
          BlockDataParser::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);
      if (failed(resolvedScalarMemref))
        return rewriter.notifyMatchFailure(
            op, "scalar pointer operand has not been converted to a memref");
      scalarMemref = *resolvedScalarMemref;
    }

    auto resTy = op.getResult().getType();
    auto idxZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto createScalarLoad = [&]() -> Value {
      auto load = rewriter.create<memref::LoadOp>(loc, resTy, scalarMemref,
                                                  idxZero.getResult());
      propagateWasBoolToInt8Attr(op.getOperation(), load.getOperation(),
                                 rewriter);
      return load.getResult();
    };
    if (!mask) {
      rewriter.replaceOp(op, createScalarLoad());
      return success();
    }

    // Keep scalar results scalar, and guard the access itself: a select after
    // an unconditional load would still read a masked-off, possibly invalid
    // address.
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{resTy},
                                           adaptor.getMask(), true);
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      rewriter.create<scf::YieldOp>(loc, createScalarLoad());

      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      Value inactiveValue = adaptor.getOther();
      if (!inactiveValue) {
        // Without `other`, the inactive value is undefined. Choose zero
        // without accessing memory or introducing a tensor temporary.
        inactiveValue = rewriter.create<arith::ConstantOp>(
            loc, resTy, rewriter.getZeroAttr(resTy));
      }
      rewriter.create<scf::YieldOp>(loc, inactiveValue);
    }
    rewriter.replaceOp(op, ifOp.getResults());
    return success();
  }

  int64_t lastStride = -1;
  if (isa<BlockArgument>(ptr)) {
    auto u = ptr;
    while (auto blkArg = dyn_cast<BlockArgument>(u)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blkArg.getOwner()->getParentOp())) {
        auto prt = forOp->getOperand(3 + blkArg.getArgNumber() - 1);
        u = prt;
      } else {
        u = nullptr;
        break;
      }
    }
    if (u && isa<memref::ReinterpretCastOp>(u.getDefiningOp())) {
      auto ret = mlir::ConverterUtils::getLastStrideOfReinterpretCastOp(
          dyn_cast<memref::ReinterpretCastOp>(u.getDefiningOp()));
      if (ret.has_value())
        lastStride = *ret;
    }
  }

  // handling no mask
  auto memRefType = dyn_cast<MemRefType>(ptr.getType());
  if (!memRefType) {
    return rewriter.notifyMatchFailure(
        op, "LoadOp expects a memref, not a memref of pointers");
  }
  if (!op->hasAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG)) {
    auto memrefOp = dyn_cast<memref::ReinterpretCastOp>(ptr.getDefiningOp());
    auto ret = mlir::ConverterUtils::getLastStrideOfReinterpretCastOp(memrefOp);
    if (ret.has_value())
      lastStride = *ret;
  }
  bool mayImplicitTransposeWithLastAxis =
      (existDotFlag) &&
      (!op->hasAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG)) &&
      (lastStride != 1 &&
       mlir::ConverterUtils::isaPermutedMemRefType(memRefType));
  auto memRefShape = memRefType.getShape();
  auto memRefElementType = memRefType.getElementType();

  Value allocOp;
  Value allocOpTmp;
  auto parentLoop = dyn_cast<scf::ForOp>(op->getParentOp());
  bool hasValidDiscreteLoop = op->hasAttr(ConverterUtils::discreteAttrName) &&
                              parentLoop &&
                              parentLoop->hasAttr("ExtractedLoadOrStore");
  if (hasValidDiscreteLoop) {
    Operation *loop = op->getParentOp();
    int extractedLoopCount = 1;
    for (auto parentOp = loop->getParentOp();
         parentOp->hasAttr("ExtractedLoadOrStore");
         parentOp = parentOp->getParentOp()) {
      loop = parentOp;
      extractedLoopCount++;
    }
    rewriter.setInsertionPoint(loop);
    auto loopOp = cast<scf::ForOp>(loop);
    auto fullMemRefShape =
        cast<RankedTensorType>(loopOp.getInitArgs()[0].getType()).getShape();
    auto fullMemRefType = MemRefType::get(fullMemRefShape, memRefElementType);
    bool isIndexSelectScenario =
        (extractedLoopCount == 1) && (fullMemRefShape.size() > 1u);
    if (isIndexSelectScenario)
      loopOp->setAttr("hivm.parallel_loop", rewriter.getUnitAttr());
    allocOp = rewriter.create<memref::AllocOp>(loc, fullMemRefType);
    allocOpTmp = allocOp;
    rewriter.setInsertionPointAfter(loop);
    auto toTensorOp = rewriter.create<bufferization::ToTensorOp>(
        loc, RankedTensorType::get(fullMemRefShape, memRefElementType), allocOp,
        true, true);
    rewriter.replaceAllUsesWith(loopOp->getResult(0), toTensorOp->getResult(0));
    tensor::InsertSliceOp insertSliceOp = nullptr;
    for (auto *user : op->getUsers()) {
      if (auto targetOp = dyn_cast<tensor::InsertSliceOp>(user)) {
        insertSliceOp = targetOp;
        break;
      }
    }
    auto offsets = insertSliceOp.getMixedOffsets();
    auto sizes = insertSliceOp.getMixedSizes();
    auto strides = insertSliceOp.getMixedStrides();
    auto allocType = memref::SubViewOp::inferResultType(fullMemRefType, offsets,
                                                        sizes, strides);
    rewriter.setInsertionPoint(op);
    allocOp = rewriter.create<memref::SubViewOp>(
        loc, cast<MemRefType>(allocType), allocOp, offsets, sizes, strides);
    rewriter.replaceAllUsesExcept(insertSliceOp.getResult(),
                                  insertSliceOp.getDest(), insertSliceOp);
    rewriter.eraseOp(insertSliceOp);
  } else {
    allocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(memRefShape, memRefElementType));
  }

  auto tensorType = RankedTensorType::get(memRefShape, memRefElementType);
  // boundary check
  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    auto makeTensorPtrOp = op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    auto boundarySizes = mlir::ConverterUtils::getBoundarySizes(
        boundaryCheck, /*remapped*/ ptr, loc, rewriter);
    // handle the padding
    auto padding = op.getPadding();
    SmallVector<OpFoldResult> srcOffsets(boundarySizes.size(),
                                         rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> dstOffsets;
    if (makeTensorPtrOp) {
      auto zeroVal = rewriter.createOrFold<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(0));
      for (auto [idx, offVal] : llvm::enumerate(makeTensorPtrOp.getOffsets())) {
        if (llvm::find(boundaryCheck, idx) == boundaryCheck.end()) {
          dstOffsets.push_back(srcOffsets[idx]);
          continue;
        }
        Value offset =
            rewriter.createOrFold<arith::SubIOp>(loc, zeroVal, offVal);
        Value size =
            getValueOrCreateConstantIndexOp(rewriter, loc, boundarySizes[idx]);
        offset = rewriter.createOrFold<arith::MaxSIOp>(loc, offset, zeroVal);
        offset = rewriter.createOrFold<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offset);
        OpFoldResult ofr;
        if (auto constOp = offset.getDefiningOp<arith::ConstantOp>()) {
          ofr = constOp.getValue();
        } else {
          ofr = offset;
        }
        ofr = minOpFoldResult(ofr, size, loc, rewriter);
        boundarySizes[idx] = subOpFoldResult(size, ofr, loc, rewriter);
        dstOffsets.push_back(ofr);
      }
    } else {
      dstOffsets = srcOffsets;
    }
    if (padding.has_value()) {
      TypedAttr padAttr = rewriter.getZeroAttr(memRefElementType);
      // triton already ensure only NAN and ZERO are passed in
      if (padding.value() == triton::PaddingOption::PAD_NAN) {
        // FIXME: Why NaN requires elemTy to be non-int or non-index?
        assert(!memRefElementType.isIntOrIndex());
        auto apNaN = llvm::APFloat::getNaN(
            cast<FloatAttr>(padAttr).getValue().getSemantics());
        padAttr = rewriter.getFloatAttr(memRefElementType, apNaN);
      }
      auto padVal = rewriter.create<arith::ConstantOp>(loc, padAttr);

      fillTensorWithOtherForMaskScenario(padVal, allocOp, boundarySizes,
                                         rewriter);
    }
    auto srcSubView = mlir::ConverterUtils::makeSubViewOp(
        ptr, srcOffsets, boundarySizes, loc, rewriter);
    auto dstSubview = mlir::ConverterUtils::makeSubViewOp(
        allocOp, dstOffsets, boundarySizes, loc, rewriter);
    auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubView, dstSubview);
    propagateWasBoolToInt8Attr(op.getOperation(), copyOp.getOperation(),
                               rewriter);
    if (mayImplicitTransposeWithLastAxis) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, dstSubview);
      markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                      UnitAttr::get(rewriter.getContext()));
    }
    return this->toTensorAndReplace(op, tensorType, allocOp,
                                    mayImplicitTransposeWithLastAxis, loc,
                                    rewriter);
  }

  if (!mask) {
    assert(!other && "can not input 'other' when 'mask' is not set");
    if (auto unrealizedCastOp =
            ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
      // TODO : not support handle  associate with "module"
      // hint : can be handled in Linearize
      op->emitError("meeting unexpected UCC in LoadConverter!");
      return failure();
    } else {
      // If last dimension stride equals 2, try deinterleave optimization.
      auto [ptrStrides, ptrOffsets] = memRefType.getStridesAndOffset();
      if (ptrStrides.back() == 2 && (memRefShape.back() % 2 == 0) &&
          !skipNonComplementaryStaticDeinterleave(op) &&
          mlir::triton::DeinterleaveStatusOptimization(op, adaptor, rewriter)
              .succeeded()) {
        return success();
      }
      auto copyOp = rewriter.create<memref::CopyOp>(loc, ptr, allocOp);
      propagateWasBoolToInt8Attr(op.getOperation(), copyOp.getOperation(),
                                 rewriter);
      if (mayImplicitTransposeWithLastAxis &&
          allocOp.getDefiningOp<memref::AllocOp>()) {
        auto markOp = rewriter.create<annotation::MarkOp>(loc, allocOp);
        markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                        UnitAttr::get(rewriter.getContext()));
      } else if (mayImplicitTransposeWithLastAxis &&
                 allocOp.getDefiningOp<memref::SubViewOp>()) {
        auto markOp = rewriter.create<annotation::MarkOp>(loc, allocOpTmp);
        markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                        UnitAttr::get(rewriter.getContext()));
      }
    }

    return this->toTensorAndReplace(op, tensorType, allocOp,
                                    mayImplicitTransposeWithLastAxis, loc,
                                    rewriter);
  }

  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);
  if (isContMask.failed()) {
    return rewriter.notifyMatchFailure(
        op, "can not lower uncontinuout masked loads");
  }

  Value tensorOtherBase;
  if (other) {
    auto scalarOther =
        mlir::ConverterUtils::getScalarValue(other, loc, rewriter);
    if (scalarOther) {
      fillTensorWithOtherForMaskScenario(scalarOther, allocOp, mstate.dims,
                                         rewriter);
    } else if (isa<RankedTensorType>(other.getType())) {
      tensorOtherBase = other;
    } else {
      return rewriter.notifyMatchFailure(
          op, "other value used in masked load produced by unsupported "
              "instruction");
    }
  }

  // To enable deinterleave optimization with mask load, mask state along last
  // dimension couldn't be split, which means `dims.back()` must be equal to
  // origin type last dimension constant size and `offsets.back()` must be 0.
  //
  // The basis is that last dimension range comparison would generate
  // unaccepted discontinuous mask.
  if (!tensorOtherBase && mstate.getRank() == memRefType.getRank() &&
      isConstantIntValue(mstate.offsets.back(), 0) &&
      isConstantIntValue(mstate.dims.back(), memRefType.getShape().back())) {
    auto [ptrStrides, ptrOffsets] = memRefType.getStridesAndOffset();
    if (ptrStrides.back() == 2 && (memRefType.getShape().back() % 2 == 0) &&
        DeinterleaveStatusWithMaskOptimization(op, adaptor, rewriter, mstate,
                                               allocOp)
            .succeeded()) {
      return success();
    }
  }

  if (auto unrealizedCastOp = ptr.getDefiningOp<UnrealizedConversionCastOp>()) {
    // TODO : not support handle  associate with "module"
    // hint : can be handled in Linearize
    op->emitError("meeting unexpected UCC in LoadConverter!");
    return failure();
  } else {
    if (mstate.isMemrefSubviewValid(ptr, rewriter)) {
      memref::SubViewOp srcSubView = mstate.getSubview(ptr, loc, rewriter);
      memref::SubViewOp dstSubView = mstate.getSubview(allocOp, loc, rewriter);
      MemRefType dstSubViewType = mlir::cast<MemRefType>(dstSubView.getType());

      auto [srcStrides, srcOffset] = dstSubViewType.getStridesAndOffset();
      MemRefType castType = MemRefType::get(
          dstSubViewType.getShape(), dstSubViewType.getElementType(),
          makeStridedLinearLayoutMap(srcStrides, srcOffset,
                                     rewriter.getContext()));
      auto castOp = rewriter.create<memref::CastOp>(loc, castType, dstSubView);
      auto copyOp = rewriter.create<memref::CopyOp>(loc, srcSubView, castOp);
      propagateWasBoolToInt8Attr(op.getOperation(), copyOp.getOperation(),
                                 rewriter);
    }

    if (mayImplicitTransposeWithLastAxis &&
        allocOp.getDefiningOp<memref::AllocOp>()) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, allocOp);
      markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                      UnitAttr::get(rewriter.getContext()));
    } else if (mayImplicitTransposeWithLastAxis &&
               allocOp.getDefiningOp<memref::SubViewOp>()) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, allocOpTmp);
      markOp->setAttr(MayImplicitTransposeWithLastAxisTAG,
                      UnitAttr::get(rewriter.getContext()));
    }
  }

  if (!tensorOtherBase) {
    return this->toTensorAndReplace(op, tensorType, allocOp,
                                    mayImplicitTransposeWithLastAxis, loc,
                                    rewriter);
  }

  return replaceMaskedLoadWithTensorOther(
      op, allocOp, mayImplicitTransposeWithLastAxis, rewriter);
}

AtomicRMWConverter::AtomicRMWConverter(MLIRContext *context)
    : OpConversionPattern<triton::AtomicRMWOp>(context) {}

// lowering tt.atomicRMW to linalg.generic
// If atomic op's return value is used by other op as it's the old value stored
// at the ptrwe will use tt.load to get it
//
// example:
// input:
//  %return_value = tt.atomic_rmw fadd, acq_rel, gpu,
//     %output_memref, %input_tensor, %mask :
//             (tensor<256x!tt.ptr<f32>>, tensor<256xf32>, tensor<256xi1>)
//                       -> tensor<256xf32>
//
// output:
//  memref.copy %output_memref, %ub_buf : memref<?xf32> to memref<?xf32>
//  %17 = bufferization.to_tensor %alloc_3 restrict writable : memref<256xf32>
//  linalg.generic
//    {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]}
//    ins(%output_memref, %masked_input_memref : memref<?xf32>, memref<?xf32>)
//    outs(%subview_2 : memref<?xf32>)
//    attrs = {GenericAtomicRMW = "fadd", MemSemantic = "acq_rel",
//                                        MemSyncScope = "gpu"} {
//    ^bb0(%in: f32, %in_9: f32, %out: f32):
//      %25 = arith.addf %in, %in_9 : f32
//      linalg.yield %25 : f32
//    }
LogicalResult
AtomicRMWConverter::matchAndRewrite(triton::AtomicRMWOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  auto ptr = adaptor.getPtr();
  auto val = op.getVal();
  auto loc = op.getLoc();
  auto mask = op.getMask();
  auto rmwOp = op.getAtomicRmwOp();
  auto resType = dyn_cast<TensorType>(op.getResult().getType());
  insertDebugNop(loc, rewriter);
  insertDebugNopForMask(mask, rewriter);

  FailureOr<Value> resolvedPtr =
      resolveMemoryPointer(op.getPtr(), ptr, rewriter);
  if (failed(resolvedPtr))
    return rewriter.notifyMatchFailure(
        op, "unable to materialize the atomic RMW pointer as a memref");
  ptr = *resolvedPtr;
  auto ptrType = dyn_cast<MemRefType>(ptr.getType());
  if (!resType)
    return rewriter.notifyMatchFailure(
        op, "atomicRMWConverter: scalar will be handled by "
            "ScalarAtomicRMWCanonicalizer");
  if (!ptrType)
    return rewriter.notifyMatchFailure(
        op, "AtomicRMWOp expects a memref, not a memref of pointers");

  const std::map<RMWOp, hivm::AtomicKind> atomicKindMap = {
      {RMWOp::ADD, hivm::AtomicKind::ADD},
      {RMWOp::FADD, hivm::AtomicKind::ADD},
      {RMWOp::OR, hivm::AtomicKind::OR},
      {RMWOp::XOR, hivm::AtomicKind::XOR},
      {RMWOp::AND, hivm::AtomicKind::AND},
      {RMWOp::MIN, hivm::AtomicKind::MIN},
      {RMWOp::UMIN, hivm::AtomicKind::UMIN},
      {RMWOp::MAX, hivm::AtomicKind::MAX},
      {RMWOp::UMAX, hivm::AtomicKind::UMAX},
      {RMWOp::XCHG, hivm::AtomicKind::XCHG},
  };
  const std::map<RMWOp, hfusion::AtomicKind> hfusionAtomicKindMap = {
      {RMWOp::ADD, hfusion::AtomicKind::ADD},
      {RMWOp::FADD, hfusion::AtomicKind::ADD},
      {RMWOp::OR, hfusion::AtomicKind::OR},
      {RMWOp::XOR, hfusion::AtomicKind::XOR},
      {RMWOp::AND, hfusion::AtomicKind::AND},
      {RMWOp::MIN, hfusion::AtomicKind::MIN},
      {RMWOp::UMIN, hfusion::AtomicKind::UMIN},
      {RMWOp::MAX, hfusion::AtomicKind::MAX},
      {RMWOp::UMAX, hfusion::AtomicKind::UMAX},
      {RMWOp::XCHG, hfusion::AtomicKind::XCHG},
  };
  assert(atomicKindMap.find(rmwOp) != atomicKindMap.end());
  auto atomicKind =
      hivm::AtomicKindAttr::get(rewriter.getContext(), atomicKindMap.at(rmwOp));
  assert(hfusionAtomicKindMap.find(rmwOp) != hfusionAtomicKindMap.end());
  auto hfusionAtomicKind = hfusion::AtomicKindAttr::get(
      rewriter.getContext(), hfusionAtomicKindMap.at(rmwOp));

  auto dstMemref = ptr;
  Value inputVal = val;

  // Lazily materialize a memref view only when we truly need buffer
  // semantics (e.g., mask subview or XCHG lowering). Otherwise keep tensor
  // inputs to avoid redundant to_memref conversions before hivm.store.
  auto getInputMemref = [&]() -> Value {
    if (isa<MemRefType>(inputVal.getType()))
      return inputVal;
    return rewriter.create<bufferization::ToBufferOp>(loc, ptrType, inputVal);
  };
  auto inputMemref = getInputMemref();
  auto inputMemrefType = cast<MemRefType>(inputMemref.getType());
  auto elementType = inputMemrefType.getElementType();
  auto isHardwareSupported =
      (rmwOp == RMWOp::ADD || rmwOp == RMWOp::FADD || rmwOp == RMWOp::MAX ||
       rmwOp == RMWOp::MIN) &&
      (elementType.isF16() || elementType.isBF16() || elementType.isF32() ||
       elementType.isInteger(8) || elementType.isInteger(16) ||
       elementType.isInteger(32));

  bool isDiscreteMask = false;
  bool hasContinuousMaskSubview = false;
  bool hasUsedReturn = !op.getResult().use_empty();
  MaskState mstate;
  if (mask) {
    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (constantMask && !isConstantMaskTrue(mask)) {
      rewriter.eraseOp(op);
      return success();
    }
    isDiscreteMask = mstate.parse(mask, loc, rewriter).failed();
    if (!constantMask && !isDiscreteMask) {
      // For dstMemref (store output), use subview to maintain reference to
      // original memref. For inputVal (store input), use tensor.extract_slice
      // to keep tensor semantics.
      dstMemref = mstate.getSubview(ptr, loc, rewriter);
      hasContinuousMaskSubview = true;
      if (isHardwareSupported) {
        auto inputTensorType = RankedTensorType::get(
            inputMemrefType.getShape(), inputMemrefType.getElementType());
        if (!isa<RankedTensorType>(inputVal.getType()))
          inputVal = rewriter.create<bufferization::ToTensorOp>(
              loc, inputTensorType, inputMemref, true, true);
        inputVal = mstate.getExtractSlice(inputVal, loc, rewriter);
      } else {
        inputMemref = mstate.getSubview(inputMemref, loc, rewriter);
      }
    }
  }

  bool needsReturnValueLock =
      hasUsedReturn && hasContinuousMaskSubview && isHardwareSupported;
  Value returnValueLock;
  if (hasUsedReturn) {
    auto tensorType =
        RankedTensorType::get(ptrType.getShape(), ptrType.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(ptrType.getShape(), ptrType.getElementType()));
    Value copySrc = ptr;
    Value copyDst = alloc;
    if (hasContinuousMaskSubview) {
      // Masked-off atomic results are undefined. Copy only the active region
      // so the old-value read and the atomic store use the same address view.
      copySrc = dstMemref;
      copyDst = mstate.getSubview(alloc, loc, rewriter);
    }
    if (needsReturnValueLock) {
      auto lockType = MemRefType::get({1}, rewriter.getI64Type());
      auto lockVar =
          rewriter.create<hivm::CreateSyncBlockLockOp>(loc, lockType, Value());
      returnValueLock = lockVar.getResult();
      rewriter.create<hivm::SyncBlockLockOp>(loc, returnValueLock);
    }
    rewriter.create<memref::CopyOp>(loc, copySrc, copyDst);
    Value tensorToReplace = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensorToReplace);
  }

  if (isDiscreteMask) {
    if (rmwOp != RMWOp::XCHG) {
      return op.emitError(
          "Discrete mask is only expected for XCHG; other atomics "
          "should be lowered without discrete masks");
    }
    Value memrefMask = mask;
    if (auto maskTypeT = dyn_cast<TensorType>(mask.getType())) {
      MemRefType maskTypeM =
          MemRefType::get(maskTypeT.getShape(), maskTypeT.getElementType());
      memrefMask =
          rewriter.create<bufferization::ToBufferOp>(loc, maskTypeM, mask);
    }
    rewriter.create<hfusion::AtomicXchgOp>(op.getLoc(), TypeRange(),
                                           inputMemref, dstMemref, memrefMask);
  } else {
    if (isHardwareSupported) {
      auto storeOp = rewriter.create<hivm::StoreOp>(
          op.getLoc(), TypeRange{}, inputVal, dstMemref, atomicKind);
      if (needsReturnValueLock) {
        // The return-value copy and the atomic update must be one critical
        // section. Mark the store so HIVM does not add another lock.
        storeOp->setAttr(kAlreadySyncAttr, rewriter.getUnitAttr());
        rewriter.create<hivm::SyncBlockUnlockOp>(loc, returnValueLock);
      }
    } else if (rmwOp == RMWOp::XCHG)
      rewriter.create<hfusion::AtomicXchgOp>(op.getLoc(), TypeRange(),
                                             inputMemref, dstMemref);
    else {
      if (rmwOp == RMWOp::OR || rmwOp == RMWOp::XOR || rmwOp == RMWOp::AND) {
        if (!elementType.isSignlessIntOrIndex()) {
          return op->emitOpError()
                 << "must be signless-integer-like, but got " << elementType;
        }
      }
      // Currently, for atomic kind and element type that is not supported by
      // the hardware, we use software to simulate the computation. However,
      // decompose now happens in both HFusion and HIVM, and is not consistent
      // for 910B and 91095. Therefore, we convert to different atomic/store ops
      // for now. This should be unified and refactored later.
      if (compileOn91095Flag) {
        rewriter.create<hfusion::StoreOp>(
            op.getLoc(), TypeRange{}, ValueRange{inputMemref},
            ValueRange{dstMemref}, hfusionAtomicKind,
            ArrayRef<NamedAttribute>{});
      } else {
        rewriter.create<hfusion::AtomicRMWOp>(op.getLoc(), TypeRange(),
                                              inputMemref, dstMemref,
                                              hfusionAtomicKind);
      }
    }
  }

  if (op.getResult().use_empty()) {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult
AtomicCASConverter::matchAndRewrite(triton::AtomicCASOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  // If the result of AtomicCASOp is not used, we don't need to load the old
  // data stored at the ptr
  auto ptr = adaptor.getPtr();
  auto cmp = op.getCmp();
  auto val = op.getVal();
  auto loc = op.getLoc();

  FailureOr<Value> resolvedPtr =
      resolveMemoryPointer(op.getPtr(), ptr, rewriter);
  if (failed(resolvedPtr))
    return rewriter.notifyMatchFailure(
        op, "unable to materialize the atomic CAS pointer as a memref");
  ptr = *resolvedPtr;
  auto resType = dyn_cast<TensorType>(op.getResult().getType());
  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, "atomicCASConverter: scalar will be handled by "
            "ScalarAtomicCASCanonicalizer");
  }

  // 1. Simple case where no mask is used.
  auto type = dyn_cast<MemRefType>(ptr.getType());
  if (!type) {
    // Seen when implicit broadcasting is done late in a chain of
    // operations. The workaround is to broadcast the pointers early in the
    // address calculation. A proper fix is complicated, but at least we can
    // provide a better error message.
    return rewriter.notifyMatchFailure(
        op, "AtomicCASOp expects a memref, not a memref of pointers");
  }

  auto dstMemref = ptr;
  // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
  // any more in latest LLVM(triton LLVM dependency has involed this), so we
  // need to convert tensor to buffer early.
  auto dstOriType = cast<MemRefType>(dstMemref.getType());
  MemRefType dstType =
      MemRefType::get(dstOriType.getShape(), dstOriType.getElementType());
  Value inputMemref =
      rewriter.create<bufferization::ToBufferOp>(loc, dstType, val);

  Value cmpMemref =
      rewriter.create<bufferization::ToBufferOp>(loc, dstType, cmp);

  // create element-wise map
  int64_t rank = type.getRank();
  SmallVector<AffineExpr> inputDims;
  auto context = rewriter.getContext();

  for (int i = 0; i < rank; i++) {
    inputDims.push_back(getAffineDimExpr(i, context));
  }

  SmallVector<AffineMap> indexingMaps;
  // As mask has been erased for now
  // the number of input must be 2
  // the input memref is also the output memref
  // Thus, there are a total of four inputs and outputs.
  // so here we have 4 map to create
  for (int i = 0; i < 4; i++) { // 4: 3 input and 1 output
    indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
  }

  if (!op.getResult().use_empty()) {
    auto tensorType =
        RankedTensorType::get(type.getShape(), type.getElementType());
    auto alloc = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(type.getShape(), type.getElementType()));

    // For the return value, don't need to care about mask for now
    // this op don't support other, so we best not fill it
    rewriter.create<memref::CopyOp>(loc, ptr, alloc);
    Value tensor = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, alloc, true /* restrict */, true /* writable */);
    rewriter.replaceOp(op, tensor);
  }

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, ValueRange{dstMemref, cmpMemref, inputMemref},
      mlir::ValueRange{dstMemref}, indexingMaps,
      mlir::ConverterUtils::getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value lhs = blockArgs[0];
        Value rhs = blockArgs[1];
        Value setValue = blockArgs[2];
        Value cond;
        if (mlir::isa<mlir::FloatType>(lhs.getType())) {
          cond = nestedBuilder.create<arith::CmpFOp>(
              nestedLoc, arith::CmpFPredicate::UEQ, lhs, rhs);
        } else {
          cond = nestedBuilder.create<arith::CmpIOp>(
              nestedLoc, arith::CmpIPredicate::eq, lhs, rhs);
        }
        auto ifOp = nestedBuilder.create<scf::IfOp>(
            nestedLoc, TypeRange{setValue.getType()}, cond, true);
        {
          OpBuilder::InsertionGuard guard(nestedBuilder);
          nestedBuilder.setInsertionPointToEnd(&ifOp.getThenRegion().front());
          nestedBuilder.create<scf::YieldOp>(nestedLoc, setValue);
        }
        {
          OpBuilder::InsertionGuard guard(nestedBuilder);
          nestedBuilder.setInsertionPointToEnd(&ifOp.getElseRegion().front());
          nestedBuilder.create<scf::YieldOp>(nestedLoc, lhs);
        }
        nestedBuilder.setInsertionPointToEnd(nestedBuilder.getBlock());
        nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                    ifOp.getResult(0));
      });

  const StringRef genericAtomicRMW = "GenericAtomicRMW";
  const StringRef memSemantic = "MemSemantic";
  const StringRef memSyncScope = "MemSyncScope";
  auto attr = mlir::StringAttr::get(context, "cas");

  linalgOp->setAttr(genericAtomicRMW, attr);
  linalgOp->setAttr(memSemantic,
                    rewriter.getStringAttr(stringifyEnum(op.getSem())));
  linalgOp->setAttr(memSyncScope,
                    rewriter.getStringAttr(stringifyEnum(op.getScope())));

  linalgOp->setAttr("Software", rewriter.getUnitAttr());

  // tt.atomicRMW op has two part of feature
  // 1. load the old data at the ptr
  // 2. atomically store the data on ub to the ptr
  //    at the same time it perform the action it has been assigned
  // So we lower this op to load + atomically store
  //
  // The first part is not necessary when the returned value of atomic op
  // is not used, it will be deleted cause it's meaningless
  // Here, we preemptively determine whether it will be used
  // and decide whether it is necessary to create the load process based on
  // this assessment.
  //
  // logic of handling is copied
  if (op.getResult().use_empty()) {
    rewriter.eraseOp(op);
  }
  return success();
}

LogicalResult
ScalarStoreCanonicalizer::matchAndRewrite(triton::StoreOp op,
                                          PatternRewriter &rewriter) const {
  if (!op.getValue().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarStoreCanonicalizer handles scalar store scene!");
  }
  auto ptr = op.getPtr();
  auto mask = op.getMask();
  auto value = op.getValue();
  if (mask) {
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::StoreOp>(loc, ptr, value, op.getCache(),
                                    op.getEvict());
          b.create<scf::YieldOp>(loc);
        });
    return success();
  }

  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, value.getType());
  auto valSplat = rewriter.create<triton::SplatOp>(op.getLoc(), valTy, value);
  auto newStoreOp = rewriter.create<triton::StoreOp>(
      op.getLoc(), ptrSplat, valSplat, op.getCache(), op.getEvict());
  rewriter.replaceOp(op, newStoreOp);
  return success();
}

LogicalResult
ScalarAtomicRMWCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicRMWCanonicalizer handles scalar atomic rmw op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());
  auto maskTy = RankedTensorType::get({(int64_t)1}, op.getMask().getType());
  auto maskSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), maskTy, op.getMask());

  auto newAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valTy, op.getAtomicRmwOp(), ptrSplat, valSplat, maskSplat,
      op.getSem(), op.getScope());
  auto idxZero =
      rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp,
                                                 ValueRange({idxZero}));
  return success();
}

LogicalResult
ScalarAtomicCASCanonicalizer::matchAndRewrite(triton::AtomicCASOp op,
                                              PatternRewriter &rewriter) const {
  if (!op.getVal().getType().isIntOrIndexOrFloat() &&
      !op.getCmp().getType().isIntOrIndexOrFloat()) {
    return rewriter.notifyMatchFailure(
        op, "ScalarAtomicCASCanonicalizer handles scalar atomic cas op scene!");
  }

  auto ptr = op.getPtr();
  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto cmpTy = RankedTensorType::get({(int64_t)1}, op.getCmp().getType());
  auto cmpSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), cmpTy, op.getCmp());
  auto valTy = RankedTensorType::get({(int64_t)1}, op.getVal().getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, op.getVal());

  auto newAtomicOp = rewriter.create<triton::AtomicCASOp>(
      op.getLoc(), valTy, ptrSplat, cmpSplat, valSplat, op.getSem(),
      op.getScope());
  auto idxZero =
      rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp,
                                                 ValueRange({idxZero}));
  return success();
}

// The atomic max op with float input will be devided into
// two atomic max ops with integer input
// One handles the part of the tensor greater than zero
// the other deals with the part less than zero
// It will lead to maskAnalysis failure
// So here we need to revert the procedures in semantics.py
// The triton IR is like
//
// %cst_0 = arith.constant dense<0.000000e+00> : tensor<1x256xf32>
// %1 = tt.bitcast %value : tensor<1x256xf32> -> tensor<1x256xi32>
// %2 = tt.bitcast %ptr : tensor<1x256x!tt.ptr<f32>> ->
// tensor<1x256x!tt.ptr<i32>> %3 = arith.cmpf oge, %1, %cst_0 %4 = arith.cmpf
// olt, %1, %cst_0 %5 = arith.andi %8, %3 %6 = tt.atomic_rmw max, acq_rel, gpu,
// %2, %1, %5 :
//    (tensor<1x256x!tt.ptr<i32>>, tensor<1x256xi32>, tensor<1x256xi1>) ->
//    tensor<1x256xi32>
// %7 = arith.andi %8, %4
// %8 = tt.atomic_rmw umin, acq_rel, gpu, %2, %1, %7 :
//    (tensor<1x256x!tt.ptr<i32>>, tensor<1x256xi32>, tensor<1x256xi1>) ->
//    tensor<1x256xi32>
//
// it's hard to handle and meaningless complicated for our device
// so we revert it to
// %0 = tt.atomic_rmw max, acq_rel, gpu, %23, %21, %8 :
//    (tensor<1x256x!tt.ptr<f32>>, tensor<1x256xf32>, tensor<1x256xi1>) ->
//    tensor<1x256xf32>
LogicalResult
AtomicMaxMinCanonicalizer::matchAndRewrite(triton::AtomicRMWOp op,
                                           PatternRewriter &rewriter) const {
  // Revert the op to its original form
  auto ptrBitcastOp = op.getPtr().getDefiningOp<triton::BitcastOp>();
  triton::SplatOp ptrSplatOp;
  if (!ptrBitcastOp) {
    // For tensor atomics on a scalar base pointer, semantic.py emits
    //   splat(bitcast(ptr<f32> -> ptr<i32>))
    // instead of
    //   bitcast(tensor<ptr<f32>> -> tensor<ptr<i32>>).
    // Accept both forms so the expanded integer atomics can be fused before
    // discrete-mask conversion.
    ptrSplatOp = op.getPtr().getDefiningOp<triton::SplatOp>();
    if (ptrSplatOp)
      ptrBitcastOp = ptrSplatOp.getSrc().getDefiningOp<triton::BitcastOp>();
  }
  auto valueBitcastOp = op.getVal().getDefiningOp<triton::BitcastOp>();
  if (!ptrBitcastOp || !valueBitcastOp) {
    return failure();
  }

  auto eraseDeadPointerCasts = [&]() {
    if (ptrSplatOp && ptrSplatOp->use_empty())
      rewriter.eraseOp(ptrSplatOp);
    if (ptrBitcastOp->use_empty())
      rewriter.eraseOp(ptrBitcastOp);
  };

  // We only need to handle the op when the element type is float
  auto elementType = getElementTypeOrSelf(valueBitcastOp.getSrc().getType());
  if (!isa<FloatType>(elementType)) {
    return failure();
  }

  auto rmwOp = op.getAtomicRmwOp();
  // here we know that atomic UMAX/UMIN
  // is created by special logic of triton right now
  // so we can simply delete it
  if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
    // if the return value of op is used, we can't simply erase it
    if (op.getResult().use_empty()) {
      rewriter.eraseOp(op);
      eraseDeadPointerCasts();
      return success();
    }
    return failure();
  }

  if (rmwOp != triton::RMWOp::MAX && rmwOp != triton::RMWOp::MIN) {
    return failure();
  }

  // 1. Though semantic interpreter will generate full true tensor as original
  // mask if atomicrmwOp don't have it, above float devision process will also
  // generate positive and negative comparison mask, which will cause to fold
  // true mask.
  // 2. While if atomicrmwOp has original mask, there exists andiop between
  // original mask and positive/negative comparison mask
  //
  // Here wanna extract original mask
  Value originalMask = op.getMask();
  auto createAllTrueMask = [&]() -> Value {
    Type maskType = op.getMask().getType();
    if (auto shapedMaskType = dyn_cast<ShapedType>(maskType)) {
      return rewriter.create<arith::ConstantOp>(
          op->getLoc(), DenseElementsAttr::get(shapedMaskType, true));
    }
    if (maskType.isInteger(1)) {
      return rewriter.create<arith::ConstantOp>(op->getLoc(),
                                                rewriter.getBoolAttr(true));
    }
    return {};
  };

  if (auto andOp = originalMask.getDefiningOp<arith::AndIOp>())
    // LHS is convention in semantic interpreter
    originalMask = andOp.getLhs();
  else if (auto xorOp = originalMask.getDefiningOp<arith::XOrIOp>()) {
    // Current f32 atomic_min uses !signbit as the positive mask:
    //   shrui(value_bits, 31) -> cmpi ne 0 -> xori true.
    if ((rmwOp != triton::RMWOp::MIN && rmwOp != triton::RMWOp::MAX) ||
        (!elementType.isF32() && !elementType.isF64()) ||
        !matchPattern(xorOp.getRhs(), m_One())) {
      return failure();
    }

    auto cmpOp = xorOp.getLhs().getDefiningOp<arith::CmpIOp>();
    if (!cmpOp || cmpOp.getPredicate() != arith::CmpIPredicate::ne ||
        !matchPattern(cmpOp.getRhs(), m_Zero()))
      return op->emitError("Illegal mask for atomicrmwOp of float type");

    auto shiftOp = cmpOp.getLhs().getDefiningOp<arith::ShRUIOp>();
    if (!shiftOp || shiftOp.getLhs() != valueBitcastOp.getResult())
      return op->emitError("Illegal mask for atomicrmwOp of float type");

    // Restore the implicit all-true mask.
    originalMask = createAllTrueMask();
    if (!originalMask)
      return failure();
  } else if (auto cmpOp = originalMask.getDefiningOp<arith::CmpIOp>()) {
    // Scalar floating-point atomic max/min represents !signbit as:
    //   shrui(value_bits, 31/63) -> cmpi eq 0.
    if (cmpOp.getPredicate() != arith::CmpIPredicate::eq ||
        !matchPattern(cmpOp.getRhs(), m_Zero()))
      return failure();

    auto shiftOp = cmpOp.getLhs().getDefiningOp<arith::ShRUIOp>();
    if (!shiftOp || shiftOp.getLhs() != valueBitcastOp.getResult())
      return failure();

    originalMask = createAllTrueMask();
    if (!originalMask)
      return failure();
  } else if (auto cmpOp = originalMask.getDefiningOp<arith::CmpFOp>()) {
    if (cmpOp.getPredicate() != mlir::arith::CmpFPredicate::OGE ||
        !matchPattern(cmpOp.getRhs(),
                      /*positive float zero matcher*/ m_PosZeroFloat()))
      // Here recheck frontend interpreter generation in no manual mask state
      return op->emitError("Illegal mask for atomicrmwOp of float type");
    // Restore original true mask
    originalMask = createAllTrueMask();
    if (!originalMask)
      return failure();
  } else
    return op->emitError("Illegal mask for atomicrmwOp of float type");

  Value originalPtr = ptrBitcastOp.getSrc();
  if (ptrSplatOp) {
    auto ptrTensorType = dyn_cast<RankedTensorType>(op.getPtr().getType());
    if (!ptrTensorType)
      return failure();
    auto originalPtrType = RankedTensorType::get(
        ptrTensorType.getShape(), ptrBitcastOp.getSrc().getType(),
        ptrTensorType.getEncoding());
    originalPtr = rewriter.create<triton::SplatOp>(op.getLoc(), originalPtrType,
                                                   ptrBitcastOp.getSrc());
  }

  auto originAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valueBitcastOp.getSrc().getType(), op.getAtomicRmwOp(),
      originalPtr, valueBitcastOp.getSrc(), originalMask, op.getSem(),
      op.getScope());

  // if the return value of op is used
  // we need to handle its usage
  // In semantic.py, if the atomic Max/Min with float input is used
  // It will use select + bitcast to get float value
  // so here we need to revert it too
  //
  // For example:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask1 :
  // (tensor<32x!tt.ptr<i32>>... %1 = tt.atomic_rmw umin, acq_rel, gpu, %gm,
  // %input, %mask2 : (tensor<32x!tt.ptr<i32>>... %2 = arith.select
  // %devidedMask, %0, %1 : tensor<32xi1>, tensor<32xi32> %3 = tt.bitcast %2 :
  // tensor<32xi32> -> tensor<32xf32> tt.store %outputMemref, %3 :
  // tensor<32x!tt.ptr<f32>>
  //
  // will be revert to:
  // %0 = tt.atomic_rmw max, acq_rel, gpu, %gm, %input, %mask :
  // (tensor<32x!tt.ptr<f32>>... tt.store %outputMemref, %0 :
  // tensor<32x!tt.ptr<f32>>
  //
  if (!op.getResult().use_empty()) {
    for (OpOperand &use : op->getUses()) {
      auto selectOp = dyn_cast<arith::SelectOp>(use.getOwner());
      if (!selectOp)
        continue;

      for (OpOperand &selectUse : selectOp->getUses()) {
        if (auto bitcastOp =
                dyn_cast<triton::BitcastOp>(selectUse.getOwner())) {
          bitcastOp.getResult().replaceAllUsesWith(originAtomicOp);
        }
      }
    }
    rewriter.replaceOp(op, originAtomicOp);
  } else {
    rewriter.eraseOp(op);
  }

  // The restored atomic uses the original floating-point GM pointer. Remove
  // the expanded pointer splat/bitcast once the paired integer atomic is gone.
  eraseDeadPointerCasts();

  return success();
}

StoreConverter::StoreConverter(MLIRContext *context)
    : OpConversionPattern<triton::StoreOp>(context) {}

LogicalResult
StoreConverter::matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {

  // triton store op basic
  auto mask = op.getMask();
  auto loc = op.getLoc();
  insertDebugNopForMask(mask, rewriter);
  auto ptr = adaptor.getPtr();
  auto val = adaptor.getValue();

  FailureOr<Value> resolvedPtr =
      resolveMemoryPointer(op.getPtr(), ptr, rewriter);
  if (failed(resolvedPtr))
    return rewriter.notifyMatchFailure(
        op, "unable to materialize the store pointer as a memref");
  ptr = *resolvedPtr;
  // 1. boundary size check
  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    auto makeTensorPtrOp = op.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    auto boundarySizes = mlir::ConverterUtils::getBoundarySizes(
        boundaryCheck, /*remapped*/ ptr, loc, rewriter);
    SmallVector<OpFoldResult> srcOffsets;
    SmallVector<OpFoldResult> dstOffsets(boundarySizes.size(),
                                         rewriter.getIndexAttr(0));
    if (makeTensorPtrOp) {
      auto zeroVal = rewriter.createOrFold<arith::ConstantOp>(
          loc, rewriter.getI32IntegerAttr(0));
      for (auto [idx, offVal] : llvm::enumerate(makeTensorPtrOp.getOffsets())) {
        if (llvm::find(boundaryCheck, idx) == boundaryCheck.end()) {
          srcOffsets.push_back(dstOffsets[idx]);
          continue;
        }
        Value offset =
            rewriter.createOrFold<arith::SubIOp>(loc, zeroVal, offVal);
        Value size =
            getValueOrCreateConstantIndexOp(rewriter, loc, boundarySizes[idx]);
        offset = rewriter.createOrFold<arith::MaxSIOp>(loc, offset, zeroVal);
        offset = rewriter.createOrFold<arith::IndexCastOp>(
            loc, rewriter.getIndexType(), offset);
        OpFoldResult ofr;
        if (auto constOp = offset.getDefiningOp<arith::ConstantOp>()) {
          ofr = constOp.getValue();
        } else {
          ofr = offset;
        }
        ofr = minOpFoldResult(ofr, size, loc, rewriter);
        boundarySizes[idx] = subOpFoldResult(size, ofr, loc, rewriter);
        srcOffsets.push_back(ofr);
      }
    } else {
      srcOffsets = dstOffsets;
    }
    auto srcSlice = mlir::ConverterUtils::makeExtractSliceOp(
        val, srcOffsets, boundarySizes, loc, rewriter);
    auto dstSubview = mlir::ConverterUtils::makeSubViewOp(
        ptr, dstOffsets, boundarySizes, loc, rewriter);
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, srcSlice, dstSubview);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }

  // 2. Simple load with no mask
  if (!mask) {
    auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
        loc, val, ptr);
    storeOp.setWritable(true);
    rewriter.eraseOp(op);
    return success();
  }

  // 3. Continuous masked stores.
  // Analyze the mask operand to determine at runtime the size of the data we
  // are moving.
  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);

  if (isContMask.failed()) {
    return failure();
  }
  LLVM_DEBUG({ llvm::dbgs() << *getModuleOpFromOperation(op) << "\n"; });
  auto srcSlice = mstate.getExtractSlice(val, loc, rewriter);
  auto dstSubview = mstate.getSubview(ptr, loc, rewriter);
  auto storeOp = rewriter.create<bufferization::MaterializeInDestinationOp>(
      loc, srcSlice, dstSubview);
  storeOp.setWritable(true);
  rewriter.eraseOp(op);
  return success();
}

bool ReinterpretCastStrideCanonicalizer::hasFixableZeroStride(
    memref::ReinterpretCastOp op) {
  auto staticSizes = op.getStaticSizes();
  auto staticStrides = op.getStaticStrides();
  auto dynamicStrides = op.getStrides();

  if (staticSizes.size() != staticStrides.size())
    return false;

  // now handle: size all static
  if (llvm::any_of(staticSizes, ShapedType::isDynamic))
    return false;

  unsigned dynStrideIdx = 0;
  for (unsigned i = 0; i < staticStrides.size(); ++i) {
    // now handle: dynamic stride 0 with static size 1
    if (!ShapedType::isDynamic(staticStrides[i]))
      continue;

    if (dynStrideIdx >= dynamicStrides.size())
      return false;

    Value st = dynamicStrides[dynStrideIdx];
    dynStrideIdx++;
    if (staticSizes[i] == 1 && mlir::isZero(OpFoldResult(st)))
      return true;
  }
  return false;
}

LogicalResult ReinterpretCastStrideCanonicalizer::matchAndRewrite(
    memref::ReinterpretCastOp op, PatternRewriter &rewriter) const {
  if (!hasFixableZeroStride(op))
    return failure();

  auto staticSizes = op.getStaticSizes();
  auto staticStrides = op.getStaticStrides();
  auto dynamicStrides = op.getStrides();

  SmallVector<Value> newDynamicStrides;
  newDynamicStrides.reserve(dynamicStrides.size());

  unsigned dynStrideIdx = 0;
  bool changed = false;
  Value c1 =
      rewriter.create<arith::ConstantOp>(op.getLoc(), rewriter.getIndexAttr(1));

  for (unsigned i = 0, e = staticStrides.size(); i < e; ++i) {
    if (!ShapedType::isDynamic(staticStrides[i]))
      continue;

    if (dynStrideIdx >= dynamicStrides.size())
      return failure();

    Value oldStride = dynamicStrides[dynStrideIdx];
    dynStrideIdx++;
    if (staticSizes[i] == 1 && mlir::isZero(OpFoldResult(oldStride))) {
      newDynamicStrides.push_back(c1);
      changed = true;
    } else {
      newDynamicStrides.push_back(oldStride);
    }
  }

  // all dynStride should be visited, and at least one should be changed
  if (dynStrideIdx != dynamicStrides.size())
    return failure();
  if (!changed)
    return failure();

  auto newReinterpretCast = rewriter.create<memref::ReinterpretCastOp>(
      op.getLoc(), cast<MemRefType>(op.getResult().getType()), op.getSource(),
      op.getOffsets(), op.getSizes(), newDynamicStrides, op.getStaticOffsets(),
      op.getStaticSizes(), op.getStaticStrides());

  rewriter.replaceOp(op, newReinterpretCast.getResult());

  return success();
}

} // namespace LoadStoreConverter
