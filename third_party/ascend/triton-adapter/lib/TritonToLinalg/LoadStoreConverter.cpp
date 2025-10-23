//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "TritonToLinalg/LoadStoreConverter.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

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

const std::string MayImplicitTransposeWithLastAxisTAG = "MayImplicitTransposeWithLastAxis";

LogicalResult
AddPtrConverter::matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                 ConversionPatternRewriter &rewriter) const {
  llvm::SmallDenseMap<Value, BlockData> known;
  BlockDataParser::rewriteAddPtr(op, adaptor, rewriter, known);
  return success();
}

LogicalResult LoadConverter::toTensorAndReplace(
    triton::LoadOp &op, RankedTensorType &tensorType, memref::AllocOp &allocOp,
    bool mayImplicitTransposeWithLastAxis, const Location &loc, ConversionPatternRewriter &rewriter) const {
  
  Value loadedTensor = rewriter.create<bufferization::ToTensorOp>(
      loc, tensorType, allocOp, true, true);
  if(mayImplicitTransposeWithLastAxis){
    auto markOp = rewriter.create<annotation::MarkOp>(loc, loadedTensor);
    markOp->setAttr(MayImplicitTransposeWithLastAxisTAG, UnitAttr::get(rewriter.getContext()));
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
    Value other, memref::AllocOp localMem, ArrayRef<OpFoldResult> maskDim,
    ConversionPatternRewriter &rewriter) const {
  auto loc = localMem->getLoc();
  MemRefType originalType = localMem.getType();
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
          loc, cast<IntegerAttr>(maskDim[i].get<Attribute>()));
    else
      maskDimVal = maskDim[i].get<Value>();

    auto curCmp = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 maskDimVal, shapeVal);

    fillFlag = rewriter.create<arith::OrIOp>(loc, fillFlag, curCmp.getResult())
                   .getResult();
  }

  rewriter.create<scf::IfOp>(
      loc, fillFlag, [&](OpBuilder &builder, Location loc) {
        builder.create<linalg::FillOp>(loc, ValueRange{other},
                                       ValueRange{localMem});
        builder.create<scf::YieldOp>(loc);
      });
}

LoadConverter::LoadConverter(MLIRContext *context)
    : OpConversionPattern<triton::LoadOp>(context) {}

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

  // handling scalar
  if (!isa<ShapedType>(op.getResult().getType())) {
    auto scalarMemref =
        BlockDataParser::getScalarMemRef(op.getPtr(), ptr, loc, rewriter);
    auto resTy = op.getResult().getType();
    auto idxZero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
    auto allocOp = rewriter.create<memref::AllocOp>(loc, MemRefType::get({1}, resTy));
    auto srcSubView = mlir::ConverterUtils::makeSubViewOp(ptr, {rewriter.getIndexAttr(1)}, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcSubView, allocOp);
    auto loadedTensor =  rewriter.create<bufferization::ToTensorOp>(
           loc, RankedTensorType::get({1}, resTy), allocOp, true /* restrict */, true /* writable */);
    auto loadedValue = rewriter.create<tensor::ExtractOp>(loc, resTy, loadedTensor, ValueRange({idxZero})).getResult();
    if (mask && other) {
      mask = rewriter.create<triton::SplatOp>(loc, RankedTensorType::get({1}, mask.getType()), mask);
      loadedValue = rewriter.create<triton::SplatOp>(loc, RankedTensorType::get({1}, loadedValue.getType()), loadedValue);
      other = rewriter.create<triton::SplatOp>(loc, RankedTensorType::get({1}, other.getType()), other);
      loadedValue = rewriter.create<arith::SelectOp>(loc, mask, loadedValue, other);
      rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, loadedValue, ValueRange({idxZero}));
    } else {
      rewriter.replaceOp(op, loadedValue);
    }
    return success();
  }

  int64_t lastStride=-1;
  if (isa<BlockArgument>(ptr)) {
    auto u = ptr;
    while (auto blkArg = dyn_cast<BlockArgument>(u)) {
      if (auto forOp = dyn_cast<scf::ForOp>(blkArg.getOwner()->getParentOp())) {
        auto prt = forOp->getOperand(3+blkArg.getArgNumber()-1);
        u = prt;
      } else {
        u=nullptr;
        break;
      }
    }
    if (u && isa<memref::ReinterpretCastOp>(u.getDefiningOp())) {
      auto ret = mlir::ConverterUtils::getLastStrideOfReinterpretCastOp(dyn_cast<memref::ReinterpretCastOp>(u.getDefiningOp()));
      if (ret.has_value()) lastStride = *ret;
    }
  }

  // handling no mask
  auto memRefType = dyn_cast<MemRefType>(ptr.getType());
  if (!memRefType) {
    return rewriter.notifyMatchFailure(
        op, "LoadOp expects a memref, not a memref of pointers");
  }
  bool mayImplicitTransposeWithLastAxis = (existDotFlag) && (!op->hasAttr(ConverterUtils::GeneratedByMakeTensorPtrTAG)) &&
    (lastStride != 1 && mlir::ConverterUtils::isaPermutedMemRefType(memRefType));
  auto memRefShape = memRefType.getShape();
  auto memRefElementType = memRefType.getElementType();

  auto allocOp = rewriter.create<memref::AllocOp>(
      loc, MemRefType::get(memRefShape, memRefElementType));

  auto tensorType = RankedTensorType::get(memRefShape, memRefElementType);
  // boundary check
  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    auto boundarySizes = mlir::ConverterUtils::getBoundarySizes(
        boundaryCheck, /*remapped*/ ptr, loc, rewriter);
    // handle the padding
    auto padding = op.getPadding();
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

    auto srcSubView =
        mlir::ConverterUtils::makeSubViewOp(ptr, boundarySizes, loc, rewriter);
    auto dstSubview = mlir::ConverterUtils::makeSubViewOp(
        allocOp, boundarySizes, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcSubView, dstSubview);
    if (mayImplicitTransposeWithLastAxis) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, dstSubview);
      markOp->setAttr(MayImplicitTransposeWithLastAxisTAG, UnitAttr::get(rewriter.getContext()));
    }
    return this->toTensorAndReplace(op, tensorType, allocOp, mayImplicitTransposeWithLastAxis, loc, rewriter);
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
      auto [ptrStrides, ptrOffsets] = getStridesAndOffset(memRefType);
      if (ptrStrides.back() == 2 && (memRefShape.back() % 2 == 0) &&
          mlir::triton::DeinterleaveStatusOptimization(op, adaptor, rewriter)
              .succeeded()) {
        return success();
      }
      rewriter.create<memref::CopyOp>(loc, ptr, allocOp);
      if (mayImplicitTransposeWithLastAxis) {
        auto markOp = rewriter.create<annotation::MarkOp>(loc, allocOp);
        markOp->setAttr(MayImplicitTransposeWithLastAxisTAG, UnitAttr::get(rewriter.getContext()));
      }
    }

    return this->toTensorAndReplace(op, tensorType, allocOp, mayImplicitTransposeWithLastAxis, loc, rewriter);
  }

  MaskState mstate;
  auto isContMask = mstate.parse(mask, loc, rewriter);
  if (isContMask.failed()) {
    return rewriter.notifyMatchFailure(
        op, "can not lower uncontinuout masked loads");
  }

  if (other) {
    auto scalarOther =
        mlir::ConverterUtils::getScalarValue(other, loc, rewriter);
    assert(
        scalarOther &&
        "other value used in masked load produced by unsupported instruction!");

    fillTensorWithOtherForMaskScenario(scalarOther, allocOp, mstate.dims,
                                       rewriter);
  }

  // To enable deinterleave optimization with mask load, mask state along last
  // dimension couldn't be split, which means `dims.back()` must be equal to
  // origin type last dimension constant size and `offsets.back()` must be 0.
  //
  // The basis is that last dimension range comparison would generate
  // unaccepted discontinuous mask.
  if (mstate.getRank() == memRefType.getRank() &&
      isConstantIntValue(mstate.offsets.back(), 0) &&
      isConstantIntValue(mstate.dims.back(), memRefType.getShape().back())) {
    auto [ptrStrides, ptrOffsets] = getStridesAndOffset(memRefType);
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
    memref::SubViewOp srcSubView = mstate.getSubview(ptr, loc, rewriter);
    memref::SubViewOp dstSubView = mstate.getSubview(allocOp, loc, rewriter);
    rewriter.create<memref::CopyOp>(loc, srcSubView, dstSubView);
    if (mayImplicitTransposeWithLastAxis) {
      auto markOp = rewriter.create<annotation::MarkOp>(loc, dstSubView);
      markOp->setAttr(MayImplicitTransposeWithLastAxisTAG, UnitAttr::get(rewriter.getContext()));
    }
  }
  return this->toTensorAndReplace(op, tensorType, allocOp, mayImplicitTransposeWithLastAxis, loc, rewriter);
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
  // If the result of AtomicRMWOp is not used, we don't need to load the old
  // data stored at the ptr
  auto ptr = adaptor.getPtr();
  auto val = op.getVal();
  auto loc = op.getLoc();

  auto resType = dyn_cast<TensorType>(op.getResult().getType());
  if (!resType) {
    return rewriter.notifyMatchFailure(
        op, "atomicRMWConverter: scalar will be handled by "
            "ScalarAtomicRMWCanonicalizer");
  }

  auto rmwOp = op.getAtomicRmwOp();
  if (rmwOp == triton::RMWOp::UMAX || rmwOp == triton::RMWOp::UMIN) {
    return rewriter.notifyMatchFailure(
        op, "AtomicRMWConverter: unsupported atomic kind for now");
  }

  // 1. Simple case where no mask is used.
  auto type = dyn_cast<MemRefType>(ptr.getType());
  if (!type) {
    // Seen when implicit broadcasting is done late in a chain of
    // operations. The workaround is to broadcast the pointers early in the
    // address calculation. A proper fix is complicated, but at least we can
    // provide a better error message.
    return rewriter.notifyMatchFailure(
        op, "AtomicRMWOp expects a memref, not a memref of pointers");
  }

  auto dstMemref = ptr;
  // Well, linalg structure op wouldn't support mixed tensor/buffer semantics
  // any more in latest LLVM(triton LLVM dependency has involed this), so we
  // need to convert tensor to buffer early.
  auto dstOriType = cast<MemRefType>(dstMemref.getType());
  MemRefType dstType = MemRefType::get(dstOriType.getShape(), dstOriType.getElementType());
  Value inputMemref =
      rewriter.create<bufferization::ToMemrefOp>(loc, dstType, val);

  // 2. handle the mask for the atomic op
  // When the dsl do not pass the mask to this op like
  // `tl.atomic_add(out_ptr0 + xindex, tmp2)`, it will create a constant mask
  // for this op by default, which is not supported by maskAnalysis, so we
  // need to handle this situation
  //
  // This logic come from semantic.py:
  //
  // if not mask:
  //     mask_ir = builder.get_int1(True)
  //     mask_ty = tl.int1
  //     if ptr.type.is_block():
  //         mask_ir = \
  //             builder.create_splat(mask_ir, ptr.type.get_block_shapes())
  //         mask_ty = tl.block_type(tl.int1, ptr.type.get_block_shapes())
  //     mask = tl.tensor(mask_ir, mask_ty)
  //
  // ...
  //
  // return ptr, val, mask
  //
  if (auto mask = op.getMask()) {
    MaskState mstate;
    auto constantMask = mask.getDefiningOp<arith::ConstantOp>();
    if (!constantMask) {
      auto isContMask = mstate.parse(mask, loc, rewriter);

      if (isContMask.failed()) {
        return rewriter.notifyMatchFailure(
            op, "Cannot lower continuous masked loads");
      }
      dstMemref = mstate.getSubview(ptr, loc, rewriter);
      inputMemref = mstate.getSubview(inputMemref, loc, rewriter);
    } else {
      if (!isConstantMaskTrue(mask)) {
        rewriter.eraseOp(op);
        return success();
      }
    }
  }

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
  // Thus, there are a total of three inputs and outputs.
  // so here we have 3 map to create
  for (int i = 0; i < 3; i++) {
    indexingMaps.push_back(AffineMap::get(rank, 0, inputDims, context));
  }

  auto linalgOp = rewriter.create<linalg::GenericOp>(
      loc, /* operands */ ValueRange{dstMemref, inputMemref},
      ValueRange{dstMemref}, indexingMaps,
      mlir::ConverterUtils::getNParallelLoopsAttrs(rank),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange blockArgs) {
        Value opResult = createAtomicBinaryOps(nestedBuilder, nestedLoc, op,
                                               type.getElementType(),
                                               blockArgs[0], blockArgs[1]);
        nestedBuilder.create<linalg::YieldOp>(nestedLoc, opResult);
      });

  // "library_call"
  // indicating the actual semantic of this op
  // TODO: If the hardware support the MemSemantic/MemSyncScope
  //       We pass them down
  //       otherwise they need to be deleted
  const StringRef genericAtomicRMW = "GenericAtomicRMW";
  const StringRef memSemantic = "MemSemantic";
  const StringRef memSyncScope = "MemSyncScope";
  linalgOp->setAttr(genericAtomicRMW,
                    rewriter.getStringAttr(stringifyEnum(op.getAtomicRmwOp())));
  linalgOp->setAttr(memSemantic,
                    rewriter.getStringAttr(stringifyEnum(op.getSem())));
  linalgOp->setAttr(memSyncScope,
                    rewriter.getStringAttr(stringifyEnum(op.getScope())));

  // Mark atomic_and/or/xor specially which need software simulation in terms
  // of backend restriction
  if (softwareAtomicKinds.contains(op.getAtomicRmwOp()))
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
  // TODO: decoupling the logic of load, put it in the Utils
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
  } else {
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
  MemRefType dstType = MemRefType::get(dstOriType.getShape(), dstOriType.getElementType());
  Value inputMemref =
      rewriter.create<bufferization::ToMemrefOp>(loc, dstType, val);

  Value cmpMemref =
      rewriter.create<bufferization::ToMemrefOp>(loc, dstType, cmp);

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
  } else {
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
    rewriter.replaceOpWithNewOp<scf::IfOp>(op, mask,
      [&](OpBuilder &b, Location loc) {
        b.create<triton::StoreOp>(
          loc, ptr, value, op.getCache(), op.getEvict());
        b.create<scf::YieldOp>(loc);
    });
    return success();
  }

  auto ptrTy = RankedTensorType::get({(int64_t)1}, ptr.getType());
  auto ptrSplat = rewriter.create<triton::SplatOp>(op.getLoc(), ptrTy, ptr);
  auto valTy = RankedTensorType::get({(int64_t)1}, value.getType());
  auto valSplat =
      rewriter.create<triton::SplatOp>(op.getLoc(), valTy, value);
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
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp, ValueRange({idxZero}));
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
  rewriter.replaceOpWithNewOp<tensor::ExtractOp>(op, newAtomicOp, ValueRange({idxZero}));
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
  auto valueBitcastOp = op.getVal().getDefiningOp<triton::BitcastOp>();
  if (!ptrBitcastOp || !valueBitcastOp) {
    return failure();
  }

  // We only need to handle the op when the element type is float
  auto elementType =
      dyn_cast<TensorType>(valueBitcastOp.getSrc().getType()).getElementType();
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
  if (auto andOp = originalMask.getDefiningOp<arith::AndIOp>())
    // LHS is convention in semantic interpreter
    originalMask = andOp.getLhs();
  else if (auto cmpOp = originalMask.getDefiningOp<arith::CmpFOp>()) {
    if (cmpOp.getPredicate() != mlir::arith::CmpFPredicate::OGE ||
        !matchPattern(cmpOp.getRhs(),
                      /*positive float zero matcher*/ m_PosZeroFloat()))
      // Here recheck frontend interpreter generation in no manual mask state
      return op->emitError("Illegal mask for atomicrmwOp of float type");
    // Restore original true mask
    originalMask = rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        /*typed attr*/ DenseElementsAttr::get(
            cast<ShapedType>(originalMask.getType()), true));
  } else
    return op->emitError("Illegal mask for atomicrmwOp of float type");

  auto originAtomicOp = rewriter.create<triton::AtomicRMWOp>(
      op.getLoc(), valueBitcastOp.getSrc().getType(), op.getAtomicRmwOp(),
      ptrBitcastOp.getSrc(), valueBitcastOp.getSrc(), originalMask, op.getSem(),
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
  auto ptr = adaptor.getPtr();
  auto val = adaptor.getValue();

  // 1. boundary size check
  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
      auto boundarySizes = mlir::ConverterUtils::getBoundarySizes(
        boundaryCheck, /*remapped*/ ptr, loc, rewriter);
      auto srcSlice = mlir::ConverterUtils::makeExtractSliceOp(val, boundarySizes, loc, rewriter);
      auto dstSubview = mlir::ConverterUtils::makeSubViewOp(ptr, boundarySizes, loc, rewriter);
      auto storeOp =
      rewriter.create<bufferization::MaterializeInDestinationOp>(
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
} // namespace LoadStoreConverter