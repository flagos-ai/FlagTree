#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, ArrayRef<Value> iterIdx,
    PatternRewriter &rewriter) const {
  if (!value)
    return value;
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, iterIdx);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
MemAccOpTy UnstructuredMemAccessConverter<MemAccOpTy>::createMemAccOp(
    MemAccOpTy op, Value ptrToAccess, Location loc, ArrayRef<Value> iterIdx,
    PatternRewriter &rewriter) const {
  llvm_unreachable("Unhandled discrete memory access operation");
}

template <>
triton::LoadOp UnstructuredMemAccessConverter<triton::LoadOp>::createMemAccOp(
    triton::LoadOp op, Value ptrToAccess, Location loc, ArrayRef<Value> iterIdx,
    PatternRewriter &rewriter) const {
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, op.getCache(),
                                         op.getEvict(), false);
}

template <>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    ArrayRef<Value> iterIdx, PatternRewriter &rewriter) const {
  auto extractedValue = createExtractOp(loc, op.getVal(), iterIdx, rewriter);
  auto extractedMask = createExtractOp(loc, op.getMask(), iterIdx, rewriter);
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
  auto scalarLikeType =
      RankedTensorType::get(scalarLikeShape, resultType.getElementType());
  auto splatedPtrToAccess = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
      ptrToAccess);
  auto splatedExtractedValue = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
      extractedValue);
  auto splatedExtractedMask = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
      extractedMask);
  return rewriter.create<triton::AtomicRMWOp>(
      loc, scalarLikeType, op.getAtomicRmwOpAttr(), splatedPtrToAccess,
      splatedExtractedValue, splatedExtractedMask, op.getSemAttr(),
      op.getScopeAttr());
}

template <>
triton::AtomicCASOp
UnstructuredMemAccessConverter<triton::AtomicCASOp>::createMemAccOp(
    triton::AtomicCASOp op, Value ptrToAccess, Location loc,
    ArrayRef<Value> iterIdx, PatternRewriter &rewriter) const {
  auto extractedCmp = createExtractOp(loc, op.getCmp(), iterIdx, rewriter);
  auto extractedValue = createExtractOp(loc, op.getVal(), iterIdx, rewriter);
  auto resultType = cast<RankedTensorType>(op.getResult().getType());
  SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
  auto scalarLikeType =
      RankedTensorType::get(scalarLikeShape, resultType.getElementType());
  auto splatedPtrToAccess = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
      ptrToAccess);
  auto splatedExtractedCmp = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
      extractedCmp);
  auto splatedExtractedValue = rewriter.create<triton::SplatOp>(
      loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
      extractedValue);
  return rewriter.create<triton::AtomicCASOp>(
      loc, scalarLikeType, splatedPtrToAccess, splatedExtractedCmp,
      splatedExtractedValue, op.getSemAttr(), op.getScopeAttr());
}

template <>
triton::StoreOp UnstructuredMemAccessConverter<triton::StoreOp>::createMemAccOp(
    triton::StoreOp op, Value ptrToAccess, Location loc,
    ArrayRef<Value> iterIdx, PatternRewriter &rewriter) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), iterIdx, rewriter);
  auto extractedMask = createExtractOp(loc, op.getMask(), iterIdx, rewriter);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask);
}

template <>
template <>
void UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, int rank,
                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  SmallVector<Value> idx(
      rank, rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
  auto extractedPtr = createExtractOp(loc, op.getPtr(), idx, rewriter);
  Value mask = op.getMask();
  Value other = op.getOther();
  Value loadedValue = rewriter.create<triton::LoadOp>(
      loc, extractedPtr, /*mask=*/nullptr, /*other=*/nullptr,
      /*boundaryCheck=*/ArrayRef<int32_t>(),
      /*PaddingOptionAttr=*/nullptr);
  loadedValue = rewriter.create<triton::SplatOp>(loc, op.getResult().getType(),
                                                 loadedValue);
  if (mask)
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue, other);
  else
    rewriter.replaceOp(op, loadedValue);
}

template <typename MemAccOpTy>
void UnstructuredMemAccessConverter<MemAccOpTy>::AddAssertForAddPtr(
  MemAccOpTy op, const Value &opoffset, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto opoffsetType = opoffset.getType();
  Value constantZero;

  op->setAttr("Negative", UnitAttr::get(rewriter.getContext()));
  if (auto tensorType = dyn_cast<RankedTensorType>(opoffsetType)) {
    constantZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(tensorType));
  } else {
    constantZero = rewriter.create<arith::ConstantIntOp>(
        loc, 0, opoffset.getType());
  }
  Value cmpResult = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sge, opoffset, constantZero);

  mlir::StringAttr assertMsg = rewriter.getStringAttr(
      "AddPtr offset (from subi) must be >= 0");

  rewriter.create<triton::AssertOp>(loc, cmpResult, assertMsg);
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap)
    : OpRewritePattern<MemAccOpTy>(context), offsetMap(offsetMap) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName)
      || op->hasAttr("Negative")) {
    return failure();
  }
  if (!offsetMap.contains(ptr))
    return op.emitError() << "PtrOffsetInfo should be computed\n" << ptr;

  auto ptrOffsetInfo = offsetMap.at(ptr);
  bool flag = false;
  if (ptrOffsetInfo.isNegativeFlag()) {
    flag = true;
  }

  if (ptrOffsetInfo.isStructured() &&
      (!ptrOffsetInfo.isScalarLike() ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; }))) {
    if (flag) {
      AddAssertForAddPtr(op, ptrOffsetInfo.getOffset(), rewriter);
      return success();
    } else {
      return failure();
    }
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    os << "isStructured = " << ptrOffsetInfo.isStructured() \
      << ",isScalarLike = " << ptrOffsetInfo.isScalarLike() \
      << ", isNegativeFlag = " << ptrOffsetInfo.isNegativeFlag() << "\n";
  });

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>)
    if (ptrOffsetInfo.isScalarLike()) {
     if (flag) {
        AddAssertForAddPtr(op, ptrOffsetInfo.getOffset(), rewriter);
      }
      splatAndLoadScenario(op, ptrOffsetInfo.getRank(), rewriter);
      return success();
    }

  if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    if (op->hasAttr(ConverterUtils::discreteMaskAttrName)) {
      auto selectOp = op.getValue().template getDefiningOp<arith::SelectOp>();
      op = rewriter.replaceOpWithNewOp<triton::StoreOp>(
          op, op.getPtr(), selectOp.getTrueValue(), selectOp.getCondition(),
          op.getCache(), op.getEvict());
      rewriter.setInsertionPoint(op);
    }
  }

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto offset = ptrOffsetInfo.getOffset();

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType =
      cast<triton::PointerType>(ptrType.getElementType()).getPointeeType();

  Value iterArg = nullptr;

  // Only load case
  if (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> dims(resultShape.size(), rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> strides;
  SmallVector<Value> iterIdx;

  SmallVector<int64_t> localMemStrides(1, 1);

  for (auto size : llvm::reverse(resultShape)) {
    localMemStrides.push_back(localMemStrides.back() * size);
  }
  localMemStrides.pop_back();

  std::reverse(localMemStrides.begin(), localMemStrides.end());
  bool isExtractedAttrInserted = false;
  for (const auto &[size, localMemStride] :
       llvm::zip_equal(resultShape, localMemStrides)) {
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(localMemStride));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    scf::ForOp forOp;
    if (isLoadLike) {
      forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx,
                                          ValueRange({iterArg}));
      if (!newOpResult) {
        newOpResult = forOp->getResult(0);
      } else {
        rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
      }
      iterArg = forOp.getRegionIterArg(0);
    } else {
      forOp = rewriter.create<scf::ForOp>(loc, zeroIdx, sizeVal, oneIdx);
    }
    offsets.push_back(forOp.getInductionVar());
    iterIdx.push_back(forOp.getInductionVar());
    forOp->setAttr("ExtractedLoadOrStore",
                   UnitAttr::get(rewriter.getContext()));
    rewriter.setInsertionPointToStart(forOp.getBody());
  }

  auto scalarLikeShape = SmallVector<int64_t>(dims.size(), 1);
  auto scalarLikeType =
      RankedTensorType::get(scalarLikeShape, resultElementType);

  auto extractedOffset = createExtractOp(loc, offset, iterIdx, rewriter);
  if (flag) {
    AddAssertForAddPtr(op, extractedOffset, rewriter);
  }
  if (isa<RankedTensorType>(srcPtr.getType())) {
    srcPtr = createExtractOp(loc, srcPtr, iterIdx, rewriter);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);

  MemAccOpTy accessedValue =
      createMemAccOp(op, ptrToAccess, loc, iterIdx, rewriter);
  accessedValue->setAttr(ConverterUtils::discreteAttrName,
                         UnitAttr::get(rewriter.getContext()));

  if (isLoadLike) {
    assert(iterArg && "Load case must have iterArg in for loop");

    Value splatedValue = accessedValue->getResult(0);
    if (!isa<RankedTensorType>(splatedValue.getType())) {
      splatedValue =
          rewriter.create<triton::SplatOp>(loc, scalarLikeType, splatedValue);
    }
    auto result = rewriter.create<tensor::InsertSliceOp>(
        loc, splatedValue, iterArg, offsets, dims, strides);
    rewriter.create<scf::YieldOp>(loc, result->getResult(0))
        ->setAttr(ConverterUtils::discreteAttrName,
                  UnitAttr::get(rewriter.getContext()));

    rewriter.restoreInsertionPoint(insertPoint);
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      if (op.getMask() && op.getOther()) {
        rewriter
            .replaceOpWithNewOp<arith::SelectOp>(op, op.getMask(), newOpResult,
                                                 op.getOther())
            ->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
        ;
      } else {
        rewriter.replaceOp(op, newOpResult);
      }
    } else {
      rewriter.replaceOp(op, newOpResult);
    }
  } else {
    rewriter.eraseOp(op);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

void exchangeValueWithOffset(Value value, Value ptr, const Location &loc,
                             RewriterBase &rewriter,
                             llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap) {
  RewriterBase::InsertionGuard guard(rewriter);
  if (auto blockArgument = dyn_cast<BlockArgument>(value)) {
    rewriter.setInsertionPointToStart(blockArgument.getOwner());
  } else {
    rewriter.setInsertionPointAfter(value.getDefiningOp());
  }
  auto tempVar = rewriter
                     .create<UnrealizedConversionCastOp>(loc, value.getType(),
                                                         ValueRange({}))
                     ->getResult(0);
  auto valueType = cast<RankedTensorType>(value.getType());
  auto offsetType =
      RankedTensorType::get(valueType.getShape(), rewriter.getIntegerType(64));
  rewriter.replaceAllUsesWith(value, tempVar);
  value.setType(offsetType);
  auto splatedPtr = rewriter.create<triton::SplatOp>(loc, valueType, ptr);
  auto newPtr =
      rewriter.create<triton::AddPtrOp>(loc, valueType, splatedPtr, value);
  parseAddPtr(newPtr, loc, rewriter, offsetMap);
  rewriter.replaceAllUsesWith(tempVar, newPtr);
}

void TritonToUnstructurePass::runPreparse(LoopLikeOpInterface op) {
  IRRewriter rewriter(&getContext());
  auto loopResults = op.getLoopResults();
  if (!loopResults)
    return;
  for (OpResult res : *loopResults) {
    if (auto tensorType = dyn_cast<RankedTensorType>(res.getType())) {
      auto loc = op.getLoc();
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
      });
      parse(res, loc, rewriter, offsetMapForLoopArgs);

      BlockArgument regionIterArg;
      auto resOffsetInfo = offsetMapForLoopArgs.at(res);
      if (!resOffsetInfo.isStructured() &&
          isa<triton::PointerType>(tensorType.getElementType())) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << "Handling special case\n" << op << '\n';
        });
        // Get initArg
        OpOperand *initArg = op.getTiedLoopInit(res);
        PtrOffsetInfo initOffsetInfo = offsetMapForLoopArgs.at(initArg->get());
        // Get regionIterArg
        regionIterArg = op.getTiedLoopRegionIterArg(res);
        PtrOffsetInfo regionIterArgOffsetInfo =
            offsetMapForLoopArgs.at(regionIterArg);
        // Get yield
        OpOperand *yieldedValue = op.getTiedLoopYieldedValue(regionIterArg);
        PtrOffsetInfo yieldOffsetInfo =
            offsetMapForLoopArgs.at(yieldedValue->get());
        // Exchange iter arg with offset
        exchangeValueWithOffset(regionIterArg, initOffsetInfo.getPtr(), loc,
                                rewriter, offsetMapForLoopArgs);
        rewriter.replaceAllUsesWith(regionIterArgOffsetInfo.getOffset(),
                                    regionIterArg);
        yieldedValue->set(yieldOffsetInfo.getOffset());
        initArg->set(initOffsetInfo.getOffset());
        exchangeValueWithOffset(res, initOffsetInfo.getPtr(), loc, rewriter,
                                offsetMapForLoopArgs);
      }

      regionIterArg = op.getTiedLoopRegionIterArg(res);
      offsetMap[regionIterArg] = PtrOffsetInfo(resOffsetInfo.getPtr());
      SmallVector<bool> &regionIterArgOffset =
          offsetMap[regionIterArg].getStructuredRef();
      SmallVector<bool> &resOffset =
          resOffsetInfo.getStructuredRef();
      regionIterArgOffset.resize(resOffset.size());
      for (size_t i = 0; i < resOffset.size(); i++)
        regionIterArgOffset[i] = resOffset[i];
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing result of\n" << regionIterArg << "\nis ";
        for (size_t i = 0; i < regionIterArgOffset.size(); i++)
          os << regionIterArgOffset[i];
        os << '\n';
      });
    }
  }
}

template <typename MemAccOpTy, typename>
void TritonToUnstructurePass::runParse(MemAccOpTy op) {
  IRRewriter rewriter(&getContext());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  parse(op.getPtr(), op.getLoc(), rewriter, offsetMap);
}

void TritonToUnstructurePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  moduleOp->walk([this](LoopLikeOpInterface op) { runPreparse(op); });
  moduleOp->walk([this](Operation *op) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      runParse(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      runParse(storeOp);
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      runParse(atomicRMWOp);
    } else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op)) {
      runParse(atomicCASOp);
    }
  });

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(ctx,
                                                                    offsetMap);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructurePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToUnstructurePass() {
  return std::make_unique<TritonToUnstructurePass>();
}