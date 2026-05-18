#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSAGPUToLLVM/Allocation.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToMTGPU/GPUToMTGPUPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include <algorithm>
#include <optional>

#include "PatternTritonGPUOpToLLVM.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONMUSAGPUTOLLVM
#include "TritonMUSAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using mlir::triton::MUSA::TargetInfo;
namespace {

enum class InplaceLoadDataKind {
  Unsupported,
  Integer,
  Float,
};

static StringRef getInplaceLoadIntrinsicName(InplaceLoadDataKind dataKind) {
  switch (dataKind) {
  case InplaceLoadDataKind::Integer:
    return "llvm.musa.lsu.ld.cache.hint.i";
  case InplaceLoadDataKind::Float:
    return "llvm.musa.lsu.ld.cache.hint.f";
  default:
    return {};
  }
}

static SmallVector<Value>
buildInplaceLoadCacheHintOperands(Value ptr, Location loc,
                                  PatternRewriter &rewriter) {
  Type i32Ty = rewriter.getI32Type();
  Value innerPersist = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                rewriter.getI32IntegerAttr(0));
  Value outerPersist = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                rewriter.getI32IntegerAttr(2));
  Value chrnt = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                         rewriter.getI32IntegerAttr(1));
  Value slc = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                       rewriter.getI32IntegerAttr(1));
  return {ptr, innerPersist, outerPersist, chrnt, slc};
}

static std::optional<unsigned> getTypeBitWidth(Type type) {
  if (auto vecTy = dyn_cast<VectorType>(type)) {
    Type elemTy = vecTy.getElementType();
    if (elemTy.isIntOrFloat())
      return vecTy.getNumElements() * elemTy.getIntOrFloatBitWidth();
    return std::nullopt;
  }
  if (type.isIntOrFloat())
    return type.getIntOrFloatBitWidth();
  return std::nullopt;
}

static InplaceLoadDataKind getInplaceLoadDataKind(Type type) {
  Type elemTy = type;
  if (auto vecTy = dyn_cast<VectorType>(type))
    elemTy = vecTy.getElementType();

  if (elemTy.isIntOrIndex())
    return InplaceLoadDataKind::Integer;
  if (isa<FloatType>(elemTy))
    return InplaceLoadDataKind::Float;
  return InplaceLoadDataKind::Unsupported;
}

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<cf::ControlFlowDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<triton::musa::MUSADialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

class PredicatedCallOpConversion : public RewritePattern {
public:
  explicit PredicatedCallOpConversion(MLIRContext *context,
                                      int32_t computeCapability)
      : RewritePattern(LLVM::CallOp::getOperationName(), 1, context),
        computeCapability(computeCapability) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto callOp = dyn_cast<LLVM::CallOp>(op);
    if (!callOp || !callOp.getCallee())
      return failure();
    auto callee = callOp.getCallee().value();
    if (callee.contains(LLVM::MUSA::Predicated_Load))
      return rewritePredicatedLoad(callOp, rewriter, /*useCacheHint=*/false);
    if (callee.contains(LLVM::MUSA::Predicated_InplaceLoad))
      return rewritePredicatedLoad(callOp, rewriter,
                                   /*useCacheHint=*/computeCapability == 31);
    if (callee.contains(LLVM::MUSA::Predicated_Store))
      return rewritePredicatedStore(callOp, rewriter);
    return failure();
  }

private:
  static LogicalResult rewritePredicatedStore(LLVM::CallOp callOp,
                                              PatternRewriter &rewriter) {
    Location loc = callOp.getLoc();
    auto operands = callOp.getOperands();
    if (operands.size() != 3)
      return failure();
    Value ptr = operands[0];
    Value val = operands[1];
    Value pred = operands[2];

    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterStore =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    Block *trueBlock = rewriter.createBlock(afterStore);

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, pred, trueBlock, afterStore);

    rewriter.setInsertionPointToStart(trueBlock);
    LLVM::StoreOp::create(rewriter, loc, val, ptr);
    LLVM::BrOp::create(rewriter, loc, afterStore);

    rewriter.setInsertionPointToStart(afterStore);
    rewriter.eraseOp(callOp);
    return success();
  }

  static LogicalResult rewritePredicatedLoad(LLVM::CallOp callOp,
                                             PatternRewriter &rewriter,
                                             bool useCacheHint) {
    Location loc = callOp.getLoc();
    auto operands = callOp.getOperands();
    if (operands.size() != 3)
      return failure();
    Value ptr = operands[0];
    Value pred = operands[1];
    Value falseVal = operands[2];

    Type elemTy = callOp.getResult().getType();
    Block *currentBlock = rewriter.getInsertionBlock();
    Block *afterLoad =
        rewriter.splitBlock(currentBlock, rewriter.getInsertionPoint());
    afterLoad->addArgument(elemTy, loc);
    Block *trueBlock = rewriter.createBlock(afterLoad);
    Block *falseBlock =
        rewriter.splitBlock(trueBlock, rewriter.getInsertionPoint());

    rewriter.setInsertionPointToEnd(currentBlock);
    LLVM::CondBrOp::create(rewriter, loc, pred, trueBlock, falseBlock);

    rewriter.setInsertionPointToStart(trueBlock);
    Value loaded;
    std::optional<unsigned> typeBits = getTypeBitWidth(elemTy);
    InplaceLoadDataKind dataKind = getInplaceLoadDataKind(elemTy);
    StringRef intrinsic = getInplaceLoadIntrinsicName(dataKind);
    if (useCacheHint && typeBits && *typeBits == 128 && !intrinsic.empty()) {
      SmallVector<Value> operands =
          buildInplaceLoadCacheHintOperands(ptr, loc, rewriter);
      loaded = LLVM::createLLVMIntrinsicCallOp(rewriter, loc, intrinsic,
                                               TypeRange{elemTy}, operands)
                   .getResult(0);
    } else {
      loaded = LLVM::LoadOp::create(rewriter, loc, elemTy, ptr);
    }
    LLVM::BrOp::create(rewriter, loc, loaded, afterLoad);

    rewriter.setInsertionPointToStart(falseBlock);
    LLVM::BrOp::create(rewriter, loc, falseVal, afterLoad);

    rewriter.setInsertionPointToStart(afterLoad);
    rewriter.replaceOp(callOp, afterLoad->getArgument(0));
    return success();
  }

  int32_t computeCapability;
};

class CancelRedundantBFloatRoundTripPattern
    : public OpRewritePattern<LLVM::CallIntrinsicOp> {
public:
  using OpRewritePattern<LLVM::CallIntrinsicOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::CallIntrinsicOp call,
                                PatternRewriter &rewriter) const override {
    if (call.getIntrin() != "llvm.musa.bfloat162float")
      return failure();
    if (call.getArgs().size() != 1)
      return failure();

    auto producer = call.getArgs()[0].getDefiningOp<LLVM::CallIntrinsicOp>();
    if (!producer || producer.getIntrin() != "llvm.musa.float2bfloat16")
      return failure();
    if (producer.getArgs().size() != 1 || !producer->hasOneUse())
      return failure();

    rewriter.replaceOp(call, producer.getArgs()[0]);
    rewriter.eraseOp(producer);
    return success();
  }
};

std::optional<int64_t>
inferElemBytesFromMemDesc(triton::gpu::MemDescType type) {
  int bitWidth = type.getElementTypeBitWidth();
  if (bitWidth <= 0)
    return std::nullopt;
  return static_cast<int64_t>((bitWidth + 7) / 8);
}

bool inferRowMajorFromMemDesc(triton::gpu::MemDescType type) {
  auto order = triton::gpu::getOrder(type);
  if (order.empty())
    return true;
  return order.front() + 1 == type.getShape().size();
}

unsigned getSqmmaSwizzleAlignment(ModuleOp mod) {
  // Shared memory alignment should satisfy all SQMMA swizzle requirements.
  unsigned maxAlignment = 256;
  mod.walk([&](triton::gpu::LocalAllocOp localAllocOp) {
    auto maybeOpIdx = triton::musa::getSqmmaOpIdx(localAllocOp.getOperation());
    if (!maybeOpIdx)
      return;

    auto memDescTy = cast<triton::gpu::MemDescType>(localAllocOp.getType());
    auto maybeElemBytes =
        triton::musa::getSqmmaElemBytes(localAllocOp.getOperation());
    if (!maybeElemBytes)
      maybeElemBytes = inferElemBytesFromMemDesc(memDescTy);
    if (!maybeElemBytes || *maybeElemBytes <= 0)
      return;

    bool isRowMajor = triton::musa::getSqmmaRowMajor(
        localAllocOp.getOperation(), inferRowMajorFromMemDesc(memDescTy));

    int64_t opIdx = *maybeOpIdx;
    bool isMNMajor =
        ((opIdx == 0) && !isRowMajor) || ((opIdx == 1) && isRowMajor);
    unsigned sg = 16;
    if (*maybeElemBytes == 2)
      sg = isMNMajor ? 32 : 16;
    else if (*maybeElemBytes == 4)
      sg = isMNMajor ? 64 : 16;

    unsigned alignment = 256 * (256 / sg);
    maxAlignment = std::max(maxAlignment, alignment);
  });
  return maxAlignment;
}

struct ConvertTritonMUSAGPUToLLVM
    : public triton::impl::ConvertTritonMUSAGPUToLLVMBase<
          ConvertTritonMUSAGPUToLLVM> {
  using ConvertTritonMUSAGPUToLLVMBase::ConvertTritonMUSAGPUToLLVMBase;

  ConvertTritonMUSAGPUToLLVM() = default;
  ConvertTritonMUSAGPUToLLVM(int32_t computeCapability)
      : ConvertTritonMUSAGPUToLLVMBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability);

    size_t numOps = 0;
    size_t numConvertLayoutOps = 0;
    mod.walk([&](Operation *op) {
      numOps++;
      if (isa<triton::gpu::ConvertLayoutOp>(op))
        numConvertLayoutOps++;
    });
    bool isLargeKernel = numOps > 500 || numConvertLayoutOps > 10;

    auto groupedTMELoadWalk = mod.walk(
        [&](triton::musa::AsyncTMECopyGlobalToLocalOp op) -> WalkResult {
          if (failed(
                  triton::musa::recoverAndVerifyGroupedTMELoadConsumerContract(
                      op)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (groupedTMELoadWalk.wasInterrupted())
      return signalPassFailure();

    ModuleAllocation allocation(
        mod, mlir::triton::musa_gpu::getMusaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    ModuleMembarAnalysis membarPass(&allocation);
    membarPass.run();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    typeConverter.addConversion(
        [&](triton::mtgpu::SqmmaAccumulatorType type) -> std::optional<Type> {
          auto info = LLVM::MUSA::getSqmmaAccumulatorCarrierInfo(type);
          if (failed(info))
            return std::nullopt;
          return info->carrierType;
        });

    TritonLLVMFunctionConversionTarget funcTarget(*context);
    RewritePatternSet funcPatterns(context);
    mlir::triton::populateFuncOpConversionPattern(
        typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
    if (failed(
            applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
      return signalPassFailure();

    initSharedMemory(typeConverter, targetInfo);
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    mlir::triton::MUSA::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);
    mlir::triton::MUSA::populateDotOpToLLVMPatterns(typeConverter, patterns,
                                                    benefit);
    mlir::triton::MUSA::populateMUSAOpsToLLVMPatterns(typeConverter, patterns,
                                                      benefit);
    mlir::triton::MUSA::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, computeCapability,
        targetInfo, benefit);
    mlir::triton::MUSA::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, computeCapability, patterns,
        axisInfoAnalysis, benefit);
    mlir::triton::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateGatherOpToLLVMPatterns(typeConverter, patterns,
                                                 targetInfo, benefit);
    mlir::triton::MUSA::populateBarrierOpToLLVMPatterns(typeConverter, patterns,
                                                        benefit, targetInfo);
    mlir::triton::MUSA::populateTensorPtrOpsToLLVMPatterns(typeConverter,
                                                           patterns, benefit);
    mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);
    mlir::triton::MUSA::populateSPMDOpToLLVMPatterns(typeConverter, patterns,
                                                     benefit);
    mlir::triton::MUSA::populateThreadIdOpToLLVMPattern(typeConverter, patterns,
                                                        benefit);
    mlir::triton::MUSA::populateWarpIdOpToLLVMPattern(typeConverter, patterns,
                                                      benefit);
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    // Native lowering patterns.
    mlir::populateGpuToMTGPUConversionPatterns(typeConverter, patterns);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::populateInstrumentationToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    RewritePatternSet cleanupPatterns(context);
    cleanupPatterns.add<CancelRedundantBFloatRoundTripPattern>(context);

    GreedyRewriteConfig cleanupConfig;
    if (isLargeKernel) {
      cleanupConfig.setMaxIterations(3)
          .setMaxNumRewrites(1000)
          .setStrictness(GreedyRewriteStrictness::ExistingOps)
          .setUseTopDownTraversal(true)
          .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    }

    if (failed(applyPatternsGreedily(mod, std::move(cleanupPatterns),
                                     cleanupConfig)))
      return signalPassFailure();

    RewritePatternSet predicatedPatterns(context);
    predicatedPatterns.add<PredicatedCallOpConversion>(context,
                                                       computeCapability);

    GreedyRewriteConfig predicatedConfig;
    if (isLargeKernel) {
      predicatedConfig.setMaxIterations(3)
          .setMaxNumRewrites(1000)
          .setStrictness(GreedyRewriteStrictness::ExistingOps)
          .setUseTopDownTraversal(true)
          .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Disabled);
    }

    if (failed(applyPatternsGreedily(mod, std::move(predicatedPatterns),
                                     predicatedConfig)))
      return signalPassFailure();

    TritonLLVMFunctionConversionTarget cfTarget(*context);
    cfTarget.markUnknownOpDynamicallyLegal([&](Operation *op) {
      return op->getDialect() !=
             context->getLoadedDialect<cf::ControlFlowDialect>();
    });
    RewritePatternSet cfPatterns(context);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          cfPatterns);
    if (failed(applyPartialConversion(mod, cfTarget, std::move(cfPatterns))))
      return signalPassFailure();

    fixUpLoopAnnotation(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter,
                        const TargetInfo &targetInfo) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    unsigned alignment = getSqmmaSwizzleAlignment(mod);
    LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/alignment,
        static_cast<unsigned>(targetInfo.getSharedAddressSpace()));
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMUSAGPUToLLVMPass() {
  return std::make_unique<ConvertTritonMUSAGPUToLLVM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonMUSAGPUToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonMUSAGPUToLLVM>(computeCapability);
}

} // namespace mlir::triton
