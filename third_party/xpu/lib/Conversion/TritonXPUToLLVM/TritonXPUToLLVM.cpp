#include "triton/Conversion/TritonXPUToLLVM/Passes.h"
// clang-format off
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
// DLTI: pulled in by the `triton_xpu.raw` payload import (see Passes.td).
#include "mlir/Dialect/DLTI/DLTI.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "PatternTritonXPUOpToLLVM.h"

// #include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"

// clang-format on

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_CONVERTTRITONXPUTOLLVM
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only
#include "triton/Conversion/TritonXPUToLLVM/Passes.h.inc"

} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<LLVM::XPU::LLVMXPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::xpu::TritonXPUDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    // addLegalDialect<index::IndexDialect>(); // TODO[dyq]: necessary?
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<LLVM::XPU::LLVMXPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertTritonXPUToLLVM
    : public triton::impl::ConvertTritonXPUToLLVMBase<ConvertTritonXPUToLLVM> {
  using ConvertTritonXPUToLLVMBase::ConvertTritonXPUToLLVMBase;

  ConvertTritonXPUToLLVM(uint32_t xpu_arch, uint32_t buffer_size,
                         bool isUseMaskZero)
      : ConvertTritonXPUToLLVMBase({xpu_arch, buffer_size, isUseMaskZero}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    // XPU has no warp concept, but upstream conversion patterns
    // (e.g. for llvm.mlir.addressof / shared memory base) call
    // triton::gpu::lookupNumWarps(op), which fatal-errors when the
    // surrounding module lacks the "ttg.num-warps" attribute. Set it to 1
    // here so those patterns succeed.
    if (!mod->hasAttr(mlir::triton::gpu::AttrNumWarpsName)) {
      OpBuilder b(mod);
      mod->setAttr(mlir::triton::gpu::AttrNumWarpsName, b.getI32IntegerAttr(1));
    }

    // Which module this invocation sees, for cases where the MLIR dump
    // sections do not line up with the pass input (sample_dirichlet: the
    // dumped module converts cleanly, yet the pipeline reports a failure).
    if (std::getenv("TRITONXPU_LOG_CONVERT_ENTRY")) {
      unsigned numFuncs = 0, numReduces = 0, numWhiles = 0;
      mod.walk([&](Operation *op) {
        if (isa<triton::FuncOp, LLVM::LLVMFuncOp>(op))
          ++numFuncs;
        else if (isa<triton::xpu::ReduceOp>(op))
          ++numReduces;
        else if (isa<scf::WhileOp>(op))
          ++numWhiles;
      });
      llvm::errs() << "[convert-entry] funcs=" << numFuncs
                   << " xpuReduces=" << numReduces << " whiles=" << numWhiles
                   << "\n";
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonXPUToLLVMTypeConverter typeConverter(context,
                                               option); // we can reuse it
    TritonLLVMConversionTarget convTarget(*context);

    // A bare signalPassFailure() only produces the pipeline-level "Failures
    // have been detected" line, which says nothing about which op or pattern
    // gave up (sample_dirichlet). Name what is left, or say that nothing is.
    // Off by default: the module is large. Set
    // TRITONXPU_DUMP_ON_CONVERT_FAILURE to get the state the pass actually
    // failed on (the MLIR dump sections do not line up one-to-one with the pass
    // input).
    auto dumpOnFailure = [&]() {
      if (std::getenv("TRITONXPU_DUMP_ON_CONVERT_FAILURE"))
        mod.print(llvm::errs());
    };
    auto reportUnconverted = [&](llvm::StringRef stage) {
      unsigned reported = 0;
      mod.walk([&](Operation *op) {
        if (reported >= 8)
          return;
        llvm::StringRef ns = op->getName().getDialectNamespace();
        if (ns != "tt" && ns != "triton_xpu" && ns != "ttg")
          return;
        op->emitError() << stage << ": left unconverted";
        ++reported;
      });
      if (reported == 0)
        mod.emitError() << stage
                        << ": conversion failed with no Triton/XPU op left; a "
                           "pattern signalled failure (use "
                           "--mlir-print-ir-after-all)";
    };

    // Allocate shared memory and set barrier
    // TODO[dyq]: necessary to open?
    // ModuleAllocation allocation(mod);
    // ModuleMembarAnalysis membarPass(&allocation);
    // membarPass.run();

    // Lower functions
    {
      mlir::LowerToLLVMOptions option(context);
      option.overrideIndexBitwidth(32);
      TritonXPUToLLVMTypeConverter typeConverter(context, option);
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      // TODO[dyq]: add [nvvm.maxntid, nvvm.kernel] attr
      mlir::triton::xpu::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(applyPartialConversion(mod, funcTarget,
                                        std::move(funcPatterns)))) {
        dumpOnFailure();
        reportUnconverted("function conversion");
        return signalPassFailure();
      }
    }

    // LLVM_DEBUG(llvm::dbgs() << "\nAfter Lower Functions:\n" << mod << "\n");

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of
    // each function
    if (failed(stampSharedMemOffsets(mod))) {
      dumpOnFailure();
      mod.emitError() << "shared-memory offset stamping failed";
      return signalPassFailure();
    }
    if (failed(initSharedMemory(typeConverter))) {
      dumpOnFailure();
      return signalPassFailure();
    }
    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    triton::xpu::TargetInfo targetInfo(xpu_arch, buffer_size, isUseMaskZero);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for XPU specific patterns higher so they apply before common
    // patterns
    int xpuBenefit = benefit + 1;

    // TODO[dyq]: Open allToLLVMPatterns
    mlir::triton::xpu::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // TODO[dyq]: XPUSDNN-CHECK add DotOp Lowering Pattern
    // mlir::triton::xpu::populateDotOpToLLVMPatterns(typeConverter, patterns,
    //                                                benefit);
    mlir::triton::xpu::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    mlir::triton::populateElementwiseOpToLLVMPatterns(
        typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit);

    mlir::triton::xpu::populateTTXPUVectorizedOpToLLVMConversionPatterns(
        typeConverter, targetInfo, patterns, benefit);

    // TODO[dyq]:
    mlir::triton::xpu::populateTTXPUUtilityOpToLLVMConversionPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    mlir::triton::xpu::populateLoadStoreOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, axisInfoAnalysis, benefit);

    mlir::triton::xpu::populateGPUToXPUConversionPatterns(
        typeConverter, patterns, targetInfo, benefit);

    mlir::triton::xpu::populateReduceOpToLLVMPatterns(typeConverter, patterns,
                                                      targetInfo, benefit);

    mlir::triton::xpu::populateScanOpToLLVMPatterns(typeConverter, patterns,
                                                    targetInfo, benefit);

    // mlir::triton::populateHistogramOpToLLVMPatterns(typeConverter, patterns,
    //                                                 targetInfo, benefit);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, benefit);
    mlir::triton::xpu::populateXPUPrintOpToLLVMPattern(typeConverter, patterns,
                                                       targetInfo, benefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::triton::xpu::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                                   benefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, benefit);

    mlir::triton::xpu::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                                    xpuBenefit);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    // mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, targetInfo,
    //                                             patterns, benefit);
    mlir::triton::xpu::populateMakeRangeOpToLLVMPattern(
        typeConverter, targetInfo, patterns, benefit);

    // tle.raw on the cluster path: declare the payload and call it; the body is
    // merged into the module at the LLVM 19 stage (tle/raw/merge.py).
    mlir::triton::xpu::populateRawOpToLLVMPatterns(typeConverter, patterns,
                                                   benefit);

    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      dumpOnFailure();
      reportUnconverted("op conversion");
      return signalPassFailure();
    }
  }

private:
  Value smem;

  /// Assign every `#triton_xpu.smem` `ttg.local_alloc` a byte offset inside the
  /// cluster-wide shared block, stamped as `xpu.smem_offset`.
  /// XPUTLELocalAllocOpConversion then lowers the alloc to
  /// `gep(gep(global_smem, coreId * perCoreBytes), smem_offset)`, so a region
  /// is `perCoreBytes * kCoresPerCluster` wide -- SM, unlike LM, is not
  /// replicated per core.
  ///
  /// Regions are laid out top-down from the budget so that they stay clear of
  /// the bottom of the block, where anything growing upwards would sit. Note
  /// that `tritonxpu-loop-invariant-staging` also reserves SM top-down at
  /// runtime (see third_party/xpu/docs/XPUTLEPipelineDesignDoc.md:601); the two
  /// schemes would collide, and today only do not because a TLE kernel never
  /// stages.
  LogicalResult stampSharedMemOffsets(ModuleOp mod) {
    unsigned groupSize = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);
    unsigned nextTop = triton::xpu::kSharedMemBudgetBytes;
    LogicalResult res = success();
    mod.walk([&](triton::gpu::LocalAllocOp alloc) {
      if (!triton::xpu::isSharedMemDesc(alloc.getType()))
        return;
      unsigned bytes = triton::xpu::getSharedMemRegionBytes(alloc, groupSize);
      if (bytes > nextTop) {
        alloc.emitError("out of shared memory: this #triton_xpu.smem buffer "
                        "needs ")
            << bytes << " bytes but only " << nextTop << " of the "
            << triton::xpu::kSharedMemBudgetBytes
            << "-byte cluster block are left";
        res = failure();
        return;
      }
      nextTop -= bytes;
      OpBuilder b(alloc);
      alloc->setAttr(triton::xpu::kSharedMemOffsetAttrName,
                     b.getI32IntegerAttr(static_cast<int32_t>(nextTop)));
    });
    return res;
  }

  /// Materialise the `global_smem` base pointer into the `smem` member.
  ///
  /// Why: every shared-memory conversion pattern reads `smem`, so a module
  /// this cannot be run on has to stop the pass before the patterns are
  /// applied -- `signalPassFailure` alone does not, it only marks the
  /// pipeline failed after `runOnOperation` returns.
  LogicalResult initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.

    // XPU: the shared space used by Triton will be put at the end of
    // SHARED_MEMORY section since only pointer is used without the real
    // allocation
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = b.create<LLVM::GlobalOp>(
        loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::Internal,
        "global_smem", /*value=*/Attribute(), /*alignment=*/8, 2);

    // TODO[dyq]: llvm-18 don't need to set pointer type, will this logic be
    // changed?
    SmallVector<LLVM::LLVMFuncOp> funcs;
    mod.walk([&](LLVM::LLVMFuncOp func) { funcs.push_back(func); });
    // Name the leftovers instead of aborting: a module that still carries a
    // helper (or a second kernel) cannot be told apart from a pass bug once the
    // assert fires, and the pipeline only reports "Failures have been detected"
    // for a pass that returns quietly.
    if (funcs.size() != 1) {
      for (LLVM::LLVMFuncOp func : funcs)
        func->emitError()
            << "ConvertTritonXPUToLLVM expects exactly one function after "
               "inlining, found "
            << funcs.size();
      return failure();
    }
    b.setInsertionPointToStart(&funcs[0].getBody().front());
    smem = b.create<LLVM::AddressOfOp>(loc, global);

    // TODO[dyq]: llvm-18 don't need to set pointer type, this type maybe cause
    // error
    // auto ptrTy =
    // LLVM::LLVMPointerType::get(typeConverter.convertType(b.getI8Type()), 2);
    auto ptrTy = LLVM::LLVMPointerType::get(mod.getContext(), 2);
    smem = b.create<LLVM::BitcastOp>(loc, ptrTy, smem);
    return success();
  }
};

} // namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonXPUToLLVMPass() {
  return std::make_unique<ConvertTritonXPUToLLVM>();
}

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonXPUToLLVMPass(uint32_t xpu_arch, uint32_t buffer_size,
                                 bool isUseMaskZero) {
  return std::make_unique<ConvertTritonXPUToLLVM>(xpu_arch, buffer_size,
                                                  isUseMaskZero);
}

} // namespace triton
} // namespace mlir
