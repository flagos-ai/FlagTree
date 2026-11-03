#include "TritonHCUGPUToLLVM/Passes.h"

#include "AsyncUtility.h"
#include "Analysis/HCUGPUAllocation.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "TargetInfo.h"
#include "TritonHCUGPUToLLVM/MembarUtility.h"
#include "TritonHCUGPUToLLVM/WaitCntHCUUtility.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#ifdef __TLE__
#include "tle/dialect/include/Conversion/TleToLLVM/ExclusiveCumsumOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/LocalPointersOpToLLVM.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#endif
#include "third_party/hcu/include/Analysis/AxisInfoExt.h"
#include "third_party/hcu/include/Dialect/TritonHCUGPU/IR/Dialect.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Analysis/Membar.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::HCU {
LogicalResult verifyLegacyMlsV0ClosedModule(ModuleOp mod, StringRef arch);
}

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONHCUGPUTOLLVM
#include "TritonHCUGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

#include "WavePrivateSyncStaging.h"
#include "WholePacketRegions.h"
#include "CompactPrefixOwner.h"
#include "TerminalReadonlyPublication.h"

namespace {

class TritonLLVMFunctionConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMFunctionConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
#ifdef __TLE__
    addIllegalDialect<triton::tle::TleDialect>();
#endif
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<triton::hcugpu::InstructionSchedHint>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

#ifdef __TLE__
class TleLLVMConversionTarget : public ConversionTarget {
public:
  explicit TleLLVMConversionTarget(MLIRContext &ctx,
                                   LLVMTypeConverter &typeConverter)
      : ConversionTarget(ctx) {
    addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, ROCDL::ROCDLDialect,
                    mlir::scf::SCFDialect>();
    addIllegalDialect<triton::tle::TleDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    markUnknownOpDynamicallyLegal([](Operation *) -> bool { return true; });
  }
};
#endif

class TritonHCUGPUToLLVMTypeConverter : public TritonGPUToLLVMTypeConverter {
public:
  TritonHCUGPUToLLVMTypeConverter(MLIRContext *ctx,
                                  const LowerToLLVMOptions &options,
                                  const TargetInfoBase &targetInfo,
                                  const DataLayoutAnalysis *analysis = nullptr)
      : TritonGPUToLLVMTypeConverter(ctx, options, targetInfo, analysis) {
    addConversion([&](TensorDescType type) -> std::optional<Type> {
      return convertTensorDescType(type);
    });
  }

  Type convertTensorDescType(triton::TensorDescType type) {
    auto ctx = type.getContext();
    auto blockType = type.getBlockType();
    auto shape = blockType.getShape();

    // Determine the number of dwords based on tensor dimensions
    // 2D tensors: group0 (4) + group1 (8) = 12 dwords
    // 3D-5D tensors: group0 (4) + group1 (8) + group2 (4) + group3 (4) = 20
    // dwords
    int numDwords = (shape.size() > 2) ? (4 + 8 + 4 + 4) : (4 + 8);

    auto types = SmallVector<Type>(numDwords, IntegerType::get(ctx, 32));
    return LLVM::LLVMStructType::getLiteral(ctx, types);
  }
};

struct ConvertTritonHCUGPUToLLVM
    : public triton::impl::ConvertTritonHCUGPUToLLVMBase<
          ConvertTritonHCUGPUToLLVM> {
  explicit ConvertTritonHCUGPUToLLVM(StringRef targetArch, bool ftz) {
    this->arch = targetArch.str();
    this->ftz = ftz;
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<LLVM::LLVMDialect, NVVM::NVVMDialect, mlir::ROCDL::ROCDLDialect,
                mlir::triton::hcugpu::TritonHCUGPUDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    HCU::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == HCU::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    if (failed(triton::HCU::verifyLegacyMlsV0ClosedModule(mod, targetInfo.getArch())))
      return signalPassFailure();

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonHCUGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    // Owner-global prefix proof must finish before shared allocation.
    unsigned compactedOwners = HCU::compactPrefixOwners(mod, targetInfo.getArch());

    // Allocate shared memory and set barrier
    ModuleAllocation allocation(
        mod, compactedOwners ? HCU::HCUAllocationAnalysisScratchSizeFn
                             : triton::defaultAllocationAnalysisScratchSizeFn);
    // An earlier allocation pass emitted sizes/offsets for the old geometry.
    // Refresh all roots and temporary buffers using the standard allocator.
    if (compactedOwners)
      triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
    bool invalidLegacyV0Frame = false;
    mod.walk([&](triton::gpu::LocalAllocOp alloc) {
      auto enc = dyn_cast<triton::gpu::HCUMlsSharedEncodingAttr>(alloc.getType().getEncoding());
      if (!enc || enc.getVersion() != 0)
        return;
      auto function = dyn_cast<FunctionOpInterface>(alloc->getParentOp());
      if (!function) {
        invalidLegacyV0Frame = true;
        return;
      }
      // The preflight has checked each packet is4096 16-bit elements.
      // Match both allocation analysis and emitted metadata to the real root.
      auto rootShape = alloc.getType().getShape();
      // Two/four real prefix slots: the existing allocator places ordinary A.
      // Check emitted and recomputed bytes/offset, never request-only metadata.
      if (rootShape.size() == 3 && (rootShape[0] == 2 || rootShape[0] == 4)) {
        uint64_t frameBytes = uint64_t(rootShape[0]) * 8192;
        auto *data = allocation.getFuncData(function);
        auto id = data->getBufferId(alloc.getResult());
        auto offset = alloc->getAttrOfType<IntegerAttr>("allocation.offset");
        auto size = mod->getAttrOfType<IntegerAttr>("ttg.shared");
        uint64_t total = allocation.getSharedMemorySize();
        invalidLegacyV0Frame |= id == Allocation::InvalidBufferId || !offset || !size ||
            offset.getInt() < 0 || uint64_t(size.getInt()) != total || total > 65536;
        if (id != Allocation::InvalidBufferId && offset) {
          uint64_t base = data->getOffset(id);
          invalidLegacyV0Frame |= data->getAllocatedSize(id) != frameBytes ||
              uint64_t(offset.getInt()) != base || base % 8192 ||
              base > total || total - base < frameBytes;
        }
        // N4's captured A root is also four whole8192B slots. Validate its
        // independently recomputed allocation and emitted offset; never alias B.
        if (rootShape[0] == 4 && id != Allocation::InvalidBufferId) {
          uint64_t bBase = data->getOffset(id);
          unsigned aRoots = 0;
          function->walk([&](triton::gpu::LocalAllocOp a) {
            if (a == alloc) return;
            ++aRoots;
            auto aId = data->getBufferId(a.getResult());
            auto aOffset = a->getAttrOfType<IntegerAttr>("allocation.offset");
            invalidLegacyV0Frame |= aId == Allocation::InvalidBufferId || !aOffset ||
                (aOffset && !aOffset.getValue().isSignedIntN(64));
            if (aId == Allocation::InvalidBufferId || !aOffset ||
                !aOffset.getValue().isSignedIntN(64)) return;
            uint64_t aBase = data->getOffset(aId);
            invalidLegacyV0Frame |= aOffset.getInt() < 0 ||
                uint64_t(aOffset.getInt()) != aBase || aBase % 8192 ||
                data->getAllocatedSize(aId) != frameBytes ||
                aBase > total || total - aBase < frameBytes ||
                !(aBase + frameBytes <= bBase || bBase + frameBytes <= aBase);
          });
          invalidLegacyV0Frame |= aRoots != 1;
        }
        return;
      }
      if (rootShape.size() != 3 || (rootShape[0] != 1 && rootShape[0] != 8)) {
        invalidLegacyV0Frame = true;
        return;
      }
      uint64_t expectedFrameBytes = uint64_t(rootShape[0]) * 8192;
      // Private diagnostic only: unchanged8KiB B frame and disjoint proved A.
      auto joint = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_joint_phase");
      const bool jointPhase = joint && joint.getInt() == 1;
      auto blockptr = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr");
      auto blockptrLoop = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr_loop");
      invalidLegacyV0Frame |= jointPhase &&
          (targetInfo.getArch() != "gfx936" || expectedFrameBytes != 8192 ||
           !blockptr || blockptr.getInt() != 1 ||
           !blockptrLoop || blockptrLoop.getInt() != 1);
      uint64_t expectedABytes = 2048;
      if (jointPhase)
        function->walk([&](triton::gpu::LocalAllocOp candidate) {
          if (!candidate.getSrc())
            return;
          auto shape = candidate.getType().getShape();
          // The strict staged-A preflight already checked the unique root,
          // fullM types, initialized Swizzled layout and exact value chain.
          if ((rootShape == ArrayRef<int64_t>({1, 32, 128}) &&
               shape == ArrayRef<int64_t>({128, 32})) ||
              (rootShape == ArrayRef<int64_t>({1, 16, 256}) &&
               shape == ArrayRef<int64_t>({256, 16})))
            expectedABytes = 8192;
        });
      const uint64_t expectedTotalBytes =
          jointPhase ? expectedFrameBytes + expectedABytes : expectedFrameBytes;
      const uint64_t expectedAOffset = jointPhase ? 8192 : 0;
      auto *data = allocation.getFuncData(function);
      auto id = data->getBufferId(alloc.getResult());
      // Check both recomputed analysis and the attributes actually consumed
      // by getSharedMemoryBase / the LLVM kernel metadata.
      auto emittedOffset = alloc->getAttrOfType<IntegerAttr>("allocation.offset");
      auto emittedSize = mod->getAttrOfType<IntegerAttr>("ttg.shared");
      invalidLegacyV0Frame |= id == Allocation::InvalidBufferId ||
          allocation.getSharedMemorySize() != expectedTotalBytes || !emittedOffset ||
          emittedOffset.getInt() != 0 || !emittedSize || emittedSize.getInt() != expectedTotalBytes;
      if (id != Allocation::InvalidBufferId)
        invalidLegacyV0Frame |= data->getOffset(id) != 0 || data->getAllocatedSize(id) != expectedFrameBytes;
      // AFTER successful strict matchLegacyV0StagedA preflight, before membar/lowering.
      // B remains offset0,size8192; the private joint phase checks real10/16KiB total.
      // No new pass or analysis framework. All barriers/DS waits remain untouched.
      if (auto enabled = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_blockptr");
          enabled && enabled.getInt() == 1) {
        SmallVector<triton::gpu::LocalAllocOp> otherRoots;
        function->walk([&](triton::gpu::LocalAllocOp candidate) {
          if (candidate != alloc)
            otherRoots.push_back(candidate);
        });
        if (expectedFrameBytes != 8192 || otherRoots.size() != 1) {
          invalidLegacyV0Frame = true;
        } else {
          auto aAlloc = otherRoots.front(); // Exact initialized Swizzled A from preflight.
          auto aId = data->getBufferId(aAlloc.getResult());
          auto aEmittedOffset = aAlloc->getAttrOfType<IntegerAttr>("allocation.offset");
          invalidLegacyV0Frame |= aId == Allocation::InvalidBufferId ||
              !aEmittedOffset || aEmittedOffset.getInt() != expectedAOffset;
          if (aId != Allocation::InvalidBufferId)
            invalidLegacyV0Frame |= data->getOffset(aId) != expectedAOffset ||
                data->getAllocatedSize(aId) != expectedABytes;
        }
      }
    });
    if (invalidLegacyV0Frame) {
      auto joint = mod->getAttrOfType<IntegerAttr>("ttg.hcu.v0_joint_phase");
      if (joint && joint.getInt() == 1)
        mod.emitError("legacy-v0 joint phase requires B0/8192,A8192/2048,total10240 or proved wide-M A8192/8192,total16384");
      else
        mod.emitError("legacy-v0 prototype requires a zero-origin8KiB/64KiB frame matching the root");
      return signalPassFailure();
    }

    if (targetInfo.requiresAliasInfoForAsyncOps())
      HCU::annotateLocalLoadsSyncedViaAsyncWait(mod);

    HCU::addLocalBarrierAfterAmdGpuAsyncWait(mod);
    WavePrivateSyncStaging wavePrivate(mod, allocation, targetInfo);
    wavePrivate.prepare();
    SmallVector<std::pair<Operation *, Operation *>> twoSlotWAR;
    SmallVector<std::pair<Operation *, Operation *>> asyncSlotWAR;
    SmallVector<std::pair<Operation *, Operation *>> legacyV0WAR;
    SmallVector<std::pair<Operation *, Operation *>> legacyV0LP1RAW;
    SmallVector<std::pair<Operation *, Operation *>> legacyV0LP1AWAR;
    SmallVector<std::pair<Operation *, Operation *>> indexedABWAR;
    if (targetInfo.getArch() == "gfx936") {
      twoSlotWAR = HCU::prepareTwoSlotPrefetchWAR(mod);
      asyncSlotWAR = HCU::prepareAsyncSlotPrefetchWAR(mod);
      legacyV0WAR = HCU::prepareLegacyV0Prefix2WAR(mod, allocation);
      legacyV0LP1RAW = HCU::prepareLegacyV0LocalPrefetchRAW(mod);
      legacyV0LP1AWAR = HCU::prepareLegacyV0LocalPrefetchAWAR(
          mod, allocation, legacyV0LP1RAW);
      indexedABWAR = HCU::prepareLegacyV0IndexedABWAR(mod, allocation);
    }
    HCU::WholePacketRegions packetRegions(mod, allocation, targetInfo.getArch());
    auto membarFilter = [&](Operation *before, Operation *after) {
      if (packetRegions.disjoint(before, after))
        return true;
      // The helper proves these exact main-body pairs immediately above;
      // membar only inserts barriers and leaves their operands unchanged.
      auto load = dyn_cast<triton::gpu::LocalLoadOp>(before);
      auto store = dyn_cast<triton::gpu::LocalStoreOp>(after);
      if (load && store && load.getSrc() == store.getDst() &&
          llvm::is_contained(twoSlotWAR, std::make_pair(before, after)))
        return true;
      // Only the independently proved current-A read -> future-A store.
      // Different views/slots intentionally do not require identical SSA src/dst.
      if (load && store && llvm::is_contained(
              legacyV0LP1AWAR, std::make_pair(before, after)))
        return true;
      // This read-only proof was computed on the same CF immediately above.
      // Membar inserts barriers but never changes these operations/operands.
      if (load && isa<triton::hcugpu::BufferLoadToLocalOp>(after) &&
          llvm::is_contained(asyncSlotWAR, std::make_pair(before, after)))
        return true;
      if (load && isa<triton::hcugpu::BufferLoadToLocalOp,
                      triton::hcugpu::MatrixLoadToLocalOp>(after) &&
          llvm::is_contained(legacyV0WAR, std::make_pair(before, after)))
        return true;
      if (load && isa<triton::hcugpu::MatrixLoadToLocalOp>(after) &&
          llvm::is_contained(indexedABWAR, std::make_pair(before, after)))
        return true;
      // The read and future write are physically distinct in THIS iteration.
      // Keep the reverse WAR direction and every other dependency unchanged.
      if (isa<triton::hcugpu::MatrixLoadToLocalOp>(before) &&
          isa<triton::gpu::LocalLoadOp>(after) &&
          llvm::is_contained(legacyV0LP1RAW, std::make_pair(before, after)))
        return true;
      return wavePrivate.filter(before, after);
    };
    ModuleMembarAnalysis membarPass(&allocation, membarFilter);
    membarPass.run();
    if (failed(HCU::scheduleLegacyV0LocalPrefetchReadFirst(mod, legacyV0LP1RAW)))
      return signalPassFailure();

    HCU::coalesceTerminalReadonlyPublication(mod, targetInfo.getArch());

    AMD::addWaitCntHCU(mod);

    // Lower functions
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      mlir::triton::populateFuncOpConversionPattern(
          typeConverter, funcPatterns, targetInfo, patternBenefitDefault);
      mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                            funcPatterns);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    // initSharedMemory is run before the conversion of call and ret ops,
    // because the call op has to know the shared memory base address of each
    // function
    initSharedMemory(typeConverter);

    // Convert call and ret ops
    {
      TritonLLVMFunctionConversionTarget funcTarget(*context);
      RewritePatternSet funcPatterns(context);
      if (failed(
              applyPartialConversion(mod, funcTarget, std::move(funcPatterns))))
        return signalPassFailure();
    }

    HCU::ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

#ifdef __TLE__
    {
      TleLLVMConversionTarget tleTarget(*context, typeConverter);
      RewritePatternSet tlePatterns(context);
      mlir::triton::tle::populateExtractTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateInsertTileOpToLLVMPatterns(
          typeConverter, tlePatterns, targetInfo,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateLocalPointersOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateExclusiveCumsumOpToLLVMPatterns(
          typeConverter, targetInfo, tlePatterns,
          patternBenefitPrioritizeOverLLVMConversions);
      mlir::triton::tle::populateMemDescAliasOpToLLVMPatterns(
          typeConverter, tlePatterns,
          patternBenefitPrioritizeOverLLVMConversions);
      if (failed(
              applyPartialConversion(mod, tleTarget, std::move(tlePatterns))))
        return signalPassFailure();
    }
#endif

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    // Make benefit for HCU specific patterns higher so they apply before common
    // patterns
    int HCUBenefit = commonBenefit + 1;
    auto populatePatterns1 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, axisInfoAnalysis, allocation,
                   benefit);
    };

    auto populatePatterns5 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, benefit);
    };

    auto populatePatterns6 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, axisInfoAnalysis, allocation,
                   targetInfo, benefit);
    };

    auto populatePatterns7 = [&](auto populateFunc, int benefit) {
      populateFunc(typeConverter, patterns, targetInfo, benefit);
    };

    HCU::populateConvertLayoutOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, HCUBenefit);
    mlir::triton::populateConvertLayoutOpToLLVMPatterns(
        typeConverter, targetInfo, patterns, commonBenefit);
    HCU::populateDotOpToLLVMPatterns(typeConverter, patterns, axisInfoAnalysis,
                                     HCUBenefit);
    HCU::populateElementwiseOpToLLVMPatterns(typeConverter, patterns, ftz,
                                             axisInfoAnalysis, allocation,
                                             targetInfo, HCUBenefit);
    HCU::populateLoadStoreOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                           axisInfoAnalysis, HCUBenefit);
    HCU::populateMaskedOpsToLLVMPatterns(patterns, targetInfo);
    HCU::populateBarrierOpToLLVMPatterns(typeConverter, patterns, HCUBenefit);
    HCU::populateTensorPtrOpsToLLVMPatterns(typeConverter, patterns,
                                            HCUBenefit);
    HCUBenefit += 1;
    HCU::populateMLSOpToLLVMPatterns(typeConverter, targetInfo, patterns,
                                     axisInfoAnalysis, HCUBenefit);

    populatePatterns7(mlir::triton::populateReduceOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateScanOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns5(mlir::triton::populateViewOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateHistogramOpToLLVMPatterns,
                      commonBenefit);
    populatePatterns7(mlir::triton::populateGatherOpToLLVMPatterns,
                      commonBenefit);

    HCU::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        HCUBenefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    HCU::populateSPMDOpToLLVMPattern(typeConverter, patterns, HCUBenefit);

    mlir::triton::HCU::populateTritonHCUGPUToLLVMPatterns(typeConverter,
                                                          patterns, HCUBenefit);
    mlir::triton::HCU::populateUpcastMXFPToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo, HCUBenefit);
    mlir::triton::HCU::populateFp4ToFpToLLVMPatterns(typeConverter, patterns,
                                                     targetInfo, HCUBenefit);
    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->arch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid HCUGPU chipset name: " + this->arch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populatePrintOpToLLVMPattern(typeConverter, patterns,
                                               targetInfo, commonBenefit);
    mlir::ub::populateUBToLLVMConversionPatterns(typeConverter, patterns);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }

    HCU::adjustModeRegister(mod, targetInfo);
    fixUpLoopAnnotation(mod);
  }

private:
  void initSharedMemory(LLVMTypeConverter &typeConverter) {
    ModuleOp mod = getOperation();
    OpBuilder b(mod.getBodyRegion());
    auto ctx = mod.getContext();
    auto loc = mod.getLoc();
    auto elemTy = typeConverter.convertType(b.getIntegerType(8));
    // Set array size 0 and external linkage indicates that we use dynamic
    // shared allocation to allow a larger shared memory size for each kernel.
    //
    // Ask for 16B alignment on global_smem because that's the largest we should
    // ever need (4xi32).
    auto arrayTy = LLVM::LLVMArrayType::get(elemTy, 0);
    auto global = LLVM::GlobalOp::create(
        b, loc, arrayTy, /*isConstant=*/false, LLVM::Linkage::External,
        "global_smem", /*value=*/Attribute(), /*alignment=*/16,
        // Add ROCm support.
        static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonHCUGPUToLLVMPass(StringRef targetArch, bool ftz) {
  return std::make_unique<ConvertTritonHCUGPUToLLVM>(targetArch, ftz);
}

} // namespace mlir::triton
