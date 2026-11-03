#include "Dialect/TritonHCUGPU/IR/Dialect.h"
#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "TritonHCUGPUTransforms/Passes.h"
#include "hcu/lib/TritonHCUGPUToLLVM/AsyncUtility.h"
#include "hcu/lib/TritonHCUGPUToLLVM/Utility.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/hcu/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include <type_traits>

#undef DEBUG_TYPE
#define DEBUG_TYPE "tritonhcugpu-coalesce-async-copy"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace ttg = triton::gpu;
namespace tta = triton::hcugpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUCOALESCEASYNCCOPY
#include "TritonHCUGPUTransforms/Passes.h.inc"

namespace {

// Adapter changes only which distributed tensor is permuted. BUFFER's base
// pointer and stride stay scalar and unchanged; all other layout logic is shared.
template <typename CopyOp> struct CopyOperands {
  static constexpr bool isBuffer = std::is_same_v<CopyOp, tta::BufferLoadToLocalOp>;
  static Value input(CopyOp op) {
    if constexpr (isBuffer) return op.getOffsets();
    else return op.getSrc();
  }
  static Value dest(CopyOp op) {
    if constexpr (isBuffer) return op.getDest();
    else return op.getResult();
  }
  static void setInput(CopyOp op, Value value) {
    if constexpr (isBuffer) op.getOffsetsMutable().assign(value);
    else op.getSrcMutable().assign(value);
  }
};

// Reuse AMD's padded-copy linear-component construction, but reserve the
// consumer's pure batch bases for warp IDs. The remaining bases still give
// consecutive registers/lanes in LDS, with the SAME proven vector width.
// This pattern changes data ownership only; all waits/barriers remain untouched.
template <typename StoreOp>
struct PreferWavePrivateSyncStores : public OpRewritePattern<StoreOp> {
  PreferWavePrivateSyncStores(
      const triton::HCU::TargetInfo &targetInfo,
      MLIRContext *ctx)
      : OpRewritePattern<StoreOp>(ctx, PatternBenefit(2)), targetInfo(targetInfo) {}

  LogicalResult matchAndRewrite(StoreOp copy,
                                PatternRewriter &rewriter) const override {
    Value input = copy.getSrc();
    if (!input)
      return failure();
    // As in AMD/HCU coalescing, change the producer's load layout and
    // rematerialize offsets/masks, rather than materializing a data shuffle.
    auto load = input.getDefiningOp<tta::BufferLoadOp>();
    if (!load || !input.hasOneUse())
      return failure();
    auto srcTy = cast<RankedTensorType>(input.getType());
    auto dstTy = [&]() {
      if constexpr (std::is_same_v<StoreOp, ttg::LocalStoreOp>)
        return copy.getDst().getType();
      else
        return copy.getType();
    }();
    auto blocked = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    ModuleOp module = copy->template getParentOfType<ModuleOp>();
    auto opt = module->getAttrOfType<IntegerAttr>("ttg.hcu.wave_private_sync");
    if (!opt || opt.getInt() != 1)
      return failure();
    bool async = false;
    module.walk([&](Operation *op) {
      auto name = op->getName().getStringRef();
      if (name.contains("async") || name.contains("tdm") ||
          isa<tta::BufferLoadToLocalOp, tta::MatrixLoadToLocalOp>(op))
        async = true;
    });
    if (async)
      return failure();
    auto ctas = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
    if (targetInfo.getArch() != "gfx936" || !blocked || !shared ||
        srcTy.getRank() != 3 || dstTy.getRank() != 3 ||
        !(dstTy.getElementType().isF16() || dstTy.getElementType().isBF16()) ||
        !ctas || ctas.getInt() != 1 ||
        shared.getOrder().back() != 0 ||
        blocked.getOrder().front() != shared.getOrder().front())
      return failure();
    int warps = ttg::lookupNumWarps(copy);
    int lanes = ttg::TritonGPUDialect::getThreadsPerWarp(module);
    if (lanes != 64 || warps <= 1 || !llvm::isPowerOf2_32(warps) ||
        srcTy.getShape()[0] < warps)
      return failure();

    // Restrict this preference to homogeneous legacy batched dots whose
    // entire warp distribution is on the independent batch dimension.
    bool seen = false, compatible = true;
    module.walk([&](triton::DotOp dot) {
      auto type = dyn_cast<RankedTensorType>(dot.getResult().getType());
      auto mma = type ? dyn_cast<ttg::AMDMfmaEncodingAttr>(type.getEncoding())
                      : ttg::AMDMfmaEncodingAttr{};
      if (!type || type.getRank() != 3 || !mma ||
          mma.getMmacLayout() != ttg::MmacLayout::LEGACY ||
          type.getShape()[0] != srcTy.getShape()[0] ||
          !llvm::equal(mma.getWarpsPerCTA(),
                       ArrayRef<unsigned>({unsigned(warps), 1, 1})))
        compatible = false;
      else
        seen = true;
    });
    if (!seen || !compatible)
      return failure();

    // Preserve the original global-contiguous register vector. Regular LDS
    // stores support per-lane addresses and may split it at a shared swizzle;
    // direct-to-LDS packet/scatter restrictions do not apply here.
    unsigned width = std::min<unsigned>(
        ttg::getContigPerThread(srcTy)[blocked.getOrder().front()], 8);
    if (!width || !llvm::isPowerOf2_32(width))
      return failure();

    auto offset = rewriter.getStringAttr("offset");
    // Match the existing AMD/HCU swizzled-copy lowering: distribute the
    // flat shared layout and let lowering permute offsets/masks within a wave.
    // Swizzled bases can mix dimensions and are not legal distributed bases.
    auto flatEncoding = ttg::SwizzledSharedEncodingAttr::get(
        copy.getContext(), 1, 1, 1, shared.getOrder(), shared.getCTALayout());
    auto flatType = ttg::MemDescType::get(
        dstTy.getShape(), dstTy.getElementType(), flatEncoding,
        dstTy.getMemorySpace(), dstTy.getMutableMemory());
    auto flatLayout = ttg::toLinearLayout(flatType);
    auto bases = flatLayout.getBases().lookup(offset);
    std::vector<std::vector<int32_t>> rest, warpBases;
    for (const auto &basis : bases) {
      if (basis[0] > 0 && unsigned(basis[0]) < unsigned(warps) &&
          basis[1] == 0 && basis[2] == 0)
        warpBases.push_back(basis);
      else
        rest.push_back(basis);
    }
    unsigned regBits = llvm::Log2_32(width);
    unsigned laneBits = llvm::Log2_32(lanes);
    if (warpBases.size() != llvm::Log2_32(warps) ||
        rest.size() < regBits + laneBits)
      return failure();
    // Batch must not leak into register/lanes before all lane bits are filled.
    for (const auto &basis : ArrayRef<std::vector<int32_t>>(rest).take_front(regBits + laneBits))
      if (basis[0] != 0)
        return failure();
    std::vector<std::vector<int32_t>> regBases(
        rest.begin(), rest.begin() + regBits);
    std::vector<std::vector<int32_t>> laneBases(
        rest.begin() + regBits, rest.begin() + regBits + laneBits);
    regBases.insert(regBases.end(), rest.begin() + regBits + laneBits, rest.end());
    auto *ctx = copy.getContext();
    triton::LinearLayout candidate(
        {{rewriter.getStringAttr("register"), regBases},
         {rewriter.getStringAttr("lane"), laneBases},
         {rewriter.getStringAttr("warp"), warpBases}},
        triton::standardOutDimNames(ctx, 3));
    candidate = ttg::combineCtaCgaWithShape(
        candidate, blocked.getCTALayout(), srcTy.getShape());
    // Vectorize global reads only when the new register group remains
    // contiguous AND aligned on the original global-contiguous dimension.
    // A shared swizzle may otherwise XOR a lane bit into a vector's base.
    unsigned minor = blocked.getOrder().front();
    for (unsigned bit = 0; bit < regBits; ++bit)
      for (unsigned dim = 0; dim < 3; ++dim)
        if (regBases[bit][dim] != (dim == minor ? (1 << bit) : 0))
          return failure();
    unsigned vectorMask = width - 1;
    for (const auto &basis : laneBases)
      if (unsigned(basis[minor]) & vectorMask)
        return failure();
    for (const auto &basis : warpBases)
      if (unsigned(basis[minor]) & vectorMask)
        return failure();
    for (const auto &basis : ArrayRef<std::vector<int32_t>>(regBases).drop_front(regBits))
      if (unsigned(basis[minor]) & vectorMask)
        return failure();
    auto mapping = candidate.invertAndCompose(flatLayout);
    if (mapping.getNumConsecutiveInOut() < width)
      return failure();
    auto encoding = ttg::LinearEncodingAttr::get(ctx, candidate);
    rewriter.setInsertionPoint(load);
    auto convertOperand = [&](Value value) -> Value {
      if (!value)
        return value;
      auto type = cast<RankedTensorType>(value.getType());
      return ttg::ConvertLayoutOp::create(
          rewriter, load.getLoc(), type.cloneWithEncoding(encoding), value);
    };
    auto offsets = convertOperand(load.getOffsets());
    auto mask = convertOperand(load.getMask());
    auto other = convertOperand(load.getOther());
    auto newLoad = tta::BufferLoadOp::create(
        rewriter, load.getLoc(), srcTy.cloneWithEncoding(encoding),
        load.getPtr(), offsets, load.getStride(), load.getCache(), mask, other);
    // Preserve advisory load attributes, including range/alignment facts.
    for (auto attr : load->getAttrs())
      if (!newLoad->hasAttr(attr.getName()))
        newLoad->setAttr(attr.getName(), attr.getValue());
    rewriter.replaceOp(load, newLoad.getResult());
    return success();
  }

private:
  const triton::HCU::TargetInfo &targetInfo;
};

// On gfx9 global and buffer loads directly to shared memory need to write
// coalesced. This pattern converts the layout of the src, mask and other to
// ensure the owned data per thread is contiguous and does no exceed the
// supported load vector size.
template <typename CopyOp>
struct CoalesceAsyncCopyWrites : public OpRewritePattern<CopyOp> {
  using Base = OpRewritePattern<CopyOp>;
  using Operands = CopyOperands<CopyOp>;
  CoalesceAsyncCopyWrites(const triton::HCU::TargetInfo &targetInfo,
                          const DenseMap<Operation *, unsigned> &asyncCopyContiguity,
                          MLIRContext *ctx)
      : Base(ctx), targetInfo{targetInfo},
        asyncCopyContiguity{std::move(asyncCopyContiguity)} {}

  LogicalResult matchAndRewrite(CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    auto src = Operands::input(copyOp);
    auto dst = Operands::dest(copyOp);
    Value mask = copyOp.getMask();
    Value other = copyOp.getOther();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<ttg::MemDescType>(dst.getType());

    auto blockedEnc = dyn_cast<ttg::BlockedEncodingAttr>(srcTy.getEncoding());
    if (!blockedEnc)
      return rewriter.notifyMatchFailure(copyOp,
                                         "src encoding must be #blocked");

    bool linearShared = targetInfo.getArch() == "gfx936" &&
                        isa<ttg::SharedLinearEncodingAttr>(dstTy.getEncoding());
    if (!linearShared &&
        !isa<ttg::SwizzledSharedEncodingAttr, ttg::PaddedSharedEncodingAttr>(
            dstTy.getEncoding())) {
      return rewriter.notifyMatchFailure(
          copyOp, "dst encoding must be #swizzled, #padded, or gfx936 "
                  "#shared_linear");
    }

    // We start from the precomputed contiguity we got from AxisAnalysis.
    unsigned loadContig = 0;
    if (auto it = asyncCopyContiguity.find(copyOp.getOperation());
        it != asyncCopyContiguity.end())
      loadContig = it->second;
    else
      return copyOp->emitError()
             << "No contiguity information about the copy op";
    assert(loadContig > 0);

    // Further restrict the contiguity based on the contiguity of the src to dst
    // layout e.g. if the order of the blocked and shared encoding is different
    // we can only load one element at a time or if the shared encoding is
    // swizzled we cannot exceed the vector size of the swizzling pattern
    LinearLayout regLayout = triton::gpu::toLinearLayout(srcTy);
    LinearLayout sharedLayout;
    auto paddedEnc =
        dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(dstTy);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);
    loadContig = std::min<unsigned>(loadContig,
                                    regToSharedLayout.getNumConsecutiveInOut());

    // Select the largest supported load width equal or smaller than loadContig
    auto elemBitWidth = dstTy.getElementTypeBitWidth();
    loadContig =
        fitToValidDirectToLdsVecSize(loadContig, elemBitWidth, targetInfo,
                                    Operands::isBuffer);

    if (loadContig == 0) {
      return rewriter.notifyMatchFailure(
          copyOp, "could not find layout config to create coalesced writes");
    }

    // Do not rewrite if we already use the correct contiguity (could be from a
    // previous rewrite)
    auto mod = copyOp->template getParentOfType<ModuleOp>();
    int numWarps = triton::gpu::lookupNumWarps(copyOp);
    int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(mod);

    ttg::DistributedEncodingTrait newDistEnc;

    if (LLVM::HCU::canCoalesceWriteIntoSharedMemory(
            rewriter, regToSharedLayout, threadsPerWarp, loadContig)) {
      if (copyOp.getContiguity() != loadContig) {
        rewriter.modifyOpInPlace(copyOp,
                                [&]() { copyOp.setContiguity(loadContig); });
        return success();
      }
      return rewriter.notifyMatchFailure(copyOp, "already writes coalesced");
    }

    if (isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding())) {
      // For swizzled layouts we apply the swizzling during lowering so we only
      // adjust the sizePerThread of the blocked encoding to avoid strided
      // writes into LDS
      auto contigPerThread = ttg::getContigPerThread(srcTy);
      auto srcElemContig = contigPerThread[blockedEnc.getOrder()[0]];
      assert(srcElemContig >= loadContig);
      contigPerThread[blockedEnc.getOrder()[0]] = loadContig;
      auto copyOrder = llvm::to_vector(blockedEnc.getOrder());
      if constexpr (Operands::isBuffer) {
        auto sharedOrder = cast<ttg::SwizzledSharedEncodingAttr>(
            dstTy.getEncoding()).getOrder();
        // BUFFER-to-LDS cannot scatter a wave across batch planes.
        if (targetInfo.getArch() == "gfx936" && srcTy.getRank() == 3 &&
            copyOrder.front() == sharedOrder.front() && sharedOrder.back() == 0)
          copyOrder.assign(sharedOrder.begin(), sharedOrder.end());
      }
      newDistEnc = BlockedEncodingAttr::get(
          copyOp.getContext(), srcTy.getShape(), contigPerThread,
          copyOrder, numWarps, threadsPerWarp,
          blockedEnc.getCTALayout());
    } else if (paddedEnc || linearShared) {
      // Padded and shared-linear layouts map LDS offsets to n-D tensor indices
      // through a linear component. It might reorder elements resulting in
      // scattered writes into LDS which is not supported on GFX9. To ensure
      // coalesced writes we change the src layout to a linear encoding which
      // effectivly copies/mimicks the linear_component so each warp (reg+lane
      // bases) map to consecutive LDS offsets resulting in coalesced writes
      // The new linear encoding is build by taking bases from the
      // linear_component and assigning them to reg/lane/warp bases in the
      // following steps:
      // 1) Take log2(loadContig) bases as reg bases to ensure our registers per
      // load instruction point to contiguous elements in LDS.
      // 2) Take log2(threadsPerWarp) as lane bases to ensure lanes write
      // contiguous into LDS.
      // 3) Take log2(numWarps) as warp bases or add braodcasting bases if we
      // run out of bases
      // 4) Take any remaining bases as additional reg bases

      auto *ctx = srcTy.getContext();
      StringAttr kOffset = StringAttr::get(ctx, "offset");

      auto rank = srcTy.getRank();

      auto offsetBases = sharedLayout.getBases().lookup(kOffset);

      int log2LoadContig = llvm::Log2_32(loadContig);
      int log2ThreadsPerWarp = llvm::Log2_32(threadsPerWarp);
      int log2NumWarps = llvm::Log2_32(numWarps);

      if (offsetBases.size() < log2LoadContig + log2ThreadsPerWarp) {
        return rewriter.notifyMatchFailure(
            copyOp, "dst shape is too small. We require at least loadContig * "
                    "threadsPerWarp elements");
      }

      auto remainingBases = ArrayRef(offsetBases);
      auto takeN = [&remainingBases](size_t n) {
        auto take = std::min(remainingBases.size(), n);
        auto v = remainingBases.take_front(take).vec();
        remainingBases = remainingBases.drop_front(take);
        return v;
      };

      auto regBases = takeN(log2LoadContig);
      auto laneBases = takeN(log2ThreadsPerWarp);
      auto warpBases = takeN(log2NumWarps);
      warpBases.resize(log2NumWarps, std::vector<int32_t>(rank, 0));
      append_range(regBases, remainingBases);

      triton::LinearLayout newRegLayout(
          {
              {StringAttr::get(ctx, "register"), regBases},
              {StringAttr::get(ctx, "lane"), laneBases},
              {StringAttr::get(ctx, "warp"), warpBases},
          },
          triton::standardOutDimNames(ctx, rank));

      newRegLayout = triton::gpu::combineCtaCgaWithShape(
          newRegLayout, blockedEnc.getCTALayout(), srcTy.getShape());

      auto newRegToShared = newRegLayout.invertAndCompose(sharedLayout);
      if (newRegToShared.getNumConsecutiveInOut() < loadContig) {
        return rewriter.notifyMatchFailure(
            copyOp, "could not coalesce global addresses based on the linear "
                    "component of the shared encoding");
      }

      // A truthful shared-linear layout may contain XOR swizzles whose bases
      // move in multiple tensor dimensions. Distributed LinearEncodingAttr
      // cannot express those bases. Leave such copies to the legality fallback
      // below instead of constructing an invalid source encoding.
      if (linearShared)
        for (const auto &dimBases :
             llvm::make_second_range(newRegLayout.getBases()))
          for (const auto &basis : dimBases)
            if (llvm::count_if(basis, [](int32_t x) { return x != 0; }) > 1)
              return rewriter.notifyMatchFailure(
                  copyOp, "shared-linear copy requires an unsupported "
                          "mixed-dimension source basis");

      newDistEnc = ttg::LinearEncodingAttr::get(ctx, newRegLayout);
    } else {
      assert(false && "Unsupported layout");
    }

    if (newDistEnc == srcTy.getEncoding()) {
      // A gfx936 BUFFER copy may already use the selected swizzled layout.
      // Preserve the width proven above even when no layout cast is needed;
      // the final legality check still validates the source and LDS mapping.
      if constexpr (Operands::isBuffer) {
        if (targetInfo.getArch() == "gfx936" &&
            isa<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding()) &&
            copyOp.getContiguity() != loadContig) {
          rewriter.modifyOpInPlace(
              copyOp, [&]() { copyOp.setContiguity(loadContig); });
          return success();
        }
      }
      return rewriter.notifyMatchFailure(
          copyOp, "Unable to find a new src layout to coalesce writes to LDS");
    }

    // Convert layout of src, mask and other to new encoding
    auto convertLayout = [&rewriter](auto loc, Value old, auto newEnc) {
      auto oldTy = cast<RankedTensorType>(old.getType());
      RankedTensorType newSrcTy = oldTy.cloneWithEncoding(newEnc);
      return ttg::ConvertLayoutOp::create(rewriter, loc, newSrcTy, old);
    };

    auto loc = copyOp->getLoc();
    Value cvtSrc = convertLayout(loc, src, newDistEnc);

    if (mask)
      mask = convertLayout(loc, mask, newDistEnc);
    if (other)
      other = convertLayout(loc, other, newDistEnc);

    rewriter.modifyOpInPlace(copyOp, [&]() {
      Operands::setInput(copyOp, cvtSrc);
      if (mask)
        copyOp.getMaskMutable().assign(mask);
      if (other)
        copyOp.getOtherMutable().assign(other);
      copyOp.setContiguity(loadContig);
    });
    return success();
  }

private:
  const triton::HCU::TargetInfo &targetInfo;
  const DenseMap<Operation *, unsigned> &asyncCopyContiguity;
};

// Check the final GLOBAL operation after source-layout rewrites. A wider
// BUFFER opcode does not make this operation legal. Unknown shared models
// use the ordinary register-load/store path instead.
static void fallbackInvalidGlobalCopies(
    ModuleOp module, const triton::HCU::TargetInfo &targetInfo) {
  if (targetInfo.getArch() != "gfx936")
    return;
  HCU::ModuleAxisInfoAnalysis axisInfo(module);
  IRRewriter rewriter(module.getContext());
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> fallback;
  module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
    auto srcTy = copy.getSrc().getType();
    auto dstTy = copy.getResult().getType();
    auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    // GLOBAL async lowering does not implement volatile. Keep nonzero/unknown
    // other and eviction hints on the established synchronous load path too.
    // For now even explicit zero other is handled conservatively here.
    bool legal = !copy.getIsVolatile() && !copy.getOther() &&
                 copy.getEvict() == triton::EvictionPolicy::NORMAL &&
                 (shared || padded) &&
                 !LLVM::SharedMemoryObject::isAffineSharedMemoryAccess(dstTy) &&
                 llvm::equal(dstTy.getShape(), dstTy.getAllocShape());
    if (!legal) {
      fallback.push_back(copy);
      return;
    }
    unsigned contig = LLVM::HCU::getVectorSize(copy.getSrc(), axisInfo);
    if (auto mask = copy.getMask())
      contig = std::min<unsigned>(contig, axisInfo.getMaskAlignment(mask));
    // Match the lowering's effective vector width, including its hint. Do not
    // let a hint override the source/mask facts established by this pass.
    unsigned vec = std::max(contig, copy.getContiguity());
    legal = vec && llvm::isPowerOf2_32(vec) &&
            copy.getContiguity() <= contig &&
            targetInfo.supportsDirectToLdsLoadBitWidth(
                vec * dstTy.getElementTypeBitWidth(), /*isBuffer=*/false);
    if (!legal) {
      fallback.push_back(copy);
      return;
    }
    auto srcLayout = ttg::toLinearLayout(srcTy);
    auto dstLayout = padded ? padded.getLinearComponent()
                            : ttg::toLinearLayout(dstTy);
    auto mapping = srcLayout.invertAndCompose(dstLayout);
    if (!targetInfo.supportsDirectToLDSScattering()) {
      auto flatTy = dstTy;
      if (shared && shared.getMaxPhase() != 1) {
        legal &= LLVM::HCU::doesSwizzleInsideWarp(
            rewriter, mapping, targetInfo.getWarpSize());
        auto flat = ttg::SwizzledSharedEncodingAttr::get(
            module.getContext(), shared.getVec(), 1, 1, shared.getOrder(),
            shared.getCTALayout());
        flatTy = ttg::MemDescType::get(dstTy.getShape(), dstTy.getElementType(),
                                     flat, dstTy.getMemorySpace());
      }
      auto flatLayout = padded ? padded.getLinearComponent()
                               : ttg::toLinearLayout(flatTy);
      auto flatMapping = srcLayout.invertAndCompose(flatLayout);
      legal &= LLVM::HCU::canCoalesceWriteIntoSharedMemory(
          rewriter, flatMapping, targetInfo.getWarpSize(), vec);
      if (padded) {
        unsigned packet = targetInfo.getWarpSize() * vec;
        legal &= padded.getMinInterval() >= packet;
        auto regBases = mapping.getBases().lookup(rewriter.getStringAttr("register"));
        for (const auto &basis : llvm::drop_begin(regBases, llvm::Log2_32(vec)))
          legal &= (basis[0] & (packet - 1)) == 0;
      }
    }
    if (!legal)
      fallback.push_back(copy);
  });
  for (auto copy : fallback) {
    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    auto load = triton::LoadOp::create(
        rewriter, loc, copy.getSrc(), copy.getMask(), copy.getOther(),
        copy.getCache(), copy.getEvict(), copy.getIsVolatile(),
        rewriter.getStringAttr(""));
    load->setAttr("tt.load.async", rewriter.getBoolAttr(false));
    ttg::LocalStoreOp::create(rewriter, loc, load.getResult(), copy.getResult());
    // Preserve every existing commit, wait, LocalLoad token, and merged chain.
    // This dependency is completed; it does not erase other async operations.
    auto completed = ttg::AsyncCommitGroupOp::create(rewriter, loc, ValueRange{});
    rewriter.replaceOp(copy, completed.getResult());
  }
}

template <typename CopyOp>
static void fallbackInvalidCopies(ModuleOp m,
    const triton::HCU::TargetInfo &targetInfo,
    const DenseMap<Operation *, unsigned> &contiguity) {
  // A valid instruction width alone does not guarantee a legal gfx9 LDS
  // destination. In particular, a minor-dimension subslice can have holes
  // between rows that a non-scattering wave-wide load cannot write.
  // Keep such copies correct via registers instead of leaving an operation
  // that fails conversion to LLVM (or applying the contiguous-view swizzle
  // calculation to a strided view).
  IRRewriter rewriter(m.getContext());
  SmallVector<CopyOp> fallbackCopies;
  m.walk([&](CopyOp copy) {
    auto srcTy = cast<RankedTensorType>(CopyOperands<CopyOp>::input(copy).getType());
    auto dstTy = cast<ttg::MemDescType>(CopyOperands<CopyOp>::dest(copy).getType());
    auto shared = dyn_cast<ttg::SwizzledSharedEncodingAttr>(dstTy.getEncoding());
    auto padded = dyn_cast<ttg::PaddedSharedEncodingAttr>(dstTy.getEncoding());
    bool linearShared = targetInfo.getArch() == "gfx936" &&
                        isa<ttg::SharedLinearEncodingAttr>(dstTy.getEncoding());
    if (!shared && !padded && !linearShared)
      return;
    auto srcLayout = ttg::toLinearLayout(srcTy);
    auto dstLayout = padded ? padded.getLinearComponent()
                            : ttg::toLinearLayout(dstTy);
    auto mapping = srcLayout.invertAndCompose(dstLayout);
    unsigned vec = copy.getContiguity();
    bool swizzled = shared && shared.getMaxPhase() != 1;
    bool legal = targetInfo.supportsDirectToLdsLoadBitWidth(
        vec * dstTy.getElementTypeBitWidth(), CopyOperands<CopyOp>::isBuffer);
    if (swizzled) {
      legal &= llvm::equal(dstTy.getShape(), dstTy.getAllocShape()) &&
               LLVM::HCU::doesSwizzleInsideWarp(
                   rewriter, mapping, targetInfo.getWarpSize());
    } else {
      legal &= LLVM::HCU::canCoalesceWriteIntoSharedMemory(
          rewriter, mapping, targetInfo.getWarpSize(), vec);
    }
    if constexpr (CopyOperands<CopyOp>::isBuffer) {
      auto it = contiguity.find(copy.getOperation());
      legal &= it != contiguity.end() && it->second >= vec;
    }
    if (!legal)
      fallbackCopies.push_back(copy);
  });
  for (auto copy : fallbackCopies) {
    rewriter.setInsertionPoint(copy);
    auto loc = copy.getLoc();
    Value loaded;
    if constexpr (CopyOperands<CopyOp>::isBuffer) {
      auto offsetsTy = cast<RankedTensorType>(copy.getOffsets().getType());
      auto dstTy = copy.getDest().getType();
      auto loadTy = offsetsTy.clone(dstTy.getElementType());
      auto load = tta::BufferLoadOp::create(rewriter, loc, loadTy, copy.getPtr(),
          copy.getOffsets(), copy.getStride(), copy.getCache(), copy.getMask(), copy.getOther());
      loaded = load.getResult();
    } else {
      auto load = triton::LoadOp::create(rewriter, loc, copy.getSrc(),
          copy.getMask(), copy.getOther(), copy.getCache(), copy.getEvict(),
          copy.getIsVolatile(), rewriter.getStringAttr(""));
      load->setAttr("tt.load.async", rewriter.getBoolAttr(false));
      loaded = load.getResult();
    }
    ttg::LocalStoreOp::create(rewriter, loc, loaded, CopyOperands<CopyOp>::dest(copy));
    // An empty group supplies a completed dependency token; existing waits
    // remain valid, including chains containing other, genuinely async copies.
    auto completed = ttg::AsyncCommitGroupOp::create(rewriter, loc, ValueRange{});
    rewriter.replaceOp(copy, completed.getResult());
  }
}

} // anonymous namespace

class TritonHCUGPUCoalesceAsyncCopyPass
    : public impl::TritonHCUGPUCoalesceAsyncCopyBase<
          TritonHCUGPUCoalesceAsyncCopyPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    MLIRContext *context = &getContext();

    triton::HCU::TargetInfo targetInfo(archGenerationName);

    mlir::RewritePatternSet patterns(context);

    if (!llvm::is_contained({HCU::ISAFamily::CDNA3, HCU::ISAFamily::CDNA4},
                            targetInfo.getISAFamily()))
      return; // This pass is CDNA3 and CDNA4 specific.

    // Precompute the contiguity of all AsyncCopy ops based on the src and
    // mask contiguity/alignment to avoid rebuilding ModuleAxisInfoAnalysis
    // after every IR change.
    HCU::ModuleAxisInfoAnalysis axisAnalysis(m);
    DenseMap<Operation *, unsigned> asyncCopyContiguity;
    bool coalesceBuffer = targetInfo.getArch() == "gfx936";
    m->walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
      unsigned contiguity =
          mlir::LLVM::HCU::getContiguity(copyOp.getSrc(), axisAnalysis);
      if (auto mask = copyOp.getMask()) {
        contiguity =
            std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
      }
      asyncCopyContiguity.insert({copyOp.getOperation(), contiguity});
    });
    if (coalesceBuffer)
      m.walk([&](tta::BufferLoadToLocalOp copy) {
        unsigned contiguity = LLVM::HCU::getContiguity(
            copy.getPtr(), copy.getOffsets(), axisAnalysis);
        if (auto mask = copy.getMask())
          contiguity = std::min<unsigned>(contiguity, axisAnalysis.getMaskAlignment(mask));
        asyncCopyContiguity.insert({copy.getOperation(), contiguity});
      });
    patterns.add<CoalesceAsyncCopyWrites<ttg::AsyncCopyGlobalToLocalOp>>(
        targetInfo, asyncCopyContiguity, context);
    if (coalesceBuffer)
      patterns.add<CoalesceAsyncCopyWrites<tta::BufferLoadToLocalOp>>(
          targetInfo, asyncCopyContiguity, context);

    patterns.add<PreferWavePrivateSyncStores<ttg::LocalStoreOp>,
                 PreferWavePrivateSyncStores<ttg::LocalAllocOp>>(targetInfo, context);

    if (applyPatternsGreedily(m, std::move(patterns)).failed()) {
      signalPassFailure();
      return;
    }

    if (targetInfo.getArch() == "gfx936")
      fallbackInvalidGlobalCopies(m, targetInfo);
    else
      fallbackInvalidCopies<ttg::AsyncCopyGlobalToLocalOp>(m, targetInfo, asyncCopyContiguity);
    if (coalesceBuffer)
      fallbackInvalidCopies<tta::BufferLoadToLocalOp>(m, targetInfo, asyncCopyContiguity);
  }
};

} // namespace mlir
