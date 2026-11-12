#include "TritonHCUGPUTransforms/Passes.h"
#include "Utility.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "hcu/lib/TritonHCUGPUToLLVM/AsyncUtility.h"
#include "hcu/lib/TritonHCUGPUToLLVM/TargetInfo.h"
#include "hcu/lib/TritonHCUGPUTransforms/PipelineUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include <memory>
#include <variant>

#define DEBUG_TYPE "tritonhcugpu-pipeline-lower-loops"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

//===----------------------------------------------------------------------===//
// This file will conditionally allocate lds memory, create local/async load
// operations, and create schedule for these operations. After lowerLoops,
// schedule will be passed to expandLoops and eventually to PipelineExpander.
//===----------------------------------------------------------------------===//

using mlir::triton::HCU::AttrBypassLDS;

namespace mlir {
struct StreamCopyChainOps {
  tt::LoadOp loadOp;
  ttg::MemDescIndexOp subviewOp;
  ttg::LocalStoreOp localStoreOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

struct AsyncCopyChainOps {
  ttg::AsyncCopyGlobalToLocalOp copyOp;
  ttg::AsyncCommitGroupOp commitOp;
  ttg::AsyncWaitOp waitOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

using StreamOpVariant = std::variant<StreamCopyChainOps, AsyncCopyChainOps>;
using LoadToStreamOpMap = llvm::MapVector<Operation *, StreamOpVariant>;

// The per-load policy overrides the backend-wide default. Keep that default
// separate from the loop's buffer policy so one explicit async load does not
// turn its unannotated neighbours into async loads as well.
static bool shouldUseAsyncCopy(Operation *load, bool useAsyncCopy) {
  if (auto attr = load->getAttrOfType<BoolAttr>("tt.load.async"))
    return attr.getValue();
  return useAsyncCopy;
}

static bool hasAsyncLoad(const LoadToInfoMap &loads, bool useAsyncCopy) {
  return llvm::any_of(loads, [&](const auto &entry) {
    return shouldUseAsyncCopy(entry.first, useAsyncCopy);
  });
}

// Joint producer/shared selection: the dot fragment's minimum vector need not
// cap a globally proven 16B transfer at 8B. Keep this opt-in while validating
// its read cost against the existing matrix-read charts.
static bool preferWideAsyncCopies(
    scf::ForOp forOp, LoadToInfoMap &loads, bool useAsyncCopy,
    triton::HCU::ModuleAxisInfoAnalysis &axisInfo,
    const triton::HCU::TargetInfo &target) {
  auto module = forOp->getParentOfType<ModuleOp>();
  auto enabled = module->getAttrOfType<IntegerAttr>(
      "ttg.hcu.prefer_wide_async_copy");
  if (target.getArch() != "gfx936" || !enabled || enabled.getInt() != 1)
    return false;
  bool changed = false;
  for (auto &[operation, info] : loads) {
    auto load = dyn_cast<tt::LoadOp>(operation);
    auto shared = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
        info.sharedEncoding);
    if (!load || !shared || !shouldUseAsyncCopy(load, useAsyncCopy))
      continue;
    auto type = dyn_cast<RankedTensorType>(load.getType());
    auto blocked = type ? dyn_cast<ttg::BlockedEncodingAttr>(type.getEncoding())
                        : ttg::BlockedEncodingAttr{};
    if (!type || type.getRank() != 2 || !blocked ||
        !(type.getElementType().isF16() || type.getElementType().isBF16()) ||
        shared.getOrder().front() != blocked.getOrder().front() ||
        axisInfo.getAlignment(load.getPtr()) < 8 ||
        (load.getMask() && axisInfo.getMaskAlignment(load.getMask()) < 8))
      continue;
    unsigned axis = blocked.getOrder().front();
    auto shape = type.getShape();
    if (shape[axis] < 8 || !llvm::isPowerOf2_64(shape[axis]))
      continue;
    unsigned maxPhase =
        std::min<unsigned>(shared.getMaxPhase(), shape[axis] / 8);
    auto widerShared = ttg::SwizzledSharedEncodingAttr::get(
        load.getContext(), 8, shared.getPerPhase(), maxPhase, shared.getOrder(),
        shared.getCTALayout());
    auto elements = llvm::to_vector(blocked.getSizePerThread());
    elements[axis] = std::max<unsigned>(elements[axis], 8);
    auto encoding = ttg::BlockedEncodingAttr::get(
        load.getContext(), shape, elements, blocked.getOrder(),
        ttg::lookupNumWarps(load),
        ttg::TritonGPUDialect::getThreadsPerWarp(module), blocked.getCTALayout());
    if (encoding == blocked && widerShared == shared)
      continue;
    IRRewriter builder(load);
    // Keep the actual load operation stable: the deserialized schedule uses
    // its identity. Preserve all logical operands, cache/padding attributes
    // and existing users through an ordinary ownership conversion.
    for (auto [index, operand] : llvm::enumerate(load->getOperands())) {
      if (auto tensor = dyn_cast<RankedTensorType>(operand.getType())) {
        auto converted = ttg::ConvertLayoutOp::create(
            builder, load.getLoc(), tensor.cloneWithEncoding(encoding), operand);
        load->setOperand(index, converted.getResult());
      }
    }
    load.getResult().setType(type.cloneWithEncoding(encoding));
    builder.setInsertionPointAfter(load);
    auto restored = ttg::ConvertLayoutOp::create(builder, load.getLoc(), type,
                                                load.getResult());
    load.getResult().replaceAllUsesExcept(restored.getResult(),
                                        restored.getOperation());
    info.sharedEncoding = widerShared;
    changed = true;
  }
  return changed;
}

// Private linear-copy experiment. These are the GPU-verified, zero-control
// canonical charts of two gfx936 matrix-read producer families. They do not
// model the vendor's non-linear circular lane rotation.
static void selectCanonicalMatrixSharedLayout(
    scf::ForOp forOp, LoadToInfoMap &loads, int numStages,
    const tt::HCU::TargetInfo &targetInfo) {
  auto module = forOp->getParentOfType<ModuleOp>();
  auto hint = module->getAttrOfType<IntegerAttr>("ttg.hcu.canonical_matrix");
  auto alt = module->getAttrOfType<IntegerAttr>("ttg.hcu.alt_matrix");
  auto ctas = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  auto warps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto lanes = module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  if (!hint || (hint.getInt() != 1 && hint.getInt() != 2) ||
      (alt && alt.getInt() != 0) ||
      targetInfo.getArch() != "gfx936" || !ctas || ctas.getInt() != 1 ||
      !warps || (warps.getInt() != 4 && warps.getInt() != 8) ||
      !lanes || lanes.getInt() != 64 || loads.size() != 2 ||
      (numStages != 2 && numStages != 3))
    return;
  auto dots = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
  if (dots.size() != 1 ||
      llvm::to_vector(forOp.getBody()->getOps<tt::LoadOp>()).size() != 2 ||
      llvm::any_of(loads, [&](const auto &entry) {
        return !isa<tt::LoadOp>(entry.first) || !entry.second.sharedEncoding ||
               entry.second.distToUse != 0 ||
               entry.second.use != dots.front().getOperation();
      }))
    return;
  auto dot = dots.front();
  auto aCvt = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
  auto bCvt = dot.getB().getDefiningOp<ttg::ConvertLayoutOp>();
  auto aLoad = aCvt ? aCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
  auto bLoad = bCvt ? bCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
  if (!aLoad || !bLoad || aLoad == bLoad ||
      !aLoad.getResult().hasOneUse() || !bLoad.getResult().hasOneUse() ||
      !aCvt.getResult().hasOneUse() || !bCvt.getResult().hasOneUse() ||
      !loads.count(aLoad.getOperation()) || !loads.count(bLoad.getOperation()) ||
      !isa<RankedTensorType>(aLoad.getPtr().getType()) ||
      !isa<RankedTensorType>(bLoad.getPtr().getType()))
    return;
  auto ty = cast<RankedTensorType>(dot.getB().getType());
  auto enc = dyn_cast<ttg::DotOperandEncodingAttr>(ty.getEncoding());
  auto mma = enc ? dyn_cast<ttg::AMDMfmaEncodingAttr>(enc.getParent())
                 : ttg::AMDMfmaEncodingAttr{};
  if (!mma || enc.getOpIdx() != 1 || ty.getRank() != 2 ||
      mma.getMmacLayout() != ttg::MmacLayout::LEGACY ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      !(ty.getElementType().isF16() || ty.getElementType().isBF16()) ||
      (enc.getKWidth() != 4 && enc.getKWidth() != 8) ||
      ttg::getOrderForMemory(cast<RankedTensorType>(bLoad.getType())).front() != 1)
    return;
  auto shape = ty.getShape();
  SmallVector<unsigned> kBases;
  if (shape[0] == 32 && shape[1] == 128)
    kBases = {2, 8, 1, 16, 4};
  else if (shape[0] == 16 && shape[1] == 256)
    kBases = {2, 1, 4, 8};
  else
    return;
  std::vector<std::vector<int32_t>> bases;
  for (int64_t n = 1; n < shape[1]; n *= 2)
    bases.push_back({0, int32_t(n)});
  for (unsigned k : kBases)
    bases.push_back({int32_t(k), 0});
  auto *ctx = forOp.getContext();
  tt::LinearLayout canonical(
      {{StringAttr::get(ctx, "offset"), bases},
       {StringAttr::get(ctx, "block"), {}}},
      {StringAttr::get(ctx, "dim0"), StringAttr::get(ctx, "dim1")});
  loads[bLoad.getOperation()].sharedEncoding =
      ttg::SharedLinearEncodingAttr::get(ctx, canonical, 16);

  // The separate private mode also tests the true linear chart of A. Keep
  // mode 1's B-only experiment unchanged. Ordinary raw G2L copies use the
  // chart's actual source distribution; no indexed descriptor or m0 rotation.
  if (hint.getInt() != 2)
    return;
  auto aTy = cast<RankedTensorType>(dot.getA().getType());
  auto aEnc = dyn_cast<ttg::DotOperandEncodingAttr>(aTy.getEncoding());
  if (!aEnc || aEnc.getOpIdx() != 0 || aEnc.getParent() != mma ||
      aTy.getRank() != 2 || aTy.getElementType() != ty.getElementType() ||
      (aEnc.getKWidth() != 4 && aEnc.getKWidth() != 8) ||
      ttg::getOrderForMemory(cast<RankedTensorType>(aLoad.getType())).front() != 1)
    return;
  if (auto canonical = getHCUCanonicalASharedEncoding(ctx, aTy.getShape()))
    loads[aLoad.getOperation()].sharedEncoding = canonical;
}

// Private ALT experiment. Keep the ordinary synchronous two-load GEMM path
// and its existing producer lowering; only replace B's shared encoding.
static void selectLegacyAltSharedLayout(
    scf::ForOp forOp, LoadToInfoMap &loads, bool useAsyncCopy, int numStages,
    const tt::HCU::TargetInfo &targetInfo) {
  auto module = forOp->getParentOfType<ModuleOp>();
  auto attr = module->getAttrOfType<IntegerAttr>("ttg.hcu.alt_matrix");
  auto canonical =
      module->getAttrOfType<IntegerAttr>("ttg.hcu.canonical_matrix");
  if ((canonical && (canonical.getInt() == 1 || canonical.getInt() == 2)) ||
      !attr || (attr.getInt() != 2 && attr.getInt() != 3) ||
      targetInfo.getArch() != "gfx936" || loads.size() != 2 ||
      (numStages != 2 && numStages != 3) || hasAsyncLoad(loads, useAsyncCopy))
    return;
  auto numCTAs = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  auto numWarps = module->getAttrOfType<IntegerAttr>("ttg.num-warps");
  auto prefetch = module->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
  // The verified-WAR experiment currently proves SwizzledShared only.
  if (!numCTAs || numCTAs.getInt() != 1 || !numWarps ||
      (numWarps.getInt() != 4 && numWarps.getInt() != 8) ||
      (prefetch && prefetch.getInt() != 0 && prefetch.getInt() != 2))
    return;
  auto dots = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
  if (dots.size() != 1 ||
      llvm::to_vector(forOp.getBody()->getOps<tt::LoadOp>()).size() != 2 ||
      llvm::any_of(loads, [&](const auto &entry) {
        return !isa<tt::LoadOp>(entry.first) || !entry.second.sharedEncoding ||
               entry.second.distToUse != 0 ||
               entry.second.use != dots.front().getOperation();
      }))
    return;
  auto dot = dots.front();
  auto aCvt = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
  auto bCvt = dot.getB().getDefiningOp<ttg::ConvertLayoutOp>();
  auto aLoad = aCvt ? aCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
  auto bLoad = bCvt ? bCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
  if (!aLoad || !bLoad || aLoad == bLoad ||
      !aLoad.getResult().hasOneUse() || !bLoad.getResult().hasOneUse() ||
      !aCvt.getResult().hasOneUse() || !bCvt.getResult().hasOneUse() ||
      !loads.count(aLoad.getOperation()) || !loads.count(bLoad.getOperation()) ||
      !isa<RankedTensorType>(aLoad.getPtr().getType()) ||
      !isa<RankedTensorType>(bLoad.getPtr().getType()))
    return;
  auto bTy = cast<RankedTensorType>(dot.getB().getType());
  auto enc = dyn_cast<ttg::DotOperandEncodingAttr>(bTy.getEncoding());
  auto mma = enc ? dyn_cast<ttg::AMDMfmaEncodingAttr>(enc.getParent())
                 : ttg::AMDMfmaEncodingAttr{};
  if (!mma || mma.getMmacLayout() != ttg::MmacLayout::LEGACY ||
      mma.getIsTransposed() ||
      enc.getOpIdx() != 1 || bTy.getRank() != 2 ||
      !(bTy.getElementType().isF16() || bTy.getElementType().isBF16()) ||
      (enc.getKWidth() != 4 && enc.getKWidth() != 8) ||
      mma.getInstrShape() != ArrayRef<unsigned>({16, 16, 16}) ||
      llvm::any_of(mma.getTilesPerWarp(), [](unsigned n) { return n != 1; }))
    return;
  auto shape = bTy.getShape();
  if (shape[0] < 16 || shape[0] > 64 || shape[1] < 64 || shape[1] > 256 ||
      !llvm::isPowerOf2_64(shape[0]) || !llvm::isPowerOf2_64(shape[1]) ||
      (enc.getKWidth() == 8 && shape[0] < 32))
    return;

  auto *ctx = forOp.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto dotLL = ttg::toLinearLayout(bTy);
  unsigned chosen = attr.getInt();
  if (chosen >= dotLL.getInDimSizeLog2(reg))
    return;
  auto basis = dotLL.getBasis(reg, chosen);
  unsigned flatBasis = basis[0] * shape[1] + basis[1];
  if (!llvm::isPowerOf2_32(flatBasis))
    return;
  unsigned pivot = llvm::Log2_32(flatBasis);
  if (pivot < 3)
    return;
  // SharedLinear is physical offset -> logical [K,N]. Move the chosen
  // row-major bit to physical bit0; this is the inverse permutation.
  std::vector<std::vector<int32_t>> bases;
  unsigned elements = shape[0] * shape[1];
  for (unsigned bit = 0; bit < llvm::Log2_32(elements); ++bit) {
    unsigned originalBit = bit == 0 ? pivot : (bit <= pivot ? bit - 1 : bit);
    unsigned logical = 1u << originalBit;
    bases.push_back({int32_t(logical / shape[1]),
                     int32_t(logical % shape[1])});
  }
  auto offset = StringAttr::get(ctx, "offset");
  auto block = StringAttr::get(ctx, "block");
  tt::LinearLayout sharedLL({{offset, bases}, {block, {}}},
                            {StringAttr::get(ctx, "dim0"),
                             StringAttr::get(ctx, "dim1")});
  loads[bLoad.getOperation()].sharedEncoding =
      ttg::SharedLinearEncodingAttr::get(ctx, sharedLL, 16);
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::HCU::TargetInfo &targetInfo);

AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
                                  Value extractIdx, int contiguity) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  auto copyOp = ttg::AsyncCopyGlobalToLocalOp::create(
      builder, loc, loadOp.getPtr(), viewLoad, loadOp.getMask(),
      loadOp.getOther(), loadOp.getCache(), loadOp.getEvict(),
      loadOp.getIsVolatile(), contiguity);
  auto commitOp =
      ttg::AsyncCommitGroupOp::create(builder, loc, copyOp->getResult(0));
  ttg::AsyncWaitOp waitOp =
      ttg::AsyncWaitOp::create(builder, loc, commitOp->getResult(0), 0);

  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, loadOp->getResult(0), viewLoad, waitOp);

  return {copyOp, commitOp, waitOp, maybeSharedLoad};
}

void scheduleLocalLoad(ttg::LocalLoadOp localLoadOp,
                       tt::CoarseSchedule &schedule, int stage,
                       const tt::CoarseSchedule::Cluster &cluster) {
  schedule.insert(localLoadOp, stage, cluster);
  // If its only user is a ConvertLayout, we place it into the same stage so
  // it can be folded by a later pass
  if (localLoadOp->hasOneUse()) {
    auto cvt = *localLoadOp->getUsers().begin();
    if (isa<ttg::ConvertLayoutOp>(cvt)) {
      schedule.insert(cvt, stage, cluster);
    }
  }
}

StreamCopyChainOps createStreamCopy(tt::LoadOp loadOp, Value alloc,
                                    Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  tt::LoadOp newLoadOp = cast<tt::LoadOp>(builder.clone(*loadOp));
  // The stream pipeline has already selected synchronous fallback.
  newLoadOp->setAttr("tt.load.async", builder.getBoolAttr(false));
  auto storeOp = ttg::LocalStoreOp::create(builder, loc, newLoadOp, viewLoad);
  auto maybeLocalLoad =
      tt::replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  return {newLoadOp, viewLoad, storeOp, maybeLocalLoad};
}

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// and |vecSize| with which dot operand |inputValue| is fed into if possible.
ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue, unsigned *opIdx,
                                        unsigned *vecSize) {
  if (!inputValue.hasOneUse())
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  LDBG("getDotEncoding user: " << *user);
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto operandType = cast<RankedTensorType>(inputValue.getType());
    *vecSize = ttg::toLinearLayout(operandType).getNumConsecutiveInOut();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }

  return getDotEncoding(user->getResult(0), opIdx, vecSize);
}

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
std::optional<ttg::SharedEncodingTrait> getSharedEncIfAllUsersAreDotEnc(
    Operation *loadOp, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
    const tt::HCU::TargetInfo &targetInfo, bool useAsyncCopy) {
  assert(loadOp);
  Value loadedValue = loadOp->getResult(0);
  llvm::SmallVector<ttg::SharedEncodingTrait> sharedEncs;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SharedEncodingTrait tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SharedEncodingTrait>(memDesc.getEncoding());
      // If the immediate user is ttg::LocalAllocOp, likely it's created in
      // TritonHCUGPUOptimizeDotOperands. We should just respect it.
      if (!isa<ttg::LocalAllocOp>(user) &&
          !getSharedEncIfAllUsersAreDotEnc(user, axisInfoAnalysis, targetInfo,
                                           useAsyncCopy)) {
        return std::nullopt;
      }
      LDBG("Deduced shared encoding candidate from memDesc: " << tempAttr);
      sharedEncs.push_back(tempAttr);
    } else {
      if (!(isa<ttg::ConvertLayoutOp>(user) ||
            user->hasTrait<OpTrait::LocalLoadTrait>()))
        return std::nullopt;

      auto srcTy = cast<ttg::TensorOrMemDesc>(loadedValue.getType());
      auto ctaLayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = getOrderForMemory(srcTy);
      unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
      SmallVector<unsigned> sharedOrder;
      int rank = order.size();
      // TODO rework this when shared -> dotOperand conversions support
      // arbitrary shared memory ordering
      if (rank == 3) {
        // Move the batch dimension (dim #0) to be the last so that it will be
        // the slowest varying dimension.
        for (unsigned i = 0; i < rank; ++i)
          if (order[i] != 0)
            sharedOrder.emplace_back(order[i]);
        sharedOrder.emplace_back(0);
      } else {
        sharedOrder = order;
      }

      auto userResEnc = cast<ttg::TensorOrMemDesc>(userResType).getEncoding();
      if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(userResEnc)) {
        // Determine if we can use padded layouts and fallback to swizzled
        // layouts if not
        bool canUseAsyncCopy = false;
        if (useAsyncCopy && isa<tt::LoadOp>(loadOp)) {
          // We pass numBuffers=2 because we assume the schedule will not
          // determine a single buffer (which does not work with AsyncCopy)
          canUseAsyncCopy = canBeConvertedToAsyncLoad(
              2, cast<tt::LoadOp>(loadOp), {}, axisInfoAnalysis, targetInfo);
        }
        tempAttr = composePaddedLayout(targetInfo, dotOpEnc, srcTy, sharedOrder,
                                       canUseAsyncCopy);
        if (!tempAttr) {
          tempAttr = ttg::SwizzledSharedEncodingAttr::get(
              loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
              ctaLayout, bitWidth, /*needTrans=*/false);
        }
        auto mma = dyn_cast<ttg::AMDMfmaEncodingAttr>(dotOpEnc.getParent());
        if (!useAsyncCopy && targetInfo.getArch() == "gfx936" && mma &&
            mma.getMmacLayout() == ttg::MmacLayout::LEGACY &&
            dotOpEnc.getOpIdx() == 1 && rank == 2 && sharedOrder[0] == 1 &&
            (srcTy.getElementType().isF16() ||
             srcTy.getElementType().isBF16()) &&
            srcTy.getShape()[0] >= 16 && srcTy.getShape()[1] >= 64) {
          // Legacy MMAC distributes four K groups across a wave. For an
          // N-contiguous B tile, the generic MFMA no-swizzle rule maps those
          // groups to the same LDS banks. Spread groups of four K rows over
          // 16-column chunks while preserving vectorized producer stores.
          // Async copies retain their own wave-coalesced destination policy.
          tempAttr = ttg::SwizzledSharedEncodingAttr::get(
              loadedValue.getContext(), 16, 4, 4, sharedOrder, ctaLayout);
        }
        LDBG("Deduced shared encoding candidate from dot layout: " << tempAttr);
        sharedEncs.push_back(tempAttr);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        unsigned vecSize;
        if (auto mfmaEnc = getDotEncoding(userResult, &opIdx, &vecSize)) {
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = mfmaEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
          LDBG("Deduced shared encoding candidate from mfma layout: "
               << tempAttr);
          sharedEncs.push_back(tempAttr);
        }
      }
    }
  }

  auto equalSharedEncIgnoreVec = [](ttg::SwizzledSharedEncodingAttr a,
                                    ttg::SwizzledSharedEncodingAttr b) {
    if (!a || !b)
      return false;
    return (a.getPerPhase() == b.getPerPhase() &&
            a.getMaxPhase() == b.getMaxPhase() &&
            a.getOrder() == b.getOrder() &&
            a.getCTALayout() == b.getCTALayout());
  };
  if (sharedEncs.empty() || !sharedEncs.front())
    return std::nullopt;
  auto maxVecSharedEnc = sharedEncs.front();

  // TODO add support for padded layouts. Right now they will use a separate
  // allocation
  for (auto sharedEnc : llvm::drop_begin(sharedEncs, 1)) {
    auto maybeSwizzShared =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(sharedEnc);
    auto maybeSwizzMaxVec =
        dyn_cast<ttg::SwizzledSharedEncodingAttr>(maxVecSharedEnc);

    if (!equalSharedEncIgnoreVec(maybeSwizzShared, maybeSwizzMaxVec)) {
      LDBG("Incompatible shared encodings");
      return std::nullopt;
    }
    if (maybeSwizzShared.getVec() > maybeSwizzMaxVec.getVec()) {
      maxVecSharedEnc = sharedEnc;
    }
  }

  LDBG("Deduced shared encoding: " << maxVecSharedEnc);

  return maxVecSharedEnc;
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               ttg::SharedEncodingTrait sharedEnc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::HCU::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  using tt::HCU::ISAFamily;
  if (sharedEnc && llvm::is_contained({ISAFamily::CDNA3, ISAFamily::CDNA4},
                                      targetInfo.getISAFamily())) {
    // Compute the final vecSize we can use for the combination of
    // sourceEncoding and sharedEncoding. We can only use AsyncCopy if the
    // target supports the requested or a smaller vecSize because we cannot
    // stride when loading directly to lds on GFX9
    auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
    auto regLayout = triton::gpu::toLinearLayout(srcTy);
    // It's the allocation so we trim the multibuffer dimension
    auto srcShape = srcTy.getShape();
    triton::LinearLayout sharedLayout;
    auto paddedEnc = dyn_cast<triton::gpu::PaddedSharedEncodingAttr>(sharedEnc);
    if (paddedEnc) {
      sharedLayout = paddedEnc.getLinearComponent();
    } else {
      sharedLayout = triton::gpu::toLinearLayout(srcShape, sharedEnc);
    }
    auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

    unsigned elemBitWidth = tt::getPointeeBitWidth(srcTy);
    unsigned vecSize = regToSharedLayout.getNumConsecutiveInOut();
    if (paddedEnc)
      vecSize = std::min(vecSize, paddedEnc.getMinInterval());

    if (fitToValidDirectToLdsVecSize(vecSize, elemBitWidth, targetInfo) == 0)
      return false;
  }

  // Checks whether the global pointer's contiguity and mask alignment allows
  // for at least 32 bit wide loads
  return triton::canBeConvertedToAsyncLoad(loadOp, axisInfoAnalysis);
}

// Convert load ops into shared memory allocation loads and apply
// multi-buffering based on the required number of buffers.
LoadToStreamOpMap
createStreamOps(const LoadToInfoMap &loadToInfo, scf::ForOp &forOp,
                const int &numBuffers, bool useAsyncCopy,
                tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  IRRewriter builder(forOp);
  Location loc = forOp.getLoc();
  Value minusOne = arith::ConstantIntOp::create(builder, loc, -1, 32);
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 32);
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      arith::ConstantIntOp::create(builder, loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = arith::AddIOp::create(builder, loc, extractIdx, one);
  Value cndExt = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                       extractIdx, numBuffersVal);
  extractIdx = arith::SelectOp::create(builder, loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[l, info] : loadToInfo) {
    if (!info.sharedEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    // Create an allocation that can hold distance number of loadOp shapes.
    auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
    Value alloc = triton::createAlloc(forOp, ty, loadOp->getLoc(),
                                      info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    auto arch = getHCUArch(loadOp->getParentOfType<ModuleOp>());
    triton::HCU::TargetInfo targetInfo(arch ? arch->str() : "");

    // Replace the old load with multi-buffered loads
    if (shouldUseAsyncCopy(loadOp, useAsyncCopy) &&
        canBeConvertedToAsyncLoad(numBuffers, loadOp, info.sharedEncoding,
                                  axisInfoAnalysis, targetInfo)) {
      unsigned vec = axisInfoAnalysis.getContiguity(loadOp.getPtr());
      if (auto mask = loadOp.getMask())
        vec = std::min<unsigned>(vec, axisInfoAnalysis.getMaskAlignment(mask));
      loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx, vec);
    } else {
      loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
    }
  }

  return loadToStreamOp;
}

static void dumpSchedule(tt::CoarseSchedule &schedule, llvm::StringRef msg) {
  LLVM_DEBUG({
    llvm::dbgs() << "\n";
    LDBG(msg);
    schedule.dump();
  });
};

namespace SingleDotSchedule {
using namespace mlir::SingleDotSchedule;
using ClusterMap = DenseMap<tt::CoarseSchedule::ClusterHash, int>;

ClusterMap createClusterMap(tt::CoarseSchedule &schedule) {
  DenseMap<tt::CoarseSchedule::ClusterHash, int> clusterMap;
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(cluster);
    clusterMap[clusterHash] = *cluster;
  }

  return clusterMap;
}

// Remap global and compute clusters to the right place
void remapClusters(tt::CoarseSchedule &schedule, ClusterMap clusterMap,
                   Clusters &clusters) {
  for (auto &[op, stageAndCluster] : schedule.opToStageAndCluster) {
    auto [stage, cluster] = stageAndCluster;
    tt::CoarseSchedule::ClusterHash clusterHash =
        tt::CoarseSchedule::hashCluster(stageAndCluster.second);
    int oldClusterId = clusterMap[clusterHash];
    if (oldClusterId == 0) {
      stageAndCluster.second = clusters[SCHED_GLOBAL_LOAD];
    } else {
      assert(oldClusterId == 1);
      stageAndCluster.second = clusters[SCHED_COMPUTE];
    }
  }
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int &numBuffers, bool useAsyncCopy, bool waitAtTail,
                           int globalPrefetch, int localPrefetch,
                           Clusters &clusters, tt::CoarseSchedule &schedule) {
  LDBG("Init SingleDotSchedule");
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool pairedGlobalLoadLocalStore = stages[SCHED_LOCAL_STORE] == 0;
  stages[SCHED_LOCAL_STORE] += maxDist;
  if (waitAtTail) {
    stages[SCHED_ASYNC_WAIT] = std::max(0, stages[SCHED_LOCAL_LOAD] - 1);
  }

  LDBG(
      "Stage schedule:" << "  GLOBAL_LOAD stage = " << stages[SCHED_GLOBAL_LOAD]
                        << ", LOCAL_STORE stage = " << stages[SCHED_LOCAL_STORE]
                        << ", LOCAL_LOAD stage = " << stages[SCHED_LOCAL_LOAD]
                        << ", COMPUTE stage = " << stages[SCHED_COMPUTE]
                        << ", ASYNC_WAIT stage = " << stages[SCHED_ASYNC_WAIT]
                        << "; total = " << numStages);

  if (stages[SCHED_LOCAL_STORE] >= numStages ||
      stages[SCHED_LOCAL_STORE] > stages[SCHED_LOCAL_LOAD]) {
    LDBG("Invalid stage schedule");
    return failure();
  }

  // Calculate the number of buffers needed for each load.
  // TODO: Use the precise number of buffers needed by the particular load.
  numBuffers =
      std::max(1, stages[SCHED_LOCAL_LOAD] - stages[SCHED_LOCAL_STORE]);
  // If we use AsyncCopy we need one more buffer since we are not using a
  // register buffer
  if (useAsyncCopy) {
    numBuffers += 1;
  }

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  // In case we use async_copy with pingpong, we need to place async_wait at
  // the end of the previous iteration, so it can guarantee the correct
  // dependency when warp0 and warp1 are pipelined.
  int asyncWaitCluster = waitAtTail ? 4 : 0;
  // If tt.load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else
  //   Initiate ttg.local_store before tt.load
  int globalLoadCluster = 1;
  int localStoreCluster = 3;
  if (!pairedGlobalLoadLocalStore) {
    globalLoadCluster = 3;
    localStoreCluster = 2;
  }

  // If ttg.local_load and ttg.local_store are in the same stage
  //   spread them apart to allow overlap with compute
  // else if they share the buffer
  //   ttg.local_load must come first
  // else
  //   schedule ttg.local_load in the middle
  int localLoadCluster = globalLoadCluster;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_LOCAL_STORE]) {
    localLoadCluster = std::max(3, localStoreCluster + 1);
  } else if (numBuffers == 1 && localLoadCluster >= localStoreCluster) {
    // For 1 buffer, ttg.local_load must occur before ttg.local_store
    localLoadCluster = localStoreCluster - 1;
  }

  // Schedule compute with ttg.local_load if paired
  // otherwise, schedule in the middle
  int computeCluster = 2;
  if (stages[SCHED_LOCAL_LOAD] == stages[SCHED_COMPUTE]) {
    computeCluster = localLoadCluster;
  }

  if (globalPrefetch == 1 && localPrefetch == 1 && numStages == 3) {
    // Keep the next global prefetch after the LDS publication barrier, so
    // its vmcnt need not be drained before the prefetched LDS read.
    localStoreCluster = 1;
    localLoadCluster = 2;
    globalLoadCluster = 3;
    computeCluster = 4;
  }

  // Create a hash map to associate cluster hash in old schedule with its
  // clusterID
  ClusterMap clusterMap = createClusterMap(schedule);

  // Make assignments
  Clusters clusterVec;
  schedule.clusters.clear();
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  remapClusters(schedule, clusterMap, clusters);

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
  auto [loadStage, loadCluster] = schedule[loadOp];
  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);
  // If the LocalLoads are scheduled to a later stage than AsyncCopy we need to
  // place the AsyncCopy prefetches after the AsyncWaits which create a barrier
  // to ensure all warps are finished reading the shared buffer we will write
  // into. This is done by scheduling AsyncWait as the first cluster.
  // If AsyncCopy and LocalLoads are in the same stage we do not assign a
  // schdule so they are placed before the LocalLoads
  if (loadStage != stages[SCHED_LOCAL_LOAD])
    schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
                    clusters[SCHED_ASYNC_WAIT]);

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps,
                        tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                        const Stages &stages, const Clusters &clusters) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];

  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE]) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, stages, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, stages, clusters);
    }
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::HCU::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    int numStages, bool useAsyncCopy, bool waitAtTail) {
  LDBG("SingleDotSchedule::updateSchedule");
  Stages stages;
  Clusters clusters;

  int maxDist = 0;
  for (auto &[l, info] : loadToInfo) {
    maxDist = std::max(maxDist, info.distToUse);
  }

  bool loopUsesAsyncCopy = hasAsyncLoad(loadToInfo, useAsyncCopy);
  int globalPrefetch = 0;
  int localPrefetch = 0;
  auto module = forOp->getParentOfType<ModuleOp>();
  auto prefetchAttr =
      module->getAttrOfType<IntegerAttr>("ttg.hcu.local_prefetch");
  auto arch = getHCUArch(module);
  // Experimental opt-in for the direct, synchronous two-load GEMM path.
  // Reuse StreamPipeline's local-prefetch schedule: LDS reads produce the
  // next iteration's dot operands, carried in VGPRs by PipelineExpander.
  // Leave unsupported loops and every default compilation unchanged.
  bool localOnly = prefetchAttr && prefetchAttr.getInt() == 1 && numStages == 2;
  bool publicThreeStage =
      prefetchAttr && prefetchAttr.getInt() == 1 && numStages == 3;
  bool doubleBuffered =
      prefetchAttr &&
      (prefetchAttr.getInt() == 3 || prefetchAttr.getInt() == 4) && numStages == 3;
  bool twoLevel =
      publicThreeStage ||
      (prefetchAttr && prefetchAttr.getInt() == 2 && numStages == 3) ||
      doubleBuffered;
  if ((localOnly || twoLevel) && arch &&
      *arch == "gfx936" && !loopUsesAsyncCopy &&
      (!publicThreeStage || !useAsyncCopy) &&
      maxDist == 0 && loadToInfo.size() == 2) {
    auto dots = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
    if (dots.size() == 1 && dots.front().getType().getRank() == 2 &&
        llvm::all_of(loadToInfo, [&](const auto &entry) {
          auto load = dyn_cast<tt::LoadOp>(entry.first);
          if (!load || !entry.second.sharedEncoding ||
              entry.second.use != dots.front().getOperation())
            return false;
          if (publicThreeStage) {
            if (load.getIsVolatile())
              return false;
            if (Attribute policy = load->getAttr("tt.load.async")) {
              auto flag = dyn_cast<BoolAttr>(policy);
              if (!flag || flag.getValue())
                return false;
            }
          }
          return true;
        })) {
      globalPrefetch = twoLevel ? 1 : 0;
      localPrefetch = 1;
    }
  }

  // Private async diagnostic: use the existing StreamPipeline stage model
  // with one iteration of LDS-to-VGPR prefetch. Keep the buffer formula and
  // complete generic RAW/WAR barrier analysis unchanged.
  auto ctas = module->getAttrOfType<IntegerAttr>("ttg.num-ctas");
  auto lanes = module->getAttrOfType<IntegerAttr>("ttg.threads-per-warp");
  if (prefetchAttr && prefetchAttr.getInt() == 5 && numStages == 3 &&
      ctas && ctas.getInt() == 1 && lanes && lanes.getInt() == 64 &&
      !waitAtTail && arch && *arch == "gfx936" && loopUsesAsyncCopy && maxDist == 0 &&
      loadToInfo.size() == 2) {
    auto dots = llvm::to_vector(forOp.getBody()->getOps<tt::DotOp>());
    if (dots.size() == 1 &&
        llvm::to_vector(forOp.getBody()->getOps<tt::LoadOp>()).size() == 2 &&
        llvm::all_of(forOp.getBody()->without_terminator(), [](Operation &op) {
          return op.getNumRegions() == 0 &&
                 (isa<tt::LoadOp>(op) || isMemoryEffectFree(&op));
        })) {
      auto dot = dots.front();
      auto aCvt = dot.getA().getDefiningOp<ttg::ConvertLayoutOp>();
      auto bCvt = dot.getB().getDefiningOp<ttg::ConvertLayoutOp>();
      auto a = aCvt ? aCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
      auto b = bCvt ? bCvt.getSrc().getDefiningOp<tt::LoadOp>() : tt::LoadOp{};
      auto mma = dyn_cast<ttg::AMDMfmaEncodingAttr>(dot.getType().getEncoding());
      auto aTy = a ? dyn_cast<RankedTensorType>(a.getType()) : RankedTensorType{};
      auto bTy = b ? dyn_cast<RankedTensorType>(b.getType()) : RankedTensorType{};
      if (a && b && aTy && bTy && a != b && !a.getIsVolatile() &&
          !b.getIsVolatile() && isa<RankedTensorType>(a.getPtr().getType()) &&
          isa<RankedTensorType>(b.getPtr().getType()) &&
          a.getResult().hasOneUse() &&
          b.getResult().hasOneUse() && aCvt.getResult().hasOneUse() &&
          bCvt.getResult().hasOneUse() && dot.getType().getRank() == 2 &&
          mma && mma.getMmacLayout() == ttg::MmacLayout::LEGACY &&
          mma.getInstrShape() == ArrayRef<unsigned>({16, 16, 16}) &&
          aTy.getElementType() == bTy.getElementType() &&
          (aTy.getElementType().isF16() || aTy.getElementType().isBF16()) &&
          llvm::all_of(loadToInfo, [&](const auto &entry) {
            return (entry.first == a.getOperation() ||
                    entry.first == b.getOperation()) &&
                   entry.second.sharedEncoding &&
                   entry.second.use == dot.getOperation();
          })) {
        localPrefetch = 1;
      }
    }
  }

  int numBuffers = 1;
  if (failed(initSchedule(maxDist, stages, numStages, numBuffers,
                          loopUsesAsyncCopy, waitAtTail,
                          globalPrefetch, localPrefetch,
                          clusters, schedule)))
    return;

  // Private diagnostic: retain the two-level schedule and all barrier analysis,
  // but alternate two LDS slots. LS and LL remain in the same stage, so this
  // allocation override does not change their cluster ordering.
  if (doubleBuffered && globalPrefetch == 1 && localPrefetch == 1)
    numBuffers = 2;

  // Reuse the existing synchronous local-prefetch schedule, without adding
  // another stage of global-load register payloads. Generic membar remains
  // conservative unless its final-CF two-slot proof recognizes this loop.
  auto syncBuffers = module->getAttrOfType<IntegerAttr>("ttg.hcu.sync_lds_buffers");
  if (syncBuffers && syncBuffers.getInt() == 2 && localOnly && arch &&
      *arch == "gfx936" && !loopUsesAsyncCopy && globalPrefetch == 0 &&
      localPrefetch == 1 && maxDist == 0)
    numBuffers = 2;

  // Convert the loads into shared memory allocations and loads from them.
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);

  scheduleStreamOps(loadToStreamOps, schedule, stages, clusters);
  dumpSchedule(schedule, "Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dependencies:");
  ttg::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dist 1:");
  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  ttg::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule(schedule, "Final coarse schedule:");
}
} // namespace SingleDotSchedule

namespace ChainedDotSchedule {
using namespace mlir::ChainedDotSchedule;

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;

  schedule.insert(copyOp, loadStage, loadCluster);
  // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
  // later UpdateAsyncWaitCount pass can deduce better waitcnts
  schedule.insert(commitOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_1, clusters[CLUSTER_ASYNC_WAIT_1]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_1,
                        clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(waitOp, STAGE_LOCAL_LOAD_2, clusters[CLUSTER_ASYNC_WAIT_2]);
    if (maybeLocalLoadOp)
      scheduleLocalLoad(maybeLocalLoadOp, schedule, STAGE_LOCAL_LOAD_2,
                        clusters[CLUSTER_LOCAL_LOAD_2]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps, tt::LoadOp loadOp,
                        tt::CoarseSchedule &schedule,
                        const ChainedDotClusters &clusters) {
  auto [loadStage, loadCluster] = schedule[loadOp];
  auto [copyOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  schedule.insert(copyOp, loadStage, loadCluster);

  if (loadStage == STAGE_GLOBAL_LOAD_1) {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_1,
                    clusters[CLUSTER_LOCAL_WRITE_1]);

    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_1,
                      clusters[CLUSTER_LOCAL_LOAD_1]);
  } else {
    schedule.insert(subviewOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    schedule.insert(localStoreOp, STAGE_LOCAL_WRITE_2,
                    clusters[CLUSTER_LOCAL_WRITE_2]);
    if (maybeLocalLoadOp)
      schedule.insert(maybeLocalLoadOp, STAGE_LOCAL_LOAD_2,
                      clusters[CLUSTER_LOCAL_LOAD_2]);
  }

  if (maybeLocalLoadOp) {
    if (auto cvt = dyn_cast<ttg::ConvertLayoutOp>(
            *maybeLocalLoadOp->getUsers().begin())) {
      auto [localLoadStage, localLoadCluster] = schedule[maybeLocalLoadOp];
      schedule.insert(cvt, localLoadStage, localLoadCluster);
    }
  }
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule,
                       const ChainedDotClusters &clusters) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    if (!loadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, loadOp, schedule, clusters);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, clusters);
    }
  }
}

void updateSchedule(scf::ForOp &forOp, const LoadToInfoMap &loadToInfo,
                    tt::CoarseSchedule &schedule,
                    triton::HCU::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                    bool useAsyncCopy) {
  LDBG("ChainedDotSchedule::updateSchedule");
  ChainedDotClusters clusters;
  int cnt = clusters.size() - schedule.clusters.size();
  for (int i = 0; i < cnt; i++) {
    schedule.clusters.newAtBack();
  }
  auto it = schedule.clusters.begin();
  for (int i = 0; i < clusters.size(); i++, it++) {
    clusters[i] = it;
  }

  // Convert the loads into shared memory allocations and loads from them.
  // TODO support different numBuffers
  int numBuffers = hasAsyncLoad(loadToInfo, useAsyncCopy) ? 2 : 1;
  auto loadToStreamOps = createStreamOps(loadToInfo, forOp, numBuffers,
                                         useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOps, schedule, clusters);

  for (auto [l, _] : loadToInfo) {
    schedule.erase(l);
    l->erase();
  }

  scheduleDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule(schedule, "Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster lastCluster = clusters.back();
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, lastCluster);
  dumpSchedule(schedule, "Final coarse schedule:");
}
} // namespace ChainedDotSchedule

void lowerLoop(scf::ForOp forOp,
               triton::HCU::ModuleAxisInfoAnalysis &axisInfoAnalysis,
               bool useAsyncCopy, bool usePingpong) {
  tt::CoarseSchedule schedule;
  if (failed(schedule.deSerialize(forOp, /*normalizeClusterId=*/false))) {
    return;
  }

  dumpSchedule(schedule, "[lowerLoops]deserialized schedule:");

  int numStages = schedule.getNumStages();

  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      getIndirectLevel(axisInfoAnalysis, forOp, numStages);

  auto arch = getHCUArch(forOp->getParentOfType<ModuleOp>());
  triton::HCU::TargetInfo targetInfo(arch ? arch->str() : "");

  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    if (load->hasAttrOfType<BoolAttr>(AttrBypassLDS)) {
      load->removeAttr(AttrBypassLDS);
      loadToInfo[load] = {nullptr, distance, use};
    } else {
      LDBG("Deduce shared encoding for: " << *load);
      auto sharedEncoding =
          getSharedEncIfAllUsersAreDotEnc(load, axisInfoAnalysis, targetInfo,
                                          shouldUseAsyncCopy(load, useAsyncCopy))
              .value_or(nullptr);
      loadToInfo[load] = {sharedEncoding, distance, use};
      LDBG("Populate loadInfo with shared encoding: " << sharedEncoding);
    }
  }

  selectCanonicalMatrixSharedLayout(forOp, loadToInfo, numStages, targetInfo);
  selectLegacyAltSharedLayout(forOp, loadToInfo, useAsyncCopy, numStages,
                              targetInfo);
  // New ownership conversions introduce pointer/mask SSA values. Recompute
  // facts before asking whether those new values can use direct-to-LDS copy.
  std::unique_ptr<triton::HCU::ModuleAxisInfoAnalysis> refreshedAxisInfo;
  if (preferWideAsyncCopies(forOp, loadToInfo, useAsyncCopy, axisInfoAnalysis,
                            targetInfo))
    refreshedAxisInfo = std::make_unique<triton::HCU::ModuleAxisInfoAnalysis>(
        forOp->getParentOfType<ModuleOp>());
  auto &scheduleAxisInfo =
      refreshedAxisInfo ? *refreshedAxisInfo : axisInfoAnalysis;
  bool waitAtTail = usePingpong && (numStages == 3) &&
                    hasAsyncLoad(loadToInfo, useAsyncCopy);
  if (succeeded(mlir::ChainedDotSchedule::checkPreconditions(forOp, numStages,
                                                             loadToInfo))) {
    ChainedDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                       scheduleAxisInfo, useAsyncCopy);
  } else {
    SingleDotSchedule::updateSchedule(forOp, loadToInfo, schedule,
                                      scheduleAxisInfo, numStages, useAsyncCopy,
                                      waitAtTail);
  }

  dumpSchedule(schedule, "[lowerLoops]updated schedule:");

  schedule.serialize(forOp);
}

void lowerLoops(ModuleOp moduleOp, bool useAsyncCopy, bool usePingpong) {
  triton::HCU::ModuleAxisInfoAnalysis axisInfoAnalysis(moduleOp);
  SmallVector<scf::ForOp> loops;
  moduleOp->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
  if (loops.empty())
    return;
  for (auto forOp : loops) {
    lowerLoop(forOp, axisInfoAnalysis, useAsyncCopy, usePingpong);
  }
}

} // namespace mlir
