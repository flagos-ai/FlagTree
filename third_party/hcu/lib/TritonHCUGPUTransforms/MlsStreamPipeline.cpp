#include "TritonHCUGPUTransforms/Passes.h"
#include "hcu/lib/TritonHCUGPUToLLVM/AsyncUtility.h"
#include "hcu/lib/TritonHCUGPUToLLVM/TargetInfo.h"
#include "third_party/hcu/include/Analysis/AxisInfoExt.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipelineExpander.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <variant>

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create stream operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop and epilogue.
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "tritonhcugpu-mls-stream-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace tta = mlir::triton::hcugpu;

namespace mlir {

#define GEN_PASS_DEF_TRITONHCUGPUMLSSTREAMPIPELINE
#include "TritonHCUGPUTransforms/Passes.h.inc"

namespace {

Operation *streamPredication(RewriterBase &rewriter, Operation *op,
                             Value pred) {
  // The epilogue peeling generates a select for the stage output. This causes
  // too much register pressure with the loop result and the epilogue-dot in
  // regs for the select. Conditionally executing the dot will allow the backend
  // to optimize the select away as redundant.
  if (auto dotOp = dyn_cast<tt::DotOpInterface>(op)) {
    auto loc = dotOp->getLoc();
    auto ifOp = rewriter.create<scf::IfOp>(loc, dotOp->getResult(0).getType(),
                                           pred, /*withElseRegion=*/true);
    auto thenB = ifOp.getThenBodyBuilder();
    auto yield = thenB.create<scf::YieldOp>(loc, dotOp->getResult(0));
    dotOp->moveBefore(yield);
    ifOp.getElseBodyBuilder().create<scf::YieldOp>(loc, dotOp->getOperand(2));
    return ifOp;
  }
  if (isa<tta::MatrixLoadToLocalOp, tt::MatrixLoadOp>(op)) {
    // TODO: some extreme cases may cause out-of-bounds access,
    // for example, a for-loop that exec zero times.
    return op;
  }

  return tt::wrapInMaskOp(rewriter, op, pred);
}

//===----------------------------------------------------------------------===//
// Software pipelining generally works by anchoring on global load ops in the
// main loop and rotating the loop to schedule global load ops for future loop
// iterations together with compute for the current iteration. In this way, we
// can 1) issue memory operations earlier to hide the latency and 2) break the
// strong dependency inside on loop iteration to give backends flexibility to
// better interleave instructions for better instruction-level parallelism.
//
// The code here creates the pipelining schedule and calls the
// PipelineExpander to rewrite the `scf.for` loop accordingly. A schedule
// consists of multiple stages, where ops from different stages can overlap
// executions because the dependencies are loop carried.
//
// The general flow of this process is:
//
// 1. The user provides a `num_stages` that specifies how many stages the
//    pipeline will have. The number of stages must be larger than the distance
//    from the first independent load to the compute in order to pipeline.
//    1.a. User may also specify `global_prefetch=<s>` to set the number of
//         stages between tt.load and ttg.local_store ops.
//    1.b. User may also specify `local_prefetch=<s>` to set the number of
//         stages between ttg.local_load and compute.
// 2. A schedule is created based on the distance between the global loads
//    in the first stages and the compute that uses the loaded values in the
//    last stage (num_stages - 1). Each operation will be clustered in the
//    order to best overlap with other operations (see details below in the
//    initSchedule methods).
// 3. When the compute is a tt.dot, the scheduler will insert a shared
//    memory allocation between the global load and tt.dot. The global load
//    value will be saved to shared memory, via ttg.local_store or via
//    ttg.async_copy_global_to_local writing directly to shared memory, and the
//    ttg.local_load will load the relevant tiles for the tt.dot. These
//    operations will be scheduled according to various scheduling schemes
//    outlined below in the initSchedule methods (see details there).
// 4. Finally the schedule will be passed to the PipelineExpander to rewrite
//    accordingly. The new implementation will consist of:
//    a. Prologue: containing the ramp-up of num_stages-1 stages for
//       iteratorions i=[0, num_stages-1).
//    b. New loop: ordered by cluster and iterated on each operation by
//       `i + (num_stages-op_stage)`.
//    c. Epilogue: ramp-down of the last `num_stages-1` iterations for the
//       ops in stages 1 to last_stage. This must consider that the loop
//       bounds may be shorter than num_stages. In this case, the epilogue
//       iterations must align with the prologue.
//

struct LoadInfo {
  // Shared layout is used for loads feeding into dot ops.
  ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
  // The distance of this load's stage to its use' stage.
  int distToUse = 0;
  Operation *use = nullptr;

  ttg::HCUMlsSharedEncodingAttr mlsEncoding = nullptr;
  bool isMatrixLoad = false;

  LoadInfo() = default;
  LoadInfo(ttg::SwizzledSharedEncodingAttr sharedEncoding, int distToUse,
           Operation *use)
      : sharedEncoding(sharedEncoding), distToUse(distToUse), use(use) {}
  LoadInfo(ttg::HCUMlsSharedEncodingAttr mlsEncoding, int distToUse,
           Operation *use)
      : mlsEncoding(mlsEncoding), distToUse(distToUse), use(use),
        isMatrixLoad(true) {}
};
using LoadToInfoMap = llvm::MapVector<Operation *, LoadInfo>;

struct StreamCopyChainOps {
  tt::LoadOp loadOp;
  ttg::MemDescIndexOp subviewOp;
  ttg::LocalStoreOp localStoreOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

struct AsyncCopyChainOps {
  // ttg::AsyncCopyGlobalToLocalOp copyOp;
  tta::MatrixLoadToLocalOp matrixCopyOp;
  ttg::AsyncCommitGroupOp commitOp;
  ttg::AsyncWaitOp waitOp;
  ttg::LocalLoadOp maybeLocalLoadOp;
};

using StreamOpVariant = std::variant<StreamCopyChainOps, AsyncCopyChainOps>;
using LoadToStreamOpMap = llvm::MapVector<Operation *, StreamOpVariant>;

// AsyncCopyChainOps createAsyncCopy(tt::LoadOp loadOp, Value alloc,
//                                   Value extractIdx) {
//   OpBuilder builder(loadOp);
//   Location loc = loadOp.getLoc();

//   // Extract local subview from shared allocation
//   auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
//                       .getDefiningOp<ttg::MemDescIndexOp>();

//   auto copyOp = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
//       loc, loadOp.getPtr(), viewLoad, loadOp.getMask(), loadOp.getOther(),
//       loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
//   auto commitOp =
//       builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));
//   ttg::AsyncWaitOp waitOp =
//       builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

//   auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
//       builder, loadOp->getResult(0), viewLoad, waitOp);

//   return {copyOp, commitOp, waitOp, maybeSharedLoad};
// }

AsyncCopyChainOps createAsyncCopy(tt::MatrixLoadOp loadOp, Value alloc,
                                  Value extractIdx) {
  OpBuilder builder(loadOp);
  Location loc = loadOp.getLoc();

  // Extract local subview from shared allocation
  auto viewLoad = triton::createSingleBufferView(builder, alloc, extractIdx)
                      .getDefiningOp<ttg::MemDescIndexOp>();

  auto copyOp = builder.create<tta::MatrixLoadToLocalOp>(
      loc, loadOp.getBase(), loadOp.getShape(), loadOp.getStrides(),
      loadOp.getTensorShape(), loadOp.getIndices(), loadOp.getBoundaryCheck(),
      loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile(), viewLoad);

  auto mlsAttr = loadOp->getAttrOfType<tta::MlsEncodingAttr>(
      tta::MlsEncodingAttr::getMnemonic());
  copyOp->setAttr(tta::MlsEncodingAttr::getMnemonic(), mlsAttr);

  auto commitOp =
      builder.create<ttg::AsyncCommitGroupOp>(loc, copyOp->getResult(0));
  ttg::AsyncWaitOp waitOp =
      builder.create<ttg::AsyncWaitOp>(loc, commitOp->getResult(0), 0);

  // HCU TODO: if loadLoad with AsyncWait, membarFilter will stop insert
  // barrier for WAR and lead to incorrect result.
  auto maybeSharedLoad = tt::replaceUsesWithLocalLoad(
      builder, loadOp->getResult(0), viewLoad /*, waitOp */);

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
  auto storeOp = builder.create<ttg::LocalStoreOp>(loc, newLoadOp, viewLoad);
  auto maybeLocalLoad =
      tt::replaceUsesWithLocalLoad(builder, loadOp->getResult(0), viewLoad);

  return {newLoadOp, viewLoad, storeOp, maybeLocalLoad};
}

// Returns the given |inputValue|'s dot user result encoding and updates |opIdx|
// with which dot operand |inputValue| is fed into if possible.
ttg::AMDMfmaEncodingAttr getDotEncoding(Value inputValue, unsigned *opIdx) {
  if (!inputValue.hasOneUse())
    return nullptr;

  Operation *user = *inputValue.getUsers().begin();
  if (user->getNumResults() != 1 ||
      user->getBlock() != inputValue.getParentBlock())
    return nullptr;

  if (auto dotOp = dyn_cast<tt::DotOpInterface>(user)) {
    OpOperand &use = *inputValue.getUses().begin();
    *opIdx = use.getOperandNumber();
    auto dotType = cast<RankedTensorType>(dotOp->getResult(0).getType());
    return dyn_cast<ttg::AMDMfmaEncodingAttr>(dotType.getEncoding());
  }
  return getDotEncoding(user->getResult(0), opIdx);
}

// Adapted from
// lib/Dialect/TritonGPU/Transforms/Utility.cpp::getSharedEncIfAllUsersAreDotEnc
// to support AMDMfmaEncodingAttr.
// TODO(max): figure out how to refactor to use upstream
//
// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreDotEnc(Value loadedValue) {
  ttg::SwizzledSharedEncodingAttr attr;
  for (Operation *user : loadedValue.getUsers()) {
    LDBG(" getSharedEncIfAllUsersAreDotEnc current user: " << *user);
    if (user->getNumResults() != 1)
      return std::nullopt;

    ttg::SwizzledSharedEncodingAttr tempAttr;
    Value userResult = user->getResult(0);
    Type userResType = userResult.getType();
    if (auto memDesc = dyn_cast<ttg::MemDescType>(userResType)) {
      // First time we find a shared encoding in the chain, save it and try to
      // use it if it is compatible with the other users.
      tempAttr = cast<ttg::SwizzledSharedEncodingAttr>(memDesc.getEncoding());
      if (!getSharedEncIfAllUsersAreDotEnc(userResult).has_value())
        return std::nullopt;
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
        tempAttr = ttg::SwizzledSharedEncodingAttr::get(
            loadedValue.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder,
            ctaLayout, bitWidth, /*needTrans=*/false);
      } else if (auto llEnc = dyn_cast<ttg::LinearEncodingAttr>(userResEnc)) {
        // We use linear layout directly for scaled dot fp8 operands. For such
        // cases, we need to look further down the def-use chain to find the dot
        // op for the mfma layout to deduce operand index and other information.
        unsigned opIdx;
        if (auto dotEnc = getDotEncoding(userResult, &opIdx)) {
          unsigned vecSize = llEnc.getLinearLayout().getNumConsecutiveInOut();
          LDBG("deduced opIdx: " << opIdx << "; deduced vecSize: " << vecSize);
          tempAttr = dotEnc.composeSharedLayoutForOperand(
              ctaLayout, opIdx, srcTy.getShape(), order, vecSize, bitWidth,
              /*needTrans=*/false);
        }
      }
    }
    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

std::optional<ttg::HCUMlsSharedEncodingAttr>
getMlsEncIfAllUsersAreDotEnc(Value val) {
  auto matrixOp = cast<tt::MatrixLoadOp>(val.getDefiningOp());
  auto matrixTy = cast<RankedTensorType>(matrixOp.getType());
  auto mlsAttr = matrixOp->getAttrOfType<tta::MlsEncodingAttr>(
      tta::MlsEncodingAttr::getMnemonic());
  auto ctaLayout = ttg::getCTALayout(matrixTy.getEncoding());
  auto attr = ttg::HCUMlsSharedEncodingAttr::get(
      matrixOp.getContext(), mlsAttr.getOpIdx(), mlsAttr.getMlsTile(),
      mlsAttr.getElemBitWidth(), mlsAttr.getAlt2Kind(), mlsAttr.getVersion(),
      mlsAttr.getOrder(), ctaLayout);
  return attr;
}

bool canBeConvertedToAsyncLoad(unsigned numBuffers, tt::LoadOp loadOp,
                               Value alloc,
                               tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                               const tt::HCU::TargetInfo &targetInfo) {
  // If we have a single buffer we would require another barrier after the
  // local_reads so instead we fall back to pipeline with registers
  // Removing this check will create incorrect IR, see
  // MembarUtility.h:membarFilter
  if (numBuffers <= 1)
    return false;

  // Compute the final vecSize we can use for the combination of sourceEncoding
  // and sharedEncoding. We can only use AsyncCopy if the target supports the
  // requested or a smaller vecSize because we cannot stride when loading
  // directly to lds
  auto srcTy = cast<RankedTensorType>(loadOp.getPtr().getType());
  auto dstTy = cast<ttg::MemDescType>(alloc.getType());
  auto regLayout = triton::gpu::toLinearLayout(srcTy);
  // It's the allocation so we trim the multibuffer dimension
  auto srcShape = dstTy.getShape().take_back(srcTy.getRank());
  auto sharedLayout =
      triton::gpu::toLinearLayout(srcShape, dstTy.getEncoding());
  auto regToSharedLayout = regLayout.invertAndCompose(sharedLayout);

  unsigned vecSize = regToSharedLayout.getNumConsecutiveInOut();
  unsigned elemBitWidth = dstTy.getElementTypeBitWidth();

  if (fitToValidDirectToLdsVecSize(vecSize, elemBitWidth, targetInfo) == 0)
    return false;

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
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value extractIdx = minusOne;
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);

  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependency.
  forOp = addIterArgsToLoop(builder, forOp, {extractIdx});

  // Create one counter for the extract indices to avoid creating long
  // live range.
  extractIdx = forOp.getBody()->getArgument(newOperandIndex);

  builder.setInsertionPoint(forOp.getBody(), forOp.getBody()->begin());
  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);

  // Patch the yield with the updated counter.
  appendToForOpYield(forOp, {extractIdx});

  LoadToStreamOpMap loadToStreamOp;
  for (auto &[l, info] : loadToInfo) {
    if (!info.sharedEncoding && !info.mlsEncoding)
      continue;

    auto loadOp = dyn_cast<tt::LoadOp>(l);
    auto matrixLoadOp = dyn_cast<tt::MatrixLoadOp>(l);
    if (!loadOp && !matrixLoadOp)
      continue;

    bool isLoadOp = loadOp != nullptr;

    // Create an allocation that can hold distance number of loadOp shapes.
    Value alloc;
    if (isLoadOp) {
      auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
      alloc = triton::createAlloc(forOp, ty, loadOp->getLoc(),
                                  info.sharedEncoding, numBuffers);
    } else if (matrixLoadOp) {
      auto ty = cast<RankedTensorType>(matrixLoadOp->getResultTypes()[0]);
      alloc = triton::createAlloc(forOp, ty, matrixLoadOp->getLoc(),
                                  info.mlsEncoding, numBuffers);
    }
    assert(alloc && "Failed to create alloc for the async load.");
    // auto arch = getHCUArch(loadOp->getParentOfType<ModuleOp>());
    // triton::HCU::TargetInfo targetInfo(arch ? arch->str() : "");

    // Replace the old load with multi-buffered loads
    if (isLoadOp) {
      if (0 /* useAsyncCopy &&
          canBeConvertedToAsyncLoad(numBuffers, loadOp, alloc, axisInfoAnalysis,
                                    targetInfo) */) {
        // loadToStreamOp[loadOp] = createAsyncCopy(loadOp, alloc, extractIdx);
      } else {
        loadToStreamOp[loadOp] = createStreamCopy(loadOp, alloc, extractIdx);
      }
    } else if (matrixLoadOp) {
      loadToStreamOp[matrixLoadOp] =
          createAsyncCopy(matrixLoadOp, alloc, extractIdx);
    }
  }

  return loadToStreamOp;
}

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
llvm::MapVector<Operation *, std::pair<int, Operation *>>
loadOpsToIndirectionLevel(scf::ForOp forOp, bool pipelineWithoutDot,
                          tt::ModuleAxisInfoAnalysis &axisInfoAnalysis,
                          int numStages, bool filterSmall) {
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel;
  DenseSet<Operation *> seen;
  DenseSet<Operation *> excluded;

  std::function<void(Operation *, Operation *, int)> dfs =
      [&](Operation *op, Operation *finalUser, int distance) {
        if (!seen.insert(op).second || excluded.count(op))
          return;
        if (isa<tt::LoadOp, tt::MatrixLoadOp>(op)) {
          if (filterSmall && isa<tt::LoadOp>(op) &&
              !canBeConvertedToAsyncLoad(cast<tt::LoadOp>(op),
                                         axisInfoAnalysis)) {
            return;
          }
          if (loadOpToIndLevel.count(op)) {
            int level = loadOpToIndLevel[op].first;
            if (level != distance) {
              // If we have multiple uses at different distances, we don't
              // know which one to pick.
              LDBG("Load " << *op
                           << " has multiple uses at different distances:"
                           << level << " and " << distance);
              loadOpToIndLevel.erase(op);
              excluded.insert(op);
              return;
            }
          } else {
            LDBG("Load " << *op << " considered for pipelining with distance "
                         << distance);
            loadOpToIndLevel[op] = {distance, finalUser};
          }
          finalUser = op;
          distance++;
        }

        for (Value operand : getNestedOperands(op)) {
          if (isa<mlir::triton::DotOpInterface>(op)) {
            // Heuristic: only pipeline A and B operands of the dot op.
            if (operand == op->getOperand(2))
              continue;
          }
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, finalUser, distance);
          }
        }
      };

  bool seenDot = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    // Arbitrary heuristic. TMEMStoreOp is included to keep logic consistent
    // with legacy code when we weren't hoisting tmem allocas.
    if (!isa<mlir::triton::DotOpInterface>(op))
      continue;
    seenDot = true;
    seen.clear();
    dfs(&op, &op, 0);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (pipelineWithoutDot) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<tt::LoadOp>(op))
        dfs(&op, &op, 0);
    }
  }

  // We assume loads with different dist are assigned to different stages.
  // If numStages is 2, we will have no stage available for indirect loads
  // with dist >= 1. In general, when dist is equal to numStages - 1, we
  // should not pipeline it.
  for (auto iter = loadOpToIndLevel.begin(); iter != loadOpToIndLevel.end();) {
    if (iter->second.first >= numStages - 1)
      iter = loadOpToIndLevel.erase(iter);
    else
      ++iter;
  }

  return loadOpToIndLevel;
}

LoadToInfoMap
preprocessLoop(triton::HCU::ModuleAxisInfoAnalysis &axisInfoAnalysis,
               scf::ForOp &forOp, int numStages) {
  auto arch = getHCUArch(forOp->getParentOfType<ModuleOp>());
  triton::HCU::ISAFamily isaFamily = triton::HCU::ISAFamily::Unknown;
  if (arch)
    isaFamily = triton::HCU::deduceISAFamily(*arch);

  bool pipelineWithoutDot = forOp->hasAttr(mlir::triton::kNumStagesAttrName);
  bool filterSmallVectors = isaFamily != triton::HCU::ISAFamily::CDNA4;
  llvm::MapVector<Operation *, std::pair<int, Operation *>> loadOpToIndLevel =
      loadOpsToIndirectionLevel(forOp, pipelineWithoutDot, axisInfoAnalysis,
                                numStages, filterSmallVectors);

  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevel.size() << " loads to pipeline:");
    for (const auto &[l, i] : loadOpToIndLevel) {
      LDBG("  - load: " << *l);
      LDBG("    at distance: " << i.first);
      LDBG("    used by op: " << *i.second);
    }
  });

  LoadToInfoMap loadToInfo;
  for (const auto &[load, info] : loadOpToIndLevel) {
    auto [distance, use] = info;
    if (isa<tt::LoadOp>(load)) {
      auto sharedEncoding =
          getSharedEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
      loadToInfo[load] = {sharedEncoding, distance, use};
    } else if (isa<tt::MatrixLoadOp>(load)) {
      auto mlsEncoding =
          getMlsEncIfAllUsersAreDotEnc(load->getResult(0)).value_or(nullptr);
      loadToInfo[load] = {mlsEncoding, distance, use};
    }
  }

  return loadToInfo;
}

namespace SingleDotSchedule {
// Define categories of scheduling details per Operation types.
// The SingleDotSchedule schedules 5 types of operations:
// 1. GLOBAL_LOAD: tt.load / ttg.async_copy_global_to_local
// 2. LOCAL_STORE: ttg.local_store
// 3. LOCAL_LOAD:  ttg.local_load
// 4. COMPUTE:     ops that use the loaded data
// 5. ASYNC_WAIT:  ttg.async_wait
// Note that ttg ops mentioned in the above list are created during scheduling.
enum SchedType {
  SCHED_GLOBAL_LOAD,
  SCHED_LOCAL_STORE,
  SCHED_LOCAL_LOAD,
  SCHED_COMPUTE,
  SCHED_ASYNC_WAIT,
  SCHED_SIZE
};

using Clusters = std::array<tt::CoarseSchedule::Cluster, SCHED_SIZE>;
using Stages = std::array<int, SCHED_SIZE>;

LogicalResult initScheduleSingleBuf(int maxDist, Stages &stages, int numStages,
                                    int &numBuffers, int globalPrefetch,
                                    int localPrefetch, bool useAsyncCopy,
                                    bool waitAtTail, bool &forceSchedLocalLoad,
                                    Clusters &clusters,
                                    tt::CoarseSchedule &schedule) {
  bool pairedGlobalLoadLocalStore = true;
  stages[SCHED_LOCAL_STORE] = stages[SCHED_GLOBAL_LOAD];

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
  assert(numBuffers == 1 && "numBuffers should be 1 for single buffer case");

  LDBG("deduced max shared memory buffer number = " << numBuffers);

  // We place async wait as the first cluster because we want to have it being
  // the first in the main loop after pipelining.
  int asyncWaitCluster = 0;
  int localLoadCluster = 1;
  int globalLoadCluster = 2;
  int computeCluster = 3;
  int localStoreCluster = 4;

  // Make assignments
  Clusters clusterVec;
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  // Streaming Schema cluster order and staging for single-buffer.
  // case 1:  both matrix_load:
  //     for i in (...):
  //       async_wait:      stage=i     cluster=0
  //       local_load:      stage=i     cluster=1
  //       matrix_load_lds: stage=i+1   cluster=2
  //       compute:         stage=i     cluster=3
  //       tail:            stage=i     cluster=3
  //
  // case 2:  mix matrix_load(b) and load(a):               // and after
  // ReorderInstrcutions pass
  //     for i in (...):
  //                                                        global_loads(a):
  //                                                        stage=i+1
  //       async_wait(b):      stage=i    cluster=0         async_wait(b):
  //       stage=i local_load(a, b):   stage=i    cluster=1 local_load(a, b):
  //       stage=i global_loads(a):    stage=i+1  cluster=2   -->
  //       matrix_load_lds(b): stage=i+1  cluster=2         matrix_load_lds(b):
  //       stage=i+1 compute:            stage=i    cluster=3         compute:
  //       stage=i tail:               stage=i    cluster=3         tail:
  //       stage=i local_store(a):     stage=i+1  cluster=4 local_store(a):
  //       stage=i+1
  //
  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);
  forceSchedLocalLoad = true;
  return success();
}

// Init Schedule Config based on settings and loop characteristics.
// Create clusters in order of ops in loop. This can interleave ops
// from different stages in the same cluster to achieve better backend
// scheduling.
//   WARNING: Changing the order of schedule.clusters.newAtBack() calls
//            can cause invalid schedules to be produced.
LogicalResult initSchedule(int maxDist, Stages &stages, int numStages,
                           int &numBuffers, int globalPrefetch,
                           int localPrefetch, bool useAsyncCopy,
                           bool waitAtTail, bool asyncCopySingleBuffer,
                           bool &forceSchedLocalLoad, Clusters &clusters,
                           tt::CoarseSchedule &schedule) {
  int lastStage = numStages - 1;
  stages[SCHED_GLOBAL_LOAD] = 0;
  stages[SCHED_LOCAL_STORE] = globalPrefetch;
  stages[SCHED_LOCAL_LOAD] = lastStage - localPrefetch;
  stages[SCHED_COMPUTE] = lastStage;
  stages[SCHED_ASYNC_WAIT] = stages[SCHED_LOCAL_LOAD];

  bool _asyncCopySingleBuffer = maxDist == 0 ? asyncCopySingleBuffer : false;
  if (_asyncCopySingleBuffer) {
    return initScheduleSingleBuf(
        maxDist, stages, numStages, numBuffers, globalPrefetch, localPrefetch,
        useAsyncCopy, waitAtTail, forceSchedLocalLoad, clusters, schedule);
  }

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

  // num_stages = 3 with prefetch case
  if (numBuffers == 2 &&
      stages[SCHED_GLOBAL_LOAD] != stages[SCHED_LOCAL_STORE] &&
      stages[SCHED_LOCAL_STORE] != stages[SCHED_LOCAL_LOAD]) {
    asyncWaitCluster = 0;
    localLoadCluster = 1;
    globalLoadCluster = 2;
    localStoreCluster = 3;
    computeCluster = 4;

    forceSchedLocalLoad = true;
  }

  // Make assignments
  Clusters clusterVec;
  std::generate(clusterVec.begin(), clusterVec.end(),
                [&]() { return schedule.clusters.newAtBack(); });

  // Streaming Schema cluster order and staging for multi-buffer.
  //
  // note: the prefetch affect the stages of local_store and all the clusters,
  // so for both matrix_load situation, the prefetch has no effect. for mix_load
  // situation, the prefetch has effect.
  //
  // case 1:  both matrix_load:
  //
  //    Stream Prefetch = False(num_stages = 2, numBuffers = 2):
  //       for i in (...):
  //         async_wait:      stage=i     cluster=0
  //         matrix_load_lds: stage=i+1   cluster=1
  //         local_load:      stage=i     cluster=1
  //         compute:         stage=i     cluster=1
  //         tail:            stage=i     cluster=1
  //    Stream Prefetch = True(num_stages = 2, numBuffers = 2):
  //        Same as Stream Prefetch = False.
  //
  //
  // case 2:  mix matrix_load(b) and load(a):
  //    Stream Prefetch = False(num_stages = 2, numBuffers = 2):
  //       for i in (...):                             // and after
  //       ReorderInstrcutions pass
  //                                                          global_loads(a):
  //                                                          stage=i+1
  //         async_wait(b):      stage=i    cluster=0         async_wait(b):
  //         stage=i global_loads(a):    stage=i+1  cluster=1   -->
  //         matrix_load_lds(b): stage=i+1  cluster=1 matrix_load_lds(b):
  //         stage=i+1 local_load(a, b):   stage=i    cluster=1 local_load(a,
  //         b):   stage=i compute:            stage=i    cluster=1 compute:
  //         stage=i tail:               stage=i    cluster=1         tail:
  //         stage=i local_store(a):     stage=i+1  cluster=3 local_store(a):
  //         stage=i+1
  //
  //
  //    Stream Prefetch = True(num_stages = 2, numBuffers = 2):
  //       for i in (...):                             // and after
  //       ReorderInstrcutions pass
  //                                                          global_loads(a):
  //                                                          stage=i+1
  //         async_wait(b):      stage=i    cluster=0         async_wait(b):
  //         stage=i local_store(a):     stage=i    cluster=2 local_store(a):
  //         stage=i global_loads(a):    stage=i+1  cluster=3   -->
  //         matrix_load_lds(b): stage=i+1  cluster=3 matrix_load_lds(b):
  //         stage=i+1 local_load(a, b):   stage=i    cluster=3 local_load(a,
  //         b):   stage=i compute:            stage=i    cluster=3 compute:
  //         stage=i tail:               stage=i    cluster=3         tail:
  //         stage=i
  //
  // case 3: mix matrix_load(b) and load(a) with num_stages = 3 && prefetch
  // case:
  //    Stream Prefetch = True(num_stages = 3, numBuffers = 2):
  //       for i in (...):                             // and after
  //       ReorderInstrcutions pass
  //                                                          global_loads(a):
  //                                                          stage=i+2
  //         async_wait(b):      stage=i    cluster=0         async_wait(b):
  //         stage=i local_load(a, b):   stage=i    cluster=1 local_load(a, b):
  //         stage=i global_loads(a):    stage=i+2  cluster=2   -->
  //         matrix_load_lds(b): stage=i+2  cluster=2 matrix_load_lds(b):
  //         stage=i+2 local_store(a):     stage=i+1  cluster=3 local_store(a):
  //         stage=i+1 compute:            stage=i    cluster=4         compute:
  //         stage=i tail:               stage=i    cluster=4         tail:
  //         stage=i
  //
  clusters[SCHED_GLOBAL_LOAD] = clusterVec[globalLoadCluster];
  clusters[SCHED_LOCAL_STORE] = clusterVec[localStoreCluster];
  clusters[SCHED_LOCAL_LOAD] = clusterVec[localLoadCluster];
  clusters[SCHED_COMPUTE] = clusterVec[computeCluster];
  clusters[SCHED_ASYNC_WAIT] = clusterVec[asyncWaitCluster];

  LDBG("Cluster schedule:" << "  GLOBAL_LOAD cluster = " << globalLoadCluster
                           << ", LOCAL_STORE cluster = " << localStoreCluster
                           << ", LOCAL_LOAD cluster = " << localLoadCluster
                           << ", COMPUTE cluster = " << computeCluster
                           << ", ASYNC_WAIT cluster = " << asyncWaitCluster
                           << "; total = " << SCHED_SIZE);

  return success();
}

// void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps, tt::LoadOp loadOp,
//                        tt::CoarseSchedule &schedule, const Stages &stages,
//                        const Clusters &clusters) {
//   auto [copyOp, commitOp, waitOp, maybeLocalLoadOp] = asyncOps;
//   auto [loadStage, loadCluster] = schedule[loadOp];
//   schedule.insert(copyOp, loadStage, loadCluster);
//   // Place ttg.async_commit_group op following AsyncCopyGlobalToLocal so the
//   // later UpdateAsyncWaitCount pass can deduce better waitcnts
//   schedule.insert(commitOp, loadStage, loadCluster);
//   // If the LocalLoads are scheduled to a later stage than AsyncCopy we need
//   to
//   // place the AsyncCopy prefetches after the AsyncWaits which create a
//   barrier
//   // to ensure all warps are finished reading the shared buffer we will write
//   // into. This is done by scheduling AsyncWait as the first cluster.
//   // If AsyncCopy and LocalLoads are in the same stage we do not assign a
//   // schdule so they are placed before the LocalLoads
//   if (loadStage != stages[SCHED_LOCAL_LOAD])
//     schedule.insert(waitOp, stages[SCHED_ASYNC_WAIT],
//                     clusters[SCHED_ASYNC_WAIT]);

//   if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE])
//   {
//     scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
//                       clusters[SCHED_LOCAL_LOAD]);
//   }
// }

void scheduleAsyncCopy(const AsyncCopyChainOps &asyncOps,
                       tt::MatrixLoadOp loadOp, tt::CoarseSchedule &schedule,
                       const Stages &stages, const Clusters &clusters,
                       bool forceSchedLocalLoad) {
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

  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] ||
      forceSchedLocalLoad) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

void scheduleStreamCopy(const StreamCopyChainOps &streamOps,
                        tt::LoadOp oldLoadOp, tt::CoarseSchedule &schedule,
                        const Stages &stages, const Clusters &clusters,
                        bool forceSchedLocalLoad) {
  auto [newLoadOp, subviewOp, localStoreOp, maybeLocalLoadOp] = streamOps;
  auto [loadStage, loadCluster] = schedule[oldLoadOp];

  schedule.insert(newLoadOp, loadStage, loadCluster);
  schedule.insert(subviewOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  schedule.insert(localStoreOp, stages[SCHED_LOCAL_STORE],
                  clusters[SCHED_LOCAL_STORE]);
  if (maybeLocalLoadOp && stages[SCHED_LOCAL_LOAD] != stages[SCHED_COMPUTE] ||
      forceSchedLocalLoad) {
    scheduleLocalLoad(maybeLocalLoadOp, schedule, stages[SCHED_LOCAL_LOAD],
                      clusters[SCHED_LOCAL_LOAD]);
  }
}

LogicalResult scheduleLoads(const LoadToInfoMap &loadToInfo, int maxDist,
                            int maxDistLoad, int numStages,
                            const Stages &stages, const Clusters &clusters,
                            tt::CoarseSchedule &schedule) {
  // The stage gap between chained loads--this allows us to "spread" loads
  // with a non-one step in case the number of stages given by the user is
  // large.
  assert(numStages >= 2 && "requires num_stages=2 at least");
  unsigned stagesBetweenLoads = llvm::divideCeil(numStages - 2, maxDist + 1);
  LDBG("stagesBetweenLoads = " << stagesBetweenLoads);

  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, info] : loadToInfo) {
    // Non-LoadOp(s) are the (final) root uses of all LoadOp(s).
    if (!isa<tt::LoadOp, tt::MatrixLoadOp>(info.use))
      schedule.insert(info.use, stages[SCHED_COMPUTE], clusters[SCHED_COMPUTE]);
  }

  bool hasSameIndirectLevel = maxDist == maxDistLoad;
  // Assign stages to the loads.
  for (auto [loadOp, info] : loadToInfo) {
    if (isa<tt::MatrixLoadOp>(loadOp)) {
      int stage = (maxDist - info.distToUse) * stagesBetweenLoads;
      schedule.insert(loadOp, stages[stage], clusters[SCHED_GLOBAL_LOAD]);
    } else if (isa<tt::LoadOp>(loadOp)) {
      int stage = (maxDistLoad - info.distToUse) * stagesBetweenLoads;
      schedule.insert(loadOp, stages[stage], clusters[SCHED_GLOBAL_LOAD]);
    }
  }

  return success();
}

void scheduleStreamOps(const LoadToStreamOpMap &loadToStreamOp,
                       tt::CoarseSchedule &schedule, const Stages &stages,
                       const Clusters &clusters, bool forceSchedLocalLoad) {
  for (auto [l, streamOps] : loadToStreamOp) {
    auto loadOp = dyn_cast<tt::LoadOp>(l);
    auto matrixLoadOp = dyn_cast<tt::MatrixLoadOp>(l);
    if (!loadOp && !matrixLoadOp)
      continue;

    if (auto asyncOps = std::get_if<AsyncCopyChainOps>(&streamOps)) {
      scheduleAsyncCopy(*asyncOps, matrixLoadOp, schedule, stages, clusters,
                        forceSchedLocalLoad);
    } else if (auto sOps = std::get_if<StreamCopyChainOps>(&streamOps)) {
      scheduleStreamCopy(*sOps, loadOp, schedule, stages, clusters,
                         forceSchedLocalLoad);
    }
  }
}

tt::CoarseSchedule
buildSchedule(scf::ForOp &forOp, int numStages, const LoadToInfoMap &loadToInfo,
              int globalPrefetch, int localPrefetch, bool useAsyncCopy,
              bool waitAtTail, bool asyncCopySingleBuffer,
              triton::HCU::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
  tt::CoarseSchedule schedule(numStages);
  Stages stages;
  Clusters clusters;

  auto dumpSchedule = [&](llvm::StringRef msg) {
    LLVM_DEBUG({
      llvm::dbgs() << "\n";
      LDBG(msg);
      schedule.dump();
    });
  };

  int maxDist = 0;
  int maxDistLoad = 0;
  for (auto &[l, info] : loadToInfo) {
    if (info.isMatrixLoad)
      maxDist = std::max(maxDist, info.distToUse);
    else
      maxDistLoad = std::max(maxDistLoad, info.distToUse);
  }

  int numBuffers = 1;
  bool forceSchedLocalLoad = false;
  if (failed(initSchedule(maxDist, stages, numStages, numBuffers,
                          globalPrefetch, localPrefetch, useAsyncCopy,
                          waitAtTail, asyncCopySingleBuffer,
                          forceSchedLocalLoad, clusters, schedule)))
    return {};

  if (failed(scheduleLoads(loadToInfo, maxDist, maxDistLoad, numStages, stages,
                           clusters, schedule)))
    return {};
  dumpSchedule("Coarse schedule loads only:");

  // Convert the loads into shared memory allocations and loads from them.
  auto loadToStreamOp = createStreamOps(loadToInfo, forOp, numBuffers,
                                        useAsyncCopy, axisInfoAnalysis);
  scheduleStreamOps(loadToStreamOp, schedule, stages, clusters,
                    forceSchedLocalLoad);
  dumpSchedule("Coarse schedule stream ops:");

  scheduleDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dependencies:");

  triton::gpu::scheduleDistanceOneDependencies(forOp, schedule);
  dumpSchedule("Coarse schedule with dist 1:");

  tt::CoarseSchedule::Cluster computeCluster = clusters[SCHED_COMPUTE];
  triton::gpu::scheduleRemainingToLastStage(forOp, schedule, computeCluster);
  dumpSchedule("Final coarse schedule:");

  std::vector<std::pair<Operation *, unsigned>> coarseSchedule =
      schedule.createFinalSchedule(forOp);

  return schedule;
}
} // namespace SingleDotSchedule

FailureOr<scf::ForOp> pipelineLoop(scf::ForOp forOp, int numStages,
                                   int globalPrefetch, int localPrefetch,
                                   bool useAsyncCopy, bool waitAtTail,
                                   bool asyncCopySingleBuffer) {

  triton::HCU::ModuleAxisInfoAnalysis axisInfoAnalysis(
      forOp->getParentOfType<ModuleOp>());

  LoadToInfoMap loadToInfo = preprocessLoop(axisInfoAnalysis, forOp, numStages);

  if (loadToInfo.empty()) {
    LDBG("couldn't find any pipeline-able loads:\n" << *forOp);
    return failure();
  }

  tt::CoarseSchedule schedule;

  schedule = SingleDotSchedule::buildSchedule(
      forOp, numStages, loadToInfo, globalPrefetch, localPrefetch, useAsyncCopy,
      waitAtTail, asyncCopySingleBuffer, axisInfoAnalysis);

  if (schedule.empty()) {
    return failure();
  }

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  auto coarseSchedule = schedule.createFinalSchedule(forOp);

  tt::PipeliningOption options;
  options.supportDynamicLoops = true;
  options.peelEpilogue = true;
  options.predicateFn = streamPredication;
  // Annotate loadOp in prologue for further moving up
  options.annotateFn = [](Operation *op,
                          tt::PipeliningOption::PipelinerPart part,
                          unsigned stage) {
    if (part != tt::PipeliningOption::PipelinerPart::Prologue)
      return;

    auto annotateLoad = [](Operation *loadOp) {
      loadOp->setAttr("hcu.pipeliner_part",
                      StringAttr::get(loadOp->getContext(), "prologue"));
    };

    if (auto loadOp = dyn_cast<tt::LoadOp>(op)) {
      annotateLoad(loadOp);
      return;
    }
    // loadOp may be wrapped by a MaskOp as predicateFn execution
    // precedes annotation
    if (auto maskOp = dyn_cast<ttg::MaskOp>(op)) {
      for (auto &innerOp : maskOp.getBody()->without_terminator()) {
        if (auto loadOp = dyn_cast<tt::LoadOp>(&innerOp)) {
          annotateLoad(loadOp);
          return;
        }
      }
    }
  };
  // Set the final schedule as our scheduling function
  options.getScheduleFn =
      [coarseSchedule](scf::ForOp,
                       std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(coarseSchedule);
      };

  LDBG("Loop before sending to expander:\n" << *forOp);

  IRRewriter rewriter(forOp);
  return tt::pipelineForLoop(rewriter, forOp, options);
}

// HCU: Skip loop without matrix_load ops and leave it handled by
// StreamPipelinePass
bool skipLoopWithOutMatrixLoadOp(scf::ForOp forOp) {
  return llvm::all_of(forOp.getBody()->without_terminator(),
                      [](Operation &op) { return !isa<tt::MatrixLoadOp>(op); });
}

} // namespace

struct MlsPipelinePass
    : impl::TritonHCUGPUMlsStreamPipelineBase<MlsPipelinePass> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    // check numStages
    if (numStages == 1)
      return;

    // check numStages
    if (globalPrefetch < 0 || globalPrefetch >= numStages) {
      moduleOp.emitWarning("global prefetch control must be in [0, ")
          << numStages << "); " << globalPrefetch
          << " is out of range, fallback to 0";
      globalPrefetch = 0;
    }

    constexpr int localPrefetch = 0;
    constexpr bool useAsyncCopy = true;

    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with num_stage <= 1.
      if (tt::getNumStagesOrDefault(forOp, numStages) > 1)
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      if (!triton::gpu::isSafeToPipeline(forOp)) {
        LDBG("Loop not safe to pipeline:\n" << *forOp);
        continue;
      }

      if (skipLoopWithOutMatrixLoadOp(forOp)) {
        LDBG("Skip loop with matrix_load or matrix_load_to_local ops:\n"
             << *forOp);
        continue;
      }

      // i.e., we can still disable `waitAtTail` by explicitly disabling
      // pingpong, which is the only use case of this scheduling variant.
      int numStagesThis = tt::getNumStagesOrDefault(forOp, numStages);
      bool waitAtTail =
          false; // usePingpong && (numStagesThis == 3) && useAsyncCopy;
      (void)pipelineLoop(forOp, numStagesThis, globalPrefetch, localPrefetch,
                         useAsyncCopy, waitAtTail, asyncCopySingleBuffer != 0);
    }

    // NOTE: Leave empty for now, until we utilize customEpiloguePeeling
    DenseSet<ttg::MaskOp> peeledMaskOps;
    tt::resolveMaskOp(moduleOp);

    if (useAsyncCopy) {
      llvm::SmallSetVector<ttg::AsyncWaitOp, 8> waitOps;
      moduleOp.walk([&](ttg::AsyncWaitOp waitOp) { waitOps.insert(waitOp); });
      tt::combineRedundantWaitOps(waitOps);
    }
  }
};

} // namespace mlir
