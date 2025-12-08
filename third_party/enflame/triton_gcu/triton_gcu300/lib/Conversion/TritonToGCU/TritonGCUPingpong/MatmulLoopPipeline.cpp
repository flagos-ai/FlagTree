/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "Utils.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include <list>
#include <utility>
#include <vector>

#define DEBUG_TYPE "triton-matmul-loop-pipeline"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

#define int_attr(num) builder.getI64IntegerAttr(num)

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

namespace {

struct LoadInfo {
  // Layout of the data in the shared memory.
  ttg::SwizzledSharedEncodingAttr sharedEncoding = nullptr;
  int distToUse = 0;
  // always preload
  bool usedByDot = true;
};
class CoarseSchedule {
public:
  class ClusterList {
    std::list<int> orderClusters;

  public:
    using iterator = decltype(orderClusters)::iterator;
    ClusterList() = default;
    iterator begin() { return orderClusters.begin(); }
    iterator end() { return orderClusters.end(); }
    size_t size() { return orderClusters.size(); }
    iterator newAtBack() {
      orderClusters.push_back(orderClusters.size());
      return std::prev(orderClusters.end());
    }
    iterator newAtFront() {
      orderClusters.push_front(-1);
      for (auto &clusterId : orderClusters) {
        clusterId++;
      }
      return orderClusters.begin();
    }
    iterator newBefore(iterator cluster) {
      auto ret = orderClusters.insert(cluster, *cluster);
      for (auto &clusterId : llvm::make_range(cluster, orderClusters.end())) {
        clusterId++;
      }
      return ret;
    }
  };

  explicit CoarseSchedule(int numStages) : numStages(numStages) {
    LDBG("CoarseSchedule "
         << "#######################################################\n");
  }
  int numStages;
  ClusterList clusters;
  using Cluster = decltype(clusters)::iterator;

  DenseMap<Operation *, std::pair<int, Cluster>> opToStageAndCluster;

  void insert(Operation *op, int stage, Cluster cluster) {
    opToStageAndCluster[op] = {stage, cluster};
  }

  bool insertIfAbsent(Operation *op, int stage, Cluster cluster) {
    if (opToStageAndCluster.count(op))
      return false;
    insert(op, stage, cluster);
    return true;
  }

  void insertDepsOfOp(Operation *op, int stage, CoarseSchedule::Cluster cluster,
                      bool includeArg) {
    for (Value operand : op->getOperands()) {
      Value v = operand;
      llvm::DenseSet<Value> seen;
      while (auto arg = dyn_cast<BlockArgument>(v)) {
        if (!includeArg)
          break;
        if (!seen.insert(v).second)
          break;
        if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
          auto yieldOp = op->getBlock()->getTerminator();
          v = yieldOp->getOperand(arg.getArgNumber() - 1);
          continue;
        }
        break;
      }
      Operation *defOp = v.getDefiningOp();
      if (defOp && defOp->getBlock() == op->getBlock()) {
        if (insertIfAbsent(defOp, stage, cluster)) {
          insertDepsOfOp(defOp, stage, cluster, includeArg);
        }
      }
    }
  }

  void erase(Operation *op) { opToStageAndCluster.erase(op); }

  int count(Operation *op) { return opToStageAndCluster.count(op); }

  std::pair<int, Cluster> operator[](Operation *op) {
    return opToStageAndCluster[op];
  }

  SmallVector<std::tuple<Operation *, int, Cluster>>
  getOpsInOrder(scf::ForOp forOp) {
    LLVM_DEBUG({ llvm::dbgs() << "getOpsInOrder enter\n"; });

    SmallVector<SmallVector<std::tuple<Operation *, int, Cluster>>, 8>
        orderClusters(clusters.size());
    for (auto &op : forOp.getBody()->without_terminator()) {
      if (opToStageAndCluster.count(&op) == 0) {
        continue;
      }
      assert(opToStageAndCluster[&op].first < numStages &&
             "Op with invalid stage!");
      int clusterId = *opToStageAndCluster[&op].second;
      assert(clusterId == std::distance(clusters.begin(),
                                        opToStageAndCluster[&op].second) &&
             "Cluster ID mismatch!");
      LLVM_DEBUG({
        llvm::dbgs() << "orderClusters push\n";
        op.dump();
      });
      orderClusters[clusterId].push_back(
          make_tuple(&op, opToStageAndCluster[&op].first,
                     opToStageAndCluster[&op].second));
    }
    SmallVector<std::tuple<Operation *, int, Cluster>> opsInOrder;
    for (size_t i = 0; i < orderClusters.size(); i++) {
      for (auto [op, stage, cluster] : orderClusters[i]) {
        opsInOrder.push_back({op, stage, cluster});
      }
    }

    return opsInOrder;
  }

  std::vector<std::pair<Operation *, unsigned>>
  createFinalSchedule(scf::ForOp forOp) {
    SmallVector<std::tuple<Operation *, int, Cluster>> opsInOrder =
        getOpsInOrder(forOp);
    std::vector<std::pair<Operation *, unsigned>> schedule;
    for (auto [op, stage, cluster] : opsInOrder) {
      (void)cluster;
      LDBG("Adding op to FinalSchedule at stage" << stage << " cluster "
                                                 << *cluster << ":" << *op);
      schedule.push_back({op, stage});
    }
    return schedule;
  }

  void dump() {
    for (int i = 0; i < numStages; i++) {
      LDBG("- Ops in stage " << i);
      for (auto &[op, stageAndCluster] : opToStageAndCluster) {
        if (i == stageAndCluster.first) {
          llvm::outs() << " cluster: " << *stageAndCluster.second << " ";
          op->dump();
        }
      }
    }
  }
};
} // namespace

// Replace the ForOp's yield with a new one with the given operands appended.
static void appendToYield(scf::ForOp forOp, ArrayRef<Value> newOperands) {
  // Fix up the yield op.
  Operation *yieldOp = forOp.getBody()->getTerminator();
  SmallVector<Value> operands(yieldOp->getOperands());
  operands.append(newOperands.begin(), newOperands.end());

  OpBuilder builder(yieldOp);
  builder.create<scf::YieldOp>(yieldOp->getLoc(), operands);
  yieldOp->erase();
}

static void createAsyncCopy(scf::ForOp &forOp, triton::gcu::LoadOp loadOp,
                            Value alloc, Value insertIdx, Value extractIdx,
                            CoarseSchedule &schedule,
                            CoarseSchedule::Cluster prefetchCluster,
                            llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
                            int numStages) {
  OpBuilder builder(forOp);
  Value zero = builder.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
  // Replace the load with insert/extract slice.
  builder.setInsertionPoint(loadOp);
  Location loc = loadOp.getLoc();

  ttg::MemDescType allocTy = cast<ttg::MemDescType>(alloc.getType());
  SmallVector<Value> copyOffsets(allocTy.getRank(), zero);
  copyOffsets[0] = insertIdx;
  ttg::MemDescType subviewTy = ttg::MemDescType::get(
      allocTy.getShape().drop_front(), allocTy.getElementType(),
      allocTy.getEncoding(), allocTy.getMemorySpace(), /*mutableMemory=*/true);
  auto view =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, copyOffsets);
  Operation *copy = builder.create<triton::gcu::AsyncLoadGlobalToShareOp>(
      loadOp.getLoc(), loadOp.getPtr(), loadOp.getShape(), loadOp.getStrides(),
      loadOp.getOffsets(), view, loadOp.getDefaultValue(),
      loadOp.getOrderHint());
  Operation *wait =
      builder.create<triton::gcu::AsyncWaitOp>(loc, copy->getResult(0));
  auto [stage, cluster] = schedule[loadOp];
  schedule.erase(loadOp);
  schedule.insert(copy, stage, cluster);

  // Extract part.
  SmallVector<Value> loadOffsets(allocTy.getRank(), zero);
  loadOffsets[0] = extractIdx;
  auto viewLoad =
      builder.create<ttg::MemDescSubviewOp>(loc, subviewTy, alloc, loadOffsets);

  auto loadUsers = loadOp->getResult(0).getUsers();
  auto userNumber = std::distance(loadUsers.begin(), loadUsers.end());
  if (userNumber == 0) {
    loadOp.dump();
    llvm::report_fatal_error(
        "[Error] there more than one loadUser of load that can't do pinpong\n");
  }
  auto loadUser = *loadUsers.begin();
  if (userNumber == 1 &&
      llvm::isa_and_nonnull<triton::gpu::ConvertLayoutOp>(loadUser)) {
    auto converLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(loadUser);
    auto localLoad = builder.create<triton::gcu::LocalLoadOp>(
        loc, converLayout.getType(), viewLoad, wait->getResult(0));
    auto result = localLoad->getResults();
    converLayout->getResult(0).replaceAllUsesWith(result[0]);
    converLayout.erase();
    loadOp.erase();

  } else {
    auto localLoad = builder.create<triton::gcu::LocalLoadOp>(
        loc, loadOp.getType(), viewLoad, wait->getResult(0));
    loadOp->getResult(0).replaceAllUsesWith(localLoad->getResult(0));
    loadOp.erase();
  }
  // Prefetch load if is not MMAV3 and is used by the dot.
  schedule.insert(wait, numStages - 2, prefetchCluster);
  schedule.insert(viewLoad, numStages - 2, prefetchCluster);
}

// If all the transitive uses of the given value have are used by a convert to
// the same dot operand encoding, return true and get the shared encoding that
// needs to be used to be compatible with users' layouts.
static std::optional<ttg::SwizzledSharedEncodingAttr>
getSharedEncIfAllUsersAreSameEnc(Value val) {
  ttg::SwizzledSharedEncodingAttr attr;
  for (Operation *user : val.getUsers()) {
    ttg::SwizzledSharedEncodingAttr tempAttr;
    if (user->getNumResults() != 1)
      return std::nullopt;
    if (!isa<ttg::ConvertLayoutOp>(user))
      return std::nullopt;
    int64_t bufferSize = 1;
    if (auto tType = dyn_cast<RankedTensorType>(val.getType())) {
      bufferSize = triton::gcu::getBpe(tType.getElementType());
      auto shapes = tType.getShape();
      for (int32_t i = 0; i < tType.getRank(); i++) {
        bufferSize = bufferSize * shapes[i];
      }
    }
    auto converLayout = dyn_cast<triton::gpu::ConvertLayoutOp>(user);
    auto srcNumElems =
        triton::gcu::getElemsPerThread(converLayout.getSrc().getType());
    // if no warp share and buffer is small than 16k skip pinpong
    auto dstNumElems = triton::gcu::getElemsPerThread(converLayout.getType());
    // only for 300 now
    if (srcNumElems == dstNumElems && bufferSize < 32 * 1024) {
      return std::nullopt;
    }
    auto srcTy = cast<triton::gpu::TensorOrMemDesc>(val.getType());
    auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
    auto order = ttg::getOrder(srcTy);
    unsigned bitWidth = srcTy.getElementType().getIntOrFloatBitWidth();
    SmallVector<unsigned> sharedOrder;
    unsigned rank = order.size();
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
    if (auto dotOpEnc = dyn_cast<ttg::DotOperandEncodingAttr>(
            cast<triton::gpu::TensorOrMemDesc>(user->getResult(0).getType())
                .getEncoding())) {
      tempAttr = ttg::SwizzledSharedEncodingAttr::get(
          val.getContext(), dotOpEnc, srcTy.getShape(), sharedOrder, CTALayout,
          bitWidth, /*needTrans=*/false);
    } else {
      tempAttr = ttg::SwizzledSharedEncodingAttr::get(val.getContext(), 1, 1, 1,
                                                      sharedOrder, CTALayout);
    }

    // Check that the shared encodings needed by the users are compatible.
    if (!tempAttr || (attr != nullptr && attr != tempAttr))
      return std::nullopt;
    attr = tempAttr;
  }
  return attr;
}

// Create a map from load ops to their indirection level and the
// final use of the load op (another load op, or a dot op).
// Indirection level is "0" for the load op directly used by the dot op,
// "1" for the load op used by the load op used by the dot op, and so on.
static llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
loadOpsToIndirectionLevelAndUse(scf::ForOp forOp) {
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse;
  DenseSet<Operation *> seen;

  std::function<void(Operation * op, int, Operation *)> dfs =
      [&](Operation *op, int distance, Operation *use) {
        if (!seen.insert(op).second)
          return;
        if (isa<triton::gcu::LoadOp>(op)) {
          // TODO(xingxing): What if there are multiple uses at different
          // distances?
          loadOpToIndLevelAndUse.push_back(std::make_tuple(op, distance, use));
          use = op;
          distance++;
        }
        for (Value operand : op->getOperands()) {
          Value v = operand;
          Operation *defOp = v.getDefiningOp();
          if (defOp && defOp->getBlock() == op->getBlock()) {
            dfs(defOp, distance, use);
          }
        }
      };

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isa<tt::DotOp, triton::gcu::MatmulOp>(op))
      continue;
    seen.clear();
    dfs(&op, 0, &op);
  }

  // If the loop has numStages attribute, also consider pipelining other loads
  // that are not directly used by dot ops.
  if (forOp->hasAttr("tt.num_stages")) {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (!isa<triton::gcu::LoadOp>(op))
        dfs(&op, 0, &op);
    }
  }

  return loadOpToIndLevelAndUse;
}

static llvm::MapVector<Operation *, LoadInfo>
assignMemoryLayouts(llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
                        &loadOpToIndLevelAndUse) {
  llvm::MapVector<Operation *, LoadInfo> loadToInfo;

  for (auto &[op, dist, use] : loadOpToIndLevelAndUse) {
    (void)dist;
    if (loadToInfo.count(op))
      // TODO(pawel): err, we'd need to verify that the distance is the same
      continue;
    LoadInfo loadInfo;
    // for dot
    if (isa<tt::DotOp, triton::gcu::MatmulOp>(use) &&
        isa<triton::gcu::LoadOp>(op)) {
      loadInfo.usedByDot = true;
      loadInfo.sharedEncoding =
          getSharedEncIfAllUsersAreSameEnc(op->getResult(0)).value_or(nullptr);
    } else if (auto loadOp = dyn_cast<tt::LoadOp>(use)) {
      // The use of this loadOp is another loadOp. If the use is not in the
      // loadsToPipeline already, it means that the use is not valid for
      // pipelining for some reason. We should skip this loadOp, too. Note
      // that we have an assumption that distAndUse.second (i.e. the use of
      // this loadOp) has already be processed in a previous loop iteration.
      // This assumption is held by how loadOpsToIndirectionLevelAndUse
      // recursively collects loadOpToIndLevelAndUse using DFS.
      if (loadToInfo.count(loadOp) == 0) {
        continue;
      }
    } else {
      loadInfo.usedByDot = false;
      if (!isa<triton::gcu::LoadOp>(op)) {
        continue;
      }
      int64_t bufferSize = 1;
      if (auto tType = dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
        bufferSize = triton::gcu::getBpe(tType.getElementType());
        auto shapes = tType.getShape();
        for (int32_t i = 0; i < tType.getRank(); i++) {
          bufferSize = bufferSize * shapes[i];
        }
      }
      if (bufferSize < 1024) {
        continue;
      }
      auto gcuLoad = dyn_cast<triton::gcu::LoadOp>(op);
      auto srcTy = dyn_cast<RankedTensorType>(gcuLoad.getType());
      auto CTALayout = ttg::getCTALayout(srcTy.getEncoding());
      auto order = ttg::getOrder(srcTy);
      loadInfo.sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
          srcTy.getContext(), 1, 1, 1, order, CTALayout);
    }
    // If that still didn't work, bail on pipelining this load.
    if (!loadInfo.sharedEncoding) {
      continue;
    }
    loadToInfo[op] = loadInfo;
  }

  return loadToInfo;
}

static llvm::MapVector<Operation *, LoadInfo>
scheduleLoads(scf::ForOp forOp, CoarseSchedule &schedule,
              DenseSet<Operation *> &rootUsers, int numStages) {
  // Get all loads that are (transitively) used by dot ops and their distance
  // to the dot op.
  llvm::SmallVector<std::tuple<Operation *, int, Operation *>>
      loadOpToIndLevelAndUse = loadOpsToIndirectionLevelAndUse(forOp);
  LLVM_DEBUG({
    LDBG("Found " << loadOpToIndLevelAndUse.size() << " loads to pipeline:");
    for (const auto &[l, i, u] : loadOpToIndLevelAndUse) {
      LDBG("  - load: " << *l);
      LDBG("    at indirection level: " << i);
      LDBG("    used by op: " << *u);
    }
  });
  if (loadOpToIndLevelAndUse.empty())
    return {};

  // Check which loads are good for pipelining, and assign them
  // memory layouts.
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      assignMemoryLayouts(loadOpToIndLevelAndUse);

  if (loadToInfo.empty()) {
    LLVM_DEBUG({ LDBG("loadToInfo.empty \n"); });
    return {};
  }

  // Calculate the stage distance between applicable loads.
  int maxIndirectionLevel = -1;
  for (auto [loadOp, dist, use] : loadOpToIndLevelAndUse) {
    (void)use;
    if (loadToInfo.count(loadOp) == 0)
      continue;
    maxIndirectionLevel = std::max(maxIndirectionLevel, dist);
  }
  unsigned stagesBetweenLoads =
      ceil<unsigned>(numStages - 2, maxIndirectionLevel + 1);
  LLVM_DEBUG({
    LDBG("scheduleLoads::stagesBetweenLoads " << stagesBetweenLoads << "\n");
  });
  CoarseSchedule::Cluster rootUsersCluster = schedule.clusters.newAtFront();
  // Put the root uses of the loads in the last stage.
  for (auto &[loadOp, dist, use] : loadOpToIndLevelAndUse) {
    (void)dist;
    if (loadToInfo.count(loadOp) == 0)
      continue;
    // Non-LoadOp(s) are the root uses of all LoadOp(s) and should be
    // always present in the opInfo
    if (!isa<tt::LoadOp>(use)) {
      schedule.insert(use, numStages - 1, rootUsersCluster);
      rootUsers.insert(use);
    }
  }

  SmallVector<CoarseSchedule::Cluster> loadsClusters;
  for (int i = 0; i < maxIndirectionLevel + 1; i++) {
    loadsClusters.push_back(schedule.clusters.newAtBack());
  }
  // Assign stages to the loads.
  for (auto [loadOp, indLevel, _] : loadOpToIndLevelAndUse) {
    (void)_;
    if (loadToInfo.count(loadOp) == 0)
      continue;
    int stage = (maxIndirectionLevel - indLevel) * stagesBetweenLoads;
    schedule.insert(loadOp, stage, loadsClusters[indLevel]);
  }

  // Distance from the load to the use.
  for (auto [loadOp, _, use] : loadOpToIndLevelAndUse) {
    (void)_;
    if (loadToInfo.count(loadOp) == 0)
      continue;
    loadToInfo[loadOp].distToUse = schedule[use].first - schedule[loadOp].first;
  }

  return loadToInfo;
}

// Schedule the prologue and epilogue `if` ops in the loop, pushing them as
// close to the loop boundaries as possible. Return the cluster after the
// prologue (or the beginning of the loop if there is no prologue).
static CoarseSchedule::Cluster
schedulePrologueAndEpilogue(scf::ForOp forOp, CoarseSchedule &schedule,
                            DenseSet<Operation *> &rootUsers, int numStages) {
  CoarseSchedule::Cluster afterPrologue = schedule.clusters.begin();

  // Look for the IfOp that is in the backward slice any of the currently
  // scheduled ops and put it at the beginning of the loop.
  DenseMap<scf::IfOp, int> ifsToStage;
  // Go stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, _] : schedule.getOpsInOrder(forOp)) {
      (void)_;
      if (stage_ != stage)
        continue;
      SetVector<Operation *> backwardSlice;
      BackwardSliceOptions opt;
      opt.omitBlockArguments = true;
      getBackwardSlice(reinterpret_cast<Operation *>(op), &backwardSlice, opt);

      for (auto op : backwardSlice) {
        if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
          ifsToStage.insert({ifOp, stage});
        }
      }
    }
  }
  CoarseSchedule::Cluster prologueCluster = schedule.clusters.newAtFront();
  for (auto [ifOp, stage] : ifsToStage) {
    schedule.insert(ifOp, stage, prologueCluster);
  }

  // Look for the IfOp that is in the forward slice of the root users and put it
  // at the end of the loop.
  CoarseSchedule::Cluster epilogueCluster = schedule.clusters.newAtBack();
  for (auto rootUser : rootUsers) {
    SetVector<Operation *> forwardSlice;
    getForwardSlice(rootUser, &forwardSlice);

    int stage = schedule[rootUser].first;
    for (auto op : forwardSlice) {
      scf::IfOp ifOp = dyn_cast<scf::IfOp>(op);
      if (ifOp == nullptr) {
        // check if the op is in the body of an if op that's part of the loop
        auto parentOp = op->getParentOp();
        if (parentOp != nullptr &&
            parentOp->getParentOp() == forOp.getOperation()) {
          ifOp = dyn_cast<scf::IfOp>(parentOp);
        }
      }
      if (ifOp) {
        schedule.insertIfAbsent(ifOp, stage,
                                epilogueCluster); // after prefetch extracts
      }
    }
  }
  return afterPrologue;
}

// Add dependencies of anchor ops to the coarse schedule. Schedule them to
// the same stage and ordering cluster as the anchor op.
static void scheduleDependencies(scf::ForOp forOp, CoarseSchedule &schedule,
                                 int numStages) {
  SmallVector<std::tuple<Operation *, int, CoarseSchedule::Cluster>>
      opsInOrder = schedule.getOpsInOrder(forOp);
  // Schedule dependencies stage by stage.
  for (int stage = 0; stage < numStages; stage++) {
    for (auto [op, stage_, cluster] : opsInOrder) {
      if (stage_ != stage)
        continue;
      schedule.insertDepsOfOp(op, stage, cluster, false);
    }
  }
}

// Find dependencies with distance of 1. They will go to the next stage,
// but in the cluster before the current op.
static void scheduleDistanceOneDependencies(scf::ForOp forOp,
                                            CoarseSchedule &schedule,
                                            int numStages) {
  auto getNestedOperands = [](Operation *op) -> SmallVector<Value> {
    SmallVector<Value> operands;
    op->walk([&](Operation *nestedOp) {
      for (Value operand : nestedOp->getOperands()) {
        if (operand.getParentBlock()->getParentOp()->isAncestor(nestedOp))
          operands.push_back(operand);
      }
    });
    return operands;
  };

  // Mapping from the cluster to the cluster before it.
  DenseMap<CoarseSchedule::Cluster *, CoarseSchedule::Cluster> dist1Cluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0)
      continue;
    auto [stage, cluster] = schedule[&op];
    // Can't schedule past the last stage.
    if (stage == numStages - 1)
      continue;
    for (Value operand : getNestedOperands(&op)) {
      if (auto arg = dyn_cast<BlockArgument>(operand)) {
        if (arg.getArgNumber() > 0 && arg.getOwner() == op.getBlock()) {
          auto yieldOp = op.getBlock()->getTerminator();
          Value v = yieldOp->getOperand(arg.getArgNumber() - 1);
          Operation *defOp = v.getDefiningOp();
          if (defOp && schedule.count(defOp) == 0) {
            if (isa<tt::LoadOp>(defOp)) {
              // Exception: Schedule loads with a distance of 1 together
              // with the current op.
              schedule.insertIfAbsent(defOp, stage, cluster);
              schedule.insertDepsOfOp(defOp, stage, cluster, true);
            } else {
              if (dist1Cluster.count(&cluster) == 0) {
                dist1Cluster[&cluster] = schedule.clusters.newBefore(cluster);
              }
              schedule.insertIfAbsent(defOp, stage + 1, dist1Cluster[&cluster]);
              schedule.insertDepsOfOp(defOp, stage + 1, dist1Cluster[&cluster],
                                      true);
            }
          }
        }
      }
    }
  }
}

static void scheduleRemainingToLastStage(scf::ForOp forOp,
                                         CoarseSchedule &schedule,
                                         CoarseSchedule::Cluster afterPrologue,
                                         int numStages) {
  // Assign the rest of the ops to the last stage.
  // Take care of the ordering of the ops - uses cannot be scheduled to the
  // cluster before the definition.
  DenseMap<Operation *, CoarseSchedule::Cluster> opToCluster;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (schedule.count(&op) == 0) {
      opToCluster[&op] = afterPrologue;
    }
  }
  SmallVector<Operation *> queue;
  for (auto [op, stage, _] : schedule.getOpsInOrder(forOp)) {
    (void)_;
    // We really only care about the producers from the last stage.
    // Others will be scheduled before these ops anyway.
    if (stage == numStages - 1) {
      queue.push_back(op);
    }
  }
  while (!queue.empty()) {
    Operation *op = queue.pop_back_val();
    for (auto user : op->getUsers()) {
      if (opToCluster.count(user)) {
        CoarseSchedule::Cluster userCluster = opToCluster[user];
        CoarseSchedule::Cluster opCluster = schedule[op].second;
        if (*userCluster < *opCluster) {
          opToCluster[user] = opCluster;
          queue.push_back(user);
        }
      }
    }
  }
  for (auto [op, cluster] : opToCluster) {
    schedule.insert(op, numStages - 1, cluster);
  }
}

// Create an allocation that can hold distance number of loadOp shapes.
static Value createAlloc(scf::ForOp &forOp, Operation *loadOp,
                         ttg::SwizzledSharedEncodingAttr sharedEnc,
                         unsigned distance) {
  OpBuilder builder(forOp);
  auto ty = cast<RankedTensorType>(loadOp->getResultTypes()[0]);
  SmallVector<int64_t> bufferShape(ty.getShape().begin(), ty.getShape().end());
  bufferShape.insert(bufferShape.begin(), distance);
  auto memorySpace = mlir::IntegerAttr::get(
      mlir::IntegerType::get(loadOp->getContext(), 64), 2);
  Type memdescType = mlir::triton::gpu::MemDescType::get(
      bufferShape, ty.getElementType(), sharedEnc, memorySpace,
      /*mutableMemory*/ true);
  Value alloc = builder.create<mlir::triton::gcu::ShareAllocOp>(
      loadOp->getLoc(), memdescType, Value());
  return alloc;
}

struct AsyncLoad {
  AsyncLoad(Operation *loadOp, Value alloc) : loadOp(loadOp), alloc(alloc) {}
  Operation *loadOp;
  Value alloc;
  Value barrier;
  Operation *waitOp = nullptr;
  bool isTMALoad = false;
};

// Convert load ops into their asyn version and apply multi-buffering based on
// the required number of buffers.
static SmallVector<Value>
createAsyncOps(scf::ForOp &forOp, CoarseSchedule &schedule,
               llvm::MapVector<Operation *, LoadInfo> &loadToInfo,
               SmallVector<Value> &barriers, int numStages) {
  // Calculate the number of buffers needed for each load.
  // int numBuffers = 0;
  int numBuffers =
      llvm::max_element(llvm::make_second_range(loadToInfo), [](auto &lhs,
                                                                auto &rhs) {
        return lhs.distToUse < rhs.distToUse;
      })->distToUse;
  LLVM_DEBUG({
    llvm::dbgs() << "createshareAlloc:numBuffers=" << numBuffers << "\n";
  });
  SmallVector<AsyncLoad> asyncLoads;
  SmallVector<Value> allocs;
  bool hasTMALoad = false;
  for (auto &[loadOp, info] : loadToInfo) {
    assert(info.sharedEncoding && "LoadOp shared encoding not defined.");
    Value alloc = createAlloc(forOp, loadOp, info.sharedEncoding, numBuffers);
    assert(alloc && "Failed to create alloc for the async load.");
    allocs.push_back(alloc);
    asyncLoads.emplace_back(loadOp, alloc);
  }

  IRRewriter builder(forOp.getContext());
  builder.setInsertionPoint(forOp);

  Location loc = forOp.getLoc();
  // Create two new counters to index into the allocs.
  Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, 32);
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
  Value one = builder.create<arith::ConstantIntOp>(loc, 1, 32);
  Value insertIdx = minusOne;
  Value extractIdx = minusOne;
  Value phase = Value();
  Value numBuffersVal =
      builder.create<arith::ConstantIntOp>(loc, numBuffers, 32);
  SmallVector<Value> newOperands;
  newOperands.push_back(insertIdx);
  newOperands.push_back(extractIdx);
  if (hasTMALoad) {
    LLVM_DEBUG({ llvm::dbgs() << "createAsyncOps:has tma load\n"; });
    phase = builder.create<arith::ConstantIntOp>(loc, 0, 32);
    newOperands.push_back(phase);
  }
  unsigned newOperandIndex = forOp.getBody()->getNumArguments();
  // Patch the loop to add the new loop carried dependencies.
  scf::ForOp newForOp =
      replaceForOpWithNewSignature(builder, forOp, newOperands);
  forOp.erase();
  forOp = newForOp;
  insertIdx = newForOp.getBody()->getArgument(newOperandIndex);
  extractIdx = newForOp.getBody()->getArgument(newOperandIndex + 1);
  if (phase) {
    phase = newForOp.getBody()->getArgument(newOperandIndex + 2);
  }

  // Create two counters for the insert and extract indices to avoid creating
  // long liverange.
  builder.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
  insertIdx = builder.create<arith::AddIOp>(loc, insertIdx, one);
  Value cndIns = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               insertIdx, numBuffersVal);
  insertIdx = builder.create<arith::SelectOp>(loc, cndIns, insertIdx, zero);

  extractIdx = builder.create<arith::AddIOp>(loc, extractIdx, one);
  Value cndExt = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               extractIdx, numBuffersVal);
  extractIdx = builder.create<arith::SelectOp>(loc, cndExt, extractIdx, zero);
  if (phase) {
    Value nextPhase = builder.create<arith::XOrIOp>(loc, phase, one);
    phase = builder.create<arith::SelectOp>(loc, cndExt, phase, nextPhase);
  }
  // createTMABarrierAndWait(forOp, asyncLoads, insertIdx, extractIdx, phase,
  //                         numBuffers, schedule, barriers, loadToInfo);

  // Create a cluster for the prefetches. It may end up being empty, but this
  // is OK.
  CoarseSchedule::Cluster prefetchCluster = schedule.clusters.newAtBack();

  for (AsyncLoad &asyncLoad : asyncLoads) {
    if (auto loadOp = dyn_cast<triton::gcu::LoadOp>(asyncLoad.loadOp)) {
      createAsyncCopy(forOp, loadOp, asyncLoad.alloc, insertIdx, extractIdx,
                      schedule, prefetchCluster, loadToInfo, numStages);
    }
  }
  SmallVector<Value> newYieldOperands = {insertIdx, extractIdx};
  if (phase)
    newYieldOperands.push_back(phase);
  // Patch the yield with the updated counters.
  appendToYield(forOp, newYieldOperands);

  return allocs;
}

bool mlir::triton::gcu::preProcessLoopAndGetSchedule(
    scf::ForOp &forOp, int numStages,
    mlir::triton::gcu::PipeliningOption &options) {
  // Schedule the loads and root ops (dot ops) in the loop. This will give us
  // a scaffold for the final schedule.
  DenseSet<Operation *> rootUsers;
  CoarseSchedule coarseSchedule(numStages);
  llvm::MapVector<Operation *, LoadInfo> loadToInfo =
      scheduleLoads(forOp, coarseSchedule, rootUsers, numStages);
  if (loadToInfo.empty())
    return false;
  LLVM_DEBUG({
    LDBG("Coarse schedule loads only:");
    coarseSchedule.dump();
  });

  SmallVector<Value> barriers; // only for TMA
  // Convert the loads into async loads and create the allocs.
  SmallVector<Value> allocs =
      createAsyncOps(forOp, coarseSchedule, loadToInfo, barriers, numStages);

  LLVM_DEBUG({
    LDBG("Coarse schedule with async loads:");
    coarseSchedule.dump();
  });

  CoarseSchedule::Cluster afterPrologue =
      schedulePrologueAndEpilogue(forOp, coarseSchedule, rootUsers, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with prologue and epilogue:");
    coarseSchedule.dump();
  });

  scheduleDependencies(forOp, coarseSchedule, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dependencies:");
    coarseSchedule.dump();
  });

  scheduleDistanceOneDependencies(forOp, coarseSchedule, numStages);
  LLVM_DEBUG({
    LDBG("Coarse schedule with dist 1:");
    coarseSchedule.dump();
  });

  scheduleRemainingToLastStage(forOp, coarseSchedule, afterPrologue, numStages);
  LLVM_DEBUG({
    LDBG("Final coarse schedule:");
    coarseSchedule.dump();
  });

  // Create the final schedule for the kernel loop. This will dictate the
  // stages and order of operations to the pipeline expander.
  std::vector<std::pair<Operation *, unsigned>> schedule =
      coarseSchedule.createFinalSchedule(forOp);

  // Fill out the pipeline options.
  options.getScheduleFn =
      [schedule](scf::ForOp forOp,
                 std::vector<std::pair<Operation *, unsigned>> &s) {
        s = std::move(schedule);
      };
  options.peelEpilogue = false;
  options.predicateFn = triton::gcu::predicateOp;
  options.supportDynamicLoops = true;
  options.annotateFn =
      [](Operation *op, mlir::triton::gcu::PipeliningOption::PipelinerPart part,
         unsigned iteration) {};
  // Insert a wait 0 after the loop
  OpBuilder builder(forOp);
  builder.setInsertionPointAfter(forOp);
  // we need skip last async_load_global_to_share
  // builder.create<ttg::AsyncWaitOp>(forOp.getLoc(), ValueRange({}), 0);
  // Invalidate any mbarrier create
  // no need for gcu
  // invalidateBarriers(builder, barriers);
  // Explicitly deallocate allocated tensors after the wait op
  for (auto alloc : allocs)
    builder.create<triton::gcu::ShareDeallocOp>(forOp.getLoc(), alloc);
  return true;
}
