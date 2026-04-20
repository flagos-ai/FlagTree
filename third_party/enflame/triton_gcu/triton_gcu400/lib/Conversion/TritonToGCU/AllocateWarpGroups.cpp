/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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

#include "Conversion/Passes.h"

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-allocate-warp-groups"

namespace mlir {
#define GEN_PASS_DEF_TRITONGCUALLOCATEWARPGROUPS
#include "Conversion/Passes.h.inc"
} // namespace mlir

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

// 1 Given a `ttg.warp_specialize` with a certain number of existing warps, pad
// it with extra warps until it has the same number of full warp groups as the
// largest partitioning. This ensures that all threads can be present to
// surrender registers.
// 2 set warpGroupStartIds
// 3 set actualRegisters
static void padToMaxWarpGroups(WarpSpecializeOp op, int numExtraWarpGroups,
                               int numWarps) {
  int numExtraWarps = op.getTotalPartitionWarps();
  int warpsToAdd = numExtraWarpGroups * numWarps - numExtraWarps;
  assert(warpsToAdd >= 0);

  // Fill it with powers of 2.
  SmallVector<int> paddingPartitionSizes;
  while (warpsToAdd > 0) {
    int paddingSize = llvm::NextPowerOf2(warpsToAdd) / 2;
    paddingPartitionSizes.push_back(paddingSize);
    warpsToAdd -= paddingSize;
  }

  auto partitions = cast<WarpSpecializePartitionsOp>(
      op.getPartitionOpHolder().front().front());
  OperationState state(partitions.getLoc(), partitions.getOperationName());
  for (Region *region : partitions.getRegions())
    state.addRegion()->takeBody(*region);

  SmallVector<int32_t> partitionNumWarps(op.getPartitionNumWarps());
  for (int paddingSize : paddingPartitionSizes) {
    partitionNumWarps.push_back(paddingSize);

    Block &body = state.addRegion()->emplaceBlock();
    for (Value capture : op.getExplicitCaptures())
      body.addArgument(capture.getType(), capture.getLoc());
    OpBuilder b(op.getContext());
    b.setInsertionPointToStart(&body);
    b.create<WarpReturnOp>(op.getLoc());
  }
  op.setPartitionNumWarps(partitionNumWarps);

  // Set the requested registers to low for the padded partitions that do
  // nothing.
  if (auto reqRegs = op.getRequestedRegisters()) {
    SmallVector<int32_t> newReqRegs(*reqRegs);
    newReqRegs.append(paddingPartitionSizes.size(), 16);
    op.setRequestedRegisters(newReqRegs);
  }

  OpBuilder b(partitions);
  b.create(state);
  partitions.erase();
}

class AllocateWarpGroupsPass
    : public mlir::impl::TritonGCUAllocateWarpGroupsBase<
          AllocateWarpGroupsPass> {
public:
  using mlir::impl::TritonGCUAllocateWarpGroupsBase<
      AllocateWarpGroupsPass>::TritonGCUAllocateWarpGroupsBase;

  void runOnOperation() override {
    mlir::gpu::GPUModuleOp mod = getOperation();

    // Warp number and register count are hardcoded for now.
    const unsigned fixedNumWarps = 4;
    // const unsigned totalRegCount = 768;
    const unsigned minRegCount = 8;

    int numWarps = ttg::lookupNumWarps(mod);
    int baseNumWarps = numWarps;

    // First determine the maximum number of extra warps.
    // Notice: currently only support one WarpSpecializeOp per function.
    int maxExtraWarps = 0;
    mod.walk([&](ttg::WarpSpecializeOp op) {
      maxExtraWarps = std::max<int>(maxExtraWarps, op.getTotalPartitionWarps());
    });
    LDBG() << "maxExtraWarps: " << maxExtraWarps;

    // Round this up to the nearest warpgroup (multiple of 4) and then pad each
    // `ttg.warp_specialize` to the nearest warpgroup.
    int numExtraWarpGroups = llvm::divideCeil(maxExtraWarps, numWarps);
    mod.walk([&](ttg::WarpSpecializeOp op) {
      padToMaxWarpGroups(op, numExtraWarpGroups, numWarps);
    });

    // Determine the maximum number of registers per thread. This may have
    // been set by the user.
    // int maxnreg = totalRegCount;
    // if (auto maxnregAttr =
    //         mod->getAttrOfType<IntegerAttr>(AttrMaxRegistersName)) {
    //   maxnreg = maxnregAttr.getInt();
    // }
    // LDBG() << "maxnreg: " << maxnreg;

    // Define the data structures for the warp group information.
    struct WarpGroupInfo {
      SmallVector<Region *> partitions;
      int maxRequestedRegs = 0;
      unsigned numWarps = 0;
    };
    struct WarpGroupPartition {
      int startId;
      Region *partition;
      int32_t estRegs;
      int numWarps;
    };

    // Compute the total number of warps required at any given time.
    mod.walk([&](ttg::WarpSpecializeOp op) {
      ArrayRef<int32_t> arr = op.getPartitionNumWarps();

      // Allocate the start IDs such that the largest warpgroups have lower
      // starting warp IDs.
      SmallVector<std::pair<unsigned, int32_t>> idxAndSize;
      for (auto [i, size] : llvm::enumerate(arr))
        idxAndSize.emplace_back(i, size);
      llvm::sort(idxAndSize,
                 [&](auto lhs, auto rhs) { return lhs.second > rhs.second; });

      SmallVector<int32_t> startIds(arr.size());
      int startId = baseNumWarps;
      for (auto [i, size] : idxAndSize) {
        startIds[i] = startId;
        startId += size;
      }
      op.setWarpGroupStartIds(startIds);

      // Require that an estimate has been set and that we have even warpgroups.
      auto regsAttr = op.getRequestedRegisters();
      if (!regsAttr || op.getTotalPartitionWarps() % fixedNumWarps != 0)
        return;

      // Group the partitions into warpgroups.
      SmallVector<WarpGroupPartition> orderedPartitions;
      for (auto [startId, partition, estRegs, numWarps] :
           llvm::zip(startIds, op.getPartitionRegions(), *regsAttr, arr))
        orderedPartitions.push_back({startId, partition, estRegs, numWarps});
      llvm::sort(orderedPartitions,
                 [&](auto lhs, auto rhs) { return lhs.startId < rhs.startId; });

      // Iterate over the partitions and assign them to warp groups. Determine
      // the maximum number of requested registers per warp group.
      SmallVector<WarpGroupInfo> warpGroups;
      for (auto [startId, partition, estRegs, numWarps] : orderedPartitions) {
        if (startId % fixedNumWarps == 0) {
          warpGroups.push_back(WarpGroupInfo{});
        }
        warpGroups.back().partitions.push_back(partition);
        // Round up the nearest multiple of minRegCount.
        int estRegsCeil = llvm::divideCeil(estRegs, minRegCount) * minRegCount;
        warpGroups.back().maxRequestedRegs =
            std::max<int>(warpGroups.back().maxRequestedRegs, estRegsCeil);
        warpGroups.back().numWarps += numWarps;
      }

      LLVM_DEBUG({
        LDBG() << "warpGroups info:";
        for (const WarpGroupInfo &wg : warpGroups) {
          LDBG() << "  numWarps: " << wg.numWarps;
          LDBG() << "  maxRequestedRegs: " << wg.maxRequestedRegs;
          for (Region *region : wg.partitions) {
            LDBG() << "  region: " << region->getRegionNumber();
          }
        }
      });

      // Generate setmaxnreg in each partition according to its warp group.
      SmallVector<int32_t> maxnregsPerPartition(1 + arr.size());
      for (const WarpGroupInfo &wg : warpGroups) {
        for (Region *region : wg.partitions) {
          maxnregsPerPartition[1 + region->getRegionNumber()] =
              wg.maxRequestedRegs;
        }
      }

      // Set the register usage for the default warp group.
      maxnregsPerPartition.front() = 0;
      op.setActualRegisters(maxnregsPerPartition);

      // // Set the initial max number of registers. This is needed for PTXAS to
      // // cooperate.
      // mod->setAttr(AttrMaxRegistersName,
      //              Builder(op.getContext()).getI32IntegerAttr(maxnreg));
    });

    Builder b(&getContext());
    mod->setAttr(
        "ttg.total-num-warps",
        b.getI32IntegerAttr(baseNumWarps + numExtraWarpGroups * numWarps));
  }
};

} // namespace
