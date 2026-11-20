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

// Materialize the TLE memory-space hint (`tle.gpu.memory_space(...)`) into
// concrete GCU IR.
//
// The frontend only attaches `tt.memory_space` to the defining op of the
// hinted value and nothing in the GCU pipeline consumed it. This pass turns
// the hint into IR that the existing TritonToGCU lowerings already understand:
//
//   shared_memory:
//     %v    = <producer>
//     %buf  = ttg.local_alloc %v : !ttg.memdesc<..., #shared, #smem, mutable>
//     %hint = ttg.local_load %buf : tensor<...>       // same type as %v
//     ... uses of %v replaced by %hint ...
//     and the producer is marked so its lowering writes straight into the
//     smem tile (kTleResultInSmemTile, read by the elementwise and
//     elementwise-fusion lowerings) instead of writing a private buffer and
//     deslicing it.
//
//   register:
//     the producer is marked so its output buffer is a memref.alloca
//     (vector.load/vector.store) instead of a memref.alloc (tar path).
//
// `tt.memory_space` is an op-level attribute - the frontend's
// `value.set_attr()` has no way to attach it to a single result - so it can
// only describe a producer with exactly one ranked-tensor result. That covers
// single-output ops and `tle.exclusive_cumsum`, whose other result is the scan
// total (a scalar); a producer with several tensor results would not be able to
// say which one was hinted, so its hint is dropped with a remark.
//
// The pass runs once, after gcu-triton-fusion and before
// triton-gcu-local-mem-optimize. That single position satisfies all the
// constraints:
//   * after triton-gpu-to-triton-gcu, whose rewriteLocalLoad turns any
//     pre-existing local_alloc(src)+local_load pair into a plain
//     ttg.convert_layout (which would erase the staging);
//   * after the DMA conversion, so a hinted load is already a
//     triton_gcu.load and the staging pair written here is still folded by
//     FuseLoadLocalAllocPattern into a zero-copy copy_global_to_local (that is
//     why the pair must be created before triton-gcu-local-mem-optimize);
//   * after gcu-triton-fusion, so elementwise producers already live inside
//     triton_gcu.elementwise_fusion_region ops and the hint is applied to the
//     corresponding *region result*, recorded with an index attr on the region.

#include <optional>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "Conversion/TritonToGCU/Constants.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

// Defined in triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h. Forward
// declared here to avoid a build dependency on the TritonGPU transforms
// library, the same way GCUTleLowerAsyncLoad.cpp does.
namespace mlir::triton {
gpu::SharedEncodingTrait getSharedEncoding(Operation *loadOp);
} // namespace mlir::triton

namespace mlir {
#define GEN_PASS_DEF_GCUTLEASSIGNMEMORYSPACE
#include "Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "gcu-tle-assign-memory-space"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcu = mlir::triton::gcu;

namespace {

// If `value` is produced inside an elementwise fusion region and observed
// outside as a region result, return the region op and that result index.
// The fusion pass clones the inner ops (attributes included) and yields their
// results, so a hint attached to an inner op reaches this pass on the inner op
// while the value that consumers see is the region result.
static std::optional<std::pair<Operation *, unsigned>>
getFusionRegionResult(Value value) {
  for (Operation *user : value.getUsers()) {
    auto yieldOp = dyn_cast<ttgcu::YieldOp>(user);
    if (!yieldOp)
      continue;
    Operation *parent = yieldOp->getParentOp();
    if (!isa<ttgcu::ElementwiseFusionRegionOp>(parent))
      continue;
    auto operands = yieldOp->getOperands();
    auto it = llvm::find(operands, value);
    assert(it != operands.end() && "yield user without matching operand");
    return std::make_pair(parent, it - operands.begin());
  }
  return std::nullopt;
}

// Record `index` in the DenseI64ArrayAttr `attrName` on `region`, keeping the
// list sorted and duplicate-free.
static void markResultIndex(Operation *region, StringRef attrName,
                            unsigned index) {
  SmallVector<int64_t> indices;
  if (auto old = region->getAttrOfType<DenseI64ArrayAttr>(attrName))
    llvm::append_range(indices, old.asArrayRef());
  if (!llvm::is_contained(indices, static_cast<int64_t>(index)))
    indices.push_back(index);
  llvm::sort(indices);
  region->setAttr(attrName,
                  DenseI64ArrayAttr::get(region->getContext(), indices));
}

// Consume the frontend hint. Only this pass may look at the `tt.memory_space`
// attribute, and it must not survive into the lowerings.
static void dropHint(Operation *op) { op->removeAttr(kTleHint); }

class GCUTleAssignMemorySpacePass
    : public impl::GCUTleAssignMemorySpaceBase<GCUTleAssignMemorySpacePass> {
public:
  using impl::GCUTleAssignMemorySpaceBase<
      GCUTleAssignMemorySpacePass>::GCUTleAssignMemorySpaceBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();

    // Collect first: the rewrite inserts ops and rewires uses.
    SmallVector<Operation *> candidates;
    moduleOp.walk([&](Operation *op) {
      auto spaceAttr = op->getAttrOfType<StringAttr>(kTleHint);
      if (!spaceAttr)
        return;
      StringRef space = spaceAttr.getValue();
      if (space != kTleHintShared && space != kTleHintRegister) {
        op->emitWarning() << "unknown " << kTleHint << " value '" << space
                          << "', ignored";
        return;
      }
      candidates.push_back(op);
    });

    OpBuilder builder(moduleOp.getContext());
    for (Operation *op : candidates)
      rewriteHint(builder, op);
  }

private:
  void rewriteHint(OpBuilder &builder, Operation *op) {
    StringRef space = op->getAttrOfType<StringAttr>(kTleHint).getValue();

    // `tt.memory_space` is an op-level attribute: the frontend's
    // `value.set_attr()` (tle.gpu.memory_space) has no way to attach it to a
    // single result, so a producer with several results cannot say which one
    // is hinted. Materialize the hint only when the op has exactly one
    // ranked-tensor result - a single-output op, or a multi-output one like
    // `tle.exclusive_cumsum`, which returns the scanned tensor and the scan
    // total (a scalar).
    Value value;
    for (Value result : op->getResults()) {
      if (!isa<RankedTensorType>(result.getType()))
        continue;
      if (value) {
        value = Value();
        break;
      }
      value = result;
    }
    if (!value) {
      op->emitRemark() << kTleHint
                       << " is only materialized for a producer with exactly "
                          "one tensor result, ignored";
      dropHint(op);
      return;
    }

    auto fusion = getFusionRegionResult(value);

    // A hint on an op inside a fusion region whose value is not yielded: the
    // value never escapes the region (its consumer was fused in as well, e.g.
    // a store), so there is no region result to attach the region-level
    // markers to, and inserting a staging pair inside the body would only
    // fight the fusion. The producer is already reading/writing registers, so
    // the hint is effectively honored by the existing lowering.
    if (!fusion && op->getParentOfType<ttgcu::ElementwiseFusionRegionOp>()) {
      op->emitRemark() << kTleHint << " ignored: the value stays inside its "
                       << ttgcu::ElementwiseFusionRegionOp::getOperationName()
                       << " and does not escape it";
      dropHint(op);
      return;
    }

    if (space == kTleHintRegister) {
      // The marker is a UnitAttr on the producer itself (it has exactly one
      // tensor result, checked above), or the result index the value escapes
      // as when the producer sits inside a fusion region.
      if (fusion)
        markResultIndex(fusion->first, kTleResultInAlloca, fusion->second);
      else
        op->setAttr(kTleResultInAlloca, builder.getUnitAttr());
      dropHint(op);
      return;
    }

    assert(space == kTleHintShared);
    // The staging pair must be built on the value consumers see: inside a
    // fusion region that is the region result, not the inner op result (which
    // is only visible to the region's yield and cannot be used outside).
    if (fusion)
      value = fusion->first->getResult(fusion->second);
    // Do not stack a second staging level on top of an explicit one.
    if (llvm::any_of(value.getUsers(), [](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        })) {
      op->emitRemark() << kTleHint << " = " << kTleHintShared
                       << " ignored: the value is already staged by a "
                       << ttg::LocalAllocOp::getOperationName();
      dropHint(op);
      return;
    }

    // Tell the producer lowering it can compute directly into the smem tile.
    Operation *anchor = op;
    if (fusion) {
      markResultIndex(fusion->first, kTleResultInSmemTile, fusion->second);
      anchor = fusion->first;
    } else {
      op->setAttr(kTleResultInSmemTile, builder.getUnitAttr());
    }

    auto tensorType = cast<RankedTensorType>(value.getType());
    auto sharedEncoding = tt::getSharedEncoding(anchor);
    auto memDescType = ttg::MemDescType::get(
        tensorType.getShape(), tensorType.getElementType(), sharedEncoding,
        ttg::SharedMemorySpaceAttr::get(op->getContext()),
        /*mutableMemory=*/true);

    builder.setInsertionPointAfter(anchor);
    auto alloc =
        builder.create<ttg::LocalAllocOp>(op->getLoc(), memDescType, value);
    alloc->setAttr(kTleResultInSmemTile, builder.getUnitAttr());
    auto localLoad =
        builder.create<ttg::LocalLoadOp>(op->getLoc(), tensorType, alloc);
    value.replaceUsesWithIf(localLoad.getResult(), [&](OpOperand &use) {
      return use.getOwner() != alloc.getOperation();
    });

    LDBG("staged " << value << " in shared memory");
    dropHint(op);
  }
};

} // namespace
