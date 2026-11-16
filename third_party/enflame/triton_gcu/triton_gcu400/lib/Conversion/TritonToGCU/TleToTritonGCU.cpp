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

#include <utility>

#include "Constants.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Utils/TritonVersionCompat.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tle-to-triton-gcu"

namespace mlir {
#define GEN_PASS_DEF_TLETOTRITONGCUPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// ===----------------------------------------------------------------------===
// Phase 1: Insert barriers between local_ptr store→load transitions.
//
// Adapted from the official TleInsertLocalPointerBarriers pass.
// Runs BEFORE tle.local_pointers lowering so we can directly track
// the high-level tle.local_pointers result values.
//
// The key idea: multiple tle.local_ptr calls on the same tle.alloc
// buffer form a "barrier group". A tt.store to any pointer in the group
// marks the group dirty; a subsequent tt.load from any pointer in the
// group triggers a gpu.barrier insertion and clears the dirty flag.
// ===----------------------------------------------------------------------===

struct BarrierGroupMaps {
  llvm::DenseMap<Value, int64_t> ptrToGroup;
  llvm::DenseMap<Value, int64_t> memdescToGroup;
};

/// Assign a group ID for a memdesc, reusing an existing one if present.
static int64_t getOrCreateGroup(llvm::DenseMap<Value, int64_t> &memdescToGroup,
                                Value memdesc, int64_t &nextGroupId) {
  auto it = memdescToGroup.find(memdesc);
  if (it != memdescToGroup.end())
    return it->second;
  int64_t groupId = nextGroupId++;
  memdescToGroup[memdesc] = groupId;
  return groupId;
}

/// Trace through MemDesc view ops (subslice, trans, reshape, index) to
/// the root local_alloc, so barrier groups are keyed on the underlying
/// allocation rather than the view op result.
static Value traceToRootAlloc(Value memdesc) {
  while (auto *def = memdesc.getDefiningOp()) {
    if (isa<triton::gpu::MemDescSubsliceOp, triton::gpu::MemDescTransOp,
            triton::gpu::MemDescReshapeOp, triton::gpu::MemDescIndexOp>(def))
      memdesc = def->getOperand(0);
    else
      break;
  }
  return memdesc;
}

/// Collect all tle.local_pointers result values and their derived values
/// (through broadcast, convert_layout), grouped by their source memdesc.
/// Also registers memdesc values from ttg.tma_copy into the same groups
/// so that Phase 1 can insert barriers across tma_copy / local_ptr
/// interactions on the same shared memory buffer.
static BarrierGroupMaps collectLocalPointerGroups(gpu::GPUModuleOp module) {
  BarrierGroupMaps maps;
  int64_t nextGroupId = 0;

  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "tle.local_pointers")
      return;
    if (op->getNumOperands() < 1 || op->getNumResults() != 1)
      return;

    Value memdesc = traceToRootAlloc(op->getOperand(0));
    int64_t groupId =
        getOrCreateGroup(maps.memdescToGroup, memdesc, nextGroupId);
    maps.ptrToGroup[op->getResult(0)] = groupId;
  });

  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "ttg.tma_copy")
      return;
    if (op->getNumOperands() < 2)
      return;
    Value src = op->getOperand(0), dst = op->getOperand(1);
    Value memdesc;
    if (isa<triton::TensorDescType>(src.getType()) &&
        isa<triton::gpu::MemDescType>(dst.getType()))
      memdesc = dst;
    else if (isa<triton::gpu::MemDescType>(src.getType()) &&
             isa<triton::TensorDescType>(dst.getType()))
      memdesc = src;
    else
      return;
    getOrCreateGroup(maps.memdescToGroup, memdesc, nextGroupId);
  });

  // Propagate through broadcast / convert_layout chains.
  SmallVector<Value> worklist;
  for (auto &kv : maps.ptrToGroup)
    worklist.push_back(kv.first);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    int64_t groupId = maps.ptrToGroup[current];
    for (OpOperand &use : current.getUses()) {
      Operation *owner = use.getOwner();
      Value derived;
      if (auto bcast = dyn_cast<triton::BroadcastOp>(owner))
        derived = bcast.getResult();
      else if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(owner))
        derived = cvt.getResult();
      else
        continue;
      if (maps.ptrToGroup.insert({derived, groupId}).second)
        worklist.push_back(derived);
    }
  }
  return maps;
}

static void processBlockForBarriers(Block &block, const BarrierGroupMaps &maps,
                                    llvm::DenseMap<int64_t, bool> &dirty);

static void processRegionForBarriers(Region &region,
                                     const BarrierGroupMaps &maps,
                                     llvm::DenseMap<int64_t, bool> &dirty) {
  for (Block &block : region)
    processBlockForBarriers(block, maps, dirty);
}

static void processBlockForBarriers(Block &block, const BarrierGroupMaps &maps,
                                    llvm::DenseMap<int64_t, bool> &dirty) {
  for (Operation &op : block) {
    if (auto store = dyn_cast<triton::StoreOp>(&op)) {
      auto it = maps.ptrToGroup.find(store.getPtr());
      if (it != maps.ptrToGroup.end())
        dirty[it->second] = true;
    } else if (auto load = dyn_cast<triton::LoadOp>(&op)) {
      auto it = maps.ptrToGroup.find(load.getPtr());
      if (it != maps.ptrToGroup.end() && dirty.lookup(it->second)) {
        OpBuilder builder(load);
        builder.create<gpu::BarrierOp>(load.getLoc());
        dirty[it->second] = false;
      }
    } else if (op.getName().getStringRef() == "ttg.tma_copy") {
      if (op.getNumOperands() >= 2) {
        Value src = op.getOperand(0), dst = op.getOperand(1);
        bool isG2L = isa<triton::TensorDescType>(src.getType()) &&
                     isa<triton::gpu::MemDescType>(dst.getType());
        bool isL2G = isa<triton::gpu::MemDescType>(src.getType()) &&
                     isa<triton::TensorDescType>(dst.getType());
        Value memdesc = isG2L ? dst : src;
        auto it = maps.memdescToGroup.find(memdesc);
        if (it != maps.memdescToGroup.end()) {
          if (isG2L) {
            dirty[it->second] = true;
          } else if (isL2G && dirty.lookup(it->second)) {
            OpBuilder builder(&op);
            builder.create<gpu::BarrierOp>(op.getLoc());
            dirty[it->second] = false;
          }
        }
      }
    } else if (isa<gpu::BarrierOp>(&op)) {
      dirty.clear();
    }

    for (Region &nested : op.getRegions())
      processRegionForBarriers(nested, maps, dirty);
  }
}

/// Phase 1 entry: insert gpu.barrier between store→load transitions
/// on tle.local_pointers and ttg.tma_copy sharing the same memdesc
/// (barrier group).
static void insertLocalPointerBarriers(gpu::GPUModuleOp module) {
  auto maps = collectLocalPointerGroups(module);
  if (maps.ptrToGroup.empty() && maps.memdescToGroup.empty())
    return;

  llvm::DenseMap<int64_t, bool> dirty;
  for (Operation &op : module.getBody()->getOperations())
    for (Region &region : op.getRegions())
      processRegionForBarriers(region, maps, dirty);
}

/// Trace a value back to the `tle.local_pointers` op that produced it, looking
/// through the broadcast / convert_layout chains the lowering inserts.
static Operation *traceToLocalPointers(Value ptr) {
  while (Operation *def = ptr.getDefiningOp()) {
    if (def->getName().getStringRef() == "tle.local_pointers")
      return def;
    if (isa<triton::BroadcastOp, triton::gpu::ConvertLayoutOp>(def)) {
      ptr = def->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Trace a stored value back to the `tt.load` that produced it, looking through
/// layout-only ops (`tt.broadcast` / `ttg.convert_layout`) that may be inserted
/// between the load and the store.
static triton::LoadOp traceToLoad(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto load = dyn_cast<triton::LoadOp>(def))
      return load;
    if (isa<triton::BroadcastOp, triton::gpu::ConvertLayoutOp>(def)) {
      value = def->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Step 2b: mark the global loads that fill a TLE pipe slot as async.
///
/// What it does:
///   Attaches `tt.load.async` to every global `tt.load` whose result is stored
///   into a buffer backing a `tle.pipe`, so that the later DTE lowering issues
///   the producer's A and B copies as overlapped asynchronous transfers.
///
/// Why it is needed:
///   `TTSmemLowerToGCU` picks between a private *async* DTE tag and the single
///   shared *sync* tag from `tt.load.async` plus a `firstUse` wait anchor.
///   Only the async branch calls `pTagPool.setMap(firstUse, tag)`, and that is
///   what defers the `dma_wait` to the producer commit.  With the sync tag
///   every copy waits for its own transfer right after issuing it, which
///   serializes the producer's A and B DTEs (`loadA, waitA, loadB, waitB`)
///   instead of letting them overlap (`loadA, loadB, waitA, waitB`).
///
///   Auto-WS gets the attribute for free: `BlockedToMMA`
///   (tritongcu-accelerate-matmul) sets it on the loads behind the dot
///   operands while the loop is still plain `tt.load -> tt.dot`.  In the TLE
///   pipeline `tle-lower-pipe-to-gcuws` has already decoupled producer from
///   consumer by the time that pass runs, so nothing behind the dot operand
///   traces back to a `tt.load`; the attribute is never set and the pipe
///   producer falls back to the shared sync tag.
///
///   A store of a globally loaded tile into a TLE pipe slot *is* that
///   producer -> consumer handoff, so it is marked here.  This runs before
///   `ConvertLocalPointersOp` rewrites the store into `ttg.local_store`, and
///   `TritonGCULocalMemOptimize::FuseLoadLocalStorePattern` already propagates
///   the attribute onto the resulting `copy_global_to_local`.
///
/// How it works:
///   (a) Collect the shared buffers behind every `tle.pipe.*` op, funnelling
///       views through their root alloc.  Both operands and results are
///       inspected: the default region refers to the `ttg.local_alloc`
///       directly, while a worker partition sees the same memdesc as a block
///       argument.
///   (b) Keep the stores whose pointer traces back to a `tle.local_pointers`
///       on a pipe buffer and whose value traces back to a *global* `tt.load`
///       (address space != shared).  Shared -> pipe copies are skipped because
///       they must stay synchronous.
///   (c) Attach `tt.load.async` to the collected loads.
static void markPipeFillLoadsAsync(gpu::GPUModuleOp module) {
  constexpr int kSharedAS = 3;

  // (a) Shared buffers backing a TLE pipe, keyed on their root alloc.
  llvm::DenseSet<Value> pipeMemdescs;
  module.walk([&](Operation *op) {
    if (!op->getName().getStringRef().starts_with("tle.pipe."))
      return;
    for (Value operand : op->getOperands())
      if (isa<triton::gpu::MemDescType>(operand.getType()))
        pipeMemdescs.insert(traceToRootAlloc(operand));
    for (Value result : op->getResults())
      if (isa<triton::gpu::MemDescType>(result.getType()))
        pipeMemdescs.insert(traceToRootAlloc(result));
  });
  if (pipeMemdescs.empty())
    return;

  // (b) Collect the global loads stored into a pipe slot.
  SmallVector<triton::LoadOp> loads;
  module.walk([&](triton::StoreOp store) {
    auto *localPointers = traceToLocalPointers(store.getPtr());
    if (!localPointers || localPointers->getNumOperands() < 1)
      return;
    if (!pipeMemdescs.contains(traceToRootAlloc(localPointers->getOperand(0))))
      return;
    triton::LoadOp loadOp = traceToLoad(store.getValue());
    if (!loadOp)
      return;
    // Only global -> shared staging moves through the DTE; a shared memory
    // read stored back into a pipe slot has to stay synchronous.
    auto ptrTy = dyn_cast<RankedTensorType>(loadOp.getPtr().getType());
    if (!ptrTy)
      return;
    auto elemTy = dyn_cast<triton::PointerType>(ptrTy.getElementType());
    if (!elemTy || elemTy.getAddressSpace() == kSharedAS)
      return;
    loads.push_back(loadOp);
  });

  // (c) Mark only after the walk is done, so the IR is not mutated while
  // iterating.
  for (triton::LoadOp loadOp : loads)
    loadOp->setAttr(kLoadAsync, BoolAttr::get(module.getContext(), true));
}

// ===----------------------------------------------------------------------===
// Shared helpers for canonicalization and optimization passes.
// ===----------------------------------------------------------------------===

/// Strip index wrappers (convert_layout, ext/trunc) to expose the underlying
/// make_range or constant for subslice pattern matching.
static Value stripSubsliceIndexWrappers(Value v) {
  while (true) {
    if (auto cvt = v.getDefiningOp<triton::gpu::ConvertLayoutOp>()) {
      v = cvt.getSrc();
      continue;
    }
    if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
      v = ext.getIn();
      continue;
    }
    if (auto ext = v.getDefiningOp<arith::ExtUIOp>()) {
      v = ext.getIn();
      continue;
    }
    if (auto tr = v.getDefiningOp<arith::TruncIOp>()) {
      v = tr.getIn();
      continue;
    }
    break;
  }
  return v;
}

/// Try to match `value` as make_range(start, start+extent), optionally
/// offset by a constant addend. Sets `offset` to the effective start.
static bool matchRangeWithOffset(Value value, int64_t extent, int64_t &offset) {
  Value current = stripSubsliceIndexWrappers(value);
  if (auto range = current.getDefiningOp<triton::MakeRangeOp>()) {
    offset = range.getStart();
    return range.getEnd() - range.getStart() == extent;
  }

  auto add = current.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto tryMatch = [&](Value lhs, Value rhs) -> bool {
    Value lhsStripped = stripSubsliceIndexWrappers(lhs);
    // Also strip expand_dims to handle addi(expand_dims(range), const),
    // e.g. `1 + tl.arange(0, N)[None, :]` -> addi(expand_dims(range), splat(1))
    while (auto expand = lhsStripped.getDefiningOp<triton::ExpandDimsOp>())
      lhsStripped = stripSubsliceIndexWrappers(expand.getSrc());
    auto range = lhsStripped.getDefiningOp<triton::MakeRangeOp>();
    if (!range)
      return false;
    Value rhsStripped = stripSubsliceIndexWrappers(rhs);
    if (auto cst = rhsStripped.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
        if (dense.isSplat()) {
          offset =
              range.getStart() + dense.getSplatValue<APInt>().getSExtValue();
          return range.getEnd() - range.getStart() == extent;
        }
      }
    }
    if (auto cstInt = rhsStripped.getDefiningOp<arith::ConstantIntOp>()) {
      offset = range.getStart() + cstInt.value();
      return range.getEnd() - range.getStart() == extent;
    }
    return false;
  };

  return tryMatch(add.getLhs(), add.getRhs()) ||
         tryMatch(add.getRhs(), add.getLhs());
}

/// Match one index of a multi-dim tle.local_pointers against the
/// "full range along a single axis" pattern the TLE frontend emits.
///
/// The index tensor has the full result shape, but its data only varies
/// along `axis`:
///
///   // result shape MxN.
///   // axis-0 index (varies along dim 0):
///   %r    = tt.make_range(offset0, offset0+M)          : tensor<Mxi32>
///   %re   = tt.expand_dims(%r) {axis = 1}              : tensor<Mx1xi32>
///   %idx0 = tt.broadcast(%re)                          : tensor<MxNxi32>

///   // axis-1 index (varies along dim 1):
///   %c    = tt.make_range(offset1, offset1+N)          : tensor<Nxi32>
///   %ce   = tt.expand_dims(%c) {axis = 0}              : tensor<1xNxi32>
///   %idx1 = tt.broadcast(%ce)                          : tensor<MxNxi32>
///   %ptr  = tle.local_pointers %md[%idx0, %idx1] : tensor<MxNx!tt.ptr>
///
/// So the index must be defined by a BroadcastOp: any other producer
/// (e.g. an addi of two broadcasts forming a flat row-major iota, which
/// Case D handles separately) is not a per-axis range and must be
/// rejected here, or Cases B/C would misread it and silently drop the
/// real offsets. The broadcast source must be size 1 in every dim except
/// `axis` (only then does the index vary along exactly one axis); we then
/// peel the expand_dims chain and match the underlying make_range (or
/// addi(range, const) for a non-zero offset). On mismatch the caller
/// falls through to Path B, keeping the original offset-carrying indices.
static bool matchFullIndexForAxis(Value index, size_t axis,
                                  ArrayRef<int64_t> shape, int64_t &offset) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger())
    return false;
  if (indexTy.getShape() != shape)
    return false;

  Value current = stripSubsliceIndexWrappers(index);
  if (shape.size() == 1)
    return matchRangeWithOffset(current, shape.front(), offset);

  auto bcast = current.getDefiningOp<triton::BroadcastOp>();
  if (!bcast)
    return false;

  auto bcastSrcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
  if (!bcastSrcTy || bcastSrcTy.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    const int64_t expected = dim == axis ? dimSize : 1;
    if (bcastSrcTy.getShape()[dim] != expected)
      return false;
  }

  current = stripSubsliceIndexWrappers(bcast.getSrc());
  while (auto expand = current.getDefiningOp<triton::ExpandDimsOp>())
    current = stripSubsliceIndexWrappers(expand.getSrc());

  auto rangeTy = dyn_cast<RankedTensorType>(current.getType());
  if (!rangeTy)
    return false;
  // For 1D: check shape directly. For multi-dim (e.g. addi(expand_dims,
  // const)): matchRangeWithOffset handles it by stripping expand_dims from addi
  // operands.
  if (rangeTy.getRank() == 1) {
    if (rangeTy.getShape()[0] != shape[axis])
      return false;
  }

  return matchRangeWithOffset(current, shape[axis], offset);
}

/// Check if a mask value is provably all-true (trivially true).
/// Matches: arith.constant dense<true>, tt.splat(arith.constant true).
static bool isTriviallyTrueMask(Value mask) {
  if (!mask)
    return true;

  if (auto constOp = mask.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat() && denseAttr.getElementType().isInteger(1) &&
          denseAttr.getSplatValue<bool>())
        return true;
    }
  }

  if (auto splatOp = mask.getDefiningOp<triton::SplatOp>()) {
    if (auto constSrc =
            splatOp.getSrc().getDefiningOp<arith::ConstantIntOp>()) {
      return constSrc.value() != 0;
    }
    if (auto constSrc = splatOp.getSrc().getDefiningOp<arith::ConstantOp>()) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constSrc.getValue()))
        return intAttr.getValue().getBoolValue();
    }
  }

  return false;
}

/// Whether any user of the tle.local_pointers result is a tt.load with a
/// non-trivially-true mask. Gates Cases B/C/D: such loads take Path B and
/// are later converted to a DTE (slice_pad/slice_deslice) transfer whose
/// source shape/size derivation assumes a ROOT buffer base.
///
/// Example (offset [4, 0] sub-tile of a 8x16 root, masked load):
///   %md   = ttg.local_alloc : !ttg.memdesc<8x16xf32>        // ROOT
///   %r    = tt.make_range(4, 8)         : tensor<4xi32>
///   %re   = tt.expand_dims(%r) {axis=1} : tensor<4x1xi32>
///   %idx0 = tt.broadcast(%re)           : tensor<4x8xi32>
///   %c    = tt.make_range(0, 8)         : tensor<8xi32>
///   %ce   = tt.expand_dims(%c) {axis=0} : tensor<1x8xi32>
///   %idx1 = tt.broadcast(%ce)           : tensor<4x8xi32>
///   %ptr  = tle.local_pointers %md[%idx0, %idx1] : tensor<4x8x!tt.ptr<f32>>
///   %val  = tt.load %ptr, %mask        // mask = col < 4 (non-trivial)
///
/// Without this gate Case B rewrites to %sub = ttg.memdesc_subslice
/// %md[4, 0] + zero-index local_pointers. The masked load then takes
/// Path B: memdesc_to_ptr folds the view offset into a mid-buffer base
/// (element [4, 0]) and the view size information is lost, so the DTE
/// conversion derives transfer sizes from the ROOT shape (8x16) and
/// reads rows 4..12 of an 8-row root -> out-of-bounds, device crash.
///
/// With this gate Cases B/C/D are skipped: the offset-carrying indices
/// keep Path B on the ROOT base, and the DTE lowering handles the offset
/// safely by placing it in the DTE source offsets ([4, 0]) instead of the
/// base pointer (rows 4..8 stay in bounds).
///
/// Unmasked (or trivially-true-mask) loads do not need this gate: they
/// take Path A, a structured ttg.local_load on the sub-view memdesc whose
/// reduced shape (4x8) bounds the transfer, so Cases B/C/D are safe for
/// them. Only non-trivial masks force Path B, hence the filter above.
static bool hasNonTriviallyMaskedLoadUser(Operation *op) {
  Value ptrTensor = op->getResult(0);
  for (auto *user : ptrTensor.getUsers()) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      if (loadOp.getPtr() == ptrTensor && loadOp.getMask() &&
          !isTriviallyTrueMask(loadOp.getMask()))
        return true;
    }
  }
  return false;
}

// ===----------------------------------------------------------------------===
// Phase 0: Canonicalize tle.local_pointers - MemDesc view canonicalization
//
// Unified handling of memdesc view transformations on tle.local_pointers.
// Runs BEFORE barrier insertion so that barrier groups track the simplified
// tle.local_pointers and the produced memdesc_xxx ops are available to all
// downstream phases.
//
// Supported cases (applied in priority order, iteratively):
//
// Case A - memdesc_index (dim-0 scalar index peeling):
//   When bufferRank > resultRank and the first index is a splat/constant
//   scalar, extract it into ttg.memdesc_index to reduce the memdesc rank.
//
//   %ptr = tle.local_pointers %md[%splat_idx, %idx1]
//            : !ttg.memdesc<4x128xf16> -> tensor<128x!tt.ptr<f16>>
//   =>
//   %indexed = ttg.memdesc_index %md[%scalar]
//            : !ttg.memdesc<4x128xf16> -> !ttg.memdesc<128xf16>
//   %ptr = tle.local_pointers %indexed[%idx1]
//            : !ttg.memdesc<128xf16> -> tensor<128x!tt.ptr<f16>>
//
// Case B - memdesc_subslice (static sub-tile offset extraction):
//   When all indices match make_range(offset, offset+extent) with static
//   offsets and the result shape is a proper sub-tile of the memdesc,
//   extract the constant offsets into ttg.memdesc_subslice. The remaining
//   tle.local_pointers gets zero-based indices covering the sub-tile.
//
//   %ptr = tle.local_pointers %md[%range_off64, %range_off32]
//            : !ttg.memdesc<128x64xf16> -> tensor<64x32x!tt.ptr<f16>>
//   =>
//   %sub = ttg.memdesc_subslice %md[64, 32]
//            : !ttg.memdesc<128x64xf16> -> !ttg.memdesc<64x32xf16>
//   %ptr = tle.local_pointers %sub[%range_0_64, %range_0_32]
//            : !ttg.memdesc<64x32xf16> -> tensor<64x32x!tt.ptr<f16>>
//
// Case C - memdesc_trans (transposed access pattern detection):
//   When a 2D tle.local_pointers has its axis-0 and axis-1 indices swapped
//   relative to the standard row-major pattern, produce ttg.memdesc_trans.
//   (Currently limited to 2D rank-preserving cases.)
//
//   %ptr = tle.local_pointers %md[%idx_col, %idx_row]
//            : !ttg.memdesc<M x N x f16> -> tensor<N x M x !tt.ptr<f16>>
//   =>
//   %transposed = ttg.memdesc_trans %md {order = [1, 0]}
//            : !ttg.memdesc<M x N x f16> -> !ttg.memdesc<N x M x f16>
//   %ptr = tle.local_pointers %transposed[%idx_row, %idx_col]
//            : !ttg.memdesc<N x M x f16> -> tensor<N x M x !tt.ptr<f16>>
//
// Case D - memdesc_reshape (1D buffer with multi-dim access):
//   When a 1D memdesc is accessed with a multi-dim index tensor (e.g.,
//   memdesc<256xf16> + tensor<16x16xi32>), produce ttg.memdesc_reshape
//   so downstream passes can use structured DTE ops instead of flat
//   pointer arithmetic.
//
//   %ptr = tle.local_pointers %md[%flat_idx]
//            : !ttg.memdesc<256xf16> -> tensor<16x16x!tt.ptr<f16>>
//   =>
//   %reshaped = ttg.memdesc_reshape %md
//            : !ttg.memdesc<256xf16> -> !ttg.memdesc<16x16xf16>
//   %ptr = tle.local_pointers %reshaped[%idx0, %idx1]
//            : !ttg.memdesc<16x16xf16> -> tensor<16x16x!tt.ptr<f16>>
//
//   NOTE: Case D requires the flat index to be decomposable into per-axis
//   make_range patterns. This is a best-effort optimization; unmatched
//   patterns fall through to ConvertLocalPointersOp's Path B.
// ===----------------------------------------------------------------------===

/// Case A: Peel dim-0 scalar index into ttg.memdesc_index.
/// Returns the rewritten tle.local_pointers op (or nullptr).
static Operation *canonicalizeDim0Index(Operation *op, OpBuilder &builder) {
  Value memDescVal = op->getOperand(0);
  auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
  auto bufferShape = memDescTy.getShape();
  unsigned bufferRank = memDescTy.getRank();
  unsigned numIndices = op->getNumOperands() - 1;

  if (numIndices == 0 || numIndices != bufferRank)
    return nullptr;

  unsigned resultRank = 0;
  if (auto rtt = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
    resultRank = rtt.getRank();

  if (resultRank != bufferRank - 1)
    return nullptr;

  Value dim0Index = op->getOperand(1);
  Value scalarIndex;

  if (auto splatOp = dim0Index.getDefiningOp<triton::SplatOp>()) {
    scalarIndex = splatOp.getSrc();
  } else if (auto constOp = dim0Index.getDefiningOp<arith::ConstantOp>()) {
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
      if (denseAttr.isSplat()) {
        auto splatVal = denseAttr.getSplatValue<Attribute>();
        scalarIndex = builder.create<arith::ConstantOp>(
            op->getLoc(), builder.getI32Type(),
            builder.getI32IntegerAttr(cast<IntegerAttr>(splatVal).getInt()));
      }
    }
  }

  if (!scalarIndex)
    return nullptr;

  if (!scalarIndex.getType().isInteger(32))
    scalarIndex = builder.create<arith::TruncIOp>(
        op->getLoc(), builder.getI32Type(), scalarIndex);

  auto reducedShape = bufferShape.drop_front(1);

  Attribute reducedEncoding = memDescTy.getEncoding();
  if (auto encTrait =
          dyn_cast<triton::gpu::LayoutEncodingTrait>(reducedEncoding)) {
    unsigned encRank = encTrait.getRank();
    unsigned newShapeSize = reducedShape.size();
    if (!(encRank == newShapeSize || encRank == newShapeSize - 1)) {
      if (auto swizzled = dyn_cast<triton::gpu::SwizzledSharedEncodingAttr>(
              reducedEncoding)) {
        auto oldOrder = swizzled.getOrder();
        SmallVector<unsigned> newOrder;
        for (unsigned o : oldOrder) {
          if (o == 0)
            continue;
          newOrder.push_back(o - 1);
        }
        auto reducedCTA = triton_gcu::compat::getDefaultCGALayout(
            builder.getContext(), newOrder.size());
        reducedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
            builder.getContext(), swizzled.getVec(), swizzled.getPerPhase(),
            swizzled.getMaxPhase(), newOrder, reducedCTA);
      } else if (auto nvmmaShared =
                     dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
                         reducedEncoding)) {
        auto reducedCTA = triton_gcu::compat::getDefaultCGALayout(
            builder.getContext(), newShapeSize);
        reducedEncoding = triton::gpu::NVMMASharedEncodingAttr::get(
            builder.getContext(), nvmmaShared.getSwizzlingByteWidth(),
            nvmmaShared.getTransposed(), nvmmaShared.getElementBitWidth(),
            nvmmaShared.getFp4Padded(), reducedCTA);
      }
    }
  }

  auto reducedMemDescTy = triton::gpu::MemDescType::get(
      reducedShape, memDescTy.getElementType(), reducedEncoding,
      memDescTy.getMemorySpace(), memDescTy.getMutableMemory());

  auto indexOp = builder.create<triton::gpu::MemDescIndexOp>(
      op->getLoc(), reducedMemDescTy, memDescVal, scalarIndex);

  SmallVector<Value> newOperands;
  newOperands.push_back(indexOp.getResult());
  for (unsigned i = 2; i < op->getNumOperands(); ++i)
    newOperands.push_back(op->getOperand(i));

  OperationState state(op->getLoc(), "tle.local_pointers");
  state.addOperands(newOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
  op->erase();
  return newOp;
}

/// Case B: Extract static sub-tile offsets into ttg.memdesc_subslice.
/// Returns the rewritten tle.local_pointers op (or nullptr).
static Operation *canonicalizeSubslice(Operation *op, OpBuilder &builder) {
  Value memDescVal = op->getOperand(0);
  auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
  auto memDescShape = memDescTy.getShape();
  unsigned bufferRank = memDescTy.getRank();
  unsigned numIndices = op->getNumOperands() - 1;

  if (numIndices == 0 || numIndices != bufferRank)
    return nullptr;

  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTy || resultTy.getRank() != bufferRank)
    return nullptr;

  auto ptrElemTy = dyn_cast<triton::PointerType>(resultTy.getElementType());
  if (!ptrElemTy)
    return nullptr;

  auto resultShape = resultTy.getShape();

  // Every axis must match make_range(offset, offset+extent).
  SmallVector<int32_t> offsets(bufferRank, 0);
  bool hasNonZeroOffset = false;
  for (unsigned axis = 0; axis < bufferRank; ++axis) {
    Value index = op->getOperand(axis + 1);
    int64_t offset = 0;
    if (!matchFullIndexForAxis(index, axis, resultShape, offset))
      return nullptr;
    if (offset < 0 || offset + resultShape[axis] > memDescShape[axis])
      return nullptr;
    offsets[axis] = static_cast<int32_t>(offset);
    if (offset != 0)
      hasNonZeroOffset = true;
  }

  // Skip full-view (all offsets zero AND shape matches buffer exactly).
  // Sub-shapes (offsets non-zero or result shape != buffer shape) are valid
  // subslice targets — TTMemDescSubsliceOpLowering handles them via
  // memref::SubViewOp with StridedLayoutAttr to preserve source strides.
  bool isFullView = !hasNonZeroOffset && llvm::equal(resultShape, memDescShape);
  if (isFullView)
    return nullptr;

  // MemDescSubsliceOp verifier requires offset to be aligned to the
  // result size in each split dimension: offset & (resultSize - 1) == 0.
  // Skip canonicalization if any offset is unaligned (falls through to Path B).
  for (unsigned axis = 0; axis < bufferRank; ++axis) {
    if (resultShape[axis] > 0 && (offsets[axis] & (resultShape[axis] - 1)))
      return nullptr;
  }

  auto subTy = triton::gpu::MemDescType::get(
      resultShape, memDescTy.getElementType(), memDescTy.getEncoding(),
      memDescTy.getMemorySpace(), memDescTy.getMutableMemory(),
      memDescTy.getAllocShape());
  auto subslice = builder.create<triton::gpu::MemDescSubsliceOp>(
      op->getLoc(), subTy, memDescVal, offsets);

  // Rebuild tle.local_pointers with zero-based indices on the sub-tile.
  // The zero-based form (no indices) is the simplest; Step 2's
  // ConvertLocalPointersOp recognizes it.
  SmallVector<Value> newOperands;
  newOperands.push_back(subslice.getResult());

  OperationState state(op->getLoc(), "tle.local_pointers");
  state.addOperands(newOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
  op->erase();
  return newOp;
}

/// Case C: Detect transposed 2D access and produce ttg.memdesc_trans.
/// Returns the rewritten tle.local_pointers op (or nullptr).
///
/// Detects when a 2D tle.local_pointers accesses memdesc<MxN> but
/// the index pattern indicates column-major order (axis swap). Limited
/// to 2D rank-preserving cases.
static Operation *canonicalizeTranspose(Operation *op, OpBuilder &builder) {
  Value memDescVal = op->getOperand(0);
  auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
  unsigned bufferRank = memDescTy.getRank();
  unsigned numIndices = op->getNumOperands() - 1;

  if (bufferRank != 2 || numIndices != 2)
    return nullptr;

  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTy || resultTy.getRank() != 2)
    return nullptr;

  auto resultShape = resultTy.getShape();
  auto memDescShape = memDescTy.getShape();

  // Transposed access: result[Nr, Mr] from a sub-region of memdesc[Ms, Ns],
  // i.e. the result dims may be smaller than the swapped buffer dims.
  if (resultShape[0] > memDescShape[1] || resultShape[1] > memDescShape[0])
    return nullptr;

  // Transposed access: index[0] varies along axis 1 of the result (covers
  // the buffer's dim 0), and index[1] varies along axis 0 of the result
  // (covers the buffer's dim 1). This is the opposite of the standard
  // row-major pattern where index[0] varies along axis 0 and index[1]
  // along axis 1.
  int64_t offset0 = 0, offset1 = 0;
  if (!matchFullIndexForAxis(op->getOperand(1), /*axis=*/1, resultShape,
                             offset0))
    return nullptr;
  if (!matchFullIndexForAxis(op->getOperand(2), /*axis=*/0, resultShape,
                             offset1))
    return nullptr;

  // Sub-region bounds: dim 0 covers [offset0, offset0 + resultShape[1]),
  // dim 1 covers [offset1, offset1 + resultShape[0]).
  int64_t subM = resultShape[1];
  int64_t subN = resultShape[0];
  if (offset0 < 0 || offset0 + subM > memDescShape[0])
    return nullptr;
  if (offset1 < 0 || offset1 + subN > memDescShape[1])
    return nullptr;

  // MemDescSubsliceOp verifier: a split dim's offset must be aligned to the
  // result size (offset & (size - 1) == 0); a non-split dim's offset must
  // be zero.
  if (subM != memDescShape[0] && (offset0 & (subM - 1)))
    return nullptr;
  if (subM == memDescShape[0] && offset0 != 0)
    return nullptr;
  if (subN != memDescShape[1] && (offset1 & (subN - 1)))
    return nullptr;
  if (subN == memDescShape[1] && offset1 != 0)
    return nullptr;

  // Emit subslice first (when the region is not the full buffer), then the
  // transpose on top of it: trans(sub(md))[r, c] = md[offset0 + c,
  // offset1 + r], which is exactly the transposed sub-region access.
  Value view = memDescVal;
  if (subM != memDescShape[0] || subN != memDescShape[1]) {
    SmallVector<int32_t> subOffsets = {static_cast<int32_t>(offset0),
                                       static_cast<int32_t>(offset1)};
    auto subTy = triton::gpu::MemDescType::get(
        SmallVector<int64_t>{subM, subN}, memDescTy.getElementType(),
        memDescTy.getEncoding(), memDescTy.getMemorySpace(),
        memDescTy.getMutableMemory(), memDescTy.getAllocShape());
    view = builder
               .create<triton::gpu::MemDescSubsliceOp>(op->getLoc(), subTy,
                                                       view, subOffsets)
               .getResult();
  }

  SmallVector<int32_t> order = {1, 0};
  auto transOp =
      builder.create<triton::gpu::MemDescTransOp>(op->getLoc(), view, order);

  // Rebuild with zero-index form on the transposed memdesc.
  SmallVector<Value> newOperands;
  newOperands.push_back(transOp.getResult());

  OperationState state(op->getLoc(), "tle.local_pointers");
  state.addOperands(newOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
  op->erase();
  return newOp;
}

/// Check if a contribution value matches axis d's row-major iota:
///   muli(broadcast(expand_dims(make_range(0, extent), ...)), stride_const)
/// or stride==1: broadcast(expand_dims(make_range(0, extent), ...))
/// where stride_const may be a tt.splat(arith.constant_int) or an
/// arith.constant with a dense splat elements attr.
///
/// The expand_dims chain must insert every axis except `axis` itself (so the
/// range data ends up along axis `axis` of the final shape). For rank=2,
/// axis 0 -> expand_dims axis=1 (shape [extent, 1]); axis 1 -> expand_dims
/// axis=0 (shape [1, extent]).
static bool matchAxisContribution(Value contrib, unsigned axis, unsigned rank,
                                  int64_t extent, int64_t stride) {
  // Strip outer broadcasts before looking for muli. The contribution may
  // be broadcast(muli(range, const)) from broadcast(addi(muli, ...)).
  Value current = contrib;
  while (auto bcast = current.getDefiningOp<triton::BroadcastOp>())
    current = bcast.getSrc();

  Value rangeSrc = current;
  int64_t mulStride = 1;

  // Strip an outer muli with a splat constant stride. This runs for unit
  // strides as well: the frontend may emit muli(range, 1) even when the
  // stride is 1, and the range must still be exposed for the
  // broadcast/expand_dims chain matching below.
  if (auto muli = current.getDefiningOp<arith::MulIOp>()) {
    auto trySplatRhs = [&](Value splatVal, Value otherSide) -> bool {
      // tt.splat(arith.constant_int)
      if (auto splat = splatVal.getDefiningOp<triton::SplatOp>()) {
        if (auto cst = splat.getSrc().getDefiningOp<arith::ConstantIntOp>()) {
          mulStride = cst.value();
          rangeSrc = otherSide;
          return true;
        }
        return false;
      }
      // arith.constant with dense splat elements attr
      if (auto cst = splatVal.getDefiningOp<arith::ConstantOp>()) {
        if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
          if (dense.isSplat()) {
            mulStride = dense.getSplatValue<APInt>().getSExtValue();
            rangeSrc = otherSide;
            return true;
          }
        }
      }
      return false;
    };
    if (!trySplatRhs(muli.getRhs(), muli.getLhs()) &&
        !trySplatRhs(muli.getLhs(), muli.getRhs()))
      return false;
  } else if (stride != 1) {
    return false;
  }

  if (mulStride != stride)
    return false;

  // Traverse the broadcast/expand_dims chain and collect the inserted axes.
  // The range data must end up along `axis`, so the set of expand_dims axes
  // must be exactly {0, 1, ..., rank-1} \ {axis}.
  Value src = rangeSrc;
  SmallVector<unsigned> insertedAxes;
  while (src) {
    if (auto bcast = src.getDefiningOp<triton::BroadcastOp>()) {
      src = bcast.getSrc();
      continue;
    }
    if (auto expand = src.getDefiningOp<triton::ExpandDimsOp>()) {
      insertedAxes.push_back(expand.getAxis());
      src = expand.getSrc();
      continue;
    }
    break;
  }

  if (insertedAxes.size() != rank - 1)
    return false;
  // Verify the inserted axes are exactly {0..rank-1} \ {axis}.
  SmallVector<unsigned> expectedAxes;
  for (unsigned i = 0; i < rank; ++i)
    if (i != axis)
      expectedAxes.push_back(i);
  // Order-insensitive comparison (expand_dims order may vary).
  SmallVector<unsigned> sortedInserted = insertedAxes;
  llvm::sort(sortedInserted);
  if (sortedInserted != expectedAxes)
    return false;

  if (auto makeRange = src.getDefiningOp<triton::MakeRangeOp>())
    return makeRange.getStart() == 0 &&
           makeRange.getEnd() == static_cast<uint32_t>(extent);
  return false;
}

/// Verify flatIdx is a row-major iota pattern:
///   flat_idx = sum_d(broadcast(expand_dims(arange(0, S_d), d)) * stride_d)
/// where stride_d = product(S_{d+1..rank-1}).
/// Simplified: does not handle trunci/extsi intermediate layers.
static bool isRowMajorIota(Value flatIdx, ArrayRef<int64_t> resultShape) {
  // Decompose addi chain to collect per-axis contributions.
  SmallVector<Value> contributions;
  Value current = flatIdx;
  while (auto addi = current.getDefiningOp<arith::AddIOp>()) {
    contributions.push_back(addi.getRhs());
    current = addi.getLhs();
  }
  contributions.push_back(current);

  unsigned rank = resultShape.size();
  for (unsigned d = 0; d < rank; ++d) {
    int64_t expectedExtent = resultShape[d];
    int64_t expectedStride = 1;
    for (unsigned i = d + 1; i < rank; ++i)
      expectedStride *= resultShape[i];

    bool found = false;
    for (Value contrib : contributions) {
      if (matchAxisContribution(contrib, d, rank, expectedExtent,
                                expectedStride)) {
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

/// Case D: Canonicalize 1D memdesc + higher-rank index into memdesc_reshape.
/// Returns the rewritten tle.local_pointers op (or nullptr).
///
/// Applies when: 1D buffer, single index operand, result rank > 1,
/// product(resultShape) == memdesc.shape[0], and the index tensor has
/// the same rank as the result.
///
/// Produces:
///   %reshaped = ttg.memdesc_reshape %md
///             : !ttg.memdesc<256xf16> -> !ttg.memdesc<16x16xf16>
///
/// Then rewrites the tle.local_pointers to use the reshaped memdesc
/// with zero indices (no explicit index operands). ConvertLocalPointersOp
/// Path B will synthesize full-view indices and use standard multi-dim
/// pointer arithmetic on the reshaped memdesc.
///
/// The original flat index (tensor<16x16xi32>) is dropped because it
/// is redundant for a full-view access on the reshaped memdesc. The
/// downstream `lowerViaMemDescToPtr` zero-index path generates identical
/// pointer arithmetic: base + row*N + col.
///
///   Before:
///     %ptr = tle.local_pointers %md[%flat_idx]
///              : !ttg.memdesc<256xf16> -> tensor<16x16x!tt.ptr<f16>>
///   After:
///     %reshaped = ttg.memdesc_reshape %md
///              : !ttg.memdesc<256xf16> -> !ttg.memdesc<16x16xf16>
///     %ptr = tle.local_pointers %reshaped
///              : !ttg.memdesc<16x16xf16> -> tensor<16x16x!tt.ptr<f16>>
///
/// Only applies when the flat index is a row-major iota pattern:
///   flat_idx = expand_dims(arange(0, M), 1) * N + expand_dims(arange(0, N), 0)
/// Non-iota flat indices (gather patterns) are rejected and fall through
/// to lowerViaMemDescToPtr's 1D-buffer special case.
static Operation *canonicalizeReshape(Operation *op, OpBuilder &builder) {
  Value memDescVal = op->getOperand(0);
  auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
  unsigned bufferRank = memDescTy.getRank();
  unsigned numIndices = op->getNumOperands() - 1;

  if (bufferRank != 1 || numIndices != 1)
    return nullptr;

  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTy || resultTy.getRank() <= 1)
    return nullptr;

  auto resultShape = resultTy.getShape();
  int64_t totalElems = 1;
  for (int64_t d : resultShape)
    totalElems *= d;

  if (totalElems != memDescTy.getShape()[0])
    return nullptr;

  Value flatIdx = op->getOperand(1);
  auto flatIdxTy = dyn_cast<RankedTensorType>(flatIdx.getType());
  if (!flatIdxTy || flatIdxTy.getRank() != resultTy.getRank())
    return nullptr;

  // Verify flat_idx is a row-major iota pattern. Non-iota flat indices
  // (gather patterns) are rejected - they must fall through to Path B's
  // 1D-buffer special case to preserve correct gather semantics.
  if (!isRowMajorIota(flatIdx, resultShape))
    return nullptr;

  // Produce ttg.memdesc_reshape: 1D -> multi-dim. Use the shape-inference
  // builder so the result encoding is correctly inferred from the source.
  auto reshapeOp = builder.create<triton::gpu::MemDescReshapeOp>(
      op->getLoc(), memDescVal, SmallVector<int64_t>(resultShape));

  // Rebuild tle.local_pointers with reshaped memdesc and no explicit
  // indices (zero-index form). ConvertLocalPointersOp's Path B will
  // synthesize full-view indices via synthesizeFullViewIndices() and
  // generate the equivalent pointer arithmetic on the reshaped shape.
  SmallVector<Value> newOperands;
  newOperands.push_back(reshapeOp.getResult());

  OperationState state(op->getLoc(), "tle.local_pointers");
  state.addOperands(newOperands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  Operation *newOp = builder.create(state);

  op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
  op->erase();
  return newOp;
}

/// Erase dead tle.local_pointers ops with no users.
static bool tryEraseDeadLocalPointers(Operation *op) {
  if (op->getResult(0).use_empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "canonicalizeLocalPointers: erasing dead op " << *op << "\n");
    op->erase();
    return true;
  }
  return false;
}

/// Phase 0 entry: canonicalize all tle.local_pointers in the module.
/// Applies Cases A-D with a worklist: each rewritten op is pushed back so
/// canonicalization can cascade (e.g. Case A -> Case B), and dead ops are
/// erased opportunistically. Each op appears in the worklist at most once
/// at a time, so the total work is linear in the number of rewrites
/// (instead of re-walking the whole module after every change).
static void canonicalizeLocalPointers(gpu::GPUModuleOp module) {
  SmallVector<Operation *> worklist;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tle.local_pointers" &&
        op->getNumResults() == 1)
      worklist.push_back(op);
  });

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    // DCE: erase dead ops first. Their users may include other
    // tle.local_pointers ops that are still in the worklist; those get
    // erased when popped.
    if (tryEraseDeadLocalPointers(op))
      continue;

    OpBuilder builder(op);

    // Try each canonicalization case in priority order. On success the
    // rewritten op goes back on the worklist so later cases can apply.
    // Case A is unconditional. Cases B/C/D are skipped when a user is a
    // non-trivially-masked tt.load: that load takes Path B, whose DTE
    // lowering derives the transfer sizes from the ROOT buffer shape and
    // would read out of bounds from a mid-buffer view base (see
    // hasNonTriviallyMaskedLoadUser). The original offset-carrying
    // indices keep Path B on the root base, which is safe.
    Operation *next = nullptr;
    bool viewCasesAllowed = !hasNonTriviallyMaskedLoadUser(op);
    if ((next = canonicalizeDim0Index(op, builder)) ||
        (viewCasesAllowed && (next = canonicalizeSubslice(op, builder))) ||
        (viewCasesAllowed && (next = canonicalizeTranspose(op, builder))) ||
        (viewCasesAllowed && (next = canonicalizeReshape(op, builder)))) {
      worklist.push_back(next);
    }
  }
}

// ===----------------------------------------------------------------------===
// Step 2: Lower all tle ops
// ===----------------------------------------------------------------------===

// ===----------------------------------------------------------------------===
// Step 2a: Lower tle.local_pointers
// ===----------------------------------------------------------------------===

/// Compute row-major strides from a static shape.
/// For shape [S0, S1, ..., S_{n-1}]:
///   stride[i] = S_{i+1} * S_{i+2} * ... * S_{n-1}
///   stride[n-1] = 1
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
    strides[i] = strides[i + 1] * shape[i + 1];
  return strides;
}

/// Compute the true memory strides (in root-buffer elements) of a MemDesc
/// view chain (ttg.memdesc_subslice / ttg.memdesc_trans) relative to the
/// view's first element.
///
/// The chain is walked down to a root buffer whose memory layout is
/// contiguous row-major:
///   - a plain ttg.local_alloc result (row-major by construction),
///   - a ttg.memdesc_index result (contiguous slice along dim 0),
///   - a ttg.memdesc_reshape result (Case D only fires on row-major iota
///     patterns over a contiguous buffer).
/// Subslice preserves the strides of its source; trans permutes them.
///
/// Path B (lowerViaMemDescToPtr) must use these strides instead of
/// row-major(viewShape): a 2D sub-tile of a [8,16] buffer has row stride 16,
/// not 8. The view's static offset is NOT folded here - the base pointer
/// produced by triton_gcu.memdesc_to_ptr already addresses the view's first
/// element (MemDescToPtrOpLowering folds the offset into the root pointer).
///
/// Returns failure when the chain root is not a MemDescType (caller should
/// fall back to row-major of the view shape).
static FailureOr<SmallVector<int64_t>> resolveViewStrides(Value view) {
  // Collect subslice/trans ops from the view down to the root.
  SmallVector<Operation *> chainOps;
  Value curr = view;
  while (auto *def = curr.getDefiningOp()) {
    if (isa<triton::gpu::MemDescSubsliceOp, triton::gpu::MemDescTransOp>(def)) {
      chainOps.push_back(def);
      curr = def->getOperand(0);
    } else {
      break;
    }
  }
  auto rootTy = dyn_cast<triton::gpu::MemDescType>(curr.getType());
  if (!rootTy)
    return failure();
  auto strides = computeRowMajorStrides(rootTy.getShape());

  // Fold the chain bottom-up. Subslice preserves strides; trans permutes.
  for (Operation *op : llvm::reverse(chainOps)) {
    if (auto trans = dyn_cast<triton::gpu::MemDescTransOp>(op)) {
      auto order = trans.getOrder();
      SmallVector<int64_t> permuted(strides.size());
      for (unsigned d = 0; d < strides.size(); ++d)
        permuted[d] = strides[order[d]];
      strides = std::move(permuted);
    }
    // MemDescSubsliceOp: strides unchanged.
  }
  return strides;
}

/// Strides for Path B pointer arithmetic: the true view strides when the
/// memdesc operand is a view chain, row-major of the shape otherwise.
static SmallVector<int64_t> getPointerArithStrides(Value memDescVal,
                                                   ArrayRef<int64_t> shape) {
  if (auto viewStrides = resolveViewStrides(memDescVal); succeeded(viewStrides))
    return std::move(*viewStrides);
  return computeRowMajorStrides(shape);
}

/// For zero-index full-view local_pointers, derive a per-axis encoding
/// compatible with tt.make_range + tt.expand_dims construction.
static Attribute getPerAxisIndexEncoding(Attribute resultEncoding,
                                         unsigned rank, unsigned axis,
                                         MLIRContext *ctx) {
  if (!resultEncoding)
    return {};
  auto distributed =
      dyn_cast<triton::gpu::DistributedEncodingTrait>(resultEncoding);
  if (!distributed)
    return {};
  for (int64_t dim = static_cast<int64_t>(rank) - 1; dim >= 0; --dim) {
    if (static_cast<unsigned>(dim) == axis)
      continue;
    distributed = triton::gpu::SliceEncodingAttr::get(
        ctx, dim, cast<triton::gpu::DistributedEncodingTrait>(distributed));
  }
  return distributed;
}

/// Synthesize explicit per-axis full-view index tensors for
/// `tle.local_pointers(%memdesc)` (zero-index form).
///
/// For each axis `i`, builds:
///   %range_i = tt.make_range [0, shape[i]) : tensor<shape[i]xi32, enc_i>
///   %idx_i   = tt.expand_dims %range_i ... -> tensor<1x...xshape[i]x...x1xi32>
static FailureOr<SmallVector<Value>>
synthesizeFullViewIndices(Operation *op, PatternRewriter &rewriter,
                          ArrayRef<int64_t> shape, Attribute resultEncoding) {
  SmallVector<Value> indices;
  auto loc = op->getLoc();
  auto i32Ty = rewriter.getI32Type();
  unsigned rank = shape.size();
  indices.reserve(rank);

  for (unsigned dim = 0; dim < rank; ++dim) {
    int64_t extent = shape[dim];
    if (extent < 0)
      return failure();

    Attribute indexEncoding = getPerAxisIndexEncoding(resultEncoding, rank, dim,
                                                      rewriter.getContext());
    auto rangeTy = RankedTensorType::get({extent}, i32Ty, indexEncoding);
    Value index = rewriter.create<triton::MakeRangeOp>(loc, rangeTy, 0, extent);

    for (unsigned axis = 0; axis < rank; ++axis) {
      if (axis == dim)
        continue;
      index = rewriter.create<triton::ExpandDimsOp>(loc, index, axis);
    }

    indices.push_back(index);
  }

  return indices;
}

// Check if all indices represent full-block access to the memdesc.
// Returns true for zero-index form (full-view by definition) or when
// each index[i] traces back to make_range(0, bufferShape[i]).
static bool isZeroOrFullRangeIndex(Operation *op) {
  auto memDescTy =
      dyn_cast<triton::gpu::MemDescType>(op->getOperand(0).getType());
  auto bufferShape = memDescTy.getShape();
  unsigned bufferRank = memDescTy.getRank();
  unsigned numIndices = op->getNumOperands() - 1;

  if (numIndices == 0)
    return true;

  if (numIndices != bufferRank)
    return false;

  // Check if an index value represents a full-block identity index for a
  // given dimension of shape `dimSize`. The pattern we look for is:
  //   tt.make_range(0, dimSize) [optionally + tt.expand_dims + tt.broadcast]
  // This means the index covers all elements [0, dimSize) in that dimension.
  auto isFullBlockIndex = [](Value idx, int64_t dimSize) {
    Value src = idx;
    // Strip through broadcast -> expand_dims chain
    while (src) {
      if (auto bcast = src.getDefiningOp<triton::BroadcastOp>()) {
        src = bcast.getSrc();
        continue;
      }
      if (auto expand = src.getDefiningOp<triton::ExpandDimsOp>()) {
        src = expand.getSrc();
        continue;
      }
      if (auto trunc = src.getDefiningOp<arith::TruncIOp>()) {
        src = trunc.getIn();
        continue;
      }

      break;
    }

    if (auto makeRange = src.getDefiningOp<triton::MakeRangeOp>()) {
      return makeRange.getStart() == 0 &&
             makeRange.getEnd() == static_cast<uint32_t>(dimSize);
    }
    return false;
  };

  for (unsigned dim = 0; dim < bufferRank; ++dim) {
    Value idx = op->getOperand(dim + 1);
    if (!isFullBlockIndex(idx, bufferShape[dim]))
      return false;
  }
  return true;
}

// Check if all users of a tle.local_pointers result are only load/store
// (unmasked or trivially-true mask) or remote_pointers. Non-trivial masks
// and other op types are rejected. Volatile / cache / eviction hints are
// intentionally IGNORED: the GCU backend does not model those modifiers on
// shared-memory accesses, so ttg.local_load / ttg.local_store (which have no
// such operands) are considered equivalent regardless of the hints.
static bool allUsersAreStructuredLoadStore(Operation *op) {
  auto ptrTensor = op->getResult(0);
  for (auto *user : ptrTensor.getUsers()) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      if (loadOp.getPtr() != ptrTensor)
        return false;
      if (loadOp.getMask() && !isTriviallyTrueMask(loadOp.getMask()))
        return false;
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      if (storeOp.getPtr() != ptrTensor)
        return false;
      if (storeOp.getMask() && !isTriviallyTrueMask(storeOp.getMask()))
        return false;
    } else if (user->getName().getStringRef() == "tle.remote_pointers") {
      if (user->getNumResults() != 1)
        return false;
      for (auto *remoteUser : user->getResult(0).getUsers()) {
        auto remoteLoadOp = dyn_cast<triton::LoadOp>(remoteUser);
        if (!remoteLoadOp || remoteLoadOp.getPtr() != user->getResult(0))
          return false;
        if (remoteLoadOp.getMask() &&
            !isTriviallyTrueMask(remoteLoadOp.getMask()))
          return false;
      }
      continue;
    } else {
      return false;
    }
  }
  return true;
}

// Path A: Structured lowering to ttg.local_load / ttg.local_store.
// Applies when: zero/full-range indices + resultShape matches buffer +
// all users are load/store (unmasked or trivially-true mask) or
// remote_pointers. Handles all post-canonicalize zero-index forms
// (including memdesc_index/subslice/trans/reshape results).
// Trivially-true masks are stripped (local_load/local_store have no mask).
static LogicalResult lowerToLocalLoadStore(Operation *op, Value memDescVal,
                                           PatternRewriter &rewriter) {
  SmallVector<Operation *> usersToRewrite;
  for (auto *user : op->getResult(0).getUsers())
    usersToRewrite.push_back(user);

  for (auto *user : usersToRewrite) {
    if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(storeOp);
      rewriter.create<triton::gpu::LocalStoreOp>(
          storeOp.getLoc(), storeOp.getValue(), memDescVal);
      rewriter.eraseOp(storeOp);
    } else if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(loadOp);
      auto resultTy = cast<RankedTensorType>(loadOp.getType());
      auto localLoad = rewriter.create<triton::gpu::LocalLoadOp>(
          loadOp.getLoc(), resultTy, memDescVal);
      rewriter.replaceOp(loadOp, localLoad.getResult());
    } else if (user->getName().getStringRef() == "tle.remote_pointers") {
      OpBuilder::InsertionGuard guard(rewriter);
      Value shardId = user->getOperand(1);

      // Collect tt.load users of tle.remote_pointers before mutation.
      SmallVector<Operation *> remoteLoads;
      for (auto *remoteUser : user->getResult(0).getUsers())
        remoteLoads.push_back(remoteUser);

      for (auto *remoteUser : remoteLoads) {
        auto remoteLoadOp = dyn_cast<triton::LoadOp>(remoteUser);
        if (!remoteLoadOp)
          continue;
        rewriter.setInsertionPoint(remoteLoadOp);
        auto remoteMemDesc = rewriter.create<triton::gcu::RemoteMemDescOp>(
            user->getLoc(), memDescVal.getType(), memDescVal, shardId);
        auto resultTy = cast<RankedTensorType>(remoteLoadOp.getType());
        auto localLoad = rewriter.create<triton::gpu::LocalLoadOp>(
            remoteLoadOp.getLoc(), resultTy, remoteMemDesc.getResult());
        rewriter.replaceOp(remoteLoadOp, localLoad.getResult());
      }
      rewriter.eraseOp(user);
    }
  }

  rewriter.eraseOp(op);
  return success();
}

static triton::gcu::MemDescToPtrOp createMemDescToPtr(PatternRewriter &rewriter,
                                                      Location loc,
                                                      Value memDescVal,
                                                      Type elemTy) {
  constexpr int kSharedMemAddrSpace = 3;
  auto ptrTy = triton::PointerType::get(elemTy, kSharedMemAddrSpace);
  return rewriter.create<triton::gcu::MemDescToPtrOp>(loc, ptrTy, memDescVal);
}

// Path B: Fallback lowering via memdesc_to_ptr + pointer arithmetic.
// Used when Path A conditions are not met: non-trivial masks, volatile,
// scalar dynamic indices, gather patterns, unaligned offsets, or
// non-load/store users. Downstream DTE pass handles masks natively
// on the generated pointer arithmetic.
static LogicalResult lowerViaMemDescToPtr(Operation *op, Value memDescVal,
                                          PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
  auto elemTy = memDescTy.getElementType();
  auto bufferShape = memDescTy.getShape();
  unsigned bufferRank = bufferShape.size();
  unsigned numIndices = op->getNumOperands() - 1;

  auto resultTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resultTy) {
    // Scalar result path: tle.local_pointers(%memdesc, idx...) -> !tt.ptr
    auto scalarPtrTy =
        dyn_cast<triton::PointerType>(op->getResult(0).getType());
    if (!scalarPtrTy)
      return rewriter.notifyMatchFailure(
          op, "result is neither RankedTensorType nor tt.ptr");

    if (numIndices != 0 && numIndices != bufferRank)
      return rewriter.notifyMatchFailure(
          op, "scalar: indices count does not match buffer rank");

    auto basePtr = createMemDescToPtr(rewriter, loc, memDescVal, elemTy);

    if (numIndices == 0) {
      rewriter.replaceOp(op, basePtr.getResult());
      return success();
    }

    // Use the true view strides when the memdesc is a view chain: the base
    // pointer already addresses the view's first element, and a sub-tile's
    // row stride is the root row stride, not row-major(viewShape).
    auto strides = getPointerArithStrides(memDescVal, bufferShape);
    auto i32Ty = rewriter.getI32Type();
    Value linearOffset;
    for (unsigned dim = 0; dim < bufferRank; ++dim) {
      Value idx = op->getOperand(dim + 1);
      if (idx.getType() != i32Ty)
        idx = rewriter.create<arith::TruncIOp>(loc, i32Ty, idx);

      Value contrib;
      int64_t strideVal = strides[dim];
      if (strideVal == 1) {
        contrib = idx;
      } else {
        auto strideCst = rewriter.create<arith::ConstantOp>(
            loc, rewriter.getI32IntegerAttr(static_cast<int32_t>(strideVal)));
        contrib = rewriter.create<arith::MulIOp>(loc, idx, strideCst);
      }
      linearOffset =
          linearOffset
              ? rewriter.create<arith::AddIOp>(loc, linearOffset, contrib)
              : contrib;
    }
    auto result = rewriter.create<triton::AddPtrOp>(loc, scalarPtrTy, basePtr,
                                                    linearOffset);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }

  auto resultShape = resultTy.getShape();
  unsigned resultRank = resultTy.getRank();
  auto encoding = resultTy.getEncoding();

  // Special case: single 1D buffer with a higher-rank index tensor used as
  // flat element offset.  e.g. memdesc<256xf16> + tensor<16x16xi32>
  //   -> tensor<16x16x!tt.ptr<f16,3>>
  // ptrAccum = splat(basePtr, resultShape) + index
  if (numIndices == bufferRank && bufferRank == 1 && resultRank > bufferRank) {
    Value idx = op->getOperand(1);
    auto idxTy = cast<RankedTensorType>(idx.getType());
    if (idxTy.getRank() != resultRank)
      return rewriter.notifyMatchFailure(
          op, "index tensor rank does not match result rank");

    auto ptrElemTy = dyn_cast<triton::PointerType>(resultTy.getElementType());
    if (!ptrElemTy)
      return rewriter.notifyMatchFailure(op,
                                         "result element type is not tt.ptr");

    auto basePtr = createMemDescToPtr(rewriter, loc, memDescVal, elemTy);

    auto i32Ty = rewriter.getI32Type();
    if (idxTy.getElementType() != i32Ty) {
      auto castTy = RankedTensorType::get(resultShape, i32Ty, encoding);
      idx = rewriter.create<arith::TruncIOp>(loc, castTy, idx);
    }

    auto splatPtrTy = RankedTensorType::get(resultShape, ptrElemTy, encoding);
    Value splatBase =
        rewriter.create<triton::SplatOp>(loc, splatPtrTy, basePtr);
    Value result =
        rewriter.create<triton::AddPtrOp>(loc, splatPtrTy, splatBase, idx);
    rewriter.replaceOp(op, result);
    return success();
  }

  if (numIndices != 0 && resultRank != bufferRank)
    return rewriter.notifyMatchFailure(
        op, "fancy indexing (result rank != buffer rank) is not supported; "
            "index tensors must have the same rank as the buffer");

  SmallVector<Value> indices;
  indices.reserve(bufferRank);
  if (numIndices == 0) {
    if (resultShape != bufferShape)
      return rewriter.notifyMatchFailure(
          op, "zero-index form requires result shape == buffer shape");
    auto synthesized =
        synthesizeFullViewIndices(op, rewriter, bufferShape, encoding);
    if (failed(synthesized))
      return rewriter.notifyMatchFailure(
          op, "failed to synthesize full-view indices");
    indices = std::move(*synthesized);
  } else {
    for (unsigned dim = 0; dim < bufferRank; ++dim)
      indices.push_back(op->getOperand(dim + 1));
  }

  auto ptrElemTy = dyn_cast<triton::PointerType>(resultTy.getElementType());
  if (!ptrElemTy)
    return rewriter.notifyMatchFailure(op, "result element type is not tt.ptr");

  // Use the true view strides when the memdesc is a view chain: the base
  // pointer already addresses the view's first element (the view offset is
  // folded by MemDescToPtrOpLowering), and a sub-tile's row stride is the
  // root row stride, not row-major(viewShape).
  auto strides = getPointerArithStrides(memDescVal, bufferShape);

  auto i32Ty = rewriter.getI32Type();

  // %base_ptr = triton_gcu.memdesc_to_ptr %memdesc
  auto basePtr = createMemDescToPtr(rewriter, loc, memDescVal, elemTy);

  if (bufferRank == 0) {
    auto scalarPtrTensorTy =
        RankedTensorType::get(resultShape, ptrElemTy, encoding);
    auto scalarPtr =
        rewriter.create<triton::SplatOp>(loc, scalarPtrTensorTy, basePtr);
    rewriter.replaceOp(op, scalarPtr);
    return success();
  }

  // Process dimensions from high (dim 0) to low (dim N-1).
  // At each step we maintain a running pointer tensor `ptrAccum`.
  Value ptrAccum;

  for (unsigned dim = 0; dim < bufferRank; ++dim) {
    Value idx = indices[dim];
    Value idxSrc = idx;
    auto idxSrcTy = cast<RankedTensorType>(idxSrc.getType());
    auto idxSrcShape = idxSrcTy.getShape();
    auto idxSrcEncoding = idxSrcTy.getEncoding();

    // Ensure index element type is i32.
    if (idxSrcTy.getElementType() != i32Ty) {
      auto castTy = RankedTensorType::get(idxSrcShape, i32Ty, idxSrcEncoding);
      idxSrc = rewriter.create<arith::TruncIOp>(loc, castTy, idxSrc);
      idxSrcTy = cast<RankedTensorType>(idxSrc.getType());
      idxSrcShape = idxSrcTy.getShape();
      idxSrcEncoding = idxSrcTy.getEncoding();
    }

    // Compute offset contribution: idx * stride (skip multiply if stride=1).
    Value offset;
    int64_t strideVal = strides[dim];
    if (strideVal == 1) {
      offset = idxSrc;
    } else {
      auto strideCstTy =
          RankedTensorType::get(idxSrcShape, i32Ty, idxSrcEncoding);
      auto strideCst = rewriter.create<arith::ConstantOp>(
          loc, DenseElementsAttr::get(strideCstTy,
                                      rewriter.getI32IntegerAttr(
                                          static_cast<int32_t>(strideVal))));
      offset = rewriter.create<arith::MulIOp>(loc, idxSrc, strideCst);
    }

    if (!ptrAccum) {
      auto splatPtrTy =
          RankedTensorType::get(idxSrcShape, ptrElemTy, idxSrcEncoding);
      ptrAccum = rewriter.create<triton::SplatOp>(loc, splatPtrTy, basePtr);
      ptrAccum =
          rewriter.create<triton::AddPtrOp>(loc, splatPtrTy, ptrAccum, offset);
    } else {
      auto ptrAccumTy = cast<RankedTensorType>(ptrAccum.getType());
      auto ptrAccumShape = ptrAccumTy.getShape();
      unsigned ptrAccumRank = ptrAccumShape.size();
      unsigned idxSrcRank = idxSrcShape.size();

      // Target shape: element-wise max over the ranks that exist.
      unsigned targetRank = std::max(ptrAccumRank, idxSrcRank);
      SmallVector<int64_t> targetShape(targetRank);
      for (unsigned d = 0; d < targetRank; ++d) {
        int64_t ptrDim = d < ptrAccumRank ? ptrAccumShape[d] : 1;
        int64_t idxDim = d < idxSrcRank ? idxSrcShape[d] : 1;
        targetShape[d] = std::max(ptrDim, idxDim);
      }

      auto targetPtrTy =
          RankedTensorType::get(targetShape, ptrElemTy, encoding);
      auto targetI32Ty = RankedTensorType::get(targetShape, i32Ty, encoding);

      if (ptrAccumShape != ArrayRef<int64_t>(targetShape))
        ptrAccum =
            rewriter.create<triton::BroadcastOp>(loc, targetPtrTy, ptrAccum);

      if (idxSrcShape != ArrayRef<int64_t>(targetShape))
        offset = rewriter.create<triton::BroadcastOp>(loc, targetI32Ty, offset);

      auto addPtrResultTy = cast<RankedTensorType>(ptrAccum.getType());
      ptrAccum = rewriter.create<triton::AddPtrOp>(loc, addPtrResultTy,
                                                   ptrAccum, offset);
    }
  }

  // Final broadcast to result shape if needed (should already match).
  auto ptrAccumShape = cast<RankedTensorType>(ptrAccum.getType()).getShape();
  if (ptrAccumShape != resultShape) {
    auto finalPtrTy = RankedTensorType::get(resultShape, ptrElemTy, encoding);
    ptrAccum = rewriter.create<triton::BroadcastOp>(loc, finalPtrTy, ptrAccum);
  }

  rewriter.replaceOp(op, ptrAccum);
  return success();
}

/// Rewrite `tle.local_pointers(%memdesc, %idx0, %idx1, ...)`
/// using the standard Triton pointer-arithmetic pattern:
///
/// For a 2D memdesc<S0 x S1 x elemTy>:
///
///   %base = triton_gcu.memdesc_to_ptr %memdesc -> !tt.ptr<elemTy, 3>
///
///   // dim 0 (highest): splat base to idx0's shape, addptr with idx0*stride0
///   %base_splat = tt.splat %base -> tensor<S0 x 1 x !tt.ptr<elemTy, 3>>
///   %stride0_cst = arith.constant dense<stride0> : tensor<S0 x 1 x i32>
///   %off0 = arith.muli %idx0_pre, %stride0_cst
///   %ptr0 = tt.addptr %base_splat, %off0   (shape: S0 x 1)
///
///   // broadcast ptr tensor to include dim 1
///   %ptr0_bc = tt.broadcast %ptr0 -> tensor<S0 x S1 x !tt.ptr<elemTy, 3>>
///
///   // dim 1 (lowest, stride=1): addptr with idx1
///   %idx1_bc = tt.broadcast %idx1_pre -> tensor<S0 x S1 x i32>
///   %result = tt.addptr %ptr0_bc, %idx1_bc  (shape: S0 x S1)
///
/// This mirrors the standard Triton IR pattern for global pointer arithmetic.
struct ConvertLocalPointersOp : public RewritePattern {
  explicit ConvertLocalPointersOp(MLIRContext *ctx)
      : RewritePattern("tle.local_pointers", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return failure();

    Value memDescVal = op->getOperand(0);
    auto memDescTy = dyn_cast<triton::gpu::MemDescType>(memDescVal.getType());
    if (!memDescTy)
      return rewriter.notifyMatchFailure(op, "operand 0 is not MemDescType");

    auto bufferShape = memDescTy.getShape();
    unsigned bufferRank = bufferShape.size();
    unsigned numIndices = op->getNumOperands() - 1;

    if (!isa<triton::PointerType>(
            getElementTypeOrSelf(op->getResult(0).getType())))
      return rewriter.notifyMatchFailure(op,
                                         "result element type is not tt.ptr");
    if (numIndices != 0 && numIndices != bufferRank)
      return rewriter.notifyMatchFailure(
          op, "indices count does not match buffer rank");

    if (op->getResult(0).use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }

    bool resultShapeMatchesBuffer = true;
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      resultShapeMatchesBuffer = resultTy.getShape() == bufferShape;
    }

    // Path A: Structured lowering to ttg.local_load / ttg.local_store.
    // Applies when: zero/full-range indices + resultShape matches buffer +
    // all users are load/store (unmasked or trivially-true mask) or
    // remote_pointers. Handles all post-canonicalize zero-index forms.
    if (isZeroOrFullRangeIndex(op) && allUsersAreStructuredLoadStore(op) &&
        resultShapeMatchesBuffer) {
      LLVM_DEBUG(llvm::dbgs() << "ConvertLocalPointersOp: Path A (structured) "
                              << "for " << *op << "\n");
      return lowerToLocalLoadStore(op, memDescVal, rewriter);
    }

    // Path B: Fallback via memdesc_to_ptr + pointer arithmetic.
    // Handles non-trivial masks, volatile, scalar dynamic indices,
    // gather patterns, unaligned offsets, non-load/store users.
    LLVM_DEBUG(llvm::dbgs() << "ConvertLocalPointersOp: Path B (fallback) "
                            << "for " << *op << "\n");
    return lowerViaMemDescToPtr(op, memDescVal, rewriter);
  }
};

// ===----------------------------------------------------------------------===
// Phase 2b: Lower tle.tma_copy
// ===----------------------------------------------------------------------===

// ===----------------------------------------------------------------------===
// TMA copy helpers (ported from RewriteTensorDescriptorToPointer.cpp)
// ===----------------------------------------------------------------------===

struct TmaDescriptor {
  Value base;
  SmallVector<Value> shape;
  SmallVector<Value> strides;
  Value paddingOption;
};

static TmaDescriptor unpackTmaDescriptor(triton::TensorDescType type,
                                         ValueRange pack) {
  int rank = type.getBlockType().getRank();
  TmaDescriptor res;
  res.base = pack[0];
  for (int i = 0; i < rank; i++)
    res.shape.push_back(pack[1 + i]);
  for (int i = 0; i < rank; i++)
    res.strides.push_back(pack[1 + rank + i]);
  res.paddingOption = pack[1 + 2 * rank];
  return res;
}

/// Build per-dimension offset+range tensor at full rank using the
/// SliceEncoding + MakeRange + ExpandDims (same as TritonLoadStoreToDma).
///
/// For a 2D block with dim=0, blockShape=[M,N]:
///   SliceEnc = Slice(dim=1, parent=blockedEnc) -> 1D encoding for rows
///   MakeRange(0, M) with SliceEnc -> tensor<M x i32, SliceEnc>
///   ExpandDims(dim=1) -> tensor<M x 1 x i32, blockedEnc>
///   ExtSI to i64, add scalar offset
///
/// Supports rank 1 and 2 (covers all practical TMA block shapes).
/// Returns a tensor of shape [1,...,blockShape[dim],...,1] with encoding.
static Value tmaGetOffsetRange(OpBuilder &builder, Location loc,
                               ArrayRef<int64_t> blockShape, Attribute encoding,
                               Value offset, unsigned dim) {
  int rank = blockShape.size();
  auto i32Ty = builder.getI32Type();
  auto i64Ty = builder.getI64Type();
  int64_t dimSize = blockShape[dim];
  auto blockedEnc = cast<triton::gpu::BlockedEncodingAttr>(encoding);

  Value rangeVal;
  if (rank == 1) {
    auto rangeTy = RankedTensorType::get({dimSize}, i32Ty, encoding);
    rangeVal = builder.create<triton::MakeRangeOp>(
        loc, rangeTy, 0, static_cast<int32_t>(dimSize));
  } else {
    // rank == 2: use SliceEncoding for the "other" dimension.
    unsigned otherDim = 1 - dim;
    auto sliceEnc = triton::gpu::SliceEncodingAttr::get(builder.getContext(),
                                                        otherDim, blockedEnc);
    auto range1DTy = RankedTensorType::get({dimSize}, i32Ty, sliceEnc);
    Value range1D = builder.create<triton::MakeRangeOp>(
        loc, range1DTy, 0, static_cast<int32_t>(dimSize));

    SmallVector<int64_t> expandedShape(rank, 1);
    expandedShape[dim] = dimSize;
    auto expandedI32Ty =
        RankedTensorType::get(expandedShape, i32Ty, blockedEnc);
    rangeVal = builder.create<triton::ExpandDimsOp>(loc, expandedI32Ty, range1D,
                                                    otherDim);
  }

  // Cast to i64 and add scalar offset.
  auto rangeShape = SmallVector<int64_t>(
      cast<RankedTensorType>(rangeVal.getType()).getShape());
  auto i64RangeTy = RankedTensorType::get(rangeShape, i64Ty, encoding);
  Value rangeI64 = builder.create<arith::ExtSIOp>(loc, i64RangeTy, rangeVal);
  Value splatOffset = builder.create<triton::SplatOp>(loc, i64RangeTy, offset);
  return builder.create<arith::AddIOp>(loc, splatOffset, rangeI64);
}

static Value tmaGeneratePtr(OpBuilder &builder, Location loc,
                            ArrayRef<int64_t> blockShape, Attribute encoding,
                            TmaDescriptor &desc, ValueRange offsets) {
  int rank = blockShape.size();
  auto i64TensorTy =
      RankedTensorType::get(blockShape, builder.getI64Type(), encoding);
  auto ptrType = cast<triton::PointerType>(desc.base.getType());
  auto ptrTensorTy = RankedTensorType::get(blockShape, ptrType, encoding);

  Value ptr = builder.create<triton::SplatOp>(loc, ptrTensorTy, desc.base);
  for (int i = 0; i < rank; ++i) {
    Value offsetRange =
        tmaGetOffsetRange(builder, loc, blockShape, encoding, offsets[i], i);

    // Compute offsetRange * stride and broadcast to full shape.
    SmallVector<int64_t> expandedShape(rank, 1);
    expandedShape[i] = blockShape[i];
    auto expandedI64Ty =
        RankedTensorType::get(expandedShape, builder.getI64Type(), encoding);

    Value splatStride =
        builder.create<triton::SplatOp>(loc, expandedI64Ty, desc.strides[i]);
    Value offsetWithStride =
        builder.create<arith::MulIOp>(loc, offsetRange, splatStride);
    Value broadcasted =
        builder.create<triton::BroadcastOp>(loc, i64TensorTy, offsetWithStride);
    ptr = builder.create<triton::AddPtrOp>(loc, ptrTensorTy, ptr, broadcasted);
  }
  return ptr;
}

static Value tmaGenerateMask(OpBuilder &builder, Location loc,
                             ArrayRef<int64_t> blockShape, Attribute encoding,
                             TmaDescriptor &desc, ValueRange offsets) {
  int rank = blockShape.size();
  auto maskTensorTy =
      RankedTensorType::get(blockShape, builder.getI1Type(), encoding);

  Value mask;
  for (int i = 0; i < rank; ++i) {
    Value offsetRange =
        tmaGetOffsetRange(builder, loc, blockShape, encoding, offsets[i], i);

    SmallVector<int64_t> expandedShape(rank, 1);
    expandedShape[i] = blockShape[i];
    auto expandedI64Ty =
        RankedTensorType::get(expandedShape, builder.getI64Type(), encoding);

    Value lowerBound =
        builder.create<arith::ConstantIntOp>(loc, builder.getI64Type(), 0);
    Value splatLB =
        builder.create<triton::SplatOp>(loc, expandedI64Ty, lowerBound);
    Value cmpLower = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, offsetRange, splatLB);

    Value splatUB =
        builder.create<triton::SplatOp>(loc, expandedI64Ty, desc.shape[i]);
    Value cmpUpper = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, offsetRange, splatUB);

    Value andResult = builder.create<arith::AndIOp>(loc, cmpLower, cmpUpper);
    Value broadcasted =
        builder.create<triton::BroadcastOp>(loc, maskTensorTy, andResult);

    mask =
        mask ? builder.create<arith::AndIOp>(loc, mask, broadcasted).getResult()
             : broadcasted;
  }
  return mask;
}

/// Generate the "other" value for masked loads. Uses SplatOp so that the
/// downstream TritonLoadStoreToDma pass can extract the scalar via
/// getScalarValue (which handles SplatOp but not DenseElementsAttr
/// with encoding or SelectOp).
static Value tmaGenerateOther(OpBuilder &builder, Location loc, Type elemType,
                              ArrayRef<int64_t> blockShape, Attribute encoding,
                              Value paddingOption) {
  auto blockTy = RankedTensorType::get(blockShape, elemType, encoding);
  Value scalar;
  if (paddingOption && isa<FloatType>(elemType)) {
    auto floatTy = cast<FloatType>(elemType);
    auto nanVal = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    auto nanScalar = builder.create<arith::ConstantOp>(
        loc, builder.getFloatAttr(floatTy, nanVal));
    auto zeroScalar =
        builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(floatTy));
    scalar = builder.create<arith::SelectOp>(loc, paddingOption, nanScalar,
                                             zeroScalar);
  } else {
    scalar =
        builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(elemType));
  }
  return builder.create<triton::SplatOp>(loc, blockTy, scalar);
}

/// Lower flagtree's ttg.tma_copy to GCU.
/// 1. global->shared : tt.load + ttg.local_store
/// 2. shared->global : ttg.local_load + tt.store
struct ConvertTMACopyOp : public RewritePattern {
  explicit ConvertTMACopyOp(MLIRContext *ctx)
      : RewritePattern("ttg.tma_copy", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumResults() != 0)
      return failure();

    auto loc = op->getLoc();
    Value src = op->getOperand(0);
    Value dst = op->getOperand(1);

    bool isGlobalToLocal = isa<triton::TensorDescType>(src.getType()) &&
                           isa<triton::gpu::MemDescType>(dst.getType());
    bool isLocalToGlobal = isa<triton::gpu::MemDescType>(src.getType()) &&
                           isa<triton::TensorDescType>(dst.getType());
    if (!isGlobalToLocal && !isLocalToGlobal)
      return failure();

    Value tensordescVal = isGlobalToLocal ? src : dst;
    Value memdescVal = isGlobalToLocal ? dst : src;
    auto tensorDescTy = cast<triton::TensorDescType>(tensordescVal.getType());
    auto blockTy = tensorDescTy.getBlockType();
    int rank = blockTy.getRank();
    auto blockShape = blockTy.getShape();

    // After add_rewrite_tensor_descriptor_to_pointer, the TensorDescType
    // operand is produced by an unrealized_conversion_cast from (ptr, shapes,
    // strides, padding).
    auto castOp = tensordescVal.getDefiningOp<UnrealizedConversionCastOp>();
    if (!castOp ||
        static_cast<int>(castOp.getOperands().size()) != 1 + rank + rank + 1)
      return failure();

    auto tmaDesc = unpackTmaDescriptor(tensorDescTy, castOp.getOperands());

    // Collect indices from operands 2..N and cast i32 -> i64.
    SmallVector<Value> offsets;
    auto i64Ty = rewriter.getI64Type();
    for (unsigned i = 2; i < op->getNumOperands(); i++) {
      Value idx = op->getOperand(i);
      if (idx.getType() != i64Ty)
        idx = rewriter.create<arith::ExtSIOp>(loc, i64Ty, idx);
      offsets.push_back(idx);
    }

    // Build encoding for the intermediate tensor type.
    int numWarps = triton::gpu::lookupNumWarps(op);
    SmallVector<unsigned> sizePerThread(rank, 1);
    SmallVector<unsigned> threadsPerWarp(rank, 1);
    SmallVector<unsigned> warpsPerCTA(rank, 1);
    warpsPerCTA[0] = static_cast<unsigned>(numWarps);
    SmallVector<unsigned> order(rank);
    for (int i = 0; i < rank; i++)
      order[i] = rank - 1 - i;
    auto ctaLayout = triton_gcu::compat::getCGALayoutFromSplitParams(
        op->getContext(),
        /*ctasPerCGA=*/SmallVector<unsigned>(rank, 1),
        /*ctaSplitNum=*/SmallVector<unsigned>(rank, 1),
        /*ctaOrder=*/order);
    auto blockedEnc = triton::gpu::BlockedEncodingAttr::get(
        op->getContext(), sizePerThread, threadsPerWarp, warpsPerCTA, order,
        ctaLayout);

    Value ptr =
        tmaGeneratePtr(rewriter, loc, blockShape, blockedEnc, tmaDesc, offsets);
    Value mask = tmaGenerateMask(rewriter, loc, blockShape, blockedEnc, tmaDesc,
                                 offsets);

    if (isGlobalToLocal) {
      Value other =
          tmaGenerateOther(rewriter, loc, blockTy.getElementType(), blockShape,
                           blockedEnc, tmaDesc.paddingOption);
      auto loadVal = rewriter.create<triton::LoadOp>(
          loc, ptr, mask, other, triton::CacheModifier::NONE,
          triton::EvictionPolicy::NORMAL, false);
      rewriter.create<triton::gpu::LocalStoreOp>(loc, loadVal, memdescVal);
    } else {
      auto tensorTy = RankedTensorType::get(
          blockShape, blockTy.getElementType(), blockedEnc);
      auto localLoadVal =
          rewriter.create<triton::gpu::LocalLoadOp>(loc, tensorTy, memdescVal);
      rewriter.create<triton::StoreOp>(loc, ptr, localLoadVal, mask,
                                       triton::CacheModifier::NONE,
                                       triton::EvictionPolicy::NORMAL);
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Phase 2c: Lower tle.extract_ptr -> tt.ptr_to_int + llvm.inttoptr
// Phase 3: Post-process tle.remote_pointers
//
// After Phase 2, tle.local_pointers has been lowered to either
//   Path A: ttg.local_store / ttg.local_load (memdesc stays as-is)
//   Path B: triton_gcu.memdesc_to_ptr + pointer arithmetic
//
// tle.remote_pointers survive Phase 2 because they are not matched by any
// rewrite pattern. This phase walks every remaining tle.remote_pointers:
//
//   %rp = tle.remote_pointers(%bufPtr, %shardId)
//
// We trace %bufPtr back through tt.splat / tt.addptr / tt.broadcast to find
// the triton_gcu.memdesc_to_ptr that produced the base pointer, and from
// that the source ttg.local_alloc memdesc.
//
// Case A (exclusive): if the memdesc_to_ptr result is ONLY consumed (through
//   the ptr chain) by tle.remote_pointers, we insert
//     triton_gcu.remote_memdesc(%memdesc, %shardId)
//   right after the local_alloc, replace the memdesc_to_ptr's source with
//   the remote memdesc, and replace all users of tle.remote_pointers with
//   the original buffer ptr chain. Then erase tle.remote_pointers.
//
// Case B (shared): if the memdesc_to_ptr result has non-remote-pointers users
//   too, we clone the ptr computation chain from memdesc_to_ptr forward, but
//   with a fresh memdesc_to_ptr sourced from a new remote_memdesc. Then
//   replace tle.remote_pointers users with the cloned chain's result and
//   erase tle.remote_pointers.
// ===----------------------------------------------------------------------===

/// Walk the def chain of a Value backwards through tt.splat, tt.addptr,
/// tt.broadcast to find the triton_gcu.memdesc_to_ptr that produced the
/// base shared-memory pointer. Returns nullptr if not found.
static Operation *traceToMemDescToPtr(Value v) {
  constexpr int kMaxDepth = 32;
  Value cur = v;
  for (int i = 0; i < kMaxDepth; ++i) {
    if (!cur || !cur.getDefiningOp())
      return nullptr;
    Operation *def = cur.getDefiningOp();
    if (isa<triton::gcu::MemDescToPtrOp>(def))
      return def;
    if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      cur = splat.getSrc();
      continue;
    }
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      cur = addptr.getPtr();
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      cur = bcast.getSrc();
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

/// Collect the chain of ops from memdesc_to_ptr (exclusive) down to
/// (and including) the Value `target`, in def-chain order (memdesc_to_ptr
/// result -> ... -> target). Only follows tt.splat / tt.addptr / tt.broadcast.
/// Returns true on success.
static bool collectDefChain(Value target, Operation *memdescToPtrOp,
                            SmallVectorImpl<Operation *> &chain) {
  SmallVector<Operation *> reverseChain;
  Value cur = target;
  constexpr int kMaxDepth = 32;
  for (int i = 0; i < kMaxDepth; ++i) {
    if (!cur || !cur.getDefiningOp())
      return false;
    Operation *def = cur.getDefiningOp();
    if (def == memdescToPtrOp)
      break;
    reverseChain.push_back(def);
    if (auto splat = dyn_cast<triton::SplatOp>(def))
      cur = splat.getSrc();
    else if (auto addptr = dyn_cast<triton::AddPtrOp>(def))
      cur = addptr.getPtr();
    else if (auto bcast = dyn_cast<triton::BroadcastOp>(def))
      cur = bcast.getSrc();
    else
      return false;
  }
  chain.clear();
  for (auto it = reverseChain.rbegin(); it != reverseChain.rend(); ++it)
    chain.push_back(*it);
  return true;
}

/// Clone the def chain (from memdesc_to_ptr forward) replacing the
/// memdesc_to_ptr source with `newBase` (a remote_memdesc result).
/// Returns the cloned value corresponding to `target`.
static Value cloneDefChainWithNewBase(
    OpBuilder &builder, Operation *memdescToPtrOp,
    const SmallVectorImpl<Operation *> &chain,
    Value newBase, Operation *insertBeforeOp) {
  auto loc = memdescToPtrOp->getLoc();

  // Clone the chain right before insertBeforeOp (typically the original
  // tle.remote_pointers op).  This ensures newBase dominates the cloned
  // memdesc_to_ptr, and all non-chain operands (strides, offsets) also
  // dominate because they dominated the original tle.remote_pointers.
  auto origM2P = cast<triton::gcu::MemDescToPtrOp>(memdescToPtrOp);
  builder.setInsertionPoint(insertBeforeOp);
  auto newM2P = builder.create<triton::gcu::MemDescToPtrOp>(
      loc, origM2P.getResult().getType(), newBase);

  llvm::DenseMap<Value, Value> valueMap;
  valueMap[origM2P.getResult()] = newM2P.getResult();

  for (Operation *origOp : chain) {
    Operation *cloned = builder.clone(*origOp);
    for (unsigned i = 0; i < cloned->getNumOperands(); ++i) {
      auto it = valueMap.find(cloned->getOperand(i));
      if (it != valueMap.end())
        cloned->setOperand(i, it->second);
    }
    if (origOp->getNumResults() > 0 && cloned->getNumResults() > 0)
      valueMap[origOp->getResult(0)] = cloned->getResult(0);
  }

  if (!chain.empty())
    return valueMap.lookup(chain.back()->getResult(0));
  return newM2P.getResult();
}

/// Rewrite AS 7 (remote) pointer types to AS 3 (shared) in a type.
/// For tt.ptr<T, 7> -> tt.ptr<T, 3>; for tensor<...x!tt.ptr<T, 7>> ->
/// tensor<...x!tt.ptr<T, 3>>. Returns the original type unchanged if no
/// AS 7 is present.
static Type rewriteRemoteAddrSpace(Type ty) {
  constexpr int kRemoteAS = 7;
  constexpr int kSharedAS = 3;

  if (auto ptrTy = dyn_cast<triton::PointerType>(ty)) {
    if (ptrTy.getAddressSpace() == kRemoteAS)
      return triton::PointerType::get(ptrTy.getPointeeType(), kSharedAS);
    return ty;
  }
  if (auto tensorTy = dyn_cast<RankedTensorType>(ty)) {
    Type newElem = rewriteRemoteAddrSpace(tensorTy.getElementType());
    if (newElem == tensorTy.getElementType())
      return ty;
    return RankedTensorType::get(tensorTy.getShape(), newElem,
                                 tensorTy.getEncoding());
  }
  return ty;
}

/// After replacing tle.remote_pointers result (AS 7) with a value of AS 3,
/// downstream ops still have AS 7 in their result types. Walk from
/// `startVal` through all transitive users and fix result types.
static void fixRemoteAddrSpaceTypes(Value startVal) {
  SmallVector<Value> worklist;
  llvm::DenseSet<Operation *> visited;
  worklist.push_back(startVal);

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (!visited.insert(user).second)
        continue;

      bool changed = false;
      for (unsigned i = 0; i < user->getNumResults(); ++i) {
        Type oldTy = user->getResult(i).getType();
        Type newTy = rewriteRemoteAddrSpace(oldTy);
        if (newTy != oldTy) {
          user->getResult(i).setType(newTy);
          changed = true;
          worklist.push_back(user->getResult(i));
        }
      }
      (void)changed;
    }
  }
}

/// Rewrite AS 7 (remote) -> AS 3 (shared) in tt.func parameter / result types
/// and tt.call operand / result types so that callee signatures stay consistent
/// after fixRemoteAddrSpaceTypes has rewritten all value types.
static void fixRemoteAddrSpaceFuncSignatures(gpu::GPUModuleOp module) {
  module.walk([](Operation *op) {
    if (auto funcOp = dyn_cast<triton::FuncOp>(op)) {
      auto funcTy = funcOp.getFunctionType();
      bool changed = false;

      SmallVector<Type> newInputs;
      for (Type t : funcTy.getInputs()) {
        Type nt = rewriteRemoteAddrSpace(t);
        newInputs.push_back(nt);
        if (nt != t) changed = true;
      }
      SmallVector<Type> newResults;
      for (Type t : funcTy.getResults()) {
        Type nt = rewriteRemoteAddrSpace(t);
        newResults.push_back(nt);
        if (nt != t) changed = true;
      }

      if (changed) {
        auto newFuncTy = FunctionType::get(op->getContext(), newInputs, newResults);
        funcOp.setFunctionType(newFuncTy);
        if (!funcOp.isDeclaration()) {
          Block &entry = funcOp.getBody().front();
          for (unsigned i = 0; i < entry.getNumArguments(); ++i) {
            if (entry.getArgument(i).getType() != newInputs[i]) {
              entry.getArgument(i).setType(newInputs[i]);
              fixRemoteAddrSpaceTypes(entry.getArgument(i));
            }
          }
        }
      }
    } else if (op->getName().getStringRef() == "tt.call") {
      for (unsigned i = 0; i < op->getNumResults(); ++i) {
        Type oldTy = op->getResult(i).getType();
        Type newTy = rewriteRemoteAddrSpace(oldTy);
        if (newTy != oldTy)
          op->getResult(i).setType(newTy);
      }
    }
  });
}

static void postProcessRemotePointers(gpu::GPUModuleOp module) {
  SmallVector<Operation *> remoteOps;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "tle.remote_pointers")
      remoteOps.push_back(op);
  });

  if (remoteOps.empty())
    return;

  OpBuilder builder(module.getContext());

  for (Operation *remoteOp : remoteOps) {
    Value bufferPtr = remoteOp->getOperand(0);
    Value shardId = remoteOp->getOperand(1);

    Operation *m2pOp = traceToMemDescToPtr(bufferPtr);
    if (!m2pOp) {
      LLVM_DEBUG(llvm::dbgs()
                 << "postProcessRemotePointers: cannot trace buffer ptr to "
                    "memdesc_to_ptr for "
                 << *remoteOp << "\n");
      continue;
    }

    auto m2p = cast<triton::gcu::MemDescToPtrOp>(m2pOp);
    Value localMemDesc = m2p.getSrc();

    SmallVector<Operation *> chain;
    bool chainOk = collectDefChain(bufferPtr, m2pOp, chain);
    if (!chainOk) {
      LLVM_DEBUG(llvm::dbgs()
                 << "postProcessRemotePointers: cannot collect def chain for "
                 << *remoteOp << "\n");
      continue;
    }
    // Both cases: place remote_memdesc + cloned chain right before remoteOp.
    // remoteOp's position is guaranteed to be dominated by all operands
    // (localMemDesc, shardId, and any non-chain values like strides).
    builder.setInsertionPoint(remoteOp);
    auto remoteMemDesc = builder.create<triton::gcu::RemoteMemDescOp>(
        remoteOp->getLoc(), localMemDesc.getType(), localMemDesc, shardId);
    Value replacement =
        cloneDefChainWithNewBase(builder, m2pOp, chain, remoteMemDesc,
                                 remoteOp);

    // Don't erase the original chain even if exclusive: other
    // tle.remote_pointers may still reference it.  Dead ops will be
    // cleaned up by later DCE / canonicalize passes.

    // replacement is AS 3, but remote_pointers result was AS 7.
    // Fix downstream ops that inherited AS 7 result types.
    remoteOp->getResult(0).replaceAllUsesWith(replacement);
    fixRemoteAddrSpaceTypes(replacement);
    remoteOp->erase();
  }

  // After all tle.remote_pointers have been lowered, AS 7 pointers no longer
  // exist as values.  However tt.func / tt.call signatures may still reference
  // AS 7 in their parameter and result types.  Rewrite those signatures so the
  // MLIR verifier does not flag an operand-type mismatch.
  fixRemoteAddrSpaceFuncSignatures(module);
}

// ===----------------------------------------------------------------------===
// Post-process: swap memdesc_index / remote_memdesc order
//
// After postProcessRemotePointers we may have:
//   %idx = ttg.memdesc_index %alloc[%i]       : memdesc<2xMxK> -> memdesc<MxK>
//   %rem = triton_gcu.remote_memdesc %idx, %id : memdesc<MxK>  -> memdesc<MxK>
// The downstream lowering for remote_memdesc expects its source to be the
// original local_alloc (full-rank memdesc), so we swap them:
//   %rem = triton_gcu.remote_memdesc %alloc, %id : memdesc<2xMxK>
//   %idx = ttg.memdesc_index %rem[%i]          : memdesc<2xMxK> -> memdesc<MxK>
//
// Case A: memdesc_index has no other users besides remote_memdesc -> swap in
// place.
// Case B: memdesc_index has other users -> clone memdesc_index for
// remote_memdesc.
// ===----------------------------------------------------------------------===
static void postProcessMemDescIndex(gpu::GPUModuleOp module) {
  SmallVector<triton::gcu::RemoteMemDescOp> opsToProcess;
  module.walk([&](triton::gcu::RemoteMemDescOp remoteOp) {
    auto indexOp = remoteOp.getSrc().getDefiningOp<triton::gpu::MemDescIndexOp>();
    if (!indexOp)
      return;
    opsToProcess.push_back(remoteOp);
  });

  for (auto remoteOp : opsToProcess) {
    auto indexOp = remoteOp.getSrc().getDefiningOp<triton::gpu::MemDescIndexOp>();
    Value allocMemDesc = indexOp.getSrc();
    Value sliceIdx = indexOp.getIndex();
    Value shardId = remoteOp.getShardId();

    bool exclusiveUse = indexOp.getResult().hasOneUse();

    // Place the new remote_memdesc + memdesc_index pair where all three
    // operands (allocMemDesc, shardId, sliceIdx) dominate.  Use the
    // original remoteOp position as the latest safe point.
    auto latestDef = [](std::initializer_list<Value> vals) -> Operation * {
      Operation *latest = nullptr;
      for (Value v : vals) {
        Operation *def = v.getDefiningOp();
        if (!def)
          continue;
        if (!latest) {
          latest = def;
          continue;
        }
        if (def->getBlock() != latest->getBlock()) {
          return nullptr;
        }
        if (latest->isBeforeInBlock(def))
          latest = def;
      }
      return latest;
    };

    OpBuilder builder(remoteOp.getContext());
    Operation *last = latestDef({allocMemDesc, shardId, sliceIdx});
    if (last && last->getBlock() == remoteOp->getBlock()) {
      builder.setInsertionPointAfter(last);
    } else {
      builder.setInsertionPoint(remoteOp);
    }

    auto newRemote = builder.create<triton::gcu::RemoteMemDescOp>(
        remoteOp.getLoc(), allocMemDesc.getType(), allocMemDesc, shardId);

    auto newIndex = builder.create<triton::gpu::MemDescIndexOp>(
        indexOp.getLoc(), indexOp.getType(), newRemote.getResult(), sliceIdx);

    remoteOp.getResult().replaceAllUsesWith(newIndex.getResult());
    remoteOp.erase();

    if (exclusiveUse) {
      indexOp.erase();
    }
  }
}


// ===----------------------------------------------------------------------===
// Phase 2c: Lower tle.extract_tile / tle.insert_tile
//
// tle.extract_ptr extracts a raw !llvm.ptr from a !tt.ptr<T>. We lower it
// here (before -convert-triton-to-gcu) because the partial conversion
// framework cannot correctly handle it there due to TTFuncOpLowering /
// convertRegionTypes interaction with the conversion value mapping.
// ===----------------------------------------------------------------------===
struct ConvertExtractPtrOp : public RewritePattern {
  explicit ConvertExtractPtrOp(MLIRContext *ctx)
      : RewritePattern("tle.extract_ptr", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return failure();
    auto loc = op->getLoc();
    Value ptr = op->getOperand(0);
    Value asInt =
        rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(), ptr);
    Value asPtr = rewriter.create<LLVM::IntToPtrOp>(
        loc, op->getResult(0).getType(), asInt);
    rewriter.replaceOp(op, asPtr);
    return success();
  }
};

struct TleToTritonGCUPass
    : public impl::TleToTritonGCUPassBase<TleToTritonGCUPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, triton::gcu::TritonGCUDialect,
                    mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    MLIRContext *ctx = &getContext();

    // Step 1: Canonicalize tle.local_pointers - extract memdesc view ops
    // (memdesc_index, memdesc_subslice, memdesc_trans, memdesc_reshape).
    // Runs before barrier insertion so barrier groups track simplified ops.
    canonicalizeLocalPointers(module);

    // Step 2: Insert barriers between store->load transitions on
    // tle.local_pointers sharing the same memdesc.
    insertLocalPointerBarriers(module);

    // Step 2b: Mark the global loads that fill a TLE pipe slot as async DTE,
    // so the DTE lowering gives the producer its own async tags instead of
    // serializing every copy on the shared sync tag.
    markPipeFillLoadsAsync(module);

    // Step 3: Lower tle.local_pointers via patterns.
    // Path A: structured -> ttg.local_load / ttg.local_store
    // Path B: fallback -> memdesc_to_ptr + pointer arithmetic
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertLocalPointersOp>(ctx);
    patterns.add<ConvertTMACopyOp>(ctx);
    patterns.add<ConvertExtractPtrOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();

    // post process tle.remote_pointers
    postProcessRemotePointers(module);
    // Swap memdesc_index / remote_memdesc so remote_memdesc sits on the
    // original local_alloc, making it lowerable.
    postProcessMemDescIndex(module);

    // Stamp cluster dimensions on the parent ModuleOp for downstream passes.
    if (auto mod = module->getParentOfType<ModuleOp>()) {
      auto i32Ty = IntegerType::get(ctx, 32);
      if (clusterDimZ != 1) {
        llvm::report_fatal_error("clusterDimZ must be 1 for gcu400/410");
      }
      if (clusterDimX * clusterDimY > 6) {
        llvm::report_fatal_error("only 6 block in a cluster for gcu400/410");
      }
      mod->setAttr("ttg.cluster-dims-x", IntegerAttr::get(i32Ty, clusterDimX));
      mod->setAttr("ttg.cluster-dims-y", IntegerAttr::get(i32Ty, clusterDimY));
      mod->setAttr("ttg.cluster-dims-z", IntegerAttr::get(i32Ty, clusterDimZ));
    }
  }
};

} // namespace
