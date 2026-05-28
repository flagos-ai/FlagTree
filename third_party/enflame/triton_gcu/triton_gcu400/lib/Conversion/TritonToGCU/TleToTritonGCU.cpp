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

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

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

    Value memdesc = op->getOperand(0);
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
  for (auto &[val, _] : maps.ptrToGroup) {
    (void)_;
    worklist.push_back(val);
  }

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

// ===----------------------------------------------------------------------===
// Phase 2: Lower all tle ops
// ===----------------------------------------------------------------------===

// ===----------------------------------------------------------------------===
// Phase 2a: Lower tle.local_pointers
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

/// Try to look through a tt.broadcast to find the pre-broadcast source.
static Value lookThroughBroadcast(Value v) {
  if (auto broadcastOp = v.getDefiningOp<triton::BroadcastOp>())
    return broadcastOp.getSrc();
  return v;
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
static bool isFullBlockAccess(Operation *op) {
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
      break;
    }

    // Strip truncation (i64->i32)
    if (auto trunc = src.getDefiningOp<arith::TruncIOp>())
      src = trunc.getIn();

    // Check for make_range(0, dimSize)
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

// Check if all users of a tle.local_pointers result are only unmasked
// tt.load and/or tt.store (no atomics, no masks, no other uses).
// ttg.local_store/ttg.local_load don't support masks, so masked accesses
// must go through Path B (memdesc_to_ptr + pointer arithmetic).
static bool allUsersAreUnmaskedLoadStore(Operation *op) {
  auto ptrTensor = op->getResult(0);
  for (auto *user : ptrTensor.getUsers()) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      if (loadOp.getPtr() != ptrTensor || loadOp.getMask())
        return false;
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      if (storeOp.getPtr() != ptrTensor || storeOp.getMask())
        return false;
    } else {
      return false;
    }
  }
  return true;
}

// Path A: Full-block access lowering.
// Replace tt.store(tle.local_pointers, val) -> ttg.local_store(val, memdesc)
// Replace tt.load (tle.local_pointers)      -> ttg.local_load (memdesc)
// Then erase tle.local_pointers.
static LogicalResult lowerFullBlockAccess(Operation *op, Value memDescVal,
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
    }
  }

  rewriter.eraseOp(op);
  return success();
}

// Path B: Sub-view / runtime-offset / atomic fallback lowering.
// Generate memdesc_to_ptr + pointer arithmetic.
// TODO(haizhu.shao): Implement proper subview lowering with
// memdesc_subview + local_load.
static LogicalResult lowerSubViewAccess(Operation *op, Value memDescVal,
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

    constexpr int kSharedMemAddrSpace = 3;
    auto basePtrTy = triton::PointerType::get(elemTy, kSharedMemAddrSpace);
    auto basePtr = rewriter.create<triton::gcu::MemDescToPtrOp>(loc, basePtrTy,
                                                                memDescVal);

    if (numIndices == 0) {
      rewriter.replaceOp(op, basePtr.getResult());
      return success();
    }

    auto strides = computeRowMajorStrides(bufferShape);
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
  auto encoding = resultTy.getEncoding();

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

  auto strides = computeRowMajorStrides(bufferShape);

  constexpr int kSharedMemAddrSpace = 3;
  auto basePtrTy = triton::PointerType::get(elemTy, kSharedMemAddrSpace);
  auto i32Ty = rewriter.getI32Type();

  // %base_ptr = triton_gcu.memdesc_to_ptr %memdesc
  auto basePtr =
      rewriter.create<triton::gcu::MemDescToPtrOp>(loc, basePtrTy, memDescVal);

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

    // Try to look through tt.broadcast to get the pre-broadcast source,
    // which typically has a "1" in non-owning dimensions.
    Value idxSrc = lookThroughBroadcast(idx);
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

      SmallVector<int64_t> targetShape(bufferRank);
      for (unsigned d = 0; d < bufferRank; ++d)
        targetShape[d] = std::max(ptrAccumShape[d], idxSrcShape[d]);

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
    if (numIndices != 0 && numIndices != bufferRank)
      return rewriter.notifyMatchFailure(
          op, "indices count does not match buffer rank");

    if (!isa<triton::PointerType>(
            getElementTypeOrSelf(op->getResult(0).getType())))
      return rewriter.notifyMatchFailure(op,
                                         "result element type is not tt.ptr");

    // Path A: Full-block access pattern (only for unmasked load/store users).
    if (isFullBlockAccess(op) && allUsersAreUnmaskedLoadStore(op)) {
      LLVM_DEBUG(llvm::dbgs() << "ConvertLocalPointersOp: Path A (full-block) "
                              << "for " << *op << "\n");
      return lowerFullBlockAccess(op, memDescVal, rewriter);
    }

    // Path B: Sub-view / runtime offset / atomic fallback.
    LLVM_DEBUG(llvm::dbgs() << "ConvertLocalPointersOp: Path B (sub-view) "
                            << "for " << *op << "\n");
    return lowerSubViewAccess(op, memDescVal, rewriter);
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
    auto ctaLayout = triton::gpu::CTAEncodingAttr::fromSplitParams(
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

    // Phase 1: Insert barriers between store→load transitions on
    // tle.local_pointers sharing the same memdesc.
    // This runs BEFORE lowering so we can track high-level ops directly.
    insertLocalPointerBarriers(module);

    // Phase 2: Lower TLE ops to TritonGCU ops.
    RewritePatternSet patterns(ctx);
    patterns.add<ConvertLocalPointersOp>(ctx);
    patterns.add<ConvertTMACopyOp>(ctx);
    patterns.add<ConvertExtractPtrOp>(ctx);

    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
