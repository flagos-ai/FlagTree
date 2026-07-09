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
    } else if (user->getName().getStringRef() == "tle.remote_pointers") {
      if (user->getNumResults() != 1)
        return false;
      for (auto *remoteUser : user->getResult(0).getUsers()) {
        if (!isa<triton::LoadOp>(remoteUser))
          return false;
      }
      continue;
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
  unsigned resultRank = resultTy.getRank();
  auto encoding = resultTy.getEncoding();

  // Special case: single 1D buffer with a higher-rank index tensor used as
  // flat element offset.  e.g. memdesc<256xf16> + tensor<16x16xi32>
  //   → tensor<16x16x!tt.ptr<f16,3>>
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

    constexpr int kSharedMemAddrSpace = 3;
    auto basePtrTy = triton::PointerType::get(elemTy, kSharedMemAddrSpace);
    auto basePtr = rewriter.create<triton::gcu::MemDescToPtrOp>(loc, basePtrTy,
                                                                memDescVal);

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

    // Path A: Full-block access pattern (only for unmasked load/store users).
    // TODO(TBD): support more flexible patterns (e.g. ttg.memdesc_reshape,
    // ttg.memdesc_trans) that still cover the full block.
    bool resultShapeMatchesBuffer = true;
    if (auto resultTy =
            dyn_cast<RankedTensorType>(op->getResult(0).getType())) {
      resultShapeMatchesBuffer = resultTy.getShape() == bufferShape;
    }
    if (isFullBlockAccess(op) && allUsersAreUnmaskedLoadStore(op) &&
        resultShapeMatchesBuffer) {
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
/// result → ... → target). Only follows tt.splat / tt.addptr / tt.broadcast.
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
static Value cloneDefChainWithNewBase(OpBuilder &builder,
                                      Operation *memdescToPtrOp,
                                      const SmallVectorImpl<Operation *> &chain,
                                      Value newBase,
                                      Operation *insertBeforeOp) {
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
/// For tt.ptr<T, 7> → tt.ptr<T, 3>; for tensor<...x!tt.ptr<T, 7>> →
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
    Value replacement = cloneDefChainWithNewBase(builder, m2pOp, chain,
                                                 remoteMemDesc, remoteOp);

    // Don't erase the original chain even if exclusive: other
    // tle.remote_pointers may still reference it.  Dead ops will be
    // cleaned up by later DCE / canonicalize passes.

    // replacement is AS 3, but remote_pointers result was AS 7.
    // Fix downstream ops that inherited AS 7 result types.
    remoteOp->getResult(0).replaceAllUsesWith(replacement);
    fixRemoteAddrSpaceTypes(replacement);
    remoteOp->erase();
  }
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
//   %idx = ttg.memdesc_index %rem[%i]             : memdesc<2xMxK> ->
//   memdesc<MxK>
//
// Case A: memdesc_index has no other users besides remote_memdesc → swap in
// place. Case B: memdesc_index has other users → clone memdesc_index for
// remote_memdesc.
// ===----------------------------------------------------------------------===
static void postProcessMemDescIndex(gpu::GPUModuleOp module) {
  SmallVector<triton::gcu::RemoteMemDescOp> opsToProcess;
  module.walk([&](triton::gcu::RemoteMemDescOp remoteOp) {
    auto indexOp =
        remoteOp.getSrc().getDefiningOp<triton::gpu::MemDescIndexOp>();
    if (!indexOp)
      return;
    opsToProcess.push_back(remoteOp);
  });

  for (auto remoteOp : opsToProcess) {
    auto indexOp =
        remoteOp.getSrc().getDefiningOp<triton::gpu::MemDescIndexOp>();
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
// Pre-process: peel dim-0 scalar index into ttg.memdesc_index
//
// When tle.local_pointers has bufferRank > resultRank AND the first index
// is a tt.splat of a scalar, we can split it into:
//   %slice = ttg.memdesc_index %memdesc[%scalar] : memdesc<NxMxK> ->
//   memdesc<MxK> %new   = tle.local_pointers(%slice, remaining_indices...)
// This runs BEFORE the pattern rewriter so that Phase 2 only sees
// tle.local_pointers where bufferRank == numIndices == resultRank.
// ===----------------------------------------------------------------------===
static void preProcessLocalPointers(gpu::GPUModuleOp module) {
  SmallVector<Operation *> opsToProcess;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() != "tle.local_pointers")
      return;
    if (op->getNumResults() != 1)
      return;
    Value memDescVal = op->getOperand(0);
    auto memDescTy = dyn_cast<triton::gpu::MemDescType>(memDescVal.getType());
    if (!memDescTy)
      return;
    unsigned bufferRank = memDescTy.getRank();
    unsigned numIndices = op->getNumOperands() - 1;
    if (numIndices == 0 || numIndices != bufferRank)
      return;

    // Determine result rank.
    unsigned resultRank = 0;
    if (auto rtt = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
      resultRank = rtt.getRank();

    if (resultRank == bufferRank) {
      return;
    } else if (resultRank == bufferRank - 1) {
      opsToProcess.push_back(op);
    } else {
      // resultRank > bufferRank (e.g. 1D buffer with 2D index tensor):
      // handled directly in ConvertLocalPointersOp, skip preProcess.
      return;
    }
  });

  for (Operation *op : opsToProcess) {
    Value memDescVal = op->getOperand(0);
    auto memDescTy = cast<triton::gpu::MemDescType>(memDescVal.getType());
    auto bufferShape = memDescTy.getShape();

    Value dim0Index = op->getOperand(1);
    Value scalarIndex;

    if (auto splatOp = dim0Index.getDefiningOp<triton::SplatOp>()) {
      scalarIndex = splatOp.getSrc();
    } else if (auto constOp = dim0Index.getDefiningOp<arith::ConstantOp>()) {
      if (auto denseAttr = dyn_cast<DenseElementsAttr>(constOp.getValue())) {
        if (denseAttr.isSplat()) {
          OpBuilder builder(op);
          auto splatVal = denseAttr.getSplatValue<Attribute>();
          scalarIndex = builder.create<arith::ConstantOp>(
              op->getLoc(), builder.getI32Type(),
              builder.getI32IntegerAttr(cast<IntegerAttr>(splatVal).getInt()));
        }
      }
    }

    if (!scalarIndex) {
      LLVM_DEBUG(llvm::dbgs()
                 << "preProcessLocalPointers: dim-0 index is not tt.splat "
                    "or constant splat, skipping "
                 << *op << "\n");
      continue;
    }

    OpBuilder builder(op);

    if (!scalarIndex.getType().isInteger(32))
      scalarIndex = builder.create<arith::TruncIOp>(
          op->getLoc(), builder.getI32Type(), scalarIndex);

    auto reducedShape = bufferShape.drop_front(1);

    // Encoding rank must be shape.size or shape.size - 1.
    // If the original encoding rank == bufferRank (same as original shape),
    // we need a reduced encoding for the new shape (rank - 1).
    // For swizzled_shared, reconstruct with order dropping the highest dim.
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
          auto reducedCTA = triton::gpu::CTAEncodingAttr::getDefault(
              builder.getContext(), newOrder.size());
          reducedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
              builder.getContext(), swizzled.getVec(), swizzled.getPerPhase(),
              swizzled.getMaxPhase(), newOrder, reducedCTA);
        } else if (auto nvmmaShared =
                       dyn_cast<triton::gpu::NVMMASharedEncodingAttr>(
                           reducedEncoding)) {
          auto reducedCTA = triton::gpu::CTAEncodingAttr::getDefault(
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

    // Build new tle.local_pointers with reduced memdesc + remaining indices.
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

    // Phase 0: Peel dim-0 scalar indices from tle.local_pointers with
    // bufferRank > resultRank into ttg.memdesc_index + reduced local_pointers.
    preProcessLocalPointers(module);

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
