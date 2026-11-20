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
#include "Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

#include "Utils/TritonVersionCompat.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "triton-gcu-local-mem-optimize"

namespace mlir {
#define GEN_PASS_DEF_TRITONGCULOCALMEMOPTIMIZEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {

// ===----------------------------------------------------------------------===
// Helpers
// ===----------------------------------------------------------------------===

static bool isSharedMemoryMemDesc(Value memdesc) {
  auto memdescTy = dyn_cast<triton::gpu::MemDescType>(memdesc.getType());
  return memdescTy && isa_and_nonnull<triton::gpu::SharedMemorySpaceAttr>(
                          memdescTy.getMemorySpace());
}

/// Trace through tt.addptr / tt.broadcast / tt.splat chains back to the
/// triton_gcu.memdesc_to_ptr that produced the base pointer.
static triton::gcu::MemDescToPtrOp traceToMemDescToPtr(Value ptrTensor) {
  Value cur = ptrTensor;
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (cur && cur.getDefiningOp()) {
    Operation *def = cur.getDefiningOp();
    if (!visited.insert(def).second)
      return nullptr;
    if (auto mdToPtr = dyn_cast<triton::gcu::MemDescToPtrOp>(def))
      return mdToPtr;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def))
      cur = addptr.getPtr();
    else if (auto bcast = dyn_cast<triton::BroadcastOp>(def))
      cur = bcast.getSrc();
    else if (auto splat = dyn_cast<triton::SplatOp>(def))
      cur = splat.getSrc();
    else
      return nullptr;
  }
  return nullptr;
}

/// Trace a !triton_gcu.ptr scalar through the int2ptr -> ptr_to_int ->
/// memdesc_to_ptr chain to recover the original memdesc Value.
static Value traceGcuPtrToMemDesc(Value gcuPtr) {
  if (!gcuPtr || !gcuPtr.getDefiningOp())
    return nullptr;
  auto int2ptr = dyn_cast<triton::gcu::IntToPtrOp>(gcuPtr.getDefiningOp());
  if (!int2ptr)
    return nullptr;
  auto ptrToInt = int2ptr.getValue().getDefiningOp<triton::PtrToIntOp>();
  if (!ptrToInt)
    return nullptr;
  auto mdToPtr = ptrToInt.getSrc().getDefiningOp<triton::gcu::MemDescToPtrOp>();
  if (!mdToPtr)
    return nullptr;
  return mdToPtr.getSrc();
}

/// Check whether an addptr offset traces back to tt.make_range with start=0,
/// possibly through expand_dims, broadcast, arith.muli, and arith.extsi.
static bool isSequentialOffset(Value offset) {
  if (!offset || !offset.getDefiningOp())
    return false;
  Operation *def = offset.getDefiningOp();
  if (auto makeRange = dyn_cast<triton::MakeRangeOp>(def))
    return makeRange.getStart() == 0;
  if (auto expandDims = dyn_cast<triton::ExpandDimsOp>(def))
    return isSequentialOffset(expandDims.getSrc());
  if (auto bcast = dyn_cast<triton::BroadcastOp>(def))
    return isSequentialOffset(bcast.getSrc());
  if (auto muli = dyn_cast<arith::MulIOp>(def))
    return isSequentialOffset(muli.getLhs()) ||
           isSequentialOffset(muli.getRhs());
  if (auto extsi = dyn_cast<arith::ExtSIOp>(def))
    return isSequentialOffset(extsi.getIn());
  return false;
}

/// Verify that the addptr chain from memdesc_to_ptr to the final ptr tensor
/// uses only sequential (make_range-based) offsets.
static bool hasSequentialIndexChain(Value ptrTensor,
                                    triton::gcu::MemDescToPtrOp mdToPtr) {
  Value cur = ptrTensor;
  llvm::SmallPtrSet<Operation *, 16> visited;
  while (cur && cur.getDefiningOp()) {
    Operation *def = cur.getDefiningOp();
    if (!visited.insert(def).second)
      return false;
    if (def == mdToPtr.getOperation())
      return true;
    if (auto addptr = dyn_cast<triton::AddPtrOp>(def)) {
      if (!isSequentialOffset(addptr.getOffset()))
        return false;
      cur = addptr.getPtr();
    } else if (auto bcast = dyn_cast<triton::BroadcastOp>(def)) {
      cur = bcast.getSrc();
    } else if (auto splat = dyn_cast<triton::SplatOp>(def)) {
      cur = splat.getSrc();
    } else {
      return false;
    }
  }
  return false;
}

/// Check whether the shared-memory pointer tensor covers the full memdesc
/// with sequential (identity) indexing.
static bool isFullBlockSmemAccess(Value smemPtrTensor,
                                  triton::gcu::MemDescToPtrOp mdToPtr) {
  auto memdescTy = cast<triton::gpu::MemDescType>(mdToPtr.getSrc().getType());
  auto ptrTensorTy = dyn_cast<RankedTensorType>(smemPtrTensor.getType());
  if (!ptrTensorTy)
    return false;
  if (memdescTy.getShape() != ptrTensorTy.getShape())
    return false;
  return hasSequentialIndexChain(smemPtrTensor, mdToPtr);
}

/// Check whether a triton_gcu.store writes the full tile to a memdesc-backed
/// smem pointer with row-major strides matching the memdesc shape.
static bool isFullTileGcuStoreToSmem(triton::gcu::StoreOp storeOp,
                                     Value memdesc) {
  auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
  auto valTy = dyn_cast<RankedTensorType>(storeOp.getValue().getType());
  if (!valTy)
    return false;
  if (memdescTy.getShape() != valTy.getShape())
    return false;

  auto storeShape = storeOp.getShape();
  auto storeStrides = storeOp.getStrides();
  auto mdShape = memdescTy.getShape();

  if (static_cast<int64_t>(storeShape.size()) != memdescTy.getRank())
    return false;

  for (unsigned i = 0; i < storeShape.size(); ++i) {
    auto shapeConst = storeShape[i].getDefiningOp<arith::ConstantIndexOp>();
    if (!shapeConst)
      return false;
    if (shapeConst.value() != mdShape[i])
      return false;
  }

  if (memdescTy.getRank() == 2 && storeStrides.size() == 2) {
    auto stride0 = storeStrides[0].getDefiningOp<arith::ConstantIndexOp>();
    auto stride1 = storeStrides[1].getDefiningOp<arith::ConstantIndexOp>();
    if (!stride0 || !stride1)
      return false;
    if (stride0.value() != mdShape[1] || stride1.value() != 1)
      return false;
  } else if (memdescTy.getRank() == 1 && storeStrides.size() == 1) {
    auto stride0 = storeStrides[0].getDefiningOp<arith::ConstantIndexOp>();
    if (!stride0 || stride0.value() != 1)
      return false;
  }

  return true;
}

/// Compute an adjusted base pointer by adding a byte offset derived from
/// tile element offsets within a row-major tensor.
///
/// GCU's DTE lowering ignores the `offsets` parameter of triton_gcu.load/store,
/// always starting from the base pointer. Therefore, to load/store a tile at a
/// non-zero position, we bake the byte offset into the pointer itself:
///
///   adjusted_ptr = int2ptr(ptr2int(base_ptr) + byte_offset)
///   byte_offset = linearize(tileElemOffsets, srcShape) * elemBytes
///
/// Example: srcShape=[512, 512], tileElemOffsets=[128, 128], elemBytes=4
///   elemOffset = 128*512 + 128 = 65664
///   byteOffset = 65664 * 4 = 262656
///   adjusted_ptr = base_ptr + 262656
///
static Value computeAdjustedPtr(PatternRewriter &rewriter, Location loc,
                                Value basePtr,
                                ArrayRef<int64_t> tileElemOffsets,
                                ArrayRef<int64_t> srcShape, unsigned elemBytes,
                                unsigned rank) {
  int64_t elemOffset = 0;
  if (rank == 2)
    elemOffset = tileElemOffsets[0] * srcShape[1] + tileElemOffsets[1];
  else if (rank == 1)
    elemOffset = tileElemOffsets[0];
  int64_t byteOffset = elemOffset * static_cast<int64_t>(elemBytes);

  if (byteOffset == 0)
    return basePtr;

  auto i64Ty = rewriter.getI64Type();
  auto ptrTy = basePtr.getType();
  Value ptrInt = rewriter.create<triton::gcu::PtrToIntOp>(loc, i64Ty, basePtr);
  Value offsetVal = rewriter.create<arith::ConstantIntOp>(loc, byteOffset, 64);
  Value adjusted = rewriter.create<arith::AddIOp>(loc, ptrInt, offsetVal);
  return rewriter.create<triton::gcu::IntToPtrOp>(loc, ptrTy, adjusted);
}

/// What a view-folding pattern did to the source dims of a copy. It decides how
/// `broadcast_dims` has to be translated when the copy is re-created.
enum class BroadcastMapping {
  /// Same dims in the same order: the mapping is kept as it is.
  Verbatim,
  /// The copy is transposed: permute by `transOrder`
  /// (new[i] = old[transOrder[i]]).
  Transposed,
  /// Dim 0 was folded into the view or the pointer: drop it from the mapping.
  DropDim0,
};

/// Give a re-created `copy_global_to_local` the attributes that its operands do
/// not carry, all of which a rebuild would otherwise lose:
///   * `broadcast_dims`  - how the source dims map onto the destination tile;
///   * `offsets_are_dst` - the offsets are destination tile offsets, not source
///                         offsets (see `triton_gcu.load`);
///   * `tt.load.async`   - issue the transfer without waiting for it.
///
/// `src` is the op the pattern replaces: the fused `triton_gcu.load`, or the
/// copy itself when only its view changed. An attribute is set only when `src`
/// has it, except for `tt.load.async` when the caller passes `asyncByDefault`:
/// a plain `tle.gpu.copy` has no frontend switch for it, so the pattern that
/// turns it into a GMEM->SMEM transfer asks for the asynchronous form itself.
/// TTSmemLowerToGCU degrades that request to a synchronous issue when no shared
/// tag is free, so a kernel that cannot afford one still compiles.
static void
inheritCopyAttributes(Operation *src, triton::gcu::CopyGlobalToLocalOp dst,
                      BroadcastMapping mapping = BroadcastMapping::Verbatim,
                      ArrayRef<int> transOrder = {},
                      bool asyncByDefault = false) {
  auto copyInherentAttrs = [&](auto srcOp) {
    if (auto attr = srcOp.getBroadcastDimsAttr()) {
      SmallVector<int32_t> mapped(attr.asArrayRef());
      switch (mapping) {
      case BroadcastMapping::Verbatim:
        break;
      case BroadcastMapping::Transposed:
        for (unsigned i = 0; i < mapped.size() && i < transOrder.size(); ++i)
          mapped[i] = attr.asArrayRef()[transOrder[i]];
        break;
      case BroadcastMapping::DropDim0:
        if (!mapped.empty())
          mapped.erase(mapped.begin());
        break;
      }
      if (!mapped.empty())
        dst.setBroadcastDimsAttr(
            DenseI32ArrayAttr::get(dst.getContext(), mapped));
    }
    if (auto attr = srcOp.getOffsetsAreDstAttr())
      dst.setOffsetsAreDstAttr(attr);
  };

  if (auto load = dyn_cast<triton::gcu::LoadOp>(src))
    copyInherentAttrs(load);
  else if (auto copy = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(src))
    copyInherentAttrs(copy);

  if (auto asyncAttr = src->getAttr(kLoadAsync)) {
    dst->setAttr(kLoadAsync, asyncAttr);
  } else if (asyncByDefault) {
    dst->setAttr(kLoadAsync, BoolAttr::get(dst.getContext(), true));
  }
}

/// Common precondition of the FuseMemDesc*LoadLocalAlloc patterns:
/// `memdesc` must be a pure (src-less) local_alloc result and exactly one
/// copy_global_to_local fills it. Barriers and the ops listed in
/// `skipUsers` (the view ops being folded) are ignored; any other user
/// makes the fusion unsafe. Returns the alloc + fill copy on success.
struct FusableCopy {
  triton::gpu::LocalAllocOp alloc;
  triton::gcu::CopyGlobalToLocalOp copy;
};
static std::optional<FusableCopy>
findFusableCopy(Value memdesc, ArrayRef<Operation *> skipUsers) {
  auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
  if (!memdescTy.getMutableMemory())
    return std::nullopt;
  auto localAlloc = memdesc.getDefiningOp<triton::gpu::LocalAllocOp>();
  if (!localAlloc || localAlloc.getSrc())
    return std::nullopt;
  triton::gcu::CopyGlobalToLocalOp copyOp;
  for (auto *user : memdesc.getUsers()) {
    if (llvm::is_contained(skipUsers, user) || isa<gpu::BarrierOp>(user))
      continue;
    auto copy = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(user);
    if (!copy || copy.getDstMem() != memdesc)
      return std::nullopt;
    if (copyOp)
      return std::nullopt;
    copyOp = copy;
  }
  if (!copyOp)
    return std::nullopt;
  return FusableCopy{localAlloc, copyOp};
}

// ===----------------------------------------------------------------------===
// Pattern 1: triton_gcu.load + ttg.local_store -> copy_global_to_local
// ===----------------------------------------------------------------------===

class FuseLoadLocalStorePattern
    : public OpRewritePattern<triton::gpu::LocalStoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalStoreOp localStoreOp,
                                PatternRewriter &rewriter) const override {
    auto gcuLoad = localStoreOp.getSrc().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (!isContiguousMemDescView(localStoreOp.getDst())) {
      LLVM_DEBUG(llvm::dbgs()
                 << "skip fusion: non-contiguous memdesc destination "
                 << localStoreOp.getDst() << "\n");
      return failure();
    }

    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), localStoreOp.getDst(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    // The broadcast mapping, the dst-offset marker and the async request of the
    // load all belong to the copy that replaces it. A plain `tle.gpu.copy` into
    // shared memory is asynchronous by default, see inheritCopyAttributes.
    inheritCopyAttributes(
        gcuLoad, copyOp, BroadcastMapping::Verbatim, {},
        /*asyncByDefault=*/isSharedMemoryMemDesc(copyOp.getDstMem()));

    rewriter.eraseOp(localStoreOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 2: triton_gcu.load + ttg.local_alloc(src) -> copy_global_to_local
// ===----------------------------------------------------------------------===

class FuseLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalAllocOp localAllocOp,
                                PatternRewriter &rewriter) const override {
    auto src = localAllocOp.getSrc();
    if (!src)
      return failure();

    auto gcuLoad = src.getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (auto afterAlloc = localAllocOp->getNextNode())
      if (auto copyOp = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(afterAlloc))
        if (copyOp.getDstMem() == localAllocOp.getResult())
          return failure();

    rewriter.setInsertionPointAfter(localAllocOp);
    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), localAllocOp.getResult(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    inheritCopyAttributes(gcuLoad, copyOp);

    rewriter.modifyOpInPlace(localAllocOp, [&]() {
      localAllocOp->setOperands(ValueRange{});
      auto oldType =
          cast<triton::gpu::MemDescType>(localAllocOp.getResult().getType());
      auto mutableType = triton::gpu::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), /*mutableMemory=*/true);
      localAllocOp.getResult().setType(mutableType);
    });

    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern: triton_gcu.load + tt.trans + ttg.local_alloc(src)
//          -> ttg.local_alloc() + copy_global_to_local (with transposed order)
//
// This eliminates redundant smem usage when a loaded tensor is transposed
// before being stored to shared memory (e.g., for tt.dot operand B in matmul
// with b.T). Instead of:
//   global -> register -> trans (needs temp smem) -> smem_for_dot
// We do:
//   global -> smem_for_dot (DTE transposes during copy via order_hint)
// ===----------------------------------------------------------------------===

class FuseTransLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalAllocOp localAllocOp,
                                PatternRewriter &rewriter) const override {
    auto src = localAllocOp.getSrc();
    if (!src)
      return failure();

    auto transOp = src.getDefiningOp<triton::TransOp>();
    if (!transOp)
      return failure();

    auto gcuLoad = transOp.getSrc().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (auto afterAlloc = localAllocOp->getNextNode())
      if (auto copyOp = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(afterAlloc))
        if (copyOp.getDstMem() == localAllocOp.getResult())
          return failure();

    auto transOrder = transOp.getOrder();
    unsigned rank = transOrder.size();
    if (rank != 2)
      return failure();

    // For a transpose with order [1, 0], we swap shape and stride dims so
    // that DTE writes directly into the transposed layout.
    // Original load: shape=[N, K], strides=[stride_bk, 1], order_hint=[-1, 1]
    // After swap:    shape=[K, N], strides=[1, stride_bk], order_hint=[1, -1]
    // The order_hint also needs to be permuted so it correctly describes
    // which dimension is contiguous in the new parameter ordering.
    auto origShapes = gcuLoad.getShape();
    auto origStrides = gcuLoad.getStrides();
    auto origOffsets = gcuLoad.getOffsets();
    SmallVector<Value> newShapes(rank), newStrides(rank), newOffsets(rank);
    for (unsigned i = 0; i < rank; ++i) {
      newShapes[i] = origShapes[transOrder[i]];
      newStrides[i] = origStrides[transOrder[i]];
      newOffsets[i] = origOffsets[transOrder[i]];
    }

    // Permute order_hint according to the transpose order.
    auto origOrderHint = gcuLoad.getOrderHint();
    SmallVector<int32_t> orderHint(rank);
    for (unsigned i = 0; i < rank; ++i)
      orderHint[i] = origOrderHint[transOrder[i]];

    rewriter.setInsertionPointAfter(localAllocOp);
    auto transCopyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), newShapes, newStrides, newOffsets,
        localAllocOp.getResult(), gcuLoad.getDefaultValue(), orderHint);
    inheritCopyAttributes(gcuLoad, transCopyOp, BroadcastMapping::Transposed,
                          transOrder);

    rewriter.modifyOpInPlace(localAllocOp, [&]() {
      localAllocOp->setOperands(ValueRange{});
      auto oldType =
          cast<triton::gpu::MemDescType>(localAllocOp.getResult().getType());
      auto mutableType = triton::gpu::MemDescType::get(
          oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
          oldType.getMemorySpace(), /*mutableMemory=*/true);
      localAllocOp.getResult().setType(mutableType);
    });

    if (transOp->use_empty())
      rewriter.eraseOp(transOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern: copy_global_to_local(memdesc) + ttg.local_load(memdesc) + tt.trans
//          -> copy_global_to_local(transposed_memdesc) +
//          ttg.local_load(transposed)
//
// This is the TLE counterpart of FuseTransLoadLocalAllocPattern. In TLE
// kernels, the user explicitly allocates shared memory and copies data into
// it, then loads and transposes. The IR pattern is:
//   %alloc = ttg.local_alloc()          : memdesc<NxKxf16>
//   triton_gcu.copy_global_to_local ..., %alloc, ..., [[-1, 1]]
//   gpu.barrier
//   %v = ttg.local_load %alloc          : tensor<NxKxf16>
//   %t = tt.trans %v                    : tensor<KxNxf16>
//
// We fuse into:
//   %alloc = ttg.local_alloc()          : memdesc<KxNxf16, transposed>
//   triton_gcu.copy_global_to_local ..., %alloc, ..., [[1, -1]]
//   %t = ttg.local_load %alloc          : tensor<KxNxf16>
//
// Notes: the first copy_global_to_local is derived from below IR:
//   %alloc = ttg.local_alloc()          : memdesc<NxKxf16>
//   %load = triton_gcu.load global_ptr  : ... -> tensor<NxKxf16>
//   ttg.local_store %load, %alloc
//   gpu.barrier
//   %v = ttg.local_load %alloc          : tensor<NxKxf16>
//   %t = tt.trans %v                    : tensor<KxNxf16>
// ===----------------------------------------------------------------------===

class FuseTransLocalLoadCopyPattern : public OpRewritePattern<triton::TransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::TransOp transOp,
                                PatternRewriter &rewriter) const override {
    auto transOrder = transOp.getOrder();
    unsigned rank = transOrder.size();
    if (rank != 2)
      return failure();

    auto localLoad = transOp.getSrc().getDefiningOp<triton::gpu::LocalLoadOp>();
    if (!localLoad)
      return failure();

    auto memdesc = localLoad.getSrc();
    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto fused = findFusableCopy(memdesc, {localLoad.getOperation()});
    if (!fused)
      return failure();
    auto localAlloc = fused->alloc;
    triton::gcu::CopyGlobalToLocalOp copyOp = fused->copy;

    // Ensure local_load has only one use (the trans).
    if (!localLoad.getResult().hasOneUse())
      return failure();

    // Build transposed memdesc type with NVMMAShared transposed layout.
    auto origShape = memdescTy.getShape();
    SmallVector<int64_t> transposedShape(rank);
    for (unsigned i = 0; i < rank; ++i)
      transposedShape[i] = origShape[transOrder[i]];

    // The backend (DTE engine + local_load lowering) does not consume the
    // swizzle parameters of the shared encoding, so a fixed, simple
    // transposed nv_mma_shared encoding is used here deliberately.
    auto elemTy = memdescTy.getElementType();
    unsigned elemBitWidth = elemTy.getIntOrFloatBitWidth();
    auto ctaLayout =
        triton_gcu::compat::getDefaultCGALayout(rewriter.getContext(), rank);
    auto transposedEnc = triton::gpu::NVMMASharedEncodingAttr::get(
        rewriter.getContext(),
        /*swizzlingByteWidth=*/128,
        /*transposed=*/true,
        /*elementBitWidth=*/elemBitWidth,
        /*fp4Padded=*/false, ctaLayout);
    auto transposedMemDescTy = triton::gpu::MemDescType::get(
        transposedShape, elemTy, transposedEnc, memdescTy.getMemorySpace(),
        /*mutableMemory=*/true);

    // Permute copy_global_to_local shapes, strides, offsets, order_hint.
    auto origCopyShapes = copyOp.getShape();
    auto origCopyStrides = copyOp.getStrides();
    auto origCopyOffsets = copyOp.getOffsets();
    auto origOrderHint = copyOp.getOrderHint();
    SmallVector<Value> newShapes(rank), newStrides(rank), newOffsets(rank);
    SmallVector<int32_t> newOrderHint(rank);
    for (unsigned i = 0; i < rank; ++i) {
      newShapes[i] = origCopyShapes[transOrder[i]];
      newStrides[i] = origCopyStrides[transOrder[i]];
      newOffsets[i] = origCopyOffsets[transOrder[i]];
      newOrderHint[i] = origOrderHint[transOrder[i]];
    }

    // Replace local_alloc with transposed-shape alloc.
    rewriter.setInsertionPoint(localAlloc);
    auto newAlloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAlloc.getLoc(), transposedMemDescTy, Value());

    // Replace copy_global_to_local with transposed parameters.
    rewriter.setInsertionPoint(copyOp);
    auto newCopy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), copyOp.getPtr(), newShapes, newStrides, newOffsets,
        newAlloc.getResult(), copyOp.getDefaultValue(),
        DenseI32ArrayAttr::get(rewriter.getContext(), newOrderHint));
    inheritCopyAttributes(copyOp, newCopy, BroadcastMapping::Transposed,
                          transOrder);
    rewriter.eraseOp(copyOp);

    auto transResultTy = cast<RankedTensorType>(transOp.getType());

    // Replace local_load + trans with direct local_load from transposed alloc.
    // Any downstream convert_layout (e.g. to DotOperandEncoding) is handled
    // by FuseLocalLoadConvertLayoutPattern separately.
    rewriter.setInsertionPoint(localLoad);
    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        localLoad.getLoc(), transResultTy, newAlloc.getResult());
    rewriter.replaceOp(transOp, newLoad.getResult());
    rewriter.eraseOp(localLoad);

    // Update all remaining users (gpu.barrier etc.) to use new alloc.
    rewriter.replaceOp(localAlloc, newAlloc.getResult());

    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern:
//  copy_global_to_local -> local_alloc -> memdesc_trans -> local_load
//  =>
//  copy_global_to_local(transposed) -> local_alloc(transposed) -> local_load
//
// Folds a memdesc-level transpose (ttg.memdesc_trans) into the DTE
// copy_global_to_local's order_hint, so DTE writes the transposed layout
// directly into shared memory. This is the memdesc-level counterpart of
// FuseTransLocalLoadCopyPattern (which handles tensor-level tt.trans).
//
// Before:
//   %alloc = ttg.local_alloc() : memdesc<MxNxf16>
//   ttg.copy_global_to_local %ptr, ..., [%alloc], [order_hint]
//   %trans = ttg.memdesc_trans %alloc {order = [1, 0]}
//            : memdesc<MxN> -> memdesc<NxM>
//   %load = ttg.local_load %trans : tensor<NxMxf16>
//
// After:
//   %alloc' = ttg.local_alloc() : memdesc<NxMxf16, nv_mma_shared(transposed)>
//   ttg.copy_global_to_local %ptr, ..., [%alloc'], [permuted_order_hint]
//   %load = ttg.local_load %alloc' : tensor<NxMxf16>
// ===----------------------------------------------------------------------===
class FuseMemDescTransLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::MemDescTransOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::MemDescTransOp transOp,
                                PatternRewriter &rewriter) const override {
    auto transOrder = transOp.getOrder();
    unsigned rank = transOrder.size();
    if (rank != 2)
      return failure();

    // transOp's result must have exactly one user: a local_load.
    if (!transOp.getResult().hasOneUse())
      return failure();
    auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(
        *transOp.getResult().getUsers().begin());
    if (!localLoad)
      return failure();

    // transOp's src must be a local_alloc (pure, mutable, no src), possibly
    // through a memdesc_subslice (transposed sub-region access produced by
    // canonicalizeTranspose). In the subslice case the sub-region offsets
    // are folded into the new copy's source offsets below.
    auto memdesc = transOp.getSrc();
    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    if (!memdescTy.getMutableMemory())
      return failure();

    SmallVector<int32_t> subOffsets;
    Operation *subOp = nullptr;
    if (auto sub = memdesc.getDefiningOp<triton::gpu::MemDescSubsliceOp>()) {
      subOp = sub.getOperation();
      subOffsets = SmallVector<int32_t>(sub.getOffsets());
      memdesc = sub.getSrc();
      memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
      if (!memdescTy.getMutableMemory())
        return failure();
    }
    // Shape of the view feeding the trans (the sub-region when a subslice
    // is present, the root allocation otherwise). The transposed alloc and
    // the new copy shapes are derived from this, NOT from the root: only
    // the sub-region needs to be copied.
    auto viewShape =
        cast<triton::gpu::MemDescType>(transOp.getSrc().getType()).getShape();

    auto fused = findFusableCopy(memdesc, {transOp, subOp});
    if (!fused)
      return failure();
    auto localAlloc = fused->alloc;
    triton::gcu::CopyGlobalToLocalOp copyOp = fused->copy;
    // A copy whose offsets are destination tile offsets (mask start) cannot
    // be fused with a view: the view offsets are source offsets and both
    // would have to share the offsets operand. Leave it to the generic
    // memdesc lowering, which keeps the copy and the view separate.
    if (copyOp.getOffsetsAreDst())
      return failure();

    // Ensure local_load has only one use.
    if (!localLoad.getResult().hasOneUse())
      return failure();

    // Build transposed memdesc type with NVMMAShared transposed layout.
    // The shape is the transposed VIEW shape (sub-region aware).
    SmallVector<int64_t> transposedShape(rank);
    for (unsigned i = 0; i < rank; ++i)
      transposedShape[i] = viewShape[transOrder[i]];

    // The backend (DTE engine + local_load lowering) does not consume the
    // swizzle parameters of the shared encoding, so a fixed, simple
    // transposed nv_mma_shared encoding is used here deliberately.
    auto elemTy = memdescTy.getElementType();
    unsigned elemBitWidth = elemTy.getIntOrFloatBitWidth();
    auto ctaLayout =
        triton_gcu::compat::getDefaultCGALayout(rewriter.getContext(), rank);
    auto transposedEnc = triton::gpu::NVMMASharedEncodingAttr::get(
        rewriter.getContext(),
        /*swizzlingByteWidth=*/128,
        /*transposed=*/true,
        /*elementBitWidth=*/elemBitWidth,
        /*fp4Padded=*/false, ctaLayout);
    auto transposedMemDescTy = triton::gpu::MemDescType::get(
        transposedShape, elemTy, transposedEnc, memdescTy.getMemorySpace(),
        /*mutableMemory=*/true);

    // Permute copy_global_to_local shapes, strides, offsets, order_hint.
    // The new copy describes the transposed VIEW of the global source:
    // shapes come from the view (sub-region aware), strides are permuted
    // from the original copy, and the offsets combine the original copy
    // offsets with the subslice offsets (both in original coordinates)
    // before permutation.
    auto origCopyStrides = copyOp.getStrides();
    auto origCopyOffsets = copyOp.getOffsets();
    auto origOrderHint = copyOp.getOrderHint();
    SmallVector<Value> newShapes(rank), newStrides(rank), newOffsets(rank);
    SmallVector<int32_t> newOrderHint(rank);
    for (unsigned i = 0; i < rank; ++i) {
      unsigned srcDim = transOrder[i];
      newShapes[i] = rewriter.create<arith::ConstantIndexOp>(copyOp.getLoc(),
                                                             viewShape[srcDim]);
      newStrides[i] = origCopyStrides[srcDim];
      Value off = origCopyOffsets[srcDim];
      if (!subOffsets.empty()) {
        off = rewriter.create<arith::AddIOp>(
            copyOp.getLoc(), off,
            rewriter.create<arith::ConstantIndexOp>(copyOp.getLoc(),
                                                    subOffsets[srcDim]));
      }
      newOffsets[i] = off;
      newOrderHint[i] = origOrderHint[srcDim];
    }

    // Replace local_alloc with transposed-shape alloc.
    rewriter.setInsertionPoint(localAlloc);
    auto newAlloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAlloc.getLoc(), transposedMemDescTy, Value());

    // Replace copy_global_to_local with transposed parameters.
    rewriter.setInsertionPoint(copyOp);
    auto newCopy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), copyOp.getPtr(), newShapes, newStrides, newOffsets,
        newAlloc.getResult(), copyOp.getDefaultValue(),
        DenseI32ArrayAttr::get(rewriter.getContext(), newOrderHint));
    inheritCopyAttributes(copyOp, newCopy, BroadcastMapping::Transposed,
                          transOrder);
    rewriter.eraseOp(copyOp);

    // Replace local_load + memdesc_trans with direct local_load from
    // transposed alloc. The result type is the same as the old local_load's
    // type (it was already reading from the transposed memdesc).
    auto loadResultTy = cast<RankedTensorType>(localLoad.getType());
    rewriter.setInsertionPoint(localLoad);
    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        localLoad.getLoc(), loadResultTy, newAlloc.getResult());
    rewriter.replaceOp(localLoad, newLoad.getResult());

    // Erase memdesc_trans (its only user, local_load, has been replaced)
    // and the (now dead) subslice op it wrapped, if any.
    rewriter.eraseOp(transOp);
    if (subOp)
      rewriter.eraseOp(subOp);

    // Update all remaining users (gpu.barrier etc.) to use new alloc.
    rewriter.replaceOp(localAlloc, newAlloc.getResult());

    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern:
//  copy_global_to_local -> local_alloc -> memdesc_subslice -> local_load
//  =>
//  copy_global_to_local(adjusted_ptr) -> local_alloc(subslice) -> local_load
//
// Folds a memdesc-level subslice into the DTE copy_global_to_local.
// ConfigGcuLoad uses the offsets operand on the copy path to compute
// vSlicehape (actual copy size) and passes them to SliceStartOp,
// so we pass the subslice offsets directly to the new copy.
// ===----------------------------------------------------------------------===

class FuseMemDescSubsliceLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::MemDescSubsliceOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::MemDescSubsliceOp subOp,
                                PatternRewriter &rewriter) const override {
    auto offsets = subOp.getOffsets();
    unsigned rank = offsets.size();

    if (!subOp.getResult().hasOneUse())
      return failure();
    auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(
        *subOp.getResult().getUsers().begin());
    if (!localLoad)
      return failure();

    auto memdesc = subOp.getSrc();
    auto fused = findFusableCopy(memdesc, {subOp.getOperation()});
    if (!fused)
      return failure();
    auto localAlloc = fused->alloc;
    triton::gcu::CopyGlobalToLocalOp copyOp = fused->copy;
    // A copy whose offsets are destination tile offsets (mask start) cannot
    // be fused with a view: the view offsets are source offsets and both
    // would have to share the offsets operand. Leave it to the generic
    // memdesc lowering, which keeps the copy and the view separate.
    if (copyOp.getOffsetsAreDst())
      return failure();

    if (!localLoad.getResult().hasOneUse())
      return failure();

    auto subTy = cast<triton::gpu::MemDescType>(subOp.getType());
    // Build a new MemDescType with allocShape == shape (required by
    // local_alloc verifier). The subslice type inherits allocShape from
    // the source, which may differ from the subslice shape.
    auto newAllocTy = triton::gpu::MemDescType::get(
        subTy.getShape(), subTy.getElementType(), subTy.getEncoding(),
        subTy.getMemorySpace(), subTy.getMutableMemory(),
        subTy.getShape() /* allocShape = shape */);

    // Create new local_alloc with subslice result type.
    rewriter.setInsertionPoint(localAlloc);
    auto newAlloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAlloc.getLoc(), newAllocTy, Value());

    // Create new copy_global_to_local with subslice offsets.
    // Keep original ptr/shape/strides unchanged. ConfigGcuLoad uses
    // the offsets to compute vSlicehape (actual copy size) and passes
    // them to SliceStartOp, so DTE reads only the sub-region.
    rewriter.setInsertionPoint(copyOp);
    SmallVector<Value> newOffsets;
    for (unsigned i = 0; i < rank; ++i) {
      Value subOff =
          rewriter.create<arith::ConstantIndexOp>(copyOp.getLoc(), offsets[i]);
      newOffsets.push_back(rewriter.create<arith::AddIOp>(
          copyOp.getLoc(), copyOp.getOffsets()[i], subOff));
    }
    auto newCopy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), copyOp.getPtr(), copyOp.getShape(),
        copyOp.getStrides(), newOffsets, newAlloc.getResult(),
        copyOp.getDefaultValue(),
        DenseI32ArrayAttr::get(rewriter.getContext(), copyOp.getOrderHint()));
    inheritCopyAttributes(copyOp, newCopy);
    rewriter.eraseOp(copyOp);

    // Create new local_load from new alloc.
    rewriter.setInsertionPoint(localLoad);
    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        localLoad.getLoc(), localLoad.getType(), newAlloc.getResult());
    rewriter.replaceOp(localLoad, newLoad.getResult());

    rewriter.eraseOp(subOp);
    rewriter.replaceOp(localAlloc, newAlloc.getResult());

    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern:
//  copy_global_to_local -> local_alloc -> memdesc_reshape -> local_load
//  =>
//  copy_global_to_local(reshaped) -> local_alloc(reshaped) -> local_load
//
// Folds a memdesc-level reshape into the DTE copy_global_to_local.
// Reshape does not move the base pointer; only the shape/strides change.
// ===----------------------------------------------------------------------===

class FuseMemDescReshapeLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::MemDescReshapeOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::MemDescReshapeOp reshapeOp,
                                PatternRewriter &rewriter) const override {
    if (!reshapeOp.getResult().hasOneUse())
      return failure();
    auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(
        *reshapeOp.getResult().getUsers().begin());
    if (!localLoad)
      return failure();

    auto memdesc = reshapeOp.getSrc();
    auto fused = findFusableCopy(memdesc, {reshapeOp.getOperation()});
    if (!fused)
      return failure();
    auto localAlloc = fused->alloc;
    triton::gcu::CopyGlobalToLocalOp copyOp = fused->copy;
    // A copy whose offsets are destination tile offsets (mask start) cannot
    // be fused with a view: the view offsets are source offsets and both
    // would have to share the offsets operand. Leave it to the generic
    // memdesc lowering, which keeps the copy and the view separate.
    if (copyOp.getOffsetsAreDst())
      return failure();

    if (!localLoad.getResult().hasOneUse())
      return failure();

    auto resultTy = cast<triton::gpu::MemDescType>(reshapeOp.getType());
    auto resultShape = resultTy.getShape();
    unsigned rank = resultShape.size();

    // Verify the original copy strides are row-major w.r.t. copy shape.
    // Reshape only preserves data layout when the source is contiguous
    // row-major; non-row-major sources would be silently miscompiled.
    // Guard against rank mismatch (reshape can change rank, e.g. 1D->2D).
    if (rank != copyOp.getShape().size())
      return failure();
    auto origCopyShape = copyOp.getShape();
    auto copyStrides = copyOp.getStrides();
    for (unsigned i = 0; i < rank; ++i) {
      auto strideCst = copyStrides[i].getDefiningOp<arith::ConstantIndexOp>();
      if (!strideCst)
        return failure();
      int64_t expected = 1;
      for (unsigned j = i + 1; j < rank; ++j) {
        auto dimCst = origCopyShape[j].getDefiningOp<arith::ConstantIndexOp>();
        if (!dimCst)
          return failure();
        expected *= dimCst.value();
      }
      if (strideCst.value() != expected)
        return failure();
    }

    // Build row-major strides for the reshaped shape.
    SmallVector<int64_t> strideVals(rank, 1);
    for (int i = static_cast<int>(rank) - 2; i >= 0; --i)
      strideVals[i] = strideVals[i + 1] * resultShape[i + 1];

    SmallVector<Value> newShapes, newStrides, newOffsets;
    for (unsigned i = 0; i < rank; ++i) {
      newShapes.push_back(rewriter.create<arith::ConstantIndexOp>(
          reshapeOp.getLoc(), resultShape[i]));
      newStrides.push_back(rewriter.create<arith::ConstantIndexOp>(
          reshapeOp.getLoc(), strideVals[i]));
      newOffsets.push_back(
          rewriter.create<arith::ConstantIndexOp>(reshapeOp.getLoc(), 0));
    }
    SmallVector<int32_t> orderHint(rank);
    for (unsigned i = 0; i < rank; ++i)
      orderHint[i] = static_cast<int32_t>(rank - 1 - i);

    rewriter.setInsertionPoint(localAlloc);
    auto newAlloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAlloc.getLoc(), resultTy, Value());

    rewriter.setInsertionPoint(copyOp);
    auto newCopy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), copyOp.getPtr(), newShapes, newStrides, newOffsets,
        newAlloc.getResult(), copyOp.getDefaultValue(),
        DenseI32ArrayAttr::get(rewriter.getContext(), orderHint));
    inheritCopyAttributes(copyOp, newCopy);
    rewriter.eraseOp(copyOp);

    rewriter.setInsertionPoint(localLoad);
    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        localLoad.getLoc(), localLoad.getType(), newAlloc.getResult());
    rewriter.replaceOp(localLoad, newLoad.getResult());

    rewriter.eraseOp(reshapeOp);
    rewriter.replaceOp(localAlloc, newAlloc.getResult());

    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern:
//  copy_global_to_local -> local_alloc -> memdesc_index -> local_load
//  =>
//  copy_global_to_local(adjusted_ptr) -> local_alloc(indexed) -> local_load
//
// Folds a memdesc-level index (runtime scalar) into the DTE
// copy_global_to_local. Since the index is a runtime value, it cannot be baked
// into static offsets. Instead, the base pointer is adjusted at runtime:
//   adjusted_ptr = int2ptr(ptr2int(base_ptr) + runtime_byte_offset)
//   runtime_byte_offset = index * stride[0] * elemBytes
// This enables DTE async copy for warp-specialize scenarios.
// ===----------------------------------------------------------------------===

class FuseMemDescIndexLoadLocalAllocPattern
    : public OpRewritePattern<triton::gpu::MemDescIndexOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::MemDescIndexOp indexOp,
                                PatternRewriter &rewriter) const override {
    if (!indexOp.getResult().hasOneUse())
      return failure();
    auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(
        *indexOp.getResult().getUsers().begin());
    if (!localLoad)
      return failure();

    auto memdesc = indexOp.getSrc();
    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto fused = findFusableCopy(memdesc, {indexOp.getOperation()});
    if (!fused)
      return failure();
    auto localAlloc = fused->alloc;
    triton::gcu::CopyGlobalToLocalOp copyOp = fused->copy;
    // A copy whose offsets are destination tile offsets (mask start) cannot
    // be fused with a view: the view offsets are source offsets and both
    // would have to share the offsets operand. Leave it to the generic
    // memdesc lowering, which keeps the copy and the view separate.
    if (copyOp.getOffsetsAreDst())
      return failure();

    if (!localLoad.getResult().hasOneUse())
      return failure();

    auto srcShape = memdescTy.getShape();
    unsigned srcRank = srcShape.size();
    if (srcRank < 2)
      return failure();
    auto elemTy = memdescTy.getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    auto loc = indexOp.getLoc();

    // The index value must dominate copyOp, since we insert adjustedPtr
    // ops before copyOp. In warp-specialize scenarios the index may be a
    // loop variable or pipe_iter defined after the copy.
    // isBeforeInBlock requires both ops to live in the same block, so bail
    // out first on cross-block definitions (e.g. loop-carried values).
    if (auto indexDef = indexOp.getIndex().getDefiningOp()) {
      if (indexDef->getBlock() != copyOp->getBlock())
        return failure();
      if (copyOp->isBeforeInBlock(indexDef))
        return failure();
    }

    // All new ops must be placed before copyOp so they dominate it.
    rewriter.setInsertionPoint(copyOp);

    // Compute runtime byte offset:
    //   (index + origOffsets[0]) * strides[0] * elemBytes
    // The original copy's dim-0 offset is folded into the index so the
    // adjusted pointer addresses the same source tile the original copy
    // would have read. origOffsets[0] dominates copyOp (it is an operand
    // of the copy), so the addi below is safely placed before copyOp.
    auto origOffsets = copyOp.getOffsets();
    auto indexTy = rewriter.getIndexType();
    Value idxVal =
        rewriter.create<arith::IndexCastOp>(loc, indexTy, indexOp.getIndex());
    idxVal = rewriter.create<arith::AddIOp>(loc, idxVal, origOffsets[0]);
    Value stride0 = copyOp.getStrides()[0];
    Value elemOffset = rewriter.create<arith::MulIOp>(loc, idxVal, stride0);
    Value byteOffset = rewriter.create<arith::MulIOp>(
        loc, elemOffset,
        rewriter.create<arith::ConstantIndexOp>(loc, elemBytes));

    // Adjust base pointer: int2ptr(ptr2int(base) + byte_offset)
    auto i64Ty = rewriter.getI64Type();
    auto ptrTy = copyOp.getPtr().getType();
    Value ptrInt =
        rewriter.create<triton::gcu::PtrToIntOp>(loc, i64Ty, copyOp.getPtr());
    Value offsetI64 =
        rewriter.create<arith::IndexCastOp>(loc, i64Ty, byteOffset);
    Value adjustedInt = rewriter.create<arith::AddIOp>(loc, ptrInt, offsetI64);
    Value adjustedPtr =
        rewriter.create<triton::gcu::IntToPtrOp>(loc, ptrTy, adjustedInt);

    // Drop dim 0 from shape/strides; the remaining dims keep the original
    // copy offsets (dim 0 was folded into the adjusted pointer above).
    SmallVector<Value> newShapes(copyOp.getShape().begin() + 1,
                                 copyOp.getShape().end());
    SmallVector<Value> newStrides(copyOp.getStrides().begin() + 1,
                                  copyOp.getStrides().end());
    unsigned newRank = srcRank - 1;
    SmallVector<Value> newOffsets(origOffsets.begin() + 1, origOffsets.end());

    // Renumber order_hint: drop dim 0, decrement remaining dims.
    // If any entry is Dynamic_stride_symbol, we cannot determine the
    // correct reduced order — use a simple [0, 1, ..., newRank-1] fallback.
    auto origOrder = copyOp.getOrderHint();
    SmallVector<int32_t> newOrder;
    bool hasDynamic = false;
    for (int32_t dim : origOrder) {
      if (dim == Dynamic_stride_symbol) {
        hasDynamic = true;
        break;
      }
    }
    if (hasDynamic) {
      for (unsigned i = 0; i < newRank; ++i)
        newOrder.push_back(static_cast<int32_t>(i));
    } else {
      for (int32_t dim : origOrder)
        if (dim != 0)
          newOrder.push_back(dim - 1);
    }

    auto indexedTy = cast<triton::gpu::MemDescType>(indexOp.getType());

    // newAlloc must be before copyOp so the copy can reference it.
    rewriter.setInsertionPoint(localAlloc);
    auto newAlloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAlloc.getLoc(), indexedTy, Value());

    // adjustedPtr and newCopy placed before the old copyOp.
    rewriter.setInsertionPoint(copyOp);
    auto newCopy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), adjustedPtr, newShapes, newStrides, newOffsets,
        newAlloc.getResult(), copyOp.getDefaultValue(),
        DenseI32ArrayAttr::get(rewriter.getContext(), newOrder));
    inheritCopyAttributes(copyOp, newCopy, BroadcastMapping::DropDim0);
    rewriter.eraseOp(copyOp);

    rewriter.setInsertionPoint(localLoad);
    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        localLoad.getLoc(), localLoad.getType(), newAlloc.getResult());
    rewriter.replaceOp(localLoad, newLoad.getResult());

    rewriter.eraseOp(indexOp);
    rewriter.replaceOp(localAlloc, newAlloc.getResult());

    return success();
  }
};

// When a local_load from shared memory is immediately followed by a
// convert_layout to DotOperandEncoding, fuse them into a single local_load
// that directly produces the dot operand type.  Without this, the lowering
// would:
//   1. DTE copy smem -> private (TTLocalLoadOpLowering, blocked dst)
//   2. DTE copy private -> smem (TTGConvertLayoutOpLowering, blocked->dot)
//   3. SubView smem (loadFromSharedMemForDotOperand)
//
// With this pattern the convert_layout disappears and TTLocalLoadOpLowering
// sees a SharedEncoding -> DotOperandEncoding local_load, which it lowers
// directly as a SubView — no extra DTE copies.
//
// This is the generic counterpart of the convert_layout fold that was
// previously done inside FuseTransLocalLoadCopyPattern.  By making it a
// separate pattern, FuseTransLocalLoadCopyPattern no longer needs to
// inspect its users.
// ===----------------------------------------------------------------------===

class FuseLocalLoadConvertLayoutPattern
    : public OpRewritePattern<triton::gpu::LocalLoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalLoadOp localLoad,
                                PatternRewriter &rewriter) const override {
    auto memdesc = localLoad.getSrc();
    auto memdescTy = dyn_cast<triton::gpu::MemDescType>(memdesc.getType());
    if (!memdescTy)
      return failure();

    if (!isa<triton::gpu::SharedEncodingTrait>(memdescTy.getEncoding()))
      return failure();

    if (!localLoad.getResult().hasOneUse())
      return failure();

    auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(
        *localLoad.getResult().getUsers().begin());
    if (!convertOp)
      return failure();

    auto targetTy = cast<RankedTensorType>(convertOp.getType());
    if (!isa<triton::gpu::DotOperandEncodingAttr>(targetTy.getEncoding()))
      return failure();

    auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(localLoad.getLoc(),
                                                             targetTy, memdesc);
    rewriter.replaceOp(convertOp, newLoad.getResult());
    rewriter.eraseOp(localLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 3a: triton_gcu.load + tt.store(smem_ptr) -> copy_global_to_local
// ===----------------------------------------------------------------------===

class FuseGcuLoadSmemStorePattern : public OpRewritePattern<triton::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto smemPtr = storeOp.getPtr();
    auto mdToPtr = traceToMemDescToPtr(smemPtr);
    if (!mdToPtr)
      return failure();

    auto gcuLoad = storeOp.getValue().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (!isFullBlockSmemAccess(smemPtr, mdToPtr))
      return failure();

    auto memdesc = mdToPtr.getSrc();
    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto loadResultTy = cast<RankedTensorType>(gcuLoad.getType());
    if (memdescTy.getShape() != loadResultTy.getShape())
      return failure();

    if (!isContiguousMemDescView(memdesc))
      return failure();

    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    inheritCopyAttributes(gcuLoad, copyOp);

    rewriter.eraseOp(storeOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 3b: tt.load(smem_ptr) -> ttg.local_load(memdesc)
// ===----------------------------------------------------------------------===

class ReplaceSmemLoadWithLocalLoadPattern
    : public OpRewritePattern<triton::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto smemPtr = loadOp.getPtr();
    auto mdToPtr = traceToMemDescToPtr(smemPtr);
    if (!mdToPtr)
      return failure();

    if (!isFullBlockSmemAccess(smemPtr, mdToPtr))
      return failure();

    auto memdesc = mdToPtr.getSrc();
    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
    if (!resultTy || memdescTy.getShape() != resultTy.getShape())
      return failure();

    auto localLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        loadOp.getLoc(), resultTy, memdesc);
    rewriter.replaceOp(loadOp, localLoad.getResult());
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern: tt.load(ptr_tensor) + local_alloc(src)
//          -> local_alloc() + gather_global_to_local(ptr_tensor, alloc)
// ===----------------------------------------------------------------------===

class FuseTritonLoadLocalAllocToGatherPattern
    : public OpRewritePattern<triton::gpu::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalAllocOp localAllocOp,
                                PatternRewriter &rewriter) const override {
    auto src = localAllocOp.getSrc();
    if (!src)
      return failure();

    // Already handled by FuseLoadLocalAllocPattern.
    if (src.getDefiningOp<triton::gcu::LoadOp>())
      return failure();

    auto ttLoad = src.getDefiningOp<triton::LoadOp>();
    if (!ttLoad)
      return failure();

    if (cast<RankedTensorType>(ttLoad.getPtr().getType()).getRank() >= 3)
      return failure();

    auto oldType =
        cast<triton::gpu::MemDescType>(localAllocOp.getResult().getType());
    auto mutableType = triton::gpu::MemDescType::get(
        oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
        oldType.getMemorySpace(), /*mutableMemory=*/true);

    rewriter.setInsertionPoint(localAllocOp);
    auto alloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAllocOp.getLoc(), mutableType, Value());

    rewriter.create<triton::gcu::GatherGlobalToLocalOp>(
        ttLoad.getLoc(), ttLoad.getPtr(), alloc, ttLoad.getMask(),
        ttLoad.getOther());

    rewriter.replaceOp(localAllocOp, alloc.getResult());
    if (ttLoad->use_empty())
      rewriter.eraseOp(ttLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 4: triton_gcu.load + triton_gcu.store(smem via int2ptr chain)
//            -> copy_global_to_local
// ===----------------------------------------------------------------------===

class FuseGcuLoadGcuStoreToSmemPattern
    : public OpRewritePattern<triton::gcu::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto memdesc = traceGcuPtrToMemDesc(storeOp.getPtr());
    if (!memdesc)
      return failure();

    auto gcuLoad = storeOp.getValue().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (!isFullTileGcuStoreToSmem(storeOp, memdesc))
      return failure();

    if (!isContiguousMemDescView(memdesc))
      return failure();

    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    inheritCopyAttributes(gcuLoad, copyOp);

    rewriter.eraseOp(storeOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 5: triton_gcu.load + triton_gcu.store(smem via int2ptr chain)
//            -> copy_global_to_local
//
// Relaxed variant of Pattern 4 for dynamic (non-constant) store shapes.
// When the store's shape operands come from runtime clamping (arith.minsi),
// isFullTileGcuStoreToSmem fails because it requires ConstantIndexOp shapes.
// This pattern skips the full-tile shape check: the load and store share
// the same shape/stride operands, so correctness is guaranteed by
// construction.
// ===----------------------------------------------------------------------===

class FuseGcuLoadGcuStoreDynShapeToSmemPattern
    : public OpRewritePattern<triton::gcu::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto memdesc = traceGcuPtrToMemDesc(storeOp.getPtr());
    if (!memdesc)
      return failure();

    auto gcuLoad = storeOp.getValue().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad)
      return failure();

    if (isFullTileGcuStoreToSmem(storeOp, memdesc))
      return failure();

    if (!isContiguousMemDescView(memdesc))
      return failure();

    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto loadResultTy = cast<RankedTensorType>(gcuLoad.getType());
    if (memdescTy.getShape() != loadResultTy.getShape())
      return failure();

    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    inheritCopyAttributes(gcuLoad, copyOp);

    rewriter.eraseOp(storeOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern 6: triton_gcu.load(smem via int2ptr chain) [+ convert_layout]
//            -> ttg.local_load(memdesc)
//
// When triton_gcu.load reads from DSM (its ptr traces through
// int2ptr -> ptr_to_int -> memdesc_to_ptr to a memdesc), the data is
// already in shared memory.  Loading into registers and then doing a
// convert_layout for tt.dot is redundant -- replace the whole chain
// with a single ttg.local_load that produces the target layout directly.
// ===----------------------------------------------------------------------===

class ReplaceGcuSmemLoadWithLocalLoadPattern
    : public OpRewritePattern<triton::gcu::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::LoadOp gcuLoad,
                                PatternRewriter &rewriter) const override {
    auto memdesc = traceGcuPtrToMemDesc(gcuLoad.getPtr());
    if (!memdesc)
      return failure();

    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto loadResultTy = cast<RankedTensorType>(gcuLoad.getType());
    if (memdescTy.getShape() != loadResultTy.getShape())
      return failure();

    if (!gcuLoad.getResult().hasOneUse())
      return failure();
    Operation *user = *gcuLoad.getResult().getUsers().begin();
    auto convertOp = dyn_cast<triton::gpu::ConvertLayoutOp>(user);
    if (!convertOp)
      return failure();

    auto targetTy = cast<RankedTensorType>(convertOp.getType());

    auto localLoad = rewriter.create<triton::gpu::LocalLoadOp>(
        gcuLoad.getLoc(), targetTy, memdesc);
    rewriter.replaceOp(convertOp, localLoad.getResult());
    rewriter.eraseOp(gcuLoad);
    return success();
  }
};

static triton::gpu::MemDescType buildMemDescType(MLIRContext *ctx,
                                                 RankedTensorType tensorType) {
  auto shape = tensorType.getShape();
  auto elemTy = tensorType.getElementType();
  auto encoding = tensorType.getEncoding();
  auto rank = shape.size();

  SmallVector<unsigned> order;
  if (auto blockedEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(encoding)) {
    order = {blockedEnc.getOrder().begin(), blockedEnc.getOrder().end()};
  } else {
    for (unsigned i = 0; i < rank; ++i)
      order.push_back(rank - 1 - i);
  }
  auto ctaLayout = triton_gcu::compat::getCGALayout(encoding);
  auto sharedEnc = triton::gpu::SwizzledSharedEncodingAttr::get(
      ctx, /*vec=*/1, /*perPhase=*/1, /*maxPhase=*/1, order, ctaLayout);
  auto smemSpace = triton::gpu::SharedMemorySpaceAttr::get(ctx);
  return triton::gpu::MemDescType::get(shape, elemTy, sharedEnc, smemSpace,
                                       /*mutableMemory=*/true);
}

/// Trace a tt.load's pointer tensor backwards through the IR to recover the
/// scalar base pointer and per-dimension strides.
///
/// This is used by the tt.load fallback path when triton_gcu.load is not
/// available. Given a pointer tensor produced by the standard Triton pattern:
///
///   %base = tt.splat %scalar_ptr : !tt.ptr<f32> -> tensor<MxNx!tt.ptr<f32>>
///   %off0 = arith.muli(expand_dims(make_range(0,M)), splat(stride0))
///   %ptr0 = tt.addptr %base, %off0
///   %ptr1 = tt.addptr broadcast(%ptr0), broadcast(expand_dims(...))
///
/// Example input IR (2D, 512x512 with strides [512, 1]):
///
///   %0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x1x!tt.ptr<f32>>
///   %1 = tt.expand_dims(tt.make_range(0, 512)) : -> tensor<512x1xi32>
///   %2 = arith.muli %1, dense<512> : tensor<512x1xi32>
///   %3 = tt.addptr %0, %2 : tensor<512x1x!tt.ptr<f32>>
///   %4 = tt.broadcast %3 : -> tensor<512x512x!tt.ptr<f32>>
///   %5 = tt.broadcast(tt.expand_dims(tt.make_range(0, 512))) : ->
///                                                          tensor<512x512xi32>
///   %6 = tt.addptr %4, %5 : tensor<512x512x!tt.ptr<f32>>
///
/// Output: scalarBasePtr = %arg0, dimStrides = [512, nullptr]
///         (nullptr means stride=1, i.e., the offset is just make_range itself)
///
/// Returns true on success.
static bool tracePtrTensorBaseAndStrides(Value ptrTensor, unsigned rank,
                                         Value &scalarBasePtr,
                                         SmallVector<Value> &dimStrides,
                                         PatternRewriter &rewriter,
                                         Location loc) {
  dimStrides.assign(rank, nullptr);

  if (rank == 2) {
    auto outerAddptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
    if (!outerAddptr)
      return false;

    Value off1 = outerAddptr.getOffset();
    Value innerPtr = outerAddptr.getPtr();

    if (auto bcast = innerPtr.getDefiningOp<triton::BroadcastOp>())
      innerPtr = bcast.getSrc();

    auto innerAddptr = innerPtr.getDefiningOp<triton::AddPtrOp>();
    if (!innerAddptr)
      return false;

    Value off0 = innerAddptr.getOffset();
    Value baseVal = innerAddptr.getPtr();

    auto baseSplat = baseVal.getDefiningOp<triton::SplatOp>();
    if (!baseSplat)
      return false;
    scalarBasePtr = baseSplat.getSrc();

    // Extract dim0 stride from off0.
    {
      Value stripped = off0;
      if (auto mul = stripped.getDefiningOp<arith::MulIOp>()) {
        Value lhs = mul.getLhs(), rhs = mul.getRhs();
        Value expandVal = nullptr, strideVal = nullptr;
        if (lhs.getDefiningOp<triton::ExpandDimsOp>()) {
          expandVal = lhs;
          strideVal = rhs;
        } else if (rhs.getDefiningOp<triton::ExpandDimsOp>()) {
          expandVal = rhs;
          strideVal = lhs;
        }
        if (expandVal && strideVal) {
          if (auto strideSplat = strideVal.getDefiningOp<triton::SplatOp>()) {
            dimStrides[0] = strideSplat.getSrc();
          } else if (auto constDense =
                         strideVal.getDefiningOp<arith::ConstantOp>()) {
            if (auto denseAttr =
                    dyn_cast<DenseElementsAttr>(constDense.getValue())) {
              if (denseAttr.isSplat()) {
                auto scalarVal =
                    denseAttr.getSplatValue<APInt>().getSExtValue();
                dimStrides[0] =
                    rewriter.create<arith::ConstantIntOp>(loc, scalarVal, 32);
              }
            }
          }
        }
      }
    }

    // Extract dim1 stride from off1.
    {
      Value stripped = off1;
      if (auto bcast = stripped.getDefiningOp<triton::BroadcastOp>())
        stripped = bcast.getSrc();
      if (auto mul = stripped.getDefiningOp<arith::MulIOp>()) {
        Value lhs = mul.getLhs(), rhs = mul.getRhs();
        Value expandVal = nullptr, strideVal = nullptr;
        if (lhs.getDefiningOp<triton::ExpandDimsOp>()) {
          expandVal = lhs;
          strideVal = rhs;
        } else if (rhs.getDefiningOp<triton::ExpandDimsOp>()) {
          expandVal = rhs;
          strideVal = lhs;
        }
        if (expandVal && strideVal) {
          if (auto strideSplat = strideVal.getDefiningOp<triton::SplatOp>())
            dimStrides[1] = strideSplat.getSrc();
        }
      }
    }
  } else if (rank == 1) {
    auto addptr = ptrTensor.getDefiningOp<triton::AddPtrOp>();
    if (!addptr)
      return false;
    auto baseSplat = addptr.getPtr().getDefiningOp<triton::SplatOp>();
    if (!baseSplat)
      return false;
    scalarBasePtr = baseSplat.getSrc();
    Value off = addptr.getOffset();
    if (auto mul = off.getDefiningOp<arith::MulIOp>()) {
      Value lhs = mul.getLhs(), rhs = mul.getRhs();
      if (lhs.getDefiningOp<triton::MakeRangeOp>()) {
        if (auto strideSplat = rhs.getDefiningOp<triton::SplatOp>())
          dimStrides[0] = strideSplat.getSrc();
      } else if (rhs.getDefiningOp<triton::MakeRangeOp>()) {
        if (auto strideSplat = lhs.getDefiningOp<triton::SplatOp>())
          dimStrides[0] = strideSplat.getSrc();
      }
    }
  } else {
    return false;
  }

  return scalarBasePtr != nullptr;
}

/// Build a tile-sized pointer tensor for a sub-region of a larger tensor.
///
/// Given a scalar base pointer and per-dimension strides, constructs a
/// pointer tensor that addresses a tile starting at `tileElemOffsets`.
///
/// Example: For a 512x512 source with strides=[512, 1], extracting tile at
/// element offsets [128, 128] with tile shape [128, 128]:
///
///   Input:  scalarBasePtr = %arg0 (points to source[0, 0])
///           tileShape = [128, 128]
///           tileElemOffsets = [128, 128]
///           dimStrides = [%c512, nullptr]  (nullptr = stride 1)
///
///   Output IR:
///     %splat = tt.splat %arg0 : -> tensor<128x128x!tt.ptr<f32>>
///     %range0 = tt.make_range(128, 256) : -> tensor<128xi32>
///     %off0 = arith.muli %range0, splat(512) : tensor<128xi32>
///     %exp0 = tt.expand_dims %off0 {axis=1} : -> tensor<128x1xi32>
///     %bc0 = tt.broadcast %exp0 : -> tensor<128x128xi32>
///     %p0 = tt.addptr %splat, %bc0
///     %range1 = tt.make_range(128, 256) : -> tensor<128xi32>
///     %exp1 = tt.expand_dims %range1 {axis=0} : -> tensor<1x128xi32>
///     %bc1 = tt.broadcast %exp1 : -> tensor<128x128xi32>
///     %p1 = tt.addptr %p0, %bc1
///
///   Semantics: p1[i][j] = base + (128+i)*512 + (128+j)
///
static Value buildTilePtrTensor(PatternRewriter &rewriter, Location loc,
                                Value scalarBasePtr,
                                ArrayRef<int64_t> tileShape,
                                ArrayRef<int64_t> tileElemOffsets,
                                ArrayRef<Value> dimStrides, Type ptrElemTy,
                                Attribute encoding) {
  unsigned rank = tileShape.size();
  auto i32Ty = rewriter.getI32Type();
  auto tilePtrTy = RankedTensorType::get(tileShape, ptrElemTy, encoding);

  Value tilePtrs =
      rewriter.create<triton::SplatOp>(loc, tilePtrTy, scalarBasePtr);

  auto distEnc = cast<triton::gpu::DistributedEncodingTrait>(encoding);
  for (unsigned d = 0; d < rank; ++d) {
    SmallVector<int64_t> rangeShape(1, tileShape[d]);
    // For rank-2: dimension d's range needs SliceEnc that slices away the
    // OTHER dimension, because expand_dims will re-add that dimension.
    // For rank-2, d=0: slice away dim=1 (we'll expand at axis=1)
    // For rank-2, d=1: slice away dim=0 (we'll expand at axis=0)
    unsigned sliceDim = (rank == 2) ? (1 - d) : 0;
    auto sliceEnc = triton::gpu::SliceEncodingAttr::get(rewriter.getContext(),
                                                        sliceDim, distEnc);
    auto rangeTy = RankedTensorType::get(rangeShape, i32Ty, sliceEnc);
    Value range = rewriter.create<triton::MakeRangeOp>(
        loc, rangeTy, static_cast<uint32_t>(tileElemOffsets[d]),
        static_cast<uint32_t>(tileElemOffsets[d] + tileShape[d]));

    if (dimStrides[d]) {
      auto strideSplatTy = RankedTensorType::get(rangeShape, i32Ty, sliceEnc);
      Value strideSplat =
          rewriter.create<triton::SplatOp>(loc, strideSplatTy, dimStrides[d]);
      range = rewriter.create<arith::MulIOp>(loc, range, strideSplat);
    }

    Value expanded = range;
    for (unsigned dd = 0; dd < rank; ++dd) {
      if (dd == d)
        continue;
      expanded = rewriter.create<triton::ExpandDimsOp>(loc, expanded, dd);
    }

    auto broadcastI32Ty = RankedTensorType::get(tileShape, i32Ty, encoding);
    Value broadcasted =
        rewriter.create<triton::BroadcastOp>(loc, broadcastI32Ty, expanded);

    tilePtrs = rewriter.create<triton::AddPtrOp>(loc, tilePtrTy, tilePtrs,
                                                 broadcasted);
  }

  return tilePtrs;
}

/// Convert a linear tile index into multi-dimensional tile coordinates.
///
/// The tile grid is computed as srcShape / tileShape per dimension.
/// Delinearization uses row-major order.
///
/// Example: srcShape=[512,512], tileShape=[128,128]
///   grid = [4, 4]   (4 tiles in each dimension)
///   linearIdx=5:  5 / 4 = 1 remainder 1  ->  coords = [1, 1]
///   linearIdx=0:  coords = [0, 0]
///   linearIdx=7:  7 / 4 = 1 remainder 3  ->  coords = [1, 3]
///
///   tileElemOffsets = coords * tileShape = [1*128, 1*128] = [128, 128]
///   (the tile starts at element [128, 128] in the source tensor)
///
static SmallVector<int64_t> delinearizeTileIndex(int64_t linearIdx,
                                                 ArrayRef<int64_t> srcShape,
                                                 ArrayRef<int64_t> tileShape) {
  unsigned rank = srcShape.size();
  SmallVector<int64_t> grid(rank);
  for (unsigned d = 0; d < rank; ++d)
    grid[d] = srcShape[d] / tileShape[d];

  SmallVector<int64_t> tileCoords(rank);
  int64_t remaining = linearIdx;
  for (unsigned d = 0; d < rank; ++d) {
    int64_t suffix = 1;
    for (unsigned dd = d + 1; dd < rank; ++dd)
      suffix *= grid[dd];
    tileCoords[d] = remaining / suffix;
    remaining %= suffix;
  }
  return tileCoords;
}

// ===----------------------------------------------------------------------===
// Pattern: Fuse triton_gcu.load/tt.load + tle.extract_tile (multi-warp)
//
// Preconditions:
//   - src is from tt.load or triton_gcu.load (data is in global memory)
//   - needsSmemRelay(srcTensorTy, tileShape) is true
//   - tensor rank <= 2
//
// Transformation (two strategies, prefer Path A for best performance):
//
// Path A (preferred, direct tile-sized load):
//   Rewrites to a single tile-sized DTE that reads only the needed tile.
//   Requires: static (constant) tile index.
//
//   Before:
//     %x = triton_gcu.load %ptr, [512, 512], strides=[512, 1]
//     %tile = tle.extract_tile %x[5] {tile_shape=[128, 128]}
//
//   After:
//     %adj_ptr = %ptr + byte_offset(tile[1, 1])   // 262656 bytes
//     %tile = triton_gcu.load %adj_ptr, [128, 128], strides=[512, 1]
//
//   Performance: 1 DTE load (64KB), optimal - only reads needed data.
//
// Path B (fallback, SMEM relay):
//   Loads entire source into smem, then slices from smem.
//   Used when index is dynamic OR Path A cannot apply.
//
//   Before:
//     %x = triton_gcu.load %ptr, [256, 256], strides=[256, 1]
//     %tile = tle.extract_tile %x[5] {tile_shape=[128, 128]}
//
//   After:
//     %smem = ttg.local_alloc
//     triton_gcu.copy_global_to_local %ptr -> %smem
//     %tile = triton_gcu.slice_from_local %smem[5]
//
//   Performance: 1 full DTE (256KB) + on-chip slice. Used as fallback.
// ===----------------------------------------------------------------------===

class FuseExtractTileSmemRelay : public RewritePattern {
public:
  explicit FuseExtractTileSmemRelay(MLIRContext *ctx)
      : RewritePattern("tle.extract_tile", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return failure();

    Value src = op->getOperand(0);
    Value index = op->getOperand(1);
    auto srcTensorTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTensorTy)
      return failure();

    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    if (!tileShapeAttr)
      return failure();

    if (!triton::gcu::needsSmemRelay(srcTensorTy, tileShapeAttr.asArrayRef()))
      return failure();

    if (!src.getDefiningOp<triton::LoadOp>() &&
        !src.getDefiningOp<triton::gcu::LoadOp>())
      return failure();

    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();

    // GatherGlobalToLocal lowering only supports 1D/2D tensors.
    if (srcTensorTy.getRank() >= 3)
      return failure();

    auto srcShape = srcTensorTy.getShape();
    auto tileShape = tileShapeAttr.asArrayRef();
    auto elemTy = srcTensorTy.getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    unsigned rank = srcShape.size();
    int64_t srcBytes = elemBytes;
    for (auto s : srcShape)
      srcBytes *= s;

    // --- Path A (preferred): direct tile-sized load ---
    // Try this first: only reads the needed tile from global memory.
    if (auto result =
            tryDirectTileLoad(op, src, index, srcTensorTy, srcShape, tileShape,
                              elemTy, elemBytes, rank, srcBytes, rewriter, loc);
        result.has_value()) {
      return *result;
    }

    // --- Path B (fallback): SMEM relay ---
    auto smemTy = buildMemDescType(ctx, srcTensorTy);
    auto smemAlloc = rewriter.create<triton::gpu::LocalAllocOp>(loc, smemTy);
    if (auto ttLoad = src.getDefiningOp<triton::LoadOp>()) {
      rewriter.create<triton::gcu::GatherGlobalToLocalOp>(
          ttLoad.getLoc(), ttLoad.getPtr(), smemAlloc, ttLoad.getMask(),
          ttLoad.getOther());
    } else {
      auto gcuLoad = src.getDefiningOp<triton::gcu::LoadOp>();
      auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
          gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
          gcuLoad.getStrides(), gcuLoad.getOffsets(), smemAlloc,
          gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
      inheritCopyAttributes(gcuLoad, copyOp);
    }
    if (src.getDefiningOp()->use_empty())
      rewriter.eraseOp(src.getDefiningOp());

    auto tileStridesAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_strides");
    auto resultTy = op->getResult(0).getType();
    auto newOp = rewriter.create<triton::gcu::SliceFromLocalOp>(
        loc, resultTy, smemAlloc, index, tileShapeAttr, tileStridesAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }

private:
  /// Path A implementation: emit a single tile-sized DTE load.
  /// Returns std::nullopt if preconditions not met (falls through to Path B).
  std::optional<LogicalResult>
  tryDirectTileLoad(Operation *op, Value src, Value index,
                    RankedTensorType srcTensorTy, ArrayRef<int64_t> srcShape,
                    ArrayRef<int64_t> tileShape, Type elemTy,
                    unsigned elemBytes, unsigned rank, int64_t srcBytes,
                    PatternRewriter &rewriter, Location loc) const {
    Operation *srcLoadOp = src.getDefiningOp();
    if (!srcLoadOp || !srcLoadOp->hasOneUse())
      return std::nullopt;

    auto constOp = index.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return std::nullopt;
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr)
      return std::nullopt;
    int64_t linearIdx = intAttr.getInt();

    auto tileCoords = delinearizeTileIndex(linearIdx, srcShape, tileShape);
    SmallVector<int64_t> tileElemOffsets(rank);
    for (unsigned d = 0; d < rank; ++d)
      tileElemOffsets[d] = tileCoords[d] * tileShape[d];

    auto resultTy = cast<RankedTensorType>(op->getResult(0).getType());
    auto resultEncoding = resultTy.getEncoding();
    auto tileResultTy =
        RankedTensorType::get(tileShape, elemTy, resultEncoding);

    if (auto gcuLoad = dyn_cast<triton::gcu::LoadOp>(srcLoadOp)) {
      Value loadPtr =
          computeAdjustedPtr(rewriter, loc, gcuLoad.getPtr(), tileElemOffsets,
                             srcShape, elemBytes, rank);

      SmallVector<Value> zeroOffsets(
          rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
      SmallVector<Value> tileShapeVals;
      for (unsigned d = 0; d < rank; ++d)
        tileShapeVals.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, tileShape[d]));

      Value tileLoaded = rewriter.create<triton::gcu::LoadOp>(
          loc, tileResultTy, loadPtr, tileShapeVals, gcuLoad.getStrides(),
          zeroOffsets, gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
      for (const auto &attr : gcuLoad->getAttrs()) {
        if (attr.getName().getValue() == "operandSegmentSizes")
          continue;
        tileLoaded.getDefiningOp()->setAttr(attr.getName(), attr.getValue());
      }

      rewriter.replaceOp(op, tileLoaded);
      if (srcLoadOp->use_empty())
        rewriter.eraseOp(srcLoadOp);
      return success();
    }

    auto ttLoad = dyn_cast<triton::LoadOp>(srcLoadOp);
    if (!ttLoad)
      return std::nullopt;

    Value ptrTensor = ttLoad.getPtr();
    Value scalarBasePtr;
    SmallVector<Value> dimStrides;
    if (!tracePtrTensorBaseAndStrides(ptrTensor, rank, scalarBasePtr,
                                      dimStrides, rewriter, loc))
      return std::nullopt;

    auto ptrTensorTy = cast<RankedTensorType>(ptrTensor.getType());
    Value tilePtrs = buildTilePtrTensor(
        rewriter, loc, scalarBasePtr, tileShape, tileElemOffsets, dimStrides,
        ptrTensorTy.getElementType(), resultEncoding);

    Value tileLoaded = rewriter.create<triton::LoadOp>(
        loc, tileResultTy, tilePtrs, /*mask=*/Value(), /*other=*/Value(),
#if TRITON_VERSION < 38
        /*boundaryCheck=*/ArrayRef<int32_t>{}, /*padding=*/nullptr,
#endif
        ttLoad.getCache(), ttLoad.getEvict(), ttLoad.getIsVolatile());

    rewriter.replaceOp(op, tileLoaded);
    if (ttLoad->use_empty())
      rewriter.eraseOp(ttLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern: Fuse triton_gcu.load/tt.load + tle.insert_tile + store (multi-warp)
//
// Preconditions:
//   - src is from tt.load or triton_gcu.load (data is in global memory)
//   - needsSmemRelay(srcTensorTy, tileShape) is true
//   - result has exactly one user: tt.store or triton_gcu.store
//   - tensor rank <= 2
//
// Transformation (two strategies):
//
// Path A (DSM overflow, tiled-copy):
//   Iterates over the tile grid, copying each tile from src->out via DTE.
//   At the target tile position, stores the new tile data instead.
//   Requires: static (constant) index, srcBytes > kDsmThresholdBytes.
//
//   Before:
//     %x = triton_gcu.load %src_ptr, [512, 512], strides=[512, 1]
//     %y = triton_gcu.load %tile_ptr, [128, 128], ...
//     %z = tle.insert_tile %x[5] = %y
//     triton_gcu.store %z, %out_ptr, [512, 512], strides=[512, 1]
//
//   After (16 tile-sized DTE pairs for a 4x4 grid):
//     // tile 0: load from src_ptr+0, store to out_ptr+0
//     // tile 1: load from src_ptr+512, store to out_ptr+512
//     // ...
//     // tile 5 (target): store %y to out_ptr+262656
//     // ...
//     // tile 15: load from src_ptr+3x128x512x4+3x128x4
//     // tile 15: load from src_ptr+787968, store to out_ptr+787968
//     //                        tile offset = (tile_row*512 + tile_col*128)*bpe
//     //                        787968 = (3*128*512 + 3*128)*4
//
//   Performance: N tile-sized DTEs (31 for 4x4 grid). Functional fallback
//   when DSM overflows. For better performance, reduce BLOCK_SIZE.
//
// Path B (normal, SMEM relay):
//   Loads entire source into DSM, then writes tile into DSM via deslice.
//   Used when srcBytes <= kDsmThresholdBytes.
//
//   After:
//     %smem = ttg.local_alloc
//     triton_gcu.copy_global_to_local %ptr -> %smem
//     %result = triton_gcu.deslice_to_local %smem, %tile, index
// ===----------------------------------------------------------------------===

class FuseInsertTileSmemRelay : public RewritePattern {
public:
  explicit FuseInsertTileSmemRelay(MLIRContext *ctx)
      : RewritePattern("tle.insert_tile", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return failure();

    Value src = op->getOperand(0);
    Value tile = op->getOperand(1);
    Value index = op->getOperand(2);
    auto srcTensorTy = dyn_cast<RankedTensorType>(src.getType());
    if (!srcTensorTy)
      return failure();

    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    if (!tileShapeAttr) {
      auto tileTy = dyn_cast<RankedTensorType>(tile.getType());
      if (!tileTy)
        return failure();
      tileShapeAttr =
          DenseI64ArrayAttr::get(op->getContext(), tileTy.getShape());
    }

    if (!triton::gcu::needsSmemRelay(srcTensorTy, tileShapeAttr.asArrayRef()))
      return failure();

    if (!src.getDefiningOp<triton::LoadOp>() &&
        !src.getDefiningOp<triton::gcu::LoadOp>())
      return failure();

    // GatherGlobalToLocal lowering only supports 1D/2D tensors.
    if (srcTensorTy.getRank() >= 3)
      return failure();

    auto loc = op->getLoc();
    auto *ctx = rewriter.getContext();
    auto srcShape = srcTensorTy.getShape();
    auto elemTy = srcTensorTy.getElementType();
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8;
    int64_t srcBytes = elemBytes;
    for (auto s : srcShape)
      srcBytes *= s;

    constexpr int64_t kDsmThresholdBytes = kDsmCapacityBytes * 3 / 4;

    // --- Path A: tiled-copy (when DSM overflows) ---
    if (srcBytes > kDsmThresholdBytes) {
      auto tileShape = tileShapeAttr.asArrayRef();
      unsigned rank = srcShape.size();
      return tryTiledCopy(op, src, tile, index, srcTensorTy, srcShape,
                          tileShape, elemTy, elemBytes, rank, srcBytes,
                          rewriter, loc);
    }

    // --- Path B: SMEM relay (source fits in DSM) ---
    auto smemTy = buildMemDescType(ctx, srcTensorTy);
    auto smemAlloc = rewriter.create<triton::gpu::LocalAllocOp>(loc, smemTy);
    if (auto ttLoad = src.getDefiningOp<triton::LoadOp>()) {
      rewriter.create<triton::gcu::GatherGlobalToLocalOp>(
          ttLoad.getLoc(), ttLoad.getPtr(), smemAlloc, ttLoad.getMask(),
          ttLoad.getOther());
    } else {
      auto gcuLoad = src.getDefiningOp<triton::gcu::LoadOp>();
      auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
          gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
          gcuLoad.getStrides(), gcuLoad.getOffsets(), smemAlloc,
          gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
      inheritCopyAttributes(gcuLoad, copyOp);
    }
    if (src.getDefiningOp()->use_empty())
      rewriter.eraseOp(src.getDefiningOp());

    auto tileStridesAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_strides");
    auto resultTy = op->getResult(0).getType();
    auto newOp = rewriter.create<triton::gcu::DesliceToLocalOp>(
        loc, resultTy, smemAlloc, tile, index, tileShapeAttr, tileStridesAttr);
    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }

private:
  static constexpr int64_t kDsmCapacityBytes = 896 * 1024;
  /// Path A: emit tiled-copy loop over the tile grid.
  LogicalResult tryTiledCopy(Operation *op, Value src, Value tile, Value index,
                             RankedTensorType srcTensorTy,
                             ArrayRef<int64_t> srcShape,
                             ArrayRef<int64_t> tileShape, Type elemTy,
                             unsigned elemBytes, unsigned rank,
                             int64_t srcBytes, PatternRewriter &rewriter,
                             Location loc) const {
    Operation *srcLoadOp = src.getDefiningOp();
    if (!srcLoadOp || !srcLoadOp->hasOneUse())
      return failure();

    Value result = op->getResult(0);
    if (!result.hasOneUse())
      return failure();
    Operation *userOp = *result.getUsers().begin();
    auto ttStoreOp = dyn_cast<triton::StoreOp>(userOp);
    auto gcuStoreOp = dyn_cast<triton::gcu::StoreOp>(userOp);
    if (!ttStoreOp && !gcuStoreOp)
      return failure();

    auto constOp = index.getDefiningOp<arith::ConstantOp>();
    if (!constOp)
      return failure();
    auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
    if (!intAttr)
      return failure();

    int64_t linearIdx = intAttr.getInt();

    int64_t totalTiles = 1;
    for (unsigned d = 0; d < rank; ++d)
      totalTiles *= srcShape[d] / tileShape[d];

    if (gcuStoreOp)
      return emitGcuTiledCopy(op, src, tile, srcLoadOp, gcuStoreOp, srcShape,
                              tileShape, elemTy, elemBytes, rank, srcBytes,
                              linearIdx, totalTiles, rewriter, loc);

    if (ttStoreOp)
      return emitTtTiledCopy(op, src, tile, srcLoadOp, ttStoreOp, srcTensorTy,
                             srcShape, tileShape, elemTy, elemBytes, rank,
                             srcBytes, linearIdx, totalTiles, rewriter, loc);

    return failure();
  }

  /// Emit tiled-copy for the triton_gcu.store path.
  LogicalResult
  emitGcuTiledCopy(Operation *op, Value src, Value tile, Operation *srcLoadOp,
                   triton::gcu::StoreOp gcuStoreOp, ArrayRef<int64_t> srcShape,
                   ArrayRef<int64_t> tileShape, Type elemTy, unsigned elemBytes,
                   unsigned rank, int64_t srcBytes, int64_t linearIdx,
                   int64_t totalTiles, PatternRewriter &rewriter,
                   Location loc) const {
    auto gcuLoadOp = cast<triton::gcu::LoadOp>(srcLoadOp);
    rewriter.setInsertionPoint(gcuStoreOp);

    auto tileTy = dyn_cast<RankedTensorType>(tile.getType());
    auto tileResultTy =
        tileTy ? tileTy
               : RankedTensorType::get(
                     tileShape, elemTy,
                     cast<RankedTensorType>(src.getType()).getEncoding());

    SmallVector<Value> zeroOffsets(
        rank, rewriter.create<arith::ConstantIndexOp>(loc, 0));
    SmallVector<Value> tileShapeVals;
    for (unsigned d = 0; d < rank; ++d)
      tileShapeVals.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, tileShape[d]));

    for (int64_t tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
      auto coords = delinearizeTileIndex(tileIdx, srcShape, tileShape);
      SmallVector<int64_t> offsets(rank);
      for (unsigned d = 0; d < rank; ++d)
        offsets[d] = coords[d] * tileShape[d];

      Value adjStorePtr =
          computeAdjustedPtr(rewriter, loc, gcuStoreOp.getPtr(), offsets,
                             srcShape, elemBytes, rank);

      if (tileIdx == linearIdx) {
        rewriter.create<triton::gcu::StoreOp>(
            loc, tile, adjStorePtr, tileShapeVals, gcuStoreOp.getStrides(),
            zeroOffsets, gcuStoreOp.getOrderHint());
      } else {
        Value adjLoadPtr =
            computeAdjustedPtr(rewriter, loc, gcuLoadOp.getPtr(), offsets,
                               srcShape, elemBytes, rank);
        auto chunkLoaded = rewriter.create<triton::gcu::LoadOp>(
            loc, tileResultTy, adjLoadPtr, tileShapeVals,
            gcuLoadOp.getStrides(), zeroOffsets, gcuLoadOp.getDefaultValue(),
            gcuLoadOp.getOrderHint());
        for (const auto &attr : gcuLoadOp->getAttrs()) {
          if (attr.getName().getValue() == "operandSegmentSizes")
            continue;
          chunkLoaded->setAttr(attr.getName(), attr.getValue());
        }
        rewriter.create<triton::gcu::StoreOp>(
            loc, chunkLoaded.getResult(), adjStorePtr, tileShapeVals,
            gcuStoreOp.getStrides(), zeroOffsets, gcuStoreOp.getOrderHint());
      }
    }

    emitDsmWarning(op, srcBytes, elemBytes);
    rewriter.eraseOp(gcuStoreOp);
    rewriter.eraseOp(op);
    if (srcLoadOp->use_empty())
      rewriter.eraseOp(srcLoadOp);
    return success();
  }

  /// Emit tiled-copy for the tt.store path.
  LogicalResult
  emitTtTiledCopy(Operation *op, Value src, Value tile, Operation *srcLoadOp,
                  triton::StoreOp ttStoreOp, RankedTensorType srcTensorTy,
                  ArrayRef<int64_t> srcShape, ArrayRef<int64_t> tileShape,
                  Type elemTy, unsigned elemBytes, unsigned rank,
                  int64_t srcBytes, int64_t linearIdx, int64_t totalTiles,
                  PatternRewriter &rewriter, Location loc) const {
    Value outPtrTensor = ttStoreOp.getPtr();
    auto outPtrTensorTy = cast<RankedTensorType>(outPtrTensor.getType());
    Value scalarOutPtr;
    SmallVector<Value> dimStrides;
    if (!tracePtrTensorBaseAndStrides(outPtrTensor, rank, scalarOutPtr,
                                      dimStrides, rewriter, loc))
      return failure();

    auto ptrElemTy = outPtrTensorTy.getElementType();
    auto srcEncoding = srcTensorTy.getEncoding();
    auto tileTy = dyn_cast<RankedTensorType>(tile.getType());
    auto tileEncoding = tileTy ? tileTy.getEncoding() : srcEncoding;

    auto ttLoad = dyn_cast<triton::LoadOp>(srcLoadOp);
    Value scalarSrcPtr;
    SmallVector<Value> srcDimStrides;
    if (ttLoad &&
        !tracePtrTensorBaseAndStrides(ttLoad.getPtr(), rank, scalarSrcPtr,
                                      srcDimStrides, rewriter, loc))
      return failure();

    rewriter.setInsertionPoint(ttStoreOp);
    auto cache = ttStoreOp.getCache();
    auto evict = ttStoreOp.getEvict();
    auto tileResultTy = RankedTensorType::get(tileShape, elemTy, tileEncoding);

    for (int64_t tileIdx = 0; tileIdx < totalTiles; ++tileIdx) {
      auto coords = delinearizeTileIndex(tileIdx, srcShape, tileShape);
      SmallVector<int64_t> elemOffsets(rank);
      for (unsigned d = 0; d < rank; ++d)
        elemOffsets[d] = coords[d] * tileShape[d];

      Value storeTilePtrs =
          buildTilePtrTensor(rewriter, loc, scalarOutPtr, tileShape,
                             elemOffsets, dimStrides, ptrElemTy, tileEncoding);

      if (tileIdx == linearIdx) {
        rewriter.create<triton::StoreOp>(loc, storeTilePtrs, tile,
                                         /*mask=*/Value(),
#if TRITON_VERSION >= 38
                                         cache, evict);
#else
                                         /*boundaryCheck=*/ArrayRef<int32_t>{},
                                         cache, evict);
#endif
      } else {
        Value loadTilePtrs = buildTilePtrTensor(
            rewriter, loc, scalarSrcPtr, tileShape, elemOffsets, srcDimStrides,
            ptrElemTy, tileEncoding);
        Value chunkLoaded = rewriter.create<triton::LoadOp>(
            loc, tileResultTy, loadTilePtrs, /*mask=*/Value(),
            /*other=*/Value(),
#if TRITON_VERSION < 38
            /*boundaryCheck=*/ArrayRef<int32_t>{}, /*padding=*/nullptr,
#endif
            cache, evict, ttLoad.getIsVolatile());
        rewriter.create<triton::StoreOp>(loc, storeTilePtrs, chunkLoaded,
                                         /*mask=*/Value(),
#if TRITON_VERSION >= 38
                                         cache, evict);
#else
                                         /*boundaryCheck=*/ArrayRef<int32_t>{},
                                         cache, evict);
#endif
      }
    }

    emitDsmWarning(op, srcBytes, elemBytes);
    rewriter.eraseOp(ttStoreOp);
    rewriter.eraseOp(op);
    if (srcLoadOp->use_empty())
      rewriter.eraseOp(srcLoadOp);
    return success();
  }

  static void emitDsmWarning(Operation *op, int64_t srcBytes,
                             unsigned elemBytes) {
    op->emitWarning()
        << "tle.insert_tile source tensor (" << srcBytes
        << " bytes) exceeds DSM capacity. Applying tiled-copy optimization. "
        << "For best performance, consider reducing BLOCK_SIZE to fit within "
        << (kDsmCapacityBytes / elemBytes) << " elements.";
  }
};

// ===----------------------------------------------------------------------===
// Pattern: triton_gcu.load + tt.broadcast + ttg.local_alloc
//          -> ttg.local_alloc() + copy_global_to_local (broadcast inferred
//             at lowering from shape != dstMem shape)
//
// When the broadcast source is not a triton_gcu.load, fall back to
// ttg.local_alloc() + ttg.local_store(broadcast, alloc).
// ===----------------------------------------------------------------------===
class FuseBroadcastLocalAllocPattern
    : public OpRewritePattern<triton::gpu::LocalAllocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gpu::LocalAllocOp localAllocOp,
                                PatternRewriter &rewriter) const override {
    auto src = localAllocOp.getSrc();
    if (!src)
      return failure();
    auto broadcastOp = src.getDefiningOp<triton::BroadcastOp>();
    if (!broadcastOp)
      return failure();

    auto oldType =
        cast<triton::gpu::MemDescType>(localAllocOp.getResult().getType());
    auto mutableType = triton::gpu::MemDescType::get(
        oldType.getShape(), oldType.getElementType(), oldType.getEncoding(),
        oldType.getMemorySpace(), /*mutableMemory=*/true);

    auto splitToLocalStore = [&]() {
      rewriter.setInsertionPoint(localAllocOp);
      auto alloc = rewriter.create<triton::gpu::LocalAllocOp>(
          localAllocOp.getLoc(), mutableType, Value());
      rewriter.create<triton::gpu::LocalStoreOp>(localAllocOp.getLoc(), src,
                                                 alloc.getResult());
      rewriter.replaceOp(localAllocOp, alloc.getResult());
    };

    auto gcuLoad = broadcastOp.getSrc().getDefiningOp<triton::gcu::LoadOp>();
    if (!gcuLoad) {
      splitToLocalStore();
      return success();
    }

    auto orderHint = gcuLoad.getOrderHint();
    if (llvm::any_of(orderHint,
                     [](int32_t o) { return o == Dynamic_stride_symbol; })) {
      splitToLocalStore();
      return success();
    }

    // Fuse into copy_global_to_local. The source (load) shape differs from
    // the dst memdesc shape; the broadcast is inferred at lowering time.
    rewriter.setInsertionPoint(localAllocOp);
    auto alloc = rewriter.create<triton::gpu::LocalAllocOp>(
        localAllocOp.getLoc(), mutableType, Value());
    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), alloc.getResult(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    inheritCopyAttributes(gcuLoad, copyOp);

    rewriter.replaceOp(localAllocOp, alloc.getResult());
    if (broadcastOp->use_empty())
      rewriter.eraseOp(broadcastOp);
    if (gcuLoad->use_empty())
      rewriter.eraseOp(gcuLoad);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pattern: triton_gcu.load -> ttg.convert_layout(dot_operand) [xN]
//          => copy_global_to_local + ttg.local_load(dot_operand) [xN]
//
// A global load whose result is converted to dot operands via convert_layout
// would lower EACH convert_layout as a private->smem->private round-trip
// (storeToSharedMem + loadFromSharedMemForDotOperand). Instead, load directly
// into shared memory once (copy_global_to_local) and produce each dot operand
// with a local_load (a cheap SubView at lowering time). Any remaining
// non-dot-operand uses (elementwise) get a local_load back to the blocked
// layout. This avoids re-storing the same tile to smem once per convert_layout.
// ===----------------------------------------------------------------------===

class FuseGcuLoadDotOperandPattern
    : public OpRewritePattern<triton::gcu::LoadOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::gcu::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    auto result = loadOp.getResult();
    auto tensorTy = dyn_cast<RankedTensorType>(result.getType());
    if (!tensorTy)
      return failure();

    if (triton::gcu::getNumWarps(loadOp.getOperation()) <= 1)
      return failure();

    // TODO(Support) Generate CopyGlobalToLocalOp in WarpSpecialize Pattern
    if (loadOp->getParentOfType<triton::gpu::WarpSpecializeOp>())
      return failure();

    SmallVector<triton::gpu::ConvertLayoutOp> dotConverts;
    SmallVector<OpOperand *> otherUses;
    for (OpOperand &use : result.getUses()) {
      if (auto convert =
              dyn_cast<triton::gpu::ConvertLayoutOp>(use.getOwner())) {
        auto convertTy = dyn_cast<RankedTensorType>(convert.getType());
        if (convertTy &&
            isa<triton::gpu::DotOperandEncodingAttr>(convertTy.getEncoding()))
          dotConverts.push_back(convert);
        else
          otherUses.push_back(&use);
      } else {
        otherUses.push_back(&use);
      }
    }

    if (dotConverts.empty())
      return failure();

    // When the load also feeds non-dot-operand (elementwise) users, routing
    // those through shared memory is only valid if the blocked encoding splits
    // just the highest (outermost) dim — then each warp's smem slice is a
    // simple contiguous row block and the elementwise can read it with only an
    // offset fixup. Otherwise leave the elementwise use on its original path.
    if (!otherUses.empty()) {
      if (auto blockedEnc = dyn_cast<triton::gpu::BlockedEncodingAttr>(
              tensorTy.getEncoding())) {
        auto warpsPerCTA = triton::gcu::getWarpsPerCTA(blockedEnc);
        for (unsigned i = 1; i < warpsPerCTA.size(); ++i)
          if (warpsPerCTA[i] != 1)
            return failure();
      }
    }

    auto loc = loadOp.getLoc();
    auto memdescTy = buildMemDescType(rewriter.getContext(), tensorTy);
    auto alloc = rewriter.create<triton::gpu::LocalAllocOp>(loc, memdescTy);
    auto copy = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        loc, loadOp.getPtr(), loadOp.getShape(), loadOp.getStrides(),
        loadOp.getOffsets(), alloc.getResult(), loadOp.getDefaultValue(),
        loadOp.getOrderHint());
    // The dot operand is prefetched: the copy is issued asynchronously and the
    // wait is placed at the first use of the shared buffer.
    inheritCopyAttributes(loadOp, copy, BroadcastMapping::Verbatim, {},
                          /*asyncByDefault=*/true);

    for (auto convert : dotConverts) {
      rewriter.setInsertionPoint(convert);
      auto newLoad = rewriter.create<triton::gpu::LocalLoadOp>(
          convert.getLoc(), convert.getType(), alloc.getResult());
      rewriter.replaceOp(convert, newLoad.getResult());
    }

    // Insert ONE blocked local_load at the earliest point that dominates all
    // elementwise uses.
    if (!otherUses.empty()) {
      Operation *earliest = nullptr;
      for (auto *use : otherUses) {
        Operation *owner = use->getOwner();
        while (owner->getBlock() != loadOp->getBlock())
          owner = owner->getParentOp();
        if (!earliest || owner->isBeforeInBlock(earliest))
          earliest = owner;
      }
      rewriter.setInsertionPoint(earliest);
      auto blockedLoad = rewriter.create<triton::gpu::LocalLoadOp>(
          loc, result.getType(), alloc.getResult());
      for (auto *use : otherUses)
        use->set(blockedLoad.getResult());
    }

    rewriter.eraseOp(loadOp);
    return success();
  }
};

// ===----------------------------------------------------------------------===
// Pass definition
// ===----------------------------------------------------------------------===

struct TritonGCULocalMemOptimizePass
    : public impl::TritonGCULocalMemOptimizePassBase<
          TritonGCULocalMemOptimizePass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::TritonDialect, triton::gcu::TritonGCUDialect,
                    triton::gpu::TritonGPUDialect, mlir::gpu::GPUDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns
        .add<FuseLoadLocalAllocPattern, FuseBroadcastLocalAllocPattern,
             FuseTransLoadLocalAllocPattern, FuseTransLocalLoadCopyPattern,
             FuseMemDescTransLoadLocalAllocPattern,
             FuseMemDescSubsliceLoadLocalAllocPattern,
             FuseMemDescReshapeLoadLocalAllocPattern,
             FuseMemDescIndexLoadLocalAllocPattern,
             FuseLocalLoadConvertLayoutPattern, FuseGcuLoadDotOperandPattern,
             FuseLoadLocalStorePattern, FuseGcuLoadSmemStorePattern,
             ReplaceSmemLoadWithLocalLoadPattern,
             ReplaceGcuSmemLoadWithLocalLoadPattern,
             FuseGcuLoadGcuStoreToSmemPattern,
             FuseGcuLoadGcuStoreDynShapeToSmemPattern,
             FuseTritonLoadLocalAllocToGatherPattern, FuseExtractTileSmemRelay,
             FuseInsertTileSmemRelay>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
