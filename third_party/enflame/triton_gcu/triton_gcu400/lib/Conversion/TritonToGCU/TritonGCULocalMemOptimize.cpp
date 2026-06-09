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

    auto copyOp = rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), localStoreOp.getDst(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
    if (gcuLoad->hasAttr(kLoadAsync)) {
      copyOp->setAttr(kLoadAsync, gcuLoad->getAttr(kLoadAsync));
    }

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
    if (gcuLoad->hasAttr(kLoadAsync)) {
      copyOp->setAttr(kLoadAsync, gcuLoad->getAttr(kLoadAsync));
    }

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
    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), newShapes, newStrides, newOffsets,
        localAllocOp.getResult(), gcuLoad.getDefaultValue(), orderHint);

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
    if (!memdescTy.getMutableMemory())
      return failure();

    auto localAlloc = memdesc.getDefiningOp<triton::gpu::LocalAllocOp>();
    if (!localAlloc || localAlloc.getSrc())
      return failure();

    // Find the unique copy_global_to_local that fills this alloc.
    triton::gcu::CopyGlobalToLocalOp copyOp;
    for (auto *user : memdesc.getUsers()) {
      if (user == localLoad || isa<gpu::BarrierOp>(user))
        continue;
      auto copy = dyn_cast<triton::gcu::CopyGlobalToLocalOp>(user);
      if (!copy || copy.getDstMem() != memdesc)
        return failure();
      if (copyOp)
        return failure();
      copyOp = copy;
    }
    if (!copyOp)
      return failure();

    // Ensure local_load has only one use (the trans).
    if (!localLoad.getResult().hasOneUse())
      return failure();

    // Build transposed memdesc type with NVMMAShared transposed layout.
    auto origShape = memdescTy.getShape();
    SmallVector<int64_t> transposedShape(rank);
    for (unsigned i = 0; i < rank; ++i)
      transposedShape[i] = origShape[transOrder[i]];

    auto elemTy = memdescTy.getElementType();
    unsigned elemBitWidth = elemTy.getIntOrFloatBitWidth();
    auto ctaLayout =
        triton::gpu::CTAEncodingAttr::getDefault(rewriter.getContext(), rank);
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
    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        copyOp.getLoc(), copyOp.getPtr(), newShapes, newStrides, newOffsets,
        newAlloc.getResult(), copyOp.getDefaultValue(), newOrderHint);
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
// Pattern: ttg.local_load(memdesc) + ttg.convert_layout(dot_operand)
//          -> ttg.local_load(memdesc) : dot_operand_type
//
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

    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());

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

    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());

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

    auto memdescTy = cast<triton::gpu::MemDescType>(memdesc.getType());
    auto loadResultTy = cast<RankedTensorType>(gcuLoad.getType());
    if (memdescTy.getShape() != loadResultTy.getShape())
      return failure();

    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), memdesc,
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());

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
  auto ctaLayout = triton::gpu::getCTALayout(encoding);
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
      rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
          gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
          gcuLoad.getStrides(), gcuLoad.getOffsets(), smemAlloc,
          gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
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
        /*boundaryCheck=*/ArrayRef<int32_t>{}, /*padding=*/nullptr,
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
      rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
          gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
          gcuLoad.getStrides(), gcuLoad.getOffsets(), smemAlloc,
          gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());
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
        rewriter.create<triton::StoreOp>(
            loc, storeTilePtrs, tile, /*mask=*/Value(),
            /*boundaryCheck=*/ArrayRef<int32_t>{}, cache, evict);
      } else {
        Value loadTilePtrs = buildTilePtrTensor(
            rewriter, loc, scalarSrcPtr, tileShape, elemOffsets, srcDimStrides,
            ptrElemTy, tileEncoding);
        Value chunkLoaded = rewriter.create<triton::LoadOp>(
            loc, tileResultTy, loadTilePtrs, /*mask=*/Value(),
            /*other=*/Value(), /*boundaryCheck=*/ArrayRef<int32_t>{},
            /*padding=*/nullptr, cache, evict, ttLoad.getIsVolatile());
        rewriter.create<triton::StoreOp>(
            loc, storeTilePtrs, chunkLoaded, /*mask=*/Value(),
            /*boundaryCheck=*/ArrayRef<int32_t>{}, cache, evict);
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
    patterns.add<FuseTransLoadLocalAllocPattern, FuseTransLocalLoadCopyPattern,
                 FuseLocalLoadConvertLayoutPattern, FuseLoadLocalStorePattern,
                 FuseLoadLocalAllocPattern, FuseGcuLoadSmemStorePattern,
                 ReplaceSmemLoadWithLocalLoadPattern,
                 FuseGcuLoadGcuStoreToSmemPattern,
                 FuseGcuLoadGcuStoreDynShapeToSmemPattern,
                 ReplaceGcuSmemLoadWithLocalLoadPattern,
                 FuseTritonLoadLocalAllocToGatherPattern,
                 FuseExtractTileSmemRelay, FuseInsertTileSmemRelay>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
