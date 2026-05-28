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

#include "llvm/Support/ErrorHandling.h"

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

    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), localStoreOp.getDst(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());

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
    rewriter.create<triton::gcu::CopyGlobalToLocalOp>(
        gcuLoad.getLoc(), gcuLoad.getPtr(), gcuLoad.getShape(),
        gcuLoad.getStrides(), gcuLoad.getOffsets(), localAllocOp.getResult(),
        gcuLoad.getDefaultValue(), gcuLoad.getOrderHint());

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

// ===----------------------------------------------------------------------===
// Pattern: SMEM relay for tle.extract_tile (multi-warp)
//
// Matches tle.extract_tile by op name. Applies when src is a tensor,
// needsSmemRelay is true, AND src comes from tt.load or triton_gcu.load.
//
// tt.load/triton_gcu.load(ptr_tensor) + tle.extract_tile
// ->
// smem = local_alloc()
// copy/gather_global_to_local(ptr_tensor, smem)
// triton_gcu.slice_from_local(smem, ...)
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
};

// ===----------------------------------------------------------------------===
// Pattern: SMEM relay for tle.insert_tile (multi-warp)
//
// Matches tle.insert_tile by op name. Applies when src is a tensor,
// needsSmemRelay is true, AND src comes from tt.load or triton_gcu.load.
//
// tt.load/triton_gcu.load(ptr_tensor) + tle.insert_tile
// ->
// smem = local_alloc()
// copy/gather_global_to_local(ptr_tensor, smem)
// triton_gcu.deslice_to_local(smem, ...)
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
        .add<FuseLoadLocalStorePattern, FuseLoadLocalAllocPattern,
             FuseGcuLoadSmemStorePattern, ReplaceSmemLoadWithLocalLoadPattern,
             FuseGcuLoadGcuStoreToSmemPattern,
             FuseTritonLoadLocalAllocToGatherPattern, FuseExtractTileSmemRelay,
             FuseInsertTileSmemRelay>(ctx);
    if (failed(applyPatternsGreedily(module, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
