// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEPROMOTELOCALSTORESTAGING
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

static bool hasPipelineStages(scf::ForOp forOp) {
  auto stageAttr = forOp->getAttrOfType<IntegerAttr>("tt.num_stages");
  return stageAttr && stageAttr.getInt() > 1;
}

static bool hasSameShapeAndElementType(Value tensor, Value memDesc) {
  auto tensorTy = dyn_cast<RankedTensorType>(tensor.getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!tensorTy || !memDescTy)
    return false;
  return tensorTy.getShape() == memDescTy.getShape() &&
         tensorTy.getElementType() == memDescTy.getElementType();
}

static bool isDeadLocalPointerUser(Operation *user) {
  return isa<tle::LocalPointersOp>(user) && user->use_empty();
}

static bool isDirectGlobalLoad(Value value) {
  auto load = value.getDefiningOp<triton::LoadOp>();
  if (!load)
    return false;

  Type ptrTy = load.getPtr().getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(ptrTy))
    ptrTy = tensorTy.getElementType();
  auto ttPtrTy = dyn_cast<triton::PointerType>(ptrTy);
  return ttPtrTy && ttPtrTy.getAddressSpace() != 3;
}

static bool isSafeToPromote(ttg::LocalStoreOp store, scf::ForOp forOp,
                            DominanceInfo &domInfo) {
  Value dst = store.getDst();
  Value src = store.getSrc();
  if (!hasSameShapeAndElementType(src, dst))
    return false;
  if (!isDirectGlobalLoad(src))
    return false;

  Operation *dstDef = dst.getDefiningOp();
  if (!dstDef || forOp->isAncestor(dstDef))
    return false;

  for (Operation *user : dst.getUsers()) {
    if (user == store.getOperation() || isDeadLocalPointerUser(user))
      continue;
    if (!forOp->isAncestor(user))
      return false;
    if (!domInfo.properlyDominates(store.getOperation(), user))
      return false;
  }
  return true;
}

static void eraseDeadLocalPointerUsers(Value memDesc) {
  SmallVector<Operation *> deadUsers;
  for (Operation *user : llvm::make_early_inc_range(memDesc.getUsers()))
    if (isDeadLocalPointerUser(user))
      deadUsers.push_back(user);
  for (Operation *user : deadUsers)
    user->erase();
}

static void eraseBarriersWithoutPrecedingLocalStore(scf::ForOp forOp) {
  bool sawLocalStore = false;
  SmallVector<mlir::gpu::BarrierOp> barriersToErase;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (isa<ttg::LocalStoreOp>(op)) {
      sawLocalStore = true;
      continue;
    }
    if (auto barrier = dyn_cast<mlir::gpu::BarrierOp>(op)) {
      if (!sawLocalStore)
        barriersToErase.push_back(barrier);
      sawLocalStore = false;
    }
  }

  for (mlir::gpu::BarrierOp barrier : barriersToErase)
    barrier.erase();
}

class PromoteLocalStoreStagingPass
    : public impl::TritonTlePromoteLocalStoreStagingBase<
          PromoteLocalStoreStagingPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<scf::ForOp> loops;
    module.walk([&](scf::ForOp forOp) {
      if (hasPipelineStages(forOp))
        loops.push_back(forOp);
    });

    for (scf::ForOp forOp : loops) {
      DominanceInfo domInfo(forOp);
      llvm::DenseMap<Value, ttg::LocalStoreOp> storeByDst;
      llvm::SmallPtrSet<Value, 8> duplicateStores;

      for (Operation &op : forOp.getBody()->without_terminator()) {
        auto store = dyn_cast<ttg::LocalStoreOp>(&op);
        if (!store)
          continue;
        Value dst = store.getDst();
        if (storeByDst.contains(dst)) {
          duplicateStores.insert(dst);
          continue;
        }
        storeByDst[dst] = store;
      }

      for (auto &[dst, store] : storeByDst) {
        if (!store || duplicateStores.contains(dst))
          continue;
        if (!isSafeToPromote(store, forOp, domInfo))
          continue;

        OpBuilder builder(store);
        auto stage = builder.create<ttg::LocalAllocOp>(
            store.getLoc(), store.getDst().getType(), store.getSrc());

        dst.replaceUsesWithIf(stage.getResult(), [&](OpOperand &use) {
          Operation *owner = use.getOwner();
          if (owner == store.getOperation())
            return false;
          if (!forOp->isAncestor(owner))
            return false;
          return domInfo.properlyDominates(store.getOperation(), owner);
        });

        store.erase();
        eraseDeadLocalPointerUsers(dst);
        if (dst.use_empty())
          dst.getDefiningOp()->erase();
      }
      eraseBarriersWithoutPrecedingLocalStore(forOp);
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
