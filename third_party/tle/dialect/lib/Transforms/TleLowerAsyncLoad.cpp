
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"
#include <iostream>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWERASYNCLOAD
#include "tle/dialect/include/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-tle-lower-async-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {}

class TritonTleLowerAsyncLoadPass
    : public impl::TritonTleLowerAsyncLoadBase<TritonTleLowerAsyncLoadPass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());
    m.walk([&, this](triton::LoadOp srcOp) {
      auto asyncAttr =
          llvm::cast_if_present<BoolAttr>(srcOp->getAttr("tt.load.async"));
      if (!asyncAttr || !asyncAttr.getValue() || !canBeAsyncLoad(srcOp))
        return;
      builder.setInsertionPointAfter(srcOp);
      // Replace the load with a local alloc + local load
      auto localAlloc = createLocalAllocForLoad(builder, srcOp);
      auto asyncCopy = createAsyncCopy(builder, srcOp, localAlloc);
      auto localLoad = createLocalLoad(builder, srcOp, localAlloc, asyncCopy);
      srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
        return use.getOwner() != localAlloc;
      });

      for (auto localLoadUser : localLoad->getUsers()) {
        if (auto localAllocOp =
                llvm::dyn_cast<triton::gpu::LocalAllocOp>(localLoadUser)) {
          for (auto allocUser : localAllocOp->getUsers()) {
            if (auto otherLocalLoadOp =
                    llvm::dyn_cast<triton::gpu::LocalLoadOp>(allocUser)) {
              // Replace new local load + local alloc + local load with the new
              // local load
              if (otherLocalLoadOp.getType() == localLoad.getType()) {
                if (otherLocalLoadOp != localLoad)
                  otherLocalLoadOp->replaceAllUsesWith(localLoad);
              } else {
                builder.setInsertionPointAfter(otherLocalLoadOp);
                auto newLocalLoad = createLocalLoad(builder, otherLocalLoadOp,
                                                    localAlloc, asyncCopy);
                otherLocalLoadOp->replaceAllUsesWith(newLocalLoad);
              }
            } else if (auto otherMemDescTransOp =
                           llvm::dyn_cast<triton::gpu::MemDescTransOp>(
                               allocUser)) {
              // Replace new local alloc + local load + local alloc + trans with
              // the new local alloc + new trans
              builder.setInsertionPointAfter(otherMemDescTransOp);
              auto newMemDescTransOp =
                  builder.create<triton::gpu::MemDescTransOp>(
                      otherMemDescTransOp.getLoc(), localAlloc,
                      otherMemDescTransOp.getOrder());
              otherMemDescTransOp->replaceAllUsesWith(newMemDescTransOp);
            } else if (auto otherWarpGroupDotOp =
                           llvm::dyn_cast<triton::nvidia_gpu::WarpGroupDotOp>(
                               allocUser)) {
              for (size_t idx = 0; idx < otherWarpGroupDotOp->getNumOperands();
                   ++idx)
                if (otherWarpGroupDotOp->getOperand(idx) == localAllocOp)
                  otherWarpGroupDotOp.setOperand(idx, localAlloc);
            }
          }
        }
      }

      srcOp->removeAttr("tt.load.async");
    });
  }

  triton::gpu::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder,
                                                    Value loadOp) {
    auto loc = loadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(loadOp.getType());
    auto order = triton::gpu::getOrder(type);
    auto ctaLayout = triton::gpu::getCTALayout(type.getEncoding());
    auto sharedEncoding = triton::getSharedEncoding(loadOp.getDefiningOp());
    auto sharedMemSpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    auto memDescType =
        triton::gpu::MemDescType::get(type.getShape(), type.getElementType(),
                                      sharedEncoding, sharedMemSpace, true);

    auto allocOp = builder.create<triton::gpu::LocalAllocOp>(loc, memDescType);
    return allocOp;
  }

  triton::gpu::AsyncWaitOp createAsyncCopy(OpBuilder &builder,
                                           triton::LoadOp loadOp,
                                           Value localAllocOp) {
    auto loc = loadOp.getLoc();
    Value src = loadOp.getPtr();
    Value mask = loadOp.getMask();
    Value other = loadOp.getOther();
    auto allocTy = cast<triton::gpu::MemDescType>(localAllocOp.getType());

    auto copyAsync = builder.create<triton::gpu::AsyncCopyGlobalToLocalOp>(
        loc, src, localAllocOp, mask, other, loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    auto commit = builder.create<triton::gpu::AsyncCommitGroupOp>(
        loc, copyAsync->getResult(0));
    // insert wait before the first use of loadop
    Operation *firstUse = nullptr;
    for (Operation *user : loadOp->getResult(0).getUsers()) {
      if (user == loadOp)
        continue;
      if (!firstUse)
        firstUse = user;
      else if (user->getBlock() == firstUse->getBlock() &&
               user->isBeforeInBlock(firstUse))
        firstUse = user;
    }

    if (firstUse)
      builder.setInsertionPoint(firstUse);
    else
      builder.setInsertionPointAfter(commit);

    auto wait =
        builder.create<triton::gpu::AsyncWaitOp>(loc, commit->getResult(0), 0);
    return wait;
  }

  triton::gpu::LocalLoadOp createLocalLoad(OpBuilder &builder, Value loadOp,
                                           Value localAllocOp,
                                           Value token = nullptr) {
    auto loc = loadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(loadOp.getType());

    auto localLoadOp = builder.create<triton::gpu::LocalLoadOp>(
        loc, type, localAllocOp, token);
    return localLoadOp;
  }
};
} // namespace mlir::triton::tle
