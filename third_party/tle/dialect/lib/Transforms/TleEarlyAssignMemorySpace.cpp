#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEEARLYASSIGNMEMORYSPACE
#include "tle/dialect/include/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-tle-early-assign-memory-space"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {}

class EarlyAssignMemorySpacePass
    : public impl::TritonTleEarlyAssignMemorySpaceBase<
          EarlyAssignMemorySpacePass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());
    m.walk([&, this](Operation *srcOp) {
      if (srcOp->getNumResults() == 1) {
        auto srcValue = srcOp->getResult(0);
        auto memorySpaceAttr = llvm::cast_if_present<StringAttr>(
            srcOp->getAttr("tt.memory_space"));
        if (isa<RankedTensorType>(srcValue.getType()) && memorySpaceAttr &&
            memorySpaceAttr.getValue() == "shared_memory") {
          builder.setInsertionPointAfter(srcOp);
          if (auto loadOp = dyn_cast<triton::LoadOp>(srcOp)) {
            // Replace the load with a local alloc + local load
            auto localAlloc = createLocalAllocForLoad(builder, loadOp);
            auto asyncCopy = createAsyncCopy(builder, loadOp, localAlloc);
            auto localLoad =
                createLocalLoad(builder, srcValue, localAlloc, asyncCopy);
            srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
              return use.getOwner() != localAlloc;
            });
          } else {
            auto localAlloc = createLocalAllocForNonLoad(builder, srcValue);
            auto localLoad = createLocalLoad(builder, srcValue, localAlloc);
            srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
              return use.getOwner() != localAlloc;
            });
          }
          srcOp->removeAttr("tt.memory_space");
        }
      }
    });
  }

  triton::gpu::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder,
                                                    Value loadOp) {
    auto loc = loadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(loadOp.getType());
    auto order = triton::gpu::getOrder(type);
    auto ctaLayout = triton::gpu::getCTALayout(type.getEncoding());
    auto sharedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        builder.getContext(), 1, 1, 1, order, ctaLayout);
    auto sharedMemSpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    auto memDescType =
        triton::gpu::MemDescType::get(type.getShape(), type.getElementType(),
                                      sharedEncoding, sharedMemSpace, true);

    auto allocOp = builder.create<triton::gpu::LocalAllocOp>(loc, memDescType);
    return allocOp;
  }

  triton::gpu::LocalAllocOp createLocalAllocForNonLoad(OpBuilder &builder,
                                                       Value nonLoadOp) {
    auto loc = nonLoadOp.getLoc();
    auto type = llvm::cast<RankedTensorType>(nonLoadOp.getType());
    auto order = triton::gpu::getOrder(type);
    auto ctaLayout = triton::gpu::getCTALayout(type.getEncoding());
    auto sharedEncoding = triton::gpu::SwizzledSharedEncodingAttr::get(
        builder.getContext(), 1, 1, 1, order, ctaLayout);
    auto sharedMemSpace =
        triton::gpu::SharedMemorySpaceAttr::get(builder.getContext());
    auto memDescType = triton::gpu::MemDescType::get(
        type.getShape(), type.getElementType(), sharedEncoding, sharedMemSpace);

    auto allocOp =
        builder.create<triton::gpu::LocalAllocOp>(loc, memDescType, nonLoadOp);
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
