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

#include "Transforms/Passes.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {
bool canBeAsyncLoad(Operation *op);
gpu::SharedEncodingTrait getSharedEncoding(Operation *loadOp);
} // namespace mlir::triton

namespace mlir {
#define GEN_PASS_DEF_GCUTLELOWERASYNCLOAD
#include "Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "gcu-tle-lower-async-load"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcu = mlir::triton::gcu;

namespace {

class GCUTleLowerAsyncLoadPass
    : public impl::GCUTleLowerAsyncLoadBase<GCUTleLowerAsyncLoadPass> {
public:
  using impl::GCUTleLowerAsyncLoadBase<
      GCUTleLowerAsyncLoadPass>::GCUTleLowerAsyncLoadBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());

    moduleOp.walk([&](ttgcu::LoadOp srcOp) {
      auto asyncAttr =
          llvm::cast_if_present<BoolAttr>(srcOp->getAttr("tt.load.async"));
      if (!asyncAttr || !asyncAttr.getValue())
        return;

      builder.setInsertionPointAfter(srcOp);
      auto localAlloc = createLocalAllocForLoad(builder, srcOp);

      createAsyncCopy(builder, srcOp, localAlloc);

      auto localLoad = createLocalLoad(builder, srcOp, localAlloc);

      srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
        return use.getOwner() != localAlloc;
      });

      for (auto localLoadUser : localLoad->getUsers()) {
        if (auto localAllocOp = dyn_cast<ttg::LocalAllocOp>(localLoadUser)) {
          for (auto allocUser : localAllocOp->getUsers()) {
            if (auto otherLocalLoadOp = dyn_cast<ttg::LocalLoadOp>(allocUser)) {
              // Replace local load + local alloc + local load with the new
              // local load when the types already match.
              if (otherLocalLoadOp.getType() == localLoad.getType()) {
                if (otherLocalLoadOp != localLoad)
                  otherLocalLoadOp->replaceAllUsesWith(localLoad);
              } else {
                builder.setInsertionPointAfter(otherLocalLoadOp);
                auto newLocalLoad =
                    createLocalLoad(builder, otherLocalLoadOp, localAlloc);
                otherLocalLoadOp->replaceAllUsesWith(newLocalLoad);
              }
            } else if (auto otherMemDescTransOp =
                           dyn_cast<ttg::MemDescTransOp>(allocUser)) {
              builder.setInsertionPointAfter(otherMemDescTransOp);
              auto newMemDescTransOp = builder.create<ttg::MemDescTransOp>(
                  otherMemDescTransOp.getLoc(), localAlloc,
                  otherMemDescTransOp.getOrder());
              otherMemDescTransOp->replaceAllUsesWith(newMemDescTransOp);
            } else if (auto otherWarpGroupDotOp =
                           dyn_cast<ttgcu::WarpGroupDotOp>(allocUser)) {
              for (size_t idx = 0; idx < otherWarpGroupDotOp->getNumOperands();
                   ++idx) {
                if (otherWarpGroupDotOp->getOperand(idx) == localAllocOp)
                  otherWarpGroupDotOp.setOperand(idx, localAlloc);
              }
            }
          }
        }
      }

      srcOp->removeAttr("tt.load.async");
    });
  }

  ttg::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder, Value loadOp) {
    auto loc = loadOp.getLoc();
    auto type = cast<RankedTensorType>(loadOp.getType());
    auto sharedEncoding = tt::getSharedEncoding(loadOp.getDefiningOp());
    auto sharedMemSpace = ttg::SharedMemorySpaceAttr::get(builder.getContext());
    auto memDescType = ttg::MemDescType::get(
        type.getShape(), type.getElementType(), sharedEncoding, sharedMemSpace,
        /*mutableMemory=*/true);

    return builder.create<ttg::LocalAllocOp>(loc, memDescType);
  }

  void createAsyncCopy(OpBuilder &builder, ttgcu::LoadOp loadOp,
                       Value localAllocOp) {
    auto copyAsync = builder.create<ttgcu::CopyGlobalToLocalOp>(
        loadOp.getLoc(), loadOp.getPtr(), loadOp.getShape(),
        loadOp.getStrides(), loadOp.getOffsets(), localAllocOp,
        loadOp.getDefaultValue(), loadOp.getOrderHint());

    copyAsync->setAttr("tt.load.async", builder.getBoolAttr(true));

    return;
  }

  ttg::LocalLoadOp createLocalLoad(OpBuilder &builder, Value loadOp,
                                   Value localAllocOp) {
    // Insert wait before the first use of the original load.
    Operation *firstUse = nullptr;
    for (Operation *user : loadOp.getDefiningOp()->getResult(0).getUsers()) {
      if (user == loadOp.getDefiningOp())
        continue;
      if (!firstUse || (user->getBlock() == firstUse->getBlock() &&
                        user->isBeforeInBlock(firstUse))) {
        firstUse = user;
      }
    }

    if (firstUse)
      builder.setInsertionPoint(firstUse);

    auto loc = loadOp.getLoc();
    auto type = cast<RankedTensorType>(loadOp.getType());
    return builder.create<ttg::LocalLoadOp>(loc, type, localAllocOp);
  }
};

} // namespace
