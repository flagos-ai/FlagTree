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

// flagtree tle

#include "Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::iluvatar_tle {

#define GEN_PASS_DEF_TRITONILUVATARTLEEARLYASSIGNMEMORYSPACE
#include "Transforms/Passes.h.inc"

namespace {

constexpr StringLiteral kMemorySpaceAttr = "tt.memory_space";

// A minimal swizzled shared encoding (vec=1, perPhase=1, maxPhase=1) matching
// the tensor's own iteration order; good enough to hold a staged tile.
static ttg::MemDescType getSharedMemDescType(OpBuilder &builder,
                                             RankedTensorType type,
                                             bool mutableMemory) {
  auto order = ttg::getOrder(type);
  auto ctaLayout = ttg::getCTALayout(type.getEncoding());
  auto sharedEncoding = ttg::SwizzledSharedEncodingAttr::get(
      builder.getContext(), 1, 1, 1, order, ctaLayout);
  auto sharedMemSpace = ttg::SharedMemorySpaceAttr::get(builder.getContext());
  return ttg::MemDescType::get(type.getShape(), type.getElementType(),
                               sharedEncoding, sharedMemSpace, mutableMemory);
}

} // namespace

class EarlyAssignMemorySpacePass
    : public impl::TritonIluvatarTleEarlyAssignMemorySpaceBase<
          EarlyAssignMemorySpacePass> {
  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());
    m.walk([&, this](Operation *srcOp) {
      if (srcOp->getNumResults() != 1)
        return;
      Value srcValue = srcOp->getResult(0);
      auto memorySpaceAttr =
          llvm::cast_if_present<StringAttr>(srcOp->getAttr(kMemorySpaceAttr));
      if (!isa<RankedTensorType>(srcValue.getType()) || !memorySpaceAttr ||
          memorySpaceAttr.getValue() != "shared_memory")
        return;

      builder.setInsertionPointAfter(srcOp);
      if (auto loadOp = dyn_cast<triton::LoadOp>(srcOp)) {
        // Loads unconditionally go through the async global-to-shared copy
        // path; the iluvatar backend now lowers that chain to hardware.
        auto localAlloc = createLocalAllocForLoad(builder, loadOp);
        auto wait = createAsyncCopy(builder, loadOp, localAlloc);
        auto localLoad = createLocalLoad(builder, srcValue, localAlloc, wait);
        srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
          return use.getOwner() != localAlloc;
        });
      } else {
        // Any other producer is materialized into an initialized shared alloc
        // followed by a local load.
        auto localAlloc = createLocalAllocForNonLoad(builder, srcValue);
        auto localLoad = createLocalLoad(builder, srcValue, localAlloc);
        srcOp->replaceUsesWithIf(localLoad, [&](OpOperand &use) {
          return use.getOwner() != localAlloc;
        });
      }
      srcOp->removeAttr(kMemorySpaceAttr);
    });
  }

  ttg::LocalAllocOp createLocalAllocForLoad(OpBuilder &builder, Value loadOp) {
    auto loc = loadOp.getLoc();
    auto type = cast<RankedTensorType>(loadOp.getType());
    auto memDescType = getSharedMemDescType(builder, type, /*mutable=*/true);
    return ttg::LocalAllocOp::create(builder, loc, memDescType);
  }

  ttg::LocalAllocOp createLocalAllocForNonLoad(OpBuilder &builder,
                                               Value nonLoadOp) {
    auto loc = nonLoadOp.getLoc();
    auto type = cast<RankedTensorType>(nonLoadOp.getType());
    auto memDescType = getSharedMemDescType(builder, type, /*mutable=*/false);
    return ttg::LocalAllocOp::create(builder, loc, memDescType, nonLoadOp);
  }

  ttg::AsyncWaitOp createAsyncCopy(OpBuilder &builder, triton::LoadOp loadOp,
                                   Value localAllocOp) {
    auto loc = loadOp.getLoc();
    Value src = loadOp.getPtr();
    Value mask = loadOp.getMask();
    Value other = loadOp.getOther();

    auto copyAsync = ttg::AsyncCopyGlobalToLocalOp::create(
        builder, loc, src, localAllocOp, mask, other, loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());
    auto commit =
        ttg::AsyncCommitGroupOp::create(builder, loc, copyAsync->getResult(0));

    // Insert the wait right before the first use of the original load.
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

    return ttg::AsyncWaitOp::create(builder, loc, commit->getResult(0), 0);
  }

  ttg::LocalLoadOp createLocalLoad(OpBuilder &builder, Value loadOp,
                                   Value localAllocOp, Value token = nullptr) {
    auto loc = loadOp.getLoc();
    auto type = cast<RankedTensorType>(loadOp.getType());
    return ttg::LocalLoadOp::create(builder, loc, type, localAllocOp, token);
  }
};

} // namespace mlir::triton::iluvatar_tle
