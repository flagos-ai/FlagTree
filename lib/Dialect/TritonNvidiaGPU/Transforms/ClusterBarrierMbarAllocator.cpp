/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/TritonNvidiaGPU/Transforms/ClusterBarrierMbarAllocator.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace triton {
namespace nvidia_gpu {
#define GEN_PASS_DEF_TRITONNVIDIAGPUCLUSTERBARRIERMBARALLOCATORPASS
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"
} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace mlir::triton::nvidia_gpu {

namespace {

bool atomicNeedsClusterBarrier(Operation *op) {
  if (!isa<AtomicCASOp, AtomicRMWOp>(op) || op->getResult(0).use_empty() ||
      gpu::lookupNumCTAs(op) == 1)
    return false;
  auto tensorTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!tensorTy)
    return true;
  auto kBlock = StringAttr::get(op->getContext(), "block");
  return gpu::toLinearLayout(tensorTy).getFreeVariableMasks().lookup(kBlock);
}

struct ClusterBarrierMbarAllocatorPass
    : public impl::TritonNvidiaGPUClusterBarrierMbarAllocatorPassBase<
          ClusterBarrierMbarAllocatorPass> {
  void runOnOperation() override {
    runClusterBarrierMbarAllocator(getOperation());
  }
};

} // namespace

bool needsClusterBarrier(Operation *op) {
  if (isa<ClusterBarrierOp>(op))
    return true;
  if (auto cvt = dyn_cast<gpu::ConvertLayoutOp>(op)) {
    auto kBlock = StringAttr::get(op->getContext(), "block");
    return !isCvtDimSync(gpu::toLinearLayout(cvt.getSrc().getType()),
                         gpu::toLinearLayout(cvt.getType()), kBlock);
  }
  if (auto reduce = dyn_cast<ReduceOp>(op))
    return !ReduceOpHelper(reduce).isReduceWithinCTA();
  return atomicNeedsClusterBarrier(op);
}

void runClusterBarrierMbarAllocator(ModuleOp mod) {
  auto funcs = mod.getOps<triton::FuncOp>();
  auto kernelIt = llvm::find_if(
      funcs, [](triton::FuncOp func) { return triton::isKernel(func); });
  if (kernelIt == funcs.end())
    return;
  triton::FuncOp kernel = *kernelIt;

  auto sharedAttr = mod->getAttrOfType<IntegerAttr>("ttg.shared");
  int64_t shared = sharedAttr ? sharedAttr.getInt() : 0;
  int64_t nextOffset = llvm::alignTo(shared, int64_t{8});
  DenseMap<Region *, IntegerAttr> regionOffsets;
  Builder builder(mod.getContext());

  kernel.walk([&](Operation *op) {
    if (!needsClusterBarrier(op))
      return;
    if (!op->getParentOfType<gpu::WarpSpecializeOp>())
      return;
    Region *region = op->getParentRegion();
    while (!isa<gpu::WarpSpecializeOp, gpu::WarpSpecializePartitionsOp>(
        region->getParentOp()))
      region = region->getParentOp()->getParentRegion();

    auto [it, inserted] = regionOffsets.try_emplace(region);
    if (inserted) {
      it->second = builder.getI32IntegerAttr(nextOffset);
      nextOffset += kClusterBarrierMbarAllocationSize;
    }
    op->setAttr(kClusterBarrierMbarOffsetAttrName, it->second);
  });

  if (regionOffsets.empty())
    mod->removeAttr(kWSClusterBarrierCountAttrName);
  else {
    mod->setAttr(kWSClusterBarrierCountAttrName,
                 builder.getI32IntegerAttr(regionOffsets.size()));
    mod->setAttr("ttg.shared", builder.getI32IntegerAttr(nextOffset));
  }
}

} // namespace mlir::triton::nvidia_gpu
