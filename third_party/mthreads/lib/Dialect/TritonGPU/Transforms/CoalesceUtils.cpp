

#include "triton/Dialect/TritonGPU/Transforms/CoalesceUtils.h"
#include "mlir/Support/LLVM.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/StrUtil.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-coalesce"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::gpu {
static bool isCompatibleOrder(ArrayRef<int64_t> contiguity,
                              ArrayRef<unsigned> order) {
  for (unsigned i = 1; i < order.size(); ++i) {
    if (contiguity[order[i - 1]] < contiguity[order[i]])
      return false;
  }
  return true;
}

BlockedEncodingAttr
buildCoalescedEncoding(ModuleAxisInfoAnalysis &axisInfoAnalysis, Operation *op,
                       int numWarps, int threadsPerWarp,
                       triton::gpu::CGAEncodingAttr cgaLayout,
                       SmallVector<int64_t> shapePerCTA) {
  Value ptr = getMemAccessPtr(op);
  auto refTensorType = cast<RankedTensorType>(ptr.getType());

  LDBG("Considering op: " << *op);
  LLVM_DEBUG({
    DBGS() << "axis info of pointer: ";
    axisInfoAnalysis.getAxisInfo(ptr)->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });

  auto contiguity = axisInfoAnalysis.getAxisInfo(ptr)->getContiguity();
  SmallVector<unsigned> order = getOrderFromContiguity(contiguity);
  LDBG("order=[" << triton::join(order, ", ") << "]");

  auto matchesShape = [&refTensorType](const Value &val) {
    auto rttType = dyn_cast<RankedTensorType>(val.getType());
    return rttType && rttType.getShape() == refTensorType.getShape();
  };

  llvm::SmallSetVector<Operation *, 32> memAccessesSameShape;
  memAccessesSameShape.insert(op);
  if (ptr.getDefiningOp()) {
    for (Operation *sliceOp : mlir::getSlice(op)) {
      Value val = getMemAccessPtr(sliceOp);
      if (!val || !matchesShape(val) || memAccessesSameShape.contains(sliceOp))
        continue;
      LDBG("multi-root-slice: insert to memAccessesSameShape " << *sliceOp);
      memAccessesSameShape.insert(sliceOp);
    }
  }

  LDBG("shapePerCTA=[" << triton::join(shapePerCTA, ", ") << "]");

  int numElems = product<int64_t>(shapePerCTA);
  int numThreads = numWarps * threadsPerWarp;

  SmallVector<SmallVector<unsigned>> candidateOrders;
  for (Operation *access : memAccessesSameShape) {
    auto accessPtr = getMemAccessPtr(access);
    auto accessContiguity =
        axisInfoAnalysis.getAxisInfo(accessPtr)->getContiguity();
    auto candidate = getOrderFromContiguity(accessContiguity);
    if (llvm::find(candidateOrders, candidate) == candidateOrders.end())
      candidateOrders.push_back(std::move(candidate));
  }

  unsigned bestPerThread = 0;
  for (const auto &candidateOrder : candidateOrders) {
    if (!isCompatibleOrder(contiguity, candidateOrder))
      continue;
    unsigned candidatePerThread = 0;
    for (Operation *access : memAccessesSameShape) {
      auto accessPtr = getMemAccessPtr(access);
      auto accessContiguity =
          axisInfoAnalysis.getAxisInfo(accessPtr)->getContiguity();
      if (!isCompatibleOrder(accessContiguity, candidateOrder))
        continue;
      SmallVector<unsigned> candidate(candidateOrder);
      candidatePerThread =
          std::max(candidatePerThread,
                   getNumElementsPerThread(access, candidate, axisInfoAnalysis,
                                           shapePerCTA));
    }
    if (candidatePerThread > bestPerThread) {
      order.assign(candidateOrder.begin(), candidateOrder.end());
      bestPerThread = candidatePerThread;
    }
  }

  unsigned perThread = bestPerThread;
  perThread = std::min<int>(perThread, std::max(numElems / numThreads, 1));
  LDBG("perThread: " << perThread);

  if (!dyn_cast<triton::LoadOp>(op)) {
    // For ops that can result in a global memory write, we should enforce
    // that each thread handles at most 128 bits, which is the widest
    // available vectorized store op; otherwise, the store will have "gaps"
    // in the memory write at the warp level, resulting in worse performance.
    // For loads, we can expect that the gaps won't matter due to the L1
    // cache.
    perThread = std::min<int>(
        perThread,
        getNumElementsPerThread(op, order, axisInfoAnalysis, shapePerCTA));
  }
  SmallVector<unsigned> sizePerThread(refTensorType.getRank(), 1);
  sizePerThread[order[0]] = perThread;
  return BlockedEncodingAttr::get(op->getContext(), refTensorType.getShape(),
                                  sizePerThread, order, numWarps,
                                  threadsPerWarp, cgaLayout);
}
} // namespace mlir::triton::gpu
