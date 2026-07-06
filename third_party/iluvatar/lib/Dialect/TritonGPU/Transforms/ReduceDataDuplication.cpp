#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"

namespace mlir {
namespace triton {
namespace gpu {

#define GEN_PASS_DEF_TRITONGPUREDUCEDATADUPLICATION
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUReduceDataDuplicationPass
    : public impl::TritonGPUReduceDataDuplicationBase<
          TritonGPUReduceDataDuplicationPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mod.walk([&](triton::gpu::ConvertLayoutOp cvtOp) -> void {
      OpBuilder builder(cvtOp);
      auto srcType = cast<RankedTensorType>(cvtOp.getSrc().getType());
      auto dstType = cast<RankedTensorType>(cvtOp.getType());
      auto srcEncoding = srcType.getEncoding();
      if (isa<triton::gpu::SharedEncodingTrait>(srcEncoding))
        return;
      auto dstDotOp =
          dyn_cast<triton::gpu::DotOperandEncodingAttr>(dstType.getEncoding());
      if (!dstDotOp)
        return;
      auto srcBlocked =
          dyn_cast<triton::gpu::BlockedEncodingAttr>(srcType.getEncoding());
      bool forceSharedForSme =
          srcBlocked && srcBlocked.getIsSme() && dstDotOp.getUseSme() != 0;
      if (!cvtNeedsSharedMemory(srcType, dstType) && !forceSharedForSme)
        return;
      // SME operands are forced through shared memory. Reuse an identical
      // dominating materialization so multiple dot sites (e.g. a loop and its
      // remainder) share one local_load. Besides avoiding duplication, the
      // multiple users prevent ReorderInstructions from sinking the load into
      // the loop and extending the source memdesc lifetime across the loop.
      if (forceSharedForSme) {
        DominanceInfo domInfo(mod);
        for (Operation *user : cvtOp.getSrc().getUsers()) {
          auto existingAlloc = dyn_cast<LocalAllocOp>(user);
          if (!existingAlloc)
            continue;
          for (Operation *allocUser : existingAlloc.getResult().getUsers()) {
            auto existingLoad = dyn_cast<LocalLoadOp>(allocUser);
            if (!existingLoad || existingLoad.getType() != dstType)
              continue;
            if (!domInfo.properlyDominates(existingLoad.getOperation(),
                                           cvtOp.getOperation()))
              continue;
            cvtOp.replaceAllUsesWith(existingLoad.getResult());
            cvtOp.erase();
            return;
          }
        }
      }
      auto order = getOrderForMemory(srcType);
      auto sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(srcType.getContext());
      auto tmpType = triton::gpu::MemDescType::get(
          dstType.getShape(), dstType.getElementType(),
          triton::gpu::SwizzledSharedEncodingAttr::get(
              mod.getContext(), dstDotOp, srcType.getShape(), order,
              triton::gpu::getCTALayout(srcEncoding), srcType.getElementType()),
          sharedMemorySpace);
      auto tmp = triton::gpu::LocalAllocOp::create(builder, cvtOp.getLoc(),
                                                   tmpType, cvtOp.getSrc());
      auto newConvert = triton::gpu::LocalLoadOp::create(
          builder, cvtOp.getLoc(), dstType, tmp);
      cvtOp.replaceAllUsesWith(newConvert.getResult());
      cvtOp.erase();
    });
  }
};

} // namespace gpu
} // namespace triton
} // namespace mlir
