
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>
using namespace mlir;
using triton::DotOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::LocalAllocOp;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"
class TritonMETAXGPUChangeLayoutForInt8Pass
    : public TritonMETAXGPUChangeLayoutForInt8Base<
          TritonMETAXGPUChangeLayoutForInt8Pass> {
public:
  TritonMETAXGPUChangeLayoutForInt8Pass() = default;
  TritonMETAXGPUChangeLayoutForInt8Pass(int numStages, std::string pipeline) {
    this->numStages = numStages;
    this->pipeline = pipeline;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (this->numStages != 4 || this->pipeline != "cpasync") {
      return;
    }
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      OpBuilder builder(forOp);
      SmallVector<tt::DotOp> dots;
      SmallVector<tt::LoadOp> loads;
      tt::LoadOp targetLoad;
      for (Operation &op : forOp) {
        if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
          loads.push_back(loadOp);
        } else if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
          dots.push_back(dotOp);
        }
      }
      if (dots.size() != 1)
        return;
      // check whether to convert thread layout conditions.
      {
        // support only NT matmul
        Value a = dots[0].getA();
        Value b = dots[0].getB();
        Operation *aOp = a.getDefiningOp();
        Operation *bOp = b.getDefiningOp();
        if (!aOp) {
          return;
        }
        Value aSrc = nullptr;
        if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(aOp)) {
          aSrc = convertLayout.getSrc();
        } else if (auto convertLayout =
                       llvm::dyn_cast<ttg::ConvertLayoutOp>(aOp)) {
          aSrc = convertLayout.getSrc();
        } else {
          return;
        }
        if (!bOp) {
          return;
        }
        Value bSrc = nullptr;
        if (auto convertLayout = llvm::dyn_cast<ttg::LocalLoadOp>(bOp)) {
          bSrc = convertLayout.getSrc();
        } else if (auto convertLayout =
                       llvm::dyn_cast<ttg::ConvertLayoutOp>(bOp)) {
          bSrc = convertLayout.getSrc();
        } else {
          return;
        }
        if (aSrc && bSrc) {
          auto aorder = getOrder(aSrc);
          auto border = getOrder(bSrc);
          if (aorder.size() == 0 || border.size() == 0) {
            return;
          }
          auto orders = std::make_pair(aorder, border);
          auto layout = layouttable.at(orders);
          if (layout != Layout::TN) {
            return;
          }
        } else {
          return;
        }
      }
      // check whether to convert thread layout conditions.
      for (auto loadOp : loads) {
        if (loadOp.getResult().hasOneUse()) {
          Operation *use = *loadOp.getResult().getUsers().begin();
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            return;
          if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
            auto convertLayoutTy =
                cast<RankedTensorType>(convertLayout.getType());
            auto elemType = convertLayoutTy.getElementType();
            if (!elemType.isInteger(8)) {
              return;
            }
            auto convertShape = convertLayoutTy.getShape();
            auto result = use->getResult(0);
            auto cvtDstTy = cast<RankedTensorType>(result.getType());
            auto dotEnc =
                dyn_cast<ttg::DotOperandEncodingAttr>(cvtDstTy.getEncoding());
            auto opIdx = dotEnc.getOpIdx();
            // For tile {128, 128}, we need to convert the thread layout to
            // bring it into pipelinTN, for other tile_sizes, no thread layout
            // is required.
            if (opIdx == 0) {
              if (convertShape[0] != 64 || convertShape[1] != 64) {
                return;
              }
            }
            if (opIdx == 1) {
              if (convertShape[0] != 64 || convertShape[1] != 128) {
                return;
              }
            }
          } else {
            return;
          }
        } else {
          return;
        }
      }
      // change thread layout
      for (auto loadOp : loads) {
        targetLoad = loadOp;
        if (!targetLoad)
          return;
        auto targetTrans = llvm::dyn_cast<tt::TransOp>(
            *targetLoad.getResult().getUsers().begin());
        auto tensorType =
            cast<RankedTensorType>(targetLoad.getResult().getType());
        auto elemType = tensorType.getElementType();
        if (!elemType.isInteger(8)) {
          return;
        }
        auto resShape = tensorType.getShape();
        BlockedEncodingAttr originBlockedLayout =
            cast<BlockedEncodingAttr>(tensorType.getEncoding());
        auto sizePerThread =
            llvm::to_vector(originBlockedLayout.getSizePerThread());
        auto threadsPerWarp =
            llvm::to_vector(originBlockedLayout.getThreadsPerWarp());
        auto order = originBlockedLayout.getOrder();
        if (order.size() != 2)
          return;
        // check if num of replica and totalElemsPerThread of
        // originBlockedLayout can be divided by new tN
        auto originShapePerCTA = ttg::getShapePerCTATile(tensorType);
        threadsPerWarp[0] = 8;
        threadsPerWarp[1] = 8;
        sizePerThread[order[0]] = 8;
        BlockedEncodingAttr newBlockedLayout = ttg::BlockedEncodingAttr::get(
            tensorType.getContext(), sizePerThread, threadsPerWarp,
            originBlockedLayout.getWarpsPerCTA(), order,
            originBlockedLayout.getCTALayout());
        // add convertLayoutOp befor ptr, mask and other
        builder.setInsertionPointAfter(targetLoad);
        Value oldPtr = targetLoad.getPtr();
        auto oldPtrTy = cast<RankedTensorType>(oldPtr.getType());
        auto newPtrType = RankedTensorType::get(
            oldPtrTy.getShape(), oldPtrTy.getElementType(), newBlockedLayout);
        auto newPtr = builder.create<ttg::ConvertLayoutOp>(oldPtr.getLoc(),
                                                           newPtrType, oldPtr);
        Value oldMsk = targetLoad.getMask();
        Value newMsk;
        if (oldMsk) {
          auto oldMskTy = cast<RankedTensorType>(oldMsk.getType());
          auto newMskType = RankedTensorType::get(
              oldMskTy.getShape(), oldMskTy.getElementType(), newBlockedLayout);
          newMsk = builder.create<ttg::ConvertLayoutOp>(oldMsk.getLoc(),
                                                        newMskType, oldMsk);
        }
        Value oldOther = targetLoad.getOther();
        Value newOther;
        if (oldOther) {
          auto oldOtherTy = cast<RankedTensorType>(oldOther.getType());
          auto newOtherType = RankedTensorType::get(oldOtherTy.getShape(),
                                                    oldOtherTy.getElementType(),
                                                    newBlockedLayout);
          newOther = builder.create<ttg::ConvertLayoutOp>(
              oldOther.getLoc(), newOtherType, oldOther);
        }
        auto newResType = RankedTensorType::get(
            resShape, tensorType.getElementType(), newBlockedLayout);
        // create new loadOp
        auto newLoadOp = builder.create<triton::LoadOp>(
            targetLoad.getLoc(), newResType, newPtr, newMsk, newOther,
            targetLoad.getBoundaryCheckAttr(), targetLoad.getPaddingAttr(),
            targetLoad.getCache(), targetLoad.getEvict(),
            targetLoad.getIsVolatile());
        if (!targetTrans) {
          targetLoad.getResult().replaceAllUsesWith(newLoadOp.getResult());
        } else {
          auto newTrans = builder.create<triton::TransOp>(
              targetTrans.getLoc(), newLoadOp, targetTrans.getOrder());
          targetTrans.getResult().replaceAllUsesWith(newTrans.getResult());
        }
      }
    });
  }
};
std::unique_ptr<Pass>
mlir::createTritonMETAXGPUChangeLayoutForInt8Pass(int numStages,
                                                  std::string pipeline) {
  return std::make_unique<TritonMETAXGPUChangeLayoutForInt8Pass>(numStages,
                                                                 pipeline);
}
