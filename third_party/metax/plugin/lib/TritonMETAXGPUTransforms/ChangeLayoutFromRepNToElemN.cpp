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
using triton::gpu::MACAMmaEncodingAttr;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUChangeLayoutFromRepNToElemNPass
    : public TritonMETAXGPUChangeLayoutFromRepNToElemNBase<
          TritonMETAXGPUChangeLayoutFromRepNToElemNPass> {
public:
  TritonMETAXGPUChangeLayoutFromRepNToElemNPass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    // mma layout = [tM, tN, tk]
    // if tN == RepN == numstage == 4 of mmaLayout
    // then change loadOp layout sizePerThread [1, tk] -> sizePerThread [4, tk]
    // create loadOp input cvt layout

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

      auto mmaLayout = dyn_cast<ttg::MACAMmaEncodingAttr>(
          cast<RankedTensorType>(dots[0].getResult().getType()).getEncoding());
      if (!mmaLayout)
        return;
      llvm::ArrayRef<unsigned> elemMNK = mmaLayout.getElementsMNK();
      int tN = elemMNK[1];
      // auto warpsPerCTA = mmaLayout.getWarpsPerCTA();
      // auto shapePerCTA = getShapePerCTATile(mmaLayout);
      auto shape =
          cast<RankedTensorType>(dots[0].getResult().getType()).getShape();
      // Only consider N for now!!
      // check if tN = RepN of mmaLayout
      // if we change tN to 1, and new RepN == tN, then this is legal
      // 16 * tN * getWarpsPerCTA()[1] = 16*4*2 = 128
      // 16 * 1 * 2 * repN = 128 => repN = 4
      // repN = shape[1]/(shapePerCTA[1]/tN)
      // if (shape[1] != shapePerCTA[1] || tN != 4)
      //   return;

      // change B BlockedLayout sizePerThread to [tN, 8]
      // like PipelinePass checkOpUses
      for (auto loadOp : loads) {
        // check if loadOp is B
        if (loadOp.getResult().hasOneUse()) {
          Operation *use = *loadOp.getResult().getUsers().begin();
          if (use->getNumResults() != 1 || !use->getResult(0).hasOneUse())
            break;
          if (auto convertLayout = llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
            auto convertUse = *convertLayout.getResult().getUsers().begin();
            if (convertLayout.getResult() == dots[0].getB())
              targetLoad = loadOp;
            else if (auto trans = llvm::dyn_cast<tt::TransOp>(convertUse)) {
              if (trans.getResult() == dots[0].getB()) {
                targetLoad = loadOp;
              }
            }
          } else if (auto trans = llvm::dyn_cast<tt::TransOp>(use)) {
            use = *trans.getResult().getUsers().begin();
            if (auto convertLayout =
                    llvm::dyn_cast<ttg::ConvertLayoutOp>(use)) {
              if (convertLayout.getResult() == dots[0].getB())
                targetLoad = loadOp;
            }
          }
        }
      }

      if (!targetLoad)
        return;
      auto targetTrans = llvm::dyn_cast<tt::TransOp>(
          *targetLoad.getResult().getUsers().begin());
      llvm::SmallVector<unsigned> sizePerThread;
      auto tensorType =
          cast<RankedTensorType>(targetLoad.getResult().getType());
      auto resShape = tensorType.getShape();
      BlockedEncodingAttr originBlockedLayout =
          cast<BlockedEncodingAttr>(tensorType.getEncoding());
      sizePerThread = llvm::to_vector(originBlockedLayout.getSizePerThread());
      auto threadsPerWarp = originBlockedLayout.getThreadsPerWarp();
      auto order = originBlockedLayout.getOrder();
      if (order.size() != 2)
        return;

      // check if num of replica and totalElemsPerThread of originBlockedLayout
      // can be divided by new tN
      auto originShapePerCTA = ttg::getShapePerCTATile(tensorType);

      auto originRepN =
          ceil<unsigned>(resShape[order[1]], originShapePerCTA[order[1]]);
      auto totalElemsPerThreadN = ttg::getElemsPerThread(tensorType)[order[1]];
      if (sizePerThread[order[1]] == tN || originRepN % tN != 0 ||
          totalElemsPerThreadN % tN != 0)
        return;

      // new blocked layout [1, tk] => [tN, tk]
      sizePerThread[order[1]] = tN;

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
        newOther = builder.create<ttg::ConvertLayoutOp>(oldOther.getLoc(),
                                                        newOtherType, oldOther);
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

      return;
    });
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUChangeLayoutFromRepNToElemNPass() {
  return std::make_unique<TritonMETAXGPUChangeLayoutFromRepNToElemNPass>();
}
