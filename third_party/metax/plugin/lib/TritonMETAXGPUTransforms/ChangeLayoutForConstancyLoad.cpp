
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include <memory>
using namespace mlir;
using triton::DotOp;
using triton::gpu::BlockedEncodingAttr;
using triton::gpu::LocalAllocOp;
using triton::gpu::MACAMmaEncodingAttr;
using triton::gpu::MemDescType;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"
/**
 * step 1. For every convertLayoutOp ()->(dotOp) inside ForOp, collect all
 * loadOps in its parentOps step 2. For every loadOps, check its constancy and
 * contiguity along continuous dimension (constancy>1 and contiguity==1), call
 * calSizePerThreadWithConstancy step 3. Find pattern like divsiOp, update
 * contiguityInterConstGroup and calculate SizePerThread step 4. For all loadOps
 * belongs to same cvtOp, get max SizePerThread among all loadOps with same
 * shape and order step 5. Create new loadOps with updated blocked encoding
 */
class TritonMETAXGPUChangeLayoutForConstancyLoadPass
    : public TritonMETAXGPUChangeLayoutForConstancyLoadBase<
          TritonMETAXGPUChangeLayoutForConstancyLoadPass> {
private:
  void replaceLoadWithLayout(tt::LoadOp targetLoad,
                             ttg::BlockedEncodingAttr newBlockedLayout) {
    // create new LoadOp with new BlockedLayout, and replace old loadOp
    auto tensorType = cast<RankedTensorType>(targetLoad.getResult().getType());
    auto resShape = tensorType.getShape();
    BlockedEncodingAttr originBlockedLayout =
        cast<BlockedEncodingAttr>(tensorType.getEncoding());
    OpBuilder builder(targetLoad);
    builder.setInsertionPointAfter(targetLoad);
    Value oldPtr = targetLoad.getPtr();
    auto oldPtrTy = cast<RankedTensorType>(oldPtr.getType());
    auto newPtrType = RankedTensorType::get(
        oldPtrTy.getShape(), oldPtrTy.getElementType(), newBlockedLayout);
    auto newPtr = builder.create<ttg::ConvertLayoutOp>(oldPtr.getLoc(),
                                                       newPtrType, oldPtr);
    auto newResType = RankedTensorType::get(
        resShape, tensorType.getElementType(), newBlockedLayout);
    // TODO: mask and other
    auto newLoadOp = builder.create<triton::LoadOp>(
        targetLoad.getLoc(), newResType, newPtr, Value(), Value(),
        targetLoad.getBoundaryCheckAttr(), targetLoad.getPaddingAttr(),
        targetLoad.getCache(), targetLoad.getEvict(),
        targetLoad.getIsVolatile(), targetLoad.getContiguityInterConstGroup());
    auto newLoadCvtRes = builder.create<ttg::ConvertLayoutOp>(
        oldPtr.getLoc(), tensorType, newLoadOp);
    targetLoad.getResult().replaceAllUsesWith(newLoadCvtRes.getResult());
  }
  /**
   *   getDivcontiguityInterConstGroup supports pattern:
   *
   *   %1 = divsi %0 %cst
   *	          \
   *   %2 = addptr %arg %1
   *               \
   *       %3 = broadcast %2     %4 = broadcast(don't care order!=0)
   *                 \              /
   *                  \            /
   *                  %5 = addptr %3 %4
   *                        |
   *                      load %5
   *
   *   and:
   *  	                                      %1 = divsi %0 %cst
   *                                              /
   *   %2 = braodcast(don't care order!=0)   %3 = broadcast %1
   *                           \             /
   *                            \           /
   *                           %4 = addptr %2 %3
   *                                 |
   *                            %5 = load %4
   */
  int getDivContiguityInterConstGroup(
      Value v, int dim, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    // BlockArgument
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      Block *block = blockArg.getOwner();
      Operation *parentOp = block->getParentOp();
      if (auto forOp = dyn_cast<scf::ForOp>(*parentOp)) {
        int argIndex = blockArg.getArgNumber() - forOp.getNumInductionVars();
        auto init_oprand = forOp.getInitArgs()[argIndex];
        int res =
            getDivContiguityInterConstGroup(init_oprand, dim, axisInfoAnalysis);
        return res;
      }
    }
    auto op = v.getDefiningOp();
    if (auto broadcastOp = dyn_cast<tt::BroadcastOp>(op)) {
      auto srcTy = cast<RankedTensorType>(broadcastOp.getSrc().getType());
      if (!srcTy)
        return -1;
      auto srcOrder = ttg::getOrder(srcTy);
      // if broadcast along dim, meaning constant value along dim, so
      // contiguityInterConstGroup = 1 for example, dim=1, %3 = broadcast %1
      // (64x1 -> 64x64)
      if (srcOrder[dim] != 0 || srcTy.getShape()[dim] == 1) {
        return 1;
      }
      // else call getDivcontiguityInterConstGroup for operand
      return getDivContiguityInterConstGroup(broadcastOp.getSrc(), dim,
                                             axisInfoAnalysis);
    } else if (dyn_cast<ttg::ConvertLayoutOp>(op) ||
               dyn_cast<ttg::LocalAllocOp>(op) ||
               dyn_cast<ttg::LocalLoadOp>(op)) {
      return 1;
    } else if (auto addPtr = dyn_cast<tt::AddPtrOp>(op)) {
      Value ptr = addPtr.getPtr();
      Value offset = addPtr.getOffset();
      auto shape = cast<RankedTensorType>(ptr.getType()).getShape();
      int rank = shape.size();
      // TODO: Support expand_dims pattern, which addPtr rank==1
      if (rank != 2)
        return 1;
      tt::AxisInfo &ptrAxisInfo = *(axisInfoAnalysis.getAxisInfo(ptr));
      int ptrContiguity =
          getDivContiguityInterConstGroup(ptr, dim, axisInfoAnalysis);
      int ptrConstancy = ptrAxisInfo.getConstancy()[dim];
      tt::AxisInfo &offsetAxisInfo = *(axisInfoAnalysis.getAxisInfo(offset));
      int offsetContiguity =
          getDivContiguityInterConstGroup(offset, dim, axisInfoAnalysis);
      int offsetConstancy = offsetAxisInfo.getConstancy()[dim];
      if (ptrConstancy == shape[dim] && offsetContiguity > 1) {
        // For dimension dim, same ptr addr along dim, offset has
        // contiguityInterConstGroup
        return offsetContiguity;
      } else if (offsetConstancy == shape[dim] && ptrContiguity > 1) {
        // For dimension dim, same offset along dim, ptr addr has
        // contiguityInterConstGroup
        return ptrContiguity;
      } else {
        return 1;
      }
    } else if (auto divOp = dyn_cast<arith::DivSIOp>(op)) {
      // cal contiguityInterConstGroup
      // %1 = divsi %0 %cst
      // contiguityInterConstGroup_of_%1 = contiguity_of_%0 / cst_value
      Value lhs = divOp.getLhs();
      Value rhs = divOp.getRhs();
      if (auto constOp = dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
        tt::AxisInfo &constAxisInfo = *(axisInfoAnalysis.getAxisInfo(rhs));
        int constVal = constAxisInfo.getConstantValue().value();
        tt::AxisInfo &dividedAxisInfo = *(axisInfoAnalysis.getAxisInfo(lhs));
        return dividedAxisInfo.getContiguity()[dim] / constVal;
      } else {
        return 1;
      }
    }
    return 1;
  }
  int calSizePerThreadWithConstancy(
      tt::LoadOp loadOp, tt::ModuleAxisInfoAnalysis &axisInfoAnalysis) {
    Value ptr = loadOp.getPtr();
    auto ptrTy = cast<RankedTensorType>(ptr.getType());
    auto layout = dyn_cast<BlockedEncodingAttr>(ptrTy.getEncoding());
    auto order = layout.getOrder();
    auto shapePerCTA = triton::gpu::getShapePerCTA(ptrTy);
    int totalNumElems = product<int64_t>(shapePerCTA);
    int numWarps = product<unsigned>(layout.getWarpsPerCTA());
    int threadsPerWarp = product<unsigned>(layout.getThreadsPerWarp());
    int totalNumThreads = numWarps * threadsPerWarp;
    // get constancy
    tt::AxisInfo &valInfo = *(axisInfoAnalysis.getAxisInfo(ptr));
    auto constancy = valInfo.getConstancy(order[0]);
    auto contiguity = valInfo.getContiguity();
    assert(valInfo.getRank() == ptrTy.getRank() &&
           "axisInfor rank must equal to shape rank");
    // get contiguityInterConstGroup and set attr to loadOp
    SmallVector<int64_t> contiguityInterConstGroup = contiguity;
    // find pattern
    contiguityInterConstGroup[order[0]] =
        getDivContiguityInterConstGroup(ptr, order[0], axisInfoAnalysis);
    loadOp.setContiguityInterConstGroupAttr(mlir::DenseI64ArrayAttr::get(
        ptrTy.getContext(), contiguityInterConstGroup));
    auto elemBitsWidth = getElementBitWidth(ptrTy);
    int maxElemNumPerThread =
        std::min<int>(totalNumElems / totalNumThreads, shapePerCTA[order[0]]);
    // loadPerThread is the max elemNum can be loaded in one ldgbxx
    int loadPerThread =
        std::min<int>(128 / elemBitsWidth, contiguityInterConstGroup[order[0]]);
    // sizePerThread is the max elemNum handled by one thread
    int sizePerThread =
        std::min<int>(loadPerThread * constancy, maxElemNumPerThread);
    return sizePerThread;
  }

public:
  TritonMETAXGPUChangeLayoutForConstancyLoadPass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    tt::ModuleAxisInfoAnalysis axisInfoAnalysis(m);
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      OpBuilder builder(forOp);
      int loadNum = 0;
      // tt::LoadOp targetLoad;
      llvm::MapVector<Value, SetVector<Operation *>> dotOpToLoadsMap;
      // collect LoadOp inside ForOp, from convertLayout ()->(dotOp)
      for (Operation &op : forOp) {
        if (auto cvtOp = dyn_cast<ttg::ConvertLayoutOp>(&op)) {
          if (auto resTy =
                  cast<RankedTensorType>(cvtOp.getResult().getType())) {
            if (dyn_cast<ttg::DotOperandEncodingAttr>(resTy.getEncoding())) {
              SetVector<Operation *> backwardSlice;
              BackwardSliceOptions opt;
              opt.omitBlockArguments = true;
              auto backwardFilter = [&forOp](Operation *candidateOp) {
                return forOp.getRegion().isAncestor(
                    candidateOp->getParentRegion());
              };
              opt.filter = {backwardFilter};
              (void)getBackwardSlice(&op, &backwardSlice, opt);
              dotOpToLoadsMap[cvtOp.getResult()] = {};
              for (auto bwdop : backwardSlice) {
                if (auto loadop = dyn_cast<tt::LoadOp>(*bwdop)) {
                  // TODO: support mask and other
                  if (loadop.getMask() || loadop.getOther())
                    continue;
                  if (!dyn_cast<RankedTensorType>(loadop.getPtr().getType()))
                    continue;
                  dotOpToLoadsMap[cvtOp.getResult()].insert(loadop);
                }
              }
            }
          }
        }
      }
      // traverse (dotOp, Set(loadOp))
      // if contiguity == 1 and constancy > 1
      //      calSizePerThread(loadOp)
      // traverse (dotOp, Set(loadOp))
      //      getMaxSizePerThread among same shape and order loadOp
      //      set newBlockedLayout
      // calSizePerThread(loadOp)
      // -> get contiguityInterConstGroup (find divsiOp, default==contiguity)
      // -> cal sizePerThread
      llvm::MapVector<ArrayRef<int64_t>,
                      llvm::MapVector<ArrayRef<unsigned>, int>>
          shapeToSizePerThreadMap;
      SmallVector<tt::LoadOp> constancyLoads;
      for (auto &[_, loadsVec] : dotOpToLoadsMap) {
        shapeToSizePerThreadMap.clear();
        constancyLoads.clear();
        for (auto op : loadsVec) {
          auto loadOp = dyn_cast<tt::LoadOp>(*op);
          Value ptr = loadOp.getPtr();
          auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
          if (!ptrTy)
            continue;
          auto layout = dyn_cast<BlockedEncodingAttr>(ptrTy.getEncoding());
          if (!layout)
            continue;
          auto order = layout.getOrder();
          auto shape = ptrTy.getShape();
          tt::AxisInfo &valInfo = *axisInfoAnalysis.getAxisInfo(ptr);
          auto constancy = valInfo.getConstancy(order[0]);
          int bestPerThread = layout.getSizePerThread()[order[0]];
          if (constancy > 1 && valInfo.getContiguity(order[0]) == 1) {
            bestPerThread =
                calSizePerThreadWithConstancy(loadOp, axisInfoAnalysis);
            // int bestPerThread = layout.getSizePerThread()[order[0]];
            constancyLoads.push_back(loadOp);
          }
          if (!shapeToSizePerThreadMap.contains(shape) ||
              !(shapeToSizePerThreadMap[shape].contains(order))) {
            std::pair<ArrayRef<unsigned>, int> orderToPerThread =
                std::make_pair(order, bestPerThread);
            shapeToSizePerThreadMap[shape].insert(orderToPerThread);
          } else {
            shapeToSizePerThreadMap[shape][order] = std::max<int>(
                bestPerThread, shapeToSizePerThreadMap[shape][order]);
          }
        }
        // no loads layout to be changed
        if (constancyLoads.empty())
          continue;
        // According to shape and order, get max sizePerThread globally,
        // so every loadOp with same shape and order has same sizePerThread
        // and change blocked layout
        for (auto op : loadsVec) {
          auto loadOp = dyn_cast<tt::LoadOp>(*op);
          Value ptr = loadOp.getPtr();
          auto ptrTy = dyn_cast<RankedTensorType>(ptr.getType());
          if (!ptrTy)
            continue;
          auto layout = dyn_cast<BlockedEncodingAttr>(ptrTy.getEncoding());
          if (!layout)
            continue;
          auto order = layout.getOrder();
          auto shape = ptrTy.getShape();
          // shapeAndOrder curShapeAndOrder(shape, order);
          SmallVector<unsigned> sizePerThread(ptrTy.getRank(), 1);
          sizePerThread[order[0]] = shapeToSizePerThreadMap[shape][order];
          int numWarps = product<unsigned>(layout.getWarpsPerCTA());
          int threadsPerWarp = product<unsigned>(layout.getThreadsPerWarp());
          auto newBlockedLayout = triton::gpu::BlockedEncodingAttr::get(
              ptrTy.getContext(), shape, sizePerThread, order, numWarps,
              threadsPerWarp, layout.getCTALayout());
          replaceLoadWithLayout(loadOp, newBlockedLayout);
        }
      }
      return;
    });
  }
};
std::unique_ptr<Pass>
mlir::createTritonMETAXGPUChangeLayoutForConstancyLoadPass() {
  return std::make_unique<TritonMETAXGPUChangeLayoutForConstancyLoadPass>();
}
