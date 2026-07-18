
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include <memory>
using namespace mlir;
using triton::DotOp;
using triton::gpu::LocalAllocOp;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"
class TritonMETAXGPUChangeTransOpGraphPass
    : public TritonMETAXGPUChangeTransOpGraphBase<
          TritonMETAXGPUChangeTransOpGraphPass> {
public:
  TritonMETAXGPUChangeTransOpGraphPass() = default;
  void runOnOperation() override {
    ModuleOp m = getOperation();
    getOperation()->walk([&](scf::ForOp forOp) -> void {
      OpBuilder builder(forOp);
      SmallVector<tt::DotOp> dots;
      SmallVector<tt::LoadOp> loads;
      triton::TransOp targetLoadA;
      triton::TransOp targetLoadB;
      for (Operation &op : forOp) {
        if (auto loadOp = dyn_cast<tt::LoadOp>(&op)) {
          loads.push_back(loadOp);
        } else if (auto dotOp = dyn_cast<tt::DotOp>(&op)) {
          dots.push_back(dotOp);
        }
      }
      if (dots.size() != 1)
        return;
      auto dotAOp = dots[0].getA();
      auto dotBOp = dots[0].getB();
      if (auto transOp =
              llvm::dyn_cast_or_null<triton::TransOp>(dotAOp.getDefiningOp())) {
        if (auto cvtOp = llvm::dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
                transOp.getSrc().getDefiningOp())) {
          if (auto loadOp = llvm::dyn_cast_or_null<triton::LoadOp>(
                  cvtOp.getSrc().getDefiningOp())) {
            targetLoadA = transOp;
          }
        }
      }
      if (auto transOp =
              llvm::dyn_cast_or_null<triton::TransOp>(dotBOp.getDefiningOp())) {
        if (auto cvtOp = llvm::dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
                transOp.getSrc().getDefiningOp())) {
          if (auto loadOp = llvm::dyn_cast_or_null<triton::LoadOp>(
                  cvtOp.getSrc().getDefiningOp())) {
            targetLoadB = transOp;
          }
        }
      }
      if (!targetLoadA && !targetLoadB)
        return;
      // TODO(wyx):When both dependencies of A and B contains transOp,
      // the accuracy is incorrect after ChangeTransGraphPass and need to check.
      if (targetLoadA && targetLoadB &&
          std::getenv("TRITON_ENABLE_TRANS_DOTAb") == nullptr)
        return;
      if (targetLoadA) {
        builder.setInsertionPointAfter(targetLoadA);
        if (auto transOp = llvm::dyn_cast_or_null<triton::TransOp>(
                dotAOp.getDefiningOp())) {
          auto dstTy = dyn_cast_or_null<triton::gpu::TensorOrMemDesc>(
              transOp.getSrc().getType());
          SmallVector<int64_t> bufferShape(dstTy.getShape().begin(),
                                           dstTy.getShape().end());
          auto transDstTy =
              cast<RankedTensorType>(transOp.getResult().getType());
          auto dotEnc = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
              transDstTy.getEncoding());
          if (auto cvtOp = llvm::dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
                  transOp.getSrc().getDefiningOp())) {
            if (auto loadOp = llvm::dyn_cast_or_null<triton::LoadOp>(
                    cvtOp.getSrc().getDefiningOp())) {
              MLIRContext *ctx = loadOp.getContext();
              auto ty = cast<RankedTensorType>(loadOp.getType());
              auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
              auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
                  ty.getContext(), dotEnc, dstTy.getShape(),
                  ttg::getOrder(dstTy), ttg::getCTALayout(dstTy.getEncoding()),
                  dstTy.getElementType(), /*needTrans*/ true);
              auto createBufferType = ttg::MemDescType::get(
                  bufferShape, dstTy.getElementType(), sharedEnc,
                  sharedMemorySpace, /*mutable*/ true);
              auto localAllocOp = builder.create<ttg::LocalAllocOp>(
                  targetLoadA.getLoc(), createBufferType, loadOp.getResult());
              Value newTrans =
                  builder.create<mlir::triton::gpu::MemDescTransOp>(
                      targetLoadA.getLoc(), localAllocOp, transOp.getOrder());
              auto localLoadOp = builder.create<ttg::LocalLoadOp>(
                  targetLoadA.getLoc(), transDstTy, newTrans);
              transOp.getResult().replaceAllUsesWith(localLoadOp.getResult());
            }
          }
        }
      }
      if (targetLoadB) {
        builder.setInsertionPointAfter(targetLoadB);
        if (auto transOp = llvm::dyn_cast_or_null<triton::TransOp>(
                dotBOp.getDefiningOp())) {
          auto dstTy = dyn_cast_or_null<triton::gpu::TensorOrMemDesc>(
              transOp.getSrc().getType());
          SmallVector<int64_t> bufferShape(dstTy.getShape().begin(),
                                           dstTy.getShape().end());
          auto transDstTy =
              cast<RankedTensorType>(transOp.getResult().getType());
          auto dotEnc = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
              transDstTy.getEncoding());
          if (auto cvtOp = llvm::dyn_cast_or_null<triton::gpu::ConvertLayoutOp>(
                  transOp.getSrc().getDefiningOp())) {
            if (auto loadOp = llvm::dyn_cast_or_null<triton::LoadOp>(
                    cvtOp.getSrc().getDefiningOp())) {
              MLIRContext *ctx = loadOp.getContext();
              auto ty = cast<RankedTensorType>(loadOp.getType());
              auto sharedMemorySpace = ttg::SharedMemorySpaceAttr::get(ctx);
              auto sharedEnc = ttg::SwizzledSharedEncodingAttr::get(
                  ty.getContext(), dotEnc, dstTy.getShape(),
                  ttg::getOrder(dstTy), ttg::getCTALayout(dstTy.getEncoding()),
                  dstTy.getElementType(), /*needTrans*/ true);
              auto createBufferType = ttg::MemDescType::get(
                  bufferShape, dstTy.getElementType(), sharedEnc,
                  sharedMemorySpace, /*mutable*/ true);
              auto localAllocOp = builder.create<ttg::LocalAllocOp>(
                  targetLoadB.getLoc(), createBufferType, loadOp.getResult());
              Value newTrans =
                  builder.create<mlir::triton::gpu::MemDescTransOp>(
                      targetLoadB.getLoc(), localAllocOp, transOp.getOrder());
              auto localLoadOp = builder.create<ttg::LocalLoadOp>(
                  targetLoadB.getLoc(), transDstTy, newTrans, nullptr, true);
              transOp.getResult().replaceAllUsesWith(localLoadOp.getResult());
            }
          }
        }
      }
      return;
    });
  }
};
std::unique_ptr<Pass> mlir::createTritonMETAXGPUChangeTransOpGraphPass() {
  return std::make_unique<TritonMETAXGPUChangeTransOpGraphPass>();
}
