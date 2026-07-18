#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Tools/GenericSwizzling.h"
#include <memory>

using namespace mlir;
namespace ttg = triton::gpu;
namespace tt = triton;
#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

class TritonMETAXGPUOptimizeSmemUsage
    : public TritonMETAXGPUOptimizeSmemUsageBase<
          TritonMETAXGPUOptimizeSmemUsage> {
public:
  TritonMETAXGPUOptimizeSmemUsage() = default;
  TritonMETAXGPUOptimizeSmemUsage(bool reduce_smem_usage) {
    this->forceNoVectorize = reduce_smem_usage;
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();
    if (!this->forceNoVectorize) {
      return;
    }
    m->walk([&](ttg::ConvertLayoutOp op) -> void {
      RankedTensorType srcTy = op.getSrc().getType();
      RankedTensorType dstTy = op.getType();
      Attribute srcLayout = srcTy.getEncoding();
      Attribute dstLayout = dstTy.getEncoding();
      // triton3.6 convertLayout will use all dims including rep dims to
      // maximaximize lds/srs vec size, wich may result in smem size increasing
      // and reducing parallsm, this scenerio set vec size to 1 forcely to
      // prevent this condition. condition: 64x256 f32 mma->block condition in
      // fused moe case
      // TODO(MACA): support calculating cvtLayout smem usage and fallback to
      // low smem size cost algorithm

      if (isa<ttg::MACAMmaEncodingAttr>(srcLayout) &&
          isa<BlockedEncodingAttr>(dstLayout)) {
        OpBuilder builder(op->getContext());
        bool hasAttr = op->hasAttr(tt::AttrSharedMemForceNoVec);
        if (!hasAttr) {
          op->setAttr(tt::AttrSharedMemForceNoVec, builder.getUnitAttr());
        }
      }

      return;
    });
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUOptimizeSmemUsage(bool reduce_smem_usage) {
  return std::make_unique<TritonMETAXGPUOptimizeSmemUsage>(reduce_smem_usage);
}
