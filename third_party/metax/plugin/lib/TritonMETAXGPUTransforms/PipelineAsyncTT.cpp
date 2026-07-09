#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/MapVector.h"
#ifdef USE_MACA
#include "TritonMETAXGPUTransforms/MACACommon.h"
#include "TritonMETAXGPUTransforms/Passes.h"
#endif

using llvm::MapVector;
using namespace mlir;
using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::MACAMmaEncodingAttr;
namespace ttg = triton::gpu;
namespace tt = triton;

#define int_attr(num) builder.getI64IntegerAttr(num)

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct TritonMETAXGPUPipelineAsyncTTPass
    : public TritonMETAXGPUPipelineAsyncTTBase<
          TritonMETAXGPUPipelineAsyncTTPass> {
  TritonMETAXGPUPipelineAsyncTTPass() = default;
  TritonMETAXGPUPipelineAsyncTTPass(int numStages) {
    this->numStages = numStages;
  }

  void runOnOperation() override { return; }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncTTPass(int numStages) {
  return std::make_unique<TritonMETAXGPUPipelineAsyncTTPass>(numStages);
}
