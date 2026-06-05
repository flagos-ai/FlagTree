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
namespace ttg = triton::gpu;
namespace tt = mlir::triton;

#define int_attr(num) builder.getI64IntegerAttr(num)

namespace {

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"

struct TritonMETAXGPUPipelineAsyncBasePass
    : public TritonMETAXGPUPipelineAsyncNTBase<
          TritonMETAXGPUPipelineAsyncBasePass> {
  TritonMETAXGPUPipelineAsyncBasePass() = default;
  TritonMETAXGPUPipelineAsyncBasePass(int numStages, bool isFullStage,
                                      bool mixed) {
    this->numStages = numStages;
    this->isFullStage = isFullStage;
    this->mixed = mixed;
  }

  void runOnOperation() override { return; }
};
} // anonymous namespace

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncBasePass(int numStages, bool isFullStage,
                                                bool mixed) {
  return std::make_unique<TritonMETAXGPUPipelineAsyncBasePass>(
      numStages, isFullStage, mixed);
}
