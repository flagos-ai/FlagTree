
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

#define GEN_PASS_CLASSES
#include "TritonMETAXGPUTransforms/Passes.h.inc"
struct TritonMETAXGPUAddPtrOptPass
    : public TritonMETAXGPUAddPtrOptPassBase<TritonMETAXGPUAddPtrOptPass> {
  TritonMETAXGPUAddPtrOptPass() = default;
  TritonMETAXGPUAddPtrOptPass(int numStages, bool isFullStage, bool mixed) {
    this->numStages = numStages;
    this->isFullStage = isFullStage;
    this->mixed = mixed;
  }
  void runOnOperation() override { return; }
};

std::unique_ptr<Pass> mlir::createTritonMETAXGPUAddPtrOptPass(int numStages,
                                                              bool isFullStage,
                                                              bool mixed) {
  return std::make_unique<TritonMETAXGPUAddPtrOptPass>(numStages, isFullStage,
                                                       mixed);
}
