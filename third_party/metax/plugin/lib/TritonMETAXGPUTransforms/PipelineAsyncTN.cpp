// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
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

struct TritonMETAXGPUPipelineAsyncTNPass
    : public TritonMETAXGPUPipelineAsyncTNBase<
          TritonMETAXGPUPipelineAsyncTNPass> {
  TritonMETAXGPUPipelineAsyncTNPass() = default;
  TritonMETAXGPUPipelineAsyncTNPass(int numStages, int innerStageM,
                                    int innerStageN) {
    this->numStages = numStages;
    this->innerStageM = innerStageM;
    this->innerStageN = innerStageN;
  }

  void runOnOperation() override { return; }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUPipelineAsyncTNPass(int numStages, int innerStageM,
                                              int innerStageN) {
  return std::make_unique<TritonMETAXGPUPipelineAsyncTNPass>(
      numStages, innerStageM, innerStageN);
}
