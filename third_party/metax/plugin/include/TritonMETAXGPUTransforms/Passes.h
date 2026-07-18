// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createTritonMETAXGPUAccelerateMatmulPass(
    int numStages = 2, bool disablePrefetch = false, bool storeCoalesce = false,
    int computeCapability = 80);

std::unique_ptr<Pass> createTritonMETAXGPUPipelineMACAPass(
    int numStages = 2, int pipelineLoadNum = -1, bool isFullStage = false,
    bool isSingleShm = false);

std::unique_ptr<Pass> createTritonMETAXGPUPipelineAsyncBasePass(
    int numStages = 2, bool isFullStage = false, bool mixed = false);
std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTNPass(int numStages = 2, int innerStageM = 0,
                                        int innerStageN = 0);
std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTTPass(int numStages = 2);
std::unique_ptr<Pass>
createTritonMETAXGPUAddPtrOptPass(int numStages = 2, bool isFullStage = false,
                                  bool mixed = false);

std::unique_ptr<Pass> createTritonMETAXGPUChangeTransOpGraphPass();

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutFromRepNToElemNPass();

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForConstancyLoadPass();

std::unique_ptr<Pass> createTritonMETAXGPUOptimizeCStorePass(int numStages = 2);

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForInt8Pass(
    int numStages = 2, std::string pipeline = std::string());

std::unique_ptr<Pass>
createTritonMETAXGPUOptimizeSmemUsage(bool forceNoVectorize = false);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMETAXGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
