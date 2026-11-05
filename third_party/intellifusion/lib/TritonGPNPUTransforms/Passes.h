/*
 * Copyright (c) 2023 NVIDIA Corporation & Affiliates. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef TRITON_GPNPU_TRANSFORMS_PASSES_H
#define TRITON_GPNPU_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir::triton::Intellifusion {

std::unique_ptr<Pass> createGPNPUPlanCTAPass();
std::unique_ptr<Pass> createGPNPUAccelerateMatmulPass();
std::unique_ptr<Pass> createGPNPUOptimizeDescriptorEncodingPass();
std::unique_ptr<Pass> createGPNPUPromoteLHSToTMemPass();
std::unique_ptr<Pass> createGPNPURemoveTMEMTokensPass();
std::unique_ptr<Pass> createGPNPUOptimizeTMemLayoutsPass();
std::unique_ptr<Pass> createGPNPUFuseTMEMLoadReducePass();
std::unique_ptr<Pass> createGPNPUTMALoweringPass();
std::unique_ptr<Pass> createGPNPUInterleaveTMemPass();
std::unique_ptr<Pass> createGPNPUFenceInsertionPass();
std::unique_ptr<Pass> createGPNPUMMALoweringPass();
std::unique_ptr<Pass> createGPNPUTensorMemoryAllocationPass();
std::unique_ptr<Pass> createGPNPUProxyFenceInsertionPass();
std::unique_ptr<Pass> createGPNPUCheckMatmulTwoCTAPass();
std::unique_ptr<Pass> createGPNPUTMemBarrierInsertionPass();

// Select and prepare the GPNPU dot lowering strategy before the generic matmul
// acceleration pass. The pass contains the MMAv5 pad/dot/crop transform and
// currently selects between existing FMA and directly legal MMAv5 paths.
std::unique_ptr<Pass> createGPNPUDotStrategyPass(bool forcePadding = false,
                                                 int numWarps = 4,
                                                 int numCTAs = 1);

// Restore the shared-memory staging required by GPNPU's NVFP4 scale-vector
// MMAv5 instruction before shared-memory allocation assigns offsets.
std::unique_ptr<Pass> createGPNPUStageScaledFp4Pass();

// Register GPNPU-owned transforms for textual --pass-pipeline parsing.
void registerGPNPUTransformPasses();

} // namespace mlir::triton::Intellifusion

#endif // TRITON_GPNPU_TRANSFORMS_PASSES_H
