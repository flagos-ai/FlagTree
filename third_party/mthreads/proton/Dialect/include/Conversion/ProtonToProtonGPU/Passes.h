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

#ifndef PROTON_TO_PROTONGPU_PASSES_H
#define PROTON_TO_PROTONGPU_PASSES_H

#include "mlir/Pass/Pass.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"

namespace mlir::triton::proton {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertProtonToProtonGPUPass(
    MetricType metricType = MetricType::CYCLE,
    SamplingStrategy samplingStrategy = SamplingStrategy::NONE,
    llvm::StringRef samplingOptions = "",
    gpu::Granularity granularity = gpu::Granularity::WARP,
    gpu::BufferStrategy bufferStrategy = gpu::BufferStrategy::CIRCULAR,
    gpu::BufferType bufferType = gpu::BufferType::SHARED,
    int32_t bufferSize = 0, int32_t maxSharedMemSize = 32768,
    int64_t profileScratchSize = 32768, int32_t profileScratchAlignment = 128,
    bool clkExt = false);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h.inc"

} // namespace mlir::triton::proton

#endif // PROTON_TO_PROTONGPU_PASSES_H
