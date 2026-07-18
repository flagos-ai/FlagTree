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

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_

#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"

namespace mlir {

// Generate the pass class declarations.
#define GEN_PASS_DECL
#include "TritonAMDGPUTransforms/Passes.h.inc"

} // namespace mlir

namespace mlir::triton::amdgpu {

// Generate the pass class declarations.
#define GEN_PASS_DECL_TRITONAMDGPUOPTIMIZEDOTOPERANDS
#include "TritonAMDGPUTransforms/Passes.h.inc"

void registerTritonAMDGPUOptimizeDotOperands();
} // namespace mlir::triton::amdgpu

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonAMDGPUTransforms/Passes.h.inc"
} // namespace mlir

#endif // TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTRANSFORMS_PASSES_H_
