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

#ifndef TRITON_TILEIR_TRANSFORMS_PASSES_H_
#define TRITON_TILEIR_TRANSFORMS_PASSES_H_

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include <optional>

namespace mlir {
namespace triton {
std::unique_ptr<Pass>
createRewriteTensorPointerToMemrefPass(int computeCapability,
                                       std::optional<int> numStages);
std::unique_ptr<Pass> createRewriteAssumeWithCudaTilePass();
std::unique_ptr<Pass> createLiftTTCFToSCFPass();
std::unique_ptr<Pass> createAutoGenMemoryTokenPass();
std::unique_ptr<Pass>
createAutoGenMemoryTokenPass(bool enable_autogen_alias_mem_token);

// Generate the pass class declarations (and options structs).
#define GEN_PASS_DECL
#include "Transform/Passes.h.inc"

// Generate the pass registration.
#define GEN_PASS_REGISTRATION
#include "Transform/Passes.h.inc"

} // namespace triton

} // namespace mlir

#endif // TRITON_TILEIR_TRANSFORMS_PASSES_H_
