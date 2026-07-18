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

#ifndef TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H
#define TRITON_CONVERSION_NVGPU_TO_LLVM_PASS_H

#include <string>
#include <utility>
#include <vector>

#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

namespace nvgpu {

using Constraints = std::vector<std::string>;
using OperandsAndConstraints = std::vector<std::pair<Value, std::string>>;

LogicalResult
rewriteAsPtxAsm(mlir::Operation *op, mlir::PatternRewriter &rewriter,
                std::string ptxAsm,
                const OperandsAndConstraints &operandsAndConstraints = {},
                const Constraints &outputConstraints = {});

} // namespace nvgpu

} // namespace triton

} // namespace mlir

#endif
