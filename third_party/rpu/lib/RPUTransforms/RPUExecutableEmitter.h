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

#ifndef RPU_EXECUTABLE_EMITTER_H
#define RPU_EXECUTABLE_EMITTER_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include <string>

namespace mlir {
namespace rpu {

struct RPUExecutableEmissionResult {
  std::string kernelName;
  std::string sourceKind;
  std::string source;
};

FailureOr<RPUExecutableEmissionResult>
emitRPUDSLFromExecutableModule(ModuleOp module);

} // namespace rpu
} // namespace mlir

#endif // RPU_EXECUTABLE_EMITTER_H
