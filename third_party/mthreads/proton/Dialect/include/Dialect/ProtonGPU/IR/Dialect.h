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

#ifndef DIALECT_PROTONGPU_IR_DIALECT_H_
#define DIALECT_PROTONGPU_IR_DIALECT_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h.inc"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Types.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#define GET_OP_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Ops.h.inc"

#define GET_ATTRDEF_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.h.inc"

namespace mlir {
namespace triton {
namespace proton {
namespace gpu {

const int getBytesPerClockEntry();

const int getCircularHeaderSize();

const int getTotalNumWarps(ModuleOp mod);

} // namespace gpu
} // namespace proton
} // namespace triton
} // namespace mlir

#endif // DIALECT_PROTONGPU_IR_DIALECT_H_
