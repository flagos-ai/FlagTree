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

#include "mctle/dialect/include/IR/Dialect.h"
#include "mctle/dialect/include/IR/Dialect.cpp.inc"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mctle/dialect/include/IR/Ops.cpp.inc"

namespace mlir::triton::mctle {
void McTleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mctle/dialect/include/IR/Ops.cpp.inc"
      >();
}
} // namespace mlir::triton::mctle
