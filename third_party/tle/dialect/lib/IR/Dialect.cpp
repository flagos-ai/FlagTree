/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
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

#include "tle/dialect/include/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "tle/dialect/include/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "tle/dialect/include/IR/Ops.cpp.inc"

#ifdef FLAGCX_ENABLED
#define GET_OP_CLASSES
#include "tle/dialect/include/IR/FlagCxOps.cpp.inc"
#endif

namespace mlir::triton::tle {
void TleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tle/dialect/include/IR/Ops.cpp.inc"
      >();

#ifdef FLAGCX_ENABLED
  addOperations<
#define GET_OP_LIST
#include "tle/dialect/include/IR/FlagCxOps.cpp.inc"
      >();
#endif
}
} // namespace mlir::triton::tle
