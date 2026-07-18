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

#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.cpp.inc"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/AttrDefs.cpp.inc"

using namespace mlir;

const int mlir::triton::proton::gpu::getBytesPerClockEntry() { return 8; }
const int mlir::triton::proton::gpu::getCircularHeaderSize() { return 40; }

void mlir::triton::proton::gpu::ProtonGPUDialect::initialize() {
  registerTypes();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/ProtonGPU/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "Dialect/ProtonGPU/IR/Ops.cpp.inc"
      >();
}

const int mlir::triton::proton::gpu::getTotalNumWarps(ModuleOp mod) {
  int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
  if (auto totalNumWarps =
          mod->getAttrOfType<IntegerAttr>("ttg.total-num-warps"))
    numWarps = totalNumWarps.getInt();
  return numWarps;
}
