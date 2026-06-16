/*
 * Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include <limits>

// clang-format off
#include "Dialect/TritonILUVATARGPU/IR/Dialect.h"
#include "Dialect/TritonILUVATARGPU/IR/Dialect.cpp.inc"
// clang-format on

#include "Dialect/TritonILUVATARGPU/Utility/CommonUtils.h"

using namespace mlir;
using namespace mlir::triton::iluvatargpu;

void mlir::triton::iluvatargpu::TritonILUVATARGPUDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonILUVATARGPU/IR/Ops.cpp.inc"
      >();

  addInterfaces<TritonInlinerInterface>();
}

#define GET_OP_CLASSES
#include "Dialect/TritonILUVATARGPU/IR/Ops.cpp.inc"
