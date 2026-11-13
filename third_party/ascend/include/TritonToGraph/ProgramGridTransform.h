/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H
#define TRITON_TO_GRAPH_PROGRAM_GRID_TRANSFORM_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>

namespace mlir {
namespace triton {
class FuncOp;
namespace cfg {

inline constexpr llvm::StringLiteral kProgramGridTransformsAttr =
    "hacc.program_grid_transforms";
inline constexpr int64_t kProgramGridTransformsVersion = 2;

struct ProgramGridTransform {
  int32_t order = 0;
  int32_t axis = 0;
  int64_t factor = 1;
  bool persistentCoverage = false;
  bool gridStrideAbiVerified = false;
};

struct ProgramGridTransformContract {
  int64_t version = kProgramGridTransformsVersion;
  SmallVector<ProgramGridTransform> transforms;
};

LogicalResult addProgramGridHiddenExtentArguments(triton::FuncOp function);
bool hasProgramGridHiddenExtentArguments(triton::FuncOp function);

LogicalResult commitProgramGridFunctionFromSandbox(triton::FuncOp destination,
                                                   triton::FuncOp source);

FailureOr<ProgramGridTransformContract>
parseProgramGridTransformContract(Attribute attribute);

DictionaryAttr serializeProgramGridTransformContract(
    MLIRContext *context, const ProgramGridTransformContract &contract);

LogicalResult
setProgramGridTransformContract(ModuleOp module,
                                const ProgramGridTransformContract &contract);

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif
