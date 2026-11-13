
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "mlir/IR/Operation.h"
#include "llvm/ADT/StringRef.h"

using namespace mlir;
using namespace llvm;

void triton::setOpBlockId(mlir::Operation *op, int blockId) {
  static constexpr int kIntegerBitWidth = 32;
  op->setAttr(
      CVPipeline::kBlockId,
      IntegerAttr::get(IntegerType::get(op->getContext(), kIntegerBitWidth),
                       blockId));
}

void triton::setOpCoreType(mlir::Operation *op, llvm::StringRef coreType) {
  op->setAttr(CVPipeline::kCoreType,
              StringAttr::get(op->getContext(), coreType));
}

bool triton::isScalarDependency(mlir::Value depValue) {
  return isa<FloatType, IntegerType>(depValue.getType());
}

bool triton::is1DTensorDependency(mlir::Value depValue) {
  static constexpr int SHAPE_1D_LENGTH = 1;
  auto tensorTy = dyn_cast<TensorType>(depValue.getType());
  if (tensorTy && tensorTy.getRank() == SHAPE_1D_LENGTH) {
    return true;
  }
  return false;
}
