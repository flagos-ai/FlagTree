/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include "npu/Dialect/TritonAscend/IR/TritonAscendDialect.h"

namespace mlir {
namespace triton {

//-- MakeTensorDescOp --
void MakeTensorDescOp::build(OpBuilder &builder, OperationState &state,
                             Value base, ValueRange shape, ValueRange strides,
                             ArrayRef<int32_t> blockShape,
                             bool isSignedInteger) {
  auto ptrTy = dyn_cast<triton::PointerType>(base.getType());
  if (!ptrTy) {
    llvm::report_fatal_error("Expected pointer type");
  }
  auto elemTy = ptrTy.getPointeeType();
  SmallVector<int64_t> blockShape64(blockShape);
  auto blockTy = RankedTensorType::get(blockShape64, elemTy);
  auto descTy =
      TensorDescType::get(builder.getContext(), blockTy, isSignedInteger);
  return build(builder, state, descTy, base, shape, strides);
}

// -- DescriptorLoadOp --
static LogicalResult verifyDescriptorLoadStoreType(Operation *op,
                                                   TensorDescType desc,
                                                   RankedTensorType tensor) {
  RankedTensorType block = desc.getSignlessBlockType();
  ArrayRef<int64_t> blockShape = block.getShape();
  ArrayRef<int64_t> tensorShape = tensor.getShape();
  if (blockShape.size() > tensorShape.size()) {
    // Allow ranked reduced load if the leading dimensions are all 1s.
    for (int i = 0; i < blockShape.size() - tensorShape.size(); ++i) {
      if (blockShape[i] != 1)
        return op->emitOpError(
            "ranked reduce load only allowed for unit dimension leading dim.");
    }
    blockShape = blockShape.take_back(tensorShape.size());
  }

  if (blockShape == tensorShape &&
      block.getElementType() == tensor.getElementType()) {
    return success();
  }
  return op->emitOpError("tensor descriptor block and tensor types must match");
}

LogicalResult DescriptorLoadOp::verify() {
  return verifyDescriptorLoadStoreType(*this, getDesc().getType(), getType());
}

// -- DescriptorStoreOp --
LogicalResult DescriptorStoreOp::verify() {
  return verifyDescriptorLoadStoreType(*this, getDesc().getType(),
                                       getSrc().getType());
}

// -- GatherOp --
LogicalResult GatherOp::verify() {
  RankedTensorType indicesTy = getIndices().getType();
  RankedTensorType srcTy = getSrc().getType();
  RankedTensorType resTy = getResult().getType();

  if (indicesTy.getShape() != resTy.getShape()) {
    return emitOpError("indices and output shapes must match");
  }
  if (indicesTy.getEncoding() != resTy.getEncoding()) {
    return emitOpError("indices and output encodings must match");
  }
  if (srcTy.getElementType() != resTy.getElementType()) {
    return emitOpError("input and output element types must match");
  }
  if (srcTy.getRank() != indicesTy.getRank()) {
    return emitOpError("input and indices ranks must match");
  }
  if (getAxis() >= srcTy.getRank()) {
    return emitOpError("gather dimension must be less than the input rank");
  }
  for (int dim = 0; dim < indicesTy.getRank(); ++dim) {
    if (dim == getAxis())
      continue;
    if (indicesTy.getShape()[dim] != srcTy.getShape()[dim]) {
      return emitOpError("indices dimension ")
             << dim << " must match the corresponding input dimension";
    }
  }

  return success();
}

LogicalResult GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  GatherOpAdaptor adaptor(operands, attributes, properties, regions);
  auto indicesType = cast<RankedTensorType>(adaptor.getIndices().getType());
  auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());

  // Shape and encoding of the indices with the element type of the src.
  inferredReturnTypes.push_back(
      RankedTensorType::get(indicesType.getShape(), srcType.getElementType(),
                            indicesType.getEncoding()));
  return success();
}

} // namespace triton
} // namespace mlir
