//===- TritonAscendOps.cpp - TritonAscend dialect operations --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>


using namespace mlir;
using namespace mlir::triton;


namespace mlir::triton::ascend {

void GatherOutToUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>> &effects)
{
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

void IndirectLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

//-- IndexSelectSimdOp --
LogicalResult IndexSelectSimdOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  
  // Get operands using adaptor
  IndexSelectSimdOpAdaptor adaptor(operands, attributes, properties, regions);
  
  // Get element type from src pointer
  Type elemType;
  if (auto ptrType = dyn_cast<triton::PointerType>(adaptor.getSrc().getType())) {
    elemType = ptrType.getPointeeType();
  } else {
    return failure();
  }
  
  // Get index shape to determine the size of dim
  auto indicesType = dyn_cast<RankedTensorType>(adaptor.getIndex().getType());
  if (!indicesType)
    return failure();
  int64_t numIndices = indicesType.getShape()[0];
  
  // Use adaptor to get attributes - this is the compatible way
  int32_t dim = adaptor.getDim();
  auto readShapeAttr = adaptor.getReadShape();
  
  // Build result shape: read_shape but with dim replaced by numIndices
  SmallVector<int64_t> resultShape;
  for (size_t i = 0; i < readShapeAttr.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      resultShape.push_back(numIndices);
    } else {
      resultShape.push_back(readShapeAttr[i]);
    }
  }
  
  // Create result tensor type
  inferredReturnTypes.push_back(RankedTensorType::get(resultShape, elemType));
  
  return success();
}

// FlipOp
LogicalResult FlipOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location, ValueRange operands,
                                       DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
                                       SmallVectorImpl<Type> &inferredReturnTypes)
{
    auto inputTy = dyn_cast<RankedTensorType>(operands[0].getType());
    if (!inputTy) {
        if (location)
            return emitOptionalError(location, "expected ranked tensor for flip input");
        return failure();
    }
    inferredReturnTypes.push_back(inputTy);
    return success();
}

//-- SortOp --
LogicalResult SortOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
    {
    if (operands.size() != 1) {
        return emitOptionalError(location, "expected exactly one operand for SortOp");
    }

    if (!isa<RankedTensorType>(operands[0].getType())) {
        return emitOptionalError(location, "operand must be a ranked tensor type for SortOp");
    }

    Value src = operands[0];
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcShape = srcTy.getShape();
    auto srcEnc = srcTy.getEncoding();

    if (srcShape.empty()) {
    return emitOptionalError(location, "input tensor must have rank >= 1");
    }

    Type sortedTy = RankedTensorType::get(srcShape, srcTy.getElementType(), srcEnc);

    inferredReturnTypes.push_back(sortedTy);

    return success();
}

//-- Conv1dOp --
LogicalResult Conv1dOp::verify()
{
    auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
    auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());

    constexpr int64_t dim2 = 2;
    constexpr int64_t dim3 = 3;
    auto inputRank = inputType.getShape().size();
    if (inputRank != dim2 && inputRank != dim3) {
      return emitOpError("input tensor must be 2D or 3D, but got rank ")
            << inputRank;
    }
    if (weightType.getShape().size() != dim3) {
      return emitOpError("weight tensor must be 3D, but got rank ")
            << weightType.getShape().size();
    }
    
    Value biasValue = getBias();
    if (biasValue) {
      auto biasType = dyn_cast<RankedTensorType>(biasValue.getType());
      if (!biasType || biasType.getRank() != 1) {
        return emitOpError("bias must be a 1D ranked tensor");
      }
      if (biasType.getElementType() != inputType.getElementType()) {
        return emitOpError("bias must have the same element type as input and weight");
      }
      if (biasType.getShape()[0] != weightType.getShape()[0]) {
        return emitOpError("bias size must match weight's output channel dimension");
      }
    }

    int64_t C_in = inputType.getShape()[inputRank - 2];
    int64_t C_out = weightType.getShape()[0];
    int64_t weight_C_in = weightType.getShape()[1];

    if (C_in % getGroups() != 0) {
      return emitOpError("input channels must be divisible by groups");
    }
    if (C_out % getGroups() != 0) {
      return emitOpError("output channels must be divisible by groups");
    }
    if (weight_C_in != C_in / getGroups()) {
      return emitOpError("weight's input channel dimension must equal input channels / groups");
    }

    return success();
}

LogicalResult Conv1dOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location,
    ValueRange operands, DictionaryAttr attributes,
    OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes)
    {
    Conv1dOpAdaptor adaptor(operands, attributes, properties, regions);

    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto weightType = dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
    if (!inputType || !weightType) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    bool isBatched = inputShape.size() == 3;
    if (isBatched) {
      int64_t N = inputShape[0];
    }
    int64_t C_in = inputShape[isBatched ? 1 : 0];
    int64_t L_in = inputShape[isBatched ? 2 : 1];
    
    ArrayRef<int64_t> weightShape = weightType.getShape();
    int64_t C_out = weightShape[0];
    int64_t K = weightShape[2];

    int64_t stride = adaptor.getStride();
    int64_t padding_size = adaptor.getPaddingSize();
    int64_t dilation = adaptor.getDilation();

    if (stride == 0) {
      return failure();
    }
    double l_out_double = static_cast<double>(L_in + 2 * padding_size - dilation * (K - 1) - 1) / stride + 1;
    int64_t L_out = static_cast<int64_t>(std::floor(l_out_double));
    
    constexpr int64_t dim3 = 3;
    SmallVector<int64_t, dim3> outputShape;
    if (isBatched) {
      outputShape.push_back(inputShape[0]);
    }
    outputShape.push_back(weightShape[0]);
    outputShape.push_back(L_out);

    Type elementType = inputType.getElementType();
    auto returnType = RankedTensorType::get(outputShape, elementType);
    inferredReturnTypes.push_back(returnType);
    return success();
}

} // namespace mlir::triton::ascend
