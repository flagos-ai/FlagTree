//===- TritonAscendOps.cpp - TritonAscend dialect operations --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/Utils/Utils.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace mlir::triton;

namespace mlir::triton::ascend {

void GatherOutToUbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

void IndirectLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

void StrideLoadOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSrcMutable(),
                       triton::GlobalMemory::get());
}

//-- DotOp --
// Fractal (zN) block geometry (kFractalBlock / kDotAccIntWidth) comes from
// Utils.h, shared with DotConverter's lowering.
LogicalResult
DotOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                        ValueRange operands, DictionaryAttr attributes,
                        OpaqueProperties properties, RegionRange regions,
                        SmallVectorImpl<Type> &inferredReturnTypes) {
  DotOpAdaptor adaptor(operands, attributes, properties, regions);
  auto aTy = dyn_cast<RankedTensorType>(adaptor.getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(adaptor.getB().getType());
  if (!aTy || !bTy)
    return failure();

  // A fractal operand's ND shape. Both operands are zN (non-transpose):
  // lhs [K1,M1,16,b] -> [M,K], rhs [N1,K1,16,b] -> [K,N], each [d1*d2, d0*d3].
  auto toND = [](ArrayRef<int64_t> shape, bool fractal,
                 bool isLhs) -> SmallVector<int64_t> {
    if (!fractal || shape.size() != 4)
      return {shape[0], shape[1]};
    return {shape[1] * shape[2], shape[0] * shape[3]};
  };
  int64_t m = toND(aTy.getShape(), adaptor.getFractalA(), /*isLhs=*/true)[0];
  int64_t n = toND(bTy.getShape(), adaptor.getFractalB(), /*isLhs=*/false)[1];

  // The result carries the cube accumulator dtype (f32 for float inputs, i32
  // for int8), not the input dtype. fractal_c produces the L0C accumulator
  // fractal, whose block is 16x16 regardless of dtype.
  Type inElemTy = aTy.getElementType();
  Type accTy =
      isa<IntegerType>(inElemTy)
          ? static_cast<Type>(IntegerType::get(context, kDotAccIntWidth))
          : static_cast<Type>(Float32Type::get(context));
  SmallVector<int64_t> resShape{m, n};
  if (adaptor.getFractalC())
    // ND [M,N] -> zN accumulator [N1, M1, 16, 16].
    resShape = {n / kFractalBlock, m / kFractalBlock, kFractalBlock,
                kFractalBlock};
  inferredReturnTypes.push_back(RankedTensorType::get(resShape, accTy));
  return success();
}

// Verify operand ranks match their fractal flags: a fractal operand must be 4D
// (zN), a non-fractal operand must be 2D (ND). Anything else (e.g. rank > 4) is
// a compile error.
LogicalResult DotOp::verify() {
  auto aTy = dyn_cast<RankedTensorType>(getA().getType());
  auto bTy = dyn_cast<RankedTensorType>(getB().getType());
  if (!aTy || !bTy)
    return emitOpError("operands must be ranked tensors");
  auto checkRank = [&](RankedTensorType ty, bool fractal,
                       StringRef name) -> LogicalResult {
    int64_t rank = ty.getRank();
    if (fractal && rank != 4)
      return emitOpError() << "fractal operand '" << name
                           << "' must be 4D (zN), got rank " << rank;
    if (!fractal && rank != 2)
      return emitOpError() << "non-fractal operand '" << name
                           << "' must be 2D (ND), got rank " << rank;
    return success();
  };
  if (failed(checkRank(aTy, getFractalA(), "a")) ||
      failed(checkRank(bTy, getFractalB(), "b")))
    return failure();
  return success();
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
  if (auto ptrType =
          dyn_cast<triton::PointerType>(adaptor.getSrc().getType())) {
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
LogicalResult
FlipOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  auto inputTy = dyn_cast<RankedTensorType>(operands[0].getType());
  if (!inputTy) {
    if (location)
      return emitOptionalError(location,
                               "expected ranked tensor for flip input");
    return failure();
  }
  inferredReturnTypes.push_back(inputTy);
  return success();
}

//-- SortOp --
LogicalResult
SortOp::inferReturnTypes(MLIRContext *context, std::optional<Location> location,
                         ValueRange operands, DictionaryAttr attributes,
                         OpaqueProperties properties, RegionRange regions,
                         SmallVectorImpl<Type> &inferredReturnTypes) {
  if (operands.size() != 1) {
    return emitOptionalError(location,
                             "expected exactly one operand for SortOp");
  }

  if (!isa<RankedTensorType>(operands[0].getType())) {
    return emitOptionalError(location,
                             "operand must be a ranked tensor type for SortOp");
  }

  Value src = operands[0];
  auto srcTy = cast<RankedTensorType>(src.getType());
  auto srcShape = srcTy.getShape();
  auto srcEnc = srcTy.getEncoding();

  if (srcShape.empty()) {
    return emitOptionalError(location, "input tensor must have rank >= 1");
  }

  Type sortedTy =
      RankedTensorType::get(srcShape, srcTy.getElementType(), srcEnc);

  inferredReturnTypes.push_back(sortedTy);

  return success();
}

//-- Conv1dOp --
LogicalResult Conv1dOp::verify() {
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
      return emitOpError(
          "bias must have the same element type as input and weight");
    }
    if (biasType.getShape()[0] != weightType.getShape()[0]) {
      return emitOpError(
          "bias size must match weight's output channel dimension");
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
    return emitOpError(
        "weight's input channel dimension must equal input channels / groups");
  }

  return success();
}

LogicalResult Conv1dOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
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
  double l_out_double =
      static_cast<double>(L_in + 2 * padding_size - dilation * (K - 1) - 1) /
          stride +
      1;
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

//-- Conv2dOp --

// Parse a conv2d parameter attribute: an integer (uniform scalar) or an
// integer array. Returns the raw values, or std::nullopt for an invalid
// attribute.
static std::optional<SmallVector<int64_t>>
parseConv2dParamAttr(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr))
    return SmallVector<int64_t>{intAttr.getInt()};
  if (auto arrayAttr = dyn_cast<DenseI32ArrayAttr>(attr)) {
    SmallVector<int64_t> values;
    for (int32_t v : arrayAttr.asArrayRef())
      values.push_back(static_cast<int64_t>(v));
    return values;
  }
  return std::nullopt;
}

LogicalResult Conv2dOp::verify() {
  auto inputType = dyn_cast<RankedTensorType>(getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(getWeight().getType());

  constexpr int64_t dim3 = 3;
  constexpr int64_t dim4 = 4;
  auto inputRank = inputType.getShape().size();
  if (inputRank != dim3 && inputRank != dim4) {
    return emitOpError(
               "input tensor must be 3D (C,H,W) or 4D (N,C,H,W), but got rank ")
           << inputRank;
  }
  if (weightType.getShape().size() != dim4) {
    return emitOpError("weight tensor must be 4D (C_out, C_in/groups, kH, kW), "
                       "but got rank ")
           << weightType.getShape().size();
  }

  Value biasValue = getBias();
  if (biasValue) {
    auto biasType = dyn_cast<RankedTensorType>(biasValue.getType());
    if (!biasType || biasType.getRank() != 1) {
      return emitOpError("bias must be a 1D ranked tensor");
    }
    if (biasType.getElementType() != inputType.getElementType()) {
      return emitOpError(
          "bias must have the same element type as input and weight");
    }
    if (biasType.getShape()[0] != weightType.getShape()[0]) {
      return emitOpError(
          "bias size must match weight's output channel dimension");
    }
  }

  auto stride = parseConv2dParamAttr(getStride());
  auto padding = parseConv2dParamAttr(getPadding());
  auto dilation = parseConv2dParamAttr(getDilation());
  if (!stride || stride->empty() || stride->size() > 2) {
    return emitOpError(
        "stride must be an integer or a 2-element array [stride_h, stride_w]");
  }
  if (!padding || padding->empty() || padding->size() > 4) {
    return emitOpError(
        "padding must be an integer, a 2-element array [pad_h, pad_w], or a "
        "4-element array [pad_top, pad_bottom, pad_left, pad_right]");
  }
  if (!dilation || dilation->empty() || dilation->size() > 2) {
    return emitOpError(
        "dilation must be an integer or a 2-element array [dil_h, dil_w]");
  }
  // Uniform scalar or pair: front/back covers both forms.
  if (stride->front() == 0 || stride->back() == 0) {
    return emitOpError("stride elements must be > 0");
  }

  int64_t C_in = inputType.getShape()[inputRank - 3];
  int64_t C_out = weightType.getShape()[0];
  int64_t weight_C_in = weightType.getShape()[1];

  if (C_in % getGroups() != 0) {
    return emitOpError("input channels must be divisible by groups");
  }
  if (C_out % getGroups() != 0) {
    return emitOpError("output channels must be divisible by groups");
  }
  if (weight_C_in != C_in / getGroups()) {
    return emitOpError(
        "weight's input channel dimension must equal input channels / groups");
  }

  return success();
}

LogicalResult Conv2dOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  Conv2dOpAdaptor adaptor(operands, attributes, properties, regions);

  auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto weightType = dyn_cast<RankedTensorType>(adaptor.getWeight().getType());
  if (!inputType || !weightType) {
    return failure();
  }

  ArrayRef<int64_t> inputShape = inputType.getShape();
  bool isBatched = inputShape.size() == 4;
  int64_t C_in = inputShape[isBatched ? 1 : 0];
  int64_t H_in = inputShape[isBatched ? 2 : 1];
  int64_t W_in = inputShape[isBatched ? 3 : 2];

  ArrayRef<int64_t> weightShape = weightType.getShape();
  int64_t C_out = weightShape[0];
  int64_t kH = weightShape[2];
  int64_t kW = weightShape[3];

  auto strideAttr = parseConv2dParamAttr(adaptor.getStride());
  auto paddingAttr = parseConv2dParamAttr(adaptor.getPadding());
  auto dilationAttr = parseConv2dParamAttr(adaptor.getDilation());
  if (!strideAttr || strideAttr->empty() || strideAttr->size() > 2 ||
      !paddingAttr || paddingAttr->empty() || paddingAttr->size() > 4 ||
      !dilationAttr || dilationAttr->empty() || dilationAttr->size() > 2) {
    return failure();
  }
  auto &stride = *strideAttr;
  auto &padding = *paddingAttr;
  auto &dilation = *dilationAttr;

  auto computeOutDim = [](int64_t in_size, int64_t pad_before,
                          int64_t pad_after, int64_t dil, int64_t k,
                          int64_t str) -> int64_t {
    double val = static_cast<double>(in_size + pad_before + pad_after -
                                     dil * (k - 1) - 1) /
                     str +
                 1;
    return static_cast<int64_t>(std::floor(val));
  };

  // Expand padding to [pad_top, pad_bottom, pad_left, pad_right].
  SmallVector<int64_t, 4> pads;
  if (padding.size() == 1) {
    pads.assign(4, padding[0]);
  } else if (padding.size() == 2) {
    pads = {padding[0], padding[0], padding[1], padding[1]};
  } else {
    pads.assign(padding.begin(), padding.end());
  }
  // Uniform scalar or pair: front/back covers both forms.
  int64_t strideH = stride.front();
  int64_t strideW = stride.back();
  int64_t dilH = dilation.front();
  int64_t dilW = dilation.back();

  // pads = [pad_top, pad_bottom, pad_left, pad_right]
  int64_t H_out = computeOutDim(H_in, pads[0], pads[1], dilH, kH, strideH);
  int64_t W_out = computeOutDim(W_in, pads[2], pads[3], dilW, kW, strideW);

  SmallVector<int64_t, 4> outputShape;
  if (isBatched) {
    outputShape.push_back(inputShape[0]);
  }
  outputShape.push_back(weightShape[0]);
  outputShape.push_back(H_out);
  outputShape.push_back(W_out);

  Type elementType = inputType.getElementType();
  auto returnType = RankedTensorType::get(outputShape, elementType);
  inferredReturnTypes.push_back(returnType);
  return success();
}

} // namespace mlir::triton::ascend
