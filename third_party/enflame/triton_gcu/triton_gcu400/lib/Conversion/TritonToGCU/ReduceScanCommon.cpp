/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ReduceScanCommon.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Vectorize.h"
#include "mlir/IR/Builders.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace triton {
namespace gcu {

Fold3DResult foldTo3D(ArrayRef<unsigned> elemsPerThread, unsigned axis) {
  std::array<int64_t, 3> dims = {1, 1, 1};
  int64_t foldedAxis = 2;
  for (int i = elemsPerThread.size() - 1, j = 2; i >= 0; i--) {
    if (static_cast<unsigned>(i) == axis) {
      if (dims[j] == 1)
        dims[j] = elemsPerThread[i];
      else
        dims[--j] = elemsPerThread[i];
      foldedAxis = j;
      --j;
    } else {
      dims[j] *= elemsPerThread[i];
    }
  }
  return {dims, foldedAxis};
}

static void registerCombineOpVectorizeHandlers(VectorizeEngine &engine) {
  engine.addHandler(std::make_unique<SelectVectorizeHandler>());
}

static const VectorizeEngine &getCombineOpVectorizeEngine() {
  static const VectorizeEngine &engine = []() {
    VectorizeEngine engine;
    registerCombineOpVectorizeHandlers(engine);
    return engine;
  }();
  return engine;
}

std::optional<vector::CombiningKind>
CombineOpDesc::matchCombiningKind(Region &combineOp) {
  if (combineOp.empty() || !combineOp.hasOneBlock()) {
    return std::nullopt;
  }
  auto &block = combineOp.front();
  if (!llvm::hasSingleElement(block.without_terminator())) {
    return std::nullopt;
  }
  auto &elementWiseOp = block.front();
  if (auto externElementwiseOp =
          dyn_cast<triton::ExternElementwiseOp>(elementWiseOp)) {
    return llvm::StringSwitch<std::optional<vector::CombiningKind>>(
               externElementwiseOp.getSymbol())
        .Case("__nv_fmaxf", vector::CombiningKind::MAXIMUMF)
        .Case("__nv_max", vector::CombiningKind::MAXSI)
        .Case("__nv_umax", vector::CombiningKind::MAXUI)
        .Case("__nv_fminf", vector::CombiningKind::MINIMUMF)
        .Case("__nv_min", vector::CombiningKind::MINSI)
        .Case("__nv_umin", vector::CombiningKind::MINUI)
        .Default(std::nullopt);
  }
  return TypeSwitch<Operation *, std::optional<vector::CombiningKind>>(
             &elementWiseOp)
      .Case<arith::AddIOp, arith::AddFOp>(
          [&](auto op) { return vector::CombiningKind::ADD; })
      .Case<arith::MulIOp, arith::MulFOp>(
          [&](auto op) { return vector::CombiningKind::MUL; })
      .Case<arith::MaxSIOp>(
          [&](auto op) { return vector::CombiningKind::MAXSI; })
      .Case<arith::MaxUIOp>(
          [&](auto op) { return vector::CombiningKind::MAXUI; })
      .Case<arith::MaxNumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXNUMF; })
      .Case<arith::MaximumFOp>(
          [&](auto op) { return vector::CombiningKind::MAXIMUMF; })
      .Case<arith::MinSIOp>(
          [&](auto op) { return vector::CombiningKind::MINSI; })
      .Case<arith::MinUIOp>(
          [&](auto op) { return vector::CombiningKind::MINUI; })
      .Case<arith::MinNumFOp>(
          [&](auto op) { return vector::CombiningKind::MINNUMF; })
      .Case<arith::MinimumFOp>(
          [&](auto op) { return vector::CombiningKind::MINIMUMF; })
      .Case<arith::AndIOp>([&](auto op) { return vector::CombiningKind::AND; })
      .Case<arith::OrIOp>([&](auto op) { return vector::CombiningKind::OR; })
      .Case<arith::XOrIOp>([&](auto op) { return vector::CombiningKind::XOR; })
      .Default([&](auto op) { return std::nullopt; });
}

SmallVector<Value>
CombineOpDesc::applyScalarCombine(OpBuilder &builder, Location loc,
                                  ValueRange operands) const {
  assert(combineOp.hasOneBlock() && "combine op must have exactly one block");
  auto combineBody = &combineOp.front();
  assert(combineBody->getNumArguments() == operands.size() &&
         "number of operands must match number of combine op arguments");
  IRMapping mapper;
  for (auto [arg, operand] :
       llvm::zip_equal(combineBody->getArguments(), operands)) {
    mapper.map(arg, operand);
  }
  for (auto &op : combineBody->without_terminator()) {
    auto cloneOp = builder.clone(op, mapper);
    for (auto [result, newResult] :
         llvm::zip_equal(op.getResults(), cloneOp->getResults())) {
      mapper.map(result, newResult);
    }
  }
  return llvm::to_vector(
      llvm::map_range(combineBody->getTerminator()->getOperands(), [&](auto v) {
        auto mappingValue = mapper.lookupOrNull(v);
        assert(mappingValue && "mapping value must not be null");
        return mappingValue;
      }));
}

SmallVector<Value>
CombineOpDesc::applyVectorizedCombine(OpBuilder &builder, Location loc,
                                      ValueRange operands,
                                      unsigned vectorLength) const {
  assert(combineOp.hasOneBlock() && "combine op must have exactly one block");
  auto combineBody = &combineOp.front();
  assert(combineBody->getNumArguments() == operands.size() &&
         "number of operands must match number of combine op arguments");
  IRMapping mapper;
  for (auto [arg, operand] :
       llvm::zip_equal(combineBody->getArguments(), operands)) {
    mapper.map(arg, operand);
  }
  auto &engine = getCombineOpVectorizeEngine();
  VectorizeContext ctx{builder, loc, mapper, vectorLength};
  for (auto &op : combineBody->without_terminator()) {
    for (auto operand : op.getOperands()) {
      if (mapper.lookupOrNull(operand)) {
        continue;
      }
      auto constantOp = operand.getDefiningOp<arith::ConstantOp>();
      if (!constantOp) {
        continue;
      }

      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointAfter(constantOp);
      auto operandType = operand.getType();
      auto vectorType =
          VectorType::get(ArrayRef<int64_t>{vectorLength}, operandType);
      if (operandType.isInteger(1)) {
        auto boolAttr = dyn_cast<BoolAttr>(constantOp.getValue());
        auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
        bool isFalse = (boolAttr && !boolAttr.getValue()) ||
                       (integerAttr && integerAttr.getValue().isZero());
        auto mask = builder.create<vector::ConstantMaskOp>(
            loc, vectorType,
            DenseI64ArrayAttr::get(
                builder.getContext(),
                ArrayRef<int64_t>{isFalse ? 0 : vectorLength}));
        mapper.map(operand, mask);
      } else {
        auto broadcast =
            builder.create<vector::BroadcastOp>(loc, vectorType, operand);
        mapper.map(operand, broadcast);
      }
    }
    engine.vectorize(&op, ctx);
  }

  return llvm::to_vector(
      llvm::map_range(combineBody->getTerminator()->getOperands(), [&](auto v) {
        auto mappingValue = mapper.lookupOrNull(v);
        assert(mappingValue && "mapping value must not be null");
        return mappingValue;
      }));
}

FailureOr<SmallVector<TypedAttr>>
CombineOpDesc::inferIdentityAttrs(OpBuilder &builder) const {
  if (!combiningKind) {
    return failure();
  }
  auto elementType = operandElementTypes.front();
  if (elementType.isInteger(1)) {
    elementType = builder.getI8Type();
  }
  switch (*combiningKind) {
  case vector::CombiningKind::ADD:
    return SmallVector<TypedAttr>{builder.getZeroAttr(elementType)};
  case vector::CombiningKind::OR:
  case vector::CombiningKind::XOR: {
    if (auto integerType = dyn_cast<IntegerType>(elementType)) {
      return SmallVector<TypedAttr>{builder.getIntegerAttr(
          elementType, APInt::getZero(integerType.getWidth()))};
    }
    return failure();
  }
  case vector::CombiningKind::AND: {
    if (auto integerType = dyn_cast<IntegerType>(elementType)) {
      return SmallVector<TypedAttr>{builder.getIntegerAttr(
          elementType, APInt::getAllOnes(integerType.getWidth()))};
    }
    return failure();
  }
  case vector::CombiningKind::MAXSI: {
    auto integerType = dyn_cast<IntegerType>(elementType);
    if (!integerType) {
      return failure();
    }
    return SmallVector<TypedAttr>{builder.getIntegerAttr(
        elementType, APInt::getSignedMinValue(integerType.getWidth()))};
  }
  case vector::CombiningKind::MINSI: {
    auto integerType = dyn_cast<IntegerType>(elementType);
    if (!integerType) {
      return failure();
    }
    return SmallVector<TypedAttr>{builder.getIntegerAttr(
        elementType, APInt::getSignedMaxValue(integerType.getWidth()))};
  }
  case vector::CombiningKind::MAXNUMF: {
    auto floatType = dyn_cast<FloatType>(elementType);
    if (!floatType) {
      return failure();
    }
    return SmallVector<TypedAttr>{builder.getFloatAttr(
        elementType, APFloat::getInf(floatType.getFloatSemantics(), true))};
  }
  case vector::CombiningKind::MINNUMF: {
    auto floatType = dyn_cast<FloatType>(elementType);
    if (!floatType) {
      return failure();
    }
    return SmallVector<TypedAttr>{builder.getFloatAttr(
        elementType, APFloat::getInf(floatType.getFloatSemantics(), false))};
  }
  case vector::CombiningKind::MAXUI: {
    auto integerType = dyn_cast<IntegerType>(operandElementTypes.front());
    if (!integerType) {
      return failure();
    }
    return SmallVector<TypedAttr>{builder.getIntegerAttr(
        operandElementTypes.front(), APInt::getZero(integerType.getWidth()))};
  }
  case vector::CombiningKind::MINUI: {
    auto integerType = dyn_cast<IntegerType>(operandElementTypes.front());
    if (!integerType) {
      return failure();
    }
    return SmallVector<TypedAttr>{
        builder.getIntegerAttr(operandElementTypes.front(),
                               APInt::getAllOnes(integerType.getWidth()))};
  }
  default:
    return failure();
  }
}

SmallVector<Value> reduceVectorLanes(OpBuilder &builder, Location loc,
                                     const CombineOpDesc &combineOpDesc,
                                     ValueRange vecValues) {
  assert(vecValues.size() == combineOpDesc.getNumOperands() &&
         "number of operands must match number of combine op operands");
  assert(llvm::all_of(vecValues.getTypes(),
                      [&](auto ty) {
                        return isa<VectorType>(ty) &&
                               cast<VectorType>(ty).getRank() == 1;
                      }) &&
         "all input vectors must have rank 1");
  if (combineOpDesc.hasFastReduceLanesImpl()) {
    return {builder.create<vector::ReductionOp>(
        loc, *combineOpDesc.getCombiningKind(), vecValues.front())};
  }
  SmallVector<Value> accumulators(vecValues.begin(), vecValues.end());
  SmallVector<Value> combineOperands;
  combineOperands.reserve(accumulators.size() * 2);
  auto vectorLength =
      cast<VectorType>(vecValues.front().getType()).getDimSize(0);
  auto numElements = vectorLength;
  while (numElements != 1) {
    combineOperands.clear();
    combineOperands.append(accumulators);
    auto shiftVal = builder.create<arith::ConstantIntOp>(
        loc, builder.getI32Type(), numElements / 2);
    for (auto accumulator : accumulators) {
      combineOperands.push_back(builder.create<mlir::gcu::VectorShiftOp>(
          loc, mlir::gcu::VectorShiftDirection::LEFT, accumulator, shiftVal));
    }
    accumulators = combineOpDesc.applyVectorizedCombine(
        builder, loc, combineOperands, vectorLength);
    numElements /= 2;
  }
  return llvm::to_vector(llvm::map_range(accumulators, [&](auto accumulator) {
    return builder.create<vector::ExtractOp>(loc, accumulator, 0).getResult();
  }));
}

} // namespace gcu
} // namespace triton
} // namespace mlir
