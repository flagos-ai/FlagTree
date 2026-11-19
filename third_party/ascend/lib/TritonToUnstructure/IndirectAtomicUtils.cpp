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

#include "TritonToUnstructure/IndirectAtomicUtils.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "triton-indirect-atomic-utils"

using namespace mlir;
using namespace triton;

namespace {

constexpr llvm::StringLiteral kIndirectAtomicBuiltin =
    "__builtin_indirect_atomic";
constexpr unsigned kInt8BitWidth = 8;
constexpr unsigned kInt16BitWidth = 16;

bool hasStaticShape(RankedTensorType tensorType) {
  return tensorType && tensorType.hasStaticShape();
}

bool hasStaticOffsetShape(Value offsetValue) {
  if (auto offsetTensorType = dyn_cast<RankedTensorType>(offsetValue.getType()))
    return offsetTensorType.hasStaticShape();
  return offsetValue.getType().isIntOrIndex();
}

bool isUnsupportedCasOrXchgElementType(Type elementType) {
  return elementType.isF16() || elementType.isBF16();
}

bool canUseIndirectAtomicFastPathForElementType(Type elementType) {
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return intType.getWidth() != kInt8BitWidth &&
           intType.getWidth() != kInt16BitWidth;
  return true;
}

bool canUseIndirectAtomicFastPathForOffset(Value offsetValue) {
  if (!offsetValue) {
    return true;
  }
  return hasStaticOffsetShape(offsetValue);
}

int64_t getNumElements(ArrayRef<int64_t> tensorShape) {
  return std::accumulate(tensorShape.begin(), tensorShape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

FailureOr<Attribute> getZeroAttr(Type elementType, PatternRewriter &rewriter) {
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return rewriter.getFloatAttr(floatType, 0.0);
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return rewriter.getIntegerAttr(intType, 0);
  return failure();
}

FailureOr<Value> createSplatConstantTensor(Location loc,
                                           RankedTensorType tensorType,
                                           Attribute splatAttr,
                                           PatternRewriter &rewriter) {
  if (!tensorType.hasStaticShape()) {
    return failure();
  }
  auto constant = rewriter.create<arith::ConstantOp>(
      loc, tensorType, DenseElementsAttr::get(tensorType, splatAttr));
  return constant.getResult();
}

FailureOr<Value> createZeroTensor(Location loc, RankedTensorType tensorType,
                                  PatternRewriter &rewriter) {
  auto zeroAttr = getZeroAttr(tensorType.getElementType(), rewriter);
  if (failed(zeroAttr)) {
    return failure();
  }
  return createSplatConstantTensor(loc, tensorType, *zeroAttr, rewriter);
}

// Convert i1 mask to i8 before reshape to avoid hardware alignment issues.
// Directly reshaping 2x3xi1 to 6xi1 without casting leads to incorrect memory
// layout: Expected: 1 1 1 1 1 1 0 0 Actual (due to i8 alignment padding): 1 1 1
// 0 0 0 0 0, 1 1 1 0 0 0 0 0 This layout mismatch breaks mask matching and
// causes accuracy errors. Casting to i8 first ensures proper element packing
// without alignment padding.
Value castMaskToI8(Location loc, Value maskValue, PatternRewriter &rewriter) {
  Type maskType = maskValue.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(maskType)) {
    auto targetType =
        RankedTensorType::get(tensorType.getShape(), rewriter.getI8Type());
    return rewriter.create<arith::ExtUIOp>(loc, targetType, maskValue);
  }
  return rewriter.create<arith::ExtUIOp>(loc, rewriter.getI8Type(), maskValue);
}

// Converts inputValue into a 1D tensor of type flatTensorType.
// This 1D flattening is mandatory: the low-level SIMT execution model only
// supports 1D tensor. If inputValue is scalar-like, it is first splatted to
// expandedTensorShape and then reshaped into the flattened 1D tensor type.
FailureOr<Value> createFlattenedTensorValue(
    Location loc, Value inputValue, ArrayRef<int64_t> expandedTensorShape,
    RankedTensorType flatTensorType, PatternRewriter &rewriter) {
  Value flattenedValue = inputValue;
  if (auto inputTensorType = dyn_cast<RankedTensorType>(inputValue.getType())) {
    if (!inputTensorType.hasStaticShape()) {
      return failure();
    }
    if (inputTensorType != flatTensorType) {
      if (inputTensorType.getNumElements() != flatTensorType.getNumElements()) {
        return failure();
      }
      flattenedValue =
          rewriter.create<triton::ReshapeOp>(loc, flatTensorType, inputValue);
    }
    return flattenedValue;
  }

  auto expandedTensorType = RankedTensorType::get(
      expandedTensorShape, flatTensorType.getElementType());
  Value splatValue =
      rewriter.create<triton::SplatOp>(loc, expandedTensorType, inputValue);
  if (expandedTensorType == flatTensorType) {
    return splatValue;
  }
  auto reshape =
      rewriter.create<triton::ReshapeOp>(loc, flatTensorType, splatValue);
  return reshape.getResult();
}

// Restores a flattened 1D tensor result to targetTensorType, which is the
// destination result tensor type expected by the original Triton op.
Value restoreFlattenedTensorShape(Location loc, Value flatTensorValue,
                                  RankedTensorType targetTensorType,
                                  PatternRewriter &rewriter) {
  if (cast<RankedTensorType>(flatTensorValue.getType()) == targetTensorType) {
    return flatTensorValue;
  }
  return rewriter.create<triton::ReshapeOp>(loc, targetTensorType,
                                            flatTensorValue);
}

std::optional<llvm::StringRef> normalizeRmwOperate(RMWOp op) {
  switch (op) {
  case RMWOp::ADD:
  case RMWOp::FADD:
    return llvm::StringRef("add");
  case RMWOp::MAX:
  case RMWOp::UMAX:
    return llvm::StringRef("max");
  case RMWOp::MIN:
  case RMWOp::UMIN:
    return llvm::StringRef("min");
  case RMWOp::XCHG:
    return llvm::StringRef("xchg");
  case RMWOp::AND:
    return llvm::StringRef("and");
  case RMWOp::OR:
    return llvm::StringRef("or");
  case RMWOp::XOR:
    return llvm::StringRef("xor");
  default:
    return std::nullopt;
  }
}

Value createIndirectAtomicCustom(Location loc,
                                 RankedTensorType customResultType,
                                 ValueRange inputs, ValueRange outputs,
                                 llvm::StringRef operate,
                                 PatternRewriter &rewriter) {
  auto custom = rewriter.create<hivm::CustomOp>(
      loc, TypeRange{customResultType}, kIndirectAtomicBuiltin, inputs, outputs,
      ValueRange{});
  custom->setAttr("symbol", rewriter.getStringAttr(kIndirectAtomicBuiltin));
  custom->setAttr("extra_attr",
                  rewriter.getStringAttr(("operate=" + operate).str()));
  custom->setAttr("hivm.pipe", hivm::PipeAttr::get(rewriter.getContext(),
                                                   hivm::PIPE::PIPE_V));
  custom->setAttr(
      "hivm.tcore_type",
      hivm::TCoreTypeAttr::get(rewriter.getContext(), hivm::TCoreType::VECTOR));
  custom->setAttr("hivm.vf_mode", hivm::VFModeAttr::get(rewriter.getContext(),
                                                        hivm::VFMode::SIMT));
  return custom->getResult(0);
}

} // namespace

namespace IndirectAtomicUtils {

bool canUseIndirectAtomicFastPath(triton::AtomicRMWOp op, Value offsetValue) {
  auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!hasStaticShape(resultTensorType)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Indirect atomic RMW falls back to legacy path: result "
                      "shape is not static.\n";
    });
    return false;
  }
  Type elementType = resultTensorType.getElementType();
  if (!canUseIndirectAtomicFastPathForElementType(elementType)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Indirect atomic RMW falls back to legacy path: element type "
          << elementType << " is not supported by SIMT fast path.\n";
    });
    return false;
  }
  if (op.getAtomicRmwOp() == RMWOp::XCHG &&
      isUnsupportedCasOrXchgElementType(elementType)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Indirect atomic RMW falls back to legacy path: XCHG "
                      "with element type "
                   << elementType << " is not supported by SIMT fast path.\n";
    });
    return false;
  }
  if (!canUseIndirectAtomicFastPathForOffset(offsetValue)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Indirect atomic RMW falls back to legacy path: offset "
                      "shape is not static.\n";
    });
    return false;
  }
  return true;
}

bool canUseIndirectAtomicFastPath(triton::AtomicCASOp op, Value offsetValue) {
  auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!hasStaticShape(resultTensorType)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Indirect atomic CAS falls back to legacy path: result "
                      "shape is not static.\n";
    });
    return false;
  }
  Type elementType = resultTensorType.getElementType();
  if (!canUseIndirectAtomicFastPathForElementType(elementType)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Indirect atomic CAS falls back to legacy path: element type "
          << elementType << " is not supported by SIMT fast path.\n";
    });
    return false;
  }
  if (isUnsupportedCasOrXchgElementType(elementType)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Indirect atomic CAS falls back to legacy path: element type "
          << elementType << " is not supported by SIMT fast path.\n";
    });
    return false;
  }
  if (!canUseIndirectAtomicFastPathForOffset(offsetValue)) {
    LLVM_DEBUG({
      llvm::dbgs() << "Indirect atomic CAS falls back to legacy path: offset "
                      "shape is not static.\n";
    });
    return false;
  }
  return true;
}

FailureOr<Value>
tryConvertAtomicRmwToIndirectCustom(triton::AtomicRMWOp op, Value srcPtr,
                                    Value offsetValue,
                                    PatternRewriter &rewriter) {
  auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!hasStaticShape(resultTensorType)) {
    return failure();
  }

  auto operate = normalizeRmwOperate(op.getAtomicRmwOp());
  if (!operate) {
    return failure();
  }

  auto resultTensorShape = resultTensorType.getShape();
  int64_t flatNumel = getNumElements(resultTensorShape);
  auto flatValueTensorType =
      RankedTensorType::get({flatNumel}, resultTensorType.getElementType());
  auto flatOffsetTensorType =
      RankedTensorType::get({flatNumel}, rewriter.getI64Type());

  auto flatOffsetValue =
      createFlattenedTensorValue(op.getLoc(), offsetValue, resultTensorShape,
                                 flatOffsetTensorType, rewriter);
  auto flatUpdateValue =
      createFlattenedTensorValue(op.getLoc(), op.getVal(), resultTensorShape,
                                 flatValueTensorType, rewriter);
  if (failed(flatOffsetValue) || failed(flatUpdateValue)) {
    return failure();
  }

  SmallVector<Value> inputs{srcPtr, *flatOffsetValue, *flatUpdateValue};
  if (Value maskValue = op.getMask()) {
    Value maskValueI8 = castMaskToI8(op.getLoc(), maskValue, rewriter);
    auto flatMaskTensorType =
        RankedTensorType::get({flatNumel}, rewriter.getI8Type());
    auto flatMaskValue =
        createFlattenedTensorValue(op.getLoc(), maskValueI8, resultTensorShape,
                                   flatMaskTensorType, rewriter);
    if (failed(flatMaskValue)) {
      return failure();
    }
    inputs.push_back(*flatMaskValue);
  }

  auto initOutFlatValue =
      createZeroTensor(op.getLoc(), flatValueTensorType, rewriter);
  if (failed(initOutFlatValue)) {
    return failure();
  }

  Value flatOldValue = createIndirectAtomicCustom(
      op.getLoc(), flatValueTensorType, inputs, ValueRange{*initOutFlatValue},
      *operate, rewriter);
  return restoreFlattenedTensorShape(op.getLoc(), flatOldValue,
                                     resultTensorType, rewriter);
}

FailureOr<Value>
tryConvertAtomicCasToIndirectCustom(triton::AtomicCASOp op, Value srcPtr,
                                    Value offsetValue,
                                    PatternRewriter &rewriter) {
  auto resultTensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!hasStaticShape(resultTensorType)) {
    return failure();
  }

  auto resultTensorShape = resultTensorType.getShape();
  int64_t flatNumel = getNumElements(resultTensorShape);
  auto flatValueTensorType =
      RankedTensorType::get({flatNumel}, resultTensorType.getElementType());
  auto flatOffsetTensorType =
      RankedTensorType::get({flatNumel}, rewriter.getI64Type());

  auto flatOffsetValue =
      createFlattenedTensorValue(op.getLoc(), offsetValue, resultTensorShape,
                                 flatOffsetTensorType, rewriter);
  auto flatCompareValue =
      createFlattenedTensorValue(op.getLoc(), op.getCmp(), resultTensorShape,
                                 flatValueTensorType, rewriter);
  auto flatUpdateValue =
      createFlattenedTensorValue(op.getLoc(), op.getVal(), resultTensorShape,
                                 flatValueTensorType, rewriter);
  if (failed(flatOffsetValue) || failed(flatCompareValue) ||
      failed(flatUpdateValue)) {
    return failure();
  }

  auto initOutFlatValue =
      createZeroTensor(op.getLoc(), flatValueTensorType, rewriter);
  if (failed(initOutFlatValue)) {
    return failure();
  }

  Value flatOldValue = createIndirectAtomicCustom(
      op.getLoc(), flatValueTensorType,
      ValueRange{srcPtr, *flatOffsetValue, *flatCompareValue, *flatUpdateValue},
      ValueRange{*initOutFlatValue}, "cas", rewriter);
  return restoreFlattenedTensorShape(op.getLoc(), flatOldValue,
                                     resultTensorType, rewriter);
}

} // namespace IndirectAtomicUtils
