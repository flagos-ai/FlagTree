//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.cpp.inc"

namespace mlir {
namespace xsmt {

using namespace mlir;
using namespace mlir::triton;

static int64_t ceilDivI64(int64_t lhs, int64_t rhs) {
  return (lhs + rhs - 1) / rhs;
}

// ===----------------------------------------------------------------------===//
// PackOp
// ===----------------------------------------------------------------------===//

void PackOp::build(OpBuilder &builder, OperationState &state, Value base,
                   ValueRange offsets, ArrayRef<int32_t> shape,
                   ArrayRef<int32_t> packed_size) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  SmallVector<int64_t> resultShape = {ceilDivI64(shape[0], packed_size[0]),
                                      ceilDivI64(shape[1], packed_size[1]),
                                      static_cast<int64_t>(packed_size[0]),
                                      static_cast<int64_t>(packed_size[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, /*destination=*/Value(),
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(packed_size));
}

void PackOp::build(OpBuilder &builder, OperationState &state, Value base,
                   ValueRange offsets, Value destination,
                   ArrayRef<int32_t> shape, ArrayRef<int32_t> packed_size) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  SmallVector<int64_t> resultShape = {ceilDivI64(shape[0], packed_size[0]),
                                      ceilDivI64(shape[1], packed_size[1]),
                                      static_cast<int64_t>(packed_size[0]),
                                      static_cast<int64_t>(packed_size[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, destination,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(packed_size));
}

LogicalResult PackOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  auto packedSizeAttr = getPackedSize();
  if (packedSizeAttr.size() != shapeAttr.size()) {
    return emitOpError("packed_size dimensions (")
           << packedSizeAttr.size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  return success();
}

// ===----------------------------------------------------------------------===//
// UnpackOp
// ===----------------------------------------------------------------------===//

void UnpackOp::build(OpBuilder &builder, OperationState &state, Value base,
                     ValueRange offsets, ArrayRef<int32_t> shape) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  // Result is 2D: the target shape
  SmallVector<int64_t> resultShape = {static_cast<int64_t>(shape[0]),
                                      static_cast<int64_t>(shape[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, /*destination=*/Value(),
        builder.getDenseI32ArrayAttr(shape));
}

void UnpackOp::build(OpBuilder &builder, OperationState &state, Value base,
                     ValueRange offsets, Value destination,
                     ArrayRef<int32_t> shape) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  SmallVector<int64_t> resultShape = {static_cast<int64_t>(shape[0]),
                                      static_cast<int64_t>(shape[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, destination,
        builder.getDenseI32ArrayAttr(shape));
}

LogicalResult UnpackOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  return success();
}

// ===----------------------------------------------------------------------===//
// RepackOp
// ===----------------------------------------------------------------------===//

void RepackOp::build(OpBuilder &builder, OperationState &state, Value base,
                     ValueRange offsets, ArrayRef<int32_t> shape,
                     ArrayRef<int32_t> packed_size) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  SmallVector<int64_t> resultShape = {ceilDivI64(shape[0], packed_size[0]),
                                      ceilDivI64(shape[1], packed_size[1]),
                                      static_cast<int64_t>(packed_size[0]),
                                      static_cast<int64_t>(packed_size[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, /*destination=*/Value(),
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(packed_size));
}

void RepackOp::build(OpBuilder &builder, OperationState &state, Value base,
                     ValueRange offsets, Value destination,
                     ArrayRef<int32_t> shape, ArrayRef<int32_t> packed_size) {
  Type baseType = base.getType();
  RankedTensorType baseRankedType = cast<RankedTensorType>(baseType);
  Type elementType = baseRankedType.getElementType();

  SmallVector<int64_t> resultShape = {ceilDivI64(shape[0], packed_size[0]),
                                      ceilDivI64(shape[1], packed_size[1]),
                                      static_cast<int64_t>(packed_size[0]),
                                      static_cast<int64_t>(packed_size[1])};

  auto resultType = RankedTensorType::get(resultShape, elementType);
  build(builder, state, resultType, base, offsets, destination,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(packed_size));
}

LogicalResult RepackOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  auto packedSizeAttr = getPackedSize();
  if (packedSizeAttr.size() != shapeAttr.size()) {
    return emitOpError("packed_size dimensions (")
           << packedSizeAttr.size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  return success();
}

// ===----------------------------------------------------------------------===//
// SubviewOp
// ===----------------------------------------------------------------------===//

void SubviewOp::build(OpBuilder &builder, OperationState &state, Value base,
                      ValueRange offsets, ArrayRef<int32_t> shape) {
  Type baseType = base.getType();
  auto ptrType = cast<PointerType>(baseType);
  RankedTensorType baseRankedType =
      cast<RankedTensorType>(ptrType.getPointeeType());
  Type elementType = baseRankedType.getElementType();
  unsigned addrSpace = ptrType.getAddressSpace();

  // Subview preserves the packing of the base. If base is 4D, compute
  // the outer dims from shape and the packed tile from base dims [2],[3].
  auto baseShape = baseRankedType.getShape();
  SmallVector<int64_t> resultShape;
  if (baseShape.size() == 4) {
    int64_t tile0 = baseShape[2];
    int64_t tile1 = baseShape[3];
    resultShape = {ceilDivI64(shape[0], tile0), ceilDivI64(shape[1], tile1),
                   tile0, tile1};
  } else {
    for (int32_t s : shape)
      resultShape.push_back(static_cast<int64_t>(s));
  }

  auto resultRankedType = RankedTensorType::get(resultShape, elementType);
  Type resultType = PointerType::get(resultRankedType, addrSpace);
  build(builder, state, resultType, base, offsets,
        builder.getDenseI32ArrayAttr(shape));
}

LogicalResult SubviewOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  return success();
}

// ===----------------------------------------------------------------------===//
// SubviewPackOp
// ===----------------------------------------------------------------------===//

void SubviewPackOp::build(OpBuilder &builder, OperationState &state, Value base,
                          ValueRange offsets, ArrayRef<int32_t> shape,
                          ArrayRef<int32_t> packed_size) {
  Type baseType = base.getType();
  auto ptrType = cast<PointerType>(baseType);
  RankedTensorType baseRankedType =
      cast<RankedTensorType>(ptrType.getPointeeType());
  Type elementType = baseRankedType.getElementType();
  unsigned addrSpace = ptrType.getAddressSpace();

  SmallVector<int64_t> resultShape = {ceilDivI64(shape[0], packed_size[0]),
                                      ceilDivI64(shape[1], packed_size[1]),
                                      static_cast<int64_t>(packed_size[0]),
                                      static_cast<int64_t>(packed_size[1])};

  auto resultRankedType = RankedTensorType::get(resultShape, elementType);
  Type resultType = PointerType::get(resultRankedType, addrSpace);
  build(builder, state, resultType, base, offsets,
        builder.getDenseI32ArrayAttr(shape),
        builder.getDenseI32ArrayAttr(packed_size));
}

LogicalResult SubviewPackOp::verify() {
  auto shapeAttr = getShape();
  if (getOffsets().size() != shapeAttr.size()) {
    return emitOpError("number of offsets (")
           << getOffsets().size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  auto packedSizeAttr = getPackedSize();
  if (packedSizeAttr.size() != shapeAttr.size()) {
    return emitOpError("packed_size dimensions (")
           << packedSizeAttr.size() << ") must match shape dimensions ("
           << shapeAttr.size() << ")";
  }
  return success();
}

void DescriptorLoadViewOp::build(OpBuilder &builder, OperationState &state,
                                 Value base, ValueRange offsets,
                                 ArrayRef<int32_t> shape,
                                 ArrayRef<int32_t> packed_size) {
  auto pointerType = cast<mlir::triton::PointerType>(base.getType());
  auto pointeeType = pointerType.getPointeeType();

  Type elementType = pointeeType;
  while (auto tensorType = dyn_cast<RankedTensorType>(elementType)) {
    elementType = tensorType.getElementType();
  }

  if (isa<RankedTensorType>(elementType)) {
    llvm::report_fatal_error("Cannot determine scalar element type for tensor");
  }

  SmallVector<int64_t> resultShape;
  resultShape.push_back(ceilDivI64(shape[0], packed_size[0]));
  resultShape.push_back(ceilDivI64(shape[1], packed_size[1]));
  resultShape.push_back(packed_size[0]);
  resultShape.push_back(packed_size[1]);

  auto resultType = RankedTensorType::get(resultShape, elementType);

  return build(builder, state, resultType, base, offsets,
               builder.getDenseI32ArrayAttr(shape),
               builder.getDenseI32ArrayAttr(packed_size));
}

void BufferTensorViewOp::build(OpBuilder &builder, OperationState &state,
                               Value buffer, Value bufferIdx) {
  auto bufferType = cast<BufferType>(buffer.getType());
  auto shape = bufferType.getShape();

  llvm::SmallVector<int64_t> resultShape;
  if (bufferType.getCopies() == 0) {
    if (shape.size() == 1) {
      resultShape = {1};
    } else {
      resultShape.assign(shape.begin() + 1, shape.end());
    }
  } else {
    resultShape.assign(shape.begin() + 1, shape.end());
  }

  auto tensorType =
      RankedTensorType::get(resultShape, bufferType.getElementType());

  auto resultType = triton::PointerType::get(tensorType, 1);

  state.addTypes(resultType);
  state.addOperands({buffer, bufferIdx});
}

// ===----------------------------------------------------------------------===//
// Custom assembly format helpers for destination-style ops
// ===----------------------------------------------------------------------===//

/// Print helper for ops with optional destination:
///   %r = xsmt.pack %base, [%off0, %off1], [shape], [packed_size]
///         {into %dest} attr-dict : type($base) -> type($result)
template <bool HasPackedSize, typename OpTy>
static void printDstStyleOp(OpTy op, OpAsmPrinter &p) {
  p << ' ' << op.getBase() << ", [";
  llvm::interleaveComma(op.getOffsets(), p);
  p << "]";
  if (op.getDestination()) {
    p << " into(" << op.getDestination() << " : "
      << op.getDestination().getType() << ")";
  }
  p << ", [" << op.getShape() << "]";
  if constexpr (HasPackedSize) {
    p << ", [" << op.getPackedSize() << "]";
  }
  p.printOptionalAttrDict(
      op->getAttrs(),
      /*elidedAttrs=*/{"shape", "packed_size", "operandSegmentSizes"});
  p << " : " << op.getBase().getType() << " -> " << op.getResult().getType();
}

/// Parse helper for ops with optional destination.
/// Format:
///   %base, [%off0, %off1] (into(%dest : type))? , [shape] (, [packed_size])?
///   ... : type -> type
template <bool HasPackedSize>
static ParseResult parseDstStyleOp(OpAsmParser &parser,
                                   OperationState &result) {
  OpAsmParser::UnresolvedOperand base;
  SmallVector<OpAsmParser::UnresolvedOperand> offsets;
  OpAsmParser::UnresolvedOperand dest;
  Type destType;
  bool hasDest = false;
  Type baseType, resultType;

  // Parse: %base , [ offsets ]
  if (parser.parseOperand(base) || parser.parseComma() ||
      parser.parseLSquare() || parser.parseOperandList(offsets) ||
      parser.parseRSquare())
    return failure();

  // Parse optional: into(%dest : type)
  if (succeeded(parser.parseOptionalKeyword("into"))) {
    if (parser.parseLParen() || parser.parseOperand(dest) ||
        parser.parseColon() || parser.parseType(destType) ||
        parser.parseRParen())
      return failure();
    hasDest = true;
  }

  // Parse: , [shape]
  SmallVector<int32_t> shapeVals;
  if (parser.parseComma() || parser.parseLSquare())
    return failure();
  {
    int32_t val;
    if (parser.parseInteger(val))
      return failure();
    shapeVals.push_back(val);
    while (succeeded(parser.parseOptionalComma())) {
      if (parser.parseInteger(val))
        return failure();
      shapeVals.push_back(val);
    }
  }
  if (parser.parseRSquare())
    return failure();
  result.addAttribute("shape",
                      parser.getBuilder().getDenseI32ArrayAttr(shapeVals));

  // Parse optional: , [packed_size]
  if constexpr (HasPackedSize) {
    SmallVector<int32_t> packedSizeVals;
    if (parser.parseComma() || parser.parseLSquare())
      return failure();
    {
      int32_t val;
      if (parser.parseInteger(val))
        return failure();
      packedSizeVals.push_back(val);
      while (succeeded(parser.parseOptionalComma())) {
        if (parser.parseInteger(val))
          return failure();
        packedSizeVals.push_back(val);
      }
    }
    if (parser.parseRSquare())
      return failure();
    result.addAttribute("packed_size", parser.getBuilder().getDenseI32ArrayAttr(
                                           packedSizeVals));
  }

  // Parse: attr-dict : baseType -> resultType
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseColon() ||
      parser.parseType(baseType) || parser.parseArrow() ||
      parser.parseType(resultType))
    return failure();

  // Resolve operands
  if (parser.resolveOperand(base, baseType, result.operands))
    return failure();

  auto i32Type = parser.getBuilder().getIntegerType(32);
  for (auto &off : offsets) {
    if (parser.resolveOperand(off, i32Type, result.operands))
      return failure();
  }

  if (hasDest) {
    if (parser.resolveOperand(dest, destType, result.operands))
      return failure();
  }

  // operandSegmentSizes: [base=1, offsets=N, destination=0or1]
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(offsets.size()), hasDest ? 1 : 0}));

  result.addTypes(resultType);
  return success();
}

// ===----------------------------------------------------------------------===//
// PackOp print/parse
// ===----------------------------------------------------------------------===//

void PackOp::print(OpAsmPrinter &p) { printDstStyleOp<true>(*this, p); }

ParseResult PackOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp</*HasPackedSize=*/true>(parser, result);
}

// ===----------------------------------------------------------------------===//
// UnpackOp print/parse
// ===----------------------------------------------------------------------===//

void UnpackOp::print(OpAsmPrinter &p) { printDstStyleOp<false>(*this, p); }

ParseResult UnpackOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp</*HasPackedSize=*/false>(parser, result);
}

// ===----------------------------------------------------------------------===//
// RepackOp print/parse
// ===----------------------------------------------------------------------===//

void RepackOp::print(OpAsmPrinter &p) { printDstStyleOp<true>(*this, p); }

ParseResult RepackOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseDstStyleOp</*HasPackedSize=*/true>(parser, result);
}

} // namespace xsmt
} // namespace mlir
