//===- AddressDialect.cpp - Address dialect ---------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Address dialect.
//
//===----------------------------------------------------------------------===//

#include "Address/Dialect/IR/AddressDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::addr;

//===----------------------------------------------------------------------===//
// Address dialect
//===----------------------------------------------------------------------===//

void AddressDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Address/Dialect/IR/AddressOps.cpp.inc"
      >();
  registerTypes();
}

void AddressDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Address/Dialect/IR/AddressOpsTypes.cpp.inc"
      >();
}

namespace {
ParseResult parseAddressType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getType<AddressType>();
  return success();
}
void printAddressType(OpAsmPrinter &p, Operation *op, AddressType ty) {
  if (ty.getAddressSpace() != nullptr)
    p << " : " << ty;
}

ParseResult parseIntType(OpAsmParser &parser, Type &ty) {
  if (succeeded(parser.parseOptionalColon()) && parser.parseType(ty))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  if (!ty)
    ty = parser.getBuilder().getIndexType();
  return success();
}
void printIntType(OpAsmPrinter &p, Operation *op, Type ty) {
  if (!ty.isIndex())
    p << " : " << ty;
}
} // namespace

//===----------------------------------------------------------------------===//
// CastInt Op
//===----------------------------------------------------------------------===//

bool CastIntOp::areCastCompatible(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  return isa<AddressType>(lhs.front()) != isa<AddressType>(rhs.front());
}

//===----------------------------------------------------------------------===//
// Constant Op
//===----------------------------------------------------------------------===//

void ConstantOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       int64_t value, Attribute addressSpace) {
  build(odsBuilder, odsState, odsBuilder.getType<AddressType>(addressSpace),
        odsBuilder.getIndexAttr(value));
}

void ConstantOp::getAsmResultNames(OpAsmSetValueNameFn setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream name(buffer);
  name << "addr" << getValueAttr().getValue();
  setNameFn(getResult(), name.str());
}

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return adaptor.getValueAttr();
}

//===----------------------------------------------------------------------===//
// TypeOffset Op
//===----------------------------------------------------------------------===//

OpFoldResult TypeOffsetOp::fold(FoldAdaptor adaptor) {
  return adaptor.getBaseTypeAttr();
}

//===----------------------------------------------------------------------===//
// Cast Op
//===----------------------------------------------------------------------===//

LogicalResult CastOp::canonicalize(CastOp op, PatternRewriter &rewriter) {
  if (op.getInput().getType() == op.getType()) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// FromMemRef Op
//===----------------------------------------------------------------------===//

LogicalResult FromMemRefOp::verify() {
  if (getType().getAddressSpace() != getInput().getType().getMemorySpace())
    return emitError("address space mismatch");
  return success();
}

LogicalResult FromMemRefOp::canonicalize(FromMemRefOp op,
                                         PatternRewriter &rewriter) {
  // Collapse the following patterns to an address:
  // 1) Result %a = %addr
  //   %m = addr.to_memref %addr base %base : memref<i32>
  //   %a = addr.from_memref [%m : memref<i32>]
  // 2) Result %a = %base
  //   %m = addr.to_memref %addr base %base : memref<i32>
  //   %a = addr.from_memref extract_base [%m : memref<i32>]
  // 3) Result %a = %addr
  //   %m = addr.to_memref %addr : memref<i32>
  //   %a = addr.from_memref extract_base [%m : memref<i32>]
  auto input = dyn_cast_or_null<ToMemRefOp>(op.getInput().getDefiningOp());
  if (!input)
    return failure();
  // Handle cases 1 & 3
  if (!op.getExtractBase() || !input.getBase())
    rewriter.replaceOp(op, input.getAddress());
  else
    rewriter.replaceOp(op, input.getBase());
  return success();
}

namespace {
ParseResult parseFromMemRef(OpAsmParser &parser, Type &inputTy,
                            Type &resultTy) {
  if (parser.parseColonType(inputTy))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  auto memrefTy = dyn_cast<MemRefType>(inputTy);
  assert(memrefTy && "Expected a memref type.");
  resultTy =
      parser.getBuilder().getType<AddressType>(memrefTy.getMemorySpace());
  return success();
}
void printFromMemRef(OpAsmPrinter &p, Operation *op, Type inputTy,
                     Type resultTy) {
  p << " : " << inputTy;
}

ParseResult parseFromUnrankedMemRef(OpAsmParser &parser, Type &inputTy,
                                    Type &resultTy) {
  if (parser.parseColonType(inputTy))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  auto memrefTy = dyn_cast<UnrankedMemRefType>(inputTy);
  assert(memrefTy && "Expected a memref type.");
  resultTy =
      parser.getBuilder().getType<AddressType>(memrefTy.getMemorySpace());
  return success();
}
void printFromUnrankedMemRef(OpAsmPrinter &p, Operation *op, Type inputTy,
                             Type resultTy) {
  p << " : " << inputTy;
}
} // namespace

namespace {} // namespace

//===----------------------------------------------------------------------===//
// ToMemRef Op
//===----------------------------------------------------------------------===//

void ToMemRefOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                       MemRefType type, Value address) {
  build(odsBuilder, odsState, type, address, nullptr);
}

void ToUnrankedMemRefOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                               UnrankedMemRefType type, Value address) {
  build(odsBuilder, odsState, type, address, nullptr);
}

LogicalResult ToMemRefOp::verify() {
  Attribute inputAS = getAddress().getType().getAddressSpace();
  if (inputAS != getType().getMemorySpace() ||
      (getBase() &&
       dyn_cast<AddressType>(getBase().getType()).getAddressSpace() != inputAS))
    return emitError("address space mismatch");
  return success();
}

LogicalResult ToMemRefOp::canonicalize(ToMemRefOp op,
                                       PatternRewriter &rewriter) {
  // Collapse the following pattern to a memref, where %m = %memref:
  //   %a = addr.from_memref [%memref : memref<i32>]
  //   %b = addr.from_memref extract_base [%memref : memref<i32>]
  //   %m = addr.to_memref %a base %b : memref<i32>
  auto address =
      dyn_cast_or_null<FromMemRefOp>(op.getAddress().getDefiningOp());
  // Fail if the address doesn't come from a `from_memref` or if the Op doesn't
  // have a base.
  if (!address || !op.getBase())
    return failure();
  auto base = dyn_cast_or_null<FromMemRefOp>(op.getBase().getDefiningOp());
  // Fail if the base doesn't come from a `from_memref` or if the base is
  // unknown.
  if (!base || address.getInput() != base.getInput() || !base.getExtractBase())
    return failure();
  rewriter.replaceOp(op, address.getInput());
  return success();
}

namespace {
ParseResult parseToMemRef(OpAsmParser &parser,
                          std::optional<OpAsmParser::UnresolvedOperand> &base,
                          Type &baseTy, Type &addressTy, Type &resultTy) {
  if (succeeded(parser.parseOptionalKeyword("base")) &&
      parser.parseOperand(base.emplace()))
    return parser.emitError(parser.getNameLoc(), "expected an operand");
  if (parser.parseColonType(resultTy))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  auto memrefTy = dyn_cast<MemRefType>(resultTy);
  assert(memrefTy && "Expected a memref type.");
  addressTy =
      parser.getBuilder().getType<AddressType>(memrefTy.getMemorySpace());
  if (base.has_value())
    baseTy = addressTy;
  return success();
}
void printToMemRef(OpAsmPrinter &p, Operation *op, Value base, Type baseTy,
                   Type addressTy, Type memrefTy) {
  if (base)
    p << "base " << base;
  p << " : " << memrefTy;
}

ParseResult
parseToUnrankedMemRef(OpAsmParser &parser,
                      std::optional<OpAsmParser::UnresolvedOperand> &base,
                      Type &baseTy, Type &addressTy, Type &resultTy) {
  if (succeeded(parser.parseOptionalKeyword("base")) &&
      parser.parseOperand(base.emplace()))
    return parser.emitError(parser.getNameLoc(), "expected an operand");
  if (parser.parseColonType(resultTy))
    return parser.emitError(parser.getNameLoc(), "expected a type");
  auto memrefTy = dyn_cast<UnrankedMemRefType>(resultTy);
  assert(memrefTy && "Expected a memref type.");
  addressTy =
      parser.getBuilder().getType<AddressType>(memrefTy.getMemorySpace());
  if (base.has_value())
    baseTy = addressTy;
  return success();
}
void printToUnrankedMemRef(OpAsmPrinter &p, Operation *op, Value base,
                           Type baseTy, Type addressTy, Type memrefTy) {
  if (base)
    p << "base " << base;
  p << " : " << memrefTy;
}
} // namespace

#include "Address/Dialect/IR/AddressOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Address/Dialect/IR/AddressOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Address/Dialect/IR/AddressOpsTypes.cpp.inc"
