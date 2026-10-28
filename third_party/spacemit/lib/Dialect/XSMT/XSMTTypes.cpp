//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::xsmt;

#define GET_TYPEDEF_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.cpp.inc"
#undef GET_TYPEDEF_CLASSES

void XSMTDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.cpp.inc"
#undef GET_TYPEDEF_LIST
      >();
}

mlir::Type BufferType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  SmallVector<int64_t> shape;
  if (parser.parseKeyword("shape") || parser.parseEqual() ||
      parser.parseLSquare())
    return Type();

  int64_t dim;
  if (parser.parseInteger(dim))
    return Type();
  shape.push_back(dim);
  while (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(dim))
      return Type();
    shape.push_back(dim);
  }

  if (parser.parseRSquare())
    return Type();
  Type elementType;
  if (parser.parseComma() || parser.parseKeyword("elementType") ||
      parser.parseEqual() || parser.parseType(elementType))
    return Type();

  int64_t copies64 = 0;
  StringRef kw;
  if (parser.parseComma() || parser.parseKeyword(&kw) || parser.parseEqual() ||
      parser.parseInteger(copies64))
    return Type();
  if (kw != "copies" && kw != "numBuffers") {
    parser.emitError(parser.getCurrentLocation(),
                     "expected copies=... or numBuffers=...");
    return Type();
  }
  int copies = (int)copies64;

  StringAttr scopeKind;
  if (parser.parseComma() || parser.parseKeyword("scopeKind") ||
      parser.parseEqual() || parser.parseAttribute(scopeKind))
    return Type();

  if (parser.parseGreater())
    return Type();

  return BufferType::get(parser.getContext(), shape, elementType, copies,
                         scopeKind);
}

void BufferType::print(AsmPrinter &printer) const {
  printer << "<shape=[";
  llvm::interleaveComma(getShape(), printer);
  printer << "], elementType=" << getElementType();
  printer << ", copies=" << getCopies();
  printer << ", scopeKind=" << getScopeKind();
  printer << ">";
}

mlir::ShapedType
BufferType::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
                      mlir::Type elementType) const {
  auto newTy = BufferType::get(getContext(), shape.value_or(getShape()),
                               elementType, getCopies(), getScopeKind());
  return mlir::cast<mlir::ShapedType>(newTy);
}

bool BufferType::hasRank() const { return true; }
