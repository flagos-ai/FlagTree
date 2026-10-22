#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`
#include <optional>
#include <utility>

using namespace mlir;
using namespace mlir::triton;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Triton/IR/Types.cpp.inc"
      >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  int addressSpace = 1;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return PointerType::get(pointeeType, addressSpace);
}

void PointerType::print(AsmPrinter &printer) const {
  if (getAddressSpace() == 1) {
    printer << "<" << getPointeeType() << ">";
  } else {
    printer << "<" << getPointeeType() << ", " << getAddressSpace() << ">";
  }
}

namespace mlir {

namespace triton {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = dyn_cast<RankedTensorType>(pointeeType))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

std::optional<std::pair<int64_t, int64_t>> getTypeSizeAlign(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type)) {
    int64_t bytes = std::max<int64_t>(1, (intTy.getWidth() + 7) / 8);
    return std::make_pair(bytes, bytes);
  }
  if (auto floatTy = dyn_cast<FloatType>(type)) {
    int64_t bytes = std::max<int64_t>(1, (int64_t)floatTy.getWidth() / 8);
    return std::make_pair(bytes, bytes);
  }
  if (isa<PointerType>(type)) {
    // The host layout is authoritative; supported targets are 64-bit.
    return std::make_pair((int64_t)8, (int64_t)8);
  }
  if (auto arrTy = dyn_cast<ArrayType>(type)) {
    auto elem = getTypeSizeAlign(arrTy.getElementType());
    if (!elem)
      return std::nullopt;
    return std::make_pair(elem->first * arrTy.getCount(), elem->second);
  }
  if (auto structTy = dyn_cast<StructType>(type)) {
    return std::make_pair(structTy.getSize(), structTy.getAlign());
  }
  return std::nullopt;
}

} // namespace triton
} // namespace mlir

// !tt.struct<{name: type:offset, ...}, size = N, align = N>
Type StructType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<StringAttr> fieldNames;
  SmallVector<Type> fieldTypes;
  SmallVector<int64_t> fieldOffsets;

  if (parser.parseLBrace())
    return Type();
  if (failed(parser.parseOptionalRBrace())) {
    do {
      StringRef name;
      Type fieldType;
      int64_t offset = 0;
      if (parser.parseKeyword(&name) || parser.parseColon() ||
          parser.parseType(fieldType) || parser.parseColon() ||
          parser.parseInteger(offset))
        return Type();
      fieldNames.push_back(StringAttr::get(parser.getContext(), name));
      fieldTypes.push_back(fieldType);
      fieldOffsets.push_back(offset);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRBrace())
      return Type();
  }

  int64_t size = 0;
  int64_t align = 0;
  if (parser.parseComma() || parser.parseKeyword("size") ||
      parser.parseEqual() || parser.parseInteger(size) || parser.parseComma() ||
      parser.parseKeyword("align") || parser.parseEqual() ||
      parser.parseInteger(align))
    return Type();

  if (parser.parseGreater())
    return Type();

  return StructType::get(parser.getContext(), fieldNames, fieldTypes,
                         fieldOffsets, size, align);
}

void StructType::print(AsmPrinter &printer) const {
  auto names = getFieldNames();
  auto types = getFieldTypes();
  auto offsets = getFieldOffsets();
  printer << "<{";
  for (unsigned i = 0; i < names.size(); ++i) {
    if (i)
      printer << ", ";
    printer << names[i].getValue() << ": " << types[i] << ":" << offsets[i];
  }
  printer << "}, size = " << getSize() << ", align = " << getAlign() << ">";
}

LogicalResult ArrayType::verify(function_ref<InFlightDiagnostic()> emitError,
                                Type elementType, int64_t count) {
  if (count < 1)
    return emitError() << "array element count must be >= 1";
  if (!getTypeSizeAlign(elementType))
    return emitError() << "unsupported array element type " << elementType;
  return success();
}

LogicalResult StructType::verify(function_ref<InFlightDiagnostic()> emitError,
                                 ArrayRef<StringAttr> fieldNames,
                                 ArrayRef<Type> fieldTypes,
                                 ArrayRef<int64_t> fieldOffsets, int64_t size,
                                 int64_t align) {
  if (fieldNames.size() != fieldTypes.size() ||
      fieldNames.size() != fieldOffsets.size())
    return emitError()
           << "struct field name/type/offset lists must have equal length";
  if (fieldNames.empty())
    return emitError() << "struct must have at least one field";
  if (size <= 0)
    return emitError() << "struct size must be positive";
  if (align <= 0)
    return emitError() << "struct alignment must be positive";

  for (unsigned i = 0; i < fieldNames.size(); ++i)
    for (unsigned j = i + 1; j < fieldNames.size(); ++j)
      if (fieldNames[i] == fieldNames[j])
        return emitError() << "duplicate struct field name '"
                           << fieldNames[i].getValue() << "'";

  int64_t prevEnd = 0;
  for (unsigned i = 0; i < fieldNames.size(); ++i) {
    int64_t offset = fieldOffsets[i];
    auto sizeAlign = getTypeSizeAlign(fieldTypes[i]);
    if (!sizeAlign)
      return emitError() << "field '" << fieldNames[i].getValue()
                         << "' has unsupported type " << fieldTypes[i];
    if (offset < 0 || offset >= size)
      return emitError() << "field '" << fieldNames[i].getValue() << "' offset "
                         << offset << " is out of range [0, " << size << ")";
    if (offset % sizeAlign->second != 0)
      return emitError() << "field '" << fieldNames[i].getValue() << "' offset "
                         << offset << " is not a multiple of its alignment "
                         << sizeAlign->second;
    if (i > 0 && offset < prevEnd)
      return emitError() << "field '" << fieldNames[i].getValue()
                         << "' overlaps the previous field";
    prevEnd = offset + sizeAlign->first;
  }

  if (prevEnd > size)
    return emitError() << "fields require at least " << prevEnd
                       << " bytes but struct size is " << size;
  if (size % align != 0)
    return emitError() << "struct size " << size
                       << " is not a multiple of alignment " << align;
  return success();
}

namespace mlir {

namespace triton {

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i1Type);
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto ptrType = dyn_cast<PointerType>(tensorTy.getElementType());
    Type pointeeType = ptrType.getPointeeType();
    return tensorTy.clone(pointeeType);
  } else if (auto ptrType = dyn_cast<PointerType>(type)) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i32Type);
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorTy.getElementType();
    PointerType ptrType = PointerType::get(elementType, 1);
    return tensorTy.clone(ptrType);
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerTypeToElement(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  PointerType ptrType = PointerType::get(elementType, 1);
  return ptrType;
}

// upstream Triton only uses address space 1 for Pointer Type
Type getPointerType(Type type, int addressSpace) {
  return PointerType::get(type, addressSpace);
}

int getAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return ptrType.getAddressSpace();
  return 1;
}

bool isTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return isa<RankedTensorType>(ptrType.getPointeeType());
  return false;
}

bool isTensorOrTensorPointerType(Type type) {
  return isa<RankedTensorType>(type) || isTensorPointerType(type);
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return tensorTy.getElementType();
  return {};
}

} // namespace triton

} // namespace mlir
