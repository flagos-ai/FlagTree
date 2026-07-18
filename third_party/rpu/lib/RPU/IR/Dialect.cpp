#include "RPU/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"

#include "RPU/IR/RPUDialect.cpp.inc"

using namespace mlir;
using namespace mlir::rpu::exec;

void RPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "RPU/IR/RPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "RPU/IR/RPUOps.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "RPU/IR/RPUTypes.cpp.inc"

Type TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  SmallVector<int64_t, 2> dimensions;
  if (parser.parseDimensionList(dimensions, /*allowDynamic=*/false))
    return Type();
  if (dimensions.empty() || dimensions.size() > 2)
    return Type();

  Type elementType;
  if (parser.parseType(elementType))
    return Type();

  if (parser.parseGreater())
    return Type();

  return TileType::get(parser.getContext(), dimensions, elementType);
}

void TileType::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleave(getShape(), printer, "x");
  printer << "x" << getElementType() << ">";
}
