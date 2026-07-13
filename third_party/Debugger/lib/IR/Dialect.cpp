#include "Debugger/IR/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Operation.h"

namespace mlir::flagtree::debugger {

void FlagTreeDebugDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Debugger/IR/Ops.cpp.inc"
      >();
}

::mlir::Attribute
FlagTreeDebugDialect::parseAttribute(::mlir::DialectAsmParser &parser,
                                     ::mlir::Type) const {
  parser.emitError(parser.getCurrentLocation(),
                   "unknown attribute for flagtree_debug dialect");
  return {};
}

void FlagTreeDebugDialect::printAttribute(::mlir::Attribute attr,
                                          ::mlir::DialectAsmPrinter &os) const {
  (void)attr;
  os << "<<flagtree_debug attribute>>";
}

} // namespace mlir::flagtree::debugger

#define GET_DIALECT_CLASSES
#include "Debugger/IR/Dialect.cpp.inc"
