#include "mctle/dialect/include/IR/Dialect.h"
#include "mctle/dialect/include/IR/Dialect.cpp.inc"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/InliningUtils.h"

#define GET_ATTRDEF_CLASSES
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mctle/dialect/include/IR/Ops.cpp.inc"

namespace mlir::triton::mctle {
namespace {
struct TensorTileInlinerInterface : DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *op, Region *, bool, IRMapping &) const final {
    return isa<ExtractTileOp, InsertTileOp>(op);
  }
};
} // namespace

void McTleDialect::initialize() {
  addInterfaces<TensorTileInlinerInterface>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mctle/dialect/include/IR/Ops.cpp.inc"
      >();
}
} // namespace mlir::triton::mctle
