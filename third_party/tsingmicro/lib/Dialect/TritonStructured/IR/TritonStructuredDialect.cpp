#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"

using namespace mlir;
using namespace mlir::tts;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void TritonStructuredDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredOps.cpp.inc"

#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.cpp.inc"
