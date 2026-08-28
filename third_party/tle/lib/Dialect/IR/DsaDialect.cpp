//===- DsaDialect.cpp - TLE DSA dialect -------------------------*- C++ -*-===//
//
// Template dialect for TLE-Struct style DSA extensions.
//
//===----------------------------------------------------------------------===//

#include "tle-dsa/Dialect/IR/DsaDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::dsa {

void DsaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tle-dsa/Dialect/IR/DsaOps.cpp.inc"
      >();
  registerTypes();
}

void DsaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tle-dsa/Dialect/IR/DsaOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::dsa

#include "tle-dsa/Dialect/IR/DsaOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "tle-dsa/Dialect/IR/DsaOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tle-dsa/Dialect/IR/DsaOpsTypes.cpp.inc"

LogicalResult mlir::dsa::BitcastOp::verify() {
  auto srcTy = dyn_cast<RankedTensorType>(getSrc().getType());
  auto dstTy = dyn_cast<RankedTensorType>(getResult().getType());
  if (!srcTy || !dstTy)
    return emitOpError("expects ranked tensor src/result");
  auto srcElem = srcTy.getElementType();
  auto dstElem = dstTy.getElementType();
  if (!srcElem.isIntOrFloat() || !dstElem.isIntOrFloat())
    return emitOpError("element types must be int or float");
  int64_t srcBits = srcTy.getNumElements() * srcElem.getIntOrFloatBitWidth();
  int64_t dstBits = dstTy.getNumElements() * dstElem.getIntOrFloatBitWidth();
  if (srcBits != dstBits)
    return emitOpError("src and result must have the same total bit size, got ")
           << srcBits << " vs " << dstBits;
  return success();
}

