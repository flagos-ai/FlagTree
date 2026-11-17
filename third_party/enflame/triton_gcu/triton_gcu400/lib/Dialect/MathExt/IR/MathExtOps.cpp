/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "mlir/Support/LogicalResult.h"

#include "Dialect/MathExt/IR/MathExt.h"

#define GET_OP_CLASSES
#include "Dialect/MathExt/IR/MathExtOps.cpp.inc"

namespace mlir {

// Match an integer constant produced by `arith.constant`.
static bool matchConstantInt(Value v, int64_t &out) {
  auto cst = v.getDefiningOp<arith::ConstantOp>();
  if (!cst)
    return false;
  auto attr = dyn_cast<IntegerAttr>(cst.getValue());
  if (!attr)
    return false;
  out = attr.getInt();
  return true;
}

// True when the histogram region is the identity `^bb(%e): yield %e`, i.e.
// the op counts raw bin indices and the region carries no arithmetic.
static bool isIdentityDigitRegion(Region &region) {
  if (!llvm::hasSingleElement(region))
    return false;
  Block &block = region.front();
  if (block.getNumArguments() != 1 || block.getOperations().size() != 1)
    return false;
  auto yield = dyn_cast<math_ext::YieldOp>(block.getTerminator());
  return yield && yield.getNumOperands() == 1 &&
         yield.getOperand(0) == block.getArgument(0);
}

namespace math_ext {

LogicalResult HistogramOp::verify() {
  auto keyTy = dyn_cast<MemRefType>(getKey().getType());
  auto resTy = dyn_cast<MemRefType>(getResult().getType());
  if (!keyTy || keyTy.getRank() != 1)
    return emitOpError() << "histogram key must be a rank-1 memref";
  if (!resTy || resTy.getRank() != 1 || !resTy.hasStaticShape() ||
      resTy.getShape()[0] <= 0)
    return emitOpError()
           << "histogram result must be a static rank-1 memref with a "
              "positive length";

  if (!getNumBins().getType().isInteger(32) ||
      !getSizeIn().getType().isInteger(32))
    return emitOpError() << "histogram num_bins and size_in must be i32";

  int64_t numBins = 0;
  bool numBinsIsConst = matchConstantInt(getNumBins(), numBins);
  if (numBinsIsConst && numBins <= 0)
    return emitOpError() << "histogram num_bins must be positive";

  int64_t sizeIn = 0;
  if (matchConstantInt(getSizeIn(), sizeIn) && sizeIn < 0)
    return emitOpError() << "histogram size_in must be non-negative";

  if (Value mask = getMask()) {
    auto maskTy = dyn_cast<MemRefType>(mask.getType());
    if (!maskTy || maskTy.getRank() != 1)
      return emitOpError() << "histogram mask must be a rank-1 memref";
    if (!maskTy.getElementType().isInteger(1) &&
        !maskTy.getElementType().isInteger(8))
      return emitOpError() << "histogram mask must have i1 or i8 elements";
    if (maskTy.hasStaticShape() && keyTy.hasStaticShape() &&
        maskTy.getShape() != keyTy.getShape())
      return emitOpError() << "histogram mask length must match the key buffer";
  }

  Region &body = getBody();
  if (!llvm::hasSingleElement(body))
    return emitOpError() << "histogram requires a single-block region";
  Block &block = body.front();
  Type keyElemTy = keyTy.getElementType();
  if (block.getNumArguments() != 1 ||
      block.getArgument(0).getType() != keyElemTy)
    return emitOpError() << "histogram region must take one scalar "
                         << keyElemTy << " key element";
  auto yield = dyn_cast<YieldOp>(block.getTerminator());
  if (!yield || yield.getNumOperands() != 1)
    return emitOpError() << "histogram region must yield exactly one value";

  bool identity = isIdentityDigitRegion(body);
  Type yieldedTy = yield.getOperand(0).getType();
  if (identity) {
    if (yieldedTy != keyElemTy)
      return emitOpError() << "the identity histogram region must yield one "
                           << keyElemTy << " value";
  } else if (!yieldedTy.isInteger(32)) {
    return emitOpError() << "a non-identity histogram region must yield one "
                            "i32 digit";
  }

  // A non-identity region is always counted by the MRADIX unit; the identity
  // form only is when `use_mradix` says so (otherwise the generic builtin
  // counts the raw values).  The common builtin currently only exists for
  // i32 keys and results.
  bool wantsMradix = identity ? static_cast<bool>(getUseMradix()) : true;
  if (wantsMradix) {
    // Raw-index MRADIX counting consumes integer bin indices; only a
    // non-identity digit region may start from a 32-bit float key (the region
    // bitcasts it before the digit arithmetic).
    bool keyOk = identity ? keyElemTy.isInteger(32)
                          : keyElemTy.getIntOrFloatBitWidth() == 32;
    if (!keyOk || !resTy.getElementType().isInteger(32))
      return emitOpError() << "the MRADIX histogram path requires 32-bit keys "
                              "and i32 result elements";
    if (numBinsIsConst && numBins > 32)
      return emitOpError() << "the MRADIX histogram path requires num_bins in "
                              "(0, 32]";
    if (resTy.getShape()[0] > 128)
      return emitOpError() << "the MRADIX histogram path requires the result "
                              "to fit in one OACC line (128 i32)";
  } else {
    if (!keyElemTy.isInteger(32) || !resTy.getElementType().isInteger(32))
      return emitOpError() << "the common histogram path requires i32 key and "
                              "result elements";
  }
  return success();
}

LogicalResult InclusiveScanOp::verify() {
  auto resultType = getResult().getType();
  auto operandType = getOperand().getType();
  if (resultType.getRank() != 1)
    return emitOpError() << "only 1D memrefs are supported (result rank="
                         << resultType.getRank() << ")";
  if (operandType.getRank() != 1)
    return emitOpError() << "only 1D memrefs are supported (operand rank="
                         << operandType.getRank() << ")";
  if (resultType.getShape() != operandType.getShape())
    return emitOpError() << "result and operand must have the same shape";
  if (getUseMiota() && !operandType.getElementType().isInteger(32))
    return emitOpError() << "miota acceleration requires i32 element type";
  return success();
}
} // namespace math_ext
} // namespace mlir
