/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/Gluon/IR/Dialect.h"

#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Interfaces.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::triton::gpu;
namespace gluon = mlir::triton::gluon;

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/Gluon/IR/Dialect.cpp.inc"
#include "triton/Dialect/Gluon/IR/GluonAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/Gluon/IR/Ops.cpp.inc"

namespace {

// Layout inference for AutoEncodingAttr -> always propagate AutoEncodingAttr to
// results
struct GluonInferLayoutInterface : public triton::DialectInferLayoutInterface {
  using DialectInferLayoutInterface::DialectInferLayoutInterface;

  LogicalResult inferAutoEncoding(Attribute operandEncoding,
                                  Attribute &resultEncoding) const {
    if (!isa<gluon::AutoEncodingAttr, gluon::CoalescedEncodingAttr>(
            operandEncoding))
      return failure();
    resultEncoding = operandEncoding;
    return success();
  }

  LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> location) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute resultEncoding,
                     std::optional<Location> location) const override {
    return inferAutoEncoding(operandEncoding, resultEncoding);
  }

  LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const override {
    return success();
  }

  LogicalResult verifyCatOpEncodingCompatibility(Operation *op) const override {
    return success();
  }

  LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got,
                        std::optional<Location> loc) const override {
    return success(expected == got);
  }

  LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc, bool,
                         std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }

  LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute srcEnc,
                         Attribute &dstEnc, bool fwdInference,
                         std::optional<Location> loc) const override {
    return inferAutoEncoding(srcEnc, dstEnc);
  }
};
} // namespace

namespace mlir::triton::gluon {

void GluonDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/Gluon/IR/GluonAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/Gluon/IR/Ops.cpp.inc"
      >();
  addInterfaces<TritonInlinerInterface>();
  addInterfaces<GluonInferLayoutInterface>();
}

void SetAutoLayoutOp::build(OpBuilder &builder, OperationState &state,
                            Attribute enc, Value value) {
  auto resTy = cast<RankedTensorType>(value.getType()).cloneWithEncoding(enc);
  return build(builder, state, resTy, value);
}

LogicalResult SetAutoLayoutOp::verify() {
  if (!isa<gluon::AutoEncodingAttr>(getSrc().getType().getEncoding())) {
    return emitOpError("input tensor must have an auto layout type");
  }
  auto dstEncoding = getType().getEncoding();
  if (!dstEncoding)
    return emitOpError("result tensor must have an encoding");
  if (isa<gluon::AutoEncodingAttr>(dstEncoding))
    return emitOpError("result type must not be auto layout");
  return success();
}

} // namespace mlir::triton::gluon
