// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_DIALECT_TRITON_IR_DIALECT_H_
#define TRITON_DIALECT_TRITON_IR_DIALECT_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h.inc"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/OpsEnums.h.inc"
#include "triton/Dialect/Triton/IR/Traits.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define GET_OP_CLASSES
#include "triton/Dialect/Triton/IR/Ops.h.inc"

namespace mlir {
namespace triton {

struct GlobalMemory : public SideEffects::Resource::Base<GlobalMemory> {
  StringRef getName() final { return "<GlobalMemory>"; }
};
#ifdef __TLE__
struct SharedMemory : public SideEffects::Resource::Base<SharedMemory> {
  StringRef getName() final { return "<SharedMemory>"; }
};
#endif

class DialectInferLayoutInterface
    : public DialectInterface::Base<DialectInferLayoutInterface> {
public:
  DialectInferLayoutInterface(Dialect *dialect) : Base(dialect) {}

  virtual LogicalResult
  inferTransOpEncoding(Attribute operandEncoding, ArrayRef<int64_t> shape,
                       ArrayRef<int32_t> order, Attribute &resultEncoding,
                       std::optional<Location> loc) const = 0;

  virtual LogicalResult
  inferReduceOpEncoding(Attribute operandEncoding, unsigned axis,
                        Attribute &resultEncoding,
                        std::optional<Location> loc) const = 0;

  virtual LogicalResult
  inferExpandDimsOpEncoding(Attribute operandEncoding, unsigned axis,
                            Attribute &resultEncoding,
                            std::optional<Location> loc) const = 0;

  // Note: This function only verifies the operand encoding.  It doesn't infer
  // the result encoding.
  virtual LogicalResult
  inferDotOpEncoding(Attribute operandEncoding, unsigned opIdx,
                     Attribute retEncoding,
                     std::optional<Location> loc) const = 0;

  // Tries to compute the encoding for the result of a reshape operation that
  // makes the reshape a "nop", i.e. the same GPU threads contain the same
  // elements as before the reshape using legacy layouts.  This is not always
  // possible (in which case we fallback to using LinearLayouts)
  // In the future we'll always use LinearLayouts
  virtual LogicalResult
  inferReshapeOpEncoding(ArrayRef<int64_t> srcShape, Attribute srcEnc,
                         ArrayRef<int64_t> dstShape, Attribute &dstEnc,
                         std::optional<Location> loc) const = 0;

  // Check if two layouts are structurally the same, even if their names are
  // different
  virtual LogicalResult
  verifyLayoutsAreEqual(ArrayRef<int64_t> shape, Attribute expected,
                        Attribute got, std::optional<Location> loc) const = 0;

  virtual LogicalResult
  inferDefaultJoinOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                             ArrayRef<int64_t> shape,
                             std::optional<Location> loc) const = 0;

  virtual LogicalResult
  inferSplitOpEncoding(Attribute srcEnc, Attribute &dstEnc,
                       ArrayRef<int64_t> shape,
                       std::optional<Location> loc) const = 0;

  // Verify that the encoding are compatible to be used together in a dot
  // operation
  virtual LogicalResult
  verifyDotOpEncodingCompatibility(Operation *op, Attribute operandEncodingA,
                                   Attribute operandEncodingB) const = 0;

  virtual LogicalResult
  inferFp4ToFpOpEncoding(ArrayRef<int64_t> shape, int axis, Attribute inEnc,
                         Attribute &outEnc, bool fwdInference,
                         std::optional<Location> loc) const = 0;
};

class DialectVerifyTensorLayoutInterface
    : public DialectInterface::Base<DialectVerifyTensorLayoutInterface> {
public:
  DialectVerifyTensorLayoutInterface(Dialect *dialect) : Base(dialect) {}

  virtual LogicalResult
  verifyTensorLayout(Attribute layout, RankedTensorType type, Operation *op,
                     function_ref<InFlightDiagnostic()> emitError) const = 0;

  virtual LogicalResult
  verifyMemDescLayout(Attribute layout, Type type, Operation *op,
                      function_ref<InFlightDiagnostic()> emitError) const = 0;
};

// Descriptor gather and scatter have restrictions on the tile sizes.
LogicalResult verifyGatherScatterOp(Operation *op, ShapedType blockType,
                                    ShapedType resultType,
                                    ShapedType indicesType);
LogicalResult verifyDescriptorLoadStoreOp(Operation *op,
                                          TensorDescInterface desc,
                                          ShapedType tensor);

} // namespace triton
} // namespace mlir

#endif // TRITON_IR_DIALECT_H_
