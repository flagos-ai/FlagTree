/*
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>

#include "tle/dialect/include/IR/VerifyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include <iostream>
#include <optional>

namespace mlir::triton::tle {
namespace {
std::optional<int64_t> getConstantIntValue(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return std::nullopt;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return std::nullopt;
  return integer.getInt();
}
} // namespace

namespace RemotePointers {
llvm::LogicalResult verifyDeviceSpace(mlir::Value src, mlir::Value result) {
  if (!src)
    return success();

  if (auto tensorTy = dyn_cast<RankedTensorType>(result.getType())) {
    auto ptr = dyn_cast<triton::PointerType>(tensorTy.getElementType());
    if (!ptr)
      return failure();
    return success();
  }
  return success();
}

llvm::LogicalResult verifyNodeSpace(RemotePointersOp op) {
  if (Value result = op.getResult()) {
    auto requireMarkerOperand = [&](Value value,
                                    StringRef name) -> LogicalResult {
      if (!value)
        return op.emitOpError()
               << "node remote pointer marker requires " << name
               << " operand";
      return success();
    };
    if (failed(requireMarkerOperand(op.getSrc(), "src")) ||
        failed(requireMarkerOperand(op.getComm(), "comm")) ||
        failed(requireMarkerOperand(op.getNetIdx(), "net_idx")))
      return failure();
    if (op.getDstMem() || op.getOffset() || op.getDstOffset() ||
        op.getNelems() || op.getElemBytesAttr() || op.getTransferKindAttr())
      return op.emitOpError()
             << "node remote pointer marker does not accept transfer operands "
                "or attributes";
    if (!op.getSrc().getType().isSignlessInteger(64) ||
        !op.getComm().getType().isSignlessInteger(64))
      return op.emitOpError()
             << "expects node marker src and comm to be i64 handles";
    if (!op.getNetIdx().getType().isSignlessInteger(32))
      return op.emitOpError() << "expects node marker net_idx to be i32";
    auto coopKindAttr = op.getCoopkindAttr();
    if (!coopKindAttr || coopKindAttr.getInt() < 0 ||
        coopKindAttr.getInt() > 2)
      return op.emitOpError()
             << "expects coopkind to be THREAD(0), WARP(1), or BLOCK(2)";
    auto ptrTy = dyn_cast<triton::PointerType>(result.getType());
    if (!ptrTy || ptrTy.getAddressSpace() != 1)
      return op.emitOpError()
             << "node remote pointer marker must produce a scalar global "
                "pointer to an integer or floating-point element";
    Type pointeeTy = ptrTy.getPointeeType();
    unsigned bitWidth = 0;
    if (auto intTy = dyn_cast<IntegerType>(pointeeTy))
      bitWidth = intTy.getWidth();
    else if (auto floatTy = dyn_cast<FloatType>(pointeeTy))
      bitWidth = floatTy.getWidth();
    if (bitWidth < 8 || bitWidth % 8 != 0)
      return op.emitOpError()
             << "node remote pointer marker element type must be "
                "byte-addressable";
    if (std::optional<int64_t> peer = getConstantIntValue(op.getShardId());
        peer && *peer < 0)
      return op.emitOpError() << "expects constant peer to be >= 0";
    if (std::optional<int64_t> netIdx = getConstantIntValue(op.getNetIdx());
        netIdx && *netIdx < 0)
      return op.emitOpError() << "expects constant net_idx to be >= 0";
    return success();
  }

  auto transferKindAttr = op->getAttrOfType<StringAttr>("transfer_kind");
  if (!transferKindAttr || (transferKindAttr.getValue() != "put" &&
                            transferKindAttr.getValue() != "get"))
    return op.emitOpError() << "expects transfer_kind to be 'put' or 'get'";

  auto requireOperand = [&](Value value, StringRef name) -> LogicalResult {
    if (!value)
      return op.emitOpError() << "node space requires " << name << " operand";
    return success();
  };
  if (failed(requireOperand(op.getSrc(), "src")) ||
      failed(requireOperand(op.getDstMem(), "dst_mem")) ||
      failed(requireOperand(op.getComm(), "comm")) ||
      failed(requireOperand(op.getOffset(), "offset")) ||
      failed(requireOperand(op.getDstOffset(), "dst_offset")) ||
      failed(requireOperand(op.getNelems(), "nelems")) ||
      failed(requireOperand(op.getNetIdx(), "net_idx")))
    return failure();

  if (!op.getSrc().getType().isSignlessInteger(64))
    return op.emitOpError()
           << "expects node source to be an i64 registered-memory handle";

  auto elemBytesAttr = op->getAttrOfType<IntegerAttr>("elem_bytes");
  if (!elemBytesAttr || elemBytesAttr.getInt() <= 0)
    return op.emitOpError() << "expects elem_bytes to be > 0";

  auto coopKindAttr = op.getCoopkindAttr();
  if (!coopKindAttr || coopKindAttr.getInt() < 0 || coopKindAttr.getInt() > 2)
    return op.emitOpError()
           << "expects coopkind to be THREAD(0), WARP(1), or BLOCK(2)";

  auto verifyNonNegativeConstant = [&](Value value,
                                       StringRef name) -> LogicalResult {
    if (std::optional<int64_t> constant = getConstantIntValue(value);
        constant && *constant < 0)
      return op.emitOpError() << "expects constant " << name << " to be >= 0";
    return success();
  };

  if (failed(verifyNonNegativeConstant(op.getShardId(), "peer")) ||
      failed(verifyNonNegativeConstant(op.getOffset(), "offset")) ||
      failed(verifyNonNegativeConstant(op.getDstOffset(), "dst_offset")) ||
      failed(verifyNonNegativeConstant(op.getNetIdx(), "net_idx")))
    return failure();

  if (std::optional<int64_t> nelems = getConstantIntValue(op.getNelems());
      nelems && *nelems < 0)
    return op.emitOpError() << "expects constant nelems to be >= 0";

  Type offsetTy = op.getOffset().getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(offsetTy)) {
    if (!tensorTy.getShape().empty())
      return op.emitOpError() << "expects offset to be a scalar";
    offsetTy = tensorTy.getElementType();
  }
  if (!offsetTy.isSignlessInteger(64))
    return op.emitOpError() << "expects offset to be i64";

  return success();
}
} // namespace RemotePointers

namespace DistributedBarrier {
llvm::LogicalResult verifyDeviceSpace(mlir::Operation *op, mlir::Value src) {

  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto barrierTypeAttr = op->getAttrOfType<StringAttr>("barrier_type");
  auto orderAttr = op->getAttrOfType<StringAttr>("order");

  if (kindAttr && barrierTypeAttr && orderAttr)
    return success();
  else
    return op->emitOpError()
           << "expects src, group_kind, barrier_type and order attributes to "
              "be present for device space distributed barrier";
}

} // namespace DistributedBarrier

namespace Signal {
std::optional<std::string> verifySignalOp(SignalOpKind kind,
                                          mlir::Value value) {
  switch (kind) {
  case SignalOpKind::INC:
    if (value)
      return "value shouldn't be provided when op is 'inc'";
    break;
  case SignalOpKind::ADD:
    if (!value)
      return "value must be provided when op is 'add'";
    break;
  }
  return std::nullopt;
}

std::optional<std::string> verifySignalWaitOp(SignalWaitKind kind,
                                              mlir::Value target) {
  switch (kind) {
  case SignalWaitKind::SIGNAL:
  case SignalWaitKind::COUNTER:
    if (!target)
      return "target must be provided when wait_kind is signal or counter";
    break;
  case SignalWaitKind::SHADOW:
    if (target)
      return "target shouldn't be provided when wait_kind is shadow";
    break;
  }
  return std::nullopt;
}
} // namespace Signal

} // namespace mlir::triton::tle
