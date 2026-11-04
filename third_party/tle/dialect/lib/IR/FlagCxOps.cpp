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

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>
#include <optional>

#include "tle/dialect/include/IR/VerifyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

namespace mlir::triton::tle {

LogicalResult GetLocalRankOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult FlagCxBarrierOp::verify() {
  auto *op = getOperation();

  auto barrierTypeAttr = getBarrierTypeAttr();
  auto indexAttr = getIndexAttr();
  auto contextIdAttr = getContextIdAttr();

  auto emitInvalidStrAttr = [&](StringRef attrName, StringRef value,
                                StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " '" << value
                             << "', expected one of: " << expected;
  };

  StringRef barrierType = barrierTypeAttr.getValue();
  bool validBarrierType = llvm::StringSwitch<bool>(barrierType)
                              .Case("arrive", true)
                              .Case("wait", true)
                              .Case("sync", true)
                              .Default(false);
  if (!validBarrierType)
    return emitInvalidStrAttr("barrier_type", barrierType,
                              "arrive, wait, sync");

  if (indexAttr.getInt() < 0)
    return op->emitOpError() << "index must be non-negative";
  if (contextIdAttr.getInt() < 0)
    return op->emitOpError() << "context_id must be non-negative";

  return success();
}

LogicalResult DeviceIntraBarrierOp::verify() {
  auto *op = getOperation();

  auto indexAttr = getIndexAttr();
  auto countAttr = getBarrierCountAttr();
  if (static_cast<bool>(getDynamicIndex()) == static_cast<bool>(indexAttr))
    return emitOpError("requires exactly one of index or dynamic_index");
  if (getDynamicIndex() && !countAttr)
    return emitOpError("dynamic_index requires a positive barrier_count");
  if (countAttr && countAttr.getInt() <= 0)
    return emitOpError("barrier_count must be positive");
  std::optional<int64_t> constantIndex;
  if (indexAttr) {
    constantIndex = indexAttr.getInt();
  } else {
    APInt value;
    if (matchPattern(getDynamicIndex(), m_ConstantInt(&value)))
      constantIndex = value.getSExtValue();
  }
  if (constantIndex && (*constantIndex < 0 ||
                        (countAttr && *constantIndex >= countAttr.getInt())))
    return emitOpError(
        "constant barrier index is outside the declared capacity");

  auto barrierTypeAttr = getBarrierTypeAttr();
  auto coopKindAttr = getCoopKindAttr();
  auto orderAttr = getOrderAttr();

  auto emitInvalidIntAttr = [&](StringRef attrName, int64_t value,
                                StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " (" << value
                             << "), expected one of: " << expected;
  };

  auto emitInvalidStrAttr = [&](StringRef attrName, StringRef value,
                                StringRef expected) -> LogicalResult {
    return op->emitOpError() << "invalid " << attrName << " '" << value
                             << "', expected one of: " << expected;
  };

  // barrier_type
  if (barrierTypeAttr) {
    StringRef barrierType = barrierTypeAttr.getValue();

    bool valid = llvm::StringSwitch<bool>(barrierType)
                     .Case("arrive", true)
                     .Case("wait", true)
                     .Case("sync", true)
                     .Default(false);

    if (!valid)
      return emitInvalidStrAttr("barrier_type", barrierType,
                                "arrive, wait, sync");
  }

  // coop_kind
  if (coopKindAttr) {
    if (coopKindAttr.getInt() < 0 || coopKindAttr.getInt() > 4)
      return emitInvalidIntAttr(
          "coop_kind", coopKindAttr.getInt(),
          "Thread(0), Warp(1), Block(2), TileSpan(3), Lanes(4)");
  }

  // order
  if (orderAttr) {
    if (orderAttr.getInt() < 0 || orderAttr.getInt() > 3)
      return emitInvalidIntAttr(
          "order", orderAttr.getInt(),
          "Relaxed(0), Acquire(1), Release(2), AcqRel(3)");
  }

  return success();
}

LogicalResult FlagCxSignalOp::verify() {
  if (auto err = Signal::verifySignalOp(getSignalOp(), getValue(), getScope()))
    return emitOpError() << *err;
  return success();
}

LogicalResult FlagCxSignalWaitOp::verify() {
  if (auto err =
          Signal::verifySignalWaitOp(getWaitKind(), getTarget(), getOrder()))
    return emitOpError() << *err;
  return success();
}
} // namespace mlir::triton::tle
