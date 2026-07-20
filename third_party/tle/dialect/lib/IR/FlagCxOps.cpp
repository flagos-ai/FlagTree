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

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include <cctype>
#include <limits>

#include "tle/dialect/include/IR/VerfiyUtils.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"

enum class CoopKind : int32_t {
  Thread = 0,
  Warp = 1,
  Block = 2,
  TileSpan = 3,
  Lanes = 4,
};

enum class MemoryOrder : int32_t {
  Relaxed = 0,
  Acquire = 1,
  Release = 2,
  AcqRel = 3,
};

namespace mlir::triton::tle {

LogicalResult GetLocalRankOp::verify() {
  auto resultTy = getResult().getType();

  if (!resultTy.isInteger(32))
    return emitOpError("result type must be i32");

  return success();
}

LogicalResult DeviceIntraBarrierOp::verify() {
  auto *op = getOperation();

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
    auto coopKind = static_cast<CoopKind>(coopKindAttr.getInt());

    switch (coopKind) {
    case CoopKind::Thread:
    case CoopKind::Warp:
    case CoopKind::Block:
    case CoopKind::TileSpan:
    case CoopKind::Lanes:
      break;
    default:
      return emitInvalidIntAttr(
          "coop_kind", coopKindAttr.getInt(),
          "Thread(0), Warp(1), Block(2), TileSpan(3), Lanes(4)");
    }
  }

  // order
  if (orderAttr) {
    auto order = static_cast<MemoryOrder>(orderAttr.getInt());

    switch (order) {
    case MemoryOrder::Relaxed:
    case MemoryOrder::Acquire:
    case MemoryOrder::Release:
    case MemoryOrder::AcqRel:
      break;
    default:
      return emitInvalidIntAttr(
          "order", orderAttr.getInt(),
          "Relaxed(0), Acquire(1), Release(2), AcqRel(3)");
    }
  }

  return success();
}
} // namespace mlir::triton::tle
