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

namespace mlir::triton::tle {
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
} // namespace RemotePointers

namespace DistributedBarrier {
llvm::LogicalResult verifyFlagCxSpace(mlir::Operation *op, mlir::Value src) {
  if (!src)
    return op->emitOpError()
           << "expects src to be present for a FlagCX distributed barrier";

  auto kindAttr = op->getAttrOfType<StringAttr>("group_kind");
  auto barrierTypeAttr = op->getAttrOfType<StringAttr>("barrier_type");
  auto orderAttr = op->getAttrOfType<StringAttr>("order");
  auto indexAttr = op->getAttrOfType<IntegerAttr>("barrier_index");
  auto contextIdAttr = op->getAttrOfType<IntegerAttr>("context_id");
  auto scopeAttr = op->getAttrOfType<StringAttr>("memory_scope");

  if (op->hasAttr("group_rank") || op->hasAttr("group_shape") ||
      op->hasAttr("group_axes") || op->hasAttr("group_mask"))
    return op->emitOpError()
           << "FlagCX distributed barriers do not accept mesh group metadata";

  if (!kindAttr || !barrierTypeAttr || !orderAttr || !indexAttr ||
      !contextIdAttr || !scopeAttr)
    return op->emitOpError()
           << "expects src, group_kind, barrier_type, order, barrier_index, "
              "context_id and memory_scope to be present for a FlagCX "
              "distributed barrier";

  StringRef kind = kindAttr.getValue();
  if (kind != "thread" && kind != "warp" && kind != "block")
    return op->emitOpError()
           << "FlagCX group_kind must be 'thread', 'warp', or 'block', got '"
           << kind << "'";

  StringRef barrierType = barrierTypeAttr.getValue();
  if (barrierType != "arrive" && barrierType != "wait" && barrierType != "sync")
    return op->emitOpError()
           << "FlagCX barrier_type must be 'arrive', 'wait', or 'sync', got '"
           << barrierType << "'";

  StringRef order = orderAttr.getValue();
  if (order != "relaxed" && order != "acquire" && order != "release" &&
      order != "acqrel")
    return op->emitOpError()
           << "FlagCX order must be 'relaxed', 'acquire', 'release', or "
              "'acqrel', got '"
           << order << "'";

  StringRef scope = scopeAttr.getValue();
  if (scope != "system" && scope != "device" && scope != "block" &&
      scope != "thread")
    return op->emitOpError()
           << "FlagCX memory_scope must be 'system', 'device', 'block', or "
              "'thread', got '"
           << scope << "'";

  if (indexAttr.getInt() < 0)
    return op->emitOpError() << "barrier_index must be non-negative";
  if (contextIdAttr.getInt() < 0)
    return op->emitOpError() << "context_id must be non-negative";
  return success();
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
