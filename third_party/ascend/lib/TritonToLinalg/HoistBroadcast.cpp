/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/TritonToLinalg/HoistBroadcast.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/ValueRange.h"

namespace HoistBroadcast {
using namespace mlir;
using namespace triton;

LogicalResult
BroadcastConverter::matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                                    ConversionPatternRewriter &rewriter) const {
  assert(op->getNumResults() == 1 && "BroadcastOp assumes single result");

  if (!isa<triton::PointerType>(op.getType().getElementType())) {
    return rewriter.notifyMatchFailure(
        op, "only support hoist broadcast for tt.ptr tensor right now.");
  }
  auto loc = op.getLoc();
  BroadcastHoister hoister(op);

  if (!hoister.canBroadcast()) {
    return failure();
  }

  if (hoister.parse(op.getSrc(), loc, rewriter).failed()) {
    return failure();
  }
  if (hoister.replaceBroadcastOp(op, rewriter).failed()) {
    return failure();
  }
  return success();
}

BroadcastHoister::BroadcastHoister(triton::BroadcastOp op) {
  source = nullptr;
  if (findSrc(op.getSrc()).failed()) {
    LLVM_DEBUG({ llvm::dbgs() << "No legal source found for broadcast op\n"; });
  }
  opToHoist = op;
  auto resultType = dyn_cast<RankedTensorType>(op.getType());
  for (size_t i = 0; i < resultType.getShape().size(); ++i) {
    tensorSizes.push_back(resultType.getShape()[i]);
  }
}

LogicalResult BroadcastHoister::findSrc(Value operand) {
  // ptr tensor can only be defined by AddPtrOp or SplatOp in this converter
  // another broadcast to be complemented
  if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return findSrc(op.getPtr());
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    source = op.getSrc();
    return success();
  } else {
    LLVM_DEBUG({
      llvm::dbgs() << "Unsupported operation in BroadcastHoister::findSrc: "
                   << *operand.getDefiningOp() << "\n";
    });
    return failure();
  }
}

LogicalResult BroadcastHoister::parse(Value operand, const Location &loc,
                                      ConversionPatternRewriter &rewriter) {
  if (auto op = operand.getDefiningOp<triton::AddPtrOp>()) {
    return parseAddptr(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::SplatOp>()) {
    return parseSplat(op, loc, rewriter);
  } else if (auto op = operand.getDefiningOp<triton::BroadcastOp>()) {
    return parseBroadcast(op, loc, rewriter);
  } else {
    // Handle other cases or throw an error
    LLVM_DEBUG({
      llvm::dbgs() << "Unsupported operation in BroadcastHoister::parse: "
                   << *operand.getDefiningOp() << "\n";
    });
    return failure();
  }
}

LogicalResult
BroadcastHoister::parseAddptr(triton::AddPtrOp addptrOp, const Location &loc,
                              ConversionPatternRewriter &rewriter) {
  // Implementation for parsing AddptrOp
  if (parse(addptrOp.getPtr(), loc, rewriter).failed()) {
    return failure();
  }
  auto broadcastedPtr = broadcastMap[addptrOp.getPtr()];

  RankedTensorType offsetType =
      dyn_cast<RankedTensorType>(addptrOp.getOffset().getType());
  if (!offsetType || !offsetType.hasStaticShape()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Offset must be a ranked tensor with static shape.\n";
    });
    return failure();
  }

  auto elementType = offsetType.getElementType();
  auto broadcastType = RankedTensorType::get({tensorSizes}, elementType);
  auto broadcastedOffset = rewriter.create<triton::BroadcastOp>(
      loc, broadcastType, addptrOp.getOffset());

  auto ptrType = dyn_cast<triton::PointerType>(source.getType());
  auto ptrTensorType = RankedTensorType::get({tensorSizes}, ptrType);

  auto newAddPtrOp = rewriter.create<triton::AddPtrOp>(
      loc, ptrTensorType, broadcastedPtr, broadcastedOffset);

  size_t hoistDim = -1;
  for (size_t i = 0; i < offsetType.getShape().size(); ++i) {
    if (offsetType.getShape()[i] == 1 &&
        offsetType.getShape()[i] != tensorSizes[i]) {
      hoistDim = i;
      break;
    }
  }
  if (hoistDim == -1) {
    LLVM_DEBUG({
      llvm::dbgs() << "No dimension to hoist found in AddPtrOp offset.\n";
    });
  }
  newAddPtrOp->setAttr(
      "hoist_dim", rewriter.getI64IntegerAttr(static_cast<int64_t>(hoistDim)));
  broadcastMap[addptrOp.getResult()] = newAddPtrOp.getResult();
  return success();
}

LogicalResult
BroadcastHoister::parseSplat(triton::SplatOp splatOp, const Location &loc,
                             ConversionPatternRewriter &rewriter) {
  // End of parse: splat for ptr
  auto src = splatOp.getSrc();
  auto dst = splatOp.getResult();
  if (!isa<triton::PointerType>(src.getType())) {
    LLVM_DEBUG(
        { llvm::dbgs() << "SplatOp source must be of pointer type.\n"; });
    return failure();
  }

  auto ptrType = dyn_cast<triton::PointerType>(source.getType());
  auto ptrTensorType = RankedTensorType::get({tensorSizes}, ptrType);
  auto newSplatOp = rewriter.create<triton::SplatOp>(loc, ptrTensorType, src);
  broadcastMap[splatOp.getResult()] = newSplatOp.getResult();
  return success();
}

LogicalResult
BroadcastHoister::parseBroadcast(triton::BroadcastOp broadcastOp,
                                 const Location &loc,
                                 ConversionPatternRewriter &rewriter) {
  // Another broadcast for ptr tensor
  // To be fixed if needed in future
  LLVM_DEBUG({
    llvm::dbgs() << "Now cannot handle multi broadcast for ptr tensor.\n";
  });
  return failure();
}

LogicalResult
BroadcastHoister::replaceBroadcastOp(triton::BroadcastOp op,
                                     ConversionPatternRewriter &rewriter) {
  auto newOp = broadcastMap[op.getSrc()];
  rewriter.replaceOp(op, newOp);
  return success();
}

bool BroadcastHoister::canBroadcast() {
  auto resultType = dyn_cast<RankedTensorType>(opToHoist.getType());
  auto srcType = dyn_cast<RankedTensorType>(opToHoist.getSrc().getType());
  int broadcastedDims = 0;
  for (size_t i = 0; i < resultType.getShape().size(); ++i) {
    if (srcType.getShape()[i] == 1 && resultType.getShape()[i] != 1) {
      broadcastedDims++;
    }
  }
  if (broadcastedDims != 1) {
    LLVM_DEBUG({
      llvm::dbgs() << "Now cannot handle broadcast for ptr tensor with multi "
                      "broadcasted dimension.\n";
    });
    return false;
  }
  return source != nullptr && isa<triton::PointerType>(source.getType());
}
} // namespace HoistBroadcast
