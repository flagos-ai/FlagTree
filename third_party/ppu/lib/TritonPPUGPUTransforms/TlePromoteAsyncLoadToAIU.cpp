/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
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

#include "TritonPPUGPUTransforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/CacheControls.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {

#define GEN_PASS_DEF_TLEPROMOTEASYNCLOADTOAIUPASS
#include "TritonPPUGPUTransforms/Passes.h.inc"

namespace {

// The AIU bulk copy only moves 1/2/4-byte elements (.b8/.b16/.b32); other
// widths stay ordinary tt.load.
static bool isAIULoadElementType(Type elemTy) {
  if (!elemTy.isIntOrFloat())
    return false;
  unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
  return bitWidth == 8 || bitWidth == 16 || bitWidth == 32;
}

static triton::MakeTensorPtrOp findTensorPointer(Value pointer) {
  llvm::SmallDenseSet<Value, 8> seen;
  while (seen.insert(pointer).second) {
    if (auto make = pointer.getDefiningOp<triton::MakeTensorPtrOp>())
      return make;
    if (auto advance = pointer.getDefiningOp<triton::AdvanceOp>()) {
      pointer = advance.getPtr();
      continue;
    }
    if (auto argument = dyn_cast<BlockArgument>(pointer)) {
      auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
      if (loop && argument.getArgNumber() > 0) {
        pointer = loop.getInitArgs()[argument.getArgNumber() - 1];
        continue;
      }
    }
    return {};
  }
  return {};
}

// PPU0010's existing shared layout is two-dimensional. Keep rank-3 copies
// on native AIU by issuing one 2D transfer per leading tile slice; each copy
// retains the original policy and the normal AIU completion protocol.
static LogicalResult splitRankThreeLoad(triton::LoadOp load,
                                        OpBuilder &builder) {
  auto ptr = load.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
  if (!ptr || ptr.getOrder() != ArrayRef<int32_t>({2, 1, 0}))
    return load.emitError(
        "PPU rank-3 AIU copy requires a direct row-major tensor pointer");
  builder.setInsertionPoint(load);
  auto loc = load.getLoc();
  auto type = cast<RankedTensorType>(load.getType());
  auto tile = type.getShape();
  if (!llvm::isPowerOf2_64(tile[0]))
    return load.emitError(
        "PPU rank-3 AIU copy requires a power-of-two leading tile extent");
  SmallVector<Value> shape(ptr.getShape().begin() + 1, ptr.getShape().end());
  SmallVector<Value> strides(ptr.getStrides().begin() + 1,
                             ptr.getStrides().end());
  SmallVector<int32_t> block{static_cast<int32_t>(tile[1]),
                             static_cast<int32_t>(tile[2])};
  Value zero = arith::ConstantIntOp::create(builder, loc, 0, 64);
  Value start = arith::ExtSIOp::create(builder, loc, builder.getI64Type(),
                                       ptr.getOffsets()[0]);
  Value outOfBoundsRow =
      arith::TruncIOp::create(builder, loc, builder.getI32Type(), shape[0]);
  SmallVector<Value> planes;
  for (int64_t i = 0; i < tile[0]; ++i) {
    Value index = arith::AddIOp::create(
        builder, loc, start, arith::ConstantIntOp::create(builder, loc, i, 64));
    Value lower = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::sge,
                                        index, zero);
    Value upper = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                        index, ptr.getShape()[0]);
    Value valid = arith::AndIOp::create(builder, loc, lower, upper);
    Value safeIndex = arith::SelectOp::create(builder, loc, valid, index, zero);
    Value baseOffset =
        arith::MulIOp::create(builder, loc, safeIndex, ptr.getStrides()[0]);
    Value base = triton::AddPtrOp::create(builder, loc, ptr.getBase().getType(),
                                          ptr.getBase(), baseOffset);
    Value row = arith::SelectOp::create(builder, loc, valid,
                                        ptr.getOffsets()[1], outOfBoundsRow);
    SmallVector<Value> offsets{row, ptr.getOffsets()[2]};
    auto planePtr = triton::MakeTensorPtrOp::create(builder, loc, base, shape,
                                                    strides, offsets, block,
                                                    ArrayRef<int32_t>({1, 0}));
    auto plane = triton::LoadOp::create(
        builder, loc, planePtr.getResult(), ArrayRef<int32_t>({0, 1}),
        triton::PaddingOption::PAD_ZERO, load.getCache(), load.getEvict(),
        false);

    planes.push_back(plane.getResult());
  }
  // Join adds a minor plane bit. Reverse those bits into leading dimensions
  // before reshaping, preserving plane order without a reorderable CatOp.
  while (planes.size() > 1) {
    SmallVector<Value> joined;
    for (size_t i = 0; i < planes.size(); i += 2) {
      joined.push_back(
          triton::JoinOp::create(builder, loc, planes[i], planes[i + 1]));
    }
    planes = std::move(joined);
  }
  Value joined = planes.front();
  unsigned rank = cast<RankedTensorType>(joined.getType()).getRank();
  if (rank > 2) {
    SmallVector<int32_t> order;
    for (int i = rank - 1; i >= 2; --i)
      order.push_back(i);
    order.append({0, 1});
    joined = triton::TransOp::create(builder, loc, joined, order);
  }
  auto result = triton::ReshapeOp::create(builder, loc, tile, joined, false);
  load.replaceAllUsesWith(result.getResult());
  load.erase();
  return success();
}

class TlePromoteAsyncLoadToAIUPass
    : public impl::TlePromoteAsyncLoadToAIUPassBase<
          TlePromoteAsyncLoadToAIUPass> {
public:
  using impl::TlePromoteAsyncLoadToAIUPassBase<
      TlePromoteAsyncLoadToAIUPass>::TlePromoteAsyncLoadToAIUPassBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();
    OpBuilder builder(m.getContext());

    bool invalid = false;
    SmallVector<triton::LoadOp> rankThreeLoads;
    m.walk([&](triton::LoadOp load) {
      auto type = dyn_cast<RankedTensorType>(load.getType());
      if (triton::isPPUAIUCache(load.getCache()) && type && type.getRank() == 3)
        rankThreeLoads.push_back(load);
    });
    for (auto load : rankThreeLoads)
      if (failed(splitRankThreeLoad(load, builder)))
        invalid = true;
    if (invalid) {
      signalPassFailure();
      return;
    }
    SmallVector<triton::LoadOp> toPromote;
    m.walk([&](triton::LoadOp loadOp) {
      auto asyncAttr =
          llvm::cast_if_present<BoolAttr>(loadOp->getAttr("tt.load.async"));
      bool nativeAIU = triton::isPPUAIUCache(loadOp.getCache());
      if (nativeAIU) {
        auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
        if (!triton::isTensorPointerType(loadOp.getPtr().getType()) ||
            !resultTy || resultTy.getRank() != 2 ||
            !isAIULoadElementType(resultTy.getElementType())) {
          loadOp.emitError("PPU AIU cache policy requires a rank-2 block "
                           "pointer with 8/16/32-bit elements");
          invalid = true;
          return;
        }
        auto ptr = findTensorPointer(loadOp.getPtr());
        if (!ptr) {
          loadOp.emitError(
              "PPU AIU cache policy requires a traceable make_block_ptr; "
              "function arguments and control-flow merges are unsupported");
          invalid = true;
          return;
        }
        auto order = ptr.getOrder();
        unsigned minor = order[0], major = order[1];
        auto strides = ptr.getStrides();
        auto shape = ptr.getShape();
        APInt minorStride, majorStride, minorShape;
        bool minorKnown =
            matchPattern(strides[minor], m_ConstantInt(&minorStride));
        bool majorKnown =
            matchPattern(strides[major], m_ConstantInt(&majorStride));
        bool shapeKnown =
            matchPattern(shape[minor], m_ConstantInt(&minorShape));
        if ((minorKnown && !minorStride.isOne()) ||
            (majorKnown && shapeKnown &&
             majorStride.getSExtValue() != minorShape.getSExtValue())) {
          loadOp.emitError("PPU AIU cache policy requires contiguous "
                           "block-pointer strides matching its order");
          invalid = true;
          return;
        }
        // Host descriptors carry independent runtime shape/stride arguments.
        // Guard their native AIU layout contract instead of discarding the
        // requested cache mode or assuming that the arguments are contiguous.
        builder.setInsertionPoint(loadOp);
        Value contiguous;
        if (!minorKnown) {
          Value one =
              arith::ConstantIntOp::create(builder, loadOp.getLoc(), 1, 64);
          contiguous = arith::CmpIOp::create(builder, loadOp.getLoc(),
                                             arith::CmpIPredicate::eq,
                                             strides[minor], one);
        }
        // Compare the values used by the descriptor, including any narrowing.
        if (strides[major] != shape[minor] && !(majorKnown && shapeKnown)) {
          Value packed = arith::CmpIOp::create(builder, loadOp.getLoc(),
                                               arith::CmpIPredicate::eq,
                                               strides[major], shape[minor]);
          contiguous = contiguous
                           ? arith::AndIOp::create(builder, loadOp.getLoc(),
                                                   contiguous, packed)
                           : packed;
        }
        if (contiguous)
          triton::AssertOp::create(builder, loadOp.getLoc(), contiguous,
                                   "PPU AIU cache policy requires contiguous "
                                   "block-pointer strides matching its order");
        auto tile = resultTy.getShape();
        auto elemBytes = resultTy.getElementTypeBitWidth() / 8;
        if (tile[major] < 16 || tile[major] > 2048 ||
            tile[minor] * elemBytes < 32) {
          loadOp.emitError("PPU AIU cache policy requires 16..2048 rows and at "
                           "least 32 contiguous bytes per tile row");
          invalid = true;
          return;
        }
        for (Value dim : shape) {
          APInt value;
          if (matchPattern(dim, m_ConstantInt(&value))) {
            if (value.getSExtValue() < 1 || value.getSExtValue() > (1 << 23)) {
              loadOp.emitError("PPU0010 AIU dimensions must be in 1..8388608");
              invalid = true;
              return;
            }
          } else {
            auto loc = loadOp.getLoc();
            Value one = arith::ConstantIntOp::create(builder, loc, 1, 64);
            Value maximum =
                arith::ConstantIntOp::create(builder, loc, 1 << 23, 64);
            Value lower = arith::CmpIOp::create(
                builder, loc, arith::CmpIPredicate::sge, dim, one);
            Value upper = arith::CmpIOp::create(
                builder, loc, arith::CmpIPredicate::sle, dim, maximum);
            Value valid = arith::AndIOp::create(builder, loc, lower, upper);
            triton::AssertOp::create(
                builder, loc, valid,
                "PPU0010 AIU dimensions must be in 1..8388608");
          }
        }
      }
      if (!nativeAIU && (!asyncAttr || !asyncAttr.getValue()))
        return;
      if (!triton::isTensorPointerType(loadOp.getPtr().getType()))
        return;
      auto resultTy = dyn_cast<RankedTensorType>(loadOp.getType());
      if (!resultTy || !isAIULoadElementType(resultTy.getElementType()))
        return;
      toPromote.push_back(loadOp);
    });

    if (invalid) {
      signalPassFailure();
      return;
    }
    for (auto loadOp : toPromote) {
      builder.setInsertionPoint(loadOp);
      auto aiuLoad = triton::AIULoadOp::create(
          builder, loadOp.getLoc(), loadOp.getType(), loadOp.getPtr(),
          loadOp.getCache(), loadOp.getEvict());
      loadOp.replaceAllUsesWith(aiuLoad.getResult());
      loadOp.erase();
    }
  }
};

} // namespace
} // namespace mlir
