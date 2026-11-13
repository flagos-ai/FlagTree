/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include "TritonToGraph/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <limits>

#define DEBUG_TYPE "merge-concat-load-buffer"

namespace mlir {
namespace triton {
namespace cfg {
#define GEN_PASS_DEF_MERGECONCATLOADBUFFER
#include "ascend/include/TritonToGraph/Passes.h.inc"
} // namespace cfg
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

//===----------------------------------------------------------------------===//
// OpFoldResult / box geometry
//===----------------------------------------------------------------------===//

static bool isSameOFR(OpFoldResult a, OpFoldResult b) {
  if (a == b)
    return true;
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  return ca && cb && *ca == *cb;
}

static bool allUnitStrides(ArrayRef<OpFoldResult> strides) {
  return llvm::all_of(strides,
                      [](OpFoldResult s) { return isConstantIntValue(s, 1); });
}

static std::optional<int64_t> addConst(OpFoldResult offset, OpFoldResult size) {
  std::optional<int64_t> off = getConstantIntValue(offset);
  std::optional<int64_t> sz = getConstantIntValue(size);
  if (!off || !sz)
    return std::nullopt;
  // No real tile comes anywhere near overflowing here, but a box that does is
  // reported as unknown rather than wrapping around into a bogus bound.
  if (*off < 0 || *sz < 0 || *off > std::numeric_limits<int64_t>::max() - *sz)
    return std::nullopt;
  return *off + *sz;
}

struct Box {
  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;

  unsigned rank() const { return offsets.size(); }
};

static Box makeFullBox(ArrayRef<int64_t> shape, MLIRContext *ctx) {
  Builder b(ctx);
  Box box;
  box.offsets.resize(shape.size(), b.getIndexAttr(0));
  for (int64_t dim : shape)
    box.sizes.push_back(b.getIndexAttr(dim));
  return box;
}

// Prove [wOff, wOff+wSz) ⊆ [iOff, iOff+iSz). Dynamic write sizes are assumed
// to be in-bounds for a well-formed subview of a static alloc when the insert
// region covers that entire dimension.
static bool intervalContained(OpFoldResult wOff, OpFoldResult wSz,
                              OpFoldResult iOff, OpFoldResult iSz,
                              int64_t allocDim) {
  if (isSameOFR(wOff, iOff) && isSameOFR(wSz, iSz))
    return true;

  std::optional<int64_t> wOffC = getConstantIntValue(wOff);
  std::optional<int64_t> wSzC = getConstantIntValue(wSz);
  std::optional<int64_t> iOffC = getConstantIntValue(iOff);
  std::optional<int64_t> iSzC = getConstantIntValue(iSz);
  if (wOffC && wSzC && iOffC && iSzC)
    return *wOffC >= *iOffC && (*wOffC + *wSzC) <= (*iOffC + *iSzC);

  if (isSameOFR(wOff, iOff) && isConstantIntValue(wOff, 0) && iSzC &&
      *iSzC == allocDim)
    return true;
  return false;
}

static bool boxContainedIn(const Box &inner, const Box &outer,
                           ArrayRef<int64_t> allocShape) {
  if (inner.rank() != outer.rank() || inner.rank() != allocShape.size())
    return false;
  for (unsigned d = 0; d < inner.rank(); ++d) {
    if (!intervalContained(inner.offsets[d], inner.sizes[d], outer.offsets[d],
                           outer.sizes[d], allocShape[d]))
      return false;
  }
  return true;
}

static bool intervalsDisjoint(OpFoldResult aOff, OpFoldResult aSz,
                              OpFoldResult bOff, OpFoldResult bSz) {
  std::optional<int64_t> aEnd = addConst(aOff, aSz);
  std::optional<int64_t> bEnd = addConst(bOff, bSz);
  std::optional<int64_t> aOffC = getConstantIntValue(aOff);
  std::optional<int64_t> bOffC = getConstantIntValue(bOff);
  if (aEnd && bOffC && *aEnd <= *bOffC)
    return true;
  if (bEnd && aOffC && *bEnd <= *aOffC)
    return true;
  return false;
}

static bool boxesDisjoint(const Box &a, const Box &b) {
  if (a.rank() != b.rank())
    return false;
  for (unsigned d = 0; d < a.rank(); ++d) {
    if (intervalsDisjoint(a.offsets[d], a.sizes[d], b.offsets[d], b.sizes[d]))
      return true;
  }
  return false;
}

// Union boxes that differ on exactly one dimension and abut on that dimension.
static std::optional<Box> unionBoxesAlongOneDim(ArrayRef<Box> boxes,
                                                MLIRContext *ctx) {
  if (boxes.empty())
    return std::nullopt;
  if (boxes.size() == 1)
    return Box(boxes.front());

  unsigned rank = boxes.front().rank();
  for (const Box &box : boxes) {
    if (box.rank() != rank)
      return std::nullopt;
  }

  SmallVector<unsigned> diffDims;
  for (unsigned d = 0; d < rank; ++d) {
    bool same = llvm::all_of(boxes, [&](const Box &box) {
      return isSameOFR(box.offsets[d], boxes.front().offsets[d]) &&
             isSameOFR(box.sizes[d], boxes.front().sizes[d]);
    });
    if (!same)
      diffDims.push_back(d);
  }
  if (diffDims.size() != 1)
    return std::nullopt;

  unsigned dim = diffDims.front();
  struct Interval {
    int64_t start;
    int64_t end;
  };
  SmallVector<Interval> intervals;
  intervals.reserve(boxes.size());
  for (const Box &box : boxes) {
    std::optional<int64_t> start = getConstantIntValue(box.offsets[dim]);
    std::optional<int64_t> end = addConst(box.offsets[dim], box.sizes[dim]);
    if (!start || !end || *end < *start)
      return std::nullopt;
    intervals.push_back({*start, *end});
  }
  llvm::sort(intervals,
             [](Interval lhs, Interval rhs) { return lhs.start < rhs.start; });
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].start != intervals[i - 1].end)
      return std::nullopt;
  }

  Builder b(ctx);
  Box result = boxes.front();
  result.offsets[dim] = b.getIndexAttr(intervals.front().start);
  result.sizes[dim] =
      b.getIndexAttr(intervals.back().end - intervals.front().start);
  return result;
}

//===----------------------------------------------------------------------===//
// Alloc user collection
//===----------------------------------------------------------------------===//

struct FillInfo {
  linalg::FillOp fill;
  scf::IfOp wrappingIf;
  Value fillValue;
  bool isBare = false;
};

struct AllocInfo {
  memref::AllocOp alloc;
  bufferization::ToTensorOp toTensor;
  SmallVector<Box> writeBoxes;
  SmallVector<memref::CopyOp> copies;
  FillInfo fill;
};

static bool sameFillValue(Value a, Value b) {
  if (a == b)
    return true;
  auto ca = a.getDefiningOp<arith::ConstantOp>();
  auto cb = b.getDefiningOp<arith::ConstantOp>();
  return ca && cb && ca.getValue() == cb.getValue();
}

static scf::IfOp wrappingTrivialFillIf(linalg::FillOp fill) {
  auto ifOp = dyn_cast<scf::IfOp>(fill->getParentOp());
  if (!ifOp || ifOp.getNumResults() != 0)
    return nullptr;

  unsigned thenCount = 0;
  for (Operation &op : ifOp.getThenRegion().front()) {
    if (isa<scf::YieldOp>(op))
      continue;
    if (&op != fill.getOperation())
      return nullptr;
    ++thenCount;
  }
  if (thenCount != 1)
    return nullptr;

  if (!ifOp.getElseRegion().empty()) {
    for (Operation &op : ifOp.getElseRegion().front()) {
      if (!isa<scf::YieldOp>(op))
        return nullptr;
    }
  }
  return ifOp;
}

// Recursively find the unique memref.copy that writes `v` (looking through
// memref.cast). Returns null if `v` has any other user or is used as a copy
// source.
static memref::CopyOp getUniqueCopyWriting(Value v) {
  memref::CopyOp found;
  for (Operation *user : v.getUsers()) {
    if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (copy.getTarget() != v || found)
        return nullptr;
      found = copy;
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(user)) {
      memref::CopyOp inner = getUniqueCopyWriting(cast.getResult());
      if (!inner || found)
        return nullptr;
      found = inner;
      continue;
    }
    return nullptr;
  }
  return found;
}

// A whole-buffer fill only acts as padding if it runs before the copies that
// overwrite parts of it. A fill placed after them is what the concatenated
// value actually observes, so it is neither dead nor safe to reorder.
static bool fillPrecedesCopies(const FillInfo &fill,
                               ArrayRef<memref::CopyOp> copies) {
  if (!fill.fill)
    return true;
  scf::IfOp wrappingIf = fill.wrappingIf;
  linalg::FillOp fillOp = fill.fill;
  Operation *fillPoint =
      wrappingIf ? wrappingIf.getOperation() : fillOp.getOperation();
  for (memref::CopyOp copy : copies) {
    Operation *copyPoint = copy.getOperation();
    if (fillPoint->getBlock() != copyPoint->getBlock() ||
        !fillPoint->isBeforeInBlock(copyPoint))
      return false;
  }
  return true;
}

static std::optional<AllocInfo> collectAllocInfo(Value mem) {
  auto alloc = mem.getDefiningOp<memref::AllocOp>();
  if (!alloc || !alloc.getDynamicSizes().empty())
    return std::nullopt;

  auto memType = dyn_cast<MemRefType>(alloc.getType());
  if (!memType || !memType.hasStaticShape())
    return std::nullopt;

  AllocInfo info;
  info.alloc = alloc;

  for (Operation *user : alloc.getResult().getUsers()) {
    if (auto fill = dyn_cast<linalg::FillOp>(user)) {
      if (info.fill.fill)
        return std::nullopt;
      if (fill.getDpsInits().empty() || fill.getDpsInputs().empty() ||
          fill.getDpsInits().front() != alloc.getResult())
        return std::nullopt;
      info.fill.fill = fill;
      info.fill.fillValue = fill.getDpsInputs().front();
      info.fill.wrappingIf = wrappingTrivialFillIf(fill);
      info.fill.isBare = fill->getBlock() == alloc->getBlock();
      continue;
    }

    if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
      if (info.toTensor)
        return std::nullopt;
      if (!toTensor.getRestrict() || !toTensor.getWritable())
        return std::nullopt;
      if (toTensor.getBuffer() != alloc.getResult())
        return std::nullopt;
      info.toTensor = toTensor;
      continue;
    }

    if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
      if (!allUnitStrides(subview.getMixedStrides()))
        return std::nullopt;
      memref::CopyOp copy = getUniqueCopyWriting(subview.getResult());
      if (!copy)
        return std::nullopt;
      info.copies.push_back(copy);
      info.writeBoxes.push_back(
          Box{subview.getMixedOffsets(), subview.getMixedSizes()});
      continue;
    }

    return std::nullopt;
  }

  if (!info.toTensor || info.writeBoxes.empty())
    return std::nullopt;
  if (!fillPrecedesCopies(info.fill, info.copies))
    return std::nullopt;
  return info;
}

static void eraseFill(PatternRewriter &rewriter, const FillInfo &fill) {
  if (!fill.fill)
    return;
  if (fill.wrappingIf)
    rewriter.eraseOp(fill.wrappingIf);
  else
    rewriter.eraseOp(fill.fill);
}

//===----------------------------------------------------------------------===//
// Rewrite pattern
//===----------------------------------------------------------------------===//

class MergeConcatLoadBufferPattern
    : public OpRewritePattern<tensor::InsertSliceOp> {
public:
  using OpRewritePattern<tensor::InsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertOp,
                                PatternRewriter &rewriter) const override {
    if (!allUnitStrides(insertOp.getMixedStrides()))
      return rewriter.notifyMatchFailure(insertOp, "non-unit insert strides");

    auto extractOp =
        insertOp.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!extractOp || !extractOp->hasOneUse())
      return rewriter.notifyMatchFailure(insertOp,
                                         "source is not a single-use extract");
    if (!allUnitStrides(extractOp.getMixedStrides()))
      return rewriter.notifyMatchFailure(insertOp, "non-unit extract strides");

    // Identity move: extract and insert must use the same offsets/sizes so
    // the source buffer can be RAUW'd onto the dest buffer without shifting.
    if (extractOp.getMixedOffsets().size() !=
            insertOp.getMixedOffsets().size() ||
        extractOp.getMixedSizes().size() != insertOp.getMixedSizes().size())
      return failure();
    for (auto [eOff, iOff] :
         llvm::zip(extractOp.getMixedOffsets(), insertOp.getMixedOffsets())) {
      if (!isSameOFR(eOff, iOff))
        return rewriter.notifyMatchFailure(insertOp,
                                           "extract/insert offsets differ");
    }
    for (auto [eSz, iSz] :
         llvm::zip(extractOp.getMixedSizes(), insertOp.getMixedSizes())) {
      if (!isSameOFR(eSz, iSz))
        return rewriter.notifyMatchFailure(insertOp,
                                           "extract/insert sizes differ");
    }

    auto toTensorA =
        extractOp.getSource().getDefiningOp<bufferization::ToTensorOp>();
    auto toTensorB =
        insertOp.getDest().getDefiningOp<bufferization::ToTensorOp>();
    if (!toTensorA || !toTensorB || toTensorA == toTensorB)
      return failure();
    if (!toTensorA->hasOneUse() || !toTensorB->hasOneUse())
      return rewriter.notifyMatchFailure(
          insertOp, "to_tensor must be used only by this concat");

    std::optional<AllocInfo> infoA = collectAllocInfo(toTensorA.getBuffer());
    std::optional<AllocInfo> infoB = collectAllocInfo(toTensorB.getBuffer());
    if (!infoA || !infoB)
      return rewriter.notifyMatchFailure(insertOp,
                                         "alloc users are not a load buffer");
    if (infoA->toTensor != toTensorA || infoB->toTensor != toTensorB)
      return failure();

    auto typeA = dyn_cast<MemRefType>(infoA->alloc.getType());
    auto typeB = dyn_cast<MemRefType>(infoB->alloc.getType());
    if (!typeA || !typeB || typeA != typeB)
      return rewriter.notifyMatchFailure(insertOp, "alloc types differ");

    if (infoA->alloc->getBlock() != infoB->alloc->getBlock() ||
        insertOp->getBlock() != infoA->alloc->getBlock() ||
        extractOp->getBlock() != insertOp->getBlock() ||
        toTensorA->getBlock() != insertOp->getBlock() ||
        toTensorB->getBlock() != insertOp->getBlock())
      return rewriter.notifyMatchFailure(insertOp, "ops not in the same block");

    Box insertBox{insertOp.getMixedOffsets(), insertOp.getMixedSizes()};
    ArrayRef<int64_t> shape = typeA.getShape();
    for (const Box &write : infoA->writeBoxes) {
      if (!boxContainedIn(write, insertBox, shape))
        return rewriter.notifyMatchFailure(
            insertOp, "source write is not inside the insert region");
    }
    for (const Box &write : infoB->writeBoxes) {
      if (!boxesDisjoint(write, insertBox))
        return rewriter.notifyMatchFailure(
            insertOp, "dest write overlaps the insert region");
    }

    SmallVector<memref::CopyOp> allCopies;
    allCopies.append(infoA->copies.begin(), infoA->copies.end());
    allCopies.append(infoB->copies.begin(), infoB->copies.end());
    for (memref::CopyOp copy : allCopies) {
      if (copy->getBlock() != toTensorB->getBlock() ||
          !copy->isBeforeInBlock(toTensorB))
        return rewriter.notifyMatchFailure(
            insertOp, "to_tensor B does not dominate all copies");
    }

    // Read region of the concatenated value: extract_slice users contribute
    // their boxes; any other user conservatively reads the whole buffer.
    // to_tensor B is required to have a single use (the insert), so only
    // insert result users matter.
    SmallVector<Box> readBoxes;
    bool readsFull = false;
    for (Operation *user : insertOp.getResult().getUsers()) {
      if (auto readExtract = dyn_cast<tensor::ExtractSliceOp>(user)) {
        if (!allUnitStrides(readExtract.getMixedStrides())) {
          readsFull = true;
          break;
        }
        readBoxes.push_back(
            Box{readExtract.getMixedOffsets(), readExtract.getMixedSizes()});
        continue;
      }
      readsFull = true;
      break;
    }
    if (readsFull) {
      readBoxes.clear();
      readBoxes.push_back(makeFullBox(shape, insertOp.getContext()));
    }

    SmallVector<Box> writeUnionInput;
    writeUnionInput.append(infoA->writeBoxes.begin(), infoA->writeBoxes.end());
    writeUnionInput.append(infoB->writeBoxes.begin(), infoB->writeBoxes.end());
    std::optional<Box> writeUnion =
        unionBoxesAlongOneDim(writeUnionInput, insertOp.getContext());

    bool fillDead = false;
    if (writeUnion) {
      fillDead = llvm::all_of(readBoxes, [&](const Box &read) {
        return boxContainedIn(read, *writeUnion, shape);
      });
    }

    // Keep the earlier alloc so RAUW cannot create uses-before-def.
    bool aIsEarlier = infoA->alloc->isBeforeInBlock(infoB->alloc);
    memref::AllocOp survivor = aIsEarlier ? infoA->alloc : infoB->alloc;
    memref::AllocOp dropped = aIsEarlier ? infoB->alloc : infoA->alloc;
    const FillInfo &survivorFill = aIsEarlier ? infoA->fill : infoB->fill;
    const FillInfo &droppedFill = aIsEarlier ? infoB->fill : infoA->fill;

    bool keepSingleFill = false;
    if (!fillDead) {
      // Once merged, the surviving fill initialises the shared buffer, so it
      // must run before every copy and not just before the ones that used to
      // target its own buffer.
      if (infoA->fill.isBare && infoB->fill.isBare && infoA->fill.fill &&
          infoB->fill.fill &&
          sameFillValue(infoA->fill.fillValue, infoB->fill.fillValue) &&
          fillPrecedesCopies(survivorFill, allCopies)) {
        keepSingleFill = true;
      } else if (infoA->fill.fill || infoB->fill.fill) {
        return rewriter.notifyMatchFailure(
            insertOp, "cannot prove fill dead and cannot merge fills");
      }
    }

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] matched at "
                            << insertOp.getLoc() << ": fillDead=" << fillDead
                            << " keepSingleFill=" << keepSingleFill << "\n");

    if (fillDead) {
      eraseFill(rewriter, infoA->fill);
      eraseFill(rewriter, infoB->fill);
    } else if (keepSingleFill) {
      eraseFill(rewriter, droppedFill);
    }

    rewriter.replaceOp(insertOp, toTensorB.getResult());
    rewriter.eraseOp(extractOp);
    rewriter.eraseOp(toTensorA);
    rewriter.replaceAllUsesWith(dropped.getResult(), survivor.getResult());
    rewriter.eraseOp(dropped);
    return success();
  }
};

struct MergeConcatLoadBufferPass
    : public mlir::triton::cfg::impl::MergeConcatLoadBufferBase<
          MergeConcatLoadBufferPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, bufferization::BufferizationDialect,
                func::FuncDialect, linalg::LinalgDialect, memref::MemRefDialect,
                scf::SCFDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MergeConcatLoadBufferPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::cfg::createMergeConcatLoadBufferPass() {
  return std::make_unique<MergeConcatLoadBufferPass>();
}
