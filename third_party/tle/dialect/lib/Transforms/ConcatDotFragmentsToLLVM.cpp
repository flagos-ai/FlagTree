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

#include "TleTileToLLVMUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
namespace ttg = mlir::triton::gpu;
using namespace mlir::triton::tle;

// concat_dot_fragments lowering: pure per-thread register gather. For each
// result register slot the authoritative LinearLayout of result and tile tells
// which (tile, tile-register-slot) feeds it, and that register is copied over.
static LogicalResult
lowerConcatDotFragmentsRegister(ConcatDotFragmentsOp op,
                                ConcatDotFragmentsOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter,
                                const LLVMTypeConverter *typeConverter) {
  Location loc = op->getLoc();
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto tileTy = cast<RankedTensorType>(op.getTiles()[0].getType());
  int64_t dim = op.getDim();
  int64_t numTiles = op.getTiles().size();

  MLIRContext *ctx = op.getContext();
  StringAttr kReg = StringAttr::get(ctx, "register");

  // Layouts other than dot_op place lanes differently as `dim` grows, so the
  // per-thread relabel below would silently move the wrong registers.
  auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(tileTy.getEncoding());
  if (!dotEnc)
    return op.emitError("concat_dot_fragments: expects dot_op encoded "
                        "operands, but got ")
           << tileTy.getEncoding()
           << "; the result must be consumed only by a dot";

  // Only the contraction axis can be extended in place: growing M or N would
  // change which lane owns an element. opIdx 0 contracts along the last dim,
  // opIdx 1 along the one before it.
  int64_t rank = tileTy.getRank();
  int64_t kDim = dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2;
  if (dim != kDim)
    return op.emitError("concat_dot_fragments: dim ")
           << dim << " is not the contraction axis (expected " << kDim
           << " for opIdx " << dotEnc.getOpIdx() << ")";

  // Below 8 * kWidth the layout stops scaling with K - the per-thread element
  // count stays flat - so the tile is not a K-slice of the wider fragment and
  // cannot be relabeled into it. kWidth is 0 for layouts that do not use it.
  unsigned kWidth = dotEnc.getKWidth();
  int64_t minK = 8 * static_cast<int64_t>(kWidth);
  if (kWidth != 0 && tileTy.getShape()[dim] < minK)
    return op.emitError("concat_dot_fragments: tile K extent ")
           << tileTy.getShape()[dim] << " is below the minimum " << minK
           << " (8 * kWidth) required for this dot_op layout";

  // Authoritative layouts.
  LinearLayout dstLL = ttg::toLinearLayout(dstTy);
  LinearLayout tileLL = ttg::toLinearLayout(tileTy);

  unsigned dstRegs = ttg::getTotalElemsPerThread(dstTy);
  unsigned tileRegs = ttg::getTotalElemsPerThread(tileTy);
  if (dstRegs != tileRegs * numTiles)
    return op.emitError("concat_dot_fragments: result per-thread elems must "
                        "equal N * tile elems");

  // The register mapping below is derived on lane 0 and applied to every
  // thread, which only holds if both layouts distribute elements over lanes,
  // warps and blocks identically - a valid concat adds register bases and
  // nothing else.
  for (StringAttr inDim :
       {StringAttr::get(ctx, "lane"), StringAttr::get(ctx, "warp"),
        StringAttr::get(ctx, "block")}) {
    if (dstLL.hasInDim(inDim) != tileLL.hasInDim(inDim) ||
        (dstLL.hasInDim(inDim) &&
         dstLL.getBases().lookup(inDim) != tileLL.getBases().lookup(inDim)))
      return op.emitError("concat_dot_fragments: ")
             << inDim.getValue()
             << " basis differs between tile and result layout; the "
                "concatenation would move elements across threads";
  }

  // Output dim names in a stable order (dim0, dim1, ...).
  SmallVector<StringAttr> outDims(llvm::to_vector(dstLL.getOutDimNames()));

  // Enumerate tile register slots once, recording in-tile coord -> slot, so we
  // can invert result coords back to the feeding tile register.
  auto dimIndexOf = [&](StringAttr d) -> int {
    for (int i = 0; i < (int)outDims.size(); ++i)
      if (outDims[i] == d)
        return i;
    return -1;
  };
  auto coordKey = [&](ArrayRef<int64_t> c) {
    int64_t key = 0;
    for (int i = 0; i < rank; ++i)
      key = key * tileTy.getShape()[i] + c[i];
    return key;
  };

  llvm::DenseMap<int64_t, unsigned> tileCoordToSlot;
  for (unsigned r = 0; r < tileRegs; ++r) {
    auto outs = tileLL.apply({{kReg, r},
                              {StringAttr::get(ctx, "lane"), 0},
                              {StringAttr::get(ctx, "warp"), 0},
                              {StringAttr::get(ctx, "block"), 0}});
    SmallVector<int64_t> c(rank, 0);
    for (auto &kv : outs) {
      int idx = dimIndexOf(kv.first);
      if (idx >= 0)
        c[idx] = kv.second;
    }
    // Two registers mapping to the same coordinate means the layout broadcasts
    // along a dim we index by, so the reverse lookup below would drop one of
    // them and duplicate the other. Bail out instead of relabeling silently.
    if (!tileCoordToSlot.try_emplace(coordKey(c), r).second)
      return op.emitError("concat_dot_fragments: tile layout maps several "
                          "registers to the same element; it cannot be "
                          "relabeled by index");
  }

  SmallVector<SmallVector<Value>> allTileVals;
  allTileVals.reserve(numTiles);
  for (int64_t t = 0; t < numTiles; ++t)
    allTileVals.push_back(
        unpackLLElements(loc, adaptor.getTiles()[t], rewriter));

  SmallVector<Value> resultVals;
  resultVals.reserve(dstRegs);
  for (unsigned r = 0; r < dstRegs; ++r) {
    auto outs = dstLL.apply({{kReg, r},
                             {StringAttr::get(ctx, "lane"), 0},
                             {StringAttr::get(ctx, "warp"), 0},
                             {StringAttr::get(ctx, "block"), 0}});
    SmallVector<int64_t> c(rank, 0);
    for (auto &kv : outs) {
      int idx = dimIndexOf(kv.first);
      if (idx >= 0)
        c[idx] = kv.second;
    }
    // Split the concat-axis coord into (tile index, in-tile coord).
    int64_t tileExtent = tileTy.getShape()[dim];
    int64_t tileIdx = c[dim] / tileExtent;
    c[dim] = c[dim] % tileExtent;
    if (tileIdx < 0 || tileIdx >= numTiles)
      return op.emitError(
          "concat_dot_fragments: concat-axis coord out of range");
    auto it = tileCoordToSlot.find(coordKey(c));
    if (it == tileCoordToSlot.end())
      return op.emitError(
          "concat_dot_fragments: no matching tile register slot; "
          "layouts are not a clean K-superset");
    resultVals.push_back(allTileVals[tileIdx][it->second]);
  }

  Value ret = packLLElements(loc, typeConverter, resultVals, rewriter, dstTy);
  rewriter.replaceOp(op, ret);
  return success();
}

struct ConcatDotFragmentsOpConversion
    : public ConvertOpToLLVMPattern<ConcatDotFragmentsOp> {
  // The relabel is expressed entirely through LinearLayout, so unlike the other
  // tile ops this one needs no target-specific information.
  ConcatDotFragmentsOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern<ConcatDotFragmentsOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(ConcatDotFragmentsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstTy = dyn_cast<RankedTensorType>(op.getType());
    auto tileTy = dyn_cast<RankedTensorType>(op.getTiles()[0].getType());
    if (!dstTy || !tileTy)
      return op.emitError(
          "concat_dot_fragments operands must be ranked tensors");
    if (!dstTy.getEncoding() || !tileTy.getEncoding())
      return op.emitError(
          "concat_dot_fragments requires tensors with encoding");
    return lowerConcatDotFragmentsRegister(op, adaptor, rewriter,
                                           this->getTypeConverter());
  }
};

} // anonymous namespace

namespace mlir::triton::tle {
void populateConcatDotFragmentsOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    [[maybe_unused]] const TargetInfoBase &targetInfo, unsigned benefit) {
  patterns.add<ConcatDotFragmentsOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::tle
