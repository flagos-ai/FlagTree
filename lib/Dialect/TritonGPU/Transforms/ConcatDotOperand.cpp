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

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "tritongpu-concat-dot-operand"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace mlir::triton::gpu {
namespace {

// Walk through the layout-only converts the TTIR to TTGPU conversion leaves
// between every step of the chain.
static Value skipConverts(Value v) {
  while (auto cvt = v.getDefiningOp<ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

// Collect the leaves of a perfectly balanced JoinOp tree, in the fragment order
// the joins encode. Every join must feed only the tree: one that is also used
// elsewhere stays live after the rewrite and recomputes the same values.
static bool collectJoinLeaves(Value v, int depth, SmallVectorImpl<Value> &out) {
  v = skipConverts(v);
  if (depth == 0) {
    out.push_back(v);
    return true;
  }
  auto join = v.getDefiningOp<triton::JoinOp>();
  if (!join || !join->hasOneUse())
    return false;
  return collectJoinLeaves(join.getLhs(), depth - 1, out) &&
         collectJoinLeaves(join.getRhs(), depth - 1, out);
}

// The join tree appends `levels` trailing axes of extent 2, outermost join
// last. Flattening them into an ordered concat needs the transpose to move
// those axes in front of the in-fragment concat coordinate, most significant
// first.
static bool isConcatOrder(ArrayRef<int32_t> order, int64_t fragRank,
                          int64_t concatDim, int levels) {
  if ((int64_t)order.size() != fragRank + levels)
    return false;
  SmallVector<int32_t> expected;
  for (int64_t d = 0; d < concatDim; ++d)
    expected.push_back(d);
  for (int i = levels; i >= 1; --i)
    expected.push_back(fragRank + i - 1);
  expected.push_back(concatDim);
  for (int64_t d = concatDim + 1; d < fragRank; ++d)
    expected.push_back(d);
  return ArrayRef<int32_t>(expected) == order;
}

// An op only feeds the chain if its single use, after skipping converts, does
// not fan out anywhere else.
static bool onlyFeedsChain(Operation *op) {
  while (op->hasOneUse()) {
    Operation *user = *op->getUsers().begin();
    if (!isa<ConvertLayoutOp>(user))
      return true;
    op = user;
  }
  return false;
}

// Every use of the concatenated value must reach a dot: only a dot gives it the
// dot_op layout the relabel needs.
static bool onlyFeedsDot(Value v) {
  SmallVector<Value> worklist{v};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (cur.use_empty())
      return false;
    for (Operation *user : cur.getUsers()) {
      if (isa<triton::DotOpInterface>(user))
        continue;
      // Layout-only ops keep the operand a candidate; follow them through.
      if (isa<ConvertLayoutOp>(user)) {
        if (seen.insert(user).second)
          worklist.push_back(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

struct MatchJoinTreeConcat : public OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern<triton::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    // A reordering reshape says nothing about element order, so the chain
    // cannot be proven to be an ordered concat.
    if (reshape.getAllowReorder())
      return failure();

    // Match the interface rather than tt.trans: the conversion to TTGPU may
    // leave the transpose as any op implementing it.
    auto trans =
        skipConverts(reshape.getSrc()).getDefiningOp<TransposeOpInterface>();
    if (!trans)
      return failure();
    auto joinRoot =
        skipConverts(trans.getSrc()).getDefiningOp<triton::JoinOp>();
    if (!joinRoot)
      return failure();

    auto rootTy = cast<RankedTensorType>(joinRoot.getType());
    auto dstTy = cast<RankedTensorType>(reshape.getType());

    // The reshape collapses the join axes back into one fragment dim, so the
    // fragment rank is the destination rank and the tree depth is what the
    // joins added on top of it.
    int64_t fragRank = dstTy.getRank();
    int levels = rootTy.getRank() - fragRank;
    if (fragRank < 1 || levels < 1)
      return failure();
    for (int i = 0; i < levels; ++i)
      if (rootTy.getShape()[rootTy.getRank() - 1 - i] != 2)
        return failure();

    SmallVector<Value> fragments;
    if (!collectJoinLeaves(joinRoot, levels, fragments))
      return failure();
    int64_t numFrags = 1LL << levels;
    if ((int64_t)fragments.size() != numFrags)
      return failure();

    auto fragTy = cast<RankedTensorType>(fragments.front().getType());
    if (fragTy.getRank() != fragRank)
      return failure();
    for (Value f : fragments) {
      auto ty = dyn_cast<RankedTensorType>(f.getType());
      if (!ty || ty != fragTy)
        return failure();
    }
    if (dstTy.getElementType() != fragTy.getElementType())
      return failure();

    // Exactly one dim grows by the fragment count; that is the concat axis.
    int64_t concatDim = -1;
    for (int64_t d = 0; d < fragRank; ++d) {
      int64_t expect = fragTy.getShape()[d];
      if (dstTy.getShape()[d] == expect)
        continue;
      if (dstTy.getShape()[d] != expect * numFrags || concatDim >= 0)
        return failure();
      concatDim = d;
    }
    if (concatDim < 0)
      return failure();

    if (!isConcatOrder(trans.getOrder(), fragRank, concatDim, levels))
      return failure();

    // Replacing the chain must not leave the joins live and duplicate the work.
    if (!onlyFeedsChain(trans) || !onlyFeedsChain(joinRoot))
      return failure();

    // Only rewrite when the result actually reaches a dot; see onlyFeedsDot.
    if (!onlyFeedsDot(reshape.getResult()))
      return failure();

    LDBG("rewriting an ordered concat of " << numFrags << " fragments on dim "
                                           << concatDim);

    // The op requires its result encoding to match the fragments; the convert
    // bridging to the reshape's encoding is folded away once layout propagation
    // pulls dot_op back through the concat.
    auto resTy = RankedTensorType::get(dstTy.getShape(), dstTy.getElementType(),
                                       fragTy.getEncoding());
    Value concat =
        ConcatDotOperandOp::create(rewriter, reshape.getLoc(), resTy, fragments,
                                   rewriter.getI32IntegerAttr(concatDim));
    if (resTy != dstTy)
      concat =
          ConvertLayoutOp::create(rewriter, reshape.getLoc(), dstTy, concat);
    rewriter.replaceOp(reshape, concat);
    return success();
  }
};

// The encoding a reshape from `srcTy` to `dstShape` infers, or null when the
// dialect cannot infer one.
static Attribute inferReshapeEncoding(RankedTensorType srcTy,
                                      ArrayRef<int64_t> dstShape,
                                      Location loc) {
  Attribute srcEnc = srcTy.getEncoding();
  if (!srcEnc)
    return {};
  Attribute dstEnc;
  if (failed(cast<triton::DialectInferLayoutInterface>(&srcEnc.getDialect())
                 ->inferReshapeOpEncoding(srcTy.getShape(), srcEnc, dstShape,
                                          dstEnc, loc)))
    return {};
  return dstEnc;
}

// Undo the rewrite when the operand did not end up with a layout the relabel
// works on.
struct ExpandUnlowerableConcat : public OpRewritePattern<ConcatDotOperandOp> {
  using OpRewritePattern<ConcatDotOperandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatDotOperandOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::pair<unsigned, unsigned>> unused;
    if (succeeded(getConcatDotOperandRegisterMap(op, unused)))
      return failure();

    LDBG("expanding a concat back into a join tree: operand layout cannot be "
         "relabeled in registers");

    Location loc = op.getLoc();
    auto fragTy = cast<RankedTensorType>(op.getFragments()[0].getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    int64_t dim = op.getDim();
    int64_t rank = fragTy.getRank();

    // Only a power-of-two count pairs up into a balanced tree; the matcher
    // never builds anything else, so bail rather than drop fragments.
    size_t numFrags = op.getFragments().size();
    if (numFrags < 2 || !llvm::isPowerOf2_64(numFrags))
      return failure();

    // Rebuild the balanced join tree the matcher folded away.
    SmallVector<Value> level(op.getFragments().begin(),
                             op.getFragments().end());
    SmallVector<Value> joins;
    int levels = 0;
    while (level.size() > 1) {
      SmallVector<Value> next;
      for (size_t i = 0; i < level.size(); i += 2) {
        next.push_back(
            triton::JoinOp::create(rewriter, loc, level[i], level[i + 1]));
        joins.push_back(next.back());
      }
      level = std::move(next);
      ++levels;
    }

    // Move the fragment index ahead of the in-fragment concat coord, then
    // flatten, mirroring isConcatOrder.
    SmallVector<int32_t> order;
    for (int64_t d = 0; d < dim; ++d)
      order.push_back(d);
    for (int i = levels; i >= 1; --i)
      order.push_back(rank + i - 1);
    order.push_back(dim);
    for (int64_t d = dim + 1; d < rank; ++d)
      order.push_back(d);

    Value trans = triton::TransOp::create(rewriter, loc, level.front(), order);

    // The chain carries whatever layout the joins infer, not the operand
    // encoding the concat had, so reshape into the inferred one and let a
    // convert restore it.
    auto transTy = cast<RankedTensorType>(trans.getType());
    Attribute reshapeEnc = inferReshapeEncoding(transTy, dstTy.getShape(), loc);
    if (!reshapeEnc) {
      // Nothing consumes the tree yet, so drop it and leave the concat in
      // place.
      rewriter.eraseOp(trans.getDefiningOp());
      for (Value v : llvm::reverse(joins))
        rewriter.eraseOp(v.getDefiningOp());
      return failure();
    }

    Value flat = triton::ReshapeOp::create(
        rewriter, loc,
        RankedTensorType::get(dstTy.getShape(), dstTy.getElementType(),
                              reshapeEnc),
        trans, /*allowReorder=*/false, /*efficientLayout=*/false);
    rewriter.replaceOpWithNewOp<ConvertLayoutOp>(op, dstTy, flat);
    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUCONCATDOTOPERAND
#define GEN_PASS_DEF_TRITONGPUEXPANDCONCATDOTOPERAND
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

class TritonGPUConcatDotOperandPass
    : public impl::TritonGPUConcatDotOperandBase<
          TritonGPUConcatDotOperandPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatchJoinTreeConcat>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

class TritonGPUExpandConcatDotOperandPass
    : public impl::TritonGPUExpandConcatDotOperandBase<
          TritonGPUExpandConcatDotOperandPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ExpandUnlowerableConcat>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu

namespace mlir::triton {
namespace ttg = mlir::triton::gpu;

// Defined here rather than in Transforms/Utility.cpp: backends that substitute
// their own Utility.cpp would otherwise have to carry this too.
LogicalResult getConcatDotOperandRegisterMap(
    ttg::ConcatDotOperandOp op,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &resultRegToFragmentReg) {
  auto dstTy = cast<RankedTensorType>(op.getType());
  auto fragTy = cast<RankedTensorType>(op.getFragments()[0].getType());
  int64_t dim = op.getDim();
  int64_t rank = fragTy.getRank();
  int64_t numFrags = op.getFragments().size();

  // Layouts other than dot_op place lanes differently as `dim` grows, so the
  // relabel would move the wrong registers.
  auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(fragTy.getEncoding());
  if (!dotEnc)
    return failure();

  // Only the contraction axis can be extended in place: growing M or N would
  // change which lane owns an element.
  if (dim != (dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2))
    return failure();

  // Below 8 * kWidth the layout stops scaling with K, so the fragment is not a
  // K-slice of the wider one. kWidth is 0 for layouts that do not use it.
  unsigned kWidth = dotEnc.getKWidth();
  if (kWidth != 0 && fragTy.getShape()[dim] < 8 * (int64_t)kWidth)
    return failure();

  // The mapping is derived on lane 0 and applied to every thread, which only
  // holds if both layouts spread elements over lanes, warps and blocks
  // identically; a valid concat adds register bases and nothing else.
  MLIRContext *ctx = op.getContext();
  LinearLayout dstLL = ttg::toLinearLayout(dstTy);
  LinearLayout fragLL = ttg::toLinearLayout(fragTy);
  for (StringAttr inDim :
       {StringAttr::get(ctx, "lane"), StringAttr::get(ctx, "warp"),
        StringAttr::get(ctx, "block")}) {
    if (dstLL.hasInDim(inDim) != fragLL.hasInDim(inDim))
      return failure();
    if (dstLL.hasInDim(inDim) &&
        dstLL.getBases().lookup(inDim) != fragLL.getBases().lookup(inDim))
      return failure();
  }

  // Coordinates of each register on lane 0. toLinearLayout always names its out
  // dims dim0..dimN in order, so the results line up with the tensor axes.
  StringAttr kReg = StringAttr::get(ctx, "register");
  auto offsetsOf = [&](const LinearLayout &ll) {
    SmallVector<SmallVector<unsigned>> offsets;
    for (int reg = 0; reg < ll.getInDimSize(kReg); ++reg) {
      auto idxs = ll.apply({{kReg, reg},
                            {StringAttr::get(ctx, "lane"), 0},
                            {StringAttr::get(ctx, "warp"), 0},
                            {StringAttr::get(ctx, "block"), 0}});
      assert((int64_t)idxs.size() == rank);
      offsets.push_back(
          llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
    }
    return offsets;
  };
  SmallVector<SmallVector<unsigned>> dstOffsets = offsetsOf(dstLL);
  SmallVector<SmallVector<unsigned>> fragOffsets = offsetsOf(fragLL);
  if (dstOffsets.size() != fragOffsets.size() * numFrags)
    return failure();

  // Flatten a coordinate to index the fragment by. Every coordinate stays
  // within the fragment extents: all dims but `dim` share them with the result,
  // and `dim` is taken modulo below.
  auto coordKey = [&](ArrayRef<unsigned> coord) {
    int64_t key = 0;
    for (int64_t i = 0; i < rank; ++i)
      key = key * fragTy.getShape()[i] + coord[i];
    return key;
  };

  // Enumerate the fragment slots so result coordinates can be inverted back to
  // the register feeding them. Two registers on one coordinate means the layout
  // broadcasts along a dim being indexed, which would drop one and duplicate
  // the other.
  llvm::DenseMap<int64_t, unsigned> fragCoordToReg;
  for (auto [reg, coord] : llvm::enumerate(fragOffsets))
    if (!fragCoordToReg.try_emplace(coordKey(coord), reg).second)
      return failure();

  resultRegToFragmentReg.clear();
  resultRegToFragmentReg.reserve(dstOffsets.size());
  int64_t extent = fragTy.getShape()[dim];
  for (SmallVector<unsigned> &coord : dstOffsets) {
    int64_t fragIdx = coord[dim] / extent;
    coord[dim] %= extent;
    if (fragIdx >= numFrags)
      return failure();
    auto it = fragCoordToReg.find(coordKey(coord));
    if (it == fragCoordToReg.end())
      return failure();
    resultRegToFragmentReg.emplace_back(fragIdx, it->second);
  }
  return success();
}

} // namespace mlir::triton
