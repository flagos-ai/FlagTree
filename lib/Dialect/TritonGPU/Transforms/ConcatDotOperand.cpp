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
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
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

// The single dot_op-encoded type an operand actually has *at the dot*, reached
// from `v` through layout-only converts, or null when it reaches none or more
// than one.
//
// The type has to be read off the dot's own operand rather than off the first
// dot_op encoding on the way there. Layout propagation leaves behind transient
// converts to `dot_op` whose parent is still a blocked layout; no mma ever
// reads those. Stopping at the first one builds extracts against a layout the
// dot does not consume, so the wide value rebuilt from them no longer matches
// the concat result type and the identity fold stops applying, leaving the
// fragments for ReduceDataDuplication to stage through shared memory.
static RankedTensorType getDotOperandType(Value v) {
  RankedTensorType found;
  SmallVector<Value> worklist{v};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (isa<triton::DotOpInterface>(user)) {
        // Only the A and B positions are contraction operands; reaching the
        // accumulator says nothing about how a fragment must be sliced, so skip
        // it rather than give up, which would make the result depend on the
        // order `getUses()` happens to walk.
        if (use.getOperandNumber() > 1)
          continue;
        auto ty = dyn_cast<RankedTensorType>(cur.getType());
        if (!ty || !isa_and_nonnull<DotOperandEncodingAttr>(ty.getEncoding()))
          return {};
        if (found && found != ty)
          return {};
        found = ty;
        continue;
      }
      // Layout-only ops keep the search going; anything else cannot be seen
      // through and is handled by the caller's own use checks.
      if (!isa<ConvertLayoutOp>(user))
        continue;
      if (seen.insert(user->getResult(0)).second)
        worklist.push_back(user->getResult(0));
    }
  }
  return found;
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

// One leaf of a balanced split tree, with the path taken to reach it: one bit
// per level, outermost split first.
struct SplitLeaf {
  Value value;
  SmallVector<unsigned> path;
};

// The single SplitOp consuming `v`, looking through the layout-only converts
// the pipeline inserts between tree levels. Null when `v` is used any other
// way: a value that also escapes elsewhere stays live after the rewrite, so
// folding the tree would not remove it.
static triton::SplitOp getSoleSplitUser(Value v) {
  while (true) {
    if (!v.hasOneUse())
      return {};
    Operation *user = *v.getUsers().begin();
    if (auto split = dyn_cast<triton::SplitOp>(user))
      return split;
    if (!isa<ConvertLayoutOp>(user))
      return {};
    v = user->getResult(0);
  }
}

// Collect the leaves of a perfectly balanced split tree rooted at `v`, the dual
// of collectJoinLeaves. `v` is the value being split, so the recursion walks
// forward through results rather than backward through operands.
static bool collectSplitLeaves(Value v, int depth,
                               SmallVectorImpl<unsigned> &path,
                               SmallVectorImpl<SplitLeaf> &out) {
  if (depth == 0) {
    out.push_back(SplitLeaf{v, llvm::to_vector(ArrayRef<unsigned>(path))});
    return true;
  }
  triton::SplitOp split = getSoleSplitUser(v);
  if (!split)
    return false;
  for (unsigned side : {0u, 1u}) {
    path.push_back(side);
    bool ok = collectSplitLeaves(split->getResult(side), depth - 1, path, out);
    path.pop_back();
    if (!ok)
      return false;
  }
  return true;
}

// Which contraction-axis slice a leaf holds.
//
// A reshape splits K into `levels` axes of extent 2, most significant first,
// and a transpose then moves them to the back in some order. Splits peel the
// trailing axis first, so a leaf's path is indexed by peel order while the K
// index needs bit significance. `significanceOfLevel` bridges the two.
//
// Skipping this mapping stays invisible in testing: when both operands are
// permuted the same way the K-sums still commute to the right result, and only
// a mixed chain, or one operand reaching a wide mma, exposes the wrong slice.
static int64_t getSplitLeafKIndex(ArrayRef<unsigned> path,
                                  ArrayRef<unsigned> significanceOfLevel) {
  int64_t index = 0;
  for (auto [level, side] : llvm::enumerate(path))
    index |= static_cast<int64_t>(side) << significanceOfLevel[level];
  return index;
}

// Undo `reshape -> trans` to learn, for each split level, the significance of
// the K bit that level peels. Returns false when the chain is not a pure
// fragment split of the contraction axis.
//
// `order` is the transpose permutation, mapping result positions to source
// positions. The reshape put the fragment axes at positions
// [concatDim, concatDim + levels), most significant first; the transpose must
// leave every other axis in place and move all fragment axes to the tail.
static bool getSplitLevelSignificance(ArrayRef<int32_t> order, int64_t fragRank,
                                      int64_t concatDim, int levels,
                                      SmallVectorImpl<unsigned> &out) {
  if ((int64_t)order.size() != fragRank + levels)
    return false;

  // Non-fragment axes keep their relative order and stay ahead of the tail.
  for (int64_t d = 0; d < fragRank; ++d) {
    int64_t src = order[d];
    int64_t expect = d < concatDim ? d : d + levels;
    if (src != expect)
      return false;
  }

  // The tail spells the peel order. Splits consume the last axis first, so
  // reverse it: the last tail entry is peeled by level 0 of the recursion.
  out.assign(levels, 0);
  SmallVector<bool> seen(levels, false);
  for (int i = 0; i < levels; ++i) {
    int64_t src = order[fragRank + i];
    // Fragment axes occupy [concatDim, concatDim + levels) in the reshape.
    int64_t axis = src - concatDim;
    if (axis < 0 || axis >= levels || seen[axis])
      return false;
    seen[axis] = true;
    // The reshape lists fragment axes most significant first.
    unsigned significance = levels - 1 - axis;
    // A split consumes the trailing axis, so the outermost split (level 0)
    // peels the last tail position.
    out[levels - 1 - i] = significance;
  }
  return true;
}

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

// Rewrite `reshape -> trans -> split tree` into per-slice extracts of the wide
// operand.
//
// This is the inverse of MatchJoinTreeConcat and the producer the segmented-dot
// rewrites need: a user who slices one wide K tile into fragments and feeds a
// dot chain writes exactly this chain, and the extracts state that all
// fragments are slices of one already-live value. Without it the wide operand
// is materialized only through the transpose chain, and the register-lifetime
// proof in hasNonRegressingRegisterLifetime has nothing to match.
//
// The rewrite keeps element order intact, so it is sound regardless of whether
// the chain is later merged; when the merge does not apply, the extracts lower
// on their own as register selects.
struct MatchSplitTreeExtracts : public OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern<triton::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    // A reordering reshape says nothing about element order, so the fragment
    // axes cannot be tied back to K coordinates.
    if (reshape.getAllowReorder())
      return failure();

    auto srcTy = cast<RankedTensorType>(reshape.getSrc().getType());
    auto splitTy = cast<RankedTensorType>(reshape.getType());
    int64_t fragRank = srcTy.getRank();
    int levels = splitTy.getRank() - fragRank;
    if (fragRank < 1 || levels < 1)
      return failure();

    // The reshape must factor exactly one axis into `levels` extent-2 axes
    // followed by the fragment extent, leaving every other axis alone.
    if (srcTy.getElementType() != splitTy.getElementType())
      return failure();
    int64_t concatDim = 0;
    while (concatDim < fragRank &&
           srcTy.getShape()[concatDim] == splitTy.getShape()[concatDim])
      ++concatDim;
    if (concatDim >= fragRank)
      return failure();

    // Factoring dim `concatDim` inserts the fragment axes at
    // [concatDim, concatDim + levels) and leaves the fragment extent behind
    // them, so the extent lives at concatDim + levels.
    int64_t numFrags = 1LL << levels;
    int64_t fragExtent = splitTy.getShape()[concatDim + levels];
    if (srcTy.getShape()[concatDim] != fragExtent * numFrags)
      return failure();
    for (int i = 0; i < levels; ++i)
      if (splitTy.getShape()[concatDim + i] != 2)
        return failure();
    // Axes after the factored one shift by `levels` but keep their extents.
    for (int64_t d = concatDim + 1; d < fragRank; ++d)
      if (srcTy.getShape()[d] != splitTy.getShape()[d + levels])
        return failure();

    // The transpose moving the fragment axes to the tail, then the split tree.
    // Layout-only converts may sit at either step.
    Value transVal = reshape.getResult();
    while (transVal.hasOneUse() &&
           isa<ConvertLayoutOp>(*transVal.getUsers().begin()))
      transVal = (*transVal.getUsers().begin())->getResult(0);
    if (!transVal.hasOneUse())
      return failure();
    auto trans = dyn_cast<TransposeOpInterface>(*transVal.getUsers().begin());
    if (!trans)
      return failure();

    SmallVector<unsigned> significance;
    if (!getSplitLevelSignificance(trans.getOrder(), fragRank, concatDim,
                                   levels, significance))
      return failure();

    SmallVector<unsigned> path;
    SmallVector<SplitLeaf> leaves;
    if (!collectSplitLeaves(trans->getResult(0), levels, path, leaves))
      return failure();
    if ((int64_t)leaves.size() != numFrags)
      return failure();

    // Every leaf must have the expected fragment shape and one encoding, and
    // must reach a dot. The leaves carry whatever layout the split infers; the
    // dot_op encoding is only assigned further down the convert chain, so match
    // on reaching a dot rather than on the encoding already being dot_op.
    SmallVector<int64_t> fragShape(srcTy.getShape());
    fragShape[concatDim] = fragExtent;
    auto leafTy = dyn_cast<RankedTensorType>(leaves.front().value.getType());
    if (!leafTy || leafTy.getShape() != ArrayRef<int64_t>(fragShape))
      return failure();
    for (const SplitLeaf &leaf : leaves) {
      if (leaf.value.getType() != leafTy)
        return failure();
      if (!onlyFeedsDot(leaf.value))
        return failure();
    }

    // Extract in the dot_op encoding the dots actually consume, not the leaf's
    // own layout: only the operand layout is guaranteed to slice into
    // per-thread register subsets, which is what the extract lowering needs.
    auto dotOpTy = getDotOperandType(leaves.front().value);
    if (!dotOpTy || dotOpTy.getShape() != leafTy.getShape())
      return failure();
    for (const SplitLeaf &leaf : leaves)
      if (getDotOperandType(leaf.value) != dotOpTy)
        return failure();

    auto wideTy = RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), dotOpTy.getEncoding());

    // Check the layout admits the register subset the extracts need before
    // touching the IR: the proof only reads types, and bailing out after the
    // ops exist would leave a failed match to clean up after itself. Keeping an
    // unlowerable chain from reaching the backend is what avoids a hard error
    // there.
    SmallVector<std::pair<unsigned, unsigned>> regMap;
    if (failed(getDotOperandSliceRegisterMap(wideTy, dotOpTy, concatDim,
                                             numFrags, regMap)))
      return failure();

    rewriter.setInsertionPoint(reshape);
    Value wide = ConvertLayoutOp::create(rewriter, reshape.getLoc(), wideTy,
                                         reshape.getSrc());

    SmallVector<Value> extracts;
    for (const SplitLeaf &leaf : leaves) {
      int64_t index = getSplitLeafKIndex(leaf.path, significance);
      extracts.push_back(ExtractDotOperandOp::create(
          rewriter, leaf.value.getLoc(), dotOpTy, wide, concatDim, index));
    }

    LDBG("rewriting a split tree of " << numFrags << " fragments on dim "
                                      << concatDim);

    // The leaves keep their own layout, so convert back and let
    // RemoveLayoutConversions collapse the round trip into the dots.
    for (auto [leaf, extract] : llvm::zip(leaves, extracts)) {
      Value restored = ConvertLayoutOp::create(rewriter, leaf.value.getLoc(),
                                               leafTy, extract);
      rewriter.replaceAllUsesWith(leaf.value, restored);
    }
    return success();
  }
};

// A concat over a complete, ordered set of extracts is an identity. The fold
// itself lives on the op as a canonicalization (see
// `ConcatDotOperandOp::canonicalize`), because the extracts a segmented chain
// leaves behind only become recognizable as a cover of one root after layout
// propagation has unified their sources -- which happens well after this pass.
// Running it here as well gives a chain that already has a same-typed cover a
// zero-cost way to recover its wide operand within one fixpoint.
struct FoldConcatOfExtracts : public OpRewritePattern<ConcatDotOperandOp> {
  using OpRewritePattern<ConcatDotOperandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatDotOperandOp op,
                                PatternRewriter &rewriter) const override {
    return ConcatDotOperandOp::canonicalize(op, rewriter);
  }
};

namespace ttng = mlir::triton::nvidia_gpu;

enum class SharedSliceKind { IndexedCarrier, WideSubslice };
enum class SharedOperandSide { A, B };

struct SharedSlice {
  SharedSliceKind kind;
  Value root;
  int64_t index;
  int64_t numSlices;
  int64_t sliceK;
  Attribute wideEncoding;
};

// A tile-written rank-3 carrier can only be reinterpreted as one wide NVMMA
// operand when the physical swizzle picture is identical. A rank-2 wide
// operand reached through memdesc_subslice needs no such proof: the original
// descriptor is already the exact wide view.
static Attribute getWideCarrierEncoding(MemDescType carrierTy) {
  auto nvmma = dyn_cast<NVMMASharedEncodingAttr>(carrierTy.getEncoding());
  if (!nvmma || nvmma.getTransposed())
    return {};
  int64_t innerBytes = carrierTy.getShape().back() *
                       carrierTy.getElementType().getIntOrFloatBitWidth() / 8;
  if (nvmma.getSwizzlingByteWidth() != 0 &&
      innerBytes > nvmma.getSwizzlingByteWidth())
    return {};
  CTAEncodingAttr cta = nvmma.getCTALayout();
  if (!cta || cta.getRank() <= 2)
    return carrierTy.getEncoding();
  if (llvm::any_of(cta.getCTAsPerCGA(), [](unsigned v) { return v != 1; }) ||
      llvm::any_of(cta.getCTASplitNum(), [](unsigned v) { return v != 1; }))
    return {};
  MLIRContext *ctx = carrierTy.getContext();
  return NVMMASharedEncodingAttr::get(
      ctx, nvmma.getSwizzlingByteWidth(), nvmma.getTransposed(),
      nvmma.getElementBitWidth(), nvmma.getFp4Padded(),
      CTAEncodingAttr::getDefault(ctx, 2));
}

static std::optional<SharedSlice> getSharedSlice(Value operand,
                                                 SharedOperandSide side) {
  // The rank-3 carrier representation is currently only physically
  // composable for B: [tile, K, N] can be reinterpreted as [tile*K, N].
  // An A carrier [tile, M, K] would require interleaving the tile and K axes,
  // which is not a plain descriptor reinterpretation.
  if (side == SharedOperandSide::B)
    if (auto index = operand.getDefiningOp<MemDescIndexOp>()) {
      APInt constantIndex;
      if (!matchPattern(index.getIndex(), m_ConstantInt(&constantIndex)))
        return std::nullopt;
      auto rootTy = cast<MemDescType>(index.getSrc().getType());
      auto sliceTy = cast<MemDescType>(operand.getType());
      if (rootTy.getRank() != 3 || sliceTy.getRank() != 2 ||
          rootTy.getElementType() != sliceTy.getElementType() ||
          rootTy.getMemorySpace() != sliceTy.getMemorySpace() ||
          rootTy.getMutableMemory() != sliceTy.getMutableMemory() ||
          rootTy.getShape()[1] != sliceTy.getShape()[0] ||
          rootTy.getShape()[2] != sliceTy.getShape()[1])
        return std::nullopt;
      Attribute wideEncoding = getWideCarrierEncoding(rootTy);
      if (!wideEncoding || sliceTy.getEncoding() != wideEncoding)
        return std::nullopt;
      return SharedSlice{SharedSliceKind::IndexedCarrier, index.getSrc(),
                         constantIndex.getSExtValue(),    rootTy.getShape()[0],
                         sliceTy.getShape()[0],           wideEncoding};
    }

  if (auto subslice = operand.getDefiningOp<MemDescSubsliceOp>()) {
    auto rootTy = cast<MemDescType>(subslice.getSrc().getType());
    auto sliceTy = cast<MemDescType>(operand.getType());
    if (rootTy.getRank() != 2 || sliceTy.getRank() != 2 ||
        rootTy.getElementType() != sliceTy.getElementType() ||
        rootTy.getEncoding() != sliceTy.getEncoding() ||
        rootTy.getMemorySpace() != sliceTy.getMemorySpace() ||
        rootTy.getMutableMemory() != sliceTy.getMutableMemory())
      return std::nullopt;
    ArrayRef<int32_t> offsets = subslice.getOffsets();
    if (offsets.size() != 2)
      return std::nullopt;
    int64_t index;
    int64_t numSlices;
    int64_t sliceK;
    if (side == SharedOperandSide::A) {
      if (rootTy.getShape()[0] != sliceTy.getShape()[0] || offsets[0] != 0 ||
          rootTy.getShape()[1] % sliceTy.getShape()[1] != 0 ||
          offsets[1] % sliceTy.getShape()[1] != 0)
        return std::nullopt;
      sliceK = sliceTy.getShape()[1];
      index = offsets[1] / sliceK;
      numSlices = rootTy.getShape()[1] / sliceK;
    } else {
      if (rootTy.getShape()[1] != sliceTy.getShape()[1] || offsets[1] != 0 ||
          rootTy.getShape()[0] % sliceTy.getShape()[0] != 0 ||
          offsets[0] % sliceTy.getShape()[0] != 0)
        return std::nullopt;
      sliceK = sliceTy.getShape()[0];
      index = offsets[0] / sliceK;
      numSlices = rootTy.getShape()[0] / sliceK;
    }
    return SharedSlice{SharedSliceKind::WideSubslice,
                       subslice.getSrc(),
                       index,
                       numSlices,
                       sliceK,
                       rootTy.getEncoding()};
  }
  return std::nullopt;
}

struct SegmentedDotChain {
  SmallVector<ttng::WarpGroupDotOp> dots;
  SmallVector<Value> aFragments;
  std::optional<SharedSlice> aSlices;
  std::optional<SharedSlice> bSlices;
};

// The backend distinguishes logical dots from physical MMA instructions. A
// wide K operand still lowers to the same fixed-granularity instruction count;
// merging is profitable only when logical bookkeeping is removed without
// increasing another hardware cost.
struct DotOperandMergePlan {
  int64_t logicalDotsBefore;
  int64_t logicalDotsAfter = 1;
  int64_t physicalMmaBefore;
  int64_t physicalMmaAfter;
  bool registerLifetimeNonIncreasing;
  bool sharedViewCompatible;

  bool isProfitable() const {
    if (!registerLifetimeNonIncreasing || !sharedViewCompatible ||
        physicalMmaAfter > physicalMmaBefore)
      return false;
    return physicalMmaAfter < physicalMmaBefore ||
           logicalDotsAfter < logicalDotsBefore;
  }
};

// Return the K granularity of the native instruction used by the v2/v3
// NVIDIA paths. This deliberately models only the instruction families this
// pass can lower: MMAv2 register dots and Hopper WGMMA (MMAv3). MMAv5/TMEM is
// a different operation family and must not silently enter this model.
static std::optional<int64_t> getNativeMmaK(Type elementTy,
                                            InputPrecision inputPrecision,
                                            NvidiaMmaEncodingAttr mma) {
  // Turing's native half precision instruction is m16n8k8 (the integer
  // variant is m8n8k16). Ampere MMAv2 uses k16 for f16/bf16 and k32 for
  // int8/fp8. Hopper WGMMA is handled through the MMAv3 caller below.
  if (mma.isTuring()) {
    if (elementTy.isF16())
      return 8;
    if (elementTy.isInteger(8))
      return 16;
    return std::nullopt;
  }
  if (elementTy.isF16() || elementTy.isBF16())
    return 16;
  if (elementTy.isF32())
    return inputPrecision == InputPrecision::TF32 ? std::optional<int64_t>(8)
                                                  : std::nullopt;
  if (elementTy.isInteger(8) ||
      llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elementTy))
    return 32;
  // MMAv2 has additional packed integer/fp4 forms, but this pass does not
  // rewrite their accumulation boundaries yet. Keep them segmented until a
  // target-specific numerical proof is added.
  return std::nullopt;
}

static std::optional<int64_t> getNativeMmaK(triton::DotOp dot,
                                            RankedTensorType fragmentTy) {
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(fragmentTy.getEncoding());
  if (!dotEnc)
    return std::nullopt;
  auto parent = dyn_cast<NvidiaMmaEncodingAttr>(dotEnc.getParent());
  if (!parent || parent.getVersionMajor() != 2)
    return std::nullopt;
  if (fragmentTy.getElementType().isF64()) {
    ModuleOp module = dot->getParentOfType<ModuleOp>();
    if (!module)
      return std::nullopt;
    int capability = getNVIDIAComputeCapability(module);
    if (capability == 90)
      return 16;
    if (capability >= 80 && capability < 90)
      return 4;
    return std::nullopt;
  }
  return getNativeMmaK(fragmentTy.getElementType(), dot.getInputPrecision(),
                       parent);
}

static std::optional<int64_t> getNativeMmaK(ttng::WarpGroupDotOp dot,
                                            RankedTensorType fragmentTy) {
  auto dotEnc = dyn_cast<DotOperandEncodingAttr>(fragmentTy.getEncoding());
  if (!dotEnc)
    return std::nullopt;
  auto parent = dyn_cast<NvidiaMmaEncodingAttr>(dotEnc.getParent());
  if (!parent || parent.getVersionMajor() != 3)
    return std::nullopt;
  return getNativeMmaK(fragmentTy.getElementType(), dot.getInputPrecision(),
                       parent);
}

static std::optional<int64_t> getNativeMmaK(ttng::WarpGroupDotOp dot) {
  auto parent =
      dyn_cast<NvidiaMmaEncodingAttr>(dot.getD().getType().getEncoding());
  if (!parent || parent.getVersionMajor() != 3)
    return std::nullopt;
  Type elementTy = cast<TensorOrMemDesc>(dot.getA().getType()).getElementType();
  return getNativeMmaK(elementTy, dot.getInputPrecision(), parent);
}

static bool mergedWarpGroupDotNeedsPartialAccumulator(ttng::WarpGroupDotOp dot,
                                                      int64_t mergedK) {
  Type elementTy = cast<TensorOrMemDesc>(dot.getA().getType()).getElementType();
  bool isFp8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType, Float8E5M2FNUZType,
                         Float8E4M3FNUZType>(elementTy);
  bool accF32 = dot.getD().getType().getElementType().isF32();
  return isFp8 && accF32 && dot.getMaxNumImpreciseAcc() <= mergedK;
}

static std::optional<DotOperandMergePlan>
makeMergePlan(ArrayRef<int64_t> fragmentKs, int64_t nativeK,
              bool registerLifetimeNonIncreasing, bool sharedViewCompatible) {
  if (fragmentKs.empty() || nativeK <= 0)
    return std::nullopt;
  int64_t physicalInstructions = 0;
  int64_t totalK = 0;
  for (int64_t fragmentK : fragmentKs) {
    // Do not hide an implicit padding/partial-accumulation step in a merge.
    // All currently supported v2/v3 instruction shapes expose an integral K
    // tile at this point in the pipeline.
    if (fragmentK <= 0 || fragmentK % nativeK != 0)
      return std::nullopt;
    physicalInstructions += fragmentK / nativeK;
    totalK += fragmentK;
  }
  if (totalK % nativeK != 0)
    return std::nullopt;
  int64_t mergedInstructions = totalK / nativeK;
  DotOperandMergePlan plan{static_cast<int64_t>(fragmentKs.size()),
                           1,
                           physicalInstructions,
                           mergedInstructions,
                           registerLifetimeNonIncreasing,
                           sharedViewCompatible};
  return plan;
}

static bool isMmaV2OrV3(DotOperandEncodingAttr encoding,
                        unsigned expectedVersion) {
  auto parent = dyn_cast<NvidiaMmaEncodingAttr>(encoding.getParent());
  return parent && parent.getVersionMajor() == expectedVersion;
}

// DotOperandEncoding is still defined over the power-of-two linear-layout
// domain. A non-power-of-two segmented chain is therefore valid input to this
// pass, but it must remain segmented unless a future layout implementation can
// represent the merged register tensor. Check this before asking
// toLinearLayout() for a mapping so the conservative fallback cannot assert.
static bool isRepresentableMmaShape(RankedTensorType type) {
  return llvm::all_of(type.getShape(), [](int64_t extent) {
    return extent > 0 && llvm::isPowerOf2_64(extent);
  });
}

static bool isCompleteExtractCover(ArrayRef<Value> fragments, int64_t dim,
                                   Value &src) {
  if (fragments.empty())
    return false;
  int64_t offset = 0;
  for (auto [i, fragment] : llvm::enumerate(fragments)) {
    // Layout-only converts may sit between the extract and the dot; they do not
    // move data between threads, so look through them.
    auto extract = skipConverts(fragment).getDefiningOp<ExtractDotOperandOp>();
    if (!extract || extract.getDim() != dim ||
        extract.getIndex() != static_cast<int64_t>(i))
      return false;
    // The producer builds one layout convert per extract over the same wide
    // value, so the extract sources are distinct SSA values that alias one
    // root. Compare the roots, not the converts, or a cover that is complete
    // by construction reads as several unrelated fragments.
    Value extractSrc = skipConverts(extract.getSrc());
    if (!src)
      src = extractSrc;
    else if (src != extractSrc)
      return false;
    // ExtractDotOperand currently encodes equal-sized slices. Keep the
    // explicit offset accumulation here so this proof remains correct when
    // unequal static slices are added later.
    auto srcTy = dyn_cast<RankedTensorType>(extract.getSrc().getType());
    auto sliceTy = dyn_cast<RankedTensorType>(fragment.getType());
    if (!srcTy || !sliceTy || srcTy.getShape()[dim] <= offset ||
        offset + sliceTy.getShape()[dim] > srcTy.getShape()[dim])
      return false;
    offset += sliceTy.getShape()[dim];
  }
  return src && offset == cast<RankedTensorType>(src.getType()).getShape()[dim];
}

// A complete extract cover is the only case in which gathering all register
// operands is guaranteed not to extend a live range.
static bool hasNonRegressingRegisterLifetime(ArrayRef<Value> fragments,
                                             int64_t dim) {
  Value src;
  return isCompleteExtractCover(fragments, dim, src);
}

// Moving several shared-memory reads to the chain tail is only valid when no
// intervening operation can mutate observable state. Pure address/view and
// register operations are harmless; barriers, copies, stores and unknown
// side effects make the original read timing semantically significant.
static bool hasOnlyPureOpsBetween(Operation *from, Operation *to) {
  for (Operation *op = from->getNextNode(); op && op != to;
       op = op->getNextNode())
    if (!isMemoryEffectFree(op))
      return false;
  return from->getBlock() == to->getBlock();
}

// Discardable attributes are allowed to carry scheduling and synchronization
// contracts. Their semantics are not part of the dot op's generated builder
// and cannot in general be combined when several dots become one. Run this
// pass before those annotations are introduced; if an input already carries
// any, keep the chain unchanged rather than silently dropping a contract.
static bool hasNoRewriteSensitiveAttrs(Operation *op) {
  return op->getDiscardableAttrs().empty();
}

static LogicalResult matchSegmentedDotChain(ttng::WarpGroupDotOp tail,
                                            SegmentedDotChain &chain) {
  if (tail.needsPartialAccumulator() || tail.getUseC() ||
      !hasNoRewriteSensitiveAttrs(tail))
    return failure();
  std::optional<SharedSlice> tailBSlice =
      getSharedSlice(tail.getB(), SharedOperandSide::B);
  if (!tailBSlice || tailBSlice->numSlices < 2 ||
      tailBSlice->index != tailBSlice->numSlices - 1)
    return failure();

  bool aInShared = isa<MemDescType>(tail.getA().getType());
  std::optional<SharedSlice> tailASlice;
  if (aInShared) {
    tailASlice = getSharedSlice(tail.getA(), SharedOperandSide::A);
    if (!tailASlice || tailASlice->kind != SharedSliceKind::WideSubslice ||
        tailASlice->index != tailBSlice->index ||
        tailASlice->numSlices != tailBSlice->numSlices ||
        tailASlice->sliceK != tailBSlice->sliceK)
      return failure();
  }

  SmallVector<ttng::WarpGroupDotOp> reversed{tail};
  ttng::WarpGroupDotOp current = tail;
  for (int64_t expected = tailBSlice->numSlices - 2; expected >= 0;
       --expected) {
    auto previous = current.getC().getDefiningOp<ttng::WarpGroupDotOp>();
    if (!previous || !previous.getD().hasOneUse() || previous.getUseC() ||
        previous.needsPartialAccumulator() ||
        !hasNoRewriteSensitiveAttrs(previous) ||
        previous->getBlock() != current->getBlock() ||
        !hasOnlyPureOpsBetween(previous, current) ||
        previous.getIsAsync() != current.getIsAsync() ||
        previous.getInputPrecision() != current.getInputPrecision() ||
        previous.getMaxNumImpreciseAcc() != current.getMaxNumImpreciseAcc())
      return failure();
    std::optional<SharedSlice> bSlice =
        getSharedSlice(previous.getB(), SharedOperandSide::B);
    if (!bSlice || bSlice->kind != tailBSlice->kind ||
        bSlice->root != tailBSlice->root || bSlice->index != expected ||
        bSlice->numSlices != tailBSlice->numSlices ||
        bSlice->sliceK != tailBSlice->sliceK)
      return failure();
    if (aInShared) {
      std::optional<SharedSlice> aSlice =
          getSharedSlice(previous.getA(), SharedOperandSide::A);
      if (!aSlice || aSlice->kind != tailASlice->kind ||
          aSlice->root != tailASlice->root || aSlice->index != expected ||
          aSlice->numSlices != tailASlice->numSlices ||
          aSlice->sliceK != tailASlice->sliceK)
        return failure();
    }
    reversed.push_back(previous);
    current = previous;
  }

  for (ttng::WarpGroupDotOp dot : llvm::reverse(reversed))
    chain.dots.push_back(dot);
  int64_t fragmentK = tailBSlice->sliceK;
  if (aInShared) {
    Type aTy = chain.dots.front().getA().getType();
    for (ttng::WarpGroupDotOp dot : chain.dots)
      if (dot.getA().getType() != aTy)
        return failure();
    chain.aSlices = *tailASlice;
  } else {
    auto fragmentTy =
        dyn_cast<RankedTensorType>(chain.dots.front().getA().getType());
    if (!fragmentTy || !isa<DotOperandEncodingAttr>(fragmentTy.getEncoding()) ||
        fragmentTy.getShape().back() != fragmentK)
      return failure();
    auto dotEncoding = cast<DotOperandEncodingAttr>(fragmentTy.getEncoding());
    auto mmaParent = dyn_cast<NvidiaMmaEncodingAttr>(dotEncoding.getParent());
    if (!mmaParent || mmaParent.getVersionMajor() != 3)
      return failure();
    for (ttng::WarpGroupDotOp dot : chain.dots) {
      if (dot.getA().getType() != fragmentTy)
        return failure();
      chain.aFragments.push_back(dot.getA());
    }
    if (!hasNonRegressingRegisterLifetime(chain.aFragments,
                                          fragmentTy.getRank() - 1)) {
      LDBG("keeping segmented WGMMA: merging would extend independent A "
           "fragment lifetimes");
      return failure();
    }
  }

  auto nativeK = getNativeMmaK(chain.dots.front());
  if (!nativeK)
    return failure();
  SmallVector<int64_t> fragmentKs(chain.dots.size(), fragmentK);
  auto plan = makeMergePlan(fragmentKs, *nativeK,
                            /*registerLifetimeNonIncreasing=*/true,
                            /*sharedViewCompatible=*/true);
  if (!plan || !plan->isProfitable())
    return failure();
  if (mergedWarpGroupDotNeedsPartialAccumulator(
          chain.dots.front(), plan->physicalMmaAfter * *nativeK))
    return failure();
  LDBG("WGMMA merge plan: logical " << plan->logicalDotsBefore << " -> "
                                    << plan->logicalDotsAfter << ", physical "
                                    << plan->physicalMmaBefore << " -> "
                                    << plan->physicalMmaAfter);

  if (!aInShared) {
    auto fragmentTy =
        cast<RankedTensorType>(chain.aFragments.front().getType());
    SmallVector<int64_t> wideShape(fragmentTy.getShape());
    wideShape.back() *= tailBSlice->numSlices;
    auto wideTy = RankedTensorType::get(wideShape, fragmentTy.getElementType(),
                                        fragmentTy.getEncoding());
    if (!isRepresentableMmaShape(wideTy))
      return failure();
    SmallVector<std::pair<unsigned, unsigned>> unused;
    if (failed(getDotOperandSliceRegisterMap(wideTy, fragmentTy,
                                             fragmentTy.getRank() - 1,
                                             tailBSlice->numSlices, unused)))
      return failure();
  }
  chain.bSlices = *tailBSlice;
  return success();
}

struct MergeSegmentedWarpGroupDot
    : public OpRewritePattern<ttng::WarpGroupDotOp> {
  using OpRewritePattern<ttng::WarpGroupDotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ttng::WarpGroupDotOp tail,
                                PatternRewriter &rewriter) const override {
    SegmentedDotChain chain;
    if (failed(matchSegmentedDotChain(tail, chain)))
      return failure();

    Location loc = tail.getLoc();
    Value wideA;
    if (chain.aSlices) {
      wideA = chain.aSlices->root;
    } else {
      auto fragmentTy =
          cast<RankedTensorType>(chain.aFragments.front().getType());
      SmallVector<int64_t> wideShape(fragmentTy.getShape());
      wideShape.back() *= chain.dots.size();
      auto wideATy = RankedTensorType::get(
          wideShape, fragmentTy.getElementType(), fragmentTy.getEncoding());
      wideA = ConcatDotOperandOp::create(
          rewriter, loc, wideATy, chain.aFragments,
          rewriter.getI32IntegerAttr(fragmentTy.getRank() - 1));
    }

    assert(chain.bSlices && "matched chain must have shared slices");
    Value wideB = chain.bSlices->root;
    if (chain.bSlices->kind == SharedSliceKind::IndexedCarrier) {
      auto carrierTy = cast<MemDescType>(chain.bSlices->root.getType());
      auto wideBTy = MemDescType::get(
          {carrierTy.getShape()[0] * carrierTy.getShape()[1],
           carrierTy.getShape()[2]},
          carrierTy.getElementType(), chain.bSlices->wideEncoding,
          carrierTy.getMemorySpace(), carrierTy.getMutableMemory());
      wideB = MemDescReinterpretOp::create(rewriter, loc, wideBTy,
                                           chain.bSlices->root);
    }

    auto merged = ttng::WarpGroupDotOp::create(
        rewriter, loc, tail.getD().getType(), wideA, wideB,
        chain.dots.front().getC(), Value(), tail.getInputPrecision(),
        tail.getMaxNumImpreciseAcc(), tail.getIsAsync());
    rewriter.replaceOp(tail, merged.getD());
    for (ttng::WarpGroupDotOp dot :
         llvm::reverse(ArrayRef(chain.dots).drop_back()))
      rewriter.eraseOp(dot);
    LDBG("merged a non-regressing " << chain.dots.size()
                                    << "-segment WGMMA chain");
    return success();
  }
};

struct SegmentedMmaChain {
  SmallVector<triton::DotOp> dots;
  SmallVector<Value> aFragments;
  SmallVector<Value> bFragments;
  // Every op strictly between the head and the tail of the chain: the dots that
  // the merged one replaces, plus the accumulator converts that link them.
  // Stored consumer-before-producer so erasing in order never hits a live use.
  SmallVector<Operation *> deadInterior;
};

// A chain built from `tl.split` leaves reaches this pass before layout
// conversions are cleaned up, so the accumulator often hops through a pair of
// value-preserving converts between two dots. Walk them, but only while each
// result has a single use: a partial sum consumed elsewhere stays observable
// and must keep being computed.
static triton::DotOp
skipAccumulatorConverts(Value acc, SmallVectorImpl<Operation *> &converts) {
  while (auto cvt = acc.getDefiningOp<ConvertLayoutOp>()) {
    if (!cvt.getResult().hasOneUse())
      return {};
    converts.push_back(cvt);
    acc = cvt.getSrc();
  }
  return acc.getDefiningOp<triton::DotOp>();
}

// Unlike the WGMMA path, this needs no partial-accumulator check for FP8 with
// an f32 accumulator: `maxNumImpreciseAcc` only drives the chunked accumulation
// in the WGMMA lowering, and the MMAv2 lowering never reads it. A wider K
// therefore introduces no new accumulation boundary here, so precision does not
// depend on how the K tile is split.
static bool hasSupportedMmaV2Numerics(triton::DotOp dot) {
  Type a = dot.getA().getType().getElementType();
  Type b = dot.getB().getType().getElementType();
  Type d = dot.getD().getType().getElementType();
  if (a.isF16() && b.isF16())
    return d.isF16() || d.isF32();
  if (a.isBF16() && b.isBF16())
    return d.isF32();
  if (a.isF32() && b.isF32())
    return d.isF32() && dot.getInputPrecision() == InputPrecision::TF32;
  if (a.isF64() && b.isF64())
    return d.isF64();
  if (a.isInteger(8) && b.isInteger(8))
    return d.isInteger(32);
  bool aFp8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(a);
  bool bFp8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(b);
  return aFp8 && bFp8 && (d.isF16() || d.isF32());
}

static LogicalResult matchSegmentedMmaChain(triton::DotOp tail,
                                            SegmentedMmaChain &chain) {
  if (!hasSupportedMmaV2Numerics(tail) || !hasNoRewriteSensitiveAttrs(tail))
    return failure();

  SmallVector<triton::DotOp> reversed{tail};
  // Interior ops in the order the walk meets them, which is already
  // consumer-before-producer. A step that is later rejected leaves its
  // converts here, so remember how far the accepted prefix reaches.
  SmallVector<Operation *> interior;
  size_t acceptedInterior = 0;
  triton::DotOp current = tail;
  while (auto previous = skipAccumulatorConverts(current.getC(), interior)) {
    if (!previous.getD().hasOneUse() || !hasNoRewriteSensitiveAttrs(previous) ||
        previous->getBlock() != current->getBlock() ||
        !hasOnlyPureOpsBetween(previous, current) ||
        previous.getInputPrecision() != current.getInputPrecision() ||
        previous.getMaxNumImpreciseAcc() != current.getMaxNumImpreciseAcc() ||
        previous.getA().getType() != tail.getA().getType() ||
        previous.getB().getType() != tail.getB().getType())
      break;
    reversed.push_back(previous);
    interior.push_back(previous);
    acceptedInterior = interior.size();
    current = previous;
  }
  if (reversed.size() < 2)
    return failure();
  for (triton::DotOp dot : llvm::reverse(reversed)) {
    chain.dots.push_back(dot);
    chain.aFragments.push_back(dot.getA());
    chain.bFragments.push_back(dot.getB());
  }
  // The merged dot takes over the head's accumulator operand, so the head is
  // replaced like the rest. Converts pushed past it belong to a rejected step
  // and are left alone.
  interior.resize(acceptedInterior);
  chain.deadInterior = std::move(interior);

  auto aTy = cast<RankedTensorType>(chain.aFragments.front().getType());
  auto bTy = cast<RankedTensorType>(chain.bFragments.front().getType());
  auto aEncoding = dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding());
  auto bEncoding = dyn_cast<DotOperandEncodingAttr>(bTy.getEncoding());
  if (!aEncoding || !bEncoding || aEncoding.getOpIdx() != 0 ||
      bEncoding.getOpIdx() != 1 || !isMmaV2OrV3(aEncoding, 2) ||
      !isMmaV2OrV3(bEncoding, 2))
    return failure();
  int64_t aDim = aTy.getRank() - 1;
  int64_t bDim = bTy.getRank() - 2;
  if (!hasNonRegressingRegisterLifetime(chain.aFragments, aDim) ||
      !hasNonRegressingRegisterLifetime(chain.bFragments, bDim)) {
    LDBG("keeping segmented MMA: merging would extend operand lifetimes");
    return failure();
  }

  auto nativeK = getNativeMmaK(chain.dots.front(), aTy);
  if (!nativeK)
    return failure();
  SmallVector<int64_t> fragmentKs;
  fragmentKs.reserve(chain.aFragments.size());
  for (Value a : chain.aFragments)
    fragmentKs.push_back(cast<RankedTensorType>(a.getType()).getShape()[aDim]);
  auto plan = makeMergePlan(fragmentKs, *nativeK,
                            /*registerLifetimeNonIncreasing=*/true,
                            /*sharedViewCompatible=*/true);
  if (!plan || !plan->isProfitable())
    return failure();
  LDBG("MMA merge plan: logical " << plan->logicalDotsBefore << " -> "
                                  << plan->logicalDotsAfter << ", physical "
                                  << plan->physicalMmaBefore << " -> "
                                  << plan->physicalMmaAfter);

  SmallVector<int64_t> wideAShape(aTy.getShape());
  SmallVector<int64_t> wideBShape(bTy.getShape());
  wideAShape[aDim] *= chain.dots.size();
  wideBShape[bDim] *= chain.dots.size();
  auto wideATy = RankedTensorType::get(wideAShape, aTy.getElementType(),
                                       aTy.getEncoding());
  auto wideBTy = RankedTensorType::get(wideBShape, bTy.getElementType(),
                                       bTy.getEncoding());
  if (!isRepresentableMmaShape(wideATy) || !isRepresentableMmaShape(wideBTy))
    return failure();
  SmallVector<std::pair<unsigned, unsigned>> unused;
  if (failed(getDotOperandSliceRegisterMap(wideATy, aTy, aDim,
                                           chain.dots.size(), unused)) ||
      failed(getDotOperandSliceRegisterMap(wideBTy, bTy, bDim,
                                           chain.dots.size(), unused)))
    return failure();
  return success();
}

struct MergeSegmentedMma : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::DotOp tail,
                                PatternRewriter &rewriter) const override {
    SegmentedMmaChain chain;
    if (failed(matchSegmentedMmaChain(tail, chain)))
      return failure();

    Location loc = tail.getLoc();
    auto aTy = cast<RankedTensorType>(chain.aFragments.front().getType());
    auto bTy = cast<RankedTensorType>(chain.bFragments.front().getType());
    int64_t aDim = aTy.getRank() - 1;
    int64_t bDim = bTy.getRank() - 2;
    SmallVector<int64_t> wideAShape(aTy.getShape());
    SmallVector<int64_t> wideBShape(bTy.getShape());
    wideAShape[aDim] *= chain.dots.size();
    wideBShape[bDim] *= chain.dots.size();
    auto wideATy = RankedTensorType::get(wideAShape, aTy.getElementType(),
                                         aTy.getEncoding());
    auto wideBTy = RankedTensorType::get(wideBShape, bTy.getElementType(),
                                         bTy.getEncoding());
    Value wideA =
        ConcatDotOperandOp::create(rewriter, loc, wideATy, chain.aFragments,
                                   rewriter.getI32IntegerAttr(aDim));
    Value wideB =
        ConcatDotOperandOp::create(rewriter, loc, wideBTy, chain.bFragments,
                                   rewriter.getI32IntegerAttr(bDim));
    Value merged = triton::DotOp::create(
        rewriter, loc, tail.getD().getType(), wideA, wideB,
        chain.dots.front().getC(), tail.getInputPrecision(),
        tail.getMaxNumImpreciseAcc());
    rewriter.replaceOp(tail, merged);
    // Already ordered consumer-before-producer, so every use is gone by the
    // time an op's turn comes.
    for (Operation *op : chain.deadInterior)
      rewriter.eraseOp(op);
    LDBG("merged a non-regressing " << chain.dots.size()
                                    << "-segment MMA chain");
    return success();
  }
};

} // namespace

#define GEN_PASS_DEF_TRITONGPUCONCATDOTOPERAND
#define GEN_PASS_DEF_TRITONGPUEXPANDCONCATDOTOPERAND
#define GEN_PASS_DEF_TRITONGPUMERGESEGMENTEDDOT
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

class TritonGPUMergeSegmentedDotPass
    : public impl::TritonGPUMergeSegmentedDotBase<
          TritonGPUMergeSegmentedDotPass> {
public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    // MatchSplitTreeExtracts turns a user-written fragment split into the
    // extract cover the two merge patterns look for, so it must run in the same
    // greedy fixpoint rather than a later pass.
    patterns.add<MatchSplitTreeExtracts, MergeSegmentedWarpGroupDot,
                 MergeSegmentedMma, FoldConcatOfExtracts>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu

namespace mlir::triton {
namespace ttg = mlir::triton::gpu;

// Defined here rather than in Transforms/Utility.cpp: backends that substitute
// their own Utility.cpp would otherwise have to carry this too.
LogicalResult getDotOperandSliceRegisterMap(
    RankedTensorType dstTy, RankedTensorType fragTy, int64_t dim,
    int64_t numFrags,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &resultRegToFragmentReg) {
  int64_t rank = fragTy.getRank();

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
  MLIRContext *ctx = dstTy.getContext();
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

LogicalResult getConcatDotOperandRegisterMap(
    ttg::ConcatDotOperandOp op,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &resultRegToFragmentReg) {
  return getDotOperandSliceRegisterMap(
      cast<RankedTensorType>(op.getType()),
      cast<RankedTensorType>(op.getFragments()[0].getType()), op.getDim(),
      op.getFragments().size(), resultRegToFragmentReg);
}

LogicalResult
getExtractDotOperandRegisterMap(ttg::ExtractDotOperandOp op,
                                SmallVectorImpl<unsigned> &resultRegToSrcReg) {
  auto srcTy = cast<RankedTensorType>(op.getSrc().getType());
  auto resultTy = cast<RankedTensorType>(op.getType());
  int64_t dim = op.getDim();
  int64_t numSlices = srcTy.getShape()[dim] / resultTy.getShape()[dim];

  SmallVector<std::pair<unsigned, unsigned>> srcRegToSliceReg;
  if (failed(getDotOperandSliceRegisterMap(srcTy, resultTy, dim, numSlices,
                                           srcRegToSliceReg)))
    return failure();

  size_t numSliceRegs = srcRegToSliceReg.size() / numSlices;
  resultRegToSrcReg.assign(numSliceRegs, ~0u);
  for (auto [srcReg, sliceAndReg] : llvm::enumerate(srcRegToSliceReg)) {
    if (sliceAndReg.first != static_cast<unsigned>(op.getIndex()))
      continue;
    resultRegToSrcReg[sliceAndReg.second] = srcReg;
  }
  if (llvm::is_contained(resultRegToSrcReg, ~0u))
    return failure();
  return success();
}

} // namespace mlir::triton
