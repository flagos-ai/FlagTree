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

//===----------------------------------------------------------------------===//
//
// This file implements the three passes that fold a K-tile reassembly into
// `ttg.concat_dot_operand` and materialize it again.
//
// A kernel that builds its K tile in fragments, e.g. a dequantizer emitting one
// register fragment per packed word, has to glue them back together before the
// mma. Triton has no concat primitive, so the spelling is a `tl.join` tree, a
// `tl.permute` moving the fragment axes in front of the in-fragment K
// coordinate, and a `tl.reshape` that flattens them:
//
//   q = tl.join(tl.join(b0, b1), tl.join(b2, b3))     # (K_PACK, N, 2, 2)
//   b = tl.reshape(tl.permute(q, (3, 2, 0, 1)),       # (BLOCK_K, BLOCK_N)
//                  (BLOCK_K, BLOCK_N))
//   acc = tl.dot(a, b, acc)
//
// `tritongpu-concat-dot-operand` translates that to
//
//   %b = ttg.concat_dot_operand %b0, %b1, %b2, %b3 {dim = 0 : i32}
//   %acc = tt.dot %a, %b, %acc
//
// Moving the permute behind the K coordinate states the interleaved map,
// `k = kIn * numFrags + i`, which is what plain sub-byte packing produces; the
// `interleaved` attribute selects it. This pass runs before any operand layout
// is assigned, so the encoding is chosen for one wide operand rather than for
// the rank-5 join chain.
//
// The same fact can be stated with no join tree anywhere, as a chain of dots
// over adjacent K segments, or backwards, as one wide operand split into a dot
// per leaf:
//
//   qk = tl.dot(q_nope, k_c)
//   qk = tl.dot(q_pe, k_pe, acc=qk)
//
// `tritongpu-merge-segmented-dot` matches both of those on the final TTGIR,
// merging the chain into one wide dot and rewriting the split leaves into
// `ttg.extract_dot_operand`. It runs after `tritongpu-accelerate-matmul`, since
// deciding whether a chain can merge needs the mma layout.
//
// `tritongpu-expand-concat-dot-operand` runs last, once every layout decision
// is final, and rebuilds the join tree for the operands the register relabel in
// ViewOpToLLVM cannot serve: those staged through shared memory, and every
// interleaved concat.
//
//===----------------------------------------------------------------------===//

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
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#endif
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#define DEBUG_TYPE "tritongpu-concat-dot-operand"
#define LDBG(X) LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << X << "\n")

namespace mlir::triton::gpu {
namespace {

//===--------------------------------------------------------------------===//
// Shared helpers
//===--------------------------------------------------------------------===//

// Walk through the layout-only converts the TTIR to TTGPU conversion leaves
// between every step of the chain.
static Value skipConverts(Value v) {
  while (auto cvt = v.getDefiningOp<ConvertLayoutOp>())
    v = cvt.getSrc();
  return v;
}

// An op only feeds the chain if its single use, after skipping converts, does
// not fan out anywhere else. A producer that also escapes stays live after the
// rewrite, so folding it would duplicate the work rather than remove it.
static bool onlyFeedsChain(Operation *op) {
  while (op->hasOneUse()) {
    Operation *user = *op->getUsers().begin();
    if (!isa<ConvertLayoutOp>(user))
      return true;
    op = user;
  }
  return false;
}

// The concatenated value has to reach a contraction operand: only the A or B
// position of a dot carries the `dot_op` encoding the register relabel needs.
// A `ttg.concat_dot_operand` user also counts, which is how a staged kernel
// qualifies: its inner chain is consumed by the outer concat, and that outer
// concat only exists because it reached a dot.
static bool feedsOnlyDotOperand(Value v) {
  SmallVector<Value> worklist{v};
  DenseSet<Operation *> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    if (cur.use_empty())
      return false;
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (isa<triton::DotOpInterface>(user)) {
        // Reaching the accumulator says nothing about how a K tile is split,
        // and an accumulator never carries a `dot_op` encoding, so a concat
        // built for it could never be relabeled.
        if (use.getOperandNumber() > 1)
          return false;
        continue;
      }
      if (isa<ConcatDotOperandOp>(user))
        continue;
      // Layout-only ops preserve the logical tensor value, so follow them
      // through. WGMMA/SS may already have staged it through local_alloc.
      if (isa<ConvertLayoutOp, LocalAllocOp, LocalLoadOp>(user)) {
        if (seen.insert(user).second)
          worklist.push_back(user->getResult(0));
        continue;
      }
      return false;
    }
  }
  return true;
}

//===--------------------------------------------------------------------===//
// Recognition: join tree -> ttg.concat_dot_operand
//===--------------------------------------------------------------------===//

// Collect the leaves of a perfectly balanced JoinOp tree, in the fragment order
// the joins encode. `tt.join` appends a trailing axis of extent 2 whose index
// selects between its two operands, so walking lhs-before-rhs at every level
// enumerates the leaves in exactly the order the flattened axis will read them.
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

// The permutation that turns the join tree into an ordered reassembly.
//
// The tree appends `levels` trailing axes of extent 2, outermost join last, so
// the fragment index is `sum(bit_i << i)`. Concatenation flattens that to
// `k = index * fragExtent + kIn`, moving every fragment axis in front of the
// in-fragment coordinate, most significant first; interleaving flattens to
// `k = kIn * numFrags + index`, moving them directly behind it. Other axes are
// left alone. Interleaving on a trailing dim is the identity permutation, so
// the matcher must also accept a join tree with no transpose at all.
static SmallVector<int32_t> expectedJoinOrder(int64_t fragRank,
                                              int64_t concatDim, int levels,
                                              bool interleaved) {
  SmallVector<int32_t> expected;
  for (int64_t d = 0; d < concatDim; ++d)
    expected.push_back(d);
  if (interleaved)
    expected.push_back(concatDim);
  for (int i = levels; i >= 1; --i)
    expected.push_back(fragRank + i - 1);
  if (!interleaved)
    expected.push_back(concatDim);
  for (int64_t d = concatDim + 1; d < fragRank; ++d)
    expected.push_back(d);
  return expected;
}

static bool isConcatOrder(ArrayRef<int32_t> order, int64_t fragRank,
                          int64_t concatDim, int levels, bool interleaved) {
  if ((int64_t)order.size() != fragRank + levels)
    return false;
  return ArrayRef<int32_t>(expectedJoinOrder(fragRank, concatDim, levels,
                                             interleaved)) == order;
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
    // leave the transpose as any op implementing it. The transpose is optional;
    // a join tree feeding the reshape directly is the identity-order case, i.e.
    // interleaving on the trailing axis, spelled `reshape(join(lo, hi))`.
    Value reshapeSrc = skipConverts(reshape.getSrc());
    auto trans = reshapeSrc.getDefiningOp<TransposeOpInterface>();
    auto joinRoot =
        trans ? skipConverts(trans.getSrc()).getDefiningOp<triton::JoinOp>()
              : reshapeSrc.getDefiningOp<triton::JoinOp>();
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
    for (Value f : fragments)
      if (f.getType() != fragTy)
        return failure();
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

    // The interleaved form for a trailing-axis concat is the identity
    // permutation, so the matcher must also accept a join tree with no
    // transpose at all. Detect which mode applies.
    SmallVector<int32_t> transOrder;
    if (trans)
      transOrder = llvm::to_vector(trans.getOrder());
    else
      for (int64_t d = 0; d < rootTy.getRank(); ++d)
        transOrder.push_back(d);
    bool interleaved = false;
    if (!isConcatOrder(transOrder, fragRank, concatDim, levels, false)) {
      if (!isConcatOrder(transOrder, fragRank, concatDim, levels, true))
        return failure();
      interleaved = true;
    }

    // Replacing the chain must not leave the joins live and duplicate the work.
    if ((trans && !onlyFeedsChain(trans)) || !onlyFeedsChain(joinRoot))
      return failure();

    // Only rewrite when the result really is a contraction operand; see
    // feedsOnlyDotOperand.
    if (!feedsOnlyDotOperand(reshape.getResult()))
      return failure();

    LDBG("folding an ordered " << (interleaved ? "interleave" : "concat")
                               << " of " << numFrags << " fragments on dim "
                               << concatDim);

    // The op requires its result encoding to match the fragments; the convert
    // bridging to the reshape's encoding is folded away once layout propagation
    // pulls dot_op back through the concat.
    auto resTy = RankedTensorType::get(dstTy.getShape(), dstTy.getElementType(),
                                       fragTy.getEncoding());
    Value concat = ConcatDotOperandOp::create(
        rewriter, reshape.getLoc(), resTy, fragments,
        rewriter.getI32IntegerAttr(concatDim),
        interleaved ? rewriter.getUnitAttr() : UnitAttr{});
    if (resTy != dstTy)
      concat =
          ConvertLayoutOp::create(rewriter, reshape.getLoc(), dstTy, concat);
    rewriter.replaceOp(reshape, concat);
    return success();
  }
};

//===--------------------------------------------------------------------===//
// Split tree -> extracts (inverse view)
//===--------------------------------------------------------------------===//

struct SplitLeaf {
  Value value;
  SmallVector<unsigned> path;
};

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

static int64_t getSplitLeafKIndex(ArrayRef<unsigned> path,
                                  ArrayRef<unsigned> significanceOfLevel) {
  int64_t index = 0;
  for (auto [level, side] : llvm::enumerate(path))
    index |= static_cast<int64_t>(side) << significanceOfLevel[level];
  return index;
}

static bool getSplitLevelSignificance(ArrayRef<int32_t> order, int64_t fragRank,
                                      int64_t concatDim, int levels,
                                      SmallVectorImpl<unsigned> &out) {
  if (static_cast<int64_t>(order.size()) != fragRank + levels)
    return false;
  for (int64_t d = 0; d < fragRank; ++d) {
    int64_t src = order[d];
    int64_t expect = d < concatDim ? d : d + levels;
    if (src != expect)
      return false;
  }
  out.assign(levels, 0);
  SmallVector<bool> seen(levels, false);
  for (int i = 0; i < levels; ++i) {
    int64_t axis = order[fragRank + i] - concatDim;
    if (axis < 0 || axis >= levels || seen[axis])
      return false;
    seen[axis] = true;
    unsigned significance = levels - 1 - axis;
    out[levels - 1 - i] = significance;
  }
  return true;
}

static RankedTensorType getDotOperandType(Value v) {
  RankedTensorType found;
  SmallVector<Value> worklist{v};
  DenseSet<Value> seen;
  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (OpOperand &use : cur.getUses()) {
      Operation *user = use.getOwner();
      if (isa<triton::DotOpInterface>(user)) {
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
      if (!isa<ConvertLayoutOp>(user))
        continue;
      if (seen.insert(user->getResult(0)).second)
        worklist.push_back(user->getResult(0));
    }
  }
  return found;
}

struct MatchSplitTreeExtracts : public OpRewritePattern<triton::ReshapeOp> {
  using OpRewritePattern<triton::ReshapeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ReshapeOp reshape,
                                PatternRewriter &rewriter) const override {
    if (reshape.getAllowReorder())
      return failure();
    auto srcTy = cast<RankedTensorType>(reshape.getSrc().getType());
    auto splitTy = cast<RankedTensorType>(reshape.getType());
    int64_t fragRank = srcTy.getRank();
    int levels = splitTy.getRank() - fragRank;
    if (fragRank < 1 || levels < 1 ||
        srcTy.getElementType() != splitTy.getElementType())
      return failure();
    int64_t concatDim = 0;
    while (concatDim < fragRank &&
           srcTy.getShape()[concatDim] == splitTy.getShape()[concatDim])
      ++concatDim;
    if (concatDim >= fragRank)
      return failure();
    int64_t numFrags = 1LL << levels;
    int64_t fragExtent = splitTy.getShape()[concatDim + levels];
    if (srcTy.getShape()[concatDim] != fragExtent * numFrags)
      return failure();
    for (int i = 0; i < levels; ++i)
      if (splitTy.getShape()[concatDim + i] != 2)
        return failure();
    for (int64_t d = concatDim + 1; d < fragRank; ++d)
      if (srcTy.getShape()[d] != splitTy.getShape()[d + levels])
        return failure();

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
    if (!collectSplitLeaves(trans->getResult(0), levels, path, leaves) ||
        static_cast<int64_t>(leaves.size()) != numFrags)
      return failure();
    SmallVector<int64_t> fragShape(srcTy.getShape());
    fragShape[concatDim] = fragExtent;
    auto leafTy = dyn_cast<RankedTensorType>(leaves.front().value.getType());
    if (!leafTy || leafTy.getShape() != ArrayRef<int64_t>(fragShape))
      return failure();
    for (const SplitLeaf &leaf : leaves) {
      if (leaf.value.getType() != leafTy || !feedsOnlyDotOperand(leaf.value))
        return failure();
    }
    auto dotOpTy = getDotOperandType(leaves.front().value);
    if (!dotOpTy || dotOpTy.getShape() != leafTy.getShape())
      return failure();
    for (const SplitLeaf &leaf : leaves)
      if (getDotOperandType(leaf.value) != dotOpTy)
        return failure();
    auto wideTy = RankedTensorType::get(
        srcTy.getShape(), srcTy.getElementType(), dotOpTy.getEncoding());
    SmallVector<std::pair<unsigned, unsigned>> regMap;
    if (failed(getDotOperandSliceRegisterMap(wideTy, dotOpTy, concatDim,
                                             numFrags, regMap)))
      return failure();
    LDBG("rewriting a split tree of "
         << numFrags << " leaves into extracts on dim " << concatDim);
    rewriter.setInsertionPoint(reshape);
    Value wide = ConvertLayoutOp::create(rewriter, reshape.getLoc(), wideTy,
                                         reshape.getSrc());
    SmallVector<Value> extracts;
    for (const SplitLeaf &leaf : leaves) {
      int64_t index = getSplitLeafKIndex(leaf.path, significance);
      extracts.push_back(ExtractDotOperandOp::create(
          rewriter, leaf.value.getLoc(), dotOpTy, wide, concatDim, index));
    }
    for (auto [leaf, extract] : llvm::zip(leaves, extracts)) {
      Value restored = ConvertLayoutOp::create(rewriter, leaf.value.getLoc(),
                                               leafTy, extract);
      rewriter.replaceAllUsesWith(leaf.value, restored);
    }
    return success();
  }
};

struct FoldConcatOfExtracts : public OpRewritePattern<ConcatDotOperandOp> {
  using OpRewritePattern<ConcatDotOperandOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(ConcatDotOperandOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(ConcatDotOperandOp::canonicalize(op, rewriter)))
      return failure();
    LDBG("folding a complete cover of extracts back to its source");
    return success();
  }
};

// `concat(concat(a, b), concat(c, d))` is `concat(a, b, c, d)`.
//
// A kernel that builds its K tile in stages spells each stage as its own
// join/permute/reshape, which the recognizer folds into nested concats.
// Flattening lets the staged spelling reach the same wide operand as the
// one-shot one. Every fragment must concat over the same axis, and every leaf
// must share the element type and encoding, since that is what the flat op is
// verified against; extents along the axis may differ. A partially staged
// chain is left nested and lowers one level at a time.
struct FlattenNestedConcats : public OpRewritePattern<ConcatDotOperandOp> {
  using OpRewritePattern<ConcatDotOperandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatDotOperandOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getInterleaved())
      return failure();
    int64_t dim = op.getDimAttr().getValue().getSExtValue();
    auto outTy = cast<RankedTensorType>(op.getType());

    SmallVector<Value> leaves;
    RankedTensorType leafTy;
    for (Value fragment : op.getFragments()) {
      auto inner = skipConverts(fragment).getDefiningOp<ConcatDotOperandOp>();
      if (!inner || inner.getDimAttr().getValue().getSExtValue() != dim)
        return failure();
      // Nested interleaving is not flat interleaving: stacking pairs and then
      // stacking the results gives a0 c0 b0 d0, while one four-way interleave
      // gives a0 b0 c0 d0. Leave both levels alone.
      if (inner.getInterleaved())
        return failure();
      // An inner concat that also escapes stays live, so flattening would
      // recompute rather than replace it.
      if (!onlyFeedsChain(inner))
        return failure();
      for (Value leaf : inner.getFragments()) {
        auto leafFragTy = cast<RankedTensorType>(leaf.getType());
        if (!leafTy)
          leafTy = leafFragTy;
        else if (leafFragTy.getElementType() != leafTy.getElementType() ||
                 leafFragTy.getEncoding() != leafTy.getEncoding() ||
                 leafFragTy.getRank() != leafTy.getRank())
          return failure();
        leaves.push_back(leaf);
      }
    }
    if (!leafTy || leaves.size() < 2)
      return failure();

    // A non-power-of-two count has no join-tree inverse, so flattening into
    // one would trade a nested value that always expands for a flat one that
    // could not be expanded at all.
    if (!llvm::isPowerOf2_64(leaves.size()))
      return failure();

    LDBG("flattening " << op.getFragments().size() << " staged concats into "
                       << leaves.size() << " fragments on dim " << dim);

    auto flatTy = RankedTensorType::get(
        outTy.getShape(), outTy.getElementType(), leafTy.getEncoding());
    Value flat = ConcatDotOperandOp::create(rewriter, op.getLoc(), flatTy,
                                            leaves, op.getDimAttr());
    if (flatTy != outTy)
      flat = ConvertLayoutOp::create(rewriter, op.getLoc(), outTy, flat);
    rewriter.replaceOp(op, flat);
    return success();
  }
};

//===--------------------------------------------------------------------===//
// Materialization: ttg.concat_dot_operand -> join tree
//===--------------------------------------------------------------------===//

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

// The type a balanced join tree over `fragTy` ends up with, or null when the
// dialect cannot infer an encoding for some level.
//
// This mirrors `JoinOp::build`, which reports inference failure with
// `llvm_unreachable`. Asking the same question up front keeps a layout the
// dialect cannot join from aborting the compiler, and decides the expansion
// from types alone, so nothing is built until it is known to be buildable.
static RankedTensorType inferJoinTreeType(RankedTensorType fragTy, int levels,
                                          Location loc) {
  RankedTensorType cur = fragTy;
  for (int i = 0; i < levels; ++i) {
    SmallVector<int64_t> shape(cur.getShape());
    shape.push_back(2);
    Attribute enc;
    if (Attribute srcEnc = cur.getEncoding()) {
      if (failed(cast<triton::DialectInferLayoutInterface>(&srcEnc.getDialect())
                     ->inferDefaultJoinOpEncoding(srcEnc, enc, cur.getShape(),
                                                  loc)))
        return {};
    }
    cur = RankedTensorType::get(shape, cur.getElementType(), enc);
  }
  return cur;
}

// Rebuild the join tree for a concat whose operand did not end up in a form the
// register relabel can serve.
//
// This is the expected path for the shared-memory forms, not a failure path:
// the tree is rebuilt after every layout decision is final, so it carries the
// layout chosen for one wide operand rather than for a rank-5 join chain.
// Landing the fragments directly in a shared subslice instead measures slower
// despite issuing less work, because the register form is what lets the
// pipeliner give each fragment its own double-buffered `cp.async`.
struct ExpandConcatToJoinTree : public OpRewritePattern<ConcatDotOperandOp> {
  using OpRewritePattern<ConcatDotOperandOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcatDotOperandOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<std::pair<unsigned, unsigned>> unused;
    if (succeeded(getConcatDotOperandRegisterMap(op, unused)))
      return failure();

    Location loc = op.getLoc();
    auto fragTy = cast<RankedTensorType>(op.getFragments()[0].getType());
    auto dstTy = cast<RankedTensorType>(op.getType());
    int64_t dim = op.getDimAttr().getValue().getSExtValue();
    int64_t rank = fragTy.getRank();

    // The exact inverse is a join tree, whose operands must have equal shapes
    // and whose count must be a power of two. Heterogeneous or non-power-of-two
    // segments take the register relabel path instead; the pass reports any
    // that failed both once the fixpoint settles.
    size_t numFrags = op.getFragments().size();
    if (numFrags < 2 || !llvm::isPowerOf2_64(numFrags))
      return failure();
    for (Value fragment : op.getFragments())
      if (cast<RankedTensorType>(fragment.getType()).getShape() !=
          fragTy.getShape())
        return failure();
    int levels = llvm::Log2_64(numFrags);

    // Decide the whole expansion from types first. Nothing is built until every
    // encoding on the way to the flat result is known to be inferable, so a
    // concat that cannot be expanded is simply left alone.
    RankedTensorType rootTy = inferJoinTreeType(fragTy, levels, loc);
    if (!rootTy)
      return failure();

    // The inverse of whichever order the recognizer matched.
    bool interleaved = op.getInterleaved();
    SmallVector<int32_t> order =
        expectedJoinOrder(rank, dim, levels, interleaved);
    bool identityOrder = true;
    for (auto [i, src] : llvm::enumerate(order))
      identityOrder &= src == static_cast<int32_t>(i);

    RankedTensorType transTy = rootTy;
    if (!identityOrder) {
      Attribute transEnc;
      if (Attribute rootEnc = rootTy.getEncoding()) {
        if (failed(
                cast<triton::DialectInferLayoutInterface>(&rootEnc.getDialect())
                    ->inferTransOpEncoding(rootEnc, rootTy.getShape(), order,
                                           transEnc, loc)))
          return failure();
      }
      SmallVector<int64_t> transShape;
      for (int32_t src : order)
        transShape.push_back(rootTy.getShape()[src]);
      transTy =
          RankedTensorType::get(transShape, rootTy.getElementType(), transEnc);
    }

    // The chain carries whatever layout the joins infer, not the operand
    // encoding the concat had, so the reshape lands in the inferred one and a
    // convert restores the concat's type.
    Attribute reshapeEnc = inferReshapeEncoding(transTy, dstTy.getShape(), loc);
    if (!reshapeEnc)
      return failure();

    LDBG("expanding a " << (interleaved ? "interleave" : "concat") << " of "
                        << numFrags
                        << " fragments back into a join tree: the operand "
                           "layout is not a register relabel");

    SmallVector<Value> level(op.getFragments().begin(),
                             op.getFragments().end());
    while (level.size() > 1) {
      SmallVector<Value> next;
      for (size_t i = 0; i < level.size(); i += 2)
        next.push_back(
            triton::JoinOp::create(rewriter, loc, level[i], level[i + 1]));
      level = std::move(next);
    }
    Value trans = level.front();
    if (!identityOrder)
      trans = triton::TransOp::create(rewriter, loc, trans, order);

    auto flatTy = RankedTensorType::get(dstTy.getShape(),
                                        dstTy.getElementType(), reshapeEnc);
    Value flat = triton::ReshapeOp::create(rewriter, loc, flatTy, trans,
                                           /*allowReorder=*/false,
                                           /*efficientLayout=*/false);
    if (flatTy != dstTy)
      flat = ConvertLayoutOp::create(rewriter, loc, dstTy, flat);
    rewriter.replaceOp(op, flat);
    return success();
  }
};

//===--------------------------------------------------------------------===//
// Generic segmented-dot merging
//===--------------------------------------------------------------------===//

namespace ttng = mlir::triton::nvidia_gpu;

// Segmented kernels do not necessarily spell a join tree. A common lowering
// instead emits a chain of dots whose accumulator is the previous dot and whose
// operands are adjacent K slices. The helpers below inspect the final TTGPU IR
// rather than the source spelling.

static bool hasNoRewriteSensitiveAttrs(Operation *op) {
  return op->getDiscardableAttrs().empty();
}

static bool hasOnlyPureOpsBetween(Operation *from, Operation *to) {
  if (from->getBlock() != to->getBlock())
    return false;
  for (Operation *op = from->getNextNode(); op && op != to;
       op = op->getNextNode()) {
    // Async WGMMA ordering operations are semantically significant even
    // though they do not model ordinary memory effects. Do not move a merged
    // dot across a wait/commit boundary.
    if (isa<ttng::WarpGroupDotWaitOp, ttng::WarpGroupDotCommitOp>(op))
      return false;
#ifdef __TLE__
    // A shared-operand fence is an ordering boundary too: it establishes
    // visibility from generic-proxy writes to the async WGMMA proxy.
    if (isa<mlir::triton::tle::WGMMASharedOperandFenceOp>(op))
      return false;
#endif
    // Backend staging may materialize the next register fragment between two
    // dots. The rewrite leaves those ops in place and only removes the
    // accumulator dots, so they are not a reorder-sensitive boundary.
    if (auto localLoad = dyn_cast<LocalLoadOp>(op)) {
      // Token-bearing local loads are asynchronous and need their own
      // dependency chain; never move a merged contraction across one.
      if (localLoad.getToken())
        return false;
      continue;
    }
    if (isa<LocalAllocOp>(op))
      continue;
    if (!isMemoryEffectFree(op))
      return false;
  }
  return true;
}

static std::optional<int64_t> getNativeMmaK(Type elementTy,
                                            InputPrecision precision,
                                            NvidiaMmaEncodingAttr mma) {
  if (mma.isTuring())
    return elementTy.isF16()        ? std::optional<int64_t>(8)
           : elementTy.isInteger(8) ? std::optional<int64_t>(16)
                                    : std::nullopt;
  if (elementTy.isF16() || elementTy.isBF16())
    return 16;
  if (elementTy.isF32() && precision == InputPrecision::TF32)
    return 8;
  if (elementTy.isInteger(8))
    return 32;
  if (llvm::isa<Float8E5M2Type, Float8E4M3FNType>(elementTy))
    return 32;
  return std::nullopt;
}

static std::optional<int64_t> getNativeMmaK(triton::DotOp dot,
                                            RankedTensorType aTy) {
  auto enc = dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding());
  auto mma = enc ? dyn_cast<NvidiaMmaEncodingAttr>(enc.getParent())
                 : NvidiaMmaEncodingAttr();
  if (!mma || mma.getVersionMajor() != 2)
    return std::nullopt;
  if (aTy.getElementType().isF64()) {
    ModuleOp module = dot->getParentOfType<ModuleOp>();
    if (!module)
      return std::nullopt;
    int capability = getNVIDIAComputeCapability(module);
    return capability == 90   ? std::optional<int64_t>(16)
           : capability >= 80 ? std::optional<int64_t>(4)
                              : std::nullopt;
  }
  return getNativeMmaK(aTy.getElementType(), dot.getInputPrecision(), mma);
}

static bool hasSupportedMmaV2Numerics(triton::DotOp dot) {
  Type a = dot.getA().getType().getElementType();
  Type b = dot.getB().getType().getElementType();
  Type d = dot.getD().getType().getElementType();
  if (a != b)
    return false;
  if (a.isF16())
    return d.isF16() || d.isF32();
  if (a.isBF16())
    return d.isF32();
  if (a.isF32())
    return d.isF32() && dot.getInputPrecision() == InputPrecision::TF32;
  if (a.isF64())
    return d.isF64();
  if (a.isInteger(8))
    return d.isInteger(32);
  bool fp8 = llvm::isa<Float8E5M2Type, Float8E4M3FNType>(a);
  if (fp8)
    return d.isF16() || d.isF32();
  return false;
}

// Independently produced fragments are safe to coalesce when each one is
// single-use along the dot chain, which keeps the rewrite from duplicating a
// producer. Values with fan-out are left segmented: their lifetime and
// pressure trade-off is not recoverable from local IR alone.
static bool hasSingleUseDotFragments(ArrayRef<Value> fragments) {
  for (Value fragment : fragments) {
    Value v = fragment;
    while (auto cvt = v.getDefiningOp<ConvertLayoutOp>()) {
      if (!cvt.getResult().hasOneUse())
        return false;
      v = cvt.getSrc();
    }
    if (!fragment.hasOneUse())
      return false;
  }
  return true;
}

// `isBeforeInBlock` asserts when the two operations do not share a block, so
// every ordering question carries the block check with it. A producer in
// another block, e.g. a convert licm hoisted out of the loop the dots live in,
// is a normal shape rather than a malformed one.
static bool isBefore(Operation *op, Operation *anchor) {
  return anchor && op->getBlock() == anchor->getBlock() &&
         op->isBeforeInBlock(anchor);
}

// A merged dot is emitted at the chain's replacement point, so fragments may
// be materialized between the original dots only when every producer dominates
// the old tail. This keeps a dot from moving ahead of the load or compute that
// produces a later fragment.
static bool fragmentsAvailableBefore(Operation *firstDot,
                                     ArrayRef<Value> fragments,
                                     Operation *lastUse = nullptr) {
  for (Value fragment : fragments) {
    Value value = fragment;
    while (auto cvt = value.getDefiningOp<ConvertLayoutOp>()) {
      if (!isBefore(cvt, firstDot) && !isBefore(cvt, lastUse))
        return false;
      value = cvt.getSrc();
    }
    if (auto def = value.getDefiningOp()) {
      // A local_load materializes a fragment from shared memory. Its register
      // pressure is governed by the shared tile rather than by extending a
      // scalar producer's live range, so allow it.
      if (auto localLoad = dyn_cast<LocalLoadOp>(def)) {
        // Token-bearing local loads are asynchronous and need their own
        // dependency chain; never move a merged contraction across one.
        if (localLoad.getToken())
          return false;
        // A backend-staged fragment may be loaded between segmented dots. It
        // is safe to delay the merged dot until the old tail (the caller's
        // replacement point), but never past that point.
        if (isBefore(def, firstDot) || isBefore(def, lastUse))
          continue;
      }
      if (!isBefore(def, firstDot))
        return false;
    }
  }
  return true;
}

static bool sameFragmentLayoutExceptK(Type lhs, Type rhs, int64_t kDim) {
  auto a = dyn_cast<RankedTensorType>(lhs);
  auto b = dyn_cast<RankedTensorType>(rhs);
  if (!a || !b || a.getRank() != b.getRank() ||
      a.getElementType() != b.getElementType() ||
      a.getEncoding() != b.getEncoding())
    return false;
  for (int64_t i = 0; i < a.getRank(); ++i)
    if (i != kDim && a.getShape()[i] != b.getShape()[i])
      return false;
  return true;
}

struct SegmentedMmaChain {
  SmallVector<triton::DotOp> dots;
  SmallVector<Value> aFragments, bFragments;
  SmallVector<Operation *> deadInterior;
};

static triton::DotOp
skipAccumulatorConverts(Value acc, SmallVectorImpl<Operation *> &interior) {
  while (auto cvt = acc.getDefiningOp<ConvertLayoutOp>()) {
    if (!cvt.getResult().hasOneUse())
      return {};
    interior.push_back(cvt);
    acc = cvt.getSrc();
  }
  return acc.getDefiningOp<triton::DotOp>();
}

static LogicalResult matchSegmentedMmaChain(triton::DotOp tail,
                                            SegmentedMmaChain &chain) {
  if (!hasSupportedMmaV2Numerics(tail) || !hasNoRewriteSensitiveAttrs(tail))
    return failure();
  SmallVector<triton::DotOp> reversed{tail};
  SmallVector<Operation *> interior;
  size_t accepted = 0;
  triton::DotOp current = tail;
  while (auto previous = skipAccumulatorConverts(current.getC(), interior)) {
    if (!previous.getD().hasOneUse() || !hasNoRewriteSensitiveAttrs(previous) ||
        previous->getBlock() != current->getBlock() ||
        !hasOnlyPureOpsBetween(previous, current) ||
        previous.getInputPrecision() != current.getInputPrecision() ||
        previous.getMaxNumImpreciseAcc() != current.getMaxNumImpreciseAcc())
      break;
    reversed.push_back(previous);
    interior.push_back(previous);
    accepted = interior.size();
    current = previous;
  }
  if (reversed.size() < 2)
    return failure();
  interior.resize(accepted);
  for (auto dot : llvm::reverse(reversed)) {
    chain.dots.push_back(dot);
    chain.aFragments.push_back(dot.getA());
    chain.bFragments.push_back(dot.getB());
  }
  auto aTy = dyn_cast<RankedTensorType>(chain.aFragments.front().getType());
  auto bTy = dyn_cast<RankedTensorType>(chain.bFragments.front().getType());
  auto aEnc = aTy ? dyn_cast<DotOperandEncodingAttr>(aTy.getEncoding())
                  : DotOperandEncodingAttr();
  auto bEnc = bTy ? dyn_cast<DotOperandEncodingAttr>(bTy.getEncoding())
                  : DotOperandEncodingAttr();
  auto aParent = aEnc ? dyn_cast<NvidiaMmaEncodingAttr>(aEnc.getParent())
                      : NvidiaMmaEncodingAttr();
  auto bParent = bEnc ? dyn_cast<NvidiaMmaEncodingAttr>(bEnc.getParent())
                      : NvidiaMmaEncodingAttr();
  if (!aTy || !bTy || !aEnc || !bEnc || !aParent || !bParent ||
      aParent.getVersionMajor() != 2 || bParent.getVersionMajor() != 2 ||
      aEnc.getOpIdx() != 0 || bEnc.getOpIdx() != 1)
    return failure();
  int64_t aDim = aTy.getRank() - 1, bDim = bTy.getRank() - 2;
  if (aDim < 0 || bDim < 0)
    return failure();
  // A complete register relabel is required; otherwise widening would change
  // live ranges or lane ownership. This naturally rejects unsupported
  // non-power-of-two register shapes while allowing shared paths below.
  if (!hasSingleUseDotFragments(chain.aFragments) ||
      !hasSingleUseDotFragments(chain.bFragments))
    return failure();
  if (!fragmentsAvailableBefore(chain.dots.front(), chain.aFragments,
                                chain.dots.back()) ||
      !fragmentsAvailableBefore(chain.dots.front(), chain.bFragments,
                                chain.dots.back()))
    return failure();
  for (auto dot : chain.dots)
    if (!sameFragmentLayoutExceptK(dot.getA().getType(), aTy, aDim) ||
        !sameFragmentLayoutExceptK(dot.getB().getType(), bTy, bDim))
      return failure();
  auto nativeK = getNativeMmaK(chain.dots.front(), aTy);
  if (!nativeK)
    return failure();
  for (auto dot : chain.dots) {
    auto dotA = cast<RankedTensorType>(dot.getA().getType());
    auto dotB = cast<RankedTensorType>(dot.getB().getType());
    if (dotA.getShape()[aDim] % *nativeK != 0 ||
        dotB.getShape()[bDim] % *nativeK != 0)
      return failure();
  }
  SmallVector<int64_t> wideAShape(aTy.getShape()), wideBShape(bTy.getShape());
  wideAShape[aDim] = 0;
  wideBShape[bDim] = 0;
  for (auto dot : chain.dots) {
    wideAShape[aDim] +=
        cast<RankedTensorType>(dot.getA().getType()).getShape()[aDim];
    wideBShape[bDim] +=
        cast<RankedTensorType>(dot.getB().getType()).getShape()[bDim];
  }
  auto wideATy = RankedTensorType::get(wideAShape, aTy.getElementType(),
                                       aTy.getEncoding());
  auto wideBTy = RankedTensorType::get(wideBShape, bTy.getElementType(),
                                       bTy.getEncoding());
  SmallVector<RankedTensorType> aTypes, bTypes;
  for (auto dot : chain.dots) {
    aTypes.push_back(cast<RankedTensorType>(dot.getA().getType()));
    bTypes.push_back(cast<RankedTensorType>(dot.getB().getType()));
  }
  SmallVector<std::pair<unsigned, unsigned>> map;
  if (failed(getDotOperandConcatRegisterMap(wideATy, aTypes, aDim, map)) ||
      failed(getDotOperandConcatRegisterMap(wideBTy, bTypes, bDim, map)))
    return failure();
  chain.deadInterior = std::move(interior);
  return success();
}

struct MergeSegmentedMma : public OpRewritePattern<triton::DotOp> {
  using OpRewritePattern<triton::DotOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(triton::DotOp tail,
                                PatternRewriter &rewriter) const override {
    SegmentedMmaChain chain;
    if (failed(matchSegmentedMmaChain(tail, chain)))
      return failure();
    auto aTy = cast<RankedTensorType>(chain.aFragments.front().getType());
    auto bTy = cast<RankedTensorType>(chain.bFragments.front().getType());
    int64_t aDim = aTy.getRank() - 1, bDim = bTy.getRank() - 2;
    SmallVector<int64_t> aShape(aTy.getShape()), bShape(bTy.getShape());
    aShape[aDim] = bShape[bDim] = 0;
    for (auto dot : chain.dots) {
      auto dotA = cast<RankedTensorType>(dot.getA().getType());
      auto dotB = cast<RankedTensorType>(dot.getB().getType());
      aShape[aDim] += dotA.getShape()[aDim];
      bShape[bDim] += dotB.getShape()[bDim];
    }
    rewriter.setInsertionPoint(tail);
    auto wideA =
        RankedTensorType::get(aShape, aTy.getElementType(), aTy.getEncoding());
    auto wideB =
        RankedTensorType::get(bShape, bTy.getElementType(), bTy.getEncoding());
    Value a = ConcatDotOperandOp::create(rewriter, tail.getLoc(), wideA,
                                         chain.aFragments,
                                         rewriter.getI32IntegerAttr(aDim));
    Value b = ConcatDotOperandOp::create(rewriter, tail.getLoc(), wideB,
                                         chain.bFragments,
                                         rewriter.getI32IntegerAttr(bDim));
    LDBG("merging " << chain.dots.size() << " segmented dots into one");
    Value merged = triton::DotOp::create(
        rewriter, tail.getLoc(), tail.getD().getType(), a, b,
        chain.dots.front().getC(), tail.getInputPrecision(),
        tail.getMaxNumImpreciseAcc());
    rewriter.replaceOp(tail, merged);
    // Erase from the end of the SSA chain towards its head. The matcher may
    // record layout converts before their defining dot, so a forward walk would
    // try to erase a still-used convert. Repeatedly picking a now-dead op keeps
    // the order correct with several converts or other pure intermediates.
    SmallVector<Operation *> dead(chain.deadInterior);
    while (!dead.empty()) {
      auto it = llvm::find_if(dead, [](Operation *op) {
        return llvm::all_of(op->getResults(),
                            [](Value result) { return result.use_empty(); });
      });
      if (it == dead.end())
        break;
      rewriter.eraseOp(*it);
      dead.erase(it);
    }
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
    // One fixpoint for both: folding an outer stage turns the stage below it
    // into a concat operand, which then folds in turn.
    patterns.add<MatchJoinTreeConcat, FlattenNestedConcats>(&getContext());
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
    patterns.add<ExpandConcatToJoinTree>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    // Last pass that can act on the op, so anything left has to lower as a
    // register relabel. Reporting here, while the fragment shapes are still
    // available, says why; ViewOpToLLVM would fail on the same condition with
    // less to go on. Reaching this means a later pass retagged an operand the
    // matcher had proven, e.g. by widening kWidth past a fragment's K extent.
    getOperation().walk(
        [&](ConcatDotOperandOp op) {
          SmallVector<std::pair<unsigned, unsigned>> unused;
          if (succeeded(getConcatDotOperandRegisterMap(op, unused)))
            return;
          InFlightDiagnostic diag =
              op.emitError()
              << "concat_dot_operand: the operand layout allows neither a "
                 "per-thread register relabel nor a join-tree expansion";
          diag.attachNote() << "expansion needs a power-of-two count of "
                               "equally shaped fragments; this op concatenates "
                            << op.getFragments().size() << " fragments on dim "
                            << op.getDim();
          signalPassFailure();
        });
  }
};

class TritonGPUMergeSegmentedDotPass
    : public impl::TritonGPUMergeSegmentedDotBase<
          TritonGPUMergeSegmentedDotPass> {
public:
  void runOnOperation() override {
    // Two stages: both matchers can serve a split tree whose leaves feed a dot
    // chain, but they key on different ops, the reshape at the tree's head
    // against the dot at its tail, so the greedy driver's order would decide.
    // Run the split rewrite first; otherwise the leaves stop being dot operands
    // and the tree survives with a concat stacked back on top of it.
    RewritePatternSet splitPatterns(&getContext());
    splitPatterns.add<MatchSplitTreeExtracts>(&getContext());
    if (failed(
            applyPatternsGreedily(getOperation(), std::move(splitPatterns)))) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<MergeSegmentedMma, FoldConcatOfExtracts>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::triton::gpu

namespace mlir::triton {
namespace ttg = mlir::triton::gpu;

// Defined here rather than in Transforms/Utility.cpp: backends that substitute
// their own Utility.cpp would otherwise have to carry this too.
LogicalResult getDotOperandConcatRegisterMap(
    RankedTensorType dstTy, ArrayRef<RankedTensorType> fragmentTypes,
    int64_t dim,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &resultRegToFragmentReg,
    bool interleaved) {
  if (fragmentTypes.size() < 2 || dim < 0 || dim >= dstTy.getRank())
    return failure();
  // Interleaving is only defined when every fragment contributes the same
  // number of elements; the verifier enforces it, this keeps the proof honest
  // when the map is run on types alone before any op exists.
  if (interleaved && llvm::any_of(fragmentTypes, [&](RankedTensorType ty) {
        return ty.getShape()[dim] != fragmentTypes.front().getShape()[dim];
      }))
    return failure();
  int64_t rank = dstTy.getRank();
  Type elementType = fragmentTypes.front().getElementType();
  Attribute encoding = fragmentTypes.front().getEncoding();
  if (!encoding || !isa<ttg::DotOperandEncodingAttr>(encoding))
    return failure();
  SmallVector<int64_t> expectedShape(dstTy.getShape());
  expectedShape[dim] = 0;
  for (RankedTensorType fragmentTy : fragmentTypes) {
    if (fragmentTy.getRank() != rank ||
        fragmentTy.getElementType() != elementType ||
        fragmentTy.getEncoding() != encoding)
      return failure();
    for (int64_t i = 0; i < rank; ++i) {
      if (i == dim)
        continue;
      if (fragmentTy.getShape()[i] != dstTy.getShape()[i])
        return failure();
    }
    expectedShape[dim] += fragmentTy.getShape()[dim];
  }
  if (expectedShape != SmallVector<int64_t>(dstTy.getShape()))
    return failure();
  auto dotEnc = cast<ttg::DotOperandEncodingAttr>(encoding);
  if (dim != (dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2))
    return failure();
  unsigned kWidth = dotEnc.getKWidth();
  if (kWidth != 0 && llvm::any_of(fragmentTypes, [&](RankedTensorType ty) {
        return ty.getShape()[dim] < 8 * static_cast<int64_t>(kWidth);
      }))
    return failure();
  auto representable = [](RankedTensorType ty) {
    return llvm::all_of(ty.getShape(), [](int64_t extent) {
      return extent > 0 && llvm::isPowerOf2_64(extent);
    });
  };
  if (!representable(dstTy) ||
      llvm::any_of(fragmentTypes, [&](auto ty) { return !representable(ty); }))
    return failure();

  MLIRContext *ctx = dstTy.getContext();
  auto inDim = [&](StringRef name) { return StringAttr::get(ctx, name); };
  LinearLayout dstLL = ttg::toLinearLayout(dstTy);
  SmallVector<LinearLayout> fragmentLLs;
  fragmentLLs.reserve(fragmentTypes.size());
  for (RankedTensorType ty : fragmentTypes)
    fragmentLLs.push_back(ttg::toLinearLayout(ty));
  for (StringRef name : {"lane", "warp", "block"}) {
    StringAttr attr = inDim(name);
    for (const LinearLayout &ll : fragmentLLs) {
      if (dstLL.hasInDim(attr) != ll.hasInDim(attr))
        return failure();
      if (dstLL.hasInDim(attr) &&
          dstLL.getBases().lookup(attr) != ll.getBases().lookup(attr))
        return failure();
    }
  }
  StringAttr kReg = inDim("register");
  auto offsetsOf = [&](const LinearLayout &ll) {
    SmallVector<SmallVector<unsigned>> offsets;
    for (int reg = 0; reg < ll.getInDimSize(kReg); ++reg) {
      auto idxs = ll.apply({{kReg, reg},
                            {inDim("lane"), 0},
                            {inDim("warp"), 0},
                            {inDim("block"), 0}});
      if (static_cast<int64_t>(idxs.size()) != rank)
        return SmallVector<SmallVector<unsigned>>();
      offsets.push_back(
          llvm::to_vector_of<unsigned>(llvm::make_second_range(idxs)));
    }
    return offsets;
  };
  auto coordKey = [&](ArrayRef<unsigned> coord, RankedTensorType ty) {
    int64_t key = 0;
    for (int64_t i = 0; i < rank; ++i)
      key = key * ty.getShape()[i] + coord[i];
    return key;
  };
  SmallVector<DenseMap<int64_t, unsigned>> fragmentMaps;
  fragmentMaps.reserve(fragmentTypes.size());
  for (size_t i = 0; i < fragmentTypes.size(); ++i) {
    auto offsets = offsetsOf(fragmentLLs[i]);
    if (offsets.empty() && fragmentLLs[i].getInDimSize(kReg) != 0)
      return failure();
    DenseMap<int64_t, unsigned> map;
    for (auto [reg, coord] : llvm::enumerate(offsets))
      if (!map.try_emplace(coordKey(coord, fragmentTypes[i]), reg).second)
        return failure();
    fragmentMaps.push_back(std::move(map));
  }
  auto dstOffsets = offsetsOf(dstLL);
  if (dstOffsets.empty() && dstLL.getInDimSize(kReg) != 0)
    return failure();
  // Where along `dim` each fragment starts, for the concatenating form. The
  // interleaved form needs no boundaries: every fragment is present at every
  // stride, so the coordinate divides instead.
  //
  // The interleaved branch never resolves on any layout in the dialect today,
  // since a `dot_op` thread owns its K elements contiguously. Do not drop it:
  // running the concatenating arithmetic over an interleaved op finds a
  // co-located register for every coordinate and reports a relabel that reads
  // the wrong elements.
  SmallVector<int64_t> boundaries;
  boundaries.push_back(0);
  for (RankedTensorType ty : fragmentTypes)
    boundaries.push_back(boundaries.back() + ty.getShape()[dim]);
  int64_t numFragments = fragmentTypes.size();
  resultRegToFragmentReg.clear();
  resultRegToFragmentReg.reserve(dstOffsets.size());
  for (auto coord : dstOffsets) {
    size_t fragment;
    if (interleaved) {
      fragment = coord[dim] % numFragments;
      coord[dim] /= numFragments;
    } else {
      auto it = llvm::upper_bound(boundaries, static_cast<int64_t>(coord[dim]));
      fragment = std::max<size_t>(1, it - boundaries.begin()) - 1;
      if (fragment >= fragmentTypes.size())
        return failure();
      coord[dim] -= boundaries[fragment];
    }
    auto found =
        fragmentMaps[fragment].find(coordKey(coord, fragmentTypes[fragment]));
    if (found == fragmentMaps[fragment].end())
      return failure();
    resultRegToFragmentReg.emplace_back(fragment, found->second);
  }
  return success();
}

LogicalResult getDotOperandSliceRegisterMap(
    RankedTensorType dstTy, RankedTensorType fragTy, int64_t dim,
    int64_t numFrags,
    SmallVectorImpl<std::pair<unsigned, unsigned>> &resultRegToFragmentReg) {
  int64_t rank = fragTy.getRank();

  if (numFrags < 2 || dstTy.getRank() != rank || dim < 0 || dim >= rank)
    return failure();
  for (int64_t i = 0; i < rank; ++i) {
    int64_t expected = fragTy.getShape()[i] * (i == dim ? numFrags : 1);
    if (dstTy.getShape()[i] != expected)
      return failure();
  }

  // Layouts other than dot_op place lanes differently as `dim` grows, so the
  // relabel would move the wrong registers. This is also what sends the
  // shared-memory forms to the join-tree expansion: an operand staged through
  // `ttg.local_alloc` never carries a dot_op encoding.
  auto dotEnc = dyn_cast<ttg::DotOperandEncodingAttr>(fragTy.getEncoding());
  if (!dotEnc)
    return failure();

  // Only the contraction axis can be extended in place: growing M or N would
  // change which lane owns an element.
  if (dim != (dotEnc.getOpIdx() == 0 ? rank - 1 : rank - 2))
    return failure();

  // Below 8 * kWidth the dot_op layout replicates elements across lanes
  // instead of scaling with K, so the fragment is not a K-slice of the wider
  // one and the relabel would read a duplicate. kWidth is 0 when unused.
  unsigned kWidth = dotEnc.getKWidth();
  if (kWidth != 0 && fragTy.getShape()[dim] < 8 * (int64_t)kWidth)
    return failure();

  // Distributed layouts are defined over the power-of-two linear-layout domain,
  // and toLinearLayout asserts rather than fails outside it, so check the
  // shapes first.
  auto isRepresentable = [](RankedTensorType ty) {
    return llvm::all_of(ty.getShape(), [](int64_t extent) {
      return extent > 0 && llvm::isPowerOf2_64(extent);
    });
  };
  if (!isRepresentable(dstTy) || !isRepresentable(fragTy))
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

  // Flatten a coordinate to index a fragment register. Every non-contraction
  // coordinate is shared with the result; the contraction coordinate is
  // translated to the fragment-local range by the caller below.
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
  SmallVector<RankedTensorType> fragmentTypes;
  fragmentTypes.reserve(op.getFragments().size());
  for (Value fragment : op.getFragments())
    fragmentTypes.push_back(cast<RankedTensorType>(fragment.getType()));
  return getDotOperandConcatRegisterMap(
      cast<RankedTensorType>(op.getType()), fragmentTypes, op.getDim(),
      resultRegToFragmentReg, op.getInterleaved());
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
