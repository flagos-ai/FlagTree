//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tritonxpu-legalize"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULEGALIZE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// get greatest common divisor
size_t gcd(size_t a, size_t b) {
  if (b == 0)
    return a;
  else
    return gcd(b, a % b);
}

size_t getLCM(llvm::SmallVector<size_t> &datas) {
  for (int i = 1; i < datas.size(); ++i) {
    datas[i] = datas[i - 1] / gcd(datas[i - 1], datas[i]) * datas[i];
  }
  return datas.back();
}

// A DenseElementsAttr of `targetTy` holding the leading elements of `attr`,
// or nullptr when `attr` is too short to cover it.
//
// Why: getFromRawBuffer() wants a buffer sized exactly for the target type.
// Handing it the original, longer buffer only trips an assert, which a release
// build drops, and the attribute then carries the wrong length -- the values
// read back shifted. Shrinking a constant is intended here (the dropped
// elements are the ones the other slices use), so copy the prefix
// element-wise, which also covers a bit-packed element type such as i1.
static DenseElementsAttr reshapeDenseAttrTo(DenseElementsAttr attr,
                                            ShapedType targetTy) {
  if (attr.isSplat())
    return DenseElementsAttr::get(targetTy, attr.getSplatValue<Attribute>());
  int64_t wanted = targetTy.getNumElements();
  if (wanted > attr.getNumElements())
    return nullptr;
  auto values = attr.getValues<Attribute>();
  llvm::SmallVector<Attribute> prefix(values.begin(), values.begin() + wanted);
  return DenseElementsAttr::get(targetTy, prefix);
}

// Make a triton_xpu.convert_layout result carry its source shape.
//
// Why: convert_layout only changes the encoding, so once a region-body slice
// narrows its source the result shape has to follow, or the op fails its own
// verifier ("convert_layout op requires the same shape for all operands and
// results"). sparse_broadcast_to's `combo` select is the case this hits.
static void alignConvertLayoutShape(triton::xpu::ConvertLayoutOp cvtOp) {
  auto srcType = mlir::dyn_cast<RankedTensorType>(cvtOp.getSrc().getType());
  auto resType = mlir::dyn_cast<RankedTensorType>(cvtOp.getType());
  if (!srcType || !resType || srcType.getShape() == resType.getShape())
    return;
  cvtOp.getResult().setType(RankedTensorType::get(
      srcType.getShape(), resType.getElementType(), resType.getEncoding()));
}

Type getSlicedType(const Type &type, llvm::SmallVector<size_t> iterCount,
                   bool isInner, bool needNewEncoding = false,
                   std::optional<size_t> rank1Count = std::nullopt) {
  // Slice tensor according iteration count
  if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type)) {
    auto tensorShape = tensorType.getShape();
    llvm::SmallVector<int64_t> slicedShape(tensorShape.size(), 1u);
    if (tensorShape.size() == iterCount.size()) {
      for (int i = 0; i < tensorShape.size(); ++i) {
        slicedShape[i] = std::max(tensorShape[i] / iterCount[i], size_t(1));
      }
    } else if (tensorShape.size() < iterCount.size()) {
      // A rank-1 value facing an N-D schedule: the inner chain takes the last
      // dimension (the one a reduce keeps) and the outer chain the first,
      // which generalises the 2-D case this used to hard-code as
      // iterCount[1] / iterCount[0]. A rank-0 tensor has nothing to slice.
      if (!tensorShape.empty()) {
        // The caller may know which dimension this value indexes (the frozen
        // rank-1 table). That answer wins; otherwise fall back to the chain
        // rule, which for a 2-D schedule is exactly iterCount[1] / [0].
        size_t count = rank1Count.value_or(isInner ? iterCount.back()
                                                   : iterCount.front());
        slicedShape[0] = std::max(tensorShape[0] / count, size_t(1));
      }
    } else {
      // A tensor that out-ranks the iteration schedule. iterCount_ is sized
      // from the load/store tensors alone (see tensorDim in computeIterCount),
      // so an intermediate tensor can carry more dims than the schedule has
      // entries for: histogramdd_from_bin_tensors loads rank-1 coordinates and
      // materialises the `edges[None, :] <= coord[:, None]` comparison as
      // rank-2, with the point axis -- the axis the rank-1 tensors carry --
      // first. The schedule covers the leading dims, so slice those by
      // iterCount and leave the trailing dims alone.
      slicedShape.assign(tensorShape.begin(), tensorShape.end());
      for (size_t i = 0; i < iterCount.size(); ++i)
        slicedShape[i] = std::max(tensorShape[i] / iterCount[i], size_t(1));
    }
    ArrayRef<int64_t> sliceTensorShape(slicedShape);
    Attribute encoding;
    if (needNewEncoding) {
      int rank = sliceTensorShape.size();
      llvm::SmallVector<unsigned> order(rank);
      std::iota(order.begin(), order.end(), 0);
      encoding = triton::xpu::ClusterLayoutAttr::get(
          tensorType.getContext(), sliceTensorShape, order, 128, 64);
    } else {
      encoding = tensorType.getEncoding();
    }
    return RankedTensorType::get(sliceTensorShape, tensorType.getElementType(),
                                 encoding);
  }
  return type;
}

// The tile dimension a rank-1 value spans, read off its own layout.
//
// Why: ttg.slice<{dim = D}> says the value is the parent tile sliced along D,
// so the vector itself spans the OTHER dimension. A reduce result is the one
// rank-1 value whose dimension is stated outright this way -- expand_dims never
// touches it, so widenedTileDimOfRank1() answers Unknown for it.
std::optional<unsigned> sliceEncodingTileDim(Value value) {
  auto ty = mlir::dyn_cast<RankedTensorType>(value.getType());
  if (!ty || ty.getShape().size() != 1)
    return std::nullopt;
  auto sliceEnc =
      mlir::dyn_cast_or_null<triton::gpu::SliceEncodingAttr>(ty.getEncoding());
  if (!sliceEnc)
    return std::nullopt;
  auto parentEnc =
      mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(sliceEnc.getParent());
  if (!parentEnc || parentEnc.getSizePerCore().size() != 2)
    return std::nullopt;
  return 1u - sliceEnc.getDim();
}

// Walk a rank-1 value back through layout conversions to one that names its
// tile dimension.
std::optional<unsigned> tileDimBehindConversions(Value value) {
  // `seen` visits each value once, so the finite def chain is walked to its end
  // without a fixed step cap -- a longer conversion chain is still followed.
  llvm::SmallPtrSet<Value, 8> seen;
  while (seen.insert(value).second) {
    if (auto dim = sliceEncodingTileDim(value))
      return dim;
    Operation *defOp = value.getDefiningOp();
    if (!defOp)
      return std::nullopt;
    Value onlyRank1;
    for (Value operand : defOp->getOperands()) {
      auto opTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!opTy || opTy.getShape().size() != 1)
        continue;
      if (onlyRank1)
        return std::nullopt; // ambiguous, two rank-1 operands
      onlyRank1 = operand;
    }
    if (!onlyRank1)
      return std::nullopt;
    value = onlyRank1;
  }
  return std::nullopt;
}

// The tile dimension a rank-1 store pointer must span, taken from the value the
// store writes.
//
// Why: a store's pointer is never widened into the tile, so
// widenedTileDimOfRank1() answers Unknown for it and the caller falls back to
// chain membership. That fallback picks the reduce axis, which is wrong when
// the pointer is built from an index vector the other axis shares: a SQUARE
// tile CSEs `tl.arange(0, BM)` and `tl.arange(0, BN)` into a single range, and
// the rank-1 output pointer then came out one element wide while the reduce
// result it stores kept the full surviving-axis length. Non-square tiles never
// hit this -- the two ranges have different lengths and do not CSE.
//
// How:
//   - follow `user` forward through rank-1-preserving ops to the stores it
//     feeds as the pointer operand;
//   - read each store's value operand through tileDimBehindConversions();
//   - require every store reached to agree.
std::optional<unsigned> tileDimOfStoredValue(Value value) {
  llvm::SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> seen;
  std::optional<unsigned> answer;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *consumer : current.getUsers()) {
      if (!seen.insert(consumer).second)
        continue;
      bool isStore =
          isa<mlir::triton::StoreOp, mlir::triton::xpu::StoreOp,
              mlir::triton::xpu::LM2GMOp, mlir::triton::xpu::LM2GMMaskOp,
              mlir::triton::xpu::SM2GMOp>(consumer);
      if (isStore) {
        if (consumer->getNumOperands() < 2 ||
            consumer->getOperand(0) != current)
          continue; // not the pointer operand
        auto dim = tileDimBehindConversions(consumer->getOperand(1));
        if (!dim)
          continue;
        if (answer && *answer != *dim)
          return std::nullopt;
        answer = dim;
        continue;
      }
      if (consumer->getNumResults() != 1)
        continue;
      auto resTy =
          mlir::dyn_cast<RankedTensorType>(consumer->getResult(0).getType());
      if (resTy && resTy.getShape().size() == 1)
        worklist.push_back(consumer->getResult(0));
    }
  }
  return answer;
}

// Which tile dimension a rank-1 value's length indexes, when the IR says so.
//
// Why this cannot be read off the chains: partitionChains() builds innerChain_
// as an axis-unaware backward closure from the axis-0 anchors, so a rank-1
// value that reaches the tile through expand_dims(axis=1) -- a dim-0 value --
// is swept in together with the dim-1 ones. getSlicedType()'s rank-1 branch
// then answers iterCount[1] for it and leaves it at full length while the tile
// around it is sliced along dim 0.
//
// The answer is structural: follow the value through rank-1-preserving ops
// until it is widened into the tile, and read the dimension off that widening.
// expand_dims(axis=0) turns a length-K vector into the K column of the tile
// (dim 1); expand_dims(axis=1) turns it into the K row (dim 0).
//
// The three answers the walk below can give. "Conflict" and "unknown" are kept
// apart because they mean opposite things to the caller: a value that is widened
// into BOTH tile dimensions is genuinely axis-ambiguous and must keep its
// chain-based split, while a value that is never widened at all may still take
// its dimension from a sibling (see mappedTileDimOfRank1).
enum class WidenedDim { Unknown, Conflict, Known };

// Which tile dimension this rank-1 value -- or a rank-1 value it feeds -- is
// widened into.
WidenedDim widenedTileDimOfRank1(Value value, unsigned &dim) {
  llvm::SmallVector<Value> worklist{value};
  llvm::SmallPtrSet<Operation *, 16> seen;
  bool found = false;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    for (Operation *user : current.getUsers()) {
      if (!seen.insert(user).second || user->getNumResults() != 1)
        continue;
      auto resTy =
          mlir::dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (!resTy)
        continue;
      auto shape = resTy.getShape();
      if (shape.size() == 1) {
        // Still a length vector: keep looking for the widening.
        worklist.push_back(user->getResult(0));
        continue;
      }
      if (shape.size() != 2)
        continue;
      auto expandOp = mlir::dyn_cast<triton::ExpandDimsOp>(user);
      if (!expandOp)
        continue;
      unsigned widenedDim = expandOp.getAxis() == 0 ? 1u : 0u;
      if (found && dim != widenedDim)
        return WidenedDim::Conflict;
      dim = widenedDim;
      found = true;
    }
  }
  return found ? WidenedDim::Known : WidenedDim::Unknown;
}

// computeRank1TileDim(): widenedTileDimOfRank1(), plus a one-step fallback
// through a rank-1 SIBLING.
//
// Why the fallback is needed: not every rank-1 value is widened itself. A
// store's own mask and pointer operands never are -- they only ever meet the
// tile through the index vector they are compared with or added to:
//
//   %offs_n_2 = arith.addi %offs_n_1, %offs_n_0 : tensor<32xi32, #cluster>
//   %ptrs_4   = triton_xpu.convert_layout %offs_n_2 : ... -> tensor<32xi32, slice dim=0>
//   %ptrs_5   = tt.expand_dims %ptrs_4 {axis = 0} -> tensor<1x32xi32, #cluster>   // dim 1
//   ...
//   %mask = arith.cmpi slt, %offs_n_2, %splat_N : tensor<32xi1>                   // never widened
//   %op   = tt.addptr %out_splat, %offs_n_2 : tensor<32x!tt.ptr<f32>>             // never widened
//
// With only the widening walk those two fall back to chain membership, which
// for an axis-0 reduce tree answers iterCount[0] and leaves them at width 1
// while %offs_n_2 is width 32 -- the arith.cmpi then fails to verify.
//
// How:
//   - only when the widening walk found nothing;
//   - look at the rank-1 values this one is combined with by its users, take
//     their widened dimension, and require it to be unambiguous.
//
// It is deliberately one step and NOT a connected-component walk: a rank-1
// value that feeds both axes (a make_range shared by two expand_dims) would
// otherwise merge the two axes into one component and answer nullopt for both,
// which is what a first attempt at this did -- it regressed the lit suite
// (139 -> 137) and the R03 shapes (11/11 -> 7/11).
std::optional<unsigned> computeRank1TileDim(
    Value value, llvm::DenseMap<Value, std::optional<unsigned>> &memo,
    llvm::SmallPtrSet<void *, 32> &inProgress) {
  // Memoised and cycle-safe. The fallbacks below recurse, so without the memo a
  // long rank-1 chain would be walked once per value on it, and a loop-carried
  // value can reach itself through its own loop.
  auto cached = memo.find(value);
  if (cached != memo.end())
    return cached->second;
  if (!inProgress.insert(value.getAsOpaquePointer()).second)
    return std::nullopt;

  auto remember = [&](std::optional<unsigned> dim) {
    inProgress.erase(value.getAsOpaquePointer());
    memo[value] = dim;
    return dim;
  };

  unsigned dim = 0;
  switch (widenedTileDimOfRank1(value, dim)) {
  case WidenedDim::Known:
    return remember(dim);
  case WidenedDim::Conflict:
    // Genuinely on both axes: leave it to the chain split.
    return remember(std::nullopt);
  case WidenedDim::Unknown:
    break;
  }

  for (Operation *user : value.getUsers()) {
    if (user->getNumResults() != 1)
      continue;
    auto userTy = mlir::dyn_cast<RankedTensorType>(user->getResult(0).getType());
    if (!userTy || userTy.getShape().size() != 1)
      continue;
    bool found = false;
    unsigned siblingDim = 0;
    for (Value operand : user->getOperands()) {
      if (operand == value)
        continue;
      auto opTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!opTy || opTy.getShape().size() != 1)
        continue;
      auto operandDim = computeRank1TileDim(operand, memo, inProgress);
      if (!operandDim)
        continue;
      if (found && siblingDim != *operandDim)
        return remember(std::nullopt);
      siblingDim = *operandDim;
      found = true;
    }
    if (found)
      return remember(siblingDim);
  }

  // Nothing usable among the users either -- the value's own users may all be
  // sinks (a store's mask compare is only ever consumed by the store). Then the
  // dimension is whatever the rank-1 operands it is computed from carry.
  if (Operation *defOp = value.getDefiningOp()) {
    bool found = false;
    unsigned operandDim = 0;
    for (Value operand : defOp->getOperands()) {
      auto operandTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy || operandTy.getShape().size() != 1)
        continue;
      auto dim = computeRank1TileDim(operand, memo, inProgress);
      if (!dim)
        continue;
      if (found && operandDim != *dim)
        return remember(std::nullopt);
      operandDim = *dim;
      found = true;
    }
    if (found)
      return remember(operandDim);
  }

  // Last resort: the value is a store operand that nothing else pins down.
  if (auto dim = tileDimOfStoredValue(value))
    return remember(dim);
  return remember(std::nullopt);
}

// The tile dimension an expand_dims widens its source into.
//
// Why: an expand_dims states its axis outright, so the dimension of the rank-1
// value it consumes is not a guess -- axis 0 turns a length-K vector into the
// 1xK column (the source spans dim 1), axis 1 turns it into the Kx1 row (dim
// 0). The source itself is rank 1, so the general rank-1 refinements that read
// a user's own result type skip an expand_dims (its result is rank 2) and fall
// back to chain membership. partitionChains() deliberately drops an axis-0
// expand from innerChain_, so that fallback then mistakes a column index for a
// value that wants the sliced outer width -- a square BLOCK_M == BLOCK_N tile
// CSEs its two aranges into one, and the store pointer then collapsed onto the
// diagonal because the column index took the per-core row value.
// The narrowest shape among the ranked tensor operands of `op`, or nullopt
// when none of them carries one.
static std::optional<llvm::SmallVector<int64_t>>
tryGetNarrowestOperandShape(mlir::Operation *op) {
  std::optional<llvm::SmallVector<int64_t>> narrowest;
  for (Value operand : op->getOperands()) {
    auto tensorTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
    if (!tensorTy)
      continue;
    llvm::SmallVector<int64_t> own(tensorTy.getShape());
    if (!narrowest || product(own) < product(*narrowest))
      narrowest = own;
  }
  return narrowest;
}

// A value holding `shape` worth of `wide`.
//
// How: a splat constant is re-materialised at the narrow shape because its
// width carries no information; anything else gets a
// triton_xpu.extract_slice, which already has an LLVM lowering
// (XPUExtractSliceOpConversion).
static Value buildNarrowView(OpBuilder &builder, Location loc, Value wide,
                             llvm::ArrayRef<int64_t> shape) {
  auto wideTy = mlir::cast<RankedTensorType>(wide.getType());
  auto narrowTy = RankedTensorType::get(shape, wideTy.getElementType(),
                                        wideTy.getEncoding());
  auto constOp = wide.getDefiningOp<mlir::arith::ConstantOp>();
  auto dense = constOp ? mlir::dyn_cast<DenseElementsAttr>(constOp.getValue())
                       : DenseElementsAttr();
  if (dense && dense.isSplat())
    return builder.create<mlir::arith::ConstantOp>(
        loc, narrowTy,
        DenseElementsAttr::get(narrowTy, dense.getSplatValue<Attribute>()));
  return builder.create<triton::xpu::ExtractSliceOp>(loc, narrowTy, wide);
}

// Restate every ranked tensor result of `op` at `shape`.
//
// Why: an arith op does not implement InferTypeOpInterface, so after its
// operands are narrowed the result has to be set directly or the op fails its
// own verifier -- arith.cmpi wants the operand shape with an i1 element type,
// arith.divf wants the operand type outright. Only the shape is taken: the
// element type belongs to the result (a compare returns i1), and so does the
// encoding.
static void retypeTensorResultsTo(mlir::Operation *op,
                                  llvm::ArrayRef<int64_t> shape) {
  for (Value result : op->getResults()) {
    auto tensorTy = mlir::dyn_cast<RankedTensorType>(result.getType());
    if (!tensorTy || tensorTy.getShape() == shape)
      continue;
    result.setType(RankedTensorType::get(shape, tensorTy.getElementType(),
                                         tensorTy.getEncoding()));
  }
}

static std::optional<unsigned> expandDimUserTileDim(mlir::Operation *user) {
  if (auto expand = dyn_cast<mlir::triton::ExpandDimsOp>(user))
    return expand.getAxis() == 0 ? 1u : 0u;
  return std::nullopt;
}

mlir::Operation *findRootOp(mlir::Operation *op) {
  mlir::Operation *rootOp = op;
  while (auto parent = rootOp->getParentOp()) {
    rootOp = parent;
    if (rootOp->getParentOp() && isa<triton::FuncOp>(rootOp->getParentOp()))
      return rootOp;
  }
  return op;
}

// One dimension's share of a cluster: the elements a core-group pair covers
// along that dimension.
static size_t computePerDimClusterExtent(unsigned sizePerCore, unsigned shape,
                                         unsigned groupsPerCluster,
                                         unsigned coresPerGroup,
                                         bool unrollOpt, unsigned bufferSize) {
  if (unrollOpt)
    return groupsPerCluster * coresPerGroup;
  // The layout's sizePerCore is a claim, not a budget. ClusterLayoutAttr's
  // 4-argument builder clamps it with lm-buflen, but a layout rebuilt later
  // does not have to: getOneCoreGEncoding() (CreateGM2LM.cpp, the atomic
  // simulation) rebuilds it from the full shape on a single core, so a
  // 1024-element tile arrives claiming sizePerCore = 1024. Trusting that makes
  // the slicing decision a tautology -- the tile "fits" by definition, nothing
  // is sliced, and lm-buflen stops having any effect at all, which is why
  // fake_quantize_learnable_per_tensor_affine_backward kept 38 live
  // 1024-element f32 tiles on one core and died in the compiler's buffer-size
  // tuning loop with no knob left to turn (measured: lm-buflen 512 / 128 / 32
  // / 16 all produced the same single-trip loop).
  unsigned perCore = std::min(sizePerCore, bufferSize);
  return std::min(perCore, shape) * groupsPerCluster * coresPerGroup;
}

// How many elements one cluster covers for `type`.
//
// Why: the core/group geometry only exists on ClusterLayoutAttr, so a tensor
// carrying any other encoding (a bare SliceEncodingAttr, say) has nothing to
// size against and cannot be sliced. That used to be llvm_unreachable -- a
// process abort, and in an NDEBUG build a silent mis-slice.
//
// How:
//   - return 1 for a non-tensor type (there is nothing to distribute);
//   - reject a tensor whose encoding is not ClusterLayoutAttr, and one whose
//     rank disagrees with the layout, reporting on `anchor`;
//   - multiply the per-dimension extents together.
//
// \param anchor op the diagnostic is reported on (the tree's dataflow sink).
// \param type tensor to size.
// \param unrollOpt ignore sizePerCore and use the group spread alone.
static FailureOr<size_t> getSizePerCluster(Operation *anchor, Type &type,
                                           bool unrollOpt,
                                           unsigned bufferSize) {
  auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
  if (tensorType == nullptr)
    return size_t(1);

  auto global =
      mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(tensorType.getEncoding());
  if (global == nullptr) {
    anchor->emitError("only ClusterLayoutAttr tensors can be sliced");
    return failure();
  }

  auto tensorShape = tensorType.getShape();
  auto groupsPerCluster = global.getCoresPerGroup();
  auto coresPerGroup = global.getGroupsPerCluster();
  auto sizePerCore = global.getSizePerCore();

  if (tensorShape.size() != groupsPerCluster.size()) {
    anchor->emitError("tensor rank does not match the cluster layout rank");
    return failure();
  }

  size_t sizePerCluster = 1;
  for (auto i = 0; i < tensorShape.size(); ++i) {
    sizePerCluster *= computePerDimClusterExtent(
        sizePerCore[i], tensorShape[i], groupsPerCluster[i], coresPerGroup[i],
        unrollOpt, bufferSize);
  }
  return sizePerCluster;
}

llvm::SmallVector<int64_t> getSlicedShape(const std::vector<int64_t> &shape,
                                          size_t spaceSize,
                                          bool isReduceMultiGroup) {
  size_t dimSize = 1u;
  size_t rank = shape.size();
  llvm::SmallVector<int64_t> slicedShape(rank, 1u);
  // The loop below walks the dims from the last one backwards, so it is rank
  // agnostic. The supported domain is enforced earlier, in
  // validateTensorShapes(): rank > 2 is rejected unless allow-rank3 is set.

  if (!isReduceMultiGroup) // no opt
    spaceSize = std::min(static_cast<int64_t>(spaceSize), shape[rank - 1]);

  for (int i = rank - 1; i >= 0; --i) {
    dimSize *= shape[i];
    const double sliceNum = static_cast<double>(dimSize) / spaceSize;
    if (sliceNum > 1) {
      slicedShape[i] = std::ceil(shape[i] / sliceNum);
      break;
    }
    slicedShape[i] = shape[i];
  }
  return slicedShape;
}

// Per-iteration context used by the slicing handlers below. Bundling the
// captured locals lets each `if`/`else if` branch live in its own member
// function instead of one giant inline lambda block.
/// Whether `reduceOp`'s result is broadcast back along the axis it reduced.
///
/// Why: inside a slicing loop that result is only the running prefix until
/// the last slice, and StoreControl guards a store derived from it to the
/// last iteration -- so `x - tl.sum(x, axis=0)` writes the last row slice
/// and leaves the others at whatever the output buffer held. Lowering this
/// shape correctly needs the slicing loop split in two, which is not done
/// yet; this predicate only finds the shape so the damage can be counted.
/// See third_party/xpu/docs/axis0_reduce_broadcast_back.md.
///
/// How:
///   - walk the result forward through layout conversions and reshapes;
///   - answer yes when a tt.expand_dims restores the axis that was reduced;
///   - bound the walk, a few links are enough for the shapes seen so far.
static bool collectBroadcastBack(mlir::triton::ReduceOp reduceOp,
                          llvm::SmallVectorImpl<mlir::Operation *> &expands) {
  // Walk the finite SSA use-graph to its end; `seen` visits each value once,
  // so no fixed depth cap is needed -- a normalization chain longer than
  // today's is still followed, not silently cut (that cut once lowered
  // `x - mean` as a running prefix).
  llvm::SmallVector<mlir::Value> worklist;
  llvm::SmallPtrSet<mlir::Value, 32> seen;
  for (mlir::Value res : reduceOp->getResults())
    worklist.push_back(res);
  while (!worklist.empty()) {
    mlir::Value value = worklist.pop_back_val();
    if (!seen.insert(value).second)
      continue;
    for (mlir::Operation *user : value.getUsers()) {
      if (auto expand = dyn_cast<mlir::triton::ExpandDimsOp>(user)) {
        if (expand.getAxis() == reduceOp.getAxis())
          expands.push_back(expand);
        continue;
      }
      // Follow the aggregate forward through layout/shape passthroughs AND
      // through any op that keeps it rank-1. A mean is `sum / N`, i.e. an
      // arith.divf sitting between the reduce and the expand_dims; stopping at
      // the divf missed the whole normalization family and the slicing loop
      // then lowered `x - mean` as a running prefix (silently wrong). The
      // expand_dims below still gates on the reduced axis, and
      // isBroadcastBackOntoTile()'s load-behind check still rejects the benign
      // store-only form, so widening the walk only adds the cases a scalar
      // rescale had been hiding.
      bool passthrough =
          isa<mlir::triton::xpu::ConvertLayoutOp, mlir::triton::ReshapeOp,
              mlir::triton::SplatOp>(user);
      bool keepsRank1 = false;
      if (user->getNumResults() == 1)
        if (auto resTy = mlir::dyn_cast<RankedTensorType>(
                user->getResult(0).getType()))
          keepsRank1 = resTy.getShape().size() == 1;
      if (passthrough || keepsRank1)
        for (mlir::Value res : user->getResults())
          worklist.push_back(res);
    }
  }
  return !expands.empty();
}

/// Whether a load is reachable backwards from `value` without entering
/// `stopAt`.
///
/// Why: the real question is "does this value still carry the tile that was
/// reduced", and asking it by SSA identity against the reduce source does
/// not work here -- Legalize has already replaced those values, so the
/// comparison is against stale ones (see
/// third_party/xpu/docs/axis0_reduce_broadcast_back.md). Treating the reduce
/// as a leaf and looking for a load instead is stable: the only route to
/// that tile which does not pass through the reduce is the load that brought
/// it in.
///
/// How:
///   - walk operands backwards, bounded;
///   - never descend into `stopAt`, so the reduce is not a path to its own
///     input;
///   - answer yes on the first load.
static bool hasLoadBehind(mlir::Value value, mlir::Operation *stopAt) {
  // `seen` already bounds this to the finite SSA graph, so walk the whole
  // chain -- a load deeper than any fixed cap must still be found.
  llvm::SmallPtrSet<mlir::Value, 32> seen;
  llvm::SmallVector<mlir::Value> work;
  work.push_back(value);
  while (!work.empty()) {
    mlir::Value current = work.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    mlir::Operation *def = current.getDefiningOp();
    if (!def || def == stopAt)
      continue;
    if (isa<mlir::triton::LoadOp, mlir::triton::xpu::LoadOp,
            mlir::triton::xpu::GM2LMOp>(def))
      return true;
    for (mlir::Value operand : def->getOperands())
      work.push_back(operand);
  }
  return false;
}

/// Whether the broadcast-back result meets the reduced tile again.
///
/// Why: carrying the shape is not enough to be wrong. `nansum` and `sum_dim`
/// at dim 0 broadcast an axis-0 reduce result back and are correct to 1e-6
/// because the only consumer is the store of that aggregate; `orgqr`,
/// `logsumexp` and `linalg_householder_product` feed it into arithmetic
/// together with the tile that was reduced, and all three are wrong. Only
/// the second form needs the slicing loop split, so a rewrite or a rejection
/// has to tell them apart.
///
/// How:
///   - for each expand_dims restoring the reduced axis, follow its users
///     through broadcast and layout conversions;
///   - skip stores, which is the benign form;
///   - answer yes when another operand of such a consumer still has a load
///     behind it.
static bool isBroadcastBackOntoTile(mlir::triton::ReduceOp reduceOp) {
  llvm::SmallVector<mlir::Operation *> expands;
  if (!collectBroadcastBack(reduceOp, expands))
    return false;
  llvm::SmallVector<mlir::Operation *> producers;
  for (mlir::Operation *expand : expands)
    producers.push_back(expand);
  llvm::SmallPtrSet<mlir::Operation *, 16> visited;
  while (!producers.empty()) {
    mlir::Operation *producer = producers.pop_back_val();
    if (!visited.insert(producer).second)
      continue;
    for (mlir::Operation *user : producer->getUsers()) {
      if (isa<mlir::triton::StoreOp, mlir::triton::xpu::StoreOp,
              mlir::triton::xpu::LM2GMOp>(user))
        continue;
      // Check this consumer, then keep following it: the tile does not
      // always meet the broadcast value at the first arithmetic op.
      // logsumexp goes expand -> cmpf -> select -> `inp - safe_m`, and only
      // that last one has the tile as its other operand.
      for (mlir::Value operand : user->getOperands()) {
        if (operand.getDefiningOp() == producer)
          continue;
        if (hasLoadBehind(operand, reduceOp))
          return true;
      }
      mlir::Dialect *dialect = user->getDialect();
      bool elementwise =
          dialect && (dialect->getNamespace() == "arith" ||
                      dialect->getNamespace() == "math");
      if (elementwise || isa<mlir::triton::xpu::BroadcastOp,
                             mlir::triton::xpu::ConvertLayoutOp>(user))
        producers.push_back(user);
    }
  }
  return false;
}

/// Whether the peel may take this reduce into its own slicing loop.
///
/// Why: the peel exists for a reduce along the dimension the loop slices --
/// only that one sees a running prefix instead of the whole reduction (see
/// third_party/xpu/docs/axis0_reduce_broadcast_back.md). Two conditions bound
/// what it can act on:
///   - the slicing loop cuts dim 0, so only an axis-0 reduce has the defect.
///     An axis-1 reduce is complete within every slice, and peeling it is both
///     pointless and broken: `addmv` crashed in replaceAllUsesWith() on a
///     carried total that lowerSlicingLoop() never produced for that shape.
///   - the peel hands back exactly one value -- lowerSlicingLoop() sets a
///     single `carriedReduceTotal_` and the split site rewires result 0 to it.
///     A Welford reduce returns three (mean, m2, weight), which inductor
///     builds every fused layer_norm from, so results 1 and 2 would keep their
///     users while the reduce is erased; MLIR then destroys an operation that
///     still has uses and crashes inside the diagnostic for it.
///
/// TODO(@yanxin11): accept multi-result reduces once lowerSlicingLoop() carries
/// one total per result. The axis-0 bound is not a gap to close -- an axis-1
/// reduce has nothing to peel.
static bool canPeelReduceIntoOwnLoop(mlir::triton::ReduceOp reduceOp) {
  if (reduceOp.getAxis() != 0)
    return false;
  return reduceOp->getNumResults() == 1;
}

/// Collect the ops of `reduceOp`'s chain that live in its own block.
///
/// Why: the accumulate loop needs its own copy of the chain -- `lower()` moves
/// ops rather than cloning them, so the two loops cannot share one. Keeping to
/// one block makes the copy well defined: program order gives the clone order
/// (`isBeforeInBlock` requires one block) and the insertion point is just ahead
/// of the first op.
///
/// Cross-block chains were tried and reverted: treating out-of-block
/// definitions as available values is sound on paper -- they dominate the
/// insertion point -- but Legalize then fails outright on orgqr, whose reduce
/// sits in its k loop. See third_party/xpu/docs/axis0_reduce_broadcast_back.md.
///
/// How:
///   - walk operands backwards from the reduce;
///   - give up as soon as a definition sits in another block;
///   - hand the chain back in program order.
static bool collectReduceChainInBlock(
    mlir::triton::ReduceOp reduceOp,
    llvm::SmallVectorImpl<mlir::Operation *> &chain,
    const llvm::SmallPtrSetImpl<mlir::Value> *stopValues = nullptr) {
  mlir::Block *block = reduceOp->getBlock();
  llvm::SetVector<mlir::Operation *> seen;
  llvm::SmallVector<mlir::Operation *> work;
  work.push_back(reduceOp.getOperation());
  while (!work.empty()) {
    mlir::Operation *op = work.pop_back_val();
    if (op->getBlock() != block)
      return false;
    if (!seen.insert(op))
      continue;
    for (mlir::Value operand : op->getOperands()) {
      // Stop at a carried total: it is a loop-invariant leaf, not part of this
      // reduce's sliceable chain (see the chained-accumulator split).
      if (stopValues && stopValues->contains(operand))
        continue;
      if (mlir::Operation *def = operand.getDefiningOp())
        work.push_back(def);
    }
  }
  chain.assign(seen.begin(), seen.end());
  llvm::sort(chain, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return true;
}

/// How many times the schedule cuts the dimension a rank-1 result indexes.
///
/// Why: for a rank-3 schedule a rank-1 value can index dim 0, 1 or 2, and the
/// inner/outer boolean cannot express that -- `isInner ? back : front` always
/// answers dim 2 or dim 0. conv2d feeds a 3-D tile through expand_dims(axis=0)
/// and a rank-3 tt.splat of the same tile; the splat was sliced per-dim while
/// the expand_dims chain kept `back`, so the cmpi that joins them had a 1x32x1
/// operand against a 1x1x1 one.
///
/// How:
///   - use the frozen rank-1 table when it has an answer and the dim is in
///     range;
///   - otherwise keep the chain rule (inner -> last dim, outer -> first),
///     which leaves every 2-D tree bit-for-bit unchanged.
static std::optional<size_t> rank1CountForResult(
    const llvm::DenseMap<Value, std::optional<unsigned>> &rank1Dims,
    llvm::ArrayRef<size_t> iterCount, Operation *op, unsigned resIdx) {
  // Only a schedule with more than two dims needs this. For a 2-D tree the
  // chain answer is the pinned authority (xpu_legalize_loop_invariant_arange
  // pins it), and `isInner ? iterCount[1] : iterCount[0]` already expresses
  // every dimension such a schedule has.
  if (iterCount.size() <= 2)
    return std::nullopt;
  auto resTy =
      mlir::dyn_cast<RankedTensorType>(op->getResult(resIdx).getType());
  if (!resTy || resTy.getShape().size() != 1)
    return std::nullopt;
  auto it = rank1Dims.find(op->getResult(resIdx));
  if (it == rank1Dims.end() || !it->second || *it->second >= iterCount.size())
    return std::nullopt;
  return iterCount[*it->second];
}

/// Whether a rank-1 result of `op` has to be sliced by iterCount[1] (dim 1)
/// rather than by the chain membership the caller passed in.
///
/// Why: iterCount is per dimension, and a rank-1 value indexes exactly one of
/// them. getSlicedType()'s rank-1 branch resolves that with
/// `isInner ? iterCount[1] : iterCount[0]`, i.e. it takes chain membership as
/// the answer. That holds for every tree whose inner chain really is dim 1,
/// but an axis-0 reduce makes dim 0 the axis the tree is built around, and the
/// backward closure then also claims the dim-0 helper values. Those keep their
/// full length while the tile around them is sliced, and the first op that
/// combines the two fails to verify. computeRank1Dims() answers the question
/// structurally instead; only 2-D trees are affected, because that is where
/// getSlicedType()'s rank-1 branch is reachable at all.
static bool isInnerForRank1Result(
    const llvm::DenseMap<Value, std::optional<unsigned>> &rank1Dims,
    llvm::ArrayRef<size_t> iterCount, Operation *op, unsigned resIdx,
    bool isInner) {
  if (iterCount.size() <= 1)
    return isInner;
  auto resTy =
      mlir::dyn_cast<RankedTensorType>(op->getResult(resIdx).getType());
  if (!resTy || resTy.getShape().size() != 1)
    return isInner;
  auto it = rank1Dims.find(op->getResult(resIdx));
  if (it != rank1Dims.end() && it->second)
    return *it->second == 1;
  return isInner;
}

struct SliceLoopCtx {
  llvm::SetVector<Operation *> &sortedOpTree;
  llvm::SmallVector<size_t> &iterCount;
  size_t outIterCount;
  llvm::SetVector<Operation *> &innerChain;
  llvm::SetVector<Operation *> &outerChain;
  // Frozen before lowering: see OpTree::rank1Dims_.
  const llvm::DenseMap<Value, std::optional<unsigned>> &rank1Dims;
  size_t reduceGroupSize;
  size_t reduceRowsPerCore;
  OpBuilder &builder;
  Value idx;
  Value upper;

  // The sliced replacements of every tt.make_range this lowering rewrote,
  // grouped by the range's (start, end). splitSharedMakeRanges() clones a
  // shared range once per chain, so one logical range can end up with a
  // 1-wide loop-carried copy AND a full-width static one; a consumer that
  // got the wrong one is repaired by looking its twin up here (see
  // tryGetRangeTwinAtWidth).
  llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::SmallVector<Value>>
      rangeFamilyByBounds{};

  // The rank-1 values this lowering already settled from a dim-stated operand.
  // Why they are kept: the width a reduce result states has to survive the
  // chain below it, and only the first consumer reads the reduce itself.
  // linalg_eig chains two selects on the same column -- `xv = where(support,
  // col_k, 0)` then `xp = where(idx == k + 1, xv, 0)` -- and for the second
  // one the only dim-stated operand is the first one's result. Without this
  // the fallback in tryGetResultRank1Width() wins and the authority flips to
  // the stale pre-slice width, which then tries to widen the reduce result.
  llvm::DenseSet<Value> dimStatedRank1Values{};

  // Thin wrappers over the shared rank-1 width rules, so the lowering that
  // leaves ops in the root block (lowerSlicingLoopInsideFor) can apply the
  // very same ones. Why they exist: see rank1CountForResult() and
  // isInnerForRank1Result() above.
  std::optional<size_t> rank1CountFor(Operation *op, unsigned resIdx,
                                      bool isInner) const {
    return rank1CountForResult(rank1Dims, iterCount, op, resIdx);
  }

  bool isInnerForResult(Operation *op, unsigned resIdx, bool isInner) {
    return isInnerForRank1Result(rank1Dims, iterCount, op, resIdx, isInner);
  }

  void setSlicedResTy(Operation *op, bool isInner = false,
                      bool needNewEncoding = false) {
    for (auto [i, resTy] : llvm::enumerate(op->getResultTypes())) {
      auto slicedResTy =
          getSlicedType(resTy, iterCount, isInnerForResult(op, i, isInner),
                        needNewEncoding, rank1CountFor(op, i, isInner));
      op->getResult(i).setType(slicedResTy);
    }
  }

  // The type an elementwise / compare / select result must take, read off the
  // op's already-final operands.
  //
  // These ops all require their result to have the same shape as their operands,
  // so the result cannot be sliced from iterCount and the op's own chain
  // membership alone -- those can disagree with an operand that something else
  // retyped (a loop whose yield set its result, a reduction, ...). The operand
  // is the authority because producers are sliced before their users.
  //
  // A non-constant operand wins over a constant one: a constant is retyped by
  // its own chain membership in handleConstantInLoop() and only reconciled with
  // its siblings afterwards, so reading it here can freeze the wrong width.
  // When every operand is a constant the largest shape is taken, and the
  // reconciliation pass brings the others up.
  std::optional<Type> resultTypeFromOperands(Operation *op) {
    // The trait covers the elementwise arithmetic ops; the compares and the
    // select carry the same contract through a custom verifier instead.
    // Both traits appear in this dialect: the shape-only one covers the
    // casts, the type one covers the arithmetic ops (arith.subf and friends).
    // The compares and the select carry the same contract through a custom
    // verifier instead.
    if (!op->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() &&
        !op->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() &&
        !isa<mlir::arith::SelectOp, mlir::arith::CmpFOp, mlir::arith::CmpIOp>(op))
      return std::nullopt;
    Type constTy;
    for (Value operand : op->getOperands()) {
      auto operandTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy)
        continue;
      if (!isa_and_nonnull<mlir::arith::ConstantOp>(operand.getDefiningOp()))
        return operand.getType();
      if (!constTy ||
          mlir::cast<RankedTensorType>(constTy).getNumElements() <
              operandTy.getNumElements())
        constTy = operand.getType();
    }
    if (!constTy)
      return std::nullopt;
    return constTy;
  }

  void handleXpuPrintOp(XPUPrintOp xpuPrintOp) {
    OpBuilder replacer(xpuPrintOp);
    auto loc = xpuPrintOp->getLoc();
    Value bound = replacer.create<mlir::arith::IndexCastOp>(
        loc, replacer.getI64Type(), upper);
    Value idx64 =
        replacer.create<arith::ExtSIOp>(loc, replacer.getI64Type(), idx);
    xpuPrintOp.setOperand(4, idx64);
    xpuPrintOp.setOperand(6, bound);
  }

  // Rewrite tt.reduce → triton_xpu.reduce, and
  // tt.reduce_return → triton_xpu.reduce_return.
  //
  // Additional parameters carried by triton_xpu.reduce vs tt.reduce:
  //   - outIterCount: total number of slicing loop iterations. The backend
  //     uses this to know when all partial results are ready for final combine.
  //   - idx: current loop iteration index. The backend uses this to decide
  //     whether to initialize or accumulate the partial reduce result.
  //
  // The combine region's terminator must be triton_xpu.reduce_return (enforced
  // by MLIR's HasParent trait on triton_xpu.ReduceOp).
  //
  // Example IR (before, inside slicing loop):
  //   %r = tt.reduce(%src) axis=1 : tensor<4x256xf32> -> tensor<4xf32>
  //     combineOp { ^bb(%a, %b): tt.reduce.return %add(%a,%b) }
  //
  // Example IR (after):
  //   %r = triton_xpu.reduce(%src) axis=1, iterCount=8, idx=%i
  //       : tensor<4x32xf32> -> tensor<4xf32>
  //     combineOp { ^bb(%a, %b): triton_xpu.reduce.return %add(%a,%b) }
  //
  // @param erasedOps  If non-null, collect replaced ops for deferred erasure
  //                   (needed when called inside a block iteration).
  /// Report the shape above when TRITONXPU_REPORT_REDUCE_BROADCAST is set.
  ///
  /// Why: the fix is a structural change to the slicing loop, so the decision
  /// between fixing it and rejecting it needs to know how many real kernels
  /// are affected. Reporting is off by default and changes nothing else.
  void reportBroadcastBack(mlir::triton::ReduceOp reduceOp) {
    static const bool report =
        ::getenv("TRITONXPU_REPORT_REDUCE_BROADCAST") != nullptr;
    if (!report)
      return;
    llvm::SmallVector<mlir::Operation *> expands;
    if (!collectBroadcastBack(reduceOp, expands))
      return;
    llvm::errs() << "[reduce-broadcast-back] axis=" << reduceOp.getAxis()
                 << (isBroadcastBackOntoTile(reduceOp) ? " onto-tile=yes"
                                                       : " onto-tile=no")
                 << " at " << reduceOp.getLoc() << "\n";
  }

  /// Give every result of `op` the shape its operands agree on.
  ///
  /// Why: inside a region the only width rule the body walk has is chain
  /// membership, which answers per op and not per value. A rank-1 chain that
  /// starts at a reduce result therefore keeps the pre-slice width while the
  /// reduce itself is sliced -- slogdet's `convert_layout` of the axis-1
  /// reduce result is the first op to fail ("requires the same shape for all
  /// operands and results"). The operands are the authority here, exactly as
  /// in lower()'s closing sweep over the root block.
  ///
  /// \param hasChangedResult set to true when a result was retyped, so the
  /// caller can iterate to a fixpoint.
  void setResultShapesFromOperands(mlir::Operation *op,
                                   bool &hasChangedResult) {
    auto operandTy = resultTypeFromOperands(op);
    if (!operandTy)
      return;
    auto shapeTy = mlir::dyn_cast<RankedTensorType>(*operandTy);
    if (!shapeTy)
      return;
    for (mlir::Value res : op->getResults()) {
      auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
      if (!resTy || resTy.getShape() == shapeTy.getShape())
        continue;
      // Shape only: the element type and the encoding stay the result's own
      // (arith.extsi widens the element type, a slice layout differs from its
      // parent).
      res.setType(RankedTensorType::get(shapeTy.getShape(),
                                        resTy.getElementType(),
                                        resTy.getEncoding()));
      hasChangedResult = true;
    }
  }

  /// Rebuild `value` as a rank-1 value of `width` elements, or null.
  ///
  /// Why: the operand a consumer needs may not exist yet at that width. Only
  /// the shapes the slicing loop can produce for free are rebuilt: a range has
  /// a twin, a splat of a scalar and a dense constant are width-agnostic, and
  /// an elementwise op is rebuilt from rebuilt operands. Anything else is left
  /// alone, so the verifier still reports the case instead of this quietly
  /// inventing data.
  ///
  /// How: twin first, then clone-and-retype, then recurse through an
  /// elementwise op; `depth` stops a pathological chain.
  Value tryRebuildRank1AtWidth(Value value, int64_t width, unsigned depth) {
    if (hasRank1Width(value, width))
      return value;
    if (depth == 0)
      return Value();
    if (Value twin = tryGetRangeTwinAtWidth(value, width))
      return twin;
    Operation *def = value.getDefiningOp();
    if (!def || def->getNumResults() != 1)
      return Value();
    return tryRebuildDefAtWidth(def, width, depth);
  }

  /// A splat constant of `width` elements with the same value, or null.
  Value tryRebuildSplatConstantAtWidth(mlir::arith::ConstantOp constOp,
                                       int64_t width) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!attr || !attr.isSplat())
      return Value();
    auto resTy = mlir::cast<RankedTensorType>(constOp.getType());
    auto newTy = RankedTensorType::get({width}, resTy.getElementType(),
                                       resTy.getEncoding());
    OpBuilder constBuilder(constOp);
    auto newAttr = mlir::DenseElementsAttr::get(
        newTy, attr.getSplatValue<mlir::Attribute>());
    return constBuilder
        .create<mlir::arith::ConstantOp>(constOp.getLoc(), newTy, newAttr)
        .getResult();
  }

  /// Clone `def` so its rank-1 result has `width` elements, or null.
  Value tryRebuildDefAtWidth(Operation *def, int64_t width, unsigned depth) {
    if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
      return tryRebuildSplatConstantAtWidth(constOp, width);
    // Same contract as resultTypeFromOperands(): the shape-only trait covers
    // the casts, the type trait covers the arithmetic ops (arith.andi carries
    // only the type one), and the compares and the select state it in a
    // custom verifier instead.
    if (!isa<triton::SplatOp>(def) &&
        !def->hasTrait<mlir::OpTrait::SameOperandsAndResultShape>() &&
        !def->hasTrait<mlir::OpTrait::SameOperandsAndResultType>() &&
        !isa<mlir::arith::CmpIOp, mlir::arith::CmpFOp, mlir::arith::SelectOp>(
            def))
      return Value();
    OpBuilder cloneBuilder(def);
    Operation *clone = cloneBuilder.clone(*def);
    if (failed(retypeCloneOperands(clone, width, depth))) {
      clone->erase();
      return Value();
    }
    auto resTy = mlir::cast<RankedTensorType>(clone->getResult(0).getType());
    clone->getResult(0).setType(RankedTensorType::get(
        {width}, resTy.getElementType(), resTy.getEncoding()));
    return clone->getResult(0);
  }

  /// Point every rank-1 operand of `clone` at a value of `width` elements.
  LogicalResult retypeCloneOperands(Operation *clone, int64_t width,
                                    unsigned depth) {
    for (mlir::OpOperand &use : clone->getOpOperands()) {
      auto operandTy = mlir::dyn_cast<RankedTensorType>(use.get().getType());
      if (!operandTy || operandTy.getShape().size() != 1)
        continue;
      Value rebuilt = tryRebuildRank1AtWidth(use.get(), width, depth - 1);
      if (!rebuilt)
        return failure();
      use.set(rebuilt);
    }
    return success();
  }

  /// Swap one mismatched rank-1 operand for a value of `width` elements.
  /// \param hasRebuiltOperand set to true when the operand was replaced, so
  /// the caller only retypes the result once the operands really agree.
  void rebuildMismatchedRank1Operand(mlir::OpOperand &use, int64_t width,
                                     bool &hasRebuiltOperand) {
    auto ty = mlir::dyn_cast<RankedTensorType>(use.get().getType());
    if (!ty || ty.getShape().size() != 1 || ty.getShape()[0] == width)
      return;
    // Bound the rebuild recursion by the enclosing function's op count: the
    // rank-1 def chain cannot be longer than that, so this never cuts a valid
    // chain short the way a fixed 8 could.
    unsigned depthBound = 1;
    if (auto funcOp =
            use.getOwner()->getParentOfType<mlir::FunctionOpInterface>())
      funcOp->walk([&](mlir::Operation *) { ++depthBound; });
    Value rebuilt = tryRebuildRank1AtWidth(use.get(), width, depthBound);
    if (!rebuilt)
      return;
    use.set(rebuilt);
    hasRebuiltOperand = true;
  }

  /// Point a region's mismatched rank-1 operands at the copy of the right
  /// width.
  ///
  /// Why: with a square tile the splits leave one logical arange as several
  /// sliced copies and a consumer inside the region can hold the wrong one.
  /// slogdet's `tl.where(cols < N, diag, 1.0)` is the case: the condition was
  /// built on the 1-wide row copy while `diag` -- an axis-0 reduce result, so
  /// its dimension is stated by its layout -- is 16 wide, and arith.select
  /// rejected the pair.
  ///
  /// How: for each op whose rank-1 operands disagree with a known-dimension
  /// operand, rebuild the disagreeing ones at that width (range twin,
  /// width-agnostic splat/constant, or an elementwise op of those).
  void alignRank1OperandsToKnownDim(mlir::Block &block) {
    for (mlir::Operation &op : block)
      alignOneOpToKnownDim(&op);
  }

  /// Realign one op whose rank-1 operands disagree with a known-dimension one.
  ///
  /// How:
  ///   - skip an op that does not require its operands to agree: a dim-0 index
  ///     meeting a dim-1 value at an addptr is two chains at two legitimate
  ///     widths;
  ///   - rebuild the disagreeing operands at the authoritative width;
  ///   - only when one was really rebuilt, take the result from the operands
  ///     and settle the chain below. Retyping the result of an op whose
  ///     operand could NOT be rebuilt freezes the wrong width instead -- that
  ///     is what broke xpu_legalize_axis0_index_both_chains, whose addf has a
  ///     1-wide loop-carried accumulator beside a 32-wide reduce result.
  void alignOneOpToKnownDim(mlir::Operation *op) {
    if (!resultTypeFromOperands(op))
      return;
    std::optional<int64_t> width = tryGetAuthoritativeRank1Width(op);
    if (!width)
      return;
    bool hasRebuiltOperand = false;
    for (mlir::OpOperand &use : op->getOpOperands())
      rebuildMismatchedRank1Operand(use, *width, hasRebuiltOperand);
    if (!hasRebuiltOperand)
      return;
    bool hasChangedResult = false;
    setResultShapesFromOperands(op, hasChangedResult);
    if (!hasChangedResult)
      return;
    recordDimStatedResults(op);
    settleAlignedWidthDownstream(op);
  }

  /// Settle the chain below an op whose operands were just realigned.
  ///
  /// Why: changing one op's width leaves every op below it stale, including
  /// the constants beside them and the ops that come earlier in program order
  /// than the sibling they must agree with (slogdet's `safe_diag` select sits
  /// ahead of the compare that feeds it). This is deliberately stronger than
  /// propagateReduceWidthToUsers(): there the authority is a reduce that the
  /// accumulator chain must NOT follow.
  ///
  /// How: collect the users in the block, then sweep constants-then-results to
  /// a fixpoint, following the ops that changed.
  void settleAlignedWidthDownstream(mlir::Operation *op) {
    llvm::SetVector<mlir::Operation *> users;
    collectBlockUsers(op, users);
    // Sweep to a fixpoint. settleOneAlignedUser() only ever widens a user
    // forward and can still grow `users`, so loop until a whole sweep changes
    // nothing. A fixed round count risked stopping before a freshly-added user
    // settled; the monotone no-change return below bounds the loop instead.
    while (true) {
      bool hasChangedAny = false;
      for (unsigned i = 0; i < users.size(); ++i)
        settleOneAlignedUser(users[i], users, hasChangedAny);
      if (!hasChangedAny)
        return;
    }
  }

  /// One op of the settle sweep: its constants, then its result.
  void settleOneAlignedUser(mlir::Operation *user,
                            llvm::SetVector<mlir::Operation *> &users,
                            bool &hasChangedAny) {
    reshapeConstOperandsToMatchSiblings(user, /*skipIfInTree=*/false);
    bool hasChangedResult = false;
    setResultShapesFromOperands(user, hasChangedResult);
    if (!hasChangedResult)
      return;
    recordDimStatedResults(user);
    collectBlockUsers(user, users);
    hasChangedAny = true;
  }

  /// Whether `value` is a rank-1 tensor whose tile dimension is stated by the
  /// IR itself (a reduce result, through its slice encoding).
  bool isKnownDimRank1(Value value) {
    auto ty = mlir::dyn_cast<RankedTensorType>(value.getType());
    if (!ty || ty.getShape().size() != 1)
      return false;
    return tileDimBehindConversions(value).has_value() ||
           dimStatedRank1Values.contains(value);
  }

  /// Remember `op`'s rank-1 results as dim-stated, so the width keeps flowing.
  void recordDimStatedResults(mlir::Operation *op) {
    for (mlir::Value res : op->getResults()) {
      auto ty = mlir::dyn_cast<RankedTensorType>(res.getType());
      if (ty && ty.getShape().size() == 1)
        dimStatedRank1Values.insert(res);
    }
  }

  /// The width of the rank-1 operand whose tile dimension is known, if the
  /// operands that have one agree.
  std::optional<int64_t> tryGetKnownDimRank1Width(Operation *op) {
    std::optional<int64_t> width;
    for (Value operand : op->getOperands()) {
      if (!isKnownDimRank1(operand))
        continue;
      int64_t operandWidth =
          mlir::cast<RankedTensorType>(operand.getType()).getShape()[0];
      if (width && *width != operandWidth)
        return std::nullopt;
      width = operandWidth;
    }
    return width;
  }

  /// Whether `op` reads a rank-1 value of a width other than `width`.
  bool hasRank1OperandOtherThan(Operation *op, int64_t width) {
    for (Value operand : op->getOperands()) {
      auto ty = mlir::dyn_cast<RankedTensorType>(operand.getType());
      if (ty && ty.getShape().size() == 1 && ty.getShape()[0] != width)
        return true;
    }
    return false;
  }

  /// The width of `op`'s own rank-1 result.
  ///
  /// Why: the known-dimension rule needs a reduce among the operands.
  /// linalg_eig's `support = (idx >= k + 1) & (idx < N)` has none -- both sides
  /// are index chains over the same CSE-shared arange -- but the compare's own
  /// result was already sliced by its chain membership, and that is the width
  /// its operands have to meet:
  ///   'arith.cmpi' op failed to verify that result type has i1 element type
  ///   and same shape as operands
  ///   : (tensor<1xi32>, tensor<8xi32>) -> tensor<8xi1>
  std::optional<int64_t> tryGetResultRank1Width(Operation *op) {
    if (op->getNumResults() != 1)
      return std::nullopt;
    auto resTy = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resTy || resTy.getShape().size() != 1)
      return std::nullopt;
    return resTy.getShape()[0];
  }

  /// The width `op`'s rank-1 operands must agree on, when one of them says so.
  ///
  /// Why: a square tile cannot tell dim 0 from dim 1 by layout alone, so the
  /// copies of a CSE-shared arange look interchangeable. A reduce result is the
  /// one rank-1 value whose dimension is stated outright, so it is the
  /// authority when the operands disagree -- slogdet's
  /// `tl.where(cols < N, diag, 1.0)` has `diag` from an axis-0 reduce at 16
  /// against a condition built on the 1-wide row copy of that same arange.
  std::optional<int64_t> tryGetAuthoritativeRank1Width(Operation *op) {
    std::optional<int64_t> authoritative = tryGetKnownDimRank1Width(op);
    if (!authoritative)
      authoritative = tryGetResultRank1Width(op);
    if (!authoritative || !hasRank1OperandOtherThan(op, *authoritative))
      return std::nullopt;
    return authoritative;
  }

  /// The twin of `value` from the same split range, at `width` elements.
  ///
  /// Why: splitSharedMakeRanges() clones a shared tt.make_range once per
  /// chain, so one logical range becomes several sliced values -- a 1-wide
  /// loop-carried out_range for the index that moves with the slicing loop and
  /// a full-width static make_range for the index that does not. A consumer
  /// inside a region can end up holding the wrong one (slogdet: `cols < N`
  /// got the 1-wide row copy), and the twin is what it should have read.
  ///
  /// How: find the family `value` belongs to (same (start, end) bounds) and
  /// return its member of the requested width.
  Value tryGetRangeTwinAtWidth(Value value, int64_t width) {
    for (auto &family : rangeFamilyByBounds) {
      if (!llvm::is_contained(family.second, value))
        continue;
      for (Value twin : family.second)
        if (hasRank1Width(twin, width))
          return twin;
      return tryFindUntouchedRange(value, family.first, width);
    }
    // `value` is in no family: it is the copy this lowering never rewrote, so
    // it is still a range op and carries its own bounds. The sliced copy of
    // the same logical range is in the family under those bounds -- that is
    // the direction linalg_eig needs, from the full-width `idx` down to the
    // 1-wide row index the reduce result is at.
    return tryGetFamilyMemberByBounds(value, width);
  }

  /// The (start, end) of `value` when it is a range op, for the family key.
  std::optional<std::pair<uint32_t, uint32_t>> tryGetRangeBounds(Value value) {
    if (auto ttRange = value.getDefiningOp<triton::MakeRangeOp>())
      return std::make_pair(ttRange.getStart(), ttRange.getEnd());
    if (auto xpuRange = value.getDefiningOp<triton::xpu::MakeRangeOp>())
      return std::make_pair(xpuRange.getStart(), xpuRange.getEnd());
    return std::nullopt;
  }

  /// A sliced copy of `value`'s own range at `width` elements, or null.
  ///
  /// How: key the family by the bounds the range op carries. The members were
  /// built at the top of the slicing loop body, so they dominate every use
  /// inside it, including the ones nested in an scf.while.
  Value tryGetFamilyMemberByBounds(Value value, int64_t width) {
    auto bounds = tryGetRangeBounds(value);
    if (!bounds)
      return Value();
    auto family = rangeFamilyByBounds.find(*bounds);
    if (family == rangeFamilyByBounds.end())
      return Value();
    for (Value twin : family->second)
      if (hasRank1Width(twin, width))
        return twin;
    return Value();
  }

  /// A still-unlowered tt.make_range of the same bounds and `width`, or null.
  ///
  /// Why: only the copies this lowering rewrote are in rangeFamilyByBounds.
  /// The copy that keeps the full width belongs to the other chain, is never
  /// moved into this slicing loop, and is still a tt.make_range at this point
  /// -- lowerRemainingMakeRanges() converts it at the end of the pass.
  ///
  /// How: scan the enclosing function for a tt.make_range with the family's
  /// bounds and the requested width, and take it only from outside the use's
  /// own block, where it dominates every use inside the loop.
  Value tryFindUntouchedRange(Value value, std::pair<uint32_t, uint32_t> bounds,
                              int64_t width) {
    auto funcOp = value.getParentRegion()->getParentOfType<triton::FuncOp>();
    if (!funcOp)
      return Value();
    mlir::Block *useBlock = value.getParentBlock();
    Value found;
    funcOp.walk([&](triton::MakeRangeOp rangeOp) {
      if (rangeOp.getStart() != bounds.first ||
          rangeOp.getEnd() != bounds.second)
        return;
      if (rangeOp->getBlock() == useBlock ||
          !hasRank1Width(rangeOp.getResult(), width))
        return;
      found = rangeOp.getResult();
    });
    return found;
  }

  /// Whether `value` is a rank-1 tensor of exactly `width` elements.
  static bool hasRank1Width(Value value, int64_t width) {
    auto ty = mlir::dyn_cast<RankedTensorType>(value.getType());
    return ty && ty.getShape().size() == 1 && ty.getShape()[0] == width;
  }

  /// Collect the users of `op` that live in its own block.
  void collectBlockUsers(mlir::Operation *op,
                         llvm::SetVector<mlir::Operation *> &users) {
    for (mlir::Value result : op->getResults())
      for (mlir::Operation *user : result.getUsers())
        if (user->getBlock() == op->getBlock())
          users.insert(user);
  }

  /// Hand the sliced reduce width to the chain that reads the reduce.
  ///
  /// Why: inside a region the body walk sizes every op from chain membership,
  /// which answers per op and not per value, so the ops reading a reduce
  /// result keep the pre-slice width. slogdet's convert_layout of an axis-1
  /// reduce result is the first to fail: "'triton_xpu.convert_layout' op
  /// requires the same shape for all operands and results".
  ///
  /// Only the consumers of THIS reduce are touched. A blanket
  /// operand-authority sweep over the whole body also pulls the accumulator
  /// chain to the reduce width, which broke
  /// xpu_legalize_axis0_index_both_chains (iter_arg at 1 against a reduce at
  /// 32) -- measured, not feared.
  ///
  /// How: breadth-first over the users inside the reduce's own block, taking
  /// each result shape from its operands and following only the ops that
  /// actually changed.
  void propagateReduceWidthToUsers(triton::xpu::ReduceOp reduceOp) {
    llvm::SetVector<mlir::Operation *> users;
    collectBlockUsers(reduceOp, users);
    for (unsigned i = 0; i < users.size(); ++i) {
      bool hasChangedResult = false;
      setResultShapesFromOperands(users[i], hasChangedResult);
      if (hasChangedResult)
        collectBlockUsers(users[i], users);
    }
  }

  /// Let every reduce in `block` hand its width to the chain that reads it.
  ///
  /// Why: this has to run after the body walk, not from inside
  /// rewriteReduceInRegion(): during the walk a consumer is still ahead of the
  /// cursor, and setSlicedResTy() would overwrite the width again.
  void propagateBlockReduceWidths(mlir::Block &block) {
    for (mlir::Operation &op : block)
      if (auto xpuReduce = mlir::dyn_cast<triton::xpu::ReduceOp>(op))
        propagateReduceWidthToUsers(xpuReduce);
  }

  /// Whether a reduce's rank-1 result spans the inner dimension (dim 1).
  ///
  /// Why: setSlicedResTy() resolves the width of a rank-1 result through the
  /// frozen rank-1 table and falls back to the caller's chain membership. The
  /// rewritten reduce is a brand new op, so the table cannot hold its result,
  /// and chain membership answers the chain the REDUCE is on, not the
  /// dimension its RESULT spans: an axis-1 reduce sits on the inner chain
  /// while its result spans dim 0. slogdet / linalg_tensorinv hit exactly that
  /// -- the source was sliced to tensor<1x16xf32> and the result kept
  /// tensor<16xf32>, so the reduce's own verifier rejected the pair with
  /// "inferred type(s) tensor<1xf32> are incompatible with return type(s)
  /// tensor<16xf32>".
  ///
  /// How: a rank-2 source leaves exactly the other dimension, so the result
  /// spans 1 - axis; anything else keeps the caller's answer (a rank-1 source
  /// reduces to a scalar, a rank-3 one to a rank-2 tile, and neither reaches
  /// getSlicedType()'s rank-1 branch).
  bool isInnerForReduceResult(triton::ReduceOp reduceOp, bool isInner) const {
    auto srcTy =
        mlir::dyn_cast<RankedTensorType>(reduceOp.getSrcs()[0].getType());
    if (!srcTy || srcTy.getShape().size() != 2)
      return isInner;
    return reduceOp.getAxis() == 0;
  }

  void rewriteReduceInRegion(triton::ReduceOp reduceOp, bool isInner,
                             SetVector<Operation *> *erasedOps = nullptr) {
    reportBroadcastBack(reduceOp);
    OpBuilder b(reduceOp);
    auto newReduceIdxOp = b.create<triton::xpu::ReduceOp>(
        reduceOp.getLoc(), reduceOp->getResultTypes(), reduceOp.getSrcs(),
        reduceOp.getAxis(), outIterCount, idx);
    auto &newCombineOp = newReduceIdxOp.getCombineOp();
    b.cloneRegionBefore(reduceOp.getCombineOp(), newCombineOp,
                        newCombineOp.end());
    setSlicedResTy(newReduceIdxOp, isInnerForReduceResult(reduceOp, isInner));
    reduceOp.replaceAllUsesWith(newReduceIdxOp->getResults());
    if (erasedOps)
      erasedOps->insert(reduceOp);
    // Replace tt.reduce_return → triton_xpu.reduce_return inside the cloned
    // combine region (HasParent trait requires matching terminator type).
    for (auto &opInCombine : newCombineOp.getOps()) {
      if (auto redReturnOp =
              dyn_cast<mlir::triton::ReduceReturnOp>(opInCombine)) {
        auto saved = b.saveInsertionPoint();
        b.setInsertionPoint(redReturnOp);
        auto newRedReturnOp = b.create<triton::xpu::ReduceReturnOp>(
            redReturnOp.getLoc(), redReturnOp.getOperands());
        b.restoreInsertionPoint(saved);
        redReturnOp->replaceAllUsesWith(newRedReturnOp->getResults());
        if (erasedOps)
          erasedOps->insert(redReturnOp);
        else
          opInCombine.erase(); // avoid the HasParent Trait
        break;                 // stop the loop in combine region
      }
    }
  }

  // Rewrite tt.scan → triton_xpu.scan, and
  // tt.scan_return → triton_xpu.scan_return.
  //
  // Additional parameters carried by triton_xpu.scan vs tt.scan:
  //   - idx: current loop iteration index. The backend uses this to compute
  //     the correct prefix-sum offset across sliced tiles.
  //   (reverse is preserved from the original tt.scan.)
  //
  // The combine region's terminator must be triton_xpu.scan_return (HasParent
  // trait on triton_xpu.ScanOp).
  //
  // Example IR (before):
  //   %s = tt.scan(%src) axis=1, reverse=false : tensor<4x256xf32>
  //     combineOp { ^bb(%a, %b): tt.scan.return %add(%a,%b) }
  //
  // Example IR (after):
  //   %s = triton_xpu.scan(%src) axis=1, reverse=false, idx=%i
  //       : tensor<4x32xf32>
  //     combineOp { ^bb(%a, %b): triton_xpu.scan.return %add(%a,%b) }
  //
  // @param erasedOps  If non-null, collect replaced ops for deferred erasure.
  void rewriteScanInRegion(triton::ScanOp scanOp, bool isInner,
                           SetVector<Operation *> *erasedOps = nullptr) {
    OpBuilder b(scanOp);
    auto newScanIdxOp = b.create<triton::xpu::ScanOp>(
        scanOp.getLoc(), scanOp->getResultTypes(), scanOp.getSrcs(),
        scanOp.getAxis(), scanOp.getReverse(), idx);
    auto &newCombineOp = newScanIdxOp.getCombineOp();
    b.cloneRegionBefore(scanOp.getCombineOp(), newCombineOp,
                        newCombineOp.end());
    setSlicedResTy(newScanIdxOp, isInner);
    scanOp.replaceAllUsesWith(newScanIdxOp->getResults());
    if (erasedOps)
      erasedOps->insert(scanOp);
    for (auto &opInCombine : newCombineOp.getOps()) {
      if (auto scanReturnOp =
              dyn_cast<mlir::triton::ScanReturnOp>(opInCombine)) {
        auto saved = b.saveInsertionPoint();
        b.setInsertionPoint(scanReturnOp);
        auto newScanReturnOp = b.create<triton::xpu::ScanReturnOp>(
            scanReturnOp.getLoc(), scanReturnOp.getOperands());
        b.restoreInsertionPoint(saved);
        scanReturnOp->replaceAllUsesWith(newScanReturnOp->getResults());
        if (erasedOps)
          erasedOps->insert(scanReturnOp);
        else
          opInCombine.erase();
        break;
      }
    }
  }

  // After slicing shrinks non-constant operands, constant operands may still
  // have the original (larger) shape, causing a shape mismatch. This function
  // finds such mismatched constants and creates a new constant with the correct
  // (sliced) shape derived from a sibling non-constant operand.
  //
  // Example:
  //   %c = arith.constant dense<0.0> : tensor<256xf32>   // not sliced
  //   %add = arith.addf %sliced, %c : tensor<32xf32>, tensor<256xf32>  // BAD
  //   →
  //   %c_new = arith.constant dense<0.0> : tensor<32xf32>
  //   %add = arith.addf %sliced, %c_new : tensor<32xf32>, tensor<32xf32>
  //
  // @param skipIfInTree  Skip constants already in sortedOpTree (they are
  //                      handled separately by handleConstantInLoop).
  void reshapeConstOperandsToMatchSiblings(Operation *op, bool skipIfInTree) {
    // Ops whose operands have independent shapes — sibling-based inference
    // would incorrectly cross-contaminate them. Skip entirely.
    // scf.while's init operands map positionally onto the before-region block
    // arguments, and scf.condition forwards the loop-carried values by
    // position; both are like scf.for's initArgs / scf.yield, already listed
    // here. Inferring one of their constant operands from an arbitrary sibling
    // rebuilds it at the sibling's shape and breaks the positional match.
    if (isa<mlir::scf::ForOp, mlir::scf::IfOp, mlir::scf::YieldOp,
            mlir::scf::WhileOp, mlir::scf::ConditionOp,
            triton::ScanOp, triton::ReduceOp, triton::xpu::ScanOp,
            triton::xpu::ReduceOp>(op))
      return;
    for (unsigned opIdx = 0; opIdx < op->getNumOperands(); ++opIdx) {
      auto operand = op->getOperand(opIdx);
      auto constOp = operand.getDefiningOp<mlir::arith::ConstantOp>();
      if (!constOp)
        continue;
      if (skipIfInTree && sortedOpTree.contains(constOp))
        continue;
      auto constTensorTy = mlir::dyn_cast<RankedTensorType>(constOp.getType());
      if (!constTensorTy)
        continue;
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (!attr)
        continue;
      Type targetTy;
      for (unsigned sibIdx = 0; sibIdx < op->getNumOperands(); ++sibIdx) {
        if (sibIdx == opIdx)
          continue;
        auto sibTensorTy =
            mlir::dyn_cast<RankedTensorType>(op->getOperand(sibIdx).getType());
        if (!sibTensorTy)
          continue;
        if (sibTensorTy.getRank() != constTensorTy.getRank())
          continue;
        if (sibTensorTy.getShape() != constTensorTy.getShape()) {
          targetTy = RankedTensorType::get(sibTensorTy.getShape(),
                                           constTensorTy.getElementType(),
                                           constTensorTy.getEncoding());
          break;
        }
      }
      if (!targetTy)
        continue;
      auto slicedShapedTy = mlir::dyn_cast<ShapedType>(targetTy);
      if (!slicedShapedTy)
        continue;
      auto newValue = reshapeDenseAttrTo(attr, slicedShapedTy);
      if (!newValue)
        continue;
      OpBuilder constBuilder(op);
      auto newConstOp = constBuilder.create<mlir::arith::ConstantOp>(
          constOp.getLoc(), targetTy, newValue);
      op->setOperand(opIdx, newConstOp);
    }
  }

  // scf.yield operands map positionally to the parent op's results, so
  // reshapeConstOperandsToMatchSiblings (which skips YieldOp and infers shapes
  // from arbitrary siblings) cannot be used here. A constant yield operand that
  // still carries the pre-slice shape must be re-created to match the parent
  // result type at the same index.
  //
  // This happens for the welford first-iteration branch, whose then-region
  // yields function-level init constants (mean=0, weight=1) shared with the
  // reduce loop's iter_args: handleConstantInLoop only rewrites the in-tree
  // iter_arg use, leaving the nested yield referencing the original shape.
  void reshapeConstYieldOperandsToMatchResults(mlir::scf::YieldOp yieldOp,
                                               TypeRange resultTypes) {
    for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
      auto constOp =
          yieldOp.getOperand(i).getDefiningOp<mlir::arith::ConstantOp>();
      if (!constOp)
        continue;
      auto targetTy = mlir::dyn_cast<RankedTensorType>(resultTypes[i]);
      auto constTy = mlir::dyn_cast<RankedTensorType>(constOp.getType());
      if (!targetTy || !constTy)
        continue;
      if (targetTy.getShape() == constTy.getShape())
        continue;
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (!attr)
        continue;
      auto newValue =
          reshapeDenseAttrTo(attr, mlir::cast<ShapedType>(targetTy));
      if (!newValue)
        continue;
      OpBuilder constBuilder(yieldOp);
      auto newConstOp = constBuilder.create<mlir::arith::ConstantOp>(
          constOp.getLoc(), targetTy, newValue);
      yieldOp.setOperand(i, newConstOp);
    }
  }

  // Rewrite tt.make_range into XPU backend range ops for the slicing loop:
  //  - outerChain range (row-index, feeds expand_dims axis=1):
  //      → triton_xpu.out_range(groupSize, rowsPerCore, idx)
  //        Computes per-iteration row indices based on reduce group topology.
  //  - other range (column-index or inner):
  //      → triton_xpu.make_range(start, end, realSize, index, null)
  //        index = idx (outer, offset shifts per iteration) or null (inner, static).
  //
  // @param makeRangeOp  The tt.make_range to rewrite.
  // @param isInner      Whether this op belongs to the inner (reduce-axis) chain.
  // Whether `op` is part of this tree: one of the ops lower() walks, or an op
  // nested inside one of their regions.
  bool isInTree(Operation *op) {
    if (sortedOpTree.contains(op))
      return true;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp())
      if (sortedOpTree.contains(parent))
        return true;
    return false;
  }

  // Rewire the uses of a replaced value, but only inside this tree.
  //
  // Why: one tt.make_range can feed more than one OpTree, and each tree slices
  // it with its own iterCount. replaceAllUsesWith() hands this tree's sliced
  // value to the other tree's ops, which were sliced differently -- the
  // fractional_max_pool3d shape, where one tree cut the range to width 1 and the
  // other kept width 4. handleConstantInLoop() guards against the same hazard
  // for constants.
  void replaceUsesWithinTree(Value oldValue, Value newValue) {
    for (OpOperand &use : llvm::make_early_inc_range(oldValue.getUses()))
      if (isInTree(use.getOwner()))
        use.set(newValue);
  }

  // Re-derive a transpose from its already-sliced source.
  //
  // Why: a transpose carries no iterCount of its own -- its result dim i is the
  // source dim order[i], so slicing the result per-dim by iterCount would cut
  // it on the wrong axis. The shape and the layout both follow the permutation
  // of the (already sliced) source, which is also what keeps the op a register
  // rename for the LLVM lowering.
  //
  // How:
  //   - permute the sliced source shape by the order;
  //   - ask the layout interface for the permuted encoding of the SLICED
  //     source encoding (ClusterLayoutAttr::inferTransOpEncoding);
  //   - retype the result in place, keeping the element type.
  //
  // param op the transpose to retype.
  void handleTransInLoop(triton::TransOp op) {
    auto srcTy = mlir::dyn_cast<RankedTensorType>(op.getSrc().getType());
    if (!srcTy)
      return;
    auto order = op.getOrder();
    llvm::SmallVector<int64_t> retShape;
    retShape.reserve(order.size());
    for (auto axis : order)
      retShape.push_back(srcTy.getShape()[axis]);

    Attribute retEnc;
    if (auto srcEnc = srcTy.getEncoding()) {
      auto *infer = mlir::dyn_cast<triton::DialectInferLayoutInterface>(
          &srcEnc.getDialect());
      if (!infer ||
          failed(infer->inferTransOpEncoding(srcEnc, srcTy.getShape(), order,
                                             retEnc, op.getLoc())))
        return;
    }
    op.getResult().setType(
        RankedTensorType::get(retShape, srcTy.getElementType(), retEnc));
  }

  void handleMakeRangeInLoop(triton::MakeRangeOp makeRangeOp, bool isInner) {
    auto type = makeRangeOp.getType();
    auto loc = makeRangeOp->getLoc();
    if (outerChain.count(makeRangeOp)) {
      auto slicedResTy =
          getSlicedType(type, iterCount, false, /*needNewEncoding=*/false,
                        rank1CountFor(makeRangeOp, /*resIdx=*/0, false));
      auto newOutRangeOp = builder.create<triton::xpu::OutRangeOp>(
          loc, slicedResTy, reduceGroupSize, reduceRowsPerCore, idx);
      rangeFamilyByBounds[{makeRangeOp.getStart(), makeRangeOp.getEnd()}]
          .push_back(newOutRangeOp.getResult());
      replaceUsesWithinTree(makeRangeOp.getResult(),
                            newOutRangeOp.getResult());
    } else {
      Value index = isInner ? Value() : idx;
      uint32_t start = makeRangeOp.getStart();
      uint32_t end = makeRangeOp.getEnd();
      uint32_t realSize = end - start;
      auto newMakeRangeOp = builder.create<triton::xpu::MakeRangeOp>(
          loc, type, builder.getI32IntegerAttr(start),
          builder.getI32IntegerAttr(end), builder.getI32IntegerAttr(realSize),
          index, Value());
      // The replacement op is brand new, so the frozen rank-1 table has no
      // entry for it. For a rank-3 schedule take the dimension answer from the
      // op being replaced (rank1CountFor returns nullopt for 2-D, so that path
      // is unchanged).
      newMakeRangeOp->getResult(0).setType(
          getSlicedType(type, iterCount, isInner, /*needNewEncoding=*/false,
                        rank1CountFor(makeRangeOp, /*resIdx=*/0, isInner)));
      uint32_t newEnd = start + product(newMakeRangeOp.getType().getShape());
      newMakeRangeOp.setEnd(newEnd);
      rangeFamilyByBounds[{makeRangeOp.getStart(), makeRangeOp.getEnd()}]
          .push_back(newMakeRangeOp.getResult());
      replaceUsesWithinTree(makeRangeOp.getResult(),
                            newMakeRangeOp.getResult());
    }
  }

  void handleReduceInLoop(triton::ReduceOp reduceOp, bool isInner,
                          SetVector<Operation *> &toErase) {
    rewriteReduceInRegion(reduceOp, isInner, &toErase);
  }

  void handleScanInLoop(triton::ScanOp scanOp, bool isInner,
                        SetVector<Operation *> &toErase) {
    rewriteScanInRegion(scanOp, isInner, &toErase);
  }

  void handleConstantInLoop(mlir::arith::ConstantOp constOp, bool isInner) {
    auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!attr)
      return;
    auto slicedResTy = getSlicedType(constOp.getType(), iterCount, isInner);
    ShapedType slicedShapedType = mlir::cast<ShapedType>(slicedResTy);
    auto newValue = reshapeDenseAttrTo(attr, slicedShapedType);
    if (!newValue)
      return;
    auto newConstOp =
        builder.create<mlir::arith::ConstantOp>(constOp->getLoc(), slicedResTy, newValue);
    // Only replace uses within the current sortedOpTree to avoid affecting
    // constants shared across different op trees (which may have different
    // iterCounts).
    for (auto &use : llvm::make_early_inc_range(constOp->getUses())) {
      if (sortedOpTree.contains(use.getOwner()))
        use.set(newConstOp->getResult(0));
    }
  }

  // Reconcile an scf.for's iter_arg types with the loop's sliced result types.
  //
  // Why the arg type cannot simply be copied from the init operand: an
  // arith.constant is never retyped in place. handleConstantInLoop() clones it
  // at the sliced shape and rewires only the users inside sortedOpTree, and a
  // loop nested in another loop's body is not in that set -- so the clone is
  // left dead and the nested loop keeps the constant's pre-slice shape as its
  // init operand. Its body, meanwhile, is sliced normally, and the loop's own
  // result types were sliced by setSlicedResTy() above. That leaves the two
  // ends of one iter_arg at two widths, and the first body op that combines the
  // iter_arg with a body value fails to verify.
  //
  // Why the result type is the target: setSlicedResTy() already sliced it with
  // the loop's own chain membership, and scf.for requires
  // iter_arg == result == init operand. Taking the arg from the init operand
  // instead let the init operand win over the loop, which is backwards.
  //
  // How:
  //   - copy the result type onto the matching iter_arg;
  //   - re-create a constant init operand at that type, the same way
  //     reshapeConstYieldOperandsToMatchResults() treats a yielded constant.
  void alignForIterArgsWithResults(mlir::scf::ForOp forOp,
                                   bool useYieldType = false) {
    auto yieldOp =
        dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      // What the body computes decides the width, not the loop's own chain
      // membership. The loop op is never reached from its own body (a
      // loop-carried value enters through a block argument, which has no
      // defining op to walk back from), so its membership says nothing about
      // what the body needs -- and the two kernels below need opposite answers:
      //   _col_sum_kernel: the iter_arg is added to an axis-0 reduce result
      //     (inner, width 32), so the arg must be 32;
      //   loop_invariant_arange: it is added to a splat of the induction
      //     variable (outer, width 1), so the arg must be 1.
      // Before the body walk the yielded value has not been sliced yet, so its
      // type is the pre-slice one -- only the final pass may read it.
      Type targetTy = forOp.getResult(i).getType();
      if (useYieldType && yieldOp) {
        Value yielded = yieldOp.getOperand(i);
        if (Operation *defOp = yielded.getDefiningOp())
          if (forOp->isProperAncestor(defOp) &&
              !isa<mlir::arith::ConstantOp>(defOp))
            targetTy = yielded.getType();
      }
      forOp.getResult(i).setType(targetTy);
      forOp.getRegionIterArg(i).setType(targetTy);

      // Compare the INIT OPERAND, not the block argument. A block argument
      // starts out carrying the loop's original type, which is also what the
      // result type is when the loop needs no slicing -- so testing the
      // argument skipped the re-creation in exactly the case that needs it.
      // _col_sum_kernel's accumulator is that case: the for's result is already
      // tensor<32xf32>, but handleConstantInLoop() had already sliced the
      // init constant %acc to tensor<1xf32> because the constant is not in
      // innerChain_ (the closure reaches it from the loop's body, and a loop is
      // not walked from its own body).
      if (forOp.getInitArgs()[i].getType() == targetTy)
        continue;

      auto constOp =
          forOp.getInitArgs()[i].getDefiningOp<mlir::arith::ConstantOp>();
      auto targetTensorTy = mlir::dyn_cast<RankedTensorType>(targetTy);
      if (!constOp || !targetTensorTy)
        continue;
      auto attr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
      if (!attr)
        continue;
      // A splat works at any shape, and both directions occur here: the
      // histogramdd seed shrank (64 -> 1) and _col_sum_kernel's accumulator
      // grows (1 -> 32) because handleConstantInLoop() had already sliced it
      // down. Re-materialise the splat at the target shape rather than reusing
      // a raw buffer, whose length is the OLD element count.
      mlir::DenseElementsAttr newValue;
      if (attr.isSplat()) {
        newValue = mlir::DenseElementsAttr::get(
            mlir::cast<ShapedType>(targetTy), attr.getSplatValue<Attribute>());
      } else {
        // Otherwise the raw buffer must cover the smaller sliced shape; the
        // extra elements are the ones the other slices would have used, and
        // taking the prefix is what the other handlers do too.
        newValue = reshapeDenseAttrTo(attr, mlir::cast<ShapedType>(targetTy));
        if (!newValue)
          continue;
      }
      OpBuilder constBuilder(forOp);
      auto newConstOp = constBuilder.create<mlir::arith::ConstantOp>(
          constOp.getLoc(), targetTy, newValue);
      forOp.getInitArgsMutable().slice(i, 1).assign(ValueRange(newConstOp));
    }
  }

  /// Retype every op of a region body that the slicing loop carries.
  ///
  /// Why: the ops nested in a region are not in sortedOpTree -- that set is
  /// filtered to the root block -- so lower() never dispatches them. Without
  /// this walk they keep their pre-slice types while the tile around them is
  /// sliced, and the first op that mixes the two fails its verifier. scf.for
  /// has had this walk all along; scf.while had nothing, which is what left
  /// linalg_eig's QR iteration (`while k < N - 2`) full of mismatched rank-1
  /// chains.
  ///
  /// How: dispatch per op the same way lower() does for the root block --
  /// reduces and scans are rewritten to the XPU ops, nested regions recurse,
  /// everything else gets its result sliced and its constant operands
  /// reconciled -- then settle the widths the per-op rule cannot decide alone.
  void retypeRegionBodyOps(mlir::Block &block) {
    SetVector<Operation *> erasedOps;
    for (auto &inBlockOp : block) {
      if (erasedOps.contains(&inBlockOp))
        continue;
      bool inBlockIsInner = inOpChain(innerChain, &inBlockOp);
      if (auto reduceOpInFor = mlir::dyn_cast<triton::ReduceOp>(inBlockOp)) {
        rewriteReduceInRegion(reduceOpInFor, inBlockIsInner, &erasedOps);
      } else if (auto scanOpInFor = mlir::dyn_cast<triton::ScanOp>(inBlockOp)) {
        rewriteScanInRegion(scanOpInFor, inBlockIsInner, &erasedOps);
      } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(inBlockOp)) {
        // Set IfOp's childOp Arg Type(Then)
        auto &thenBlock = ifOp.getThenRegion().front();
        for (auto &thenOp : thenBlock) {
          bool isInnerOp = inOpChain(innerChain, &thenOp);
          if (handleShapeContractOp(&thenOp))
            continue;
          for (auto thenRes : thenOp.getResults()) {
            auto slicedOpType =
                getSlicedType(thenRes.getType(), iterCount, isInnerOp);
            thenRes.setType(slicedOpType);
          }
          if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(thenOp))
            handleXpuPrintOp(xpuPrintOp);
        }
        // Set IfOp's childOp Arg Type(Else)
        auto &elseRegion = ifOp.getElseRegion();
        if (!elseRegion.empty()) {
          for (auto &elseOp : elseRegion.front()) {
            bool isInnerOp = inOpChain(innerChain, &elseOp);
            if (handleShapeContractOp(&elseOp))
              continue;
            if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(elseOp))
              handleXpuPrintOp(xpuPrintOp);
          }
          for (auto ifRes : ifOp->getResults()) {
            auto slicedOpType = getSlicedType(ifRes.getType(), iterCount, true);
            ifRes.setType(slicedOpType);
          }
        }
        // After if/else body and result types are sliced, constant operands
        // referenced inside the regions may still carry the pre-slice shape.
        // This occurs for the welford first-iteration branch, whose init
        // constants (mean=0, weight=1) are function-level values shared with
        // the reduce loop's iter_args; handleConstantInLoop only rewrites the
        // in-tree iter_arg use, leaving the nested if/yield stale. Fix
        // elementwise ops via sibling inference and yields via positional
        // result matching.
        for (auto &thenOp : thenBlock)
          if (!isa<mlir::arith::ConstantOp>(thenOp))
            reshapeConstOperandsToMatchSiblings(&thenOp, /*skipIfInTree=*/false);
        if (auto thenYield =
                dyn_cast<mlir::scf::YieldOp>(thenBlock.getTerminator()))
          reshapeConstYieldOperandsToMatchResults(thenYield,
                                                  ifOp.getResultTypes());
        if (!elseRegion.empty()) {
          auto &elseBlock = elseRegion.front();
          for (auto &elseOp : elseBlock)
            if (!isa<mlir::arith::ConstantOp>(elseOp))
              reshapeConstOperandsToMatchSiblings(&elseOp,
                                                  /*skipIfInTree=*/false);
          if (auto elseYield =
                  dyn_cast<mlir::scf::YieldOp>(elseBlock.getTerminator()))
            reshapeConstYieldOperandsToMatchResults(elseYield,
                                                    ifOp.getResultTypes());
        }
      } else if (auto nestedWhile = dyn_cast<mlir::scf::WhileOp>(inBlockOp)) {
        setSlicedResTy(&inBlockOp, inBlockIsInner);
        handleWhileInLoop(nestedWhile);
      } else if (auto nestedFor = dyn_cast<mlir::scf::ForOp>(inBlockOp)) {
        // A loop nested inside this one is not in slicingOps_: that set is
        // filtered to the root op's own block, and a nested loop lives in this
        // loop's body block. lower() therefore never dispatches it, and without
        // this branch its body kept its pre-slice types while the enclosing
        // body was sliced -- so a value computed outside the nested loop and
        // one computed inside met at different shapes.
        handleForInLoop(nestedFor, inBlockIsInner);
      } else if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(inBlockOp)) {
        handleXpuPrintOp(xpuPrintOp);
      } else if (auto transOpInFor =
                     mlir::dyn_cast<triton::TransOp>(inBlockOp)) {
        handleTransInLoop(transOpInFor);
      } else if (handleShapeContractOp(&inBlockOp)) {
      } else {
        // Skip setSlicedResTy for constants that were already sliced by the
        // main sortedOpTree loop (via replaceAllUsesWith).
        if (!mlir::isa<mlir::arith::ConstantOp>(inBlockOp))
          setSlicedResTy(&inBlockOp, inBlockIsInner);
        // Constants may have been pre-sliced by the main sortedOpTree loop,
        // so getSlicedType would double-slice. Derive target shape from a
        // non-constant sibling operand instead.
        reshapeConstOperandsToMatchSiblings(&inBlockOp, /*skipIfInTree=*/false);
      }
    }
    for (auto op : erasedOps) {
      if (op->use_empty())
        op->erase();
    }
    // The walk above decides each op's width from chain membership alone;
    // this settles the rank-1 chains that have to follow a reduce ...
    propagateBlockReduceWidths(block);
    // ... and the consumers that hold the wrong copy of a shared range.
    alignRank1OperandsToKnownDim(block);
  }

  void handleForInLoop(mlir::scf::ForOp forOp, bool isInner) {
    if (iterCount.back() != 1) {
      if (auto stepOp = dyn_cast<mlir::arith::ConstantOp>(
              forOp.getStep().getDefiningOp())) {
        setSlicedResTy(stepOp, isInner);
      }
    }

    // Set forOp Result Type
    setSlicedResTy(forOp, isInner);

    // The loop's result types are final now, so make the iter_args (and any
    // constant init operand) agree with them. This has to happen before the
    // body walk: the walk fixes stale constant operands by sibling inference
    // against the iter_arg types, and it can only see the right sibling shape
    // once the iter_args are retyped.
    alignForIterArgsWithResults(forOp);

    retypeRegionBodyOps(forOp.getRegion().front());
  }

  /// Settle the rank-1 widths inside an scf.while the slicing loop carries.
  ///
  /// Why: the body walk knows scf.for and scf.if; `scf.while` has no handler at
  /// all (grep for WhileOp in this file came back empty), so the ops in its two
  /// regions keep whatever width they were given while the tile around them is
  /// sliced. linalg_eig's Gauss/QR iteration is one long `while k < N - 2`, and
  /// its `support = (idx >= k + 1) & (idx < N)` ends up comparing a 1-wide copy
  /// of the shared arange with an 8-wide one:
  ///   'arith.cmpi' op failed to verify that result type has i1 element type
  ///   and same shape as operands
  ///
  /// How: run the same three settling steps the for body gets -- stale
  /// constants against their siblings, the reduce widths down their consumer
  /// chains, and the operands that hold the wrong copy of a shared range -- on
  /// both regions. The op types themselves are already sliced by the tree walk;
  /// what is missing is the agreement between the chains.
  ///
  void handleWhileInLoop(mlir::scf::WhileOp whileOp) {
    if (whileOp.getBefore().empty() || whileOp.getAfter().empty())
      return;
    alignWhileArgsWithOperands(whileOp);
    retypeRegionBodyOps(whileOp.getBefore().front());
    retypeRegionBodyOps(whileOp.getAfter().front());
    // The walk can retype what scf.condition forwards, so take the arguments
    // and the results from it once more -- and then settle the ops that read
    // those arguments, which the walk sized before they changed. Re-walking
    // instead would double-slice: setSlicedResTy() divides the type it finds.
    alignWhileArgsWithOperands(whileOp);
    settleRegionBodyWidths(whileOp.getBefore().front());
    settleRegionBodyWidths(whileOp.getAfter().front());
  }

  /// Re-settle a region body whose block arguments changed after the walk.
  ///
  /// How: the three steps that only ever copy a width from one value to
  /// another -- stale constants against their siblings, the reduce widths down
  /// their consumer chains, and the operands holding the wrong copy of a
  /// shared range. None of them re-slices a type, so this is safe to run
  /// after the walk.
  void settleRegionBodyWidths(mlir::Block &block) {
    for (mlir::Operation &op : block)
      reshapeConstOperandsToMatchSiblings(&op, /*skipIfInTree=*/false);
    propagateBlockReduceWidths(block);
    alignRank1OperandsToKnownDim(block);
  }

  /// Make an scf.while's region arguments and results agree with the values
  /// that feed them.
  ///
  /// Why: the before-region arguments come from the init operands, and the
  /// after-region arguments and the while's own results both come from what
  /// scf.condition forwards. Once the operands are sliced those three sets are
  /// stale, and the while's verifier fires before anything else --
  /// linalg_eig's `while k < N - 2` carries the 8x8 tile and the loop index.
  void alignWhileArgsWithOperands(mlir::scf::WhileOp whileOp) {
    for (auto [arg, init] :
         llvm::zip(whileOp.getBeforeArguments(), whileOp.getInits()))
      arg.setType(init.getType());
    auto conditionOp = whileOp.getConditionOp();
    for (auto [arg, forwarded] :
         llvm::zip(whileOp.getAfterArguments(), conditionOp.getArgs()))
      arg.setType(forwarded.getType());
    for (auto [result, forwarded] :
         llvm::zip(whileOp->getResults(), conditionOp.getArgs()))
      result.setType(forwarded.getType());
  }

  // An scf.while inside the slicing loop needs the same treatment as the
  // nested scf.for above, and for the same reason: getOpTreeBwd keeps only
  // root-block ops, so nothing inside the while's regions reaches
  // sortedOpTree and lower() never dispatches it. Without this handler the
  // while fell to handleDefaultInLoop, which slices the op's own results and
  // stops there. A tt.reduce in the body then survives to
  // ConvertTritonXPUToLLVM, where the whole triton dialect is illegal, and the
  // only diagnostic is "failed to legalize operation 'tt.reduce'".
  // FlagGems' index_reduce writes exactly this shape: a compare-and-swap retry
  // loop whose exit test is tl.sum(stop) == BLOCK.
  //
  // scf.while carries two regions. `before` takes the init values as block
  // arguments and ends in scf.condition; `after` takes what scf.condition
  // forwards and ends in scf.yield. Both are sliced the way handleForInLoop
  // slices the single for body.
  void handleWhileInLoop(mlir::scf::WhileOp whileOp, bool isInner) {
    setSlicedResTy(whileOp, isInner);
    reshapeConstOperandsToMatchSiblings(whileOp, /*skipIfInTree=*/true);

    SetVector<Operation *> erasedOps;
    for (Region *region : {&whileOp.getBefore(), &whileOp.getAfter()}) {
      if (region->empty())
        continue;
      Block &block = region->front();
      for (auto blockArg : block.getArguments()) {
        bool isInnerArg = inOpChain(innerChain, blockArg.getDefiningOp());
        blockArg.setType(
            getSlicedType(blockArg.getType(), iterCount, isInnerArg));
      }
      for (auto &inBlockOp : block) {
        if (erasedOps.contains(&inBlockOp))
          continue;
        bool inBlockIsInner = inOpChain(innerChain, &inBlockOp);
        if (auto reduceOp = mlir::dyn_cast<triton::ReduceOp>(inBlockOp)) {
          rewriteReduceInRegion(reduceOp, inBlockIsInner, &erasedOps);
        } else if (auto scanOp = mlir::dyn_cast<triton::ScanOp>(inBlockOp)) {
          rewriteScanInRegion(scanOp, inBlockIsInner, &erasedOps);
        } else if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(inBlockOp)) {
          handleIfInLoop(ifOp, inBlockIsInner);
        } else if (auto nestedFor =
                       mlir::dyn_cast<mlir::scf::ForOp>(inBlockOp)) {
          handleForInLoop(nestedFor, inBlockIsInner);
        } else if (auto nestedWhile =
                       mlir::dyn_cast<mlir::scf::WhileOp>(inBlockOp)) {
          handleWhileInLoop(nestedWhile, inBlockIsInner);
        } else if (auto xpuPrintOp = mlir::dyn_cast<XPUPrintOp>(inBlockOp)) {
          handleXpuPrintOp(xpuPrintOp);
        } else {
          if (!mlir::isa<mlir::arith::ConstantOp>(inBlockOp))
            setSlicedResTy(&inBlockOp, inBlockIsInner);
          reshapeConstOperandsToMatchSiblings(&inBlockOp,
                                              /*skipIfInTree=*/false);
        }
      }
    }
    for (auto op : erasedOps) {
      if (op->use_empty())
        op->erase();
    }
  }

  void handleIfInLoop(mlir::scf::IfOp ifOp, bool isInner) {
    // Set IfOp Result Type
    setSlicedResTy(ifOp, isInner);

    // Set IfOp Arg Type
    for (auto ifArg : ifOp.getBody()->getArguments()) {
      bool isInnerArg = inOpChain(innerChain, ifArg.getDefiningOp());
      auto slicedArgType =
          getSlicedType(ifArg.getType(), iterCount, isInnerArg);
      ifArg.setType(slicedArgType);
    }

    // Set IfOp's childOp Arg Type(Then)
    auto &thenBlock = ifOp.getThenRegion().front();
    SetVector<Operation *> thenToErase;
    for (auto &thenOp : thenBlock) {
      bool isInnerOp = inOpChain(innerChain, &thenOp);
      for (auto thenRes : thenOp.getResults()) {
        auto slicedOpType =
            getSlicedType(thenRes.getType(), iterCount, isInnerOp);
        thenRes.setType(slicedOpType);
      }
      if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(thenOp)) {
        handleXpuPrintOp(xpuPrintOp);
      } else if (auto reduceOpInIf = mlir::dyn_cast<triton::ReduceOp>(thenOp)) {
        rewriteReduceInRegion(reduceOpInIf, isInnerOp, &thenToErase);
      }
    }
    for (auto &thenOp : thenBlock) {
      if (isa<mlir::arith::ConstantOp>(thenOp))
        continue;
      if (thenToErase.contains(&thenOp))
        continue;
      reshapeConstOperandsToMatchSiblings(&thenOp, /*skipIfInTree=*/false);
    }
    // scf.yield operands map positionally to the if's results, so a value that
    // is yielded from the region but defined outside it -- a function-level
    // constant, say -- is never reached by the loops above. Without this the
    // if result is sliced while the yielded constant keeps the pre-slice shape,
    // and the scf.if successor-operand check fires. handleForInLoop() already
    // does this for an if nested in a for; the top-level path did not.
    if (auto thenYield =
            dyn_cast<mlir::scf::YieldOp>(thenBlock.getTerminator()))
      reshapeConstYieldOperandsToMatchResults(thenYield,
                                              ifOp.getResultTypes());
    for (auto *op : thenToErase)
      op->erase();

    // Set IfOp's childOp Arg Type(Else)
    auto &elseRegion = ifOp.getElseRegion();
    if (!elseRegion.empty()) {
      for (auto &elseOp : elseRegion.front()) {
        bool isInnerOp = inOpChain(innerChain, &elseOp);
        for (auto elseRes : elseOp.getResults()) {
          auto slicedOpType =
              getSlicedType(elseRes.getType(), iterCount, isInnerOp);
          elseRes.setType(slicedOpType);
        }
        if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(elseOp))
          handleXpuPrintOp(xpuPrintOp);
      }
      for (auto &elseOp : elseRegion.front()) {
        if (!isa<mlir::arith::ConstantOp>(elseOp))
          reshapeConstOperandsToMatchSiblings(&elseOp, /*skipIfInTree=*/false);
      }
      if (auto elseYield = dyn_cast<mlir::scf::YieldOp>(
              elseRegion.front().getTerminator()))
        reshapeConstYieldOperandsToMatchResults(elseYield,
                                                ifOp.getResultTypes());
    }
  }

  // Fix reshape result type after slicing. Only handles the case where reshape
  // flattens to 1D (e.g., tensor<4x64xf32> → tensor<256xf32>).
  //
  // Why special-cased: setSlicedResTy divides each dim by iterCount, but
  // reshape output shape doesn't decompose along the same axes as input.
  // Instead we compute: new_size = product(original_res_shape) / iterCount[0].
  //
  // Note: only 1D result is handled; multi-dim reshape results are unsupported.
  //
  // Example:
  //   Before: reshape tensor<4x64xf32> → tensor<256xf32>, iterCount[0]=4
  //   After:  result type becomes tensor<64xf32> (= 256/4)
  void handleReshapeInLoop(mlir::triton::ReshapeOp reshapeOp) {
    auto reshapeResTy =
        dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
    if (!reshapeResTy)
      return;
    auto reshapeResShape = reshapeResTy.getShape();
    if (reshapeResShape.size() != 1)
      return;
    auto reshapeSrcTy =
        cast<RankedTensorType>(reshapeOp.getOperand().getType());
    auto reshapeResSize = product(reshapeResShape);
    llvm::SmallVector<int64_t> slicedShape(1, 1u);
    slicedShape[0] =
        std::ceil(static_cast<double>(reshapeResSize / iterCount[0]));
    auto slicedReshapeSrcTy = RankedTensorType::get(
        slicedShape, reshapeSrcTy.getElementType(), reshapeResTy.getEncoding());
    reshapeOp.getResult().setType(slicedReshapeSrcTy);
  }

  void handleConvertLayoutInLoop(triton::xpu::ConvertLayoutOp cvtOp) {
    // ConvertLayoutOp has the SameOperandsAndResultShape trait, so its
    // result shape must match its operand shape. After slicing, the operand
    // may already have a reduced shape (e.g., tensor<1xi64> instead of
    // tensor<16xi64>). Derive result shape from operand.
    auto srcType = mlir::dyn_cast<RankedTensorType>(cvtOp.getSrc().getType());
    auto resType =
        mlir::dyn_cast<RankedTensorType>(cvtOp.getResult().getType());
    if (srcType && resType && srcType.getShape() != resType.getShape()) {
      auto newResType = RankedTensorType::get(
          srcType.getShape(), resType.getElementType(), resType.getEncoding());
      cvtOp.getResult().setType(newResType);
    }
  }

  // Fix expand_dims result type after slicing shrinks its operand shape.
  //
  // Why: setSlicedResTy only updates the result type based on iterCount, but
  // expand_dims semantics require result = insert dim-1 at axis into src shape.
  // After slicing reduces the src (e.g., tensor<16xi32> → tensor<1xi32>), the
  // result must be recomputed from the actual sliced src shape, not from the
  // original result shape divided by iterCount (which may be wrong for dim-1 axes).
  //
  // Example IR:
  //   Before slicing: %e = tt.expand_dims %r {axis=0} :
  //                       tensor<16xi32> → tensor<1x16xi32>
  //   After slicing src: %r has type tensor<1xi32>
  //   Correct result:    tensor<1x1xi32> (insert dim-1 at axis=0 into [1])
  //   Wrong (naive):     tensor<1x16xi32> / iterCount would give wrong shape
  void handleExpandDimsInLoop(triton::ExpandDimsOp expandOp) {
    auto srcType =
        mlir::dyn_cast<RankedTensorType>(expandOp.getSrc().getType());
    auto resType =
        mlir::dyn_cast<RankedTensorType>(expandOp.getResult().getType());
    if (!srcType || !resType)
      return;
    auto srcShape = srcType.getShape();
    int axis = expandOp.getAxis();
    llvm::SmallVector<int64_t> newShape(srcShape.begin(), srcShape.end());
    newShape.insert(newShape.begin() + axis, 1);
    auto newResType = RankedTensorType::get(newShape, resType.getElementType(),
                                            resType.getEncoding());
    expandOp.getResult().setType(newResType);
  }

  // Whether `op` has a shape contract that per-dimension slicing would break,
  // and it was handled here.
  //
  // Why: expand_dims derives its result from its source, so the per-dimension
  // getSlicedType() retyping used on a nested block gives it the wrong shape --
  // sparse_semi_structured_mm builds `b0[None, :, :]` from a source that the
  // same walk sliced to 1x32, and the result has to become 1x1x32 rather than
  // stay 1x32x32.
  bool handleShapeContractOp(Operation *op) {
    if (auto expandOp = mlir::dyn_cast<triton::ExpandDimsOp>(op)) {
      handleExpandDimsInLoop(expandOp);
      return true;
    }
    return false;
  }

  void handleDefaultInLoop(Operation *op, bool isInner) {
    setSlicedResTy(op, isInner);
    // Handle constant operands that are not in the sorted op tree. Derive
    // the target shape from a sibling operand to avoid double-slicing when
    // the constant was already processed.
    reshapeConstOperandsToMatchSiblings(op, /*skipIfInTree=*/true);
  }

  // Iterate over the sorted op tree in reverse, hoisting each op into the
  // outer for-loop body and dispatching to the per-op handler above.
  void lower(Operation *yieldOp) {
    SetVector<Operation *> toErase;
    for (auto op : llvm::reverse(sortedOpTree)) {
      bool isInner = inOpChain(innerChain, op);
      if (!isa<scf::YieldOp>(op))
        op->moveBefore(yieldOp);

      if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        handleMakeRangeInLoop(makeRangeOp, isInner);
      } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
        handleReduceInLoop(reduceOp, isInner, toErase);
      } else if (auto scanOp = dyn_cast<triton::ScanOp>(op)) {
        handleScanInLoop(scanOp, isInner, toErase);
      } else if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
        handleConstantInLoop(constOp, isInner);
      } else if (auto forOp = dyn_cast<mlir::scf::ForOp>(op)) {
        handleForInLoop(forOp, isInner);
      } else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
        handleWhileInLoop(whileOp, isInner);
      } else if (auto ifOp = dyn_cast<mlir::scf::IfOp>(op)) {
        handleIfInLoop(ifOp, isInner);
      } else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(op)) {
        handleDefaultInLoop(op, isInner);
        handleWhileInLoop(whileOp);
      } else if (auto reshapeOp = dyn_cast<mlir::triton::ReshapeOp>(op)) {
        handleReshapeInLoop(reshapeOp);
      } else if (auto xpuPrintOp = dyn_cast<XPUPrintOp>(op)) {
        handleXpuPrintOp(xpuPrintOp);
      } else if (auto cvtOp = dyn_cast<triton::xpu::ConvertLayoutOp>(op)) {
        handleConvertLayoutInLoop(cvtOp);
      } else if (auto expandOp = dyn_cast<triton::ExpandDimsOp>(op)) {
        handleExpandDimsInLoop(expandOp);
      } else if (auto transOp = dyn_cast<triton::TransOp>(op)) {
        handleTransInLoop(transOp);
      } else {
        handleDefaultInLoop(op, isInner);
      }
    }
    // Second pass: fix constant operands whose shapes became stale after
    // sibling ops were sliced later in the reverse-topological traversal.
    for (auto op : sortedOpTree) {
      if (toErase.contains(op))
        continue;
      if (isa<mlir::arith::ConstantOp>(op))
        continue;
      reshapeConstOperandsToMatchSiblings(op, /*skipIfInTree=*/false);
    }
    // Settle the shapes that the per-op traversal cannot decide on its own.
    // All of it must stay BEFORE the erasure below: toErase ops are still in
    // sortedOpTree afterwards, and dyn_cast on a freed operation is a
    // use-after-free (the pre-existing pass after the erasure only gets away
    // with it because the pool has not reused the memory yet).
    //
    // 1. Every loop's iter_args: a constant that seeds one is sliced by its own
    //    chain membership, and the traversal can reach it after the loop, so
    //    handleForInLoop()'s earlier alignment saw the pre-slice type and had
    //    nothing to fix. The authority is the body's yield.
    for (auto op : sortedOpTree) {
      if (toErase.contains(op))
        continue;
      if (auto forOp = dyn_cast<mlir::scf::ForOp>(op))
        alignForIterArgsWithResults(forOp, /*useYieldType=*/true);
    }
    // 2. Elementwise / compare / select results, from their now-final operands
    //    -- the loop results above can have just been retyped, which leaves
    //    every consumer of the loop's rank-1 chain at the other width
    //    (histogramdd_col_range's post-loop select/cmpf/subf chain).
    // Run to a fixpoint: sortedOpTree is not in producer-first order, so a
    // single sweep can visit a consumer before the producer whose type it has
    // to follow. Each sweep only ever propagates a shape forward, so loop
    // until a whole sweep changes nothing: the forward-only propagation
    // settles, and no fixed round count can cut a long chain short.
    while (true) {
      bool changed = false;
      for (auto op : sortedOpTree) {
        if (toErase.contains(op))
          continue;
        auto operandTy = resultTypeFromOperands(op);
        auto shapeTy = operandTy ? mlir::dyn_cast<RankedTensorType>(*operandTy)
                                 : RankedTensorType();
        if (!shapeTy)
          continue;
        for (auto res : op->getResults()) {
          auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
          if (!resTy || resTy.getShape() == shapeTy.getShape())
            continue;
          // Take the operand's SHAPE only, and keep the result's own element
          // type and encoding. Copying the operand type wholesale would turn
          // arith.extsi's tensor<32xi32> -> tensor<32xi64> into i32 -> i32, and
          // an operand may legitimately carry a different encoding (an
          // expand_dims source is a slice, its result is not).
          res.setType(RankedTensorType::get(shapeTy.getShape(),
                                            resTy.getElementType(),
                                            resTy.getEncoding()));
          changed = true;
        }
      }
      if (!changed)
        break;
    }
    // 3. Constants against their now-final siblings.
    for (auto op : sortedOpTree) {
      if (toErase.contains(op) || isa<mlir::arith::ConstantOp>(op))
        continue;
      reshapeConstOperandsToMatchSiblings(op, /*skipIfInTree=*/false);
    }
    // Erase replaced ops after all passes over sortedOpTree are done.
    for (auto *op : toErase) {
      op->erase();
    }
    // Last: the consumers that hold the wrong copy of a CSE-shared arange.
    // This runs on the slicing loop's own body, the twin of the call in
    // handleForInLoop() for a nested one.
    alignRank1OperandsToKnownDim(*yieldOp->getBlock());
  }
};

// Encapsulates all per-tree data and operations for the Legalize pass.
// Each OpTree corresponds to one dataflow sink (LM2GM/SM2GM/AtomicRMW) and
// holds the backward op closure, chain partitioning, iteration schedule,
// and reduce group topology needed to generate the slicing loop.
struct OpTree {
  // Block-filtered backward closure (sorted topologically). These ops will
  // be physically moved into the slicing loop.
  llvm::SetVector<Operation *> slicingOps_;

  // Full backward closure (no block filter). Used for analysis: chain
  // classification and tensor type collection.
  llvm::SetVector<Operation *> analysisOps_;

  // Representative tensor types from loads/stores that determine tile shapes.
  llvm::SmallVector<Type> tensorTypes_;

  // Reduce/scan multi-core group topology for this tree.
  size_t reduceNGroup_ = 1;
  size_t reduceGroupSize_ = 64;
  size_t reduceRowsPerCore_ = 1;

  // Per-dimension slicing loop iteration counts (LCM across all tensors).
  llvm::SmallVector<size_t> iterCount_;

  // Inner chain: reduce-axis ops (not sliced by the loop).
  llvm::SetVector<Operation *> innerChain_;
  // Outer chain: iteration-axis ops (sliced by the loop).
  llvm::SetVector<Operation *> outerChain_;

  // Maps a range clone back to the original it was cloned from; see
  // recordRangeClone().
  DenseMap<Operation *, Operation *> rangeCloneToOriginal_;

  // Rank-1 value -> the tile dimension its length indexes (see
  // computeRank1TileDim), or nullopt when the IR does not say.
  //
  // Filled by computeRank1Dims() while every op is still alive, and only READ
  // afterwards. The walk behind it follows def-use edges, and lower() erases
  // ops as it goes, so recomputing it lazily during lowering walked into freed
  // operations: the same tree lowered fine on its own but killed the process
  // when a batch harness compiled several kernels in a row.
  //
  // A miss means "an op created during lowering" -- those keep the chain-based
  // answer, which is what they got before the walk existed too (a fresh op has
  // no users yet, so the walk answered nullopt for it).
  llvm::DenseMap<Value, std::optional<unsigned>> rank1Dims_;

  /// When set, this tree exists only to compute that reduce, and the slicing
  /// loop hands its result back as a loop result.
  ///
  /// Why: the consumer tree needs the *complete* reduction. The reduce already
  /// accumulates across slices through the SM loop result cache, so the value
  /// it produces on the last iteration is the total, and an scf.for yields its
  /// last iteration value as its result -- one iter_arg carrying the reduce
  /// result is enough, with no combine region to clone. See
  /// third_party/xpu/docs/axis0_reduce_broadcast_back.md.
  Operation *carryReduceResult_ = nullptr;

  /// Loop result holding the complete reduction; set by lowerSlicingLoop().
  Value carriedReduceTotal_;

  /// The slicing loop lowerSlicingLoopInsideFor() built, so lowerSlicingLoop()
  /// can split it in two for a broadcast-back reduce.
  mlir::scf::ForOp flippedSliceLoop_;

  void computeRank1Dims() {
    rank1Dims_.clear();
    // Run to a fixpoint. A nullopt can be produced by a cycle that only exists
    // because a dependency had not been resolved yet, and the memo then hands
    // that nullopt to every later query -- which is what left smm's pointer
    // splats at the outer width while the offsets they are added to were at the
    // inner one. A Known answer never becomes nullopt, so dropping the nullopt
    // entries and recomputing converges.
    // Loop until a sweep gains no new Known answer. That monotone gain bounds
    // the loop; a count taken here would read the zero that rank1Dims_.clear()
    // leaves above and stop after one round, leaving the smm and orgqr rank-1
    // cycle unresolved.
    while (true) {
      llvm::SmallVector<Value> unknowns;
      for (auto &entry : rank1Dims_)
        if (!entry.second)
          unknowns.push_back(entry.first);
      for (Value v : unknowns)
        rank1Dims_.erase(v);

      llvm::SmallPtrSet<void *, 32> inProgress;
      bool gained = false;
      // slicingOps_ plus everything nested inside it, not slicingOps_ alone:
      // that set is filtered to the root op's own block, so a value computed
      // inside a nested loop body is missing from it and falls back to chain
      // membership -- orgqr's `splat %k_idx` lives in the reduction loop's body
      // and came out at the inner width while the accumulator it is added to
      // was at the outer one. analysisOps_ is too wide: it also covers values
      // whose structural dimension disagrees with the chain answer that smm
      // depends on.
      llvm::SetVector<Operation *> dimOps(slicingOps_);
      for (Operation *op : slicingOps_)
        op->walk([&](Operation *inner) { dimOps.insert(inner); });
      for (Operation *op : dimOps)
        for (Value res : op->getResults()) {
          auto resTy = mlir::dyn_cast<RankedTensorType>(res.getType());
          if (!resTy || resTy.getShape().size() != 1)
            continue;
          if (rank1Dims_.count(res))
            continue; // already Known
          if (computeRank1TileDim(res, rank1Dims_, inProgress))
            gained = true;
        }
      if (!gained)
        break;
    }
  }

  std::optional<unsigned> tileDimOfRank1(Value value) const {
    auto it = rank1Dims_.find(value);
    return it == rank1Dims_.end() ? std::nullopt : it->second;
  }

  //=== Static factory ===//

  static std::optional<OpTree>
  build(Operation *endOp, llvm::SetVector<Operation *> &visitedSlicingOps,
        llvm::SetVector<Operation *> &visitedAnalysisOps,
        llvm::SetVector<Operation *> &storeOps,
        const llvm::SmallPtrSetImpl<mlir::Value> *stopValues = nullptr);

  //=== Per-tree methods ===//

  bool computeReduceGroupInfo(unsigned passGroupsPerCluster, unsigned coreNum);

  /// Reduces of this tree that the peel may take into their own loop.
  ///
  /// How:
  ///   - keep the ones whose result is broadcast back onto the reduced tile;
  ///   - drop the ones whose result the peel cannot carry out, see
  ///     canPeelReduceIntoOwnLoop().
  llvm::SmallVector<mlir::triton::ReduceOp> collectPeelableReduces() {
    llvm::SmallVector<mlir::triton::ReduceOp> peelable;
    for (Operation *op : analysisOps_) {
      auto reduceOp = dyn_cast<mlir::triton::ReduceOp>(op);
      if (!reduceOp || !isBroadcastBackOntoTile(reduceOp))
        continue;
      if (!canPeelReduceIntoOwnLoop(reduceOp))
        continue;
      peelable.push_back(reduceOp);
    }
    return peelable;
  }

  /// The reduce this tree can be split around, with its chain, or null.
  ///
  /// Why: splitting means running the reduce chain in its own loop first, so
  /// the consumer sees the complete reduction instead of the running prefix
  /// (see third_party/xpu/docs/axis0_reduce_broadcast_back.md). The first
  /// chain must hold no other reduce, so stacked reduces are peeled one per
  /// round rather than split at once.
  mlir::triton::ReduceOp
  findSplittableReduce(llvm::SmallVectorImpl<Operation *> &chain,
                       const llvm::SmallPtrSetImpl<mlir::Value> *carried =
                           nullptr) {
    // Take the earliest flagged reduce whose own chain holds no other reduce.
    // logsumexp stacks two of them -- the sum consumes the tile that the max
    // was subtracted from -- and splitting the inner one first, then asking
    // again, peels them one at a time: once the max has its own loop, the sum
    // chain reads the carried total and no longer contains a reduce.
    llvm::SmallVector<mlir::triton::ReduceOp> candidates =
        collectPeelableReduces();
    for (mlir::triton::ReduceOp candidate : candidates) {
      chain.clear();
      if (!collectReduceChainInBlock(candidate, chain, carried))
        continue;
      bool nested = llvm::any_of(chain, [&](Operation *op) {
        return op != candidate.getOperation() &&
               isa<mlir::triton::ReduceOp>(op);
      });
      if (!nested)
        return candidate;
    }
    return {};
  }

  /// Reject a tree whose axis-0 reduce result is fed back onto the tile it
  /// reduced.
  ///
  /// Why: the slicing loop lowers that shape wrongly and silently -- the reduce
  /// result is only the running prefix until the last slice and the store is
  /// guarded to the last iteration, so `x - tl.sum(x, axis=0)` writes one slice
  /// and leaves the rest at whatever the output held. Three real operators are
  /// wrong because of it (orgqr, logsumexp at dim 0, linalg_householder_product)
  /// and nothing told their callers. A compile error is worse to look at and
  /// better to have than wrong numbers.
  ///
  /// TODO(@yanxin11): drop this once lowerSlicingLoop() splits the loop in two
  /// (accumulate, then consume); the plan and its constraints are in
  /// third_party/xpu/docs/axis0_reduce_broadcast_back.md.
  bool rejectBroadcastBackOntoTile() {
    // Opt-in: rejecting the shape also stops Legalize before the widening and
    // chain-splitting work, which xpu_legalize_axis0_orgqr_splat_both_axes.mlir
    // uses this very shape to pin. Turning it on by default would blind that
    // test to the three fixes it guards, so the switch is left to whoever is
    // shipping - loud failure for users, lowering for the test suite.
    static const bool reject =
        ::getenv("TRITONXPU_REJECT_REDUCE_BROADCAST") != nullptr;
    if (!reject)
      return false;
    bool rejected = false;
    for (Operation *op : analysisOps_) {
      auto reduceOp = dyn_cast<mlir::triton::ReduceOp>(op);
      if (!reduceOp || !isBroadcastBackOntoTile(reduceOp))
        continue;
      reduceOp->emitError(
          "axis-0 reduce result is broadcast back onto the tile it reduced; "
          "the slicing loop cannot lower this yet (it would store only the "
          "last slice). See third_party/xpu/docs/"
          "axis0_reduce_broadcast_back.md");
      rejected = true;
    }
    return rejected;
  }

  LogicalResult computeIterCount(unsigned bufferSize, unsigned coreNum,
                                 unsigned passGroupsPerCluster,
                                 bool hasGm2lmFeedingReduce,
                                 int32_t maxGm2lmLrie, bool allowRank3);
  void partitionChains();
  void splitSharedMakeRanges();
  void splitSharedChainValues();
  void lowerSlicingLoop();
  LogicalResult lowerSlicingLoopInsideFor(mlir::scf::ForOp hostFor);

  /// One `for (...) { acc = acc + tl.sum(tile, axis=0) }` accumulator.
  struct FoldedAccumulator {
    mlir::scf::ForOp forOp;
    unsigned argIdx;
    mlir::Operation *updateOp;
  };
  void
  collectFoldedAccumulators(llvm::SmallVectorImpl<FoldedAccumulator> &accs);
  bool isPerIterationReduce(mlir::Operation *op);
  mlir::triton::ReduceOp findFlipBroadcastReduce();
  llvm::SetVector<mlir::Operation *> collectRootOpsToMove();
  bool growRootOpsToMove(llvm::SetVector<mlir::Operation *> &moving);

  unsigned countReduceOps() const;
  unsigned countScanOps() const;

private:
  // Helpers for splitSharedMakeRanges
  static bool isUsedByInnerAxisExpand(Operation *user);
  bool isSharedAcrossChains(Operation *op);
  void cloneSharedValueForOuter(Operation *op,
                                DenseMap<Operation *, Operation *> &clones,
                                bool addToSlicingOps = true);
  void swapRangeInChains(triton::MakeRangeOp rangeOp,
                         triton::MakeRangeOp clonedRangeOp);
  void recordRangeClone(Operation *cloned, Operation *original);
  void cloneDominatingRanges();
  mlir::Value cloneRangeIntoTree(mlir::Value range, mlir::Block *treeBlock,
                                 DenseMap<mlir::Value, mlir::Value> &clones);
  void cloneRangeWithExtsi(triton::MakeRangeOp rangeOp,
                           triton::MakeRangeOp clonedRangeOp,
                           arith::ExtSIOp extsiOp);
  void cloneRangeDirectUser(triton::MakeRangeOp rangeOp,
                            triton::MakeRangeOp clonedRangeOp, Operation *user);
};

//===----------------------------------------------------------------------===//
// OpTree implementation
//===----------------------------------------------------------------------===//

std::optional<OpTree>
OpTree::build(Operation *endOp, llvm::SetVector<Operation *> &visitedSlicingOps,
              llvm::SetVector<Operation *> &visitedAnalysisOps,
              llvm::SetVector<Operation *> &storeOps,
              const llvm::SmallPtrSetImpl<mlir::Value> *stopValues) {
  if (visitedSlicingOps.contains(endOp) || inSameSCFIfBlock(storeOps, endOp))
    return std::nullopt;
  storeOps.insert(endOp);

  OpTree tree;

  auto slicingRootOp = findRootOp(endOp);
  auto slicingRootBlock = slicingRootOp->getBlock();
  llvm::SetVector<Operation *> rawSlicing;
  getOpTreeBwd(rawSlicing, visitedSlicingOps, slicingRootOp, slicingRootBlock,
               stopValues);
  tree.slicingOps_ = sortOpTreeBwd(rawSlicing);
  tree.cloneDominatingRanges();

  getOpTreeBwd(tree.analysisOps_, visitedAnalysisOps, endOp, stopValues);

  for (auto op : tree.analysisOps_) {
    if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[tensorTypes emplace_back]-loadResType: "
                              << loadOp.getResult().getType() << "\n");
      tree.tensorTypes_.emplace_back(loadOp.getResult().getType());
    } else if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[tensorTypes emplace_back]-storeValType: "
                              << storeOp.getValue().getType() << "\n");
      tree.tensorTypes_.emplace_back(storeOp.getValue().getType());
    } else if (auto storeOp = dyn_cast<triton::xpu::LM2GMMaskOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "[tensorTypes emplace_back]-storeValType: "
                              << storeOp.getValue().getType() << "\n");
      tree.tensorTypes_.emplace_back(storeOp.getValue().getType());
    }
  }

  return tree;
}

bool OpTree::computeReduceGroupInfo(unsigned passGroupsPerCluster,
                                    unsigned coreNum) {
  if (passGroupsPerCluster > 1) {
    reduceNGroup_ = passGroupsPerCluster;
    if (coreNum % reduceNGroup_ != 0) {
      if (!slicingOps_.empty())
        (*slicingOps_.begin())
            ->emitError()
            << "groups_per_cluster must divide core_num, but got "
            << reduceNGroup_ << " and " << coreNum;
      return false;
    }
    reduceGroupSize_ = ceil<size_t>(coreNum, reduceNGroup_);
    return true;
  }

  // Infer from ClusterLayoutAttr on reduce ops in slicingOps_.
  bool ok = true;
  bool isFirst = true;
  auto getGroupInfoFromLayout = [&](triton::xpu::ClusterLayoutAttr &encoding,
                                    Operation *anchor) {
    int _ngroup = product(encoding.getGroupsPerCluster());
    int _groupsize = product(encoding.getCoresPerGroup());
    int _rowspercore = encoding.getSizePerCore()[0];
    if (isFirst) {
      reduceNGroup_ = _ngroup;
      reduceGroupSize_ = _groupsize;
      reduceRowsPerCore_ = _rowspercore;
      isFirst = false;
      return;
    }
    // One tree carries one cluster geometry. Two reduces that disagree leave
    // the slicer without a loop shape, so report the second one instead of
    // aborting -- the assert that used to live here was the only report, and
    // an NDEBUG build silently kept the first geometry.
    if (reduceNGroup_ != static_cast<size_t>(_ngroup) ||
        reduceGroupSize_ != static_cast<size_t>(_groupsize) ||
        reduceRowsPerCore_ != static_cast<size_t>(_rowspercore)) {
      if (anchor)
        anchor->emitError()
            << "reduce cluster geometry is not consistent: expected "
               "ngroup/groupsize/rowspercore "
            << reduceNGroup_ << "/" << reduceGroupSize_ << "/"
            << reduceRowsPerCore_ << ", got " << _ngroup << "/" << _groupsize
            << "/" << _rowspercore;
      ok = false;
    }
  };

  for (auto op : slicingOps_) {
    op->walk([&](triton::ReduceOp reduceOp) {
      auto tensorType =
          mlir::dyn_cast<RankedTensorType>(reduceOp.getOperandTypes()[0]);
      if (!tensorType)
        return;
      if (auto reshapeOp =
              reduceOp.getSrcs()[0].getDefiningOp<triton::ReshapeOp>()) {
        auto reshapeResTy =
            dyn_cast<RankedTensorType>(reshapeOp.getResult().getType());
        if (!reshapeResTy || reshapeResTy.getShape().size() != 1)
          return;
        tensorType = cast<RankedTensorType>(reshapeOp.getOperand().getType());
      }
      // Only a 2-D reduce describes the tile's cluster geometry. A rank-1
      // reduce's sizePerCore[0] is the whole vector's length, not a per-core
      // row count, so letting it vote makes the topology "inconsistent" as soon
      // as one tree holds both kinds -- histogramdd_col_range has four 2-D
      // axis-0 reduces (sizePerCore=[1,32]) and one rank-1 one (sizePerCore=
      // [32]), which tripped the rowspercore assert. A tree whose reduces are
      // all rank-1 leaves isFirst set and falls through to the 2-D inference
      // below, exactly as before.
      if (tensorType.getShape().size() != 2)
        return;
      if (auto globalEncoding = mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
              tensorType.getEncoding()))
        getGroupInfoFromLayout(globalEncoding, reduceOp);
    });
  }

  if (!isFirst)
    return ok;

  int64_t maxNumElements = 0;
  triton::xpu::ClusterLayoutAttr maxEncoding;
  auto updateMax2DTensorEncoding = [&](Type type) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
    // getGroupInfo() below only ever reads front() and back(), so a rank-3
    // tensor is usable here; keeping the old rank-2 filter would leave a pure
    // rank-3 tree with no geometry at all.
    if (!tensorType || tensorType.getShape().size() < 2)
      return;
    auto globalEncoding = mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(
        tensorType.getEncoding());
    if (!globalEncoding)
      return;
    auto shape = tensorType.getShape();
    if (shape[0] <= 0 || shape[1] <= 0)
      return;
    int64_t numElements = shape[0] * shape[1];
    if (numElements <= maxNumElements)
      return;
    maxNumElements = numElements;
    maxEncoding = globalEncoding;
  };

  for (auto op : slicingOps_) {
    op->walk([&](Operation *op) {
      if (isa<triton::xpu::ConvertLayoutOp, triton::ExpandDimsOp,
              triton::xpu::BroadcastOp>(op))
        return;
      for (auto result : op->getResults())
        updateMax2DTensorEncoding(result.getType());
      for (auto operand : op->getOperands())
        updateMax2DTensorEncoding(operand.getType());
    });
  }
  if (maxEncoding)
    getGroupInfoFromLayout(
        maxEncoding, slicingOps_.empty() ? nullptr : *slicingOps_.begin());
  return ok;
}

// Reject the load/store tensor shapes the slicer cannot express.
//
// Why: getSlicedShape() only understands rank<=2 tensors, and a 2D tensor whose
// inner dim exceeds the local-memory budget cannot be sliced either. Both used
// to be llvm_unreachable: a process abort, and in an NDEBUG build a silent
// mis-slice.
//
// How:
//   - reject rank > 2;
//   - reject 2D with shape[1] > coreNum * bufferSize.
//
// \param anchor op the diagnostic is reported on (the tree's dataflow sink).
// Reject a rank > 2 tensor, or say out loud that accepting it is pass-local.
//
// Why: `allow-rank3` is a test switch, not a capability. Stating that only in
// the option description lets a pipeline turn it on and get silently wrong
// results, because CoreTiling, Vectorize and TritonXPUToLLVM still assume
// rank <= 2 (the open points are listed in docs/rank3_handoff.md sections
// 2-4). A warning puts it in the build log instead, where a pipeline that
// enabled the switch by accident can see it. It stays a warning rather than an
// error because the switch exists precisely so the Legalize-only lit cases can
// run.
//
// \param warned set once per tree so a wide module does not print one line per
//        tensor.
static LogicalResult checkRank3Allowed(Operation *anchor, size_t rank,
                                       bool allowRank3, bool &warned) {
  if (rank <= 2)
    return success();
  if (!allowRank3)
    return anchor->emitError(
        "3D Shape Unsupported. (pass --tritonxpu-legalize=allow-rank3 to let "
        "Legalize slice rank-3 IR; the downstream passes still assume "
        "rank <= 2)");
  if (!warned) {
    anchor->emitWarning()
        << "allow-rank3 accepted a rank-" << rank
        << " tensor: CoreTiling, Vectorize and TritonXPUToLLVM still assume "
           "rank <= 2, so nothing past tritonxpu-legalize is verified for it "
           "(see third_party/xpu/docs/rank3_handoff.md)";
    warned = true;
  }
  return success();
}

static LogicalResult validateTensorShapes(Operation *anchor,
                                          ArrayRef<Type> tensorTypes,
                                          unsigned coreNum,
                                          unsigned bufferSize,
                                          bool allowRank3) {
  bool warnedRank3 = false;
  for (Type type : tensorTypes) {
    auto tensorType = mlir::dyn_cast<RankedTensorType>(type);
    if (!tensorType)
      continue;
    auto shape = tensorType.getShape();
    if (failed(checkRank3Allowed(anchor, shape.size(), allowRank3,
                                 warnedRank3)))
      return failure();
    if (shape.size() == 2 && shape[1] > coreNum * bufferSize)
      return anchor->emitError("2D Shape[-1] <= core_num * buffer_size Limit. "
                               "Please Adjust COL_BLOCK_SIZE");
  }
  return success();
}

LogicalResult OpTree::computeIterCount(unsigned bufferSize, unsigned coreNum,
                                       unsigned passGroupsPerCluster,
                                       bool hasGm2lmFeedingReduce,
                                       int32_t maxGm2lmLrie, bool allowRank3) {
  // Decline a tree that carries no load or store: it has no shaped tensor to
  // tile, so there is no iteration schedule to compute.
  //
  // Why: tensorTypes_ is seeded only from LoadOp/LM2GMOp (build()), so the
  // accumulate tree a split reduce clones from a reduce whose input tile was
  // produced by a *prior* loop -- weight_norm's `tl.sum` over a loop-carried
  // accumulator -- holds the reduce but no load/store and leaves tensorTypes_
  // empty. getLCM() below reads .back() of a per-dim list sized by
  // tensorTypes_, so an empty one aborts the whole compiler (Issue #2860).
  // Returning failure turns that crash into the caller's existing fallback:
  // the reduce-split peel drops the clone and keeps the reduce whole, and a
  // top-level tree reports a clean `failed to legalize`.
  //
  // TODO(@yanxin11): lift this once the slicer can tile an accumulate loop
  // whose source tile is a prior loop's carried result; see
  // third_party/xpu/docs/reduce_source_prior_loop_empty_schedule.md.
  if (tensorTypes_.empty())
    return failure();

  // Whether the whole tile maps onto a single core, which is what the atomic
  // simulation path needs. A tensor with no cluster geometry cannot be mapped
  // at all, so it answers false -- getSizePerCluster() rejects it with a
  // diagnostic further down.
  auto isMapToSingleCore = [](RankedTensorType tensorTy) {
    auto clusterEncoding =
        mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
    if (!clusterEncoding)
      return false;
    auto coresPerGroup = clusterEncoding.getCoresPerGroup();
    auto groupsPerCluster = clusterEncoding.getGroupsPerCluster();
    return llvm::all_of(coresPerGroup, [](unsigned n) { return n == 1; }) &&
           llvm::all_of(groupsPerCluster, [](unsigned n) { return n == 1; });
  };

  auto isAtomicSimMode = [&](size_t &totalElements) -> bool {
    bool found = false;
    for (auto tensorTy : tensorTypes_) {
      auto rankedTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy);
      if (!rankedTensorTy)
        continue;
      if (!isMapToSingleCore(rankedTensorTy))
        return false;
      found = true;
      totalElements = product(rankedTensorTy.getShape());
    }
    return found;
  };

  bool unrollOpt = false;
  size_t totalElements = 0;
  bool isAtomicSim = isAtomicSimMode(totalElements);

  bool isReduceMultiGroup =
      passGroupsPerCluster > 1 ? true : (reduceNGroup_ > 1);

  // Compute per-dim iteration counts (LCM across all tensors in this tree).
  size_t tensorDim = 1;
  for (auto &type : tensorTypes_) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(type))
      tensorDim = std::max(tensorDim, tensorType.getShape().size());
  }

  llvm::SmallVector<llvm::SmallVector<size_t>> dimIterCounts(
      tensorDim, llvm::SmallVector<size_t>(tensorTypes_.size(), 1u));

  // build() seeds analysisOps_ with the dataflow sink, so front() is the sink
  // and gives the diagnostic a kernel location instead of an unknown one.
  if (failed(validateTensorShapes(analysisOps_.front(), tensorTypes_, coreNum,
                                  bufferSize, allowRank3)))
    return failure();

  for (size_t i = 0; i < tensorTypes_.size(); ++i) {
    if (auto tensorType = mlir::dyn_cast<RankedTensorType>(tensorTypes_[i])) {
      auto spaceSize =
          getSizePerCluster(analysisOps_.front(), tensorTypes_[i], unrollOpt,
                            bufferSize);
      if (failed(spaceSize))
        return failure();
      auto tensorShape = tensorType.getShape();
      llvm::SmallVector<int64_t> slicedShape =
          getSlicedShape(tensorShape, *spaceSize, isReduceMultiGroup);
      for (size_t j = 0; j < tensorShape.size(); ++j) {
        dimIterCounts[j][i] =
            std::ceil(static_cast<double>(tensorShape[j]) / slicedShape[j]);
      }
    }
  }

  iterCount_.resize(tensorDim, 1u);
  for (size_t j = 0; j < dimIterCounts.size(); ++j)
    iterCount_[j] = getLCM(dimIterCounts[j]);

  if (isAtomicSim && hasGm2lmFeedingReduce)
    isAtomicSim = false;
  if (isAtomicSim)
    iterCount_.assign(iterCount_.size(),
                     mlir::ceil<size_t>(totalElements, maxGm2lmLrie));
  return success();
}

// Partition analysisOps_ into innerChain_ and outerChain_.
//
// innerChain_: ops on the reduce axis (axis=0). Collected by tracing backward
//   from ExpandDims(axis=0) and Broadcast(dim0 changes), and forward from
//   Reduce/Scan(axis=0). These ops are NOT sliced by the outer iteration loop.
//
// outerChain_: only records tt.make_range ops that feed ExpandDims(axis=1),
//   i.e., row-index ranges. Used by handleMakeRangeInLoop to emit
//   triton_xpu.out_range instead of triton_xpu.make_range.
//
// All other ops are implicitly outer (sliced by the loop). The downstream
// SliceLoopCtx checks `inOpChain(innerChain_, op)` to decide per-op handling.
void OpTree::partitionChains() {
  for (auto op : analysisOps_) {
    // MakeRange feeding ExpandDims(axis=1) → row index → outer chain.
    if (auto rangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
      auto expandDimsOps = findAllTypeUserOps<triton::ExpandDimsOp>(rangeOp);
      for (auto e : expandDimsOps) {
        auto expandDimOp = cast<triton::ExpandDimsOp>(e);
        if (expandDimOp.getAxis() == 1)
          outerChain_.insert(rangeOp);
      }
    }

    // ExpandDims(axis=0) → column index → trace backward into inner chain.
    if (auto expandDimOp = dyn_cast<triton::ExpandDimsOp>(op)) {
      auto srcTy =
          mlir::dyn_cast<RankedTensorType>(expandDimOp.getSrc().getType());
      auto resTy =
          mlir::dyn_cast<RankedTensorType>(expandDimOp.getResult().getType());
      if (srcTy && resTy && expandDimOp.getAxis() == 0) {
        getOpChainBwd(innerChain_, expandDimOp);
        innerChain_.remove(expandDimOp);
      }
    }

    // Broadcast with different dim0 shape → reduce-axis broadcast →
    // trace backward into inner chain.
    if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(op)) {
      auto srcTy =
          mlir::dyn_cast<RankedTensorType>(broadcastOp.getSrc().getType());
      auto resTy =
          mlir::dyn_cast<RankedTensorType>(broadcastOp.getResult().getType());
      if (srcTy && resTy && srcTy.getShape()[0] != resTy.getShape()[0]) {
        getOpChainBwd(innerChain_, broadcastOp);
        innerChain_.remove(broadcastOp);
      }
    }

    // Reduce/Scan along axis=0 → trace forward into inner chain.
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      if (reduceOp.getAxis() == 0)
        getOpChainFwd(innerChain_, reduceOp);
    }

    if (auto scanOp = dyn_cast<triton::ScanOp>(op)) {
      if (scanOp.getAxis() == 0)
        getOpChainFwd(innerChain_, scanOp);
    }
  }

}

// Whether every ExpandDimsOp reachable from `user` sits on the inner axis.
//
// Why: findUserOp() returns the first match in DFS order, so a value that
// reaches both axes would be classified by traversal order alone. The value is
// only inner-owned when nothing reachable from it expands on the outer axis.
bool OpTree::isUsedByInnerAxisExpand(Operation *user) {
  auto expandDimsOps = findAllTypeUserOps<triton::ExpandDimsOp>(user);
  if (expandDimsOps.empty())
    return false;
  return llvm::all_of(expandDimsOps, [](Operation *op) {
    return cast<triton::ExpandDimsOp>(op).getAxis() == 0;
  });
}

void OpTree::swapRangeInChains(triton::MakeRangeOp rangeOp,
                               triton::MakeRangeOp clonedRangeOp) {
  innerChain_.insert(clonedRangeOp);
  innerChain_.remove(rangeOp);
  slicingOps_.insert(clonedRangeOp);
  recordRangeClone(clonedRangeOp, rangeOp);
}

// Remember which original a range clone came from.
//
// Why: cloneSharedValueForOuter() clones a value for the outer chain by copying
// its operands, but by then splitSharedMakeRanges() has already rewired the
// original onto the *inner* range clone. Without this map the outer copy would
// inherit the inner range and end up with operands of the inner width --
// fractional_max_pool3d's arith.cmpi: result sliced to 1, operands left at 4.
void OpTree::recordRangeClone(Operation *cloned, Operation *original) {
  if (cloned && original)
    rangeCloneToOriginal_[cloned] = original;
}

/// Whether `operand` is a tt.make_range defined outside `treeBlock`.
///
/// Why: getOpTreeBwd() stops at the tree's own block, so a range the frontend
/// hoisted into a dominating block never joins the tree.
static bool isRangeFromOtherBlock(mlir::Value operand, mlir::Block *treeBlock) {
  auto rangeOp = operand.getDefiningOp<triton::MakeRangeOp>();
  return rangeOp && rangeOp->getBlock() != treeBlock;
}

/// Give the tree its own copy of every range it reads from another block.
///
/// Why: a producer outside the tree's block is never visited by lower(), so it
/// keeps its declared width while the ops around it are sliced.
/// top_k_per_row_decode hoists `lane = tl.arange(0, BLOCK_SIZE)` above the
/// `if row_len <= TOPK` branch, which leaves the range in the predecessor block
/// and the stores that index with it in the branch: the first
/// `arith.cmpi slt, %lane, %take_row` then compares a 512-wide operand with the
/// 1-wide splat the slicing loop produced and fails its verifier.
///
/// How:
///   - take the operands of every tree op and keep the ranges from another
///     block;
///   - clone the range at the start of the tree's block -- it has no operands,
///     so no dominance to preserve -- and rewire only this tree's use, leaving
///     whatever else reads the original at its own width;
///   - re-sort the tree so the clone is visited before its users.
///
/// TODO(@claude): only tt.make_range is pulled across. A splat or an index
/// computation hoisted the same way would need its operands cloned too, which
/// is the shape of splitSharedChainValues() rather than of this walk.
void OpTree::cloneDominatingRanges() {
  if (slicingOps_.empty())
    return;
  mlir::Block *treeBlock = (*slicingOps_.begin())->getBlock();
  DenseMap<mlir::Value, mlir::Value> clones;
  llvm::SmallVector<Operation *> users(slicingOps_.begin(), slicingOps_.end());
  for (Operation *user : users)
    for (mlir::OpOperand &use : user->getOpOperands())
      if (isRangeFromOtherBlock(use.get(), treeBlock))
        use.set(cloneRangeIntoTree(use.get(), treeBlock, clones));
  if (!clones.empty())
    slicingOps_ = sortOpTreeBwd(slicingOps_);
}

/// The tree-local copy of `range`, created on first use.
mlir::Value
OpTree::cloneRangeIntoTree(mlir::Value range, mlir::Block *treeBlock,
                           DenseMap<mlir::Value, mlir::Value> &clones) {
  auto known = clones.find(range);
  if (known != clones.end())
    return known->second;
  mlir::OpBuilder builder(treeBlock, treeBlock->begin());
  Operation *clone = builder.clone(*range.getDefiningOp());
  slicingOps_.insert(clone);
  clones[range] = clone->getResult(0);
  return clone->getResult(0);
}

void OpTree::cloneRangeWithExtsi(triton::MakeRangeOp rangeOp,
                                 triton::MakeRangeOp clonedRangeOp,
                                 arith::ExtSIOp extsiOp) {
  OpBuilder builder(rangeOp);
  builder.setInsertionPointAfter(clonedRangeOp);
  auto newExtsiOp = builder.create<arith::ExtSIOp>(
      builder.getUnknownLoc(), extsiOp.getType(), clonedRangeOp);

  SmallVector<Operation *> extsiUsers(extsiOp.getResult().getUsers().begin(),
                                      extsiOp.getResult().getUsers().end());
  for (Operation *user : extsiUsers) {
    if (inOpChain(innerChain_, user))
      user->replaceUsesOfWith(extsiOp.getResult(), newExtsiOp.getResult());
  }

  swapRangeInChains(rangeOp, clonedRangeOp);
  recordRangeClone(newExtsiOp, extsiOp);
  innerChain_.insert(newExtsiOp);
  innerChain_.remove(extsiOp);
  slicingOps_.insert(newExtsiOp);
}

void OpTree::cloneRangeDirectUser(triton::MakeRangeOp rangeOp,
                                  triton::MakeRangeOp clonedRangeOp,
                                  Operation *user) {
  user->replaceUsesOfWith(rangeOp.getResult(), clonedRangeOp.getResult());
  swapRangeInChains(rangeOp, clonedRangeOp);
}

// Split a tt.make_range that appears in BOTH outerChain_ and innerChain_.
// Such a range feeds expand_dims on two axes simultaneously, but downstream
// the outer copy becomes triton_xpu.out_range while the inner copy becomes
// triton_xpu.make_range — they cannot share the same SSA value.
//
// This pass clones an inner-side copy and rewires inner-chain users to it.
//
// Example IR (before):
//   %r = tt.make_range {start=0, end=32} : tensor<32xi32>
//   %outer = tt.expand_dims %r {axis=1} : tensor<32x1xi32>  // outer use
//   %inner = tt.expand_dims %r {axis=0} : tensor<1x32xi32>  // inner use
//
// Example IR (after):
//   %r       = tt.make_range {start=0, end=32}  // stays in outerChain_
//   %r_inner = tt.make_range {start=0, end=32}  // cloned into innerChain_
//   %outer = tt.expand_dims %r       {axis=1}
//   %inner = tt.expand_dims %r_inner {axis=0}
//
void OpTree::splitSharedMakeRanges() {
  for (Operation *outerOp : outerChain_) {
    auto rangeOp = dyn_cast<triton::MakeRangeOp>(outerOp);
    if (!rangeOp)
      continue;
    if (!inOpChain(innerChain_, outerOp))
      continue;

    OpBuilder builder(rangeOp);
    triton::MakeRangeOp clonedInnerRange;
    SetVector<Operation *> rangeUsers(rangeOp->getUsers().begin(),
                                      rangeOp->getUsers().end());
    for (Operation *user : rangeUsers) {
      // The chain membership says whether the user belongs to this tree's
      // inner chain, but NOT which axis it consumes the range as: an axis-0
      // reduce makes the whole producer side inner, including the dim-0 index
      // vectors, so membership alone rewires both kinds of user onto the inner
      // clone and the dim-0 one keeps its full length. When the IR says which
      // dimension the user widens into, that is the answer; with no such
      // evidence fall back to membership, which is what the trees the slicer
      // already supports rely on.
      bool userIsInner = inOpChain(innerChain_, user);
      if (iterCount_.size() > 1 && user->getNumResults() == 1) {
        if (auto userResTy = mlir::dyn_cast<RankedTensorType>(
                user->getResult(0).getType());
            userResTy && userResTy.getShape().size() == 1)
          if (auto dim = tileDimOfRank1(user->getResult(0)))
            userIsInner = (*dim == 1);
      }
      if (!userIsInner)
        continue;
      if (!clonedInnerRange)
        clonedInnerRange = cast<triton::MakeRangeOp>(builder.clone(*rangeOp));
      if (auto extsiOp = dyn_cast<arith::ExtSIOp>(user))
        cloneRangeWithExtsi(rangeOp, clonedInnerRange, extsiOp);
      else
        cloneRangeDirectUser(rangeOp, clonedInnerRange, user);
      slicingOps_ = sortOpTreeBwd(slicingOps_);
    }
  }
}

// Whether `op` is a 1-D value that both chains slice at once.
//
// Why: SliceLoopCtx::lower() picks ONE chain membership per op
// (`isInner = inOpChain(innerChain, op)`) and getSlicedType() slices a 1-D value
// by iterCount[1] when it is inner, by iterCount[0] otherwise. A value that
// innerChain_ owns but that also feeds an op outside the chain is therefore
// sliced at one width while the other consumers expect the other, and the two
// meet at a cmpi/andi/addi that requires equal operand types.
//
// How:
//   - the value must be inner-owned, single-result and rank 1;
//   - and at least one user must sit outside innerChain_.
bool OpTree::isSharedAcrossChains(Operation *op) {
  if (!inOpChain(innerChain_, op))
    return false;
  // Only a value can be split, never a region. Cloning an scf.for/scf.if here
  // would duplicate the whole loop (its body included) and leave two copies for
  // lower() to rewrite independently -- which it did, once the loop op itself
  // started landing in innerChain_ (see partitionChains).
  if (op->getNumRegions() != 0)
    return false;
  if (op->getNumResults() != 1)
    return false;
  auto resTy = mlir::dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!resTy || resTy.getShape().size() != 1)
    return false;

  // The clone exists to give the outer consumers the OTHER width, so it is only
  // needed when they really want one. When this value's tile dimension is
  // structurally known AND every consumer outside the inner chain resolves to
  // that same dimension on its own, one width serves everybody and the clone
  // would only add a second copy sliced at iterCount[0].
  //
  // _col_sum_kernel is the first case: the column index %offs_n_2 is dim 1, and
  // its outer consumers are the store's own cmpi/addptr, which take dim 1 from
  // %offs_n_2 through the sibling step -- the clone kept width 1 while the index
  // vector stayed at 32. xpu_legalize_loop_invariant_arange is the second: the
  // loop-invariant arange is dim 1 through expand_dims(axis=0), but its outer
  // consumer is the scf.for itself, whose iter_arg width comes from the loop's
  // own chain membership and does not resolve to dim 1 -- so there the split is
  // load-bearing.
  if (iterCount_.size() > 1) {
    // A value that the widening walk finds on BOTH axes has two consumers that
    // want two widths -- which is exactly what a split is for. The
    // user-membership rule below cannot see it: orgqr's `splat %N` is reached
    // through `rows * N` (dim 0) and through `cols < N` (dim 1), yet the chain
    // closure classifies BOTH of those users inner, so the rule answers "no
    // outer user" and leaves one width for two dimensions.
    unsigned widenedDim = 0;
    if (widenedTileDimOfRank1(op->getResult(0), widenedDim) ==
        WidenedDim::Conflict)
      return true;
    if (auto dim = tileDimOfRank1(op->getResult(0))) {
      // Does this consumer need the value at a width other than `dim`?
      //   - a region boundary (scf.for / scf.if) takes its iter_arg width from
      //     its own chain membership, not from what it is combined with, so it
      //     always wants its own copy;
      //   - a plain rank-1 op takes the width of whatever it is combined with,
      //     so it wants another width only if it is combined with a rank-1
      //     value of a different, known dimension.
      auto wantsOtherWidth = [&](Operation *user) {
        if (isa<mlir::scf::ForOp, mlir::scf::IfOp>(user))
          return true;
        // An expand_dims names the dimension it widens into; it wants another
        // width only when that dimension differs from this value's own.
        if (auto userDim = expandDimUserTileDim(user))
          return *userDim != *dim;
        if (user->getNumResults() != 1)
          return true;
        auto userTy =
            mlir::dyn_cast<RankedTensorType>(user->getResult(0).getType());
        if (!userTy || userTy.getShape().size() != 1)
          return true;
        for (Value operand : user->getOperands()) {
          if (operand == op->getResult(0))
            continue;
          auto operandTy = mlir::dyn_cast<RankedTensorType>(operand.getType());
          if (!operandTy || operandTy.getShape().size() != 1)
            continue;
          unsigned operandDim = 0;
          if (widenedTileDimOfRank1(operand, operandDim) == WidenedDim::Known &&
              operandDim != *dim)
            return true;
        }
        return false;
      };
      bool outerWantsAnotherWidth = false;
      for (Operation *user : op->getUsers()) {
        if (inOpChain(innerChain_, user))
          continue;
        if (wantsOtherWidth(user)) {
          outerWantsAnotherWidth = true;
          break;
        }
      }
      if (!outerWantsAnotherWidth)
        return false;
    }
  }

  for (Operation *user : op->getUsers())
    if (!inOpChain(innerChain_, user))
      return true;
  return false;
}

// Clone a chain-shared value for the outer chain and rewire its outer users.
//
// Why: the two chains need the value at two different widths, so they cannot
// share one SSA value. The clone is deliberately left out of innerChain_, which
// is exactly what makes lower() slice it by iterCount[0].
//
// How:
//   - resolve an operand to its own outer clone when it has one, so a chain of
//     shared values splits as a unit;
//   - clone at the original's position and rewire the outside users;
//   - record the clone in `clones` and in slicingOps_ so lower() visits it.
void OpTree::cloneSharedValueForOuter(
    Operation *op, DenseMap<Operation *, Operation *> &clones,
    bool addToSlicingOps) {
  OpBuilder builder(op);
  IRMapping mapping;
  for (Value operand : op->getOperands()) {
    Operation *producer = operand.getDefiningOp();
    auto it = clones.find(producer);
    if (it != clones.end()) {
      mapping.map(operand, it->second->getResult(0));
      continue;
    }
    // The producer may be a range clone made for the inner chain; the outer
    // copy has to read the original instead, or it would be sliced twice.
    auto original = rangeCloneToOriginal_.find(producer);
    if (original != rangeCloneToOriginal_.end())
      mapping.map(operand, original->second->getResult(0));
  }
  Operation *clone = builder.clone(*op, mapping);
  clones[op] = clone;

  // Rewire the consumers that need the value at the OTHER width. Chain
  // membership alone is not enough: a value widened into both dimensions can
  // have every consumer classified inner and still need two widths -- orgqr's
  // `splat %N` is consumed by `cols < N` (dim 1) and by `rows * N` (dim 0), and
  // both of those ops sit in the inner chain.
  const unsigned selfDim = inOpChain(innerChain_, op) ? 1u : 0u;
  for (Operation *user : llvm::make_early_inc_range(op->getUsers())) {
    // Which side of the split this user needs.
    //
    // Why: splitSharedMakeRanges() takes the same decision from the frozen
    // rank-1 table, and chain membership alone would undo it. A user whose
    // result is a dim-1 value must keep the inner copy even when its chain
    // membership says outer -- linalg_tensorinv rewires `cols = j0 + arange`
    // onto the inner range there, and rewiring it back here left the addi with
    // a 1-wide operand next to its 32-wide result. The mirror case matters too:
    // a dim-0 user must take the outer copy even when it is classified inner.
    // A region boundary is different: its width comes from the yielded value,
    // not from the chain (xpu_legalize_loop_invariant_arange pins the for
    // result at dim 1 while the init operand must take the outer copy), so it
    // keeps the membership rule.
    //
    // How: read the user result dimension from the frozen table when it is
    // there, and fall back to chain membership otherwise.
    bool rewire = !inOpChain(innerChain_, user);
    if (user->getNumResults() == 1 && user->getNumRegions() == 0) {
      auto userTy =
          mlir::dyn_cast<RankedTensorType>(user->getResult(0).getType());
      if (userTy && userTy.getShape().size() == 1)
        if (auto userDim = tileDimOfRank1(user->getResult(0)))
          rewire = (*userDim != selfDim);
    }
    // An expand_dims result is rank 2, so the refinement above skips it; its
    // axis names the dimension directly, so a column index (axis 0) keeps the
    // inner copy instead of being sent to the sliced outer one.
    if (auto userDim = expandDimUserTileDim(user))
      rewire = (*userDim != selfDim);
    if (rewire)
      user->replaceUsesOfWith(op->getResult(0), clone->getResult(0));
  }
  // A clone made for a value in a nested block stays out of slicingOps_:
  // that set is filtered to the root op block and sortOpTreeBwd() rejects a
  // set spanning two regions. The handleForInLoop body walk retypes it instead.
  if (addToSlicingOps)
    slicingOps_.insert(clone);
  recordRangeClone(clone, op);
}

// Split every 1-D value that both chains would slice at once.
//
// Why: splitSharedMakeRanges() above only covers tt.make_range. Every other
// shared 1-D producer -- a tt.splat of n_rows, an index addi, ... -- hits the
// same mismatch. corrcoef shares its n_rows splat, not its range.
//
// How:
//   - walk slicingOps_ producers-first so an operand is split before its user;
//   - values in a nested block are not in that set, so clone them in place
//     instead of adding them: adding one would put two regions in one set,
//     which sortOpTreeBwd() rejects;
//   - clone each shared value for the outer chain;
//   - re-sort slicingOps_ once at the end.
//
// NOTE: the sibling shapes of this family that are a region boundary rather
// than a shared value are handled where the boundary is:
//   - handleIfInLoop() re-creates a constant yielded out of an scf.if at the
//     if's sliced result shape;
//   - handleForInLoop() takes each iter_arg's type from its init operand.
// See third_party/xpu/docs/legalize_cluster_failure_modes.md.
void OpTree::splitSharedChainValues() {
  DenseMap<Operation *, Operation *> clones;
  llvm::SetVector<Operation *> ordered = sortOpTreeBwd(slicingOps_);
  bool grewSlicingOps = false;
  for (Operation *op : llvm::reverse(ordered))
    if (isSharedAcrossChains(op)) {
      cloneSharedValueForOuter(op, clones);
      grewSlicingOps = true;
    }

  // Nested blocks. slicingOps_ is filtered to the root op block, so a shared
  // value computed inside a loop body is never visited by the loop above. It
  // then keeps its declared width while the consumers that resolve to the
  // other tile dimension are sliced, and the first of those fails its
  // verifier: the conv2d `splat %s` in the kw loop and the
  // sparse_semi_structured_mm `offs_k * 4 + k_offset * 4` in the k_offset loop
  // both show up as a convert_layout with a 32-wide operand and a 1-wide
  // result. Clone such a value in place rather than moving it into
  // slicingOps_, which sortOpTreeBwd() would reject as spanning two regions.
  //
  // Block order is enough here: walk() is pre-order, so inside a block a
  // producer is visited before its user, and every value this branch rewires
  // shares a block with the consumers it splits.
  llvm::SetVector<Operation *> nested;
  for (Operation *root : slicingOps_)
    root->walk([&](Operation *inner) {
      if (inner != root && !slicingOps_.contains(inner))
        nested.insert(inner);
    });
  for (Operation *op : nested)
    if (isSharedAcrossChains(op))
      cloneSharedValueForOuter(op, clones, /*addToSlicingOps=*/false);

  if (grewSlicingOps)
    slicingOps_ = sortOpTreeBwd(slicingOps_);
}

/// Whether `init` can be reproduced ahead of the slicing loop.
///
/// Why: moving a folded accumulator onto the slicing loop means seeding the
/// slicing loop with this value. A constant (or any value that already
/// dominates the loop) is fine; an op with operands could be moved inside by
/// lower() and then cannot be cloned back out, so such an accumulator is left
/// alone rather than carried half-way.
static bool isReproducibleInit(mlir::Value init) {
  if (mlir::Operation *initDef = init.getDefiningOp())
    return initDef->getNumOperands() == 0;
  return true;
}

/// Whether `reduceOp` combines with the same operation `update` does.
///
/// Why: the reduce result can only stand in for the accumulated value when the
/// two accumulate the same way -- a max-reduce folded into an add is a
/// different computation.
static bool hasMatchingCombine(mlir::triton::ReduceOp reduceOp,
                               mlir::Operation *update) {
  auto combineRet = dyn_cast_or_null<mlir::triton::ReduceReturnOp>(
      reduceOp.getCombineOp().front().getTerminator());
  if (!combineRet || combineRet.getNumOperands() != 1)
    return false;
  mlir::Operation *inner = combineRet.getOperand(0).getDefiningOp();
  return inner && inner->getName() == update->getName();
}

/// The reduce feeding `update`, looking past everything but the accumulator.
static mlir::triton::ReduceOp
findReduceBehindUpdate(mlir::Operation *update, mlir::Value acc) {
  for (mlir::Value operand : update->getOperands()) {
    if (operand == acc)
      continue;
    if (auto reduceOp = findDefOpBwd<mlir::triton::ReduceOp>(operand))
      return reduceOp;
  }
  return {};
}

/// Match iter_arg `argIdx` of `forOp` as `acc = acc + tl.sum(tile, axis=0)`.
static std::optional<OpTree::FoldedAccumulator>
matchFoldedAccumulator(mlir::scf::ForOp forOp, mlir::scf::YieldOp yield,
                       unsigned argIdx) {
  mlir::Operation *update = yield.getOperand(argIdx).getDefiningOp();
  if (!update || !isa<mlir::arith::AddFOp, mlir::arith::AddIOp>(update))
    return std::nullopt;
  auto reduceOp =
      findReduceBehindUpdate(update, forOp.getRegionIterArg(argIdx));
  if (!reduceOp)
    return std::nullopt;
  if (!isReproducibleInit(forOp.getInitArgs()[argIdx]))
    return std::nullopt;
  if (!hasMatchingCombine(reduceOp, update))
    return std::nullopt;
  return OpTree::FoldedAccumulator{forOp, argIdx, update};
}

/// Whether `reduceOp`'s result goes through the shared-memory loop cache.
///
/// Why: only that path accumulates across slicing-loop iterations, so only it
/// can be mis-accumulated by the nesting. The rule has to match
/// isNeedLoopCacheResult() in ReduceOpToLLVM.cpp: axis 1 never caches, and of
/// the axis-0 shapes only a rank-1 result (or a scalar one) does.
static bool isLoopCachedResult(mlir::triton::ReduceOp reduceOp) {
  if (reduceOp.getAxis() != 0)
    return false;
  auto resTy =
      mlir::dyn_cast<RankedTensorType>(reduceOp.getResult()[0].getType());
  return !resTy || resTy.getRank() == 1;
}

/// Collect this tree's folded accumulators.
///
/// Why: `for (...) { acc = acc + tl.sum(tile, axis=0) }` with the store after
/// the loop has to move its accumulation onto the slicing loop, because the
/// slicing loop replaces the source loop's trip count. Left inside, every
/// slice restarts it from the loop's init: a masked tail slice stores that
/// init value, and a multi-chunk loop adds the reduce's running result to a
/// per-slice accumulator (double count).
///
/// How:
///   - take every scf.for in the closure that carries iter_args;
///   - keep the args whose yielded value is an add fed by a reduce;
///   - require an init that is reproducible before the slicing loop, and a
///     combine op that matches the add.
void OpTree::collectFoldedAccumulators(
    llvm::SmallVectorImpl<FoldedAccumulator> &foldedAccs) {
  for (auto op : slicingOps_) {
    auto forOp = dyn_cast<mlir::scf::ForOp>(op);
    if (!forOp || forOp.getNumRegionIterArgs() == 0)
      continue;
    auto yield = dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
    if (!yield)
      continue;
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i)
      if (auto acc = matchFoldedAccumulator(forOp, yield, i))
        foldedAccs.push_back(*acc);
  }
}

/// Whether a store inside the loop observes `reduceOp`'s total.
///
/// Why: a reduce that only feeds its loop's carry does not need the total
/// within the iteration. `acc = acc + tl.sum(...)` is moved onto the slicing
/// loop by collectFoldedAccumulators(), and an idempotent carry (min, max)
/// tolerates the per-slice prefix because the last slice's prefix already is
/// the total -- histogramdd_col_range carries four of those. A store is the
/// opposite case: it writes whatever the total is at that point of the
/// iteration, so the accumulation has to be finished by then.
///
/// How:
///   - walk the users forward through everything;
///   - stop at an scf.yield: that is carry, not observation;
///   - answer yes at a store sink (the ops collectEndOps() collects).
static bool isReduceObservedInIteration(mlir::triton::ReduceOp reduceOp) {
  llvm::SetVector<mlir::Operation *> seen;
  llvm::SmallVector<mlir::Operation *> work{reduceOp.getOperation()};
  while (!work.empty()) {
    mlir::Operation *op = work.pop_back_val();
    if (!seen.insert(op))
      continue;
    for (mlir::Operation *user : op->getUsers()) {
      if (isa<mlir::scf::YieldOp>(user))
        continue;
      if (isa<triton::xpu::LM2GMOp, triton::xpu::LM2GMMaskOp,
              triton::xpu::SM2GMOp, triton::AtomicRMWOp>(user))
        return true;
      work.push_back(user);
    }
  }
  return false;
}

/// Whether any operand of `op` is produced by an op in `moving`.
static bool isFedByMovingOp(mlir::Operation *op,
                            const llvm::SetVector<mlir::Operation *> &moving) {
  return llvm::any_of(op->getOperands(), [&](mlir::Value v) {
    mlir::Operation *def = v.getDefiningOp();
    return def && moving.contains(def);
  });
}

/// Add the root-block closure ops fed by `moving` to it.
///
/// \returns whether anything was added, so the caller can iterate to a
/// fixpoint (a chain expand_dims -> muli -> broadcast takes one round each).
bool OpTree::growRootOpsToMove(llvm::SetVector<Operation *> &moving) {
  bool grew = false;
  for (auto op : slicingOps_) {
    if (moving.contains(op) || isa<mlir::scf::ForOp>(op))
      continue;
    if (isFedByMovingOp(op, moving))
      grew |= moving.insert(op);
  }
  return grew;
}

/// The root-block closure ops that have to move into the host loop body.
///
/// Why: a make range becomes slice-index dependent inside the slicing loop --
/// handleMakeRangeInLoop() replaces it with an out_range of that index -- so
/// everything computed from it is only valid in there. Leaving a consumer in
/// the root block would have it read a value defined inside the loop, which is
/// how the flip used to be blocked for `for k: tl.sum(x, axis=0)`, whose whole
/// index computation sits ahead of the source loop.
///
/// How:
///   - seed with the closure's make ranges;
///   - add the closure ops that consume one, to a fixpoint;
///   - hand them back in program order, which the move preserves.
llvm::SetVector<mlir::Operation *> OpTree::collectRootOpsToMove() {
  llvm::SetVector<mlir::Operation *> moving;
  for (auto op : slicingOps_)
    if (isa<triton::MakeRangeOp>(op))
      moving.insert(op);
  while (growRootOpsToMove(moving))
    ;
  llvm::SmallVector<mlir::Operation *> sorted(moving.begin(), moving.end());
  llvm::sort(sorted, [](mlir::Operation *a, mlir::Operation *b) {
    return a->isBeforeInBlock(b);
  });
  return {sorted.begin(), sorted.end()};
}

/// Whether `op` is a reduce whose complete result is consumed inside the
/// source loop that holds it.
///
/// Why: the default lowering wraps the slicing loop AROUND that source loop,
/// so the reduce runs once per (slice, source iteration) while its
/// shared-memory accumulator is keyed by the slice index alone. Every slice
/// but the first is then added once per source iteration: `for k: tl.sum(x,
/// axis=0)` over 8 slices with K=2 yields 2*sum - x[0] instead of sum (orgqr
/// and linalg_householder_product are wrong for this reason). Nesting the
/// slicing loop inside the source loop restarts the accumulator per source
/// iteration, because the slice index is then the inner one.
///
/// How:
///   - require a source loop around the reduce, and the loop cache path;
///   - leave a single-slice tree alone: the nesting cannot be observed;
///   - require a store inside the loop to observe the total: a reduce that
///     only feeds the loop's carry is handled elsewhere, see
///     isReduceObservedInIteration().
bool OpTree::isPerIterationReduce(mlir::Operation *op) {
  auto reduceOp = dyn_cast<mlir::triton::ReduceOp>(op);
  if (!reduceOp || !op->getParentOfType<mlir::scf::ForOp>())
    return false;
  if (iterCount_.empty() || iterCount_[0] <= 1)
    return false;
  if (!isLoopCachedResult(reduceOp))
    return false;
  return isReduceObservedInIteration(reduceOp);
}

/// Build a zero constant matching a ranked tensor type.
static mlir::Value makeZeroTensor(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::RankedTensorType ty) {
  mlir::Type elemTy = ty.getElementType();
  mlir::TypedAttr zero;
  if (elemTy.isIntOrIndex())
    zero = mlir::DenseElementsAttr::get(
        ty, mlir::APInt(elemTy.getIntOrFloatBitWidth(), 0));
  else
    zero = mlir::DenseElementsAttr::get(
        ty, mlir::APFloat::getZero(
                mlir::cast<mlir::FloatType>(elemTy).getFloatSemantics()));
  return b.create<mlir::arith::ConstantOp>(loc, ty, zero);
}

/// Whether `op` transitively consumes `red`'s result.
static bool opConsumesReduce(mlir::Operation *op, mlir::Operation *red) {
  llvm::SmallPtrSet<mlir::Operation *, 32> seen;
  llvm::SmallVector<mlir::Operation *> work{op};
  while (!work.empty()) {
    mlir::Operation *cur = work.pop_back_val();
    if (!seen.insert(cur).second)
      continue;
    for (mlir::Value v : cur->getOperands())
      if (mlir::Operation *def = v.getDefiningOp()) {
        if (def == red)
          return true;
        work.push_back(def);
      }
  }
  return false;
}

/// Split a flipped slicing loop whose reduce is broadcast back onto its tile.
///
/// Why: after the flip nests the slicing loop inside the source loop, the
/// reduce accumulates across slices through the SM loop cache, but the consumer
/// in the same slice reads the running prefix -- only the last slice holds the
/// complete sum (orgqr / linalg_householder_product, both still wrong after the
/// flip alone). An axis-0 broadcast-back needs two passes over the slices:
/// accumulate the total, then consume it. See
/// third_party/xpu/docs/axis0_reduce_broadcast_back.md.
///
/// How:
///   - clone the slicing-loop body (minus the stores) into an accumulate loop
///     placed just ahead of it, carrying the reduce result out as a loop
///     result: the last slice's value is the complete reduction;
///   - rewrite the original (consumer) loop to read that total in place of its
///     own reduce, then erase the reduce -- its now-dead inputs and the dead
///     consumer clones in the accumulate loop are removed by later DCE.
/// Whether `op` writes a tile back to global or shared memory.
static bool isStoreLike(mlir::Operation *op) {
  return isa<mlir::triton::xpu::StoreOp, mlir::triton::xpu::LM2GMOp,
             mlir::triton::xpu::LM2GMMaskOp, mlir::triton::xpu::SM2GMOp>(op);
}

/// Lift the stores that ran before the host loop back out of the nested
/// slicing loop.
///
/// Why: the flip puts the slicing loop INSIDE the source scf.for so the slice
/// index restarts per source iteration, and SliceLoopCtx::lower() moves every
/// op of the tree in -- including the stores that sat in FRONT of that loop.
/// orgqr and linalg_householder_product initialise Q to the identity there, so
/// running it per source iteration reset the tile the loop accumulates into:
/// every iteration read the identity back and only the last reflector survived.
/// Q came out orthonormal (a single Householder reflector is) but wrong, with
/// only column 0 correct, which is why an orthonormality check passed it.
///
/// How:
///   - take the stores recorded before the move as living outside `hostFor`;
///   - clone the slicing loop in front of `hostFor` carrying only those stores
///     and the feeders they need;
///   - erase them from the in-host copy and let DCE drop what they fed;
///   - bail out when a feeder depends on the host induction variable, which
///     cannot be evaluated before the loop.
static void hoistPreHostStores(mlir::scf::ForOp sliceFor,
                               mlir::scf::ForOp hostFor,
                               llvm::ArrayRef<mlir::Operation *> preStores) {
  llvm::SmallVector<mlir::Operation *> hoisted;
  for (mlir::Operation *op : preStores)
    if (sliceFor->isProperAncestor(op))
      hoisted.push_back(op);
  if (hoisted.empty())
    return;

  // Backward closure, split by the block each feeder sits in: the ones inside
  // the slicing loop are cloned into the new loop, the ones in the host body
  // in front of it.
  mlir::Block *sliceBody = sliceFor.getBody();
  mlir::Block *hostBody = hostFor.getBody();
  llvm::SmallPtrSet<mlir::Operation *, 32> keepSlice, keepHost;
  llvm::SmallVector<mlir::Operation *> work(hoisted.begin(), hoisted.end());
  while (!work.empty()) {
    mlir::Operation *cur = work.pop_back_val();
    if (cur->getBlock() == sliceBody) {
      if (!keepSlice.insert(cur).second)
        continue;
    } else if (cur->getBlock() == hostBody) {
      if (!keepHost.insert(cur).second)
        continue;
    } else {
      continue; // already dominates hostFor
    }
    for (mlir::Value operand : cur->getOperands()) {
      if (operand == hostFor.getInductionVar())
        return; // not computable ahead of the host loop
      if (mlir::Operation *def = operand.getDefiningOp())
        work.push_back(def);
    }
  }
  for (mlir::BlockArgument arg : hostBody->getArguments())
    for (mlir::Operation *user : arg.getUsers())
      if (keepSlice.contains(user) || keepHost.contains(user))
        return; // depends on an iter_arg

  auto upperBound = mlir::getConstantIntValue(sliceFor.getUpperBound());
  if (!upperBound)
    return;

  mlir::Location loc = sliceFor.getLoc();
  mlir::OpBuilder builder(hostFor);
  mlir::IRMapping map;
  for (mlir::Operation &op : *hostBody)
    if (keepHost.contains(&op))
      builder.clone(op, map);

  auto low = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto upper =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(*upperBound));
  auto step = builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
  auto initFor = builder.create<mlir::scf::ForOp>(loc, low, upper, step);
  builder.setInsertionPointToStart(initFor.getBody());
  map.map(sliceFor.getInductionVar(), initFor.getInductionVar());
  for (mlir::Operation &op : sliceBody->without_terminator())
    if (keepSlice.contains(&op))
      builder.clone(op, map);

  for (mlir::Operation *op : hoisted)
    op->erase();
}

static void splitFlippedLoopForBroadcastBack(mlir::scf::ForOp sliceFor,
                                             mlir::triton::xpu::ReduceOp red) {
  // Only the store-inside form is handled: an escaping iter_arg would have to
  // be carried through both loops, which this rewrite does not do.
  if (sliceFor.getNumRegionIterArgs() != 0)
    return;
  auto rty = mlir::dyn_cast<mlir::RankedTensorType>(red.getResult()[0].getType());
  if (!rty)
    return;

  mlir::Location loc = sliceFor.getLoc();
  mlir::OpBuilder b(sliceFor);
  mlir::Value zero = makeZeroTensor(b, loc, rty);
  auto loopA = b.create<mlir::scf::ForOp>(
      loc, sliceFor.getLowerBound(), sliceFor.getUpperBound(),
      sliceFor.getStep(), mlir::ValueRange{zero});

  // Clone the body into the accumulate loop, dropping the stores. The induction
  // variable is remapped; everything else the body reads is defined before
  // sliceFor and dominates loopA.
  mlir::IRMapping map;
  map.map(sliceFor.getInductionVar(), loopA.getInductionVar());
  b.setInsertionPointToStart(loopA.getBody());
  mlir::Operation *clonedRed = nullptr;
  for (mlir::Operation &op : sliceFor.getBody()->without_terminator()) {
    bool isStore =
        isa<mlir::triton::xpu::LM2GMOp, mlir::triton::xpu::LM2GMMaskOp,
            mlir::triton::xpu::SM2GMOp, mlir::triton::xpu::StoreOp>(op);
    // Drop only the stores that write the reduce's broadcast-back result; a
    // reduce-independent store (an init like householder's Q = I) has to stay,
    // or the accumulate loop's own load reads the buffer before it is written.
    if (isStore && opConsumesReduce(&op, red.getOperation()))
      continue;
    mlir::Operation *c = b.clone(op, map);
    if (&op == red.getOperation())
      clonedRed = c;
  }
  if (!clonedRed) {
    loopA.erase();
    return;
  }

  // Carry the reduce result out; its last-slice value is the total.
  b.setInsertionPointToEnd(loopA.getBody());
  b.create<mlir::scf::YieldOp>(loc, clonedRed->getResult(0));
  mlir::Type cty = clonedRed->getResult(0).getType();
  loopA.getRegionIterArgs()[0].setType(cty);
  loopA.getResult(0).setType(cty);

  // The consumer loop now reads the complete reduction; drop its reduce.
  red.getResult()[0].replaceAllUsesWith(loopA.getResult(0));
  red.erase();
}


/// The broadcast-back reduce that will go through the flip, or null.
///
/// Why: lowerSlicingLoop() has to decide BEFORE lowering whether the flipped
/// loop will need the two-pass split; after lowering the original tt.reduce is
/// gone. Only a reduce that is both broadcast back onto its tile and consumed
/// inside its source loop needs it (orgqr / linalg_householder_product).
mlir::triton::ReduceOp OpTree::findFlipBroadcastReduce() {
  for (mlir::Operation *op : analysisOps_)
    if (auto r = dyn_cast<mlir::triton::ReduceOp>(op))
      if (isBroadcastBackOntoTile(r) && isPerIterationReduce(r))
        return r;
  return {};
}

void OpTree::lowerSlicingLoop() {
  // Flip detection: when the tree's scan -- or a reduce whose total is needed
  // per source iteration -- executes inside a host scf.for that is itself a
  // slicing op (e.g. chunked cumsum, orgqr's k loop), the default lowering
  // would wrap the slicing loop AROUND the host loop: every slice iteration
  // would re-run the whole host loop and re-zero its carry iter_args, and the
  // reduce would add each slice once per host iteration. Route to the inside
  // variant, which nests the slicing loop within the host loop.
  bool needsFlip = false;
  bool flip = true;
  mlir::scf::ForOp hostFor;
  for (auto op : analysisOps_) {
    if (!isa<triton::ScanOp>(op) && !isPerIterationReduce(op))
      continue;
    needsFlip = true;
    mlir::scf::ForOp ancFor;
    for (Operation *anc = op->getParentOp(); anc && !isa<triton::FuncOp>(anc);
         anc = anc->getParentOp()) {
      auto forAnc = dyn_cast<mlir::scf::ForOp>(anc);
      if (forAnc && slicingOps_.contains(forAnc)) {
        ancFor = forAnc;
        break;
      }
    }
    if (!ancFor) {
      flip = false;
      break;
    }
    if (hostFor && hostFor != ancFor) {
      flip = false;
      break;
    }
    hostFor = ancFor;
  }
  mlir::triton::ReduceOp flipBcastReduce = findFlipBroadcastReduce();
  // TRITONXPU_TRACE_FLIP reports why a tree did or did not get the flip. The
  // decision is spread over four early returns below, and a kernel that comes
  // out numerically wrong gives no clue which one fired.
  if (::getenv("TRITONXPU_TRACE_FLIP"))
    llvm::errs() << "[flip] needsFlip=" << needsFlip << " flip=" << flip
                 << " bcast=" << (flipBcastReduce != nullptr) << "\n";
  if (needsFlip && flip && succeeded(lowerSlicingLoopInsideFor(hostFor))) {
    // A broadcast-back reduce still reads the per-slice prefix inside the
    // flipped loop; split it into accumulate + consume.
    // TRITONXPU_DISABLE_BCAST_SPLIT leaves the single flipped loop in place, so
    // the red->green of the split can be shown without rebuilding.
    if (flipBcastReduce && flippedSliceLoop_ &&
        !::getenv("TRITONXPU_DISABLE_BCAST_SPLIT")) {
      mlir::triton::xpu::ReduceOp loweredRed;
      flippedSliceLoop_.walk([&](mlir::triton::xpu::ReduceOp r) {
        if (!loweredRed)
          loweredRed = r;
      });
      if (loweredRed)
        splitFlippedLoopForBroadcastBack(flippedSliceLoop_, loweredRed);
    }
    return;
  }

  size_t outIterCount = iterCount_[0];
  auto endOp = slicingOps_[0];
  OpBuilder builder(endOp);
  auto loc = endOp->getLoc();

  llvm::SmallVector<FoldedAccumulator> foldedAccs;
  collectFoldedAccumulators(foldedAccs);

  auto low =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto upper = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIndexAttr(outIterCount));
  auto step =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
  llvm::SmallVector<mlir::Value> foldedInits;
  for (auto &acc : foldedAccs)
    foldedInits.push_back(acc.forOp.getInitArgs()[acc.argIdx]);
  // A carried reduce result needs its iter_arg from the start: scf.for cannot
  // grow one later. The type is fixed up below, once lower() has sliced the
  // reduce.
  unsigned carryIdx = foldedInits.size();
  if (carryReduceResult_) {
    Type carryTy = carryReduceResult_->getResult(0).getType();
    foldedInits.push_back(builder.create<mlir::arith::ConstantOp>(
        loc, carryTy, cast<mlir::TypedAttr>(builder.getZeroAttr(carryTy))));
  }
  auto forLoopOp =
      builder.create<scf::ForOp>(loc, low, upper, step, foldedInits);
  if (!foldedInits.empty()) {
    // A for with iter_args is built without a terminator; seed the yield with
    // the carried values so the body is well formed before lower() runs.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(forLoopOp.getBody());
    builder.create<scf::YieldOp>(loc, forLoopOp.getRegionIterArgs());
  }
  builder.setInsertionPointToStart(forLoopOp.getBody());

  Value idx = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getI32Type(), forLoopOp.getInductionVar());
  Operation *yieldOp = forLoopOp.getBody()->getTerminator();

  SliceLoopCtx ctx{slicingOps_,
                   iterCount_,
                   outIterCount,
                   innerChain_,
                   outerChain_,
                   rank1Dims_,
                   reduceGroupSize_,
                   reduceRowsPerCore_,
                   builder,
                   idx,
                   upper};
  ctx.lower(yieldOp);

  // Hand the complete reduction back as the loop result. lower() has replaced
  // the tt.reduce with the XPU op and sliced its type, so read both off the op
  // that is live now.
  if (carryReduceResult_) {
    Operation *liveReduce = nullptr;
    for (Operation &op : *forLoopOp.getBody())
      if (isa<triton::xpu::ReduceOp>(op))
        liveReduce = &op;
    if (liveReduce) {
      Value carried = liveReduce->getResult(0);
      forLoopOp.getRegionIterArg(carryIdx).setType(carried.getType());
      forLoopOp.getResult(carryIdx).setType(carried.getType());
      yieldOp->setOperand(carryIdx, carried);
      carriedReduceTotal_ = forLoopOp.getResult(carryIdx);
    }
  }

  // Move each folded accumulator onto the slicing loop: the source loop starts
  // from the running value instead of its own init, and its update collapses to
  // the reduce result (which already carries the total across slices).
  for (auto [n, acc] : llvm::enumerate(foldedAccs)) {
    mlir::Value iterArg = acc.forOp.getRegionIterArg(acc.argIdx);
    mlir::Value reduced;
    for (mlir::Value operand : acc.updateOp->getOperands())
      if (operand != iterArg)
        reduced = operand;
    if (!reduced)
      continue;
    mlir::Type carriedTy = iterArg.getType();
    forLoopOp.getRegionIterArg(n).setType(carriedTy);
    forLoopOp.getResult(n).setType(carriedTy);
    // The slicing loop must start from the source loop's init. lower() moves
    // the op that defines it (the zero splat) into the loop body, so materialise
    // an equivalent before the loop first; otherwise the init stops dominating
    // the loop.
    mlir::Value carriedInit = acc.forOp.getInitArgs()[acc.argIdx];
    if (mlir::Operation *def = carriedInit.getDefiningOp()) {
      if (forLoopOp->isProperAncestor(def)) {
        OpBuilder initBuilder(forLoopOp);
        carriedInit = initBuilder.clone(*def)->getResult(0);
      }
    }
    forLoopOp.getInitArgsMutable()[n].assign(carriedInit);
    acc.forOp.getInitArgsMutable()[acc.argIdx].assign(
        forLoopOp.getRegionIterArg(n));
    // The carried accumulator now owns the cross-slice total, so the reduce
    // must hand back its own partial -- the update keeps its `acc + reduce`
    // form, it is the carry that was missing. Keeping the SM loop result cache
    // as well would add the running total to the carried value (and the cache
    // cannot tell two invocations inside slice 0 apart). loopNum stays the
    // slicing loop trip count: StoreControl still needs it for the store guard.
    // rewriteReduceInRegion() has turned the tt.reduce into the XPU op by now.
    if (auto xpuReduce =
            findDefOpBwd<mlir::triton::xpu::ReduceOp>(reduced))
      xpuReduce->setAttr("loopCache",
                         mlir::BoolAttr::get(xpuReduce.getContext(), false));
    yieldOp->setOperand(n, acc.forOp.getResult(acc.argIdx));
  }
}

/// The host-body ops that read the slicing loop, directly or through others.
///
/// Why: hoistFeedersAboveSliceLoop() must not move such an op above the loop,
/// and the escape rewrite must not hand one of them a value it then feeds back
/// into the loop -- both produce a use before its definition.
///
/// How: walk the body in program order, so a producer is classified before its
/// consumers and one pass closes the set.
static llvm::SetVector<mlir::Operation *>
collectSliceLoopConsumers(mlir::Block *hostBody, mlir::scf::ForOp sliceFor) {
  llvm::SetVector<mlir::Operation *> consumers;
  for (mlir::Operation &op : *hostBody) {
    if (&op == sliceFor.getOperation() || sliceFor->isAncestor(&op))
      continue;
    bool reads = llvm::any_of(op.getOperands(), [&](mlir::Value v) {
      mlir::Operation *def = v.getDefiningOp();
      return def && (def == sliceFor.getOperation() || consumers.contains(def));
    });
    if (reads)
      consumers.insert(&op);
  }
  return consumers;
}

/// Whether `op` feeds anything in `movedSet`, directly or not.
///
/// Why: an escape value's external user that feeds the moved closure back
/// would read the slicing loop's result from inside that same loop. orgqr is
/// this shape -- `q -= tau_k * v[:, None] * dots[None, :]` consumes the
/// complete reduction the loop is still accumulating -- and it needs the loop
/// split in two (see axis0_reduce_broadcast_back.md), not just a reorder.
static bool feedsMovedSet(mlir::Operation *op,
                          const llvm::SetVector<mlir::Operation *> &movedSet) {
  llvm::SetVector<mlir::Operation *> seen;
  llvm::SmallVector<mlir::Operation *> work{op};
  while (!work.empty()) {
    mlir::Operation *cur = work.pop_back_val();
    if (!seen.insert(cur))
      continue;
    for (mlir::Operation *user : cur->getUsers()) {
      for (mlir::Operation *anc = user; anc; anc = anc->getParentOp())
        if (movedSet.contains(anc))
          return true;
      work.push_back(user);
    }
  }
  return false;
}

/// Move the host-body ops the slicing loop reads above the loop.
///
/// Why: the slicing loop is built at the top of the host body, so every body
/// op that stayed outside the closure now sits BELOW it. A closure op reading
/// one of those would use a value defined later in the parent region, which
/// the verifier rejects ("operand #0 does not dominate this use") -- orgqr's
/// `tau_k * v[:, None] * dots[None, :]` reads a scalar load the tree does not
/// claim. Hoisting is sound because none of them depends on the slice index;
/// they are host-loop invariant or derived from the host induction variable.
///
/// How:
///   - walk the body backwards, so a consumer is seen before its producer;
///   - take an op whose users are in the loop or already hoisted;
///   - re-insert them above the loop in program order (the walk collected the
///     reverse).
static void hoistFeedersAboveSliceLoop(mlir::Block *hostBody,
                                       mlir::scf::ForOp sliceFor) {
  llvm::SetVector<mlir::Operation *> consumers =
      collectSliceLoopConsumers(hostBody, sliceFor);
  llvm::SetVector<mlir::Operation *> hoist;
  for (mlir::Operation &op : llvm::reverse(*hostBody)) {
    if (sliceFor->isAncestor(&op) || isa<mlir::scf::YieldOp>(op))
      continue;
    // Reading the loop and feeding it cannot both be satisfied by a position.
    if (consumers.contains(&op))
      continue;
    bool feeds = llvm::any_of(op.getUsers(), [&](mlir::Operation *user) {
      return sliceFor->isAncestor(user) || hoist.contains(user);
    });
    if (feeds)
      hoist.insert(&op);
  }
  for (mlir::Operation *op : llvm::reverse(hoist))
    op->moveBefore(sliceFor);
}

// Variant of lowerSlicingLoop for trees whose scan -- or whose per-iteration
// reduce (see isPerIterationReduce) -- executes inside a host scf.for. The
// host loop walks data chunks and carries the running state in iter_args;
// wrapping the slicing loop around it would re-run every chunk per slice and
// re-zero the carry each time, and would leave the reduce accumulating one
// slice once per host iteration. Instead the slicing loop is placed at
// the top of the host loop body, so each host iteration executes the slices
// of one chunk: scan's SM carry and reduce's loop cache chain within a chunk
// (reset when idx wraps to 0) while the host iter_args carry across chunks.
//
//   before:                        after:
//   <entry ops>                    <entry ops, types sliced in place>
//   %carry = scf.for ... () {      %carry = scf.for ... (sliced types) {
//     <closure body ops>             %esc = scf.for %i = 0..outIterCount
//     scf.yield %v                       iter_args(%z = zero, never read) {
//   }                                    <closure body ops, sliced>
//                                        scf.yield %v
//                                      }
//                                      scf.yield %esc
//                                   }
//
// Values produced by the moved closure but consumed outside it (the host
// yield, or body ops left behind) escape through the slicing loop results;
// their iter_arg inits are zero constants that no op ever reads, because the
// moved body always recomputes them from the host iter_args.
// Returns failure (without mutating anything) when a shape shows up that this
// rewrite does not handle; the caller then falls back to the default path.
LogicalResult OpTree::lowerSlicingLoopInsideFor(mlir::scf::ForOp hostFor) {
  // Captured before the move: lower() relocates these into the nested slicing
  // loop and their block no longer says which side of hostFor they ran on.
  llvm::SmallVector<mlir::Operation *> preHostStores;
  for (mlir::Operation *op : slicingOps_)
    if (isStoreLike(op) && !hostFor->isProperAncestor(op))
      preHostStores.push_back(op);

  size_t outIterCount = iterCount_[0];
  Block *hostBody = hostFor.getBody();
  auto hostYield = cast<mlir::scf::YieldOp>(hostBody->getTerminator());
  auto loc = hostFor->getLoc();

  // Moved set: closure ops living in the host body. Entry-block closure ops
  // stay in place and are type-sliced where they stand.
  SetVector<Operation *> movedSet;
  for (auto op : analysisOps_) {
    if (op->getBlock() == hostBody && op != hostYield)
      movedSet.insert(op);
  }

  // When the broadcast-back split will run, pull the reduce's forward closure
  // (within the host body) into the moved set. orgqr's reduce-result layout
  // conversions land outside analysisOps_ -- the identity-init store and the
  // consumer store build two trees and the chain is split between them -- so
  // without this they look like escapes that feed the closure back, and
  // feedsMovedSet() rejects the flip. The split replaces the reduce result
  // with the accumulate loop's total, so these ops must move into the loop
  // with the rest of the closure, not become loop results.
  if (mlir::triton::ReduceOp bcastRed = findFlipBroadcastReduce()) {
    llvm::SmallVector<mlir::Operation *> work{bcastRed.getOperation()};
    llvm::SmallPtrSet<mlir::Operation *, 32> seen;
    while (!work.empty()) {
      mlir::Operation *cur = work.pop_back_val();
      if (!seen.insert(cur).second)
        continue;
      for (mlir::Operation *user : cur->getUsers()) {
        if (user->getBlock() == hostBody && user != hostYield)
          movedSet.insert(user);
        work.push_back(user);
      }
    }
  }

  // Escape channels: results of moved ops consumed outside the moved set.
  // Each distinct value gets one slicing-loop result slot; every external
  // user is rewired to that result after ctx.lower.
  SmallVector<Value> escapeVals;
  DenseMap<Value, size_t> escapeSlot;
  SmallVector<std::pair<Operation *, Value>> externalUses;
  auto isExternalUser = [&](Operation *user) {
    for (Operation *a = user; a; a = a->getParentOp())
      if (movedSet.contains(a))
        return false;
    return true;
  };
  for (auto op : movedSet) {
    for (auto res : op->getResults()) {
      for (auto user : res.getUsers()) {
        if (!isExternalUser(user))
          continue;
        // Only users inside the host loop can legally reference the slicing
        // loop result; entry-block or foreign-tree users block the flip.
        if (!hostFor->isAncestor(user)) {
          if (::getenv("TRITONXPU_TRACE_FLIP"))
            llvm::errs() << "[flip] fail: escape user not in hostFor\n";
          return failure();
        }
        // Non-tensor escapes would need scalar zero inits, which is not
        // handled here; keep the rewrite tensor-only.
        if (!isa<RankedTensorType>(res.getType())) {
          if (::getenv("TRITONXPU_TRACE_FLIP"))
            llvm::errs() << "[flip] fail: non-tensor escape\n";
          return failure();
        }
        // A consumer that feeds the closure back would have to read the
        // slicing loop's result from inside that loop.
        if (feedsMovedSet(user, movedSet)) {
          if (::getenv("TRITONXPU_TRACE_FLIP"))
            llvm::errs() << "[flip] fail feedsMovedSet def="
                         << res.getDefiningOp()->getName() << " user="
                         << user->getName() << "\n";
          return failure();
        }
        if (!escapeSlot.contains(res)) {
          escapeSlot[res] = escapeVals.size();
          escapeVals.push_back(res);
        }
        externalUses.emplace_back(user, res);
      }
    }
  }

  // Root-block closure ops move into the host body first: the moved set has to
  // be single-block (sortOpTreeBwd requires a common region), and anything
  // derived from a make range is only valid inside the slicing loop. ctx.lower
  // then lowers them into it like the body ops.
  llvm::SetVector<Operation *> movedRoot = collectRootOpsToMove();
  for (Operation *op : movedRoot)
    for (Operation *user : op->getUsers())
      if (!hostFor->isAncestor(user) && !movedRoot.contains(user)) {
        if (::getenv("TRITONXPU_TRACE_FLIP"))
          llvm::errs() << "[flip] fail: root-op user escapes; op="
                       << op->getName() << " user=" << user->getName() << "\n";
        return failure();
      }
  Operation *rootIp = nullptr;
  for (Operation *op : movedRoot) {
    if (rootIp)
      op->moveAfter(rootIp);
    else
      op->moveBefore(hostBody, hostBody->begin());
    rootIp = op;
    // The op moved after movedSet was collected; register it so ctx.lower
    // moves it into the slicing loop with the rest of the closure.
    movedSet.insert(op);
  }

  // Slice the host loop frame: step constant, result types, iter_arg types
  // (mirrors handleForInLoop, minus its body walk).
  bool isInnerHost = inOpChain(innerChain_, hostFor);
  if (iterCount_.back() != 1) {
    if (auto stepOp = dyn_cast<mlir::arith::ConstantOp>(
            hostFor.getStep().getDefiningOp())) {
      for (auto [i, resTy] : llvm::enumerate(stepOp->getResultTypes()))
        stepOp->getResult(i).setType(
            getSlicedType(resTy, iterCount_, isInnerHost));
    }
  }
  for (auto [i, resTy] : llvm::enumerate(hostFor->getResultTypes()))
    hostFor->getResult(i).setType(
        getSlicedType(resTy, iterCount_, isInnerHost));
  for (auto forArg : hostBody->getArguments()) {
    bool isInnerArg = inOpChain(innerChain_, forArg.getDefiningOp());
    forArg.setType(getSlicedType(forArg.getType(), iterCount_, isInnerArg));
  }

  // Entry-block closure ops keep their position. Constants are re-materalized
  // at the sliced type with only this tree's uses rewired (mirrors
  // handleConstantInLoop's scoped rewrite, extended to host loop operands).
  for (auto op : slicingOps_) {
    if (op == hostFor || movedRoot.contains(op))
      continue;
    bool isInnerOp = inOpChain(innerChain_, op);
    if (auto constOp = dyn_cast<mlir::arith::ConstantOp>(op)) {
      auto attr = mlir::dyn_cast<DenseElementsAttr>(constOp.getValue());
      if (!attr)
        continue;
      Type slicedTy = getSlicedType(
          constOp.getType(), iterCount_,
          isInnerForRank1Result(rank1Dims_, iterCount_, constOp, 0, isInnerOp),
          /*needNewEncoding=*/false,
          rank1CountForResult(rank1Dims_, iterCount_, constOp, 0));
      if (slicedTy == constOp.getType())
        continue;
      auto newValue =
          reshapeDenseAttrTo(attr, mlir::cast<ShapedType>(slicedTy));
      if (!newValue)
        continue;
      OpBuilder constBuilder(constOp);
      auto newConstOp = constBuilder.create<mlir::arith::ConstantOp>(
          constOp->getLoc(), slicedTy, newValue);
      for (auto &use : llvm::make_early_inc_range(constOp->getUses())) {
        Operation *owner = use.getOwner();
        if (movedSet.contains(owner) || slicingOps_.contains(owner))
          use.set(newConstOp->getResult(0));
      }
    } else {
      for (auto [i, resTy] : llvm::enumerate(op->getResultTypes()))
        op->getResult(i).setType(getSlicedType(
            resTy, iterCount_,
            isInnerForRank1Result(rank1Dims_, iterCount_, op, i, isInnerOp),
            /*needNewEncoding=*/false,
            rank1CountForResult(rank1Dims_, iterCount_, op, i)));
    }
  }

  OpBuilder builder(hostBody, hostBody->begin());

  // Zero-typed inits for the escape slots. Never read: the moved body
  // recomputes every escape value from the host iter_args, so these only
  // satisfy scf.for's init/result type contract.
  auto getZeroInit = [&](Value v) -> Value {
    // Slice the init by the escape value's own chain membership, matching how
    // ctx.lower retypes that value's producer (inOpChain is used everywhere
    // else). A hardcoded isInner=false mis-sizes an inner-chain escape's init
    // (getSlicedType divides a 1D shape by iterCount[0] vs [1]) and breaks
    // scf.for's init==iter_arg==yield==result contract when iterCount[0]!=[1].
    Type slicedTy = getSlicedType(v.getType(), iterCount_,
                                  inOpChain(innerChain_, v.getDefiningOp()));
    ShapedType shapedTy = mlir::cast<ShapedType>(slicedTy);
    TypedAttr zeroAttr;
    if (shapedTy.getElementType().isIntOrIndex())
      zeroAttr = DenseElementsAttr::get(
          shapedTy,
          APInt(shapedTy.getElementType().getIntOrFloatBitWidth(), 0));
    else
      zeroAttr = DenseElementsAttr::get(
          shapedTy,
          APFloat::getZero(mlir::cast<FloatType>(shapedTy.getElementType())
                               .getFloatSemantics()));
    return builder.create<mlir::arith::ConstantOp>(v.getLoc(), slicedTy,
                                                   zeroAttr);
  };
  SmallVector<Value> inits;
  for (auto v : escapeVals)
    inits.push_back(getZeroInit(v));

  // The slicing loop at the top of the host body. Handler-created ops (e.g.
  // the idx-dependent make_range) land right after idx; ctx.lower places the
  // moved ops before the yield.
  auto low =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto upper = builder.create<mlir::arith::ConstantOp>(
      loc, builder.getIndexAttr(outIterCount));
  auto step =
      builder.create<mlir::arith::ConstantOp>(loc, builder.getIndexAttr(1));
  auto sliceFor = builder.create<scf::ForOp>(loc, low, upper, step, inits);
  flippedSliceLoop_ = sliceFor;
  Block *sliceBody = sliceFor.getBody();
  builder.setInsertionPointToStart(sliceBody);
  Value idx = builder.create<mlir::arith::IndexCastOp>(
      loc, builder.getI32Type(), sliceFor.getInductionVar());
  Operation *sliceYieldOp;
  if (inits.empty()) {
    // No iter_args: the builder auto-creates an empty terminator.
    sliceYieldOp = sliceBody->getTerminator();
  } else {
    // With iter_args the terminator is left to the caller; yield the escape
    // values, whose producers ctx.lower is about to move in front of it.
    SmallVector<Value> yieldOperands(escapeVals);
    builder.setInsertionPoint(sliceBody, sliceBody->end());
    builder.create<scf::YieldOp>(loc, yieldOperands);
    sliceYieldOp = sliceBody->getTerminator();
  }
  builder.setInsertionPointAfter(idx.getDefiningOp());

  SetVector<Operation *> sortedMoved = sortOpTreeBwd(movedSet);
  SliceLoopCtx ctx{sortedMoved,       iterCount_,
                   outIterCount,      innerChain_,
                   outerChain_,       rank1Dims_,
                   reduceGroupSize_,
                   reduceRowsPerCore_, builder,
                   idx,               upper};
  ctx.lower(sliceYieldOp);

  // Re-route the external uses to the slicing loop results.
  for (auto [user, res] : externalUses)
    user->replaceUsesOfWith(res, sliceFor.getResult(escapeSlot[res]));

  hoistFeedersAboveSliceLoop(hostBody, sliceFor);
  hoistPreHostStores(sliceFor, hostFor, preHostStores);

  // Body ops left outside the closure keep their position; give them the
  // default sliced treatment. Constants are skipped here: the ones just
  // created must not be re-sliced, and stale operand constants are fixed by
  // sibling inference inside handleDefaultInLoop.
  for (auto &inBlockOp : llvm::make_early_inc_range(*hostBody)) {
    if (sliceFor->isAncestor(&inBlockOp))
      continue;
    if (isa<mlir::scf::YieldOp>(inBlockOp) ||
        isa<mlir::arith::ConstantOp>(inBlockOp))
      continue;
    bool isInnerOp = inOpChain(innerChain_, &inBlockOp);
    ctx.handleDefaultInLoop(&inBlockOp, isInnerOp);
  }

  // Host yield operands that bypass the closure (entry constants, iter_arg
  // pass-through) may still carry the pre-slice shape; fix them positionally
  // against the host loop's sliced result types.
  ctx.reshapeConstYieldOperandsToMatchResults(hostYield,
                                              hostFor->getResultTypes());
  return success();
}

unsigned OpTree::countReduceOps() const {
  return llvm::count_if(
      slicingOps_, [](Operation *op) { return isa<triton::ReduceOp>(op); });
}

unsigned OpTree::countScanOps() const {
  return llvm::count_if(slicingOps_,
                        [](Operation *op) { return isa<triton::ScanOp>(op); });
}

} // namespace

struct TritonXPULegalizePass
    : public impl::TritonXPULegalizeBase<TritonXPULegalizePass> {

  using impl::TritonXPULegalizeBase<
      TritonXPULegalizePass>::TritonXPULegalizeBase;

  TritonXPULegalizePass() = default;
  TritonXPULegalizePass(unsigned bufferSize, unsigned coreNum,
                        unsigned groupsPerCluster) {
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
    this->groupsPerCluster = groupsPerCluster;
  }

  //===--------------------------------------------------------------------===//
  // Per-run state (populated by runOnOperation()'s helper steps)
  //===--------------------------------------------------------------------===//

  // Reverse closure rooted at every SM2GM[Mask] op; will be hoisted before
  // func.return so only core 0 executes it.
  llvm::SetVector<Operation *> sm2gmPtrLenOpChain_;

  // Ops that terminate a data flow (LM2GM[Mask] / SM2GM / AtomicRMW).
  llvm::SetVector<Operation *> endOps_;

  // Cross-tree dedup set for store ops.
  llvm::SetVector<Operation *> storeOps_;

  // Module-level GM2LM info for atomicSim trip-count computation.
  bool hasGm2lmFeedingReduce_ = false;
  int32_t maxGm2lmLrie_ = 1;

  void removeOpsOutsideRegion(llvm::SetVector<mlir::Operation *> &opChain,
                              mlir::Region *region) {
    if (!region) {
      opChain.clear();
      return;
    }

    llvm::SmallVector<mlir::Operation *, 32> opsToRemove;
    opsToRemove.reserve(opChain.size());

    for (mlir::Operation *op : opChain) {
      if (!op || op->getParentRegion() != region) {
        opsToRemove.push_back(op);
      }
    }

    for (mlir::Operation *op : opsToRemove) {
      opChain.remove(op);
    }
  }

  bool areOperandsAvailableBefore(Operation *opToMove, Operation *insertBefore,
                                  mlir::DominanceInfo &dom) {
    if (!opToMove || !insertBefore)
      return false;

    for (mlir::Value operand : opToMove->getOperands()) {
      if (auto blockArg = llvm::dyn_cast<mlir::BlockArgument>(operand)) {
        if (!dom.dominates(blockArg.getOwner(), insertBefore->getBlock()))
          return false;
      } else {
        Operation *defOp = operand.getDefiningOp();
        if (!defOp)
          return false;
        if (!dom.dominates(defOp, insertBefore))
          return false;
      }
    }
    return true;
  }

  //===--------------------------------------------------------------------===//
  // runOnOperation() helpers
  //===--------------------------------------------------------------------===//

  // Collect the def-use closure rooted at every SM2GM[Mask] op; this whole
  // chain will later be hoisted before func.return because the ptr/len
  // computation only needs to run on core 0.
  void
  collectSm2gmPtrLenChain(mlir::ModuleOp m,
                          llvm::SetVector<Operation *> *sm2gmPtrLenOpChain) {
    if (this->isUseMaskZero) {
      m.walk([&](triton::xpu::SM2GMMaskOp sm2gmOp) {
        sm2gmPtrLenOpChain->insert(sm2gmOp);
        getOpChainBwd(*sm2gmPtrLenOpChain, sm2gmOp.getPtr().getDefiningOp());
        if (sm2gmOp.getMask())
          getOpChainBwd(*sm2gmPtrLenOpChain, sm2gmOp.getMask().getDefiningOp());
      });
      return;
    }

    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) {
      sm2gmPtrLenOpChain->insert(sm2gmOp);
      getOpChainBwd(*sm2gmPtrLenOpChain, sm2gmOp.getPtr().getDefiningOp());
      if (sm2gmOp.getLen())
        getOpChainBwd(*sm2gmPtrLenOpChain, sm2gmOp.getLen().getDefiningOp());
    });
  }

  // Collect ops that terminate a data flow (LM2GM[Mask] / SM2GM / AtomicRMW).
  //
  // A 2D+ reduce on axis 0 used to be rejected here (XPUTC-7119). The slicer
  // now legalizes it, so the rejection is gone; what made it look impossible
  // was that a rank-1 value's width was taken from chain membership, which is
  // axis-unaware -- see computeRank1TileDim().
  void collectEndOps(mlir::ModuleOp m, SetVector<Operation *> *endOps) {
    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) { endOps->insert(lm2gmOp); });
    m.walk([&](triton::xpu::LM2GMMaskOp lm2gmOp) { endOps->insert(lm2gmOp); });
    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) { endOps->insert(sm2gmOp); });
    m.walk([&](triton::AtomicRMWOp rmwOp) { endOps->insert(rmwOp); });
    // tt.atomic_cas writes global memory just like tt.atomic_rmw, so it is a
    // slicing root too. Without it a kernel whose only global write is a CAS
    // has no sink at all, so no OpTree is built and this pass does nothing:
    // every tt.reduce/tt.scan in it stays unconverted and fails to legalize in
    // ConvertTritonXPUToLLVM. FlagGems' index_reduce prod is such a kernel.
    m.walk([&](triton::AtomicCASOp casOp) { endOps->insert(casOp); });
  }

  // Reject a tt.atomic_cas that shares a loop trip with a cross-core
  // reduce/scan, because the pair deadlocks on xpu3 instead of computing a
  // result.
  //
  // Two facts combine. xpu::atomicCAS (device/xpu3/trigonometric.xpu) is not a
  // hardware CAS: it wraps read-compare-write in xtransLock() on GM or
  // ticket_lock_mix() on SM, both spin locks. And rewriteReduceInRegion lowers
  // a reduce to triton_xpu.reduce, whose LLVM lowering emits xpu_barrier()
  // (ReduceOpToLLVM.cpp), which LLVMXPUOps.td expands to mfence.v2 plus
  // csr_set_sync_group(0xffff) -- a whole-cluster barrier. A core spinning on
  // the lock cannot reach the barrier its peers are waiting at.
  //
  // Measured on xpu3 (card7, FlagGems index_reduce "prod"): CAS alone in a
  // fixed-trip loop completes, a cross-core reduce alone in a fixed-trip loop
  // completes, replacing the CAS with a non-atomic load+store completes, and
  // only the two together hang -- the process stops in synchronize() and does
  // not respond to SIGTERM. An iteration cap does not help, because a core
  // parked at the barrier never reaches the cap.
  //
  // kUnsupportedOnXPU matches the prefix make_ttxir looks for in
  // third_party/xpu/backend/compiler.py, so the driver reports this as an
  // unsupported lowering rather than a resource error.
  LogicalResult validateNoCASWithLoopReduce(mlir::ModuleOp m) {
    auto regionHasClusterReduce = [](Region &region) -> Operation * {
      Operation *found = nullptr;
      region.walk([&](Operation *op) {
        if (found)
          return;
        if (isa<triton::ReduceOp, triton::ScanOp, triton::xpu::ReduceOp,
                triton::xpu::ScanOp>(op))
          found = op;
      });
      return found;
    };

    LogicalResult result = success();
    m.walk([&](triton::AtomicCASOp casOp) {
      // Any loop that encloses the CAS puts the lock and the barrier in the
      // same trip, so check every enclosing loop, not just the innermost one.
      for (Operation *parent = casOp->getParentOp(); parent;
           parent = parent->getParentOp()) {
        // An scf.while evaluates both of its regions once per trip: a Python
        // `while cond: body` puts the exit test in before and the body in
        // after, so a reduce in either one shares the trip with the CAS.
        SmallVector<Region *, 2> loopRegions;
        if (auto forOp = dyn_cast<mlir::scf::ForOp>(parent))
          loopRegions.push_back(&forOp.getRegion());
        else if (auto whileOp = dyn_cast<mlir::scf::WhileOp>(parent))
          loopRegions.append({&whileOp.getBefore(), &whileOp.getAfter()});
        if (loopRegions.empty())
          continue;
        for (Region *loopRegion : loopRegions) {
          if (Operation *reduceOp = regionHasClusterReduce(*loopRegion)) {
            casOp->emitError()
                << "[unsupported-on-xpu] tt.atomic_cas shares a loop body "
                   "with a cross-core "
                << reduceOp->getName()
                << "; xpu::atomicCAS takes a spin lock while the reduce emits "
                   "a whole-cluster barrier, and the two deadlock on xpu3";
            result = failure();
            return;
          }
        }
      }
    });
    return result;
  }

  // Collect module-level GM2LM info used by atomicSim trip-count logic:
  // whether any GM2LM feeds a reduce, and the max lrie across all GM2LM ops.
  void collectGm2lmInfo(mlir::ModuleOp m) {
    m.walk([&](triton::xpu::GM2LMOp op) {
      if (findUserOp<triton::ReduceOp, triton::xpu::ReduceOp>(op))
        hasGm2lmFeedingReduce_ = true;
      maxGm2lmLrie_ = std::max(maxGm2lmLrie_, op.getLrie());
    });
    m.walk([&](triton::xpu::GM2LMMaskOp op) {
      if (findUserOp<triton::ReduceOp, triton::xpu::ReduceOp>(op))
        hasGm2lmFeedingReduce_ = true;
      maxGm2lmLrie_ = std::max(maxGm2lmLrie_, op.getLrie());
    });
  }

  // Rewrite a tensor result's encoding to single-core layout (all dims mapped
  // to one core). Handles both ClusterLayoutAttr and SliceEncodingAttr.
  void rewriteToSingleCoreEncoding(Operation *op, OpResult res) {
    auto resTensorTy = mlir::dyn_cast<RankedTensorType>(res.getType());
    if (!resTensorTy)
      return;
    auto *context = op->getContext();
    auto resShape = resTensorTy.getShape();
    auto elemTy = resTensorTy.getElementType();
    auto encoding = resTensorTy.getEncoding();

    if (auto clusterEnc =
            mlir::dyn_cast<triton::xpu::ClusterLayoutAttr>(encoding)) {
      auto order = clusterEnc.getOrder();
      SmallVector<unsigned> newCoresPerGroup(
          clusterEnc.getCoresPerGroup().size(), 1);
      SmallVector<unsigned> newGroupsPerCluster(
          clusterEnc.getGroupsPerCluster().size(), 1);
      SmallVector<unsigned> newSizePerCore(resShape.begin(), resShape.end());

      auto newEncoding = triton::xpu::ClusterLayoutAttr::get(
          context, newSizePerCore, newCoresPerGroup, newGroupsPerCluster,
          order);
      auto newResTy = RankedTensorType::get(resShape, elemTy, newEncoding);
      if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
        if (auto attr =
                mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
          // Retype the constant only when its value can follow. A non-splat
          // constant cannot grow to a larger target -- reshapeDenseAttrTo()
          // returns null -- and changing the type without reshaping the value
          // would leave the attr at the old shape and fail the verifier.
          auto newValue = reshapeDenseAttrTo(attr, newResTy);
          if (!newValue) {
            // SM2GM runs on core 0 alone, so this result must become
            // single-core. Leaving res at its old multi-core encoding would be
            // wrong, not merely unoptimised, so fail loudly instead of
            // returning quietly. reshapeDenseAttrTo only returns null for a
            // non-splat constant that must grow, which this same-shape
            // re-encoding never asks for.
            op->emitError("cannot rewrite a non-splat constant to the "
                          "single-core SM2GM layout");
            signalPassFailure();
            return;
          }
          res.setType(newResTy);
          constOp.setValueAttr(newValue);
          return;
        }
      }
      res.setType(newResTy);
      return;
    }

    auto sliceEnc =
        mlir::dyn_cast<triton::gpu::SliceEncodingAttr>(encoding);
    assert(sliceEnc && "Unexpected tensor encoding in SM Optimization");
    auto parentEnc =
        mlir::cast<triton::xpu::ClusterLayoutAttr>(sliceEnc.getParent());
    auto dim = sliceEnc.getDim();
    auto sizePerCore = parentEnc.getSizePerCore();
    auto order = parentEnc.getOrder();

    SmallVector<unsigned> newCoresPerGroup(parentEnc.getCoresPerGroup().size(), 1);
    SmallVector<unsigned> newGroupsPerCluster(
        parentEnc.getGroupsPerCluster().size(), 1);
    SmallVector<unsigned> newSizePerCore(sizePerCore.size(), 1);
    assert(resShape.size() <= sizePerCore.size() &&
           "the sliced result must not out-rank its parent layout");
    if (sizePerCore.size() == 2 && resShape.size() == 1) {
      newSizePerCore[1 - dim] = resShape[0];
    } else {
      for (size_t i = 0; i < resShape.size(); ++i)
        newSizePerCore[i] = resShape[i];
    }

    auto newGlobalEncoding = triton::xpu::ClusterLayoutAttr::get(
        context, newSizePerCore, newCoresPerGroup, newGroupsPerCluster, order);
    auto newEncoding = triton::gpu::SliceEncodingAttr::get(
        context, sliceEnc.getDim(), newGlobalEncoding);
    res.setType(RankedTensorType::get(resShape, elemTy, newEncoding));
  }

  // Move the SM2GM ptr/len chain right before func.return and rewrite each
  // result's encoding to a single-core layout (only core 0 runs SM2GM).
  void hoistSm2gmPtrLenChain(mlir::ModuleOp m,
                             llvm::SetVector<Operation *> *sm2gmPtrLenOpChain) {
    if (sm2gmPtrLenOpChain->empty())
      return;

    SmallVector<func::ReturnOp> funcReturns;
    m.walk([&](func::ReturnOp r) { funcReturns.push_back(r); });
    assert(funcReturns.size() == 1 &&
           "Only one func.return is expected in the module");

    auto sortedChain = sortOpTreeBwd(*sm2gmPtrLenOpChain);
    for (auto op : llvm::reverse(sortedChain)) {
      op->moveBefore(funcReturns[0]);
      for (auto res : op->getResults())
        rewriteToSingleCoreEncoding(op, res);
    }
  }

  // Assign unique IDs and total counts to reduce/scan ops so downstream
  // passes can allocate non-overlapping shared-memory slots for each.
  // Rewrite tt.reduce ops that no OpTree owns.
  //
  // Why: Legalize rewrites the reduces inside an OpTree into triton_xpu.reduce,
  // but a reduce that feeds no store -- sample_dirichlet computes
  // tl.sum(~done) just to drive an scf.while condition -- belongs to no tree
  // and stayed a tt.reduce. ConvertTritonXPUToLLVM marks the Triton dialect
  // illegal and has no pattern for it, so the pipeline failed with
  //   failed to legalize operation tt.reduce
  // A reduce outside a slicing loop combines each core partial exactly once,
  // which is outIterCount = 1 with index 0: the lowering then skips its
  // loop-cache branch (loopIndex == 0) and only does the cross-core combine.
  //
  // How:
  //   - collect the remaining tt.reduce ops (walk covers nested regions such
  //     as the while body);
  //   - create triton_xpu.reduce with the same result types;
  //   - move the combine region over and replace the op.
  //
  // \param m module after the slicing loops have been lowered.
  // \returns the number of rewritten reduces (they count towards reduceNum).
  unsigned rewriteTreeExternalReduces(mlir::ModuleOp m) {
    SmallVector<triton::ReduceOp> reduces;
    m.walk([&](triton::ReduceOp op) { reduces.push_back(op); });
    for (auto op : reduces) {
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value zero = b.create<mlir::arith::ConstantOp>(
          loc, b.getI32Type(), b.getI32IntegerAttr(0));
      auto newOp = b.create<triton::xpu::ReduceOp>(
          loc, op->getResultTypes(), op.getSrcs(), op.getAxis(),
          /*outIterCount=*/1u, zero);
      auto &newCombine = newOp.getCombineOp();
      b.cloneRegionBefore(op.getCombineOp(), newCombine, newCombine.end());
      // HasParent requires triton_xpu.reduce.return inside the new op.
      for (auto &innerOp : llvm::make_early_inc_range(newCombine.getOps())) {
        auto ret = dyn_cast<triton::ReduceReturnOp>(innerOp);
        if (!ret)
          continue;
        OpBuilder rb(ret);
        auto newRet =
            rb.create<triton::xpu::ReduceReturnOp>(ret.getLoc(), ret.getOperands());
        ret->replaceAllUsesWith(newRet->getResults());
        ret.erase();
      }
      op.replaceAllUsesWith(newOp->getResults());
      op.erase();
    }
    return reduces.size();
  }

  // Rewrite tt.scan that sits outside every OpTree (e.g. a tl.cumprod inside an
  // scf.if body) into triton_xpu.scan, mirroring rewriteTreeExternalReduces.
  //
  // Why: OpTree building stops at the func entry block, so a scan nested in a
  // conditional never joins a tree and keeps its tt.scan form; ConvertTriton-
  // XPUToLLVM marks the whole triton dialect illegal and has no pattern for it,
  // so it fails to legalize (cumprod_backward's tl.cumprod at cumprod.py:500).
  // A tree-external scan runs once, so idx = 0 -- the lowering's slice-prefix
  // offset collapses to the single pass, exactly like outIterCount = 1 does for
  // the external reduce above.
  unsigned rewriteTreeExternalScans(mlir::ModuleOp m) {
    SmallVector<triton::ScanOp> scans;
    m.walk([&](triton::ScanOp op) { scans.push_back(op); });
    for (auto op : scans) {
      OpBuilder b(op);
      Location loc = op.getLoc();
      Value zero = b.create<mlir::arith::ConstantOp>(
          loc, b.getI32Type(), b.getI32IntegerAttr(0));
      auto newOp = b.create<triton::xpu::ScanOp>(
          loc, op->getResultTypes(), op.getSrcs(), op.getAxis(),
          op.getReverse(), zero);
      auto &newCombine = newOp.getCombineOp();
      b.cloneRegionBefore(op.getCombineOp(), newCombine, newCombine.end());
      // HasParent requires triton_xpu.scan.return inside the new op.
      for (auto &innerOp : llvm::make_early_inc_range(newCombine.getOps())) {
        auto ret = dyn_cast<triton::ScanReturnOp>(innerOp);
        if (!ret)
          continue;
        OpBuilder rb(ret);
        auto newRet = rb.create<triton::xpu::ScanReturnOp>(ret.getLoc(),
                                                           ret.getOperands());
        ret->replaceAllUsesWith(newRet->getResults());
        ret.erase();
      }
      op.replaceAllUsesWith(newOp->getResults());
      op.erase();
    }
    return scans.size();
  }

  void assignReduceScanIds(mlir::ModuleOp m, unsigned reduceNum,
                           unsigned scanNum) {
    unsigned reduceId = 0;
    m.walk([&](triton::xpu::ReduceOp redOp) {
      ReduceOpHelper helper(redOp);
      helper.setReduceId(reduceId);
      helper.setReduceNum(reduceNum);
      reduceId++;
    });

    unsigned scanId = 0;
    m.walk([&](triton::xpu::ScanOp scanOp) {
      ScanLoweringHelper helper(scanOp);
      helper.setScanId(scanId);
      helper.setScanNum(scanNum);
      scanId++;
    });
  }

  // Rewrite any tt.make_range that was NOT already handled by OpTree slicing
  // loops (i.e., ranges outside every endOp's backward closure).
  //
  // Classification:
  //   - "out range": feeds an expand_dims(axis=1), meaning it produces row
  //     indices for the outer (iteration) dimension.
  //     → replaced by triton_xpu.out_range(groupSize, rowsPerCore, idx=0)
  //   - otherwise: a plain index range (column or scalar).
  //     → replaced by triton_xpu.make_range(start, end, realSize, null, null)
  //
  // Example IR transformation:
  //
  //   Before:
  //     %r = tt.make_range {start = 0, end = 32} : tensor<32xi32, #enc>
  //     %e = tt.expand_dims %r {axis = 1} : ... -> tensor<32x1xi32, #enc2d>
  //
  //   After:
  //     %r = triton_xpu.out_range {groupSize = 2, rowsPerCore = 1, idx = 0}
  //           : tensor<32xi32, #enc>
  //     %e = tt.expand_dims %r {axis = 1} : ... -> tensor<32x1xi32, #enc2d>
  //
  //   Before:
  //     %c = tt.make_range {start = 0, end = 256} : tensor<256xi32, #enc>
  //
  //   After:
  //     %c = triton_xpu.make_range {start = 0, end = 256, realSize = 256}
  //           : tensor<256xi32, #enc>
  //
  /// Settle every width the per-tree walks could not finish, module-wide.
  ///
  /// Why: the slicing runs once per dataflow sink, and tree B can narrow a
  /// value tree A had already settled -- the surviving mismatch therefore only
  /// exists after the last tree, which no per-tree walk can see. The D-AMS
  /// layernorm welford kernel ends there:
  ///   'arith.subf' op requires the same type for all operands and results
  /// on the `pre - mean` of `norm = (pre - mean) * rsqrt(var + eps)`, where
  /// `pre` is the sliced 2-D tile and `mean` the reduce result broadcast back.
  ///
  /// How: run to a fixpoint; the single rule only copies a shape from one
  /// value to another, so it cannot oscillate -- loop until a pass narrows
  /// nothing, with no round cap.
  void settleModuleWidths(mlir::ModuleOp m) {
    while (narrowArithOperandUses(m)) {
    }
  }

  /// Give every arith op that mixes two widths a narrow VIEW of the wide
  /// operand, rather than retyping the wide value where it is defined.
  ///
  /// Why: retyping in place cascades -- the value also feeds ops that are
  /// already consistent, and its own producer (an scf result, a reduce) does
  /// not follow, so one mismatch turns into several. Rewriting only this use
  /// leaves the wide value and its other consumers untouched.
  ///
  /// How: collect the offending ops first, then rewrite -- inserting ops while
  /// walking would revisit what was just built.
  bool narrowArithOperandUses(mlir::ModuleOp m) {
    llvm::SmallVector<mlir::Operation *> work;
    m.walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::arith::ArithDialect>(op->getDialect()) &&
          isFailingVerifier(op) && hasMixedTensorWidths(op))
        work.push_back(op);
    });
    for (mlir::Operation *op : work)
      narrowOneArithOp(op);
    return !work.empty();
  }

  /// Whether `op` currently fails its own verifier.
  ///
  /// Why: the narrowing below is a repair, not a normalisation. Running it on
  /// every arith op that merely mixes widths is wrong -- a shared arange
  /// legitimately has a dim-0 and a dim-1 copy at different widths, and
  /// forcing them together breaks it. Gating on the verifier means only IR
  /// that would otherwise refuse to compile is touched.
  ///
  /// How: verify this op alone with diagnostics swallowed, so probing does not
  /// print the error we are about to fix.
  bool isFailingVerifier(mlir::Operation *op) {
    mlir::ScopedDiagnosticHandler handler(
        op->getContext(), [](mlir::Diagnostic &) { return success(); });
    return failed(mlir::verify(op, /*verifyRecursively=*/false));
  }

  /// Whether a tensor operand or result of `op` is wider than its narrowest
  /// tensor operand.
  bool hasMixedTensorWidths(mlir::Operation *op) {
    auto shape = tryGetNarrowestOperandShape(op);
    if (!shape)
      return false;
    for (Value operand : op->getOperands())
      if (isWiderThan(operand, *shape))
        return true;
    for (Value result : op->getResults())
      if (isWiderThan(result, *shape))
        return true;
    return false;
  }

  /// Whether `value` is a ranked tensor whose shape differs from `shape`.
  bool isWiderThan(Value value, llvm::ArrayRef<int64_t> shape) {
    auto ty = mlir::dyn_cast<RankedTensorType>(value.getType());
    return ty && ty.getShape() != shape;
  }

  /// Rewrite the wide uses of `op` at its narrowest operand width.
  ///
  /// FIXME(@yanxin11): the narrow view takes the leading elements, which is
  /// numerically wrong for a value the slicing loop should index per
  /// iteration -- handleMakeRangeInLoop() rebuilds a range per slice with a
  /// different start, so each slice covers a different element range, and
  /// every slice now reads the same leading elements. This repair is here to
  /// turn a hard compile failure into a kernel that builds and runs; the
  /// numeric gap is tracked separately.
  void narrowOneArithOp(mlir::Operation *op) {
    auto shape = tryGetNarrowestOperandShape(op);
    if (!shape)
      return;
    OpBuilder builder(op);
    for (mlir::OpOperand &use : op->getOpOperands())
      if (isWiderThan(use.get(), *shape))
        use.set(buildNarrowView(builder, op->getLoc(), use.get(), *shape));
    retypeTensorResultsTo(op, *shape);
  }

  void lowerRemainingMakeRanges(mlir::ModuleOp m) {
    SetVector<Operation *> outMakeRanges;
    m.walk([&](triton::ExpandDimsOp expandDimOp) {
      if (expandDimOp.getAxis() != 1)
        return;
      if (auto mrOp = findDefOpBwd<triton::MakeRangeOp>(expandDimOp.getSrc()))
        outMakeRanges.insert(mrOp);
    });

    m.walk([&](triton::MakeRangeOp mrOp) {
      // A region-body slice narrowed the result in place without touching
      // `end`; realign it here so the op (and the xpu range it becomes)
      // verify. Idempotent for ranges whose width was never sliced.
      if (auto rangeTy = mlir::dyn_cast<RankedTensorType>(mrOp.getType())) {
        uint32_t newEnd = mrOp.getStart() + rangeTy.getNumElements();
        mrOp.setEnd(newEnd);
      }
      OpBuilder builder(mrOp);
      auto loc = mrOp->getLoc();
      uint32_t start = mrOp.getStart();
      uint32_t end = mrOp.getEnd();
      uint32_t realSize = end - start;
      if (outMakeRanges.count(mrOp)) {
        size_t groupSize = std::ceil(this->coreNum / this->groupsPerCluster);
        size_t rowsPerCore = 1; // TODO: Set Real Data
        auto idx = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI32IntegerAttr(0));
        auto newOutRangeOp = builder.create<triton::xpu::OutRangeOp>(
            loc, mrOp.getType(), groupSize, rowsPerCore, idx);
        mrOp->replaceAllUsesWith(newOutRangeOp->getResults());
      } else {
        auto newMakeRangeOp = builder.create<triton::xpu::MakeRangeOp>(
            loc, mrOp.getType(), builder.getI32IntegerAttr(start),
            builder.getI32IntegerAttr(end), builder.getI32IntegerAttr(realSize),
            Value(), Value());
        mrOp->replaceAllUsesWith(newMakeRangeOp->getResults());
      }
    });
  }

  // Hoist cacheable GM2LM[Mask] chains to the entry of their func so the
  // GM->LM transfer happens once and the loop body reuses LM contents.
  void hoistGm2lmCacheChains(mlir::ModuleOp m) {
    SmallVector<SetVector<Operation *>> lmCacheChains;

    auto collectCacheChain = [&](auto gm2lmOp) {
      SetVector<Operation *> chain;
      getOpChainBwd(chain, gm2lmOp);
      removeOpsOutsideRegion(chain, gm2lmOp->getParentRegion());
      lmCacheChains.emplace_back(sortOpTreeBwd(chain));
    };
    m.walk([&](triton::xpu::GM2LMOp op) {
      if (op.getCache())
        collectCacheChain(op);
    });
    m.walk([&](triton::xpu::GM2LMMaskOp op) {
      if (op.getCache())
        collectCacheChain(op);
    });

    for (auto &chain : llvm::reverse(lmCacheChains)) {
      for (auto op : chain) {
        // Walk up to the enclosing FuncOp.
        Operation *ancestorOp = op;
        while (!isa<triton::FuncOp>(ancestorOp))
          ancestorOp = ancestorOp->getBlock()->getParentOp();
        auto func = cast<triton::FuncOp>(ancestorOp);
        mlir::DominanceInfo dom(func);
        Operation *firstOp = &(*func.getBody().front().begin());

        if (!areOperandsAvailableBefore(op, firstOp, dom))
          continue;
        op->moveBefore(firstOp);
      }
    }
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // Collect the SM2GM pointer/length def-chain. SM2GM (shared-memory to
    // global-memory) writes are single-core (core 0 only), so their address
    // computation must be excluded from the multi-core slicing loop and hoisted
    // to run once before func.return.
    collectSm2gmPtrLenChain(m, &sm2gmPtrLenOpChain_);

    // Find all data-flow sinks (LM2GM, SM2GM, AtomicRMW) as slicing roots,
    // and validate that no 2D+ reduce has axis==0 (unsupported without
    // transpose).
    collectEndOps(m, &endOps_);

    // A tt.atomic_cas in the same loop trip as a cross-core reduce/scan
    // deadlocks on device, so reject it before any slicing happens.
    if (failed(validateNoCASWithLoopReduce(m)))
      return signalPassFailure();

    // Collect module-level GM2LM info (hasGm2lmFeedingReduce_, maxGm2lmLrie_)
    // used by atomicSim trip-count logic below.
    collectGm2lmInfo(m);

    // Build all OpTrees from dataflow sinks.
    llvm::SmallVector<OpTree, 0> opTrees;
    llvm::SetVector<Operation *> visitedSlicingOps;
    llvm::SetVector<Operation *> visitedAnalysisOps;
    for (auto endOp : endOps_) {
      if (auto tree = OpTree::build(endOp, visitedSlicingOps,
                                    visitedAnalysisOps, storeOps_))
        opTrees.push_back(std::move(*tree));
    }

    // Split the trees whose axis-0 reduce result is fed back onto the tile it
    // reduced: the reduce chain is cloned into its own slicing loop ahead of
    // the consumer, and the consumer reads the loop result -- the complete
    // reduction -- instead of the running prefix it would see inside one loop.
    // See third_party/xpu/docs/axis0_reduce_broadcast_back.md.
    llvm::SmallVector<OpTree, 0> accumTrees;
    unsigned accumReduceNum = 0;
    // Totals already carried out by an earlier peel. A later reduce whose input
    // tile is built from one of these (var-mean: v = sum((x - mean)^2)) is a
    // CHAINED accumulator -- its accumulate tree reads a prior loop result
    // broadcast onto the 2-D tile, which the slicer cannot cut yet and would
    // lower into an invalid make_range. Such a reduce is left un-peeled below.
    llvm::SmallPtrSet<mlir::Value, 8> carriedTotals;
    for (auto &tree : opTrees) {
      // Peel one reduce per round: splitting the inner one can expose an outer
      // one whose chain no longer holds a reduce (logsumexp has two).
      // Peel every splittable reduce; the loop exits when none remain, so no
      // fixed round count can cut it short (logsumexp needs 2).
      while (true) {
      llvm::SmallVector<Operation *> chain;
      auto splittable = tree.findSplittableReduce(chain, &carriedTotals);
      if (!splittable)
        break;

      // A chained accumulator (var-mean: v = sum((x - mean)^2)) reads a total an
      // earlier round carried out. findSplittableReduce() stopped its chain at
      // those totals, so the carried mean is a loop-invariant leaf here and the
      // accumulate tree below slices only the tile -- see
      // third_party/xpu/docs/axis0_reduce_broadcast_back.md.

      // Clone the chain ahead of its first op, so the accumulate loop lands
      // before the consumer loop and dominates it. For a chained accumulator the
      // chain also reads carried totals (the mean's loop result); the clone
      // must land after the loop that produced each, or that result will not
      // dominate it. Push the insertion point past the latest such producer.
      Operation *insertBefore = chain.front();
      for (Operation *op : chain)
        for (mlir::Value operand : op->getOperands())
          if (carriedTotals.contains(operand))
            if (Operation *def = operand.getDefiningOp())
              if (def->getBlock() == insertBefore->getBlock() &&
                  insertBefore->isBeforeInBlock(def) && def->getNextNode())
                insertBefore = def->getNextNode();
      OpBuilder cloneBuilder(insertBefore);
      IRMapping map;
      Operation *clonedReduce = nullptr;
      llvm::SmallVector<Operation *> clones;
      for (Operation *op : chain) {
        Operation *clone = cloneBuilder.clone(*op, map);
        clones.push_back(clone);
        if (op == splittable.getOperation())
          clonedReduce = clone;
      }

      auto dropClones = [&]() {
        for (Operation *clone : llvm::reverse(clones))
          clone->erase();
      };
      if (!clonedReduce) {
        dropClones();
        break;
      }

      llvm::SetVector<Operation *> accumSlicing, accumAnalysis;
      auto accumTree = OpTree::build(clonedReduce, accumSlicing,
                                     accumAnalysis, storeOps_, &carriedTotals);
      if (!accumTree) {
        dropClones();
        break;
      }
      accumTree->carryReduceResult_ = clonedReduce;
      bool ok = accumTree->computeReduceGroupInfo(this->groupsPerCluster,
                                                  this->coreNum);
      ok &= succeeded(accumTree->computeIterCount(
          this->bufferSize, this->coreNum, this->groupsPerCluster,
          hasGm2lmFeedingReduce_, maxGm2lmLrie_, this->allowRank3));
      if (!ok) {
        dropClones();
        break;
      }
      accumTree->partitionChains();
      accumTree->computeRank1Dims();
      accumTree->splitSharedMakeRanges();
      accumTree->splitSharedChainValues();
      accumTree->computeRank1Dims();
      accumReduceNum += accumTree->countReduceOps();
      accumTree->lowerSlicingLoop();
      if (!accumTree->carriedReduceTotal_) {
        dropClones();
        break;
      }

      // The consumer now reads the complete reduction, and its own reduce is
      // dead: drop it from the tree before the derivations below run, so the
      // iteration counts and chains are computed without it.
      carriedTotals.insert(accumTree->carriedReduceTotal_);
      splittable->getResult(0).replaceAllUsesWith(
          accumTree->carriedReduceTotal_);
      tree.slicingOps_.remove(splittable);
      tree.analysisOps_.remove(splittable);
      splittable.erase();
      accumTrees.push_back(std::move(*accumTree));
      }
    }

    // Per-tree analysis: reduce group info, iteration counts, chain partition.
    // Nothing past this block may run once a tree is rejected: the rejection
    // reasons are shape constraints the slicer has no representation for, and
    // rewriting a rejected tree would emit malformed IR.
    unsigned reduceNum = accumReduceNum, scanNum = 0;
    bool sliceable = true;
    for (auto &tree : opTrees) {
      sliceable &= tree.computeReduceGroupInfo(this->groupsPerCluster,
                                               this->coreNum);
      sliceable &= succeeded(tree.computeIterCount(
          this->bufferSize, this->coreNum, this->groupsPerCluster,
          hasGm2lmFeedingReduce_, maxGm2lmLrie_, this->allowRank3));
      tree.partitionChains();
      tree.computeRank1Dims();
      tree.splitSharedMakeRanges();
      tree.splitSharedChainValues();
      // The split above added clones; give them their dimension too.
      tree.computeRank1Dims();
      reduceNum += tree.countReduceOps();
      if (tree.rejectBroadcastBackOntoTile())
        sliceable = false;
      if (::getenv("TRITONXPU_REPORT_REDUCE_BROADCAST")) {
        llvm::SmallVector<Operation *> chain;
        if (auto splittable = tree.findSplittableReduce(chain))
          llvm::errs() << "[reduce-split] candidate chain=" << chain.size()
                       << " ops at " << splittable.getLoc() << "\n";
      }
      scanNum += tree.countScanOps();
    }
    if (!sliceable) {
      signalPassFailure();
      return;
    }

    // Per-tree lowering: wrap each tree in a scf.for slicing loop.
    for (auto &tree : opTrees)
      tree.lowerSlicingLoop();

    // Move SM2GM ptr/len chain to right before func.return. SM2GM is a
    // single-core (core 0) operation that must execute after the multi-core
    // slicing loop completes.
    hoistSm2gmPtrLenChain(m, &sm2gmPtrLenOpChain_);

    // Reduces outside every tree still have to become triton_xpu.reduce, or the
    // LLVM conversion has no pattern for them.
    reduceNum += rewriteTreeExternalReduces(m);

    // Scans outside every tree likewise must become triton_xpu.scan.
    scanNum += rewriteTreeExternalScans(m);

    // Assign sequential IDs and total counts to reduce/scan ops.
    assignReduceScanIds(m, reduceNum, scanNum);

    // One OpTree can narrow a value a previous tree had already settled, so
    // the last mismatch only exists once every tree is done. This is the only
    // point that sees the whole module.
    settleModuleWidths(m);

    // A region-body slice may narrow a convert_layout's source after the op's
    // own result was set; realign result shape to source (encoding-only op).
    // Runs after settleModuleWidths so every tree-driven narrowing is in; it
    // does not catch a source narrowed by a later step (the expand_dims / splat
    // class, which needs in-region handling and is tracked separately).
    m.walk([&](triton::xpu::ConvertLayoutOp cvtOp) {
      alignConvertLayoutShape(cvtOp);
    });
    // Lower any tt.make_range that wasn't consumed by the slicing loop.
    lowerRemainingMakeRanges(m);

    // Hoist loop-invariant GM2LM[Mask] chains to FuncOp entry block.
    hoistGm2lmCacheChains(m);
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir