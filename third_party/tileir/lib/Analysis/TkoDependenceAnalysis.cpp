#include "Analysis/TkoDependenceAnalysis.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

#include <optional>

using namespace mlir;
using namespace mlir::triton::tileir;

//===----------------------------------------------------------------------===//
// Local affine subset
//===----------------------------------------------------------------------===//

// This file deliberately implements a bounded local proof, not a general
// dependence solver. It recognizes a small affine subset that is useful for
// AutoGenMemoryToken's parallel-store optimization, and every unsupported or
// over-budget case falls back to "not proven disjoint".
//
// View proofs are rank-generic but require exactly one loop-IV-dependent index
// dimension. Pointer proofs either reduce to an IV stride plus a lane interval,
// or enumerate a finite set of per-lane affine offsets. The finite-set path is
// capped by element count, trip count, and total membership checks to avoid
// compile-time cliffs.

namespace {

/// Internal witness for the proof family that established disjointness.
struct DisjointWitness {
  enum class Kind {
    /// `partition_view` selects a unique sub-tile per partition index.
    PartitionUniqueness,
    /// `strided_view` advances far enough along the IV-indexed dimension that
    /// the covered tile footprints do not overlap across iterations.
    AffineStrideDominates,
    /// `store_ptr_tko` pointer arithmetic was reduced to
    /// IV-stride-plus-lane-interval, and the IV stride dominates the interval.
    PointerAffineStrideDominates,
    /// `store_ptr_tko` pointer arithmetic was reduced to a finite set of
    /// per-lane affine offsets, and no set element collides with any shifted
    /// copy reachable by another loop iteration.
    PointerAffineSetDisjoint,
  };
  Kind kind;
};

enum class DependenceResultKind {
  /// A same-root collision was proven impossible.
  ProvenDisjoint,
  /// The access relation was understood and may collide.
  PossibleCollision,
  /// The access relation is outside the supported affine subset.
  Unsupported,
};

struct DependenceResult {
  DependenceResultKind kind = DependenceResultKind::Unsupported;
  std::optional<DisjointWitness> witness;

  static DependenceResult proven(std::optional<DisjointWitness> witness) {
    return {DependenceResultKind::ProvenDisjoint, witness};
  }
  static DependenceResult possible() {
    return {DependenceResultKind::PossibleCollision, std::nullopt};
  }
  static DependenceResult unsupported() {
    return {DependenceResultKind::Unsupported, std::nullopt};
  }
};

/// Small affine summary used for loop-carried self-dependence proofs.
///
/// The expression represented here is:
///   ivCoeff * loopIV + [minOffset, maxOffset] + loop-invariant terms
///
/// Loop-invariant terms are deliberately not represented: for a self-query of
/// the same operation across two iterations, they are shared by both sides and
/// cancel out of the collision equation. Terms that can vary per lane stay in
/// the interval.
struct LinearExpr {
  int64_t ivCoeff = 0;
  int64_t minOffset = 0;
  int64_t maxOffset = 0;
};

/// Exact per-lane affine expression used for tiled pointer arithmetic. Each
/// element is `ivCoeff * loopIV + offset`.
struct ElementExpr {
  int64_t ivCoeff = 0;
  int64_t offset = 0;
};

struct TileExpr {
  SmallVector<int64_t> shape;
  SmallVector<ElementExpr> elements;
};

bool isLoopInvariant(Value v, cuda_tile::ForOp loop) {
  if (auto blockArg = dyn_cast<BlockArgument>(v))
    return blockArg.getOwner() != loop.getBody();
  Operation *def = v.getDefiningOp();
  return !def || !loop->isProperAncestor(def);
}

bool isSingleElementTile(Type type) {
  auto tileTy = dyn_cast<cuda_tile::TileType>(type);
  if (!tileTy)
    return false;
  int64_t count = 1;
  for (int64_t dim : tileTy.getShape()) {
    if (dim <= 0)
      return false;
    count *= dim;
  }
  return count == 1;
}

int64_t getTileElementCount(Type type) {
  auto tileTy = dyn_cast<cuda_tile::TileType>(type);
  if (!tileTy)
    return -1;
  int64_t count = 1;
  for (int64_t dim : tileTy.getShape()) {
    if (dim <= 0)
      return -1;
    count *= dim;
  }
  return count;
}

std::optional<SmallVector<int64_t>> getTileShape(Type type) {
  auto tileTy = dyn_cast<cuda_tile::TileType>(type);
  if (!tileTy)
    return std::nullopt;
  SmallVector<int64_t> shape;
  for (int64_t dim : tileTy.getShape()) {
    if (dim <= 0)
      return std::nullopt;
    shape.push_back(dim);
  }
  return shape;
}

int64_t getElementCount(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t dim : shape)
    count *= dim;
  return count;
}

SmallVector<int64_t> getIndices(int64_t linear, ArrayRef<int64_t> shape) {
  SmallVector<int64_t> indices(shape.size(), 0);
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    indices[i] = linear % shape[i];
    linear /= shape[i];
  }
  return indices;
}

int64_t getLinearIndex(ArrayRef<int64_t> indices, ArrayRef<int64_t> shape) {
  int64_t linear = 0;
  for (auto [index, dim] : llvm::zip_equal(indices, shape))
    linear = linear * dim + index;
  return linear;
}

std::optional<std::pair<int64_t, int64_t>>
getDenseIntMinMax(DenseIntElementsAttr attr) {
  if (!attr)
    return std::nullopt;

  std::optional<int64_t> minValue;
  std::optional<int64_t> maxValue;
  for (APInt value : attr.getValues<APInt>()) {
    int64_t sext = value.getSExtValue();
    minValue = minValue ? std::min(*minValue, sext) : sext;
    maxValue = maxValue ? std::max(*maxValue, sext) : sext;
  }
  if (!minValue || !maxValue)
    return std::nullopt;
  return std::pair<int64_t, int64_t>{*minValue, *maxValue};
}

std::optional<std::pair<int64_t, int64_t>> getConstantMinMax(Value v) {
  if (auto constant = v.getDefiningOp<cuda_tile::ConstantOp>()) {
    return getDenseIntMinMax(
        dyn_cast<DenseIntElementsAttr>(constant.getValue()));
  }
  return std::nullopt;
}

std::optional<SmallVector<int64_t>> getConstantValues(Value v) {
  auto appendDense =
      [](DenseIntElementsAttr attr) -> std::optional<SmallVector<int64_t>> {
    if (!attr)
      return std::nullopt;
    SmallVector<int64_t> values;
    for (APInt value : attr.getValues<APInt>())
      values.push_back(value.getSExtValue());
    return values;
  };

  if (auto constant = v.getDefiningOp<cuda_tile::ConstantOp>())
    return appendDense(dyn_cast<DenseIntElementsAttr>(constant.getValue()));

  return std::nullopt;
}

std::optional<int64_t> getSplatConstant(Value v) {
  std::optional<std::pair<int64_t, int64_t>> minMax = getConstantMinMax(v);
  if (!minMax || minMax->first != minMax->second)
    return std::nullopt;
  return minMax->first;
}

std::optional<TileExpr> getExactTileExpr(Value v, cuda_tile::ForOp loop);

TileExpr makeSplatTileExpr(Type type, ElementExpr expr) {
  TileExpr out;
  out.shape = *getTileShape(type);
  out.elements.assign(getElementCount(out.shape), expr);
  return out;
}

std::optional<TileExpr> reshapeTileExpr(TileExpr expr, Type resultType) {
  std::optional<SmallVector<int64_t>> resultShape = getTileShape(resultType);
  if (!resultShape ||
      getElementCount(*resultShape) != int64_t(expr.elements.size()))
    return std::nullopt;
  expr.shape = *resultShape;
  return expr;
}

std::optional<TileExpr> broadcastTileExpr(TileExpr source, Type resultType) {
  std::optional<SmallVector<int64_t>> resultShape = getTileShape(resultType);
  if (!resultShape || source.shape.size() != resultShape->size())
    return std::nullopt;

  TileExpr out;
  out.shape = *resultShape;
  out.elements.reserve(getElementCount(out.shape));
  for (int64_t linear = 0, e = getElementCount(out.shape); linear < e;
       ++linear) {
    SmallVector<int64_t> indices = getIndices(linear, out.shape);
    SmallVector<int64_t> sourceIndices;
    sourceIndices.reserve(indices.size());
    for (auto [index, sourceDim, resultDim] :
         llvm::zip_equal(indices, source.shape, out.shape)) {
      if (sourceDim == resultDim) {
        sourceIndices.push_back(index);
        continue;
      }
      if (sourceDim != 1)
        return std::nullopt;
      sourceIndices.push_back(0);
    }
    out.elements.push_back(
        source.elements[getLinearIndex(sourceIndices, source.shape)]);
  }
  return out;
}

std::optional<TileExpr> combineSameShape(
    TileExpr lhs, TileExpr rhs,
    llvm::function_ref<std::optional<ElementExpr>(ElementExpr, ElementExpr)>
        combine) {
  if (lhs.shape != rhs.shape || lhs.elements.size() != rhs.elements.size())
    return std::nullopt;

  TileExpr out;
  out.shape = lhs.shape;
  out.elements.reserve(lhs.elements.size());
  for (auto [lhsElement, rhsElement] :
       llvm::zip_equal(lhs.elements, rhs.elements)) {
    std::optional<ElementExpr> element = combine(lhsElement, rhsElement);
    if (!element)
      return std::nullopt;
    out.elements.push_back(*element);
  }
  return out;
}

std::optional<ElementExpr> mulElement(ElementExpr lhs, ElementExpr rhs) {
  if (lhs.ivCoeff != 0 && rhs.ivCoeff != 0)
    return std::nullopt;
  if (lhs.ivCoeff != 0)
    return ElementExpr{lhs.ivCoeff * rhs.offset, lhs.offset * rhs.offset};
  if (rhs.ivCoeff != 0)
    return ElementExpr{rhs.ivCoeff * lhs.offset, rhs.offset * lhs.offset};
  return ElementExpr{0, lhs.offset * rhs.offset};
}

std::optional<TileExpr> getExactBinaryTileExpr(Operation *op,
                                               cuda_tile::ForOp loop) {
  if (auto addOp = dyn_cast<cuda_tile::AddIOp>(op)) {
    std::optional<TileExpr> lhs = getExactTileExpr(addOp.getLhs(), loop);
    std::optional<TileExpr> rhs = getExactTileExpr(addOp.getRhs(), loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineSameShape(*lhs, *rhs, [](ElementExpr lhs, ElementExpr rhs) {
      return ElementExpr{lhs.ivCoeff + rhs.ivCoeff, lhs.offset + rhs.offset};
    });
  }
  if (auto subOp = dyn_cast<cuda_tile::SubIOp>(op)) {
    std::optional<TileExpr> lhs = getExactTileExpr(subOp.getLhs(), loop);
    std::optional<TileExpr> rhs = getExactTileExpr(subOp.getRhs(), loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineSameShape(*lhs, *rhs, [](ElementExpr lhs, ElementExpr rhs) {
      return ElementExpr{lhs.ivCoeff - rhs.ivCoeff, lhs.offset - rhs.offset};
    });
  }
  if (auto mulOp = dyn_cast<cuda_tile::MulIOp>(op)) {
    std::optional<TileExpr> lhs = getExactTileExpr(mulOp.getLhs(), loop);
    std::optional<TileExpr> rhs = getExactTileExpr(mulOp.getRhs(), loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return combineSameShape(*lhs, *rhs, mulElement);
  }
  return std::nullopt;
}

std::optional<TileExpr> getExactTileExpr(Value v, cuda_tile::ForOp loop) {
  std::optional<SmallVector<int64_t>> shape = getTileShape(v.getType());
  if (!shape)
    return std::nullopt;

  if (v == loop.getBody()->getArgument(0))
    return makeSplatTileExpr(v.getType(), ElementExpr{/*ivCoeff=*/1,
                                                      /*offset=*/0});

  if (std::optional<SmallVector<int64_t>> constants = getConstantValues(v)) {
    int64_t count = getElementCount(*shape);
    if (constants->size() != size_t(count))
      return std::nullopt;
    TileExpr out;
    out.shape = *shape;
    out.elements.reserve(count);
    for (int64_t value : *constants)
      out.elements.push_back(ElementExpr{/*ivCoeff=*/0, value});
    return out;
  }

  if (auto iotaOp = v.getDefiningOp<cuda_tile::IotaOp>()) {
    int64_t count = getElementCount(*shape);
    TileExpr out;
    out.shape = *shape;
    out.elements.reserve(count);
    for (int64_t i = 0; i < count; ++i)
      out.elements.push_back(ElementExpr{/*ivCoeff=*/0, i});
    return out;
  }

  if (Operation *def = v.getDefiningOp()) {
    if (auto assumeOp = dyn_cast<cuda_tile::AssumeOp>(def))
      return getExactTileExpr(assumeOp.getValue(), loop);
    if (auto reshapeOp = dyn_cast<cuda_tile::ReshapeOp>(def)) {
      std::optional<TileExpr> source =
          getExactTileExpr(reshapeOp.getSource(), loop);
      if (!source)
        return std::nullopt;
      return reshapeTileExpr(*source, reshapeOp.getResult().getType());
    }
    if (auto broadcastOp = dyn_cast<cuda_tile::BroadcastOp>(def)) {
      std::optional<TileExpr> source =
          getExactTileExpr(broadcastOp.getSource(), loop);
      if (!source)
        return std::nullopt;
      return broadcastTileExpr(*source, broadcastOp.getResult().getType());
    }
    if (std::optional<TileExpr> binary = getExactBinaryTileExpr(def, loop))
      return binary;
  }

  return std::nullopt;
}

LinearExpr add(LinearExpr lhs, LinearExpr rhs) {
  return {lhs.ivCoeff + rhs.ivCoeff, lhs.minOffset + rhs.minOffset,
          lhs.maxOffset + rhs.maxOffset};
}

LinearExpr sub(LinearExpr lhs, LinearExpr rhs) {
  return {lhs.ivCoeff - rhs.ivCoeff, lhs.minOffset - rhs.maxOffset,
          lhs.maxOffset - rhs.minOffset};
}

LinearExpr mulByConstant(LinearExpr expr, int64_t constant) {
  int64_t a = expr.minOffset * constant;
  int64_t b = expr.maxOffset * constant;
  return {expr.ivCoeff * constant, std::min(a, b), std::max(a, b)};
}

std::optional<LinearExpr> getLinearExpr(Value v, cuda_tile::ForOp loop);

std::optional<LinearExpr> getBinaryLinearExpr(Operation *op,
                                              cuda_tile::ForOp loop) {
  if (auto addOp = dyn_cast<cuda_tile::AddIOp>(op)) {
    std::optional<LinearExpr> lhs = getLinearExpr(addOp.getLhs(), loop);
    std::optional<LinearExpr> rhs = getLinearExpr(addOp.getRhs(), loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return add(*lhs, *rhs);
  }
  if (auto subOp = dyn_cast<cuda_tile::SubIOp>(op)) {
    std::optional<LinearExpr> lhs = getLinearExpr(subOp.getLhs(), loop);
    std::optional<LinearExpr> rhs = getLinearExpr(subOp.getRhs(), loop);
    if (!lhs || !rhs)
      return std::nullopt;
    return sub(*lhs, *rhs);
  }
  if (auto mulOp = dyn_cast<cuda_tile::MulIOp>(op)) {
    if (std::optional<int64_t> lhsConst = getSplatConstant(mulOp.getLhs())) {
      std::optional<LinearExpr> rhs = getLinearExpr(mulOp.getRhs(), loop);
      if (!rhs)
        return std::nullopt;
      return mulByConstant(*rhs, *lhsConst);
    }
    if (std::optional<int64_t> rhsConst = getSplatConstant(mulOp.getRhs())) {
      std::optional<LinearExpr> lhs = getLinearExpr(mulOp.getLhs(), loop);
      if (!lhs)
        return std::nullopt;
      return mulByConstant(*lhs, *rhsConst);
    }
    return std::nullopt;
  }
  return std::nullopt;
}

std::optional<LinearExpr> getLinearExpr(Value v, cuda_tile::ForOp loop) {
  Value loopIv = loop.getBody()->getArgument(0);
  if (v == loopIv)
    return LinearExpr{/*ivCoeff=*/1, /*minOffset=*/0, /*maxOffset=*/0};

  if (std::optional<std::pair<int64_t, int64_t>> minMax =
          getConstantMinMax(v)) {
    return LinearExpr{/*ivCoeff=*/0, minMax->first, minMax->second};
  }

  if (auto iotaOp = v.getDefiningOp<cuda_tile::IotaOp>()) {
    int64_t count = getTileElementCount(iotaOp.getResult().getType());
    if (count < 1)
      return std::nullopt;
    return LinearExpr{/*ivCoeff=*/0, /*minOffset=*/0,
                      /*maxOffset=*/count - 1};
  }

  if (Operation *def = v.getDefiningOp()) {
    if (auto assumeOp = dyn_cast<cuda_tile::AssumeOp>(def))
      return getLinearExpr(assumeOp.getValue(), loop);
    if (auto broadcastOp = dyn_cast<cuda_tile::BroadcastOp>(def))
      return getLinearExpr(broadcastOp.getSource(), loop);
    if (auto reshapeOp = dyn_cast<cuda_tile::ReshapeOp>(def))
      return getLinearExpr(reshapeOp.getSource(), loop);
    if (std::optional<LinearExpr> binary = getBinaryLinearExpr(def, loop))
      return binary;
  }

  // Scalar loop-invariant values are shared by the two sides of a
  // loop-carried self-query, so they cancel out of the collision equation.
  // Vector invariants can vary per lane; unless handled above as constants or
  // iota-like lane intervals, they are not a safe affine interval.
  if (isLoopInvariant(v, loop) && isSingleElementTile(v.getType()))
    return LinearExpr{};

  return std::nullopt;
}

std::optional<int64_t> getLoopStep(cuda_tile::ForOp loop) {
  std::optional<int64_t> step = getSplatConstant(loop.getStep());
  if (!step || *step == 0)
    return std::nullopt;
  return *step;
}

DependenceResult proveIntervalDisjoint(LinearExpr expr, int64_t loopStep,
                                       DisjointWitness witness) {
  if (expr.ivCoeff == 0)
    return DependenceResult::possible();

  int64_t width = expr.maxOffset - expr.minOffset + 1;
  if (width <= 0)
    return DependenceResult::unsupported();

  int64_t minIvSeparation = std::abs(expr.ivCoeff * loopStep);
  if (minIvSeparation >= width)
    return DependenceResult::proven(witness);
  return DependenceResult::possible();
}

std::optional<int64_t> getLoopTripCount(cuda_tile::ForOp loop) {
  std::optional<int64_t> lower = getSplatConstant(loop.getLowerBound());
  std::optional<int64_t> upper = getSplatConstant(loop.getUpperBound());
  std::optional<int64_t> step = getLoopStep(loop);
  if (!lower || !upper || !step)
    return std::nullopt;

  if (*step > 0) {
    if (*upper <= *lower)
      return 0;
    return llvm::divideCeil(*upper - *lower, *step);
  }

  if (*lower <= *upper)
    return 0;
  return llvm::divideCeil(*lower - *upper, -*step);
}

DependenceResult proveExactSetDisjoint(TileExpr expr, int64_t loopStep,
                                       std::optional<int64_t> tripCount) {
  constexpr int64_t kMaxExactElements = 4096;
  constexpr int64_t kMaxTripCount = 256;
  constexpr int64_t kMaxExactChecks = 1'000'000;

  if (expr.elements.empty() ||
      int64_t(expr.elements.size()) > kMaxExactElements)
    return DependenceResult::unsupported();

  int64_t ivCoeff = expr.elements.front().ivCoeff;
  for (ElementExpr element : expr.elements) {
    if (element.ivCoeff != ivCoeff)
      return DependenceResult::unsupported();
  }
  if (ivCoeff == 0)
    return DependenceResult::possible();

  int64_t shiftPerIter = ivCoeff * loopStep;
  if (shiftPerIter == 0)
    return DependenceResult::possible();

  if (!tripCount)
    return DependenceResult::unsupported();
  if (*tripCount <= 1) {
    return DependenceResult::proven(
        DisjointWitness{DisjointWitness::Kind::PointerAffineSetDisjoint});
  }
  if (*tripCount > kMaxTripCount)
    return DependenceResult::unsupported();
  if (int64_t(expr.elements.size()) * *tripCount > kMaxExactChecks)
    return DependenceResult::unsupported();

  llvm::DenseSet<int64_t> offsets;
  offsets.reserve(expr.elements.size());
  for (ElementExpr element : expr.elements)
    offsets.insert(element.offset);

  for (int64_t iterDelta = 1; iterDelta < *tripCount; ++iterDelta) {
    int64_t shift = iterDelta * shiftPerIter;
    for (int64_t offset : offsets) {
      if (offsets.contains(offset + shift))
        return DependenceResult::possible();
    }
  }

  return DependenceResult::proven(
      DisjointWitness{DisjointWitness::Kind::PointerAffineSetDisjoint});
}

/// Return the unique IV-dependent view index dimension. Non-IV dimensions may
/// contain loop-invariant dynamic values; they are shared by both sides of the
/// self-query and do not affect cross-iteration uniqueness.
std::optional<std::pair<unsigned, int64_t>>
findUniqueIvIndexedDim(cuda_tile::StoreViewTkoOp storeOp,
                       cuda_tile::ForOp loop) {
  std::optional<std::pair<unsigned, int64_t>> ivDim;
  for (auto [dim, index] : llvm::enumerate(storeOp.getIndex())) {
    std::optional<LinearExpr> expr = getLinearExpr(index, loop);
    if (!expr)
      return std::nullopt;
    if (expr->ivCoeff == 0)
      continue;
    if (ivDim)
      return std::nullopt;
    ivDim = std::pair<unsigned, int64_t>(dim, expr->ivCoeff);
  }
  return ivDim;
}

DependenceResult provePartitionViewStore(cuda_tile::StoreViewTkoOp storeOp,
                                         cuda_tile::ForOp loop) {
  std::optional<std::pair<unsigned, int64_t>> ivDim =
      findUniqueIvIndexedDim(storeOp, loop);
  if (!ivDim)
    return DependenceResult::possible();
  std::optional<int64_t> step = getLoopStep(loop);
  if (!step)
    return DependenceResult::unsupported();
  if (ivDim->second * *step == 0)
    return DependenceResult::possible();
  return DependenceResult::proven(
      DisjointWitness{DisjointWitness::Kind::PartitionUniqueness});
}

DependenceResult proveStridedViewStore(cuda_tile::StoreViewTkoOp storeOp,
                                       cuda_tile::ForOp loop,
                                       cuda_tile::StridedViewType viewTy) {
  std::optional<std::pair<unsigned, int64_t>> ivDim =
      findUniqueIvIndexedDim(storeOp, loop);
  if (!ivDim)
    return DependenceResult::possible();

  std::optional<int64_t> step = getLoopStep(loop);
  if (!step)
    return DependenceResult::unsupported();

  ArrayRef<int32_t> tileShape = viewTy.getTileShape();
  ArrayRef<int32_t> traversalStrides = viewTy.getTraversalStrides();
  unsigned dim = ivDim->first;
  if (dim >= tileShape.size() || dim >= traversalStrides.size())
    return DependenceResult::unsupported();

  int64_t movement =
      std::abs(ivDim->second * *step * int64_t(traversalStrides[dim]));
  if (movement >= int64_t(tileShape[dim]))
    return DependenceResult::proven(
        DisjointWitness{DisjointWitness::Kind::AffineStrideDominates});
  return DependenceResult::possible();
}

Value stripPointerPassthrough(Value v) {
  while (Operation *def = v.getDefiningOp()) {
    if (auto assumeOp = dyn_cast<cuda_tile::AssumeOp>(def)) {
      v = assumeOp.getValue();
      continue;
    }
    if (auto ptrToPtrOp = dyn_cast<cuda_tile::PtrToPtrOp>(def)) {
      v = ptrToPtrOp.getSource();
      continue;
    }
    if (auto broadcastOp = dyn_cast<cuda_tile::BroadcastOp>(def)) {
      v = broadcastOp.getSource();
      continue;
    }
    if (auto reshapeOp = dyn_cast<cuda_tile::ReshapeOp>(def)) {
      v = reshapeOp.getSource();
      continue;
    }
    break;
  }
  return v;
}

std::optional<LinearExpr> getPointerOffsetExpr(Value ptr,
                                               cuda_tile::ForOp loop) {
  ptr = stripPointerPassthrough(ptr);
  if (auto offsetOp = ptr.getDefiningOp<cuda_tile::OffsetOp>()) {
    std::optional<LinearExpr> base =
        getPointerOffsetExpr(offsetOp.getPtr(), loop);
    std::optional<LinearExpr> offset =
        getLinearExpr(offsetOp.getOffset(), loop);
    if (!base || !offset)
      return std::nullopt;
    return add(*base, *offset);
  }

  if (!isLoopInvariant(ptr, loop))
    return std::nullopt;

  // A loop-invariant vector of base pointers may contain arbitrary per-lane
  // addresses, so shifting it by an IV-derived scalar is not enough to prove
  // lane-wise non-overlap. Require the invariant base to be scalar/splat-like
  // after stripping reshape/broadcast.
  if (!isSingleElementTile(ptr.getType()))
    return std::nullopt;
  return LinearExpr{};
}

std::optional<TileExpr> getExactPointerOffsetExpr(Value ptr,
                                                  cuda_tile::ForOp loop) {
  Type originalType = ptr.getType();
  ptr = stripPointerPassthrough(ptr);
  if (auto offsetOp = ptr.getDefiningOp<cuda_tile::OffsetOp>()) {
    std::optional<TileExpr> base =
        getExactPointerOffsetExpr(offsetOp.getPtr(), loop);
    std::optional<TileExpr> offset =
        getExactTileExpr(offsetOp.getOffset(), loop);
    if (!base || !offset)
      return std::nullopt;
    return combineSameShape(*base, *offset,
                            [](ElementExpr lhs, ElementExpr rhs) {
                              return ElementExpr{lhs.ivCoeff + rhs.ivCoeff,
                                                 lhs.offset + rhs.offset};
                            });
  }

  if (!isLoopInvariant(ptr, loop) || !isSingleElementTile(ptr.getType()))
    return std::nullopt;

  std::optional<SmallVector<int64_t>> shape = getTileShape(originalType);
  if (!shape)
    return std::nullopt;
  TileExpr out;
  out.shape = *shape;
  out.elements.assign(getElementCount(out.shape), ElementExpr{});
  return out;
}

DependenceResult provePtrStore(cuda_tile::StorePtrTkoOp storeOp,
                               cuda_tile::ForOp loop) {
  std::optional<int64_t> step = getLoopStep(loop);
  if (!step)
    return DependenceResult::unsupported();

  if (std::optional<TileExpr> exact =
          getExactPointerOffsetExpr(storeOp.getDestination(), loop)) {
    DependenceResult result =
        proveExactSetDisjoint(*exact, *step, getLoopTripCount(loop));
    if (result.kind != DependenceResultKind::Unsupported)
      return result;
  }

  std::optional<LinearExpr> expr =
      getPointerOffsetExpr(storeOp.getDestination(), loop);
  if (!expr)
    return DependenceResult::unsupported();

  return proveIntervalDisjoint(
      *expr, *step,
      DisjointWitness{DisjointWitness::Kind::PointerAffineStrideDominates});
}

DependenceResult proveStoreLoopCarriedDisjoint(Operation *op,
                                               cuda_tile::ForOp loop) {
  if (auto storeOp = dyn_cast<cuda_tile::StoreViewTkoOp>(op)) {
    if (storeOp.getMemoryOrderingSemantics() !=
        cuda_tile::MemoryOrderingSemantics::WEAK)
      return DependenceResult::possible();
    Type viewTy = storeOp.getView().getType();
    if (isa<cuda_tile::PartitionViewType>(viewTy))
      return provePartitionViewStore(storeOp, loop);
    if (auto stridedTy = dyn_cast<cuda_tile::StridedViewType>(viewTy))
      return proveStridedViewStore(storeOp, loop, stridedTy);
    return DependenceResult::unsupported();
  }

  if (auto storeOp = dyn_cast<cuda_tile::StorePtrTkoOp>(op)) {
    if (storeOp.getMemoryOrderingSemantics() !=
        cuda_tile::MemoryOrderingSemantics::WEAK)
      return DependenceResult::possible();
    return provePtrStore(storeOp, loop);
  }

  return DependenceResult::unsupported();
}

} // namespace

//===----------------------------------------------------------------------===//
// TkoDependenceAnalysis
//===----------------------------------------------------------------------===//

/*
 * Loop-carried store proof boundary.
 *
 * This analysis proves one fact about one concrete store op: dynamic instances
 * of that store in distinct iterations of `loop` do not overlap. It does not
 * decide whether the store is the only relevant effect on a memory root.
 * AutoGenMemoryToken performs that token-specific candidate selection before
 * calling this query.
 *
 * Supported proof families are intentionally local:
 *   - partition-view stores with a loop-IV partition index;
 *   - strided-view stores whose IV stride dominates the tile footprint;
 *   - pointer stores reducible to IV-stride plus a lane interval or finite
 *     per-lane offset set.
 */
bool TkoDependenceAnalysis::isLoopCarriedStoreDisjoint(cuda_tile::ForOp loop,
                                                       Operation *storeOp) {
  if (!storeOp || storeOp->getBlock() != loop.getBody())
    return false;

  CacheKey key{loop.getOperation(), storeOp};
  auto it = loopCarriedStoreCache.find(key);
  if (it != loopCarriedStoreCache.end())
    return it->second;

  bool proven = proveStoreLoopCarriedDisjoint(storeOp, loop).kind ==
                DependenceResultKind::ProvenDisjoint;
  loopCarriedStoreCache[key] = proven;
  return proven;
}
