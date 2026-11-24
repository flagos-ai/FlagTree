#include "triton/Analysis/NewAnalysis/StructuredIdAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>

using namespace mlir;

namespace mlir {
namespace triton {
namespace sdnn {

namespace {

/// One additive term of a decoded tile: an integer literal or an i32 scalar.
///
/// Why keep literals symbolic instead of materialising them on the spot: the
/// two ramps of `offs_a[:, None] - offs_a[None, :]` are both literals, so 0 and
/// 1 cancel out during the decode and a rejected tile leaves no
/// `arith.constant` behind in the IR.  Only a term that a runtime scalar
/// actually enters has to become an SSA value.
///
/// How:
///   - `literal` is set when the term is known here and now;
///   - otherwise `value` holds an i32 scalar (a block argument, a `tt.load`
///     result, ...), which needs no proof because the EW coefficient slot takes
///     an i32 SSA value as-is.
struct Coeff {
  std::optional<int64_t> literal;
  Value value;

  bool isLiteral() const { return literal.has_value(); }
  bool isZero() const { return isLiteral() && *literal == 0; }
};

/// A tile decoded as an affine function of its own indices.
///
/// `value(x_0, ..., x_{r-1}) = constant + sum_d steps[d] * x_d`, where `r` is
/// the rank of the decoded value (0 for a scalar).
struct AffineTile {
  Coeff constant;
  SmallVector<Coeff, 2> steps;
};

static Coeff makeLiteral(int64_t value) { return Coeff{value, Value()}; }

/// Read `v` as a scalar integer literal.
static std::optional<int64_t> readIntLiteral(Value v) {
  APInt bits;
  if (!matchPattern(v, m_ConstantInt(&bits)))
    return std::nullopt;
  if (bits.getBitWidth() > 64)
    return std::nullopt;
  return bits.getSExtValue();
}

/// Read an integer literal attribute; `std::nullopt` when it is too wide.
static std::optional<int64_t> readIntLiteral(APInt bits) {
  if (bits.getBitWidth() > 64)
    return std::nullopt;
  return bits.getSExtValue();
}

/// Build the term for an existing SSA value, folding it when it is a literal.
static Coeff makeCoeff(Value v) {
  if (std::optional<int64_t> literal = readIntLiteral(v))
    return Coeff{literal, Value()};
  return Coeff{std::nullopt, v};
}

/// A tile whose every element is `constant`.
static AffineTile splatTile(Coeff constant, int64_t rank) {
  AffineTile tile;
  tile.constant = constant;
  tile.steps.assign(rank, makeLiteral(0));
  return tile;
}

/// Tell whether `type` is a value this decoder models.
///
/// Why i32 only, and rank at most 2: every coefficient handed out is an i32
/// scalar and the EW coefficient register is f32, so an i64 index chain would
/// have to be narrowed first -- and nothing in the frontend produces one
/// (`tt.make_range` is i32).  The rank bound is the generator's: `ssaux` walks
/// one row per stream, so `verifyStructuredIdShape` rejects rank 0 and rank
/// > 2.
static bool isDecodableType(Type type) {
  if (type.isInteger(32))
    return true;
  auto shaped = dyn_cast<RankedTensorType>(type);
  if (!shaped || !shaped.hasStaticShape() || shaped.getRank() > 2)
    return false;
  return shaped.getElementType().isInteger(32);
}

/// True when `src` and `dst` differ only in extents that are 1 in `src`.
static bool isBroadcastOf(ShapedType src, ShapedType dst) {
  if (src.getRank() != dst.getRank())
    return false;
  for (auto [dim, extent] : llvm::enumerate(dst.getShape()))
    if (src.getShape()[dim] != 1 && src.getShape()[dim] != extent)
      return false;
  return true;
}

/// Zero the steps of dims that only ever read index 0.
///
/// Why: a unit-extent dim contributes `step * 0` to every element, so the step
/// is dead weight in the generated op -- and on a `[M, 1]` column it would make
/// the tile look like it advances along the column axis when it does not.
static void dropUnitDimSteps(AffineTile &tile, ArrayRef<int64_t> shape) {
  for (auto [dim, extent] : llvm::enumerate(shape))
    if (extent == 1)
      tile.steps[dim] = makeLiteral(0);
}

/// Walk a whitelisted TTIR integer tile into `constant + sum_d steps[d] * x_d`.
///
/// Why a class rather than free functions: the decode needs a builder (to
/// materialise coefficients), a location and a rollback list across a
/// recursive walk, and every step has to report failure on its own so one
/// unsupported op rejects the whole tile instead of half of it.
///
/// How:
///   - `decode` dispatches on the defining op, one family at a time;
///   - each family either folds into the affine form or fails;
///   - `rollback` erases whatever the walk materialised when the tile is
///     finally rejected, so a failure leaves the IR untouched.
class TileDecoder {
public:
  TileDecoder(OpBuilder &builder, Location loc)
      : builder_(builder), loc_(loc) {}

  FailureOr<AffineTile> decode(Value tile);

  /// Erase every op the walk created, innermost first.
  void rollback();

  /// Turn a term into an i32 scalar SSA value (one op per distinct literal).
  Value materialize(Coeff coeff);

private:
  FailureOr<AffineTile> decodeShaped(Value tile);
  FailureOr<AffineTile> decodeLeaf(Value tile, Operation *def);
  FailureOr<AffineTile> decodeReshape(Value tile, Operation *def);
  FailureOr<AffineTile> decodeArithmetic(Value tile, Operation *def);

  FailureOr<AffineTile> decodeConstant(arith::ConstantOp op);
  FailureOr<AffineTile> decodeMakeRange(triton::MakeRangeOp op);
  FailureOr<AffineTile> decodeSplat(triton::SplatOp op);
  FailureOr<AffineTile> decodeExpandDims(triton::ExpandDimsOp op);
  FailureOr<AffineTile> decodeBroadcast(triton::BroadcastOp op);
  FailureOr<AffineTile> decodeExpandShape(tensor::ExpandShapeOp op);

  FailureOr<AffineTile> decodeAdd(arith::AddIOp op);
  FailureOr<AffineTile> decodeSub(arith::SubIOp op);
  FailureOr<AffineTile> decodeMul(arith::MulIOp op);

  FailureOr<AffineTile> combineTiles(const AffineTile &lhs,
                                     const AffineTile &rhs, bool isSub);
  FailureOr<AffineTile> scaleTile(const AffineTile &tile, Coeff factor);
  Coeff addCoeffs(Coeff lhs, Coeff rhs, bool isSub);
  Coeff mulCoeffs(Coeff lhs, Coeff rhs);

  OpBuilder &builder_;
  Location loc_;
  SmallVector<Operation *> materialized_;
  llvm::DenseMap<int64_t, Value> literalCache_;
};

FailureOr<AffineTile> TileDecoder::decode(Value tile) {
  if (!isDecodableType(tile.getType()))
    return failure();
  // A scalar is a uniform term whatever produced it: the coefficient slot
  // takes an i32 SSA value, so there is nothing to prove about it.
  if (!isa<ShapedType>(tile.getType()))
    return splatTile(makeCoeff(tile), /*rank=*/0);
  return decodeShaped(tile);
}

FailureOr<AffineTile> TileDecoder::decodeShaped(Value tile) {
  Operation *def = tile.getDefiningOp();
  if (!def)
    return failure();
  if (isa<arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(def))
    return decodeLeaf(tile, def);
  if (isa<triton::ExpandDimsOp, triton::BroadcastOp, tensor::ExpandShapeOp>(
          def))
    return decodeReshape(tile, def);
  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp>(def))
    return decodeArithmetic(tile, def);
  // A tensor block argument, `divsi` / `remsi`, `tt.load`, `arith.select` or
  // any op we do not know: none of them is affine by construction.
  return failure();
}

FailureOr<AffineTile> TileDecoder::decodeLeaf(Value tile, Operation *def) {
  if (auto op = dyn_cast<arith::ConstantOp>(def))
    return decodeConstant(op);
  if (auto op = dyn_cast<triton::MakeRangeOp>(def))
    return decodeMakeRange(op);
  if (auto op = dyn_cast<triton::SplatOp>(def))
    return decodeSplat(op);
  return failure();
}

FailureOr<AffineTile> TileDecoder::decodeReshape(Value tile, Operation *def) {
  if (auto op = dyn_cast<triton::ExpandDimsOp>(def))
    return decodeExpandDims(op);
  if (auto op = dyn_cast<triton::BroadcastOp>(def))
    return decodeBroadcast(op);
  if (auto op = dyn_cast<tensor::ExpandShapeOp>(def))
    return decodeExpandShape(op);
  return failure();
}

FailureOr<AffineTile> TileDecoder::decodeArithmetic(Value tile,
                                                    Operation *def) {
  if (auto op = dyn_cast<arith::AddIOp>(def))
    return decodeAdd(op);
  if (auto op = dyn_cast<arith::SubIOp>(def))
    return decodeSub(op);
  if (auto op = dyn_cast<arith::MulIOp>(def))
    return decodeMul(op);
  return failure();
}

FailureOr<AffineTile> TileDecoder::decodeConstant(arith::ConstantOp op) {
  auto shaped = dyn_cast<RankedTensorType>(op.getType());
  if (!shaped)
    return splatTile(makeCoeff(op.getResult()), /*rank=*/0);
  // Only a splat: a dense ramp (`dense<[0, 1, 2, 3]>`) is a structured id too,
  // but the frontend never emits one and accepting it would mean proving the
  // literal against the affine form element by element.
  auto dense = dyn_cast<DenseIntElementsAttr>(op.getValue());
  if (!dense || !dense.isSplat())
    return failure();
  std::optional<int64_t> splat = readIntLiteral(dense.getSplatValue<APInt>());
  if (!splat)
    return failure();
  return splatTile(Coeff{splat, Value()}, shaped.getRank());
}

FailureOr<AffineTile> TileDecoder::decodeMakeRange(triton::MakeRangeOp op) {
  auto shaped = dyn_cast<RankedTensorType>(op.getType());
  if (!shaped || shaped.getRank() != 1)
    return failure();
  // `getStart()` is generated as uint32_t; the attribute keeps the sign.
  int64_t start = op.getStartAttr().getInt();
  AffineTile tile;
  tile.constant = makeLiteral(start);
  tile.steps.push_back(makeLiteral(1));
  return tile;
}

FailureOr<AffineTile> TileDecoder::decodeSplat(triton::SplatOp op) {
  FailureOr<AffineTile> src = decode(op.getSrc());
  if (failed(src))
    return failure();
  auto shaped = dyn_cast<RankedTensorType>(op.getType());
  if (!shaped)
    return failure();
  return splatTile(src->constant, shaped.getRank());
}

FailureOr<AffineTile> TileDecoder::decodeExpandDims(triton::ExpandDimsOp op) {
  FailureOr<AffineTile> src = decode(op.getSrc());
  if (failed(src))
    return failure();
  int64_t axis = op.getAxis();
  if (axis < 0 || axis > static_cast<int64_t>(src->steps.size()))
    return failure();
  AffineTile tile = *src;
  tile.steps.insert(tile.steps.begin() + axis, makeLiteral(0));
  return tile;
}

FailureOr<AffineTile> TileDecoder::decodeBroadcast(triton::BroadcastOp op) {
  FailureOr<AffineTile> src = decode(op.getSrc());
  if (failed(src))
    return failure();
  auto dstType = dyn_cast<RankedTensorType>(op.getType());
  auto srcType = dyn_cast<RankedTensorType>(op.getSrc().getType());
  if (!dstType || !srcType || !isBroadcastOf(srcType, dstType))
    return failure();
  AffineTile tile = *src;
  for (auto [dim, extent] : llvm::enumerate(dstType.getShape()))
    // The expanded dim only ever reads index 0 of the source, so its ramp
    // contributes nothing along the new axis.
    if (srcType.getShape()[dim] == 1)
      tile.steps[dim] = makeLiteral(0);
  return tile;
}

FailureOr<AffineTile> TileDecoder::decodeExpandShape(tensor::ExpandShapeOp op) {
  FailureOr<AffineTile> src = decode(op.getSrc());
  if (failed(src))
    return failure();
  auto dstType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!dstType)
    return failure();
  // `expand_shape` keeps one reassociation group per *source* dim, listing the
  // destination dims it was split into.  A source index is the flattened
  // destination index inside its group:
  //   srcIndex = sum_m dstIndex(m) * prod_{l>m} extent(l)
  // so every destination dim of the group carries the source step scaled by
  // the extents that follow it.
  // NOTE: `getReassociationIndices()` returns the groups *by value* in this
  // MLIR, so an `ArrayRef` binding would dangle.
  SmallVector<ReassociationIndices, 4> groups = op.getReassociationIndices();
  if (groups.size() != src->steps.size())
    return failure();
  AffineTile tile;
  tile.constant = src->constant;
  tile.steps.assign(dstType.getRank(), makeLiteral(0));
  for (auto [srcDim, group] : llvm::enumerate(groups)) {
    int64_t stride = 1;
    for (int64_t member = static_cast<int64_t>(group.size()) - 1; member >= 0;
         --member) {
      int64_t dstDim = group[member];
      if (dstDim >= dstType.getRank())
        return failure();
      Coeff scaled = mulCoeffs(src->steps[srcDim], makeLiteral(stride));
      tile.steps[dstDim] =
          addCoeffs(tile.steps[dstDim], scaled, /*isSub=*/false);
      stride *= dstType.getShape()[dstDim];
    }
  }
  return tile;
}

FailureOr<AffineTile> TileDecoder::decodeAdd(arith::AddIOp op) {
  FailureOr<AffineTile> lhs = decode(op.getLhs());
  FailureOr<AffineTile> rhs = decode(op.getRhs());
  if (failed(lhs) || failed(rhs))
    return failure();
  return combineTiles(*lhs, *rhs, /*isSub=*/false);
}

FailureOr<AffineTile> TileDecoder::decodeSub(arith::SubIOp op) {
  FailureOr<AffineTile> lhs = decode(op.getLhs());
  FailureOr<AffineTile> rhs = decode(op.getRhs());
  if (failed(lhs) || failed(rhs))
    return failure();
  return combineTiles(*lhs, *rhs, /*isSub=*/true);
}

FailureOr<AffineTile> TileDecoder::decodeMul(arith::MulIOp op) {
  FailureOr<AffineTile> lhs = decode(op.getLhs());
  FailureOr<AffineTile> rhs = decode(op.getRhs());
  if (failed(lhs) || failed(rhs))
    return failure();
  // One side has to be a constant: a product of two index-dependent tiles
  // (`i * j`) is not affine, and decoding it partially would emit a tile that
  // ignores one factor.
  bool lhsIsConstant =
      llvm::all_of(lhs->steps, [](const Coeff &step) { return step.isZero(); });
  if (lhsIsConstant && lhs->constant.isLiteral())
    return scaleTile(*rhs, lhs->constant);
  bool rhsIsConstant =
      llvm::all_of(rhs->steps, [](const Coeff &step) { return step.isZero(); });
  if (rhsIsConstant && rhs->constant.isLiteral())
    return scaleTile(*lhs, rhs->constant);
  return failure();
}

FailureOr<AffineTile> TileDecoder::combineTiles(const AffineTile &lhs,
                                                const AffineTile &rhs,
                                                bool isSub) {
  if (lhs.steps.size() != rhs.steps.size())
    return failure();
  AffineTile tile;
  tile.constant = addCoeffs(lhs.constant, rhs.constant, isSub);
  for (auto [dim, step] : llvm::enumerate(lhs.steps))
    tile.steps.push_back(addCoeffs(step, rhs.steps[dim], isSub));
  return tile;
}

FailureOr<AffineTile> TileDecoder::scaleTile(const AffineTile &tile,
                                             Coeff factor) {
  AffineTile scaled;
  scaled.constant = mulCoeffs(tile.constant, factor);
  for (const Coeff &step : tile.steps)
    scaled.steps.push_back(mulCoeffs(step, factor));
  return scaled;
}

Coeff TileDecoder::addCoeffs(Coeff lhs, Coeff rhs, bool isSub) {
  if (lhs.isLiteral() && rhs.isLiteral())
    return makeLiteral(isSub ? *lhs.literal - *rhs.literal
                             : *lhs.literal + *rhs.literal);
  if (rhs.isZero())
    return lhs;
  if (!isSub && lhs.isZero())
    return rhs;
  Value lhsValue = materialize(lhs);
  Value rhsValue = materialize(rhs);
  Operation *sum =
      isSub ? builder_.create<arith::SubIOp>(loc_, lhsValue, rhsValue)
            : builder_.create<arith::AddIOp>(loc_, lhsValue, rhsValue);
  materialized_.push_back(sum);
  return Coeff{std::nullopt, sum->getResult(0)};
}

Coeff TileDecoder::mulCoeffs(Coeff lhs, Coeff rhs) {
  if (lhs.isLiteral() && rhs.isLiteral())
    return makeLiteral(*lhs.literal * *rhs.literal);
  if (lhs.isLiteral() && *lhs.literal == 1)
    return rhs;
  if (rhs.isLiteral() && *rhs.literal == 1)
    return lhs;
  if (lhs.isZero() || rhs.isZero())
    return makeLiteral(0);
  Value lhsValue = materialize(lhs);
  Value rhsValue = materialize(rhs);
  Operation *product = builder_.create<arith::MulIOp>(loc_, lhsValue, rhsValue);
  materialized_.push_back(product);
  return Coeff{std::nullopt, product->getResult(0)};
}

Value TileDecoder::materialize(Coeff coeff) {
  if (!coeff.isLiteral())
    return coeff.value;
  auto cached = literalCache_.find(*coeff.literal);
  if (cached != literalCache_.end())
    return cached->second;
  // NOTE: the cast is a wrap-around, matching what the i32 chain this replaces
  // would have computed. A coefficient that does not fit 32 bits has already
  // overflowed in the original IR, so there is no wider value to preserve.
  auto constant = builder_.create<arith::ConstantOp>(
      loc_, builder_.getI32IntegerAttr(static_cast<int32_t>(*coeff.literal)));
  materialized_.push_back(constant);
  literalCache_.insert({*coeff.literal, constant.getResult()});
  return constant.getResult();
}

void TileDecoder::rollback() {
  for (Operation *op : llvm::reverse(materialized_))
    if (op->use_empty())
      op->erase();
  materialized_.clear();
  literalCache_.clear();
}

/// The rank `sdnn.ew structured_ids` can write `tile` at, if any.
///
/// Why 1 and 2 only: `ssaux` walks one row per stream, so
/// `verifyStructuredIdShape` rejects rank 0 and rank > 2.
static std::optional<int64_t> getGeneratableTileRank(Value tile) {
  auto tileType = dyn_cast<RankedTensorType>(tile.getType());
  if (!tileType)
    return std::nullopt;
  int64_t rank = tileType.getRank();
  if (rank != 1 && rank != 2)
    return std::nullopt;
  return rank;
}

/// Put `builder` where a coefficient of `tile` has to be materialised.
///
/// Why before the root op: the coefficients are `arith.constant` i32 ops, and
/// placing them there makes them dominate both the tile and its consumer.
static void setInsertionPointForCoeffs(OpBuilder &builder, Value tile) {
  if (Operation *root = tile.getDefiningOp())
    builder.setInsertionPoint(root);
  else
    builder.setInsertionPointToStart(tile.getParentBlock());
}

} // namespace

namespace detail {

LogicalResult analyze(Value tile, StructuredId &ids) {
  std::optional<int64_t> rank = getGeneratableTileRank(tile);
  if (!rank)
    return failure();

  // The coefficients are materialised as `arith.constant` i32 ops just before
  // the tile's root op, so they dominate both the tile and its consumer.
  OpBuilder builder(tile.getContext());
  setInsertionPointForCoeffs(builder, tile);

  TileDecoder decoder(builder, tile.getLoc());
  FailureOr<AffineTile> decoded = decoder.decode(tile);
  if (failed(decoded) || static_cast<int64_t>(decoded->steps.size()) != *rank) {
    decoder.rollback();
    return failure();
  }

  dropUnitDimSteps(*decoded, cast<RankedTensorType>(tile.getType()).getShape());

  ids.base = decoder.materialize(decoded->constant);
  ids.stepCol =
      decoder.materialize(*rank == 1 ? decoded->steps[0] : decoded->steps[1]);
  // A rank-1 tile is a single row: nothing advances it along a row axis.
  ids.stepRow =
      decoder.materialize(*rank == 1 ? makeLiteral(0) : decoded->steps[0]);
  return success();
}

} // namespace detail

LogicalResult analyze(Value tile, StructuredId &ids) {
  return detail::analyze(tile, ids);
}

bool isAffineIdTile(Value tile) {
  std::optional<int64_t> rank = getGeneratableTileRank(tile);
  if (!rank)
    return false;

  OpBuilder builder(tile.getContext());
  setInsertionPointForCoeffs(builder, tile);

  TileDecoder decoder(builder, tile.getLoc());
  FailureOr<AffineTile> decoded = decoder.decode(tile);
  bool isAffine = succeeded(decoded) &&
                  static_cast<int64_t>(decoded->steps.size()) == *rank;
  // Unconditionally: the probe answers a question, it does not rewrite.  No
  // coefficient slot was materialised, so every op the decode built for an
  // intermediate sum or product is dead and `rollback` erases all of them.
  decoder.rollback();
  return isAffine;
}

} // namespace sdnn
} // namespace triton
} // namespace mlir
