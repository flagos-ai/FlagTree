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

#include "TritonToGraph/PermutationAnalysis.h"

#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// This static assertion fixes the compose order in source as well as in the
// public documentation: apply [2, 0, 1], then [1, 2, 0] == identity.
constexpr int32_t kComposeBefore[] = {2, 0, 1};
constexpr int32_t kComposeAfter[] = {1, 2, 0};
static_assert(kComposeBefore[kComposeAfter[0]] == 0 &&
                  kComposeBefore[kComposeAfter[1]] == 1 &&
                  kComposeBefore[kComposeAfter[2]] == 2,
              "Permutation::compose applies the left operand before the "
              "right operand");

struct ParsedRankOneOffset {
  ProofOutcome outcome;
  triton::MakeRangeOp range;
  // Null for a fully static offset. When non-null, `staticOffset` is the
  // residual to add to this one shared dynamic scalar origin.
  Value dynamicOrigin;
  int64_t staticOffset = 0;
};

struct OffsetBounds {
  int64_t first = 0;
  int64_t last = 0;
};

struct ParsedAffineAxis {
  ProofOutcome outcome;
  StaticAccessAxis axis;
  OffsetBounds bounds;
};

bool haveSameShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getShape() == rhs.getShape();
}

bool isUnencodedRankedTensor(RankedTensorType type) {
  return type.hasStaticShape() && !type.getEncoding();
}

bool isI32Tensor(RankedTensorType type) {
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  return elementType && elementType.getWidth() == 32;
}

bool getStaticI32Constant(Value value, int64_t &result) {
  auto integerType = dyn_cast<IntegerType>(value.getType());
  if (!integerType || integerType.getWidth() != 32)
    return false;

  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer || integer.getType() != integerType)
    return false;

  result = integer.getValue().getSExtValue();
  return true;
}

bool isSignedI32(int64_t value) {
  return value >= std::numeric_limits<int32_t>::min() &&
         value <= std::numeric_limits<int32_t>::max();
}

bool isScalarI32(Value value) {
  auto integerType = dyn_cast<IntegerType>(value.getType());
  return integerType && integerType.getWidth() == 32;
}

bool getSignedI32RangeBounds(triton::MakeRangeOp range, int64_t &start,
                             int64_t &end) {
  if (!range)
    return false;

  IntegerAttr startAttr = range.getStartAttr();
  IntegerAttr endAttr = range.getEndAttr();
  auto startType =
      startAttr ? dyn_cast<IntegerType>(startAttr.getType()) : IntegerType();
  auto endType =
      endAttr ? dyn_cast<IntegerType>(endAttr.getType()) : IntegerType();
  if (!startAttr || !endAttr || !startType || !endType ||
      startType.getWidth() != 32 || endType.getWidth() != 32)
    return false;

  start = startAttr.getInt();
  end = endAttr.getInt();
  return true;
}

bool multiplyWouldOverflow(int64_t lhs, int64_t rhs) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();

  if (lhs == 0 || rhs == 0)
    return false;
  if (lhs == -1)
    return rhs == kMin;
  if (rhs == -1)
    return lhs == kMin;
  if (lhs > 0) {
    if (rhs > 0)
      return lhs > kMax / rhs;
    return rhs < kMin / lhs;
  }
  if (rhs > 0)
    return lhs < kMin / rhs;
  return lhs < kMax / rhs;
}

bool addWouldOverflow(int64_t lhs, int64_t rhs) {
  constexpr int64_t kMin = std::numeric_limits<int64_t>::min();
  constexpr int64_t kMax = std::numeric_limits<int64_t>::max();
  if (rhs > 0)
    return lhs > kMax - rhs;
  return lhs < kMin - rhs;
}

bool checkedAddI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (addWouldOverflow(lhs, rhs))
    return false;
  result = lhs + rhs;
  return true;
}

bool checkedMulI64(int64_t lhs, int64_t rhs, int64_t &result) {
  if (multiplyWouldOverflow(lhs, rhs))
    return false;
  result = lhs * rhs;
  return true;
}

// Decompose the scalar source of a rank-one offset splat into a single
// dynamic SSA origin and one signed i32 static residual. A non-add scalar is
// intentionally treated as the opaque origin: the proof relies only on SSA
// identity, never on algebraic equivalence. Scalar addi is accepted only for
// an explicit origin + constant (in either order); multiple dynamic terms and
// flagged arithmetic are rejected rather than normalized speculatively.
ParsedRankOneOffset parseScalarRankOneOrigin(Value scalar) {
  if (!isScalarI32(scalar))
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  int64_t constant = 0;
  if (getStaticI32Constant(scalar, constant))
    return {ProofOutcome::proven(), {}, Value(), constant};

  auto add = scalar.getDefiningOp<arith::AddIOp>();
  if (!add || add.getResult() != scalar)
    return {ProofOutcome::proven(), {}, scalar, 0};
  if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {ProofOutcome::rejected(ProofReason::OverflowFlags), {}, Value(), 0};

  int64_t lhsConstant = 0;
  int64_t rhsConstant = 0;
  const bool lhsIsConstant = getStaticI32Constant(add.getLhs(), lhsConstant);
  const bool rhsIsConstant = getStaticI32Constant(add.getRhs(), rhsConstant);
  if (lhsIsConstant == rhsIsConstant)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  Value origin = lhsIsConstant ? add.getRhs() : add.getLhs();
  if (!isScalarI32(origin))
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};
  return {ProofOutcome::proven(),
          {},
          origin,
          lhsIsConstant ? lhsConstant : rhsConstant};
}

// Accept the old direct make_range form, or exactly
//   addi(splat(origin + static_i32), make_range)
// in either operand order.  The caller supplies the fully checked rank-one
// offset tensor type so this helper cannot accidentally admit a broadcast or
// a differently shaped add.
ParsedRankOneOffset parseNormalizedRankOneOffset(Value offset,
                                                 RankedTensorType offsetType) {
  if (!offsetType || !isUnencodedRankedTensor(offsetType) ||
      offsetType.getRank() != 1 || !isI32Tensor(offsetType) ||
      offset.getType() != offsetType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  if (auto range = offset.getDefiningOp<triton::MakeRangeOp>()) {
    if (range.getResult() != offset ||
        range.getResult().getType() != offsetType)
      return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
              {},
              Value(),
              0};
    return {ProofOutcome::proven(), range, Value(), 0};
  }

  auto add = offset.getDefiningOp<arith::AddIOp>();
  if (!add || add.getResult() != offset ||
      add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {ProofOutcome::rejected(add && add.getResult() == offset
                                       ? ProofReason::OverflowFlags
                                       : ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};
  if (add.getLhs().getType() != offsetType ||
      add.getRhs().getType() != offsetType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  auto lhsRange = add.getLhs().getDefiningOp<triton::MakeRangeOp>();
  auto rhsRange = add.getRhs().getDefiningOp<triton::MakeRangeOp>();
  if (lhsRange && lhsRange.getResult() != add.getLhs())
    lhsRange = triton::MakeRangeOp();
  if (rhsRange && rhsRange.getResult() != add.getRhs())
    rhsRange = triton::MakeRangeOp();
  if (static_cast<bool>(lhsRange) == static_cast<bool>(rhsRange))
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  triton::MakeRangeOp range = lhsRange ? lhsRange : rhsRange;
  Value splatValue = lhsRange ? add.getRhs() : add.getLhs();
  if (range.getResult().getType() != offsetType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  auto splat = splatValue.getDefiningOp<triton::SplatOp>();
  if (!splat || splat.getResult() != splatValue ||
      splat.getResult().getType() != offsetType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset),
            {},
            Value(),
            0};

  ParsedRankOneOffset parsedOrigin = parseScalarRankOneOrigin(splat.getSrc());
  if (!parsedOrigin.outcome.isProven())
    return parsedOrigin;
  parsedOrigin.range = range;
  return parsedOrigin;
}

ParsedAffineAxis parseAffineAxis(Value value, RankedTensorType fullOffsetType) {
  auto broadcast = value.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast || broadcast.getResult() != value)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto broadcastType = dyn_cast<RankedTensorType>(value.getType());
  if (!broadcastType || !isUnencodedRankedTensor(broadcastType) ||
      !isI32Tensor(broadcastType) ||
      !haveSameShape(broadcastType, fullOffsetType))
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto multiply = broadcast.getSrc().getDefiningOp<arith::MulIOp>();
  if (!multiply || multiply.getResult() != broadcast.getSrc())
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};
  if (multiply.getOverflowFlags() != arith::IntegerOverflowFlags::none)
    return {ProofOutcome::rejected(ProofReason::OverflowFlags), {}};

  auto strideSplat = multiply.getRhs().getDefiningOp<triton::SplatOp>();
  if (!strideSplat || strideSplat.getResult() != multiply.getRhs())
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto expandedType = dyn_cast<RankedTensorType>(multiply.getLhs().getType());
  auto strideType =
      dyn_cast<RankedTensorType>(strideSplat.getResult().getType());
  if (!expandedType || !strideType || !isUnencodedRankedTensor(expandedType) ||
      !isUnencodedRankedTensor(strideType) || !isI32Tensor(expandedType) ||
      !isI32Tensor(strideType) || expandedType != strideType ||
      multiply.getResult().getType() != expandedType)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  int64_t stride = 0;
  if (!getStaticI32Constant(strideSplat.getSrc(), stride))
    return {ProofOutcome::rejected(ProofReason::DynamicStride), {}};

  Value rangeValue = multiply.getLhs();
  unsigned outputAxis = 0;
  bool hasOutputAxis = false;
  unsigned expectedRank = fullOffsetType.getRank();
  while (auto expanded = rangeValue.getDefiningOp<triton::ExpandDimsOp>()) {
    if (expanded.getResult() != rangeValue)
      return {
          ProofOutcome::rejected(ProofReason::InvalidExpandDimsChain), {}, {}};
    auto sourceType = dyn_cast<RankedTensorType>(expanded.getSrc().getType());
    auto resultType =
        dyn_cast<RankedTensorType>(expanded.getResult().getType());
    if (!sourceType || !resultType || !isUnencodedRankedTensor(sourceType) ||
        !isUnencodedRankedTensor(resultType) || !isI32Tensor(sourceType) ||
        !isI32Tensor(resultType) ||
        resultType.getRank() != sourceType.getRank() + 1 ||
        resultType.getRank() > expectedRank ||
        expanded.getAxis() >= resultType.getRank())
      return {
          ProofOutcome::rejected(ProofReason::InvalidExpandDimsChain), {}, {}};
    if (hasOutputAxis && expanded.getAxis() <= outputAxis)
      ++outputAxis;
    else if (!hasOutputAxis) {
      outputAxis = 0;
      hasOutputAxis = true;
      if (expanded.getAxis() == 0)
        ++outputAxis;
    }
    for (unsigned axis = 0; axis < resultType.getRank(); ++axis) {
      const int64_t expected =
          axis == expanded.getAxis()
              ? 1
              : sourceType
                    .getShape()[axis < expanded.getAxis() ? axis : axis - 1];
      if (resultType.getShape()[axis] != expected)
        return {ProofOutcome::rejected(ProofReason::InvalidExpandDimsChain),
                {},
                {}};
    }
    rangeValue = expanded.getSrc();
  }

  auto range = rangeValue.getDefiningOp<triton::MakeRangeOp>();
  if (!range || range.getResult() != rangeValue)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};

  auto rangeType = dyn_cast<RankedTensorType>(range.getResult().getType());
  if (!rangeType || !isUnencodedRankedTensor(rangeType) ||
      !isI32Tensor(rangeType) || rangeType.getRank() != 1)
    return {ProofOutcome::rejected(ProofReason::UnsupportedRank), {}};

  int64_t rangeStart = 0;
  int64_t rangeEnd = 0;
  if (!getSignedI32RangeBounds(range, rangeStart, rangeEnd))
    return {ProofOutcome::rejected(ProofReason::InvalidMakeRange), {}};
  const int64_t extent = rangeType.getShape().front();
  if (rangeEnd <= rangeStart || rangeEnd - rangeStart != extent)
    return {ProofOutcome::rejected(ProofReason::InvalidMakeRange), {}};

  if (!hasOutputAxis || expandedType.getRank() != expectedRank ||
      outputAxis >= expectedRank ||
      expandedType.getShape()[outputAxis] != extent)
    return {ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}};
  for (unsigned axis = 0; axis < expectedRank; ++axis) {
    if (axis != outputAxis && expandedType.getShape()[axis] != 1)
      return {
          ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset), {}, {}};
  }

  int64_t first = 0;
  int64_t last = 0;
  if (!checkedMulI64(rangeStart, stride, first) ||
      !checkedMulI64(rangeEnd - 1, stride, last))
    return {ProofOutcome::rejected(ProofReason::OffsetOverflow), {}, {}};

  return {ProofOutcome::proven(),
          StaticAccessAxis{range, rangeStart, rangeEnd, stride, outputAxis},
          OffsetBounds{first, last}};
}

StaticAccessProof rejectAccess(ProofOutcome outcome) {
  return {outcome, std::nullopt};
}

bool isBarrierLike(Operation *operation) {
  return operation->getName().getStringRef().contains("barrier");
}

// A distinct entry pointer root removes only the aliasing question. It does
// not make predicated, boundary-checked, padded, or volatile accesses safe to
// move across a delayed store, so retain the direct-access subset accepted by
// StaticAccessAnalysis before taking the ABI no-alias fast path.
bool isSupportedInterveningLoad(triton::LoadOp load) {
  return !load.getMask() && !load.getOther() &&
         load.getBoundaryCheck().empty() && !load.getPadding() &&
         !load.getIsVolatile();
}

bool isSupportedInterveningStore(triton::StoreOp store) {
  return !store.getMask() && store.getBoundaryCheck().empty();
}

} // namespace

llvm::StringRef cfg::getProofReasonMessage(ProofReason reason) {
  switch (reason) {
  case ProofReason::None:
    return "proof established";
  case ProofReason::InvalidPermutationRank:
    return "permutation rank must be non-zero and representable";
  case ProofReason::InvalidPermutationAxis:
    return "permutation contains an out-of-range axis";
  case ProofReason::DuplicatePermutationAxis:
    return "permutation contains a duplicate axis";
  case ProofReason::PermutationRankMismatch:
    return "permutations have different ranks";
  case ProofReason::ShapeRankMismatch:
    return "shape rank does not match permutation rank";
  case ProofReason::InvalidOldAxis:
    return "old axis is out of range";
  case ProofReason::NullValue:
    return "analysis received a null value";
  case ProofReason::UnresolvedPointer:
    return "pointer has no normalized defining operation";
  case ProofReason::UnsupportedPointerType:
    return "pointer is not a ranked tensor of pointers";
  case ProofReason::UnsupportedPointerForm:
    return "pointer is not normalized as tt.splat plus tt.addptr";
  case ProofReason::UnsupportedRank:
    return "access rank is outside the supported static subset";
  case ProofReason::UnsupportedEncoding:
    return "encoded tensors are outside the pre-layout static subset";
  case ProofReason::NonSquareShape:
    return "rank-2 access must have a static square shape";
  case ProofReason::UnsupportedIndexElementType:
    return "access indices must be unencoded i32 tensors";
  case ProofReason::DynamicShape:
    return "access shape is dynamic";
  case ProofReason::UnsupportedAffineOffset:
    return "offset is outside the supported static affine form";
  case ProofReason::DynamicStride:
    return "stride is not a static integer";
  case ProofReason::NegativeStride:
    return "stride is negative";
  case ProofReason::ZeroStride:
    return "stride is zero";
  case ProofReason::DuplicateStride:
    return "strides are duplicated";
  case ProofReason::InvalidAxisProvenance:
    return "axis provenance is out of range";
  case ProofReason::DuplicateAxisProvenance:
    return "multiple access axes have the same provenance";
  case ProofReason::DuplicateRangeSource:
    return "multiple access axes reuse the same tt.make_range source";
  case ProofReason::InvalidMakeRange:
    return "tt.make_range does not match its static result shape";
  case ProofReason::InvalidExpandDimsChain:
    return "affine axis does not have a valid expand_dims chain";
  case ProofReason::OffsetOverflow:
    return "static affine offset arithmetic overflows";
  case ProofReason::OverflowFlags:
    return "affine arithmetic has overflow flags that V1 cannot preserve";
  case ProofReason::NonRowMajorContiguous:
    return "access is not logically row-major contiguous";
  case ProofReason::NonInjectiveLanes:
    return "lane address map is not proven injective";
  case ProofReason::MaskedAccess:
    return "masked or predicated access is unsupported";
  case ProofReason::BoundaryCheck:
    return "boundary or padding behavior is unsupported";
  case ProofReason::VolatileLoad:
    return "volatile load is unsupported";
  case ProofReason::NullOperation:
    return "analysis received a null operation";
  case ProofReason::DifferentBlocks:
    return "protected interval crosses MLIR blocks";
  case ProofReason::InvalidProtectedInterval:
    return "protected interval endpoints are not in program order";
  case ProofReason::RegionOperation:
    return "protected interval contains a region operation";
  case ProofReason::CallOperation:
    return "protected interval contains a call";
  case ProofReason::BarrierOperation:
    return "protected interval contains a barrier";
  case ProofReason::UnknownMemoryEffect:
    return "protected interval contains an operation with unknown memory "
           "effects";
  case ProofReason::InterveningMemoryEffect:
    return "protected interval contains a memory effect";
  case ProofReason::DifferentAccessBase:
    return "accesses do not share the same proven SSA base";
  case ProofReason::OverlappingAccessRange:
    return "static access ranges overlap";
  case ProofReason::UnsupportedInterveningMemoryAccess:
    return "intervening memory access is unsupported or cannot be proven "
           "static";
  }
  return "unknown proof reason";
}

ProofOutcome Permutation::validate(llvm::ArrayRef<int32_t> perm) {
  if (perm.empty() ||
      perm.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
    return ProofOutcome::rejected(ProofReason::InvalidPermutationRank);

  llvm::SmallVector<unsigned char, 4> seen(perm.size(), 0);
  for (int32_t oldAxis : perm) {
    if (oldAxis < 0 || static_cast<size_t>(oldAxis) >= perm.size())
      return ProofOutcome::rejected(ProofReason::InvalidPermutationAxis);
    if (seen[oldAxis])
      return ProofOutcome::rejected(ProofReason::DuplicatePermutationAxis);
    seen[oldAxis] = 1;
  }
  return ProofOutcome::proven();
}

FailureOr<Permutation> Permutation::create(llvm::ArrayRef<int32_t> perm) {
  if (!validate(perm).isProven())
    return failure();
  return Permutation(perm);
}

Permutation Permutation::inverse() const {
  llvm::SmallVector<int32_t, 4> oldToNew(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    oldToNew[newToOld[newAxis]] = static_cast<int32_t>(newAxis);
  return Permutation(oldToNew);
}

FailureOr<Permutation> Permutation::compose(const Permutation &after) const {
  if (rank() != after.rank())
    return failure();

  llvm::SmallVector<int32_t, 4> combined;
  combined.reserve(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    combined.push_back(newToOld[after.newToOld[newAxis]]);
  return Permutation(combined);
}

int32_t Permutation::mapOldAxisToNew(int32_t oldAxis) const {
  if (oldAxis < 0 || static_cast<unsigned>(oldAxis) >= rank())
    return -1;
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis) {
    if (newToOld[newAxis] == oldAxis)
      return static_cast<int32_t>(newAxis);
  }
  return -1;
}

FailureOr<llvm::SmallVector<int64_t>>
Permutation::permuteShape(llvm::ArrayRef<int64_t> shape) const {
  if (shape.size() != rank())
    return failure();

  llvm::SmallVector<int64_t> result;
  result.reserve(rank());
  for (unsigned newAxis = 0; newAxis < rank(); ++newAxis)
    result.push_back(shape[newToOld[newAxis]]);
  return result;
}

ProofOutcome StaticAccessAnalysis::proveLaneInjectivity(
    llvm::ArrayRef<int64_t> shape, llvm::ArrayRef<int64_t> strides,
    llvm::ArrayRef<unsigned> axisProvenance) {
  constexpr int64_t kSignedI32Max = std::numeric_limits<int32_t>::max();
  if (shape.empty() || shape.size() != strides.size() ||
      shape.size() != axisProvenance.size())
    return ProofOutcome::rejected(ProofReason::UnsupportedRank);

  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    if (shape[axis] < 0)
      return ProofOutcome::rejected(ProofReason::DynamicShape);
    if (shape[axis] == 0)
      return ProofOutcome::rejected(ProofReason::UnsupportedRank);
    if (strides[axis] < 0)
      return ProofOutcome::rejected(ProofReason::NegativeStride);
    if (strides[axis] == 0)
      return ProofOutcome::rejected(ProofReason::ZeroStride);
    if (strides[axis] > kSignedI32Max)
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    if (axisProvenance[axis] >= shape.size())
      return ProofOutcome::rejected(ProofReason::InvalidAxisProvenance);

    for (unsigned previous = 0; previous < axis; ++previous) {
      if (axisProvenance[previous] == axisProvenance[axis])
        return ProofOutcome::rejected(ProofReason::DuplicateAxisProvenance);
      if (strides[previous] == strides[axis])
        return ProofOutcome::rejected(ProofReason::DuplicateStride);
    }
  }

  llvm::SmallVector<unsigned, 4> axes;
  axes.reserve(shape.size());
  for (unsigned axis = 0; axis < shape.size(); ++axis)
    axes.push_back(axis);
  std::sort(axes.begin(), axes.end(), [&](unsigned lhs, unsigned rhs) {
    return strides[lhs] < strides[rhs];
  });

  int64_t reachableSpan = 0;
  for (unsigned axis : axes) {
    if (shape[axis] == 1)
      continue;
    if (strides[axis] <= reachableSpan)
      return ProofOutcome::rejected(ProofReason::NonInjectiveLanes);

    const int64_t extent = shape[axis] - 1;
    if (multiplyWouldOverflow(extent, strides[axis]))
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    const int64_t contribution = extent * strides[axis];
    // The IR offsets are i32 tensors.  i64 host arithmetic is used only to
    // prove the expression safely; it must not authorize an address map that
    // would overflow the actual TTIR arithmetic.  Existing tensor-size limits
    // keep current fixtures below this bound, but the check is intentional for
    // future configurations as well.
    if (contribution > kSignedI32Max ||
        reachableSpan > kSignedI32Max - contribution)
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    if (addWouldOverflow(reachableSpan, contribution))
      return ProofOutcome::rejected(ProofReason::OffsetOverflow);
    reachableSpan += contribution;
  }

  return ProofOutcome::proven();
}

bool StaticAccess::isLogicalRowMajorContiguous() const {
  if (!lanesInjective || shape.empty() || shape.size() != strides.size() ||
      shape.size() != axes.size() || elementCount <= 0)
    return false;

  int64_t expectedStride = 1;
  for (size_t index = shape.size(); index != 0; --index) {
    const size_t axis = index - 1;
    if (shape[axis] <= 0 || axes[axis].outputAxis != axis ||
        strides[axis] != expectedStride)
      return false;
    if (multiplyWouldOverflow(expectedStride, shape[axis]))
      return false;
    expectedStride *= shape[axis];
  }
  return expectedStride == elementCount;
}

StaticAccessProof StaticAccessAnalysis::analyzePointer(Value pointer) const {
  if (!pointer)
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullValue));

  auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
  if (!pointerType)
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerType));
  if (pointerType.getRank() < 1)
    return rejectAccess(ProofOutcome::rejected(ProofReason::UnsupportedRank));
  if (!pointerType.hasStaticShape())
    return rejectAccess(ProofOutcome::rejected(ProofReason::DynamicShape));
  if (pointerType.getEncoding())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedEncoding));

  auto pointerElement =
      dyn_cast<triton::PointerType>(pointerType.getElementType());
  if (!pointerElement ||
      triton::isTensorPointerType(pointerType.getElementType()))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerType));

  auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr) {
    if (!pointer.getDefiningOp())
      return rejectAccess(
          ProofOutcome::unknown(ProofReason::UnresolvedPointer));
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  }

  auto baseSplat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!baseSplat)
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  Type baseType = baseSplat.getSrc().getType();
  if (!isa<triton::PointerType>(baseType) ||
      triton::isTensorPointerType(baseType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));

  auto baseSplatType =
      dyn_cast<RankedTensorType>(baseSplat.getResult().getType());
  auto offsetType = dyn_cast<RankedTensorType>(addPtr.getOffset().getType());
  if (!baseSplatType || !offsetType || !baseSplatType.hasStaticShape() ||
      !offsetType.hasStaticShape() ||
      !haveSameShape(pointerType, baseSplatType) ||
      !haveSameShape(pointerType, offsetType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedPointerForm));
  if (baseSplatType.getEncoding() || offsetType.getEncoding())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedEncoding));

  if (pointerType.getRank() == 1) {
    ParsedRankOneOffset parsedOffset =
        parseNormalizedRankOneOffset(addPtr.getOffset(), offsetType);
    if (!parsedOffset.outcome.isProven())
      return rejectAccess(parsedOffset.outcome);

    auto rangeType =
        dyn_cast<RankedTensorType>(parsedOffset.range.getResult().getType());
    if (!rangeType || rangeType.getRank() != 1 || !rangeType.hasStaticShape() ||
        rangeType.getEncoding() || !haveSameShape(pointerType, rangeType))
      return rejectAccess(ProofOutcome::rejected(ProofReason::UnsupportedRank));

    int64_t rangeStart = 0;
    int64_t rangeEnd = 0;
    if (!getSignedI32RangeBounds(parsedOffset.range, rangeStart, rangeEnd))
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));
    if (rangeEnd <= rangeStart ||
        rangeType.getShape().front() != rangeEnd - rangeStart)
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));

    int64_t firstOffset = 0;
    int64_t lastOffset = 0;
    if (!checkedAddI64(rangeStart, parsedOffset.staticOffset, firstOffset) ||
        !checkedAddI64(rangeEnd - 1, parsedOffset.staticOffset, lastOffset) ||
        !isSignedI32(firstOffset) || !isSignedI32(lastOffset))
      return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));

    StaticAccess access;
    access.pointer = pointer;
    access.offset = addPtr.getOffset();
    access.base = baseSplat.getSrc();
    access.dynamicOrigin = parsedOffset.dynamicOrigin;
    access.shape.push_back(rangeType.getShape().front());
    access.strides.push_back(1);
    access.axisProvenance.push_back(0);
    access.axes.push_back(
        StaticAccessAxis{parsedOffset.range, rangeStart, rangeEnd, 1, 0});

    ProofOutcome injectivity = proveLaneInjectivity(
        access.shape, access.strides, access.axisProvenance);
    if (!injectivity.isProven())
      return rejectAccess(injectivity);

    access.firstOffset = firstOffset;
    access.lastOffset = lastOffset;
    access.elementCount = rangeType.getShape().front();
    access.lanesInjective = true;
    return {ProofOutcome::proven(), std::move(access)};
  }

  if (!isI32Tensor(offsetType))
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedIndexElementType));
  SmallVector<Value, 4> pendingTerms = {addPtr.getOffset()};
  SmallVector<Value, 4> affineTerms;
  while (!pendingTerms.empty()) {
    Value term = pendingTerms.pop_back_val();
    if (auto add = term.getDefiningOp<arith::AddIOp>()) {
      if (add.getResult() != term)
        return rejectAccess(
            ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset));
      if (add.getOverflowFlags() != arith::IntegerOverflowFlags::none)
        return rejectAccess(ProofOutcome::rejected(ProofReason::OverflowFlags));
      pendingTerms.push_back(add.getLhs());
      pendingTerms.push_back(add.getRhs());
      continue;
    }
    affineTerms.push_back(term);
  }
  if (affineTerms.size() != pointerType.getRank())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::UnsupportedAffineOffset));

  SmallVector<StaticAccessAxis, 4> axes(pointerType.getRank());
  SmallVector<OffsetBounds, 4> bounds(pointerType.getRank());
  SmallVector<bool, 4> seenOutputAxis(pointerType.getRank(), false);
  for (Value term : affineTerms) {
    ParsedAffineAxis parsedAxis = parseAffineAxis(term, offsetType);
    if (!parsedAxis.outcome.isProven())
      return rejectAccess(parsedAxis.outcome);
    if (parsedAxis.axis.outputAxis >= pointerType.getRank() ||
        seenOutputAxis[parsedAxis.axis.outputAxis])
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::DuplicateAxisProvenance));
    seenOutputAxis[parsedAxis.axis.outputAxis] = true;
    axes[parsedAxis.axis.outputAxis] = parsedAxis.axis;
    bounds[parsedAxis.axis.outputAxis] = parsedAxis.bounds;
  }

  StaticAccess access;
  access.pointer = pointer;
  access.offset = addPtr.getOffset();
  access.base = baseSplat.getSrc();
  access.shape.append(pointerType.getShape().begin(),
                      pointerType.getShape().end());
  access.axes = std::move(axes);

  int64_t firstOffset = 0;
  int64_t lastOffset = 0;
  int64_t expectedStride = 1;
  for (size_t index = access.shape.size(); index != 0; --index) {
    const unsigned axis = static_cast<unsigned>(index - 1);
    const StaticAccessAxis &parsedAxis = access.axes[axis];
    if (!seenOutputAxis[axis] || !parsedAxis.range ||
        parsedAxis.outputAxis != axis ||
        parsedAxis.rangeEnd <= parsedAxis.rangeStart ||
        parsedAxis.rangeEnd - parsedAxis.rangeStart != access.shape[axis])
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::InvalidMakeRange));
    if (parsedAxis.stride != expectedStride)
      return rejectAccess(
          ProofOutcome::rejected(ProofReason::NonRowMajorContiguous));
    if (!checkedMulI64(expectedStride, access.shape[axis], expectedStride))
      return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));
  }

  for (unsigned axis = 0; axis < access.shape.size(); ++axis) {
    const StaticAccessAxis &parsedAxis = access.axes[axis];
    for (unsigned prior = 0; prior < axis; ++prior) {
      if (access.axes[prior].range.operator->() ==
          parsedAxis.range.operator->())
        return rejectAccess(
            ProofOutcome::rejected(ProofReason::DuplicateRangeSource));
    }
    if (!checkedAddI64(firstOffset, bounds[axis].first, firstOffset) ||
        !checkedAddI64(lastOffset, bounds[axis].last, lastOffset))
      return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));
    access.strides.push_back(parsedAxis.stride);
    access.axisProvenance.push_back(axis);
  }

  if (firstOffset < std::numeric_limits<int32_t>::min() ||
      firstOffset > std::numeric_limits<int32_t>::max() ||
      lastOffset < std::numeric_limits<int32_t>::min() ||
      lastOffset > std::numeric_limits<int32_t>::max())
    return rejectAccess(ProofOutcome::rejected(ProofReason::OffsetOverflow));

  access.firstOffset = firstOffset;
  access.lastOffset = lastOffset;
  access.elementCount = expectedStride;
  access.lanesInjective = true;
  if (!access.isLogicalRowMajorContiguous())
    return rejectAccess(
        ProofOutcome::rejected(ProofReason::NonRowMajorContiguous));
  return {ProofOutcome::proven(), std::move(access)};
}

StaticAccessProof StaticAccessAnalysis::analyzeLoad(triton::LoadOp load) const {
  if (!load.getOperation())
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullOperation));
  if (load.getMask() || load.getOther())
    return rejectAccess(ProofOutcome::rejected(ProofReason::MaskedAccess));
  if (!load.getBoundaryCheck().empty() || load.getPadding())
    return rejectAccess(ProofOutcome::rejected(ProofReason::BoundaryCheck));
  if (load.getIsVolatile())
    return rejectAccess(ProofOutcome::rejected(ProofReason::VolatileLoad));
  return analyzePointer(load.getPtr());
}

StaticAccessProof
StaticAccessAnalysis::analyzeStore(triton::StoreOp store) const {
  if (!store.getOperation())
    return rejectAccess(ProofOutcome::rejected(ProofReason::NullOperation));
  if (store.getMask())
    return rejectAccess(ProofOutcome::rejected(ProofReason::MaskedAccess));
  if (!store.getBoundaryCheck().empty())
    return rejectAccess(ProofOutcome::rejected(ProofReason::BoundaryCheck));
  return analyzePointer(store.getPtr());
}

ProofOutcome
StaticAccessAnalysis::proveSameBaseDisjoint(const StaticAccess &lhs,
                                            const StaticAccess &rhs) const {
  if (!lhs.base || !rhs.base || lhs.base != rhs.base ||
      lhs.dynamicOrigin != rhs.dynamicOrigin)
    return ProofOutcome::rejected(ProofReason::DifferentAccessBase);
  if (!lhs.lanesInjective || !rhs.lanesInjective ||
      lhs.firstOffset > lhs.lastOffset || rhs.firstOffset > rhs.lastOffset)
    return ProofOutcome::rejected(
        ProofReason::UnsupportedInterveningMemoryAccess);
  if (lhs.lastOffset < rhs.firstOffset || rhs.lastOffset < lhs.firstOffset)
    return ProofOutcome::proven();
  return ProofOutcome::rejected(ProofReason::OverlappingAccessRange);
}

ProofOutcome
ProtectedIntervalAnalysis::proveNoMemoryEffects(Operation *first,
                                                Operation *last) const {
  if (!first || !last)
    return ProofOutcome::rejected(ProofReason::NullOperation);
  if (first == last)
    return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
  if (first->getBlock() != last->getBlock())
    return ProofOutcome::rejected(ProofReason::DifferentBlocks);

  for (Operation *operation = first->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (operation == last)
      return ProofOutcome::proven();
    if (operation->getNumRegions() != 0)
      return ProofOutcome::rejected(ProofReason::RegionOperation);
    if (isa<CallOpInterface>(operation))
      return ProofOutcome::rejected(ProofReason::CallOperation);
    if (isBarrierLike(operation))
      return ProofOutcome::rejected(ProofReason::BarrierOperation);

    if (auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
          effects;
      memoryEffects.getEffects(effects);
      if (!effects.empty())
        return ProofOutcome::rejected(ProofReason::InterveningMemoryEffect);
    }
    if (!isMemoryEffectFree(operation))
      return ProofOutcome::rejected(ProofReason::UnknownMemoryEffect);
  }

  return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
}

ProofOutcome ProtectedIntervalAnalysis::proveNoConflictingLoadStoreEffects(
    Operation *first, Operation *last,
    llvm::ArrayRef<StaticAccess> protectedAccesses,
    const EntryArgPointerAliasAnalysis *entryArgPointerAliases) const {
  if (!first || !last)
    return ProofOutcome::rejected(ProofReason::NullOperation);
  if (protectedAccesses.empty() || first == last)
    return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
  if (first->getBlock() != last->getBlock())
    return ProofOutcome::rejected(ProofReason::DifferentBlocks);

  StaticAccessAnalysis accessAnalysis;
  auto proveDisjoint = [&](Value effectPointer, auto &&getProof) {
    SmallVector<const StaticAccess *, 4> sameRootAccesses;
    for (const StaticAccess &protectedAccess : protectedAccesses) {
      if (!entryArgPointerAliases) {
        sameRootAccesses.push_back(&protectedAccess);
        continue;
      }

      EntryArgPointerRelation relation =
          entryArgPointerAliases->classify(effectPointer, protectedAccess.base);
      if (relation == EntryArgPointerRelation::DistinctEntryRoots)
        continue;
      if (relation == EntryArgPointerRelation::Unknown)
        return ProofOutcome::rejected(
            ProofReason::UnsupportedInterveningMemoryAccess);
      sameRootAccesses.push_back(&protectedAccess);
    }

    // All protected stores are rooted at a distinct entry pointer. This is
    // sufficient under the StoreCoalescing ABI contract, including scalar or
    // dynamically indexed accesses that StaticAccessAnalysis cannot parse.
    if (sameRootAccesses.empty())
      return ProofOutcome::proven();

    StaticAccessProof proof = getProof();
    if (!proof.isProven())
      return ProofOutcome::rejected(
          ProofReason::UnsupportedInterveningMemoryAccess);
    for (const StaticAccess *protectedAccess : sameRootAccesses) {
      ProofOutcome outcome =
          accessAnalysis.proveSameBaseDisjoint(*proof.access, *protectedAccess);
      if (!outcome.isProven())
        return outcome;
    }
    return ProofOutcome::proven();
  };

  for (Operation *operation = first->getNextNode(); operation;
       operation = operation->getNextNode()) {
    if (operation == last)
      return ProofOutcome::proven();
    if (operation->getNumRegions() != 0)
      return ProofOutcome::rejected(ProofReason::RegionOperation);
    if (isa<CallOpInterface>(operation))
      return ProofOutcome::rejected(ProofReason::CallOperation);
    if (isBarrierLike(operation))
      return ProofOutcome::rejected(ProofReason::BarrierOperation);

    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (!isSupportedInterveningLoad(load))
        return ProofOutcome::rejected(
            ProofReason::UnsupportedInterveningMemoryAccess);
      ProofOutcome outcome = proveDisjoint(
          load.getPtr(), [&]() { return accessAnalysis.analyzeLoad(load); });
      if (!outcome.isProven())
        return outcome;
      continue;
    }
    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (!isSupportedInterveningStore(store))
        return ProofOutcome::rejected(
            ProofReason::UnsupportedInterveningMemoryAccess);
      ProofOutcome outcome = proveDisjoint(
          store.getPtr(), [&]() { return accessAnalysis.analyzeStore(store); });
      if (!outcome.isProven())
        return outcome;
      continue;
    }

    if (auto memoryEffects = dyn_cast<MemoryEffectOpInterface>(operation)) {
      llvm::SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4>
          effects;
      memoryEffects.getEffects(effects);
      if (!effects.empty())
        return ProofOutcome::rejected(ProofReason::InterveningMemoryEffect);
    }
    if (!isMemoryEffectFree(operation))
      return ProofOutcome::rejected(ProofReason::UnknownMemoryEffect);
  }

  return ProofOutcome::rejected(ProofReason::InvalidProtectedInterval);
}
