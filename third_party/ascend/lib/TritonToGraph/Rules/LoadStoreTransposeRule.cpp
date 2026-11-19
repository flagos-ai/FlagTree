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

#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/PermutationAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

// This rule deliberately does not reuse StaticAccessAnalysis.  That analysis
// is the strict, static, row-major proof used by StoreCoalescing. Here we
// need a separate fail-closed affine-layout proof which accepts symbolic
// physical strides (for example N and 256 * N), masks, and scf loop-carried
// pointers. Keeping it local also prevents a load/store layout extension from
// changing StoreCoalescing admission semantics.
struct ScaleExpression {
  int64_t constantFactor = 1;
  SmallVector<Value, 4> symbols;
};

struct AffineLaneAxis {
  unsigned outputAxis = 0;
  // The rank-R value carrying this lane before its final broadcast.  Keeping
  // the provenance lets the dynamic-mask proof bind a symbolic bound to the
  // exact inner lane rather than merely observe that the mask mentions N.
  Value lane;
  ScaleExpression scale;
};

struct AffineLayoutAccess {
  Value pointer;
  Value base;
  Value offset;
  RankedTensorType type;
  SmallVector<AffineLaneAxis, 4> axes;
  SmallVector<int32_t, 4> permutation;
};

struct ReductionPort {
  triton::ReduceOp reduce;
  unsigned forResultIndex = 0;
};

// A tt.assert is a side use rather than a result-producing node, so it is not
// discovered by the ordinary backwards Value clone.  Keep it as an explicit
// port of the layout closure.  The assert itself is deliberately retained: we
// only replace its condition after the complete replacement closure verifies.
struct ExternalAssertPort {
  Operation *assertOperation = nullptr;
  Value oldCondition;
};

struct LayoutPermutationCandidate {
  scf::ForOp loop;
  // A local basic-block candidate is used when the layout closure does not
  // cross a structured-control-flow port (for example a load/store pair in an
  // scf.if branch).  `loop` and `block` are mutually exclusive.
  Block *block = nullptr;
  Operation *anchor = nullptr;
  SmallVector<int32_t, 4> permutation;
  SmallVector<Value, 16> externalTensorValues;
  SmallVector<ReductionPort, 4> reductionPorts;
  SmallVector<unsigned, 8> transformedResultIndices;
  SmallVector<Operation *, 16> blockOperations;
  SmallVector<ExternalAssertPort, 8> externalAssertPorts;
  // Pure, full-rank address/guard producers whose use closure has been
  // proven closed.  They are explicitly reclaimed after the old endpoint is
  // removed; graph-optimize is not followed by a mandatory DCE pass.
  SmallVector<Operation *, 32> retireOperations;
  unsigned endpointCount = 0;
};

bool isUnencodedStaticTensor(Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.hasStaticShape() && !tensor.getEncoding();
}

bool hasLayoutRank(Type type, unsigned rank) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  return tensor && tensor.getRank() == rank;
}

bool isLayoutTensor(Type type, unsigned rank) {
  return hasLayoutRank(type, rank) && isUnencodedStaticTensor(type);
}

bool getStaticInteger(Value value, int64_t &result) {
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto integer = dyn_cast<IntegerAttr>(constant.getValue());
  if (!integer)
    return false;
  result = integer.getValue().getSExtValue();
  return true;
}

bool multiplyConstantFactor(ScaleExpression &scale, int64_t factor) {
  if (factor <= 0)
    return false;
  if (scale.constantFactor > std::numeric_limits<int64_t>::max() / factor)
    return false;
  scale.constantFactor *= factor;
  return true;
}

bool collectScaleExpression(Value value, ScaleExpression &scale,
                            DenseSet<Value> &visiting) {
  if (!value || !visiting.insert(value).second)
    return false;

  int64_t constant = 0;
  if (getStaticInteger(value, constant))
    return multiplyConstantFactor(scale, constant);

  if (auto multiply = value.getDefiningOp<arith::MulIOp>()) {
    if (multiply.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return false;
    return collectScaleExpression(multiply.getLhs(), scale, visiting) &&
           collectScaleExpression(multiply.getRhs(), scale, visiting);
  }

  if (!isa<IntegerType, IndexType>(value.getType()))
    return false;
  scale.symbols.push_back(value);
  return true;
}

bool normalizeScaleExpression(ScaleExpression &scale) {
  if (scale.constantFactor <= 0)
    return false;
  llvm::sort(scale.symbols, [](Value lhs, Value rhs) {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  });
  return true;
}

bool collectUniformTensorScale(Value value, ScaleExpression &scale) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>()) {
    DenseSet<Value> visiting;
    return collectScaleExpression(splat.getSrc(), scale, visiting) &&
           normalizeScaleExpression(scale);
  }

  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
  if (!dense || !dense.isSplat())
    return false;
  auto integer = dyn_cast<IntegerAttr>(dense.getSplatValue<Attribute>());
  return integer && multiplyConstantFactor(scale, integer.getInt()) &&
         normalizeScaleExpression(scale);
}

bool containsMakeRange(Value value, DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (value.getDefiningOp<triton::MakeRangeOp>())
    return true;

  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumRegions() != 0 ||
      !isMemoryEffectFree(operation))
    return false;
  for (Value operand : operation->getOperands()) {
    if (containsMakeRange(operand, visited))
      return true;
  }
  return false;
}

std::optional<unsigned> getSingleVaryingAxis(Value value, unsigned rank) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      type.getRank() != rank)
    return std::nullopt;

  std::optional<unsigned> varyingAxis;
  for (unsigned axis = 0; axis < rank; ++axis) {
    if (type.getShape()[axis] == 1)
      continue;
    if (varyingAxis)
      return std::nullopt;
    varyingAxis = axis;
  }
  return varyingAxis;
}

std::optional<AffineLaneAxis> parseAffineLaneTerm(Value value, unsigned rank) {
  auto broadcast = value.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast || broadcast.getResult() != value)
    return std::nullopt;

  Value laneValue = broadcast.getSrc();
  ScaleExpression scale;
  if (auto multiply = laneValue.getDefiningOp<arith::MulIOp>()) {
    ScaleExpression lhsScale;
    ScaleExpression rhsScale;
    const bool lhsUniform =
        collectUniformTensorScale(multiply.getLhs(), lhsScale);
    const bool rhsUniform =
        collectUniformTensorScale(multiply.getRhs(), rhsScale);
    if (lhsUniform == rhsUniform)
      return std::nullopt;
    if (lhsUniform) {
      scale = std::move(lhsScale);
      laneValue = multiply.getRhs();
    } else {
      scale = std::move(rhsScale);
      laneValue = multiply.getLhs();
    }
  } else if (!normalizeScaleExpression(scale)) {
    return std::nullopt;
  }

  std::optional<unsigned> axis = getSingleVaryingAxis(laneValue, rank);
  if (!axis)
    return std::nullopt;

  DenseSet<Value> visited;
  if (!containsMakeRange(laneValue, visited))
    return std::nullopt;
  return AffineLaneAxis{*axis, laneValue, std::move(scale)};
}

bool isSubsetOf(ArrayRef<Value> subset, ArrayRef<Value> superset) {
  size_t subsetIndex = 0;
  size_t supersetIndex = 0;
  while (subsetIndex < subset.size() && supersetIndex < superset.size()) {
    const void *subsetValue = subset[subsetIndex].getAsOpaquePointer();
    const void *supersetValue = superset[supersetIndex].getAsOpaquePointer();
    if (subsetValue == supersetValue) {
      ++subsetIndex;
      ++supersetIndex;
      continue;
    }
    if (subsetValue < supersetValue)
      return false;
    ++supersetIndex;
  }
  return subsetIndex == subset.size();
}

// Returns 1 if lhs is structurally a larger physical stride, -1 if rhs is,
// and 0 when the ordering cannot be certified.  We intentionally compare the
// symbolic factor structure rather than runtime values: N may be 1 at run
// time, but the masked active domain still establishes the target layout.
int compareScaleExpressions(const ScaleExpression &lhs,
                            const ScaleExpression &rhs) {
  const bool lhsContainsRhs = isSubsetOf(rhs.symbols, lhs.symbols);
  const bool rhsContainsLhs = isSubsetOf(lhs.symbols, rhs.symbols);
  if (lhsContainsRhs != rhsContainsLhs)
    return lhsContainsRhs ? 1 : -1;

  if (!lhsContainsRhs)
    return 0;
  if (lhs.constantFactor == rhs.constantFactor)
    return 0;
  return lhs.constantFactor > rhs.constantFactor ? 1 : -1;
}

bool isIdentityPermutation(ArrayRef<int32_t> permutation) {
  for (unsigned index = 0; index < permutation.size(); ++index) {
    if (permutation[index] != static_cast<int32_t>(index))
      return false;
  }
  return true;
}

bool haveSamePermutation(ArrayRef<int32_t> lhs, ArrayRef<int32_t> rhs) {
  return lhs.size() == rhs.size() && llvm::equal(lhs, rhs);
}

std::optional<AffineLayoutAccess> analyzeAffineLayout(Value pointer) {
  auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
  if (!pointerType || !pointerType.hasStaticShape() ||
      pointerType.getEncoding() || pointerType.getRank() < 2 ||
      !isa<triton::PointerType>(pointerType.getElementType()) ||
      triton::isTensorPointerType(pointerType.getElementType()))
    return std::nullopt;

  auto addPtr = pointer.getDefiningOp<triton::AddPtrOp>();
  if (!addPtr || addPtr.getResult() != pointer)
    return std::nullopt;
  auto baseSplat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  if (!baseSplat || !isa<triton::PointerType>(baseSplat.getSrc().getType()) ||
      triton::isTensorPointerType(baseSplat.getSrc().getType()))
    return std::nullopt;

  auto offsetType = dyn_cast<RankedTensorType>(addPtr.getOffset().getType());
  if (!offsetType || !offsetType.hasStaticShape() || offsetType.getEncoding() ||
      offsetType.getShape() != pointerType.getShape())
    return std::nullopt;

  SmallVector<Value, 8> pending = {addPtr.getOffset()};
  SmallVector<Value, 8> affineTerms;
  while (!pending.empty()) {
    Value term = pending.pop_back_val();
    if (auto add = term.getDefiningOp<arith::AddIOp>()) {
      pending.push_back(add.getLhs());
      pending.push_back(add.getRhs());
      continue;
    }
    affineTerms.push_back(term);
  }

  SmallVector<AffineLaneAxis, 4> axes;
  for (Value term : affineTerms) {
    if (std::optional<AffineLaneAxis> axis =
            parseAffineLaneTerm(term, pointerType.getRank())) {
      axes.push_back(std::move(*axis));
      continue;
    }

    // A full-tile splat is an invariant offset and does not affect the lane
    // permutation.  Other unrecognized terms make the proof fail closed.
    if (!term.getDefiningOp<triton::SplatOp>())
      return std::nullopt;
  }

  if (axes.size() != pointerType.getRank())
    return std::nullopt;
  for (unsigned axis = 0; axis < axes.size(); ++axis) {
    for (unsigned previous = 0; previous < axis; ++previous) {
      if (axes[axis].outputAxis == axes[previous].outputAxis)
        return std::nullopt;
    }
  }

  SmallVector<unsigned, 4> remaining;
  for (unsigned axis = 0; axis < axes.size(); ++axis)
    remaining.push_back(axis);
  SmallVector<int32_t, 4> permutation;
  permutation.reserve(axes.size());
  while (!remaining.empty()) {
    std::optional<unsigned> selected;
    for (unsigned candidate : remaining) {
      bool largerThanAll = true;
      for (unsigned other : remaining) {
        if (candidate == other)
          continue;
        if (compareScaleExpressions(axes[candidate].scale, axes[other].scale) !=
            1) {
          largerThanAll = false;
          break;
        }
      }
      if (!largerThanAll)
        continue;
      if (selected)
        return std::nullopt;
      selected = candidate;
    }
    if (!selected)
      return std::nullopt;
    permutation.push_back(static_cast<int32_t>(axes[*selected].outputAxis));
    remaining.erase(llvm::find(remaining, *selected));
  }

  if (failed(Permutation::create(permutation)) ||
      isIdentityPermutation(permutation))
    return std::nullopt;

  return AffineLayoutAccess{
      pointer,     baseSplat.getSrc(), addPtr.getOffset(),
      pointerType, std::move(axes),    std::move(permutation)};
}

const AffineLaneAxis *getAxisForOutputAxis(const AffineLayoutAccess &access,
                                           unsigned outputAxis) {
  for (const AffineLaneAxis &axis : access.axes) {
    if (axis.outputAxis == outputAxis)
      return &axis;
  }
  return nullptr;
}

bool isStaticInjectiveAccess(const AffineLayoutAccess &access) {
  int64_t innerReachableOffset = 0;
  for (int index = static_cast<int>(access.permutation.size()) - 1; index >= 0;
       --index) {
    const auto outputAxis = static_cast<unsigned>(access.permutation[index]);
    const AffineLaneAxis *axis = getAxisForOutputAxis(access, outputAxis);
    if (!axis || !axis->scale.symbols.empty() ||
        axis->scale.constantFactor <= innerReachableOffset)
      return false;

    const int64_t extent = access.type.getShape()[outputAxis];
    if (extent <= 0 ||
        axis->scale.constantFactor >
            (std::numeric_limits<int64_t>::max() - innerReachableOffset) /
                (extent - 1 == 0 ? 1 : extent - 1))
      return false;
    const int64_t contribution = axis->scale.constantFactor * (extent - 1);
    if (innerReachableOffset >
        std::numeric_limits<int64_t>::max() - contribution)
      return false;
    innerReachableOffset += contribution;
  }
  return true;
}

bool collectMaskComparisons(Value value,
                            SmallVectorImpl<arith::CmpIOp> &comparisons,
                            DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (auto comparison = value.getDefiningOp<arith::CmpIOp>()) {
    comparisons.push_back(comparison);
    return true;
  }
  if (auto conjunction = value.getDefiningOp<arith::AndIOp>()) {
    return collectMaskComparisons(conjunction.getLhs(), comparisons, visited) &&
           collectMaskComparisons(conjunction.getRhs(), comparisons, visited);
  }
  if (auto broadcast = value.getDefiningOp<triton::BroadcastOp>())
    return collectMaskComparisons(broadcast.getSrc(), comparisons, visited);
  return false;
}

bool collectUniformScaleExpression(Value value, ScaleExpression &scale) {
  if (isa<RankedTensorType>(value.getType()))
    return collectUniformTensorScale(value, scale);

  DenseSet<Value> visiting;
  return collectScaleExpression(value, scale, visiting) &&
         normalizeScaleExpression(scale);
}

std::optional<ScaleExpression> getStrictUpperBoundFor(arith::CmpIOp comparison,
                                                      Value lane) {
  const arith::CmpIPredicate predicate = comparison.getPredicate();
  if ((predicate == arith::CmpIPredicate::slt ||
       predicate == arith::CmpIPredicate::ult) &&
      comparison.getLhs() == lane) {
    ScaleExpression bound;
    if (collectUniformScaleExpression(comparison.getRhs(), bound))
      return bound;
  }
  if ((predicate == arith::CmpIPredicate::sgt ||
       predicate == arith::CmpIPredicate::ugt) &&
      comparison.getRhs() == lane) {
    ScaleExpression bound;
    if (collectUniformScaleExpression(comparison.getLhs(), bound))
      return bound;
  }
  return std::nullopt;
}

bool multiplyScaleExpressions(const ScaleExpression &lhs,
                              const ScaleExpression &rhs,
                              ScaleExpression &result) {
  if (lhs.constantFactor <= 0 || rhs.constantFactor <= 0 ||
      lhs.constantFactor >
          std::numeric_limits<int64_t>::max() / rhs.constantFactor)
    return false;
  result.constantFactor = lhs.constantFactor * rhs.constantFactor;
  result.symbols.clear();
  result.symbols.reserve(lhs.symbols.size() + rhs.symbols.size());
  result.symbols.append(lhs.symbols.begin(), lhs.symbols.end());
  result.symbols.append(rhs.symbols.begin(), rhs.symbols.end());
  return normalizeScaleExpression(result);
}

// This comparison intentionally requires identical symbolic monomials.  It
// never assumes an arbitrary SSA symbol is positive, ordered, or at least one;
// any such unproven relation must keep the candidate fail-closed.
bool dominatesScaleExpression(const ScaleExpression &lhs,
                              const ScaleExpression &rhs) {
  return lhs.constantFactor >= rhs.constantFactor &&
         lhs.symbols.size() == rhs.symbols.size() &&
         llvm::equal(lhs.symbols, rhs.symbols);
}

void collectAxisCapacities(const AffineLaneAxis &axis, int64_t staticExtent,
                           ArrayRef<arith::CmpIOp> comparisons,
                           SmallVectorImpl<ScaleExpression> &capacities) {
  if (staticExtent > 0)
    capacities.push_back(ScaleExpression{staticExtent, {}});
  for (arith::CmpIOp comparison : comparisons) {
    std::optional<ScaleExpression> bound =
        getStrictUpperBoundFor(comparison, axis.lane);
    if (bound)
      capacities.push_back(std::move(*bound));
  }
}

// Dynamic physical strides need a domain proof, not a runtime argsort.  In
// physical outer-to-inner order, prove every adjacent mixed-radix capacity:
//
//   outerStride >= innerStride * capacity(innerAxis)
//
// A static tile extent is one capacity; a mask conjunct `lane < B` supplies a
// symbolic capacity B.  By induction, the span of every suffix is strictly
// smaller than its leading stride, so active lanes remain injective.  This
// handles rank-N canonical forms such as M*N, N, 1 without assuming a runtime
// ordering of M or N.  Non-monomial scales or missing bounds still reject.
bool hasDynamicMaskProof(Value mask, const AffineLayoutAccess &access) {
  if (!mask || access.type.getRank() != access.permutation.size() ||
      access.type.getRank() < 2)
    return false;

  SmallVector<arith::CmpIOp, 4> comparisons;
  DenseSet<Value> visited;
  if (!collectMaskComparisons(mask, comparisons, visited))
    return false;

  for (unsigned outerIndex = 0; outerIndex + 1 < access.permutation.size();
       ++outerIndex) {
    const auto outerOutputAxis =
        static_cast<unsigned>(access.permutation[outerIndex]);
    const auto innerOutputAxis =
        static_cast<unsigned>(access.permutation[outerIndex + 1]);
    const AffineLaneAxis *outer = getAxisForOutputAxis(access, outerOutputAxis);
    const AffineLaneAxis *inner = getAxisForOutputAxis(access, innerOutputAxis);
    if (!outer || !inner)
      return false;

    SmallVector<ScaleExpression, 4> capacities;
    collectAxisCapacities(*inner, access.type.getShape()[innerOutputAxis],
                          comparisons, capacities);
    bool hasDominatingCapacity = false;
    for (const ScaleExpression &capacity : capacities) {
      ScaleExpression required;
      if (multiplyScaleExpressions(inner->scale, capacity, required) &&
          dominatesScaleExpression(outer->scale, required)) {
        hasDominatingCapacity = true;
        break;
      }
    }
    if (!hasDominatingCapacity)
      return false;
  }
  return true;
}

bool requiresDynamicMaskProof(const AffineLayoutAccess &access) {
  return llvm::any_of(access.axes, [](const AffineLaneAxis &axis) {
    return !axis.scale.symbols.empty();
  });
}

bool hasValidEndpointMask(Value mask, const AffineLayoutAccess &access) {
  if (!requiresDynamicMaskProof(access))
    return isStaticInjectiveAccess(access) &&
           (!mask || isLayoutTensor(mask.getType(), access.type.getRank()));
  return hasDynamicMaskProof(mask, access);
}

bool isUniformOther(Value value) {
  if (!value)
    return true;
  if (!isa<RankedTensorType>(value.getType()))
    return true;
  if (value.getDefiningOp<triton::SplatOp>())
    return true;
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
  return dense && dense.isSplat();
}

bool hasUnsupportedLoadAttributes(triton::LoadOp load) {
  if (auto volatileAttr = load->getAttrOfType<BoolAttr>("isVolatile")) {
    if (volatileAttr.getValue())
      return true;
  }
  if (load->getAttr("padding"))
    return true;
  if (auto boundary = load->getAttrOfType<DenseI32ArrayAttr>("boundaryCheck")) {
    if (!boundary.asArrayRef().empty())
      return true;
  }
  return false;
}

bool hasUnsupportedStoreAttributes(triton::StoreOp store) {
  if (auto boundary =
          store->getAttrOfType<DenseI32ArrayAttr>("boundaryCheck")) {
    if (!boundary.asArrayRef().empty())
      return true;
  }
  return false;
}

bool isSupportedClonableOperation(Operation *operation) {
  if (!operation || operation->getNumRegions() != 0)
    return false;
  if (isa<triton::LoadOp, triton::StoreOp, triton::AssertOp, triton::SplatOp,
          triton::BroadcastOp, triton::ExpandDimsOp, triton::AddPtrOp,
          triton::TransOp, arith::ConstantOp>(operation))
    return true;
  return operation->hasTrait<OpTrait::Elementwise>() &&
         isMemoryEffectFree(operation);
}

std::optional<RankedTensorType>
getPermutedTensorType(RankedTensorType type, const Permutation &permutation) {
  if (!type.hasStaticShape() || type.getEncoding() ||
      type.getRank() != permutation.rank())
    return std::nullopt;
  FailureOr<SmallVector<int64_t>> shape =
      permutation.permuteShape(type.getShape());
  if (failed(shape))
    return std::nullopt;
  return type.clone(*shape);
}

std::optional<Type> getPermutedType(Type type, const Permutation &permutation) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  if (!tensor || tensor.getRank() != permutation.rank())
    return type;
  std::optional<RankedTensorType> permuted =
      getPermutedTensorType(tensor, permutation);
  if (!permuted)
    return std::nullopt;
  return Type(*permuted);
}

bool isPermutedTensorValue(Value value, const Permutation &permutation) {
  return hasLayoutRank(value.getType(), permutation.rank());
}

// ExpandDims is not a simple axis-attribute permutation.  Its source has one
// fewer dimensions, so a rank-R permutation cannot be applied to that source
// verbatim.  For the affine layout form we accept here, an expand chain carries
// exactly one make_range-derived lane axis.  Rebuild the complete chain from
// its rank-1 source and place that lane at its mapped full-rank axis.  This is
// what makes a non-self-inverse 3D permutation such as [1, 2, 0] correct.
std::optional<Value> getExpandRankOneSource(triton::ExpandDimsOp expand) {
  Value source = expand.getSrc();
  while (auto parent = source.getDefiningOp<triton::ExpandDimsOp>())
    source = parent.getSrc();
  auto type = dyn_cast<RankedTensorType>(source.getType());
  if (!type || !type.hasStaticShape() || type.getEncoding() ||
      type.getRank() != 1)
    return std::nullopt;
  return source;
}

std::optional<Value>
rebuildFullRankExpand(IRRewriter &rewriter, triton::ExpandDimsOp expand,
                      Value rankOneSource, const Permutation &permutation,
                      SmallVectorImpl<Operation *> *created = nullptr) {
  auto oldType = dyn_cast<RankedTensorType>(expand.getResult().getType());
  std::optional<RankedTensorType> newType =
      oldType ? getPermutedTensorType(oldType, permutation) : std::nullopt;
  std::optional<unsigned> oldAxis =
      oldType ? getSingleVaryingAxis(expand.getResult(), permutation.rank())
              : std::nullopt;
  if (!newType || !oldAxis)
    return std::nullopt;
  const int32_t newVaryingAxis =
      permutation.mapOldAxisToNew(static_cast<int32_t>(*oldAxis));
  if (newVaryingAxis < 0)
    return std::nullopt;

  Value current = rankOneSource;
  auto currentType = dyn_cast<RankedTensorType>(current.getType());
  if (!currentType || currentType.getRank() != 1)
    return std::nullopt;
  for (unsigned axis = 0; axis < permutation.rank(); ++axis) {
    if (axis == static_cast<unsigned>(newVaryingAxis))
      continue;
    SmallVector<int64_t, 4> shape(currentType.getShape());
    shape.insert(shape.begin() + axis, 1);
    RankedTensorType expandedType = newType->clone(shape);
    auto rebuilt = rewriter.create<triton::ExpandDimsOp>(
        expand.getLoc(), expandedType, current, axis);
    if (created)
      created->push_back(rebuilt.getOperation());
    current = rebuilt.getResult();
    currentType = expandedType;
  }
  if (current.getType() != *newType)
    return std::nullopt;
  return current;
}

// All full-rank values in this rule carry the same layout P.  For an explicit
// TTIR transpose Q, the reindexed transpose must satisfy
//
//   P[Q'[new]] == Q[P[new]],
//
// hence Q' = P.inverse() \u2218 Q \u2218 P under the shared
// new-axis-to-old-axis convention.  This is deliberately computed with
// Permutation::compose rather than a 2-D swap shortcut: non-self-inverse 3-D
// orders are the regression case that would otherwise silently be wrong.
std::optional<SmallVector<int32_t, 4>>
getReindexedTransOrder(triton::TransOp trans, const Permutation &layout) {
  FailureOr<Permutation> original = Permutation::create(trans.getOrder());
  if (failed(original) || original->rank() != layout.rank())
    return std::nullopt;
  FailureOr<Permutation> originalAfterLayout = original->compose(layout);
  if (failed(originalAfterLayout))
    return std::nullopt;
  FailureOr<Permutation> reindexed =
      layout.inverse().compose(*originalAfterLayout);
  if (failed(reindexed))
    return std::nullopt;
  return SmallVector<int32_t, 4>(reindexed->getNewToOld().begin(),
                                 reindexed->getNewToOld().end());
}

std::optional<Operation *>
rebuildTransOperation(IRRewriter &rewriter, triton::TransOp trans, Value input,
                      Type resultType, const Permutation &permutation,
                      SmallVectorImpl<Operation *> *created = nullptr) {
  auto newResultType = dyn_cast<RankedTensorType>(resultType);
  if (!newResultType || !isLayoutTensor(input.getType(), permutation.rank()) ||
      !isLayoutTensor(newResultType, permutation.rank()))
    return std::nullopt;
  std::optional<SmallVector<int32_t, 4>> order =
      getReindexedTransOrder(trans, permutation);
  if (!order)
    return std::nullopt;
  FailureOr<Permutation> reindexedOrder = Permutation::create(*order);
  if (failed(reindexedOrder))
    return std::nullopt;
  auto inputType = cast<RankedTensorType>(input.getType());
  FailureOr<SmallVector<int64_t>> expectedShape =
      reindexedOrder->permuteShape(inputType.getShape());
  if (failed(expectedShape) || *expectedShape != newResultType.getShape())
    return std::nullopt;

  auto rebuilt = rewriter.create<triton::TransOp>(
      trans.getLoc(), newResultType, input, ArrayRef<int32_t>(*order));
  for (NamedAttribute attribute : trans->getAttrs()) {
    if (attribute.getName().getValue() != "order")
      rebuilt->setAttr(attribute.getName(), attribute.getValue());
  }
  if (created)
    created->push_back(rebuilt.getOperation());
  return rebuilt.getOperation();
}

bool isRebuildableExternalTensor(Value value, const Permutation &permutation,
                                 DenseSet<Value> &visiting) {
  if (!isPermutedTensorValue(value, permutation))
    return true;
  if (!visiting.insert(value).second)
    return false;

  // `visiting` is a recursion stack, not a memo set.  A legal external DAG can
  // fan out and feed the same full-rank value into multiple operands (for
  // example `arith.andi %mask, %mask`).  LayoutValueCloner has a value cache
  // and rebuilds that shape once, so rejecting the second DAG edge here would
  // be an unnecessary false negative.  Erase this value on every return while
  // retaining cycle detection along the active ancestry.
  auto finish = [&](bool result) {
    visiting.erase(value);
    return result;
  };
  if (!isLayoutTensor(value.getType(), permutation.rank()))
    return finish(false);

  Operation *operation = value.getDefiningOp();
  if (!operation || !isSupportedClonableOperation(operation))
    return finish(false);
  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    return finish(dense && dense.isSplat());
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
    if (isPermutedTensorValue(expand.getResult(), permutation) &&
        !getExpandRankOneSource(expand))
      return finish(false);
  }

  for (Value operand : operation->getOperands()) {
    if (!isRebuildableExternalTensor(operand, permutation, visiting))
      return finish(false);
  }
  return finish(true);
}

bool isAutomaticOverflowAssert(triton::AssertOp assertOp) {
  if (!assertOp)
    return false;
  // The message is useful diagnostic context, but not a provenance proof: a
  // user-authored device_assert can deliberately use the same text.  The
  // frontend places this private marker only on sanitize_overflow assertions.
  if (!assertOp->hasAttr("tt.auto_overflow_assert"))
    return false;
  Attribute message = assertOp.getMessageAttr();
  auto string = dyn_cast<StringAttr>(message);
  return string &&
         string.getValue().contains("overflow detected for operation");
}

bool isPermutableOverflowAssertCondition(Value condition,
                                         const Permutation &permutation) {
  auto type = dyn_cast<RankedTensorType>(condition.getType());
  if (!type || !isLayoutTensor(type, permutation.rank()))
    return false;
  auto elementType = dyn_cast<IntegerType>(type.getElementType());
  return elementType && elementType.getWidth() == 1;
}

// `isSupportedClonableOperation` intentionally accepts load/store/assert for
// the loop-body rewriter.  They cannot be part of the old address/guard slice
// we retire, because that slice is only safe to erase when it is pure.
bool isRetirablePureProducer(Operation *operation) {
  return operation && operation->getNumRegions() == 0 &&
         operation->getNumResults() == 1 &&
         !isa<triton::LoadOp, triton::StoreOp, triton::AssertOp>(operation) &&
         isSupportedClonableOperation(operation) &&
         isMemoryEffectFree(operation);
}

struct LayoutClosure {
  DenseSet<Value> values;
  llvm::SetVector<Operation *> operations;
};

// Collect the old full-rank layout slice that LayoutValueCloner will recreate.
// Rank-0/rank-1 leaves (for example N and make_range sources) remain shared
// external inputs; they neither need a layout permutation nor can they be
// retired until their ordinary users are gone.
bool collectLayoutClosure(Value value, const Permutation &permutation,
                          LayoutClosure &closure, Block *sourceBlock) {
  if (!isPermutedTensorValue(value, permutation))
    return true;
  if (!isLayoutTensor(value.getType(), permutation.rank()))
    return false;
  if (!closure.values.insert(value).second)
    return true;

  Operation *operation = value.getDefiningOp();
  if (!isRetirablePureProducer(operation) ||
      (sourceBlock && operation->getBlock() != sourceBlock))
    return false;
  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    if (!dense || !dense.isSplat())
      return false;
  }
  if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
    if (isPermutedTensorValue(expand.getResult(), permutation) &&
        !getExpandRankOneSource(expand))
      return false;
  }

  // LayoutValueCloner rebuilds every result of an operation.  Account for all
  // of its full-rank results here as well, so an unselected use of a sibling
  // result cannot escape the proof.
  for (Value result : operation->getResults()) {
    if (!isPermutedTensorValue(result, permutation))
      continue;
    if (!isLayoutTensor(result.getType(), permutation.rank()))
      return false;
    closure.values.insert(result);
  }
  closure.operations.insert(operation);

  for (Value operand : operation->getOperands()) {
    if (!collectLayoutClosure(operand, permutation, closure, sourceBlock))
      return false;
  }
  return true;
}

void mergeLayoutClosure(LayoutClosure &into, const LayoutClosure &from) {
  for (Value value : from.values)
    into.values.insert(value);
  for (Operation *operation : from.operations)
    into.operations.insert(operation);
}

bool hasSharedLayoutProvenance(Value value,
                               const DenseSet<Value> &addressValues,
                               DenseSet<Value> &visited) {
  if (addressValues.contains(value))
    return true;
  if (!value || !visited.insert(value).second)
    return false;
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumRegions() != 0)
    return false;
  return llvm::any_of(operation->getOperands(), [&](Value operand) {
    return hasSharedLayoutProvenance(operand, addressValues, visited);
  });
}

// This deliberately performs only a read-only walk.  It is used to decide
// whether a preheader assert is relevant before we impose the stronger
// same-Block/pure-producer closure proof.  In particular, an ordinary
// load/store candidate with no relevant automatic assert must retain the
// pre-existing admission behavior for externally supplied mask/pointer
// values.
void collectLayoutProvenance(Value value, const Permutation &permutation,
                             DenseSet<Value> &provenance,
                             DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return;
  if (isPermutedTensorValue(value, permutation))
    provenance.insert(value);

  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumRegions() != 0)
    return;
  for (Value operand : operation->getOperands())
    collectLayoutProvenance(operand, permutation, provenance, visited);
}

bool isClosureInBlock(const LayoutClosure &closure, Block *block) {
  return block && llvm::all_of(closure.operations, [&](Operation *operation) {
           return operation && operation->getBlock() == block;
         });
}

bool verifyLayoutClosureUses(const LayoutClosure &closure, Operation *endpoint,
                             ArrayRef<Operation *> selectedBlockOperations,
                             ArrayRef<ExternalAssertPort> assertPorts) {
  if (!endpoint)
    return false;

  DenseSet<Operation *> permittedConsumers;
  for (Operation *operation : closure.operations)
    permittedConsumers.insert(operation);
  permittedConsumers.insert(endpoint);
  for (Operation *operation : selectedBlockOperations)
    permittedConsumers.insert(operation);
  for (const ExternalAssertPort &port : assertPorts) {
    if (!port.assertOperation)
      return false;
    permittedConsumers.insert(port.assertOperation);
  }

  for (Value value : closure.values) {
    for (OpOperand &use : value.getUses()) {
      Operation *owner = use.getOwner();
      if (permittedConsumers.contains(owner))
        continue;
      return false;
    }
  }
  return true;
}

// Discover preheader (or pre-anchor) automatic overflow assertions that are
// side users of the same pointer-layout provenance.  Any other side use of a
// layout value in the resulting closure makes the whole candidate fail closed.
bool collectAutomaticOverflowAssertClosure(
    LayoutPermutationCandidate &candidate, Block *assertBlock, Operation *limit,
    ArrayRef<Value> pointerRoots, const Permutation &permutation) {
  if (!assertBlock || !limit || limit->getBlock() != assertBlock)
    return false;

  DenseSet<Value> pointerProvenance;
  DenseSet<Value> provenanceVisited;
  for (Value value : pointerRoots) {
    collectLayoutProvenance(value, permutation, pointerProvenance,
                            provenanceVisited);
  }

  SmallVector<ExternalAssertPort, 8> ports;
  bool foundLimit = false;
  for (Operation &operation : *assertBlock) {
    if (&operation == limit) {
      foundLimit = true;
      break;
    }
    auto assertOp = dyn_cast<triton::AssertOp>(operation);
    if (!assertOp || !isAutomaticOverflowAssert(assertOp))
      continue;

    Value condition = assertOp.getCondition();
    DenseSet<Value> conditionVisited;
    if (!hasSharedLayoutProvenance(condition, pointerProvenance,
                                   conditionVisited))
      continue;
    if (!isPermutableOverflowAssertCondition(condition, permutation))
      return false;
    ports.push_back(ExternalAssertPort{assertOp.getOperation(), condition});
  }
  if (!foundLimit)
    return false;
  // No relevant marked auto assert: preserve the existing candidate behavior.
  // Do not require an external mask/pointer to be a locally-retirable guard
  // producer merely because this extension is enabled.
  if (ports.empty())
    return true;

  LayoutClosure pointerClosure;
  for (Value value : pointerRoots) {
    if (!collectLayoutClosure(value, permutation, pointerClosure, assertBlock))
      return false;
  }

  LayoutClosure guardClosure;
  for (const ExternalAssertPort &port : ports) {
    LayoutClosure conditionClosure;
    if (!collectLayoutClosure(port.oldCondition, permutation, conditionClosure,
                              assertBlock))
      return false;
    mergeLayoutClosure(guardClosure, conditionClosure);
  }

  LayoutClosure addressClosure = pointerClosure;
  mergeLayoutClosure(addressClosure, guardClosure);

  // A pointer-layout producer is commonly shared by the external mask (and
  // occasionally by `other`/loop init values) that the ordinary rewrite will
  // clone into the new endpoint.  Admit only those external slices that
  // actually depend on the address/guard closure.  This restores their valid
  // use as closed consumers without imposing any same-Block/pure requirement
  // on unrelated external inputs when no automatic assert is present.
  for (Value value : candidate.externalTensorValues) {
    DenseSet<Value> externalVisited;
    if (!hasSharedLayoutProvenance(value, addressClosure.values,
                                   externalVisited))
      continue;
    LayoutClosure externalConsumerClosure;
    if (!collectLayoutClosure(value, permutation, externalConsumerClosure,
                              assertBlock))
      return false;
    mergeLayoutClosure(addressClosure, externalConsumerClosure);
  }

  // A guarded block/loop may not require a layout value through a Region port.
  // The source-block proof above is intentionally exact: this rule does not
  // synthesize scf.if/scf.while value ports for a type-changing rewrite.
  if (!isClosureInBlock(addressClosure, assertBlock))
    return false;

  Operation *endpoint =
      candidate.loop ? candidate.loop.getOperation() : nullptr;
  if (!endpoint && candidate.block)
    endpoint = candidate.anchor;
  if (!verifyLayoutClosureUses(addressClosure, endpoint,
                               candidate.blockOperations, ports))
    return false;

  // Some loop-only external tensors (for example the full-rank pointer
  // increment splat) do not share address provenance with a guard, but become
  // dead solely because the old endpoint is replaced.  Reclaim them when the
  // same closed-use proof succeeds; an unrelated/external value simply stays
  // outside the retirement set and cannot make this assert extension reject a
  // valid candidate.
  LayoutClosure retireClosure = addressClosure;
  for (Value value : candidate.externalTensorValues) {
    if (addressClosure.values.contains(value))
      continue;
    LayoutClosure auxiliaryClosure;
    if (!collectLayoutClosure(value, permutation, auxiliaryClosure,
                              assertBlock) ||
        !isClosureInBlock(auxiliaryClosure, assertBlock) ||
        !verifyLayoutClosureUses(auxiliaryClosure, endpoint,
                                 candidate.blockOperations, ports))
      continue;
    mergeLayoutClosure(retireClosure, auxiliaryClosure);
  }

  candidate.externalAssertPorts = std::move(ports);
  candidate.retireOperations.append(retireClosure.operations.begin(),
                                    retireClosure.operations.end());
  return true;
}

bool isSupportedLoopBodyOperation(Operation *operation) {
  if (isa<scf::YieldOp>(operation))
    return true;
  return isSupportedClonableOperation(operation);
}

bool valueDependsOnLoad(Value value, const DenseSet<Value> &loadResults,
                        DenseSet<Value> &visited) {
  if (!value || !visited.insert(value).second)
    return false;
  if (loadResults.contains(value))
    return true;
  Operation *operation = value.getDefiningOp();
  if (!operation || operation->getNumRegions() != 0)
    return false;
  for (Value operand : operation->getOperands()) {
    if (valueDependsOnLoad(operand, loadResults, visited))
      return true;
  }
  return false;
}

std::optional<unsigned> getForIterArgIndex(scf::ForOp loop, Value value) {
  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument || blockArgument.getOwner() != loop.getBody() ||
      blockArgument.getArgNumber() == 0)
    return std::nullopt;
  const unsigned index = blockArgument.getArgNumber() - 1;
  if (index >= loop.getInitArgs().size())
    return std::nullopt;
  return index;
}

std::optional<Type> getReducedResultType(RankedTensorType inputType,
                                         int32_t axis) {
  if (axis < 0 || static_cast<unsigned>(axis) >= inputType.getRank())
    return std::nullopt;
  SmallVector<int64_t, 4> shape(inputType.getShape());
  shape.erase(shape.begin() + axis);
  return inputType.clone(shape);
}

std::optional<LayoutPermutationCandidate> matchLoopCandidate(scf::ForOp loop) {
  if (!loop || !loop.getBody() || !loop.getBody()->hasNoPredecessors() ||
      loop.getBody()->getNumArguments() != loop.getInitArgs().size() + 1)
    return std::nullopt;

  Block *body = loop.getBody();
  SmallVector<triton::LoadOp, 4> loads;
  SmallVector<triton::StoreOp, 4> stores;
  SmallVector<AffineLayoutAccess, 8> accesses;
  DenseSet<Value> loadResults;
  std::optional<SmallVector<int32_t, 4>> permutation;

  for (Operation &operation : *body) {
    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (hasUnsupportedLoadAttributes(load) ||
          !isUniformOther(load.getOther()))
        return std::nullopt;
      std::optional<unsigned> iterArgIndex =
          getForIterArgIndex(loop, load.getPtr());
      auto pointerType = dyn_cast<RankedTensorType>(load.getPtr().getType());
      if (!iterArgIndex || !pointerType || !pointerType.hasStaticShape() ||
          pointerType.getEncoding() || pointerType.getRank() < 2)
        return std::nullopt;
      std::optional<AffineLayoutAccess> access =
          analyzeAffineLayout(loop.getInitArgs()[*iterArgIndex]);
      if (!access ||
          !isLayoutTensor(load.getResult().getType(), access->type.getRank()) ||
          !hasValidEndpointMask(load.getMask(), *access))
        return std::nullopt;
      if (permutation &&
          !haveSamePermutation(*permutation, access->permutation))
        return std::nullopt;
      permutation = access->permutation;
      loadResults.insert(load.getResult());
      loads.push_back(load);
      accesses.push_back(std::move(*access));
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (hasUnsupportedStoreAttributes(store))
        return std::nullopt;
      std::optional<unsigned> iterArgIndex =
          getForIterArgIndex(loop, store.getPtr());
      if (!iterArgIndex)
        return std::nullopt;
      std::optional<AffineLayoutAccess> access =
          analyzeAffineLayout(loop.getInitArgs()[*iterArgIndex]);
      if (!access ||
          !isLayoutTensor(store.getValue().getType(), access->type.getRank()) ||
          !hasValidEndpointMask(store.getMask(), *access))
        return std::nullopt;
      if (permutation &&
          !haveSamePermutation(*permutation, access->permutation))
        return std::nullopt;
      permutation = access->permutation;
      stores.push_back(store);
      accesses.push_back(std::move(*access));
    }
  }

  if (loads.empty() || stores.empty() || !permutation)
    return std::nullopt;
  FailureOr<Permutation> layoutPermutation = Permutation::create(*permutation);
  if (failed(layoutPermutation) || isIdentityPermutation(*permutation))
    return std::nullopt;

  bool hasDataPath = false;
  for (triton::StoreOp store : stores) {
    DenseSet<Value> visited;
    if (valueDependsOnLoad(store.getValue(), loadResults, visited)) {
      hasDataPath = true;
      break;
    }
  }
  if (!hasDataPath)
    return std::nullopt;

  llvm::SetVector<Value> externalValues;
  for (Value init : loop.getInitArgs()) {
    if (isPermutedTensorValue(init, *layoutPermutation))
      externalValues.insert(init);
  }

  for (Operation &operation : *body) {
    if (auto assertOp = dyn_cast<triton::AssertOp>(operation)) {
      // External automatic overflow assertions are handled by the dedicated
      // preheader closure below.  Do not silently transpose a tensor-valued
      // user assertion inside the loop body.
      if (isPermutedTensorValue(assertOp.getCondition(), *layoutPermutation))
        return std::nullopt;
    }
    if (!isSupportedLoopBodyOperation(&operation))
      return std::nullopt;
    for (Value result : operation.getResults()) {
      if (isPermutedTensorValue(result, *layoutPermutation) &&
          !isLayoutTensor(result.getType(), layoutPermutation->rank()))
        return std::nullopt;
    }
    for (Value operand : operation.getOperands()) {
      if (!isPermutedTensorValue(operand, *layoutPermutation))
        continue;
      if (!isLayoutTensor(operand.getType(), layoutPermutation->rank()))
        return std::nullopt;
      auto blockArgument = dyn_cast<BlockArgument>(operand);
      if (blockArgument && blockArgument.getOwner() == body)
        continue;
      Operation *definingOperation = operand.getDefiningOp();
      if (definingOperation && definingOperation->getBlock() == body)
        continue;
      externalValues.insert(operand);
    }
  }

  for (Value value : externalValues) {
    DenseSet<Value> visiting;
    if (!isRebuildableExternalTensor(value, *layoutPermutation, visiting))
      return std::nullopt;
  }

  LayoutPermutationCandidate candidate;
  candidate.loop = loop;
  candidate.anchor = loads.front().getOperation();
  candidate.permutation = *permutation;
  candidate.externalTensorValues.append(externalValues.begin(),
                                        externalValues.end());
  // The loop rewrite clones exactly these direct body operations.  Record them
  // explicitly for the guard use-closure proof instead of treating an
  // arbitrary descendant Region operation as an old-loop consumer.
  for (Operation &operation : *body)
    candidate.blockOperations.push_back(&operation);
  candidate.endpointCount = loads.size() + stores.size();

  for (auto [index, result] : llvm::enumerate(loop.getResults())) {
    if (!isPermutedTensorValue(result, *layoutPermutation))
      continue;
    candidate.transformedResultIndices.push_back(index);
    for (OpOperand &use : result.getUses()) {
      auto reduce = dyn_cast<triton::ReduceOp>(use.getOwner());
      if (!reduce || reduce.getNumOperands() != 1 ||
          use.getOperandNumber() != 0 || reduce->getNumResults() != 1)
        return std::nullopt;
      const int32_t newAxis =
          layoutPermutation->mapOldAxisToNew(reduce.getAxis());
      auto oldInputType = dyn_cast<RankedTensorType>(result.getType());
      FailureOr<SmallVector<int64_t>> newInputShape =
          layoutPermutation->permuteShape(oldInputType.getShape());
      if (newAxis < 0 || failed(newInputShape))
        return std::nullopt;
      RankedTensorType newInputType = oldInputType.clone(*newInputShape);
      std::optional<Type> expectedResultType =
          getReducedResultType(newInputType, newAxis);
      if (!expectedResultType ||
          *expectedResultType != reduce->getResult(0).getType())
        return std::nullopt;
      candidate.reductionPorts.push_back(
          ReductionPort{reduce, static_cast<unsigned>(index)});
    }
  }

  llvm::SetVector<Value> pointerRoots;
  for (const AffineLayoutAccess &access : accesses)
    pointerRoots.insert(access.pointer);
  if (!collectAutomaticOverflowAssertClosure(
          candidate, loop->getBlock(), loop.getOperation(),
          SmallVector<Value, 8>(pointerRoots.begin(), pointerRoots.end()),
          *layoutPermutation))
    return std::nullopt;

  return candidate;
}

// A local candidate never crosses a Region port.  This covers a plain function
// block and a block nested in scf.if/scf.while/index_switch when every value
// that carries the rewritten layout is consumed in that same Block.  scf.for
// is handled by the dedicated complete port adapter above.
bool isSafeExternalBlockTensor(Value value, const Permutation &permutation) {
  DenseSet<Value> rebuildVisiting;
  if (!isRebuildableExternalTensor(value, permutation, rebuildVisiting))
    return false;

  SmallVector<Value, 8> pending = {value};
  DenseSet<Value> visited;
  while (!pending.empty()) {
    Value current = pending.pop_back_val();
    if (!isPermutedTensorValue(current, permutation) ||
        !visited.insert(current).second)
      continue;
    Operation *operation = current.getDefiningOp();
    if (!operation ||
        isa<triton::LoadOp, triton::StoreOp, triton::AssertOp>(operation))
      return false;
    for (Value operand : operation->getOperands())
      pending.push_back(operand);
  }
  return true;
}

std::optional<LayoutPermutationCandidate> matchBlockCandidate(Block *block) {
  if (!block || block->empty())
    return std::nullopt;
  if (auto loop = dyn_cast_or_null<scf::ForOp>(block->getParentOp())) {
    if (loop.getBody() == block)
      return std::nullopt;
  }

  SmallVector<triton::LoadOp, 4> loads;
  SmallVector<triton::StoreOp, 4> stores;
  SmallVector<AffineLayoutAccess, 8> accesses;
  DenseSet<Value> loadResults;
  std::optional<SmallVector<int32_t, 4>> permutation;

  auto recordAccess = [&](Value pointer, Value mask, Type valueType) -> bool {
    auto pointerType = dyn_cast<RankedTensorType>(pointer.getType());
    // Rank-0/rank-1 operations are outside this layout closure and can remain
    // in the block unchanged (for example a post-reduction scalar store).
    if (!pointerType || pointerType.getRank() < 2)
      return true;
    std::optional<AffineLayoutAccess> access = analyzeAffineLayout(pointer);
    if (!access || !isLayoutTensor(valueType, access->type.getRank()) ||
        !hasValidEndpointMask(mask, *access))
      return false;
    if (permutation && !haveSamePermutation(*permutation, access->permutation))
      return false;
    permutation = access->permutation;
    accesses.push_back(std::move(*access));
    return true;
  };

  for (Operation &operation : *block) {
    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (hasUnsupportedLoadAttributes(load) ||
          !isUniformOther(load.getOther()) ||
          !recordAccess(load.getPtr(), load.getMask(),
                        load.getResult().getType()))
        return std::nullopt;
      auto pointerType = dyn_cast<RankedTensorType>(load.getPtr().getType());
      if (pointerType && pointerType.getRank() >= 2) {
        loads.push_back(load);
        loadResults.insert(load.getResult());
      }
      continue;
    }
    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (hasUnsupportedStoreAttributes(store) ||
          !recordAccess(store.getPtr(), store.getMask(),
                        store.getValue().getType()))
        return std::nullopt;
      auto pointerType = dyn_cast<RankedTensorType>(store.getPtr().getType());
      if (pointerType && pointerType.getRank() >= 2)
        stores.push_back(store);
    }
  }

  if (loads.empty() || stores.empty() || !permutation)
    return std::nullopt;
  FailureOr<Permutation> layoutPermutation = Permutation::create(*permutation);
  if (failed(layoutPermutation) || isIdentityPermutation(*permutation))
    return std::nullopt;

  bool hasDataPath = false;
  for (triton::StoreOp store : stores) {
    DenseSet<Value> visited;
    if (valueDependsOnLoad(store.getValue(), loadResults, visited)) {
      hasDataPath = true;
      break;
    }
  }
  if (!hasDataPath)
    return std::nullopt;

  DenseSet<Operation *> selected;
  for (triton::LoadOp load : loads)
    selected.insert(load.getOperation());
  for (triton::StoreOp store : stores)
    selected.insert(store.getOperation());

  bool changed = true;
  while (changed) {
    changed = false;
    for (Operation &operation : *block) {
      if (selected.contains(&operation))
        continue;
      bool usesSelectedValue =
          llvm::any_of(operation.getOperands(), [&](Value operand) {
            Operation *definingOperation = operand.getDefiningOp();
            return definingOperation && selected.contains(definingOperation);
          });
      if (!usesSelectedValue)
        continue;
      // Assert is a zero-result side use, not part of the forward data DAG.
      // In particular, a user assertion must not be cloned/deleted as an
      // incidental consumer of a transformed tensor.
      if (isa<triton::AssertOp>(&operation))
        return std::nullopt;
      if (!isSupportedClonableOperation(&operation))
        return std::nullopt;
      selected.insert(&operation);
      changed = true;
    }
  }

  llvm::SetVector<Value> externalValues;
  SmallVector<Operation *, 16> blockOperations;
  for (Operation &operation : *block) {
    if (!selected.contains(&operation))
      continue;
    blockOperations.push_back(&operation);
    if (!isSupportedClonableOperation(&operation))
      return std::nullopt;
    for (Value result : operation.getResults()) {
      if (isPermutedTensorValue(result, *layoutPermutation) &&
          !isLayoutTensor(result.getType(), layoutPermutation->rank()))
        return std::nullopt;
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != block ||
            !selected.contains(use.getOwner()))
          return std::nullopt;
      }
    }
    for (Value operand : operation.getOperands()) {
      if (!isPermutedTensorValue(operand, *layoutPermutation))
        continue;
      if (!isLayoutTensor(operand.getType(), layoutPermutation->rank()))
        return std::nullopt;
      Operation *definingOperation = operand.getDefiningOp();
      if (!definingOperation || !selected.contains(definingOperation))
        externalValues.insert(operand);
    }
  }

  for (Value value : externalValues) {
    if (!isSafeExternalBlockTensor(value, *layoutPermutation))
      return std::nullopt;
  }

  LayoutPermutationCandidate candidate;
  candidate.block = block;
  candidate.anchor = loads.front().getOperation();
  candidate.permutation = *permutation;
  candidate.externalTensorValues.append(externalValues.begin(),
                                        externalValues.end());
  candidate.blockOperations = std::move(blockOperations);
  candidate.endpointCount = loads.size() + stores.size();
  llvm::SetVector<Value> pointerRoots;
  for (const AffineLayoutAccess &access : accesses)
    pointerRoots.insert(access.pointer);
  // Unlike scf.for's dedicated preheader adapter, a generic Block rewrite
  // does not synthesize Region arguments/results for a type-changing pointer.
  // A branch-local load/store may therefore only use a pointer materialized in
  // that exact Block; accepting a parent Block pointer would leave an old/new
  // layout port across scf.if/scf.while.
  for (Value pointer : pointerRoots) {
    Operation *definingOperation = pointer.getDefiningOp();
    if (!definingOperation || definingOperation->getBlock() != block)
      return std::nullopt;
  }
  if (!collectAutomaticOverflowAssertClosure(
          candidate, block, candidate.anchor,
          SmallVector<Value, 8>(pointerRoots.begin(), pointerRoots.end()),
          *layoutPermutation))
    return std::nullopt;
  return candidate;
}

class LayoutValueCloner {
public:
  LayoutValueCloner(IRRewriter &rewriter, const Permutation &permutation,
                    SmallVectorImpl<Operation *> &created)
      : rewriter(rewriter), permutation(permutation), created(created) {}

  std::optional<Value> clone(Value value) {
    if (!isPermutedTensorValue(value, permutation))
      return value;
    if (auto found = values.find(value); found != values.end())
      return found->second;
    if (!isLayoutTensor(value.getType(), permutation.rank()) ||
        !visiting.insert(value).second)
      return std::nullopt;

    Operation *operation = value.getDefiningOp();
    if (!operation || !isSupportedClonableOperation(operation))
      return std::nullopt;

    std::optional<Operation *> rebuilt = rebuildOperation(operation);
    if (!rebuilt)
      return std::nullopt;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), (*rebuilt)->getResults()))
      values[oldResult] = newResult;
    return values.lookup(value);
  }

  std::optional<Value> get(Value value) const {
    if (!isPermutedTensorValue(value, permutation))
      return value;
    auto found = values.find(value);
    if (found == values.end())
      return std::nullopt;
    return found->second;
  }

private:
  std::optional<Value> mapExternalOperand(Value value) {
    if (!isPermutedTensorValue(value, permutation))
      return value;
    return clone(value);
  }

  std::optional<Operation *> rebuildOperation(Operation *operation) {
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
      auto oldType = dyn_cast<RankedTensorType>(constant.getResult().getType());
      std::optional<RankedTensorType> newType =
          oldType ? getPermutedTensorType(oldType, permutation) : std::nullopt;
      if (!dense || !dense.isSplat() || !newType)
        return std::nullopt;
      SmallVector<Attribute, 1> elements = {dense.getSplatValue<Attribute>()};
      auto newDense = DenseElementsAttr::get(*newType, elements);
      auto rebuilt = rewriter.create<arith::ConstantOp>(operation->getLoc(),
                                                        *newType, newDense);
      created.push_back(rebuilt.getOperation());
      return rebuilt.getOperation();
    }

    SmallVector<Value, 4> operands;
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      std::optional<Value> mapped = mapExternalOperand(operand);
      if (!mapped)
        return std::nullopt;
      operands.push_back(*mapped);
    }

    SmallVector<Type, 2> resultTypes;
    resultTypes.reserve(operation->getNumResults());
    for (Value result : operation->getResults()) {
      std::optional<Type> type = getPermutedType(result.getType(), permutation);
      if (!type)
        return std::nullopt;
      resultTypes.push_back(*type);
    }

    if (auto trans = dyn_cast<triton::TransOp>(operation)) {
      if (operands.size() != 1 || resultTypes.size() != 1)
        return std::nullopt;
      return rebuildTransOperation(rewriter, trans, operands.front(),
                                   resultTypes.front(), permutation, &created);
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
      if (isPermutedTensorValue(expand.getResult(), permutation)) {
        std::optional<Value> source = getExpandRankOneSource(expand);
        if (!source)
          return std::nullopt;
        std::optional<Value> mappedSource = mapExternalOperand(*source);
        if (!mappedSource)
          return std::nullopt;
        std::optional<Value> rebuilt = rebuildFullRankExpand(
            rewriter, expand, *mappedSource, permutation, &created);
        if (!rebuilt)
          return std::nullopt;
        return rebuilt->getDefiningOp();
      }

      if (resultTypes.size() != 1)
        return std::nullopt;
      auto rebuilt = rewriter.create<triton::ExpandDimsOp>(
          operation->getLoc(), cast<RankedTensorType>(resultTypes.front()),
          operands.front(), expand.getAxis());
      created.push_back(rebuilt.getOperation());
      return rebuilt.getOperation();
    }

    OperationState state(operation->getLoc(), operation->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(operation->getAttrs());
    Operation *rebuilt = rewriter.create(state);
    created.push_back(rebuilt);
    return rebuilt;
  }

  IRRewriter &rewriter;
  const Permutation &permutation;
  SmallVectorImpl<Operation *> &created;
  DenseMap<Value, Value> values;
  DenseSet<Value> visiting;
};

std::optional<Value> mapLoopValue(Value value, IRMapping &mapping,
                                  LayoutValueCloner &externalValues,
                                  const Permutation &permutation) {
  if (mapping.contains(value))
    return mapping.lookup(value);
  if (!isPermutedTensorValue(value, permutation))
    return value;
  return externalValues.get(value);
}

std::optional<Operation *>
cloneLoopOperation(IRRewriter &rewriter, Operation *operation,
                   IRMapping &mapping, LayoutValueCloner &externalValues,
                   const Permutation &permutation,
                   SmallVectorImpl<Operation *> *created = nullptr) {
  if (!isSupportedClonableOperation(operation))
    return std::nullopt;

  if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
    auto dense = dyn_cast<DenseElementsAttr>(constant.getValue());
    auto oldType = dyn_cast<RankedTensorType>(constant.getResult().getType());
    if (!dense || !dense.isSplat() || !oldType)
      return std::nullopt;
    std::optional<RankedTensorType> newType =
        getPermutedTensorType(oldType, permutation);
    if (!newType)
      return std::nullopt;
    SmallVector<Attribute, 1> elements = {dense.getSplatValue<Attribute>()};
    auto rebuilt = rewriter.create<arith::ConstantOp>(
        operation->getLoc(), *newType,
        DenseElementsAttr::get(*newType, elements));
    if (created)
      created->push_back(rebuilt.getOperation());
    mapping.map(constant.getResult(), rebuilt.getResult());
    return rebuilt.getOperation();
  }

  SmallVector<Value, 4> operands;
  operands.reserve(operation->getNumOperands());
  for (Value operand : operation->getOperands()) {
    std::optional<Value> mapped =
        mapLoopValue(operand, mapping, externalValues, permutation);
    if (!mapped)
      return std::nullopt;
    operands.push_back(*mapped);
  }

  SmallVector<Type, 2> resultTypes;
  resultTypes.reserve(operation->getNumResults());
  for (Value result : operation->getResults()) {
    std::optional<Type> type = getPermutedType(result.getType(), permutation);
    if (!type)
      return std::nullopt;
    resultTypes.push_back(*type);
  }

  Operation *rebuilt = nullptr;
  if (auto trans = dyn_cast<triton::TransOp>(operation)) {
    if (operands.size() != 1 || resultTypes.size() != 1)
      return std::nullopt;
    std::optional<Operation *> rebuiltTrans =
        rebuildTransOperation(rewriter, trans, operands.front(),
                              resultTypes.front(), permutation, created);
    if (!rebuiltTrans)
      return std::nullopt;
    rebuilt = *rebuiltTrans;
  } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
    if (isPermutedTensorValue(expand.getResult(), permutation)) {
      std::optional<Value> source = getExpandRankOneSource(expand);
      if (!source)
        return std::nullopt;
      std::optional<Value> mappedSource =
          mapLoopValue(*source, mapping, externalValues, permutation);
      if (!mappedSource)
        return std::nullopt;
      std::optional<Value> rebuiltValue = rebuildFullRankExpand(
          rewriter, expand, *mappedSource, permutation, created);
      if (!rebuiltValue)
        return std::nullopt;
      rebuilt = rebuiltValue->getDefiningOp();
    } else {
      if (resultTypes.size() != 1)
        return std::nullopt;
      rebuilt = rewriter
                    .create<triton::ExpandDimsOp>(
                        operation->getLoc(),
                        cast<RankedTensorType>(resultTypes.front()),
                        operands.front(), expand.getAxis())
                    .getOperation();
      if (created)
        created->push_back(rebuilt);
    }
  } else {
    OperationState state(operation->getLoc(), operation->getName());
    state.addOperands(operands);
    state.addTypes(resultTypes);
    state.addAttributes(operation->getAttrs());
    rebuilt = rewriter.create(state);
    if (created)
      created->push_back(rebuilt);
  }

  for (auto [oldResult, newResult] :
       llvm::zip(operation->getResults(), rebuilt->getResults()))
    mapping.map(oldResult, newResult);
  return rebuilt;
}

std::optional<triton::ReduceOp> cloneReduction(IRRewriter &rewriter,
                                               triton::ReduceOp reduce,
                                               Value newInput,
                                               const Permutation &permutation) {
  if (reduce.getNumOperands() != 1 || reduce->getNumResults() != 1)
    return std::nullopt;
  const int32_t newAxis = permutation.mapOldAxisToNew(reduce.getAxis());
  if (newAxis < 0)
    return std::nullopt;

  auto rebuilt = rewriter.create<triton::ReduceOp>(
      reduce.getLoc(), ValueRange{newInput}, static_cast<int>(newAxis));
  rewriter.cloneRegionBefore(reduce.getCombineOp(), rebuilt.getCombineOp(),
                             rebuilt.getCombineOp().end());
  for (NamedAttribute attribute : reduce->getAttrs()) {
    if (!rebuilt->hasAttr(attribute.getName()))
      rebuilt->setAttr(attribute.getName(), attribute.getValue());
  }
  return rebuilt;
}

using AssertConditionReplacement = std::pair<Operation *, Value>;

std::optional<SmallVector<AssertConditionReplacement, 8>>
cloneExternalAssertConditions(IRRewriter &rewriter,
                              ArrayRef<ExternalAssertPort> assertPorts,
                              LayoutValueCloner &externalValues,
                              const Permutation &permutation) {
  SmallVector<AssertConditionReplacement, 8> replacements;
  replacements.reserve(assertPorts.size());
  for (const ExternalAssertPort &port : assertPorts) {
    auto assertOp = dyn_cast_or_null<triton::AssertOp>(port.assertOperation);
    if (!assertOp || assertOp.getCondition() != port.oldCondition)
      return std::nullopt;
    std::optional<Type> expectedType =
        getPermutedType(port.oldCondition.getType(), permutation);
    if (!expectedType)
      return std::nullopt;

    // Keep the assert in place.  Inserting its replacement condition directly
    // before it preserves both the original lexical assertion order and the
    // overflow check's required dominance over the checked i32 operation.
    rewriter.setInsertionPoint(assertOp);
    std::optional<Value> condition = externalValues.clone(port.oldCondition);
    if (!condition || condition->getType() != *expectedType)
      return std::nullopt;
    replacements.push_back({assertOp.getOperation(), *condition});
  }
  return replacements;
}

bool verifyCreatedOperations(ArrayRef<Operation *> created) {
  return llvm::all_of(created, [](Operation *operation) {
    return operation && succeeded(mlir::verify(operation));
  });
}

void commitAssertConditionReplacements(
    IRRewriter &rewriter, ArrayRef<AssertConditionReplacement> replacements) {
  for (const AssertConditionReplacement &replacement : replacements) {
    rewriter.modifyOpInPlace(replacement.first, [&] {
      replacement.first->setOperand(0, replacement.second);
    });
  }
}

void eraseDeadPureClosure(IRRewriter &rewriter,
                          ArrayRef<Operation *> retireOperations) {
  SmallVector<Operation *, 32> pending(retireOperations.begin(),
                                       retireOperations.end());
  bool erasedAny = true;
  while (erasedAny) {
    erasedAny = false;
    for (Operation *&operation : llvm::reverse(pending)) {
      if (!operation || !isRetirablePureProducer(operation))
        continue;
      if (llvm::any_of(operation->getResults(),
                       [](Value result) { return !result.use_empty(); }))
        continue;
      rewriter.eraseOp(operation);
      operation = nullptr;
      erasedAny = true;
    }
  }
}

bool canCommitLoopRewrite(const LayoutPermutationCandidate &candidate,
                          scf::ForOp oldFor, scf::ForOp newFor) {
  if (!oldFor || !newFor || oldFor.getNumResults() != newFor.getNumResults())
    return false;

  DenseSet<unsigned> transformed(candidate.transformedResultIndices.begin(),
                                 candidate.transformedResultIndices.end());
  DenseSet<Operation *> expectedReductions;
  for (const ReductionPort &port : candidate.reductionPorts) {
    if (!port.reduce || port.forResultIndex >= oldFor.getNumResults())
      return false;
    expectedReductions.insert(static_cast<Operation *>(port.reduce));
  }

  for (auto [index, oldResult] : llvm::enumerate(oldFor.getResults())) {
    if (!transformed.contains(index)) {
      if (oldResult.getType() != newFor.getResult(index).getType())
        return false;
      continue;
    }
    for (OpOperand &use : oldResult.getUses()) {
      if (!expectedReductions.contains(use.getOwner()))
        return false;
    }
  }
  return true;
}

bool canCommitBlockRewrite(const LayoutPermutationCandidate &candidate,
                           const IRMapping &mapping) {
  for (Operation *operation : candidate.blockOperations) {
    if (!operation || operation->getBlock() != candidate.block)
      return false;
    for (Value result : operation->getResults()) {
      if (!mapping.contains(result))
        return false;
      for (OpOperand &use : result.getUses()) {
        if (use.getOwner()->getBlock() != candidate.block ||
            !llvm::is_contained(candidate.blockOperations, use.getOwner()))
          return false;
      }
    }
  }
  return true;
}

void eraseCreatedOperations(IRRewriter &rewriter,
                            ArrayRef<Operation *> created) {
  for (Operation *operation : llvm::reverse(created)) {
    if (operation)
      rewriter.eraseOp(operation);
  }
}

LogicalResult applyCandidate(IRRewriter &rewriter,
                             const LayoutPermutationCandidate &candidate) {
  FailureOr<Permutation> permutation =
      Permutation::create(candidate.permutation);
  if (failed(permutation))
    return failure();

  scf::ForOp oldFor = candidate.loop;
  if (!oldFor || !oldFor.getBody())
    return failure();

  SmallVector<Operation *, 32> created;
  rewriter.setInsertionPoint(oldFor);
  LayoutValueCloner externalValues(rewriter, *permutation, created);
  std::optional<SmallVector<AssertConditionReplacement, 8>> assertReplacements =
      cloneExternalAssertConditions(rewriter, candidate.externalAssertPorts,
                                    externalValues, *permutation);
  if (!assertReplacements) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  rewriter.setInsertionPoint(oldFor);
  for (Value value : candidate.externalTensorValues) {
    if (!externalValues.clone(value)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
  }

  SmallVector<Value, 8> newInitArgs;
  newInitArgs.reserve(oldFor.getInitArgs().size());
  for (Value init : oldFor.getInitArgs()) {
    std::optional<Value> mapped = externalValues.get(init);
    if (!mapped) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    newInitArgs.push_back(*mapped);
  }

  auto newFor = rewriter.create<scf::ForOp>(
      oldFor.getLoc(), oldFor.getLowerBound(), oldFor.getUpperBound(),
      oldFor.getStep(), newInitArgs);
  created.push_back(newFor.getOperation());
  Block *newBody = newFor.getBody();
  // The generated SCF builder may leave a newly created body empty.  Do not
  // query getTerminator() in that state: Block intentionally asserts when it
  // has not yet been marked as containing a terminator.
  if (!newBody->empty())
    rewriter.eraseOp(&newBody->back());

  IRMapping mapping;
  Block *oldBody = oldFor.getBody();
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));
  for (unsigned index = 0; index < oldFor.getInitArgs().size(); ++index)
    mapping.map(oldBody->getArgument(index + 1),
                newBody->getArgument(index + 1));

  rewriter.setInsertionPointToEnd(newBody);
  for (Operation &operation : *oldBody) {
    if (auto yield = dyn_cast<scf::YieldOp>(operation)) {
      SmallVector<Value, 8> operands;
      operands.reserve(yield.getNumOperands());
      for (Value operand : yield.getOperands()) {
        std::optional<Value> mapped =
            mapLoopValue(operand, mapping, externalValues, *permutation);
        if (!mapped) {
          eraseCreatedOperations(rewriter, created);
          return failure();
        }
        operands.push_back(*mapped);
      }
      rewriter.create<scf::YieldOp>(yield.getLoc(), operands);
      continue;
    }

    if (!cloneLoopOperation(rewriter, &operation, mapping, externalValues,
                            *permutation)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
  }

  if (failed(mlir::verify(newFor.getOperation()))) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  SmallVector<std::pair<triton::ReduceOp, triton::ReduceOp>, 4> reductions;
  for (const ReductionPort &port : candidate.reductionPorts) {
    if (!port.reduce || port.forResultIndex >= newFor.getNumResults()) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    rewriter.setInsertionPoint(port.reduce);
    std::optional<triton::ReduceOp> rebuilt =
        cloneReduction(rewriter, port.reduce,
                       newFor.getResult(port.forResultIndex), *permutation);
    if (rebuilt)
      created.push_back((*rebuilt).getOperation());
    if (!rebuilt ||
        (*rebuilt)->getResult(0).getType() !=
            port.reduce->getResult(0).getType() ||
        failed(mlir::verify((*rebuilt).getOperation()))) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    reductions.push_back({port.reduce, *rebuilt});
  }

  if (!verifyCreatedOperations(created) ||
      !canCommitLoopRewrite(candidate, oldFor, newFor)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  // All fallible work has completed.  Keep the old assert operations and only
  // change their conditions, then replace the old loop as one committed unit.
  commitAssertConditionReplacements(rewriter, *assertReplacements);
  DenseSet<unsigned> transformed(candidate.transformedResultIndices.begin(),
                                 candidate.transformedResultIndices.end());
  for (auto [index, oldResult] : llvm::enumerate(oldFor.getResults())) {
    if (transformed.contains(index))
      continue;
    oldResult.replaceAllUsesWith(newFor.getResult(index));
  }
  for (auto &pair : reductions)
    pair.first->getResult(0).replaceAllUsesWith(pair.second->getResult(0));
  for (auto &pair : reductions)
    rewriter.eraseOp(pair.first.getOperation());

  rewriter.eraseOp(oldFor.getOperation());
  eraseDeadPureClosure(rewriter, candidate.retireOperations);
  return success();
}

LogicalResult applyBlockCandidate(IRRewriter &rewriter,
                                  const LayoutPermutationCandidate &candidate) {
  if (!candidate.block || !candidate.anchor ||
      candidate.blockOperations.empty())
    return failure();
  FailureOr<Permutation> permutation =
      Permutation::create(candidate.permutation);
  if (failed(permutation))
    return failure();

  SmallVector<Operation *, 32> created;
  rewriter.setInsertionPoint(candidate.anchor);
  LayoutValueCloner externalValues(rewriter, *permutation, created);
  std::optional<SmallVector<AssertConditionReplacement, 8>> assertReplacements =
      cloneExternalAssertConditions(rewriter, candidate.externalAssertPorts,
                                    externalValues, *permutation);
  if (!assertReplacements) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  rewriter.setInsertionPoint(candidate.anchor);
  for (Value value : candidate.externalTensorValues) {
    if (!externalValues.clone(value)) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
  }

  IRMapping mapping;
  for (Operation *operation : candidate.blockOperations) {
    if (!operation || operation->getBlock() != candidate.block) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
    rewriter.setInsertionPoint(operation);
    std::optional<Operation *> rebuilt = cloneLoopOperation(
        rewriter, operation, mapping, externalValues, *permutation, &created);
    if (!rebuilt || failed(mlir::verify(*rebuilt))) {
      eraseCreatedOperations(rewriter, created);
      return failure();
    }
  }

  // Complete every validation before changing an old use.  The freshly cloned
  // operations are inserted in the same Block, so the preflight must target
  // only the original selected closure, not merely every operation in Block.
  if (!verifyCreatedOperations(created) ||
      !canCommitBlockRewrite(candidate, mapping)) {
    eraseCreatedOperations(rewriter, created);
    return failure();
  }

  commitAssertConditionReplacements(rewriter, *assertReplacements);

  // Rewire only the original selected closure.  The freshly cloned operations
  // are inserted in the same Block, so checking the Block alone would also
  // rewrite their already-correct operands.  matchBlockCandidate has proven
  // that every old full-rank result use stays inside this selected component;
  // no layout value is allowed to escape through a Region port.
  for (Operation *operation : candidate.blockOperations) {
    for (Value result : operation->getResults()) {
      Value replacement = mapping.lookup(result);
      result.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        return use.getOwner()->getBlock() == candidate.block &&
               llvm::is_contained(candidate.blockOperations, use.getOwner());
      });
    }
  }

  // The preflight above proves each old result is now dead after consumers are
  // rewired, so no fallible path remains after the assert conditions commit.
  for (Operation *operation : llvm::reverse(candidate.blockOperations)) {
    rewriter.eraseOp(operation);
  }
  eraseDeadPureClosure(rewriter, candidate.retireOperations);
  return success();
}

class LayoutPermutationPlan final : public RewritePlan {
public:
  LayoutPermutationPlan(LayoutPermutationCandidate candidate, unsigned epoch)
      : loop(candidate.loop), block(candidate.block), anchor(candidate.anchor),
        permutation(std::move(candidate.permutation)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::LoadStoreTranspose;
  }

  unsigned getBenefit() const override { return 1; }

  Operation *getAnchor() const override { return anchor; }

  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getEpoch() != epoch || !anchor || (!loop && !block) ||
        (loop && block))
      return failure();
    triton::FuncOp function = anchor->getParentOfType<triton::FuncOp>();
    if (!function || function != context.getFunction())
      return failure();
    std::optional<LayoutPermutationCandidate> current =
        loop ? matchLoopCandidate(loop) : matchBlockCandidate(block);
    if (!current || current->anchor != anchor ||
        !haveSamePermutation(current->permutation, permutation))
      return failure();
    return success();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    std::optional<LayoutPermutationCandidate> candidate =
        loop ? matchLoopCandidate(loop) : matchBlockCandidate(block);
    if (!candidate || candidate->anchor != anchor ||
        !haveSamePermutation(candidate->permutation, permutation))
      return failure();
    return loop ? applyCandidate(rewriter, *candidate)
                : applyBlockCandidate(rewriter, *candidate);
  }

private:
  scf::ForOp loop;
  Block *block;
  Operation *anchor;
  SmallVector<int32_t, 4> permutation;
  unsigned epoch;
};

class LoadStoreTransposeRule final : public GraphOptimizationRule {
public:
  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::LoadStoreTranspose;
  }

  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::None;
  }

  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    context.getFunction().walk([&](scf::ForOp loop) {
      std::optional<LayoutPermutationCandidate> candidate =
          matchLoopCandidate(loop);
      if (candidate) {
        LLVM_DEBUG({
          llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                       << static_cast<unsigned>(getId()) << " ("
                       << getGraphOptimizationRuleName(getId()) << ") at loop "
                       << candidate->anchor->getLoc() << ": perm=[";
          llvm::interleaveComma(candidate->permutation, llvm::dbgs());
          llvm::dbgs() << "] endpoints=" << candidate->endpointCount << "\n";
        });
        plans.push_back(std::make_unique<LayoutPermutationPlan>(
            std::move(*candidate), context.getEpoch()));
      }
    });

    llvm::SetVector<Block *> blocks;
    auto collectBlocks = [&](Operation *operation) {
      for (Region &region : operation->getRegions()) {
        for (Block &block : region)
          blocks.insert(&block);
      }
    };
    collectBlocks(context.getFunction().getOperation());
    context.getFunction().walk(
        [&](Operation *operation) { collectBlocks(operation); });
    for (Block *block : blocks) {
      std::optional<LayoutPermutationCandidate> candidate =
          matchBlockCandidate(block);
      if (candidate) {
        LLVM_DEBUG({
          llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                       << static_cast<unsigned>(getId()) << " ("
                       << getGraphOptimizationRuleName(getId()) << ") at block "
                       << candidate->anchor->getLoc() << ": perm=[";
          llvm::interleaveComma(candidate->permutation, llvm::dbgs());
          llvm::dbgs() << "] endpoints=" << candidate->endpointCount << "\n";
        });
        plans.push_back(std::make_unique<LayoutPermutationPlan>(
            std::move(*candidate), context.getEpoch()));
      }
    }
    return success();
  }
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createLoadStoreTransposeRule() {
  return std::make_unique<LoadStoreTransposeRule>();
}
