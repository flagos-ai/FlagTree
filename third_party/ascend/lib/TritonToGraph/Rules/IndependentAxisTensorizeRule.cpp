/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to do so, subject to the
 * following conditions:
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

#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/IndependentRowReductionAnalysis.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr llvm::StringLiteral kIndependentAxisTensorizeMarkerAttr =
    "hacc.independent_axis_tensorize";
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";
constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kCoalesceGridCeilDivAttr =
    "hacc.coalesce_grid_ceil_div";

constexpr std::array<unsigned, 3> kMergeTensorizeFactors = {2, 4, 8};
constexpr std::array<unsigned, 2> kMergeLargeTensorizeFactors = {16, 32};
constexpr std::array<unsigned, 4> kNormTensorizeFactors = {2, 4, 8, 16};
constexpr std::array<unsigned, 7> kNormPersistentBlockTCandidates = {
    2, 4, 8, 16, 32, 64, 128};

constexpr uint64_t kMergeSplitMaxTensorizedPlanes = 16;

constexpr int64_t kMergeSplitD192SourceTile = 256;
constexpr int64_t kMergeSplitD192TargetTile = 192;
constexpr unsigned kDynamicSingleMomentBlockT = 4;

enum class TensorizeForm : uint8_t {
  MergeSplit,
  NormRope,
};

enum class IATCandidateScope : uint8_t {
  Default,
  MergeSplitLarge,
};

struct IATCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  TensorizeForm form = TensorizeForm::MergeSplit;
  int32_t axis = 1;
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
  unsigned factor = 1;
  bool usesMergeSplitSubCorePolicy = false;
  bool usesMergeSplitD192Tile = false;
  bool requiresPersistentChaining = false;
  ResourceSnapshot resources;
  CandidateEvaluation evaluation;
};

struct JointProgramMappingCandidate {
  IATCandidate iat;
  unsigned blockT = 0;
  std::array<uint64_t, 3> finalLogicalGrid = {1, 1, 1};
  std::array<uint64_t, 3> finalPhysicalGrid = {1, 1, 1};
  CandidateEvaluation evaluation;
};

struct TensorizeReductionShape {
  int64_t splitExtent = 0;
  int64_t dimExtent = 0;
};

struct TensorizeFormMatch {
  TensorizeForm form;
  TensorizeReductionShape reduction;
};

struct MappedValue {
  Value value;
  bool tensorized = false;
  int64_t laneAxis = 0;
};

std::optional<triton::GetProgramIdOp> findOnlyProgramId(triton::FuncOp function,
                                                        int32_t axis) {
  std::optional<triton::GetProgramIdOp> result;
  unsigned count = 0;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() != axis)
      return;
    ++count;
    if (count == 1)
      result = pid;
  });
  return count == 1 ? result : std::nullopt;
}

std::optional<TensorizeFormMatch>
classifyTensorizeForm(triton::FuncOp function) {
  std::optional<TensorizeReductionShape> mergeReduction;
  std::optional<TensorizeReductionShape> normReduction;
  function.walk([&](triton::ReduceOp reduce) {
    if (reduce.getSrcs().size() != 1 || reduce.getResults().size() != 1 ||
        reduce.getAxis() != 0)
      return;
    auto source =
        dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
    if (!source || !source.hasStaticShape())
      return;
    Type result = reduce.getResults().front().getType();
    auto rankedResult = dyn_cast<RankedTensorType>(result);
    if (source.getRank() == 2 && rankedResult && rankedResult.getRank() == 1 &&
        rankedResult.hasStaticShape() &&
        source.getShape()[1] == rankedResult.getShape()[0]) {
      if (!mergeReduction)
        mergeReduction =
            TensorizeReductionShape{source.getShape()[0], source.getShape()[1]};
    } else if (source.getRank() == 1 && !isa<RankedTensorType>(result)) {
      if (!normReduction)
        normReduction = TensorizeReductionShape{0, source.getShape()[0]};
    }
  });
  if (mergeReduction)
    return TensorizeFormMatch{TensorizeForm::MergeSplit, *mergeReduction};
  if (normReduction)
    return TensorizeFormMatch{TensorizeForm::NormRope, *normReduction};
  return std::nullopt;
}

bool hasSupportedControlFlow(triton::FuncOp function) {
  Region &body = function.getBody();
  if (!body.hasOneBlock())
    return false;
  for (Operation &operation : body.front()) {
    if (isa<CallOpInterface>(operation))
      return false;
    StringRef name = operation.getName().getStringRef();
    if (name.contains("atomic") || name.contains("barrier") ||
        name == "cf.br" || name == "cf.cond_br" || name.starts_with("scf."))
      return false;
    if (operation.getNumRegions() != 0 &&
        !isa<triton::ReduceOp, triton::ScanOp>(operation))
      return false;
  }
  return true;
}

bool hasConflictingLaunchContract(ModuleOp module) {
  return module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
         module->hasAttr(kProgramGridTransformsAttr) ||
         module->hasAttr(kCoalesceFactorAttr) ||
         module->hasAttr(kCoalesceAxisAttr) ||
         module->hasAttr(kCoalesceGridCeilDivAttr);
}

bool hasDisjointWriteReadRoots(
    const ProgramAxisDependence &dependence,
    const EntryArgPointerAliasAnalysis &entryPointerAliases) {
  for (const StoreAddressDependence &store : dependence.stores) {
    auto storeOp = dyn_cast_or_null<triton::StoreOp>(store.store);
    if (!storeOp)
      return false;
    for (Operation *operation : dependence.dependenceClosure) {
      auto load = dyn_cast<triton::LoadOp>(operation);
      if (!load)
        continue;
      if (entryPointerAliases.classify(storeOp.getPtr(), load.getPtr()) !=
          EntryArgPointerRelation::DistinctEntryRoots)
        return false;
    }
  }
  return true;
}

bool hasTensorPointerInProgramAxisClosure(
    const ProgramAxisDependence &dependence) {
  // A tensor pointer requires a scalar base pointer. IAT turns the dependent
  // program id into a vector of lanes, and the current materializer cannot
  // reconstruct a legal tensor pointer from that vector. Reject this case
  // while discovering candidates instead of scheduling a plan that fails in
  // apply().
  return llvm::any_of(dependence.dependenceClosure, [](Operation *operation) {
    return isa<triton::MakeTensorPtrOp>(operation);
  });
}

int64_t getPreferredLaneAxis(Type originalType, const IATCandidate &candidate) {
  auto tensor = dyn_cast<RankedTensorType>(originalType);
  if (!tensor || candidate.form == TensorizeForm::NormRope)
    return 0;
  const int64_t rank = tensor.getRank();
  if (rank == 0)
    return 0;
  if (rank == 1)
    return tensor.getShape().front() == candidate.splitExtent ? 1 : 0;
  if (tensor.getShape().front() == candidate.splitExtent ||
      tensor.getShape().front() == 1)
    return 1;
  return 0;
}

std::optional<RankedTensorType>
getLiftedTensorType(Type originalType, const IATCandidate &candidate,
                    int64_t laneAxis) {
  Type elementType = originalType;
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType)) {
    if (laneAxis < 0 || laneAxis > tensor.getRank())
      return std::nullopt;
    shape.assign(tensor.getShape().begin(), tensor.getShape().end());
    elementType = tensor.getElementType();
  } else if (laneAxis != 0) {
    return std::nullopt;
  }
  shape.insert(shape.begin() + laneAxis, candidate.factor);
  return RankedTensorType::get(shape, elementType);
}

std::optional<RankedTensorType>
getSameShapeTensorType(Type originalType, ArrayRef<int64_t> shape) {
  Type elementType = originalType;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType))
    elementType = tensor.getElementType();
  return RankedTensorType::get(shape, elementType);
}

bool hasSameShape(RankedTensorType lhs, RankedTensorType rhs) {
  return lhs.getRank() == rhs.getRank() &&
         llvm::equal(lhs.getShape(), rhs.getShape());
}

Value makeZero(IRRewriter &rewriter, Location loc, Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type elementType = tensor ? tensor.getElementType() : type;
  if (auto integer = dyn_cast<IntegerType>(elementType)) {
    TypedAttr zero = IntegerAttr::get(integer, 0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  if (auto floating = dyn_cast<FloatType>(elementType)) {
    TypedAttr zero = FloatAttr::get(floating, 0.0);
    return tensor ? rewriter.create<arith::ConstantOp>(
                        loc, type, DenseElementsAttr::get(tensor, zero))
                  : rewriter.create<arith::ConstantOp>(loc, type, zero);
  }
  return Value();
}

bool rebuildTensorizedFunction(triton::FuncOp function,
                               const IATCandidate &candidate,
                               IRRewriter &rewriter) {
  if (!hasSupportedControlFlow(function))
    return false;
  if (failed(addProgramGridHiddenExtentArguments(function)))
    return false;
  Value originalExtent = function.getArgument(function.getNumArguments() - 1);
  auto originalExtentType = dyn_cast<IntegerType>(originalExtent.getType());
  if (!originalExtentType || originalExtentType.getWidth() != 32)
    return false;
  std::optional<triton::GetProgramIdOp> targetPid =
      findOnlyProgramId(function, candidate.axis);
  if (!targetPid)
    return false;

  ProgramAxisDependenceAnalysis analysis(function);
  const ProgramAxisDependence &dependence = analysis.get(candidate.axis);
  if (!dependence.isProgramMappingTransformCandidate())
    return false;
  DenseSet<Operation *> closure;
  for (Operation *operation : dependence.dependenceClosure)
    closure.insert(operation);

  Block &block = function.getBody().front();
  SmallVector<Operation *> originals;
  originals.reserve(block.getOperations().size());
  for (Operation &operation : block)
    originals.push_back(&operation);

  DenseMap<Value, MappedValue> values;
  auto lookup = [&](Value value) -> MappedValue {
    auto it = values.find(value);
    return it == values.end() ? MappedValue{value, false, 0} : it->second;
  };

  auto broadcastTo = [&](Location loc, Value value,
                         RankedTensorType target) -> Value {
    auto source = dyn_cast<RankedTensorType>(value.getType());
    if (!source || source.getElementType() != target.getElementType() ||
        source.getRank() != target.getRank())
      return Value();
    for (auto [sourceDim, targetDim] :
         llvm::zip(source.getShape(), target.getShape()))
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    return hasSameShape(source, target)
               ? value
               : rewriter.create<triton::BroadcastOp>(loc, target, value);
  };

  auto alignTensorized = [&](Location loc, MappedValue mapped,
                             RankedTensorType target,
                             int64_t targetLaneAxis) -> Value {
    auto source = dyn_cast<RankedTensorType>(mapped.value.getType());
    if (!source || mapped.laneAxis < 0 || mapped.laneAxis >= source.getRank() ||
        targetLaneAxis < 0 || targetLaneAxis >= target.getRank() ||
        source.getElementType() != target.getElementType())
      return Value();
    const int64_t sourceLaneAxis = mapped.laneAxis;
    const int64_t sourceTail = source.getRank() - sourceLaneAxis - 1;
    const int64_t targetTail = target.getRank() - targetLaneAxis - 1;
    if (sourceLaneAxis > targetLaneAxis || sourceTail > targetTail ||
        source.getShape()[sourceLaneAxis] != candidate.factor ||
        target.getShape()[targetLaneAxis] != candidate.factor)
      return Value();

    SmallVector<int64_t> sourceToTarget(source.getRank(), -1);
    for (int64_t index = 0; index < sourceLaneAxis; ++index)
      sourceToTarget[index] = index;
    sourceToTarget[sourceLaneAxis] = targetLaneAxis;
    for (int64_t index = 0; index < sourceTail; ++index)
      sourceToTarget[sourceLaneAxis + 1 + index] =
          target.getRank() - sourceTail + index;

    SmallVector<bool> targetMapped(target.getRank(), false);
    for (int64_t sourceIndex = 0; sourceIndex < source.getRank();
         ++sourceIndex) {
      const int64_t targetIndex = sourceToTarget[sourceIndex];
      if (targetIndex < 0 || targetIndex >= target.getRank() ||
          targetMapped[targetIndex])
        return Value();
      targetMapped[targetIndex] = true;
      const int64_t sourceDim = source.getShape()[sourceIndex];
      const int64_t targetDim = target.getShape()[targetIndex];
      if (sourceDim != targetDim && sourceDim != 1)
        return Value();
    }

    Value current = mapped.value;
    int64_t currentAxis = 0;
    for (int64_t targetIndex = 0; targetIndex < target.getRank();
         ++targetIndex) {
      if (targetMapped[targetIndex]) {
        ++currentAxis;
        continue;
      }
      current =
          rewriter.create<triton::ExpandDimsOp>(loc, current, currentAxis);
      ++currentAxis;
    }
    return broadcastTo(loc, current, target);
  };

  auto promoteUniform = [&](Location loc, Value value, Type oldType,
                            RankedTensorType target,
                            int64_t targetLaneAxis) -> Value {
    if (targetLaneAxis < 0 || targetLaneAxis >= target.getRank())
      return Value();
    if (auto oldTensor = dyn_cast<RankedTensorType>(oldType)) {
      if (oldTensor.getRank() != target.getRank() - 1 ||
          oldTensor.getElementType() != target.getElementType())
        return Value();
      SmallVector<int64_t> shape(target.getShape().begin(),
                                 target.getShape().end());
      shape.erase(shape.begin() + targetLaneAxis);
      auto unlaned = RankedTensorType::get(shape, target.getElementType());
      Value base = broadcastTo(loc, value, unlaned);
      if (!base)
        return Value();
      Value expanded =
          rewriter.create<triton::ExpandDimsOp>(loc, base, targetLaneAxis);
      return broadcastTo(loc, expanded, target);
    }
    return rewriter.create<triton::SplatOp>(loc, target, value);
  };

  auto liftOperandTo = [&](Value value, RankedTensorType target,
                           int64_t targetLaneAxis) -> Value {
    MappedValue mapped = lookup(value);
    return mapped.tensorized
               ? alignTensorized(value.getLoc(), mapped, target, targetLaneAxis)
               : promoteUniform(value.getLoc(), mapped.value, value.getType(),
                                target, targetLaneAxis);
  };

  auto copyMissingAttrs = [](Operation *from, Operation *to) {
    for (NamedAttribute attribute : from->getAttrs())
      if (!to->hasAttr(attribute.getName()))
        to->setAttr(attribute.getName(), attribute.getValue());
  };

  Value laneMask;
  auto maskForPointerType = [&](Location loc, RankedTensorType pointerType,
                                int64_t laneAxis) -> Value {
    if (!laneMask)
      return Value();
    auto maskType =
        RankedTensorType::get(pointerType.getShape(), rewriter.getI1Type());
    return alignTensorized(loc, MappedValue{laneMask, true, 0}, maskType,
                           laneAxis);
  };

  auto createUnchanged = [&](Operation *operation) -> bool {
    IRMapping mapping;
    for (Value operand : operation->getOperands()) {
      Value mapped = lookup(operand).value;
      if (!mapped)
        return false;
      mapping.map(operand, mapped);
    }
    Operation *replacement = rewriter.clone(*operation, mapping);
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, false, 0};
    return true;
  };

  auto originalTypesHaveSameShape = [](Type lhs, Type rhs) {
    auto left = dyn_cast<RankedTensorType>(lhs);
    auto right = dyn_cast<RankedTensorType>(rhs);
    if (!left || !right)
      return !left && !right;
    return hasSameShape(left, right);
  };

  auto createElementwise = [&](Operation *operation) -> bool {
    if (operation->getNumResults() == 0)
      return false;
    const bool isReshape = isa<triton::ReshapeOp>(operation);
    if (!operation->hasTrait<OpTrait::Elementwise>() && !isReshape)
      return false;
    if (isReshape &&
        (!originalTypesHaveSameShape(operation->getOperand(0).getType(),
                                     operation->getResult(0).getType())))
      return false;

    int64_t laneAxis =
        getPreferredLaneAxis(operation->getResult(0).getType(), candidate);
    for (Value operand : operation->getOperands()) {
      MappedValue mapped = lookup(operand);
      if (!mapped.tensorized ||
          !originalTypesHaveSameShape(operand.getType(),
                                      operation->getResult(0).getType()))
        continue;
      laneAxis = mapped.laneAxis;
      break;
    }
    std::optional<RankedTensorType> firstResultType = getLiftedTensorType(
        operation->getResult(0).getType(), candidate, laneAxis);
    if (!firstResultType)
      return false;

    SmallVector<Type> resultTypes;
    resultTypes.reserve(operation->getNumResults());
    for (Type resultType : operation->getResultTypes()) {
      std::optional<RankedTensorType> lifted =
          getLiftedTensorType(resultType, candidate, laneAxis);
      if (!lifted || !hasSameShape(*lifted, *firstResultType))
        return false;
      resultTypes.push_back(*lifted);
    }

    SmallVector<Value> operands;
    operands.reserve(operation->getNumOperands());
    for (Value operand : operation->getOperands()) {
      std::optional<RankedTensorType> operandType = getSameShapeTensorType(
          operand.getType(), firstResultType->getShape());
      if (!operandType)
        return false;
      Value lifted = liftOperandTo(operand, *operandType, laneAxis);
      if (!lifted)
        return false;
      operands.push_back(lifted);
    }

    Operation *replacement = rewriter.create(
        operation->getLoc(), operation->getName().getIdentifier(), operands,
        resultTypes, operation->getAttrs());
    if (replacement->getNumResults() != operation->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(operation->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, true, laneAxis};
    return true;
  };

  for (Operation *operation : originals) {
    rewriter.setInsertionPoint(operation);

    if (operation == targetPid->getOperation()) {
      Operation *scalarPid = rewriter.clone(*operation);
      if (scalarPid->getNumResults() != 1)
        return false;
      Value scalar = scalarPid->getResult(0);
      auto integer = dyn_cast<IntegerType>(scalar.getType());
      if (!integer || integer.getWidth() != 32)
        return false;
      Value factor = rewriter.create<arith::ConstantIntOp>(
          operation->getLoc(), candidate.factor, integer.getWidth());
      Value base =
          rewriter.create<arith::MulIOp>(operation->getLoc(), scalar, factor);
      auto laneType =
          RankedTensorType::get({candidate.factor}, scalar.getType());
      Value lane = rewriter.create<triton::MakeRangeOp>(
          operation->getLoc(), laneType, 0, candidate.factor);
      Value baseSplat =
          rewriter.create<triton::SplatOp>(operation->getLoc(), laneType, base);
      Value logicalIds =
          rewriter.create<arith::AddIOp>(operation->getLoc(), baseSplat, lane);
      Type i64 = rewriter.getI64Type();
      auto laneI64Type = RankedTensorType::get({candidate.factor}, i64);
      Value logicalIdsI64 = rewriter.create<arith::ExtUIOp>(
          operation->getLoc(), laneI64Type, logicalIds);
      Value extentI64 = rewriter.create<arith::ExtUIOp>(operation->getLoc(),
                                                        i64, originalExtent);
      Value extentSplat = rewriter.create<triton::SplatOp>(
          operation->getLoc(), laneI64Type, extentI64);
      auto runtimeExtentMask = rewriter.create<arith::CmpIOp>(
          operation->getLoc(), arith::CmpIPredicate::ult, logicalIdsI64,
          extentSplat);
      runtimeExtentMask->setAttr(
          mlir::triton::memory_access::IATRuntimeExtentUnsignedMaskTAG,
          rewriter.getUnitAttr());
      laneMask = runtimeExtentMask;
      values[operation->getResult(0)] = {logicalIds, true, 0};
      continue;
    }

    bool tensorized = closure.contains(operation);
    for (Value operand : operation->getOperands())
      tensorized |= lookup(operand).tensorized;

    if (auto splat = dyn_cast<triton::SplatOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      const int64_t laneAxis = getPreferredLaneAxis(splat.getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(splat.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(splat.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[splat.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(expand.getSrc());
      const int64_t oldLaneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(expand.getSrc().getType(), candidate);
      std::optional<RankedTensorType> sourceType = getLiftedTensorType(
          expand.getSrc().getType(), candidate, oldLaneAxis);
      if (!sourceType)
        return false;
      Value source = liftOperandTo(expand.getSrc(), *sourceType, oldLaneAxis);
      if (!source)
        return false;
      const int64_t newAxis =
          expand.getAxis() + (expand.getAxis() >= oldLaneAxis ? 1 : 0);
      const int64_t resultLaneAxis =
          oldLaneAxis + (expand.getAxis() < oldLaneAxis ? 1 : 0);
      Value result = rewriter.create<triton::ExpandDimsOp>(operation->getLoc(),
                                                           source, newAxis);
      std::optional<RankedTensorType> expected =
          getLiftedTensorType(expand.getType(), candidate, resultLaneAxis);
      if (!expected || result.getType() != *expected)
        return false;
      values[expand.getResult()] = {result, true, resultLaneAxis};
      continue;
    }

    if (auto broadcast = dyn_cast<triton::BroadcastOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue sourceMapped = lookup(broadcast.getSrc());
      const int64_t laneAxis =
          sourceMapped.tensorized
              ? sourceMapped.laneAxis
              : getPreferredLaneAxis(broadcast.getSrc().getType(), candidate);
      std::optional<RankedTensorType> resultType =
          getLiftedTensorType(broadcast.getType(), candidate, laneAxis);
      if (!resultType)
        return false;
      Value result = liftOperandTo(broadcast.getSrc(), *resultType, laneAxis);
      if (!result)
        return false;
      values[broadcast.getResult()] = {result, true, laneAxis};
      continue;
    }

    if (auto reduce = dyn_cast<triton::ReduceOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (reduce.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(reduce.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(reduce.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : reduce.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          reduce.getAxis() + (reduce.getAxis() >= laneAxis ? 1 : 0);
      const int64_t resultLaneAxis = laneAxis - (axis < laneAxis ? 1 : 0);
      auto replacement =
          rewriter.create<triton::ReduceOp>(operation->getLoc(), sources, axis);
      rewriter.cloneRegionBefore(reduce.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(reduce.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, resultLaneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, resultLaneAxis};
      }
      continue;
    }

    if (auto scan = dyn_cast<triton::ScanOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      if (scan.getSrcs().empty())
        return false;
      auto oldInputType =
          dyn_cast<RankedTensorType>(scan.getSrcs().front().getType());
      if (!oldInputType)
        return false;
      MappedValue firstSource = lookup(scan.getSrcs().front());
      const int64_t laneAxis =
          firstSource.tensorized
              ? firstSource.laneAxis
              : getPreferredLaneAxis(oldInputType, candidate);
      std::optional<RankedTensorType> firstSourceType =
          getLiftedTensorType(oldInputType, candidate, laneAxis);
      if (!firstSourceType)
        return false;
      SmallVector<Value> sources;
      for (Value source : scan.getSrcs()) {
        std::optional<RankedTensorType> sourceType =
            getLiftedTensorType(source.getType(), candidate, laneAxis);
        if (!sourceType || !hasSameShape(*sourceType, *firstSourceType))
          return false;
        Value lifted = liftOperandTo(source, *sourceType, laneAxis);
        if (!lifted)
          return false;
        sources.push_back(lifted);
      }
      const int64_t axis =
          scan.getAxis() + (scan.getAxis() >= laneAxis ? 1 : 0);
      auto replacement = rewriter.create<triton::ScanOp>(
          operation->getLoc(), sources, static_cast<uint32_t>(axis),
          scan.getReverse());
      rewriter.cloneRegionBefore(scan.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      for (auto [oldResult, newResult] :
           llvm::zip(scan.getResults(), replacement.getResults())) {
        std::optional<RankedTensorType> expected =
            getLiftedTensorType(oldResult.getType(), candidate, laneAxis);
        if (!expected || newResult.getType() != *expected)
          return false;
        values[oldResult] = {newResult, true, laneAxis};
      }
      continue;
    }

    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(load.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(load.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(load.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(load.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> maskType =
          getSameShapeTensorType(rewriter.getI1Type(), pointerType->getShape());
      Value mask = load.getMask()
                       ? liftOperandTo(load.getMask(), *maskType, laneAxis)
                       : Value();
      Value laneMaskForLoad =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForLoad)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForLoad)
                    : laneMaskForLoad;
      std::optional<RankedTensorType> otherType = getSameShapeTensorType(
          load.getResult().getType(), pointerType->getShape());
      Value other = load.getOther()
                        ? liftOperandTo(load.getOther(), *otherType, laneAxis)
                        : makeZero(rewriter, operation->getLoc(), *otherType);
      if (!pointer || !maskType || !otherType || !other)
        return false;
      auto replacement = rewriter.create<triton::LoadOp>(
          operation->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      copyMissingAttrs(operation, replacement.getOperation());
      if (replacement.getResult().getType() != *otherType)
        return false;
      values[load.getResult()] = {replacement.getResult(), true, laneAxis};
      continue;
    }

    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      if (!tensorized) {
        if (!createUnchanged(operation))
          return false;
        continue;
      }
      MappedValue pointerMapped = lookup(store.getPtr());
      const int64_t laneAxis =
          pointerMapped.tensorized
              ? pointerMapped.laneAxis
              : getPreferredLaneAxis(store.getPtr().getType(), candidate);
      std::optional<RankedTensorType> pointerType =
          getLiftedTensorType(store.getPtr().getType(), candidate, laneAxis);
      if (!pointerType)
        return false;
      Value pointer = liftOperandTo(store.getPtr(), *pointerType, laneAxis);
      std::optional<RankedTensorType> valueType = getSameShapeTensorType(
          store.getValue().getType(), pointerType->getShape());
      if (!valueType)
        return false;
      Value value = liftOperandTo(store.getValue(), *valueType, laneAxis);
      auto maskType =
          RankedTensorType::get(pointerType->getShape(), rewriter.getI1Type());
      Value mask = store.getMask()
                       ? liftOperandTo(store.getMask(), maskType, laneAxis)
                       : Value();
      Value laneMaskForStore =
          maskForPointerType(operation->getLoc(), *pointerType, laneAxis);
      if (laneMaskForStore)
        mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                     laneMaskForStore)
                    : laneMaskForStore;
      if (!pointer || !value || !laneMaskForStore)
        return false;
      auto replacement = rewriter.create<triton::StoreOp>(
          operation->getLoc(), pointer, value, mask, store.getBoundaryCheck(),
          store.getCache(), store.getEvict());
      copyMissingAttrs(operation, replacement.getOperation());
      continue;
    }

    if (isa<triton::ReturnOp>(operation)) {
      for (Value operand : operation->getOperands())
        if (lookup(operand).tensorized)
          return false;
      if (!createUnchanged(operation))
        return false;
      continue;
    }

    if (!tensorized) {
      if (!createUnchanged(operation))
        return false;
      continue;
    }
    if (!createElementwise(operation))
      return false;
  }

  if (!laneMask)
    return false;
  for (Operation *operation : llvm::reverse(originals)) {
    operation->dropAllUses();
    rewriter.eraseOp(operation);
  }
  return true;
}

LogicalResult runProgramMappingStructuralCleanup(ModuleOp module) {
  PassManager cleanup(module.getContext(), module.getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(module);
}

LogicalResult materializeIATCandidateToSandbox(ModuleOp module,
                                               triton::FuncOp function,
                                               const IATCandidate &candidate) {
  if (!module || module->hasAttr(kIndependentAxisTensorizeMarkerAttr) ||
      module->hasAttr(kProgramGridTransformsAttr))
    return failure();
  IRRewriter rewriter(module.getContext());
  if (!rebuildTensorizedFunction(function, candidate, rewriter))
    return failure();
  ProgramGridTransformContract contract;
  contract.transforms.push_back(ProgramGridTransform{
      0, candidate.axis, static_cast<int64_t>(candidate.factor), false, false});
  if (failed(setProgramGridTransformContract(module, contract)))
    return failure();
  module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                  UnitAttr::get(module.getContext()));
  return mlir::verify(module.getOperation());
}

ModuleOp createProgramMappingSandbox(ModuleOp module, triton::FuncOp function) {
  (void)module;
  ModuleOp sandbox = ModuleOp::create(function.getLoc());
  sandbox.getBody()->push_back(function->clone());
  return sandbox;
}

std::optional<IATCandidate>
analyzeDynamicSingleMomentCandidate(GraphOptimizationContext &context) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module) ||
      !hasSupportedControlFlow(function) ||
      classifyIndependentRowReduction(function) !=
          IndependentRowReductionKind::SingleMoment ||
      !findOnlyProgramId(function, 0))
    return std::nullopt;

  std::optional<TensorizeFormMatch> formMatch = classifyTensorizeForm(function);
  if (!formMatch || formMatch->form != TensorizeForm::NormRope)
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> headPid =
      findOnlyProgramId(function, 1);
  if (!headPid)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(1);
  if (!dependence.isProgramMappingTransformCandidate() ||
      hasTensorPointerInProgramAxisClosure(dependence) ||
      !hasDisjointWriteReadRoots(dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;
  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  if (!liveBytes.known)
    return std::nullopt;
  uint64_t estimatedPeak = 0;
  if (__builtin_mul_overflow(liveBytes.peakLiveBytes, uint64_t{16},
                             &estimatedPeak))
    return std::nullopt;

  IATCandidate candidate;
  candidate.function = function;
  candidate.anchor = headPid->getOperation();
  candidate.form = TensorizeForm::NormRope;
  candidate.axis = 1;
  candidate.dimExtent = formMatch->reduction.dimExtent;
  candidate.factor = 16;
  candidate.requiresPersistentChaining = true;
  candidate.resources = context.getResourceCostAnalysis().getResourceSnapshot();

  CandidateCost cost;
  cost.plan.tensorizeFactor = candidate.factor;
  cost.plan.blockT = kDynamicSingleMomentBlockT;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = "iat.ptsm.single-moment.dynamic.f16.b4";
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = estimatedPeak;
  candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
  return candidate.evaluation.accepted ? std::optional<IATCandidate>(candidate)
                                       : std::nullopt;
}

bool sameDynamicSingleMomentCandidate(const IATCandidate &lhs,
                                      const IATCandidate &rhs);

class DynamicSingleMomentProgramMappingPlan final : public RewritePlan {
public:
  DynamicSingleMomentProgramMappingPlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<IATCandidate> current =
        analyzeDynamicSingleMomentCandidate(context);
    return current && sameDynamicSingleMomentCandidate(candidate, *current)
               ? success()
               : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || hasConflictingLaunchContract(module))
      return failure();

    ModuleOp sandbox = createProgramMappingSandbox(module, candidate.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    CandidateEvaluation persistentEvaluation;
    if (!clonedFunction ||
        failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate)) ||
        failed(materializePersistentTaskStripMiningCandidate(
            sandbox, clonedFunction, candidate.resources,
            kDynamicSingleMomentBlockT, &persistentEvaluation, true)) ||
        failed(runProgramMappingStructuralCleanup(sandbox)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();

    const LiveByteEstimate baseline =
        estimatePeakLiveBytes(candidate.function.getOperation());
    const LiveByteEstimate final =
        estimatePeakLiveBytes(clonedFunction.getOperation());
    if (!baseline.known || !final.known)
      return failure();
    CandidateCost finalCost;
    finalCost.plan.tensorizeFactor = candidate.factor;
    finalCost.plan.blockT = kDynamicSingleMomentBlockT;
    finalCost.plan.staticAxisFusionFactor = 1;
    finalCost.plan.stableId = "iat.ptsm.single-moment.dynamic.f16.b4";
    finalCost.hasPeakLiveBytes = true;
    finalCost.baselinePeakLiveBytes = baseline.peakLiveBytes;
    finalCost.estimatedPeakLiveBytes = final.peakLiveBytes;
    if (!evaluateCandidateCost(candidate.resources, finalCost).accepted)
      return failure();

    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    if (!transforms)
      return failure();
    FailureOr<ProgramGridTransformContract> parsed =
        parseProgramGridTransformContract(transforms);
    if (failed(parsed) || parsed->transforms.size() != 2 ||
        !hasProgramGridHiddenExtentArguments(clonedFunction))
      return failure();

    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    module->setAttr(kPersistentTaskStripMiningMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

std::optional<IATCandidate>
analyzeDynamicMergeSplitCandidate(GraphOptimizationContext &context) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  if (!module || hasConflictingLaunchContract(module) ||
      !hasSupportedControlFlow(function))
    return std::nullopt;

  std::optional<TensorizeFormMatch> formMatch = classifyTensorizeForm(function);
  if (!formMatch || formMatch->form != TensorizeForm::MergeSplit ||
      !findOnlyProgramId(function, 0))
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> headPid =
      findOnlyProgramId(function, 1);
  if (!headPid)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(1);
  if (!dependence.isProgramMappingTransformCandidate() ||
      hasTensorPointerInProgramAxisClosure(dependence) ||
      !hasDisjointWriteReadRoots(dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;
  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  if (!liveBytes.known)
    return std::nullopt;
  uint64_t estimatedPeak = 0;
  if (__builtin_mul_overflow(liveBytes.peakLiveBytes, uint64_t{16},
                             &estimatedPeak))
    return std::nullopt;

  IATCandidate candidate;
  candidate.function = function;
  candidate.anchor = headPid->getOperation();
  candidate.form = TensorizeForm::MergeSplit;
  candidate.axis = 1;
  candidate.splitExtent = formMatch->reduction.splitExtent;
  candidate.dimExtent = formMatch->reduction.dimExtent;
  candidate.factor = 16;
  candidate.resources = context.getResourceCostAnalysis().getResourceSnapshot();

  CandidateCost cost;
  cost.plan.tensorizeFactor = candidate.factor;
  cost.plan.blockT = 1;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = "iat.merge.dynamic.f16";
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = liveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = estimatedPeak;
  candidate.evaluation = context.getResourceCostAnalysis().evaluate(cost);
  return candidate.evaluation.accepted ? std::optional<IATCandidate>(candidate)
                                       : std::nullopt;
}

bool sameDynamicMergeSplitCandidate(const IATCandidate &lhs,
                                    const IATCandidate &rhs) {
  return lhs.function == rhs.function && lhs.anchor == rhs.anchor &&
         lhs.form == rhs.form && lhs.axis == rhs.axis &&
         lhs.splitExtent == rhs.splitExtent && lhs.dimExtent == rhs.dimExtent &&
         lhs.factor == rhs.factor;
}

bool sameDynamicSingleMomentCandidate(const IATCandidate &lhs,
                                      const IATCandidate &rhs) {
  return lhs.function == rhs.function && lhs.anchor == rhs.anchor &&
         lhs.form == rhs.form && lhs.axis == rhs.axis &&
         lhs.dimExtent == rhs.dimExtent && lhs.factor == rhs.factor &&
         lhs.requiresPersistentChaining == rhs.requiresPersistentChaining;
}

class DynamicMergeSplitIATPlan final : public RewritePlan {
public:
  DynamicMergeSplitIATPlan(IATCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  unsigned getBenefit() const override { return 1; }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<IATCandidate> current =
        analyzeDynamicMergeSplitCandidate(context);
    return current && sameDynamicMergeSplitCandidate(candidate, *current)
               ? success()
               : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || hasConflictingLaunchContract(module))
      return failure();
    ModuleOp sandbox = createProgramMappingSandbox(module, candidate.function);
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction ||
        failed(materializeIATCandidateToSandbox(sandbox, clonedFunction,
                                                candidate)) ||
        failed(runProgramMappingStructuralCleanup(sandbox)) ||
        failed(mlir::verify(sandbox.getOperation())))
      return failure();
    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    if (!transforms || failed(parseProgramGridTransformContract(transforms)) ||
        !hasProgramGridHiddenExtentArguments(clonedFunction))
      return failure();

    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kIndependentAxisTensorizeMarkerAttr,
                    UnitAttr::get(module.getContext()));
    return success();
  }

private:
  IATCandidate candidate;
  unsigned epoch;
};

class DynamicIndependentAxisTensorizeRule final : public GraphOptimizationRule {
public:
  DynamicIndependentAxisTensorizeRule(bool enabledForCompileMode,
                                      bool iatAndPtsmEnabled)
      : enabledForCompileMode(enabledForCompileMode),
        iatAndPtsmEnabled(iatAndPtsmEnabled) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::IndependentAxisTensorize;
  }
  AnalysisRequirement getAnalysisRequirements() const override {
    return AnalysisRequirement::EntryArgPointerAlias |
           AnalysisRequirement::ProgramAxisDependence |
           AnalysisRequirement::ResourceCost;
  }
  LogicalResult findCandidates(
      GraphOptimizationContext &context,
      SmallVectorImpl<std::unique_ptr<RewritePlan>> &plans) override {
    if (!enabledForCompileMode)
      return success();
    std::optional<IATCandidate> candidate =
        analyzeDynamicMergeSplitCandidate(context);
    if (candidate) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<DynamicMergeSplitIATPlan>(
          std::move(*candidate), context.getEpoch()));
      return success();
    }
    if (!iatAndPtsmEnabled)
      return success();
    candidate = analyzeDynamicSingleMomentCandidate(context);
    if (candidate) {
      emitCandidateRemark(candidate->anchor, candidate->evaluation);
      plans.push_back(std::make_unique<DynamicSingleMomentProgramMappingPlan>(
          std::move(*candidate), context.getEpoch()));
    }
    return success();
  }

private:
  bool enabledForCompileMode;
  bool iatAndPtsmEnabled;
};

} // namespace

std::unique_ptr<GraphOptimizationRule> cfg::createIndependentAxisTensorizeRule(
    const IndependentAxisTensorizeRuleOptions &options) {
  return std::make_unique<DynamicIndependentAxisTensorizeRule>(
      options.enabledForCompileMode, options.iatAndPtsmEnabled);
}
