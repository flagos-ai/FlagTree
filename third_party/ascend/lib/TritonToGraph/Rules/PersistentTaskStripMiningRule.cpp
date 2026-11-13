/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToGraph/EntryArgPointerAliasAnalysis.h"
#include "TritonToGraph/GraphOptimizationRule.h"
#include "TritonToGraph/IndependentRowReductionAnalysis.h"
#include "TritonToGraph/ProgramAxisDependenceAnalysis.h"
#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
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
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

using namespace mlir;
using namespace triton;
using namespace cfg;

namespace {

constexpr int32_t kTokenAxis = 0;
constexpr llvm::StringLiteral kPersistentTaskStripMiningMarkerAttr =
    "hacc.persistent_task_strip_mining";
constexpr unsigned kSingleMomentBlockT = 4;
constexpr unsigned kTwoMomentBlockT = 64;

struct PTSMCandidate {
  triton::FuncOp function;
  Operation *anchor = nullptr;
  unsigned blockT = 0;
  unsigned existingTransformCount = 0;
  IndependentRowReductionKind reductionKind =
      IndependentRowReductionKind::Other;
  CandidateEvaluation evaluation;
};

LogicalResult
materializePersistentTaskStripMining(triton::FuncOp function,
                                     const PTSMCandidate &candidate);
LogicalResult runProgramMappingCandidateCleanup(ModuleOp module);
std::optional<LiveByteEstimate>
estimateFinalPersistentPeak(triton::FuncOp function,
                            const PTSMCandidate &candidate);

struct MappedValue {
  Value value;
  bool tensorized = false;
};

bool isPublicEntry(triton::FuncOp function) {
  auto visibility = function->getAttrOfType<StringAttr>("sym_visibility");
  return !visibility || visibility.getValue() == "public";
}

bool isOnlyPublicEntry(ModuleOp module, triton::FuncOp function) {
  unsigned entries = 0;
  for (triton::FuncOp candidate : module.getOps<triton::FuncOp>()) {
    if (!isPublicEntry(candidate))
      continue;
    ++entries;
    if (candidate != function)
      return false;
  }
  return entries == 1;
}

bool hasDirectCall(triton::FuncOp function) {
  bool hasCall = false;
  function.walk([&](Operation *operation) {
    if (isa<CallOpInterface>(operation))
      hasCall = true;
  });
  return hasCall;
}

bool hasDisallowedEffectOrControlFlow(triton::FuncOp function) {
  bool disallowed = false;
  function.walk([&](Operation *operation) {
    if (operation == function.getOperation())
      return;
    const StringRef name = operation->getName().getStringRef();
    if (isa<CallOpInterface>(operation) || name.starts_with("scf.") ||
        name.contains_insensitive("atomic") ||
        name.contains_insensitive("barrier") ||
        name.contains_insensitive("random") ||
        name.contains_insensitive("rng") ||
        name.contains_insensitive("print")) {
      disallowed = true;
      return;
    }
    if (operation->getNumRegions() != 0 &&
        !isa<triton::ReduceOp, triton::ScanOp>(operation))
      disallowed = true;
  });
  return disallowed;
}

std::optional<triton::GetProgramIdOp> findOnlyProgramId(triton::FuncOp function,
                                                        int32_t axis) {
  SmallVector<triton::GetProgramIdOp> ids;
  function.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() == axis)
      ids.push_back(pid);
  });
  if (ids.size() != 1)
    return std::nullopt;
  return ids.front();
}

bool hasVoidReturn(triton::FuncOp function) {
  if (function->getNumRegions() != 1 || function->getRegion(0).empty())
    return false;
  Block &block = function->getRegion(0).front();
  auto result = dyn_cast<triton::ReturnOp>(block.getTerminator());
  return result && result.getNumOperands() == 0;
}

bool hasPersistentRowReductionForm(triton::FuncOp function,
                                   IndependentRowReductionKind expected) {
  return classifyIndependentRowReduction(function) == expected;
}

bool hasDisjointWriteReadRoots(
    triton::FuncOp function, const ProgramAxisDependence &dependence,
    const EntryArgPointerAliasAnalysis &entryPointerAliases) {
  SmallVector<triton::LoadOp> loads;
  function.walk([&](triton::LoadOp load) { loads.push_back(load); });
  for (const StoreAddressDependence &store : dependence.stores) {
    auto storeOp = dyn_cast_or_null<triton::StoreOp>(store.store);
    if (!storeOp)
      return false;
    for (triton::LoadOp load : loads) {
      if (entryPointerAliases.classify(storeOp.getPtr(), load.getPtr()) !=
          EntryArgPointerRelation::DistinctEntryRoots)
        return false;
    }
  }
  return true;
}

std::optional<ProgramGridTransformContract>
getComposableLaunchContract(ModuleOp module) {
  Attribute attribute = module->getAttr(kProgramGridTransformsAttr);
  if (!attribute)
    return ProgramGridTransformContract{};
  FailureOr<ProgramGridTransformContract> parsed =
      parseProgramGridTransformContract(attribute);
  if (failed(parsed))
    return std::nullopt;
  for (const ProgramGridTransform &transform : parsed->transforms) {
    if (transform.axis == kTokenAxis || transform.persistentCoverage ||
        transform.gridStrideAbiVerified)
      return std::nullopt;
  }
  return *parsed;
}

CandidateCost buildResourceCandidate(const PTSMCandidate &candidate,
                                     const ProgramAxisDependence &dependence,
                                     const LiveByteEstimate &baselineLiveBytes,
                                     const LiveByteEstimate *finalLiveBytes) {
  CandidateCost cost;
  cost.plan.tensorizeFactor = 1;
  cost.plan.blockT = candidate.blockT;
  cost.plan.staticAxisFusionFactor = 1;
  cost.plan.stableId = (llvm::Twine("persistent-task-strip-mining.b") +
                        llvm::Twine(candidate.blockT))
                           .str();
  cost.launchesBefore = 1;
  cost.launchesAfter = 1;
  cost.storeCountBefore = dependence.stores.size();
  cost.storeCountAfter = dependence.stores.size();
  cost.addressCalculationsBefore = dependence.dependenceClosure.size();
  cost.addressCalculationsAfter = dependence.dependenceClosure.size();
  cost.persistent = true;

  if (!baselineLiveBytes.known || !finalLiveBytes || !finalLiveBytes->known) {
    const ResourceCostRejectReason reason =
        !baselineLiveBytes.known
            ? baselineLiveBytes.reason
            : (finalLiveBytes ? finalLiveBytes->reason
                              : ResourceCostRejectReason::UnknownResource);
    cost.hasDynamicShape = reason == ResourceCostRejectReason::DynamicShape;
    cost.hasUnknownResource = !cost.hasDynamicShape;
    return cost;
  }
  cost.hasPeakLiveBytes = true;
  cost.baselinePeakLiveBytes = baselineLiveBytes.peakLiveBytes;
  cost.estimatedPeakLiveBytes = finalLiveBytes->peakLiveBytes;
  return cost;
}

std::optional<PTSMCandidate>
analyzeCandidate(GraphOptimizationContext &context, bool emitRejectRemark,
                 std::optional<unsigned> requestedBlockT = std::nullopt,
                 bool deferIntermediateResourceRejection = false) {
  triton::FuncOp function = context.getFunction();
  ModuleOp module = function->getParentOfType<ModuleOp>();
  const IndependentRowReductionKind reductionKind =
      classifyIndependentRowReduction(function);
  if (!module || module->hasAttr(kPersistentTaskStripMiningMarkerAttr) ||
      !isPublicEntry(function) || !isOnlyPublicEntry(module, function) ||
      reductionKind == IndependentRowReductionKind::Other ||
      !hasVoidReturn(function) || hasDirectCall(function) ||
      hasDisallowedEffectOrControlFlow(function))
    return std::nullopt;

  std::optional<ProgramGridTransformContract> launchContract =
      getComposableLaunchContract(module);
  if (!launchContract)
    return std::nullopt;
  const bool hasIAT = launchContract->transforms.size() == 1;
  if (launchContract->transforms.size() > 1 ||
      (hasIAT && reductionKind != IndependentRowReductionKind::SingleMoment) ||
      (!hasIAT && reductionKind != IndependentRowReductionKind::TwoMoment))
    return std::nullopt;
  const unsigned expectedBlockT =
      hasIAT ? kSingleMomentBlockT : kTwoMomentBlockT;
  if (requestedBlockT && *requestedBlockT != expectedBlockT)
    return std::nullopt;
  std::optional<triton::GetProgramIdOp> tokenPid =
      findOnlyProgramId(function, kTokenAxis);
  if (!tokenPid)
    return std::nullopt;

  const ProgramAxisDependence &dependence =
      context.getProgramAxisDependenceAnalysis().get(kTokenAxis);
  if (!dependence.isProgramMappingTransformCandidate() ||
      !hasDisjointWriteReadRoots(function, dependence,
                                 context.getEntryArgPointerAliasAnalysis()))
    return std::nullopt;

  const LiveByteEstimate &liveBytes =
      context.getResourceCostAnalysis().getLiveByteEstimate();
  PTSMCandidate candidate;
  candidate.function = function;
  candidate.anchor = tokenPid->getOperation();
  candidate.blockT = expectedBlockT;
  candidate.existingTransformCount = launchContract->transforms.size();
  candidate.reductionKind = reductionKind;
  std::optional<LiveByteEstimate> finalLiveBytes =
      estimateFinalPersistentPeak(function, candidate);
  candidate.evaluation = context.getResourceCostAnalysis().evaluate(
      buildResourceCandidate(candidate, dependence, liveBytes,
                             finalLiveBytes ? &*finalLiveBytes : nullptr));
  const bool mayDeferResourceRejection =
      requestedBlockT && deferIntermediateResourceRejection;
  if (!candidate.evaluation.accepted && !mayDeferResourceRejection)
    return std::nullopt;
  if (emitRejectRemark)
    emitCandidateRemark(tokenPid->getOperation(), candidate.evaluation);
  return candidate;
}

LogicalResult
applyPersistentCandidateToSandbox(ModuleOp module, triton::FuncOp function,
                                  const PTSMCandidate &candidate) {
  if (!module || module->hasAttr(kPersistentTaskStripMiningMarkerAttr))
    return failure();
  std::optional<ProgramGridTransformContract> existing =
      getComposableLaunchContract(module);
  if (!existing ||
      existing->transforms.size() != candidate.existingTransformCount)
    return failure();
  if (failed(materializePersistentTaskStripMining(function, candidate)))
    return failure();

  ProgramGridTransformContract contract = *existing;
  contract.transforms.push_back(ProgramGridTransform{
      static_cast<int32_t>(contract.transforms.size()), kTokenAxis,
      static_cast<int64_t>(candidate.blockT), true, true});
  if (failed(setProgramGridTransformContract(module, contract)))
    return failure();
  module->setAttr(kPersistentTaskStripMiningMarkerAttr,
                  UnitAttr::get(module.getContext()));
  return mlir::verify(module.getOperation());
}

bool sameCandidate(const PTSMCandidate &lhs, const PTSMCandidate &rhs) {
  return lhs.function == rhs.function && lhs.blockT == rhs.blockT &&
         lhs.existingTransformCount == rhs.existingTransformCount &&
         lhs.reductionKind == rhs.reductionKind;
}

std::optional<RankedTensorType> getBatchedTensorType(Type originalType,
                                                     unsigned blockT) {
  Type elementType = originalType;
  SmallVector<int64_t> shape;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType)) {
    if (!tensor.hasStaticShape())
      return std::nullopt;
    elementType = tensor.getElementType();
    shape.assign(tensor.getShape().begin(), tensor.getShape().end());
  }
  shape.insert(shape.begin(), blockT);
  return RankedTensorType::get(shape, elementType);
}

std::optional<RankedTensorType>
getSameShapeTensorType(Type originalType, ArrayRef<int64_t> shape) {
  Type elementType = originalType;
  if (auto tensor = dyn_cast<RankedTensorType>(originalType)) {
    if (!tensor.hasStaticShape())
      return std::nullopt;
    elementType = tensor.getElementType();
  }
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

Value alignTensor(IRRewriter &rewriter, Location loc, Value source,
                  RankedTensorType target, bool sourceIsBatched,
                  unsigned blockT) {
  auto sourceType = dyn_cast<RankedTensorType>(source.getType());
  if (!sourceType || !sourceType.hasStaticShape() ||
      sourceType.getElementType() != target.getElementType() ||
      target.getRank() == 0 || target.getShape().front() != blockT)
    return Value();
  if (sourceIsBatched &&
      (sourceType.getRank() == 0 || sourceType.getShape().front() != blockT))
    return Value();

  SmallVector<int64_t> sourceToTarget(sourceType.getRank(), -1);
  unsigned sourceStart = 0;
  unsigned targetStart = 1;
  if (sourceIsBatched) {
    sourceToTarget[0] = 0;
    sourceStart = 1;
  }
  std::function<bool(unsigned, unsigned)> mapTail =
      [&](unsigned sourceIndex, unsigned targetIndex) -> bool {
    if (sourceIndex == sourceType.getRank())
      return true;
    for (unsigned candidate = targetIndex; candidate < target.getRank();
         ++candidate) {
      const int64_t sourceDim = sourceType.getShape()[sourceIndex];
      const int64_t targetDim = target.getShape()[candidate];
      if (sourceDim != targetDim && sourceDim != 1)
        continue;
      sourceToTarget[sourceIndex] = candidate;
      if (mapTail(sourceIndex + 1, candidate + 1))
        return true;
      sourceToTarget[sourceIndex] = -1;
    }
    return false;
  };
  if (!mapTail(sourceStart, targetStart))
    return Value();

  SmallVector<bool> targetMapped(target.getRank(), false);
  for (unsigned sourceIndex = 0; sourceIndex < sourceType.getRank();
       ++sourceIndex) {
    const int64_t targetIndex = sourceToTarget[sourceIndex];
    if (targetIndex < 0 || targetMapped[targetIndex])
      return Value();
    targetMapped[targetIndex] = true;
  }
  Value current = source;
  int64_t currentAxis = 0;
  for (int64_t targetIndex = 0; targetIndex < target.getRank(); ++targetIndex) {
    if (!targetMapped[targetIndex])
      current =
          rewriter.create<triton::ExpandDimsOp>(loc, current, currentAxis);
    ++currentAxis;
  }
  auto currentType = dyn_cast<RankedTensorType>(current.getType());
  if (!currentType || currentType.getElementType() != target.getElementType() ||
      currentType.getRank() != target.getRank())
    return Value();
  for (auto [currentDim, targetDim] :
       llvm::zip(currentType.getShape(), target.getShape()))
    if (currentDim != targetDim && currentDim != 1)
      return Value();
  return hasSameShape(currentType, target)
             ? current
             : rewriter.create<triton::BroadcastOp>(loc, target, current);
}

LogicalResult
materializePersistentTaskStripMining(triton::FuncOp function,
                                     const PTSMCandidate &candidate) {
  if (!hasVoidReturn(function) ||
      !hasPersistentRowReductionForm(function, candidate.reductionKind) ||
      hasDisallowedEffectOrControlFlow(function) ||
      failed(addProgramGridHiddenExtentArguments(function)))
    return failure();
  Value originalTokenExtent =
      function.getArgument(function.getNumArguments() - 2);
  auto originalTokenExtentType =
      dyn_cast<IntegerType>(originalTokenExtent.getType());
  if (!originalTokenExtentType || originalTokenExtentType.getWidth() != 32)
    return failure();
  std::optional<triton::GetProgramIdOp> tokenPid =
      findOnlyProgramId(function, kTokenAxis);
  if (!tokenPid)
    return failure();

  ProgramAxisDependenceAnalysis axisAnalysis(function);
  const ProgramAxisDependence &dependence = axisAnalysis.get(kTokenAxis);
  if (!dependence.isProgramMappingTransformCandidate())
    return failure();
  DenseSet<Operation *> tokenClosure;
  for (Operation *operation : dependence.dependenceClosure)
    tokenClosure.insert(operation);

  Block &block = function.getBody().front();
  SmallVector<Operation *> originals;
  originals.reserve(block.getOperations().size());
  for (Operation &operation : block.without_terminator())
    originals.push_back(&operation);

  IRRewriter rewriter(function.getContext());
  rewriter.setInsertionPoint(originals.empty() ? block.getTerminator()
                                               : originals.front());
  DenseMap<Value, MappedValue> values;
  auto lookup = [&](Value value) -> MappedValue {
    auto it = values.find(value);
    return it == values.end() ? MappedValue{value, false} : it->second;
  };
  auto mapResults = [&](Operation *original, Operation *replacement,
                        bool tensorized) -> bool {
    if (!original || !replacement ||
        original->getNumResults() != replacement->getNumResults())
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip(original->getResults(), replacement->getResults()))
      values[oldResult] = {newResult, tensorized};
    return true;
  };

  Value physicalPid;
  for (Operation *operation : originals) {
    if (operation == tokenPid->getOperation()) {
      Operation *replacement = rewriter.clone(*operation);
      if (!replacement || replacement->getNumResults() != 1)
        return failure();
      physicalPid = replacement->getResult(0);
      continue;
    }
    if (tokenClosure.contains(operation))
      continue;
    IRMapping mapping;
    for (Value operand : operation->getOperands())
      mapping.map(operand, lookup(operand).value);
    Operation *replacement = rewriter.clone(*operation, mapping);
    if (!mapResults(operation, replacement, false))
      return failure();
  }
  if (!physicalPid)
    return failure();
  auto pidType = dyn_cast<IntegerType>(physicalPid.getType());
  if (!pidType || pidType.getWidth() != 32)
    return failure();

  const Location loc = tokenPid->getLoc();
  Value actualPrograms =
      rewriter.create<triton::GetNumProgramsOp>(loc, triton::ProgramIDDim::X);
  if (actualPrograms.getType() != physicalPid.getType())
    return failure();
  Type i64 = rewriter.getI64Type();
  Value originalTokenExtentI64 =
      rewriter.create<arith::ExtUIOp>(loc, i64, originalTokenExtent);
  Value blockT64 =
      rewriter.create<arith::ConstantIntOp>(loc, candidate.blockT, 64);
  Value ceilNumerator = rewriter.create<arith::AddIOp>(
      loc, originalTokenExtentI64,
      rewriter.create<arith::ConstantIntOp>(loc, candidate.blockT - 1, 64));
  Value logicalTiles =
      rewriter.create<arith::DivUIOp>(loc, ceilNumerator, blockT64);
  Value physicalPidI64 = rewriter.create<arith::ExtUIOp>(loc, i64, physicalPid);
  Value actualProgramsI64 =
      rewriter.create<arith::ExtUIOp>(loc, i64, actualPrograms);
  auto outerLoop = rewriter.create<scf::ForOp>(loc, physicalPidI64,
                                               logicalTiles, actualProgramsI64);
  Block *loopBody = outerLoop.getBody();
  if (!loopBody || !loopBody->mightHaveTerminator())
    return failure();
  rewriter.setInsertionPointToStart(loopBody);

  Value tokenBaseI64 = rewriter.create<arith::MulIOp>(
      loc, outerLoop.getInductionVar(), blockT64);
  Value tokenBase =
      rewriter.create<arith::TruncIOp>(loc, pidType, tokenBaseI64);
  auto tokenType = RankedTensorType::get(
      {static_cast<int64_t>(candidate.blockT)}, physicalPid.getType());
  Value tokenRange = rewriter.create<triton::MakeRangeOp>(
      loc, tokenType, 0, static_cast<int32_t>(candidate.blockT));
  Value tokenBaseSplat =
      rewriter.create<triton::SplatOp>(loc, tokenType, tokenBase);
  Value logicalTokens =
      rewriter.create<arith::AddIOp>(loc, tokenBaseSplat, tokenRange);
  auto tokenI64Type =
      RankedTensorType::get({static_cast<int64_t>(candidate.blockT)}, i64);
  Value logicalTokensI64 =
      rewriter.create<arith::ExtUIOp>(loc, tokenI64Type, logicalTokens);
  Value tokenExtentSplat = rewriter.create<triton::SplatOp>(
      loc, tokenI64Type, originalTokenExtentI64);
  auto tokenTailMask = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::ult, logicalTokensI64, tokenExtentSplat);
  tokenTailMask->setAttr(
      mlir::triton::memory_access::PTSMRuntimeExtentUnsignedMaskTAG,
      rewriter.getUnitAttr());
  values[tokenPid->getResult()] = {logicalTokens, true};

  auto copyMissingAttrs = [](Operation *from, Operation *to) {
    for (NamedAttribute attribute : from->getAttrs())
      if (!to->hasAttr(attribute.getName()))
        to->setAttr(attribute.getName(), attribute.getValue());
  };
  auto liftOperandTo = [&](Value original, RankedTensorType target) -> Value {
    MappedValue mapped = lookup(original);
    if (mapped.tensorized)
      return alignTensor(rewriter, original.getLoc(), mapped.value, target,
                         true, candidate.blockT);
    if (isa<RankedTensorType>(mapped.value.getType()))
      return alignTensor(rewriter, original.getLoc(), mapped.value, target,
                         false, candidate.blockT);
    if (mapped.value.getType() != target.getElementType())
      return Value();
    return rewriter.create<triton::SplatOp>(original.getLoc(), target,
                                            mapped.value);
  };
  auto tailMaskFor = [&](Location location, RankedTensorType target) -> Value {
    return alignTensor(rewriter, location, tokenTailMask, target, true,
                       candidate.blockT);
  };
  auto createElementwise = [&](Operation *operation) -> bool {
    if (operation->getNumResults() == 0)
      return false;
    const bool isReshape = isa<triton::ReshapeOp>(operation);
    if (!operation->hasTrait<OpTrait::Elementwise>() && !isReshape)
      return false;
    if (isReshape) {
      if (operation->getNumOperands() != 1)
        return false;
      auto input =
          dyn_cast<RankedTensorType>(operation->getOperand(0).getType());
      auto output =
          dyn_cast<RankedTensorType>(operation->getResult(0).getType());
      if (!input || !output || !hasSameShape(input, output))
        return false;
    }
    std::optional<RankedTensorType> firstResult = getBatchedTensorType(
        operation->getResult(0).getType(), candidate.blockT);
    if (!firstResult)
      return false;
    SmallVector<Type> resultTypes;
    for (Type result : operation->getResultTypes()) {
      std::optional<RankedTensorType> lifted =
          getBatchedTensorType(result, candidate.blockT);
      if (!lifted)
        return false;
      resultTypes.push_back(*lifted);
    }
    SmallVector<Value> operands;
    for (Value operand : operation->getOperands()) {
      std::optional<RankedTensorType> operandType =
          getSameShapeTensorType(operand.getType(), firstResult->getShape());
      if (!operandType)
        return false;
      Value lifted = liftOperandTo(operand, *operandType);
      if (!lifted)
        return false;
      operands.push_back(lifted);
    }
    Operation *replacement = rewriter.create(
        operation->getLoc(), operation->getName().getIdentifier(), operands,
        resultTypes, operation->getAttrs());
    return mapResults(operation, replacement, true);
  };

  for (Operation *operation : originals) {
    if (!tokenClosure.contains(operation) ||
        operation == tokenPid->getOperation())
      continue;
    rewriter.setInsertionPoint(loopBody->getTerminator());
    if (auto splat = dyn_cast<triton::SplatOp>(operation)) {
      std::optional<RankedTensorType> target =
          getBatchedTensorType(splat.getType(), candidate.blockT);
      if (!target)
        return failure();
      Value value = liftOperandTo(splat.getSrc(), *target);
      if (!value)
        return failure();
      values[splat.getResult()] = {value, true};
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(operation)) {
      std::optional<RankedTensorType> sourceTarget =
          getBatchedTensorType(expand.getSrc().getType(), candidate.blockT);
      std::optional<RankedTensorType> resultTarget =
          getBatchedTensorType(expand.getType(), candidate.blockT);
      if (!sourceTarget || !resultTarget)
        return failure();
      Value source = liftOperandTo(expand.getSrc(), *sourceTarget);
      if (!source)
        return failure();
      Value result = rewriter.create<triton::ExpandDimsOp>(
          operation->getLoc(), source, expand.getAxis() + 1);
      if (result.getType() != *resultTarget)
        return failure();
      values[expand.getResult()] = {result, true};
      continue;
    }
    if (auto broadcast = dyn_cast<triton::BroadcastOp>(operation)) {
      std::optional<RankedTensorType> sourceTarget =
          getBatchedTensorType(broadcast.getSrc().getType(), candidate.blockT);
      std::optional<RankedTensorType> resultTarget =
          getBatchedTensorType(broadcast.getType(), candidate.blockT);
      if (!sourceTarget || !resultTarget)
        return failure();
      Value source = liftOperandTo(broadcast.getSrc(), *sourceTarget);
      if (!source)
        return failure();
      Value result = rewriter.create<triton::BroadcastOp>(
          operation->getLoc(), *resultTarget, source);
      values[broadcast.getResult()] = {result, true};
      continue;
    }
    if (auto reduce = dyn_cast<triton::ReduceOp>(operation)) {
      if (reduce.getSrcs().empty())
        return failure();
      SmallVector<Value> sources;
      for (Value source : reduce.getSrcs()) {
        std::optional<RankedTensorType> target =
            getBatchedTensorType(source.getType(), candidate.blockT);
        if (!target)
          return failure();
        Value lifted = liftOperandTo(source, *target);
        if (!lifted)
          return failure();
        sources.push_back(lifted);
      }
      auto replacement = rewriter.create<triton::ReduceOp>(
          operation->getLoc(), sources, reduce.getAxis() + 1);
      rewriter.cloneRegionBefore(reduce.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      if (!mapResults(operation, replacement.getOperation(), true))
        return failure();
      continue;
    }
    if (auto scan = dyn_cast<triton::ScanOp>(operation)) {
      if (scan.getSrcs().empty())
        return failure();
      SmallVector<Value> sources;
      for (Value source : scan.getSrcs()) {
        std::optional<RankedTensorType> target =
            getBatchedTensorType(source.getType(), candidate.blockT);
        if (!target)
          return failure();
        Value lifted = liftOperandTo(source, *target);
        if (!lifted)
          return failure();
        sources.push_back(lifted);
      }
      auto replacement = rewriter.create<triton::ScanOp>(
          operation->getLoc(), sources,
          static_cast<uint32_t>(scan.getAxis() + 1), scan.getReverse());
      rewriter.cloneRegionBefore(scan.getCombineOp(),
                                 replacement.getCombineOp(),
                                 replacement.getCombineOp().end());
      copyMissingAttrs(operation, replacement.getOperation());
      if (!mapResults(operation, replacement.getOperation(), true))
        return failure();
      continue;
    }
    if (auto load = dyn_cast<triton::LoadOp>(operation)) {
      std::optional<RankedTensorType> pointerType =
          getBatchedTensorType(load.getPtr().getType(), candidate.blockT);
      if (!pointerType)
        return failure();
      Value pointer = liftOperandTo(load.getPtr(), *pointerType);
      auto maskType =
          RankedTensorType::get(pointerType->getShape(), rewriter.getI1Type());
      Value mask =
          load.getMask() ? liftOperandTo(load.getMask(), maskType) : Value();
      Value tailMask = tailMaskFor(operation->getLoc(), maskType);
      if (!pointer || !tailMask)
        return failure();
      mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                   tailMask)
                  : tailMask;
      std::optional<RankedTensorType> resultType = getSameShapeTensorType(
          load.getResult().getType(), pointerType->getShape());
      if (!resultType)
        return failure();
      Value other = load.getOther()
                        ? liftOperandTo(load.getOther(), *resultType)
                        : makeZero(rewriter, operation->getLoc(), *resultType);
      if (!other)
        return failure();
      auto replacement = rewriter.create<triton::LoadOp>(
          operation->getLoc(), pointer, mask, other, load.getBoundaryCheck(),
          load.getPadding(), load.getCache(), load.getEvict(),
          load.getIsVolatile());
      copyMissingAttrs(operation, replacement.getOperation());
      if (!mapResults(operation, replacement.getOperation(), true))
        return failure();
      continue;
    }
    if (auto store = dyn_cast<triton::StoreOp>(operation)) {
      std::optional<RankedTensorType> pointerType =
          getBatchedTensorType(store.getPtr().getType(), candidate.blockT);
      if (!pointerType)
        return failure();
      Value pointer = liftOperandTo(store.getPtr(), *pointerType);
      std::optional<RankedTensorType> valueType = getSameShapeTensorType(
          store.getValue().getType(), pointerType->getShape());
      if (!valueType)
        return failure();
      Value value = liftOperandTo(store.getValue(), *valueType);
      auto maskType =
          RankedTensorType::get(pointerType->getShape(), rewriter.getI1Type());
      Value mask =
          store.getMask() ? liftOperandTo(store.getMask(), maskType) : Value();
      Value tailMask = tailMaskFor(operation->getLoc(), maskType);
      if (!pointer || !value || !tailMask)
        return failure();
      mask = mask ? rewriter.create<arith::AndIOp>(operation->getLoc(), mask,
                                                   tailMask)
                  : tailMask;
      auto replacement = rewriter.create<triton::StoreOp>(
          operation->getLoc(), pointer, value, mask, store.getBoundaryCheck(),
          store.getCache(), store.getEvict());
      copyMissingAttrs(operation, replacement.getOperation());
      continue;
    }
    if (!createElementwise(operation))
      return failure();
  }

  for (Operation *operation : llvm::reverse(originals)) {
    operation->dropAllUses();
    rewriter.eraseOp(operation);
  }
  return mlir::verify(function.getOperation());
}

LogicalResult runProgramMappingCandidateCleanup(ModuleOp module) {
  PassManager cleanup(module.getContext(), module.getOperationName());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  cleanup.addPass(createLoopInvariantCodeMotionPass());
  cleanup.addPass(createCanonicalizerPass());
  cleanup.addPass(createCSEPass());
  return cleanup.run(module);
}

std::optional<LiveByteEstimate>
estimateFinalPersistentPeak(triton::FuncOp function,
                            const PTSMCandidate &candidate) {
  ModuleOp sandbox = ModuleOp::create(function.getLoc());
  sandbox.getBody()->push_back(function->clone());
  auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
  if (!clonedFunction ||
      failed(materializePersistentTaskStripMining(clonedFunction, candidate)) ||
      failed(runProgramMappingCandidateCleanup(sandbox)) ||
      failed(mlir::verify(sandbox.getOperation())))
    return std::nullopt;
  LiveByteEstimate estimate =
      estimatePeakLiveBytes(clonedFunction.getOperation());
  return estimate.known ? std::optional<LiveByteEstimate>(std::move(estimate))
                        : std::nullopt;
}

class PersistentTaskStripMiningPlan final : public RewritePlan {
public:
  PersistentTaskStripMiningPlan(PTSMCandidate candidate, unsigned epoch)
      : candidate(std::move(candidate)), epoch(epoch) {}

  GraphOptimizationRuleId getRuleId() const override {
    return GraphOptimizationRuleId::PersistentTaskStripMining;
  }
  unsigned getBenefit() const override {
    const int64_t score = candidate.evaluation.benefitScore;
    if (score <= 0)
      return 1;
    return static_cast<unsigned>(std::min<int64_t>(
        score, static_cast<int64_t>(std::numeric_limits<unsigned>::max())));
  }
  Operation *getAnchor() const override { return candidate.anchor; }
  unsigned getCreationEpoch() const override { return epoch; }

  LogicalResult revalidate(GraphOptimizationContext &context) const override {
    if (context.getFunction() != candidate.function)
      return failure();
    std::optional<PTSMCandidate> current = analyzeCandidate(context, false);
    return current && sameCandidate(candidate, *current) ? success()
                                                         : failure();
  }

  LogicalResult apply(IRRewriter &rewriter) override {
    (void)rewriter;
    ModuleOp module = candidate.function->getParentOfType<ModuleOp>();
    if (!module || module->hasAttr(kPersistentTaskStripMiningMarkerAttr))
      return failure();
    std::optional<ProgramGridTransformContract> existing =
        getComposableLaunchContract(module);
    if (!existing ||
        existing->transforms.size() != candidate.existingTransformCount)
      return failure();

    ModuleOp sandbox = ModuleOp::create(candidate.function.getLoc());
    if (Attribute oldContract = module->getAttr(kProgramGridTransformsAttr))
      sandbox->setAttr(kProgramGridTransformsAttr, oldContract);
    sandbox.getBody()->push_back(candidate.function->clone());
    auto clonedFunction = dyn_cast<triton::FuncOp>(&sandbox.getBody()->front());
    if (!clonedFunction || failed(applyPersistentCandidateToSandbox(
                               sandbox, clonedFunction, candidate)))
      return failure();

    Attribute transforms = sandbox->getAttr(kProgramGridTransformsAttr);
    if (!transforms || failed(parseProgramGridTransformContract(transforms)) ||
        !hasProgramGridHiddenExtentArguments(clonedFunction))
      return failure();
    if (failed(commitProgramGridFunctionFromSandbox(candidate.function,
                                                    clonedFunction)))
      return failure();
    module->setAttr(kProgramGridTransformsAttr, transforms);
    module->setAttr(kPersistentTaskStripMiningMarkerAttr,
                    sandbox->getAttr(kPersistentTaskStripMiningMarkerAttr));
    return success();
  }

private:
  PTSMCandidate candidate;
  unsigned epoch;
};

class PersistentTaskStripMiningRule final : public GraphOptimizationRule {
public:
  explicit PersistentTaskStripMiningRule(bool enabledForCompileMode)
      : enabledForCompileMode(enabledForCompileMode) {}

  GraphOptimizationRuleId getId() const override {
    return GraphOptimizationRuleId::PersistentTaskStripMining;
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
    std::optional<PTSMCandidate> candidate = analyzeCandidate(context, true);
    if (!candidate)
      return success();
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] matched graph optimization rule "
               << static_cast<unsigned>(getId()) << " ("
               << getGraphOptimizationRuleName(getId()) << ") in @"
               << candidate->function.getName() << ": dynamic-token-extent"
               << " block_t=" << candidate->blockT << "\n");
    plans.push_back(std::make_unique<PersistentTaskStripMiningPlan>(
        std::move(*candidate), context.getEpoch()));
    return success();
  }

private:
  bool enabledForCompileMode;
};

} // namespace

LogicalResult cfg::materializePersistentTaskStripMiningCandidate(
    ModuleOp module, triton::FuncOp function, const ResourceSnapshot &resources,
    unsigned requestedBlockT, CandidateEvaluation *evaluation,
    bool deferIntermediateResourceRejection) {
  if (!module || !function || function->getParentOfType<ModuleOp>() != module ||
      requestedBlockT == 0)
    return failure();
  GraphOptimizationContext context(function, resources);
  const AnalysisRequirement requirements =
      AnalysisRequirement::EntryArgPointerAlias |
      AnalysisRequirement::ProgramAxisDependence |
      AnalysisRequirement::ResourceCost;
  if (failed(context.ensure(requirements)))
    return failure();
  std::optional<PTSMCandidate> candidate = analyzeCandidate(
      context, false, requestedBlockT, deferIntermediateResourceRejection);
  if (!candidate ||
      failed(applyPersistentCandidateToSandbox(module, function, *candidate)))
    return failure();
  if (evaluation)
    *evaluation = candidate->evaluation;
  return success();
}

std::unique_ptr<GraphOptimizationRule> cfg::createPersistentTaskStripMiningRule(
    const PersistentTaskStripMiningRuleOptions &options) {
  return std::make_unique<PersistentTaskStripMiningRule>(
      options.enabledForCompileMode);
}
