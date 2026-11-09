#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Conversion/MLIRTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace tt = mlir::triton;

namespace {

static RankedTensorType getSqmmaAccumulatorTensorType(Type type) {
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy)
    return RankedTensorType();
  return isa_and_nonnull<tt::gpu::MUSASqmmaEncodingAttr>(tensorTy.getEncoding())
             ? tensorTy
             : RankedTensorType();
}

struct Candidate {
  unsigned iterArgIdx;
  triton::musa::SquadDotOp sqmma;
  triton::mtgpu::SqmmaAccumulatorType carrierType;
  // Direct loop yields have no in-loop completion boundary. They are only
  // accepted for a dot immediately followed by scf.yield; a final wait is
  // then materialized after the loop.
  bool directYield = false;
};

// SCF builders in the supported MLIR revisions are inconsistent about whether
// `skipRegionBuilder=true` seeds an implicit scf.yield. Keep that terminator in
// place when present and update its operands after cloning the body.
static void setOrCreateScfYield(Block &block, Location loc, ValueRange operands,
                                RewriterBase &rewriter) {
  if (!block.empty()) {
    if (auto yield = dyn_cast<scf::YieldOp>(&block.back())) {
      yield->setOperands(operands);
      return;
    }
  }
  rewriter.setInsertionPointToEnd(&block);
  scf::YieldOp::create(rewriter, loc, operands);
}

static void setInsertionPointBeforeScfTerminator(Block &block,
                                                 RewriterBase &rewriter) {
  if (!block.empty() && isa<scf::YieldOp>(&block.back())) {
    rewriter.setInsertionPoint(&block.back());
    return;
  }
  rewriter.setInsertionPointToEnd(&block);
}

// Values which carry an asynchronous SQMMA accumulator through an SCF region
// have to use the opaque native carrier type after conversion.
using CarrierValueSet = llvm::SmallDenseSet<Value, 16>;
using TensorMaterializationMap = DenseMap<Value, DenseMap<Block *, Value>>;

static triton::musa::SquadDotWaitOp getCanonicalExternalFinalWait(
    scf::ForOp forOp, const llvm::SmallDenseSet<unsigned> &candidateIdxs) {
  auto wait =
      dyn_cast_or_null<triton::musa::SquadDotWaitOp>(forOp->getNextNode());
  if (!wait || wait->getBlock() != forOp->getBlock())
    return {};

  bool dependsOnCandidate = llvm::any_of(wait.getInputs(), [&](Value input) {
    auto result = dyn_cast<OpResult>(input);
    return result && result.getOwner() == forOp.getOperation() &&
           candidateIdxs.contains(result.getResultNumber());
  });
  return dependsOnCandidate ? wait : triton::musa::SquadDotWaitOp();
}

static Value unwrapYieldedSqmmaValue(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return value;
  auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(result.getOwner());
  if (!wait)
    return value;
  unsigned idx = result.getResultNumber();
  return idx < wait.getInputs().size() ? wait.getInputs()[idx] : value;
}

// Return true when `value` is on a loop-carried SQMMA path rooted at
// `iterArg`. An scf.if result is accepted only when every present branch
// yields a value on that path. This keeps conversion fail-closed.
static bool markCarrierPath(Value value, Value iterArg,
                            CarrierValueSet &carrierIfResults,
                            llvm::SmallPtrSetImpl<void *> &visited) {
  if (!value || !visited.insert(value.getAsOpaquePointer()).second)
    return false;
  if (value == iterArg)
    return true;

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;
  Operation *def = result.getOwner();
  if (auto dot = dyn_cast<triton::musa::SquadDotOp>(def)) {
    if (dot->getOperand(2) == iterArg)
      return true;
    return markCarrierPath(dot->getOperand(2), iterArg, carrierIfResults,
                           visited);
  }

  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(def)) {
    unsigned idx = result.getResultNumber();
    return idx < wait.getInputs().size() &&
           markCarrierPath(wait.getInputs()[idx], iterArg, carrierIfResults,
                           visited);
  }

  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned idx = result.getResultNumber();
    bool found = false;
    auto markRegion = [&](Region &region) {
      if (region.empty())
        return true;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || idx >= yield.getNumOperands())
        return false;
      llvm::SmallPtrSet<void *, 16> branchVisited;
      if (!markCarrierPath(yield.getOperand(idx), iterArg, carrierIfResults,
                           branchVisited))
        return false;
      found = true;
      return true;
    };
    if (!markRegion(ifOp.getThenRegion()) ||
        (ifOp.elseBlock() && !markRegion(ifOp.getElseRegion())))
      return false;
    if (found)
      carrierIfResults.insert(value);
    return found;
  }

  // Do not treat arbitrary one-input/one-result tensor operations as carrier
  // preserving. Most such operations require an unpacked tensor and cannot
  // legally yield the opaque accumulator type through SCF.
  return false;
}

static std::optional<triton::musa::SquadDotOp>
findSqmmaOnCarrierPath(Value value, Value iterArg,
                       llvm::SmallPtrSetImpl<void *> &visited) {
  if (!value || !visited.insert(value.getAsOpaquePointer()).second)
    return std::nullopt;
  if (value == iterArg)
    return std::nullopt;
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return std::nullopt;
  Operation *def = result.getOwner();
  if (auto dot = dyn_cast<triton::musa::SquadDotOp>(def)) {
    if (dot->getOperand(2) == iterArg)
      return std::optional<triton::musa::SquadDotOp>(dot);
    return findSqmmaOnCarrierPath(dot->getOperand(2), iterArg, visited);
  }
  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(def)) {
    unsigned idx = result.getResultNumber();
    return idx < wait.getInputs().size()
               ? findSqmmaOnCarrierPath(wait.getInputs()[idx], iterArg, visited)
               : std::nullopt;
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
    unsigned idx = result.getResultNumber();
    std::optional<triton::musa::SquadDotOp> found;
    auto findRegion = [&](Region &region) {
      if (region.empty())
        return true;
      auto yield = dyn_cast<scf::YieldOp>(region.front().getTerminator());
      if (!yield || idx >= yield.getNumOperands())
        return false;
      llvm::SmallPtrSet<void *, 16> branchVisited;
      auto branchDot =
          findSqmmaOnCarrierPath(yield.getOperand(idx), iterArg, branchVisited);
      if (branchDot) {
        if (found && found != branchDot)
          return false;
        found = branchDot;
      }
      return true;
    };
    if (!findRegion(ifOp.getThenRegion()) ||
        (ifOp.elseBlock() && !findRegion(ifOp.getElseRegion())))
      return std::nullopt;
    return found;
  }
  return std::nullopt;
}

static triton::mtgpu::SQMMAEltType
convertEltType(triton::musa::SQMMAEltType type) {
  return static_cast<triton::mtgpu::SQMMAEltType>(static_cast<int>(type));
}

static triton::mtgpu::SQMMALayout
convertLayout(triton::musa::SQMMALayout layout) {
  return static_cast<triton::mtgpu::SQMMALayout>(static_cast<int>(layout));
}

static triton::mtgpu::SQMMAAccumulationMode
convertAccumulationMode(triton::musa::SQMMAAccumulationMode mode) {
  return static_cast<triton::mtgpu::SQMMAAccumulationMode>(
      static_cast<int>(mode));
}

static std::optional<Candidate>
getCandidateForIterArg(scf::ForOp forOp, unsigned iterArgIdx,
                       CarrierValueSet &carrierIfResults) {
  Value iterArg = forOp.getRegionIterArg(iterArgIdx);
  auto tensorTy = getSqmmaAccumulatorTensorType(iterArg.getType());
  if (!tensorTy)
    return std::nullopt;

  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp || iterArgIdx >= yieldOp.getNumOperands())
    return std::nullopt;

  Value originalYieldedValue = yieldOp.getOperand(iterArgIdx);
  Value yieldedValue = unwrapYieldedSqmmaValue(originalYieldedValue);
  auto sqmma = yieldedValue.getDefiningOp<triton::musa::SquadDotOp>();
  if (!sqmma) {
    llvm::SmallPtrSet<void *, 16> visited;
    auto nested =
        findSqmmaOnCarrierPath(originalYieldedValue, iterArg, visited);
    if (!nested)
      return std::nullopt;
    sqmma = *nested;
  }
  if (!sqmma)
    return std::nullopt;

  // Validate the complete yielded path even for the direct dot/wait(dot)
  // fast path. Otherwise an unrelated dot could be rebound to this loop's
  // initial accumulator solely because it happens to be yielded at the same
  // result index.
  llvm::SmallPtrSet<void *, 16> pathVisited;
  if (!markCarrierPath(originalYieldedValue, iterArg, carrierIfResults,
                       pathVisited))
    return std::nullopt;

  bool directYield = originalYieldedValue == yieldedValue;
  return Candidate{
      iterArgIdx, sqmma,
      triton::mtgpu::SqmmaAccumulatorType::get(forOp.getContext(), tensorTy),
      directYield};
}

static Value materializeTensorAccumulatorForUse(
    Value original, Location loc, IRMapping &mapping,
    TensorMaterializationMap &tensorMaterializations, RewriterBase &rewriter) {
  Value mapped = mapping.lookupOrDefault(original);
  if (!mapped || mapped.getType() == original.getType())
    return mapped;

  auto originalTensorTy = getSqmmaAccumulatorTensorType(original.getType());
  if (!originalTensorTy ||
      !isa<triton::mtgpu::SqmmaAccumulatorType>(mapped.getType()))
    return mapped;

  Block *insertionBlock = rewriter.getInsertionBlock();
  if (!insertionBlock)
    return mapped;
  auto &byBlock = tensorMaterializations[original];
  auto it = byBlock.find(insertionBlock);
  if (it != byBlock.end())
    return it->second;

  Value unpacked = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
      rewriter, loc, originalTensorTy, mapped);
  byBlock[insertionBlock] = unpacked;
  return unpacked;
}

static bool preservesSqmmaCarrierOperand(Operation *op, unsigned operandIdx) {
  if (isa<triton::mtgpu::SqmmaOp, triton::mtgpu::SqmmaWaitOp,
          triton::mtgpu::UnpackSqmmaAccumulatorOp>(op))
    return true;

  if (auto yield = dyn_cast<scf::YieldOp>(op)) {
    Operation *parent = yield->getParentOp();
    return parent && operandIdx < parent->getNumResults() &&
           isa<triton::mtgpu::SqmmaAccumulatorType>(
               parent->getResult(operandIdx).getType());
  }

  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    constexpr unsigned firstInitArgOperand = 3;
    if (operandIdx < firstInitArgOperand)
      return false;
    unsigned resultIdx = operandIdx - firstInitArgOperand;
    return resultIdx < forOp.getNumResults() &&
           isa<triton::mtgpu::SqmmaAccumulatorType>(
               forOp.getResult(resultIdx).getType());
  }

  return false;
}

static Value materializeTensorAccumulatorFromCarrier(
    Value carrier, Location loc,
    DenseMap<Value, DenseMap<Block *, Value>> &tensorMaterializations,
    RewriterBase &rewriter, Operation *insertionPoint) {
  auto carrierTy =
      dyn_cast<triton::mtgpu::SqmmaAccumulatorType>(carrier.getType());
  if (!carrierTy || !insertionPoint || !insertionPoint->getBlock())
    return carrier;

  Block *block = insertionPoint->getBlock();
  auto &byBlock = tensorMaterializations[carrier];
  auto it = byBlock.find(block);
  if (it != byBlock.end())
    return it->second;

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(insertionPoint);
  Value unpacked = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
      rewriter, loc, carrierTy.getAccumulatorType(), carrier);
  byBlock[block] = unpacked;
  return unpacked;
}

static void materializeNestedTensorAccumulatorUses(Operation *root,
                                                   RewriterBase &rewriter) {
  DenseMap<Value, DenseMap<Block *, Value>> tensorMaterializations;
  root->walk([&](Operation *op) {
    for (OpOperand &operand : op->getOpOperands()) {
      if (preservesSqmmaCarrierOperand(op, operand.getOperandNumber()))
        continue;
      Value replacement = materializeTensorAccumulatorFromCarrier(
          operand.get(), op->getLoc(), tensorMaterializations, rewriter, op);
      if (replacement != operand.get())
        operand.set(replacement);
    }
  });
}

static Operation *cloneSqmmaOp(triton::musa::SquadDotOp op, IRMapping &mapping,
                               TensorMaterializationMap &tensorMaterializations,
                               RewriterBase &rewriter) {
  auto lookupOrDefault = [&](Value value) -> Value {
    return value ? mapping.lookupOrDefault(value) : Value();
  };
  auto materializeTensorOperand = [&](Value value) -> Value {
    return materializeTensorAccumulatorForUse(value, op.getLoc(), mapping,
                                              tensorMaterializations, rewriter);
  };
  Value mappedA = materializeTensorOperand(op.getA());
  Value mappedB = materializeTensorOperand(op.getB());
  // Read C through the generic operand list. A loop cloned by an earlier
  // fixed-point iteration may already expose the opaque carrier on C; the
  // generated typed accessor only accepts ranked tensor values.
  Value originalC = op->getOperand(2);
  Value mappedC = lookupOrDefault(originalC);
  Value mappedUseC = lookupOrDefault(op.getUseC());
  if (auto carrierTy =
          dyn_cast<triton::mtgpu::SqmmaAccumulatorType>(mappedC.getType())) {
    auto newOp = triton::mtgpu::SqmmaOp::create(
        rewriter, op.getLoc(), carrierTy, mappedA, mappedB, mappedC, mappedUseC,
        op.getM(), op.getN(), op.getK(), convertEltType(op.getEltTypeC()),
        convertEltType(op.getEltTypeA()), convertEltType(op.getEltTypeB()),
        convertLayout(op.getLayoutA()), convertLayout(op.getLayoutB()),
        op.getIsAsync(), convertAccumulationMode(op.getAccMode()),
        op.getInputPrecision(), op.getMaxNumImpreciseAcc());
    newOp->setAttrs(op->getAttrs());
    return newOp;
  }

  auto newOp = triton::musa::SquadDotOp::create(
      rewriter, op.getLoc(), op.getResult().getType(), mappedA, mappedB,
      mappedC, mappedUseC, op.getM(), op.getN(), op.getK(), op.getEltTypeC(),
      op.getEltTypeA(), op.getEltTypeB(), op.getLayoutA(), op.getLayoutB(),
      op.getIsAsync(), op.getAccMode(), op.getInputPrecision(),
      op.getMaxNumImpreciseAcc());
  newOp->setAttrs(op->getAttrs());
  return newOp;
}

static Operation *cloneSqmmaWaitOp(triton::musa::SquadDotWaitOp op,
                                   IRMapping &mapping, RewriterBase &rewriter) {
  SmallVector<Value> newInputs;
  newInputs.reserve(op.getInputs().size());
  for (Value input : op.getInputs())
    newInputs.push_back(mapping.lookupOrDefault(input));
  auto newOp =
      triton::mtgpu::SqmmaWaitOp::create(rewriter, op.getLoc(), newInputs);
  newOp->setAttrs(op->getAttrs());
  return newOp;
}

static Operation *
cloneNestedSqmmaOperation(Operation &op, IRMapping &mapping,
                          TensorMaterializationMap &tensorMaterializations,
                          const CarrierValueSet &carrierIfResults,
                          RewriterBase &rewriter);

static LogicalResult
cloneSqmmaIfRegion(Region &oldRegion, Region &newRegion, IRMapping &mapping,
                   TensorMaterializationMap &tensorMaterializations,
                   const CarrierValueSet &carrierIfResults,
                   RewriterBase &rewriter) {
  if (oldRegion.empty() || newRegion.empty())
    return success();

  Block &oldBlock = oldRegion.front();
  Block &newBlock = newRegion.front();
  for (auto [oldArg, newArg] :
       llvm::zip(oldBlock.getArguments(), newBlock.getArguments()))
    mapping.map(oldArg, newArg);

  for (Operation &oldOp : oldBlock.without_terminator()) {
    setInsertionPointBeforeScfTerminator(newBlock, rewriter);
    Operation *newOp = cloneNestedSqmmaOperation(
        oldOp, mapping, tensorMaterializations, carrierIfResults, rewriter);
    if (!newOp)
      return failure();
    for (auto [oldResult, newResult] :
         llvm::zip(oldOp.getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
  }

  auto oldYield = dyn_cast<scf::YieldOp>(oldBlock.getTerminator());
  if (!oldYield)
    return failure();
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYield.getNumOperands());
  for (Value operand : oldYield.getOperands()) {
    Value mapped = mapping.lookupOrDefault(operand);
    if (!mapped)
      return failure();
    newYieldOperands.push_back(mapped);
  }
  setOrCreateScfYield(newBlock, oldYield.getLoc(), newYieldOperands, rewriter);
  return success();
}

static Operation *
cloneNestedSqmmaOperation(Operation &op, IRMapping &mapping,
                          TensorMaterializationMap &tensorMaterializations,
                          const CarrierValueSet &carrierIfResults,
                          RewriterBase &rewriter) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(&op))
    return cloneSqmmaOp(sqmma, mapping, tensorMaterializations, rewriter);
  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(&op))
    return cloneSqmmaWaitOp(wait, mapping, rewriter);

  if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
    bool carriesAccumulator =
        llvm::any_of(ifOp.getResults(), [&](Value result) {
          return carrierIfResults.contains(result);
        });
    if (carriesAccumulator) {
      SmallVector<Type> resultTypes;
      resultTypes.reserve(ifOp.getNumResults());
      for (Value result : ifOp.getResults()) {
        if (!carrierIfResults.contains(result)) {
          resultTypes.push_back(result.getType());
          continue;
        }
        auto tensorTy = getSqmmaAccumulatorTensorType(result.getType());
        if (!tensorTy)
          return nullptr;
        resultTypes.push_back(triton::mtgpu::SqmmaAccumulatorType::get(
            rewriter.getContext(), tensorTy));
      }

      Value cond = mapping.lookupOrDefault(ifOp.getCondition());
      auto newIf =
          scf::IfOp::create(rewriter, ifOp.getLoc(), resultTypes, cond,
                            /*withElseRegion=*/ifOp.elseBlock() != nullptr,
                            /*skipRegionBuilder=*/true);
      newIf->setAttrs(ifOp->getAttrs());
      if (newIf.getThenRegion().empty())
        newIf.getThenRegion().push_back(new Block);
      if (ifOp.elseBlock() && newIf.getElseRegion().empty())
        newIf.getElseRegion().push_back(new Block);

      if (failed(cloneSqmmaIfRegion(ifOp.getThenRegion(), newIf.getThenRegion(),
                                    mapping, tensorMaterializations,
                                    carrierIfResults, rewriter)))
        return nullptr;
      if (ifOp.elseBlock() &&
          failed(cloneSqmmaIfRegion(ifOp.getElseRegion(), newIf.getElseRegion(),
                                    mapping, tensorMaterializations,
                                    carrierIfResults, rewriter)))
        return nullptr;
      return newIf.getOperation();
    }
  }

  IRMapping opMapping(mapping);
  for (Value operand : op.getOperands()) {
    Value remapped = materializeTensorAccumulatorForUse(
        operand, op.getLoc(), mapping, tensorMaterializations, rewriter);
    if (remapped != mapping.lookupOrDefault(operand))
      opMapping.map(operand, remapped);
  }
  Operation *newOp = rewriter.clone(op, opMapping);
  materializeNestedTensorAccumulatorUses(newOp, rewriter);
  return newOp;
}

static bool convertLoopCarriedSqmmaAccumulator(scf::ForOp forOp,
                                               RewriterBase &rewriter) {
  CarrierValueSet carrierIfResults;
  SmallVector<Candidate> candidates;
  for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx) {
    // A loop converted by an earlier fixed-point iteration already exposes
    // opaque result and iter-arg types and must not be reconsidered.
    if (!isa<RankedTensorType>(forOp.getResult(idx).getType()) ||
        !isa<RankedTensorType>(forOp.getRegionIterArg(idx).getType()))
      continue;
    if (auto candidate = getCandidateForIterArg(forOp, idx, carrierIfResults))
      candidates.push_back(*candidate);
  }
  if (candidates.empty())
    return false;

  llvm::SmallDenseSet<unsigned> candidateIdxs;
  for (const Candidate &candidate : candidates)
    candidateIdxs.insert(candidate.iterArgIdx);
  triton::musa::SquadDotWaitOp externalFinalWait =
      getCanonicalExternalFinalWait(forOp, candidateIdxs);
  bool hasDirectYield =
      llvm::any_of(candidates, [](const Candidate &candidate) {
        return candidate.directYield;
      });

  Location loc = forOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);

  SmallVector<Value> initArgs(forOp.getInitArgs());
  SmallVector<Operation *> initPacks;
  for (const Candidate &candidate : candidates) {
    auto pack = triton::mtgpu::PackSqmmaAccumulatorOp::create(
        rewriter, loc, candidate.carrierType, initArgs[candidate.iterArgIdx]);
    initArgs[candidate.iterArgIdx] = pack;
    initPacks.push_back(pack.getOperation());
  }

  scf::ForOp newFor =
      scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                         forOp.getUpperBound(), forOp.getStep(), initArgs);

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newFor.getInductionVar());
  for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx)
    mapping.map(forOp.getRegionIterArg(idx), newFor.getRegionIterArg(idx));

  Block &oldBody = forOp.getRegion().front();
  Block &newBody = newFor.getRegion().front();
  setInsertionPointBeforeScfTerminator(newBody, rewriter);
  TensorMaterializationMap tensorMaterializations;

  for (Operation &op : oldBody.without_terminator()) {
    // Recursive cloning may leave the insertion point in a nested region.
    // Restore it before each top-level sibling.
    setInsertionPointBeforeScfTerminator(newBody, rewriter);
    if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
      Operation *newOp =
          cloneSqmmaOp(sqmma, mapping, tensorMaterializations, rewriter);
      mapping.map(sqmma.getResult(), newOp->getResult(0));
      continue;
    }
    if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(op)) {
      Operation *newOp = cloneSqmmaWaitOp(wait, mapping, rewriter);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(wait.getResults(), newOp->getResults()))
        mapping.map(oldResult, newResult);
      continue;
    }

    Operation *newOp = cloneNestedSqmmaOperation(
        op, mapping, tensorMaterializations, carrierIfResults, rewriter);
    if (!newOp) {
      newFor->dropAllReferences();
      rewriter.eraseOp(newFor);
      for (Operation *pack : llvm::reverse(initPacks))
        rewriter.eraseOp(pack);
      return false;
    }
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
  }

  auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYield.getNumOperands());
  for (Value operand : oldYield.getOperands())
    newYieldOperands.push_back(mapping.lookupOrDefault(operand));
  setOrCreateScfYield(newBody, oldYield.getLoc(), newYieldOperands, rewriter);

  rewriter.setInsertionPointAfter(newFor);
  DenseMap<Value, Value> externalWaitUnpacks;
  llvm::SmallDenseMap<unsigned, Value> waitedDirectResults;
  if (externalFinalWait || hasDirectYield) {
    auto oldWait = externalFinalWait;
    SmallVector<Value> newInputs;
    llvm::SmallDenseMap<unsigned, unsigned> directWaitResultIndices;
    if (externalFinalWait) {
      newInputs.reserve(oldWait.getInputs().size());
      for (auto [inputIndex, input] : llvm::enumerate(oldWait.getInputs())) {
        Value newInput = input;
        if (auto result = dyn_cast<OpResult>(input)) {
          if (result.getOwner() == forOp.getOperation()) {
            newInput = newFor.getResult(result.getResultNumber());
            if (candidateIdxs.contains(result.getResultNumber()))
              directWaitResultIndices[result.getResultNumber()] = inputIndex;
          } else {
            Value remapped = mapping.lookupOrDefault(input);
            if (remapped)
              newInput = remapped;
          }
        }
        newInputs.push_back(newInput);
      }
    }

    // Directly yielded carriers have no old wait to clone. Append them to the
    // final wait, including loops that mix direct and explicit-wait carriers.
    for (const Candidate &candidate : candidates) {
      if (candidate.directYield &&
          !directWaitResultIndices.contains(candidate.iterArgIdx)) {
        directWaitResultIndices[candidate.iterArgIdx] = newInputs.size();
        newInputs.push_back(newFor.getResult(candidate.iterArgIdx));
      }
    }

    Location waitLoc = externalFinalWait ? oldWait.getLoc() : forOp.getLoc();
    auto newWait =
        triton::mtgpu::SqmmaWaitOp::create(rewriter, waitLoc, newInputs);
    if (externalFinalWait)
      newWait->setAttrs(oldWait->getAttrs());

    for (const Candidate &candidate : candidates)
      if (candidate.directYield)
        waitedDirectResults[candidate.iterArgIdx] =
            newWait->getResult(directWaitResultIndices[candidate.iterArgIdx]);

    if (externalFinalWait) {
      for (unsigned idx = 0; idx < oldWait.getNumResults(); ++idx) {
        Value oldResult = oldWait.getResult(idx);
        Value newResult = newWait.getResult(idx);
        Value replacement = newResult;
        if (auto oldInput = dyn_cast<OpResult>(oldWait.getInputs()[idx])) {
          if (oldInput.getOwner() == forOp.getOperation() &&
              candidateIdxs.contains(oldInput.getResultNumber())) {
            auto it = externalWaitUnpacks.find(newResult);
            if (it == externalWaitUnpacks.end()) {
              replacement = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
                  rewriter, oldWait.getLoc(), oldResult.getType(), newResult);
              externalWaitUnpacks[newResult] = replacement;
            } else {
              replacement = it->second;
            }
          }
        }
        rewriter.replaceAllUsesWith(oldResult, replacement);
      }

      rewriter.eraseOp(oldWait);
    }
    rewriter.setInsertionPointAfter(newWait);
  }

  SmallVector<Value> replacements;
  replacements.reserve(forOp.getNumResults());
  for (unsigned idx = 0; idx < forOp.getNumResults(); ++idx) {
    Value result = newFor.getResult(idx);
    if (candidateIdxs.contains(idx)) {
      if (auto waited = waitedDirectResults.find(idx);
          waited != waitedDirectResults.end())
        result = waited->second;
      auto tensorTy = cast<RankedTensorType>(forOp.getResult(idx).getType());
      result = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
          rewriter, loc, tensorTy, result);
    }
    replacements.push_back(result);
  }
  // Replace externally visible results and drop dead source-region references
  // before erasing the old parent. Cross-region RAUW is invalid for nested
  // async dots still referenced by their source SCF yields.
  for (auto [oldResult, replacement] :
       llvm::zip(forOp.getResults(), replacements))
    oldResult.replaceAllUsesWith(replacement);
  forOp->walk([&](Operation *oldOp) {
    if (oldOp != forOp.getOperation()) {
      oldOp->dropAllUses();
      oldOp->dropAllReferences();
    }
  });
  forOp->dropAllUses();
  forOp->dropAllReferences();
  rewriter.eraseOp(forOp);
  return true;
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUCONVERTSQMMATOMTGPU
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUConvertSqmmaToMTGPUPass
    : impl::TritonMUSAGPUConvertSqmmaToMTGPUBase<
          TritonMUSAGPUConvertSqmmaToMTGPUPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());

    for (tt::FuncOp func : mod.getOps<tt::FuncOp>()) {
      bool changed = true;
      while (changed) {
        changed = false;
        SmallVector<scf::ForOp> loops;
        func.walk([&](scf::ForOp loop) { loops.push_back(loop); });
        for (scf::ForOp loop : loops) {
          if (!loop->getBlock())
            continue;
          if (convertLoopCarriedSqmmaAccumulator(loop, rewriter))
            changed = true;
        }
      }
    }
  }
};

} // namespace mlir
