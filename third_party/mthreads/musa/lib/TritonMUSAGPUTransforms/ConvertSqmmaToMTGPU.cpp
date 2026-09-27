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
// `skipRegionBuilder=true` seeds an implicit scf.yield.  Keep that terminator
// in place when present and update its operands after cloning the body.  This
// avoids relying on Block::mightHaveTerminator() (which can be false for a
// freshly-created block) and guarantees that a new yield is never inserted in
// front of an existing one.
static void setOrCreateScfYield(Block &block, Location loc,
                                ValueRange operands, RewriterBase &rewriter) {
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
// have to use the opaque native carrier type after conversion.  Keep this
// set separate from the candidate dots: an if result is a carrier boundary,
// while the dot itself is converted by cloneSqmmaOp when its C operand is a
// carrier.
using CarrierValueSet = llvm::SmallDenseSet<Value, 16>;

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
// `iterArg`.  In particular, an scf.if result is accepted only when every
// present branch yields either the loop accumulator or a dot derived from it.
// This keeps the conversion fail-closed: arbitrary tensor values cannot be
// reinterpreted as native accumulator carriers.
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
    // Follow an accumulator chain within the loop body.  A grouped rolling
    // consumer may issue several dots (and waits) before yielding the final
    // value; the first dot is still rooted at the SCF iter arg and all later
    // dots can use the same native carrier.
    if (dot->getOperand(2) == iterArg)
      return true;
    return markCarrierPath(dot->getOperand(2), iterArg, carrierIfResults,
                           visited);
  }

  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(def)) {
    unsigned idx = result.getResultNumber();
    return idx < wait.getInputs().size() &&
           markCarrierPath(wait.getInputs()[idx], iterArg,
                           carrierIfResults, visited);
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
      if (!markCarrierPath(yield.getOperand(idx), iterArg,
                           carrierIfResults, branchVisited))
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

  // Transparent layout/view operations may sit between an if result and the
  // loop yield.  They retain tensor semantics and are materialized around the
  // carrier by the existing conversion helpers.
  if (def->getNumOperands() == 1 && def->getNumResults() == 1)
    return markCarrierPath(def->getOperand(0), iterArg, carrierIfResults,
                           visited);
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
               ? findSqmmaOnCarrierPath(wait.getInputs()[idx], iterArg,
                                        visited)
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
      auto branchDot = findSqmmaOnCarrierPath(yield.getOperand(idx), iterArg,
                                               branchVisited);
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
  if (def->getNumOperands() == 1 && def->getNumResults() == 1)
    return findSqmmaOnCarrierPath(def->getOperand(0), iterArg, visited);
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
    auto nested = findSqmmaOnCarrierPath(originalYieldedValue, iterArg,
                                         visited);
    if (!nested)
      return std::nullopt;
    sqmma = *nested;
    visited.clear();
    if (!markCarrierPath(originalYieldedValue, iterArg, carrierIfResults,
                         visited))
      return std::nullopt;
  }
  if (!sqmma)
    return std::nullopt;

  bool directYield = originalYieldedValue == yieldedValue;
  return Candidate{
      iterArgIdx, sqmma,
      triton::mtgpu::SqmmaAccumulatorType::get(forOp.getContext(), tensorTy),
      directYield};
}

static Value materializeTensorAccumulatorForUse(
    Value original, Location loc, IRMapping &mapping,
    DenseMap<Value, Value> &tensorMaterializations, RewriterBase &rewriter) {
  Value mapped = mapping.lookupOrDefault(original);
  if (!mapped || mapped.getType() == original.getType())
    return mapped;

  auto originalTensorTy = getSqmmaAccumulatorTensorType(original.getType());
  if (!originalTensorTy ||
      !isa<triton::mtgpu::SqmmaAccumulatorType>(mapped.getType()))
    return mapped;

  auto it = tensorMaterializations.find(original);
  if (it != tensorMaterializations.end())
    return it->second;

  Value unpacked = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
      rewriter, loc, originalTensorTy, mapped);
  tensorMaterializations[original] = unpacked;
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
                               DenseMap<Value, Value> &tensorMaterializations,
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
  // Read C through the generic operand list.  A loop cloned by an earlier
  // fixed-point iteration may already expose the opaque carrier on C; the
  // generated typed accessor on the source MUSA op only accepts ranked tensor
  // values and would assert before we can remap it.
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

static Operation *cloneNestedSqmmaOperation(
    Operation &op, IRMapping &mapping,
    DenseMap<Value, Value> &tensorMaterializations,
    const CarrierValueSet &carrierIfResults, RewriterBase &rewriter);

static LogicalResult cloneSqmmaIfRegion(
    Region &oldRegion, Region &newRegion, IRMapping &mapping,
    DenseMap<Value, Value> &tensorMaterializations,
    const CarrierValueSet &carrierIfResults, RewriterBase &rewriter) {
  if (oldRegion.empty() || newRegion.empty())
    return success();

  Block &oldBlock = oldRegion.front();
  Block &newBlock = newRegion.front();
  for (auto [oldArg, newArg] : llvm::zip(oldBlock.getArguments(),
                                         newBlock.getArguments()))
    mapping.map(oldArg, newArg);

  // `scf::IfOp::create` may materialize a default `scf.yield` even when the
  // skip-region-builder flag is set (the exact behavior depends on the MLIR
  // revision). Remove it before cloning the source body; otherwise appending
  // the mapped terminator below leaves two yields in the new block and the
  // verifier reports that `scf.yield` is not the final operation.
  for (Operation &oldOp : oldBlock.without_terminator()) {
    // Recursive cloning of a nested scf.if changes the rewriter insertion
    // point to the nested region. Restore the outer block before each sibling
    // operation so later values are not accidentally appended to that region.
    setInsertionPointBeforeScfTerminator(newBlock, rewriter);
    Operation *newOp = cloneNestedSqmmaOperation(
        oldOp, mapping, tensorMaterializations, carrierIfResults, rewriter);
    if (!newOp)
      return failure();
    for (auto [oldResult, newResult] :
         llvm::zip(oldOp.getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    // The source region is erased after the replacement loop is built.  RAUW
    // nested results immediately so its terminator cannot retain a use of an
    // operation that is about to be destroyed with the old parent.
    for (auto [oldResult, newResult] :
         llvm::zip(oldOp.getResults(), newOp->getResults()))
      oldResult.replaceAllUsesWith(newResult);
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

static Operation *cloneNestedSqmmaOperation(
    Operation &op, IRMapping &mapping,
    DenseMap<Value, Value> &tensorMaterializations,
    const CarrierValueSet &carrierIfResults, RewriterBase &rewriter) {
  if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(&op))
    return cloneSqmmaOp(sqmma, mapping, tensorMaterializations, rewriter);
  if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(&op))
    return cloneSqmmaWaitOp(wait, mapping, rewriter);

  if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
    bool carriesAccumulator = llvm::any_of(
        ifOp.getResults(), [&](Value result) {
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
      auto newIf = scf::IfOp::create(
          rewriter, ifOp.getLoc(), resultTypes, cond,
          /*withElseRegion=*/ifOp.elseBlock() != nullptr,
          /*skipRegionBuilder=*/true);
      newIf->setAttrs(ifOp->getAttrs());
      // The skip-region builder leaves the regions empty.  Materialize one
      // block per source region before recursively cloning their bodies.
      if (newIf.getThenRegion().empty())
        newIf.getThenRegion().push_back(new Block);
      if (ifOp.elseBlock() && newIf.getElseRegion().empty())
        newIf.getElseRegion().push_back(new Block);

      if (failed(cloneSqmmaIfRegion(
              ifOp.getThenRegion(), newIf.getThenRegion(), mapping,
              tensorMaterializations, carrierIfResults, rewriter)))
        return nullptr;
      if (ifOp.elseBlock() &&
          failed(cloneSqmmaIfRegion(
              ifOp.getElseRegion(), newIf.getElseRegion(), mapping,
              tensorMaterializations, carrierIfResults, rewriter)))
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
    // A loop produced by an earlier carrier conversion already exposes an
    // opaque result type.  It must not be reconsidered by the fixed-point
    // walk below, whose unpack step expects the original ranked tensor type.
    if (!isa<RankedTensorType>(forOp.getResult(idx).getType()) ||
        !isa<RankedTensorType>(forOp.getRegionIterArg(idx).getType()))
      continue;
    if (auto candidate =
            getCandidateForIterArg(forOp, idx, carrierIfResults))
      candidates.push_back(*candidate);
  }
  if (candidates.empty())
    return false;

  llvm::SmallDenseSet<unsigned> candidateIdxs;
  for (const Candidate &candidate : candidates)
    candidateIdxs.insert(candidate.iterArgIdx);
  triton::musa::SquadDotWaitOp externalFinalWait =
      getCanonicalExternalFinalWait(forOp, candidateIdxs);
  bool hasDirectYield = llvm::any_of(candidates, [](const Candidate &candidate) {
    return candidate.directYield;
  });

  Location loc = forOp.getLoc();
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);

  SmallVector<Value> initArgs(forOp.getInitArgs());
  for (const Candidate &candidate : candidates) {
    initArgs[candidate.iterArgIdx] =
        triton::mtgpu::PackSqmmaAccumulatorOp::create(
            rewriter, loc, candidate.carrierType,
            initArgs[candidate.iterArgIdx]);
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
  // The SCF builder may seed the cloned loop body with an implicit yield.
  // Remove it before cloning the source body so the explicit mapped yield
  // below remains the sole terminator.
  setInsertionPointBeforeScfTerminator(newBody, rewriter);
  DenseMap<Value, Value> tensorMaterializations;

  for (Operation &op : oldBody.without_terminator()) {
    // Nested cloning leaves the insertion point in the nested region.  Reset
    // it before every top-level sibling so later operations stay in the loop
    // body and remain before its implicit yield, when one exists.
    setInsertionPointBeforeScfTerminator(newBody, rewriter);
    if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
      Operation *newOp =
          cloneSqmmaOp(sqmma, mapping, tensorMaterializations, rewriter);
      mapping.map(sqmma.getResult(), newOp->getResult(0));
      sqmma.getResult().replaceAllUsesWith(newOp->getResult(0));
      continue;
    }
    if (auto wait = dyn_cast<triton::musa::SquadDotWaitOp>(op)) {
      Operation *newOp = cloneSqmmaWaitOp(wait, mapping, rewriter);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(wait.getResults(), newOp->getResults()))
        mapping.map(oldResult, newResult);
      for (auto [oldResult, newResult] :
           llvm::zip_equal(wait.getResults(), newOp->getResults()))
        oldResult.replaceAllUsesWith(newResult);
      continue;
    }

    Operation *newOp = cloneNestedSqmmaOperation(
        op, mapping, tensorMaterializations, carrierIfResults, rewriter);
    if (!newOp)
      return false;
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp->getResults()))
      oldResult.replaceAllUsesWith(newResult);
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
    if (externalFinalWait) {
      newInputs.reserve(oldWait.getInputs().size());
      for (Value input : oldWait.getInputs()) {
        Value newInput = input;
        if (auto result = dyn_cast<OpResult>(input)) {
          if (result.getOwner() == forOp.getOperation()) {
            newInput = newFor.getResult(result.getResultNumber());
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
    // cloned wait as well when a loop mixes direct-yield and explicit-wait
    // accumulators, then map the appended results for the final unpack.
    for (const Candidate &candidate : candidates)
      if (candidate.directYield)
        newInputs.push_back(newFor.getResult(candidate.iterArgIdx));

    Location waitLoc = externalFinalWait ? oldWait.getLoc() : forOp.getLoc();
    auto newWait = triton::mtgpu::SqmmaWaitOp::create(rewriter, waitLoc,
                                                       newInputs);
    if (externalFinalWait)
      newWait->setAttrs(oldWait->getAttrs());

    unsigned directWaitResultIdx = externalFinalWait
                                       ? oldWait.getNumResults()
                                       : 0;
    for (const Candidate &candidate : candidates)
      if (candidate.directYield)
        waitedDirectResults[candidate.iterArgIdx] =
            newWait->getResult(directWaitResultIdx++);

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
      auto tensorTy = dyn_cast<RankedTensorType>(forOp.getResult(idx).getType());
      if (!tensorTy) {
        // Keep the source loop intact if a preceding transformation changed
        // its result to a native carrier between candidate discovery and the
        // replacement phase.  This is a conservative escape hatch for mixed
        // fixed-point pipelines; no partially cloned loop is exposed.
        rewriter.eraseOp(newFor);
        return false;
      }
      result = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
          rewriter, loc, tensorTy, result);
    }
    replacements.push_back(result);
  }

  // Replace the loop's externally visible results, then drop references in
  // the dead source regions before erasing the parent.  Nested async dots may
  // be used by their old SCF yields; cross-region RAUW is invalid here, so
  // clear those internal links explicitly after the cloned loop is complete.
  for (auto [oldResult, replacement] : llvm::zip(forOp.getResults(), replacements))
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
