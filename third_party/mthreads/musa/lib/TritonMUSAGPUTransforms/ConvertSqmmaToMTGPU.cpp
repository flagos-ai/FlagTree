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
};

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

static std::optional<Candidate> getCandidateForIterArg(scf::ForOp forOp,
                                                       unsigned iterArgIdx) {
  Value iterArg = forOp.getRegionIterArg(iterArgIdx);
  auto tensorTy = getSqmmaAccumulatorTensorType(iterArg.getType());
  if (!tensorTy)
    return std::nullopt;

  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp || iterArgIdx >= yieldOp.getNumOperands())
    return std::nullopt;

  Value yieldedValue = unwrapYieldedSqmmaValue(yieldOp.getOperand(iterArgIdx));
  auto sqmma = yieldedValue.getDefiningOp<triton::musa::SquadDotOp>();
  if (!sqmma || sqmma.getC() != iterArg)
    return std::nullopt;

  return Candidate{
      iterArgIdx, sqmma,
      triton::mtgpu::SqmmaAccumulatorType::get(forOp.getContext(), tensorTy)};
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
  Value mappedC = lookupOrDefault(op.getC());
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

static bool convertLoopCarriedSqmmaAccumulator(scf::ForOp forOp,
                                               RewriterBase &rewriter) {
  SmallVector<Candidate> candidates;
  for (unsigned idx = 0; idx < forOp.getNumRegionIterArgs(); ++idx) {
    if (auto candidate = getCandidateForIterArg(forOp, idx))
      candidates.push_back(*candidate);
  }
  if (candidates.empty())
    return false;

  llvm::SmallDenseSet<unsigned> candidateIdxs;
  for (const Candidate &candidate : candidates)
    candidateIdxs.insert(candidate.iterArgIdx);
  triton::musa::SquadDotWaitOp externalFinalWait =
      getCanonicalExternalFinalWait(forOp, candidateIdxs);

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
  rewriter.setInsertionPointToStart(&newBody);
  DenseMap<Value, Value> tensorMaterializations;

  for (Operation &op : oldBody.without_terminator()) {
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

    IRMapping opMapping(mapping);
    for (Value operand : op.getOperands()) {
      Value remapped = materializeTensorAccumulatorForUse(
          operand, op.getLoc(), mapping, tensorMaterializations, rewriter);
      if (remapped != mapping.lookupOrDefault(operand))
        opMapping.map(operand, remapped);
    }
    Operation *newOp = rewriter.clone(op, opMapping);
    for (auto [oldResult, newResult] :
         llvm::zip_equal(op.getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
  }

  auto oldYield = cast<scf::YieldOp>(oldBody.getTerminator());
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYield.getNumOperands());
  for (Value operand : oldYield.getOperands())
    newYieldOperands.push_back(mapping.lookupOrDefault(operand));
  scf::YieldOp::create(rewriter, oldYield.getLoc(), newYieldOperands);

  rewriter.setInsertionPointAfter(newFor);
  DenseMap<Value, Value> externalWaitUnpacks;
  if (externalFinalWait) {
    auto oldWait = externalFinalWait;
    SmallVector<Value> newInputs;
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

    auto newWait = triton::mtgpu::SqmmaWaitOp::create(
        rewriter, oldWait.getLoc(), newInputs);
    newWait->setAttrs(oldWait->getAttrs());

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
    rewriter.setInsertionPointAfter(newWait);
  }

  SmallVector<Value> replacements;
  replacements.reserve(forOp.getNumResults());
  for (unsigned idx = 0; idx < forOp.getNumResults(); ++idx) {
    Value result = newFor.getResult(idx);
    if (candidateIdxs.contains(idx)) {
      auto tensorTy = cast<RankedTensorType>(forOp.getResult(idx).getType());
      result = triton::mtgpu::UnpackSqmmaAccumulatorOp::create(
          rewriter, loc, tensorTy, result);
    }
    replacements.push_back(result);
  }
  rewriter.replaceOp(forOp, replacements);
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
