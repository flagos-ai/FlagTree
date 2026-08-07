#ifndef TRITONMUSA_COMMON_SQMMA_OPERAND_PLAN_H
#define TRITONMUSA_COMMON_SQMMA_OPERAND_PLAN_H

#include "TritonMUSACommon/MMAOperandUtils.h"
#include "TritonMUSACommon/MemDescUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir::triton::musa {
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

enum class SqmmaTensorViewKind {
  Trans,
  Reshape,
  ConvertLayout,
  UnrealizedCast,
};

struct SqmmaTensorViewStep {
  SqmmaTensorViewKind kind;
  Operation *op = nullptr;
};

struct SqmmaOperandPlan {
  tt::LoadOp loadOp;
  ttg::LocalAllocOp oldLocalAlloc;
  SmallVector<SqmmaTensorViewStep> tensorViewChain;
  RecoveredSqmmaConsumerContract contract;
  ttg::MemDescType finalViewTy;
  ttg::MemDescType sourceMemDescTy;
};

struct SqmmaLandingGroup {
  tt::LoadOp loadOp;
  ttg::MemDescType sourceMemDescTy;
  SmallVector<SqmmaOperandPlan> plans;
};

inline bool isSqmmaTensorViewBridgeOp(Operation *op) {
  return isa<tt::TransOp, tt::ReshapeOp, ttg::ConvertLayoutOp,
             UnrealizedConversionCastOp>(op);
}

inline bool areSqmmaTensorBridgeLayoutsEquivalent(Operation *op) {
  if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto dstTy = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!srcTy || !dstTy || srcTy.getShape() != dstTy.getShape() ||
      srcTy.getElementType() != dstTy.getElementType())
    return false;

  Attribute srcEncoding = srcTy.getEncoding();
  Attribute dstEncoding = dstTy.getEncoding();
  if (srcEncoding == dstEncoding)
    return true;
  auto srcLayout = dyn_cast_or_null<ttg::LayoutEncodingTrait>(srcEncoding);
  auto dstLayout = dyn_cast_or_null<ttg::LayoutEncodingTrait>(dstEncoding);
  return srcLayout && dstLayout &&
         ttg::areLayoutsEquivalent(srcTy.getShape(), srcLayout, dstLayout);
}

inline bool isSqmmaNoOpTensorBridgeForMemDesc(Operation *op,
                                              ttg::MemDescType memDescTy,
                                              bool memDescMatchesResult) {
  if (!memDescTy || !areSqmmaTensorBridgeLayoutsEquivalent(op))
    return false;
  auto tensorTy = cast<RankedTensorType>(
      (memDescMatchesResult ? op->getResult(0) : op->getOperand(0)).getType());
  return memDescTy.getShape() == tensorTy.getShape() &&
         memDescTy.getElementType() == tensorTy.getElementType();
}

inline FailureOr<ttg::MemDescType>
inferSqmmaOperandSourceMemDescType(ArrayRef<SqmmaTensorViewStep> chain,
                                   ttg::MemDescType finalViewTy) {
  ttg::MemDescType currentTy = finalViewTy;
  for (const SqmmaTensorViewStep &step : llvm::reverse(chain)) {
    switch (step.kind) {
    case SqmmaTensorViewKind::Trans: {
      auto transOp = cast<tt::TransOp>(step.op);
      currentTy = inferTransformedMemDescSourceType(transOp, currentTy);
      break;
    }
    case SqmmaTensorViewKind::Reshape: {
      auto reshapeOp = cast<tt::ReshapeOp>(step.op);
      currentTy = inferReshapedMemDescSourceType(reshapeOp, currentTy);
      break;
    }
    case SqmmaTensorViewKind::ConvertLayout:
    case SqmmaTensorViewKind::UnrealizedCast: {
      if (!isSqmmaNoOpTensorBridgeForMemDesc(step.op, currentTy, true))
        return failure();
      break;
    }
    }
    if (!currentTy)
      return failure();
  }
  return currentTy;
}

inline FailureOr<ttg::MemDescType>
inferSqmmaOperandFinalMemDescType(ArrayRef<SqmmaTensorViewStep> chain,
                                  ttg::MemDescType sourceTy) {
  ttg::MemDescType currentTy = sourceTy;
  for (const SqmmaTensorViewStep &step : chain) {
    switch (step.kind) {
    case SqmmaTensorViewKind::Trans: {
      auto transOp = cast<tt::TransOp>(step.op);
      Attribute dstEncoding;
      Dialect &dialect = currentTy.getEncoding().getDialect();
      auto inferLayoutInterface =
          dyn_cast<tt::DialectInferLayoutInterface>(&dialect);
      if (!inferLayoutInterface ||
          failed(inferLayoutInterface->inferTransOpEncoding(
              currentTy.getEncoding(), currentTy.getShape(), transOp.getOrder(),
              dstEncoding, transOp.getLoc())))
        return failure();
      SmallVector<int64_t> dstAllocShape;
      if (!currentTy.getAllocShape().empty()) {
        dstAllocShape = applyPermutation(
            currentTy.getAllocShape().take_back(transOp.getOrder().size()),
            transOp.getOrder());
        dstAllocShape.insert(
            dstAllocShape.begin(), currentTy.getAllocShape().begin(),
            currentTy.getAllocShape().end() - transOp.getOrder().size());
      }
      currentTy = ttg::MemDescType::get(
          applyPermutation(currentTy.getShape(), transOp.getOrder()),
          currentTy.getElementType(), dstEncoding, currentTy.getMemorySpace(),
          currentTy.getMutableMemory(), dstAllocShape);
      break;
    }
    case SqmmaTensorViewKind::Reshape: {
      auto reshapeOp = cast<tt::ReshapeOp>(step.op);
      auto resultTy = cast<RankedTensorType>(reshapeOp.getResult().getType());
      ttg::MemDescType reshapedTy;
      if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
              reshapeOp.getContext(), reshapeOp.getLoc(), currentTy,
              resultTy.getShape(), reshapedTy)))
        return failure();
      currentTy = reshapedTy;
      break;
    }
    case SqmmaTensorViewKind::ConvertLayout:
    case SqmmaTensorViewKind::UnrealizedCast: {
      if (!isSqmmaNoOpTensorBridgeForMemDesc(step.op, currentTy, false))
        return failure();
      break;
    }
    }
    if (!currentTy)
      return failure();
  }
  return currentTy;
}

inline LogicalResult mergeSqmmaLandingConsumerContract(
    std::optional<RecoveredSqmmaConsumerContract> &contract,
    std::optional<RecoveredSqmmaConsumerContract> candidate) {
  if (!candidate)
    return success();
  if (contract && !(*contract == *candidate))
    return failure();
  contract = *candidate;
  return success();
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverSqmmaLandingConsumerContract(Value memDesc);

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverSqmmaLandingConsumerContractFromTensor(Value tensor) {
  if (!isa<RankedTensorType>(tensor.getType()))
    return failure();

  std::optional<RecoveredSqmmaConsumerContract> contract;
  SmallVector<Value> worklist{tensor};
  llvm::SmallPtrSet<void *, 16> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    if (!isa<RankedTensorType>(current.getType()))
      return failure();

    for (Operation *user : current.getUsers()) {
      if (isSqmmaTensorViewBridgeOp(user)) {
        for (Value result : user->getResults())
          if (isa<RankedTensorType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        auto nestedContract =
            recoverSqmmaLandingConsumerContract(localAlloc.getResult());
        if (failed(nestedContract) || failed(mergeSqmmaLandingConsumerContract(
                                          contract, *nestedContract)))
          return failure();
      }
    }
  }

  return contract;
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverSqmmaLandingConsumerContract(Value memDesc) {
  if (!isa<ttg::MemDescType>(memDesc.getType()))
    return failure();

  std::optional<RecoveredSqmmaConsumerContract> contract;
  SmallVector<Value> worklist{memDesc};
  llvm::SmallPtrSet<void *, 16> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    auto currentTy = dyn_cast<ttg::MemDescType>(current.getType());
    if (!currentTy)
      return failure();

    auto currentContract = recoverSqmmaProducerContractFromMemDesc(current);
    if (failed(currentContract))
      return failure();

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (isa<triton::musa::AsyncTMECopyGlobalToLocalOp,
              triton::musa::AsyncTMECopyLocalToGlobalOp, ttg::LocalDeallocOp>(
              user))
        continue;

      if (isMemDescSqmmaContractBridgeOp(user)) {
        for (Value result : user->getResults())
          if (isa<ttg::MemDescType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (auto waitOp = dyn_cast<triton::musa::SquadDotWaitOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(waitOp.getInputs())) {
          if (operand != current)
            continue;
          Value passthrough = waitOp.getResult(idx);
          if (isa<ttg::MemDescType>(passthrough.getType()))
            worklist.push_back(passthrough);
        }
        continue;
      }
      if (auto waitOp = dyn_cast<triton::mtgpu::SqmmaWaitOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(waitOp.getInputs())) {
          if (operand != current)
            continue;
          Value passthrough = waitOp.getResult(idx);
          if (isa<ttg::MemDescType>(passthrough.getType()))
            worklist.push_back(passthrough);
        }
        continue;
      }

      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(user)) {
        auto tensorContract = recoverSqmmaLandingConsumerContractFromTensor(
            localLoad.getResult());
        if (failed(tensorContract) || failed(mergeSqmmaLandingConsumerContract(
                                          contract, *tensorContract)))
          return failure();
        continue;
      }

      std::optional<RecoveredSqmmaConsumerContract> candidate;
      if (isa<triton::musa::SquadDotOp, triton::mtgpu::SqmmaOp>(user)) {
        unsigned operandIdx = use.getOperandNumber();
        if (operandIdx > 1)
          return failure();
        auto expected =
            getExpectedSqmmaOperandContract(user, operandIdx, currentTy);
        if (failed(expected))
          return failure();
        candidate = *expected;
        if (*currentContract && !(**currentContract == *candidate))
          return failure();
      } else {
        auto userContract = getSqmmaContractFromAnnotatedOp(
            user, inferSharedRowMajor(currentTy));
        if (failed(userContract))
          return failure();
        candidate = *userContract;
      }

      if (failed(mergeSqmmaLandingConsumerContract(contract, candidate)))
        return failure();
    }
  }

  return contract;
}

inline bool hasReachableSqmmaConsumer(Value value) {
  SmallVector<Value> worklist;
  llvm::SmallPtrSet<void *, 16> visited;
  worklist.push_back(value);

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (isa<triton::musa::SquadDotOp, triton::mtgpu::SqmmaOp>(user))
        return true;

      if (isMemDescSqmmaContractBridgeOp(user) ||
          isSqmmaTensorViewBridgeOp(user)) {
        for (Value result : user->getResults())
          if (isa<ttg::MemDescType, RankedTensorType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (auto localLoad = dyn_cast<ttg::LocalLoadOp>(user)) {
        worklist.push_back(localLoad.getResult());
        continue;
      }

      if (auto waitOp = dyn_cast<triton::musa::SquadDotWaitOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(waitOp.getInputs()))
          if (operand == current)
            worklist.push_back(waitOp.getResult(idx));
        continue;
      }
      if (auto waitOp = dyn_cast<triton::mtgpu::SqmmaWaitOp>(user)) {
        for (auto [idx, operand] : llvm::enumerate(waitOp.getInputs()))
          if (operand == current)
            worklist.push_back(waitOp.getResult(idx));
        continue;
      }
    }
  }

  return false;
}

inline FailureOr<std::optional<SqmmaOperandPlan>>
buildSqmmaOperandPlan(tt::LoadOp loadOp, ttg::LocalAllocOp localAlloc,
                      ArrayRef<SqmmaTensorViewStep> chain) {
  auto finalViewTy = dyn_cast<ttg::MemDescType>(localAlloc.getType());
  if (!finalViewTy)
    return failure();

  auto consumerContract =
      recoverSqmmaLandingConsumerContract(localAlloc.getResult());
  if (failed(consumerContract)) {
    localAlloc.emitOpError("requires a unique SQMMA consumer contract");
    return failure();
  }
  if (!*consumerContract) {
    if (!hasSqmmaOpIdxAttr(localAlloc.getOperation()) &&
        !hasReachableSqmmaConsumer(localAlloc.getResult()))
      return std::optional<SqmmaOperandPlan>{};
    localAlloc.emitOpError("requires a unique SQMMA consumer contract");
    return failure();
  }
  const RecoveredSqmmaConsumerContract &consumerContractValue =
      consumerContract->value();

  auto annotatedContract = getSqmmaContractFromAnnotatedOp(
      localAlloc.getOperation(), consumerContractValue.rowMajor);
  if (failed(annotatedContract)) {
    localAlloc.emitOpError("has invalid SQMMA producer attributes");
    return failure();
  }
  if (*annotatedContract && !(**annotatedContract == consumerContractValue)) {
    localAlloc.emitOpError(
        "producer attributes conflict with recovered SQMMA consumer contract");
    return failure();
  }

  auto sourceTy = inferSqmmaOperandSourceMemDescType(chain, finalViewTy);
  if (failed(sourceTy)) {
    localAlloc.emitOpError("cannot infer SQMMA source memdesc type");
    return failure();
  }

  auto loadTy = dyn_cast<RankedTensorType>(loadOp.getType());
  if (!loadTy || sourceTy->getShape() != loadTy.getShape() ||
      sourceTy->getElementType() != loadTy.getElementType() ||
      !isa_and_nonnull<ttg::SharedEncodingTrait>(sourceTy->getEncoding())) {
    localAlloc.emitOpError(
        "inferred SQMMA source memdesc type is incompatible with load result");
    return failure();
  }

  auto simulatedFinalTy = inferSqmmaOperandFinalMemDescType(chain, *sourceTy);
  if (failed(simulatedFinalTy)) {
    localAlloc.emitOpError("cannot simulate final SQMMA memdesc view type");
    return failure();
  }
  if (!areMemDescTypesCompatible(*simulatedFinalTy, finalViewTy) &&
      !areMemDescTypesLayoutEquivalent(*simulatedFinalTy, finalViewTy)) {
    localAlloc.emitOpError(
        "final SQMMA memdesc view type is incompatible with local_alloc type; "
        "simulated type is ")
        << *simulatedFinalTy << ", local_alloc type is " << finalViewTy;
    return failure();
  }

  return std::optional<SqmmaOperandPlan>{SqmmaOperandPlan{
      loadOp, localAlloc,
      SmallVector<SqmmaTensorViewStep>(chain.begin(), chain.end()),
      consumerContractValue, finalViewTy, *sourceTy}};
}

inline FailureOr<SmallVector<SqmmaLandingGroup>>
collectSqmmaOperandLandingGroups(tt::LoadOp loadOp) {
  SmallVector<SqmmaOperandPlan> plans;
  if (loadOp.getOther() && !isZeroConst(loadOp.getOther()))
    return SmallVector<SqmmaLandingGroup>{};

  struct WorkItem {
    Value value;
    SmallVector<SqmmaTensorViewStep> chain;
  };

  SmallVector<WorkItem> worklist;
  llvm::SmallPtrSet<void *, 16> visited;
  worklist.push_back({loadOp.getResult(), {}});

  while (!worklist.empty()) {
    WorkItem item = std::move(worklist.pop_back_val());
    if (!visited.insert(item.value.getAsOpaquePointer()).second)
      continue;

    for (Operation *user : item.value.getUsers()) {
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        if (localAlloc.getSrc() != item.value)
          continue;
        if (!isa<ttg::MemDescType>(localAlloc.getType()))
          continue;
        auto plan = buildSqmmaOperandPlan(loadOp, localAlloc, item.chain);
        if (failed(plan)) {
          localAlloc.emitOpError(
              "unable to recover SQMMA operand landing plan");
          return failure();
        }
        if (*plan)
          plans.push_back(std::move(**plan));
        continue;
      }

      if (!isSqmmaTensorViewBridgeOp(user))
        continue;

      SqmmaTensorViewKind kind;
      if (isa<tt::TransOp>(user))
        kind = SqmmaTensorViewKind::Trans;
      else if (isa<tt::ReshapeOp>(user))
        kind = SqmmaTensorViewKind::Reshape;
      else if (isa<ttg::ConvertLayoutOp>(user))
        kind = SqmmaTensorViewKind::ConvertLayout;
      else
        kind = SqmmaTensorViewKind::UnrealizedCast;

      for (Value result : user->getResults()) {
        if (!isa<RankedTensorType>(result.getType()))
          continue;
        WorkItem next{result, item.chain};
        next.chain.push_back({kind, user});
        worklist.push_back(std::move(next));
      }
    }
  }

  SmallVector<SqmmaLandingGroup> groups;
  for (SqmmaOperandPlan &plan : plans) {
    auto it = llvm::find_if(groups, [&](const SqmmaLandingGroup &group) {
      return areMemDescTypesCompatible(group.sourceMemDescTy,
                                       plan.sourceMemDescTy) ||
             areMemDescTypesLayoutEquivalent(group.sourceMemDescTy,
                                             plan.sourceMemDescTy);
    });
    if (it == groups.end()) {
      groups.push_back(SqmmaLandingGroup{loadOp, plan.sourceMemDescTy, {}});
      it = std::prev(groups.end());
    }
    it->plans.push_back(std::move(plan));
  }

  return groups;
}

inline std::optional<RecoveredSqmmaConsumerContract>
getUniqueSqmmaLandingGroupContract(const SqmmaLandingGroup &group) {
  std::optional<RecoveredSqmmaConsumerContract> contract;
  for (const SqmmaOperandPlan &plan : group.plans) {
    if (!contract) {
      contract = plan.contract;
      continue;
    }
    if (!(*contract == plan.contract))
      return std::nullopt;
  }
  return contract;
}

inline void setSqmmaAttrsFromContract(Operation *op,
                                      RecoveredSqmmaConsumerContract contract) {
  setSqmmaAttrs(op, contract.sqmmaOpIdx, contract.elemBytes, contract.rowMajor);
}

inline FailureOr<Value>
materializeSqmmaOperandPlan(RewriterBase &rewriter,
                            const SqmmaOperandPlan &plan, Value sourceMemDesc) {
  Value current = sourceMemDesc;
  if (current.getType() != plan.sourceMemDescTy) {
    current = adaptMemDescValue(rewriter, plan.oldLocalAlloc->getLoc(), current,
                                plan.sourceMemDescTy, nullptr);
    if (!current)
      return failure();
  }

  for (const SqmmaTensorViewStep &step : plan.tensorViewChain) {
    switch (step.kind) {
    case SqmmaTensorViewKind::Trans:
      current = materializeTransformedMemDesc(
          rewriter, cast<tt::TransOp>(step.op), current, nullptr);
      break;
    case SqmmaTensorViewKind::Reshape:
      current = materializeReshapedMemDesc(
          rewriter, cast<tt::ReshapeOp>(step.op), current, nullptr);
      break;
    case SqmmaTensorViewKind::ConvertLayout:
    case SqmmaTensorViewKind::UnrealizedCast:
      if (!isSqmmaNoOpTensorBridgeForMemDesc(
              step.op, dyn_cast<ttg::MemDescType>(current.getType()), false))
        return failure();
      break;
    }
    if (!current)
      return failure();
  }

  if (current.getType() != plan.finalViewTy) {
    current = adaptMemDescValue(rewriter, plan.oldLocalAlloc->getLoc(), current,
                                plan.finalViewTy, nullptr);
    if (!current)
      return failure();
  }

  if (Operation *defOp = current.getDefiningOp())
    setSqmmaAttrsFromContract(defOp, plan.contract);
  else
    return failure();
  return current;
}

inline LogicalResult replaceSqmmaOperandPlan(RewriterBase &rewriter,
                                             const SqmmaOperandPlan &plan,
                                             Value sourceMemDesc) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(plan.oldLocalAlloc);
  auto replacement = materializeSqmmaOperandPlan(rewriter, plan, sourceMemDesc);
  if (failed(replacement))
    return failure();
  replaceUsesAndPropagateType(rewriter, plan.oldLocalAlloc, *replacement);
  rewriter.eraseOp(plan.oldLocalAlloc);

  for (const SqmmaTensorViewStep &step : llvm::reverse(plan.tensorViewChain)) {
    if (step.op->use_empty())
      rewriter.eraseOp(step.op);
  }
  return success();
}

} // namespace mlir::triton::musa

#endif
