#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#ifdef __TLE__
#include "Dialect/MUSATLE/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#endif
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

#ifdef __TLE__
struct MusaTleEncodingInfo {
  Attribute encoding;
  // A varying hint may be replaced by a hard hint. Hard hints represent an
  // explicit set_layout result and must not be silently changed.
  bool mayVary = false;

  explicit operator bool() const { return bool(encoding); }
};

static bool musaTleEncodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp, triton::CatOp,
             triton::TransOp>(op);
}

static LogicalResult mergeMusaTleEncodingInfo(MusaTleEncodingInfo oldInfo,
                                              MusaTleEncodingInfo newInfo,
                                              Operation *op,
                                              MusaTleEncodingInfo &merged) {
  if (!oldInfo) {
    merged = newInfo;
    return success();
  }
  if (!newInfo) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.encoding == newInfo.encoding) {
    merged = oldInfo;
    merged.mayVary = oldInfo.mayVary && newInfo.mayVary;
    return success();
  }
  if (oldInfo.mayVary && !newInfo.mayVary) {
    merged = newInfo;
    return success();
  }
  if (!oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }

  op->emitOpError("found conflicting MUSA TLE encoding hints for value:\n  ")
      << oldInfo.encoding << "\nand\n  " << newInfo.encoding;
  return failure();
}

using MusaTleEncodingMap = llvm::MapVector<Value, MusaTleEncodingInfo>;
using MusaTleEncodingWorklist = llvm::PriorityWorklist<Value>;

struct MusaTleLayoutDomainMember {
  Attribute encoding;
  Operation *op;
};

struct MusaTleLayoutProducerFamily {
  SmallVector<Value> originalOperands;
  SmallVector<MusaTleLayoutDomainMember> members;
};

using MusaTleLayoutProducerFamilies =
    llvm::MapVector<Operation *, MusaTleLayoutProducerFamily>;

static bool isMusaTleLayoutPolymorphicProducer(Operation *op) {
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
      op->getNumResults() != 1 || !isPure(op) ||
      op->hasTrait<OpTrait::IsTerminator>() ||
      isa<mlir::triton::musa_tle::SetLayoutOp, triton::CallOp>(op) ||
      getMemAccessPtr(op))
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return resultType && !resultType.getEncoding();
}

static Value
materializeMusaTleLayoutDomain(Value value, Attribute encoding,
                               MusaTleLayoutProducerFamilies &families) {
  Operation *producer = value.getDefiningOp();
  if (!producer || !isMusaTleLayoutPolymorphicProducer(producer))
    return value;

  auto [it, inserted] = families.insert(
      {producer, MusaTleLayoutProducerFamily{
                     llvm::to_vector(producer->getOperands()), {}}});
  MusaTleLayoutProducerFamily &family = it->second;
  for (const MusaTleLayoutDomainMember &member : family.members)
    if (member.encoding == encoding)
      return member.op->getResult(0);

  Operation *member = producer;
  if (!family.members.empty()) {
    OpBuilder builder(producer);
    builder.setInsertionPointAfter(family.members.back().op);
    member = builder.clone(*producer);
    member->setOperands(family.originalOperands);
  }
  family.members.push_back({encoding, member});
  SmallVector<Value> originalOperands = family.originalOperands;

  Attribute operandEncoding = inferSrcEncoding(producer, encoding);
  if (!operandEncoding)
    return member->getResult(0);

  for (auto [index, operand] : llvm::enumerate(originalOperands)) {
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    member->setOperand(index, materializeMusaTleLayoutDomain(
                                  operand, operandEncoding, families));
  }
  return member->getResult(0);
}

static void mergeMusaTleLayoutProducerDomains(
    const MusaTleLayoutProducerFamilies &families) {
  for (const auto &[_, family] : families) {
    llvm::MapVector<Type, Operation *> representatives;
    for (const MusaTleLayoutDomainMember &domain : family.members) {
      Operation *member = domain.op;
      Type resultType = member->getResult(0).getType();
      auto [it, inserted] = representatives.insert({resultType, member});
      if (inserted)
        continue;
      member->getResult(0).replaceAllUsesWith(it->second->getResult(0));
      member->erase();
    }
  }
}

static LogicalResult updateMusaTleEncoding(ArrayRef<Value> values,
                                           MusaTleEncodingInfo info,
                                           FuncOp func,
                                           MusaTleEncodingMap &valueToEncoding,
                                           MusaTleEncodingWorklist &worklist) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;

    auto [it, inserted] = valueToEncoding.insert({value, info});
    if (!inserted) {
      Operation *diagOp = value.getDefiningOp();
      if (!diagOp)
        if (auto blockArg = dyn_cast<BlockArgument>(value))
          diagOp = blockArg.getOwner()->getParentOp();
      if (!diagOp)
        diagOp = func.getOperation();
      MusaTleEncodingInfo merged;
      if (failed(mergeMusaTleEncodingInfo(it->second, info, diagOp, merged)))
        return failure();
      if (merged.encoding == it->second.encoding &&
          merged.mayVary == it->second.mayVary)
        continue;
      it->second = merged;
    }
    worklist.insert(value);
  }
  return success();
}

static LogicalResult propagateMusaTleEncodingHints(
    FuncOp func, const MusaTleEncodingMap &boundarySeeds, bool &changed) {
  SmallVector<mlir::triton::musa_tle::SetLayoutOp> setLayoutOps;
  func.walk([&](mlir::triton::musa_tle::SetLayoutOp op) {
    setLayoutOps.push_back(op);
  });
  if (setLayoutOps.empty() && boundarySeeds.empty())
    return success();

  MusaTleLayoutProducerFamilies producerFamilies;
  for (mlir::triton::musa_tle::SetLayoutOp op : setLayoutOps) {
    Value source = materializeMusaTleLayoutDomain(
        op.getSrc(), op.getTargetEncoding(), producerFamilies);
    op->setOperand(0, source);
  }

  SmallVector<std::pair<Value, MusaTleEncodingInfo>> seedEncodings;
  for (mlir::triton::musa_tle::SetLayoutOp op : setLayoutOps) {
    // The source is allowed to serve multiple explicit layout domains. The
    // result is the hard user contract introduced by set_layout.
    seedEncodings.push_back(
        {op.getSrc(), MusaTleEncodingInfo{op.getTargetEncoding(), true}});
    seedEncodings.push_back(
        {op.getResult(), MusaTleEncodingInfo{op.getTargetEncoding(), false}});
  }
  for (const auto &[value, info] : boundarySeeds)
    seedEncodings.push_back({value, info});

  MusaTleEncodingMap valueToEncoding;
  MusaTleEncodingWorklist worklist;
  for (auto &[value, info] : seedEncodings) {
    if (failed(updateMusaTleEncoding({value}, info, func, valueToEncoding,
                                     worklist)))
      return failure();
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    MusaTleEncodingInfo info = valueToEncoding[value];
    assert(info && "worklist value must have a MUSA TLE encoding");

    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();

      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        int offset = 3 * isa<scf::ForOp>(op);
        int tiedIndex = static_cast<int>(use.getOperandNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(op, tiedIndex);
        if (failed(updateMusaTleEncoding(tiedArgs, info, func, valueToEncoding,
                                         worklist)))
          return failure();
        continue;
      }

      if (isa<scf::YieldOp, scf::ConditionOp>(op)) {
        Operation *parentOp = op->getParentOp();
        if (!isa_and_nonnull<scf::ForOp, scf::WhileOp, scf::IfOp>(parentOp))
          continue;
        int offset = isa<scf::ConditionOp>(op);
        int tiedIndex = static_cast<int>(use.getOperandNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(parentOp, tiedIndex);
        if (failed(updateMusaTleEncoding(tiedArgs, info, func, valueToEncoding,
                                         worklist)))
          return failure();
        continue;
      }

      if (auto wait = dyn_cast<mlir::triton::musa_tle::SqmmaWaitOp>(op)) {
        if (failed(updateMusaTleEncoding({wait.getOutput()}, info, func,
                                         valueToEncoding, worklist)))
          return failure();
        continue;
      }

      if (auto sqmma = dyn_cast<mlir::triton::musa_tle::SqmmaOp>(op)) {
        if (use.get() == sqmma.getC() &&
            failed(updateMusaTleEncoding({sqmma.getD()}, info, func,
                                         valueToEncoding, worklist)))
          return failure();
        continue;
      }

      if (auto dot = dyn_cast<triton::DotOpInterface>(op)) {
        // A/B use dot-operand encodings, while C and D use the parent MMA
        // encoding. Only C and D are layout-equivalent across a dot.
        if (use.getOperandNumber() == 2 &&
            failed(updateMusaTleEncoding({dot.getD()}, info, func,
                                         valueToEncoding, worklist)))
          return failure();
        continue;
      }

      if (isa<mlir::triton::musa_tle::SetLayoutOp>(op))
        continue;

      if (isa<mlir::triton::musa_tle::LocalPointersOp>(op)) {
        if (failed(updateMusaTleEncoding(
                llvm::to_vector_of<Value>(op->getResults()), info, func,
                valueToEncoding, worklist)))
          return failure();
        continue;
      }

      Attribute dstEncoding = inferDstEncoding(op, info.encoding);
      if (!dstEncoding)
        continue;
      MusaTleEncodingInfo dstInfo{dstEncoding,
                                  info.mayVary || musaTleEncodingsMayVary(op)};
      if (failed(
              updateMusaTleEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                    dstInfo, func, valueToEncoding, worklist)))
        return failure();
    }

    if (auto opResult = dyn_cast<OpResult>(value)) {
      Operation *definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateMusaTleEncoding(tiedArgs, info, func, valueToEncoding,
                                         worklist)))
          return failure();
      } else if (auto wait = dyn_cast<mlir::triton::musa_tle::SqmmaWaitOp>(
                     definingOp)) {
        if (failed(updateMusaTleEncoding({wait.getInput()}, info, func,
                                         valueToEncoding, worklist)))
          return failure();
      } else if (auto sqmma =
                     dyn_cast<mlir::triton::musa_tle::SqmmaOp>(definingOp)) {
        if (failed(updateMusaTleEncoding({sqmma.getC()}, info, func,
                                         valueToEncoding, worklist)))
          return failure();
      } else if (isa<triton::DotOpInterface>(definingOp)) {
        if (failed(updateMusaTleEncoding({definingOp->getOperand(2)}, info,
                                         func, valueToEncoding, worklist)))
          return failure();
      } else if (auto localPointers =
                     dyn_cast<mlir::triton::musa_tle::LocalPointersOp>(
                         definingOp)) {
        SmallVector<Value> tensorIndices;
        for (Value index : localPointers.getIndices())
          if (isa<RankedTensorType>(index.getType()))
            tensorIndices.push_back(index);
        if (failed(updateMusaTleEncoding(tensorIndices, info, func,
                                         valueToEncoding, worklist)))
          return failure();
      } else if (!isa<mlir::triton::musa_tle::SetLayoutOp>(definingOp)) {
        Attribute srcEncoding = inferSrcEncoding(definingOp, info.encoding);
        if (srcEncoding) {
          MusaTleEncodingInfo srcInfo{
              srcEncoding, info.mayVary || musaTleEncodingsMayVary(definingOp)};
          SmallVector<Value> tensorOperands;
          for (Value operand : definingOp->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              tensorOperands.push_back(operand);
          if (failed(updateMusaTleEncoding(tensorOperands, srcInfo, func,
                                           valueToEncoding, worklist)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        int offset = isa<scf::ForOp>(parentOp);
        int tiedIndex = static_cast<int>(blockArg.getArgNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(parentOp, tiedIndex);
        if (failed(updateMusaTleEncoding(tiedArgs, info, func, valueToEncoding,
                                         worklist)))
          return failure();
      }
    }
  }

  for (auto &[value, info] : valueToEncoding) {
    auto existingType = cast<RankedTensorType>(value.getType());
    if (existingType.getEncoding() != info.encoding) {
      auto newType = existingType.cloneWithEncoding(info.encoding);
      value.setType(newType);
      changed = true;
      if (auto opResult = dyn_cast<OpResult>(value)) {
        if (auto constant = dyn_cast<arith::ConstantOp>(opResult.getOwner())) {
          if (auto elements =
                  dyn_cast<DenseElementsAttr>(constant.getValueAttr()))
            constant.setValueAttr(elements.reshape(newType));
        }
      }
    }

    if (auto opResult = dyn_cast<OpResult>(value))
      setTleExplicitResultEncoding(opResult, info.encoding);
  }

  mergeMusaTleLayoutProducerDomains(producerFamilies);

  WalkResult memoryWalk = func.walk([&](Operation *op) {
    if (!getMemAccessPtr(op))
      return WalkResult::advance();

    Attribute explicitEncoding;
    if (failed(inferTleExplicitMemoryEncoding(op, explicitEncoding)))
      return WalkResult::interrupt();
    if (explicitEncoding)
      setTleExplicitMemoryEncoding(op, explicitEncoding);
    return WalkResult::advance();
  });
  if (memoryWalk.wasInterrupted())
    return failure();

  return success();
}

using MusaTleFunctionSeeds = llvm::MapVector<Operation *, MusaTleEncodingMap>;

struct MusaTleCallSite {
  triton::CallOp call;
  FuncOp caller;
  FuncOp callee;
};

static LogicalResult addMusaTleBoundarySeed(MusaTleFunctionSeeds &seeds,
                                            FuncOp func, Value value,
                                            Attribute encoding) {
  if (!encoding || !isa<RankedTensorType>(value.getType()))
    return success();

  auto [funcIt, _] = seeds.insert({func.getOperation(), MusaTleEncodingMap{}});
  MusaTleEncodingMap &funcSeeds = funcIt->second;
  MusaTleEncodingInfo info{encoding, false};
  auto [valueIt, inserted] = funcSeeds.insert({value, info});
  if (inserted)
    return success();

  MusaTleEncodingInfo merged;
  if (failed(mergeMusaTleEncodingInfo(valueIt->second, info,
                                      func.getOperation(), merged)))
    return failure();
  valueIt->second = merged;
  return success();
}

static bool hasSameMusaTleAbiTensorType(RankedTensorType lhs,
                                        RankedTensorType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType();
}

static LogicalResult
resolveMusaTleAbiSlot(FuncOp callee, StringRef slotKind, unsigned slotIndex,
                      Type signatureType,
                      ArrayRef<std::pair<FuncOp, Value>> members,
                      MusaTleFunctionSeeds &nextSeeds) {
  auto signatureTensor = dyn_cast<RankedTensorType>(signatureType);
  if (!signatureTensor) {
    for (const auto &[_, value] : members) {
      if (value.getType() != signatureType)
        return callee.emitOpError()
               << "found incompatible MUSA TLE " << slotKind << " #"
               << slotIndex << " ABI type: expected " << signatureType
               << " but found " << value.getType();
    }
    return success();
  }

  Attribute encoding = signatureTensor.getEncoding();
  for (const auto &[_, value] : members) {
    auto valueType = dyn_cast<RankedTensorType>(value.getType());
    if (!valueType || !hasSameMusaTleAbiTensorType(signatureTensor, valueType))
      return callee.emitOpError()
             << "found incompatible MUSA TLE " << slotKind << " #" << slotIndex
             << " ABI type: expected tensor shape and element "
                "type "
             << signatureTensor << " but found " << value.getType();

    Attribute valueEncoding = valueType.getEncoding();
    if (!valueEncoding)
      continue;
    if (encoding && encoding != valueEncoding)
      return callee.emitOpError()
             << "found conflicting MUSA TLE ABI encodings for " << slotKind
             << " #" << slotIndex << ":\n  " << encoding << "\nand\n  "
             << valueEncoding;
    encoding = valueEncoding;
  }

  if (!encoding)
    return success();
  for (const auto &[func, value] : members) {
    auto valueType = cast<RankedTensorType>(value.getType());
    if (!valueType.getEncoding() &&
        failed(addMusaTleBoundarySeed(nextSeeds, func, value, encoding)))
      return failure();
  }
  return success();
}

static LogicalResult
collectMusaTleCallSites(ModuleOp mod,
                        SmallVectorImpl<MusaTleCallSite> &callSites) {
  for (FuncOp caller : mod.getOps<FuncOp>()) {
    WalkResult result = caller.walk([&](triton::CallOp call) {
      FuncOp callee = mod.lookupSymbol<FuncOp>(call.getCallee());
      if (!callee) {
        call.emitOpError("could not resolve callee while propagating MUSA "
                         "TLE ABI encodings");
        return WalkResult::interrupt();
      }
      callSites.push_back({call, caller, callee});
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }
  return success();
}

static SmallVector<ReturnOp> collectMusaTleReturns(FuncOp func) {
  SmallVector<ReturnOp> returns;
  func.walk([&](ReturnOp op) { returns.push_back(op); });
  return returns;
}

static LogicalResult
resolveMusaTleAbiBoundaries(ArrayRef<FuncOp> funcs,
                            ArrayRef<MusaTleCallSite> callSites,
                            MusaTleFunctionSeeds &nextSeeds) {
  for (FuncOp callee : funcs) {
    SmallVector<MusaTleCallSite> calleeCalls;
    for (const MusaTleCallSite &callSite : callSites)
      if (callSite.callee == callee)
        calleeCalls.push_back(callSite);

    for (unsigned index = 0; index < callee.getNumArguments(); ++index) {
      SmallVector<std::pair<FuncOp, Value>> members;
      if (!callee.isExternal())
        members.push_back({callee, callee.getArgument(index)});
      for (MusaTleCallSite callSite : calleeCalls) {
        if (index >= callSite.call.getNumOperands())
          return callSite.call.emitOpError()
                 << "has too few operands for MUSA TLE ABI argument #" << index;
        members.push_back({callSite.caller, callSite.call.getOperand(index)});
      }
      if (failed(resolveMusaTleAbiSlot(callee, "argument", index,
                                       callee.getArgumentTypes()[index],
                                       members, nextSeeds)))
        return failure();
    }

    SmallVector<ReturnOp> returns = collectMusaTleReturns(callee);
    for (ReturnOp returnOp : returns)
      if (returnOp.getNumOperands() != callee.getNumResults())
        return returnOp.emitOpError()
               << "has a result count inconsistent with function @"
               << callee.getName();

    for (unsigned index = 0; index < callee.getNumResults(); ++index) {
      SmallVector<std::pair<FuncOp, Value>> members;
      for (ReturnOp returnOp : returns)
        members.push_back({callee, returnOp.getOperand(index)});
      for (MusaTleCallSite callSite : calleeCalls) {
        if (index >= callSite.call.getNumResults())
          return callSite.call.emitOpError()
                 << "has too few results for MUSA TLE ABI result #" << index;
        members.push_back({callSite.caller, callSite.call.getResult(index)});
      }
      if (failed(resolveMusaTleAbiSlot(callee, "result", index,
                                       callee.getResultTypes()[index], members,
                                       nextSeeds)))
        return failure();
    }
  }
  return success();
}

static LogicalResult
synchronizeMusaTleFunctionTypes(ArrayRef<FuncOp> funcs,
                                ArrayRef<MusaTleCallSite> callSites) {
  for (FuncOp func : funcs) {
    SmallVector<Type> argumentTypes(func.getArgumentTypes());
    SmallVector<Type> resultTypes(func.getResultTypes());
    if (!func.isExternal()) {
      argumentTypes.assign(func.getBlocks().front().getArgumentTypes().begin(),
                           func.getBlocks().front().getArgumentTypes().end());
      SmallVector<ReturnOp> returns = collectMusaTleReturns(func);
      if (!returns.empty()) {
        resultTypes.assign(returns.front().getOperandTypes().begin(),
                           returns.front().getOperandTypes().end());
        for (ReturnOp returnOp : llvm::drop_begin(returns))
          if (!llvm::equal(returnOp.getOperandTypes(), resultTypes))
            return returnOp.emitOpError()
                   << "does not match the converged MUSA TLE result ABI of "
                      "function @"
                   << func.getName();
      }
    }
    func.setFunctionType(
        FunctionType::get(func.getContext(), argumentTypes, resultTypes));
  }

  for (MusaTleCallSite callSite : callSites) {
    FunctionType calleeType = callSite.callee.getFunctionType();
    if (!llvm::equal(callSite.call.getOperandTypes(), calleeType.getInputs()))
      return callSite.call.emitOpError()
             << "operands do not match the converged MUSA TLE ABI for @"
             << callSite.callee.getName();
    if (callSite.call.getNumResults() != calleeType.getNumResults())
      return callSite.call.emitOpError()
             << "result count does not match the converged MUSA TLE ABI for @"
             << callSite.callee.getName();
    for (auto [result, type] :
         llvm::zip(callSite.call.getResults(), calleeType.getResults()))
      result.setType(type);
  }
  return success();
}

static LogicalResult applyMusaTleEncodingHints(ModuleOp mod) {
  bool hasSetLayout = false;
  mod.walk([&](mlir::triton::musa_tle::SetLayoutOp) {
    hasSetLayout = true;
    return WalkResult::interrupt();
  });
  if (!hasSetLayout)
    return success();

  SmallVector<FuncOp> funcs(llvm::to_vector(mod.getOps<FuncOp>()));
  SmallVector<MusaTleCallSite> callSites;
  if (failed(collectMusaTleCallSites(mod, callSites)))
    return failure();

  unsigned tensorValueCount = 0;
  mod.walk([&](Operation *op) {
    tensorValueCount += llvm::count_if(op->getResults(), [](Value value) {
      return isa<RankedTensorType>(value.getType());
    });
    for (Region &region : op->getRegions())
      for (Block &block : region)
        tensorValueCount +=
            llvm::count_if(block.getArguments(), [](BlockArgument argument) {
              return isa<RankedTensorType>(argument.getType());
            });
  });

  MusaTleFunctionSeeds boundarySeeds;
  for (unsigned iteration = 0; iteration <= tensorValueCount + 1; ++iteration) {
    bool changed = false;
    for (FuncOp func : funcs) {
      auto it = boundarySeeds.find(func.getOperation());
      MusaTleEncodingMap emptySeeds;
      const MusaTleEncodingMap &funcSeeds =
          it == boundarySeeds.end() ? emptySeeds : it->second;
      if (failed(propagateMusaTleEncodingHints(func, funcSeeds, changed)))
        return failure();
    }

    MusaTleFunctionSeeds nextSeeds;
    if (failed(resolveMusaTleAbiBoundaries(funcs, callSites, nextSeeds)))
      return failure();
    if (nextSeeds.empty())
      return synchronizeMusaTleFunctionTypes(funcs, callSites);
    if (!boundarySeeds.empty() && !changed) {
      mod.emitError("MUSA TLE function ABI propagation did not make "
                    "progress");
      return failure();
    }
    boundarySeeds = std::move(nextSeeds);
  }

  mod.emitError("MUSA TLE function ABI propagation did not converge");
  return failure();
}
#endif // __TLE__

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      // This is a hack. We just want to add encoding.
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::gpu::BlockedEncodingAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    auto newRank = retShape.size();
    // return encoding
    auto retSizePerThread = llvm::to_vector(argEncoding.getSizePerThread());
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = to_vector(argEncoding.getThreadsPerWarp());
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = to_vector(argEncoding.getWarpsPerCTA());
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto ctaLl = argEncoding.getCGALayout().getLinearLayout();
    auto kBlock = *ctaLl.getInDimNames().begin();
    auto *ctx = kBlock.getContext();
    auto newDim = standardOutDimNames(ctx, newRank)[newRank - 1];
    ctaLl *= LinearLayout::identity1D(1, kBlock, newDim);
    // Move last dim to op.getAxis(). nb is this a std::rotate?
    auto newOrder = to_vector(llvm::seq<int32_t>(newRank));
    for (int i = newRank - 1; i >= op.getAxis() + 1; --i) {
      std::swap(newOrder[i], newOrder[i - 1]);
    }
    ctaLl = transposeLinearLayout(ctaLl, newOrder);
    auto retCGALayout = CGAEncodingAttr::get(ctx, std::move(ctaLl));
    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCGALayout);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = argType.cloneWithEncoding(newArgEncoding);
    // construct new op
    auto newSrc = triton::gpu::ConvertLayoutOp::create(
        rewriter, op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();
    auto rank = origShape.size();
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    retSizePerThread[rank - 1] = std::min(
        retSizePerThread[rank - 1], static_cast<unsigned>(origShape[rank - 1]));
    retSizePerThread[rank - 2] = std::min(
        retSizePerThread[rank - 2], static_cast<unsigned>(origShape[rank - 2]));

    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i)
      retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType = origType.cloneWithEncoding(dEncoding);
    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType = aType.cloneWithEncoding(encoding);
      a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstType,
                                               a);
    }
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType = bType.cloneWithEncoding(encoding);
      b = triton::gpu::ConvertLayoutOp::create(rewriter, b.getLoc(), dstType,
                                               b);
    }
    c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);

#ifdef __TLE__
    Attribute explicitEncoding =
        getTleExplicitResultEncoding(op.getOperation(), 0);
    auto explicitSqmma =
        dyn_cast_or_null<triton::gpu::MUSASqmmaEncodingAttr>(explicitEncoding);
    auto resultSqmma = dyn_cast_or_null<triton::gpu::MUSASqmmaEncodingAttr>(
        origType.getEncoding());
    if (resultSqmma && explicitSqmma != resultSqmma)
      return op.emitOpError(
          "has an SQMMA result encoding without a matching explicit MUSA "
          "TLE layout contract");

    if (explicitSqmma) {
      auto newDot = triton::DotOp::create(rewriter, op.getLoc(), retType, a, b,
                                          c, adaptor.getInputPrecision(),
                                          adaptor.getMaxNumImpreciseAcc());
      addNamedAttrs(newDot, adaptor.getAttributes());
      newDot->removeAttr(getTleExplicitEncodingAttrName(0));
      setTleExplicitSqmmaEncoding(newDot.getOperation(), explicitSqmma);

      auto boundary = triton::gpu::ConvertLayoutOp::create(
          rewriter, op.getLoc(), origType, newDot.getResult());
      setTleExplicitResultEncoding(boundary.getOperation(), 0, explicitSqmma);
      rewriter.replaceOp(op, boundary.getResult());
      return success();
    }
#endif // __TLE__

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The cat op satisfy two conditions:
    // 1. output.numel = lhs.numel + rhs.numel
    // 2. output.total_elems_per_thread =
    // next_power_of_2(lhs.total_elems_per_thread + rhs.total_elems_per_thread)
    // For now, this behaves like generic, but this
    // will evolve when we add support for `can_reorder=False`.
    auto retType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    auto retEncoding =
        cast<triton::gpu::BlockedEncodingAttr>(retType.getEncoding());
    auto lhsType = adaptor.getLhs().getType();
    auto rhsType = adaptor.getRhs().getType();
    auto lhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(retType);
    auto retShape = retType.getShape();
    auto retOrder = retEncoding.getOrder();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    // Get new retSizePerThread if ret elems per thread is not enough.
    // We have to round it up to the next power of 2 due to triton's tensor size
    // constraint.
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = llvm::to_vector(retEncoding.getSizePerThread());
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton::gpu::BlockedEncodingAttr newRetEncoding =
        triton::gpu::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retEncoding.getCGALayout());
    auto newRetType = retType.cloneWithEncoding(newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, newRetType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Simply rely on type inference for this op.  (Notably, GenericOpPattern
    // does not do this, instead it assigns the default layout to the ins and
    // outs.)
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, adaptor.getLhs(), adaptor.getRhs()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    int rank = srcEnc.getOrder().size();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();

    // The operand to split must have:
    //  - a blocked layout, with
    //  - sizePerThread = 2 in the last dimension,
    //  - threadsPerWarp, warpsPerCTA, and CTAsPerCGA = 1 in the last dim, and
    //  - the last dimension minor.
    // If that's not the case, add a convert before the split.
    if (!srcEnc || srcEnc.getSizePerThread().back() != 2 ||
        srcEnc.getOrder().front() != rank - 1) {
      // If we take the default encoding for the op's result (i.e. post-split)
      // and add 1 to the end of each dim, that gives us what we want.  Other
      // than making a legal src encoding, our choice of layout doesn't matter;
      // it'll get fixed by RemoveLayoutConversions.
      auto defaultEnc = getDefaultBlockedEncoding(
          getContext(),
          cast<RankedTensorType>(op.getResult(0).getType()).getShape(),
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
          typeConverter->getNumCTAs());

      auto append = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res(vals);
        res.push_back(val);
        return res;
      };
      auto prepend = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res;
        res.push_back(val);
        res.append(vals.begin(), vals.end());
        return res;
      };

      auto layout = defaultEnc.getCGALayout().getLinearLayout();
      auto kBlock = StringAttr::get(getContext(), "block");
      auto newDim = standardOutDimNames(getContext(), rank)[rank - 1];
      layout *= LinearLayout::identity1D(1, kBlock, newDim);
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CGAEncodingAttr::get(getContext(), std::move(layout)));
      srcTy = srcTy.cloneWithEncoding(srcEnc);
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), srcTy, src);
    }

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = srcTy.getEncoding();
    if (!srcEnc)
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = op.getType().cloneWithEncoding(srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = triton::ReduceOp::create(
        rewriter, op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan =
        triton::ScanOp::create(rewriter, op.getLoc(), adaptor.getOperands(),
                               adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

struct TritonMapElementwisePattern
    : public OpConversionPattern<triton::MapElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    SmallVector<Type> resultTys;
    auto err = converter->convertTypes(op.getResults().getType(), resultTys);
    if (failed(err)) {
      return err;
    }

    auto newMapOp = triton::MapElementwiseOp::create(
        rewriter, op.getLoc(), resultTys, adaptor.getOperands(), op.getPack());
    addNamedAttrs(newMapOp, adaptor.getAttributes());

    auto &newScalarOp = newMapOp.getScalarOp();
    rewriter.cloneRegionBefore(op.getScalarOp(), newScalarOp,
                               newScalarOp.end());
    rewriter.replaceOp(op, newMapOp.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    // Convert just the entry block. The remaining unstructured control flow is
    // converted by br patterns.
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns, unsigned numCTAs) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::UnsplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>,
      TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>,
#ifdef __TLE__
      GenericOpPattern<triton::gpu::LocalAllocOp>,
#endif
      TritonExpandDimsPattern,
      TritonTransPattern,
      TritonDotPattern,
      TritonMapElementwisePattern,
      GatherScatterOpPattern<DescriptorGatherOp>,
      GatherScatterOpPattern<DescriptorScatterOp>,
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::DescriptorReduceOp>,
      // this assumes the right layout will be set later for dot scaled.
      GenericOpPattern<triton::DotScaledOp>,
      GenericOpPattern<triton::CallOp>,
      GenericOpPattern<ReturnOp>,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
}
//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = scf::WhileOp::create(rewriter, op.getLoc(), newResultTypes,
                                      adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}

#ifdef __TLE__
struct MUSATLEExtractTilePattern
    : public OpConversionPattern<mlir::triton::musa_tle::ExtractTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::musa_tle::ExtractTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding())
      return failure();

    SmallVector<int64_t> tileShape;
    llvm::append_range(tileShape, op.getTileShape());
    auto resultTy = RankedTensorType::get(tileShape, srcTy.getElementType(),
                                          srcTy.getEncoding());
    OperationState state(
        op.getLoc(), mlir::triton::musa_tle::ExtractTileOp::getOperationName());
    state.addOperands({adaptor.getSrc(), adaptor.getIndex()});
    state.addAttributes(op->getAttrs());
    state.addTypes(resultTy);
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct MUSATLEInsertTilePattern
    : public OpConversionPattern<mlir::triton::musa_tle::InsertTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::musa_tle::InsertTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcTy = dyn_cast<RankedTensorType>(adaptor.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding())
      return failure();

    OperationState state(
        op.getLoc(), mlir::triton::musa_tle::InsertTileOp::getOperationName());
    state.addOperands(
        {adaptor.getSrc(), adaptor.getTile(), adaptor.getIndex()});
    state.addAttributes(op->getAttrs());
    state.addTypes(srcTy);
    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

struct MUSATLESetLayoutPattern
    : public OpConversionPattern<mlir::triton::musa_tle::SetLayoutOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::triton::musa_tle::SetLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getResult());
    auto resultType = dyn_cast<RankedTensorType>(convertedType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    Value src = adaptor.getSrc();
    if (src.getType() == resultType) {
      if (auto srcResult = dyn_cast<OpResult>(src))
        setTleExplicitResultEncoding(srcResult, op.getTargetEncoding());
      rewriter.replaceOp(op, src);
      return success();
    }

    auto convert = rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, resultType, src);
    setTleExplicitResultEncoding(convert.getOperation(), 0,
                                 op.getTargetEncoding());
    return success();
  }
};

void populateMUSATlePatterns(TritonGPUTypeConverter &typeConverter,
                             RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<MUSATLESetLayoutPattern,
               GenericOpPattern<mlir::triton::musa_tle::LocalPointersOp>,
               GenericOpPattern<mlir::triton::musa_tle::ExclusiveCumsumOp>,
               GenericOpPattern<mlir::triton::musa_tle::SqmmaOp>,
               GenericOpPattern<mlir::triton::musa_tle::SqmmaWaitOp>,
               MUSATLEExtractTilePattern, MUSATLEInsertTilePattern>(
      typeConverter, context);
}
#endif // __TLE__

class ConvertTritonToTritonGPU
    : public triton::impl::ConvertTritonToTritonGPUBase<
          ConvertTritonToTritonGPU> {
public:
  using ConvertTritonToTritonGPUBase::ConvertTritonToTritonGPUBase;

  void runOnOperation() override {
    if (target.getValue().empty()) {
      mlir::emitError(
          getOperation().getLoc(),
          "'convert-triton-to-tritongpu' requires 'target' option to be set");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, enableSourceRemat);
    TritonGPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
#ifdef __TLE__
    populateMUSATlePatterns(typeConverter, patterns);
#endif
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

#ifdef __TLE__
    if (failed(applyMusaTleEncodingHints(mod)))
      return signalPassFailure();
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
#else
    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
#endif // __TLE__
  }
};

} // namespace
