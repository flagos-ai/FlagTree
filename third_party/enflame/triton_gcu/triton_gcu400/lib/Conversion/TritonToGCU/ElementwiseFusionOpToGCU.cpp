/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <algorithm>
#include <functional>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <string>

#include "Analysis/FirstLastUserAnalysis.h"
#include "Conversion/TritonToGCU/Constants.h"
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/MemrefExt/IR/MemrefExt.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "PatternTritonGPUOpToGCU.h"
#include "TritonGCUToGCU/TritionToGCUBase.h"
#include "Utility.h"
#include "Utils/TritonVersionCompat.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;
#if TRITON_VERSION == 35
using mlir::triton::gcu::AccPublishContext;
#endif
namespace {

bool isOffsetContiguousWithin(Operation *op, int64_t granularity) {
  auto attr = op->getAttrOfType<BoolAttr>(kIsContinual);
  if (!attr || !attr.getValue())
    return false;

  Value offset;
  if (auto loadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op))
    offset = loadOp.getOffset();
  else if (auto storeOp = dyn_cast<triton::gcu::MaskedStoreOp>(op))
    offset = storeOp.getOffset();
  else
    return false;

  SetVector<Value> worklist;
  worklist.insert(offset);
  auto isRemNoWrap = [granularity](auto remOp) {
    int64_t divisor = getConstantIntValue(remOp.getRhs()).value_or(-1);
    return divisor > 0 && divisor % granularity == 0;
  };
  for (size_t i = 0; i < worklist.size(); ++i) {
    Value value = worklist[i];
    auto defOp = value.getDefiningOp();
    if (!defOp)
      continue;
    if (auto remOp = dyn_cast<arith::RemSIOp>(defOp)) {
      if (!isRemNoWrap(remOp))
        return false;
      worklist.insert(remOp.getLhs());
      continue;
    }
    if (auto remOp = dyn_cast<arith::RemUIOp>(defOp)) {
      if (!isRemNoWrap(remOp))
        return false;
      worklist.insert(remOp.getLhs());
      continue;
    }
    for (Value operand : defOp->getOperands())
      worklist.insert(operand);
  }
  return true;
}

Value narrowToStoreVector(OpBuilder &builder, Location loc, Value v) {
  auto vecTy = cast<VectorType>(v.getType());
  auto elementTy = vecTy.getElementType();
  unsigned maxVectorLength =
      4 * kOaccSizeInBytes / mlir::triton::gcu::getBpe(elementTy);
  unsigned numElems = vecTy.getNumElements();
  if (numElems <= maxVectorLength)
    return v;
  unsigned numParts = numElems / maxVectorLength;
  SmallVector<Type> partTypes(
      numParts, VectorType::get(ArrayRef<int64_t>{maxVectorLength},
                                vecTy.getElementType()));
  return builder.create<gcu::VectorConvertOp>(loc, partTypes, v).getResult(0);
}

// If |v| is an i1 vector, convert it to i8 (the element type expected by
// vector store / tar store).  The conversion is inserted right after the
// defining op of |v| to keep the def-use order valid.  Returns |v| unchanged
// when no conversion is needed.
Value convertI1ToI8VectorForStore(OpBuilder &builder, Location loc, Value v,
                                  unsigned vectorLength) {
  if (!cast<VectorType>(v.getType()).getElementType().isInteger(1))
    return v;
  OpBuilder::InsertionGuard guard(builder);
  auto defOp = v.getDefiningOp();
  assert(defOp);
  builder.setInsertionPointAfter(defOp);
  return builder
      .create<gcu::VectorConvertOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(8)),
          v)
      .getResult(0);
}

enum class InputKind {
  MemRef,
  Ptr,
  Scalar,
};

enum class BroadcastKind {
  BroadcastOnDim0,
  BroadcastOnLastDim,
};

struct BroadcastDesc {
  BroadcastKind kind;
  unsigned elementNumPerThread;
};

struct InputDesc {
  InputKind kind;
  bool isAllocaInput = false;
  std::optional<unsigned> preferredVectorLength = std::nullopt;
  std::optional<BroadcastDesc> broadcastDesc = std::nullopt;
};

struct OutputDesc {
  bool useAlloca = false;
  std::optional<unsigned> reuseInputIndex = std::nullopt;
  bool useLocalStore = false;
};

struct ElementwiseFusionDesc {
  SmallVector<InputDesc> inputDescs;
  SmallVector<OutputDesc> outputDescs;

  unsigned vectorLength;
#if TRITON_VERSION >= 38
  unsigned maxBpe = 1;
#endif
  bool needCvtVboolDataLayout = false;

  unsigned loopDepth;
  unsigned loopUnrollCount;
  bool unrollFull = false;
};

class ElementwiseFusionAnalysis {
public:
  explicit ElementwiseFusionAnalysis(triton::gcu::ElementwiseFusionRegionOp op,
                                     ConversionPatternRewriter &rewriter)
      : op(op), rewriter(rewriter) {}
  FailureOr<ElementwiseFusionDesc> run(ArrayRef<Value> remappedOperands);

private:
  __attribute__((unused)) bool isResultOnlyUsedBy2DStore();
  Value getAllocaInputValue(Value val);
  LogicalResult populateBroadcastDescs(ElementwiseFusionDesc &desc);
  DenseSet<Type> analyzeInputs(ElementwiseFusionDesc &desc,
                               ArrayRef<Value> remappedOperands);
  void collectResultElementTypes(ElementwiseFusionDesc &desc,
                                 DenseSet<Type> &elementTypeSet);
  void analyzeOutputs(ElementwiseFusionDesc &desc,
                      const DenseSet<Type> &elementTypeSet);
  bool needsFullUnroll(const ElementwiseFusionDesc &desc) const;

private:
  triton::gcu::ElementwiseFusionRegionOp op;
  ConversionPatternRewriter &rewriter;
  bool useAllocaForVboolResult = true;
};

FailureOr<BroadcastKind> getBroadcastKind(triton::BroadcastOp op) {
  auto srcType = op.getSrc().getType();
  auto resultType = op.getType();
  auto rank = srcType.getRank();
  SmallVector<unsigned> broadcastAxes;
  for (unsigned i = 0; i < rank; ++i) {
    if (srcType.getDimSize(i) == 1 && resultType.getDimSize(i) != 1) {
      broadcastAxes.push_back(i);
    }
  }
  if (broadcastAxes.size() != 1) {
    op->emitError() << "Only support one broadcast axis";
    return failure();
  }
  if (broadcastAxes[0] == 0) {
    return BroadcastKind::BroadcastOnDim0;
  } else if (broadcastAxes[0] == rank - 1) {
    return BroadcastKind::BroadcastOnLastDim;
  } else {
    op->emitError() << "Only support broadcast on dim0 or last dim";
    return failure();
  }
}

std::optional<unsigned> inferPreferredVectorLength(Value value) {
  for (auto user : value.getUsers()) {
    if (auto storeOp = dyn_cast<vector::StoreOp>(user))
      return storeOp.getVectorType().getDimSize(0);
    if (isa<memref::ReinterpretCastOp>(user)) {
      if (auto vectorLength = inferPreferredVectorLength(user->getResult(0)))
        return vectorLength;
    }
  }
  return std::nullopt;
}

unsigned getVectorLength(unsigned bpe) { return kOaccSizeInBytes / bpe; }

#if TRITON_VERSION >= 38
TensorType getFusionTensorType(triton::gcu::ElementwiseFusionRegionOp op) {
  for (auto argument : op.getBody()->getArguments()) {
    if (auto tensorType = dyn_cast<TensorType>(argument.getType()))
      return tensorType;
  }
  for (auto &bodyOp : op.getBody()->without_terminator()) {
    for (auto type : bodyOp.getResultTypes()) {
      if (auto tensorType = dyn_cast<TensorType>(type))
        return tensorType;
    }
  }
  llvm_unreachable("elementwise fusion region must contain a tensor");
}
#endif

unsigned computeVectorLength(const DenseSet<Type> &elementTypeSet,
                             ArrayRef<InputDesc> inputDescs) {
  unsigned minBpe = 8;
  for (auto elementTy : elementTypeSet) {
    auto bpe = mlir::triton::gcu::getBpe(elementTy);
    minBpe = std::min(minBpe, bpe);
  }
  unsigned vectorLength = getVectorLength(minBpe);

  for (const auto &inputDesc : inputDescs) {
    if (inputDesc.broadcastDesc &&
        inputDesc.broadcastDesc->elementNumPerThread >
            kLoopUnrollTimes * vectorLength) {
      vectorLength =
          inputDesc.broadcastDesc->elementNumPerThread / kLoopUnrollTimes;
    }
    if (inputDesc.preferredVectorLength &&
        *inputDesc.preferredVectorLength > vectorLength)
      vectorLength = *inputDesc.preferredVectorLength;
  }
  return vectorLength;
}

bool canUseAllocaForValue(Value val) {
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned idx = 0; idx < worklist.size(); ++idx) {
    Value cur = worklist[idx];
    for (auto *user : cur.getUsers()) {
      if (isa<triton::gcu::ElementwiseFusionRegionOp>(user))
        continue;
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(user)) {
        if (reduceOp->getNumOperands() != 1) {
          return false;
        }
        auto axis = reduceOp.getAxis();
        auto tensorType = reduceOp.getInputTypes()[0];
        if (tensorType.getRank() - 1 != axis ||
            triton::gcu::getElemsPerThread(tensorType)[axis] *
                    triton::gcu::getBpe(tensorType.getElementType()) <
                kOaccSizeInBytes) {
          return false;
        }
        continue;
      }
      if (auto gatherOp = dyn_cast<triton::GatherOp>(user)) {
        if (gatherOp.getSrc() != cur)
          continue;
        return false;
      }
      if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
        if (triton::gcu::isExpensiveView(expandDimsOp))
          return false;
        worklist.insert(expandDimsOp.getResult());
        continue;
      }
      if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        for (auto [initArgIdx, initArg] :
             llvm::enumerate(forOp.getInitArgs())) {
          if (initArg == cur) {
            worklist.insert(forOp.getRegionIterArg(initArgIdx));
            worklist.insert(forOp.getResult(initArgIdx));
          }
        }
        continue;
      }
      return false;
    }
  }
  return true;
}

bool hasVboolIncompatibleOperand(triton::gcu::ElementwiseFusionRegionOp op) {
  return llvm::any_of(op.getOperands(), [&](Value operand) {
    auto type = dyn_cast<TensorType>(operand.getType());
    if (!type || !type.getElementType().isInteger(1))
      return false;
    Operation *producer = operand.getDefiningOp();
    while (isa<triton::ExpandDimsOp>(producer) &&
           !triton::gcu::isExpensiveView(cast<triton::ExpandDimsOp>(producer)))
      producer = producer->getOperand(0).getDefiningOp();
    return !isa<triton::gcu::ElementwiseFusionRegionOp>(producer);
  });
}

bool hasVboolIncompatibleOp(triton::gcu::ElementwiseFusionRegionOp op) {
  return llvm::any_of(*op.getBody(), [&](Operation &innerOp) {
    auto resultTypes = innerOp.getResultTypes();
    if (!llvm::any_of(resultTypes, [&](Type resultType) {
          return cast<TensorType>(resultType).getElementType().isInteger(1);
        }))
      return false;
    return !isa<arith::AndIOp>(&innerOp) && !isa<arith::OrIOp>(&innerOp) &&
           !isa<arith::XOrIOp>(&innerOp) && !isa<arith::CmpIOp>(&innerOp) &&
           !isa<arith::CmpFOp>(&innerOp) && !isa<arith::ConstantOp>(&innerOp) &&
           !isa<arith::SelectOp>(&innerOp) && !isa<triton::SplatOp>(&innerOp) &&
           !isa<triton::BroadcastOp>(&innerOp);
  });
}

bool canUseAllocaForVboolValue(Value val) {
  SetVector<Value> worklist;
  worklist.insert(val);
  for (unsigned idx = 0; idx < worklist.size(); ++idx) {
    Value cur = worklist[idx];
    for (auto *user : cur.getUsers()) {
      if (auto fusionOp =
              dyn_cast<triton::gcu::ElementwiseFusionRegionOp>(user)) {
        if (hasVboolIncompatibleOperand(fusionOp) ||
            hasVboolIncompatibleOp(fusionOp))
          return false;
        continue;
      }
      if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
        if (triton::gcu::isExpensiveView(expandDimsOp))
          return false;
        worklist.insert(expandDimsOp.getResult());
        continue;
      }
      return false;
    }
  }
  return true;
}

bool ElementwiseFusionAnalysis::isResultOnlyUsedBy2DStore() {
  if (op.getNumResults() != 1)
    return false;
  if (!op->hasOneUse())
    return false;
  auto storeOp = dyn_cast<triton::gcu::StoreOp>(*op->getUsers().begin());
  if (!storeOp)
    return false;
  auto tensorType = storeOp.getValue().getType();
  return tensorType && tensorType.getRank() == 2;
}

Value ElementwiseFusionAnalysis::getAllocaInputValue(Value val) {
  if (auto blockArg = dyn_cast<BlockArgument>(val)) {
    auto forOp = dyn_cast<scf::ForOp>(blockArg.getOwner()->getParentOp());
    if (!forOp)
      return nullptr;
    return getAllocaInputValue(
        forOp.getInitArgs()[blockArg.getArgNumber() -
                            forOp.getNumInductionVars()]);
  }
  auto defOp = val.getDefiningOp();
  if (isa<memref::AllocaOp>(defOp)) {
    return val;
  } else if (isa<memref::ReinterpretCastOp>(defOp)) {
    return getAllocaInputValue(defOp->getOperand(0));
  } else if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    unsigned resultIdx = cast<OpResult>(val).getResultNumber();
    auto initArg = forOp.getInitArgs()[resultIdx];
    auto remappedValue = rewriter.getRemappedValue(initArg);
    return getAllocaInputValue(remappedValue ? remappedValue : initArg);
  }
  return nullptr;
}

LogicalResult
ElementwiseFusionAnalysis::populateBroadcastDescs(ElementwiseFusionDesc &desc) {
  for (auto broadcastOp : op.getBody()->getOps<triton::BroadcastOp>()) {
    auto broadcastKind = getBroadcastKind(broadcastOp);
    if (failed(broadcastKind))
      return failure();

    auto blockArg = dyn_cast<BlockArgument>(broadcastOp.getSrc());
    if (!blockArg) {
      op->emitError() << "Only support broadcast on block argument";
      return failure();
    }

    auto argNum = blockArg.getArgNumber();
    if (desc.inputDescs[argNum].broadcastDesc) {
      if (desc.inputDescs[argNum].broadcastDesc->kind != *broadcastKind) {
        op->emitError()
            << "Broadcast on both dim 0 and the last dim in elementwise "
               "fusion is not supported";
        return failure();
      }
      continue;
    }

    BroadcastDesc broadcastDesc;
    broadcastDesc.kind = *broadcastKind;
    switch (*broadcastKind) {
    case BroadcastKind::BroadcastOnDim0:
      broadcastDesc.elementNumPerThread =
          triton::gcu::getTotalElemsPerThread(broadcastOp.getSrc().getType());
      break;
    case BroadcastKind::BroadcastOnLastDim:
      broadcastDesc.elementNumPerThread =
          triton::gcu::getElemsPerThread(broadcastOp.getType()).back();
      break;
    default:
      llvm_unreachable("Unsupported broadcast kind");
    }
    desc.inputDescs[argNum].broadcastDesc = broadcastDesc;
  }
  return success();
}

DenseSet<Type>
ElementwiseFusionAnalysis::analyzeInputs(ElementwiseFusionDesc &desc,
                                         ArrayRef<Value> remappedOperands) {
  DenseSet<Type> elementTypeSet;
  for (auto [i, remappedValue] : llvm::enumerate(remappedOperands)) {
    auto type = remappedValue.getType();
    if (auto memrefType = dyn_cast<MemRefType>(type)) {
      desc.inputDescs[i].kind = InputKind::MemRef;
      auto allocaValue = getAllocaInputValue(remappedValue);
      if (allocaValue) {
        desc.inputDescs[i].isAllocaInput = true;
        desc.inputDescs[i].preferredVectorLength =
            inferPreferredVectorLength(allocaValue);
      }
      auto elementType = memrefType.getElementType();
      if (elementType.isInteger(1)) {
        if (desc.inputDescs[i].isAllocaInput)
          continue;
        if (desc.inputDescs[i].broadcastDesc &&
            desc.inputDescs[i].broadcastDesc->kind ==
                BroadcastKind::BroadcastOnLastDim)
          continue;
        desc.needCvtVboolDataLayout = true;
      }
      elementTypeSet.insert(elementType);
    } else if (isa<gcu::PtrType>(type)) {
      desc.inputDescs[i].kind = InputKind::Ptr;
    } else {
      desc.inputDescs[i].kind = InputKind::Scalar;
    }
  }
  return elementTypeSet;
}

void ElementwiseFusionAnalysis::collectResultElementTypes(
    ElementwiseFusionDesc &desc, DenseSet<Type> &elementTypeSet) {
  for (auto &o : op.getBody()->without_terminator()) {
    for (auto type : o.getResultTypes()) {
#if TRITON_VERSION >= 38
      auto tensorType = dyn_cast<TensorType>(type);
      if (!tensorType)
        continue;
      auto elementTy = tensorType.getElementType();
#else
      auto elementTy = cast<TensorType>(type).getElementType();
#endif
      if (!elementTy.isInteger(1)) {
        elementTypeSet.insert(elementTy);
      }
    }
  }
  useAllocaForVboolResult = !desc.needCvtVboolDataLayout;
  for (auto result : op->getResults()) {
    auto elementType =
        cast<RankedTensorType>(result.getType()).getElementType();
    if (elementType.isInteger(1)) {
      if (!useAllocaForVboolResult || canUseAllocaForVboolValue(result))
        continue;
      useAllocaForVboolResult = false;
    }
    elementTypeSet.insert(elementType);
  }
}

void ElementwiseFusionAnalysis::analyzeOutputs(
    ElementwiseFusionDesc &desc, const DenseSet<Type> &elementTypeSet) {
  unsigned maxBpe = 1;
  for (auto elementTy : elementTypeSet)
    maxBpe = std::max(maxBpe, mlir::triton::gcu::getBpe(elementTy));
#if TRITON_VERSION >= 38
  desc.maxBpe = maxBpe;
  auto totalNumElems =
      triton::gcu::getTotalElemsPerThread(getFusionTensorType(op));
#else
  auto totalNumElems = triton::gcu::getTotalElemsPerThread(
      op.getBody()->front().getResultTypes().front());
#endif
  auto tripCount = ceil<unsigned>(totalNumElems, desc.vectorLength);
  if (totalNumElems < desc.vectorLength || tripCount > kLoopUnrollTimes) {
    if (!desc.needCvtVboolDataLayout &&
        llvm::any_of(
            op->getResultTypes(),
            [](auto type) {
              return cast<RankedTensorType>(type).getElementType().isInteger(1);
            }) &&
        desc.vectorLength < getVectorLength(1)) {
      desc.vectorLength = getVectorLength(1);
      tripCount = ceil<unsigned>(totalNumElems, desc.vectorLength);
    }
  } else {
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      auto elementType =
          cast<RankedTensorType>(result.getType()).getElementType();
      desc.outputDescs[i].useAlloca = elementType.isInteger(1)
                                          ? useAllocaForVboolResult
                                          : canUseAllocaForValue(result);
    }
  }
  auto getInplaceIndex = [](IntegerAttr attr) -> std::optional<unsigned> {
    if (!attr || attr.getInt() < 0)
      return std::nullopt;
    return static_cast<unsigned>(attr.getInt());
  };
  auto useLocalStore = [&]() {
    auto accStoreAttr = op->getAttrOfType<StringAttr>(kAccStore);
    return accStoreAttr && accStoreAttr.getValue() == kAccStoreLocal;
  };
  auto reuseInputIndex =
      getInplaceIndex(op->getAttrOfType<IntegerAttr>(kAccReuseInplaceOperand));
  auto reuseOutputIndex =
      getInplaceIndex(op->getAttrOfType<IntegerAttr>(kAccReuseInplaceResult));
  if (reuseInputIndex && reuseOutputIndex &&
      *reuseInputIndex < desc.inputDescs.size() &&
      *reuseOutputIndex < desc.outputDescs.size()) {
    auto &outputDesc = desc.outputDescs[*reuseOutputIndex];
    outputDesc.useAlloca = true;
    outputDesc.reuseInputIndex = *reuseInputIndex;
    if (useLocalStore() &&
        !canUseAllocaForValue(op->getResult(*reuseOutputIndex))) {
      outputDesc.useLocalStore = true;
    }
  }
  // The per-region unroll is decided by the shared OACC-pressure plan so the
  // annotator's admission check (AnnotateDotFusion::exceedOaccLimit) uses the
  // same numbers.
  desc.loopUnrollCount = triton::gcu::computeFusionUnrollPlan(
                             op, desc.vectorLength, maxBpe, tripCount)
                             .loopCnt;
}

bool ElementwiseFusionAnalysis::needsFullUnroll(
    const ElementwiseFusionDesc &desc) const {
  auto isNoneAccStoreOperand = [](Value operand) {
    auto dotOp = operand.getDefiningOp<triton::DotOp>();
    if (!dotOp)
      return false;
    auto accStoreAttr = dotOp->getAttrOfType<StringAttr>(kAccStore);
    return accStoreAttr && accStoreAttr.getValue() == kAccStoreNone;
  };
  return llvm::any_of(op->getOperands(), isNoneAccStoreOperand) ||
         llvm::any_of(
             desc.inputDescs,
             [](const InputDesc &input) { return input.isAllocaInput; }) ||
         llvm::any_of(desc.outputDescs, [](const OutputDesc &output) {
           return output.useAlloca;
         });
}

FailureOr<ElementwiseFusionDesc>
ElementwiseFusionAnalysis::run(ArrayRef<Value> remappedOperands) {
  ElementwiseFusionDesc desc;
  desc.loopDepth = 1;
  desc.inputDescs.resize(op->getNumOperands());
  desc.outputDescs.resize(op->getNumResults());
  if (failed(populateBroadcastDescs(desc)))
    return failure();
  auto elementTypeSet = analyzeInputs(desc, remappedOperands);
  collectResultElementTypes(desc, elementTypeSet);
  desc.vectorLength = computeVectorLength(elementTypeSet, desc.inputDescs);
  analyzeOutputs(desc, elementTypeSet);
  desc.unrollFull = needsFullUnroll(desc);
  return desc;
}

struct GCUElementwiseFusionOpLowering
    : SharedConversionPattern<triton::gcu::ElementwiseFusionRegionOp> {
  // Published OACC accumulator buffers, keyed by original loop result. Used
  // on Triton 3.5 so accumulator reads target the local buffer
  // TTDotOpLowering published, which the conversion mapping cannot deliver to
  // later consumers.
#if TRITON_VERSION == 35
  AccPublishContext &accPublishContext;
  GCUElementwiseFusionOpLowering(
      const TypeConverter &converter, MLIRContext *ctx,
      triton::gcu::FirstLastUserAnalysis &userAnalysis,
      std::map<Operation *, Operation *> &replaced2Origin,
      triton::gcu::DTETagPool &pTagPool, AccPublishContext &accPublishContext)
      : SharedConversionPattern(converter, ctx, userAnalysis, replaced2Origin,
                                pTagPool),
        accPublishContext(accPublishContext) {}
#else
  using SharedConversionPattern::SharedConversionPattern;
#endif

  LogicalResult
  matchAndRewrite(triton::gcu::ElementwiseFusionRegionOp op,
                  SharedConversionPattern::OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    enterTritionOp(rewriter, op);
    if (pTagPool.isExistInMap(op.getOperation())) {
      pTagPool.releaseMap(op.getOperation());
    }
    auto loc = op.getLoc();

    SmallVector<Value> remappedOperands;
#if TRITON_VERSION == 35
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      Value remappedValue =
          stripUnrealizedConversionCastOp(adaptor.getOperands()[i]);
      if (failed(mlir::triton::gcu::resolveAccumulatorOperand(
              operand, remappedValue, accPublishContext)))
        return rewriter.notifyMatchFailure(
            op,
            "expected OACC publish buffer is not yet available for operand " +
                std::to_string(i));
      remappedOperands.push_back(remappedValue);
    }
#else
    for (auto remappedValue : adaptor.getOperands())
      remappedOperands.push_back(
          stripUnrealizedConversionCastOp(remappedValue));
#endif

    Block::iterator reduceOpIter = op.getBody()->begin();
    while (reduceOpIter != op.getBody()->end()) {
      if (isa<triton::ReduceOp>(*reduceOpIter)) {
        break;
      }
      ++reduceOpIter;
    }

    if (reduceOpIter != op.getBody()->end()) {
      mlir::triton::gcu::ReduceGenerator reduceGenerator(
          cast<triton::ReduceOp>(*reduceOpIter), op.getBody()->getArguments(),
          llvm::make_range(op.getBody()->begin(), reduceOpIter));
      auto results =
          llvm::to_vector(llvm::map_range(op.getResults(), [&](auto result) {
            auto memrefType = dyn_cast<MemRefType>(
                getTypeConverter()->convertType(result.getType()));
            auto lastUser = userAnalysis.getLastUser(result);
            return syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                               replaced2Origin, memrefType);
          }));
      auto inputs =
          reduceGenerator.normalizeInputs(rewriter, loc, remappedOperands);
      auto outputs = reduceGenerator.normalizeOutputs(rewriter, loc, results);
      reduceGenerator.applyReduce(rewriter, loc, outputs, inputs);
      rewriter.replaceOp(op, results);
      return success();
    }

    ElementwiseFusionAnalysis analysis(op, rewriter);
    auto fusionDesc = analysis.run(remappedOperands);
    if (llvm::failed(fusionDesc)) {
      return failure();
    }

    auto needCvtDataLayout = fusionDesc->needCvtVboolDataLayout;

#if TRITON_VERSION >= 38
    auto totalNumElems =
        triton::gcu::getTotalElemsPerThread(getFusionTensorType(op));
#else
    auto totalNumElems = triton::gcu::getTotalElemsPerThread(
        op.getBody()->front().getResultTypes().front());
#endif
    bool isSmallSize = totalNumElems < fusionDesc->vectorLength;

    auto convertI1MemrefToI8Memref = [&](Value memref) {
      auto ptr = rewriter.create<mlir::gcu::MemRefToPtrOp>(
          loc,
          mlir::gcu::PtrType::get(rewriter.getContext(), rewriter.getI1Type()),
          memref);
      return rewriter.create<mlir::gcu::PtrToMemRefOp>(
          loc,
          MemRefType::get(ArrayRef<int64_t>{ShapedType::kDynamic},
                          rewriter.getI8Type()),
          ptr);
    };

    const auto &inputDescs = fusionDesc->inputDescs;
    const auto &outputDescs = fusionDesc->outputDescs;
    const auto vectorLength = fusionDesc->vectorLength;
    const auto loopUnrollCount = fusionDesc->loopUnrollCount;

    SmallVector<Value> results;
    SmallVector<Value> outputs;
    for (auto [i, result] : llvm::enumerate(op.getResults())) {
      const auto &desc = outputDescs[i];
      auto resultType = dyn_cast<MemRefType>(
          getTypeConverter()->convertType(result.getType()));
      auto elementTy = resultType.getElementType();
      auto outputElemTy = elementTy;
      if (desc.useAlloca) {
        if (desc.reuseInputIndex) {
          results.push_back(remappedOperands[*desc.reuseInputIndex]);
        } else {
          auto allocaOp = rewriter.create<memref::AllocaOp>(loc, resultType);
          allocaOp.setAlignment(kOaccSizeInBytes);
          results.push_back(allocaOp.getResult());
        }
      } else {
        auto lastUser = userAnalysis.getLastUser(result);
        results.push_back(syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                      replaced2Origin, resultType));
      }
      Value output = results.back();
      if (elementTy.isInteger(1) && !desc.useAlloca) {
        output = convertI1MemrefToI8Memref(output);
        outputElemTy = rewriter.getI8Type();
        if (fusionDesc->loopDepth != 1) {
          auto [strides, offset] = resultType.getStridesAndOffset();
          output = rewriter.create<memref::ReinterpretCastOp>(
              loc, MemRefType::get(resultType.getShape(), outputElemTy), output,
              offset, resultType.getShape(), strides);
        }
      }
      if (fusionDesc->loopDepth == 1) {
        output = rewriter.create<memref::ReinterpretCastOp>(
            loc,
            MemRefType::get(ArrayRef<int64_t>{totalNumElems}, outputElemTy),
            output, 0, ArrayRef<int64_t>{totalNumElems}, ArrayRef<int64_t>{1});
      }
      outputs.push_back(output);
    }

    SmallVector<Value> inputs;
    for (auto [i, operand] : llvm::enumerate(op.getOperands())) {
      const auto &desc = inputDescs[i];
      Value input = remappedOperands[i];
      if (desc.kind == InputKind::MemRef) {
        auto memrefTy = cast<MemRefType>(input.getType());
        auto elementType = memrefTy.getElementType();
        auto inputElementTy = elementType;
        if (elementType.isInteger(1) && !desc.isAllocaInput &&
            !(desc.broadcastDesc &&
              desc.broadcastDesc->kind == BroadcastKind::BroadcastOnLastDim)) {
          input = convertI1MemrefToI8Memref(input);
          inputElementTy = rewriter.getI8Type();
          if (fusionDesc->loopDepth != 1) {
            auto [strides, offset] = memrefTy.getStridesAndOffset();
            input = rewriter.create<memref::ReinterpretCastOp>(
                loc, MemRefType::get(memrefTy.getShape(), inputElementTy),
                input, offset, memrefTy.getShape(), strides);
          }
        }
        if (fusionDesc->loopDepth == 1) {
          auto numElements =
              triton::gcu::getTotalElemsPerThread(operand.getType());
          input = rewriter.create<memref::ReinterpretCastOp>(
              loc,
              MemRefType::get(ArrayRef<int64_t>{numElements}, inputElementTy),
              input, 0, ArrayRef<int64_t>{numElements}, ArrayRef<int64_t>{1});
        }
      }
      inputs.push_back(input);
    }

    auto insertPoint = rewriter.saveInsertionPoint();
    SmallVector<IRMapping> operandMaps(fusionDesc->loopUnrollCount);

    Value mask;
    llvm::MapVector<Operation *, Value> offsets;
    DenseSet<Operation *> perVecContiguousOps;
    DenseSet<Operation *> useLoadStoreInstrOps;
#if TRITON_VERSION >= 38
    if (fusionDesc->maxBpe < 8)
#endif
      useLoadStoreInstrOps =
          trySimplifyLoadStore(op, rewriter, offsets, mask,
                               fusionDesc->vectorLength, perVecContiguousOps);
    bool disableLoadStroreInstrOptimize = false;

    triton::gcu::TritonGCUBuilder b(loc, rewriter);
    SmallVector<Value> initValues;
    auto emitLoad = [&](const InputDesc &desc, Value input,
                        int offset) -> Value {
      assert(desc.broadcastDesc &&
             desc.broadcastDesc->kind == BroadcastKind::BroadcastOnDim0);
      const auto elementNumPerThread = desc.broadcastDesc->elementNumPerThread;
      auto elementType = cast<MemRefType>(input.getType()).getElementType();
      auto preferredVectorLength = desc.preferredVectorLength
                                       ? *desc.preferredVectorLength
                                       : vectorLength;
      if (vectorLength > preferredVectorLength) {
        auto length = vectorLength > elementNumPerThread ? elementNumPerThread
                                                         : vectorLength;
        auto vectorType = VectorType::get(
            ArrayRef<int64_t>{preferredVectorLength}, elementType);
        SmallVector<Value> values;
        for (unsigned i = 0; i < length / preferredVectorLength; ++i) {
          values.emplace_back(rewriter.create<vector::LoadOp>(
              loc, vectorType, input,
              ValueRange{rewriter.create<arith::ConstantIndexOp>(
                  loc, offset + i * preferredVectorLength)}));
        }
        return rewriter
            .create<gcu::VectorConvertOp>(
                loc, VectorType::get(ArrayRef<int64_t>{length}, elementType),
                values)
            ->getResult(0);
      } else {
        auto vectorType =
            VectorType::get(ArrayRef<int64_t>{vectorLength > elementNumPerThread
                                                  ? elementNumPerThread
                                                  : preferredVectorLength},
                            elementType);
        return rewriter.create<vector::LoadOp>(
            loc, vectorType, input,
            ValueRange{rewriter.create<arith::ConstantIndexOp>(loc, offset)});
      }
    };
    for (auto [argNum, input] : llvm::enumerate(inputs)) {
      const auto &desc = inputDescs[argNum];
      if (desc.broadcastDesc) {
        if (desc.broadcastDesc->kind != BroadcastKind::BroadcastOnDim0)
          continue;
        auto vectorType =
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            cast<MemRefType>(input.getType()).getElementType());
        auto elementNumPerThread = desc.broadcastDesc->elementNumPerThread;
        if (vectorLength > elementNumPerThread) {
          auto v = emitLoad(desc, input, 0);
          v = rewriter
                  .create<gcu::VectorConvertOp>(
                      loc, vectorType,
                      SmallVector<Value>(vectorLength / elementNumPerThread, v))
                  .getResult(0);
          for (unsigned i = 0; i < loopUnrollCount; ++i) {
            operandMaps[i].map(op.getRegion().getArgument(argNum), v);
          }
        } else if (vectorLength == elementNumPerThread) {
          auto v = emitLoad(desc, input, 0);
          for (unsigned i = 0; i < loopUnrollCount; ++i) {
            operandMaps[i].map(op.getRegion().getArgument(argNum), v);
          }
        } else {
          auto preferredVectorLength = desc.preferredVectorLength
                                           ? *desc.preferredVectorLength
                                           : vectorLength;
          auto tripCount = elementNumPerThread / vectorLength;
          if (preferredVectorLength > vectorLength) {
            auto numVec = preferredVectorLength / vectorLength;
            for (unsigned i = 0;
                 i < elementNumPerThread / preferredVectorLength; ++i) {
              auto convertOp = rewriter.create<gcu::VectorConvertOp>(
                  loc, SmallVector<Type>(numVec, vectorType),
                  emitLoad(desc, input, i * preferredVectorLength));
              for (unsigned j = 0; j < numVec; ++j) {
                for (unsigned k = 0; k < loopUnrollCount / tripCount; ++k) {
                  operandMaps[k * tripCount + i * numVec + j].map(
                      op.getRegion().getArgument(argNum),
                      convertOp.getResult(j));
                }
              }
            }
          } else {
            for (unsigned j = 0; j < tripCount; ++j) {
              auto v = emitLoad(desc, input, j * vectorLength);
              for (unsigned k = 0; k < loopUnrollCount / tripCount; ++k) {
                operandMaps[k * tripCount + j].map(
                    op.getRegion().getArgument(argNum), v);
              }
            }
          }
        }
      } else if (desc.kind == InputKind::Ptr) {
        Value memref = rewriter.create<mlir::gcu::PtrToMemRefOp>(
            loc,
            MemRefType::get(
                ArrayRef<int64_t>{ShapedType::kDynamic},
                cast<gcu::PtrType>(input.getType()).getElementType()),
            input);
        for (unsigned j = 0; j < loopUnrollCount; ++j) {
          operandMaps[j].map(op.getRegion().getArgument(argNum), memref);
        }
      } else if (desc.kind == InputKind::MemRef && !desc.isAllocaInput) {
        initValues.emplace_back(b.tarAddr(input));
      }
    }

    unsigned numInputTarAddrs = initValues.size();
    for (auto [idx, output] : llvm::enumerate(outputs)) {
      if (!outputDescs[idx].useAlloca)
        initValues.emplace_back(b.tarAddr(output));
    }
    auto tarStride = b.tarValue(kOaccSizeInBytes);

    DenseMap<Operation *, unsigned> iterArgIndexMap;
    for (auto makeRangeOp : op.getBody()->getOps<triton::MakeRangeOp>()) {
      auto startIdx = makeRangeOp.getStart();
      auto elementTy = makeRangeOp.getResult().getType().getElementType();
      Value start =
          rewriter.create<arith::ConstantIntOp>(loc, elementTy, startIdx)
              .getResult();
      if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
        start = rewriter.create<arith::AddIOp>(
            loc,
            rewriter.create<arith::MulIOp>(
                loc,
                rewriter.create<arith::IndexCastOp>(
                    loc, elementTy,
                    getWarpIds(rewriter, loc, makeRangeOp.getType()).front()),
                rewriter.create<arith::ConstantIntOp>(loc, elementTy,
                                                      totalNumElems)),
            start);
      }
      iterArgIndexMap[makeRangeOp] = initValues.size();
      initValues.emplace_back(
          rewriter
              .create<gcu::VectorStepOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                  start)
              .getResult());
    }

    if (!isSmallSize) {
      DenseMap<Value, Value> addrMap;
      for (auto [op, offset] : offsets) {
        if (!useLoadStoreInstrOps.contains(op))
          continue;
        auto ptr = op->getOperand(0);
        assert(isa<BlockArgument>(ptr));
        auto ptrType =
            cast<gcu::PtrType>(getTypeConverter()->convertType(ptr.getType()));
        Value addr;
        auto it = addrMap.find(ptr);
        if (it != addrMap.end()) {
          addr = it->second;
        } else {
          addr = b.tarAddr(operandMaps[0].lookup(ptr));
          addrMap[ptr] = addr;
        }
        // Extend offset to i64 before multiplication to avoid i32 overflow.
        b.tarJump(
            addr,
            b.tarValue(rewriter.create<arith::MulIOp>(
                loc,
                getElementTypeOrSelf(offset).isInteger(64)
                    ? offset
                    : rewriter
                          .create<arith::ExtSIOp>(loc, rewriter.getI64Type(),
                                                  offset)
                          .getResult(),
                rewriter.create<arith::ConstantIntOp>(
                    loc, rewriter.getI64Type(),
                    mlir::triton::gcu::getBpe(ptrType.getElementType())))));
        iterArgIndexMap[op] = initValues.size();
        initValues.emplace_back(addr);
      }
    }
    Value step;
    auto loopBody = [&](OpBuilder &builder, Location loc, Value loopIter,
                        ValueRange iterArgs) {
      SmallVector<Value> iterArgValues(iterArgs);

      // Compute the load/store offset for an alloca-backed buffer at the
      // given inner unroll index. When the outer loop runs more than once
      // (ceil(totalNumElems, vectorLength * loopCnt) > 1), the offset must
      // advance with loopIter so each outer iteration touches a distinct
      // buffer slice. loopIter's unit differs by loop shape: element
      // offset when useLoadStoreInstrOps is non-empty, iteration index
      // otherwise.
      auto computeAllocaOffset = [&](unsigned innerIdx) -> Value {
        Value offset = builder.create<arith::ConstantIndexOp>(
            loc, innerIdx * vectorLength);
        if (ceil<unsigned>(totalNumElems, vectorLength * loopUnrollCount) > 1) {
          Value baseOffset;
          if (!useLoadStoreInstrOps.empty()) {
            // Tar mode: loopIter is an element offset, add it directly.
            baseOffset = loopIter;
          } else {
            // Normal mode: loopIter is an iteration index, multiply by the
            // per-trip block size (vectorLength * loopCnt).
            baseOffset = builder.create<arith::MulIOp>(
                loc, loopIter,
                builder.create<arith::ConstantIndexOp>(loc, loopUnrollCount *
                                                                vectorLength));
          }
          offset = builder.createOrFold<arith::AddIOp>(loc, baseOffset, offset);
        }
        return offset;
      };

      for (unsigned i = 0; i < loopUnrollCount; ++i) {
        for (unsigned j = 0, k = 0; j < inputs.size(); ++j) {
          const auto &desc = inputDescs[j];
          if (desc.broadcastDesc) {
            if (desc.broadcastDesc->kind != BroadcastKind::BroadcastOnLastDim)
              continue;
            auto elementNumPerThread = desc.broadcastDesc->elementNumPerThread;
            auto elementTy =
                cast<MemRefType>(inputs[j].getType()).getElementType();
            auto rowsPerTrip =
                loopUnrollCount * vectorLength / elementNumPerThread;
            auto computeLoadOffset = [&](unsigned innerOffset) {
              return builder.createOrFold<arith::AddIOp>(
                  loc,
                  builder.createOrFold<arith::MulIOp>(
                      loc, loopIter,
                      builder.create<arith::ConstantIndexOp>(loc, rowsPerTrip)),
                  builder.create<arith::ConstantIndexOp>(loc, innerOffset));
            };
            auto broadcastLoad = [&](unsigned innerOffset,
                                     unsigned broadcastLength) {
              return builder.create<vector::BroadcastOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{broadcastLength},
                                  elementTy),
                  builder.create<memref::LoadOp>(
                      loc, inputs[j],
                      ValueRange{computeLoadOffset(innerOffset)}));
            };
            if (elementNumPerThread < vectorLength) {
              auto rowsPerVector = vectorLength / elementNumPerThread;
              SmallVector<Value> rowBroadcasts;
              for (unsigned rowIdx = 0; rowIdx < rowsPerVector; ++rowIdx)
                rowBroadcasts.emplace_back(broadcastLoad(
                    i * rowsPerVector + rowIdx, elementNumPerThread));
              auto broadcastVector =
                  builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          elementTy),
                          rowBroadcasts)
                      .getResult(0);
              operandMaps[i].map(op.getRegion().getArgument(j),
                                 broadcastVector);
            } else {
              auto vectorsPerRow = elementNumPerThread / vectorLength;
              if (i % vectorsPerRow == 0) {
                auto broadcastVector =
                    broadcastLoad(i / vectorsPerRow, vectorLength);
                for (unsigned vecIdx = 0; vecIdx < vectorsPerRow; ++vecIdx) {
                  operandMaps[i + vecIdx].map(op.getRegion().getArgument(j),
                                              broadcastVector);
                }
              }
            }
          } else if (desc.isAllocaInput) {
            auto memrefType = cast<MemRefType>(inputs[j].getType());
            auto elementTy = memrefType.getElementType();
            Value offset = computeAllocaOffset(i);
            unsigned preferredVectorLength = desc.preferredVectorLength
                                                 ? *desc.preferredVectorLength
                                                 : vectorLength;
            auto subVectorType = VectorType::get(
                ArrayRef<int64_t>{preferredVectorLength}, elementTy);
            SmallVector<Value> subVectors;
            if (preferredVectorLength >= totalNumElems) {
              Value subVector = builder.create<vector::LoadOp>(
                  loc, subVectorType, inputs[j], ValueRange{offset});
              subVectors.assign(vectorLength / preferredVectorLength,
                                subVector);
            } else if (totalNumElems < vectorLength) {
              for (unsigned vecIdx = 0;
                   vecIdx < totalNumElems / preferredVectorLength; ++vecIdx) {
                subVectors.emplace_back(builder.create<vector::LoadOp>(
                    loc, subVectorType, inputs[j],
                    ValueRange{builder.createOrFold<arith::AddIOp>(
                        loc, offset,
                        builder.create<arith::ConstantIndexOp>(
                            loc, vecIdx * preferredVectorLength))}));
              }
              Value subVector =
                  builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{totalNumElems},
                                          elementTy),
                          subVectors)
                      ->getResult(0);
              subVectors.assign(vectorLength / totalNumElems, subVector);
            } else {
              assert(vectorLength >= preferredVectorLength);
              for (unsigned vecIdx = 0;
                   vecIdx < vectorLength / preferredVectorLength; ++vecIdx) {
                subVectors.emplace_back(builder.create<vector::LoadOp>(
                    loc, subVectorType, inputs[j],
                    ValueRange{builder.createOrFold<arith::AddIOp>(
                        loc, offset,
                        builder.create<arith::ConstantIndexOp>(
                            loc, vecIdx * preferredVectorLength))}));
              }
            }
            auto loadedVector =
                subVectors.size() == 1
                    ? subVectors.front()
                    : builder
                          .create<gcu::VectorConvertOp>(
                              loc,
                              VectorType::get(ArrayRef<int64_t>{vectorLength},
                                              elementTy),
                              subVectors)
                          ->getResult(0);
            operandMaps[i].map(op.getRegion().getArgument(j), loadedVector);
          } else if (desc.kind == InputKind::MemRef) {
            auto elementTy =
                cast<MemRefType>(inputs[j].getType()).getElementType();
            Value inputTarAddr = iterArgValues[k];
            Value loadedVector =
                isSmallSize
                    ? loadSmallSizeTarInput(b, builder, elementTy, vectorLength,
                                            inputTarAddr, tarStride, loc)
                    : generateSplitTarLoad(b, builder, elementTy, vectorLength,
                                           inputTarAddr, tarStride, loc);
            operandMaps[i].map(op.getRegion().getArgument(j), loadedVector);
            iterArgValues[k] = inputTarAddr;
            ++k;
          } else if (desc.kind == InputKind::Ptr) {
            continue;
          } else {
            operandMaps[i].map(op.getRegion().getArgument(j), inputs[j]);
          }
        }
      }

      auto iterRange = op.getRegion().back().without_terminator();
      auto cur = iterRange.begin();
      while (cur != iterRange.end()) {
        auto &op = *cur;
        if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(op)) {
          auto constancy = maskedLoadOp->getAttrOfType<IntegerAttr>(kConstancy);
          bool isBroadcast = false;
          bool isFullWarpBroadcast = false;
          if (constancy) {
            auto constancyVal = constancy.getInt();
            auto totalElems = triton::gcu::getTotalElemsPerThread(
                maskedLoadOp.getResult().getType());
            if (constancyVal == static_cast<int>(totalElems)) {
              isBroadcast = true;
              isFullWarpBroadcast = true;
            } else if (constancyVal >= static_cast<int>(vectorLength) &&
                       constancyVal % static_cast<int>(vectorLength) == 0 &&
                       totalNumElems <= vectorLength * loopUnrollCount) {
              isBroadcast = true;
            }
          }
          for (unsigned i = 0; i < loopUnrollCount; ++i) {
            auto result = maskedLoadOp.getResult();
            auto elementTy = result.getType().getElementType();
            auto vecTy =
                VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
            if (isBroadcast) {
              if (isFullWarpBroadcast && i > 0) {
                operandMaps[i].map(result, operandMaps[0].lookup(result));
              } else {
                operandMaps[i].map(
                    result,
                    simplifyLoadToBroadcast(maskedLoadOp, builder,
                                            operandMaps[i], i, vectorLength));
              }
            } else if (useLoadStoreInstrOps.contains(&op) &&
                       (!disableLoadStroreInstrOptimize ||
                        !maskedLoadOp.getMask())) {
              if (iterArgIndexMap.contains(&op)) {
                if (elementTy.isInteger(64) && vectorLength == 512)
                  operandMaps[i].map(
                      result,
                      generateSplitTarLoad(b, builder, elementTy, vectorLength,
                                           iterArgValues[iterArgIndexMap[&op]],
                                           tarStride, loc));
                else
                  operandMaps[i].map(
                      result,
                      b.tarLoad(vecTy, iterArgValues[iterArgIndexMap[&op]],
                                tarStride));
              } else {
                assert(isSmallSize);
                auto mask = builder.create<vector::ConstantMaskOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength},
                                    builder.getIntegerType(1)),
                    DenseI64ArrayAttr::get(builder.getContext(),
                                           ArrayRef<int64_t>{totalNumElems}));
                operandMaps[i].map(
                    result,
                    builder.create<vector::MaskedLoadOp>(
                        loc, vecTy,
                        operandMaps[i].lookup(maskedLoadOp.getPtr()),
                        ValueRange{builder.create<arith::IndexCastOp>(
                            loc, builder.getIndexType(),
                            builder.create<arith::AddIOp>(
                                loc,
                                offsets[&op].getType().isInteger(64)
                                    ? offsets[&op]
                                    : builder.create<arith::ExtSIOp>(
                                          loc, builder.getIntegerType(64),
                                          offsets[&op]),
                                builder.create<arith::AddIOp>(
                                    loc,
                                    builder.create<arith::ConstantIntOp>(
                                        loc, builder.getIntegerType(64),
                                        i * vectorLength),
                                    builder.create<arith::IndexCastOp>(
                                        loc, builder.getIntegerType(64),
                                        loopIter))))},
                        mask,
                        builder.create<arith::ConstantOp>(
                            loc, DenseElementsAttr::get(
                                     vecTy, builder.getZeroAttr(elementTy)))));
              }
            } else if (perVecContiguousOps.contains(&op)) {
              operandMaps[i].map(
                  result, simplifyLoadToMaskedLoad(
                              maskedLoadOp, builder, operandMaps[i],
                              vectorLength, needCvtDataLayout, isSmallSize,
                              totalNumElems, loopUnrollCount, loopIter, i,
                              !useLoadStoreInstrOps.empty()));
            } else if (offsets.contains(&op)) {
              assert(maskedLoadOp.getMask());
              auto mask = operandMaps[i].lookup(maskedLoadOp.getMask());
              if (getElementTypeOrSelf(mask.getType()).isInteger(8)) {
                mask = builder
                           .create<gcu::VectorConvertOp>(
                               loc,
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(1)),
                               mask)
                           .getResult(0);
              }
              if (isSmallSize) {
                mask = builder.create<arith::AndIOp>(
                    loc,
                    builder.create<vector::ConstantMaskOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        DenseI64ArrayAttr::get(
                            builder.getContext(),
                            ArrayRef<int64_t>{totalNumElems})),
                    mask);
              }
              auto other = maskedLoadOp.getOther();
              operandMaps[i].map(
                  result,
                  builder.create<vector::MaskedLoadOp>(
                      loc, vecTy, operandMaps[i].lookup(maskedLoadOp.getPtr()),
                      ValueRange{builder.create<arith::IndexCastOp>(
                          loc, builder.getIndexType(),
                          builder.create<arith::AddIOp>(
                              loc,
                              offsets[&op].getType().isInteger(64)
                                  ? offsets[&op]
                                  : builder.create<arith::ExtSIOp>(
                                        loc, builder.getIntegerType(64),
                                        offsets[&op]),
                              builder.create<arith::AddIOp>(
                                  loc,
                                  builder.create<arith::ConstantIntOp>(
                                      loc, builder.getIntegerType(64),
                                      i * vectorLength),
                                  builder.create<arith::IndexCastOp>(
                                      loc, builder.getIntegerType(64),
                                      loopIter))))},
                      mask,
                      other ? operandMaps[i].lookup(other)
                            : builder.create<arith::ConstantOp>(
                                  loc,
                                  DenseElementsAttr::get(
                                      vecTy, builder.getZeroAttr(elementTy)))));
            } else {
              handleMaskedLoadOp(maskedLoadOp, builder, operandMaps[i],
                                 vectorLength, needCvtDataLayout);
            }
          }
          ++cur;
        }
        unsigned i = 0;
        while (i < loopUnrollCount) {
          auto iter = cur;
          while (cur != iterRange.end()) {
            auto &o = *cur;
            if (isa<triton::gcu::MaskedLoadOp>(o)) {
              break;
            } else if (auto cvtLayoutOp =
                           dyn_cast<triton::gpu::ConvertLayoutOp>(o)) {
              operandMaps[i].map(cvtLayoutOp.getResult(),
                                 operandMaps[i].lookup(cvtLayoutOp.getSrc()));
            } else if (auto reshapeOp = dyn_cast<triton::ReshapeOp>(o)) {
              operandMaps[i].map(reshapeOp.getResult(),
                                 operandMaps[i].lookup(reshapeOp.getSrc()));
            } else if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(o)) {
              operandMaps[i].map(broadcastOp.getResult(),
                                 operandMaps[i].lookup(broadcastOp.getSrc()));
            } else if (auto maskedStoreOp =
                           dyn_cast<triton::gcu::MaskedStoreOp>(o)) {
              if (useLoadStoreInstrOps.contains(&o) &&
                  (!disableLoadStroreInstrOptimize ||
                   !maskedStoreOp.getMask())) {
                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
                v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
                if (iterArgIndexMap.contains(&o)) {
                  auto vTy = cast<VectorType>(v.getType());
                  if (vTy.getElementType().isInteger(64) &&
                      vTy.getNumElements() == 512)
                    generateSplitTarStore(b, builder, v,
                                          iterArgValues[iterArgIndexMap[&o]],
                                          tarStride, loc);
                  else
                    b.tarStore(v, iterArgValues[iterArgIndexMap[&o]],
                               tarStride);
                } else {
                  assert(isSmallSize);
                  auto mask = builder.create<vector::ConstantMaskOp>(
                      loc,
                      VectorType::get(ArrayRef<int64_t>{vectorLength},
                                      builder.getIntegerType(1)),
                      DenseI64ArrayAttr::get(builder.getContext(),
                                             ArrayRef<int64_t>{totalNumElems}));
                  builder.create<vector::MaskedStoreOp>(
                      loc, operandMaps[i].lookup(maskedStoreOp.getPtr()),
                      ValueRange{builder.create<arith::IndexCastOp>(
                          loc, builder.getIndexType(),
                          builder.create<arith::AddIOp>(
                              loc,
                              offsets[&o].getType().isInteger(64)
                                  ? offsets[&o]
                                  : builder.create<arith::ExtSIOp>(
                                        loc, builder.getIntegerType(64),
                                        offsets[&o]),
                              builder.create<arith::AddIOp>(
                                  loc,
                                  builder.create<arith::ConstantIntOp>(
                                      loc, builder.getIntegerType(64),
                                      i * vectorLength),
                                  builder.create<arith::IndexCastOp>(
                                      loc, builder.getIntegerType(64),
                                      loopIter))))},
                      mask, v);
                }
              } else if (offsets.contains(&o)) {
                assert(maskedStoreOp.getMask());

                auto v = operandMaps[i].lookup(maskedStoreOp.getValue());
                v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);

                auto mask = operandMaps[i].lookup(maskedStoreOp.getMask());
                if (getElementTypeOrSelf(mask.getType()).isInteger(8)) {
                  mask =
                      builder
                          .create<gcu::VectorConvertOp>(
                              loc,
                              VectorType::get(ArrayRef<int64_t>{vectorLength},
                                              builder.getIntegerType(1)),
                              mask)
                          .getResult(0);
                }
                if (isSmallSize) {
                  mask = builder.create<arith::AndIOp>(
                      loc,
                      builder.create<vector::ConstantMaskOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          DenseI64ArrayAttr::get(
                              builder.getContext(),
                              ArrayRef<int64_t>{totalNumElems})),
                      mask);
                }
                builder.create<vector::MaskedStoreOp>(
                    loc, operandMaps[i].lookup(maskedStoreOp.getPtr()),
                    ValueRange{builder.create<arith::IndexCastOp>(
                        loc, builder.getIndexType(),
                        builder.create<arith::AddIOp>(
                            loc,
                            offsets[&o].getType().isInteger(64)
                                ? offsets[&o]
                                : builder.create<arith::ExtSIOp>(
                                      loc, builder.getIntegerType(64),
                                      offsets[&o]),
                            builder.create<arith::AddIOp>(
                                loc,
                                builder.create<arith::ConstantIntOp>(
                                    loc, builder.getIntegerType(64),
                                    i * vectorLength),
                                builder.create<arith::IndexCastOp>(
                                    loc, builder.getIntegerType(64),
                                    loopIter))))},
                    mask, v);
              } else if (perVecContiguousOps.contains(&o)) {
                simplifyStoreToMaskedStore(
                    maskedStoreOp, builder, operandMaps[i], vectorLength,
                    needCvtDataLayout, isSmallSize, totalNumElems,
                    loopUnrollCount, loopIter, i,
                    !useLoadStoreInstrOps.empty());
              } else {
                handleMaskedStoreOp(maskedStoreOp, builder, operandMaps[i],
                                    vectorLength, needCvtDataLayout);
              }
            } else if (auto bitcastOp = dyn_cast<triton::BitcastOp>(o)) {
              handleBitcastOp(bitcastOp, builder, operandMaps[i], vectorLength);
            } else if (auto splatOp = dyn_cast<triton::SplatOp>(o)) {
              if (i == 0) {
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                handleSplatOp(splatOp, builder, operandMaps[i], vectorLength,
                              needCvtDataLayout);
              } else {
                operandMaps[i].map(splatOp.getResult(),
                                   operandMaps[0].lookup(splatOp.getResult()));
              }
            } else if (auto constantOp = dyn_cast<arith::ConstantOp>(o)) {
              if (i == 0) {
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                handleConstantOp(constantOp, builder, operandMaps[i],
                                 vectorLength, needCvtDataLayout);
              } else {
                operandMaps[i].map(
                    constantOp.getResult(),
                    operandMaps[0].lookup(constantOp.getResult()));
              }
            } else if (auto externElementwiseOp =
                           dyn_cast<triton::ExternElementwiseOp>(o)) {
              handleExternElementwiseOp(externElementwiseOp, builder,
                                        operandMaps[i], vectorLength);
            } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(o)) {
              if (i == 0) {
                OpBuilder::InsertionGuard guard(builder);
                builder.restoreInsertionPoint(insertPoint);
                auto elementTy =
                    makeRangeOp.getResult().getType().getElementType();
                step = builder.create<vector::BroadcastOp>(
                    loc,
                    VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
                    builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                         vectorLength));
              }
              operandMaps[i].map(makeRangeOp.getResult(),
                                 iterArgValues[iterArgIndexMap[&o]]);
              iterArgValues[iterArgIndexMap[&o]] =
                  rewriter.create<arith::AddIOp>(
                      loc, iterArgValues[iterArgIndexMap[&o]], step);
            } else {
              handleCommonOp(o, builder, operandMaps[i], vectorLength,
                             needCvtDataLayout);
            }
            ++cur;
          }
          if (++i != loopUnrollCount) {
            cur = iter;
          }
        }
      }

      if (auto yieldOp = cast<triton::gcu::YieldOp>(
              op.getRegion().back().getTerminator())) {
        for (unsigned i = 0; i < loopUnrollCount; ++i) {
          for (unsigned j = 0, k = 0; j < yieldOp.getNumOperands(); ++j) {
            auto v = operandMaps[i].lookup(yieldOp.getOperand(j));
            if (outputDescs[j].useAlloca) {
              Value offset = computeAllocaOffset(i);
              builder.create<vector::StoreOp>(loc, v, outputs[j],
                                              ValueRange{offset});
            } else {
              v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
              Value tarAddr = iterArgValues[numInputTarAddrs + k];
              if (isSmallSize)
                v = narrowToStoreVector(builder, loc, v);
              generateSplitTarStore(b, builder, v, tarAddr, tarStride, loc);
              iterArgValues[numInputTarAddrs + k] = tarAddr;
              ++k;
            }
          }
        }
        builder.create<scf::YieldOp>(loc, iterArgValues);
      }
    };

    auto setUnrollFullAttr = [&](scf::ForOp forOp) {
      if (!fusionDesc->unrollFull)
        return;
      auto *ctx = rewriter.getContext();
      auto loopUnrollAttr = LLVM::LoopUnrollAttr::get(
          ctx, /*disable=*/{}, /*count=*/{}, /*runtimeDisable=*/{},
          /*full=*/BoolAttr::get(ctx, true), /*followupUnrolled=*/{},
          /*followupRemainder=*/{}, /*followupAll=*/{});
      auto loopAnnotation = LLVM::LoopAnnotationAttr::get(
          ctx, /*disableNonforced=*/{}, /*vectorize=*/{}, /*interleave=*/{},
          /*unroll=*/loopUnrollAttr, /*unrollAndJam=*/{}, /*licm=*/{},
          /*distribute=*/{}, /*pipeline=*/{}, /*peeled=*/{}, /*unswitch=*/{},
          /*mustProgress=*/{}, /*isVectorized=*/{}, /*startLoc=*/{},
          /*endLoc=*/{}, /*parallelAccesses=*/{});
      forOp->setAttr(
          StringAttr::get(ctx, LLVM::LoopAnnotationAttr::getMnemonic()),
          loopAnnotation);
    };

    if (!useLoadStoreInstrOps.empty()) {
      auto step = rewriter.create<arith::ConstantIndexOp>(
          loc, vectorLength * loopUnrollCount);
      auto lowerBound = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto upperBound =
          rewriter.create<arith::ConstantIndexOp>(loc, totalNumElems);

      if (mask) {
        rewriter.create<scf::IfOp>(
            loc, mask,
            [&](OpBuilder &builder, Location loc) {
              auto forOp = builder.create<scf::ForOp>(
                  loc, lowerBound, upperBound, step, initValues, loopBody);
              setUnrollFullAttr(forOp);
              builder.create<scf::YieldOp>(loc);
            },
            [&](OpBuilder &builder, Location loc) {
              disableLoadStroreInstrOptimize = true;
              auto forOp = builder.create<scf::ForOp>(
                  loc, lowerBound, upperBound, step, initValues, loopBody);
              setUnrollFullAttr(forOp);
              builder.create<scf::YieldOp>(loc);
            });
      } else {
        auto forOp = rewriter.create<scf::ForOp>(loc, lowerBound, upperBound,
                                                 step, initValues, loopBody);
        setUnrollFullAttr(forOp);
      }
    } else {
      auto forOp = rewriter.create<scf::ForOp>(
          loc, rewriter.create<arith::ConstantIndexOp>(loc, 0),
          rewriter.create<arith::ConstantIndexOp>(
              loc,
              ceil<unsigned>(totalNumElems, vectorLength * loopUnrollCount)),
          rewriter.create<arith::ConstantIndexOp>(loc, 1), initValues,
          loopBody);
      setUnrollFullAttr(forOp);
    }

    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      if (outputDescs[i].useLocalStore) {
        auto result = op.getResult(i);
        auto resultType = dyn_cast<MemRefType>(
            getTypeConverter()->convertType(result.getType()));
        auto lastUser = userAnalysis.getLastUser(result);
        Value localMemory = syncAllocOp(rewriter, loc, lastUser, userAnalysis,
                                        replaced2Origin, resultType);
        ConfigMatrixStoreLocal(rewriter, loc, resultType, localMemory,
                               results[i]);
        results[i] = localMemory;
      }
    }

    leaveTritionOp(rewriter, op);
    if (results.empty()) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, results);
    }
    return success();
  }

private:
  Value computeScalarOffsetForVector(Value offset, OpBuilder &builder,
                                     const IRMapping &argMap,
                                     unsigned unrollIdx,
                                     unsigned vectorLength) const {
    IRMapping mapper;
    SetVector<Operation *> visited;
    [[maybe_unused]] bool sawMakeRange = false;
    DenseSet<Operation *> onStack;
    SmallVector<std::pair<Operation *, unsigned>> dfsStack;
    if (auto defOp = offset.getDefiningOp()) {
      dfsStack.push_back({defOp, 0});
      onStack.insert(defOp);
    }
    while (!dfsStack.empty()) {
      Operation *cur = dfsStack.back().first;
      unsigned &operandIdx = dfsStack.back().second;
      if (operandIdx < cur->getNumOperands()) {
        auto operand = cur->getOperand(operandIdx++);
        if (auto child = operand.getDefiningOp()) {
          if (!visited.contains(child) && !onStack.contains(child)) {
            onStack.insert(child);
            dfsStack.push_back({child, 0});
          }
        }
        continue;
      }
      visited.insert(cur);
      onStack.erase(cur);
      dfsStack.pop_back();
    }
    for (auto op : visited) {
      if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
        sawMakeRange = true;
        auto startIdx = makeRangeOp.getStart();
        auto elementTy = makeRangeOp.getResult().getType().getElementType();
        auto loc = op->getLoc();
        Value start =
            builder.create<arith::ConstantIntOp>(loc, elementTy, startIdx)
                .getResult();
        if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
          auto totalNumElems = triton::gcu::getTotalElemsPerThread(
              makeRangeOp.getResult().getType());
          start = builder.create<arith::AddIOp>(
              loc,
              builder.create<arith::MulIOp>(
                  loc,
                  builder.create<arith::IndexCastOp>(
                      loc, elementTy,
                      getWarpIds(builder, loc, makeRangeOp.getType()).front()),
                  builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                       totalNumElems)),
              start);
          if (unrollIdx > 0) {
            start = builder.create<arith::AddIOp>(
                loc, start,
                builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                     unrollIdx * vectorLength));
          }
        }
        mapper.map(makeRangeOp.getResult(), start);
      } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
        mapper.map(splatOp.getResult(), argMap.lookupOrNull(splatOp.getSrc()));
      } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
        auto denseAttr = dyn_cast<DenseElementsAttr>(constantOp.getValue());
        assert(denseAttr && denseAttr.isSplat() &&
               "Constant op should have a splat value");
        mapper.map(constantOp.getResult(),
                   builder.create<arith::ConstantOp>(
                       constantOp.getLoc(), denseAttr.getElementType(),
                       denseAttr.getSplatValue<TypedAttr>()));
      } else {
        auto cloneOp = builder.clone(*op, mapper);
        for (auto [result, newResult] :
             llvm::zip(op->getResults(), cloneOp->getResults())) {
          newResult.setType(getElementTypeOrSelf(result));
          mapper.map(result, newResult);
        }
      }
    }
    assert((unrollIdx == 0 || sawMakeRange) &&
           "per-vector load with unrollIdx > 0 must be MakeRange-indexed");
    return mapper.lookupOrNull(offset);
  }

  Value simplifyLoadToBroadcast(triton::gcu::MaskedLoadOp maskedLoadOp,
                                OpBuilder &builder, const IRMapping &argMap,
                                unsigned vectorIndex,
                                unsigned vectorLength) const {
    auto offset = maskedLoadOp.getOffset();
    auto scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                     vectorIndex, vectorLength);
    auto loc = maskedLoadOp.getLoc();
    if (!scalarOffset) {
      maskedLoadOp.emitWarning()
          << "simplifyLoadToBroadcast: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }
    auto v = builder.create<memref::LoadOp>(
        loc, argMap.lookup(maskedLoadOp.getPtr()),
        ValueRange{builder
                       .create<arith::IndexCastOp>(loc, builder.getIndexType(),
                                                   scalarOffset)
                       .getResult()});
    return builder.create<vector::BroadcastOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, v.getType()), v);
  }

  // Emit a vector.maskedload for a per-vector-contiguous load.  Within each
  // OACC vector the `vectorLength` elements are contiguous (stride-1), so
  // element 0 of the offset vector gives a valid scalar base address.
  Value simplifyLoadToMaskedLoad(triton::gcu::MaskedLoadOp maskedLoadOp,
                                 OpBuilder &builder, const IRMapping &argMap,
                                 unsigned vectorLength, bool needCvtDataLayout,
                                 bool isSmallSize, unsigned totalNumElems,
                                 unsigned loopCnt, Value loopIter,
                                 unsigned unrollIdx, bool tarMode) const {
    auto offset = maskedLoadOp.getOffset();
    auto loc = maskedLoadOp.getLoc();
    Value scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                      unrollIdx, vectorLength);
    if (!scalarOffset) {
      maskedLoadOp.emitWarning()
          << "simplifyLoadToMaskedLoad: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }

    // When the outer loop has more than one iteration, advance the scalar
    // base across outer iterations.  loopIter's unit depends on the region's
    // loop mode: in Tar mode (useLoadStoreInstrOps non-empty) loopIter is an
    // element offset and is added directly; in Normal mode it is an iteration
    // index scaled by the per-trip block size (vectorLength * loopCnt).
    if (totalNumElems > vectorLength * loopCnt) {
      auto offTy = scalarOffset.getType();
      Value iterAdvance;
      if (tarMode) {
        iterAdvance = builder.create<arith::IndexCastOp>(loc, offTy, loopIter);
      } else {
        Value tripStride = builder.create<arith::ConstantIntOp>(
            loc, offTy, static_cast<int64_t>(vectorLength) * loopCnt);
        iterAdvance = builder.create<arith::MulIOp>(
            loc, tripStride,
            builder.create<arith::IndexCastOp>(loc, offTy, loopIter));
      }
      scalarOffset =
          builder.create<arith::AddIOp>(loc, scalarOffset, iterAdvance);
    }

    auto elementTy = maskedLoadOp.getResult().getType().getElementType();
    auto vecTy = VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);

    auto mask = maskedLoadOp.getMask();
    Value maskVal;
    if (!mask) {
      maskVal = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{vectorLength}));
    } else {
      maskVal = argMap.lookup(mask);
      if (needCvtDataLayout) {
        maskVal = builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          maskVal)
                      .getResult(0);
      }
      if (isSmallSize) {
        if (getElementTypeOrSelf(maskVal.getType()).isInteger(8)) {
          maskVal = builder
                        .create<gcu::VectorConvertOp>(
                            loc,
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(1)),
                            maskVal)
                        .getResult(0);
        }
        maskVal = builder.create<arith::AndIOp>(
            loc, maskVal,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{totalNumElems})));
      }
    }

    auto other = maskedLoadOp.getOther();
    Value passthru = other
                         ? argMap.lookup(other)
                         : builder.create<arith::ConstantOp>(
                               loc, DenseElementsAttr::get(
                                        vecTy, builder.getZeroAttr(elementTy)));
    return builder.create<vector::MaskedLoadOp>(
        loc, vecTy, argMap.lookup(maskedLoadOp.getPtr()),
        ValueRange{builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), scalarOffset)},
        maskVal, passthru);
  }

  // Emit a vector.maskedstore for a per-vector-contiguous store.  Within each
  // OACC vector the `vectorLength` elements are contiguous (stride-1), so
  // element 0 of the offset vector gives a valid scalar base address.
  void simplifyStoreToMaskedStore(triton::gcu::MaskedStoreOp maskedStoreOp,
                                  OpBuilder &builder, const IRMapping &argMap,
                                  unsigned vectorLength, bool needCvtDataLayout,
                                  bool isSmallSize, unsigned totalNumElems,
                                  unsigned loopCnt, Value loopIter,
                                  unsigned unrollIdx, bool tarMode) const {
    auto offset = maskedStoreOp.getOffset();
    auto loc = maskedStoreOp.getLoc();
    Value scalarOffset = computeScalarOffsetForVector(offset, builder, argMap,
                                                      unrollIdx, vectorLength);
    if (!scalarOffset) {
      maskedStoreOp.emitWarning()
          << "simplifyStoreToMaskedStore: offset scalarization returned null, "
             "falling back to vector.extract";
      auto vecOffset = argMap.lookup(offset);
      scalarOffset = builder.create<vector::ExtractOp>(loc, vecOffset,
                                                       ArrayRef<int64_t>{0});
    }

    if (totalNumElems > vectorLength * loopCnt) {
      auto offTy = scalarOffset.getType();
      Value iterAdvance;
      if (tarMode) {
        iterAdvance = builder.create<arith::IndexCastOp>(loc, offTy, loopIter);
      } else {
        Value tripStride = builder.create<arith::ConstantIntOp>(
            loc, offTy, static_cast<int64_t>(vectorLength) * loopCnt);
        iterAdvance = builder.create<arith::MulIOp>(
            loc, tripStride,
            builder.create<arith::IndexCastOp>(loc, offTy, loopIter));
      }
      scalarOffset =
          builder.create<arith::AddIOp>(loc, scalarOffset, iterAdvance);
    }

    auto mask = maskedStoreOp.getMask();
    Value maskVal;
    if (!mask) {
      maskVal = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{vectorLength}));
    } else {
      maskVal = argMap.lookup(mask);
      if (needCvtDataLayout) {
        maskVal = builder
                      .create<gcu::VectorConvertOp>(
                          loc,
                          VectorType::get(ArrayRef<int64_t>{vectorLength},
                                          builder.getIntegerType(1)),
                          maskVal)
                      .getResult(0);
      }
      if (isSmallSize) {
        if (getElementTypeOrSelf(maskVal.getType()).isInteger(8)) {
          maskVal = builder
                        .create<gcu::VectorConvertOp>(
                            loc,
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(1)),
                            maskVal)
                        .getResult(0);
        }
        maskVal = builder.create<arith::AndIOp>(
            loc, maskVal,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{totalNumElems})));
      }
    }

    auto v = argMap.lookup(maskedStoreOp.getValue());
    v = convertI1ToI8VectorForStore(builder, loc, v, vectorLength);
    builder.create<vector::MaskedStoreOp>(
        loc, argMap.lookup(maskedStoreOp.getPtr()),
        ValueRange{builder.create<arith::IndexCastOp>(
            loc, builder.getIndexType(), scalarOffset)},
        maskVal, v);
  }

  DenseSet<Operation *> trySimplifyLoadStore(
      triton::gcu::ElementwiseFusionRegionOp op, OpBuilder &builder,
      llvm::MapVector<Operation *, Value> &map, Value &mask,
      unsigned vectorLength, DenseSet<Operation *> &perVecContiguousOps) const {
    SetVector<Value> offsets;
    SetVector<Value> masks;
    llvm::MapVector<Value, SmallVector<Operation *>> mask2ops;
    DenseSet<Operation *> useLoadStoreInstrOps;
    mask = nullptr;
    perVecContiguousOps.clear();

    auto processLoadStoreOp = [&](auto loadStoreOp) {
      Value offset = loadStoreOp.getOffset();
      int64_t totalNumElems =
          triton::gcu::getTotalElemsPerThread(offset.getType());
      if (isOffsetContiguousWithin(loadStoreOp, totalNumElems)) {
        map[loadStoreOp] = offset;
        offsets.insert(offset);
        if (auto mask = loadStoreOp.getMask()) {
          masks.insert(mask);
          mask2ops[mask].push_back(loadStoreOp);
        } else {
          useLoadStoreInstrOps.insert(loadStoreOp);
        }
      } else if (isOffsetContiguousWithin(loadStoreOp, vectorLength)) {
        perVecContiguousOps.insert(loadStoreOp);
      }
    };

    for (auto &o : *op.getBody()) {
      if (auto maskedLoadOp = dyn_cast<triton::gcu::MaskedLoadOp>(o))
        processLoadStoreOp(maskedLoadOp);
      else if (auto maskedStoreOp = dyn_cast<triton::gcu::MaskedStoreOp>(o))
        processLoadStoreOp(maskedStoreOp);
    }

    auto generator =
        [&builder](
            const std::set<Operation *, bool (*)(mlir::Operation *,
                                                 mlir::Operation *)> &opSet,
            IRMapping &mapper) {
          for (auto op : opSet) {
            if (mapper.contains(op->getResult(0))) {
              continue;
            }
            if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(op)) {
              auto startIdx = makeRangeOp.getStart();
              auto elementTy =
                  makeRangeOp.getResult().getType().getElementType();
              auto loc = op->getLoc();
              Value start =
                  builder.create<arith::ConstantIntOp>(loc, elementTy, startIdx)
                      .getResult();
              if (!getSlicedAxies(makeRangeOp.getType()).empty()) {
                auto totalNumElems = triton::gcu::getTotalElemsPerThread(
                    makeRangeOp.getResult().getType());
                start = builder.create<arith::AddIOp>(
                    loc,
                    builder.create<arith::MulIOp>(
                        loc,
                        builder.create<arith::IndexCastOp>(
                            loc, elementTy,
                            getWarpIds(builder, loc, makeRangeOp.getType())
                                .front()),
                        builder.create<arith::ConstantIntOp>(loc, elementTy,
                                                             totalNumElems)),
                    start);
              }
              mapper.map(makeRangeOp.getResult(), start);
            } else if (auto splatOp = dyn_cast<triton::SplatOp>(op)) {
              mapper.map(splatOp.getResult(),
                         mapper.lookupOrNull(splatOp.getSrc()));
            } else if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
              auto loc = op->getLoc();
              auto result = constantOp.getResult();
              mapper.map(result,
                         builder.create<arith::ConstantOp>(
                             loc, getElementTypeOrSelf(result),
                             dyn_cast<DenseElementsAttr>(constantOp.getValue())
                                 .getSplatValue<TypedAttr>()));
            } else if (auto cmpOp = dyn_cast<arith::CmpIOp>(op)) {
              auto loc = op->getLoc();
              auto result = cmpOp.getResult();
              auto lhs = mapper.lookupOrNull(cmpOp.getLhs());
              auto rhs = mapper.lookupOrNull(cmpOp.getRhs());
              auto predicate = cmpOp.getPredicate();
              if (predicate == arith::CmpIPredicate::slt ||
                  predicate == arith::CmpIPredicate::ult ||
                  predicate == arith::CmpIPredicate::sle ||
                  predicate == arith::CmpIPredicate::ule) {
                Value size = builder.create<arith::SubIOp>(loc, rhs, lhs);
                if (predicate == arith::CmpIPredicate::sle ||
                    predicate == arith::CmpIPredicate::ule) {
                  size = builder.create<arith::AddIOp>(
                      loc, size,
                      builder
                          .create<arith::ConstantIntOp>(loc, size.getType(), 1)
                          .getResult());
                }
                auto totalNumElems = triton::gcu::getTotalElemsPerThread(
                    cmpOp.getResult().getType());
                mapper.map(result, builder.create<arith::CmpIOp>(
                                       loc, arith::CmpIPredicate::sge, size,
                                       builder.create<arith::ConstantIntOp>(
                                           loc, getElementTypeOrSelf(size),
                                           totalNumElems)));
              } else {
                mapper.map(result, builder.create<arith::CmpIOp>(loc, predicate,
                                                                 lhs, rhs));
              }
            } else {
              auto cloneOp = builder.clone(*op, mapper);
              for (auto [result, newResult] :
                   llvm::zip(op->getResults(), cloneOp->getResults())) {
                newResult.setType(getElementTypeOrSelf(result));
                mapper.map(result, newResult);
              }
            }
          }
        };

    std::set<Operation *, bool (*)(Operation *, Operation *)> opSet(
        [](Operation *lhs, Operation *rhs) {
          return lhs->isBeforeInBlock(rhs);
        });
    std::queue<Operation *> workList;
    IRMapping mapper;
    auto isSupportedCmpOp = [](Operation *op) {
      auto cmpOp = dyn_cast<arith::CmpIOp>(op);
      if (!cmpOp) {
        return false;
      }
      auto predicate = cmpOp.getPredicate();
      return (predicate == arith::CmpIPredicate::slt ||
              predicate == arith::CmpIPredicate::ult ||
              predicate == arith::CmpIPredicate::sge ||
              predicate == arith::CmpIPredicate::uge) &&
             isa_and_nonnull<arith::ConstantOp, triton::SplatOp>(
                 cmpOp.getRhs().getDefiningOp());
    };
    auto isSupportedOp = [](Operation *op) {
      return isa<arith::AndIOp, arith::AddIOp, arith::ExtSIOp, arith::ExtUIOp,
                 arith::ConstantOp, triton::MakeRangeOp, triton::SplatOp>(op);
    };
    for (auto mask : masks) {
      if (auto defOp = mask.getDefiningOp()) {
        if (!isSupportedCmpOp(defOp) && !isSupportedOp(defOp)) {
          mask2ops.erase(mask);
          continue;
        }
        workList.push(defOp);
        opSet.insert(defOp);
        while (!workList.empty()) {
          auto o = workList.front();
          workList.pop();
          for (auto operand : o->getOperands()) {
            auto defOp = operand.getDefiningOp();
            if (defOp) {
              if (isSupportedOp(defOp)) {
                workList.push(defOp);
                opSet.insert(defOp);
              } else if (isSupportedCmpOp(defOp)) {
                workList.push(defOp);
                opSet.insert(defOp);
              } else {
                mask2ops.erase(mask);
                opSet.clear();
                continue;
              }
            } else {
              if (!isa<IntegerType>(operand.getType())) {
                mask2ops.erase(mask);
                opSet.clear();
                continue;
              }
              mapper.map(
                  operand,
                  op.getOperand(cast<BlockArgument>(operand).getArgNumber()));
            }
          }
        }
        generator(opSet, mapper);
        opSet.clear();
      }
    }

    for (auto offset : offsets) {
      auto defOp = offset.getDefiningOp();
      assert(defOp);
      workList.push(defOp);
      opSet.insert(defOp);
      while (!workList.empty()) {
        auto o = workList.front();
        workList.pop();
        for (auto operand : o->getOperands()) {
          auto defOp = operand.getDefiningOp();
          if (defOp) {
            workList.push(defOp);
            opSet.insert(defOp);
          } else {
            assert(isa<IntegerType>(operand.getType()));
            assert(isa<BlockArgument>(operand));
            mapper.map(
                operand,
                op.getOperand(cast<BlockArgument>(operand).getArgNumber()));
          }
        }
      }
      generator(opSet, mapper);
      opSet.clear();
    }
    for (auto [k, v] : map) {
      map[k] = mapper.lookup(v);
    }
    for (auto const &[k, ops] : mask2ops) {
      if (mask) {
        mask = builder.create<arith::AndIOp>(op.getLoc(), mask,
                                             mapper.lookupOrNull(k));
      } else {
        mask = mapper.lookupOrNull(k);
      }
      for (auto op : ops) {
        useLoadStoreInstrOps.insert(op);
      }
    }
    return useLoadStoreInstrOps;
  }

  unsigned getSplitLength(unsigned splitIndex, unsigned vectorLength,
                          unsigned maxVectorLength) const {
    unsigned remainingLength = vectorLength - splitIndex * maxVectorLength;
    return std::min(remainingLength, maxVectorLength);
  }

  Value convertLoadStoreMask(OpBuilder &builder, Value mask,
                             unsigned vectorLength, Location loc) const {
    if (!mask) {
      return mask;
    }

    auto defOp = mask.getDefiningOp();
    auto maskElementType = getElementTypeOrSelf(mask.getType());
    if (isa<arith::AndIOp, arith::XOrIOp, arith::OrIOp>(defOp)) {
      auto ip = builder.saveInsertionPoint();
      builder.setInsertionPoint(defOp);
      auto lhs = builder
                     .create<gcu::VectorConvertOp>(
                         loc,
                         VectorType::get(ArrayRef<int64_t>{vectorLength},
                                         builder.getIntegerType(8)),
                         defOp->getOperand(0))
                     .getResult(0);
      auto rhs = builder
                     .create<gcu::VectorConvertOp>(
                         loc,
                         VectorType::get(ArrayRef<int64_t>{vectorLength},
                                         builder.getIntegerType(8)),
                         defOp->getOperand(1))
                     .getResult(0);

      Operation *clonedOp = builder.clone(*defOp);
      clonedOp->setOperand(0, lhs);
      clonedOp->setOperand(1, rhs);
      clonedOp->getResult(0).setType(VectorType::get(
          ArrayRef<int64_t>{vectorLength}, builder.getIntegerType(8)));
      builder.restoreInsertionPoint(ip);
      mask = builder
                 .create<gcu::VectorConvertOp>(
                     loc,
                     VectorType::get(ArrayRef<int64_t>{vectorLength},
                                     maskElementType),
                     clonedOp->getResult(0))
                 .getResult(0);
    }
    return mask;
  }

  Value loadSmallSizeTarInput(triton::gcu::TritonGCUBuilder &b,
                              OpBuilder &builder, Type elementTy,
                              unsigned vectorLength, Value &tarAddr,
                              const Value &tarStride, Location loc) const {
    unsigned maxVectorLength =
        4 * kOaccSizeInBytes / mlir::triton::gcu::getBpe(elementTy);
    if (vectorLength <= maxVectorLength) {
      return b.tarLoad(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy), tarAddr,
          tarStride);
    }

    auto chunkTy =
        VectorType::get(ArrayRef<int64_t>{maxVectorLength}, elementTy);
    Value chunk = b.tarLoad(chunkTy, tarAddr, tarStride);
    unsigned numParts = vectorLength / maxVectorLength;
    SmallVector<Value> parts;
    parts.push_back(chunk);
    auto zeroChunk = builder.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(chunkTy, builder.getZeroAttr(elementTy)));
    for (unsigned p = 1; p < numParts; ++p)
      parts.push_back(zeroChunk);
    SmallVector<Type> resultTypes;
    resultTypes.push_back(
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
    return builder.create<gcu::VectorConvertOp>(loc, resultTypes, parts)
        .getResult(0);
  }

  Value generateSplitTarLoad(triton::gcu::TritonGCUBuilder &b,
                             OpBuilder &builder, Type elementTy,
                             unsigned vectorLength, Value &tarAddr,
                             const Value &tarStride, Location loc) const {
    unsigned bpe = mlir::triton::gcu::getBpe(elementTy);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / bpe;

    if (vectorLength <= maxVectorLength) {
      return b.tarLoad(
          VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy), tarAddr,
          tarStride);
    }

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;
    SmallVector<Value> loadValues;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      Value v =
          b.tarLoad(VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy),
                    tarAddr, tarStride);
      loadValues.push_back(v);
    }

    SmallVector<Type> resultTypes;
    resultTypes.push_back(
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy));
    return builder.create<gcu::VectorConvertOp>(loc, resultTypes, loadValues)
        .getResult(0);
  }

  void generateSplitTarStore(triton::gcu::TritonGCUBuilder &b,
                             OpBuilder &builder, Value v, Value &tarAddr,
                             const Value &tarStride, Location loc) const {
    VectorType vectorType = cast<VectorType>(v.getType());
    Type elementTy = vectorType.getElementType();
    unsigned vectorLength = vectorType.getNumElements();

    unsigned bpe = mlir::triton::gcu::getBpe(elementTy);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / bpe;

    if (vectorLength <= maxVectorLength) {
      return b.tarStore(v, tarAddr, tarStride);
    }

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;
    SmallVector<Type> resultTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      resultTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy));
    }

    auto convertOp = builder.create<gcu::VectorConvertOp>(loc, resultTypes, v);

    for (unsigned i = 0; i < numSplits; ++i) {
      b.tarStore(convertOp.getResult(i), tarAddr, tarStride);
    }
    return;
  }

  void generateSplitMaskedLoadOp(triton::gcu::MaskedLoadOp op, Value mask,
                                 OpBuilder &builder, IRMapping &map,
                                 unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto elementTy = op.getResult().getType().getElementType();
    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    auto other = op.getOther();
    auto offset = map.lookup(op.getOffset());
    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);

    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (numElements < vectorLength) {
        Value constantMask = builder.create<vector::ConstantMaskOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(1)),
            DenseI64ArrayAttr::get(builder.getContext(),
                                   ArrayRef<int64_t>{numElements}));
        constantMask = builder
                           .create<gcu::VectorConvertOp>(
                               op.getLoc(),
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(8)),
                               constantMask)
                           .getResult(0);
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(8)),
                       mask)
                   .getResult(0);
        mask = builder.create<arith::AndIOp>(loc, mask, constantMask);
      }
    }

    mask = builder
               .create<gcu::VectorConvertOp>(
                   op.getLoc(),
                   VectorType::get(ArrayRef<int64_t>{vectorLength},
                                   builder.getIntegerType(8)),
                   mask)
               .getResult(0);
    auto elemBitWidth = elementTy.getIntOrFloatBitWidth();
    mask = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(elemBitWidth)),
        mask);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> offsetTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      offsetTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, offsetElementType));
    }
    auto offsets =
        builder.create<gcu::VectorConvertOp>(loc, offsetTypes, offset);

    SmallVector<Type> maskTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      maskTypes.push_back(VectorType::get(
          ArrayRef<int64_t>{currentLen}, builder.getIntegerType(elemBitWidth)));
    }
    auto masks = builder.create<gcu::VectorConvertOp>(loc, maskTypes, mask);

    SmallVector<Value> passThruValues;
    if (other) {
      Value otherVec = map.lookup(other);
      SmallVector<Type> passThruTypes;
      for (unsigned i = 0; i < numSplits; ++i) {
        unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
        passThruTypes.push_back(
            VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy));
      }
      auto passThruSplitOp =
          builder.create<gcu::VectorConvertOp>(loc, passThruTypes, otherVec);
      for (unsigned i = 0; i < numSplits; ++i) {
        passThruValues.push_back(passThruSplitOp.getResult(i));
      }
    } else {
      Value zeroScalar =
          triton::gcu::createConstantZero(builder, loc, elementTy);
      for (unsigned i = 0; i < numSplits; ++i) {
        unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
        auto splitVectorType =
            VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy);
        passThruValues.push_back(builder.create<vector::BroadcastOp>(
            loc, splitVectorType, zeroScalar));
      }
    }

    SmallVector<Value> loadValues;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      auto splitVectorType =
          VectorType::get(ArrayRef<int64_t>{currentLen}, elementTy);

      mask = builder
                 .create<gcu::VectorConvertOp>(
                     op.getLoc(),
                     VectorType::get(ArrayRef<int64_t>{currentLen},
                                     builder.getIntegerType(1)),
                     masks.getResult(i))
                 .getResult(0);

      Value gatherResult = builder.create<vector::GatherOp>(
          loc, splitVectorType, map.lookup(op.getPtr()), ValueRange{zero},
          offsets.getResult(i), mask, passThruValues[i]);
      loadValues.push_back(gatherResult);
    }

    auto mergedResult =
        builder.create<gcu::VectorConvertOp>(loc, vectorType, loadValues)
            .getResult(0);
    map.map(op.getResult(), mergedResult);
  }

  void generateSplitMaskedStoreOp(triton::gcu::MaskedStoreOp op, Value mask,
                                  OpBuilder &builder, IRMapping &map,
                                  unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto offset = map.lookup(op.getOffset());
    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());

    auto v = map.lookup(op.getValue());
    auto valueElementType = getElementTypeOrSelf(v.getType());
    auto elemBitWidth = valueElementType.getIntOrFloatBitWidth();

    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (numElements < vectorLength) {
        Value constantMask = builder.create<vector::ConstantMaskOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(1)),
            DenseI64ArrayAttr::get(builder.getContext(),
                                   ArrayRef<int64_t>{numElements}));
        constantMask = builder
                           .create<gcu::VectorConvertOp>(
                               op.getLoc(),
                               VectorType::get(ArrayRef<int64_t>{vectorLength},
                                               builder.getIntegerType(8)),
                               constantMask)
                           .getResult(0);
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(8)),
                       mask)
                   .getResult(0);
        mask = builder.create<arith::AndIOp>(loc, mask, constantMask);
      }
    }

    mask = builder
               .create<gcu::VectorConvertOp>(
                   op.getLoc(),
                   VectorType::get(ArrayRef<int64_t>{vectorLength},
                                   builder.getIntegerType(8)),
                   mask)
               .getResult(0);
    mask = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(elemBitWidth)),
        mask);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> offsetTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      offsetTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, offsetElementType));
    }
    auto offsets =
        builder.create<gcu::VectorConvertOp>(loc, offsetTypes, offset);

    SmallVector<Type> maskTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      maskTypes.push_back(VectorType::get(
          ArrayRef<int64_t>{currentLen}, builder.getIntegerType(elemBitWidth)));
    }
    auto masks = builder.create<gcu::VectorConvertOp>(loc, maskTypes, mask);

    SmallVector<Type> valueTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      valueTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, valueElementType));
    }
    auto values = builder.create<gcu::VectorConvertOp>(loc, valueTypes, v);

    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      mask = builder
                 .create<gcu::VectorConvertOp>(
                     op.getLoc(),
                     VectorType::get(ArrayRef<int64_t>{currentLen},
                                     builder.getIntegerType(1)),
                     masks.getResult(i))
                 .getResult(0);
      triton_gcu::compat::createVectorScatterOp(
          builder, loc, map.lookup(op.getPtr()), ValueRange{zero},
          offsets.getResult(i), mask, values.getResult(i));
    }
    return;
  }

  Operation *generateSplitSelectOp(arith::SelectOp op, OpBuilder &builder,
                                   IRMapping &map, unsigned vectorLength,
                                   unsigned maxVectorLength) const {
    auto loc = op.getLoc();

    Value condition = map.lookup(op.getCondition());
    Value trueValue = map.lookup(op.getTrueValue());
    Value falseValue = map.lookup(op.getFalseValue());

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> conditionTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      conditionTypes.push_back(VectorType::get(ArrayRef<int64_t>{currentLen},
                                               builder.getIntegerType(1)));
    }
    auto conditionSplits =
        builder.create<gcu::VectorConvertOp>(loc, conditionTypes, condition);

    SmallVector<Type> valueTypes;
    auto valueElementType = getElementTypeOrSelf(op.getResult().getType());
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      valueTypes.push_back(
          VectorType::get(ArrayRef<int64_t>{currentLen}, valueElementType));
    }
    auto trueSplits =
        builder.create<gcu::VectorConvertOp>(loc, valueTypes, trueValue);
    auto falseSplits =
        builder.create<gcu::VectorConvertOp>(loc, valueTypes, falseValue);

    SmallVector<Value> resultSplits;
    for (unsigned i = 0; i < numSplits; ++i) {
      Value resultSplit = builder.create<arith::SelectOp>(
          loc, conditionSplits.getResult(i), trueSplits.getResult(i),
          falseSplits.getResult(i));
      resultSplits.push_back(resultSplit);
    }

    auto mergedResult = builder.create<gcu::VectorConvertOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, valueElementType),
        resultSplits);
    map.map(op.getResult(), mergedResult.getResult(0));
    return mergedResult;
  }

  Operation *generateSplitExtUIOp(arith::ExtUIOp op, OpBuilder &builder,
                                  IRMapping &map, unsigned vectorLength,
                                  unsigned maxVectorLength) const {
    auto loc = op.getLoc();

    Value inValue = map.lookup(op.getIn());
    if (getElementTypeOrSelf(inValue.getType()).isInteger(8)) {
      inValue = builder
                    .create<gcu::VectorConvertOp>(
                        loc,
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        inValue)
                    .getResult(0);
    }

    inValue = builder.create<arith::ExtSIOp>(
        loc,
        VectorType::get(ArrayRef<int64_t>{vectorLength},
                        builder.getIntegerType(8)),
        inValue);

    unsigned numSplits = (vectorLength + maxVectorLength - 1) / maxVectorLength;

    SmallVector<Type> inTypes;
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      inTypes.push_back(VectorType::get(ArrayRef<int64_t>{currentLen},
                                        builder.getIntegerType(1)));
    }
    auto inSplits = builder.create<gcu::VectorConvertOp>(loc, inTypes, inValue);

    SmallVector<Value> resultSplits;
    auto outElementType = getElementTypeOrSelf(op.getOut().getType());
    for (unsigned i = 0; i < numSplits; ++i) {
      unsigned currentLen = getSplitLength(i, vectorLength, maxVectorLength);
      Value resultSplit = builder.create<arith::ExtUIOp>(
          loc, VectorType::get(ArrayRef<int64_t>{currentLen}, outElementType),
          inSplits.getResult(i));
      resultSplits.push_back(resultSplit);
    }

    auto mergedResult = builder.create<gcu::VectorConvertOp>(
        loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, outElementType),
        resultSplits);
    map.map(op.getResult(), mergedResult.getResult(0));
    return mergedResult;
  }

  void handleMaskedLoadOp(triton::gcu::MaskedLoadOp op, OpBuilder &builder,
                          IRMapping &map, unsigned vectorLength,
                          bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto offset = map.lookup(op.getOffset());
    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    auto elementTy = op.getResult().getType().getElementType();
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    auto other = op.getOther();
    auto mask = map.lookupOrNull(op.getMask());

    if (vectorLength > maxVectorLength) {
      mask = convertLoadStoreMask(builder, mask, vectorLength, loc);
      // tcle support load/store for i8/f8 with v8 oacc, so no need to split.
      if (mlir::triton::gcu::getBpe(elementTy) > 1) {
        return generateSplitMaskedLoadOp(op, mask, builder, map, vectorLength);
      }
    }

    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (needCvtDataLayout) {
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(1)),
                       mask)
                   .getResult(0);
      }
      if (numElements < vectorLength) {
        mask = builder.create<arith::AndIOp>(
            loc, mask,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{numElements})));
      }
    }
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    map.map(op.getResult(),
            builder.create<vector::GatherOp>(
                loc, vectorType, map.lookup(op.getPtr()), ValueRange{zero},
                map.lookup(op.getOffset()), mask,
                other ? map.lookup(other)
                      : builder.create<vector::BroadcastOp>(
                            loc, vectorType,
                            triton::gcu::createConstantZero(builder, loc,
                                                            elementTy))));
  }

  void handleMaskedStoreOp(triton::gcu::MaskedStoreOp op, OpBuilder &builder,
                           IRMapping &map, unsigned vectorLength,
                           bool needCvtDataLayout) const {
    auto v = map.lookup(op.getValue());
    auto offset = map.lookup(op.getOffset());
    auto valueElementType = getElementTypeOrSelf(v.getType());
    auto offsetElementType = getElementTypeOrSelf(offset.getType());
    unsigned offsetBpe = mlir::triton::gcu::getBpe(offsetElementType);
    unsigned maxVectorLength = 4 * kOaccSizeInBytes / offsetBpe;

    auto loc = op.getLoc();
    auto zero = builder.create<arith::ConstantIndexOp>(loc, 0);
    auto mask = map.lookupOrNull(op.getMask());

    if (vectorLength > maxVectorLength) {
      mask = convertLoadStoreMask(builder, mask, vectorLength, loc);
      // tcle support load/store for i8/f8 with v8 oacc, so no need to split.
      if (mlir::triton::gcu::getBpe(valueElementType) > 1) {
        return generateSplitMaskedStoreOp(op, mask, builder, map, vectorLength);
      }
    }

    auto numElements =
        triton::gcu::getTotalElemsPerThread(op.getOffset().getType());
    if (!mask) {
      mask = builder.create<vector::ConstantMaskOp>(
          loc,
          VectorType::get(ArrayRef<int64_t>{vectorLength},
                          builder.getIntegerType(1)),
          DenseI64ArrayAttr::get(builder.getContext(),
                                 ArrayRef<int64_t>{numElements < vectorLength
                                                       ? numElements
                                                       : vectorLength}));
    } else {
      if (needCvtDataLayout) {
        mask = builder
                   .create<gcu::VectorConvertOp>(
                       op.getLoc(),
                       VectorType::get(ArrayRef<int64_t>{vectorLength},
                                       builder.getIntegerType(1)),
                       mask)
                   .getResult(0);
      }
      if (numElements < vectorLength) {
        mask = builder.create<arith::AndIOp>(
            loc, mask,
            builder.create<vector::ConstantMaskOp>(
                loc,
                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                builder.getIntegerType(1)),
                DenseI64ArrayAttr::get(builder.getContext(),
                                       ArrayRef<int64_t>{numElements})));
      }
    }
    if (dyn_cast<VectorType>(v.getType()).getElementType().isInteger(1)) {
      OpBuilder::InsertionGuard guard(builder);
      auto defOp = v.getDefiningOp();
      assert(defOp);
      builder.setInsertionPointAfter(defOp);
      v = builder
              .create<gcu::VectorConvertOp>(
                  loc,
                  VectorType::get(ArrayRef<int64_t>{vectorLength},
                                  builder.getIntegerType(8)),
                  v)
              .getResult(0);
    }
    triton_gcu::compat::createVectorScatterOp(
        builder, loc, map.lookup(op.getPtr()), ValueRange{zero},
        map.lookup(op.getOffset()), mask, v);
  }

  void handleConstantOp(arith::ConstantOp op, OpBuilder &builder,
                        IRMapping &map, unsigned vectorLength,
                        bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto elementTy = cast<TensorType>(op.getType()).getElementType();
    auto vectorType =
        VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
    Value v;
    if (elementTy.isInteger(1)) {
      if (needCvtDataLayout) {
        v = builder.create<vector::BroadcastOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(8)),
            builder.create<arith::ExtUIOp>(
                loc, builder.getIntegerType(8),
                builder.create<arith::ConstantOp>(
                    loc, elementTy,
                    dyn_cast<DenseElementsAttr>(op.getValue())
                        .getSplatValue<TypedAttr>())));
      } else {
        if (dyn_cast<DenseElementsAttr>(op.getValue())
                .getSplatValue<APInt>()
                .isZero()) {
          v = builder.create<vector::ConstantMaskOp>(
              loc, vectorType,
              DenseI64ArrayAttr::get(builder.getContext(),
                                     ArrayRef<int64_t>{0}));
        } else {
          v = builder.create<vector::ConstantMaskOp>(
              loc, vectorType,
              DenseI64ArrayAttr::get(builder.getContext(),
                                     ArrayRef<int64_t>{vectorLength}));
        }
      }
    } else {
      v = builder.create<vector::BroadcastOp>(
          loc, vectorType,
          builder.create<arith::ConstantOp>(
              loc, elementTy,
              dyn_cast<DenseElementsAttr>(op.getValue())
                  .getSplatValue<TypedAttr>()));
    }
    map.map(op.getResult(), v);
  }

  void handleSplatOp(triton::SplatOp op, OpBuilder &builder, IRMapping &map,
                     unsigned vectorLength, bool needCvtDataLayout) const {
    auto loc = op.getLoc();
    auto elementTy = getTypeConverter()->convertType(
        dyn_cast<TensorType>(op.getType()).getElementType());
    Value v;
    if (elementTy.isInteger(1)) {
      if (needCvtDataLayout) {
        v = builder.create<vector::BroadcastOp>(
            loc,
            VectorType::get(ArrayRef<int64_t>{vectorLength},
                            builder.getIntegerType(8)),
            builder.create<arith::ExtUIOp>(loc, builder.getIntegerType(8),
                                           map.lookup(op.getSrc())));
      } else {
        auto vectorType =
            VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy);
        auto ifOp = builder.create<scf::IfOp>(
            loc, map.lookup(op.getSrc()),
            [&](OpBuilder &b, Location loc) {
              Value allTrue = b.create<vector::ConstantMaskOp>(
                  loc, vectorType,
                  DenseI64ArrayAttr::get(b.getContext(),
                                         ArrayRef<int64_t>{vectorLength}));
              b.create<scf::YieldOp>(loc, allTrue);
            },
            [&](OpBuilder &b, Location loc) {
              Value allFalse = b.create<vector::ConstantMaskOp>(
                  loc, vectorType,
                  DenseI64ArrayAttr::get(b.getContext(), ArrayRef<int64_t>{0}));
              b.create<scf::YieldOp>(loc, allFalse);
            });
        v = ifOp.getResult(0);
      }
    } else {
      v = builder.create<vector::BroadcastOp>(
          loc, VectorType::get(ArrayRef<int64_t>{vectorLength}, elementTy),
          map.lookup(op.getSrc()));
    }
    map.map(op.getResult(), v);
  }

  void handleBitcastOp(triton::BitcastOp op, OpBuilder &builder, IRMapping &map,
                       unsigned vectorLength) const {
    auto loc = op.getLoc();
    auto vectorType = VectorType::get(
        ArrayRef<int64_t>{vectorLength},
        getTypeConverter()->convertType(
            dyn_cast<TensorType>(op.getType()).getElementType()));
    auto newOp = builder.create<arith::BitcastOp>(loc, vectorType,
                                                  map.lookup(op.getOperand()));
    map.map(op.getResult(), newOp.getResult());
  }

  void handleExternElementwiseOp(triton::ExternElementwiseOp op,
                                 OpBuilder &builder, IRMapping &map,
                                 unsigned vectorLength) const {
    SmallVector<Value, 4> operands;
    auto loc = op.getLoc();
    for (auto operand : op.getOperands()) {
      operands.push_back(map.lookup(operand));
    }
    auto symbol = op.getSymbol();
    Value result;
    auto resultTy = VectorType::get(ArrayRef<int64_t>{vectorLength},
                                    getElementTypeOrSelf(op.getResult()));
    if (symbol.starts_with("__ef_")) {
      std::string efSymbol = "__ef_v";
      efSymbol += symbol.drop_front(strlen("__ef_"));
      result = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                        efSymbol);
    } else if (mlir::triton::gcu::isMixedPrecisionSymbol(symbol)) {
      result = builder.create<gcu::ExternElementwiseOp>(loc, resultTy, operands,
                                                        symbol);
    } else {
      llvm_unreachable(
          ("unsupported extern elementwise: " + symbol).str().c_str());
    }
    map.map(op.getResult(), result);
  }

  void handleCommonOp(Operation &op, OpBuilder &builder, IRMapping &map,
                      unsigned vectorLength, bool needCvtDataLayout) const {
    Operation *newOp;
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      auto condition = selectOp.getCondition();
      auto mapValue = map.lookup(condition);
      if (getElementTypeOrSelf(mapValue.getType()).isInteger(8)) {
        map.map(condition,
                builder
                    .create<gcu::VectorConvertOp>(
                        op.getLoc(),
                        VectorType::get(ArrayRef<int64_t>{vectorLength},
                                        builder.getIntegerType(1)),
                        mapValue)
                    .getResult(0));
      }

      unsigned maxVectorLength = vectorLength;
      auto conditionDefOp = condition.getDefiningOp();
      if (conditionDefOp && isa<arith::CmpIOp, arith::CmpFOp>(conditionDefOp)) {
        auto lhsValue = map.lookup(conditionDefOp->getOperand(0));
        auto rhsValue = map.lookup(conditionDefOp->getOperand(1));
        unsigned lhsBpe =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(lhsValue.getType()));
        unsigned rhsBpe =
            mlir::triton::gcu::getBpe(getElementTypeOrSelf(rhsValue.getType()));
        unsigned bpe = std::max(lhsBpe, rhsBpe);
        maxVectorLength = 4 * kOaccSizeInBytes / bpe;
      }
      if (vectorLength > maxVectorLength) {
        newOp = generateSplitSelectOp(selectOp, builder, map, vectorLength,
                                      maxVectorLength);
      } else {
        newOp = builder.clone(op, map);
        map.map(condition, mapValue);
      }
    } else if (auto cvtOp = dyn_cast<arith::ExtUIOp>(op)) {
      if (cast<TensorType>(cvtOp.getIn().getType())
              .getElementType()
              .isInteger(1) &&
          cast<TensorType>(cvtOp.getOut().getType())
              .getElementType()
              .isInteger(8)) {
        auto inValue = map.lookup(cvtOp.getIn());
        if (getElementTypeOrSelf(inValue.getType()).isInteger(1)) {
          inValue = builder
                        .create<gcu::VectorConvertOp>(
                            op.getLoc(),
                            VectorType::get(ArrayRef<int64_t>{vectorLength},
                                            builder.getIntegerType(8)),
                            inValue)
                        .getResult(0);
        }
        map.map(cvtOp.getOut(), inValue);
        return;
      } else {
        auto outElementType = getElementTypeOrSelf(cvtOp.getOut().getType());
        unsigned outBpe = mlir::triton::gcu::getBpe(outElementType);
        unsigned maxVectorLength = 4 * kOaccSizeInBytes / outBpe;
        if (vectorLength > maxVectorLength &&
            getElementTypeOrSelf(cvtOp.getIn().getType()).isInteger(1)) {
          newOp = generateSplitExtUIOp(cvtOp, builder, map, vectorLength,
                                       maxVectorLength);
        } else {
          newOp = builder.clone(op, map);
        }
      }
    } else if (isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(op)) {
      auto lhs = op.getOperand(0);
      auto rhs = op.getOperand(1);
      auto lhsValue = map.lookup(lhs);
      auto rhsValue = map.lookup(rhs);
      if (getElementTypeOrSelf(lhsValue.getType()).isInteger(1) &&
          getElementTypeOrSelf(rhsValue.getType()).isInteger(8)) {
        map.map(lhs, builder
                         .create<gcu::VectorConvertOp>(
                             op.getLoc(),
                             VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             builder.getIntegerType(8)),
                             lhsValue)
                         .getResult(0));
        newOp = builder.clone(op, map);
        map.map(lhs, lhsValue);
      } else if (getElementTypeOrSelf(lhsValue.getType()).isInteger(8) &&
                 getElementTypeOrSelf(rhsValue.getType()).isInteger(1)) {
        map.map(rhs, builder
                         .create<gcu::VectorConvertOp>(
                             op.getLoc(),
                             VectorType::get(ArrayRef<int64_t>{vectorLength},
                                             builder.getIntegerType(8)),
                             rhsValue)
                         .getResult(0));
        newOp = builder.clone(op, map);
        map.map(rhs, rhsValue);
      } else {
        newOp = builder.clone(op, map);
      }
    } else {
      newOp = builder.clone(op, map);
    }
    SmallVector<Type> resultTypes;
    auto typeInterface = dyn_cast<InferTypeOpInterface>(newOp);
    if (!typeInterface ||
        failed(typeInterface.inferReturnTypes(
            newOp->getContext(), newOp->getLoc(), newOp->getOperands(),
            newOp->getAttrDictionary(), newOp->getPropertiesStorage(),
            newOp->getRegions(), resultTypes))) {
      resultTypes.clear();
      llvm::transform(
          op.getResultTypes(), std::back_inserter(resultTypes),
          [&](auto resultType) {
            return VectorType::get(
                ArrayRef<int64_t>{vectorLength},
                getTypeConverter()->convertType(
                    dyn_cast<TensorType>(resultType).getElementType()));
          });
    }

    for (auto [resultType, result, newResult] :
         llvm::zip(resultTypes, op.getResults(), newOp->getResults())) {
      newResult.setType(resultType);
      if (isa<arith::CmpFOp, arith::CmpIOp>(op) && needCvtDataLayout &&
          llvm::any_of(op.getUsers(), [&](Operation *user) {
            if (auto selectOp = dyn_cast<arith::SelectOp>(user)) {
              return selectOp.getCondition() != op.getResult(0);
            } else if (auto maskedLoadOp =
                           dyn_cast<triton::gcu::MaskedLoadOp>(user)) {
              return maskedLoadOp.getMask() != op.getResult(0);
            } else if (auto maskedStoreOp =
                           dyn_cast<triton::gcu::MaskedStoreOp>(user)) {
              return maskedStoreOp.getMask() != op.getResult(0);
            }
            return !isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(user);
          })) {
        map.map(result, builder
                            .create<gcu::VectorConvertOp>(
                                op.getLoc(),
                                VectorType::get(ArrayRef<int64_t>{vectorLength},
                                                builder.getIntegerType(8)),
                                newResult)
                            .getResult(0));
      } else {
        map.map(result, newResult);
      }
    }
  }
};
} // namespace

#if TRITON_VERSION == 35
void mlir::triton::populateElementwiseFusionOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::DTETagPool &pTagPool,
    gcu::AccPublishContext &accPublishContext) {
  patterns.add<GCUElementwiseFusionOpLowering>(converter, patterns.getContext(),
                                               userAnalysis, replaced2Origin,
                                               pTagPool, accPublishContext);
}
#else
void mlir::triton::populateElementwiseFusionOpToGCUPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    gcu::FirstLastUserAnalysis &userAnalysis,
    std::map<Operation *, Operation *> &replaced2Origin,
    triton::gcu::DTETagPool &pTagPool) {
  patterns.add<GCUElementwiseFusionOpLowering>(converter, patterns.getContext(),
                                               userAnalysis, replaced2Origin,
                                               pTagPool);
}
#endif
