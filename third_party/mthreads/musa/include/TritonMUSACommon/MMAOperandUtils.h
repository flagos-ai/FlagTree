#ifndef TRITONMUSA_COMMON_MMA_OPERAND_UTILS_H
#define TRITONMUSA_COMMON_MMA_OPERAND_UTILS_H

#include "Dialect/MTGPU/IR/Dialect.h"
#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <optional>

namespace mlir::triton::musa {
namespace ttg = mlir::triton::gpu;

inline SmallVector<int64_t>
getMemDescPhysicalShape(ttg::MemDescType memDescTy) {
  auto allocShape = memDescTy.getAllocShape();
  if (allocShape.empty())
    return SmallVector<int64_t>(memDescTy.getShape().begin(),
                                memDescTy.getShape().end());
  if (allocShape.size() >= static_cast<size_t>(memDescTy.getRank()))
    allocShape = allocShape.take_back(memDescTy.getRank());
  return SmallVector<int64_t>(allocShape.begin(), allocShape.end());
}

inline SmallVector<unsigned> getSharedOrder(Attribute encoding,
                                            ArrayRef<int64_t> shape = {}) {
  if (auto swizzled =
          dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(encoding))
    return SmallVector<unsigned>(swizzled.getOrder().begin(),
                                 swizzled.getOrder().end());
  if (auto padded = dyn_cast_or_null<ttg::PaddedSharedEncodingAttr>(encoding))
    return SmallVector<unsigned>(padded.getOrder().begin(),
                                 padded.getOrder().end());
  if (auto shared = dyn_cast_or_null<ttg::SharedEncodingTrait>(encoding)) {
    if (!shape.empty())
      return ttg::getOrder(shared, shape);
  }
  return {};
}

inline bool isSharedEncoding(Attribute encoding) {
  return isa<ttg::SharedEncodingTrait>(encoding);
}

inline bool inferSharedRowMajor(ttg::MemDescType memDescTy,
                                ArrayRef<int64_t> physicalShape = {}) {
  SmallVector<int64_t> shape;
  if (physicalShape.empty())
    shape = getMemDescPhysicalShape(memDescTy);
  else
    shape.assign(physicalShape.begin(), physicalShape.end());
  auto order = getSharedOrder(memDescTy.getEncoding(), shape);
  return !order.empty() && order.front() + 1 == shape.size();
}

inline bool isMemDescViewLikeOp(Operation *op) {
  return isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
             ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
             ttg::MemDescReshapeOp>(op);
}

inline bool isMemDescSqmmaContractBridgeOp(Operation *op) {
  return isMemDescViewLikeOp(op) || isa<UnrealizedConversionCastOp>(op);
}

inline bool isTMEBackedMemDesc(Value memDesc) {
  llvm::SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<void *, 16> visited;
  auto enqueue = [&](Value candidate) {
    if (!candidate || !isa<ttg::MemDescType>(candidate.getType()))
      return;
    if (visited.insert(candidate.getAsOpaquePointer()).second)
      worklist.push_back(candidate);
  };

  enqueue(memDesc);
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();

    if (Operation *defOp = current.getDefiningOp()) {
      if (auto indexOp = dyn_cast<ttg::MemDescIndexOp>(defOp))
        enqueue(indexOp.getSrc());
      else if (auto subsliceOp = dyn_cast<ttg::MemDescSubsliceOp>(defOp))
        enqueue(subsliceOp.getSrc());
      else if (auto reinterpretOp = dyn_cast<ttg::MemDescReinterpretOp>(defOp))
        enqueue(reinterpretOp.getSrc());
      else if (auto transOp = dyn_cast<ttg::MemDescTransOp>(defOp))
        enqueue(transOp.getSrc());
      else if (auto reshapeOp = dyn_cast<ttg::MemDescReshapeOp>(defOp))
        enqueue(reshapeOp.getSrc());
    }

    for (Operation *user : current.getUsers()) {
      if (isa<triton::musa::AsyncTMECopyGlobalToLocalOp>(user))
        return true;
      if (!isMemDescViewLikeOp(user))
        continue;
      for (Value result : user->getResults())
        enqueue(result);
    }
  }

  return false;
}

inline bool needsSqmmaIssueBarrier(Value aMemDesc, Value bMemDesc) {
  return !(isTMEBackedMemDesc(aMemDesc) && isTMEBackedMemDesc(bMemDesc));
}

inline std::optional<ttg::SwizzledSharedEncodingAttr>
composeMusaOperandSharedLayout(ttg::DotOperandEncodingAttr dotEncoding,
                               ArrayRef<int64_t> operandShape,
                               ArrayRef<unsigned> sharedOrder,
                               ttg::CGAEncodingAttr cgaLayout,
                               unsigned elemBitWidth, bool needTrans) {
  if (!dotEncoding || elemBitWidth == 0)
    return std::nullopt;

  Attribute parent = dotEncoding.getParent();
  if (auto wmma = dyn_cast<ttg::MUSAWmmaEncodingAttr>(parent)) {
    return wmma.composeSharedLayoutForOperand(
        cgaLayout, dotEncoding.getOpIdx(), operandShape, sharedOrder,
        dotEncoding.getKWidth(), elemBitWidth, needTrans);
  }
  if (auto sqmma = dyn_cast<ttg::MUSASqmmaEncodingAttr>(parent)) {
    return sqmma.composeSharedLayoutForOperand(
        cgaLayout, dotEncoding.getOpIdx(), operandShape, sharedOrder,
        dotEncoding.getKWidth(), elemBitWidth, needTrans);
  }
  return std::nullopt;
}

inline std::optional<ttg::SwizzledSharedEncodingAttr>
composeMusaOperandSharedLayout(ttg::DotOperandEncodingAttr dotEncoding,
                               ArrayRef<int64_t> operandShape,
                               ArrayRef<unsigned> sharedOrder,
                               ttg::CGAEncodingAttr cgaLayout, Type elementType,
                               bool needTrans) {
  unsigned elemBitWidth = elementType.getIntOrFloatBitWidth();
  return composeMusaOperandSharedLayout(dotEncoding, operandShape, sharedOrder,
                                        cgaLayout, elemBitWidth, needTrans);
}

inline std::optional<int64_t> inferElemBytesFromMemDesc(ttg::MemDescType type) {
  int bitWidth = type.getElementTypeBitWidth();
  if (bitWidth <= 0)
    return std::nullopt;
  return static_cast<int64_t>((bitWidth + 7) / 8);
}

struct RecoveredSqmmaConsumerContract {
  int64_t sqmmaOpIdx = -1;
  int64_t elemBytes = 0;
  bool rowMajor = true;

  bool operator==(const RecoveredSqmmaConsumerContract &other) const {
    return sqmmaOpIdx == other.sqmmaOpIdx && elemBytes == other.elemBytes &&
           rowMajor == other.rowMajor;
  }
};

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
getSqmmaContractFromAnnotatedOp(Operation *op, bool defaultRowMajor) {
  if (!op || !hasSqmmaOpIdxAttr(op))
    return std::optional<RecoveredSqmmaConsumerContract>{};

  auto opIdx = getSqmmaOpIdx(op);
  auto elemBytes = getSqmmaElemBytes(op);
  if (!opIdx || !elemBytes || *elemBytes <= 0)
    return failure();

  return std::optional<RecoveredSqmmaConsumerContract>(
      RecoveredSqmmaConsumerContract{*opIdx, *elemBytes,
                                     getSqmmaRowMajor(op, defaultRowMajor)});
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverSqmmaProducerContractFromMemDesc(Value memDesc) {
  auto memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
  if (!memDescTy)
    return failure();

  std::optional<RecoveredSqmmaConsumerContract> contract;
  llvm::SmallVector<Value, 8> worklist{memDesc};
  llvm::SmallPtrSet<void *, 16> visited;

  auto mergeCandidate =
      [&](std::optional<RecoveredSqmmaConsumerContract> candidate)
      -> LogicalResult {
    if (!candidate)
      return success();
    if (contract && !(*contract == *candidate))
      return failure();
    contract = *candidate;
    return success();
  };

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    auto currentTy = dyn_cast<ttg::MemDescType>(current.getType());
    if (!currentTy)
      return failure();

    Operation *defOp = current.getDefiningOp();
    auto currentContract =
        getSqmmaContractFromAnnotatedOp(defOp, inferSharedRowMajor(currentTy));
    if (failed(currentContract) || failed(mergeCandidate(*currentContract)))
      return failure();
    if (!defOp)
      continue;

    auto enqueueOperand = [&](Value operand) {
      if (isa<ttg::MemDescType>(operand.getType()))
        worklist.push_back(operand);
    };

    if (isMemDescSqmmaContractBridgeOp(defOp)) {
      for (Value operand : defOp->getOperands())
        enqueueOperand(operand);
      continue;
    }

    if (auto waitOp = dyn_cast<triton::musa::SquadDotWaitOp>(defOp)) {
      for (auto [idx, result] : llvm::enumerate(waitOp->getResults())) {
        if (result != current)
          continue;
        enqueueOperand(waitOp.getInputs()[idx]);
      }
      continue;
    }
    if (auto waitOp = dyn_cast<triton::mtgpu::SqmmaWaitOp>(defOp)) {
      for (auto [idx, result] : llvm::enumerate(waitOp->getResults())) {
        if (result != current)
          continue;
        enqueueOperand(waitOp.getInputs()[idx]);
      }
      continue;
    }
  }

  return contract;
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverUniqueSqmmaConsumerContract(Value memDesc);

inline bool isTensorSqmmaContractBridgeOp(Operation *op) {
  return isa<triton::TransOp, triton::ReshapeOp, ttg::ConvertLayoutOp,
             UnrealizedConversionCastOp>(op);
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverUniqueSqmmaConsumerContractFromTensor(Value tensor) {
  if (!isa<RankedTensorType>(tensor.getType()))
    return failure();

  std::optional<RecoveredSqmmaConsumerContract> contract;
  bool sawNonSqmmaTerminal = false;
  llvm::SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<void *, 16> visited;
  worklist.push_back(tensor);

  auto mergeCandidate =
      [&](std::optional<RecoveredSqmmaConsumerContract> candidate)
      -> LogicalResult {
    if (!candidate) {
      sawNonSqmmaTerminal = true;
      return success();
    }
    if (contract && !(*contract == candidate))
      return failure();
    contract = *candidate;
    return success();
  };

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    if (!isa<RankedTensorType>(current.getType()))
      return failure();

    for (Operation *user : current.getUsers()) {
      if (isTensorSqmmaContractBridgeOp(user)) {
        for (Value result : user->getResults())
          if (isa<RankedTensorType>(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
        auto nestedContract =
            recoverUniqueSqmmaConsumerContract(localAlloc.getResult());
        if (failed(nestedContract))
          return failure();
        if (failed(mergeCandidate(*nestedContract)))
          return failure();
        continue;
      }

      sawNonSqmmaTerminal = true;
    }
  }

  if (contract && sawNonSqmmaTerminal)
    return failure();
  return contract;
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverUniqueSqmmaConsumerContract(Value memDesc) {
  if (!isa<ttg::MemDescType>(memDesc.getType()))
    return failure();

  std::optional<RecoveredSqmmaConsumerContract> contract;
  bool sawNonSqmmaTerminal = false;
  llvm::SmallVector<Value, 8> worklist;
  llvm::SmallPtrSet<void *, 16> visited;
  worklist.push_back(memDesc);

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

    for (Operation *user : current.getUsers()) {
      if (isa<triton::musa::AsyncTMECopyGlobalToLocalOp,
              triton::musa::AsyncTMECopyLocalToGlobalOp>(user))
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
        auto tensorContract =
            recoverUniqueSqmmaConsumerContractFromTensor(localLoad.getResult());
        if (failed(tensorContract))
          return failure();
        if (*tensorContract) {
          if (contract && !(*contract == *tensorContract))
            return failure();
          contract = **tensorContract;
        } else {
          sawNonSqmmaTerminal = true;
        }
        continue;
      }

      auto userContract =
          getSqmmaContractFromAnnotatedOp(user, inferSharedRowMajor(currentTy));
      if (failed(userContract))
        return failure();

      std::optional<RecoveredSqmmaConsumerContract> candidate = *userContract;
      if ((isa<triton::musa::SquadDotOp, triton::mtgpu::SqmmaOp>(user)) &&
          !candidate)
        candidate = *currentContract;

      if (!candidate) {
        if (isa<triton::musa::SquadDotOp, triton::mtgpu::SqmmaOp>(user))
          return failure();
        sawNonSqmmaTerminal = true;
        continue;
      }

      if (contract && !(*contract == candidate))
        return failure();
      contract = *candidate;
    }
  }

  if (contract && sawNonSqmmaTerminal)
    return failure();
  return contract;
}

inline LogicalResult verifyGroupedTMELoadConsumerContract(
    AsyncTMECopyGlobalToLocalOp op,
    std::optional<RecoveredSqmmaConsumerContract> contract) {
  if (!contract)
    return success();

  auto memDescTy = dyn_cast<ttg::MemDescType>(op.getResult().getType());
  if (!memDescTy || memDescTy.getShape().size() != 2) {
    return op.emitOpError("grouped SQMMA TME load requires a 2D shared "
                          "memdesc");
  }

  auto order = getSharedOrder(memDescTy.getEncoding(), memDescTy.getShape());
  if (order.empty()) {
    return op.emitOpError("grouped SQMMA TME load requires a valid shared "
                          "order");
  }

  auto maybeElemBytes = inferElemBytesFromMemDesc(memDescTy);
  if (!maybeElemBytes || *maybeElemBytes <= 0 ||
      *maybeElemBytes != contract->elemBytes) {
    return op.emitOpError("grouped SQMMA TME load recovered element size does "
                          "not match destination memdesc");
  }

  int64_t maxLeadingElems = 256 / *maybeElemBytes;
  if (maxLeadingElems <= 0) {
    return op.emitOpError("grouped SQMMA TME load has invalid leading segment "
                          "size");
  }

  return success();
}

inline FailureOr<std::optional<RecoveredSqmmaConsumerContract>>
recoverAndVerifyGroupedTMELoadConsumerContract(AsyncTMECopyGlobalToLocalOp op) {
  auto producerContract =
      recoverSqmmaProducerContractFromMemDesc(op.getResult());
  if (failed(producerContract)) {
    op.emitOpError("grouped TME load segment requires a valid SQMMA "
                   "producer contract on the destination memdesc");
    return failure();
  }

  auto consumerContract = recoverUniqueSqmmaConsumerContract(op.getResult());
  if (failed(consumerContract)) {
    op.emitOpError("grouped TME load segment requires a unique consistent "
                   "SQMMA consumer contract when high-level consumers are "
                   "still present");
    return failure();
  }

  std::optional<RecoveredSqmmaConsumerContract> contract = *producerContract;
  if (*consumerContract) {
    if (contract && !(*contract == **consumerContract)) {
      op.emitOpError("grouped TME load producer contract does not match the "
                     "unique recovered SQMMA consumer contract");
      return failure();
    }
    contract = *consumerContract;
  }

  if (failed(verifyGroupedTMELoadConsumerContract(op, contract)))
    return failure();
  return contract;
}

struct ResolvedSharedOperand {
  Value memDesc;
  Value llvmMemDesc;
  ttg::MemDescType memDescTy;
  LLVM::SharedMemoryObject sharedMemObj;
  Value affineBase;
  SmallVector<int64_t> physicalShape;
};

inline FailureOr<ResolvedSharedOperand>
resolveSharedOperandWithAffineBase(Value operand, Value adaptorOperand,
                                   Location loc,
                                   const LLVMTypeConverter *typeConverter,
                                   ConversionPatternRewriter &rewriter) {
  Value memDesc = operand;
  Value llvmMemDesc = adaptorOperand;
  auto memDescTy = dyn_cast<ttg::MemDescType>(operand.getType());

  if (!memDescTy) {
    Value source = operand;
    while (auto cvt = source.getDefiningOp<ttg::ConvertLayoutOp>())
      source = cvt.getSrc();

    auto localLoad = source.getDefiningOp<ttg::LocalLoadOp>();
    if (!localLoad)
      return failure();

    memDesc = localLoad.getSrc();
    memDescTy = dyn_cast<ttg::MemDescType>(memDesc.getType());
    if (!memDescTy)
      return failure();
    llvmMemDesc = rewriter.getRemappedValue(memDesc);
    if (!llvmMemDesc)
      return failure();
  }

  Type llvmElemTy = typeConverter->convertType(memDescTy.getElementType());
  auto sharedMemObj = LLVM::getSharedMemoryObjectFromStruct(
      loc, llvmMemDesc, llvmElemTy, rewriter);
  Value affineBase = sharedMemObj.getShmemAffineBase(loc, rewriter, memDescTy);
  return ResolvedSharedOperand{memDesc,    llvmMemDesc,
                               memDescTy,  sharedMemObj,
                               affineBase, getMemDescPhysicalShape(memDescTy)};
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MMA_OPERAND_UTILS_H
