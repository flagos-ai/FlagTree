#include "tle/dialect/include/Analysis/TlePipeEffectAnalysis.h"

#include "tle/dialect/include/Analysis/TleMemoryEffectAnalysis.h"
#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {
namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static bool isProvenGlobalPointerImpl(Value value,
                                      llvm::DenseSet<Value> &visited) {
  std::optional<int> addressSpace = getPointerAddressSpace(value);
  if (!addressSpace || *addressSpace != 1)
    return false;
  if (!visited.insert(value).second)
    return true;

  if (isa<BlockArgument>(value))
    return true;

  Operation *def = value.getDefiningOp();
  if (!def || def->getNumRegions() != 0 || !isMemoryEffectFree(def))
    return false;

  Value pointerOperand;
  for (Value operand : def->getOperands()) {
    std::optional<int> operandAddressSpace = getPointerAddressSpace(operand);
    if (!operandAddressSpace)
      continue;
    if (*operandAddressSpace != *addressSpace)
      return false;
    if (pointerOperand)
      return false;
    pointerOperand = operand;
  }

  return pointerOperand && isProvenGlobalPointerImpl(pointerOperand, visited);
}

} // namespace

bool sameIndexValue(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsCst = lhs.getDefiningOp<arith::ConstantIntOp>();
  auto rhsCst = rhs.getDefiningOp<arith::ConstantIntOp>();
  if (lhsCst && rhsCst)
    return lhsCst.value() == rhsCst.value();
  auto lhsIdx = lhs.getDefiningOp<arith::ConstantIndexOp>();
  auto rhsIdx = rhs.getDefiningOp<arith::ConstantIndexOp>();
  if (lhsIdx && rhsIdx)
    return lhsIdx.value() == rhsIdx.value();
  return false;
}

std::optional<int> getPointerAddressSpace(Value value) {
  Type type = value.getType();
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    type = tensorTy.getElementType();
  auto ptrTy = dyn_cast<tt::PointerType>(type);
  if (!ptrTy)
    return std::nullopt;
  return ptrTy.getAddressSpace();
}

bool isSharedPointer(Value value) {
  return classifyPointerAddress(value) == PointerAddressClass::Shared;
}

bool isProvenGlobalPointer(Value value) {
  llvm::DenseSet<Value> visited;
  return isProvenGlobalPointerImpl(value, visited);
}

bool isNonSharedPointer(Value value) { return isProvenGlobalPointer(value); }

Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

std::optional<LocalStoreTarget> getLocalStoreTarget(Operation *op) {
  if (auto localStore = dyn_cast<ttg::LocalStoreOp>(op))
    return LocalStoreTarget{localStore.getDst(), localStore.getSrc().getType()};

  auto store = dyn_cast<tt::StoreOp>(op);
  if (!store)
    return std::nullopt;

  Value ptr = stripConvertLayouts(store.getPtr());
  while (auto addPtr = ptr.getDefiningOp<tt::AddPtrOp>())
    ptr = stripConvertLayouts(addPtr.getPtr());
  auto localPointers = ptr.getDefiningOp<LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;
  return LocalStoreTarget{localPointers.getSrc(), store.getValue().getType()};
}

std::optional<LocalStoreTarget> getAsyncCopyTarget(Operation *op) {
  auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op);
  if (!copy)
    return std::nullopt;
  return LocalStoreTarget{copy.getResult(), copy.getSrc().getType()};
}

bool recordsCompletedAsyncCopies(ttg::AsyncWaitOp wait) {
  return wait.getNum() == 0;
}

void recordCompletedAsyncWait(ttg::AsyncWaitOp wait,
                              CompletedAsyncCopyState &state) {
  if (!recordsCompletedAsyncCopies(wait))
    return;
  if (wait.getAsyncToken().empty()) {
    state.allPriorAsyncCopiesComplete = true;
    return;
  }
  for (Value token : wait.getAsyncToken())
    state.completedTokens.insert(token);
}

void propagateCompletedAsyncCommitGroup(ttg::AsyncCommitGroupOp commit,
                                        CompletedAsyncCopyState &state) {
  if (!state.allPriorAsyncCopiesComplete &&
      !state.completedTokens.contains(commit.getAsyncToken()))
    return;
  for (Value token : commit.getInputTokens())
    state.completedTokens.insert(token);
}

bool isAsyncCopyComplete(ttg::AsyncCopyGlobalToLocalOp copy,
                         const CompletedAsyncCopyState &state) {
  return state.allPriorAsyncCopiesComplete ||
         state.completedTokens.contains(copy.getToken());
}

bool isCtaInvariantSpecialRegisterRead(Operation *op) {
  auto inlineAsm = dyn_cast<tt::ElementwiseInlineAsmOp>(op);
  if (!inlineAsm || inlineAsm.getPure() || inlineAsm->getNumOperands() != 0 ||
      inlineAsm->getNumResults() != 1)
    return false;
  if (!inlineAsm->getResult(0).getType().isIntOrIndex())
    return false;

  StringRef constraints = inlineAsm.getConstraints();
  if (constraints != "=r" && constraints != "=l")
    return false;

  StringRef asmString = inlineAsm.getAsmString().trim();
  return asmString.starts_with("mov.") && asmString.contains("$0, %") &&
         asmString.ends_with(";");
}

bool canInterleaveBeforePipeMetadataOp(Operation *op) {
  if (op->getNumRegions() != 0 || op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (isCtaInvariantSpecialRegisterRead(op))
    return true;
  if (isMemoryEffectFree(op))
    return true;
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return !load.getIsVolatile() && isNonSharedPointer(load.getPtr());
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    if (getLocalStoreTarget(op))
      return true;
    return isNonSharedPointer(store.getPtr());
  }
  if (isa<ttg::LocalStoreOp>(op))
    return true;
  return false;
}

} // namespace mlir::triton::tle
