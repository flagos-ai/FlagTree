#include "tle/dialect/include/Analysis/TleMemoryEffectAnalysis.h"

#include "tle/dialect/include/IR/Dialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {
namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Type getScalarOrElementType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.getElementType();
  return type;
}

static bool isSharedMemDesc(Type type) {
  auto memDescTy = dyn_cast<ttg::MemDescType>(type);
  return memDescTy &&
         isa<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace());
}

static bool isSharedMemDesc(Value value) {
  return value && isSharedMemDesc(value.getType());
}

static bool isSharedPointerValue(Value value) {
  return classifyPointerAddress(value) == PointerAddressClass::Shared;
}

static bool isGlobalPointerValue(Value value) {
  return classifyPointerAddress(value) == PointerAddressClass::Global;
}

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value getSharedMemDescRoot(Value value) {
  Value current = stripConvertLayouts(value);
  while (current) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = stripConvertLayouts(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = stripConvertLayouts(subslice.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = stripConvertLayouts(reinterpret.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = stripConvertLayouts(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = stripConvertLayouts(reshape.getSrc());
      continue;
    }
    break;
  }
  return current;
}

static std::optional<Value> getSharedPointerMemDescRootImpl(Value ptr) {
  Value current = stripConvertLayouts(ptr);
  while (auto addPtr = current.getDefiningOp<tt::AddPtrOp>())
    current = stripConvertLayouts(addPtr.getPtr());
  auto localPointers = current.getDefiningOp<LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;
  return getSharedMemDescRoot(localPointers.getSrc());
}

static bool memDescRootsMayAlias(Value lhs, Value rhs) {
  Value lhsRoot = getSharedMemDescRoot(lhs);
  Value rhsRoot = getSharedMemDescRoot(rhs);
  if (!lhsRoot || !rhsRoot)
    return true;
  return lhsRoot == rhsRoot;
}

static bool isWriteLike(MemoryEffects::Effect *effect) {
  return isa<MemoryEffects::Write, MemoryEffects::Allocate,
             MemoryEffects::Free>(effect);
}

static bool isReadLike(MemoryEffects::Effect *effect) {
  return isa<MemoryEffects::Read>(effect);
}

static bool isKnownGlobalResource(SideEffects::Resource *resource) {
  return isa<tt::GlobalMemory>(resource);
}

static bool effectValueProvesShared(Value value) {
  return value && (isSharedMemDesc(value) || isSharedPointerValue(value));
}

static bool effectValueProvesNonShared(Value value) {
  if (!value)
    return false;
  if (isSharedMemDesc(value) || isSharedPointerValue(value))
    return false;
  if (isGlobalPointerValue(value))
    return true;
  return !isa<ttg::MemDescType>(value.getType()) &&
         classifyPointerAddress(value) != PointerAddressClass::Unknown;
}

static bool hasDirectSharedRead(Operation *op) {
  if (auto load = dyn_cast<tt::LoadOp>(op))
    return isSharedPointerValue(load.getPtr());
  if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(op))
    return isSharedMemDesc(tmaCopy.getSrc());
  return isa<ttg::LocalLoadOp>(op);
}

static bool hasDirectSharedWrite(Operation *op) {
  if (auto store = dyn_cast<tt::StoreOp>(op))
    return isSharedPointerValue(store.getPtr());
  if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(op))
    return isSharedMemDesc(tmaCopy.getDst());
  return isa<ttg::LocalStoreOp, ttg::AsyncCopyGlobalToLocalOp,
             ttg::LocalAllocOp>(op);
}

static std::optional<Value> getDirectSharedWriteMemDesc(Operation *op) {
  if (auto localStore = dyn_cast<ttg::LocalStoreOp>(op))
    return localStore.getDst();
  if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op))
    return asyncCopy.getResult();
  if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(op))
    return localAlloc.getResult();
  if (auto tmaCopy = dyn_cast<ttg::TMACopyOp>(op)) {
    if (isSharedMemDesc(tmaCopy.getDst()))
      return tmaCopy.getDst();
    return std::nullopt;
  }
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    if (!isSharedPointerValue(store.getPtr()))
      return std::nullopt;
    return getSharedPointerMemDescRootImpl(store.getPtr());
  }
  return std::nullopt;
}

static bool mayAccessSharedMemory(Operation *op, bool writeLike) {
  if (!op)
    return true;
  if (writeLike ? hasDirectSharedWrite(op) : hasDirectSharedRead(op))
    return true;
  if (isMemoryEffectFree(op))
    return false;

  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  if (effects.empty())
    return true;

  for (MemoryEffects::EffectInstance effect : effects) {
    MemoryEffects::Effect *kind = effect.getEffect();
    if (writeLike ? !isWriteLike(kind) : !isReadLike(kind))
      continue;

    if (isa<SideEffects::DefaultResource>(effect.getResource()))
      return true;
    if (isa<ttg::SharedMemory>(effect.getResource()))
      return true;
    if (effectValueProvesShared(effect.getValue()))
      return true;

    if (isKnownGlobalResource(effect.getResource()))
      continue;
    if (effectValueProvesNonShared(effect.getValue()))
      continue;

    return true;
  }
  return false;
}

static bool isBeforeInSameBlock(Operation *before, Operation *after) {
  return before && after && before->getBlock() == after->getBlock() &&
         before->isBeforeInBlock(after);
}

} // namespace

PointerAddressClass classifyPointerAddress(Value value) {
  if (!value)
    return PointerAddressClass::Unknown;
  Type type = getScalarOrElementType(value.getType());
  auto ptrTy = dyn_cast<tt::PointerType>(type);
  if (!ptrTy)
    return PointerAddressClass::Unknown;
  if (ptrTy.getAddressSpace() == 3)
    return PointerAddressClass::Shared;
  if (ptrTy.getAddressSpace() == 1)
    return PointerAddressClass::Global;
  return PointerAddressClass::Unknown;
}

std::optional<Value> getSharedPointerMemDescRoot(Value ptr) {
  return getSharedPointerMemDescRootImpl(ptr);
}

bool mayReadSharedMemory(Operation *op) {
  return mayAccessSharedMemory(op, /*writeLike=*/false);
}

bool mayWriteSharedMemory(Operation *op) {
  return mayAccessSharedMemory(op, /*writeLike=*/true);
}

bool mayWriteSharedMemoryAlias(Operation *op, Value memdesc) {
  if (!op || !memdesc)
    return true;
  if (std::optional<Value> directMemDesc = getDirectSharedWriteMemDesc(op))
    return memDescRootsMayAlias(*directMemDesc, memdesc);
  if (auto store = dyn_cast<tt::StoreOp>(op)) {
    if (isSharedPointerValue(store.getPtr()))
      return true;
  }
  if (isMemoryEffectFree(op))
    return false;

  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  if (effects.empty())
    return true;

  for (MemoryEffects::EffectInstance effect : effects) {
    if (!isWriteLike(effect.getEffect()))
      continue;
    Value value = effect.getValue();
    if (value && isSharedMemDesc(value)) {
      if (memDescRootsMayAlias(value, memdesc))
        return true;
      continue;
    }
    if (value && isSharedPointerValue(value)) {
      std::optional<Value> ptrRoot = getSharedPointerMemDescRootImpl(value);
      if (!ptrRoot || memDescRootsMayAlias(*ptrRoot, memdesc))
        return true;
      continue;
    }
    if (isa<SideEffects::DefaultResource>(effect.getResource()) ||
        isa<ttg::SharedMemory>(effect.getResource()))
      return true;
    if (isKnownGlobalResource(effect.getResource()))
      continue;
    if (effectValueProvesNonShared(value))
      continue;
    return true;
  }
  return false;
}

bool hasInterveningSharedMemoryWrite(Operation *from, Operation *to) {
  if (!isBeforeInSameBlock(from, to))
    return true;

  for (Operation *cur = from->getNextNode(); cur && cur != to;
       cur = cur->getNextNode())
    if (mayWriteSharedMemory(cur))
      return true;
  return false;
}

bool hasInterveningSharedMemoryWriteAlias(Operation *from, Operation *to,
                                          Value memdesc) {
  if (!isBeforeInSameBlock(from, to))
    return true;

  for (Operation *cur = from->getNextNode(); cur && cur != to;
       cur = cur->getNextNode())
    if (mayWriteSharedMemoryAlias(cur, memdesc))
      return true;
  return false;
}

} // namespace mlir::triton::tle
