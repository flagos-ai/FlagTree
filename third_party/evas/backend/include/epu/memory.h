/* Copyright 2024 The EVAS Intelligence Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef EV_SUPPORT_MEMORY_H_
#define EV_SUPPORT_MEMORY_H_


#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"

#include <string>
#include <unordered_map>


namespace mlir::ev {

static constexpr llvm::StringRef scopeName = "scope";
static constexpr llvm::StringRef addrName = "addr";
static constexpr llvm::StringRef subKernelAddrName = "address";
static constexpr llvm::StringRef phyAddrName = "phyAddr";
static constexpr llvm::StringRef overflowName = "overflow";
static constexpr llvm::StringRef previewName = "preview";
static constexpr llvm::StringRef priorityName = "priority";
static constexpr llvm::StringRef schedulePrimitive = "schedule_primitive";
static constexpr llvm::StringRef prefetchName = "prefetch";
static constexpr llvm::StringRef MEMSCOPE = "mem_scope";

enum MemScope : unsigned {
  UNKNOWN = 0,
  DDR = 1,
  L2 = 2,
  MM = 3,
  PAM = 4,
  FAM = 5,
  MAX,
};

typedef enum {
  PRIOR_MIN = 0,
  PRIOR_LOW = 25,
  PRIOR_MEDIUM = 50,
  PRIOR_HIGH = 75,
  BANK_ALONE_THRESHOLD = 100,
  PRIOR_MAX = 200,
} BufferPrior;

static inline llvm::StringRef getScopedPtrName(MemScope mem) {
  switch (mem) {
  case FAM:
  case PAM:
    return "am_ptr";
  case L2:
    return "l2_ptr";
  case MM:
    return "mm_ptr";
  case DDR:
    return "ddr_ptr";
  default:
    return "";
  }
}

static inline llvm::StringRef memScopeToString(MemScope mem) {
  assert(static_cast<unsigned>(mem) < static_cast<unsigned>(MemScope::MAX) &&
         "Unexpected memory scope.");
  static llvm::StringRef names[] = {"UNKNOWN", "DDR", "L2", "MM", "PAM", "FAM"};
  return names[static_cast<unsigned>(mem)];
}

static inline llvm::StringRef LiveBufString(MemScope mem) {
  assert(static_cast<unsigned>(mem) < static_cast<unsigned>(MemScope::MAX) &&
         "Unexpected memory scope.");
  static llvm::StringRef names[] = {"liveUnknown", "liveDDR", "liveL2",
                                    "liveMM",      "livePAM", "liveFAM"};
  return names[static_cast<unsigned>(mem)];
}

static inline llvm::StringRef FixedBufString(MemScope mem) {
  assert(static_cast<unsigned>(mem) < static_cast<unsigned>(MemScope::MAX) &&
         "Unexpected memory scope.");
  static llvm::StringRef names[] = {"fixedUnknown", "fixedDDR", "fixedL2",
                                    "fixedMM",      "fixedPAM", "fixedFAM"};
  return names[static_cast<unsigned>(mem)];
}

static inline MemScope stringToMemScope(llvm::StringRef mem) {
  static std::unordered_map<std::string, MemScope> map = {
      {"DDR", MemScope::DDR},     {"L2", MemScope::L2},
      {"MM", MemScope::MM},       {"PAM", MemScope::PAM},
      {"FAM", MemScope::FAM},     {"liveDDR", MemScope::DDR},
      {"liveL2", MemScope::L2},   {"liveMM", MemScope::MM},
      {"livePAM", MemScope::PAM}, {"liveFAM", MemScope::FAM}};
  auto it = map.find(mem.str());
  return (it != map.end()) ? it->second : MemScope::UNKNOWN;
}

static inline MemScope memLower(MemScope mem) {
  assert((mem == MemScope::MM || mem == MemScope::L2) &&
         "Unexpected memory scope.");
  return MemScope(static_cast<unsigned>(mem) - 1);
}

// 0x1000000000ULL is the base offset of the DDR memory 
// 0x40000000 is 1GB space for code section and stack/heap memory
static inline int64_t ddrAddr(int64_t offset) { return 0x1000000000ULL + 0x40000000 + (offset); }

static inline int64_t memCapacity(MemScope mem) {
  assert(static_cast<unsigned>(mem) < static_cast<unsigned>(MemScope::MAX) &&
         "Unexpected memory scope.");
  static std::unordered_map<MemScope, int64_t> map = {
      {MemScope::DDR, 5 * 1024 * 1024 * 1024U},
      {MemScope::L2, 9 * 1024 * 1024U},
      {MemScope::MM, 3 * 512 * 1024U},
      {MemScope::PAM, 256 * 1024U},
      {MemScope::FAM, 256 * 1024U}};
  auto it = map.find(mem);
  return (it != map.end()) ? it->second : 0U;
}

static inline int64_t getPreferedAlignBytes() { return 128; }

static inline bool memHasBank(MemScope mem) {
  if (mem == MemScope::L2 || mem == MemScope::MM || mem == MemScope::PAM ||
      mem == MemScope::FAM)
    return true;
  return false;
}

static inline int64_t memBankAlignment(MemScope mem) {
  assert(memHasBank(mem) && "Unexpected memory scope.");
  static std::unordered_map<MemScope, int64_t> map = {
      {MemScope::L2, 3 * 1024 * 1024U},
      {MemScope::MM, 256 * 1024U},
      {MemScope::PAM, 128 * 1024U},
      {MemScope::FAM, 128 * 1024U}};
  auto it = map.find(mem);
  return (it != map.end()) ? it->second : getPreferedAlignBytes();
}

static inline MemScope getMemScope(Type type) {
  assert(isa<MemRefType>(type) && "Unexpected type");
  MemRefType memType = cast<MemRefType>(type);
  return static_cast<MemScope>(memType.getMemorySpaceAsInt());
}

static inline MemScope getMemScope(Value value) {
  return getMemScope(value.getType());
}

static inline MemScope getMemScope(Operation *op) {
  if (isa<memref::AllocOp>(op)) {
    MemRefType memType = cast<memref::AllocOp>(op).getType();
    return getMemScope(memType);
  } else if (isa<bufferization::AllocTensorOp>(op)) {
    auto memSpace = cast<bufferization::AllocTensorOp>(op).getMemorySpace();
    if (memSpace && isa<IntegerAttr>(*memSpace))
      return static_cast<MemScope>(cast<IntegerAttr>(*memSpace).getInt());
    return MemScope::UNKNOWN;
  } else {
    assert(false && "Unexpected Operation");
  }
}

static inline void setMemScope(Value value, MemScope scope) {
  assert(isa<MemRefType>(value.getType()) && "Unexpected type");
  MemRefType memType = cast<MemRefType>(value.getType());
  Type AttrType = IntegerType::get(memType.getContext(), 64);
  Attribute mem_scope = IntegerAttr::get(AttrType, scope);
  auto newType = MemRefType::Builder(memType).setMemorySpace(mem_scope);
  value.setType(newType);
}

static inline void setMemScope(Operation *op, MemScope scope) {
  if (isa<memref::AllocOp>(op)) {
    setMemScope(cast<memref::AllocOp>(op).getResult(), scope);
  } else if (isa<bufferization::AllocTensorOp>(op)) {
    Type AttrType = IntegerType::get(op->getContext(), 64);
    Attribute mem_scope = IntegerAttr::get(AttrType, scope);
    cast<bufferization::AllocTensorOp>(op).setMemorySpaceAttr(mem_scope);
  } else {
    assert(false && "Unexpected Operation");
  }
}

static inline uint32_t getMemorySize(Type type) {
  if (isa<MemRefType>(type)) {
    MemRefType memType = cast<MemRefType>(type);
    return affine::getIntOrFloatMemRefSizeInBytes(memType).value();
  } else if (isa<RankedTensorType>(type)) {
    TensorType tensorType = cast<RankedTensorType>(type);
    auto memType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());
    return affine::getIntOrFloatMemRefSizeInBytes(memType).value();
  } else {
    assert(false && "Unexpected type");
  }
}

static inline uint32_t getMemorySize(Operation *op) {
  if (isa<memref::AllocOp>(op)) {
    return getMemorySize(cast<memref::AllocOp>(op).getType());
  } else if (isa<bufferization::AllocTensorOp>(op)) {
    return getMemorySize(cast<bufferization::AllocTensorOp>(op).getType());
  } else {
    assert(false && "Unexpected Operation");
  }
}

static inline Value getMemoryValue(Operation *op) {
  if (isa<memref::AllocOp>(op)) {
    return cast<memref::AllocOp>(op).getResult();
  } else if (isa<bufferization::AllocTensorOp>(op)) {
    return cast<bufferization::AllocTensorOp>(op).getResult();
  } else {
    assert(false && "Unexpected Operation");
  }
}

static inline bool isSubkernelBufferOp(Operation *op) {
  return false;
  //return isa<func::CallOp>(op) && op->hasAttr(schedulePrimitive);
}

static inline bool isMemoryAllocOp(Operation *op) {
  return isa<memref::AllocOp>(op) || isa<bufferization::AllocTensorOp>(op);
}

static inline void
setMemoryPrior(Operation *op, int64_t priority = BufferPrior::PRIOR_MEDIUM) {
  priority = std::max<int64_t>(BufferPrior::PRIOR_MIN, priority);
  priority = std::min<int64_t>(BufferPrior::PRIOR_MAX, priority);
  Type AttrType = IntegerType::get(op->getContext(), 64);
  Attribute prior = IntegerAttr::get(AttrType, priority);
  op->setAttr(priorityName, prior);
}

static inline void
setBankAlonePrior(Operation *op, int64_t priority = BufferPrior::PRIOR_MEDIUM) {
  priority = std::max<int64_t>(BufferPrior::PRIOR_MIN, priority);
  setMemoryPrior(op, priority + BufferPrior::BANK_ALONE_THRESHOLD);
}

static inline int64_t getMemoryPrior(Operation *op) {
  if (auto prior = op->getAttrOfType<IntegerAttr>(priorityName))
    return prior.getInt();
  return BufferPrior::PRIOR_MEDIUM;
}

/// Return the func::FuncOp called by `callOp`.
static inline func::FuncOp getCalledFunction(CallOpInterface callOp) {
  SymbolRefAttr sym =
      llvm::dyn_cast_if_present<SymbolRefAttr>(callOp.getCallableForCallee());
  if (!sym)
    return nullptr;
  return dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(callOp, sym));
}

static inline void setAddrAtIndex(Operation *op, unsigned idx, int64_t addr) {
  Builder builder(op);
  auto oldAddrAttr = op->getAttrOfType<ArrayAttr>(addrName);
  SmallVector<Attribute> newAddrs;
  for (unsigned i = 0; i < oldAddrAttr.size(); ++i) {
    if (i == idx) {
      newAddrs.push_back(builder.getI64IntegerAttr(addr));
    } else {
      newAddrs.push_back(oldAddrAttr[i]);
    }
  }
  op->setAttr(addrName, builder.getArrayAttr(newAddrs));
}

} // namespace mlir::ev

#endif // EV_SUPPORT_MEMORY_H_
