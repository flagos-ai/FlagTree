//===-------------------- RegionMemAllocator.h ------------------*- C++ -*-===//
//
// Copyright 2024 EVAS Intelligence Co.,Ltd. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef EV_TRANSFORMS_REGIONMEMALLOCATOR_H
#define EV_TRANSFORMS_REGIONMEMALLOCATOR_H

#include "epu/memory.h"
#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::ev {
using namespace mlir::ev;
//===----------------------------------------------------------------------===//
// Living Buffer Class
//===----------------------------------------------------------------------===//

class LiveBuffer {
public:
  /// Create Normal Buffers
  LiveBuffer(Operation *op) : op(op) {
    memScope = getMemScope(op);
    bufferSize = getMemorySize(op);
  }
  LiveBuffer(Value bufferValue, MemScope memScope, int64_t phyAddr,
             uint64_t slotIndex, const std::vector<Operation *> &liveVector)
      : bufferValue(bufferValue), memScope(memScope), phyAddr(phyAddr),
        slotIndex(slotIndex) {
    bufferSize = getMemorySize(bufferValue.getType());
    op = bufferValue.getDefiningOp();
    assert(op != nullptr &&
           "bufferValue must be a defining value of an operation");
    priority = getMemoryPrior(op);
    setLiveInterval(liveVector);
  }
  /// Create Sub Region Buffers
  LiveBuffer(Operation *op, MemScope memScope, int64_t phyAddr, int64_t size)
      : op(op), memScope(memScope), phyAddr(phyAddr), bufferSize(size) {}
  /// Create External Buffers
  LiveBuffer(MemScope memScope, int64_t phyAddr, int64_t size)
      : memScope(memScope), phyAddr(phyAddr), bufferSize(size) {}

  void setPhyAddr(int64_t phyAddr) { this->phyAddr = phyAddr; }
  void setPriority(int64_t priority) { this->priority = priority; }
  void setSlotIndex(uint64_t slotIndex) {
    this->slotIndex = static_cast<int64_t>(slotIndex);
  }
  void setLiveInterval(Operation *liveOp) {
    liveInterval.clear();
    liveInterval.insert(liveOp);
  }
  void setLiveInterval(const std::vector<Operation *> &liveVector) {
    liveInterval = std::set<Operation *>(liveVector.begin(), liveVector.end());
  }

  // get operation order index in current region.
  int64_t getSlotIndex() { return slotIndex; }
  int64_t addr() { return phyAddr; }
  int64_t size() { return bufferSize; }
  MemScope scope() { return memScope; }
  Operation *getOperation() { return op; }
  Value value() { return bufferValue ? bufferValue : getMemoryValue(op); }
  int64_t getPriority() { return priority; }
  int64_t upperBound() { return addr() + size(); }
  std::set<Operation *> &getLiveInterval() { return liveInterval; }
  int64_t fixedAddr() {
    if (memScope == MemScope::DDR) {
      return ev::ddrAddr(addr());
    } 
    return addr();
  }
  bool isPreferBankAlone() {
    return memHasBank(memScope) &&
           priority >= BufferPrior::BANK_ALONE_THRESHOLD;
  }
  bool isOverflow() {
    return upperBound() > memCapacity(memScope) || addr() < 0;
  }
  bool isExternal() { return op == nullptr; }
  bool isSubRegionBuf() {
    return op != nullptr && !isMemoryAllocOp(op) && !isSubkernelBufferOp(op);
  }
  bool isConflictWith(std::set<Operation *> &checkInterval) {
    for (auto &checkOp : checkInterval) {
      if (liveInterval.count(checkOp))
        return true;
    }
    return false;
  }
  bool isLiveAt(Operation *checkOp) {
    if (liveInterval.count(checkOp))
      return true;
    return false;
  }

private:
  int64_t slotIndex = -1;
  Operation *op = nullptr;
  MemScope memScope = UNKNOWN;
  int64_t phyAddr = 0;
  int64_t bufferSize = 0;
  std::set<Operation *> liveInterval;
  int64_t priority = BufferPrior::PRIOR_MEDIUM;
  Value bufferValue;
};

using CompareBufferT = std::function<bool(const std::shared_ptr<LiveBuffer>,
                                          const std::shared_ptr<LiveBuffer>)>;
using LiveBufferSet = std::set<std::shared_ptr<LiveBuffer>, CompareBufferT>;

//===----------------------------------------------------------------------===//
// Buffer Info
//===----------------------------------------------------------------------===//

struct BufferInfo {
  MemScope scope;
  int64_t address;

  BufferInfo(MemScope scope = MemScope::UNKNOWN, int64_t address = -1)
      : scope(scope), address(address) {}
};

//===----------------------------------------------------------------------===//
// Basic Memory Allocator
//===----------------------------------------------------------------------===//

class MemAllocator {
private:
  bool preview;
  MemScope memScope;
  int64_t alignment;
  LiveBufferSet virtualBufs;
  LiveBufferSet phyBufsOrder;
  LiveBufferSet phyBufsReverse;

public:
  static bool LayoutOrder(const std::shared_ptr<LiveBuffer> lhs,
                          const std::shared_ptr<LiveBuffer> rhs) {
    if (lhs->addr() != rhs->addr())
      return lhs->addr() < rhs->addr();
    assert(!lhs->isExternal() && !rhs->isExternal() && "buffer overlap");
    return lhs->getSlotIndex() > rhs->getSlotIndex();
  }

  static bool LayoutReverse(const std::shared_ptr<LiveBuffer> lhs,
                            const std::shared_ptr<LiveBuffer> rhs) {
    if (lhs->upperBound() != rhs->upperBound())
      return lhs->upperBound() > rhs->upperBound();
    assert(!lhs->isExternal() && !rhs->isExternal() && "buffer overlap");
    return lhs->getSlotIndex() > rhs->getSlotIndex();
  }

  MemAllocator(CompareBufferT ToAssignOrder, MemScope memScope,
               int64_t alignment, bool preview)
      : preview(preview), memScope(memScope), alignment(alignment),
        virtualBufs(ToAssignOrder), phyBufsOrder(LayoutOrder),
        phyBufsReverse(LayoutReverse) {
    virtualBufs.clear();
    phyBufsOrder.clear();
    phyBufsReverse.clear();
  }

  virtual ~MemAllocator() {}

private:
  void initVirtualBufs(const std::vector<Operation *> &opsToAssign,
                       const Liveness &LN, uint64_t &nextSlotIndex);

  void initExternelBufs(const ArrayAttr &externalLives,
                        uint64_t &nextSlotIndex);
  // Warning: There is no guarantee for the validity of fixed physical buffers.
  void initCurrRegionPhyBufs(const std::vector<Operation *> &opsWithFixedAddr,
                             const Liveness &LN, uint64_t &nextSlotIndex);
  void initSubRegionPhyBufs(const std::vector<Operation *> &opsHasSubRegion,
                            uint64_t &nextSlotIndex);
  void visitOpsInRegion(Region *visitedRegion,
                        std::vector<Operation *> &opsToAssign,
                        std::vector<Operation *> &opsWithFixedAddr,
                        std::vector<Operation *> &opsHasSubRegion,
                        llvm::DenseMap<Value, MemScope> &subkernelBuffers);
  void initRegionBuffers(Region *region, const Liveness &LN,
                         uint64_t &nextSlotIndex);

public:
  std::shared_ptr<LiveBuffer> pickNextBuffer();
  void insertPhyBuf(const std::shared_ptr<LiveBuffer> buffer);
  void erasePhyBuf(const std::shared_ptr<LiveBuffer> buffer);
  void revertPhyBuf(const std::shared_ptr<LiveBuffer> buffer);
  void assignAddrOrder(const std::shared_ptr<LiveBuffer> visitedBuffer,
                       int64_t align);
  void assignAddrReverse(const std::shared_ptr<LiveBuffer> visitedBuffer,
                         int64_t align);
  virtual void init(Region *region, const Liveness &LN);
  virtual void rewrite();
  virtual void reset();
  virtual void allocate() = 0;

  bool isVirtBufEmpty() { return virtualBufs.empty(); }
  LiveBufferSet &getAllocResult() {
    assert(isVirtBufEmpty() && "Not finish allocate");
    return phyBufsOrder;
  }
  bool isPreview() { return preview; }
  int64_t getAlign() { return alignment; }
  MemScope getScope() { return memScope; }
};

//===----------------------------------------------------------------------===//
// Order Memory Allocator
//===----------------------------------------------------------------------===//

class OrderMemAllocator : public MemAllocator {

public:
  OrderMemAllocator(CompareBufferT ToAssignOrder, MemScope memScope,
                    int64_t alignment, bool preview)
      : MemAllocator(ToAssignOrder, memScope, alignment, preview) {}

  void allocate() override;
};

//===----------------------------------------------------------------------===//
// Dual-Directional Memory Allocator
//===----------------------------------------------------------------------===//

class DualMemAllocator : public MemAllocator {
public:
  DualMemAllocator(CompareBufferT ToAssignOrder, MemScope memScope,
                   int64_t alignment, bool preview)
      : MemAllocator(ToAssignOrder, memScope, alignment, preview) {}

  void allocate() override;
};

//===----------------------------------------------------------------------===//
// Bank Optimization Memory Allocator
//===----------------------------------------------------------------------===//

class BankOptAllocator : public MemAllocator {
public:
  BankOptAllocator(CompareBufferT ToAssignOrder, MemScope memScope,
                   int64_t alignment, bool preview)
      : MemAllocator(ToAssignOrder, memScope, alignment, preview) {}

  void allocate() override;
  void rewrite() override;
};

} // namespace mlir::triton::ev

#endif // EVOFC_TRANSFORMS_REGIONMEMALLOCATOR_H
