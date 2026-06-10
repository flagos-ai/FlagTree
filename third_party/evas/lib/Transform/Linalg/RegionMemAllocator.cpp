//===------------------- RegionMemAllocator.cpp -----------------*- C++ -*-===//
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

#include "evas/Transform/Linalg/RegionMemAllocator.h"
#include "epu/memory.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace mlir::triton::ev {

//===----------------------------------------------------------------------===//
// Basic Memory Allocator
//===----------------------------------------------------------------------===//

void MemAllocator::assignAddrOrder(
    const std::shared_ptr<LiveBuffer> visitedBuffer, int64_t align) {
  int64_t nextToAssignAddr = 0;
  for (const auto &phyBuffer : phyBufsOrder) {
    if (phyBuffer->isOverflow())
      continue;
    if (phyBuffer->isExternal() ||
        phyBuffer->isConflictWith(visitedBuffer->getLiveInterval())) {
      if (nextToAssignAddr + visitedBuffer->size() <= phyBuffer->addr())
        break;
      int64_t upBound = llvm::alignTo(phyBuffer->upperBound(), align);
      nextToAssignAddr = std::max<int64_t>(upBound, nextToAssignAddr);
    }
  }
  visitedBuffer->setPhyAddr(nextToAssignAddr);
  insertPhyBuf(visitedBuffer);
}

void MemAllocator::assignAddrReverse(
    const std::shared_ptr<LiveBuffer> visitedBuffer, int64_t align) {
  // FIXME: Improve next addresee computation with real size.
  int64_t alignedBufSize = llvm::alignTo(visitedBuffer->size(), alignment);
  int64_t nextToAssignAddr =
      memCapacity(visitedBuffer->scope()) - alignedBufSize;
  for (const auto &phyBuffer : phyBufsReverse) {
    if (phyBuffer->isOverflow())
      continue;
    if (phyBuffer->isExternal() ||
        phyBuffer->isConflictWith(visitedBuffer->getLiveInterval())) {
      if (nextToAssignAddr >= phyBuffer->upperBound())
        break;
      nextToAssignAddr = std::min<int64_t>(nextToAssignAddr,
                                           phyBuffer->addr() - alignedBufSize);
    }
  }
  visitedBuffer->setPhyAddr(nextToAssignAddr);
  insertPhyBuf(visitedBuffer);
}

void MemAllocator::insertPhyBuf(const std::shared_ptr<LiveBuffer> buffer) {
  phyBufsOrder.insert(buffer);
  phyBufsReverse.insert(buffer);
}

void MemAllocator::erasePhyBuf(const std::shared_ptr<LiveBuffer> buffer) {
  phyBufsOrder.erase(buffer);
  phyBufsReverse.erase(buffer);
}

void MemAllocator::revertPhyBuf(const std::shared_ptr<LiveBuffer> buffer) {
  erasePhyBuf(buffer);
  buffer->setPhyAddr(0);
  virtualBufs.insert(buffer);
}

std::shared_ptr<LiveBuffer> MemAllocator::pickNextBuffer() {
  assert(!virtualBufs.empty() && "No buffer to pick");
  auto visitedBuffer = *virtualBufs.begin();
  virtualBufs.erase(visitedBuffer);
  return visitedBuffer;
}

void MemAllocator::initExternelBufs(const ArrayAttr &externalLives,
                                    uint64_t &nextSlotIndex) {
  // Init externel buffers using living information at attribute.
  for (auto liveBuffer : externalLives) {
    ArrayAttr liveInfo = cast<ArrayAttr>(liveBuffer);
    int64_t phyAddr = cast<IntegerAttr>(liveInfo[0]).getInt();
    int64_t size = cast<IntegerAttr>(liveInfo[1]).getInt();
    auto phyBuf = std::make_shared<LiveBuffer>(memScope, phyAddr, size);
    phyBuf->setSlotIndex(nextSlotIndex++);
    insertPhyBuf(phyBuf);
  }
  // Checking there is no externel buffer overlap with another.
  auto isVaildInitBuffer = [](const LiveBufferSet &phyBufsOrder) {
    int64_t slidePtr = 0;
    for (const auto &phyBuffer : phyBufsOrder) {
      if (!phyBuffer->isExternal())
        continue;
      if (phyBuffer->isOverflow() || phyBuffer->addr() < slidePtr)
        return false;
      slidePtr = phyBuffer->upperBound();
    }
    return true;
  };
  assert(isVaildInitBuffer(phyBufsOrder) && "Unexpected initial state");
}

// Warning: There is no guarantee for the validity of fixed physical buffers.
void MemAllocator::initCurrRegionPhyBufs(
    const std::vector<Operation *> &opsWithFixedAddr, const Liveness &LN,
    uint64_t &nextSlotIndex) {
  for (const auto &op : opsWithFixedAddr) {
    auto fixedBuf = std::make_shared<LiveBuffer>(op);
    // set phyAddr, slotIndex, priority and liveness
    int64_t addr = op->getAttrOfType<IntegerAttr>(phyAddrName).getInt();
    fixedBuf->setPhyAddr(addr);
    fixedBuf->setSlotIndex(nextSlotIndex++);
    fixedBuf->setPriority(getMemoryPrior(op));
    // TODO: Improve Live-interval to speed up analysis
    auto liveInterval = LN.resolveLiveness(fixedBuf->value());
    fixedBuf->setLiveInterval(liveInterval);
    insertPhyBuf(fixedBuf);
  }
}

void MemAllocator::initSubRegionPhyBufs(
    const std::vector<Operation *> &opsHasSubRegion, uint64_t &nextSlotIndex) {
  // Create a merged buffer and insert this to physical buffer set
  auto initMergedBuffer = [&](Operation *op, int64_t addr, int64_t upper,
                              uint64_t &slotIndex) {
    int64_t size = upper - addr;
    auto mergedBuffer = std::make_shared<LiveBuffer>(op, memScope, addr, size);
    mergedBuffer->setLiveInterval(op);
    mergedBuffer->setSlotIndex(slotIndex++);
    insertPhyBuf(mergedBuffer);
  };
  for (const auto &opHasSubRegion : opsHasSubRegion) {
    // 1. Collect all of allocation operations nested under the given sub region
    LiveBufferSet toMergeBuffers(LayoutOrder);
    opHasSubRegion->walk([&](Operation *op) {
      if (!isMemoryAllocOp(op) || getMemScope(op) != memScope)
        return;
      if (op->hasAttr(overflowName) &&
          op->getAttrOfType<BoolAttr>(previewName).getValue())
        return;
      if (!op->hasAttr(phyAddrName))
        return;
      auto fixedBuf = std::make_shared<LiveBuffer>(op);
      int64_t addr = op->getAttrOfType<IntegerAttr>(phyAddrName).getInt();
      fixedBuf->setPhyAddr(addr);
      toMergeBuffers.insert(fixedBuf);
    });
    if (toMergeBuffers.empty())
      continue;
    // 2. Try to merge all buffers and create fake physical buffer with liveness
    int64_t mergedBufAddr = -1;
    int64_t mergedBufUpper = -1;
    for (const auto &toMergeBuffer : toMergeBuffers) {
      if (mergedBufAddr == -1 || mergedBufUpper == -1) {
        mergedBufAddr = toMergeBuffer->addr();
        mergedBufUpper = toMergeBuffer->upperBound();
      }
      if (toMergeBuffer->addr() > mergedBufUpper &&
          mergedBufUpper > mergedBufAddr) {
        initMergedBuffer(opHasSubRegion, mergedBufAddr, mergedBufUpper,
                         nextSlotIndex);
        mergedBufAddr = toMergeBuffer->addr();
        mergedBufUpper = toMergeBuffer->upperBound();
        continue;
      }
      mergedBufUpper =
          std::max<int64_t>(toMergeBuffer->upperBound(), mergedBufUpper);
    }
    if (mergedBufAddr != -1 && mergedBufUpper != -1) {
      assert(mergedBufUpper > mergedBufAddr && "Unexpected buffer");
      initMergedBuffer(opHasSubRegion, mergedBufAddr, mergedBufUpper,
                       nextSlotIndex);
    }
  }
}

void MemAllocator::initRegionBuffers(Region *region, const Liveness &LN,
                                     uint64_t &nextSlotIndex) {
  // Walk through operations to identify and map buffers
  std::shared_ptr<LiveBuffer> buffer;
  for (auto &op : region->getOps()) {
    if (isSubkernelBufferOp(&op)) {
      for (unsigned i = 0, e = op.getNumResults(); i < e; ++i) {
        auto result = op.getResult(i);
        auto scope = MemScope(
            cast<IntegerAttr>(op.getAttrOfType<ArrayAttr>(
                                  ev::MEMSCOPE)[op.getNumOperands() + i])
                .getInt());
        if (scope != memScope)
          continue;
        int64_t address = -1;
        if (op.hasAttr(ev::addrName)) {
          address =
              cast<IntegerAttr>(op.getAttrOfType<ArrayAttr>(ev::addrName)[i])
                  .getInt();
        }
        buffer = std::make_shared<LiveBuffer>(
            result, scope, address < 0 ? 0 : address, nextSlotIndex++,
            LN.resolveLiveness(result));
        if (address < 0) {
          virtualBufs.insert(std::move(buffer));
        } else {
          insertPhyBuf(buffer);
        }
      }
    } else if (isMemoryAllocOp(&op) && getMemScope(&op) == memScope) {
      int64_t address = -1;
      if (op.hasAttr(ev::phyAddrName)) {
        address = op.getAttrOfType<IntegerAttr>(ev::phyAddrName).getInt();
      }
      buffer = std::make_shared<LiveBuffer>(
          op.getResult(0), memScope, address < 0 ? 0 : address, nextSlotIndex++,
          LN.resolveLiveness(op.getResult(0)));
      if (address < 0) {
        virtualBufs.insert(std::move(buffer));
      } else {
        insertPhyBuf(buffer);
      }
    } else {
      continue;
    }
  }
}

void MemAllocator::init(Region *region, const Liveness &LN) {
  // Collect all wanted operators in mentioned region:
  //   1) 'isMemoryAllocOp' with memory space equal to 'memScope',
  //       but no attribute 'phyAddr' attached.
  //   2) 'isMemoryAllocOp' with memory space equal to 'memScope',
  //      and has attribute 'phyAddr'.
  //   3) operation has sub regions, which is necessary for liveness analysis
  uint64_t nextSlotIndex = 0;
  initRegionBuffers(region, LN, nextSlotIndex);

  // Init virtual buffers with liveness
  // initVirtualBufs(memBuffers, LN, nextSlotIndex);
  // Init external buffers.
  if (auto externalLives = region->getParentOp()->getAttrOfType<ArrayAttr>(
          LiveBufString(memScope)))
    initExternelBufs(externalLives, nextSlotIndex);
  // Init fixed physical buffers.
  // initCurrRegionPhyBufs(opsWithFixedAddr, LN, nextSlotIndex);
  // initSubRegionPhyBufs(opsHasSubRegion, nextSlotIndex);
  assert(virtualBufs.size() + phyBufsOrder.size() == nextSlotIndex &&
         "Error buffer size after init");
}

void MemAllocator::rewrite() {

  for (const auto &phyBuffer : getAllocResult()) {
    if (phyBuffer->isExternal() || phyBuffer->isSubRegionBuf())
      continue;
    Operation *alloc = phyBuffer->getOperation();
    Builder builder(alloc);
    auto setAddr = [&builder](Operation *op, int64_t addr) {
      auto phyAddrAttr = builder.getI64IntegerAttr(addr);
      op->setAttr(phyAddrName, phyAddrAttr);
    };
    if (isPreview() && !alloc->hasAttr(phyAddrName))
      alloc->setAttr(previewName, builder.getBoolAttr(true));
    if (isMemoryAllocOp(alloc)) {
      if (phyBuffer->isOverflow()) {
        auto sizeAttr = builder.getI64IntegerAttr(phyBuffer->size());
        alloc->setAttr(overflowName, sizeAttr);
        setAddr(alloc, -1);
      } else {
        setAddr(alloc, phyBuffer->fixedAddr());
      }
    } else {
      assert(isSubkernelBufferOp(alloc) &&
             "Only support subkernel buffer or memory allocation");
      auto bufValue = phyBuffer->value();
      auto outIdx = cast<OpResult>(bufValue).getResultNumber();
      // Update the address at the specific index
      if (phyBuffer->isOverflow()) {
        alloc->setAttr(overflowName,
                       builder.getI64IntegerAttr(phyBuffer->size()));
        setAddrAtIndex(alloc, outIdx, -1);
      } else {
        setAddrAtIndex(alloc, outIdx, phyBuffer->fixedAddr());
      }
    }
  }
}

void MemAllocator::reset() {
  virtualBufs.clear();
  phyBufsOrder.clear();
  phyBufsReverse.clear();
}

//===----------------------------------------------------------------------===//
// Order Memory Allocator
//===----------------------------------------------------------------------===//

void OrderMemAllocator::allocate() {
  while (!isVirtBufEmpty()) {
    auto visitedBuffer = pickNextBuffer();
    assignAddrOrder(visitedBuffer, getAlign());
    if (isPreview() || !visitedBuffer->isOverflow())
      continue;
    assert(false && "TODO: support buffer spill");
  }
}

//===----------------------------------------------------------------------===//
// Dual-Directional Memory Allocator
//===----------------------------------------------------------------------===//

void DualMemAllocator::allocate() {
  while (!isVirtBufEmpty()) {
    auto visitedBuffer = pickNextBuffer();
    // allocate ddr buffer in reverse order for now to avoid collision with input and output ddr buffers
    if (visitedBuffer->isPreferBankAlone() || visitedBuffer->scope() == MemScope::DDR) {
      assignAddrReverse(visitedBuffer, getAlign());
    } else {
      assignAddrOrder(visitedBuffer, getAlign());
    }
    if (isPreview() || !visitedBuffer->isOverflow())
      continue;
    assert(false && "TODO: support buffer spill");
  }
}

//===----------------------------------------------------------------------===//
// Bank Optimization Memory Allocator
//===----------------------------------------------------------------------===//

void BankOptAllocator::allocate() {
  auto toReallocBufs = LiveBufferSet(LayoutOrder);

  for (const auto &phyBuffer : getAllocResult()) {
    assert((isPreview() || !phyBuffer->isOverflow()) && "Unsupported Spill");
    if (phyBuffer->isExternal() || phyBuffer->isSubRegionBuf())
      continue;
    // FIXME: Maybe buffer specified by user has priority ?
    if (phyBuffer->isPreferBankAlone() && !phyBuffer->isOverflow())
      toReallocBufs.insert(phyBuffer);
  }

  assert(isVirtBufEmpty() && "All buffer must have been assigned");

  while (!toReallocBufs.empty()) {
    // 1. Revert a buffer with least physical address
    const auto &reallocBuf = *toReallocBufs.begin();
    int64_t oriPhyAddr = reallocBuf->addr();
    toReallocBufs.erase(reallocBuf);
    revertPhyBuf(reallocBuf);
    // 2. Reassign physical address
    auto visitedBuffer = pickNextBuffer();
    assert(visitedBuffer->getOperation() == reallocBuf->getOperation() &&
           "Next buffer must be reallocBuf");
    assignAddrOrder(visitedBuffer, getAlign());
    if (visitedBuffer->isOverflow()) {
      erasePhyBuf(visitedBuffer);
      visitedBuffer->setPhyAddr(oriPhyAddr);
      insertPhyBuf(visitedBuffer);
    }
  }
}

void BankOptAllocator::rewrite() {
  for (const auto &phyBuffer : getAllocResult()) {
    if (phyBuffer->isExternal() || phyBuffer->isSubRegionBuf())
      continue;
    if (!phyBuffer->isPreferBankAlone() || phyBuffer->isOverflow())
      continue;
    Operation *alloc = phyBuffer->getOperation();
    if (isMemoryAllocOp(alloc)) {
      Builder builder(alloc);
      auto phyAddrAttr = builder.getI64IntegerAttr(phyBuffer->fixedAddr());
      alloc->setAttr(phyAddrName, phyAddrAttr);
    } else {
      assert(isSubkernelBufferOp(alloc) &&
             "Only support subkernel buffer or memory allocation");
      auto bufValue = phyBuffer->value();
      auto outIdx = cast<OpResult>(bufValue).getResultNumber();
      // Update the address at the specific index
      setAddrAtIndex(alloc, outIdx, phyBuffer->fixedAddr());
    }
  }
}

} // namespace mlir::triton::ev
