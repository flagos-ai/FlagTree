//===----------------------- MemoryAlloc.cpp --------------------*- C++ -*-===//
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
#include "evas/Transform/Linalg/MemoryAlloc.h"

#define DEBUG_TYPE "memory-alloc"

namespace mlir::triton::ev {

//===----------------------------------------------------------------------===//
// BFS Region Visitor
//===----------------------------------------------------------------------===//

void BFSRegionVisitor::pushSubRegionAndUpdate(
    Region *visitedRegion, const std::shared_ptr<MemAllocator> MA) {
  Builder builder(visitedRegion->getParentOp());
  std::set<Operation *> opsToUpdate;

  visitedRegion->walk([&](Operation *op) {
    if (op->getParentRegion() != visitedRegion)
      return;
    if (isa<func::CallOp>(op)) {
      opsToUpdate.insert(op);
    } else if (op->getNumRegions() != 0) {
      for (unsigned i = 0, e = op->getNumRegions(); i < e; ++i) {
        Region *subRegion = &op->getRegion(i);
        if (!subRegion->empty()) {
          opsToUpdate.insert(op);
          toVisitRegion.push(subRegion);
        }
      }
    }
  });

  for (auto op : opsToUpdate) {
    std::vector<Attribute> liveAttrs;
    for (const auto &phyBuffer : MA->getAllocResult()) {
      if (phyBuffer->isOverflow() || phyBuffer->isSubRegionBuf())
        continue;
      if (phyBuffer->isExternal() || phyBuffer->isLiveAt(op)) {
        auto liveInfo =
            builder.getI64ArrayAttr({phyBuffer->addr(), phyBuffer->size()});
        liveAttrs.push_back(liveInfo);
      }
    }
    if (liveAttrs.empty())
      continue;
    Attribute liveAttrArrayAttr = builder.getArrayAttr(liveAttrs);
    op->setAttr(LiveBufString(MA->getScope()), liveAttrArrayAttr);
    // Update living information at 'func::FuncOp'
    if (update && isa<func::CallOp>(op)) {
      func::FuncOp func = getCalledFunction(cast<func::CallOp>(op));
      if (func && !func.isDeclaration())
        func->setAttr(LiveBufString(MA->getScope()), liveAttrArrayAttr);
    }
  }
}

void BFSRegionVisitor::visit(Region *rootRegion, const Liveness &LN,
                             const std::shared_ptr<MemAllocator> MA) {
  assert(toVisitRegion.empty() && "Unexpected queue state");
  toVisitRegion.push(rootRegion);
  while (!toVisitRegion.empty()) {
    Region *visitedRegion = toVisitRegion.front();
    toVisitRegion.pop();
    MA->reset();
    // 1. Init memory allocator
    MA->init(visitedRegion, LN);
    // 2. Assign physical address for all buffers
    MA->allocate();
    // 3. Rewrite physical address to 'isMemoryAllocOp' operators
    MA->rewrite();
    // 4. Update living information at 'func::CallOp' and 'opHasSubRegion'
    pushSubRegionAndUpdate(visitedRegion, MA);
  }
}

//===----------------------------------------------------------------------===//
// Memory Allocation Implement
//===----------------------------------------------------------------------===//

void MemoryAllocImpl::runOnFuncAtScope(MemScope memScope,
                                       FunctionOpInterface func,
                                       const Liveness &LN) {
  Builder builder(func.getOperation());
  // Prologue:
  //   1) remove 'phyAddr' attribute if 'preview = true'
  //   2) remove 'overflow' and 'preview' attribute
  std::vector<Attribute> fixedBufferAttrs;
  func.getFunctionBody().walk([&](Operation *op) {
    if (!isMemoryAllocOp(op) || getMemScope(op) != memScope)
      return;
    if (op->hasAttr(previewName) &&
        op->getAttrOfType<BoolAttr>(previewName).getValue())
      op->removeAttr(phyAddrName);
    op->removeAttr(previewName);
    op->removeAttr(overflowName);
  });

  // Main : Alloc memory for current function
  auto DMA = std::make_shared<DualMemAllocator>(ToAssignOrder, memScope,
                                                alignment, preview);
  BfsRV.visit(&func.getFunctionBody(), LN, DMA);

  // BankOpt: Realloc buffers with bank-alone preference
  if (bankopt && memHasBank(memScope)) {
    auto MRA = std::make_shared<BankOptAllocator>(
        ToAssignOrder, memScope, memBankAlignment(memScope), preview);
    BfsRV.visit(&func.getFunctionBody(), LN, MRA);
  }

  // Epilogue : remove living information in sub regions at release version
  LLVM_DEBUG(return;);
  func.getFunctionBody().walk([&](Operation *op) {
    if (op->getNumRegions() != 0)
      op->removeAttr(LiveBufString(memScope));
  });
}

void MemoryAllocImpl::runOnFunction(FunctionOpInterface func,
                                    const Liveness &LN) {
  // Allocate memory from high-level to low-level
  for (size_t id = MemScope::MAX - 1; id > MemScope::UNKNOWN; id--)
    runOnFuncAtScope(MemScope(id), func, LN);
}
} // namespace mlir::triton::ev
