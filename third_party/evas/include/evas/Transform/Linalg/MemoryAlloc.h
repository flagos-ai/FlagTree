//===------------------------- MemoryAlloc.h --------------------*- C++ -*-===//
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

#ifndef EV_TRANSFORMS_MEMORYALLOC_H
#define EV_TRANSFORMS_MEMORYALLOC_H

#include "mlir/Analysis/CallGraph.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "evas/Transform/Linalg/RegionMemAllocator.h"
#include <queue>

namespace mlir {
class FunctionOpInterface;
}

namespace mlir::triton::ev {

typedef enum {
  SIZE_PRIOR = 0,
  LIVE_RANGE_PRIOR = 1,
} AssignPrior;

typedef struct {
  size_t alignment;
  AssignPrior assignPrior;
} AllocPolicy;

//===----------------------------------------------------------------------===//
// BFS Region Visitor
//===----------------------------------------------------------------------===//

class BFSRegionVisitor {
private:
  bool update = false;
  std::queue<Region *> toVisitRegion;

public:
  BFSRegionVisitor(bool update) : update(update) {}
  void visit(Region *rootRegion, const Liveness &LN,
             const std::shared_ptr<MemAllocator> MA);

private:
  void pushSubRegionAndUpdate(Region *visitedRegion,
                              const std::shared_ptr<MemAllocator> MA);
};

//===----------------------------------------------------------------------===//
// Memory Allocation Implement
//===----------------------------------------------------------------------===//

class MemoryAllocImpl {
private:
  size_t alignment = 128;
  bool preview = false;
  bool bankopt = false;
  CompareBufferT ToAssignOrder;
  BFSRegionVisitor BfsRV;

public:
  MemoryAllocImpl(AllocPolicy policy, bool preview, bool update, bool bankopt)
      : alignment(policy.alignment), preview(preview), bankopt(bankopt),
        BfsRV(update) {
    // TODO:: Support more greedy allocation policy
    assert(policy.assignPrior == SIZE_PRIOR && "UnSupported policy.");
    ToAssignOrder = [](const std::shared_ptr<LiveBuffer> lhs,
                       const std::shared_ptr<LiveBuffer> rhs) {
      if (lhs->getPriority() != rhs->getPriority())
        return lhs->getPriority() > rhs->getPriority();
      if (lhs->size() != rhs->size())
        return lhs->size() > rhs->size();
      return lhs->getSlotIndex() < rhs->getSlotIndex();
    };
  }

  void runOnFuncAtScope(MemScope memScope, FunctionOpInterface func,
                        const Liveness &LN);
  void runOnFunction(FunctionOpInterface func, const Liveness &LN);
};

} // namespace mlir::triton::ev

namespace mlir::triton {

std::unique_ptr<InterfacePass<FunctionOpInterface>> createMemoryAllocPass();

std::unique_ptr<InterfacePass<FunctionOpInterface>>
createMemoryAllocPass(size_t memScope, size_t alignment, bool preview,
                      bool update, bool bankopt);

} // namespace mlir::triton

#endif // EVOFC_TRANSFORMS_MEMORYALLOC_H
