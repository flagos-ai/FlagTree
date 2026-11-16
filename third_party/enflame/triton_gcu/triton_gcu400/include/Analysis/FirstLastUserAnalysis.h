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
#ifndef GCU_ANALYSIS_FIRSTLASTUSERANALYSIS_H
#define GCU_ANALYSIS_FIRSTLASTUSERANALYSIS_H

#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace triton {
namespace gcu {

using namespace mlir;

class FirstLastUserAnalysis {
 public:
  using ValueToUserT = llvm::DenseMap<Value, std::pair<Operation *, int>>;

  explicit FirstLastUserAnalysis(Operation *op) :
      moduleOp(op), dominators(op), postDominators(op) {
    start();
  }

  // Get last user for BlockArgument.
  std::pair<Operation *, int> getLastUser(Value value, Region *opRegion);

  // Get last user that locate the same region with value's.
  std::pair<Operation *, int> getLastUser(Value value) const {
    if (lastUserMap.count(value) == 0) {
      llvm::errs() << "value: " << value << " has no last user\n";
      llvm::report_fatal_error("No last user found for value");
    }
    return lastUserMap.lookup(value);
  }

  // Get first user that locate the same region with value's.
  std::pair<Operation *, int> getFirstUser(Value value) const {
    if (firstUserMap.count(value) == 0) {
      llvm::errs() << "value: " << value  << " has no first user\n";
      llvm::report_fatal_error("No first user found for value");
    }
    return firstUserMap.lookup(value);
  }

 private:
  void start();

  std::pair<Operation *, int>
  getLastUserOfValue(mlir::Value value, PostDominanceInfo &postDomInfo);

  void getUsersForLast(mlir::Value value,
                       mlir::Region *opRegion,
                       PostDominanceInfo &postDomInfo,
                       llvm::SetVector<std::pair<Operation*, int>> &userList,
                       llvm::SetVector<Block*> &blockList,
                       llvm::SetVector<std::pair<Operation*, int>> &aliasList);

  std::pair<Operation *, int>
  getFirstUserOfValue(mlir::Value value, DominanceInfo &domInfo);

  void getUsersForFisrt(mlir::Value value,
                        mlir::Region *opRegion,
                        llvm::SetVector<std::pair<Operation *, int>> &userList,
                        llvm::SetVector<mlir::Block *> &blockList,
                        llvm::SetVector<std::pair<Operation*, int>> &aliasList);

 private:
  Operation *moduleOp;
  DominanceInfo dominators;
  PostDominanceInfo postDominators;

  ValueToUserT lastUserMap;
  ValueToUserT firstUserMap;
};

}  // namespace gcu
}  // namespace triton
}  // namespace mlir

#endif  // GCU_ANALYSIS_FIRSTLASTUSERANALYSIS_H
