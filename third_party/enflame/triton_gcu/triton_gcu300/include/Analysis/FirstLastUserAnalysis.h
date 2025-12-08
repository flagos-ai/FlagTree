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

#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/DenseMap.h"

namespace mlir {
namespace triton {
namespace gcu {

using namespace mlir;

class FirstLastUserAnalysis {
public:
  using OptoOpT = llvm::DenseMap<Operation *, Operation *>;

  explicit FirstLastUserAnalysis(Operation *op)
      : moduleOp(op), dominators(op), postDominators(op) {
    start();
  }

  Operation *getLastUserOp(Value value, Region *opRegion);

  Operation *getLastUserOp(Operation *op) const {
    if (lastUserMap.count(op) == 0) {
      llvm::errs() << "op: " << *op << " has no last user\n";
      llvm::report_fatal_error("No last user found for op");
    }
    return lastUserMap.lookup(op);
  }

  Operation *getFirstUserOp(Operation *op) const {
    if (firstUserMap.count(op) == 0) {
      llvm::errs() << "op: " << *op << " has no first user\n";
      llvm::report_fatal_error("No first user found for op");
    }
    return firstUserMap.lookup(op);
  }

private:
  void start();

private:
  Operation *moduleOp;
  DominanceInfo dominators;
  PostDominanceInfo postDominators;

  OptoOpT lastUserMap;
  OptoOpT firstUserMap;
};

} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // GCU_ANALYSIS_FIRSTLASTUSERANALYSIS_H
