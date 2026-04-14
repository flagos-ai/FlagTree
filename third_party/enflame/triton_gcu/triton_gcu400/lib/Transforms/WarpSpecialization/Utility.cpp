/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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
#include <algorithm>

#include "Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace mlir::triton::gcu {

//===----------------------------------------------------------------------===//
// Helper functions for async task
//===----------------------------------------------------------------------===//

SmallVector<AsyncTaskId> getAsyncTaskIds(Operation *op) {
  SmallVector<AsyncTaskId> asyncTaskIds;
  if (auto attr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id")) {
    for (AsyncTaskId asyncTaskId : attr.asArrayRef()) {
      // TODO(Arda): Remove this check once we figure out why we have duplicate
      // async task ids
      if (asyncTaskIds.empty() ||
          asyncTaskIds[asyncTaskIds.size() - 1] != asyncTaskId)
        asyncTaskIds.push_back(asyncTaskId);
    }
  }
  return asyncTaskIds;
}

bool hasAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  return llvm::is_contained(getAsyncTaskIds(op), asyncTaskId);
}

void setAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTaskIds) {
  SmallVector<AsyncTaskId> sortedAsyncTaskIds(asyncTaskIds.begin(),
                                              asyncTaskIds.end());
  sort(sortedAsyncTaskIds);
  op->setAttr("async_task_id",
              DenseI32ArrayAttr::get(op->getContext(), sortedAsyncTaskIds));
}

void labelParentOps(Operation *op) {
  auto asyncTaskIds = getAsyncTaskIds(op);
  auto parent = op->getParentOp();
  while (parent && !isa<triton::FuncOp>(parent)) {
    addAsyncTaskIds(parent, asyncTaskIds);
    parent = parent->getParentOp();
  }
}

SmallVector<AsyncTaskId> getNestedAsyncTaskIds(Operation *op) {
  SetVector<AsyncTaskId> asyncTaskIds;
  op->walk([&](Operation *curOp) {
    asyncTaskIds.insert_range(getAsyncTaskIds(curOp));
  });
  SmallVector<AsyncTaskId> res = asyncTaskIds.takeVector();
  llvm::sort(res);
  return res;
}

void addAsyncTaskIds(Operation *op, ArrayRef<AsyncTaskId> asyncTasks) {
  auto asyncTasksVec = getAsyncTaskIds(op);
  DenseSet<AsyncTaskId> asyncTasksSet(asyncTasksVec.begin(),
                                      asyncTasksVec.end());
  for (auto a : asyncTasks) {
    if (!asyncTasksSet.contains(a)) {
      asyncTasksVec.push_back(a);
    }
  }
  if (asyncTasksVec.size() > 0) {
    setAsyncTaskIds(op, asyncTasksVec);
  }
}

void removeAsyncTaskId(Operation *op, AsyncTaskId asyncTaskId) {
  auto origAsyncTaskIds = getAsyncTaskIds(op);
  llvm::erase(origAsyncTaskIds, asyncTaskId);
  if (origAsyncTaskIds.empty())
    op->removeAttr("async_task_id");
  else
    setAsyncTaskIds(op, origAsyncTaskIds);
}

void removeAsyncTaskIds(Operation *op) { op->removeAttr("async_task_id"); }

} // namespace mlir::triton::gcu
