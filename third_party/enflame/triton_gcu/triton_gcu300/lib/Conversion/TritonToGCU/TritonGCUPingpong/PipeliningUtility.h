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

#ifndef TRITON_TRIRONTOGCU_PINGPONG_UTILITY_H_
#define TRITON_TRIRONTOGCU_PINGPONG_UTILITY_H_
#include "mlir/Dialect/SCF/IR/SCF.h"
#include <utility>
#include <vector>

namespace mlir {
namespace triton {
namespace gcu {
/// Function to mask operations during scheduling.
Operation *predicateOp(RewriterBase &rewriter, Operation *op, Value pred);

/// Collect ssa dependencies of `op` in `deps`. if `includeArg` is true,
/// continue looking through loop block arguments.
void addDep(Operation *op, DenseSet<Operation *> &deps, bool includeArg = true,
            DenseSet<Operation *> *filter = nullptr);

/// Add operations from `forOp` into a pipeline schedule with the the given
/// `stage` when filter is true. This will add operation in the original loop
/// order.
void addOps(scf::ForOp forOp, int stage,
            std::vector<std::pair<Operation *, unsigned>> &schedule,
            std::function<bool(Operation *)> filter);
} // namespace gcu
} // namespace triton
} // namespace mlir

#endif // TRITON_TRIRONTOGCU_PINGPONG_UTILITY_H_
