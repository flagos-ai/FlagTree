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

#include "Constants.h"
#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Utility.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_ANNOTATEDOTALLOCAREUSEPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;
#define DEBUG_TYPE "annotate-dot-alloca-reuse"

namespace {

/// Collect all Values transitively reachable from \p source through
/// use-def chains.  The set always includes \p source itself.
static DenseSet<Value> collectReachableSet(Value source) {
  DenseSet<Value> visited;
  SmallVector<Value> worklist;
  worklist.push_back(source);
  visited.insert(source);

  while (!worklist.empty()) {
    Value cur = worklist.pop_back_val();
    for (Operation *user : cur.getUsers()) {
      for (Value result : user->getResults()) {
        if (visited.insert(result).second)
          worklist.push_back(result);
      }
    }
  }
  return visited;
}

/// Returns true if \p earlierResult is used as a direct operand by any
/// operation that appears after \p later in the same block.  This catches
/// the case where the earlier dot's OACC buffer is still read after the
/// later dot has overwritten it.  (Transitive uses are not checked because
/// those downstream values live in local memory / registers after the
/// earlier dot's configOaccStoreToLocal, not in the shared OACC buffer.)
static bool reachesOpAfter(Operation *later, Value earlierResult) {
  Block *block = later->getBlock();
  for (auto it = std::next(later->getIterator()), e = block->end(); it != e;
       ++it) {
    for (Value operand : it->getOperands()) {
      if (operand == earlierResult)
        return true;
    }
  }
  return false;
}

/// Returns true when dot \p later can reuse the oacc alloca that was
/// allocated for dot \p earlier.  Reuse is safe when:
///   0. Both dots are in the same block (alloca scoping).
///   1. The later dot's accumulator (C operand) does not depend on the
///      earlier dot's result (checked via \p earlierReachable).
///   2. No operation after the later dot directly uses earlierResult.
static bool canReuseAlloca(triton::DotOp earlier, triton::DotOp later,
                           const DenseSet<Value> &earlierReachable) {
  // Both must be 2-D matmuls
  if (earlier.getType().getRank() != 2 || later.getType().getRank() != 2)
    return false;

  // Compare only shape and element type, ignoring layout encodings.
  auto earlierTy = cast<RankedTensorType>(earlier.getType());
  auto laterTy = cast<RankedTensorType>(later.getType());
  if (earlierTy.getShape() != laterTy.getShape() ||
      earlierTy.getElementType() != laterTy.getElementType())
    return false;

  // (0) Both dots must live in the same block.  If they are in different
  //     scopes (e.g. one inside a loop, the other outside), their allocas
  //     have disjoint lifetimes and cannot be shared.
  if (earlier->getBlock() != later->getBlock())
    return false;

  // (1) If either dot has acc_reuse_candidate = "acc_reuse_oacc", its OACC
  //     buffer stays live (not stored to local).  Sharing would clobber it.
  auto isOaccReuse = [](triton::DotOp dot) {
    if (auto attr = dot->getAttrOfType<StringAttr>(kAccReuseCandidate))
      return attr.getValue() == kAccReuseOacc;
    return false;
  };
  if (isOaccReuse(earlier) || isOaccReuse(later))
    return false;

  // (2) laterC must not be reachable from earlierResult.
  Value laterC = later.getC();
  if (earlierReachable.contains(laterC))
    return false;

  // (3) No operation after laterDot must directly use earlierResult.
  //     If it did, the earlier result would still be directly needed
  //     from the OACC buffer when the later dot overwrites it.
  //     (Transitive uses are safe: after configOaccStoreToLocal the
  //     derived values live in local memory / registers.)
  Value earlierResult = earlier.getResult();
  if (reachesOpAfter(later, earlierResult))
    return false;

  LLVM_DEBUG(llvm::dbgs() << "AnnotateDotAllocaReuse: dot " << later
                          << " can reuse alloca"
                          << " from dot " << earlier << "\n");
  return true;
}

struct AnnotateDotAllocaReusePass
    : public impl::AnnotateDotAllocaReusePassBase<AnnotateDotAllocaReusePass> {
  using Base::Base;

  void runOnOperation() override {
    auto mod = getOperation();
    auto i64Ty = IntegerType::get(mod.getContext(), 64);

    // Collect all tt.dot ops in program order.
    SmallVector<triton::DotOp> dots;
    mod.walk([&](triton::DotOp dotOp) { dots.push_back(dotOp); });

    if (dots.size() < 2)
      return;

    // Pre-compute the reachable set for each dot's result so we only
    // traverse the use-def graph once per dot instead of once per pair.
    SmallVector<DenseSet<Value>> reachableSets(dots.size());
    for (unsigned i = 0; i < dots.size(); ++i)
      reachableSets[i] = collectReachableSet(dots[i].getResult());

    // Assign alloca group IDs.
    // Dots that can share an alloca get the same group ID.
    int64_t nextGroupId = 0;
    SmallVector<int64_t> groupIds(dots.size(), -1);

    for (unsigned i = 0; i < dots.size(); ++i) {
      // Search backwards for a compatible earlier dot.
      bool foundReuse = false;
      for (int j = static_cast<int>(i) - 1; j >= 0; --j) {
        if (!canReuseAlloca(dots[j], dots[i], reachableSets[j]))
          continue;

        // Verify dots[i] is also compatible with every other member
        // in dots[j]'s group.  If dots[j] shares an alloca with dots[k],
        // then dots[i] must be safe to share with dots[k] as well.
        // Example: dot0 & dot1 already share; dot2 depends on dot1 but
        // not dot0.  Without this check dot2 would join via dot0 and
        // clobber dot1's result.
        bool groupCompatible = true;
        for (unsigned k = 0; k < i; ++k) {
          if (static_cast<int>(k) == j)
            continue;
          if (groupIds[k] != groupIds[j])
            continue;
          if (!canReuseAlloca(dots[k], dots[i], reachableSets[k])) {
            groupCompatible = false;
            LLVM_DEBUG(llvm::dbgs()
                       << "AnnotateDotAllocaReuse: dot " << dots[i]
                       << " incompatible with group-mate dot " << dots[k]
                       << " (via leader dot " << dots[j] << ")\n");
            break;
          }
        }
        if (!groupCompatible)
          continue;

        groupIds[i] = groupIds[j];
        foundReuse = true;
        break;
      }
      if (!foundReuse)
        groupIds[i] = nextGroupId++;
    }

    // Annotate dots with their group ID.
    // Only annotate when the group is actually shared (>= 2 members).
    DenseMap<int64_t, unsigned> groupCount;
    for (int64_t gid : groupIds)
      if (gid >= 0)
        ++groupCount[gid];

    for (unsigned i = 0; i < dots.size(); ++i) {
      int64_t gid = groupIds[i];
      if (gid >= 0 && groupCount.lookup(gid) >= 2) {
        dots[i]->setAttr(kAllocaReuseGroup, IntegerAttr::get(i64Ty, gid));
      }
    }
  }
};

} // namespace
