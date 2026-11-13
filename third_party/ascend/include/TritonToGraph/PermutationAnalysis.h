/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#ifndef TRITON_TO_GRAPH_PERMUTATION_ANALYSIS_H
#define TRITON_TO_GRAPH_PERMUTATION_ANALYSIS_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
namespace triton {
namespace cfg {

class EntryArgPointerAliasAnalysis;

// Every analysis decision is explicit.  Callers must treat Rejected and
// Unknown identically for authorization: neither permits an IR rewrite.
enum class ProofResult : uint8_t {
  Proven,
  Rejected,
  Unknown,
};

// A stable, inspectable explanation for an intentionally fail-closed result.
// `None` is used only with ProofResult::Proven.
enum class ProofReason : uint8_t {
  None,
  InvalidPermutationRank,
  InvalidPermutationAxis,
  DuplicatePermutationAxis,
  PermutationRankMismatch,
  ShapeRankMismatch,
  InvalidOldAxis,
  NullValue,
  UnresolvedPointer,
  UnsupportedPointerType,
  UnsupportedPointerForm,
  UnsupportedRank,
  UnsupportedEncoding,
  NonSquareShape,
  UnsupportedIndexElementType,
  DynamicShape,
  UnsupportedAffineOffset,
  DynamicStride,
  NegativeStride,
  ZeroStride,
  DuplicateStride,
  InvalidAxisProvenance,
  DuplicateAxisProvenance,
  DuplicateRangeSource,
  InvalidMakeRange,
  InvalidExpandDimsChain,
  OffsetOverflow,
  OverflowFlags,
  NonRowMajorContiguous,
  NonInjectiveLanes,
  MaskedAccess,
  BoundaryCheck,
  VolatileLoad,
  NullOperation,
  DifferentBlocks,
  InvalidProtectedInterval,
  RegionOperation,
  CallOperation,
  BarrierOperation,
  UnknownMemoryEffect,
  InterveningMemoryEffect,
  DifferentAccessBase,
  OverlappingAccessRange,
  UnsupportedInterveningMemoryAccess,
};

llvm::StringRef getProofReasonMessage(ProofReason reason);

struct ProofOutcome {
  ProofResult result = ProofResult::Unknown;
  ProofReason reason = ProofReason::UnresolvedPointer;

  constexpr bool isProven() const { return result == ProofResult::Proven; }
  constexpr bool isRejected() const { return result == ProofResult::Rejected; }
  constexpr bool isUnknown() const { return result == ProofResult::Unknown; }

  static constexpr ProofOutcome proven() {
    return {ProofResult::Proven, ProofReason::None};
  }
  static constexpr ProofOutcome rejected(ProofReason reason) {
    return {ProofResult::Rejected, reason};
  }
  static constexpr ProofOutcome unknown(ProofReason reason) {
    return {ProofResult::Unknown, reason};
  }
};

// The single convention used by this class and TTIR's `tt.trans` order:
//
//   perm[newAxis] == oldAxis
//
// If P is applied first and Q is applied after P, P.compose(Q) describes the
// combined transform and has combined[newAxis] == P[Q[newAxis]].  For example,
// P=[2, 0, 1] and Q=[1, 2, 0] compose to [0, 1, 2].
class Permutation {
public:
  // Returns a diagnostic result without constructing a value.  This is useful
  // when a caller needs to report why a tt.trans order cannot be trusted.
  static ProofOutcome validate(llvm::ArrayRef<int32_t> perm);

  // Rejects empty, out-of-range, and duplicate-axis permutations.
  static FailureOr<Permutation> create(llvm::ArrayRef<int32_t> perm);

  unsigned rank() const { return newToOld.size(); }
  llvm::ArrayRef<int32_t> getNewToOld() const { return newToOld; }

  // Valid Permutation objects always have a valid inverse.
  Permutation inverse() const;

  // Returns failure when the ranks differ.  The operation means "apply this,
  // then apply after" using the new-axis-to-old-axis convention above.
  FailureOr<Permutation> compose(const Permutation &after) const;

  // Returns -1 for an invalid old axis instead of asserting or wrapping.
  int32_t mapOldAxisToNew(int32_t oldAxis) const;

  // Returns failure when shape.rank() differs from this permutation's rank.
  // Dynamic dimension values, if present, are simply moved with their axis.
  FailureOr<llvm::SmallVector<int64_t>>
  permuteShape(llvm::ArrayRef<int64_t> shape) const;

private:
  explicit Permutation(llvm::ArrayRef<int32_t> perm)
      : newToOld(perm.begin(), perm.end()) {}

  llvm::SmallVector<int32_t, 4> newToOld;
};

// One normalized affine lane term.  `range` preserves the source operation
// identity so callers can reject a DAG that reuses one logical range for two
// axes even when those axes have equal extents.
struct StaticAccessAxis {
  triton::MakeRangeOp range;
  int64_t rangeStart = 0;
  int64_t rangeEnd = 0;
  int64_t stride = 0;
  unsigned outputAxis = 0;
};

// A statically understood affine pointer access. Rank-1 supports the
// original tt.splat + tt.addptr + tt.make_range form and the narrow dynamic
// form tt.splat(origin + static_i32) + tt.make_range. In the latter case
// `dynamicOrigin` is the shared scalar origin (or null for a fully static
// access), while `firstOffset` and `lastOffset` are the static residual
// interval relative to it. Higher ranks support a deliberately narrow
// row-major normal form: one independent make_range -> expand_dims* ->
// splat(static i32) -> muli -> broadcast term per logical axis, joined by
// addi. The fields preserve source operations and axis provenance rather
// than deriving axes from equal dimension sizes.
struct StaticAccess {
  Value pointer;
  Value offset;
  Value base;
  Value dynamicOrigin;
  llvm::SmallVector<int64_t, 4> shape;
  llvm::SmallVector<int64_t, 4> strides;
  llvm::SmallVector<unsigned, 4> axisProvenance;
  llvm::SmallVector<StaticAccessAxis, 4> axes;
  int64_t firstOffset = 0;
  int64_t lastOffset = 0;
  int64_t elementCount = 0;
  bool lanesInjective = false;

  bool isRankOneContiguous() const {
    return lanesInjective && shape.size() == 1 && strides.size() == 1 &&
           strides.front() == 1;
  }

  // Proves that flattening the logical tensor in row-major order preserves
  // its address order. This is intentionally stronger than generic lane
  // injectivity and is the admission condition for UB store packing.
  bool isLogicalRowMajorContiguous() const;
};

struct StaticAccessProof {
  ProofOutcome outcome;
  std::optional<StaticAccess> access;

  bool isProven() const { return outcome.isProven() && access.has_value(); }
};

class StaticAccessAnalysis {
public:
  // Analyze only the normalized static pointer form documented above.  A
  // block argument, nested addptr, tensor pointer, dynamic stride, or any
  // unrecognized producer is rejected or unknown; it is never assumed safe.
  StaticAccessProof analyzePointer(Value pointer) const;

  // V1 access proofs accept neither masks nor boundary/padding behavior;
  // volatile loads are likewise rejected before pointer analysis.
  StaticAccessProof analyzeLoad(triton::LoadOp load) const;
  StaticAccessProof analyzeStore(triton::StoreOp store) const;

  // Proves that two statically understood accesses use the same SSA base and
  // dynamic origin (both absent for static accesses), and have disjoint
  // closed residual offset intervals. Any missing proof is rejected.
  ProofOutcome proveSameBaseDisjoint(const StaticAccess &lhs,
                                     const StaticAccess &rhs) const;

  // Proves a sufficient injectivity condition for a fully static affine lane
  // map.  All dimensions, strides, and provenance entries must have the same
  // non-zero rank.  Dynamic, negative, zero, and duplicate strides fail
  // closed, as do duplicated axis origins.  The proof also rejects maps whose
  // nonnegative terms or maximum cumulative offset exceed signed i32, because
  // the TTIR access offsets being proven are i32 tensors.
  static ProofOutcome
  proveLaneInjectivity(llvm::ArrayRef<int64_t> shape,
                       llvm::ArrayRef<int64_t> strides,
                       llvm::ArrayRef<unsigned> axisProvenance);
};

class ProtectedIntervalAnalysis {
public:
  // Proves that the open interval (first, last) is in one MLIR block and has
  // no regions, calls, barriers, unknown operations, or memory effects.
  // This method analyzes only; it never changes the IR.
  ProofOutcome proveNoMemoryEffects(Operation *first, Operation *last) const;

  // Proves that the open interval (first, last) has no effects that conflict
  // with any protected access. Without entryArgPointerAliases, only statically
  // proven tt.load/tt.store operations with the same base and disjoint ranges
  // are permitted. With it, accesses rooted at distinct entry pointers are
  // additionally permitted under the StoreCoalescing entry-argument ABI
  // contract.
  ProofOutcome proveNoConflictingLoadStoreEffects(
      Operation *first, Operation *last,
      llvm::ArrayRef<StaticAccess> protectedAccesses,
      const EntryArgPointerAliasAnalysis *entryArgPointerAliases =
          nullptr) const;
};

} // namespace cfg
} // namespace triton
} // namespace mlir

#endif // TRITON_TO_GRAPH_PERMUTATION_ANALYSIS_H
