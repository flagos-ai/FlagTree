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

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <limits>
#include <optional>

using namespace mlir;
using mlir::triton::controlflow::ControlFlowOpAnalysis;
using mlir::triton::controlflow::ControlFlowRewriteContext;
using mlir::triton::controlflow::ControlFlowRewritePlan;
using mlir::triton::controlflow::ControlFlowRewritePolicy;
using mlir::triton::controlflow::ControlFlowSlotAnalysis;
using mlir::triton::controlflow::DecomposedValue;
using mlir::triton::controlflow::kPointerDescriptorBoundaryAttr;
using mlir::triton::controlflow::kPointerDescriptorRebuildAttr;

namespace mlir::triton::controlflow {

Value ControlFlowRewriteContext::remap(Value value) const {
  if (Value mapped = valueMapping.lookupOrNull(value))
    return mapped;
  return value;
}

const DecomposedValue *ControlFlowRewriteContext::lookup(Value value) const {
  auto it = decomposedValues.find(value);
  return it == decomposedValues.end() ? nullptr : &it->second;
}

} // namespace mlir::triton::controlflow

namespace {

// Keep the mechanical structured-control-flow and Scope rewrite in one
// translation unit. These handlers are mutually recursive through
// rewriteBodyOps(), share one
// short-lived RewriteEnv, and must agree on signature expansion, nested-op
// ordering and failure cleanup. Splitting them by op kind would expose those
// private constraints through additional internal headers without creating an
// independently reusable component.
//
//===----------------------------------------------------------------------===//
// Per-rewrite state and generic component manipulation
//===----------------------------------------------------------------------===//

/// Carries everything needed to translate values from one original
/// control-flow path into its replacement path.
///
/// valueMapping answers which new SSA value replaces an old SSA value.
/// decomposedValues remembers the policy-specific pieces of an old pointer.
/// policy defines what those pieces mean and how to rebuild the pointer,
/// while plan says which control-flow values must be expanded into such pieces.
///
/// For example, after a block-pointer loop argument is expanded into scalar
/// offsets, a region-local environment can contain:
/// valueMapping:      %old_ptr_arg -> %rebuilt_ptr
/// decomposedValues:  %old_ptr_arg -> {
///   components = [%base, %shape0, %stride0, %new_offset0],
///   attributes = [order]
/// }
/// Operations cloned into that region use the mapping, while pointer
/// decomposition uses the stored components.
struct RewriteEnv {
  /// Starts a rewrite path with no old-to-new mappings or decomposed values.
  /// The referenced policy and analysis plan are shared by child environments;
  /// only the two mutable state tables above are copied for each nested region.
  ///
  /// For example, the top-level environment may enter a rewritten scf.for.
  /// The loop body copies it, then adds mappings from the old body arguments to
  /// the new body arguments without changing the state of sibling regions:
  /// RewriteEnv env(blockPtrPolicy, rewritePlan);
  /// RewriteEnv bodyEnv = env;
  /// bodyEnv.valueMapping.map(oldBodyArg, newBodyArg);
  RewriteEnv(const ControlFlowRewritePolicy &policy,
             const ControlFlowRewritePlan &plan)
      : policy(policy), plan(plan) {}

  /// Gives a decomposition policy read-only access to this path's two state
  /// tables. The context can resolve an old SSA value to its replacement and
  /// retrieve a previously stored DecomposedValue; it does not own or copy
  /// either table and therefore must not outlive this environment.
  ///
  /// For example, while the block-pointer policy decomposes this operation,
  /// it needs both the replacement delta and the saved state of the input:
  /// %next = tt.advance %old_ptr, [%old_delta]
  ///
  /// valueMapping:      %old_delta -> %new_delta
  /// decomposedValues:  %old_ptr   -> ptrInfo
  ///
  /// auto context = bodyEnv.getRewriteContext();
  /// context.remap(oldDelta); // The Value for %new_delta.
  /// context.lookup(oldPtr);  // A pointer to ptrInfo.
  ControlFlowRewriteContext getRewriteContext() const {
    return ControlFlowRewriteContext(valueMapping, decomposedValues);
  }

  /// Translates one SSA value referenced by the original IR into the value that
  /// must be used in the replacement IR. It queries valueMapping; a value
  /// defined outside the rewritten area is already valid and is returned
  /// unchanged when the table has no entry.
  ///
  /// This prevents a newly built operation from referring back to a block
  /// argument or result owned by the old region. For example:
  /// valueMapping:  %old_iter_arg -> %new_iter_arg
  ///
  /// // Original: scf.yield %old_iter_arg, %outer_value
  /// newYieldOperands = {
  ///   bodyEnv.remap(oldIterArg), // The Value for %new_iter_arg.
  ///   bodyEnv.remap(outerValue)  // The unchanged %outer_value.
  /// };
  /// The returned values are then used as operands of the replacement yield.
  Value remap(Value value) const { return getRewriteContext().remap(value); }

  /// Asks the active policy to express one high-level SSA value as the runtime
  /// components that may cross an expanded supported control-flow boundary.
  /// value is the original pointer-like SSA value. The policy
  /// receives the current rewrite context so it can reuse a known decomposition
  /// and remap operands to the replacement IR. It may use builder and loc
  /// to insert scalar address arithmetic at the correct rewrite position.
  ///
  /// For the block-pointer policy, decomposing an advance conceptually changes
  /// only its offset components; base, shape, strides, and order are preserved:
  /// %next = tt.advance %ptr, [%delta0, %delta1]
  ///
  /// decomposeValue(%next) -> DecomposedValue {
  ///   components = [%base, %shape0, %shape1, %stride0, %stride1,
  ///                 %offset0 + %delta0, %offset1 + %delta1],
  ///   attributes = [order]
  /// }
  /// The caller can put selected components into a new control-flow signature
  /// or pass the whole descriptor to policy.recompose(). Unsupported values
  /// or inconsistent component layouts return failure.
  FailureOr<DecomposedValue> decomposeValue(Value value, OpBuilder &builder,
                                            Location loc) const {
    return policy.decompose(value, getRewriteContext(), builder, loc);
  }

  /// Records both ways in which later rewriting must understand an original
  /// value. oldValue is the key from the original IR, info is its component
  /// descriptor, and rebuiltValue is the pointer-like SSA value created in the
  /// replacement IR. Pointer-aware code may encounter either SSA value, so
  /// both keys resolve to info; ordinary cloned users read rebuiltValue through
  /// valueMapping.
  ///
  /// For example, after rebuilding an expanded loop argument:
  /// %rebuilt_ptr = tt.make_tensor_ptr %base, ... %new_offset ...
  /// bodyEnv.recordDecomposition(oldPtrArg, ptrInfo, rebuiltPtr);
  ///
  /// // An ordinary cloned tt.load receives %rebuilt_ptr.
  /// bodyEnv.remap(oldPtrArg);
  ///
  /// // A later tt.advance/yield reuses the flattened ptrInfo directly.
  /// bodyEnv.getRewriteContext().lookup(oldPtrArg);
  void recordDecomposition(Value oldValue, const DecomposedValue &info,
                           Value rebuiltValue) {
    decomposedValues[oldValue] = info;
    // A later cloned operation may consume the rebuilt value directly rather
    // than the original value. Cache the same descriptor under that SSA value
    // so chained pointer producers reuse the existing decomposition when the
    // active policy says that the rebuilt representation is downstream-safe.
    if (policy.shouldCacheRebuiltDecomposition(info))
      decomposedValues[rebuiltValue] = info;
    valueMapping.map(oldValue, rebuiltValue);
  }

  // Maps values from the original region to values in the replacement region.
  IRMapping valueMapping;
  // Concrete component state keyed by original and rebuilt pointer values.
  // Keeping this alongside the mapping lets chained pointer producers and
  // nested rewrites reuse an already materialized descriptor.
  DenseMap<Value, DecomposedValue> decomposedValues;
  const ControlFlowRewritePolicy &policy;
  const ControlFlowRewritePlan &plan;
};

// RewriteEnv is copied when entering a newly built region. The copy inherits
// mappings visible at the region boundary and records additional mappings only
// for that recursive rewrite. Nothing is stored on the IR or shared between
// decomposition policies.

struct LoopPointerInfo {
  // Original iter-argument/result position before signature expansion.
  unsigned oldIndex = 0;
  // Concrete descriptor used as the reconstruction template.
  DecomposedValue initInfo;
  // Ordered schema decided by ControlFlowSlotAnalysis.
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  // Policy metadata after all loop incoming paths have been normalized.
  SmallVector<Attribute> resultAttributes;
  // Positions occupied by those components in the replacement operation.
  SmallVector<unsigned> newIndices;
};

struct IfPointerInfo {
  unsigned oldIndex = 0;
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  SmallVector<Attribute> resultAttributes;
  std::optional<DecomposedValue> thenInfo;
};

struct ScopePointerInfo {
  unsigned oldIndex = 0;
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  SmallVector<Attribute> resultAttributes;
  std::optional<DecomposedValue> returnedInfo;
};

/// Returns the terminator of the single-block Scope form supported by the
/// dedicated rewrite. The analyzer already checked these conditions, but the
/// frozen plan is consumed after analysis, so the rewrite validates them again
/// before creating replacement IR.
static FailureOr<scope::ReturnOp>
getSupportedScopeReturn(scope::ScopeOp scopeOp) {
  if (scopeOp->getNumOperands() != 0 || scopeOp->getNumRegions() != 1)
    return failure();
  Region &region = scopeOp.getBodyRegion();
  if (!llvm::hasSingleElement(region))
    return failure();
  Block &body = region.front();
  if (body.getNumArguments() != 0)
    return failure();

  auto returnOp = dyn_cast<scope::ReturnOp>(body.getTerminator());
  if (!returnOp || returnOp.getNumOperands() != scopeOp.getNumResults())
    return failure();
  for (auto [operand, result] :
       llvm::zip(returnOp.getOperands(), scopeOp.getResults())) {
    if (operand.getType() != result.getType())
      return failure();
  }
  return returnOp;
}

// Updates the downstream descriptor marker after a loop signature expansion.
// Existing slots are remapped through oldToNewStart before slots introduced by
// the active policy are merged. The array is always expressed in the
// replacement loop's iter-argument/result coordinate space.
//
// Example:
//   old slots = [1], oldToNewStart = [0, 1, 3]
//   the active policy expands old slot 2 into new slots [3, 4]
//   result = [1, 3, 4]
static LogicalResult updatePointerDescriptorBoundaryMarker(
    Operation *loop, ArrayRef<LoopPointerInfo> pointerInfos,
    ArrayRef<unsigned> oldToNewStart, const ControlFlowRewritePolicy &policy) {
  SmallVector<int32_t> descriptorSlots;
  bool ownsPointerBoundary = false;
  llvm::SmallDenseSet<unsigned> seenSlots;
  auto appendSlot = [&](unsigned slot) -> LogicalResult {
    if (slot > static_cast<unsigned>(std::numeric_limits<int32_t>::max()))
      return failure();
    if (!seenSlots.insert(slot).second)
      return failure();
    descriptorSlots.push_back(static_cast<int32_t>(slot));
    return success();
  };

  if (Attribute oldMarker = loop->getAttr(kPointerDescriptorBoundaryAttr)) {
    ownsPointerBoundary = true;
    auto oldSlots = dyn_cast<DenseI32ArrayAttr>(oldMarker);
    if (!oldSlots)
      return failure();
    for (int32_t oldSlot : oldSlots.asArrayRef()) {
      if (oldSlot < 0 ||
          static_cast<unsigned>(oldSlot) >= oldToNewStart.size() ||
          failed(appendSlot(oldToNewStart[oldSlot])))
        return failure();
    }
  }

  if (policy.requiresPointerDescriptorBoundaryMarker()) {
    ownsPointerBoundary |= !pointerInfos.empty();
    for (const LoopPointerInfo &pointerInfo : pointerInfos) {
      for (unsigned newSlot : pointerInfo.newIndices) {
        if (failed(appendSlot(newSlot)))
          return failure();
      }
    }
  }

  if (!ownsPointerBoundary) {
    loop->removeAttr(kPointerDescriptorBoundaryAttr);
    return success();
  }
  llvm::sort(descriptorSlots);
  loop->setAttr(kPointerDescriptorBoundaryAttr,
                DenseI32ArrayAttr::get(loop->getContext(), descriptorSlots));
  return success();
}

// Marks policy-created pointer descriptor reconstructions. Their defining
// operation consumes the complete descriptor, including invariant components
// omitted from a replacement SCF signature. TritonToLinalg uses these operands
// as precise preservation roots and removes the marker after conversion.
static LogicalResult
markPointerDescriptorRebuild(Value rebuiltPointer, const DecomposedValue &value,
                             const ControlFlowRewritePolicy &policy) {
  if (!policy.requiresPointerDescriptorBoundaryMarker())
    return success();
  Operation *definingOp = rebuiltPointer.getDefiningOp();
  if (!definingOp)
    return failure();
  definingOp->setAttr(kPointerDescriptorRebuildAttr,
                      UnitAttr::get(definingOp->getContext()));
  return policy.annotatePointerDescriptorRebuild(definingOp, value);
}

// Copies the values selected by indices into a new owning vector while
// preserving their input order. Indices must be unique and in bounds; this
// function reports failure instead of deduplicating or accessing invalid input.
// `sourceValues` is only borrowed while this function executes.
//
// Example: sourceValues = [shape, stride, offset] and indices = [2, 0, 1]
// produce [offset, shape, stride]. Indices [2, 0, 2] produce failure.
static FailureOr<SmallVector<Value>> gatherValues(ValueRange sourceValues,
                                                  ArrayRef<unsigned> indices) {
  SmallVector<Value> values;
  values.reserve(indices.size());
  llvm::SmallDenseSet<unsigned> seenIndices;
  for (unsigned index : indices) {
    if (index >= sourceValues.size() || !seenIndices.insert(index).second)
      return failure();
    values.push_back(sourceValues[index]);
  }
  return values;
}

// Returns a copy of decomposition with selected component values replaced.
// componentIndices and replacements are paired by position. The replacement
// fails if the ranges have different sizes, an index is out of bounds, or a
// replacement changes the component type; the input object remains unchanged.
//
// Example:
//   decomposition.components = [base, shape, stride, originalOffset]
//   componentIndices = [3], replacements = [nextOffset]
//   result.components = [base, shape, stride, nextOffset]
static FailureOr<DecomposedValue>
withReplacedComponents(DecomposedValue decomposition,
                       ArrayRef<unsigned> componentIndices,
                       ArrayRef<Value> replacements) {
  if (componentIndices.size() != replacements.size())
    return failure();
  for (auto [componentIndex, replacement] :
       llvm::zip(componentIndices, replacements)) {
    if (componentIndex >= decomposition.components.size() ||
        decomposition.components[componentIndex].getType() !=
            replacement.getType())
      return failure();
    decomposition.components[componentIndex] = replacement;
  }
  return decomposition;
}

static LogicalResult castPlannedComponents(DecomposedValue &value,
                                           ArrayRef<unsigned> componentIndices,
                                           ArrayRef<Type> componentTypes,
                                           OpBuilder &builder, Location loc) {
  if (componentIndices.size() != componentTypes.size())
    return failure();
  for (auto [index, type] : llvm::zip(componentIndices, componentTypes)) {
    if (index >= value.components.size())
      return failure();
    FailureOr<Value> component =
        castIntegerLike(builder, loc, value.components[index], type);
    if (failed(component))
      return failure();
    value.components[index] = *component;
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Shared recursive body rewrite
//===----------------------------------------------------------------------===//

// Returns the first pointer descriptor whose oldIndex matches the requested
// position in the original control-flow signature. The range is an lvalue
// reference so the returned pointer refers to caller-owned storage, and the
// return type preserves whether that range exposes mutable or const elements.
// A missing index returns nullptr; duplicate indices keep first-match behavior.
//
// Example: [{oldIndex = 1}, {oldIndex = 3}] queried with 3 returns a pointer to
// the second element, while a query for 2 returns nullptr.
template <typename InfoRange>
static auto findPointerInfoByOldIndex(InfoRange &pointerInfos,
                                      unsigned oldIndex)
    -> decltype(pointerInfos.data()) {
  for (auto &info : pointerInfos) {
    if (info.oldIndex == oldIndex)
      return &info;
  }
  return nullptr;
}

// A replacement loop carries policy-selected scalar or tensor descriptor
// components instead of the original pointer iter-argument. BlockPtr selects
// every component that analysis found dynamic; TensorPtr selects only the
// base/offset fields whose symbolic identities change at that boundary.
// Operations cloned from the original body still expect one pointer-typed
// block argument, so this function reconstructs that pointer at the replacement
// region entry.
// `pointerInfo.newIndices` selects the current component values from
// `newRegionArguments`, while
// `pointerInfo.componentIndices` identifies the descriptor fields that those
// values replace. The untouched fields come from `pointerInfo.initInfo`.
//
// On success, this function updates both parts of `regionEnv`: remapping the
// old region argument produces the rebuilt pointer, and looking up its
// decomposition returns the descriptor containing the current components.
// Operations cloned later in the same region can therefore use the complete
// pointer or its flattened state without referring to the old loop block.
//
// Example:
//   oldRegionArgument = %old_ptr
//   newRegionArguments = [%ordinary, %base, %shape, %stride, %offset]
//   pointerInfo.newIndices = [1, 2, 3, 4]
//   pointerInfo.componentIndices = [0, 1, 2, 3]
//   pointerInfo.initInfo.components =
//       [initial_base, initial_shape, initial_stride, initial_offset]
//
// The rebuilt BlockPtr descriptor replaces all four fields with the current
// loop values and records `%old_ptr -> %rebuilt_ptr` in `regionEnv`. Invalid
// indices, incompatible component types, or a policy that cannot recompose the
// descriptor return failure without recording a partial binding; the enclosing
// loop rewrite owns cleanup of inserted IR.
static LogicalResult bindLoopCarriedPointer(Value oldRegionArgument,
                                            const LoopPointerInfo &pointerInfo,
                                            ValueRange newRegionArguments,
                                            OpBuilder &builder, Location loc,
                                            RewriteEnv &regionEnv) {
  FailureOr<SmallVector<Value>> carriedComponentValues =
      gatherValues(newRegionArguments, pointerInfo.newIndices);
  if (failed(carriedComponentValues))
    return failure();

  FailureOr<DecomposedValue> argumentInfo =
      withReplacedComponents(pointerInfo.initInfo, pointerInfo.componentIndices,
                             *carriedComponentValues);
  if (failed(argumentInfo) || failed(regionEnv.policy.normalizeLoopCarriedValue(
                                  *argumentInfo, pointerInfo.componentIndices,
                                  pointerInfo.resultAttributes, builder, loc)))
    return failure();

  Value rebuiltPointer =
      regionEnv.policy.recompose(*argumentInfo, builder, loc);
  if (!rebuiltPointer || failed(markPointerDescriptorRebuild(
                             rebuiltPointer, *argumentInfo, regionEnv.policy)))
    return failure();

  regionEnv.recordDecomposition(oldRegionArgument, *argumentInfo,
                                rebuiltPointer);
  return success();
}

// Binds every original loop region argument to its replacement-region state.
// A pointer slot delegates to bindLoopCarriedPointer because one original
// pointer may occupy several component positions in the new signature. An
// ordinary slot keeps one SSA value and is mapped through oldToNewStart.
//
// Example:
//   oldRegionArguments = [%x, %old_ptr, %y]
//   newRegionArguments = [%new_x, %offset0, %offset1, %new_y]
//   pointerInfos = [{oldIndex = 1, newIndices = [1, 2]}]
//   oldToNewStart = [0, 1, 3]
//
// The resulting environment maps %x to %new_x and %y to %new_y. For
// %old_ptr, it gathers %offset0 and %offset1, rebuilds the complete pointer,
// and records both its SSA mapping and current decomposition.
//
// The function deliberately visits every argument after a binding failure.
// The surrounding builder callback may still need all available mappings to
// construct structurally valid temporary IR before its new loop is erased.
// It therefore accumulates failure and reports it only after the full range.
static LogicalResult bindLoopRegionArguments(
    ValueRange oldRegionArguments, ValueRange newRegionArguments,
    ArrayRef<LoopPointerInfo> pointerInfos, ArrayRef<unsigned> oldToNewStart,
    OpBuilder &builder, Location loc, RewriteEnv &regionEnv) {
  bool allArgumentsBound = true;
  for (auto [oldIndex, oldRegionArgument] :
       llvm::enumerate(oldRegionArguments)) {
    const LoopPointerInfo *pointerInfo =
        findPointerInfoByOldIndex(pointerInfos, oldIndex);
    if (pointerInfo) {
      if (failed(bindLoopCarriedPointer(oldRegionArgument, *pointerInfo,
                                        newRegionArguments, builder, loc,
                                        regionEnv)))
        allArgumentsBound = false;
      continue;
    }

    if (oldIndex >= oldToNewStart.size() ||
        oldToNewStart[oldIndex] >= newRegionArguments.size()) {
      allArgumentsBound = false;
      continue;
    }
    regionEnv.valueMapping.map(oldRegionArgument,
                               newRegionArguments[oldToNewStart[oldIndex]]);
  }
  return allArgumentsBound ? success() : failure();
}

// Rewrites one original loop terminator operand list to the expanded signature
// used by the replacement loop. Ordinary operands remain one value and are
// remapped through `regionEnv`. A pointer operand is decomposed, normalized to
// the component types frozen by analysis, and expanded into the components
// selected by its `LoopPointerInfo`.
//
// Example:
//   oldOperands = [%next_ptr, %sum]
//   pointerInfo = {
//     oldIndex = 0, componentIndices = [0, 1, 2, 3],
//     newIndices = [0, 1, 2, 3]
//   }
//   currentRegionArguments =
//       [%current_base, %current_shape, %current_stride, %current_offset,
//        %sum_arg]
//
// A valid `%next_ptr` decomposition produces
// `[%next_base, %next_shape, %next_stride, %next_offset, %mapped_sum]`. If
// pointer decomposition or component normalization fails, the output instead
// uses the four current descriptor arguments followed by `%mapped_sum`. The
// fallback keeps the temporary scf.yield/scf.condition structurally complete
// until the enclosing failed loop rewrite erases it.
//
// The output vector is separate from the LogicalResult intentionally. The
// function visits every old operand and fills all available fallback positions
// even after a pointer failure, then reports whether every pointer succeeded.
static LogicalResult rewriteLoopTerminatorOperands(
    ValueRange oldOperands, ValueRange currentRegionArguments,
    ArrayRef<LoopPointerInfo> pointerInfos, OpBuilder &builder, Location loc,
    RewriteEnv &regionEnv, SmallVectorImpl<Value> &newOperands) {
  bool allOperandsValid = true;
  newOperands.clear();
  newOperands.reserve(currentRegionArguments.size());

  auto appendFallbackComponents = [&](const LoopPointerInfo &pointerInfo) {
    for (unsigned newIndex : pointerInfo.newIndices)
      newOperands.push_back(currentRegionArguments[newIndex]);
  };

  for (auto [oldIndex, oldOperand] : llvm::enumerate(oldOperands)) {
    const LoopPointerInfo *pointerInfo =
        findPointerInfoByOldIndex(pointerInfos, oldIndex);
    if (!pointerInfo) {
      newOperands.push_back(regionEnv.remap(oldOperand));
      continue;
    }

    FailureOr<DecomposedValue> nextInfo =
        regionEnv.decomposeValue(oldOperand, builder, loc);
    if (failed(nextInfo) ||
        failed(castPlannedComponents(*nextInfo, pointerInfo->componentIndices,
                                     pointerInfo->componentTypes, builder,
                                     loc)) ||
        failed(regionEnv.policy.normalizeLoopCarriedValue(
            *nextInfo, pointerInfo->componentIndices,
            pointerInfo->resultAttributes, builder, loc))) {
      allOperandsValid = false;
      appendFallbackComponents(*pointerInfo);
      continue;
    }

    FailureOr<SmallVector<Value>> carriedComponentValues =
        gatherValues(nextInfo->components, pointerInfo->componentIndices);
    if (failed(carriedComponentValues)) {
      allOperandsValid = false;
      appendFallbackComponents(*pointerInfo);
      continue;
    }
    newOperands.append(carriedComponentValues->begin(),
                       carriedComponentValues->end());
  }

  return allOperandsValid ? success() : failure();
}

// Rebuilds and records every result produced by a replacement loop. Ordinary
// results still occupy one position and are mapped through `oldToNewStart`.
// A pointer result occupies the positions listed in its `LoopPointerInfo`; the
// function gathers those components, writes them into the initial descriptor,
// recomposes the complete pointer, and records both its SSA mapping and current
// decomposition in `env`.
//
// Example:
//   oldResults = [%old_sum, %old_ptr, %old_flag]
//   newResults =
//       [%new_sum, %base, %shape, %stride, %offset, %new_flag]
//   pointerInfo = {
//     oldIndex = 1, componentIndices = [0, 1, 2, 3],
//     newIndices = [1, 2, 3, 4]
//   }
//   oldToNewStart = [0, 1, 5]
//
// The function maps `%old_sum -> %new_sum` and `%old_flag -> %new_flag`. It
// inserts all four results into the pointer descriptor, rebuilds `%old_ptr`,
// and records `%old_ptr -> %rebuilt_ptr` plus that decomposition.
//
// The caller must set the builder insertion point after the replacement loop,
// so any rebuilt pointer dominates later operations. On failure this function
// returns immediately; the caller still owns and erases the replacement loop.
static LogicalResult
rebuildAndMapLoopResults(ValueRange oldResults, ValueRange newResults,
                         ArrayRef<LoopPointerInfo> pointerInfos,
                         ArrayRef<unsigned> oldToNewStart, OpBuilder &builder,
                         RewriteEnv &env) {
  for (auto [oldIndex, oldResult] : llvm::enumerate(oldResults)) {
    const LoopPointerInfo *pointerInfo =
        findPointerInfoByOldIndex(pointerInfos, oldIndex);
    if (!pointerInfo) {
      env.valueMapping.map(oldResult, newResults[oldToNewStart[oldIndex]]);
      continue;
    }

    FailureOr<SmallVector<Value>> resultComponentValues =
        gatherValues(newResults, pointerInfo->newIndices);
    if (failed(resultComponentValues))
      return failure();

    FailureOr<DecomposedValue> resultInfo = withReplacedComponents(
        pointerInfo->initInfo, pointerInfo->componentIndices,
        *resultComponentValues);
    if (failed(resultInfo) ||
        failed(env.policy.normalizeLoopCarriedValue(
            *resultInfo, pointerInfo->componentIndices,
            pointerInfo->resultAttributes, builder, oldResult.getLoc())))
      return failure();

    Value rebuiltPointer =
        env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
    if (!rebuiltPointer || failed(markPointerDescriptorRebuild(
                               rebuiltPointer, *resultInfo, env.policy)))
      return failure();

    env.recordDecomposition(oldResult, *resultInfo, rebuiltPointer);
  }

  return success();
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env);

static LogicalResult materializePointerResult(Operation &originalOp,
                                              Operation *clonedOp,
                                              OpBuilder &builder,
                                              RewriteEnv &env) {
  // Each policy decides which pointer-producing operations need their exact
  // components recorded immediately after cloning.
  if (!env.policy.shouldDecomposeOperation(&originalOp))
    return success();

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointAfter(clonedOp);

  bool decomposedAllResults = clonedOp->getNumResults() != 0;
  for (auto [oldResult, clonedResult] :
       llvm::zip(originalOp.getResults(), clonedOp->getResults())) {
    if (!env.policy.isDecompositionTarget(oldResult)) {
      decomposedAllResults = false;
      continue;
    }

    FailureOr<DecomposedValue> info =
        env.decomposeValue(clonedResult, builder, oldResult.getLoc());
    if (failed(info))
      return failure();

    Value rebuilt = env.policy.recompose(*info, builder, oldResult.getLoc());
    if (!rebuilt)
      return failure();
    if (env.policy.shouldMarkOperationRecomposition(&originalOp) &&
        failed(markPointerDescriptorRebuild(rebuilt, *info, env.policy)))
      return failure();
    env.recordDecomposition(oldResult, *info, rebuilt);
  }

  // The replacement is recorded in the SSA mapping, so a side-effect-free
  // clone whose every result was decomposed is redundant once it has no users.
  if (decomposedAllResults && clonedOp->use_empty() &&
      isMemoryEffectFree(clonedOp))
    clonedOp->erase();

  return success();
}

static LogicalResult rewriteBodyOps(Block *oldBlock, OpBuilder &builder,
                                    RewriteEnv &env) {
  // Process operations in program order. Nested control flow is rewritten
  // recursively with the same policy; ordinary operations are cloned through
  // the current SSA mapping.
  for (Operation &originalOp : oldBlock->without_terminator()) {
    if (isa<scf::ForOp, scf::WhileOp, scf::IfOp, scope::ScopeOp>(originalOp)) {
      const ControlFlowOpAnalysis *analysis = env.plan.lookup(&originalOp);
      if (!analysis)
        return failure();
      if (analysis->needsRewrite()) {
        if (failed(rewriteControlFlowOp(&originalOp, builder, env)))
          return failure();
        continue;
      }
    }
    Operation *clonedOp = builder.clone(originalOp, env.valueMapping);
    if (failed(materializePointerResult(originalOp, clonedOp, builder, env)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// scf.for rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteForOp(scf::ForOp forOp, OpBuilder &builder,
                                  RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(forOp);
  if (!analysis || !analysis->needsRewrite())
    return failure();
  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // The read-only analysis has already fixed every expanded slot and type.
  // Materialization here recovers only the concrete values for that schema.
  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    unsigned idx = slot.oldIndex;
    if (idx >= forOp.getInitArgs().size() || idx >= yieldOp.getNumOperands() ||
        !env.policy.matches(forOp.getRegionIterArgs()[idx].getType()))
      return failure();

    FailureOr<DecomposedValue> initInfo =
        env.decomposeValue(forOp.getInitArgs()[idx], builder, forOp.getLoc());
    if (failed(initInfo) ||
        failed(env.policy.normalizeLoopCarriedValue(
            *initInfo, slot.componentIndices, slot.resultAttributes, builder,
            forOp.getLoc())) ||
        failed(castPlannedComponents(*initInfo, slot.componentIndices,
                                     slot.componentTypes, builder,
                                     forOp.getLoc())))
      return failure();
    pointerInfos.push_back(LoopPointerInfo{idx,
                                           *initInfo,
                                           slot.componentIndices,
                                           slot.componentTypes,
                                           slot.resultAttributes,
                                           {}});
  }

  SmallVector<Value> newInitArgs;
  SmallVector<unsigned> oldToNewStart(forOp.getInitArgs().size(), 0);
  // Expand each owned pointer init into its runtime components. Non-pointer and
  // other-policy slots retain one position in the new signature.
  for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
    oldToNewStart[idx] = newInitArgs.size();
    if (LoopPointerInfo *info = findPointerInfoByOldIndex(pointerInfos, idx)) {
      FailureOr<SmallVector<Value>> initComponents =
          gatherValues(info->initInfo.components, info->componentIndices);
      if (failed(initComponents))
        return failure();
      for (Value component : *initComponents) {
        info->newIndices.push_back(newInitArgs.size());
        newInitArgs.push_back(component);
      }
      continue;
    }
    newInitArgs.push_back(env.remap(initArg));
  }

  bool bodyOk = true;
  auto newForOp = builder.create<scf::ForOp>(
      forOp.getLoc(), env.remap(forOp.getLowerBound()),
      env.remap(forOp.getUpperBound()), env.remap(forOp.getStep()), newInitArgs,
      [&](OpBuilder &bodyBuilder, Location loc, Value iv,
          ValueRange newRegionArgs) {
        RewriteEnv bodyEnv = env;
        bodyEnv.valueMapping.map(forOp.getInductionVar(), iv);

        // Bind pointer and ordinary iter-arguments before cloning body users.
        if (failed(bindLoopRegionArguments(
                forOp.getRegionIterArgs(), newRegionArgs, pointerInfos,
                oldToNewStart, bodyBuilder, loc, bodyEnv)))
          bodyOk = false;

        if (failed(rewriteBodyOps(forOp.getBody(), bodyBuilder, bodyEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        if (failed(rewriteLoopTerminatorOperands(
                yieldOp.getOperands(), newRegionArgs, pointerInfos, bodyBuilder,
                yieldOp.getLoc(), bodyEnv, newYieldOperands)))
          bodyOk = false;

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newForOp->setAttrs(forOp->getAttrs());
  if (analysis->rewritesOwnSignature() &&
      failed(updatePointerDescriptorBoundaryMarker(
          newForOp, pointerInfos, oldToNewStart, env.policy))) {
    newForOp.erase();
    return failure();
  }

  if (!bodyOk) {
    newForOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newForOp);
  if (failed(rebuildAndMapLoopResults(forOp.getResults(), newForOp.getResults(),
                                      pointerInfos, oldToNewStart, builder,
                                      env))) {
    newForOp.erase();
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scf.while rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteWhileOp(scf::WhileOp whileOp, OpBuilder &builder,
                                    RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(whileOp);
  if (!analysis || !analysis->needsRewrite())
    return failure();
  scf::ConditionOp conditionOp = whileOp.getConditionOp();
  scf::YieldOp yieldOp = whileOp.getYieldOp();
  SmallVector<LoopPointerInfo, 4> pointerInfos;

  // The before arguments, condition forwarded values, after arguments and
  // yield operands all consume the same precomputed positional schema.
  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    unsigned idx = slot.oldIndex;
    if (idx >= whileOp.getBeforeArguments().size() ||
        !env.policy.matches(whileOp.getBeforeArguments()[idx].getType()) ||
        idx >= whileOp.getInits().size() ||
        idx >= conditionOp.getArgs().size() || idx >= yieldOp.getNumOperands())
      return failure();

    FailureOr<DecomposedValue> initInfo =
        env.decomposeValue(whileOp.getInits()[idx], builder, whileOp.getLoc());
    if (failed(initInfo) ||
        failed(env.policy.normalizeLoopCarriedValue(
            *initInfo, slot.componentIndices, slot.resultAttributes, builder,
            whileOp.getLoc())) ||
        failed(castPlannedComponents(*initInfo, slot.componentIndices,
                                     slot.componentTypes, builder,
                                     whileOp.getLoc())))
      return failure();
    pointerInfos.push_back(LoopPointerInfo{idx,
                                           *initInfo,
                                           slot.componentIndices,
                                           slot.componentTypes,
                                           slot.resultAttributes,
                                           {}});
  }

  // Expand inits and result types in lockstep. oldToNewStart keeps untouched
  // positions addressable even when earlier pointer slots expand by rank.
  SmallVector<Value> newInits;
  SmallVector<Type> newResultTypes;
  SmallVector<unsigned> oldToNewStart(whileOp.getInits().size(), 0);
  for (auto [idx, initArg] : llvm::enumerate(whileOp.getInits())) {
    oldToNewStart[idx] = newInits.size();
    if (LoopPointerInfo *info = findPointerInfoByOldIndex(pointerInfos, idx)) {
      FailureOr<SmallVector<Value>> initComponents =
          gatherValues(info->initInfo.components, info->componentIndices);
      if (failed(initComponents))
        return failure();
      for (Value component : *initComponents) {
        info->newIndices.push_back(newInits.size());
        newInits.push_back(component);
        newResultTypes.push_back(component.getType());
      }
      continue;
    }
    newInits.push_back(env.remap(initArg));
    newResultTypes.push_back(whileOp.getResult(idx).getType());
  }

  bool bodyOk = true;
  auto newWhileOp = builder.create<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange newRegionArgs) {
        RewriteEnv beforeEnv = env;
        // Bind the before-region arguments, then rewrite the body and the
        // values forwarded by scf.condition.
        if (failed(bindLoopRegionArguments(
                whileOp.getBeforeArguments(), newRegionArgs, pointerInfos,
                oldToNewStart, bodyBuilder, loc, beforeEnv)))
          bodyOk = false;

        if (failed(rewriteBodyOps(whileOp.getBeforeBody(), bodyBuilder,
                                  beforeEnv)))
          bodyOk = false;

        SmallVector<Value> newConditionArgs;
        if (failed(rewriteLoopTerminatorOperands(
                conditionOp.getArgs(), newRegionArgs, pointerInfos, bodyBuilder,
                conditionOp.getLoc(), beforeEnv, newConditionArgs)))
          bodyOk = false;

        bodyBuilder.create<scf::ConditionOp>(
            conditionOp.getLoc(), beforeEnv.remap(conditionOp.getCondition()),
            newConditionArgs);
      },
      [&](OpBuilder &bodyBuilder, Location loc, ValueRange newRegionArgs) {
        RewriteEnv afterEnv = env;
        // Bind the after-region arguments before rewriting the body and its
        // backedge yield.
        if (failed(bindLoopRegionArguments(
                whileOp.getAfterArguments(), newRegionArgs, pointerInfos,
                oldToNewStart, bodyBuilder, loc, afterEnv)))
          bodyOk = false;

        if (failed(
                rewriteBodyOps(whileOp.getAfterBody(), bodyBuilder, afterEnv)))
          bodyOk = false;

        SmallVector<Value> newYieldOperands;
        if (failed(rewriteLoopTerminatorOperands(
                yieldOp.getOperands(), newRegionArgs, pointerInfos, bodyBuilder,
                yieldOp.getLoc(), afterEnv, newYieldOperands)))
          bodyOk = false;

        bodyBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
      });
  newWhileOp->setAttrs(whileOp->getAttrs());
  if (analysis->rewritesOwnSignature() &&
      failed(updatePointerDescriptorBoundaryMarker(
          newWhileOp, pointerInfos, oldToNewStart, env.policy))) {
    newWhileOp.erase();
    return failure();
  }

  if (!bodyOk) {
    newWhileOp.erase();
    return failure();
  }

  builder.setInsertionPointAfter(newWhileOp);
  if (failed(rebuildAndMapLoopResults(whileOp.getResults(),
                                      newWhileOp.getResults(), pointerInfos,
                                      oldToNewStart, builder, env))) {
    newWhileOp.erase();
    return failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scf.if component planning and rewrite
//===----------------------------------------------------------------------===//

static LogicalResult rewriteIfOp(scf::IfOp ifOp, OpBuilder &builder,
                                 RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(ifOp);
  if (!analysis || !analysis->needsRewrite() ||
      (!ifOp.elseBlock() && analysis->rewritesOwnSignature()))
    return failure();

  bool hasElse = static_cast<bool>(ifOp.elseBlock());
  scf::YieldOp thenYield = ifOp.thenYield();
  scf::YieldOp elseYield = hasElse ? ifOp.elseYield() : scf::YieldOp();
  SmallVector<IfPointerInfo, 4> pointerInfos;

  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    if (slot.oldIndex >= ifOp.getNumResults() ||
        !env.policy.matches(ifOp.getResult(slot.oldIndex).getType()) ||
        slot.componentIndices.size() != slot.componentTypes.size())
      return failure();
    pointerInfos.push_back(IfPointerInfo{slot.oldIndex, slot.componentIndices,
                                         slot.componentTypes,
                                         slot.resultAttributes, std::nullopt});
  }

  // Expand only result positions selected by analysis. An if with no pointer
  // result may still be rebuilt because one of its nested operations changes.
  SmallVector<Type> newResultTypes;
  for (auto [idx, result] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info =
            findPointerInfoByOldIndex(pointerInfos, idx)) {
      newResultTypes.append(info->componentTypes.begin(),
                            info->componentTypes.end());
      continue;
    }
    newResultTypes.push_back(result.getType());
  }

  bool bodyOk = true;
  auto buildBranch = [&](OpBuilder &branchBuilder,
                         bool isThen) -> LogicalResult {
    // Each branch gets an isolated environment because values defined in one
    // branch must never be visible while cloning the other branch.
    RewriteEnv branchEnv = env;
    Block *oldBlock = isThen ? ifOp.thenBlock() : ifOp.elseBlock();
    scf::YieldOp oldYield = isThen ? thenYield : elseYield;
    if (failed(rewriteBodyOps(oldBlock, branchBuilder, branchEnv)))
      return failure();

    SmallVector<Value> newYieldOperands;
    for (auto [idx, oldOperand] : llvm::enumerate(oldYield.getOperands())) {
      if (IfPointerInfo *info = findPointerInfoByOldIndex(pointerInfos, idx)) {
        FailureOr<DecomposedValue> branchInfo = branchEnv.decomposeValue(
            oldOperand, branchBuilder, oldYield.getLoc());
        if (failed(branchInfo) ||
            failed(castPlannedComponents(*branchInfo, info->componentIndices,
                                         info->componentTypes, branchBuilder,
                                         oldYield.getLoc())) ||
            failed(branchEnv.policy.normalizeControlFlowValue(
                *branchInfo, info->resultAttributes, branchBuilder,
                oldYield.getLoc())))
          return failure();
        if (isThen)
          info->thenInfo = *branchInfo;
        FailureOr<SmallVector<Value>> values =
            gatherValues(branchInfo->components, info->componentIndices);
        if (failed(values))
          return failure();
        newYieldOperands.append(values->begin(), values->end());
        continue;
      }
      newYieldOperands.push_back(branchEnv.remap(oldOperand));
    }
    branchBuilder.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
    return success();
  };

  // Create the shell first, then clone each old branch into the corresponding
  // new region with independent mappings.
  auto newIfOp = builder.create<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, env.remap(ifOp.getCondition()), hasElse);
  newIfOp->setAttrs(ifOp->getAttrs());

  // The then region always exists, including for a result-less one-arm if.
  // Rewriting it is still required when it contains affected nested SCF.
  {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.thenBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.thenBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.thenBlock());
    }
    if (failed(buildBranch(builder, /*isThen=*/true)))
      bodyOk = false;
  }
  // An else region exists only for the two-arm form. In particular, do not
  // access elseBlock() merely because the then region contains nested work.
  if (hasElse) {
    OpBuilder::InsertionGuard guard(builder);
    if (newResultTypes.empty()) {
      newIfOp.elseBlock()->getTerminator()->erase();
      builder.setInsertionPointToEnd(newIfOp.elseBlock());
    } else {
      builder.setInsertionPointToStart(newIfOp.elseBlock());
    }
    if (failed(buildBranch(builder, /*isThen=*/false)))
      bodyOk = false;
  }

  if (!bodyOk) {
    newIfOp.erase();
    return failure();
  }

  // Reassemble the pointer immediately after the replacement if. Downstream
  // operations therefore keep their original operand types; decomposition is
  // limited to the control-flow boundary itself.
  builder.setInsertionPointAfter(newIfOp);
  unsigned newResultIndex = 0;
  for (auto [idx, oldResult] : llvm::enumerate(ifOp.getResults())) {
    if (const IfPointerInfo *info =
            findPointerInfoByOldIndex(pointerInfos, idx)) {
      SmallVector<Value> componentValues;
      for (unsigned i = 0; i < info->componentIndices.size(); ++i)
        componentValues.push_back(newIfOp.getResult(newResultIndex++));
      FailureOr<DecomposedValue> resultInfo = withReplacedComponents(
          *info->thenInfo, info->componentIndices, componentValues);
      if (failed(resultInfo) || failed(env.policy.normalizeControlFlowValue(
                                    *resultInfo, info->resultAttributes,
                                    builder, oldResult.getLoc()))) {
        newIfOp.erase();
        return failure();
      }
      Value rebuilt =
          env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
      if (!rebuilt || failed(markPointerDescriptorRebuild(rebuilt, *resultInfo,
                                                          env.policy))) {
        newIfOp.erase();
        return failure();
      }
      env.recordDecomposition(oldResult, *resultInfo, rebuilt);
      continue;
    }
    env.valueMapping.map(oldResult, newIfOp.getResult(newResultIndex++));
  }

  return success();
}

//===----------------------------------------------------------------------===//
// scope.scope rewrite
//===----------------------------------------------------------------------===//

/// Erases the replacement Scope and any pointer reconstruction operations
/// inserted between it and the original Scope. Body-local operations disappear
/// with the replacement region; post-Scope operations are erased in reverse
/// order so their SSA dependencies remain valid during cleanup.
static void eraseScopeReplacement(scope::ScopeOp replacement,
                                  scope::ScopeOp original) {
  SmallVector<Operation *> followingOperations;
  for (Operation *operation = replacement->getNextNode();
       operation && operation != original.getOperation();
       operation = operation->getNextNode())
    followingOperations.push_back(operation);
  for (Operation *operation : llvm::reverse(followingOperations))
    operation->erase();
  replacement.erase();
}

static LogicalResult rewriteScopeOp(scope::ScopeOp scopeOp, OpBuilder &builder,
                                    RewriteEnv &env) {
  const ControlFlowOpAnalysis *analysis = env.plan.lookup(scopeOp);
  FailureOr<scope::ReturnOp> oldReturn = getSupportedScopeReturn(scopeOp);
  if (!analysis || !analysis->needsRewrite() || failed(oldReturn))
    return failure();

  SmallVector<ScopePointerInfo, 0> pointerInfos;
  llvm::SmallDenseSet<unsigned> claimedResultIndices;
  pointerInfos.reserve(analysis->slots.size());
  for (const ControlFlowSlotAnalysis &slot : analysis->slots) {
    if (slot.oldIndex >= scopeOp.getNumResults() ||
        slot.componentIndices.size() != slot.componentTypes.size() ||
        !claimedResultIndices.insert(slot.oldIndex).second ||
        !env.policy.isDecompositionTarget(scopeOp.getResult(slot.oldIndex)))
      return failure();
    pointerInfos.push_back(ScopePointerInfo{
        slot.oldIndex, slot.componentIndices, slot.componentTypes,
        slot.resultAttributes, std::nullopt});
  }

  SmallVector<Type> newResultTypes;
  for (auto [oldIndex, oldResult] : llvm::enumerate(scopeOp.getResults())) {
    if (const ScopePointerInfo *pointerInfo =
            findPointerInfoByOldIndex(pointerInfos, oldIndex)) {
      newResultTypes.append(pointerInfo->componentTypes.begin(),
                            pointerInfo->componentTypes.end());
      continue;
    }
    newResultTypes.push_back(oldResult.getType());
  }

  auto newScope =
      builder.create<scope::ScopeOp>(scopeOp.getLoc(), newResultTypes);
  newScope->setAttrs(scopeOp->getAttrs());
  // PointerDescriptorBoundary describes loop-carried slot positions and has no
  // meaning for an operand-free Scope result boundary.
  newScope->removeAttr(kPointerDescriptorBoundaryAttr);
  newScope.getBodyRegion().emplaceBlock();

  bool bodyOk = true;
  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(&newScope.getBodyRegion().front());
    RewriteEnv scopeEnv = env;
    if (failed(rewriteBodyOps(&scopeOp.getBodyRegion().front(), builder,
                              scopeEnv))) {
      bodyOk = false;
    } else {
      SmallVector<Value> newReturnOperands;
      newReturnOperands.reserve(newResultTypes.size());
      for (auto [oldIndex, oldOperand] :
           llvm::enumerate(oldReturn->getOperands())) {
        ScopePointerInfo *pointerInfo =
            findPointerInfoByOldIndex(pointerInfos, oldIndex);
        if (!pointerInfo) {
          newReturnOperands.push_back(scopeEnv.remap(oldOperand));
          continue;
        }

        FailureOr<DecomposedValue> returnedInfo =
            scopeEnv.decomposeValue(oldOperand, builder, oldReturn->getLoc());
        if (failed(returnedInfo) ||
            failed(castPlannedComponents(
                *returnedInfo, pointerInfo->componentIndices,
                pointerInfo->componentTypes, builder, oldReturn->getLoc())) ||
            failed(scopeEnv.policy.normalizeControlFlowValue(
                *returnedInfo, pointerInfo->resultAttributes, builder,
                oldReturn->getLoc()))) {
          bodyOk = false;
          break;
        }

        FailureOr<SmallVector<Value>> returnedComponents = gatherValues(
            returnedInfo->components, pointerInfo->componentIndices);
        if (failed(returnedComponents)) {
          bodyOk = false;
          break;
        }
        pointerInfo->returnedInfo = *returnedInfo;
        newReturnOperands.append(returnedComponents->begin(),
                                 returnedComponents->end());
      }

      if (bodyOk)
        builder.create<scope::ReturnOp>(oldReturn->getLoc(), newReturnOperands);
    }
  }

  if (!bodyOk) {
    eraseScopeReplacement(newScope, scopeOp);
    return failure();
  }

  // Pointer values are reconstructed immediately outside Scope. Scope itself
  // returns only policy-selected components and deliberately does not receive
  // PointerDescriptorBoundary; the reconstruction operation remains the
  // precise downstream ownership marker.
  builder.setInsertionPointAfter(newScope);
  unsigned newResultIndex = 0;
  for (auto [oldIndex, oldResult] : llvm::enumerate(scopeOp.getResults())) {
    ScopePointerInfo *pointerInfo =
        findPointerInfoByOldIndex(pointerInfos, oldIndex);
    if (!pointerInfo) {
      if (newResultIndex >= newScope.getNumResults()) {
        eraseScopeReplacement(newScope, scopeOp);
        return failure();
      }
      env.valueMapping.map(oldResult, newScope.getResult(newResultIndex++));
      continue;
    }

    if (!pointerInfo->returnedInfo ||
        newResultIndex + pointerInfo->componentIndices.size() >
            newScope.getNumResults()) {
      eraseScopeReplacement(newScope, scopeOp);
      return failure();
    }
    SmallVector<Value> componentValues;
    componentValues.reserve(pointerInfo->componentIndices.size());
    for (unsigned componentIndex = 0;
         componentIndex < pointerInfo->componentIndices.size();
         ++componentIndex)
      componentValues.push_back(newScope.getResult(newResultIndex++));

    FailureOr<DecomposedValue> resultInfo =
        withReplacedComponents(*pointerInfo->returnedInfo,
                               pointerInfo->componentIndices, componentValues);
    if (failed(resultInfo) || failed(env.policy.normalizeControlFlowValue(
                                  *resultInfo, pointerInfo->resultAttributes,
                                  builder, oldResult.getLoc()))) {
      eraseScopeReplacement(newScope, scopeOp);
      return failure();
    }

    Value rebuilt =
        env.policy.recompose(*resultInfo, builder, oldResult.getLoc());
    if (!rebuilt || failed(markPointerDescriptorRebuild(rebuilt, *resultInfo,
                                                        env.policy))) {
      eraseScopeReplacement(newScope, scopeOp);
      return failure();
    }
    env.recordDecomposition(oldResult, *resultInfo, rebuilt);
  }

  if (newResultIndex != newScope.getNumResults()) {
    eraseScopeReplacement(newScope, scopeOp);
    return failure();
  }
  return success();
}

static LogicalResult rewriteControlFlowOp(Operation *op, OpBuilder &builder,
                                          RewriteEnv &env) {
  // Keep the operation dispatch next to the shared recursive implementation:
  // all supported region operations must obey the same mapping and cleanup
  // rules. Pointer-specific semantics enter only through env.policy.
  if (auto forOp = dyn_cast<scf::ForOp>(op))
    return rewriteForOp(forOp, builder, env);
  if (auto whileOp = dyn_cast<scf::WhileOp>(op))
    return rewriteWhileOp(whileOp, builder, env);
  if (auto ifOp = dyn_cast<scf::IfOp>(op))
    return rewriteIfOp(ifOp, builder, env);
  if (auto scopeOp = dyn_cast<scope::ScopeOp>(op))
    return rewriteScopeOp(scopeOp, builder, env);
  return failure();
}

static FailureOr<SmallVector<Value>>
collectReplacements(Operation *op, const RewriteEnv &env) {
  SmallVector<Value> replacements;
  replacements.reserve(op->getNumResults());
  for (Value result : op->getResults()) {
    // Unlike RewriteEnv::remap(), replacement collection must not fall back to
    // the original result. Such a fallback would hide an unhandled result slot
    // and ask replaceOp to replace a value with itself.
    Value replacement = env.valueMapping.lookupOrNull(result);
    if (!replacement)
      return failure();
    replacements.push_back(replacement);
  }
  return replacements;
}

static LogicalResult
tryDecoupleControlFlowOp(Operation *op, IRRewriter &rewriter,
                         const ControlFlowRewritePolicy &policy,
                         const ControlFlowRewritePlan &plan) {
  // Build a replacement beside the original operation. The original operation
  // itself remains until every result has a valid mapped value, after which the
  // standard rewriter performs the externally visible replacement.
  // TODO: Track and erase policy materializations created outside the new SCF
  // operation if an unexpected rewrite-time validation fails. Read-only
  // analysis makes that path exceptional, but failure should still be atomic.
  RewriteEnv env(policy, plan);
  rewriter.setInsertionPoint(op);
  if (failed(rewriteControlFlowOp(op, rewriter, env)))
    return failure();

  FailureOr<SmallVector<Value>> replacements = collectReplacements(op, env);
  if (failed(replacements))
    return failure();
  rewriter.replaceOp(op, *replacements);
  return success();
}

} // namespace

namespace mlir::triton::controlflow {

LogicalResult applyControlFlowRewritePlan(
    ModuleOp module, const ControlFlowRewritePolicy &policy,
    const ControlFlowRewritePlan &plan, bool emitDiagnostics) {
  IRRewriter rewriter(module.getContext());
  // Analysis and application are consecutive and no IR mutation occurs in
  // between, so rediscover the roots from the module instead of duplicating
  // traversal state in the immutable operation plan.
  for (Operation *root : collectOutermostControlFlowOps(module)) {
    const ControlFlowOpAnalysis *rootAnalysis = plan.lookup(root);
    if (!rootAnalysis) {
      if (emitDiagnostics)
        root->emitError("missing frozen control-flow rewrite decision");
      return failure();
    }
    if (!rootAnalysis->needsRewrite())
      continue;
    if (failed(tryDecoupleControlFlowOp(root, rewriter, policy, plan))) {
      if (emitDiagnostics)
        root->emitError("failed to apply analyzed pointer decomposition");
      return failure();
    }
  }
  return success();
}

LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy,
                                 bool allowUnsupportedFallback) {
  // Tensor-pointer decomposition is an optimization, not the only legal
  // representation of a TensorPtr. Probe the complete rewrite on a detached
  // clone before mutating the real module. If analysis or application rejects
  // any root, discard the clone and keep the original TensorPtr graph intact;
  // this prevents a partially materialized descriptor from reaching T2L.
  if (allowUnsupportedFallback) {
    Operation *probeOperation = module->clone();
    auto probeModule = cast<ModuleOp>(probeOperation);
    bool probeSucceeded = false;
    {
      ScopedDiagnosticHandler suppressDiagnostics(
          module.getContext(), [](Diagnostic &) { return success(); });
      FailureOr<ControlFlowRewritePlan> probePlan =
          analyzeControlFlow(probeModule, policy);
      probeSucceeded =
          succeeded(probePlan) &&
          succeeded(applyControlFlowRewritePlan(probeModule, policy, *probePlan,
                                                /*emitDiagnostics=*/false));
    }
    probeModule->erase();
    if (!probeSucceeded)
      return success();
  }

  FailureOr<ControlFlowRewritePlan> plan = analyzeControlFlow(module, policy);
  if (failed(plan))
    return failure();
  return applyControlFlowRewritePlan(module, policy, *plan);
}

} // namespace mlir::triton::controlflow
