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

#ifndef TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
#define TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H

#include "TritonControlFlowOpt/ControlFlowAnalysis.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::triton::controlflow {

/// Handoff marker for SCF loops whose pointer slots have already been expanded
/// into policy-owned descriptor components. TritonToLinalg consumes and
/// removes it instead of applying its legacy pointer-loop decomposition a
/// second time.
inline constexpr llvm::StringLiteral kPointerDescriptorBoundaryAttr =
    "PointerDescriptorBoundary";

/// Handoff marker for a pointer rebuilt at a CFO control-flow boundary. The
/// defining operation's operands are the complete descriptor roots, including
/// invariant components omitted from the replacement SCF signature.
inline constexpr llvm::StringLiteral kPointerDescriptorRebuildAttr =
    "PointerDescriptorRebuild";

/// Optional companion on a PointerDescriptorRebuild root. The `strided_1d`
/// form means the tensor-pointer offset operand is a proven rank-1 affine
/// expression rather than an opaque complete-offset carrier.
inline constexpr llvm::StringLiteral kPointerDescriptorOffsetFormAttr =
    "PointerDescriptorOffsetForm";
inline constexpr llvm::StringLiteral kStrided1DOffsetForm = "strided_1d";

/// Dense per-axis classification attached to a descriptor rebuild. A value
/// of one means the complete contribution of that logical axis is represented
/// by its scalar stride; zero means the axis is carried by the opaque complete
/// contribution tensor.
inline constexpr llvm::StringLiteral kPointerDescriptorStructuredAxesAttr =
    "PointerDescriptorStructuredAxes";

/// Policy-owned description of one value crossing a control-flow boundary.
///
/// `components` contain every runtime value needed to rebuild the original
/// value. A policy may place a selected subset in an expanded SCF signature.
/// `attributes` retain non-SSA metadata. Both layouts are private to the
/// policy; the shared rewrite never interprets pointer-specific fields.
struct DecomposedValue {
  Type originalType;
  SmallVector<Value> components;
  SmallVector<Attribute> attributes;
};

/// Read-only view of the SSA mapping and decompositions visible at the current
/// rewrite point. Policies use it while recursively analyzing pointer
/// producers; the state exists only for one control-flow rewrite attempt.
class ControlFlowRewriteContext {
public:
  ControlFlowRewriteContext(
      const IRMapping &valueMapping,
      const llvm::DenseMap<Value, DecomposedValue> &decomposedValues)
      : valueMapping(valueMapping), decomposedValues(decomposedValues) {}

  Value remap(Value value) const;
  const DecomposedValue *lookup(Value value) const;

private:
  const IRMapping &valueMapping;
  const llvm::DenseMap<Value, DecomposedValue> &decomposedValues;
};

/// Pointer-semantics interface implemented by each decomposition policy.
///
/// The policy decides how its value is decomposed and rebuilt, which components
/// cross loop/if boundaries, and whether two decompositions share a compatible
/// non-carried schema. It carries no mutable state between policy invocations.
/// A policy that expands pointer descriptors may request a downstream handoff
/// marker; the shared rewrite owns the positional slot metadata.
class ControlFlowRewritePolicy : public ControlFlowAnalysisPolicy {
public:
  virtual ~ControlFlowRewritePolicy() = default;

  /// Whether results of this ordinary operation need immediate decomposition
  /// after cloning so later operations can reuse their exact component state.
  virtual bool shouldDecomposeOperation(Operation *op) const = 0;

  /// Whether a rebuilt value may be used as a direct decomposition-cache key.
  ///
  /// The default preserves the existing behavior for policies whose rebuilt
  /// values are already consumable by downstream conversion. A policy that
  /// rebuilds an opaque tensor-of-pointers can reject the cache entry so the
  /// next consumer re-analyzes the pointer producer and follows its normal
  /// scalar-address lowering path instead of exposing tensor.extract of a
  /// scalar Triton pointer to T2L.
  virtual bool shouldCacheRebuiltDecomposition(const DecomposedValue &) const {
    return true;
  }

  /// Whether rewritten loops owned by this policy must expose their descriptor
  /// slots to downstream conversion. The shared rewrite records the exact
  /// expanded result indices because only it knows both old and new signatures.
  virtual bool requiresPointerDescriptorBoundaryMarker() const { return false; }

  /// Whether recomposing a result of this ordinary cloned operation creates a
  /// descriptor root that downstream conversion must recognize. This is more
  /// selective than the boundary marker request above: most policies need
  /// handoff markers only for SCF argument/result reconstructions, while a
  /// policy may opt in for a specific pointer-preserving operation.
  virtual bool shouldMarkOperationRecomposition(Operation *) const {
    return false;
  }

  /// Adds policy-private metadata to a descriptor rebuild root.
  /// This is called only after the generic rewrite has attached the ownership
  /// marker, whether the root was created at an SCF boundary or by an ordinary
  /// operation recomposition explicitly selected by the policy.
  virtual LogicalResult
  annotatePointerDescriptorRebuild(Operation *, const DecomposedValue &) const {
    return success();
  }

  /// Normalizes a concrete descriptor to the schema selected by control-flow
  /// analysis. The default only replaces policy metadata; tensor-pointer
  /// policies override it to fold structured fields into an opaque carrier
  /// when an incoming edge was downgraded.
  virtual LogicalResult
  normalizeControlFlowValue(DecomposedValue &value,
                            ArrayRef<Attribute> targetAttributes, OpBuilder &,
                            Location) const {
    value.attributes.assign(targetAttributes.begin(), targetAttributes.end());
    return success();
  }

  /// Normalizes a descriptor used at a loop boundary. The default preserves
  /// the generic control-flow behavior. Policies may use the exact set of
  /// loop-carried components to retain a more compact equivalent schema that
  /// would not be valid for an if or scope result.
  virtual LogicalResult
  normalizeLoopCarriedValue(DecomposedValue &value,
                            ArrayRef<unsigned> carriedComponents,
                            ArrayRef<Attribute> targetAttributes,
                            OpBuilder &builder, Location loc) const {
    return normalizeControlFlowValue(value, targetAttributes, builder, loc);
  }

  virtual FailureOr<DecomposedValue>
  decompose(Value value, const ControlFlowRewriteContext &context,
            OpBuilder &builder, Location loc) const = 0;

  virtual Value recompose(const DecomposedValue &value, OpBuilder &builder,
                          Location loc) const = 0;
};

/// Rewrites supported SCF operations from outermost to innermost.
///
/// Applies a previously frozen plan without running value analysis again.
/// Signature expansion, region cloning, terminator rewriting, nested recursion
/// and result replacement are driven solely by operation-level decisions in
/// `plan`.
/// Set `emitDiagnostics` to false for a speculative rewrite on a detached
/// clone whose failure is intentionally treated as an unsupported case.
LogicalResult applyControlFlowRewritePlan(
    ModuleOp module, const ControlFlowRewritePolicy &policy,
    const ControlFlowRewritePlan &plan, bool emitDiagnostics = true);

/// Analyzes the complete decomposition stage before mutating the IR, then
/// applies the frozen plan from outermost to innermost. Pointer semantics
/// remain selected by `policy` so different decompositions share the same SCF
/// plumbing.
/// When `allowUnsupportedFallback` is true, a failed complete rewrite is
/// treated as an unsupported optimization and leaves the input module intact.
LogicalResult rewriteControlFlow(ModuleOp module,
                                 const ControlFlowRewritePolicy &policy,
                                 bool allowUnsupportedFallback = false);

} // namespace mlir::triton::controlflow

#endif // TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_REWRITE_H
