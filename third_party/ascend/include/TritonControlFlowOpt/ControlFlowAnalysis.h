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

#ifndef TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_ANALYSIS_H
#define TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_ANALYSIS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::triton::controlflow {

/// Classification used by the signature planner. Recomputed components are
/// intentionally not modeled yet; they can be added when IV extraction becomes
/// part of the analysis rather than the existing block-pointer rewrite helper.
enum class ComponentTransferKind { Invariant, Transferred };

/// Symbolic identity of a component during read-only analysis.
///
/// A component produced by an ordinary SSA value is identified by that value
/// and its policy-defined component index. Zero represents a canonical zero
/// offsets value without materializing an arith.constant in the source IR.
struct ComponentIdentity {
  enum class Kind { Value, Zero } kind = Kind::Value;
  Value value;
  unsigned componentIndex = 0;

  static ComponentIdentity fromValue(Value value, unsigned componentIndex) {
    return {Kind::Value, value, componentIndex};
  }
  static ComponentIdentity zero(unsigned componentIndex = 0) {
    return {Kind::Zero, {}, componentIndex};
  }

  bool operator==(const ComponentIdentity &other) const {
    return kind == other.kind && value == other.value &&
           componentIndex == other.componentIndex;
  }
  bool operator!=(const ComponentIdentity &other) const {
    return !(*this == other);
  }
};

struct AnalyzedComponent {
  Type type;
  ComponentIdentity identity;
};

/// Policy-owned abstract pointer state. Unlike DecomposedValue, this structure
/// never contains newly created IR values and is therefore safe to compute
/// before a rewrite starts.
struct AnalyzedValue {
  Type originalType;
  SmallVector<AnalyzedComponent> components;
  SmallVector<Attribute> attributes;
};

/// Signature decision for one original result/iter-argument position. The slot
/// itself means the original value is removed from the SCF signature;
/// componentIndices may be empty when every component is invariant and no
/// replacement operand/result is required.
struct ControlFlowSlotAnalysis {
  unsigned oldIndex = 0;
  SmallVector<ComponentTransferKind> componentKinds;
  SmallVector<unsigned> componentIndices;
  SmallVector<Type> componentTypes;
  /// Policy metadata after all incoming paths have been joined.
  SmallVector<Attribute> resultAttributes;
};

/// Cached decision for one structured control-flow operation.
struct ControlFlowOpAnalysis {
  SmallVector<ControlFlowSlotAnalysis, 0> slots;
  bool hasNestedRewrite = false;

  bool rewritesOwnSignature() const { return !slots.empty(); }
  bool needsRewrite() const {
    return rewritesOwnSignature() || hasNestedRewrite;
  }
};

/// Immutable operation-level contract consumed after read-only analysis.
///
/// Temporary Value -> AnalyzedValue state is deliberately excluded: those
/// values may be erased while earlier roots are rewritten, whereas a root and
/// its nested operations remain valid until that root is processed.
struct ControlFlowRewritePlan {
  llvm::DenseMap<Operation *, ControlFlowOpAnalysis> operations;

  const ControlFlowOpAnalysis *lookup(Operation *op) const;
};

class ControlFlowAnalysisContext;

/// Pointer-specific part of the read-only analysis. Implementations describe
/// their component layout and merge rules; the common analyzer owns SCF region
/// traversal, argument/result correspondence, and nested-op caching.
class ControlFlowAnalysisPolicy {
public:
  virtual ~ControlFlowAnalysisPolicy() = default;

  virtual bool matches(Type type) const = 0;

  /// Returns whether a value belongs to this decomposition stage. Pointer
  /// policies use the type-based default. A future StructuredOffsets policy
  /// can override this with the result of a backward address-demand analysis.
  virtual bool isDecompositionTarget(Value value) const {
    return matches(value.getType());
  }

  virtual FailureOr<AnalyzedValue>
  analyzeValue(Value value, ControlFlowAnalysisContext &context) const = 0;

  /// Components which may legally be carried by a loop. Other components must
  /// remain symbolically identical across the backedge.
  virtual FailureOr<SmallVector<unsigned>>
  getLoopCandidateComponents(const AnalyzedValue &value) const = 0;

  /// Components which may cross a scope.scope result boundary. Scope has no
  /// region arguments or backedge, so the first implementation returns every
  /// component that the policy already permits a loop to carry. A policy may
  /// override this hook when its Scope schema differs from its loop schema.
  virtual FailureOr<SmallVector<unsigned>>
  getScopeCandidateComponents(const AnalyzedValue &value) const {
    return getLoopCandidateComponents(value);
  }

  virtual FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const = 0;

  virtual FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const = 0;

  /// Selects the component type used in the replacement SCF signature.
  virtual FailureOr<Type> joinComponentTypes(Type lhs, Type rhs) const = 0;

  /// Joins policy-owned metadata across one control-flow boundary. The
  /// default requires exact equality, which preserves BlockPtr behavior.
  virtual FailureOr<SmallVector<Attribute>>
  mergeControlFlowAttributes(const AnalyzedValue &lhs,
                             const AnalyzedValue &rhs) const {
    if (lhs.attributes != rhs.attributes)
      return failure();
    return lhs.attributes;
  }
};

/// Stage-scoped transient cache used while computing a rewrite plan. One
/// context analyzes all outermost roots before any IR mutation; only the
/// operation-level decisions survive in ControlFlowRewritePlan.
class ControlFlowAnalysisContext {
public:
  explicit ControlFlowAnalysisContext(const ControlFlowAnalysisPolicy &policy)
      : policy(policy) {}

  FailureOr<AnalyzedValue> analyzeValue(Value value);
  FailureOr<ControlFlowOpAnalysis> analyzeControlFlowOp(Operation *op);

  ControlFlowRewritePlan takeRewritePlan() &&;

  const AnalyzedValue *lookupValue(Value value) const;
  const ControlFlowOpAnalysis *lookup(Operation *op) const;

private:
  LogicalResult analyzeNestedOperations(Block *block, bool &hasNestedRewrite);
  FailureOr<ControlFlowOpAnalysis> analyzeFor(Operation *op);
  FailureOr<ControlFlowOpAnalysis> analyzeWhile(Operation *op);
  FailureOr<ControlFlowOpAnalysis> analyzeIf(Operation *op);
  FailureOr<ControlFlowOpAnalysis> analyzeScope(Operation *op);

  void bindRegionArgument(Value argument, const AnalyzedValue &initial,
                          ArrayRef<unsigned> componentIndices);
  FailureOr<SmallVector<Type>>
  getTransferredTypes(const AnalyzedValue &lhs, const AnalyzedValue &rhs,
                      ArrayRef<unsigned> componentIndices) const;

  const ControlFlowAnalysisPolicy &policy;
  llvm::DenseMap<Value, AnalyzedValue> analyzedValues;
  llvm::DenseMap<Operation *, ControlFlowOpAnalysis> analyzedOps;
  llvm::DenseSet<Operation *> operationsBeingAnalyzed;
};

/// Returns SCF roots that are not nested in another supported SCF operation.
SmallVector<Operation *> collectOutermostControlFlowOps(ModuleOp module);

/// Analyzes every control-flow root for one decomposition stage before any IR
/// mutation and freezes only the operation-level rewrite decisions.
FailureOr<ControlFlowRewritePlan>
analyzeControlFlow(ModuleOp module, const ControlFlowAnalysisPolicy &policy);

} // namespace mlir::triton::controlflow

#endif // TRITON_ASCEND_TRITON_CONTROL_FLOW_OPT_CONTROL_FLOW_ANALYSIS_H
