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

#include "TritonControlFlowOpt/BlockPtrDecompose.h"

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "Utils/Utils.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::controlflow;

namespace {

static constexpr unsigned kBaseComponent = 0;
static constexpr unsigned getShapeStart() { return kBaseComponent + 1; }
static constexpr unsigned getStrideStart(unsigned rank) {
  return getShapeStart() + rank;
}
static constexpr unsigned getOffsetStart(unsigned rank) {
  return getStrideStart(rank) + rank;
}

/// Block-pointer component layout used only by this policy:
///
///   components = [base_address, shape..., strides..., offsets...]
///   attributes = [order]
///
/// Analysis tracks all components in this exact order, while a rewritten SCF
/// boundary carries only the components whose symbolic identities may change.
/// The base is represented as an i64 address rather than a pointer/memref so a
/// control-flow merge remains an SSA value selection and cannot be lowered as
/// a memory-object copy by the backend.
/// `order` remains policy-owned static metadata and must agree on every
/// incoming path.
///
/// Keep rank/layout checks local to this file: the generic control-flow
/// machinery intentionally does not know the descriptor format of a policy.
// Extracts the block-pointer rank from the original type shared by analysis
// and rewrite states. Invalid pointer and pointee types fail consistently.
static FailureOr<unsigned> getRank(Type originalType) {
  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  if (!pointerType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!tensorType)
    return failure();
  return tensorType.getRank();
}

// Recovers the scalar pointer type accepted by tt.make_tensor_ptr from the
// BlockPtr result type. The address space is preserved across the temporary
// integer carrier.
static FailureOr<triton::PointerType> getBasePointerType(Type originalType) {
  auto pointerType = dyn_cast<triton::PointerType>(originalType);
  if (!pointerType)
    return failure();
  auto tensorType = dyn_cast<RankedTensorType>(pointerType.getPointeeType());
  if (!tensorType)
    return failure();
  return triton::PointerType::get(tensorType.getElementType(),
                                  pointerType.getAddressSpace());
}

// Materializes the integer form of a make_tensor_ptr base at a point dominated
// by the base itself, rather than unconditionally inside the make_tensor_ptr's
// block. This matters when two scf.if branches build equivalent descriptors
// from one outer base: analysis classifies the base as invariant, so the
// concrete address reused after the if must also dominate that use.
//
// Example:
//   %base is a function argument
//   scf.if ... {
//     %a = tt.make_tensor_ptr %base, ...
//   } else {
//     %b = tt.make_tensor_ptr %base, ...
//   }
//
// The shared tt.ptr_to_int is materialized among the leading conversions of a
// block-argument owner, or immediately after the base's defining operation.
// Both positions dominate later branches and the pointer rebuilt after them.
// Keeping the explicit tt.ptr_to_int even for a base produced by tt.int_to_ptr
// preserves the native pointer provenance used during invariant-base
// recomposition.
static FailureOr<Value> materializeBaseAddress(Value base, Location loc,
                                               OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  if (auto blockArgument = dyn_cast<BlockArgument>(base)) {
    Block *owner = blockArgument.getOwner();
    Operation *lastLeadingAddress = nullptr;
    for (Operation &operation : *owner) {
      auto existing = dyn_cast<triton::PtrToIntOp>(&operation);
      if (!existing)
        break;
      if (existing.getSrc() == base &&
          existing.getResult().getType().isInteger(64))
        return existing.getResult();
      lastLeadingAddress = &operation;
    }
    if (lastLeadingAddress)
      builder.setInsertionPointAfter(lastLeadingAddress);
    else
      builder.setInsertionPointToStart(owner);
  } else if (Operation *definingOp = base.getDefiningOp()) {
    if (Operation *next = definingOp->getNextNode()) {
      if (auto existing = dyn_cast<triton::PtrToIntOp>(next);
          existing && existing.getSrc() == base &&
          existing.getResult().getType().isInteger(64))
        return existing.getResult();
    }
    builder.setInsertionPointAfter(definingOp);
  } else {
    return failure();
  }
  return builder.create<triton::PtrToIntOp>(loc, builder.getI64Type(), base)
      .getResult();
}

// Recovers the original pointer when a concrete descriptor still holds the
// exact address produced for that pointer. An invariant base keeps the direct
//
//   %address = tt.ptr_to_int %base
//
// result as component zero while offsets cross an SCF boundary. A changing
// base instead replaces component zero with an SCF result or region argument,
// so this lookup intentionally fails and recomposition must use tt.int_to_ptr.
static Value recoverNativeBase(Value baseAddress,
                               triton::PointerType expectedType) {
  auto ptrToInt = baseAddress.getDefiningOp<triton::PtrToIntOp>();
  if (!ptrToInt || ptrToInt.getSrc().getType() != expectedType)
    return nullptr;
  return ptrToInt.getSrc();
}

// Analysis and rewrite must agree on which advance dimensions change the
// descriptor. Only a literal integer zero is ignored; values that merely fold
// to zero after another rewrite remain conservative transfer candidates.
static bool isConstantZeroInteger(Value value) {
  std::optional<int64_t> constant = getConstantIntValue(value);
  return constant && *constant == 0;
}

// Validates the common block-pointer schema for either state representation.
// Their component element types differ, but this check only needs field sizes.
template <typename StateT> static bool hasValidLayout(const StateT &state) {
  FailureOr<unsigned> rank = getRank(state.originalType);
  return succeeded(rank) && state.components.size() == 1 + 3 * *rank &&
         state.attributes.size() == 1 &&
         isa<DenseI32ArrayAttr>(state.attributes.front());
}

/// Returns the complete, ordered descriptor range used to expand one
/// block-pointer control-flow slot. Keeping this policy-local prevents the
/// generic SCF rewrite from depending on the BlockPtr field layout.
static SmallVector<unsigned>
getAllComponentIndices(const AnalyzedValue &value) {
  SmallVector<unsigned> indices;
  indices.reserve(value.components.size());
  for (unsigned index = 0; index < value.components.size(); ++index)
    indices.push_back(index);
  return indices;
}

class BlockPtrPolicy final : public ControlFlowRewritePolicy {
public:
  bool matches(Type type) const override {
    // A Triton block pointer is a scalar !tt.ptr whose pointee is a ranked
    // tensor. Tensor-of-pointers are handled by their own decomposition.
    auto pointerType = dyn_cast<triton::PointerType>(type);
    return pointerType && isa<RankedTensorType>(pointerType.getPointeeType());
  }

  FailureOr<AnalyzedValue>
  analyzeValue(Value value,
               ControlFlowAnalysisContext &context) const override {
    // Region arguments and control-flow results are installed by the generic
    // analysis after merging their incoming abstract component states.
    if (const AnalyzedValue *known = context.lookupValue(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    if (auto makePtr = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
      // make_tensor_ptr exposes the complete descriptor directly. Record only
      // types and symbolic identities here; this phase must not create IR.
      AnalyzedValue result;
      result.originalType = value.getType();
      Value base = makePtr.getBase();
      result.components.push_back(
          {IntegerType::get(value.getContext(), 64),
           ComponentIdentity::fromValue(base, kBaseComponent)});
      unsigned componentIndex = getShapeStart();
      auto appendComponents = [&](ValueRange values) {
        for (Value component : values) {
          result.components.push_back(
              {component.getType(),
               ComponentIdentity::fromValue(component, componentIndex++)});
        }
      };
      appendComponents(makePtr.getShape());
      appendComponents(makePtr.getStrides());
      appendComponents(makePtr.getOffsets());
      result.attributes.push_back(makePtr.getOrderAttr());
      if (!hasValidLayout(result))
        return failure();
      return result;
    }

    auto advance = value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();
    // tt.advance preserves base/shape/strides/order and produces new offsets.
    // Give those offsets identities tied to the result so a loop/if merge can
    // detect that they differ without materializing arith.addi operations.
    FailureOr<AnalyzedValue> result = context.analyzeValue(advance.getPtr());
    if (failed(result) || !hasValidLayout(*result))
      return failure();
    unsigned rank = *getRank(result->originalType);
    if (advance.getOffsets().size() != rank)
      return failure();
    for (unsigned dimension = 0; dimension < rank; ++dimension) {
      if (isConstantZeroInteger(advance.getOffsets()[dimension]))
        continue;
      unsigned componentIndex = getOffsetStart(rank) + dimension;
      result->components[componentIndex].identity =
          ComponentIdentity::fromValue(value, componentIndex);
    }
    result->originalType = value.getType();
    return *result;
  }

  FailureOr<SmallVector<unsigned>>
  getLoopCandidateComponents(const AnalyzedValue &value) const override {
    if (!hasValidLayout(value))
      return failure();
    // Analyze the complete descriptor so changes to base, shape, stride, or
    // offset remain visible. The transfer hook below decides which candidates
    // actually become ordinary loop-carried SSA state.
    return getAllComponentIndices(value);
  }

  FailureOr<SmallVector<unsigned>>
  getLoopTransferredComponents(const AnalyzedValue &initial,
                               const AnalyzedValue &regionArgument,
                               const AnalyzedValue &next) const override {
    if (!hasValidLayout(initial) || !hasValidLayout(regionArgument) ||
        !hasValidLayout(next) ||
        initial.originalType != regionArgument.originalType ||
        initial.originalType != next.originalType ||
        initial.attributes != regionArgument.attributes ||
        initial.attributes != next.attributes)
      return failure();

    SmallVector<unsigned> transferred;
    for (unsigned index = 0; index < initial.components.size(); ++index) {
      if (failed(joinComponentTypes(initial.components[index].type,
                                    regionArgument.components[index].type)) ||
          failed(joinComponentTypes(initial.components[index].type,
                                    next.components[index].type)))
        return failure();
      const ComponentIdentity &nextIdentity = next.components[index].identity;
      bool forwardsCurrent =
          nextIdentity == regionArgument.components[index].identity;
      bool restoresInitial = nextIdentity == initial.components[index].identity;
      if (!forwardsCurrent && !restoresInitial)
        transferred.push_back(index);
    }
    return transferred;
  }

  FailureOr<SmallVector<unsigned>>
  getIfTransferredComponents(const AnalyzedValue &thenValue,
                             const AnalyzedValue &elseValue) const override {
    if (!hasValidLayout(thenValue) || !hasValidLayout(elseValue) ||
        thenValue.originalType != elseValue.originalType ||
        thenValue.attributes != elseValue.attributes)
      return failure();

    SmallVector<unsigned> transferred;
    for (unsigned index = 0; index < thenValue.components.size(); ++index) {
      if (failed(joinComponentTypes(thenValue.components[index].type,
                                    elseValue.components[index].type)))
        return failure();
      if (thenValue.components[index].identity !=
          elseValue.components[index].identity)
        transferred.push_back(index);
    }
    return transferred;
  }

  FailureOr<Type> joinComponentTypes(Type lhs, Type rhs) const override {
    // Block-pointer descriptor operands must have identical types on all
    // incoming paths. Tensor-pointer offsets use a more permissive integer
    // width join in their own policy.
    if (lhs != rhs)
      return failure();
    return lhs;
  }

  bool shouldDecomposeOperation(Operation *op) const override {
    // Recording each cloned advance lets downstream advances reuse its
    // flattened descriptor rather than walking through the rebuilt pointer.
    return isa<triton::AdvanceOp>(op);
  }

  bool requiresPointerDescriptorBoundaryMarker() const override { return true; }

  FailureOr<DecomposedValue> decompose(Value value,
                                       const ControlFlowRewriteContext &context,
                                       OpBuilder &builder,
                                       Location loc) const override {
    // Prefer decompositions recorded while rebuilding the enclosing region;
    // this is how analysis results cross nested SCF boundaries at rewrite time.
    if (const DecomposedValue *known = context.lookup(value)) {
      if (!matches(known->originalType))
        return failure();
      return *known;
    }

    value = context.remap(value);
    if (auto makePtr = value.getDefiningOp<triton::MakeTensorPtrOp>()) {
      // Materialize the concrete counterpart of analyzeValue's descriptor.
      DecomposedValue result;
      result.originalType = value.getType();
      FailureOr<Value> baseAddress =
          materializeBaseAddress(makePtr.getBase(), makePtr.getLoc(), builder);
      if (failed(baseAddress))
        return failure();
      result.components.push_back(*baseAddress);
      result.components.append(makePtr.getShape().begin(),
                               makePtr.getShape().end());
      result.components.append(makePtr.getStrides().begin(),
                               makePtr.getStrides().end());
      result.components.append(makePtr.getOffsets().begin(),
                               makePtr.getOffsets().end());
      result.attributes.push_back(makePtr.getOrderAttr());
      if (!hasValidLayout(result))
        return failure();
      return result;
    }

    auto advance = value.getDefiningOp<triton::AdvanceOp>();
    if (!advance)
      return failure();

    FailureOr<DecomposedValue> result =
        decompose(advance.getPtr(), context, builder, loc);
    if (failed(result) || !hasValidLayout(*result))
      return failure();
    FailureOr<unsigned> rank = getRank(result->originalType);
    if (advance.getOffsets().size() != *rank)
      return failure();

    // Flatten an advance into offset arithmetic at the original operation's
    // position. Base, shape, strides, and order are inherited unchanged.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(advance);
    for (auto [dim, delta] : llvm::enumerate(advance.getOffsets())) {
      unsigned component = getOffsetStart(*rank) + dim;
      Value currentOffset = result->components[component];
      if (isConstantZeroInteger(delta))
        continue;
      Value remappedDelta = context.remap(delta);
      if (!remappedDelta)
        return failure();
      Value offset = dyn_cast<Value>(
          addOpFoldResult(currentOffset, remappedDelta, advance.getLoc(),
                          builder, currentOffset.getType()));
      if (!offset)
        return failure();
      result->components[component] = offset;
    }
    result->originalType = value.getType();
    return *result;
  }

  Value recompose(const DecomposedValue &value, OpBuilder &builder,
                  Location loc) const override {
    // Rebuild the original pointer type immediately inside/after the rewritten
    // control-flow boundary so ordinary users remain untouched.
    if (!hasValidLayout(value))
      return nullptr;
    unsigned rank = *getRank(value.originalType);
    auto order = cast<DenseI32ArrayAttr>(value.attributes.front());
    FailureOr<triton::PointerType> basePointerType =
        getBasePointerType(value.originalType);
    if (failed(basePointerType) ||
        value.components[kBaseComponent].getType() != builder.getI64Type())
      return nullptr;
    Value base =
        recoverNativeBase(value.components[kBaseComponent], *basePointerType);
    if (!base)
      base = builder.create<triton::IntToPtrOp>(
          loc, *basePointerType, value.components[kBaseComponent]);
    return builder.create<triton::MakeTensorPtrOp>(
        loc, value.originalType, base,
        ValueRange(value.components).slice(getShapeStart(), rank),
        ValueRange(value.components).slice(getStrideStart(rank), rank),
        ValueRange(value.components).slice(getOffsetStart(rank), rank), order);
  }
};

} // namespace

namespace mlir::triton::controlflow {

LogicalResult runBlockPtrDecompose(ModuleOp module) {
  // Analyze the complete ordered descriptor for every BlockPtr that crosses a
  // supported SCF boundary, carry only changing components, and rebuild the
  // pointer at each use site.
  BlockPtrPolicy policy;
  return rewriteControlFlow(module, policy);
}

} // namespace mlir::triton::controlflow
