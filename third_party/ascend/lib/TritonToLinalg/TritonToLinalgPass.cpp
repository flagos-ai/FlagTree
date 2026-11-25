/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

#include <cstdlib>

#include "TritonControlFlowOpt/ControlFlowRewrite.h"
#include "TritonToGraph/LayoutMemoryOptimization.h"
#include "TritonToLinalg/BlockPtrAnalysis.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToLinalg/ArgMinMaxConverter.h"
#include "ascend/include/TritonToLinalg/CanonicalizeDebugLocationsPass.h"
#include "ascend/include/TritonToLinalg/DeduplicateDebugNopsPass.h"
#include "ascend/include/TritonToLinalg/DescriptorConverter.h"
#include "ascend/include/TritonToLinalg/DevicePrintOffsetRewrite.h"
#include "ascend/include/TritonToLinalg/FunctionConverter.h"
#include "ascend/include/TritonToLinalg/HoistBroadcast.h"
#include "ascend/include/TritonToLinalg/ImplicitPermute.h"
#include "ascend/include/TritonToLinalg/LoadStoreConverter.h"
#include "ascend/include/TritonToLinalg/MarkTensorKindPass.h"
#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/TritonToLinalg/UseAnalysis.h"
#include "ascend/include/TritonToStructured/CannonicalizerConverter.h"
#include "ascend/include/TritonToUnstructure/UnstructureConversionPass.h"
#include "ascend/include/Utils/InterleaveOptimization.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

int nd2nzFlag = 0;
bool compileOn91095Flag = false;
bool existDotFlag = false;
triton::ascend::CompileMode compileModeFlag = triton::ascend::CompileMode::Simd;

static bool containsTritonPointer(Type type) {
  if (isa<triton::PointerType>(type))
    return true;
  auto shapedType = dyn_cast<ShapedType>(type);
  return shapedType && isa<triton::PointerType>(shapedType.getElementType());
}

static bool hasPointerFreeTypes(TypeRange types) {
  return llvm::none_of(types, containsTritonPointer);
}

template <typename RangeT> static bool hasPointerFreeValues(RangeT values) {
  return llvm::none_of(values, [](auto value) {
    return containsTritonPointer(value.getType());
  });
}

// Checks every structural edge of a marker loop. Looking only at the loop
// operation's operands/results misses region arguments and terminator operands,
// which can otherwise leave a half-converted pointer boundary.
static bool hasPointerFreeControlFlowBoundary(LoopLikeOpInterface loopOp) {
  Operation *loop = loopOp.getOperation();
  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    return hasPointerFreeValues(forOp.getInitArgs()) &&
           hasPointerFreeValues(forOp.getRegionIterArgs()) &&
           hasPointerFreeValues(forOp.getYieldedValues()) &&
           hasPointerFreeTypes(forOp.getResultTypes());
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop)) {
    return hasPointerFreeValues(whileOp.getInits()) &&
           hasPointerFreeValues(whileOp.getBeforeArguments()) &&
           hasPointerFreeValues(whileOp.getConditionOp().getArgs()) &&
           hasPointerFreeValues(whileOp.getAfterArguments()) &&
           hasPointerFreeTypes(whileOp.getYieldOp().getOperandTypes()) &&
           hasPointerFreeTypes(whileOp.getResultTypes());
  }
  return false;
}

// Adds only the values that can produce one SSA result. Structured control
// flow is followed by result index so preserving one descriptor slot does not
// retain unrelated loop accumulators or sibling results.
static LogicalResult
appendProducerOperands(Value value, SmallVectorImpl<Value> &producerWorklist) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    unsigned argumentIndex = argument.getArgNumber();
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
      if (argumentIndex == 0)
        return success();
      unsigned iterIndex = argumentIndex - 1;
      if (iterIndex >= forOp.getInitArgs().size())
        return failure();
      producerWorklist.push_back(forOp.getInitArgs()[iterIndex]);
      producerWorklist.push_back(forOp.getYieldedValues()[iterIndex]);
      return success();
    }
    if (auto whileOp = dyn_cast_or_null<scf::WhileOp>(parent)) {
      if (argumentIndex >= whileOp.getInits().size())
        return failure();
      if (argument.getOwner() == whileOp.getBeforeBody()) {
        producerWorklist.push_back(whileOp.getInits()[argumentIndex]);
        producerWorklist.push_back(
            whileOp.getYieldOp().getOperand(argumentIndex));
        return success();
      }
      if (argument.getOwner() == whileOp.getAfterBody()) {
        producerWorklist.push_back(
            whileOp.getConditionOp().getArgs()[argumentIndex]);
        return success();
      }
    }
    return success();
  }

  Operation *producer = value.getDefiningOp();
  if (!producer)
    return success();

  auto result = dyn_cast<OpResult>(value);
  if (auto forOp = dyn_cast<scf::ForOp>(producer)) {
    if (!result || result.getResultNumber() >= forOp.getNumResults())
      return failure();
    unsigned index = result.getResultNumber();
    producerWorklist.push_back(forOp.getInitArgs()[index]);
    producerWorklist.push_back(forOp.getYieldedValues()[index]);
    return success();
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(producer)) {
    if (!result || result.getResultNumber() >= whileOp.getNumResults())
      return failure();
    unsigned index = result.getResultNumber();
    producerWorklist.push_back(whileOp.getInits()[index]);
    producerWorklist.push_back(whileOp.getConditionOp().getArgs()[index]);
    producerWorklist.push_back(whileOp.getYieldOp().getOperand(index));
    return success();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(producer)) {
    if (!result || result.getResultNumber() >= ifOp.getNumResults() ||
        !ifOp.elseBlock())
      return failure();
    unsigned index = result.getResultNumber();
    producerWorklist.push_back(ifOp.thenYield().getOperand(index));
    producerWorklist.push_back(ifOp.elseYield().getOperand(index));
    return success();
  }

  producerWorklist.append(producer->operand_begin(), producer->operand_end());
  return success();
}

// Visits every structural value represented by one dynamic descriptor slot.
// The result type is the slot contract; init values, region arguments and
// terminator operands must all agree with it exactly. Keeping this traversal
// shared prevents entry validation and later MetaUse preservation from
// interpreting PointerDescriptorBoundary differently.
template <typename VisitorT>
static LogicalResult
visitPointerDescriptorBoundarySlotValues(Operation *loop, int32_t slot,
                                         VisitorT &&visit) {
  if (!isa<scf::ForOp, scf::WhileOp>(loop) || slot < 0 ||
      static_cast<unsigned>(slot) >= loop->getNumResults())
    return failure();

  Type expectedType = loop->getResult(slot).getType();
  if (containsTritonPointer(expectedType))
    return failure();

  auto visitSlot = [&](auto values) {
    if (static_cast<unsigned>(slot) >= values.size() ||
        values[slot].getType() != expectedType)
      return failure();
    visit(values[slot]);
    return success();
  };

  visit(loop->getResult(slot));
  if (auto forOp = dyn_cast<scf::ForOp>(loop)) {
    return success(succeeded(visitSlot(forOp.getInitArgs())) &&
                   succeeded(visitSlot(forOp.getRegionIterArgs())) &&
                   succeeded(visitSlot(forOp.getYieldedValues())));
  }

  auto whileOp = cast<scf::WhileOp>(loop);
  return success(succeeded(visitSlot(whileOp.getInits())) &&
                 succeeded(visitSlot(whileOp.getBeforeArguments())) &&
                 succeeded(visitSlot(whileOp.getConditionOp().getArgs())) &&
                 succeeded(visitSlot(whileOp.getAfterArguments())) &&
                 succeeded(visitSlot(whileOp.getYieldOp().getOperands())));
}

// Validates one CFO-owned loop boundary without changing the IR. An empty
// DenseI32ArrayAttr is valid and means every descriptor component is invariant
// and therefore absent from the loop signature.
static LogicalResult validatePointerDescriptorBoundary(Operation *loop) {
  if (!isa<scf::ForOp, scf::WhileOp>(loop))
    return failure();

  Attribute marker = loop->getAttr(controlflow::kPointerDescriptorBoundaryAttr);
  if (!marker)
    return failure();
  auto descriptorSlots = dyn_cast<DenseI32ArrayAttr>(marker);
  if (!descriptorSlots ||
      !hasPointerFreeControlFlowBoundary(cast<LoopLikeOpInterface>(loop)))
    return failure();

  llvm::SmallDenseSet<int32_t> seenSlots;
  for (int32_t slot : descriptorSlots.asArrayRef()) {
    if (!seenSlots.insert(slot).second ||
        failed(
            visitPointerDescriptorBoundarySlotValues(loop, slot, [](Value) {})))
      return failure();
  }
  return success();
}

// Validates a complete descriptor reconstruction root without changing the
// IR. Rebuild roots need operands even when the corresponding loop marker is
// empty because those operands preserve the invariant descriptor components.
static LogicalResult validatePointerDescriptorRebuild(Operation *op) {
  Attribute marker = op->getAttr(controlflow::kPointerDescriptorRebuildAttr);
  if (!marker || !isa<UnitAttr>(marker) || op->getNumOperands() == 0 ||
      !llvm::any_of(op->getResultTypes(), containsTritonPointer))
    return failure();

  Attribute offsetForm =
      op->getAttr(controlflow::kPointerDescriptorOffsetFormAttr);
  if (!offsetForm)
    return success();
  auto form = dyn_cast<StringAttr>(offsetForm);
  auto addPtr = dyn_cast<triton::AddPtrOp>(op);
  if (!form || form.getValue() != controlflow::kStrided1DOffsetForm || !addPtr)
    return failure();

  auto resultType = dyn_cast<RankedTensorType>(addPtr.getType());
  auto offsetType = dyn_cast<RankedTensorType>(addPtr.getOffset().getType());
  auto offsetElementType =
      offsetType ? dyn_cast<IntegerType>(offsetType.getElementType())
                 : IntegerType();
  auto baseSplat = addPtr.getPtr().getDefiningOp<triton::SplatOp>();
  return success(resultType && resultType.getRank() == 1 && offsetType &&
                 isa<triton::PointerType>(resultType.getElementType()) &&
                 offsetElementType && offsetElementType.getWidth() <= 64 &&
                 resultType.getShape() == offsetType.getShape() &&
                 resultType.getEncoding() == offsetType.getEncoding() &&
                 baseSplat &&
                 isa<triton::PointerType>(baseSplat.getSrc().getType()));
}

// Returns true when a pointer consumer is reached through ordinary addptr
// operations from a CFO descriptor reconstruction root. The walk is purposely
// limited to addptr: following arbitrary pointer-producing operations would
// make the handoff contract retain computations it does not own.
static bool hasPointerDescriptorRebuildProvenance(Value pointer) {
  llvm::DenseSet<Value> visited;
  while (pointer && visited.insert(pointer).second) {
    Operation *producer = pointer.getDefiningOp();
    if (!producer)
      return false;
    if (producer->hasAttr(controlflow::kPointerDescriptorRebuildAttr))
      return true;
    auto addPtr = dyn_cast<triton::AddPtrOp>(producer);
    if (!addPtr)
      return false;
    pointer = addPtr.getPtr();
  }
  return false;
}

// Checks the complete CFO-to-TritonToLinalg handoff before any rewrite can
// fold away a malformed marker. This function is deliberately read-only so it
// can run at pass entry, before UseAnalysis has produced MetaUse attributes.
static LogicalResult
validatePointerDescriptorHandoffMetadata(ModuleOp moduleOp) {
  bool valid = true;
  moduleOp.walk([&](Operation *op) {
    if (op->hasAttr(controlflow::kPointerDescriptorBoundaryAttr) &&
        failed(validatePointerDescriptorBoundary(op)))
      valid = false;
    if (op->hasAttr(controlflow::kPointerDescriptorRebuildAttr) &&
        failed(validatePointerDescriptorRebuild(op)))
      valid = false;
    if (op->hasAttr(controlflow::kPointerDescriptorOffsetFormAttr) &&
        !op->hasAttr(controlflow::kPointerDescriptorRebuildAttr))
      valid = false;
  });
  return success(valid);
}

// PointerDescriptorBoundary identifies the dynamic loop slots used to rebuild
// pointers. PointerDescriptorRebuild operands are the complete descriptor
// roots, including invariant components omitted from an empty or minimal slot
// list. After UseAnalysis, preserve exactly those producer chains so
// MetaUseEraser cannot discard values required by conversion.
static LogicalResult preservePointerDescriptorComputations(ModuleOp moduleOp) {
  if (failed(validatePointerDescriptorHandoffMetadata(moduleOp)))
    return failure();

  bool valid = true;
  SmallVector<Value> producerWorklist;
  moduleOp.walk([&](Operation *loop) {
    Attribute marker =
        loop->getAttr(controlflow::kPointerDescriptorBoundaryAttr);
    if (!marker)
      return;
    auto descriptorSlots = cast<DenseI32ArrayAttr>(marker);

    loop->removeAttr("MetaUse");
    for (int32_t slot : descriptorSlots.asArrayRef()) {
      if (failed(visitPointerDescriptorBoundarySlotValues(
              loop, slot,
              [&](Value value) { producerWorklist.push_back(value); })))
        valid = false;
    }
  });

  moduleOp.walk([&](Operation *op) {
    if (!op->hasAttr(controlflow::kPointerDescriptorRebuildAttr))
      return;

    // The rebuild operation itself and each of its operands are exact live
    // conversion roots. This includes an invariant opaque tensor base as well
    // as a scalar address behind int_to_ptr+splat. UseAnalysis has already
    // propagated MetaUse by this point; preserving only the offset would leave
    // a live ptr operand's producer eligible for erasure. This operand-local
    // closure is still narrower than retaining all loop operands/terminators.
    op->removeAttr("MetaUse");
    producerWorklist.append(op->operand_begin(), op->operand_end());
  });

  // UseAnalysis classifies load/store masks and load fallback values as
  // pointer metadata. A descriptor rebuild, however, is converted to a memref
  // while its consumer still needs those tensor values to form subviews and
  // padding. Preserve these exact consumer operands when the pointer is owned
  // by the CFO handoff; otherwise MetaUseEraser can leave an unconvertible
  // memref-to-pointer materialization at the live memory operation.
  moduleOp.walk([&](triton::LoadOp load) {
    if (!hasPointerDescriptorRebuildProvenance(load.getPtr()))
      return;
    if (Value mask = load.getMask())
      producerWorklist.push_back(mask);
    if (Value other = load.getOther())
      producerWorklist.push_back(other);
  });
  moduleOp.walk([&](triton::StoreOp store) {
    if (hasPointerDescriptorRebuildProvenance(store.getPtr()) &&
        store.getMask())
      producerWorklist.push_back(store.getMask());
  });
  moduleOp.walk([&](triton::AtomicRMWOp atomic) {
    if (hasPointerDescriptorRebuildProvenance(atomic.getPtr()) &&
        atomic.getMask())
      producerWorklist.push_back(atomic.getMask());
  });

  llvm::DenseSet<Value> visitedValues;
  while (!producerWorklist.empty()) {
    Value value = producerWorklist.pop_back_val();
    if (!value || !visitedValues.insert(value).second)
      continue;
    if (Operation *producer = value.getDefiningOp())
      producer->removeAttr("MetaUse");
    if (failed(appendProducerOperands(value, producerWorklist)))
      valid = false;
  }
  return success(valid);
}

// The generic canonicalizer may remove forwarded scf.for/scf.while iter-args,
// which changes a marker loop's positional signature without updating its
// external descriptor-slot metadata. Marker-bearing modules therefore use this
// narrow pre-clean: CSE remains enabled, and only constant scf.if regions are
// inlined so use analysis does not visit unreachable branches.
class FoldConstantIfBeforeUseAnalysis final
    : public OpRewritePattern<scf::IfOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto condition = ifOp.getCondition().getDefiningOp<arith::ConstantOp>();
    if (!condition)
      return failure();
    auto conditionValue = dyn_cast<IntegerAttr>(condition.getValue());
    if (!conditionValue || !conditionValue.getType().isInteger(1))
      return failure();

    Block *selectedBlock =
        conditionValue.getValue().isOne() ? ifOp.thenBlock() : ifOp.elseBlock();
    if (!selectedBlock) {
      if (ifOp.getNumResults() != 0)
        return failure();
      rewriter.eraseOp(ifOp);
      return success();
    }

    auto yield = cast<scf::YieldOp>(selectedBlock->getTerminator());
    SmallVector<Value> replacements(yield.getOperands());
    rewriter.inlineBlockBefore(selectedBlock, ifOp->getBlock(),
                               ifOp->getIterator());
    rewriter.eraseOp(yield);
    rewriter.replaceOp(ifOp, replacements);
    return success();
  }
};

// Inspect Operation directly so discovery cannot silently miss an SCF marker
// or an if-only rebuild root. Any surviving handoff attribute disables generic
// pre-clean canonicalization because its validated slot positions and rebuild
// operands must remain stable until MetaUse preservation.
static bool containsPointerDescriptorHandoff(ModuleOp moduleOp) {
  bool found = false;
  moduleOp.walk([&](Operation *op) {
    if (op->hasAttr(controlflow::kPointerDescriptorBoundaryAttr) ||
        op->hasAttr(controlflow::kPointerDescriptorRebuildAttr) ||
        op->hasAttr(controlflow::kPointerDescriptorOffsetFormAttr))
      found = true;
  });
  return found;
}

// A converted make_range may leave a tensor-valued SCF state behind even
// though no loop result or loop-body operation observes it. Keep this cleanup
// deliberately narrow: it recognizes only statically shaped integer tensors
// whose entire loop-carried use chain consists of uniform integer updates.
static bool isDeadRangeUpdate(Value value, scf::ForOp loop, unsigned slot,
                              llvm::SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return false;

  auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
  bool sawYield = false;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (user == yield) {
      if (use.getOperandNumber() != slot)
        return false;
      sawYield = true;
      continue;
    }

    Value updateResult;
    Value lhs;
    Value rhs;
    if (auto add = dyn_cast<arith::AddIOp>(user)) {
      updateResult = add.getResult();
      lhs = add.getLhs();
      rhs = add.getRhs();
    } else if (auto sub = dyn_cast<arith::SubIOp>(user)) {
      updateResult = sub.getResult();
      lhs = sub.getLhs();
      rhs = sub.getRhs();
    } else {
      return false;
    }
    if (updateResult.getType() != value.getType())
      return false;

    bool usesValueAsLhs = lhs == value;
    bool usesValueAsRhs = rhs == value;
    if (usesValueAsLhs == usesValueAsRhs)
      return false;
    Value delta = usesValueAsLhs ? rhs : lhs;
    auto deltaType = dyn_cast<RankedTensorType>(delta.getType());
    if (!deltaType || deltaType != value.getType())
      return false;

    Operation *deltaProducer = delta.getDefiningOp();
    bool isUniformDelta = false;
    if (auto constant = dyn_cast_or_null<arith::ConstantOp>(deltaProducer)) {
      if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()))
        isUniformDelta = dense.isSplat();
    }
    if (auto fill = dyn_cast_or_null<linalg::FillOp>(deltaProducer))
      isUniformDelta = fill.getInputs().size() == 1 &&
                       fill.getOutputs().size() == 1 &&
                       fill.getOutputs().front().getType() == value.getType();
    if (!isUniformDelta)
      return false;

    if (!isDeadRangeUpdate(updateResult, loop, slot, visited))
      return false;
  }
  return sawYield;
}

static bool isDeadRangeCarrier(scf::ForOp loop, unsigned slot) {
  if (slot >= loop.getInitArgs().size() ||
      slot >= loop.getRegionIterArgs().size() ||
      slot >= loop.getResults().size() ||
      slot >= loop.getYieldedValues().size() ||
      !loop.getResult(slot).use_empty())
    return false;

  auto type = dyn_cast<RankedTensorType>(loop.getInitArgs()[slot].getType());
  auto iterType =
      dyn_cast<RankedTensorType>(loop.getRegionIterArgs()[slot].getType());
  if (!type || !iterType || type != iterType || !type.hasStaticShape() ||
      !isa<IntegerType>(type.getElementType()))
    return false;

  Operation *producer = loop.getInitArgs()[slot].getDefiningOp();
  if (!producer ||
      (!isa<linalg::GenericOp, linalg::FillOp, tensor::CastOp>(producer) &&
       !producer->hasAttr("tt.from_make_range") &&
       !producer->hasAttr("tt.make_range_offset") &&
       !producer->hasAttr("tt.make_range_size")))
    return false;

  llvm::SmallPtrSet<Value, 8> visited;
  return isDeadRangeUpdate(loop.getRegionIterArgs()[slot], loop, slot, visited);
}

// Remove only fully dead range-like SCF state. Body arguments and yield
// operands are removed first, then the loop is rebuilt with surviving values.
static void eraseDeadRangeCarriers(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp loop) { loops.push_back(loop); });

  for (scf::ForOp loop : loops) {
    if (!loop || loop->getParentOp() == nullptr)
      continue;
    llvm::BitVector dead(loop.getInitArgs().size());
    for (unsigned i = 0; i < dead.size(); ++i) {
      if (isDeadRangeCarrier(loop, i))
        dead.set(i);
    }
    if (dead.none())
      continue;

    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    yield->eraseOperands(dead);
    loop.getBody()->eraseArguments([&](BlockArgument arg) {
      unsigned argNumber = arg.getArgNumber();
      return argNumber != 0 && dead.test(argNumber - 1);
    });

    llvm::BitVector operandIndices(loop->getNumOperands());
    for (auto [i, init] : llvm::enumerate(loop.getInitArgsMutable())) {
      if (dead.test(i))
        operandIndices.set(init.getOperandNumber());
    }
    loop->eraseOperands(operandIndices);

    OperationState state(loop.getLoc(), loop->getName(), loop->getOperands(),
                         loop.getInitArgs().getTypes(), loop->getAttrs());
    state.addRegion()->takeBody(loop.getBodyRegion());
    OpBuilder builder(loop);
    auto newLoop = cast<scf::ForOp>(builder.create(state));

    unsigned newResultIndex = 0;
    for (auto [i, result] : llvm::enumerate(loop.getResults())) {
      if (dead.test(i)) {
        assert(result.use_empty() && "dead range result still has uses");
        continue;
      }
      result.replaceAllUsesWith(newLoop.getResult(newResultIndex++));
    }
    loop.erase();
  }
}

static LogicalResult preCleanBeforeUseAnalysis(ModuleOp moduleOp) {
  bool hasPointerDescriptorHandoff = containsPointerDescriptorHandoff(moduleOp);

  PassManager csePipeline(moduleOp.getContext(), moduleOp.getOperationName());
  csePipeline.addPass(createCSEPass());
  if (!hasPointerDescriptorHandoff)
    csePipeline.addPass(createCanonicalizerPass());
  if (failed(csePipeline.run(moduleOp)))
    return failure();

  if (!hasPointerDescriptorHandoff)
    return success();
  RewritePatternSet patterns(moduleOp.getContext());
  patterns.add<FoldConstantIfBeforeUseAnalysis>(moduleOp.getContext());
  return applyPatternsGreedily(moduleOp, std::move(patterns));
}

// Recomputes the result layouts of subviews whose source descriptor has been
// rebased. A memref.subview result layout is derived from both its mixed
// offsets/strides and its source layout. For example, changing the source from
// `memref<32xf32, strided<[1], offset: ?>>` to the equivalent rebased
// `memref<32xf32, strided<[1]>>` changes a zero-offset subview result from a
// dynamic offset to offset zero. Merely changing the source SSA value leaves
// the old result type behind and makes SubViewOp verification fail.
//
// Subviews may be chained, so every updated result becomes a new worklist
// source. Rank-reduced subviews retain their existing result shape while their
// layout is inferred again from the rebased source.
static LogicalResult propagateRebasedSubviewTypes(Value rebasedSource,
                                                  IRRewriter &rewriter) {
  SmallVector<Value> sources{rebasedSource};
  llvm::SmallPtrSet<Operation *, 8> visited;

  while (!sources.empty()) {
    Value source = sources.pop_back_val();
    for (Operation *user : source.getUsers()) {
      auto subview = dyn_cast<memref::SubViewOp>(user);
      if (!subview || !visited.insert(user).second)
        continue;

      auto sourceType = dyn_cast<MemRefType>(subview.getSource().getType());
      auto oldResultType = dyn_cast<MemRefType>(subview.getResult().getType());
      if (!sourceType || !oldResultType)
        return failure();

      Type inferredType;
      if (sourceType.getRank() == oldResultType.getRank()) {
        inferredType = memref::SubViewOp::inferResultType(
            sourceType, subview.getMixedOffsets(), subview.getMixedSizes(),
            subview.getMixedStrides());
      } else {
        inferredType = memref::SubViewOp::inferRankReducedResultType(
            oldResultType.getShape(), sourceType, subview.getMixedOffsets(),
            subview.getMixedSizes(), subview.getMixedStrides());
      }

      auto inferredMemRefType = dyn_cast<MemRefType>(inferredType);
      if (!inferredMemRefType)
        return failure();
      if (inferredMemRefType != oldResultType) {
        rewriter.modifyOpInPlace(
            subview, [&] { subview.getResult().setType(inferredMemRefType); });
      }
      sources.push_back(subview.getResult());
    }
  }
  return success();
}

// Returns true when rebasing a descriptor would change a type owned by a
// different operation. A subview chain ending in direct memref loads/stores is
// local to this rewrite. Calls, returns, SCF/CFG boundaries, and every other
// user retain their existing descriptor layout and therefore stop rebasing.
static bool reachesLayoutSensitiveBoundary(Value root) {
  SmallVector<Value> worklist{root};
  llvm::DenseSet<Value> visited;
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second)
      continue;
    for (Operation *user : value.getUsers()) {
      if (auto subview = dyn_cast<memref::SubViewOp>(user)) {
        if (subview.getSource() == value) {
          worklist.push_back(subview.getResult());
          continue;
        }
      }
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        if (load.getMemRef() == value)
          continue;
      }
      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        if (store.getMemRef() == value)
          continue;
      }
      return true;
    }
  }
  return false;
}

// Convert structured custom ops after operand type converted,
// for example tt.ptr converted to memref.
template <typename CustomOpT>
class StructuredCustomOpConverter : public OpConversionPattern<CustomOpT> {
public:
  using OpConversionPattern<CustomOpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CustomOpT op, typename CustomOpT::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    BlockDataParser::rewriteStructuredCustomOp(op, adaptor, rewriter);
    return success();
  }
};

// Collect the ops in a scan body that actually contribute to the yielded
// result: compute the backward slice from the terminator operands, then drop
// pure type-cast ops. Mirrors ReductionOpBaseConverter::getRealReductionOps so
// that auxiliary ops (e.g. the overflow asserts injected by TRITON_DEBUG)
// never feed the yielded result and are ignored here as well.
static llvm::SmallVector<Operation *> getRealScanBodyOps(triton::ScanOp op) {
  Block *body = op.getBody();
  Operation *terminator = body->getTerminator();

  llvm::DenseSet<Operation *> liveOps;
  llvm::SmallVector<Value> worklist(terminator->getOperands());
  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (auto *defOp = val.getDefiningOp()) {
      if (defOp->getBlock() == body && liveOps.insert(defOp).second) {
        for (auto operand : defOp->getOperands())
          worklist.push_back(operand);
      }
    }
  }

  llvm::SmallVector<Operation *> realOps;
  for (Operation &bodyOp : body->without_terminator()) {
    if (!liveOps.contains(&bodyOp))
      continue;
    if (isa<arith::ExtFOp, arith::TruncFOp, arith::BitcastOp>(&bodyOp))
      continue;
    realOps.push_back(&bodyOp);
  }
  return realOps;
}

// A tt.scan that is (1) a plain cumsum (combine body is a single add, matching
// ScanConverter's triton_cumsum selection) and (2) collapses to a 1-D scan
// after backend lowering, i.e. every dim except the scan axis has extent 1
// (e.g. [1,1,128,1] with axis=2). Only this case is routed to the SIMT
// (Sklansky) cumsum template; cumprod / generic scans and multi-dim cumsum stay
// on SIMD.
static bool isSimt1DCumsum(triton::ScanOp op) {
  // (1) Must be a single-add combine body (only ops that feed the yielded
  // result count, mirroring getRealReductionOps).
  llvm::SmallVector<Operation *> realOps = getRealScanBodyOps(op);
  if (realOps.size() != 1 ||
      !isa<arith::AddFOp, arith::AddIOp>(realOps.front()))
    return false;

  // (2) Must be the 1-D scenario: all non-scan dims are unit-sized.
  auto srcTy = dyn_cast<RankedTensorType>(op.getOperand(0).getType());
  if (!srcTy || !srcTy.hasRank())
    return false;
  int64_t axis = op.getAxis();
  ArrayRef<int64_t> shape = srcTy.getShape();
  if (axis < 0 || axis >= static_cast<int64_t>(shape.size()))
    return false;
  for (int64_t i = 0; i < static_cast<int64_t>(shape.size()); ++i) {
    if (i != axis && shape[i] != 1)
      return false;
  }
  return true;
}

static bool isCustomOpOperandTypesLegal(TypeRange types) {
  return llvm::all_of(types, [](Type t) {
    if (isa<triton::PointerType>(t)) {
      return false;
    }
    if (auto shapedType = dyn_cast<ShapedType>(t)) {
      return !isa<triton::PointerType>(shapedType.getElementType());
    }
    return true;
  });
}

static bool isSIMTOp(Operation *op) {
  if (auto custom_op = dyn_cast<hivm::CustomOp>(op)) {
    return custom_op.getCoreType() == hivm::TCoreType::VECTOR &&
           custom_op.getVFMode() == hivm::VFMode::SIMT;
  }

  if (isa<triton::GatherOp>(op) && compileOn91095Flag) {
    return true;
  }

  if (isa<triton::HistogramOp>(op) && compileOn91095Flag) {
    return true;
  }

  // tt.scan: only a 1-D cumsum is treated as a SIMT op (drives the kernel
  // parallel_mode -> mix_simd_simt -> enable_simt). Everything else stays SIMD.
  if (compileOn91095Flag) {
    if (auto scan = dyn_cast<triton::ScanOp>(op)) {
      return isSimt1DCumsum(scan);
    }
  }

  // math.sin / math.cos on f16/f32 inputs: downstream (Ascend 950 RegBase
  // normalize, enable-high-precision defaults to true) rewrites them into a
  // Payne-Hanek range reduction that looks up a 320xi32 2/pi limbs table with
  // two hfusion.gather ops per collapsed region.  Match that scenario here and
  // route it to the SIMT template so the table gathers run in SIMT.
  if (compileOn91095Flag && (isa<math::SinOp>(op) || isa<math::CosOp>(op))) {
    Type inElem = getElementTypeOrSelf(op->getOperand(0).getType());
    if (inElem.isF16() || inElem.isF32()) {
      return true;
    }
  }

  return isa<triton::ascend::IndexPutOp, triton::ascend::GatherOutToUbOp,
             triton::ascend::ScatterUbToOutOp, triton::ascend::IndirectLoadOp,
             triton::ascend::StrideLoadOp, triton::ascend::StrideStoreOp,
             triton::ascend::IndirectStoreOp>(op);
}

TritonTypeConverter::TritonTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion([](triton::PointerType ptrType) {
    Type elem = ptrType.getPointeeType();
    // Handling special case: ptr<i1> -> memref<?x i8>
    if (auto it = dyn_cast<IntegerType>(elem); it && it.getWidth() == 1) {
      elem = IntegerType::get(ptrType.getContext(), 8);
      LLVM_DEBUG({
        llvm::dbgs() << "[TritonTypeConverter] Normalize i1 pointer to i8 "
                        "memref. elemType="
                     << elem << "\n";
      });
    }
    return MemRefType::get({ShapedType::kDynamic}, elem);
  });

  addConversion([](TensorType tensorType) -> Type {
    auto elemType = tensorType.getElementType();
    if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
      elemType = ptrType.getPointeeType();
    }
    // Handling special case: tensor<i1> -> memref<?x i8>
    if (auto it = dyn_cast<IntegerType>(elemType); it && it.getWidth() == 1) {
      elemType = IntegerType::get(tensorType.getContext(), 8);
      LLVM_DEBUG({
        llvm::dbgs() << "[TritonTypeConverter] Normalize i1 tensor to i8 "
                        "memref. elemType="
                     << elemType << "\n";
      });
    }
    return MemRefType::get(tensorType.getShape(), elemType);
  });

  // A pointer-descriptor boundary can intentionally remain a legal SCF op
  // carrying pointer-free tensor values while function signature conversion
  // changes the corresponding argument to a memref. Materialize only the
  // canonical numerical memref-to-ranked-tensor pair produced by the
  // conversion above. The exact type check deliberately excludes pointer
  // tensors, encoded tensors, i1-to-i8 normalization, non-identity layouts,
  // and non-default memory spaces, so this does not become a general SCF
  // type-conversion path.
  addSourceMaterialization([](OpBuilder &builder, RankedTensorType resultType,
                              ValueRange inputs, Location loc) -> Value {
    if (inputs.size() != 1 || resultType.getEncoding() ||
        isa<triton::PointerType>(resultType.getElementType()))
      return nullptr;

    auto inputType = dyn_cast<MemRefType>(inputs.front().getType());
    if (!inputType || !inputType.getLayout().isIdentity() ||
        inputType.getMemorySpace())
      return nullptr;

    auto expectedInputType =
        MemRefType::get(resultType.getShape(), resultType.getElementType());
    if (inputType != expectedInputType)
      return nullptr;

    return builder.create<bufferization::ToTensorOp>(
        loc, resultType, inputs.front(), /*restrict=*/false,
        /*writable=*/false);
  });
}

void TritonToLinalgPass::addProgramInfo(triton::FuncOp func,
                                        bool globalKernel) {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newFuncType =
      b.getFunctionType(newInputTypes, origFuncType.getResults());

  func.setFunctionType(newFuncType);

  // If argument attributes exist, extend attribute list.
  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  // Append the arguments to the entry block.
  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  }

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}

LogicalResult
TritonToLinalgPass::convertMultipleBlockControlFlow(Operation *funcOp,
                                                    OpBuilder &builder) {
  if (!isa<func::FuncOp>(funcOp)) {
    funcOp->emitError(
        "convertMultipleBlockControlFlow can only process func::FuncOp!");
    return failure();
  }

  SmallVector<Operation *> candidate;
  SmallVector<Block *> eraseBlocks;
  for (Block &block : dyn_cast<func::FuncOp>(funcOp).getBody()) {
    auto curTerminator = block.getTerminator();
    if (isa<cf::CondBranchOp>(curTerminator)) {
      candidate.push_back(curTerminator);
    } else if (isa<triton::ReturnOp>(curTerminator)) {
      if (candidate.empty()) {
        curTerminator->emitError(
            "funcOp has more than one Block but got an early 'tt.return' Op.");
        return failure();
      }
    } else if (!isa<cf::BranchOp>(curTerminator)) {
      funcOp->emitError(
          "funcOp has more than one Block but found unsupported Terminator: ")
          << *curTerminator;
      return failure();
    }

    if (!block.isEntryBlock())
      eraseBlocks.push_back(&block);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << candidate.size()
                 << " candidate cond_branch operations to convert.\n";
  });

  if (candidate.empty()) {
    funcOp->emitError("funcOp has more than one Block but no candidate "
                      "Terminator was found!");
    return failure();
  }

  llvm::BitVector visitFlag(candidate.size(), false);

  // Recursive function to convert all cf::CondBranchOp to scf::IfOp
  std::function<void(Operation *, Operation *)> convertToSCF =
      [&](Operation *op, Operation *insertPosOp) -> void {
    auto condBranchOp = dyn_cast_if_present<cf::CondBranchOp>(op);
    auto iter = llvm::find(candidate, condBranchOp);
    if (!(condBranchOp && iter != candidate.end())) {
      op->emitError(
          "convertToSCF must process with condBranchOp in candidates!");
      return;
    }
    visitFlag.set(iter - candidate.begin());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(insertPosOp);

    // Well, here force to destory original control flow
    builder.create<scf::IfOp>(
        condBranchOp->getLoc(), condBranchOp.getCondition(),
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getTrueDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getTrueDest()->getTerminator();
          if (auto nextCond = dyn_cast<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError("movedOps can not be empty before entering "
                                   "convertToSCF (then)!");
              return;
            }
            convertToSCF(nextCond, movedOps.back());
          } else if (!isa<cf::BranchOp, triton::ReturnOp>(blockTerm)) {
            blockTerm->emitError(
                "Unsupported terminator in then branch after structuring");
          }

          builder.create<scf::YieldOp>(loc);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getFalseDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getFalseDest()->getTerminator();
          if (auto nextCond = dyn_cast<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError("movedOps can not be empty before entering "
                                   "convertToSCF (else)!");
              return;
            }
            convertToSCF(nextCond, movedOps.back());
          } else if (!isa<cf::BranchOp, triton::ReturnOp>(blockTerm)) {
            blockTerm->emitError(
                "Unsupported terminator in else branch after structuring");
          }
          builder.create<scf::YieldOp>(loc);
        });
  };

  Block::iterator insertOp(candidate.front());
  if (insertOp == candidate.front()->getBlock()->begin()) {
    // if the first operation is a cond_branch, we need to insert before it
    convertToSCF(candidate.front(), candidate.front());
  } else {
    --insertOp;
    convertToSCF(candidate.front(), &(*insertOp));
  }

  if (!visitFlag.all()) {
    funcOp->emitError("Not all cf.cond_br converted!");
    return failure();
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(candidate.front());
  builder.create<triton::ReturnOp>(candidate.front()->getLoc());

  for (Operation *eachTerm : candidate)
    eachTerm->erase();
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

void TritonToLinalgPass::convertTTFunc(triton::FuncOp func, const bool existDot,
                                       const bool existSIMTOp) {
  OpBuilder builder(func);

  auto name = func.getName();
  auto type = func.getFunctionType();

  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  func.getAllArgAttrs(argAttrs);
  func.getAllResultAttrs(resAttrs);

  // Special handling for bit-casted tt.ptr arguments
  SmallVector<Type> inputTypes{type.getInputs()};
  SmallVector<Type> retTypes{type.getResults()};
  if (func.getSymVisibility() == "public" && !func.isDeclaration()) {
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      auto arg = func.getArgument(i);
      // Special method for i1 arg
      if (!isa<BaseMemRefType>(arg.getType()) ||
          dyn_cast<BaseMemRefType>(arg.getType()).getElementTypeBitWidth() !=
              1) {
        continue;
      }

      SmallVector<Operation *> argVaildUser{arg.getUsers()};
      llvm::erase_if(argVaildUser, [](Operation *op) -> bool {
        return isOpTriviallyDead(op);
      });

      if (!argVaildUser.empty()) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : argVaildUser) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        if (llvm::all_of(argVaildUser, [](Operation *userOp) {
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          auto castOp = cast<UnrealizedConversionCastOp>(*argVaildUser.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
        } else {
          func->emitError(Twine("Unsupported use of func arg at index ") +
                          Twine(i));
        }
      } else {
        // Process unused bool ptr type specially, which guarantees bool pointer
        // argument's type is realistic and don't mislead backend compiler.
        // realistic memory layout of bool pointer is 8 bit width
        auto memType = dyn_cast<BaseMemRefType>(arg.getType())
                           .cloneWith(std::nullopt, builder.getI8Type());
        arg.setType(memType);
        inputTypes[i] = arg.getType();
      }
    }
  }
  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  auto kernelAttr = func->getAttr(globalKernelAttr);
  if (kernelAttr) {
    funcFunc->setAttr(globalKernelAttr, kernelAttr);
  }
  std::string kernelMixMode = "aiv";
  if (existDot) {
    // mix also works for pure cube kernel by using the same MAGIC_ELF keyword
    kernelMixMode = "mix";
  }
  // Set mix_mode in the func attrs so that the backend could know
  // the mix_mode by parse the func attrs.
  // The backend needs to know the mix_mode because the host wrapper
  // needs to set the devbin.magic. Check npu_utils.cpp.
  funcFunc->setAttr(kernelMixModeName, builder.getStringAttr(kernelMixMode));

  std::string parallelMode = "simd";
  if (existSIMTOp) {
    parallelMode = "mix_simd_simt";
  }
  funcFunc->setAttr(kernelParallelModeName,
                    builder.getStringAttr(parallelMode));

  auto autoBlockifyAttr = func->getAttr("auto_blockify_size");
  if (autoBlockifyAttr)
    funcFunc->setAttr("auto_blockify_size", autoBlockifyAttr);

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  if (!funcFuncBody.hasOneBlock()) {
    if (failed(convertMultipleBlockControlFlow(funcFunc, builder))) {
      llvm_unreachable("Encounter unsupported control flow");
    }
  }

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
  }
  func.erase();
}

void TritonToLinalgPass::addDynamicLegal(
    ConversionTarget &target, TritonTypeConverter &tritonTypeConverter) {
  target.addLegalDialect<
      func::FuncDialect, arith::ArithDialect, math::MathDialect,
      linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
      cf::ControlFlowDialect, tensor::TensorDialect, LLVM::LLVMDialect,
      bufferization::BufferizationDialect, memref::MemRefDialect,
      annotation::AnnotationDialect, hivm::HIVMDialect, hfusion::HFusionDialect,
      scope::ScopeDialect>();

  // add legal dialect on condition
  target.addLegalOp<ModuleOp>();

  // decide which ops need conversion based on uses
  target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
      [](mlir::Operation *op) {
        if (op->use_empty()) {
          return false;
        } else {
          return true;
        }
      });

  target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
    return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  // For CustomOp/CustomMacroOp, tt.ptr should be converted to memref.
  target.addDynamicallyLegalOp<hivm::CustomOp>([&](hivm::CustomOp op) {
    return isCustomOpOperandTypesLegal(op->getOperandTypes());
  });
  target.addDynamicallyLegalOp<hivm::CustomMacroOp>(
      [&](hivm::CustomMacroOp op) {
        return isCustomOpOperandTypesLegal(op->getOperandTypes());
      });

  target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
    auto res = op.getResult();
    if (!isa<RankedTensorType>(res.getType())) {
      return true;
    }

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (!denseAttr.isSplat() ||
          !isa<FloatType, IntegerType>(denseAttr.getElementType())) {
        return true;
      }
      if (res.hasOneUse() && isa<tensor::ReshapeOp>(*res.user_begin())) {
        return true;
      }
      return false;
    }
    return true;
  });

  target.addDynamicallyLegalOp<scf::IfOp>(
      [](scf::IfOp op) { return !TTOpConverters::hasScalarPointerResult(op); });

  auto controlFlowTerminatorLegal = [](Operation *op) {
    Operation *parent = op->getParentOp();
    if (parent &&
        parent->hasAttr(TTOpConverters::kScalarPointerCarrierBoundaryAttr)) {
      if (auto parentIf = dyn_cast<scf::IfOp>(parent))
        return llvm::equal(op->getOperandTypes(), parentIf.getResultTypes());
    }

    if (parent && parent->hasAttr(controlflow::kPointerDescriptorBoundaryAttr))
      return hasPointerFreeControlFlowBoundary(
          cast<LoopLikeOpInterface>(parent));

    return llvm::all_of(op->getOperandTypes(), [](Type t) {
      if (isa<triton::PointerType>(t))
        return false;
      if (auto shapedType = dyn_cast<ShapedType>(t))
        return shapedType.getElementType().isIntOrFloat();
      assert(t.isIntOrIndexOrFloat());
      return true;
    });
  };

  target.addDynamicallyLegalOp<scf::YieldOp, scf::ConditionOp>(
      controlFlowTerminatorLegal);

  auto isArithOrMathOpLegal = [this](Operation *op) {
    if (op->hasAttr("MetaUse"))
      return false;

    if (isa<arith::ConstantOp>(op))
      return true;

    bool operateOnTensors = llvm::all_of(op->getOperandTypes(), [](Type type) {
      return isa<RankedTensorType>(type);
    });

    return this->namedOps || !operateOnTensors;
  };

  // Numeric selects retain the existing Arith legality. Every scalar-pointer
  // select uses the integer-address converter so no memref object crosses it.
  target.addDynamicallyLegalOp<arith::SelectOp>(
      [isArithOrMathOpLegal](arith::SelectOp op) {
        if (TTOpConverters::isScalarPointerSelect(op))
          return false;
        return isArithOrMathOpLegal(op);
      });

  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      isArithOrMathOpLegal);
}

namespace {

/// Route the specific `splat(base) -> addptr(scan) -> addptr(constant) ->
/// store` form through the existing indirect-store operation before use
/// analysis. Lowering either addptr independently would materialize a memref
/// while the remaining operation still expects tensor<ptr>.
class FoldScanOffsetAddPtrChain : public OpRewritePattern<triton::AddPtrOp> {
public:
  using OpRewritePattern<triton::AddPtrOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::AddPtrOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->hasOneUse() || !op.getOffset().getDefiningOp<triton::ScanOp>())
      return failure();
    auto nextAddPtr = dyn_cast<triton::AddPtrOp>(*op->user_begin());
    if (!nextAddPtr ||
        !nextAddPtr.getOffset().getDefiningOp<arith::ConstantOp>() ||
        op.getOffset().getType() != nextAddPtr.getOffset().getType() ||
        !nextAddPtr->hasOneUse())
      return failure();
    auto store = dyn_cast<triton::StoreOp>(*nextAddPtr->user_begin());
    auto baseSplat = op.getPtr().getDefiningOp<triton::SplatOp>();
    if (!store || !baseSplat ||
        !isa<triton::PointerType>(baseSplat.getSrc().getType()))
      return failure();

    rewriter.setInsertionPoint(nextAddPtr);
    Value combinedOffset = rewriter.create<arith::AddIOp>(
        nextAddPtr.getLoc(), op.getOffset(), nextAddPtr.getOffset());
    rewriter.create<triton::ascend::IndirectStoreOp>(
        store.getLoc(), baseSplat.getSrc(), combinedOffset, store.getValue(),
        store.getMask());
    rewriter.eraseOp(store);
    rewriter.eraseOp(nextAddPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

// Version only the small, bufferized row-copy loops emitted for mixed-axis
// loads. The column extent is shared by every row: checking it once outside
// the loop lets full tiles recover a static copy while tails keep their mask.
// Run after memory conversion so both branches use the same outer allocation.
static void specializeFullRowCopies(ModuleOp moduleOp) {
  SmallVector<scf::ForOp> loops;
  moduleOp.walk([&](scf::ForOp loop) {
    if (loop->hasAttr("ExtractedLoadOrStore") &&
        loop->hasAttr("hivm.parallel_loop") && loop.getNumResults() == 0)
      loops.push_back(loop);
  });
  IRRewriter rewriter(moduleOp.getContext());
  for (scf::ForOp loop : loops) {
    // Do not duplicate nested control flow, allocations, or other accesses.
    memref::CopyOp copy;
    bool supported = true;
    unsigned numOps = 0;
    for (Operation &op : loop.getBody()->without_terminator()) {
      if (++numOps > 32 || op.getNumRegions() != 0) {
        supported = false;
        break;
      }
      if (auto candidate = dyn_cast<memref::CopyOp>(op)) {
        if (copy) {
          supported = false;
          break;
        }
        copy = candidate;
      } else if (!isMemoryEffectFree(&op)) {
        supported = false;
        break;
      }
    }
    if (!supported || !copy)
      continue;

    auto src = copy.getSource().getDefiningOp<memref::SubViewOp>();
    auto dst = copy.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!src || !dst || src.getSourceType().getRank() != 2 ||
        dst.getSourceType().getRank() != 2 || src.getType().getRank() != 2 ||
        dst.getType().getRank() != 2)
      continue;
    auto shape = src.getSourceType().getShape();
    if (shape[0] != 1 || ShapedType::isDynamic(shape[1]) || shape[1] <= 1 ||
        dst.getSourceType().getShape() != shape)
      continue;
    int64_t width = shape[1];
    auto isZero = [](OpFoldResult v) { return isConstantIntValue(v, 0); };
    auto isOne = [](OpFoldResult v) { return isConstantIntValue(v, 1); };
    auto isRowPrefix = [&](memref::SubViewOp view) {
      auto [strides, offset] = view.getSourceType().getStridesAndOffset();
      return strides.back() == 1 &&
             llvm::all_of(view.getMixedOffsets(), isZero) &&
             llvm::all_of(view.getMixedStrides(), isOne) &&
             isOne(view.getMixedSizes()[0]);
    };
    if (!isRowPrefix(src) || !isRowPrefix(dst) ||
        src.getMixedSizes()[1] != dst.getMixedSizes()[1])
      continue;
    auto extent = dyn_cast<Value>(src.getMixedSizes()[1]);
    if (!extent || !loop.isDefinedOutsideOfLoop(extent))
      continue;

    // Require the existing load buffer and its exact row slice. This excludes
    // stores and avoids moving/duplicating a per-row allocation into a branch.
    auto row = dst.getSource().getDefiningOp<memref::SubViewOp>();
    auto source = src.getSource().getDefiningOp<memref::ReinterpretCastOp>();
    if (!row || !source || row.getSourceType().getRank() != 2 ||
        !row.getSource().getDefiningOp<memref::AllocOp>() ||
        !loop.isDefinedOutsideOfLoop(row.getSource()) ||
        !loop.isDefinedOutsideOfLoop(source.getSource()) ||
        row.getMixedOffsets()[0] != OpFoldResult(loop.getInductionVar()) ||
        !isZero(row.getMixedOffsets()[1]) ||
        !llvm::all_of(row.getMixedStrides(), isOne) ||
        !isOne(row.getMixedSizes()[0]) ||
        !isConstantIntValue(row.getMixedSizes()[1], width))
      continue;

    rewriter.setInsertionPoint(loop);
    Location loc = copy.getLoc();
    Value fullWidth = rewriter.create<arith::ConstantIndexOp>(loc, width);
    Value full = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                extent, fullWidth);
    auto branch =
        rewriter.create<scf::IfOp>(loc, full, /*withElseRegion=*/true);
    rewriter.setInsertionPointToStart(branch.thenBlock());
    IRMapping mapping;
    auto fastLoop = cast<scf::ForOp>(rewriter.clone(*loop, mapping));
    auto fastCopy = *fastLoop.getBody()->getOps<memref::CopyOp>().begin();
    // Under extent == width the two zero-offset subviews are the complete
    // static rows. Bypass them explicitly; no branch-sensitive folding needed.
    fastCopy->setOperand(0, mapping.lookupOrDefault(src.getSource()));
    fastCopy->setOperand(1, mapping.lookupOrDefault(dst.getSource()));
    loop->moveBefore(branch.elseBlock()->getTerminator());
    // The A5 backend must be able to specialize the full-width library call
    // independently of the dynamic tail call that shares its callee.
    moduleOp->setAttr("tt.full_row_copy", rewriter.getUnitAttr());
  }
}

void TritonToLinalgPass::populateTritonToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns) {
  patterns.add<LoadStoreConverter::LoadStoreCanonicalizer<triton::LoadOp>,
               LoadStoreConverter::LoadStoreCanonicalizer<triton::StoreOp>,
               LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicRMWOp>,
               LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicCASOp>>(
      patterns.getContext());
  patterns.add<FoldScanOffsetAddPtrChain>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastCanonicalizer>(patterns.getContext());
  patterns.add<TTOpConverters::FpToFpCanonicalizer>(patterns.getContext());
  patterns.add<LoadStoreConverter::ScalarStoreCanonicalizer>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::ScalarAtomicRMWCanonicalizer>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::ScalarAtomicCASCanonicalizer>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicMaxMinCanonicalizer>(
      patterns.getContext());
  patterns.add<
      TTOpConverters::ScalarMathCanonicalizer<math::AbsFOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AcosOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AcoshOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AsinOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AsinhOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AtanOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::Atan2Op>,
      // TTOpConverters::ScalarMathCanonicalizer<math::AtanhOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::CeilOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::CosOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::CoshOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::ErfOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::ExpOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::Exp2Op>,
      // TTOpConverters::ScalarMathCanonicalizer<math::ExpM1Op>,
      TTOpConverters::ScalarMathCanonicalizer<math::FloorOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::FmaOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::LogOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::Log10Op>,
      // TTOpConverters::ScalarMathCanonicalizer<math::Log1pOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::Log2Op>,
      // TTOpConverters::ScalarMathCanonicalizer<math::PowFOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::RoundOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::RsqrtOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::SinOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::SinhOp>,
      TTOpConverters::ScalarMathCanonicalizer<math::SqrtOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::TanOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::TanhOp>,
      // TTOpConverters::ScalarMathCanonicalizer<math::TruncOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::AddFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::SubFOp>,
      TTOpConverters::ScalarMathCanonicalizer<
          arith::
              MulFOp>, // TTOpConverters::ScalarMathCanonicalizer<arith::DivFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::NegFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::RemFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::MaxNumFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::MaximumFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::MinNumFOp>,
      TTOpConverters::ScalarMathCanonicalizer<arith::MinimumFOp>
      // By test, the following ops do not need canonicalization.
      // TTOpConverters::ScalarMathCanonicalizer<arith::CmpFOp>
      // TTOpConverters::ScalarMathCanonicalizer<arith::ExtFOp>
      // TTOpConverters::ScalarMathCanonicalizer<arith::TruncFOp>
      >(patterns.getContext());
  patterns.add<TTOpConverters::ReduceSingleCanonicalizer>(
      patterns.getContext());
  if (this->enableSelectAnalysis) {
    patterns.add<TTOpConverters::SelectCanonicalizer>(patterns.getContext());
  }
}

void TritonToLinalgPass::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  nd2nzFlag = this->enableNd2nzOnVector;
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);

  patterns.add<triton::MetaUseEraser>(patterns.getContext());
  patterns.add<LoadStoreConverter::StoreConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AddPtrConverter>(patterns.getContext());
  patterns
      .add<LoadStoreConverter::MemoryPointerConverter<triton::SplatOp>,
           LoadStoreConverter::MemoryPointerConverter<triton::BitcastOp>,
           LoadStoreConverter::MemoryPointerConverter<triton::BroadcastOp>,
           LoadStoreConverter::MemoryPointerConverter<triton::ExpandDimsOp>>(
          patterns.getContext());
  patterns.add<FunctionConverter::GetProgramIDConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetNumProgramsConverter>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::LoadConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicRMWConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicCASConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MakeRangeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::UnsplatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ClampFConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PreciseDivConverter>(patterns.getContext());
  // reduce converters
  patterns.add<TTOpConverters::ArgMinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ArgMaxConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReduceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ScanConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MapElementwiseDecomposeConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::ReshapeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExpandDimsConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BroadcastConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DenseConstantConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExternElementwiseClOpConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::TritonMulhiuiConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TritonPreciseSqrtConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<TTOpConverters::AdvanceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TransposeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplitConverter>(patterns.getContext());
  patterns.add<TTOpConverters::JoinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::CatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastConverter>(patterns.getContext());
  patterns.add<TTOpConverters::LoopConverter<scf::ForOp>>(patterns.getContext(),
                                                          PatternBenefit(2));
  patterns.add<TTOpConverters::LoopConverter<scf::WhileOp>>(
      patterns.getContext(), PatternBenefit(2));
  patterns.add<TTOpConverters::YieldConverter>(patterns.getContext(),
                                               PatternBenefit(2));
  patterns.add<TTOpConverters::IfConverter>(
      typeConverter, patterns.getContext(), PatternBenefit(2));
  patterns.add<TTOpConverters::PointerSelectConverter>(typeConverter,
                                                       patterns.getContext());

  patterns.add<TTOpConverters::DeviceAssertConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MatmulConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DotConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DotScaledConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PtrToIntConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IntToPtrConverter>(typeConverter,
                                                  patterns.getContext());

  patterns.add<TTOpConverters::IndirectLoadConverter>(patterns.getContext());
  patterns.add<TTOpConverters::StrideLoadConverter>(patterns.getContext());
  patterns.add<TTOpConverters::StrideStoreConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndirectStoreConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherOutToUbConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ScatterUbToOutConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndexSelectSimdConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndexPutConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SortOpConverter>(patterns.getContext());
  patterns.add<TTOpConverters::FlipOpConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherConverter>(patterns.getContext());
  // On 950 (910B4/91095), histogram is lowered via hivm.custom builtin
  // template. On other targets, histogram is handled by TritonToHFusion pass
  // instead.
  if (compileOn91095Flag) {
    patterns.add<TTOpConverters::HistogramConverter>(patterns.getContext());
  }

  // Add convert pattern for structured custom ops.
  patterns.add<StructuredCustomOpConverter<hivm::CustomOp>,
               StructuredCustomOpConverter<hivm::CustomMacroOp>>(
      patterns.getContext());

  if (!this->namedOps) {
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
  }
}

void TritonToLinalgPass::getDependentDialects(DialectRegistry &registry) const {
  registry
      .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
              linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
              tensor::TensorDialect, bufferization::BufferizationDialect,
              memref::MemRefDialect, hfusion::HFusionDialect, hivm::HIVMDialect,
              annotation::AnnotationDialect, LLVM::LLVMDialect,
              triton::ascend::TritonAscendDialect, scope::ScopeDialect>();
}

LogicalResult
TritonToLinalgPass::processDescriptorOperations(ModuleOp moduleOp) {
  // --- ConversionTarget: dynamic legality checks ---
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::tensor::TensorDialect>();

  // Dialect-level dynamic legality: ops are legal if none of their
  // operands/results use TensorDescType.
  target.addDynamicallyLegalDialect<
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, triton::TritonDialect>(
      [](mlir::Operation *op) {
        return !DescriptorConverter::hasATensorDescriptorType(
                   op->getOperandTypes()) &&
               !DescriptorConverter::hasATensorDescriptorType(
                   op->getResultTypes());
      });
  // Function signature legality: Triton FuncOp is legal if its inputs/outputs
  // contain no TensorDescType.
  target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
    return !DescriptorConverter::hasATensorDescriptorType(
               funcOp.getFunctionType().getInputs()) &&
           !DescriptorConverter::hasATensorDescriptorType(
               funcOp.getFunctionType().getResults());
  });
  target.addLegalOp<triton::MakeTensorDescOp>();
  target.addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
                      triton::DescriptorScatterOp, triton::DescriptorGatherOp,
                      triton::DescriptorReduceOp>();

  // --- Patterns ---
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<DescriptorConverter::DescriptorLoadConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorStoreConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorScatterConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorGatherConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorReduceConverter>(
      patterns.getContext());

  mlir::ConversionConfig config;
  config.buildMaterializations = true;
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns),
                                    config))) {
    moduleOp->emitError("failed to convert tensor descriptor operations");
    return failure();
  }

  return success();
}

LogicalResult
TritonToLinalgPass::processPtrBroadcastOperations(ModuleOp moduleOp) {
  // --- ConversionTarget: dynamic legality checks ---
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<triton::SplatOp>();
  target.addLegalOp<triton::AddPtrOp>();
  target.addDynamicallyLegalOp<triton::BroadcastOp>([](triton::BroadcastOp op) {
    if (op->hasAttr("MetaUse")) {
      return true;
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    HoistBroadcast::BroadcastHoister hoister(op);
    return !(isa<triton::PointerType>(resultType.getElementType()) &&
             hoister.canBroadcast());
  });

  // --- Patterns ---
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<HoistBroadcast::BroadcastConverter>(patterns.getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError("failed to convert ptr broadcast operations");
    return failure();
  }

  return success();
}

LogicalResult
TritonToLinalgPass::processImplicitPermuteOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ImplicitPermute::LoadConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::StoreConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicRMWConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicCASConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::SplatCmpConverter>(
      patterns.getContext());

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    LLVM_DEBUG({ llvm::dbgs() << "ImplicitPermute: rewrite MemOp failed\n"; });
  }

  mlir::PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  return runPipeline(pm, getOperation());
}

LogicalResult TritonToLinalgPass::processStridedLoadStoreRewriteOperations(
    ModuleOp moduleOp) {
  // The strided-axis rewrites below only apply in 950 SIMT mode. On other
  // targets we leave strided loads to the legacy strided DMA lowering.
  if (!(compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(compileModeFlag))) {
    return success();
  }

  auto runLayoutMemoryPhase =
      [&](cfg::LayoutMemoryCompatibilityPhase phase) -> LogicalResult {
    mlir::PassManager phasePm(&getContext(), moduleOp.getOperationName());
    phasePm.addPass(cfg::createLayoutMemoryCompatibilityPass(phase));
    return runPipeline(phasePm, getOperation());
  };

  // Keep the original insertion point after ImplicitPermute.  Axis remains in
  // the pre-Diagonal slot; the current target has no Diagonal migration, so
  // the two compatibility phases run adjacently.
  if (failed(runLayoutMemoryPhase(
          cfg::LayoutMemoryCompatibilityPhase::BeforeDiagonal))) {
    return failure();
  }

  if (failed(runLayoutMemoryPhase(
          cfg::LayoutMemoryCompatibilityPhase::AfterDiagonal))) {
    return failure();
  }

  // Mirror processImplicitPermuteOperations: clean up dead IR left behind by
  // PtrAnalysis when the pattern decided not to rewrite (e.g. stride==1 case
  // returns failure() but PtrAnalysis has already inserted helper ExtSI ops).
  // Without this, downstream passes may trip on stale uses.
  mlir::PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  return runPipeline(pm, getOperation());
}

LogicalResult
TritonToLinalgPass::processLegalStrideOperations(ModuleOp moduleOp) {
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<arith::ConstantOp>();
  target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
      [](memref::ReinterpretCastOp op) {
        return !LoadStoreConverter::ReinterpretCastStrideCanonicalizer::
            hasFixableZeroStride(op);
      });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LoadStoreConverter::ReinterpretCastStrideCanonicalizer>(
      patterns.getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError(
        "failed to legalize reinterpret_cast dynamic stride(0) with size(1)");
    return failure();
  }

  return success();
}

void TritonToLinalgPass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  auto compileMode = triton::ascend::parseCompileMode(this->compileMode);
  if (!compileMode) {
    getOperation().emitError()
        << "triton-to-linalg compile-mode is invalid: " << this->compileMode;
    signalPassFailure();
    return;
  }
  compileModeFlag = *compileMode;

  auto moduleOp = getOperation();

  // Validate the CFO handoff before descriptor conversion, canonicalization,
  // or any other IR mutation can erase malformed metadata with dead code.
  if (failed(validatePointerDescriptorHandoffMetadata(moduleOp))) {
    moduleOp->emitError("invalid pointer descriptor handoff metadata");
    signalPassFailure();
    return;
  }

  // Check if the kernel contains a cube op: tl.dot / tl.dot_scaled / al.dot
  // decompose into a cube linalg.matmul, and conv1d/conv2d run on the cube
  // unit. Without any of them the kernel would be tagged as a pure AIV kernel;
  // with them it must be tagged mix mode, otherwise the cube tile-and-slice
  // fails (cbuf overflow).
  bool existDot = false;
  moduleOp.walk([&](hivm::CustomOp customOp) {
    if (customOp.getCoreType() == hivm::TCoreType::CUBE_AND_VECTOR ||
        customOp.getCoreType() == hivm::TCoreType::CUBE) {
      existDot = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  moduleOp.walk([&](hivm::CustomMacroOp customMacroOp) {
    if (customMacroOp.getCoreType() == hivm::TCoreType::CUBE_AND_VECTOR ||
        customMacroOp.getCoreType() == hivm::TCoreType::CUBE) {
      existDot = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  moduleOp.walk([&](Operation *op) {
    if (isa<triton::DotOp, triton::DotScaledOp, triton::ascend::DotOp,
            hfusion::Conv1DOp, hfusion::Conv2DOp>(op)) {
      existDot = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  existDotFlag = existDot;

  // NOTE: existSIMTOp is intentionally computed AFTER
  // processStridedLoadStoreRewriteOperations below, because that step
  // materializes triton::ascend::IndirectLoadOp/IndirectStoreOp (which
  // isSIMTOp() counts). Walking here (before the rewrite) would miss them and
  // mislabel the kernel parallel_mode as "simd" instead of "mix_simd_simt";
  // then enable_simt would be false and the launch would not reserve
  // localMemorySize for the SIMT templates -> VEC UB out-of-bounds (error 341)
  // at runtime on mix-CV kernels.
  bool existSIMTOp = false;

  // Execute tensor descriptor operations conversion
  if (failed(processDescriptorOperations(moduleOp))) {
    signalPassFailure();
  }

  // Execute implicit permute
  if (failed(processImplicitPermuteOperations(moduleOp))) {
    LLVM_DEBUG(
        { llvm::dbgs() << "Failed to process implicit permute operations\n"; });
    signalPassFailure();
  }

  // SIMT IndirectLoad fast-path rewrite (runs after ImplicitPermute so the
  // permuted access patterns have already been absorbed; this step only
  // catches non-permuted last-axis stride > 1 loads).
  if (failed(processStridedLoadStoreRewriteOperations(moduleOp))) {
    LLVM_DEBUG({
      llvm::dbgs() << "Failed to process indirect-load rewrite operations\n";
    });
    signalPassFailure();
  }

  // Detect SIMT ops AFTER the indirect-load rewrite so the freshly materialized
  // IndirectLoadOp/IndirectStoreOp are counted (drives parallel_mode ->
  // "mix_simd_simt" -> enable_simt -> launch reserves localMemorySize).
  moduleOp.walk([&](Operation *op) {
    if (isSIMTOp(op)) {
      existSIMTOp = true;
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Found SIMT op in function: ";
        os << op->getName();
        os << "\n";
      });
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // 0. Annotate Memory-Related Triton FuncOps with tensor_kind (used by
  // profiling).
  {
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(triton::createMarkTensorKindPass());
    if (failed(runPipeline(pm, moduleOp))) {
      moduleOp->emitError("failed to run LoopCanonicalizerPass");
      signalPassFailure();
      return;
    }
  }

  RewritePatternSet canonicalizerPatterns(&getContext());
  // 1. Canonicalize load/store related patterns.
  // The currently registered patterns rewrite pointer consumers but do not
  // replace tt.make_tensor_ptr/tt.addptr descriptor rebuild roots. A future
  // pattern that rewrites those producers must atomically transfer
  // PointerDescriptorRebuild to its replacement; entry validation alone
  // cannot preserve metadata across a producer rewrite.
  this->populateTritonToLinalgCanonicalizationPatterns(canonicalizerPatterns);
  if (failed(
          applyPatternsGreedily(moduleOp, std::move(canonicalizerPatterns)))) {
    moduleOp->emitError("failed to apply Canonicalizer Patterns");
    signalPassFailure();
    return;
  }

  // Stage-1 may legitimately delete an unused, already validated boundary.
  // Revalidate every surviving handoff so later phases never consume a marker
  // whose structural edges were changed without updating its slot contract.
  if (failed(validatePointerDescriptorHandoffMetadata(moduleOp))) {
    moduleOp->emitError("invalid pointer descriptor handoff metadata");
    signalPassFailure();
    return;
  }

  // 2.1 Pre-clean dead control flow before use analysis. Descriptor handoff
  // attributes require stable loop positions and rebuild operands, so marked
  // modules use a restricted cleanup that cannot rewrite either contract.
  if (failed(preCleanBeforeUseAnalysis(moduleOp))) {
    moduleOp->emitError(
        "failed to pre-clean dead control-flow before use analysis");
    signalPassFailure();
    return;
  }

  // 2. Perform use analysis on FuncOp.
  moduleOp.walk([this](triton::FuncOp op) {
    if (failed(runUseAnalysis(op))) {
      signalPassFailure();
    }
  });

  if (failed(preservePointerDescriptorComputations(moduleOp))) {
    moduleOp->emitError("invalid pointer descriptor handoff metadata");
    signalPassFailure();
    return;
  }

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  TritonTypeConverter tritonTypeConverter{};

  // 3. Mark legal dialects and operations.
  this->addDynamicLegal(target, tritonTypeConverter);

  // 4. Mark ops that must be converted explicitly (e.g. tt.scan).
  auto loopOpLegalFn = [](LoopLikeOpInterface loopOp) {
    Operation *op = loopOp.getOperation();
    if (op->hasAttr(controlflow::kPointerDescriptorBoundaryAttr)) {
      return hasPointerFreeControlFlowBoundary(loopOp);
    }
    return !op->hasAttr("UnhandledLoopOp");
  };

  target.addIllegalOp<triton::ScanOp>();
  target.addIllegalOp<triton::MapElementwiseOp>();
  target.addDynamicallyLegalOp<scf::ForOp>(loopOpLegalFn);
  target.addDynamicallyLegalOp<scf::WhileOp>(loopOpLegalFn);

  // 5. Register converters for all illegal Triton ops.
  // Execute ptr broadcast operations conversion
  if (failed(processPtrBroadcastOperations(moduleOp))) {
    signalPassFailure();
  }
  this->populateTritonToLinalgConversionPatterns(tritonTypeConverter, patterns,
                                                 LAUNCH_GRID_RANK);

  // 6. Inject program id / number of programs arguments into each Triton kernel
  // function.
  for (auto func : getOperation().getOps<triton::FuncOp>()) {
    addProgramInfo(func, globalKernel);
  }

  moduleOp.walk([this](LoopLikeOpInterface loopOp) {
    auto *op = loopOp.getOperation();
    // CFO-expanded pointer loops already carry policy-owned descriptor
    // components. They require ordinary nested-op conversion, not the legacy
    // pointer-loop decomposition a second time. Unmarked loops are delegated
    // only when their original boundary still carries Triton pointer/address
    // state. A fixed-layout memref is already a complete SCF value; the fact
    // that its init is produced by reinterpret_cast does not make it BlockData.
    bool hasExpandedPointerDescriptor =
        op->hasAttr(mlir::triton::controlflow::kPointerDescriptorBoundaryAttr);
    if (!op->hasAttr("ExtractedLoadOrStore") &&
        needsLegacyBlockDataLoopRewrite(loopOp))
      op->setAttr("UnhandledLoopOp", UnitAttr::get(op->getContext()));

    if (hasExpandedPointerDescriptor)
      return;

    for (auto res : loopOp->getResults()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(res.getType());
          tensorType &&
          !isa<triton::PointerType>(tensorType.getElementType())) {
        IRRewriter rewriter(op->getContext());
        rewriter.setInsertionPointAfter(op);
        auto newVal =
            rewriter.create<tensor::CastOp>(op->getLoc(), res.getType(), res);
        rewriter.replaceAllUsesExcept(res, newVal, newVal);
      }
    }
  });

  // 7. Convert ops.
  LogicalResult conversionResult =
      applyPartialConversion(moduleOp, target, std::move(patterns));

  if (failed(conversionResult)) {
    moduleOp->emitError("failed to apply Conversion Patterns");
    signalPassFailure();
    return;
  }

  // These markers are contracts with this conversion pass. Remove them only
  // after successful conversion so failure reproducers retain ownership,
  // exact dynamic descriptor slots, and complete rebuild roots.
  moduleOp.walk([](LoopLikeOpInterface loopOp) {
    loopOp->removeAttr(controlflow::kPointerDescriptorBoundaryAttr);
  });
  moduleOp.walk([](Operation *op) {
    op->removeAttr(controlflow::kPointerDescriptorRebuildAttr);
    op->removeAttr(controlflow::kPointerDescriptorOffsetFormAttr);
    op->removeAttr(controlflow::kPointerDescriptorStructuredAxesAttr);
  });
  moduleOp.walk([](Operation *op) {
    op->removeAttr(TTOpConverters::kScalarPointerCarrierBoundaryAttr);
  });

  // Conversion can expose a make_range carrier whose loop result is already
  // dead. Remove only the proven dead range/update chain before generic
  // canonicalization; live tensor carriers and scalar address state remain.
  eraseDeadRangeCarriers(moduleOp);

  // 7.1 Workaround: fold duplicated one-hot reconstruction emitted after
  // ArgMax lowering. The issue is not in triton::ReduceOp semantics themselves;
  // redundant value reconstruction is materialized later and can lower to
  // incorrect code on Ascend, so this is fixed post-conversion on
  // linalg::ReduceOp.
  {
    RewritePatternSet foldPatterns(&getContext());
    TTOpConverters::populatePostConversionCanonicalizationPatterns(
        foldPatterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(foldPatterns)))) {
      moduleOp->emitError("failed to fold one-hot gather after max_with_index");
      signalPassFailure();
      return;
    }
  }

  // Execute legal stride operations conversion
  if (failed(processLegalStrideOperations(moduleOp))) {
    signalPassFailure();
  }

  // 8. Convert function prologue/epilogue.
  moduleOp.walk([&](triton::FuncOp func) {
    this->convertTTFunc(func, existDot, existSIMTOp);
  });

  rewriteDevicePrintOffsets(moduleOp);

  // 9. Clean up dead code and simplify IR.
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }

  specializeFullRowCopies(moduleOp);

  // 10. Collapses call-site locations whose callee is an inlined Triton stdlib
  // helper (under site-packages) down to their caller (user-file) frame
  //     Opt-in via LLVM_EXTRACT_DI_LOCAL_VARIABLES=1.
  {
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(triton::createCanonicalizeDebugLocationsPass());
    if (failed(runPipeline(pm, moduleOp))) {
      moduleOp->emitWarning("CanonicalizeDebugLocationsPass pass failed");
    }
  }

  // 11. Deduplicate debug NOPs inserted by converters.
  //     Opt-in via LLVM_EXTRACT_DI_LOCAL_VARIABLES=1.
  {
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(triton::createDeduplicateDebugNopsPass());
    if (failed(runPipeline(pm, moduleOp))) {
      moduleOp->emitWarning("DeduplicateDebugNops pass failed");
      // Non-fatal: dedup is a quality improvement, not a correctness pass.
    }
  }

  // Calculate size of PointerCastOp precisely
  SmallVector<hivm::PointerCastOp> castOps;

  moduleOp.walk([&](hivm::PointerCastOp op) { castOps.push_back(op); });

  for (auto op : castOps) {
    SmallVector<memref::ReinterpretCastOp> reinterpretCastOps;
    for (Operation *user : op->getUsers()) {
      if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(user))
        reinterpretCastOps.push_back(reinterpretCast);
    }
    if (reinterpretCastOps.empty())
      continue;

    bool isScalarPointerCarrier = op->hasAttr(kScalarPointerCarrierAttr);
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointAfter(op);
    Value addr = op.getAddrs()[0];
    auto elementType =
        cast<MemRefType>(op.getResult().getType()).getElementType();

    for (memref::ReinterpretCastOp reinterpretCastOp : reinterpretCastOps) {
      auto sizes = reinterpretCastOp.getStaticSizes();
      auto staticStrides = reinterpretCastOp.getStaticStrides();
      auto strides = reinterpretCastOp.getStrides();
      if (reinterpretCastOp.getStaticOffsets().size() != 1) {
        reinterpretCastOp->emitError(
            "IntToPtrOp must converted to PointerCastOp of "
            "memref<?xdtype> type");
        signalPassFailure();
        return;
      }
      int64_t castOpSize = 0;
      SmallVector<int64_t> dynamicSizes;
      for (const auto &[size, stride] : llvm::zip_equal(sizes, staticStrides)) {
        assert(!ShapedType::isDynamic(size));
        if (ShapedType::isDynamic(stride))
          dynamicSizes.push_back(size);
        else
          castOpSize = size * stride;
      }
      rewriter.setInsertionPoint(reinterpretCastOp);
      Value dynamicSize = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(castOpSize));
      for (const auto &[size, stride] :
           llvm::zip_equal(dynamicSizes, strides)) {
        Value axisSize = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(size));
        axisSize =
            rewriter.create<arith::MulIOp>(op.getLoc(), stride, axisSize);
        dynamicSize =
            rewriter.create<arith::AddIOp>(op.getLoc(), dynamicSize, axisSize);
      }
      auto staticOffset = reinterpretCastOp.getStaticOffsets()[0];
      auto materializeOffset = [&](Type targetType) -> Value {
        if (ShapedType::isDynamic(staticOffset)) {
          Value offset = reinterpretCastOp.getOffsets()[0];
          if (offset.getType() != targetType)
            offset = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                                         targetType, offset);
          return offset;
        }
        if (targetType.isIndex())
          return rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                         staticOffset);
        return rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIntegerAttr(targetType, staticOffset));
      };

      auto memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
      auto createPointerCast = [&](Value address, Value capacity) {
        auto cast = rewriter.create<hivm::PointerCastOp>(
            op.getLoc(), memrefType, address, capacity);
        auto mark =
            rewriter.create<annotation::MarkOp>(op.getLoc(), cast.getResult());
        mark->setAttr(hivm::AddressSpaceAttr::getMnemonic(),
                      {hivm::AddressSpaceAttr::get(rewriter.getContext(),
                                                   hivm::AddressSpace::GM)});
        return cast;
      };

      if (!isScalarPointerCarrier) {
        // Preserve main-dev behavior for ordinary PointerCast operations. The
        // provenance-gated branches below exist specifically for scalar
        // pointer carriers introduced by this conversion pipeline.
        Value offsetValue = materializeOffset(addr.getType());
        Value elementTypeSize;
        if (auto intType = dyn_cast<IntegerType>(elementType)) {
          elementTypeSize = rewriter.create<arith::ConstantOp>(
              op.getLoc(),
              rewriter.getIntegerAttr(addr.getType(), intType.getWidth() / 8));
        } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
          elementTypeSize = rewriter.create<arith::ConstantOp>(
              op.getLoc(), rewriter.getIntegerAttr(addr.getType(),
                                                   floatType.getWidth() / 8));
        } else {
          llvm_unreachable("Cannot get memory size");
        }
        offsetValue = rewriter.create<arith::MulIOp>(op.getLoc(), offsetValue,
                                                     elementTypeSize);
        Value realAddr =
            rewriter.create<arith::AddIOp>(op.getLoc(), addr, offsetValue);
        auto newCastOp = createPointerCast(realAddr, dynamicSize);

        auto oldResultType =
            cast<MemRefType>(reinterpretCastOp.getResult().getType());
        MemRefType newResultType = oldResultType;
        if (auto stridedLayout =
                dyn_cast<StridedLayoutAttr>(oldResultType.getLayout())) {
          if (!ShapedType::isDynamic(stridedLayout.getOffset())) {
            auto newLayout =
                StridedLayoutAttr::get(rewriter.getContext(), /*offset=*/0,
                                       stridedLayout.getStrides());
            newResultType = MemRefType::get(
                oldResultType.getShape(), oldResultType.getElementType(),
                newLayout, oldResultType.getMemorySpace());
          }
        }
        rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
            reinterpretCastOp, newResultType, newCastOp, ValueRange({}),
            reinterpretCastOp.getSizes(), reinterpretCastOp.getStrides(),
            SmallVector<int64_t>({0}), reinterpretCastOp.getStaticSizes(),
            reinterpretCastOp.getStaticStrides());
        continue;
      }

      if (reachesLayoutSensitiveBoundary(reinterpretCastOp.getResult())) {
        // Keep the original offset and result type at externally typed
        // boundaries. The replacement PointerCast still starts at the old
        // address, so its capacity must include the leading view displacement.
        Value pointerCapacity = dynamicSize;
        Value offsetElements = materializeOffset(pointerCapacity.getType());
        Value leadingExtent = offsetElements;
        if (ShapedType::isDynamic(staticOffset)) {
          Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
          leadingExtent = rewriter.create<arith::MaxSIOp>(op.getLoc(),
                                                          offsetElements, zero);
        }
        if (ShapedType::isDynamic(staticOffset) || staticOffset > 0)
          pointerCapacity = rewriter.create<arith::AddIOp>(
              op.getLoc(), pointerCapacity, leadingExtent);
        auto newCastOp = createPointerCast(addr, pointerCapacity);
        rewriter.modifyOpInPlace(reinterpretCastOp, [&] {
          reinterpretCastOp.getSourceMutable().assign(newCastOp.getResult());
        });
        continue;
      }

      Value offsetValue = materializeOffset(addr.getType());
      Value elementTypeSize;
      if (auto intType = dyn_cast<IntegerType>(elementType)) {
        elementTypeSize = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
            rewriter.getIntegerAttr(addr.getType(), intType.getWidth() / 8));
      } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
        elementTypeSize = rewriter.create<arith::ConstantOp>(
            op.getLoc(),
            rewriter.getIntegerAttr(addr.getType(), floatType.getWidth() / 8));
      } else {
        llvm_unreachable("Cannot get memory size");
      }
      offsetValue = rewriter.create<arith::MulIOp>(op.getLoc(), offsetValue,
                                                   elementTypeSize);
      Value realAddr =
          rewriter.create<arith::AddIOp>(op.getLoc(), addr, offsetValue);
      auto newCastOp = createPointerCast(realAddr, dynamicSize);
      // realAddr already includes the old reinterpret-cast offset in bytes.
      // The replacement view therefore starts at offset zero, and its result
      // type must describe the same rebased layout. Reusing the old type here
      // would combine static_offsets=[0] with (for example) a type-level
      // offset of 1, which is rejected by the ReinterpretCast verifier.
      auto oldResultType =
          cast<MemRefType>(reinterpretCastOp.getResult().getType());
      SmallVector<int64_t> rebasedStrides(
          oldResultType.getStridesAndOffset().first);
      auto rebasedResultType = MemRefType::get(
          oldResultType.getShape(), oldResultType.getElementType(),
          StridedLayoutAttr::get(oldResultType.getContext(), /*offset=*/0,
                                 rebasedStrides),
          oldResultType.getMemorySpace());

      // Keep the old result and replacement types equal while RAUW updates all
      // users to the rebased descriptor type.
      rewriter.modifyOpInPlace(reinterpretCastOp, [&] {
        reinterpretCastOp.getResult().setType(rebasedResultType);
      });
      auto rebasedReinterpretCast =
          rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
              reinterpretCastOp, rebasedResultType, newCastOp, ValueRange({}),
              reinterpretCastOp.getSizes(), reinterpretCastOp.getStrides(),
              SmallVector<int64_t>({0}), reinterpretCastOp.getStaticSizes(),
              reinterpretCastOp.getStaticStrides());
      if (failed(propagateRebasedSubviewTypes(
              rebasedReinterpretCast.getResult(), rewriter))) {
        rebasedReinterpretCast.emitError(
            "failed to propagate rebased layout through subview users");
        signalPassFailure();
        return;
      }
    }
    if (op->use_empty())
      rewriter.eraseOp(op);
  }

  // ScalarPointerCarrier is a pass-local provenance marker. Keep it through
  // PointerCast post-processing so only known scalar-address carriers use the
  // new layout path. A carrier that is consumed directly (without an
  // intermediate reinterpret_cast) is not rebuilt by the loop above, so give
  // that remaining GM pointer the same address-space annotation before the
  // provenance marker is removed.
  moduleOp.walk([](hivm::PointerCastOp pointerCast) {
    if (!pointerCast->hasAttr(kScalarPointerCarrierAttr))
      return;
    OpBuilder builder(pointerCast);
    builder.setInsertionPointAfter(pointerCast);
    auto mark = builder.create<annotation::MarkOp>(pointerCast.getLoc(),
                                                   pointerCast.getResult());
    mark->setAttr(hivm::AddressSpaceAttr::getMnemonic(),
                  {hivm::AddressSpaceAttr::get(pointerCast.getContext(),
                                               hivm::AddressSpace::GM)});
    pointerCast->removeAttr(kScalarPointerCarrierAttr);
  });

  // Try interleave optimization
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>> interleaveCandidate;
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>>
      interleaveCandidateWithMask;
  moduleOp.walk([&](bufferization::MaterializeInDestinationOp materializeOp) {
    if (auto reinterpretCastOp =
            materializeOp.getDest()
                .getDefiningOp<memref::ReinterpretCastOp>()) {
      if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
          reinterpretCastOp.getStaticStrides().back() == 2) {
        interleaveCandidate[llvm::cast<BlockArgument>(
                                reinterpretCastOp.getSource())]
            .push_back(materializeOp);
      }
    }

    // Difference is that converted op chain of store with mask has
    // `memref::SubViewOp`
    if (auto subviewOp =
            materializeOp.getDest().getDefiningOp<memref::SubViewOp>()) {
      if (!llvm::isa<tensor::ExtractSliceOp>(
              materializeOp.getSource().getDefiningOp()))
        return WalkResult::advance();

      if (auto reinterpretCastOp =
              subviewOp.getSource()
                  .getDefiningOp<memref::ReinterpretCastOp>()) {
        if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
            reinterpretCastOp.getStaticStrides().back() == 2) {
          interleaveCandidateWithMask[llvm::cast<BlockArgument>(
                                          reinterpretCastOp.getSource())]
              .push_back(materializeOp);
        }
      }
    }

    return WalkResult::advance();
  });

  for (auto [blockArg, materializeVec] : interleaveCandidate) {
    // Just enable optimization where exists double materializeOp with same
    // block argument destination.
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusOptimization(materializeVec);
  }

  for (auto [blockArg, materializeVec] : interleaveCandidateWithMask) {
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusWithMaskOptimization(materializeVec);
  }

  // Force to add an argument at the beginning of function arguments, which
  // represents stub arg for workspace. Default type is memref<?xi8>
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    if (!func->hasAttr("global_kernel"))
      continue;

    auto context = func.getContext();
    constexpr int64_t syncBlockLockArgIdx = 0;
    NamedAttribute syncBlockLockArgAttr(
        StringAttr::get(context, "syncBlockLock"), UnitAttr::get(context));
    MemRefType syncBlockLockArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    llvm::LogicalResult syncBlockLockArg =
        func.insertArgument(syncBlockLockArgIdx,      // argIndex
                            syncBlockLockArgType,     // argType
                            nullptr, func->getLoc()); // dicAttr
    func->setAttr("SyncBlockLockArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   0)); // 64: 64位整型

    constexpr int64_t workspaceArgIdx = 1;
    MemRefType workspaceArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                    UnitAttr::get(context));

    llvm::LogicalResult workspaceArg =
        func.insertArgument(/*argIndex*/ workspaceArgIdx,
                            /*argType*/ workspaceArgType,
                            /*dicAttr*/ nullptr, func->getLoc());
    func->setAttr("WorkspaceArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   1)); // 64: 64位整型
  }

  // Fix the Location info
  moduleOp.walk([&](Operation *op) {
    auto loc = op->getLoc();
    if (isa<UnknownLoc>(loc)) {
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [](Operation *curOp) { return false; },
          /*stopFn*/
          [](Operation *curOp) { return !isa<UnknownLoc>(curOp->getLoc()); },
          /*actionFn*/
          nullptr, stopOps);
      if (stopOps.empty()) {
        op->emitWarning() << *op << " and its users all have no location!";
      } else {
        Operation *goodOp = *stopOps.begin();
        op->setLoc(goodOp->getLoc());
      }
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgPass(bool globalKernel, bool namedOps,
                                 bool enableNd2nzOnVector,
                                 bool enableSelectAnalysis, bool compileOn91095,
                                 const std::string &compileMode) {
  return std::make_unique<TritonToLinalgPass>(
      globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
      compileOn91095, compileMode);
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
