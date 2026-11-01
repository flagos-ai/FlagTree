#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <memory>

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPVECTORIZE
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-vectorize"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

constexpr StringLiteral kVectorFactorAttr = "tt.loop_vectorize_factor";

static RankedTensorType prependIterationDim(Type type, int64_t vectorFactor) {
  if (auto tensor = dyn_cast<RankedTensorType>(type)) {
    SmallVector<int64_t> shape(tensor.getShape());
    shape.insert(shape.begin(), vectorFactor);
    return RankedTensorType::get(shape, tensor.getElementType());
  }
  return RankedTensorType::get({vectorFactor}, type);
}

struct OperandVectorizationPlan {
  Value source;
  RankedTensorType targetType;
  bool broadcastScalar = false;
};

struct IfVectorizationPlan;

struct OperationVectorizationPlan {
  Operation *op;
  SmallVector<OperandVectorizationPlan> operands;
  SmallVector<Type> resultTypes;
  int32_t axis = 0;
  SmallVector<int32_t> order;
  std::unique_ptr<IfVectorizationPlan> ifPlan;
};

struct BlockVectorizationPlan {
  SmallVector<OperationVectorizationPlan> operations;
  SmallVector<OperandVectorizationPlan> yieldOperands;
};

struct IfVectorizationPlan {
  Value condition;
  BlockVectorizationPlan thenPlan;
  BlockVectorizationPlan elsePlan;
};

enum class LoopCarriedValueKind { Sum, LastValue };

struct LoopCarriedValuePlan {
  LoopCarriedValueKind kind;
  BlockArgument argument;
  Value initial;
  Operation *update = nullptr;
};

struct LoopVectorizationPlan {
  bool singleGroup = false;
  llvm::SetVector<Operation *> conditionDependencies;
  SmallVector<LoopCarriedValuePlan> carriedValues;
  BlockVectorizationPlan body;
};

class LoopVectorizer {
private:
  scf::ForOp loop;
  int64_t vectorFactor;
  OpBuilder builder;
  IRMapping conditionMapping;
  StringRef failureReason;
  Operation *failureOp = nullptr;

public:
  LoopVectorizer(scf::ForOp loop, int64_t vectorFactor)
      : loop(loop), vectorFactor(vectorFactor), builder(loop) {}

  LogicalResult run() {
    auto plan = buildPlan();
    if (failed(plan)) {
      if (!failureReason.empty()) {
        auto diagnostic = loop.emitWarning("loop vectorization with factor ");
        diagnostic << vectorFactor << " was not applied: " << failureReason;
        if (failureOp)
          diagnostic << " (operation '" << failureOp->getName() << "')";
        diagnostic << "; keeping the original loop";
      }
      return failure();
    }

    LDBG("Built vectorization plan with factor " << vectorFactor << " for "
                                                 << loop.getLoc());
    // Keep the original loop until the whole rewrite succeeds. Both paths
    // insert before it, so rollback also removes any strip-mining bounds.
    Operation *previous = loop->getPrevNode();
    cloneConditionDependencies(plan->conditionDependencies);
    auto result =
        plan->singleGroup ? emitSingleGroup(*plan) : emitStripMinedLoop(*plan);
    if (failed(result)) {
      eraseGeneratedOperations(previous);
      loop.emitWarning() << "loop vectorization with factor " << vectorFactor
                         << " was not applied: plan execution failed; keeping "
                            "the original loop";
      return failure();
    }

    LDBG(
        "Applied vectorization with factor "
        << vectorFactor << " using "
        << (plan->singleGroup ? "direct replacement" : "main loop and epilogue")
        << "; processed " << plan->body.operations.size() << " operations at "
        << loop.getLoc());
    loop->replaceAllUsesWith(*result);
    loop.erase();
    return success();
  }

private:
  LogicalResult reject(StringRef reason, Operation *op = nullptr) {
    failureReason = reason;
    failureOp = op;
    LDBG("Cannot vectorize loop with factor " << vectorFactor << ": "
                                              << reason);
    if (op)
      LDBG("Unsupported operation '" << op->getName() << "' at "
                                     << op->getLoc());
    return failure();
  }

  static bool isLiftableType(Type type) {
    if (auto tensor = dyn_cast<RankedTensorType>(type))
      type = tensor.getElementType();
    if (auto ptr = dyn_cast<PointerType>(type))
      return isa<IntegerType, FloatType>(ptr.getPointeeType());
    return isa<IntegerType, FloatType>(type);
  }

  LogicalResult planConditionValue(Value value,
                                   llvm::SetVector<Operation *> &operations) {
    if (!loop.getRegion().isAncestor(value.getParentRegion()))
      return success();
    Operation *op = value.getDefiningOp();
    if (!op)
      return reject(
          "if condition depends on a loop induction or carried value");
    if (operations.contains(op))
      return success();
    if (!isMemoryEffectFree(op) || !isSpeculatable(op))
      return reject("if condition cannot be safely computed outside the loop",
                    op);
    if (!llvm::all_of(op->getOperandTypes(), isLiftableType) ||
        !llvm::all_of(op->getResultTypes(), isLiftableType))
      return reject("unsupported if condition dependency type", op);

    llvm::SetVector<Value> dependencies;
    dependencies.insert(op->operand_begin(), op->operand_end());
    getUsedValuesDefinedAbove(op->getRegions(), dependencies);
    for (Value dependency : dependencies)
      if (failed(planConditionValue(dependency, operations)))
        return failure();
    operations.insert(op);
    return success();
  }

  LogicalResult planIfConditions(Block &block,
                                 llvm::SetVector<Operation *> &operations) {
    for (Operation &op : block.without_terminator()) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp)
        continue;
      if (failed(planConditionValue(ifOp.getCondition(), operations)) ||
          failed(planIfConditions(*ifOp.thenBlock(), operations)) ||
          (ifOp.elseBlock() &&
           failed(planIfConditions(*ifOp.elseBlock(), operations))))
        return failure();
    }
    return success();
  }

  LogicalResult validateCombiner(Region &combine) {
    for (Operation &op : combine.front()) {
      if (op.getNumRegions() != 0 || !isMemoryEffectFree(&op))
        return reject("combiners must be region-free and memory-effect-free",
                      &op);
    }
    return success();
  }

  LogicalResult validateOperation(Operation *op) {
    if (!llvm::all_of(op->getOperandTypes(), isLiftableType) ||
        !llvm::all_of(op->getResultTypes(), isLiftableType))
      return reject("unsupported operand or result type", op);
    if (auto reduce = dyn_cast<ReduceOp>(op))
      return validateCombiner(reduce.getCombineOp());
    if (auto scan = dyn_cast<ScanOp>(op))
      return validateCombiner(scan.getCombineOp());
    if (isa<scf::IfOp>(op)) {
      return success();
    }
    if (op->getNumRegions() != 0)
      return reject("nested regions are not supported", op);
    if (isa<LoadOp, StoreOp, arith::ConstantOp, MakeRangeOp, SplatOp,
            ExpandDimsOp, BroadcastOp, ReshapeOp, TransOp, JoinOp, SplitOp,
            UnsplatOp, GatherOp, GetProgramIdOp, GetNumProgramsOp>(op))
      return success();
    if (!op->hasTrait<OpTrait::Elementwise>() || !isMemoryEffectFree(op))
      return reject("operation has no supported tensorization", op);
    return success();
  }

  LogicalResult planLoopCarriedValues(LoopVectorizationPlan &plan) {
    auto yield = cast<scf::YieldOp>(loop.getBody()->getTerminator());
    for (auto [argument, initial, yielded] :
         llvm::zip(loop.getRegionIterArgs(), loop.getInitArgs(),
                   yield.getOperands())) {
      Type type = argument.getType();
      if (!isLiftableType(type))
        return reject("unsupported loop-carried value type");
      if (argument.use_empty()) {
        plan.carriedValues.push_back(
            {LoopCarriedValueKind::LastValue, argument, initial});
        continue;
      }
      Type elementType = type;
      if (auto tensor = dyn_cast<RankedTensorType>(type))
        elementType = tensor.getElementType();
      if (!isa<IntegerType, FloatType>(elementType))
        return reject(
            "loop-carried values must have integer or float element types");

      Operation *update = yielded.getDefiningOp();
      if (!update || !isa<arith::AddFOp, arith::AddIOp>(update) ||
          update->getBlock() != loop.getBody() ||
          (update->getOperand(0) != argument &&
           update->getOperand(1) != argument))
        return reject(
            "loop-carried values must have a single addf or addi update");
      if (!argument.hasOneUse())
        return reject("loop sum accumulator is used outside its update",
                      update);
      if (!yielded.hasOneUse())
        return reject("loop sum result is used outside its yield", update);
      plan.carriedValues.push_back(
          {LoopCarriedValueKind::Sum, argument, initial, update});
    }
    return success();
  }

  LogicalResult planOperands(OperationVectorizationPlan &plan,
                             const DenseMap<Value, Type> &plannedTypes) {
    for (Value operand : plan.op->getOperands()) {
      Type sourceType = plannedTypes.lookup(operand);
      if (!sourceType) {
        if (loop.getRegion().isAncestor(operand.getParentRegion()))
          return reject("operand has no planned definition", plan.op);
        sourceType = operand.getType();
      }

      if (auto splat = dyn_cast<SplatOp>(plan.op)) {
        RankedTensorType target =
            prependIterationDim(splat.getResult().getType(), vectorFactor);
        Type scalarType = target.getElementType();
        Type iterType = prependIterationDim(scalarType, vectorFactor);
        if (sourceType != scalarType && sourceType != iterType)
          return reject("scalar operand cannot broadcast to the planned shape",
                        plan.op);
        plan.operands.push_back({operand, target, true});
        continue;
      }

      RankedTensorType target =
          prependIterationDim(operand.getType(), vectorFactor);
      if (plannedTypes.contains(operand) && sourceType != target) {
        return reject("operand type does not match the planned shape", plan.op);
      }
      plan.operands.push_back({operand, target, false});
    }
    return success();
  }

  LogicalResult
  planBlock(Block &block, DenseMap<Value, Type> plannedTypes,
            BlockVectorizationPlan &plan,
            const llvm::SetVector<Operation *> &conditionDependencies) {
    for (Operation &op : block.without_terminator()) {
      if (conditionDependencies.contains(&op)) {
        for (Value result : op.getResults())
          plannedTypes[result] =
              prependIterationDim(result.getType(), vectorFactor);
        continue;
      }
      if (failed(validateOperation(&op)))
        return failure();
      OperationVectorizationPlan operation;
      operation.op = &op;
      for (Type type : op.getResultTypes())
        operation.resultTypes.push_back(
            prependIterationDim(type, vectorFactor));
      if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        operation.ifPlan = std::make_unique<IfVectorizationPlan>();
        auto &ifPlan = *operation.ifPlan;
        ifPlan.condition = ifOp.getCondition();
        if (failed(planBlock(*ifOp.thenBlock(), plannedTypes, ifPlan.thenPlan,
                             conditionDependencies)))
          return failure();
        if (ifOp.elseBlock() &&
            failed(planBlock(*ifOp.elseBlock(), plannedTypes, ifPlan.elsePlan,
                             conditionDependencies)))
          return failure();
      } else {
        if (auto expand = dyn_cast<ExpandDimsOp>(op)) {
          operation.axis = expand.getAxis() + 1;
        } else if (auto reduce = dyn_cast<ReduceOp>(op)) {
          operation.axis = reduce.getAxis() + 1;
        } else if (auto scan = dyn_cast<ScanOp>(op)) {
          operation.axis = scan.getAxis() + 1;
        } else if (auto gather = dyn_cast<GatherOp>(op)) {
          operation.axis = gather.getAxis() + 1;
        } else if (auto trans = dyn_cast<TransOp>(op)) {
          operation.order.push_back(0);
          for (int32_t dim : trans.getOrder())
            operation.order.push_back(dim + 1);
        }
        if (failed(planOperands(operation, plannedTypes)))
          return failure();
      }
      for (auto [result, type] :
           llvm::zip(op.getResults(), operation.resultTypes))
        plannedTypes[result] = type;
      plan.operations.push_back(std::move(operation));
    }
    OperationVectorizationPlan terminator;
    terminator.op = block.getTerminator();
    if (failed(planOperands(terminator, plannedTypes)))
      return failure();
    plan.yieldOperands = std::move(terminator.operands);
    return success();
  }

  FailureOr<LoopVectorizationPlan> buildPlan() {
    if (vectorFactor <= 1) {
      LDBG("Loop vectorization is disabled at " << loop.getLoc());
      return failure();
    }
    if (!llvm::isPowerOf2_64(vectorFactor))
      return reject("factor must be a power of two");
    LoopVectorizationPlan plan;
    if (auto tripCount = loop.getStaticTripCount()) {
      if (tripCount->isZero()) {
        LDBG("Skipping an empty loop at " << loop.getLoc());
        return failure();
      }
      if (tripCount->ult(vectorFactor))
        return reject("iteration count is smaller than vectorization factor");
      plan.singleGroup = *tripCount == vectorFactor;
    }
    if (failed(planIfConditions(*loop.getBody(), plan.conditionDependencies)) ||
        failed(planLoopCarriedValues(plan)))
      return failure();

    DenseMap<Value, Type> plannedTypes;
    plannedTypes[loop.getInductionVar()] =
        prependIterationDim(loop.getInductionVar().getType(), vectorFactor);
    for (const auto &carriedValue : plan.carriedValues)
      plannedTypes[carriedValue.argument] =
          prependIterationDim(carriedValue.argument.getType(), vectorFactor);
    if (failed(planBlock(*loop.getBody(), std::move(plannedTypes), plan.body,
                         plan.conditionDependencies)))
      return failure();
    return plan;
  }

  void cloneConditionDependencies(
      const llvm::SetVector<Operation *> &conditionDependencies) {
    for (Operation *op : conditionDependencies) {
      bool nested = false;
      for (Operation *parent = op->getParentOp(); parent != loop;
           parent = parent->getParentOp()) {
        if (conditionDependencies.contains(parent)) {
          nested = true;
          break;
        }
      }
      if (!nested)
        builder.clone(*op, conditionMapping);
    }
  }

  void eraseGeneratedOperations(Operation *previous) {
    Operation *current =
        previous ? previous->getNextNode() : &loop->getBlock()->front();
    SmallVector<Operation *> created;
    while (current != loop) {
      created.push_back(current);
      current = current->getNextNode();
    }
    for (Operation *op : llvm::reverse(created))
      op->erase();
  }

  Value broadcastScalar(Value value, RankedTensorType target) {
    if (!isa<RankedTensorType>(value.getType()))
      return SplatOp::create(builder, loop.getLoc(), target, value);
    while (cast<RankedTensorType>(value.getType()).getRank() <
           target.getRank()) {
      auto type = cast<RankedTensorType>(value.getType());
      SmallVector<int64_t> shape(type.getShape());
      shape.push_back(1);
      auto expanded = RankedTensorType::get(shape, type.getElementType());
      value = ExpandDimsOp::create(builder, loop.getLoc(), expanded, value,
                                   type.getRank());
    }
    if (value.getType() == target)
      return value;
    return BroadcastOp::create(builder, loop.getLoc(), target, value);
  }

  Value broadcastAcrossIterations(Value value, RankedTensorType target) {
    if (!isa<RankedTensorType>(value.getType()))
      return SplatOp::create(builder, loop.getLoc(), target, value);
    auto source = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape(source.getShape());
    shape.insert(shape.begin(), 1);
    auto expandedType = RankedTensorType::get(shape, source.getElementType());
    Value expanded =
        ExpandDimsOp::create(builder, loop.getLoc(), expandedType, value, 0);
    return BroadcastOp::create(builder, loop.getLoc(), target, expanded);
  }

  FailureOr<Value> emitOperand(const OperandVectorizationPlan &operand,
                               const IRMapping &mapping) {
    Value value = mapping.lookupOrNull(operand.source);
    Value source = conditionMapping.lookupOrDefault(operand.source);
    if (operand.broadcastScalar)
      value = broadcastScalar(value ? value : source, operand.targetType);
    else if (!value)
      value = broadcastAcrossIterations(source, operand.targetType);
    if (value.getType() != operand.targetType)
      return failure();
    return value;
  }

  FailureOr<SmallVector<Value>>
  emitBlockContents(const BlockVectorizationPlan &plan, IRMapping &mapping) {
    for (const auto &operation : plan.operations)
      if (failed(emitOperation(operation, mapping)))
        return failure();
    SmallVector<Value> values;
    for (const auto &operand : plan.yieldOperands) {
      auto value = emitOperand(operand, mapping);
      if (failed(value))
        return failure();
      values.push_back(*value);
    }
    return values;
  }

  LogicalResult emitBlock(const BlockVectorizationPlan &plan, Block &block,
                          IRMapping mapping) {
    OpBuilder::InsertionGuard guard(builder);
    if (!block.empty())
      block.getTerminator()->erase();
    builder.setInsertionPointToStart(&block);
    auto values = emitBlockContents(plan, mapping);
    if (failed(values))
      return failure();
    scf::YieldOp::create(builder, loop.getLoc(), *values);
    return success();
  }

  LogicalResult emitOperation(const OperationVectorizationPlan &plan,
                              IRMapping &mapping) {
    if (auto original = dyn_cast<scf::IfOp>(plan.op)) {
      const auto &ifPlan = *plan.ifPlan;
      auto vectorizedIf =
          scf::IfOp::create(builder, original.getLoc(), plan.resultTypes,
                            conditionMapping.lookupOrDefault(ifPlan.condition),
                            original.elseBlock() != nullptr);
      vectorizedIf->setAttrs(original->getAttrs());
      if (failed(
              emitBlock(ifPlan.thenPlan, *vectorizedIf.thenBlock(), mapping)) ||
          (vectorizedIf.elseBlock() &&
           failed(
               emitBlock(ifPlan.elsePlan, *vectorizedIf.elseBlock(), mapping))))
        return failure();
      mapping.map(original.getResults(), vectorizedIf.getResults());
      return success();
    }
    IRMapping operandMapping;
    SmallVector<Value> inputs;
    for (const auto &operand : plan.operands) {
      auto value = emitOperand(operand, mapping);
      if (failed(value))
        return failure();
      operandMapping.map(operand.source, *value);
      inputs.push_back(*value);
    }

    SmallVector<Value, 1> results;
    if (isa<arith::ConstantOp, MakeRangeOp, GetProgramIdOp, GetNumProgramsOp>(
            plan.op)) {
      Operation *copy = builder.clone(*plan.op);
      results.push_back(broadcastAcrossIterations(
          copy->getResult(0),
          cast<RankedTensorType>(plan.resultTypes.front())));
    } else if (isa<SplatOp>(plan.op)) {
      results.push_back(inputs.front());
    } else if (isa<UnsplatOp>(plan.op)) {
      auto type = cast<RankedTensorType>(plan.resultTypes.front());
      results.push_back(ReshapeOp::create(builder, plan.op->getLoc(),
                                          type.getShape(), inputs.front()));
    } else if (isa<ExpandDimsOp>(plan.op)) {
      results.push_back(ExpandDimsOp::create(builder, plan.op->getLoc(),
                                             plan.resultTypes.front(),
                                             inputs.front(), plan.axis));
    } else if (auto reduce = dyn_cast<ReduceOp>(plan.op)) {
      auto vectorizedReduce =
          ReduceOp::create(builder, plan.op->getLoc(), inputs, plan.axis);
      builder.cloneRegionBefore(reduce.getCombineOp(),
                                vectorizedReduce.getCombineOp(),
                                vectorizedReduce.getCombineOp().end());
      llvm::append_range(results, vectorizedReduce.getResult());
    } else if (auto scan = dyn_cast<ScanOp>(plan.op)) {
      auto vectorizedScan = ScanOp::create(builder, plan.op->getLoc(), inputs,
                                           plan.axis, scan.getReverse());
      builder.cloneRegionBefore(scan.getCombineOp(),
                                vectorizedScan.getCombineOp(),
                                vectorizedScan.getCombineOp().end());
      llvm::append_range(results, vectorizedScan.getResult());
    } else {
      Operation *copy = builder.clone(*plan.op, operandMapping);
      for (auto [newResult, type] :
           llvm::zip(copy->getResults(), plan.resultTypes))
        newResult.setType(type);
      if (auto trans = dyn_cast<TransOp>(copy))
        trans.setOrder(plan.order);
      else if (auto gather = dyn_cast<GatherOp>(copy))
        gather.setAxis(plan.axis);
      llvm::append_range(results, copy->getResults());
    }
    if (results.size() != plan.resultTypes.size())
      return failure();
    for (auto [result, type] : llvm::zip(results, plan.resultTypes)) {
      if (!result || result.getType() != type)
        return failure();
    }
    mapping.map(plan.op->getResults(), results);
    return success();
  }

  SmallVector<Value> emitLoopInitializers(const LoopVectorizationPlan &plan) {
    SmallVector<Value> initializers;
    Location loc = loop.getLoc();
    Value firstIterMask;
    for (const auto &carriedValue : plan.carriedValues) {
      auto type =
          prependIterationDim(carriedValue.argument.getType(), vectorFactor);
      if (carriedValue.kind == LoopCarriedValueKind::LastValue) {
        initializers.push_back(
            broadcastAcrossIterations(carriedValue.initial, type));
        continue;
      }
      if (!firstIterMask) {
        auto indexType =
            RankedTensorType::get({vectorFactor}, builder.getI32Type());
        Value indices = MakeRangeOp::create(builder, loc, indexType, 0,
                                            static_cast<int32_t>(vectorFactor));
        Value zero = arith::ConstantOp::create(
            builder, loc,
            DenseElementsAttr::get(indexType, builder.getI32IntegerAttr(0)));
        firstIterMask = arith::CmpIOp::create(
            builder, loc, arith::CmpIPredicate::eq, indices, zero);
      }
      Attribute identityAttr = builder.getZeroAttr(type.getElementType());
      // Count the original initial value exactly once. Floating-point sums use
      // negative zero to preserve negative-zero partial sums; integers use
      // zero.
      if (auto elementType = dyn_cast<FloatType>(type.getElementType()))
        identityAttr = builder.getFloatAttr(
            elementType,
            APFloat::getZero(elementType.getFloatSemantics(), true));
      Value identity = arith::ConstantOp::create(
          builder, loc, DenseElementsAttr::get(type, identityAttr));
      Value initial = broadcastAcrossIterations(carriedValue.initial, type);
      Value mask =
          broadcastScalar(firstIterMask, type.clone(builder.getI1Type()));
      initializers.push_back(
          arith::SelectOp::create(builder, loc, mask, initial, identity));
    }
    return initializers;
  }

  Value emitLastValue(Value value, Type originalType) {
    Location loc = loop.getLoc();
    auto type = cast<RankedTensorType>(value.getType());
    SmallVector<int64_t> shape(type.getShape());
    shape.front() = 1;
    auto indexType = RankedTensorType::get(shape, builder.getI32Type());
    Value indices = arith::ConstantOp::create(
        builder, loc,
        DenseElementsAttr::get(indexType,
                               builder.getI32IntegerAttr(vectorFactor - 1)));
    Value last = GatherOp::create(builder, loc, value, indices, 0);
    if (auto tensor = dyn_cast<RankedTensorType>(originalType))
      return ReshapeOp::create(builder, loc, tensor.getShape(), last);
    return UnsplatOp::create(builder, loc, originalType, last);
  }

  SmallVector<Value> emitLoopResults(const LoopVectorizationPlan &plan,
                                     ValueRange values) {
    SmallVector<Value> results;
    for (auto [carriedValue, value] : llvm::zip(plan.carriedValues, values)) {
      if (carriedValue.kind == LoopCarriedValueKind::LastValue) {
        results.push_back(
            emitLastValue(value, carriedValue.argument.getType()));
        continue;
      }
      auto update = carriedValue.update;
      Location loc = update->getLoc();
      auto reduce = ReduceOp::create(builder, loc, ValueRange{value}, 0);
      auto elementType =
          cast<RankedTensorType>(value.getType()).getElementType();
      {
        OpBuilder::InsertionGuard guard(builder);
        Block *combine = builder.createBlock(
            &reduce.getCombineOp(), {}, {elementType, elementType}, {loc, loc});
        Value sum;
        if (auto updateF = dyn_cast<arith::AddFOp>(update)) {
          auto add = arith::AddFOp::create(
              builder, loc, combine->getArgument(0), combine->getArgument(1));
          add.setFastmathAttr(updateF.getFastmathAttr());
          sum = add;
        } else {
          sum = arith::AddIOp::create(builder, loc, combine->getArgument(0),
                                      combine->getArgument(1));
        }
        ReduceReturnOp::create(builder, loc, ValueRange{sum});
      }
      Value result = reduce.getResult().front();
      if (result.getType() != carriedValue.argument.getType())
        result = SplatOp::create(builder, loc, carriedValue.argument.getType(),
                                 result);
      results.push_back(result);
    }
    return results;
  }

  FailureOr<SmallVector<Value>> emitBody(const LoopVectorizationPlan &plan,
                                         Value base, ValueRange carriedValues) {
    IRMapping mapping;
    for (auto [carriedValue, value] :
         llvm::zip(plan.carriedValues, carriedValues))
      mapping.map(carriedValue.argument, value);
    auto ivType = cast<IntegerType>(loop.getInductionVar().getType());
    auto type = RankedTensorType::get({vectorFactor}, ivType);
    auto indexType =
        RankedTensorType::get({vectorFactor}, builder.getI32Type());
    Value indices = MakeRangeOp::create(builder, loop.getLoc(), indexType, 0,
                                        static_cast<int32_t>(vectorFactor));
    if (ivType.getWidth() > 32)
      indices = arith::ExtSIOp::create(builder, loop.getLoc(), type, indices);
    else if (ivType.getWidth() < 32)
      indices = arith::TruncIOp::create(builder, loop.getLoc(), type, indices);
    Value step = SplatOp::create(builder, loop.getLoc(), type, loop.getStep());
    Value lower = SplatOp::create(builder, loop.getLoc(), type, base);
    Value scaled =
        arith::MulIOp::create(builder, loop.getLoc(), type, indices, step);
    Value vectorIV =
        arith::AddIOp::create(builder, loop.getLoc(), type, lower, scaled);
    mapping.map(loop.getInductionVar(), vectorIV);
    return emitBlockContents(plan.body, mapping);
  }

  FailureOr<SmallVector<Value>>
  emitSingleGroup(const LoopVectorizationPlan &plan) {
    auto initializers = emitLoopInitializers(plan);
    auto results = emitBody(plan, loop.getLowerBound(), initializers);
    if (failed(results))
      return failure();
    return emitLoopResults(plan, *results);
  }

  FailureOr<SmallVector<Value>>
  emitStripMinedLoop(const LoopVectorizationPlan &plan) {
    // Follow LoopUnroll's bounds computation in the original IV type, with
    // ceilDivPositive(distance, step) = (distance + (step - 1)) / step.
    // Keep empty loops empty even when their upper bound is below the lower.
    Location loc = loop.getLoc();
    auto ivType = cast<IntegerType>(loop.getInductionVar().getType());
    Value lower = loop.getLowerBound();
    Value upper = loop.getUpperBound();
    Value step = loop.getStep();
    Value zero =
        arith::ConstantIntOp::create(builder, loc, 0, ivType.getWidth());
    Value one =
        arith::ConstantIntOp::create(builder, loc, 1, ivType.getWidth());
    Value factor = arith::ConstantIntOp::create(builder, loc, vectorFactor,
                                                ivType.getWidth());
    Value distance = arith::SubIOp::create(builder, loc, upper, lower);
    Value nonempty = arith::CmpIOp::create(
        builder, loc, arith::CmpIPredicate::slt, lower, upper);
    Value safeDistance =
        arith::SelectOp::create(builder, loc, nonempty, distance, zero);
    Value stepMinusOne = arith::SubIOp::create(builder, loc, step, one);
    Value numerator =
        arith::AddIOp::create(builder, loc, safeDistance, stepMinusOne);
    Value tripCount = arith::DivUIOp::create(builder, loc, numerator, step);
    Value remainder = arith::RemSIOp::create(builder, loc, tripCount, factor);
    Value vectorizedTripCount =
        arith::SubIOp::create(builder, loc, tripCount, remainder);
    Value vectorDistance =
        arith::MulIOp::create(builder, loc, vectorizedTripCount, step);
    Value vectorEnd =
        arith::AddIOp::create(builder, loc, lower, vectorDistance);
    Value vectorStep = arith::MulIOp::create(builder, loc, step, factor);

    auto initializers = emitLoopInitializers(plan);
    auto vectorLoop = scf::ForOp::create(builder, loc, loop.getLowerBound(),
                                         vectorEnd, vectorStep, initializers);
    vectorLoop->setAttrs(loop->getAttrs());
    vectorLoop->removeAttr(kVectorFactorAttr);
    if (!vectorLoop.getBody()->empty())
      vectorLoop.getBody()->getTerminator()->erase();
    builder.setInsertionPointToStart(vectorLoop.getBody());
    auto results = emitBody(plan, vectorLoop.getInductionVar(),
                            vectorLoop.getRegionIterArgs());
    if (failed(results))
      return failure();
    scf::YieldOp::create(builder, loc, *results);

    builder.setInsertionPointAfter(vectorLoop);
    SmallVector<Value> epilogueInitializers;
    if (!plan.carriedValues.empty()) {
      Value hasFullGroup = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ugt, vectorizedTripCount, zero);
      auto resultIf = scf::IfOp::create(builder, loc, loop.getResultTypes(),
                                        hasFullGroup, true);
      builder.setInsertionPointToStart(resultIf.thenBlock());
      auto results = emitLoopResults(plan, vectorLoop.getResults());
      scf::YieldOp::create(builder, loc, results);
      builder.setInsertionPointToStart(resultIf.elseBlock());
      scf::YieldOp::create(builder, loc, loop.getInitArgs());
      builder.setInsertionPointAfter(resultIf);
      llvm::append_range(epilogueInitializers, resultIf.getResults());
    }
    auto epilogueLoop =
        scf::ForOp::create(builder, loc, vectorEnd, loop.getUpperBound(),
                           loop.getStep(), epilogueInitializers);
    epilogueLoop->setAttrs(loop->getAttrs());
    epilogueLoop->removeAttr(kVectorFactorAttr);
    epilogueLoop->setAttr("tt.num_stages", builder.getI32IntegerAttr(1));
    if (!epilogueLoop.getBody()->empty())
      epilogueLoop.getBody()->getTerminator()->erase();
    builder.setInsertionPointToStart(epilogueLoop.getBody());
    IRMapping epilogueMapping;
    epilogueMapping.map(loop.getInductionVar(), epilogueLoop.getInductionVar());
    epilogueMapping.map(loop.getRegionIterArgs(),
                        epilogueLoop.getRegionIterArgs());
    for (Operation &op : *loop.getBody())
      builder.clone(op, epilogueMapping);
    return SmallVector<Value>(epilogueLoop.getResults());
  }
};

class LoopVectorizePass
    : public impl::TritonLoopVectorizeBase<LoopVectorizePass> {
public:
  void runOnOperation() override {
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp loop) {
      if (loop->hasAttr(kVectorFactorAttr))
        loops.push_back(loop);
    });

    for (auto loop : loops) {
      if (auto factor = loop->getAttrOfType<IntegerAttr>(kVectorFactorAttr)) {
        if (failed(LoopVectorizer(loop, factor.getInt()).run()))
          continue;
      }
    }
  }
};

} // namespace
} // namespace mlir::triton
