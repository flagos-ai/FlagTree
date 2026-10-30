#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#ifdef __ILUVATAR_TLE__
#include "IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/PriorityWorklist.h"
#endif
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Tools/LayoutUtils.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTTRITONTOTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

static LogicalResult
validateIluvatarTleSetLayout(iluvatar_tle::SetLayoutOp op) {
  // Structural layout restrictions belong to the op verifier. This pass adds
  // the launch-context checks once the module attributes have been populated.
  if (failed(op.verify()))
    return failure();
  Attribute encoding = op.getTargetEncoding();
  Attribute parent = encoding;
  while (auto slice = dyn_cast<SliceEncodingAttr>(parent))
    parent = slice.getParent();
  auto blocked = cast<BlockedEncodingAttr>(parent);

  auto mod = op->getParentOfType<ModuleOp>();
  unsigned threadsPerWarp = product<unsigned>(blocked.getThreadsPerWarp());
  unsigned requiredThreadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
  if (threadsPerWarp != requiredThreadsPerWarp)
    return op.emitOpError("target encoding has ")
           << threadsPerWarp << " threads per warp, but the target requires "
           << requiredThreadsPerWarp;
  unsigned numWarps = product<unsigned>(blocked.getWarpsPerCTA());
  if (numWarps != static_cast<unsigned>(lookupNumWarps(op)))
    return op.emitOpError("target encoding has ")
           << numWarps << " warps per CTA, but the context requires "
           << lookupNumWarps(op);
  if (getNumCTAs(encoding) != 1 || TritonGPUDialect::getNumCTAs(mod) != 1)
    return op.emitOpError("Iluvatar set_layout requires num_ctas = 1");
  return success();
}

struct TleEncodingInfo {
  Attribute encoding;
  bool mayVary = false;

  explicit operator bool() const { return bool(encoding); }
};

static bool tleEncodingsMayVary(Operation *op) {
  return isa<triton::JoinOp, triton::SplitOp, triton::ReshapeOp, triton::CatOp,
             triton::TransOp>(op);
}

static LogicalResult mergeTleEncodingInfo(TleEncodingInfo oldInfo,
                                          TleEncodingInfo newInfo,
                                          Operation *op,
                                          TleEncodingInfo &merged) {
  if (!oldInfo) {
    merged = newInfo;
    return success();
  }
  if (!newInfo) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.encoding == newInfo.encoding) {
    merged = oldInfo;
    merged.mayVary = oldInfo.mayVary && newInfo.mayVary;
    return success();
  }
  if (oldInfo.mayVary && !newInfo.mayVary) {
    merged = newInfo;
    return success();
  }
  if (!oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }
  if (oldInfo.mayVary && newInfo.mayVary) {
    merged = oldInfo;
    return success();
  }

  op->emitOpError("found conflicting TLE encoding hints for value:\n  ")
      << oldInfo.encoding << "\nand\n  " << newInfo.encoding;
  return failure();
}

struct TleLayoutDomainMember {
  Attribute encoding;
  Operation *op;
};

struct TleLayoutProducerFamily {
  SmallVector<Value> originalOperands;
  SmallVector<TleLayoutDomainMember> members;
};

using TleLayoutProducerFamilies =
    llvm::MapVector<Operation *, TleLayoutProducerFamily>;

static bool isTleLayoutPolymorphicProducer(Operation *op) {
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0 ||
      op->getNumResults() != 1 || !isPure(op) ||
      op->hasTrait<OpTrait::IsTerminator>() ||
      isa<iluvatar_tle::SetLayoutOp, triton::CallOp>(op) || getMemAccessPtr(op))
    return false;

  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  return resultType && !resultType.getEncoding();
}

static Value materializeTleLayoutDomain(Value value, Attribute encoding,
                                        TleLayoutProducerFamilies &families) {
  Operation *producer = value.getDefiningOp();
  if (!producer || !isTleLayoutPolymorphicProducer(producer))
    return value;

  auto [it, inserted] = families.insert(
      {producer,
       TleLayoutProducerFamily{llvm::to_vector(producer->getOperands()), {}}});
  TleLayoutProducerFamily &family = it->second;
  for (const TleLayoutDomainMember &member : family.members)
    if (member.encoding == encoding)
      return member.op->getResult(0);

  Operation *member = producer;
  if (!family.members.empty()) {
    OpBuilder builder(producer);
    builder.setInsertionPointAfter(family.members.back().op);
    member = builder.clone(*producer);
    member->setOperands(family.originalOperands);
  }
  family.members.push_back({encoding, member});
  SmallVector<Value> originalOperands = family.originalOperands;

  Attribute operandEncoding = inferSrcEncoding(producer, encoding);
  if (!operandEncoding)
    return member->getResult(0);

  for (auto [index, operand] : llvm::enumerate(originalOperands)) {
    // Gather ties only its indices to the result layout; its source can use
    // an independent encoding.
    if (isa<triton::GatherOp>(producer) && index == 0)
      continue;
    if (!isa<RankedTensorType>(operand.getType()))
      continue;
    member->setOperand(
        index, materializeTleLayoutDomain(operand, operandEncoding, families));
  }
  return member->getResult(0);
}

// A pure producer can be specialized independently, but its operands may
// include shared loads or loop-carried values that cannot be cloned. Keep
// those values in their selected layout and convert at each domain boundary.
static void materializeTleLayoutProducerOperands(
    const TleLayoutProducerFamilies &families) {
  for (const auto &[_, family] : families) {
    for (const TleLayoutDomainMember &domain : family.members) {
      Operation *member = domain.op;
      auto resultType = cast<RankedTensorType>(member->getResult(0).getType());
      Attribute operandEncoding =
          inferSrcEncoding(member, resultType.getEncoding());
      if (!operandEncoding)
        continue;

      OpBuilder builder(member);
      llvm::MapVector<Value, Value> conversions;
      for (OpOperand &operand : member->getOpOperands()) {
        if (isa<triton::GatherOp>(member) && operand.getOperandNumber() == 0)
          continue;
        Value value = operand.get();
        auto type = dyn_cast<RankedTensorType>(value.getType());
        if (!type || type.getEncoding() == operandEncoding)
          continue;
        auto [it, inserted] = conversions.insert({value, Value{}});
        if (inserted)
          it->second = ConvertLayoutOp::create(
              builder, member->getLoc(),
              type.cloneWithEncoding(operandEncoding), value);
        operand.set(it->second);
      }
    }
  }
}

static void
mergeTleLayoutProducerDomains(const TleLayoutProducerFamilies &families) {
  for (const auto &[_, family] : families) {
    llvm::MapVector<Type, Operation *> representatives;
    for (const TleLayoutDomainMember &domain : family.members) {
      Operation *member = domain.op;
      Type resultType = member->getResult(0).getType();
      auto [it, inserted] = representatives.insert({resultType, member});
      if (inserted)
        continue;
      if (Attribute encoding = getTleExplicitResultEncoding(member, 0))
        setTleExplicitResultEncoding(it->second, 0, encoding);
      member->getResult(0).replaceAllUsesWith(it->second->getResult(0));
      member->erase();
    }
  }
}

static LogicalResult
updateTleEncoding(ArrayRef<Value> values, TleEncodingInfo info, FuncOp func,
                  llvm::MapVector<Value, TleEncodingInfo> &valueToEncoding,
                  llvm::PriorityWorklist<Value> &worklist) {
  for (Value value : values) {
    if (!isa<RankedTensorType>(value.getType()))
      continue;

    // Layout propagation is intraprocedural. Mutating a tensor argument or
    // call result would otherwise leave the function signature/callers with
    // the old type. Normal JIT helper functions are inlined before this pass.
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (arg.getOwner() == &func.getBody().front())
        return func.emitOpError("Iluvatar set_layout requires tensor function "
                                "boundaries to be inlined");
    }
    if (auto call = value.getDefiningOp<triton::CallOp>())
      return call.emitOpError("Iluvatar set_layout requires tensor function "
                              "boundaries to be inlined");

    auto [it, inserted] = valueToEncoding.insert({value, info});
    if (!inserted) {
      Operation *defOp = value.getDefiningOp();
      Operation *diagOp = defOp ? defOp : func.getOperation();
      TleEncodingInfo merged;
      if (failed(mergeTleEncodingInfo(it->second, info, diagOp, merged)))
        return failure();
      if (merged.encoding == it->second.encoding &&
          merged.mayVary == it->second.mayVary)
        continue;
      it->second = merged;
    }
    worklist.insert(value);
  }
  return success();
}

static LogicalResult
propagateTleEncodingHints(FuncOp func,
                          ArrayRef<iluvatar_tle::SetLayoutOp> setLayoutOps) {
  TleLayoutProducerFamilies producerFamilies;
  for (iluvatar_tle::SetLayoutOp op : setLayoutOps) {
    Value source = materializeTleLayoutDomain(
        op.getSrc(), op.getTargetEncoding(), producerFamilies);
    op->setOperand(0, source);
  }

  SmallVector<std::pair<Value, TleEncodingInfo>> seedEncodings;
  for (iluvatar_tle::SetLayoutOp op : setLayoutOps) {
    // Pure producers may serve multiple requested layouts. The set_layout
    // result is the hard contract; its source may retain another encoding.
    // Reductions and scans have restricted native result layouts. If the
    // request cannot be propagated to their inputs, convert their result at
    // the set_layout boundary instead of assigning an invalid source hint.
    Operation *producer = op.getSrc().getDefiningOp();
    if (!isa_and_nonnull<triton::ReduceOp, triton::ScanOp>(producer) ||
        inferSrcEncoding(producer, op.getTargetEncoding()))
      seedEncodings.push_back(
          {op.getSrc(), TleEncodingInfo{op.getTargetEncoding(), true}});
    seedEncodings.push_back(
        {op.getResult(), TleEncodingInfo{op.getTargetEncoding(), false}});
  }

  llvm::MapVector<Value, TleEncodingInfo> valueToEncoding;
  llvm::PriorityWorklist<Value> worklist;
  for (auto &[value, info] : seedEncodings) {
    if (failed(
            updateTleEncoding({value}, info, func, valueToEncoding, worklist)))
      return failure();
  }

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    TleEncodingInfo info = valueToEncoding[value];
    assert(info && "worklist value must have an encoding");

    for (OpOperand &use : value.getUses()) {
      Operation *op = use.getOwner();
      if (isa<triton::CallOp, triton::ReturnOp>(op))
        return op->emitOpError("Iluvatar set_layout requires tensor function "
                               "boundaries to be inlined");
      if (isa<scf::ForOp, scf::WhileOp>(op)) {
        int offset = 3 * isa<scf::ForOp>(op);
        int tiedIndex = static_cast<int>(use.getOperandNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(op, tiedIndex);
        if (failed(updateTleEncoding(tiedArgs, info, func, valueToEncoding,
                                     worklist)))
          return failure();
        continue;
      }
      if (isa<scf::YieldOp, scf::ConditionOp>(op)) {
        Operation *parentOp = op->getParentOp();
        if (!isa_and_nonnull<scf::ForOp, scf::WhileOp, scf::IfOp>(parentOp))
          continue;
        int offset = isa<scf::ConditionOp>(op);
        int tiedIndex = static_cast<int>(use.getOperandNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(parentOp, tiedIndex);
        if (failed(updateTleEncoding(tiedArgs, info, func, valueToEncoding,
                                     worklist)))
          return failure();
        continue;
      }
      if (auto dot = dyn_cast<triton::DotOpInterface>(op)) {
        // A/B use dot-operand encodings, while C and D use the parent MMA
        // encoding. Only C and D are layout-equivalent across a dot.
        if (use.getOperandNumber() == 2 &&
            failed(updateTleEncoding({dot.getD()}, info, func, valueToEncoding,
                                     worklist)))
          return failure();
        continue;
      }
      if (isa<iluvatar_tle::SetLayoutOp>(op))
        continue;
      if (isa<triton::GatherOp>(op) && use.getOperandNumber() == 0)
        continue;

      if (isa<iluvatar_tle::LocalPointersOp>(op)) {
        if (failed(
                updateTleEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                  info, func, valueToEncoding, worklist)))
          return failure();
        continue;
      }

      Attribute dstEncoding = inferDstEncoding(op, info.encoding);
      if (!dstEncoding)
        continue;
      TleEncodingInfo dstInfo{dstEncoding,
                              info.mayVary || tleEncodingsMayVary(op)};
      if (failed(updateTleEncoding(llvm::to_vector_of<Value>(op->getResults()),
                                   dstInfo, func, valueToEncoding, worklist)))
        return failure();
    }

    if (auto opResult = dyn_cast<OpResult>(value)) {
      Operation *definingOp = opResult.getOwner();
      if (isa<scf::ForOp, scf::WhileOp, scf::IfOp>(definingOp)) {
        auto tiedArgs = getTiedArgs(definingOp, opResult.getResultNumber());
        if (failed(updateTleEncoding(tiedArgs, info, func, valueToEncoding,
                                     worklist)))
          return failure();
      } else if (isa<triton::DotOpInterface>(definingOp)) {
        if (failed(updateTleEncoding({definingOp->getOperand(2)}, info, func,
                                     valueToEncoding, worklist)))
          return failure();
      } else if (auto gather = dyn_cast<triton::GatherOp>(definingOp)) {
        if (failed(updateTleEncoding({gather.getIndices()}, info, func,
                                     valueToEncoding, worklist)))
          return failure();
      } else if (auto localPointers =
                     dyn_cast<iluvatar_tle::LocalPointersOp>(definingOp)) {
        llvm::SmallVector<Value> tensorIndices;
        for (Value index : localPointers.getIndices())
          if (isa<RankedTensorType>(index.getType()))
            tensorIndices.push_back(index);
        if (failed(updateTleEncoding(tensorIndices, info, func, valueToEncoding,
                                     worklist)))
          return failure();
      } else if (!isa<iluvatar_tle::SetLayoutOp>(definingOp)) {
        Attribute srcEncoding = inferSrcEncoding(definingOp, info.encoding);
        if (srcEncoding) {
          TleEncodingInfo srcInfo{
              srcEncoding, info.mayVary || tleEncodingsMayVary(definingOp)};
          llvm::SmallVector<Value> tensorOperands;
          for (Value operand : definingOp->getOperands())
            if (isa<RankedTensorType>(operand.getType()))
              tensorOperands.push_back(operand);
          if (failed(updateTleEncoding(tensorOperands, srcInfo, func,
                                       valueToEncoding, worklist)))
            return failure();
        }
      }
    } else if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<scf::ForOp, scf::WhileOp>(parentOp)) {
        int offset = isa<scf::ForOp>(parentOp);
        int tiedIndex = static_cast<int>(blockArg.getArgNumber()) - offset;
        if (tiedIndex < 0)
          continue;
        auto tiedArgs = getTiedArgs(parentOp, tiedIndex);
        if (failed(updateTleEncoding(tiedArgs, info, func, valueToEncoding,
                                     worklist)))
          return failure();
      }
    }
  }

  for (auto &[value, info] : valueToEncoding) {
    auto existingTy = cast<RankedTensorType>(value.getType());
    if (existingTy.getEncoding() != info.encoding) {
      auto newTy = existingTy.cloneWithEncoding(info.encoding);
      value.setType(newTy);

      if (auto opResult = dyn_cast<OpResult>(value)) {
        if (auto constant = dyn_cast<arith::ConstantOp>(opResult.getOwner())) {
          if (auto elements =
                  dyn_cast<DenseElementsAttr>(constant.getValueAttr()))
            constant.setValueAttr(elements.reshape(newTy));
        }
      }
    }

    // The source hint is only a preference. Turning it into a hard anchor
    // would make stores in another requested domain spuriously conflict.
    if (!info.mayVary)
      if (auto opResult = dyn_cast<OpResult>(value))
        setTleExplicitResultEncoding(opResult, info.encoding);
  }

  materializeTleLayoutProducerOperands(producerFamilies);
  mergeTleLayoutProducerDomains(producerFamilies);

  WalkResult memoryWalk = func.walk([&](Operation *op) {
    if (!getMemAccessPtr(op))
      return WalkResult::advance();

    Attribute explicitEncoding;
    if (failed(inferTleExplicitMemoryEncoding(op, explicitEncoding)))
      return WalkResult::interrupt();

    // Block arguments cannot carry per-result attributes, but their hard
    // constraints still apply to memory operations inside loops.
    for (Value operand : op->getOperands()) {
      auto it = valueToEncoding.find(operand);
      if (it == valueToEncoding.end() || it->second.mayVary)
        continue;
      Attribute encoding = it->second.encoding;
      if (explicitEncoding && explicitEncoding != encoding) {
        op->emitOpError("has conflicting explicit TLE memory encodings:\n  ")
            << explicitEncoding << "\nand\n  " << encoding;
        return WalkResult::interrupt();
      }
      explicitEncoding = encoding;
    }

    if (explicitEncoding)
      setTleExplicitMemoryEncoding(op, explicitEncoding);
    return WalkResult::advance();
  });
  if (memoryWalk.wasInterrupted())
    return failure();

  return success();
}

// Conversion assigns default layouts to previously unconstrained pointer
// and mask computations. Reconcile them only at each memory operation, after
// all operand types have been converted, so a shared address computation can
// serve several explicit data layouts without changing its other users.
static void materializeTleMemoryOperands(FuncOp func) {
  func.walk([](Operation *op) {
    if (!getMemAccessPtr(op))
      return;
    Attribute encoding = getTleExplicitMemoryEncoding(op);
    // A load/atomic result or stored value can have a soft layout preference
    // independent of a shared address computation. Reconcile its operands
    // without turning that preference into a hard constraint on either the
    // memory operation or the address producer.
    if (!encoding && op->getNumResults() == 1)
      if (auto type = dyn_cast<RankedTensorType>(op->getResult(0).getType()))
        encoding = type.getEncoding();
    if (!encoding)
      if (auto store = dyn_cast<triton::StoreOp>(op))
        if (auto type = dyn_cast<RankedTensorType>(store.getValue().getType()))
          encoding = type.getEncoding();
    if (!encoding)
      return;
    OpBuilder builder(op);
    llvm::MapVector<Value, Value> conversions;
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!type || type.getEncoding() == encoding)
        continue;
      auto [it, inserted] = conversions.insert({value, Value{}});
      if (inserted)
        it->second = ConvertLayoutOp::create(
            builder, op->getLoc(), type.cloneWithEncoding(encoding), value);
      operand.set(it->second);
    }
  });
}

static LogicalResult
applyTleEncodingHints(ModuleOp mod,
                      SmallVectorImpl<StringAttr> &layoutFunctions) {
  WalkResult validation = mod.walk([](iluvatar_tle::SetLayoutOp op) {
    return failed(validateIluvatarTleSetLayout(op)) ? WalkResult::interrupt()
                                                    : WalkResult::advance();
  });
  if (validation.wasInterrupted())
    return failure();

  for (FuncOp func : mod.getOps<FuncOp>()) {
    SmallVector<iluvatar_tle::SetLayoutOp> setLayoutOps;
    func.walk(
        [&](iluvatar_tle::SetLayoutOp op) { setLayoutOps.push_back(op); });
    if (setLayoutOps.empty())
      continue;
    if (failed(propagateTleEncodingHints(func, setLayoutOps)))
      return failure();
    // FuncOp is replaced during conversion; retain its stable symbol name.
    layoutFunctions.push_back(func.getSymNameAttr());
  }
  return success();
}

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      // This is a hack. We just want to add encoding.
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::gpu::BlockedEncodingAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    auto newRank = retShape.size();
    // return encoding
    auto retSizePerThread = llvm::to_vector(argEncoding.getSizePerThread());
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = to_vector(argEncoding.getThreadsPerWarp());
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = to_vector(argEncoding.getWarpsPerCTA());
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto ctaLl = argEncoding.getCTALayout().getLinearLayout();
    auto kBlock = *ctaLl.getInDimNames().begin();
    auto *ctx = kBlock.getContext();
    auto newDim = standardOutDimNames(ctx, newRank)[newRank - 1];
    ctaLl *= LinearLayout::identity1D(1, kBlock, newDim);
    // Move last dim to op.getAxis(). nb is this a std::rotate?
    auto newOrder = to_vector(llvm::seq<int32_t>(newRank));
    for (int i = newRank - 1; i >= op.getAxis() + 1; --i) {
      std::swap(newOrder[i], newOrder[i - 1]);
    }
    ctaLl = transposeLinearLayout(ctaLl, newOrder);
    auto retCTALayout = CTAEncodingAttr::get(ctx, std::move(ctaLl));
    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCTALayout);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = argType.cloneWithEncoding(newArgEncoding);
    // construct new op
    auto newSrc = triton::gpu::ConvertLayoutOp::create(
        rewriter, op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();
    auto rank = origShape.size();
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    retSizePerThread[rank - 1] = std::min(
        retSizePerThread[rank - 1], static_cast<unsigned>(origShape[rank - 1]));
    retSizePerThread[rank - 2] = std::min(
        retSizePerThread[rank - 2], static_cast<unsigned>(origShape[rank - 2]));

    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i)
      retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    // Explicit blocked results must keep the exact encoding, including the
    // Iluvatar-specific fields. Rebuilding a default blocked encoding can
    // produce a distinct type and leave live source materializations behind.
    if (isa_and_nonnull<BlockedEncodingAttr>(origType.getEncoding()))
      dEncoding = origType.getEncoding();
    RankedTensorType retType = origType.cloneWithEncoding(dEncoding);
    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType = aType.cloneWithEncoding(encoding);
      a = triton::gpu::ConvertLayoutOp::create(rewriter, a.getLoc(), dstType,
                                               a);
    }
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType = bType.cloneWithEncoding(encoding);
      b = triton::gpu::ConvertLayoutOp::create(rewriter, b.getLoc(), dstType,
                                               b);
    }
    c = triton::gpu::ConvertLayoutOp::create(rewriter, c.getLoc(), retType, c);

    // Dot operands require a native parent encoding. A requested sliced
    // result is represented by a conversion after the dot, with C converted
    // to the native accumulator type above.
    if (origType.getEncoding() && origType != retType) {
      auto newDot = triton::DotOp::create(rewriter, op.getLoc(), retType, a, b,
                                          c, adaptor.getInputPrecision(),
                                          adaptor.getMaxNumImpreciseAcc());
      addNamedAttrs(newDot, adaptor.getAttributes());
      newDot->removeAttr(getTleExplicitEncodingAttrName(0));
      auto converted = ConvertLayoutOp::create(rewriter, op.getLoc(), origType,
                                               newDot.getResult());
      setTleExplicitResultEncoding(converted.getOperation(), 0,
                                   origType.getEncoding());
      rewriter.replaceOp(op, converted.getResult());
      return success();
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The cat op satisfy two conditions:
    // 1. output.numel = lhs.numel + rhs.numel
    // 2. output.total_elems_per_thread =
    // next_power_of_2(lhs.total_elems_per_thread + rhs.total_elems_per_thread)
    // For now, this behaves like generic, but this
    // will evolve when we add support for `can_reorder=False`.
    auto retType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    auto retEncoding =
        cast<triton::gpu::BlockedEncodingAttr>(retType.getEncoding());
    auto lhsType = adaptor.getLhs().getType();
    auto rhsType = adaptor.getRhs().getType();
    auto lhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(retType);
    auto retShape = retType.getShape();
    auto retOrder = retEncoding.getOrder();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    // Get new retSizePerThread if ret elems per thread is not enough.
    // We have to round it up to the next power of 2 due to triton's tensor size
    // constraint.
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = llvm::to_vector(retEncoding.getSizePerThread());
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton::gpu::BlockedEncodingAttr newRetEncoding =
        triton::gpu::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
    auto newRetType = retType.cloneWithEncoding(newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, newRetType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Simply rely on type inference for this op.  (Notably, GenericOpPattern
    // does not do this, instead it assigns the default layout to the ins and
    // outs.)
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, adaptor.getLhs(), adaptor.getRhs()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    int rank = srcEnc.getOrder().size();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();

    // The operand to split must have:
    //  - a blocked layout, with
    //  - sizePerThread = 2 in the last dimension,
    //  - threadsPerWarp, warpsPerCTA, and CTAsPerCGA = 1 in the last dim, and
    //  - the last dimension minor.
    // If that's not the case, add a convert before the split.
    if (!srcEnc || srcEnc.getSizePerThread().back() != 2 ||
        srcEnc.getOrder().front() != rank - 1) {
      // If we take the default encoding for the op's result (i.e. post-split)
      // and add 1 to the end of each dim, that gives us what we want.  Other
      // than making a legal src encoding, our choice of layout doesn't matter;
      // it'll get fixed by RemoveLayoutConversions.
      auto defaultEnc = getDefaultBlockedEncoding(
          getContext(),
          cast<RankedTensorType>(op.getResult(0).getType()).getShape(),
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
          typeConverter->getNumCTAs());

      auto append = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res(vals);
        res.push_back(val);
        return res;
      };
      auto prepend = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res;
        res.push_back(val);
        res.append(vals.begin(), vals.end());
        return res;
      };

      auto layout = defaultEnc.getCTALayout().getLinearLayout();
      auto kBlock = StringAttr::get(getContext(), "block");
      auto newDim = standardOutDimNames(getContext(), rank)[rank - 1];
      layout *= LinearLayout::identity1D(1, kBlock, newDim);
      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CTAEncodingAttr::get(getContext(), layout));
      srcTy = srcTy.cloneWithEncoding(srcEnc);
      src = ConvertLayoutOp::create(rewriter, op.getLoc(), srcTy, src);
    }

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = srcTy.getEncoding();
    if (!srcEnc)
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = op.getType().cloneWithEncoding(srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = triton::ReduceOp::create(
        rewriter, op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan =
        triton::ScanOp::create(rewriter, op.getLoc(), adaptor.getOperands(),
                               adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

struct TritonMapElementwisePattern
    : public OpConversionPattern<triton::MapElementwiseOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::MapElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    SmallVector<Type> resultTys;
    auto err = converter->convertTypes(op.getResults().getType(), resultTys);
    if (failed(err)) {
      return err;
    }

    auto newMapOp = triton::MapElementwiseOp::create(
        rewriter, op.getLoc(), resultTys, adaptor.getOperands(), op.getPack());
    addNamedAttrs(newMapOp, adaptor.getAttributes());

    auto &newScalarOp = newMapOp.getScalarOp();
    rewriter.cloneRegionBefore(op.getScalarOp(), newScalarOp,
                               newScalarOp.end());
    rewriter.replaceOp(op, newMapOp.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    // Convert just the entry block. The remaining unstructured control flow is
    // converted by br patterns.
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns, unsigned numCTAs) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::UnsplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>,
      TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>,
      TritonExpandDimsPattern,
      TritonTransPattern,
      TritonDotPattern,
      TritonMapElementwisePattern,
      GatherScatterOpPattern<DescriptorGatherOp>,
      GatherScatterOpPattern<DescriptorScatterOp>,
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::DescriptorReduceOp>,
      // this assumes the right layout will be set later for dot scaled.
      GenericOpPattern<triton::DotScaledOp>,
      GenericOpPattern<triton::CallOp>,
      GenericOpPattern<ReturnOp>,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
}
//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = scf::WhileOp::create(rewriter, op.getLoc(), newResultTypes,
                                      adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}

#ifdef __ILUVATAR_TLE__
// TLE-Raw. Unlike the other TLE ops, dsl_region carries a region whose block
// arguments mirror the operands, so the region signature has to be converted
// alongside the op itself.
class TleDSLRegionOpPattern
    : public OpConversionPattern<triton::iluvatar_tle::DSLRegionOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::iluvatar_tle::DSLRegionOp op,
                  triton::iluvatar_tle::DSLRegionOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        rewriter.cloneWithoutRegions<triton::iluvatar_tle::DSLRegionOp>(op);
    Region &body = op.getBody(), &newBody = newOp.getBody();
    rewriter.inlineRegionBefore(body, newBody, newBody.end());

    if (failed(rewriter.convertRegionTypes(&newBody, *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    newOp->setOperands(adaptor.getOperands());
    for (OpResult result : newOp->getResults()) {
      result.setType(getTypeConverter()->convertType(result.getType()));
    }

    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

class TleSetLayoutOpPattern
    : public OpConversionPattern<iluvatar_tle::SetLayoutOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(iluvatar_tle::SetLayoutOp op,
                  iluvatar_tle::SetLayoutOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type convertedType = getTypeConverter()->convertType(op.getResult());
    auto resultType = dyn_cast<RankedTensorType>(convertedType);
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor result");

    // Keep a boundary even for identical types: block arguments cannot carry
    // result attributes, and canonicalization may replace the source producer.
    // LLVM lowering removes identity conversions after layout selection.
    Attribute targetEncoding = op.getTargetEncoding();
    auto convert = rewriter.replaceOpWithNewOp<triton::gpu::ConvertLayoutOp>(
        op, resultType, adaptor.getSrc());
    convert->setAttr(getTleExplicitEncodingAttrName(0), targetEncoding);
    return success();
  }
};

void populateIluvatarTlePatterns(TritonGPUTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<TleSetLayoutOpPattern, TleDSLRegionOpPattern>(typeConverter,
                                                             context);
  patterns.add<GenericOpPattern<triton::iluvatar_tle::YieldOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractAllocatedPtrOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractAlignedPtrOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractOffsetOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractSizesOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractStridesOp>,
               GenericOpPattern<triton::iluvatar_tle::ExtractPtrOp>,
               GenericOpPattern<triton::iluvatar_tle::PackOp>>(typeConverter,
                                                               context);
  patterns.add<GenericOpPattern<triton::iluvatar_tle::ExtractTileOp>,
               GenericOpPattern<triton::iluvatar_tle::InsertTileOp>,
               GenericOpPattern<triton::iluvatar_tle::ExclusiveCumsumOp>,
               GenericOpPattern<triton::iluvatar_tle::LocalPointersOp>,
               GenericOpPattern<triton::iluvatar_tle::RemotePointersOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeCreateOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeWriterAcquireOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeWriterCommitOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeWriterCloseOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeReaderWaitOp>,
               GenericOpPattern<triton::iluvatar_tle::PipeReaderReleaseOp>>(
      typeConverter, context);
}
#endif

class ConvertTritonToTritonGPU
    : public triton::impl::ConvertTritonToTritonGPUBase<
          ConvertTritonToTritonGPU> {
public:
  using ConvertTritonToTritonGPUBase::ConvertTritonToTritonGPUBase;

  void runOnOperation() override {
    if (target.getValue().empty()) {
      mlir::emitError(
          getOperation().getLoc(),
          "'convert-triton-to-tritongpu' requires 'target' option to be set");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, enableSourceRemat);
    TritonGPUConversionTarget target(*context, typeConverter);
    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
#ifdef __ILUVATAR_TLE__
    populateIluvatarTlePatterns(typeConverter, patterns);
#endif
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    SmallVector<StringAttr> layoutFunctions;
    if (failed(applyTleEncodingHints(mod, layoutFunctions)))
      return signalPassFailure();

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
    for (StringAttr name : layoutFunctions)
      materializeTleMemoryOperands(mod.lookupSymbol<FuncOp>(name));
  }
};

} // namespace
